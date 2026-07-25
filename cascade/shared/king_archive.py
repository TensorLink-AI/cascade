"""King archive — a permanent private record of every throne-holding generator.

A generator wins the cascade king-of-the-hill and its ``repo@digest`` reigns
until dethroned. Those generator repos live on the **public** Hippius Hub
registry (content-addressed, but a miner can delete their repo at any time) and
the throne history lives only in the validator's public ``receipts/index.json``.
This module snapshots both into a **private** S3-compatible bucket (Cloudflare
R2), so cascade keeps a durable, independent record of every king even if the
upstream Hub repo disappears.

Two things are written to the archive bucket:

* ``kings/<repo>/<digest>.tar`` — the king's generator code, fetched from the Hub
  by its content-addressed ref and packed to a **deterministic** tar (the same
  reproducible packing the eval-pool snapshots use, so the tar's sha256 is
  stable). Content-addressed by the OCI digest, so the archive is inherently
  append-only and de-duplicated: an already-archived king is never re-fetched.
* ``kings/index.json`` — the "db": one entry per distinct king generator, each
  pointing back at its archived object (``archive_key`` / ``archive_url``) with
  the throne attribution (hotkey, uid, the rounds it reigned).

The source of truth for *who was king* is the validator's public
``receipts/index.json`` (see :func:`cascade.shared.hippius.read_receipt_index`);
:func:`collect_king_refs` distils the throne history from it. Everything here is
stdlib-only and pure except :func:`sync_kings`, which drives the Hub fetch + S3
writes through injectable seams so it unit-tests without a network.
"""

from __future__ import annotations

import json
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

from .hippius import (
    HubConfig,
    HubRef,
    S3Config,
    S3Store,
    StorageError,
    fetch_from_hub,
    is_hub_ref,
    pack_dir_to_tar,
    read_receipt_index,
    tar_cid_digest,
)

KING_INDEX_KEY = "kings/index.json"
KING_INDEX_SCHEMA = 1


# ───────────────────────────── who was king (pure) ──────────────────────────


def _is_scored(rnd: dict) -> bool:
    # A rejected round crowns no one; only a scored round names a king. An index
    # written before the ``status`` field existed is treated as scored.
    return str(rnd.get("status", "scored")) == "scored"


def collect_king_refs(index_doc: dict) -> OrderedDict[str, dict]:
    """Distil the throne history from a ``receipts/index.json`` document.

    Returns an ordered ``{gen_ref: attribution}`` map over every distinct
    generator that has held the throne, sorted by when it first reigned
    (``first_epoch_start_block`` then ``round_id``). A generator counts as a
    king if it was the ``king_gen_ref`` of any scored round OR the winning
    ``chal_gen_ref`` of a scored round it dethroned in — the latter catches a
    freshly-crowned king even when the round that shows it *reigning* hasn't
    been published yet (or has already scrolled off the rolling index window).

    Attribution per ref: ``repo`` / ``digest`` (parsed), the ``hotkey`` / ``uid``
    that owned it, ``reign_rounds`` (scored rounds it was the reigning king),
    the first/last round it was seen on the throne, and ``crowned_round_id`` (the
    round it took the throne, if that dethrone was in the index). Refs that don't
    parse as a Hub ``repo@digest`` are skipped.
    """
    kings: OrderedDict[str, dict] = OrderedDict()

    def observe(gen_ref, hotkey, uid, round_id, epoch_block, *, reign, crowned):
        if not gen_ref or not is_hub_ref(str(gen_ref)):
            return
        gen_ref = str(gen_ref)
        try:
            ref = HubRef.parse(gen_ref)
        except StorageError:
            return
        epoch_block = int(epoch_block or 0)
        rid = str(round_id or "")
        cur = kings.get(gen_ref)
        if cur is None:
            cur = {
                "gen_ref": gen_ref,
                "repo": ref.repo,
                "digest": ref.digest,
                "hotkey": str(hotkey) if hotkey else None,
                "uid": (int(uid) if uid is not None else None),
                "first_round_id": rid,
                "first_epoch_start_block": epoch_block,
                "last_round_id": rid,
                "last_epoch_start_block": epoch_block,
                "reign_rounds": 0,
                "crowned_round_id": None,
            }
            kings[gen_ref] = cur
        # Keep the earliest/latest reign extents.
        if epoch_block < cur["first_epoch_start_block"]:
            cur["first_epoch_start_block"] = epoch_block
            cur["first_round_id"] = rid
        if epoch_block >= cur["last_epoch_start_block"]:
            cur["last_epoch_start_block"] = epoch_block
            cur["last_round_id"] = rid
        # Prefer a concrete hotkey/uid if we didn't have one yet.
        if cur["hotkey"] is None and hotkey:
            cur["hotkey"] = str(hotkey)
        if cur["uid"] is None and uid is not None:
            cur["uid"] = int(uid)
        if reign:
            cur["reign_rounds"] += 1
        if crowned and cur["crowned_round_id"] is None:
            cur["crowned_round_id"] = rid

    rounds = index_doc.get("rounds", []) if isinstance(index_doc, dict) else []
    rounds = [r for r in rounds if isinstance(r, dict) and _is_scored(r)]
    rounds.sort(key=lambda r: (int(r.get("epoch_start_block", 0)), str(r.get("round_id", ""))))

    for rnd in rounds:
        eb = rnd.get("epoch_start_block", 0)
        rid = rnd.get("round_id", "")
        # The reigning king this round.
        observe(
            rnd.get("king_gen_ref"), rnd.get("king_hotkey"), rnd.get("king_uid"),
            rid, eb, reign=True, crowned=False,
        )
        # A challenger that won the round is the new king from here on.
        if rnd.get("challenger_wins_round") or rnd.get("dethroned"):
            observe(
                rnd.get("chal_gen_ref"), rnd.get("chal_hotkey"), rnd.get("chal_uid"),
                rid, eb, reign=False, crowned=True,
            )

    ordered = OrderedDict(
        sorted(kings.items(),
               key=lambda kv: (kv[1]["first_epoch_start_block"], kv[1]["first_round_id"]))
    )
    return ordered


# ─────────────────────────── archive addressing (pure) ──────────────────────


def archive_key_for_ref(gen_ref: str) -> str:
    """The archive object key for a king generator ref (content-addressed).

    ``<repo>@<scheme>:<hex>`` → ``kings/<repo>/<scheme>-<hex>.tar``. The OCI
    digest pins the content, so the key is stable and collision-free: the same
    generator always maps to the same object, which is what makes the archive
    append-only.
    """
    ref = HubRef.parse(gen_ref)
    return f"kings/{ref.repo}/{ref.digest.replace(':', '-')}.tar"


def archive_url(endpoint: str, bucket: str, key: str) -> str:
    """A resolvable (credentialed) URL to an archived object in the R2 bucket."""
    return f"{endpoint.rstrip('/')}/{bucket}/{key}"


# ───────────────────────────── the index "db" ───────────────────────────────


def read_king_index(store: S3Store) -> dict:
    """Read ``kings/index.json``; return an empty index if absent/malformed."""
    empty = {"schema": KING_INDEX_SCHEMA, "kings": []}
    try:
        text = store.get_text(KING_INDEX_KEY)
    except StorageError:
        return empty
    try:
        doc = json.loads(text)
    except (ValueError, TypeError):
        return empty
    if not isinstance(doc, dict) or not isinstance(doc.get("kings"), list):
        return empty
    return doc


def _attribution_fields(attr: dict) -> dict:
    """The throne-history fields of an index entry (everything but the archive
    location). Rebuilt on every sync so reign extents stay current."""
    return {
        "gen_ref": attr["gen_ref"],
        "repo": attr["repo"],
        "digest": attr["digest"],
        "hotkey": attr.get("hotkey"),
        "uid": attr.get("uid"),
        "first_round_id": attr.get("first_round_id"),
        "first_epoch_start_block": attr.get("first_epoch_start_block"),
        "last_round_id": attr.get("last_round_id"),
        "last_epoch_start_block": attr.get("last_epoch_start_block"),
        "reign_rounds": attr.get("reign_rounds", 0),
        "crowned_round_id": attr.get("crowned_round_id"),
    }


def _merge_attribution(old: dict, new: dict) -> dict:
    """Fold a prior index entry's throne history into a freshly-computed one.

    ``receipts/index.json`` is a ROLLING window, so a re-derived entry can't see
    rounds that have since scrolled off it. Merge conservatively so the db never
    regresses: keep the EARLIEST first-seen and the LATEST last-seen across both,
    and the LARGER ``reign_rounds`` (recomputation from a shrunk window can only
    undercount). ``crowned_round_id`` / ``hotkey`` / ``uid`` prefer whichever is
    already known.
    """
    merged = dict(new)
    if int(old.get("first_epoch_start_block", 1 << 62)) <= int(new.get("first_epoch_start_block", 0)):
        merged["first_epoch_start_block"] = old.get("first_epoch_start_block")
        merged["first_round_id"] = old.get("first_round_id")
    if int(old.get("last_epoch_start_block", -1)) >= int(new.get("last_epoch_start_block", 0)):
        merged["last_epoch_start_block"] = old.get("last_epoch_start_block")
        merged["last_round_id"] = old.get("last_round_id")
    merged["reign_rounds"] = max(int(old.get("reign_rounds", 0)),
                                 int(new.get("reign_rounds", 0)))
    merged["crowned_round_id"] = new.get("crowned_round_id") or old.get("crowned_round_id")
    merged["hotkey"] = new.get("hotkey") or old.get("hotkey")
    merged["uid"] = new.get("uid") if new.get("uid") is not None else old.get("uid")
    return merged


# ───────────────────────────── sync orchestration ───────────────────────────


@dataclass
class SyncResult:
    """Outcome of a :func:`sync_kings` run."""

    archived: int = 0          # kings fetched + uploaded this run
    skipped: int = 0           # already-archived kings (metadata refreshed only)
    would_archive: int = 0     # dry-run: kings that WOULD be fetched + uploaded
    failed: list[str] = field(default_factory=list)   # gen_refs that errored
    index: dict = field(default_factory=dict)         # the written kings/index.json

    @property
    def total_kings(self) -> int:
        return len(self.index.get("kings", []))


def king_archive_config(storage: object) -> tuple[S3Config, str, str]:
    """Build the ``(S3Config, endpoint, bucket)`` for the private R2 king archive.

    The endpoint/region default to the ``backup_*`` R2 values (the same account
    already configured for the manifest/receipt mirror); the bucket defaults to
    ``cascade-king-archive``. Credentials come from KING_ARCHIVE_S3_ACCESS_KEY /
    KING_ARCHIVE_S3_SECRET_KEY, falling back to the BACKUP_S3_* pair when unset —
    so an operator who already runs the R2 backup needs no new credentials.
    """
    import os

    endpoint = (getattr(storage, "king_archive_s3_endpoint", "")
                or getattr(storage, "backup_s3_endpoint", ""))
    if not endpoint:
        raise StorageError(
            "no king-archive endpoint: set [storage] king_archive_s3_endpoint "
            "(or backup_s3_endpoint) to your R2 endpoint"
        )
    region = (getattr(storage, "king_archive_s3_region", "")
              or getattr(storage, "backup_s3_region", "") or "auto")
    bucket = getattr(storage, "king_archive_bucket", "") or "cascade-king-archive"
    use_king_env = bool(os.environ.get("KING_ARCHIVE_S3_ACCESS_KEY"))
    cfg = S3Config(
        endpoint=endpoint,
        region=region,
        bucket=bucket,
        access_key_env="KING_ARCHIVE_S3_ACCESS_KEY" if use_king_env else "BACKUP_S3_ACCESS_KEY",
        secret_key_env="KING_ARCHIVE_S3_SECRET_KEY" if use_king_env else "BACKUP_S3_SECRET_KEY",
    )
    return cfg, endpoint, bucket


def sync_kings(
    *,
    manifest_store: S3Store,
    archive_store: S3Store,
    hub: HubConfig,
    endpoint: str,
    bucket: str,
    dry_run: bool = False,
    updated_at: str = "",
    fetch=fetch_from_hub,
    pack=pack_dir_to_tar,
    tmp_root: str | None = None,
    log=lambda _msg: None,
) -> SyncResult:
    """Archive every king generator that isn't already in the private bucket.

    Reads the throne history from ``manifest_store`` (``receipts/index.json``),
    then for each distinct king ref: if it's already archived (an index entry
    with a ``tar_sha256``) only its throne metadata is refreshed; otherwise the
    generator is fetched from the Hub, packed to a deterministic tar, and written
    to ``archive_store`` under a content-addressed key. Finally the merged
    ``kings/index.json`` "db" is written back (skipped on ``dry_run``).

    ``fetch`` / ``pack`` are injectable so the whole flow unit-tests without a
    Hub or S3 endpoint. Returns a :class:`SyncResult`.
    """
    index_doc = read_receipt_index(manifest_store)
    kings = collect_king_refs(index_doc)
    log(f"found {len(kings)} distinct king generator(s) in the receipt index")

    prior = read_king_index(archive_store)
    prior_by_ref = {str(e.get("gen_ref")): dict(e) for e in prior.get("kings", [])
                    if isinstance(e, dict)}

    result = SyncResult()
    # Start from the prior db so kings that have scrolled off the rolling receipt
    # window are PRESERVED (the archive is permanent; the index window is not) —
    # and a transient empty read never blanks the db. Current-window kings then
    # update or extend it.
    merged: dict[str, dict] = dict(prior_by_ref)

    for gen_ref, attr in kings.items():
        key = archive_key_for_ref(gen_ref)
        url = archive_url(endpoint, bucket, key)
        base = _attribution_fields(attr)
        base["archive_key"] = key
        base["archive_url"] = url

        old = prior_by_ref.get(gen_ref)
        if old:
            base = _merge_attribution(old, base)
            base["archive_key"] = key
            base["archive_url"] = url
        already = bool(old and old.get("tar_sha256"))
        if already:
            # Content-addressed ⇒ the tar is immutable; keep it, refresh metadata.
            base["tar_sha256"] = old.get("tar_sha256")
            base["size_bytes"] = old.get("size_bytes")
            base["archived_at"] = old.get("archived_at") or updated_at
            merged[gen_ref] = base
            result.skipped += 1
            continue

        if dry_run:
            log(f"WOULD ARCHIVE {gen_ref} -> {key}")
            base["tar_sha256"] = None
            base["size_bytes"] = None
            base["archived_at"] = None
            merged[gen_ref] = base
            result.would_archive += 1
            continue

        try:
            with tempfile.TemporaryDirectory(dir=tmp_root, prefix="king-") as td:
                dest = fetch(gen_ref, Path(td) / "gen", hub)
                tar_bytes = pack(dest)
            sha = tar_cid_digest(tar_bytes)
            archive_store.put_bytes(key, tar_bytes, content_type="application/x-tar")
            base["tar_sha256"] = sha
            base["size_bytes"] = len(tar_bytes)
            base["archived_at"] = updated_at
            merged[gen_ref] = base
            result.archived += 1
            log(f"archived {gen_ref} -> {key} ({len(tar_bytes)} bytes, sha256 {sha[:12]}…)")
        except (StorageError, OSError) as e:
            log(f"FAILED {gen_ref}: {type(e).__name__}: {e}")
            result.failed.append(gen_ref)
            # A transient failure never drops a king already recorded in the db:
            # its prior entry is still in `merged` from the seed above.

    entries = sorted(merged.values(),
                     key=lambda e: (int(e.get("first_epoch_start_block", 0)),
                                    str(e.get("first_round_id", ""))))
    doc: dict = {
        "schema": KING_INDEX_SCHEMA,
        "endpoint": endpoint,
        "bucket": bucket,
        "kings": entries,
    }
    if updated_at:
        doc["updated_at"] = updated_at
    result.index = doc

    if not dry_run:
        archive_store.put_text(
            KING_INDEX_KEY, json.dumps(doc, indent=2, sort_keys=True),
            content_type="application/json",
        )
        log(f"wrote {KING_INDEX_KEY}: {len(entries)} king(s) "
            f"({result.archived} new, {result.skipped} already archived)")

    return result
