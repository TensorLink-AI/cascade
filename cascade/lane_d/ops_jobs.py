"""Lane D recurring ops jobs: ``python -P -m cascade.lane_d.ops_jobs <job> ...`` (plan §10.4
per-settlement runbook, §10.7 monitoring, §1.4 house, §1.2 retention, §11.1).

Every job is one idempotent command: re-running it with nothing new is a
no-op, ``--dry-run`` prints what it would write and writes nothing, and it
never reads the chain, rents or deploys. Only ``king-publish`` and
``takedown`` touch a bucket or repo (the existing public mirrors and the
public manifest store, never a new one), and only without ``--dry-run`` and
without ``--vault-dir`` / ``--mirror`` (the local filesystem mode testnet and
the tests use); every other input is a local file. The systemd TEMPLATES in
``deploy/laned/systemd/`` schedule them; nothing here installs anything.

Jobs:

* ``house-pointer``: re-sign the W5f house pointer when lane A's king
  changes. The lane A king pointer is read from a local JSON (the published
  ``status/v2.json``, ``king.pointer`` by default). The house checkpoint for
  that king (its lane D manifest digest, D47's BF16 cast, and D46's
  ``reveal_ts``) comes from a ``laned_house_source/1`` file the owner prepares
  per king; without one for the CURRENT king the job exits 3 (owner step).
  The pointer is signed with the validator hotkey through
  :func:`cascade.validator.lane_d_house.sign_house_pointer`.
* ``evidence``: validate a new ``laned_readiness/1`` (W1 ``--readiness-out``)
  and/or ``laned_t7/1`` (the T7 pod run, an owner step) and install it
  atomically at the path the validator's ``--lane-d-inputs`` TOML names.
  A document that does not parse, a T7 that is not pod evidence, or one that
  disarms a currently armed context is refused (``--allow-disarm`` overrides
  the last).
* ``vault-ledger``: export the ``laned_vault_ledger/1`` the W18 sweeper reads
  (:func:`cascade.validator.lane_d_status.build_vault_ledger`) from
  ``lane_d_state.json`` and its persisted reports.
* ``king-check``: every crowned persisted report against the signed lane D
  index (and, given local copies, the vault and the mirrors): §11.1 "one
  digest, one identity" via :func:`cascade.lane_d.publish.check_one_digest`.
* ``house-source``: verify a ``laned_house_source/1`` produced on a pod by
  ``python -P -m cascade.lane_d.house_source build`` (D47 BF16 cast, D46 T)
  against its ``laned_house_provenance/1`` (stdlib re-hash, no weights read)
  and install both atomically where ``house-pointer`` reads them.
* ``king-publish``: for every crowned persisted report not yet in the signed
  index (the validator's settlement step leaves a
  ``laned_king_publish_request/1`` per crowning), copy the king from the vault
  to the Hub, R2 and HF mirrors (:func:`cascade.lane_d.publish.publish_king`,
  read-back re-hash), re-sign ``champions/lane_d/index.json`` with the
  validator hotkey, put it on the public store and write the local copy
  ``king-check`` reads. Holds one checkpoint in memory (~2.5 GB): run it where
  that fits (testnet-ctl / the builder VM), not on a loaded box.
* ``takedown``: B13, unpublish one king from the Hub and HF mirrors and
  re-sign the index (``--reason licence|rights``).

Exit codes (every job): 0 done / nothing to do / dry-run planned; 1 an input
is unreadable or invalid, or a check found a problem (nothing written);
2 usage; 3 blocked on an owner step.

Log lines (stderr, ``%(asctime)s %(levelname)s %(name)s: %(message)s``, the
cascade-ops follower's format) start with a ``lane_d_ops_*`` tag listed in
:data:`LOG_TAGS`. A JSON summary goes to stdout.

Light imports only (no torch); numpy is loaded by ``evidence`` alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("cascade.lane_d.ops")

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_BLOCKED = 0, 1, 2, 3
HOUSE_SOURCE_SCHEMA = "laned_house_source/1"
HOUSE_SOURCE_FIELDS = ("schema", "king_pointer", "digest", "reveal_ts")
SIDECAR_SUFFIX = ".source.json"

# tag -> level (the follower patterns the owner adds to cascade-ops)
LOG_TAGS: dict[str, str] = {
    "lane_d_ops_house_current": "info",
    "lane_d_ops_house_no_lane_a_king": "info",
    "lane_d_ops_house_resign_planned": "info",
    "lane_d_ops_house_resigned": "warning",
    "lane_d_ops_house_resign_blocked": "error",
    "lane_d_ops_house_failed": "error",
    "lane_d_ops_evidence_current": "info",
    "lane_d_ops_evidence_planned": "info",
    "lane_d_ops_evidence_installed": "warning",
    "lane_d_ops_evidence_refused": "error",
    "lane_d_ops_ledger_current": "info",
    "lane_d_ops_ledger_planned": "info",
    "lane_d_ops_ledger_written": "info",
    "lane_d_ops_ledger_failed": "error",
    "lane_d_ops_king_ok": "info",
    "lane_d_ops_king_unpublished": "critical",
    "lane_d_ops_king_check_failed": "error",
    "lane_d_ops_house_source_current": "info",
    "lane_d_ops_house_source_planned": "info",
    "lane_d_ops_house_source_installed": "warning",
    "lane_d_ops_house_source_refused": "error",
    "lane_d_ops_king_publish_current": "info",
    "lane_d_ops_king_publish_planned": "info",
    "lane_d_ops_king_published": "warning",
    "lane_d_ops_king_publish_blocked": "error",
    "lane_d_ops_king_publish_failed": "error",
    "lane_d_ops_takedown_planned": "info",
    "lane_d_ops_takedown_done": "warning",
    "lane_d_ops_takedown_failed": "error",
}


class JobError(Exception):
    """A job input is unreadable or invalid: exit 1, nothing written."""


class JobBlocked(Exception):
    """The job needs an owner step first: exit 3, nothing written."""


def _read_json(path: Path, what: str):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise JobError(f"{what} {path} is absent") from None
    except (OSError, UnicodeDecodeError, ValueError) as e:
        raise JobError(f"{what} {path} is unreadable ({type(e).__name__})") from None


def _write_atomic(path: Path, data: bytes) -> None:
    from ..shared.atomic import write_text_atomic

    write_text_atomic(path, data.decode("utf-8"))


def _field(doc, dotted: str):
    cur = doc
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise JobError(f"the king source has no {dotted!r}")
        cur = cur[part]
    return cur


# ── house-pointer ────────────────────────────────────────────────────────────


def _house_source(path: Path | None, king: str) -> dict:
    from ..validator.lane_d_house import LaneDHouseError, check_house

    if path is None:
        raise JobBlocked(f"lane A king changed to {king}: no --house-source given (prepare the "
                         f"{HOUSE_SOURCE_SCHEMA} for it: BF16 cast digest and D46 reveal_ts)")
    if not Path(path).exists():
        raise JobBlocked(f"lane A king changed to {king}: {path} is absent (prepare the "
                         f"{HOUSE_SOURCE_SCHEMA} for it)")
    src = _read_json(path, "house source")
    if not isinstance(src, dict) or sorted(src) != sorted(HOUSE_SOURCE_FIELDS) \
            or src.get("schema") != HOUSE_SOURCE_SCHEMA:
        raise JobError(f"{path} is not exactly {HOUSE_SOURCE_SCHEMA} {list(HOUSE_SOURCE_FIELDS)}")
    if src["king_pointer"] != king:
        raise JobBlocked(f"{path} is for lane A king {src['king_pointer']!r}, not the current "
                         f"{king!r} (prepare the house source for the new king)")
    try:
        house = check_house({"digest": src["digest"], "reveal_ts": src["reveal_ts"]})
    except LaneDHouseError as e:
        raise JobError(f"{path}: {e}") from None
    return {"schema": HOUSE_SOURCE_SCHEMA, "king_pointer": king, **house}


def house_pointer(args) -> dict:
    from ..validator.lane_d_house import (
        HOUSE_SCHEMA,
        LaneDHouseError,
        canonical_body,
        load_house_pointer,
        sign_house_pointer,
        verify_house_pointer,
    )

    king = _field(_read_json(args.king_source, "king source"), args.king_field)
    if not isinstance(king, str):
        raise JobError(f"the king source {args.king_field!r} is {type(king).__name__}, not a str")
    pointer_path = Path(args.pointer)
    sidecar_path = pointer_path.with_name(pointer_path.name + SIDECAR_SUFFIX)
    current = None
    if pointer_path.exists():
        try:
            current = load_house_pointer(pointer_path, signer_ss58=args.signer_ss58)
        except LaneDHouseError as e:
            raise JobError(f"the installed house pointer is refused ({e}); fix it by hand "
                           "first, the job never overwrites a pointer it cannot verify") from None
    sidecar = _read_json(sidecar_path, "house sidecar") if sidecar_path.exists() else None
    if not king:
        log.info("lane_d_ops_house_no_lane_a_king lane A has no king; the house stays %s",
              (current or {}).get("digest", "unset"))
        return {"job": "house-pointer", "action": "kept", "king_pointer": ""}
    if current is not None and isinstance(sidecar, dict) and sidecar.get("king_pointer") == king \
            and sidecar.get("digest") == current["digest"] \
            and sidecar.get("reveal_ts") == current["reveal_ts"]:
        log.info("lane_d_ops_house_current house %s (T=%d) is current for lane A king %s",
              current["digest"], current["reveal_ts"], king)
        return {"job": "house-pointer", "action": "current", "king_pointer": king, **current}
    src = _house_source(args.house_source, king)
    house = {"digest": src["digest"], "reveal_ts": src["reveal_ts"]}
    payload = {"schema": HOUSE_SCHEMA, **house, "signer": args.signer_ss58}
    out = {"job": "house-pointer", "king_pointer": king, "previous": current, "house": house,
           "payload": payload, "signed_bytes": canonical_body(payload).decode("ascii")}
    if args.dry_run:
        log.info("lane_d_ops_house_resign_planned would sign house %s (T=%d) for lane A king %s",
              house["digest"], house["reveal_ts"], king)
        return {**out, "action": "planned"}
    if current == house:
        pointer_doc = None                         # same house: only the sidecar is missing
    else:
        from ..shared.chain import open_wallet

        wallet = open_wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
        pointer_doc = sign_house_pointer(digest=house["digest"], reveal_ts=house["reveal_ts"],
                                         signer=wallet)
        if pointer_doc["signer"] != args.signer_ss58:
            raise JobError(f"the wallet hotkey {pointer_doc['signer']} is not the pinned house "
                           f"signer {args.signer_ss58}; nothing written")
        verify_house_pointer(pointer_doc, signer_ss58=args.signer_ss58)
        _write_atomic(pointer_path, (json.dumps(pointer_doc, sort_keys=True) + "\n").encode())
    _write_atomic(sidecar_path, (json.dumps(src, sort_keys=True) + "\n").encode())
    log.warning("lane_d_ops_house_resigned house %s (T=%d) signed for lane A king %s",
          house["digest"], house["reveal_ts"], king)
    return {**out, "action": "signed" if pointer_doc else "sidecar_written"}


# ── evidence (readiness / T7) ────────────────────────────────────────────────


def _armed(cfg, readiness: bytes | None, t7: bytes | None) -> list[int]:
    from ..eval.lane_d_scorer import armed_record

    rec = armed_record(cfg.lane_d.contexts, readiness=readiness, t7=t7,
                       min_long_series=cfg.lane_d.min_long_series)
    return list(rec["armed_contexts"])


def _bytes_or_none(path: Path | None) -> bytes | None:
    if path is None or not Path(path).exists():
        return None
    return Path(path).read_bytes()


def evidence(args) -> dict:
    from ..shared.config import load_chain_config

    if args.readiness is None and args.t7 is None:
        raise JobError("give --readiness and/or --t7 (the new documents)")
    cfg = load_chain_config(args.chain_toml)
    pinned_r, pinned_t = _bytes_or_none(args.pinned_readiness), _bytes_or_none(args.pinned_t7)
    new_r = Path(args.readiness).read_bytes() if args.readiness else None
    new_t = Path(args.t7).read_bytes() if args.t7 else None
    if new_t is not None:
        try:
            src = json.loads(new_t.decode("utf-8")).get("source")
        except (UnicodeDecodeError, ValueError, AttributeError):
            src = None
        if src != "pod":
            raise JobError(f"T7 {args.t7}: source {src!r} is not pod evidence (the T7 run is an "
                           "owner step on the eval host)")
    try:
        before = _armed(cfg, pinned_r, pinned_t) if pinned_r is not None or pinned_t is not None \
            else []
    except ValueError as e:
        before = []
        log.warning("lane_d_ops_evidence_refused the PINNED documents do not parse (%s); "
                    "replacing them", e)
    want_r = new_r if new_r is not None else pinned_r
    want_t = new_t if new_t is not None else pinned_t
    try:
        after = _armed(cfg, want_r, want_t)
    except ValueError as e:
        raise JobError(f"the new document does not parse: {e}") from None
    changes = [(p, b) for p, b, old in ((args.pinned_readiness, new_r, pinned_r),
                                       (args.pinned_t7, new_t, pinned_t))
               if b is not None and b != old]
    out = {"job": "evidence", "armed_before": before, "armed_after": after,
           "install": [str(p) for p, _ in changes],
           "sha256": {str(p): hashlib.sha256(b).hexdigest() for p, b in changes}}
    if not changes:
        log.info("lane_d_ops_evidence_current pinned documents already current; armed %s", after)
        return {**out, "action": "current"}
    lost = sorted(set(before) - set(after))
    if (lost or not after) and not args.allow_disarm:
        raise JobError(f"installing would disarm {lost or 'every context'} (armed {before} -> "
                       f"{after}); pass --allow-disarm to install anyway")
    if args.dry_run:
        log.info("lane_d_ops_evidence_planned would install %s; armed %s -> %s",
              out["install"], before, after)
        return {**out, "action": "planned"}
    for path, data in changes:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_name(f".{Path(path).name}.tmp-{os.getpid()}")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    log.warning("lane_d_ops_evidence_installed installed %s; armed %s -> %s (effective next "
          "settlement)", out["install"], before, after)
    return {**out, "action": "installed"}


# ── vault-ledger ─────────────────────────────────────────────────────────────


def _reports(state_path: Path) -> list[dict]:
    from ..validator.lane_d_state import persisted_report_boundaries, report_path

    return [_read_json(report_path(state_path, b), f"report {b}")
            for b in persisted_report_boundaries(state_path)]


def _state(state_path: Path) -> dict:
    from ..validator.lane_d_state import LaneDStateError, load_state

    if not Path(state_path).is_file():
        raise JobError(f"{state_path} is absent: lane D has not settled here")
    try:
        return load_state(state_path, {})
    except LaneDStateError as e:
        raise JobError(f"{state_path}: {e}") from None


def vault_ledger(args) -> dict:
    from ..shared.config import load_chain_config
    from ..validator.lane_d_state import persisted_report_boundaries
    from ..validator.lane_d_status import build_vault_ledger

    cfg = load_chain_config(args.chain_toml)
    state = _state(args.state)
    reports = _reports(args.state)
    block = args.block
    if block is None:
        bounds = persisted_report_boundaries(args.state)
        if not bounds:
            raise JobError("no persisted report: pass --block")
        block = bounds[-1]           # the last settled boundary: never ahead of the chain
    disputes = _read_json(args.disputes, "disputes") if args.disputes else []
    if not isinstance(disputes, list):
        raise JobError(f"{args.disputes} is not a JSON list of digests")
    try:
        ledger = build_vault_ledger(cfg, state=state, reports=reports, block=int(block),
                                    disputes=disputes)
    except ValueError as e:
        raise JobError(str(e)) from None
    data = (json.dumps(ledger, sort_keys=True, indent=1) + "\n").encode()
    out = {"job": "vault-ledger", "era": ledger["era"], "entries": len(ledger["entries"]),
           "disputes": len(ledger["disputes"]), "out": str(args.out)}
    old = _bytes_or_none(args.out)
    if old == data:
        log.info("lane_d_ops_ledger_current %s is current (era %d)", args.out, ledger["era"])
        return {**out, "action": "current"}
    if args.dry_run:
        log.info("lane_d_ops_ledger_planned would write %s (era %d, %d entries)", args.out,
              ledger["era"], len(ledger["entries"]))
        return {**out, "action": "planned", "ledger": ledger}
    _write_atomic(Path(args.out), data)
    log.info("lane_d_ops_ledger_written wrote %s (era %d, %d entries)", args.out, ledger["era"],
          len(ledger["entries"]))
    return {**out, "action": "written"}


# ── king-check ───────────────────────────────────────────────────────────────


def _mirrors(specs: list[str]) -> dict:
    from .publish import MIRRORS
    from .storage import FilesystemStorage

    class FsMirror(FilesystemStorage):
        """A filesystem mirror copy: the storage adapter plus ``delete``."""

        def delete(self, key: str) -> None:
            self._path(key).unlink(missing_ok=True)

    out = {}
    for spec in specs:
        name, sep, root = spec.partition("=")
        if not sep or name not in MIRRORS or not root:
            raise JobError(f"--mirror {spec!r}: want NAME=DIR, NAME one of {MIRRORS}")
        out[name] = FsMirror(root)
    return out


def king_check(args) -> dict:
    from ..validator.lane_d_report import verify_report
    from .publish import check_one_digest, verify_index
    from .storage import FilesystemStorage

    index = _read_json(args.index, "lane D index") if Path(args.index).exists() else None
    if index is not None and not verify_index(index, args.signer_ss58):
        raise JobError(f"{args.index} does not verify under {args.signer_ss58}")
    mirrors = _mirrors(args.mirror)
    vault = FilesystemStorage(args.vault_dir) if args.vault_dir else None
    full = vault is not None and bool(mirrors)
    problems: dict[str, list[str]] = {}
    crowned = 0
    for report in _reports(args.state):
        if report.get("crowned") is None:
            continue
        crowned += 1
        b = report.get("boundary")
        if not verify_report(report, args.signer_ss58):
            problems[f"{b}"] = ["report_unverified"]
            continue
        if index is None:
            problems[f"{b}"] = ["index_missing"]
            continue
        if full:
            found = check_one_digest(index=index, report=report, vault=vault, mirrors=mirrors)
        else:                                       # index-only check (no local copies)
            entry = next((k for k in index.get("kings") or ()
                          if isinstance(k, dict) and k.get("digest") == report["crowned"]), None)
            found = (["index_missing"] if entry is None else
                     ["crown_block_mismatch"] if entry.get("crown_block") != b else [])
        if found:
            problems[f"{b}"] = found
    out = {"job": "king-check", "crowned_reports": crowned, "scope": "full" if full else "index",
           "problems": problems}
    if problems:
        log.critical("lane_d_ops_king_unpublished %d crowned settlement(s) disagree with the "
              "published index/mirrors: %s", len(problems), problems)
        raise _CheckFailed(out)
    log.info("lane_d_ops_king_ok %d crowned settlement(s) agree with the index (%s check)",
          crowned, out["scope"])
    return out


class _CheckFailed(Exception):
    def __init__(self, out: dict):
        super().__init__("check failed")
        self.out = out


# ── house-source (D46 / D47, produced on a pod) ──────────────────────────────


def house_source(args) -> dict:
    from .house_source import HouseSourceError, check_house_documents, provenance_path

    src = _read_json(args.source, "house source")
    prov = _read_json(provenance_path(args.source), "house provenance")
    try:
        checked = check_house_documents(src, prov)
    except HouseSourceError as e:
        raise JobError(f"{args.source}: {e}") from None
    dest = Path(args.dest)
    data = (json.dumps(checked, sort_keys=True) + "\n").encode()
    pdata = (json.dumps(prov, sort_keys=True, indent=1) + "\n").encode()
    out = {"job": "house-source", "king_pointer": checked["king_pointer"],
           "digest": checked["digest"], "reveal_ts": checked["reveal_ts"],
           "reveal_block": prov["reveal_block"], "dest": str(dest)}
    if _bytes_or_none(dest) == data and _bytes_or_none(provenance_path(dest)) == pdata:
        log.info("lane_d_ops_house_source_current %s is current (house %s for %s)", dest,
                 checked["digest"], checked["king_pointer"])
        return {**out, "action": "current"}
    if args.dry_run:
        log.info("lane_d_ops_house_source_planned would install house %s (T=%d) for %s at %s",
                 checked["digest"], checked["reveal_ts"], checked["king_pointer"], dest)
        return {**out, "action": "planned"}
    _write_atomic(provenance_path(dest), pdata)
    _write_atomic(dest, data)
    log.warning("lane_d_ops_house_source_installed house %s (T=%d) for %s at %s",
                checked["digest"], checked["reveal_ts"], checked["king_pointer"], dest)
    return {**out, "action": "installed"}


# ── king-publish / takedown (W13) ────────────────────────────────────────────


def _local_index(path: Path, signer_ss58: str) -> dict | None:
    from .publish import verify_index

    if not Path(path).exists():
        return None
    index = _read_json(path, "lane D index")
    if not verify_index(index, signer_ss58):
        raise JobError(f"{path} does not verify under {signer_ss58}; the job never re-signs "
                       "over an index it cannot verify")
    return index


def _licence(intake_dir: Path, digest: str, hotkey: str) -> str:
    from .service import LICENCES, QUEUE_FILENAME

    doc = _read_json(Path(intake_dir) / QUEUE_FILENAME, "intake queue")
    rows = [r for r in (doc.get("entries") if isinstance(doc, dict) else None) or []
            if isinstance(r, dict) and r.get("digest") == digest and r.get("hotkey") == hotkey]
    lic = rows[0].get("licence") if rows else None
    if not isinstance(lic, str) or lic not in LICENCES:
        raise JobBlocked(f"king {digest[:16]}: the intake has no ready licence record for it "
                         f"(want one of {list(LICENCES)}); publish it by hand after checking")
    return lic


def _request(state_path: Path, boundary: int, digest: str) -> dict | None:
    from .publish import PUBLISH_REQUEST_SCHEMA, publish_request_path

    path = publish_request_path(state_path, boundary)
    if not path.exists():
        return None
    req = _read_json(path, "publish request")
    if not isinstance(req, dict) or req.get("schema") != PUBLISH_REQUEST_SCHEMA \
            or req.get("boundary") != boundary or req.get("digest") != digest:
        raise JobError(f"{path} is not the {PUBLISH_REQUEST_SCHEMA} of {boundary}/{digest[:16]}")
    return req


def _fs_or_live(args, cfg_needed: bool):
    from .storage import FilesystemStorage

    cfg = None
    if cfg_needed:
        if args.chain_toml is None:
            raise JobError("live publication needs --chain-toml (the [storage] names)")
        from ..shared.config import load_chain_config

        cfg = load_chain_config(args.chain_toml)
    vault = FilesystemStorage(args.vault_dir) if args.vault_dir else None
    return cfg, vault


def _mirrors_for(args, cfg, king: dict) -> dict:
    if args.mirror:
        return _mirrors(args.mirror)
    from .mirrors import open_live_mirrors

    return open_live_mirrors(cfg, context=int(king["context"]), hotkey=str(king["hotkey"]),
                             digest=str(king["digest"]))


def _index_store(args, cfg):
    from .storage import FilesystemStorage

    if args.index_store_dir:
        return FilesystemStorage(args.index_store_dir)
    from .mirrors import open_index_store

    return open_index_store(cfg)


def _signer(args):
    from ..shared.chain import open_wallet

    wallet = open_wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
    key = getattr(wallet, "hotkey", wallet)
    if str(key.ss58_address) != args.signer_ss58:
        raise JobError(f"the wallet hotkey {key.ss58_address} is not the pinned index signer "
                       f"{args.signer_ss58}; nothing written")
    return wallet


def _write_index(args, index: dict, cfg) -> None:
    from .publish import index_bytes, publish_index

    publish_index(index, _index_store(args, cfg), args.signer_ss58)
    _write_atomic(Path(args.index), index_bytes(index))


def king_publish(args) -> dict:
    from ..validator.lane_d_report import verify_report
    from .publish import PublishError, build_index, king_entry, publish_king, sentinel_entry

    index = _local_index(args.index, args.signer_ss58)
    have = {k["digest"]: k for k in (index or {}).get("kings") or ()}
    planned: list[dict] = []
    for report in _reports(args.state):
        crowned, b = report.get("crowned"), report.get("boundary")
        if crowned is None:
            continue
        if not verify_report(report, args.signer_ss58):
            raise JobError(f"report {b} does not verify under {args.signer_ss58}")
        if crowned in have:
            if have[crowned].get("crown_block") != b:
                raise JobError(f"king {crowned[:16]} is indexed at crown block "
                               f"{have[crowned].get('crown_block')}, the report says {b}")
            continue
        req = _request(args.state, b, crowned)
        try:
            sentinel = sentinel_entry(report, None if req is None else req.get("sentinel"))
        except PublishError as e:
            raise JobBlocked(f"settlement {b}: {e} (the settlement step's publish request "
                             "carries it; re-run after the sentinel record is in place)") from None
        hotkey = next((e.get("hotkey") for e in report.get("entries") or ()
                       if isinstance(e, dict) and e.get("digest") == crowned), None)
        licence = _licence(args.intake_state_dir, crowned, str(hotkey))
        try:
            planned.append(king_entry(report, licence=licence, sentinel=sentinel))
        except PublishError as e:
            raise JobError(f"settlement {b}: {e}") from None
    out = {"job": "king-publish", "indexed": len(have),
           "planned": [{"digest": k["digest"], "crown_block": k["crown_block"],
                        "hotkey": k["hotkey"], "context": k["context"]} for k in planned]}
    if not planned:
        log.info("lane_d_ops_king_publish_current every crowned settlement is in %s (%d kings)",
                 args.index, len(have))
        return {**out, "action": "current"}
    if args.dry_run:
        log.info("lane_d_ops_king_publish_planned would publish %d king(s): %s", len(planned),
                 [k["digest"][:16] for k in planned])
        return {**out, "action": "planned", "entries": planned}
    live = not args.mirror or not args.vault_dir or not args.index_store_dir
    cfg, vault = _fs_or_live(args, cfg_needed=live)
    if vault is None:
        from .mirrors import open_live_vault

        vault = open_live_vault(cfg)
    signer = _signer(args)

    def alert(doc: dict) -> None:
        log.error("lane_d_ops_king_publish_failed mirror alert %s", json.dumps(doc, sort_keys=True))

    published = []
    for king in planned:
        try:
            res = publish_king(hotkey=king["hotkey"], digest=king["digest"], vault=vault,
                               mirrors=_mirrors_for(args, cfg, king), alert=alert)
        except PublishError as e:
            raise JobError(f"king {king['digest'][:16]}: {e}") from None
        published.append(res)
    kings = [*(index or {}).get("kings", ()), *planned]
    new_index = build_index(kings, signer)
    _write_index(args, new_index, cfg)
    log.warning("lane_d_ops_king_published %d king(s) on the mirrors; index re-signed with %d "
                "king(s)", len(planned), len(new_index["kings"]))
    return {**out, "action": "published", "mirrors": published}


def takedown_job(args) -> dict:
    from .publish import PublishError, takedown

    index = _local_index(args.index, args.signer_ss58)
    if index is None:
        raise JobError(f"{args.index} is absent: nothing is published")
    king = next((k for k in index["kings"] if k["digest"] == args.digest), None)
    if king is None:
        raise JobError(f"{args.digest} is not in the index")
    out = {"job": "takedown", "digest": args.digest, "reason": args.reason, "block": args.block,
           "hotkey": king["hotkey"], "context": king["context"]}
    if isinstance(king.get("takedown"), dict):
        log.info("lane_d_ops_takedown_planned %s is already taken down (%s)", args.digest[:16],
                 king["takedown"])
        return {**out, "action": "current"}
    if args.dry_run:
        log.info("lane_d_ops_takedown_planned would unpublish %s from the Hub and HF (%s, block "
                 "%d)", args.digest[:16], args.reason, args.block)
        return {**out, "action": "planned"}
    cfg, _ = _fs_or_live(args, cfg_needed=not args.mirror or not args.index_store_dir)
    mirrors = _mirrors_for(args, cfg, king)
    try:
        new_index = takedown(index, digest=args.digest, reason=args.reason, block=args.block,
                             mirrors={k: v for k, v in mirrors.items() if k in ("hub", "hf")},
                             signer=_signer(args))
    except PublishError as e:
        raise JobError(str(e)) from None
    _write_index(args, new_index, cfg)
    log.warning("lane_d_ops_takedown_done %s unpublished from the Hub and HF (%s); index "
                "re-signed", args.digest[:16], args.reason)
    return {**out, "action": "taken_down"}


# ── CLI ──────────────────────────────────────────────────────────────────────

def _log_failed(job: str, e: Exception) -> None:
    if job == "house-pointer":
        log.error("lane_d_ops_house_failed %s; nothing written", e)
    elif job == "evidence":
        log.error("lane_d_ops_evidence_refused %s; nothing written", e)
    elif job == "vault-ledger":
        log.error("lane_d_ops_ledger_failed %s; nothing written", e)
    elif job == "house-source":
        log.error("lane_d_ops_house_source_refused %s; nothing installed", e)
    elif job == "king-publish":
        log.error("lane_d_ops_king_publish_failed %s; the index is unchanged", e)
    elif job == "takedown":
        log.error("lane_d_ops_takedown_failed %s; the index is unchanged", e)
    else:
        log.error("lane_d_ops_king_check_failed %s", e)


def _configure_logging(log_file: Path | None) -> None:
    handler: logging.Handler = (logging.FileHandler(log_file) if log_file
                                else logging.StreamHandler(sys.stderr))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    log.handlers[:] = [handler]
    log.setLevel(logging.INFO)
    log.propagate = False


def _run(args) -> int:
    _configure_logging(args.log_file)
    job = args.job
    try:
        out = args.job_func(args)
    except JobBlocked as e:              # house-pointer and king-publish block on an owner step
        if job == "king-publish":
            log.error("lane_d_ops_king_publish_blocked %s", e)
        else:
            log.error("lane_d_ops_house_resign_blocked %s", e)
        print(json.dumps({"job": job, "action": "blocked", "reason": str(e)}, sort_keys=True))
        return EXIT_BLOCKED
    except _CheckFailed as e:
        print(json.dumps({**e.out, "action": "problems"}, sort_keys=True))
        return EXIT_FAIL
    except Exception as e:  # noqa: BLE001 — JobError, a wallet or signing error: exit 1
        _log_failed(job, e)
        print(json.dumps({"job": job, "action": "failed", "reason": str(e)}, sort_keys=True))
        return EXIT_FAIL
    print(json.dumps(out, sort_keys=True, default=str))
    return EXIT_OK


def add_parser(subparsers) -> None:
    p = subparsers.add_parser("jobs", help="lane D recurring ops jobs (idempotent, --dry-run)")
    jobs = p.add_subparsers(dest="job", required=True)

    def common(sp):
        sp.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
        sp.add_argument("--log-file", type=Path, default=None,
                        help="append log lines here (default stderr / the journal)")
        sp.set_defaults(func=_run)

    h = jobs.add_parser("house-pointer", help="re-sign the house pointer on a lane A king change")
    h.add_argument("--king-source", type=Path, required=True,
                   help="local JSON naming the lane A king pointer (status/v2.json)")
    h.add_argument("--king-field", default="king.pointer", help="dotted field (default king.pointer)")
    h.add_argument("--pointer", type=Path, required=True,
                   help="the house pointer file the --lane-d-inputs TOML names (house_pointer)")
    h.add_argument("--signer-ss58", required=True, help="the pinned house_signer_ss58")
    h.add_argument("--house-source", type=Path, default=None,
                   help=f"the {HOUSE_SOURCE_SCHEMA} JSON prepared for the current lane A king")
    h.add_argument("--wallet-name", default=None)
    h.add_argument("--wallet-hotkey", default=None)
    h.add_argument("--wallet-path", default=None)
    common(h)
    h.set_defaults(job_func=house_pointer)

    e = jobs.add_parser("evidence", help="validate and install laned_readiness/1 / laned_t7/1")
    e.add_argument("--chain-toml", type=Path, required=True)
    e.add_argument("--readiness", type=Path, default=None, help="new laned_readiness/1 (W1)")
    e.add_argument("--t7", type=Path, default=None, help="new laned_t7/1 (pod run)")
    e.add_argument("--pinned-readiness", type=Path, required=True,
                   help="the inputs TOML 'readiness' path")
    e.add_argument("--pinned-t7", type=Path, required=True, help="the inputs TOML 't7' path")
    e.add_argument("--allow-disarm", action="store_true",
                   help="install even if a currently armed context disarms")
    common(e)
    e.set_defaults(job_func=evidence)

    v = jobs.add_parser("vault-ledger", help="export laned_vault_ledger/1 for the W18 sweeper")
    v.add_argument("--chain-toml", type=Path, required=True)
    v.add_argument("--state", type=Path, required=True, help="lane_d_state.json")
    v.add_argument("--out", type=Path, required=True, help="the ledger the sweeper reads")
    v.add_argument("--block", type=int, default=None,
                   help="the block the era is taken at (default: the last persisted boundary)")
    v.add_argument("--disputes", type=Path, default=None, help="JSON list of disputed digests")
    common(v)
    v.set_defaults(job_func=vault_ledger)

    k = jobs.add_parser("king-check", help="crowned reports vs the signed index (and mirrors)")
    k.add_argument("--state", type=Path, required=True, help="lane_d_state.json")
    k.add_argument("--index", type=Path, required=True, help="local champions/lane_d/index.json")
    k.add_argument("--signer-ss58", required=True, help="the pinned report / index signer")
    k.add_argument("--vault-dir", type=Path, default=None, help="local vault copy (full check)")
    k.add_argument("--mirror", action="append", default=[], help="NAME=DIR local mirror copy")
    common(k)
    k.set_defaults(job_func=king_check)

    s = jobs.add_parser("house-source", help="verify and install a pod-built laned_house_source/1")
    s.add_argument("--source", type=Path, required=True,
                   help="the produced laned_house_source/1 (its .provenance.json beside it)")
    s.add_argument("--dest", type=Path, required=True,
                   help="where house-pointer reads it (--house-source)")
    common(s)
    s.set_defaults(job_func=house_source)

    def publication(sp, *, index_help: str):
        sp.add_argument("--index", type=Path, required=True, help=index_help)
        sp.add_argument("--signer-ss58", required=True, help="the pinned report / index signer")
        sp.add_argument("--chain-toml", type=Path, default=None,
                        help="the deployed chain.toml ([storage] names for the live clients)")
        sp.add_argument("--vault-dir", type=Path, default=None,
                        help="filesystem vault (testnet / tests); default the live vault")
        sp.add_argument("--mirror", action="append", default=[],
                        help="NAME=DIR filesystem mirror (testnet / tests); default the live "
                             "Hub, R2 and HF mirrors")
        sp.add_argument("--index-store-dir", type=Path, default=None,
                        help="filesystem public store for the index; default the manifest store")
        sp.add_argument("--wallet-name", default=None)
        sp.add_argument("--wallet-hotkey", default=None)
        sp.add_argument("--wallet-path", default=None)
        common(sp)

    kp = jobs.add_parser("king-publish", help="publish crowned kings and re-sign the index")
    kp.add_argument("--state", type=Path, required=True, help="lane_d_state.json")
    kp.add_argument("--intake-state-dir", type=Path, required=True,
                    help="the intake state dir (the ready licence record)")
    publication(kp, index_help="the local champions/lane_d/index.json copy king-check reads")
    kp.set_defaults(job_func=king_publish)

    td = jobs.add_parser("takedown", help="B13: unpublish one king from the Hub and HF")
    td.add_argument("--digest", required=True)
    td.add_argument("--reason", required=True, choices=("licence", "rights"))
    td.add_argument("--block", type=int, required=True, help="the block of the claim")
    publication(td, index_help="the local champions/lane_d/index.json copy")
    td.set_defaults(job_func=takedown_job)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -P -m cascade.lane_d.ops_jobs")
    sub = parser.add_subparsers(dest="command", required=True)
    add_parser(sub)
    args = parser.parse_args(["jobs", *(sys.argv[1:] if argv is None else argv)])
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
