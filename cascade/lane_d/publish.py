"""Lane D king publication (LANE_D_PLAN §11.1, W13).

A crowned lane D king leaves the private vault for three public mirrors (the
Hippius Hub, the R2 public bucket and the HF dataset
:data:`HF_MIRROR_REPO`), is listed in the signed
``champions/lane_d/index.json`` and gets repo cards
``champions/lane_d/<digest>/{PROVENANCE.json, MODEL_CARD.md}``.

Immutability (rev6.4 §1.2): Hippius offers no write-once storage mode, so
nothing here asks a mirror for one and nothing published claims one. A
king's bytes are pinned by

* the content-addressed path: every mirror key carries the digest as one
  ``/``-separated component (``lane-d-kings/<digest>/<file>``);
* the re-hash on every read (:func:`verify_mirrors`): ``manifest.json`` must
  hash to the digest in the path and every file to its manifest sha256, so an
  overwrite in place, even a consistent rewrite of weights and manifest,
  reads as ``digest_mismatch``;
* the signed digest in the index, and the three independent mirrors.

A mirror client is any object with ``put(key, data)``, ``get(key)`` (raises
``KeyError`` when absent), ``list(prefix)`` (sorted keys), ``size(key)`` and
``delete(key)``. ``put`` takes a key and bytes and nothing else.

Takedown (rev5, B13): a licence or rights claim removes the Hub and HF copies
(:func:`takedown`); the vault, the R2 copy, the digest and the verdict record
stay, and the re-signed index marks the king ``takedown``.

Torch-free: stdlib plus the lane D intake helpers and the sr25519 verifier.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from .intake import (
    MANIFEST_OBJECT,
    VAULT_PREFIX,
    IntakeError,
    entry_prefix,
    manifest_digest,
)

MIRRORS = ("hub", "r2", "hf")
HF_MIRROR_REPO = "tensorlink-dev/cascade-lane-d-kings"
# Mirror key layout: ``<KINGS_PREFIX>/<digest>/<file>``.
KINGS_PREFIX = "lane-d-kings"
# The signed index: its path in the repo and its key in the public bucket.
INDEX_KEY = "champions/lane_d/index.json"
INDEX_SCHEMA = "lane-d-index/1"
CARDS_DIR = ("champions", "lane_d")
PROVENANCE_FILE = "PROVENANCE.json"
MODEL_CARD_FILE = "MODEL_CARD.md"
# §11.1: the fields every index entry carries, unchanged from the crowning.
KING_FIELDS = ("hotkey", "context", "digest", "crown_block", "lcb", "gate", "sentinel", "license")
# B13: the claims a takedown answers, and the mirrors it unpublishes.
TAKEDOWN_REASONS = ("licence", "rights")
TAKEDOWN_MIRRORS = ("hub", "hf")

OK, MISSING, DIGEST_MISMATCH = "ok", "missing", "digest_mismatch"

PUBLISH_CODES = (
    "mirror_missing", "mirror_digest_mismatch", "vault_digest_mismatch",
    "index_field_missing", "index_field_invalid", "index_duplicate_digest",
    "index_signature_invalid", "unknown_king", "takedown_reason_invalid",
    "takedown_incomplete",
)

_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_SIGNATURE_RE = re.compile(r"[0-9a-f]{128}")


class PublishError(ValueError):
    """A refused publication step; ``code`` is one of :data:`PUBLISH_CODES`."""

    def __init__(self, code: str, detail: str = ""):
        if code not in PUBLISH_CODES:
            raise ValueError(f"unknown publish code {code!r}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def hub_repo(ns: str, context: int, hotkey: str) -> str:
    """§11.1: the public Hub repository of a king."""
    return f"{ns}/laned-king-c{int(context)}-{str(hotkey)[:8]}"


def mirror_prefix(digest: str) -> str:
    """The mirror prefix of one king (trailing ``/``)."""
    if not isinstance(digest, str) or not _DIGEST_RE.fullmatch(digest):
        raise PublishError("index_field_invalid", f"digest {digest!r} is not 64 lowercase hex")
    return f"{KINGS_PREFIX}/{digest}/"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> bytes:
    """``sort_keys``, compact separators, ASCII, no NaN (``ValueError``)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


# ── the vault entry and its re-hash ──────────────────────────────────────────


def _vault_files(vault, hotkey: str, digest: str) -> tuple[bytes, dict, dict[str, bytes]]:
    """``(manifest bytes, manifest, {name: bytes})`` of a vault entry whose
    manifest hashes to ``digest`` and whose every file hashes to its manifest
    sha256 and size; ``PublishError("vault_digest_mismatch")`` otherwise."""
    try:
        prefix = entry_prefix(hotkey, digest)
    except IntakeError as e:
        raise PublishError("vault_digest_mismatch", str(e)) from None
    try:
        raw = bytes(vault.get(prefix + MANIFEST_OBJECT))
        manifest = json.loads(raw.decode("utf-8"))
        if manifest_digest(manifest) != digest:
            raise PublishError("vault_digest_mismatch",
                               f"{prefix}{MANIFEST_OBJECT} does not hash to {digest}")
    except KeyError:
        raise PublishError("vault_digest_mismatch", f"no {prefix}{MANIFEST_OBJECT}") from None
    except (UnicodeDecodeError, ValueError) as e:
        if isinstance(e, PublishError):
            raise
        raise PublishError("vault_digest_mismatch", f"unreadable manifest: {e}") from None
    files: dict[str, bytes] = {}
    for name, meta in sorted(manifest["files"].items()):
        try:
            data = bytes(vault.get(prefix + name))
        except KeyError:
            raise PublishError("vault_digest_mismatch", f"no {prefix}{name}") from None
        if _sha256(data) != str(meta["sha256"]) or len(data) != int(meta["size"]):
            raise PublishError("vault_digest_mismatch", f"{prefix}{name} fails its re-hash")
        files[name] = data
    return raw, manifest, files


def _vault_hotkey(vault, digest: str) -> str | None:
    """The hotkey whose vault entry is ``digest`` (``lane-d/<hk>/<digest>/``)."""
    for key in vault.list(f"{VAULT_PREFIX}/"):
        parts = key.split("/")
        if len(parts) >= 4 and parts[2] == digest:
            return parts[1]
    return None


# ── mirrors ──────────────────────────────────────────────────────────────────


def _verify_one(digest: str, mirror) -> str:
    """One mirror's verdict on ``digest``, re-read and re-hashed now."""
    prefix = mirror_prefix(digest)
    try:
        raw = bytes(mirror.get(prefix + MANIFEST_OBJECT))
    except KeyError:
        return MISSING
    try:
        manifest = json.loads(raw.decode("utf-8"))
        if manifest_digest(manifest) != digest:
            return DIGEST_MISMATCH
        files = manifest["files"]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError):
        return DIGEST_MISMATCH
    for name, meta in sorted(files.items()):
        try:
            data = bytes(mirror.get(prefix + name))
        except KeyError:
            return MISSING
        if _sha256(data) != str(meta["sha256"]) or len(data) != int(meta["size"]):
            return DIGEST_MISMATCH
    return OK


def verify_mirrors(digest: str, mirrors: Mapping) -> dict[str, str]:
    """``name -> "ok" | "missing" | "digest_mismatch"`` for every given
    mirror, re-reading and re-hashing the bytes on each call."""
    return {name: _verify_one(digest, client) for name, client in mirrors.items()}


def publish_king(*, hotkey: str, digest: str, vault, mirrors: Mapping,
                 alert: Callable[[dict], object]) -> dict:
    """Copy the vault entry ``lane-d/<hotkey>/<digest>/`` to the Hub, R2 and
    HF mirrors under ``lane-d-kings/<digest>/`` and read every mirror back.

    The vault entry is re-hashed first (``vault_digest_mismatch``, alerted,
    nothing written); all three mirrors are required (``mirror_missing``,
    nothing written); a mirror that reads back other bytes is alerted and
    refused (``mirror_digest_mismatch``). A mirror that already holds the
    king intact is not rewritten. The vault is only read."""
    absent = [name for name in MIRRORS if name not in mirrors]
    if absent:
        raise PublishError("mirror_missing", f"publishing needs every mirror; missing {absent}")
    try:
        raw_manifest, manifest, files = _vault_files(vault, hotkey, digest)
    except PublishError as e:
        alert({"code": e.code, "digest": digest, "hotkey": hotkey, "detail": e.detail})
        raise
    prefix = mirror_prefix(digest)
    for name in MIRRORS:
        client = mirrors[name]
        if _verify_one(digest, client) == OK:
            continue
        for fname in sorted(files):
            client.put(prefix + fname, files[fname])
        client.put(prefix + MANIFEST_OBJECT, raw_manifest)
    verdicts = {name: _verify_one(digest, mirrors[name]) for name in MIRRORS}
    bad = [name for name, v in verdicts.items() if v != OK]
    for name in bad:
        alert({"code": "mirror_digest_mismatch", "mirror": name, "digest": digest,
               "hotkey": hotkey, "verdict": verdicts[name]})
    if bad:
        raise PublishError("mirror_digest_mismatch",
                           f"read-back of {digest} failed on {bad}: {verdicts}")
    return {"digest": digest, "mirrors": verdicts, "files": len(manifest["files"])}


# ── the signed index ─────────────────────────────────────────────────────────


def _int(value, field: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PublishError("index_field_invalid", f"{field} {value!r} is not an int >= {minimum}")
    return value


def _check_king(king) -> dict:
    """A deep copy of one index entry, checked (§11.1 fields present and
    well formed)."""
    if not isinstance(king, Mapping):
        raise PublishError("index_field_invalid", f"a king entry is {type(king).__name__}")
    missing = [f for f in KING_FIELDS if f not in king]
    if missing:
        raise PublishError("index_field_missing", f"king entry lacks {missing}")
    entry = copy.deepcopy(dict(king))
    if not isinstance(entry["hotkey"], str) or not entry["hotkey"] or "/" in entry["hotkey"]:
        raise PublishError("index_field_invalid", f"hotkey {entry['hotkey']!r}")
    if not isinstance(entry["digest"], str) or not _DIGEST_RE.fullmatch(entry["digest"]):
        raise PublishError("index_field_invalid",
                           f"digest {entry['digest']!r} is not 64 lowercase hex")
    _int(entry["context"], "context", minimum=1)
    _int(entry["crown_block"], "crown_block", minimum=0)
    lcb = entry["lcb"]
    if isinstance(lcb, bool) or not isinstance(lcb, int | float) or not math.isfinite(lcb):
        raise PublishError("index_field_invalid", f"lcb {lcb!r} is not a finite number")
    if not isinstance(entry["gate"], Mapping):
        raise PublishError("index_field_invalid", "gate is not an object")
    sentinel = entry["sentinel"]
    if not isinstance(sentinel, Mapping):
        raise PublishError("index_field_invalid", "sentinel is not an object")
    if "suite" not in sentinel:
        raise PublishError("index_field_missing", "sentinel lacks its suite")
    if not isinstance(sentinel["suite"], str) or not sentinel["suite"]:
        raise PublishError("index_field_invalid", f"sentinel suite {sentinel['suite']!r}")
    if not isinstance(entry["license"], str) or not entry["license"]:
        raise PublishError("index_field_invalid", f"license {entry['license']!r}")
    if "takedown" in entry:
        td = entry["takedown"]
        if (not isinstance(td, Mapping) or td.get("reason") not in TAKEDOWN_REASONS
                or isinstance(td.get("block"), bool) or not isinstance(td.get("block"), int)):
            raise PublishError("index_field_invalid", f"takedown {td!r}")
    return entry


def _checked_kings(kings) -> list[dict]:
    if not isinstance(kings, list | tuple):
        raise PublishError("index_field_invalid", "kings is not a list")
    entries = [_check_king(k) for k in kings]
    seen: set[str] = set()
    for e in entries:
        if e["digest"] in seen:
            raise PublishError("index_duplicate_digest",
                               f"{e['digest']} appears twice (one digest, one identity)")
        seen.add(e["digest"])
    entries.sort(key=lambda e: (e["crown_block"], e["digest"]))
    return entries


def _signature_hex(sig) -> str:
    if isinstance(sig, bytes | bytearray):
        out = bytes(sig).hex()
    else:
        text = str(sig)
        out = bytes.fromhex(text[2:] if text[:2] in ("0x", "0X") else text).hex()
    if not _SIGNATURE_RE.fullmatch(out):
        raise ValueError(f"the signer produced {len(out) // 2} signature bytes, want 64 (sr25519)")
    return out


def index_body(index: Mapping) -> bytes:
    """The signed bytes: the index without ``signature`` (``signer`` inside)."""
    return canonical_json({k: v for k, v in index.items() if k != "signature"})


def build_index(kings, signer) -> dict:
    """The signed ``champions/lane_d/index.json`` over ``kings`` (each with
    every :data:`KING_FIELDS` field), ordered by crown block then digest.
    ``signer`` is a keypair (``.ss58_address``, ``.sign``) or a wallet whose
    ``hotkey`` is one. The input is not mutated."""
    key = getattr(signer, "hotkey", signer)
    index = {"schema": INDEX_SCHEMA, "kings": _checked_kings(kings),
             "signer": str(key.ss58_address)}
    index["signature"] = _signature_hex(key.sign(index_body(index)))
    return index


def verify_index(index, signer_ss58: str) -> bool:
    """True only for a well-formed index signed by ``signer_ss58`` (sr25519
    over :func:`index_body`). Any other input is ``False``; a verifier that
    cannot run raises ``SignatureVerifierUnavailable``."""
    if not isinstance(index, Mapping) or not isinstance(signer_ss58, str) or not signer_ss58:
        return False
    sig = index.get("signature")
    if (index.get("schema") != INDEX_SCHEMA or index.get("signer") != signer_ss58
            or not isinstance(sig, str) or not _SIGNATURE_RE.fullmatch(sig)):
        return False
    try:
        _checked_kings(index.get("kings"))
        body = index_body(index)
    except (TypeError, ValueError):
        return False
    from ..shared.sr25519 import verify_hotkey

    return verify_hotkey(signer_ss58, body, sig)


def index_bytes(index: Mapping) -> bytes:
    """The published form of the index (sorted keys, indented, newline)."""
    return (json.dumps(index, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
            + "\n").encode("utf-8")


def publish_index(index: Mapping, store, signer_ss58: str) -> str:
    """Put a verified index at :data:`INDEX_KEY` of ``store`` (the public
    bucket client); returns the key. An index that does not verify under
    ``signer_ss58`` is refused (``index_signature_invalid``)."""
    if not verify_index(index, signer_ss58):
        raise PublishError("index_signature_invalid", f"index does not verify under {signer_ss58}")
    store.put(INDEX_KEY, index_bytes(index))
    return INDEX_KEY


# ── repo cards (laned-king-sync.yml) ─────────────────────────────────────────


def provenance(king: Mapping) -> dict:
    """``PROVENANCE.json`` of one index entry."""
    out = {f: copy.deepcopy(king[f]) for f in KING_FIELDS}
    out["hf_repo"] = HF_MIRROR_REPO
    out["mirror_path"] = mirror_prefix(king["digest"])
    if "takedown" in king:
        out["takedown"] = copy.deepcopy(king["takedown"])
    for extra in ("hub_ref",):
        if extra in king:
            out[extra] = copy.deepcopy(king[extra])
    return out


def model_card(king: Mapping) -> str:
    """``MODEL_CARD.md`` of one index entry."""
    digest = king["digest"]
    sentinel = king["sentinel"]
    lines = [
        f"# cascade lane D king `{digest[:12]}`",
        "",
        f"- Digest: `{digest}`",
        f"- Licence: {king['license']}",
        f"- Hotkey: `{king['hotkey']}`",
        f"- Context: {king['context']}",
        f"- Crown block: {king['crown_block']}",
        f"- LCB: {king['lcb']!r}",
        f"- Sentinel suite: {sentinel['suite']}",
        "",
        "## Where the weights are",
        "",
        f"The checkpoint is not in git. Its files are published under `{mirror_prefix(digest)}`"
        f" on the Hippius Hub, the R2 public mirror and the Hugging Face dataset "
        f"`{HF_MIRROR_REPO}`.",
        "",
        "## Checking a copy",
        "",
        "The digest is the sha256 of the canonical `manifest.json` (sorted keys, compact "
        "separators). A copy is genuine when its `manifest.json` hashes to the digest above "
        "and every file hashes to the sha256 its manifest lists. Check this on every "
        "download; the path names the digest, so any other bytes fail the check.",
        "",
    ]
    td = king.get("takedown")
    if isinstance(td, Mapping):
        lines += [
            "## Takedown",
            "",
            f"Unpublished from the Hub and HF mirrors at block {td['block']} after a "
            f"{td['reason']} claim. The digest and the verdict record stay in the signed index.",
            "",
        ]
    return "\n".join(lines)


def _write_if_changed(path: Path, data: bytes) -> bool:
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.part-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def sync_cards(repo_root, index, signer_ss58: str) -> list[Path]:
    """Write the signed index and every king's cards under
    ``<repo_root>/champions/lane_d/``; returns the paths written (files whose
    bytes changed). ``index is None`` (lane D unarmed, nothing published)
    writes nothing. An index that does not verify under ``signer_ss58`` is
    refused (``index_signature_invalid``) before anything is written."""
    if index is None:
        return []
    if not verify_index(index, signer_ss58):
        raise PublishError("index_signature_invalid", f"index does not verify under {signer_ss58}")
    root = Path(repo_root).joinpath(*CARDS_DIR)
    planned: list[tuple[Path, bytes]] = [(root / "index.json", index_bytes(index))]
    for king in index["kings"]:
        d = root / king["digest"]
        planned.append((d / PROVENANCE_FILE, index_bytes(provenance(king))))
        planned.append((d / MODEL_CARD_FILE, model_card(king).encode("utf-8")))
    return [path for path, data in planned if _write_if_changed(path, data)]


# ── one digest, one identity ─────────────────────────────────────────────────


def check_one_digest(*, index, report, vault, mirrors: Mapping) -> list[str]:
    """§11.1: the report's crowned digest, the index, the vault and the
    mirrors must agree. Read-only (vault and mirror ``get``/``list`` only).
    Returns problem codes; ``[]`` means everything agrees. The report's
    signature is the caller's check (``verify_report``)."""
    crowned = report.get("crowned") if isinstance(report, Mapping) else None
    if crowned is None:
        return ["report_no_crown"]
    problems: list[str] = []
    kings = index.get("kings") if isinstance(index, Mapping) else None
    entry = next((k for k in kings or () if isinstance(k, Mapping) and k.get("digest") == crowned),
                 None)
    if entry is None:
        problems.append("index_missing")
    elif entry.get("crown_block") != report.get("boundary"):
        problems.append("crown_block_mismatch")
    if not isinstance(crowned, str) or not _DIGEST_RE.fullmatch(crowned):
        return [*problems, "vault_digest_mismatch", *(f"{m}_{MISSING}" for m in MIRRORS)]
    hotkey = entry.get("hotkey") if entry is not None else _vault_hotkey(vault, crowned)
    try:
        if not isinstance(hotkey, str):
            raise PublishError("vault_digest_mismatch", f"no vault entry for {crowned}")
        _vault_files(vault, hotkey, crowned)
    except PublishError:
        problems.append("vault_digest_mismatch")
    taken_down = entry is not None and isinstance(entry.get("takedown"), Mapping)
    for name in MIRRORS:
        client = mirrors.get(name)
        verdict = _verify_one(crowned, client) if client is not None else MISSING
        if verdict == MISSING and taken_down and name in TAKEDOWN_MIRRORS:
            continue
        if verdict != OK:
            problems.append(f"{name}_{verdict}")
    return problems


# ── takedown (rev5, B13) ─────────────────────────────────────────────────────


def takedown(index, *, digest: str, reason: str, block: int, mirrors: Mapping, signer) -> dict:
    """Unpublish a king from the Hub and HF mirrors after a licence or rights
    claim and return the re-signed index whose entry gains
    ``"takedown": {"reason", "block"}``. The vault, the R2 copy and every
    §11.1 field stay. The input index is verified under ``signer`` first and
    never mutated; an unknown digest is refused (``unknown_king``) before any
    mirror changes."""
    if not reason:
        raise ValueError("takedown needs a reason ('licence' or 'rights')")
    if reason not in TAKEDOWN_REASONS:
        raise PublishError("takedown_reason_invalid",
                           f"reason {reason!r} is not one of {TAKEDOWN_REASONS}")
    if isinstance(block, bool) or not isinstance(block, int) or block < 0:
        raise PublishError("index_field_invalid", f"takedown block {block!r}")
    key = getattr(signer, "hotkey", signer)
    if not verify_index(index, str(key.ss58_address)):
        raise PublishError("index_signature_invalid",
                           f"the index does not verify under {key.ss58_address}")
    kings = copy.deepcopy(list(index["kings"]))
    entry = next((k for k in kings if k["digest"] == digest), None)
    if entry is None:
        raise PublishError("unknown_king", f"{digest!r} is not in the index")
    absent = [name for name in TAKEDOWN_MIRRORS if name not in mirrors]
    if absent:
        raise PublishError("mirror_missing", f"takedown needs the {absent} mirror(s)")
    prefix = mirror_prefix(digest)
    for name in TAKEDOWN_MIRRORS:
        client = mirrors[name]
        for k in client.list(prefix):
            client.delete(k)
    left = {name: mirrors[name].list(prefix) for name in TAKEDOWN_MIRRORS}
    if any(left.values()):
        raise PublishError("takedown_incomplete", f"objects remain under {prefix}: {left}")
    entry["takedown"] = {"reason": reason, "block": block}
    return build_index(kings, signer)


__all__ = ["HF_MIRROR_REPO", "INDEX_KEY", "INDEX_SCHEMA", "KINGS_PREFIX", "KING_FIELDS",
           "MIRRORS", "PUBLISH_CODES", "PublishError", "TAKEDOWN_MIRRORS", "TAKEDOWN_REASONS",
           "build_index", "check_one_digest", "hub_repo", "index_body", "index_bytes",
           "mirror_prefix", "model_card", "provenance", "publish_index", "publish_king",
           "sync_cards", "takedown", "verify_index", "verify_mirrors"]
