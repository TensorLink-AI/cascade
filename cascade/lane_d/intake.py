"""Lane D intake: the miner's upload and ``ready`` (LANE_D_PLAN §1.2 "Flow").

* :func:`checkpoint_manifest` / :func:`manifest_digest` — a checkpoint is
  identified by the sha256 of its canonical file manifest (the entry digest,
  ``X-Content-Digest``).
* :func:`upload_checkpoint` — the miner's direct upload: every file under
  ``lane-d/<hotkey>/<digest>/<name>`` plus the manifest itself at
  ``lane-d/<hotkey>/<digest>/manifest.json`` (content addressed: the path
  carries the digest, so any later overwrite fails ingest's re-hash).
* :func:`ready` — checks the object list and sizes against the manifest
  WITHOUT downloading the weights, checks the declared context against
  ``[lane_d] contexts``, and records ``coldkey_at_ready`` and ``ready_block``
  (D22 option b: the coldkey is recorded at ready and carried in the signed
  report).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .storage import LaneDStorage

VAULT_PREFIX = "lane-d"
MANIFEST_OBJECT = "manifest.json"
# §1.1 / W4: a lane D checkpoint is exactly these files, nothing else.
ALLOWED_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")


class IntakeError(ValueError):
    """A typed intake refusal: ``code`` is the structured reason."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def entry_prefix(hotkey: str, digest: str) -> str:
    """The vault prefix of one entry (trailing ``/``)."""
    if not hotkey or "/" in hotkey or not digest or "/" in digest:
        raise IntakeError("bad_identity", f"hotkey={hotkey!r} digest={digest!r}")
    return f"{VAULT_PREFIX}/{hotkey}/{digest}/"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_manifest(directory: str | os.PathLike) -> dict:
    """``{"files": {name: {"sha256", "size"}}}`` over every regular file
    below ``directory`` (names are ``/``-relative paths)."""
    root = Path(directory)
    if not root.is_dir():
        raise IntakeError("missing_checkpoint", str(root))
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files[p.relative_to(root).as_posix()] = {"sha256": _sha256_file(p),
                                                     "size": int(p.stat().st_size)}
    if not files:
        raise IntakeError("empty_checkpoint", str(root))
    return {"files": files}


def canonical_manifest(manifest: dict) -> bytes:
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or not files:
        raise IntakeError("bad_manifest", "no files")
    body = {"files": {str(k): {"sha256": str(v["sha256"]), "size": int(v["size"])}
                      for k, v in files.items()}}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def manifest_digest(manifest: dict) -> str:
    """The entry digest: sha256 hex of the canonical manifest."""
    return hashlib.sha256(canonical_manifest(manifest)).hexdigest()


def upload_checkpoint(storage: LaneDStorage, hotkey: str, directory: str | os.PathLike) -> str:
    """Upload a checkpoint directory under its digest; returns the digest."""
    root = Path(directory)
    manifest = checkpoint_manifest(root)
    digest = manifest_digest(manifest)
    prefix = entry_prefix(hotkey, digest)
    for name in manifest["files"]:
        storage.put(prefix + name, (root / name).read_bytes())
    storage.put(prefix + MANIFEST_OBJECT, canonical_manifest(manifest))
    return digest


def read_manifest(storage: LaneDStorage, hotkey: str, digest: str) -> dict:
    """The entry's stored manifest, checked against its digest."""
    prefix = entry_prefix(hotkey, digest)
    try:
        raw = storage.get(prefix + MANIFEST_OBJECT)
    except KeyError as e:
        raise IntakeError("missing_manifest", prefix + MANIFEST_OBJECT) from e
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise IntakeError("bad_manifest", "unparseable") from e
    if manifest_digest(manifest) != digest:
        raise IntakeError("digest_mismatch", "manifest does not hash to the entry digest")
    return manifest


def check_object_list(storage: LaneDStorage, hotkey: str, digest: str, manifest: dict) -> None:
    """Every manifest file present at its size, no file outside the allowlist
    and no stray object under the prefix (sizes only; nothing downloaded)."""
    prefix = entry_prefix(hotkey, digest)
    names = set(manifest["files"])
    if names != set(ALLOWED_FILES):
        raise IntakeError("file_not_allowed",
                          f"extra={sorted(names - set(ALLOWED_FILES))} "
                          f"missing={sorted(set(ALLOWED_FILES) - names)}")
    listed = set(storage.list(prefix))
    stray = sorted(k for k in listed - {prefix + n for n in names} - {prefix + MANIFEST_OBJECT})
    if stray:
        raise IntakeError("stray_object", str(stray[:5]))
    for name, meta in manifest["files"].items():
        try:
            size = storage.size(prefix + name)
        except KeyError as e:
            raise IntakeError("missing_object", prefix + name) from e
        if size != int(meta["size"]):
            raise IntakeError("size_mismatch", f"{name}: {size} != {meta['size']}")


def ready(storage: LaneDStorage, *, hotkey: str, digest: str, context: int, coldkey: str,
          ready_block: int, cfg) -> dict:
    """``/v1/lane-d/ready``: verify the uploaded entry from storage alone and
    return its ready record."""
    if int(context) not in tuple(cfg.contexts):
        raise IntakeError("context_not_armed", f"{context} not in {list(cfg.contexts)}")
    if not coldkey:
        raise IntakeError("coldkey_unresolved", hotkey)
    if int(ready_block) < 0:
        raise IntakeError("bad_ready_block", str(ready_block))
    manifest = read_manifest(storage, hotkey, digest)
    check_object_list(storage, hotkey, digest, manifest)
    return {"hotkey": str(hotkey), "digest": str(digest), "context": int(context),
            "coldkey_at_ready": str(coldkey), "ready_block": int(ready_block),
            "weights_sha256": str(manifest["files"]["weights.safetensors"]["sha256"]),
            "ready_expires_block": int(ready_block) + int(cfg.ready_ttl_blocks)}


__all__ = [
    "ALLOWED_FILES", "IntakeError", "MANIFEST_OBJECT", "VAULT_PREFIX", "canonical_manifest",
    "check_object_list", "checkpoint_manifest", "entry_prefix", "manifest_digest",
    "read_manifest", "ready", "upload_checkpoint",
]
