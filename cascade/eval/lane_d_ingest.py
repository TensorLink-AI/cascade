"""Lane D ingest guards (LANE_D_PLAN §1.1, §1.2; W4 deepens them).

:func:`ingest_entry` takes a revealed entry from the vault to a verified local
checkpoint directory, in this order, failing closed with a typed
:class:`IngestError`:

1. the stored manifest hashes to the entry digest (``digest_mismatch``);
2. the object list is exactly the allowlisted files at their manifest sizes
   (``file_not_allowed`` / ``stray_object`` / ``missing_object``);
3. every fetched object re-hashes to its manifest sha256 (``digest_mismatch``);
4. the checkpoint guards: lane A's code / config / header rules with the
   lane-D-only BF16 dtype path (``checkpoint_tampered``), then
   ``verify_weights_finite`` (``checkpoint_diverged``).

Only after all four does the directory appear at ``dest_dir/<digest>``; a
failed ingest leaves nothing behind. No checkpoint code is ever imported.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from ..lane_d.intake import IntakeError, check_object_list, entry_prefix, read_manifest
from ..lane_d.storage import LaneDStorage

INGEST_CODES = ("digest_mismatch", "file_not_allowed", "stray_object", "missing_object",
                "size_mismatch", "missing_manifest", "bad_manifest", "bad_identity",
                "checkpoint_tampered", "checkpoint_diverged")


class IngestError(RuntimeError):
    """A typed ingest refusal (``code`` in :data:`INGEST_CODES`): the miner's
    fault, recorded as ``checkpoint_fault`` by the settlement."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def ingest_entry(storage: LaneDStorage, entry: dict, dest_dir: str | os.PathLike, *,
                 contract, dtype: str = "BF16") -> Path:
    """Fetch, re-hash and guard ``entry``; returns the verified checkpoint
    directory (it holds ``weights.safetensors``)."""
    from .checkpoint_guard import (
        CheckpointDiverged,
        CheckpointTampered,
        verify_lane_d_checkpoint,
    )

    hotkey, digest = str(entry["hotkey"]), str(entry["digest"])
    try:
        manifest = read_manifest(storage, hotkey, digest)
        check_object_list(storage, hotkey, digest, manifest)
    except IntakeError as e:
        raise IngestError(e.code, e.detail) from e
    prefix = entry_prefix(hotkey, digest)
    dest = Path(dest_dir)
    final = dest / digest
    staging = dest / f".{digest}.staging-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for name, meta in sorted(manifest["files"].items()):
            data = storage.get(prefix + name)
            if hashlib.sha256(data).hexdigest() != str(meta["sha256"]) or len(data) != int(meta["size"]):
                raise IngestError("digest_mismatch", f"{name} does not match the manifest")
            target = staging.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        try:
            verify_lane_d_checkpoint(staging, contract, dtype=dtype)
        except CheckpointTampered as e:
            raise IngestError("checkpoint_tampered", str(e)) from e
        except CheckpointDiverged as e:
            raise IngestError("checkpoint_diverged", str(e)) from e
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return final


__all__ = ["INGEST_CODES", "IngestError", "ingest_entry"]
