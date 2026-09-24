"""In-memory checkpoint digests.

Every tensor file the trainer writes is serialised in memory, hashed there,
and the digest recorded in this process before the bytes reach the disk.
The worker reports those digests in its receipt (``tensor_digests``); the
orchestrator hashes the files it harvests from the pod and refuses a
checkpoint whose tensor files differ from, or were not among, the ones the
worker wrote (``tensor_mismatches``).

``save_tensors_hashed`` writes exactly the bytes ``safetensors.torch.save_file``
would (same serialiser, no metadata): the checkpoint format is unchanged.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

TENSOR_SUFFIX = ".safetensors"

_LOCK = threading.Lock()
_DIGESTS: dict[str, str] = {}      # absolute path → sha256 hex of the bytes THIS process wrote


def _key(path: Path | str) -> str:
    return str(Path(path).resolve())


def save_tensors_hashed(tensors: dict, path: Path | str) -> str:
    """Serialise ``tensors`` in memory, hash the bytes, write them atomically
    to ``path`` and record the digest. Returns the sha256 hex digest."""
    from safetensors.torch import save

    data = save(tensors)
    digest = hashlib.sha256(data).hexdigest()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, p)
    record_digest(p, digest)
    return digest


def record_digest(path: Path | str, digest: str) -> None:
    with _LOCK:
        _DIGESTS[_key(path)] = str(digest)


def digests_for(directory: Path | str) -> dict[str, str]:
    """``{file name: sha256}`` of every tensor file this process wrote
    directly under ``directory`` (empty when none — e.g. a reused checkpoint
    trained by an earlier process; see the ``.train_complete`` marker)."""
    d = _key(directory)
    with _LOCK:
        return {os.path.basename(k): v for k, v in _DIGESTS.items()
                if os.path.dirname(k) == d}


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_mismatches(directory: Path | str, expected: dict[str, str]) -> list[str]:
    """Why the tensor files under ``directory`` are not the ones the worker
    wrote, as one reason per file (empty list = all match). A file the worker
    hashed that is missing or differs is named; so is any tensor file present
    that the worker never wrote (a planted file is a swap too)."""
    d = Path(directory)
    reasons: list[str] = []
    for name in sorted(expected):
        want = str(expected[name]).lower()
        p = d / name
        if not p.is_file():
            reasons.append(f"{name}: missing from the harvested checkpoint")
            continue
        got = file_sha256(p)
        if got != want:
            reasons.append(f"{name}: sha256 {got[:12]}… on the pod, the worker wrote "
                           f"{want[:12]}… (replaced after the save)")
    for p in sorted(d.iterdir()) if d.is_dir() else []:
        if p.is_file() and p.name.endswith(TENSOR_SUFFIX) and p.name not in expected:
            reasons.append(f"{p.name}: tensor file the worker never wrote")
    return reasons
