"""Versioned, job-bound optimizer-boundary snapshots; no pickle or executable state.

Input iterators are reconstructed by deterministic replay before RNG restoration.
A continuation is never a new warm start and never increases the remaining budget.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

from ..shared.atomic import path_lock, write_text_atomic

PROTOCOL = "cascade-continuation/1"
VERSION_RE = re.compile(r"^[0-9]{12}-[0-9a-f]{32}$")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _pack(value, tensors):
    import torch

    if isinstance(value, torch.Tensor):
        name = str(len(tensors))
        tensors[name] = value.detach().cpu().contiguous().clone()
        return {"tensor": name}
    if isinstance(value, np.ndarray):
        return {"numpy": _pack(torch.from_numpy(value.copy()), tensors)}
    if isinstance(value, dict):
        return {"dict": [[_pack(k, tensors), _pack(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {"tuple" if isinstance(value, tuple) else "list": [_pack(v, tensors) for v in value]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": repr(value)}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported continuation state type: {type(value).__name__}")


def _unpack(value, tensors):
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ValueError("invalid continuation state encoding")
    kind, data = next(iter(value.items()))
    if kind == "tensor":
        return tensors[data]
    if kind == "numpy":
        return _unpack(data, tensors).numpy().copy()
    if kind == "dict":
        return {_unpack(k, tensors): _unpack(v, tensors) for k, v in data}
    if kind in {"tuple", "list"}:
        items = [_unpack(v, tensors) for v in data]
        return tuple(items) if kind == "tuple" else items
    if kind == "float" and data in {"nan", "inf", "-inf"}:
        return float(data)
    raise ValueError("invalid continuation state tag")


class SnapshotStore:
    def __init__(self, root, binding, *, max_bytes=16 * 1024**3):
        self.root, self.binding, self.max_bytes = Path(root), binding, max_bytes

    def save(self, step, state):
        from safetensors.torch import save_file

        if step < 1:
            raise ValueError("continuation must follow a completed optimizer step")
        tensors = {}
        packed = _pack(state, tensors)
        if sum(t.numel() * t.element_size() for t in tensors.values()) > self.max_bytes:
            raise ValueError("continuation exceeds its storage limit")
        self.root.mkdir(parents=True, exist_ok=True)
        with path_lock(self.root):
            prior = self.header()
            if prior is not None and step < prior["step"]:
                raise ValueError("out-of-order continuation refused")
            version = f"{step:012d}-{uuid.uuid4().hex}"
            folder = self.root / version
            folder.mkdir(mode=0o700)
            publishing = False
            try:
                save_file(tensors, str(folder / "state.safetensors"))
                with open(folder / "state.safetensors", "rb") as f:
                    os.fsync(f.fileno())
                metadata = {
                    "protocol": PROTOCOL,
                    "step": step,
                    "binding": self.binding,
                    "state": packed,
                    "tensor_sha256": digest_file(folder / "state.safetensors"),
                    "runtime": state.get("runtime"),
                }
                write_text_atomic(folder / "state.json", canonical(metadata).decode())
                pointer = {
                    "version": version,
                    "metadata_sha256": digest_file(folder / "state.json"),
                }
                publishing = True
                write_text_atomic(self.root / "CURRENT", canonical(pointer).decode())
            except BaseException:
                if not publishing:
                    shutil.rmtree(folder, ignore_errors=True)
                raise
            # Retain the preceding generation for an in-progress mirror pull.
            folders = sorted(
                p for p in self.root.iterdir() if p.is_dir() and VERSION_RE.fullmatch(p.name)
            )
            for old in folders[:-2]:
                shutil.rmtree(old)
            return pointer

    def header(self):
        path = self.root / "CURRENT"
        if not path.exists():
            return None
        ptr = json.loads(path.read_text())
        version = ptr["version"]
        if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
            raise ValueError("invalid continuation generation")
        folder = self.root / version
        for name in ("state.json", "state.safetensors"):
            p = folder / name
            bound = min(self.max_bytes, 16 * 1024**2) if name == "state.json" else self.max_bytes
            if p.is_symlink() or not p.is_file() or p.stat().st_size > bound:
                raise ValueError("continuation file is missing or oversized")
        if folder.is_symlink() or digest_file(folder / "state.json") != ptr["metadata_sha256"]:
            raise ValueError("continuation metadata digest mismatch")
        meta = json.loads((folder / "state.json").read_text())
        if meta["protocol"] != PROTOCOL or canonical(meta["binding"]) != canonical(self.binding):
            raise ValueError("continuation belongs to a different job or protocol")
        return meta

    def load(self):
        from safetensors.torch import load_file

        meta = self.header()
        if meta is None:
            return None
        ptr = json.loads((self.root / "CURRENT").read_text())
        path = self.root / ptr["version"] / "state.safetensors"
        if digest_file(path) != meta["tensor_sha256"]:
            raise ValueError("continuation tensor digest mismatch")
        state = _unpack(meta["state"], load_file(str(path)))
        if int(state["step"]) != int(meta["step"]):
            raise ValueError("continuation cursor mismatch")
        return state


def rng_state():
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("continuation CUDA device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def batch_digest(batch):
    h = hashlib.sha256()

    def add(value):
        if isinstance(value, dict):
            for key in sorted(value):
                h.update(key.encode())
                add(value[key])
        elif hasattr(value, "__dataclass_fields__"):
            from dataclasses import asdict

            add(asdict(value))
        elif isinstance(value, np.ndarray):
            h.update(str(value.dtype).encode())
            h.update(canonical(list(value.shape)))
            h.update(np.ascontiguousarray(value).tobytes())
        elif value is not None:
            h.update(canonical(value))

    add(batch)
    return h.hexdigest()


class Continuation:
    def __init__(self, store, *, interval=300, clock=time.monotonic):
        self.store, self.interval, self.clock = store, interval, clock
        self.last_saved = clock()

    def due(self):
        return self.clock() - self.last_saved >= self.interval

    def save(self, state):
        self.store.save(state["step"], state)
        self.last_saved = self.clock()


def worker_context(args, cfg, binding):
    root = getattr(args, "continuation_root", None)
    if root is None:
        return None
    from .contract import TRAIN_IMAGE_DIGEST_ENV

    rnd = cfg.round
    if (
        not rnd.ops_continuation_interval_seconds
        or not rnd.ops_continuation_image_digest
        or os.environ.get(TRAIN_IMAGE_DIGEST_ENV) != rnd.ops_continuation_image_digest
    ):
        raise ValueError("continuation requires the explicitly qualified running image")
    return Continuation(
        SnapshotStore(root, binding, max_bytes=rnd.ops_continuation_max_bytes),
        interval=rnd.ops_continuation_interval_seconds,
    )


def marketcast_data_files(snapshot):
    """Keep the original public training snapshot, including missing-value bytes."""
    import torch

    files = {}
    for relative in ["snapshot.json", *sorted(snapshot.meta.get("files", {}))]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("invalid MarketCast snapshot member")
        files[relative] = torch.from_numpy(
            np.frombuffer((snapshot.root / relative).read_bytes(), dtype=np.uint8).copy()
        )
    return files


def restore_marketcast_data(args, cfg, expected, destination):
    """Restore pinned data before any upstream refetch can revise the experiment."""
    root = getattr(args, "continuation_root", None)
    if root is None or not (Path(root) / "CURRENT").exists():
        return False
    # A binding peek discovers the original data digest, then normal store
    # verification checks the complete metadata/tensors before using any bytes.
    pointer = json.loads((Path(root) / "CURRENT").read_text())
    version = pointer["version"]
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise ValueError("invalid MarketCast continuation generation")
    metadata = Path(root) / version / "state.json"
    if metadata.is_symlink() or metadata.stat().st_size > 1024 * 1024:
        raise ValueError("invalid MarketCast continuation metadata")
    binding = json.loads(metadata.read_text())["binding"]
    if canonical({k: v for k, v in binding.items() if k != "data"}) != canonical(expected):
        raise ValueError("MarketCast continuation belongs to a different experiment")
    context = worker_context(args, cfg, binding)
    state = context.store.load()
    files = state.get("data_files")
    if not isinstance(files, dict) or "snapshot.json" not in files:
        raise ValueError("MarketCast continuation has no pinned training data")
    destination = Path(destination)
    staging = destination.with_name(destination.name + ".restore-" + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    try:
        for relative, tensor in files.items():
            path = Path(relative)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not (
                    relative == "snapshot.json"
                    or path.parts[0] in {"bars", "coarse"}
                    and path.suffix == ".npy"
                )
            ):
                raise ValueError("invalid MarketCast continuation data member")
            target = staging / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(tensor.numpy().tobytes())
        from ..pool.marketcast import load_snapshot

        snapshot = load_snapshot(staging, verify_signature=False)
        if snapshot.digest != binding["data"]:
            raise ValueError("MarketCast continuation data digest mismatch")
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return True


def training_runtime(model):
    import torch

    parameter = next(model.parameters())
    gpu = parameter.device.type == "cuda"
    return {
        "torch": torch.__version__,
        "dtype": str(parameter.dtype),
        "device": parameter.device.type,
        "gpu_name": torch.cuda.get_device_name(parameter.device) if gpu else "",
        "cuda": torch.version.cuda if gpu else "",
        "devices": torch.cuda.device_count() if gpu else 0,
        "cpu_threads": None if gpu else torch.get_num_threads(),
    }


def assert_training_runtime(model, state):
    if state.get("runtime") != training_runtime(model):
        raise ValueError("continuation requires the same qualified training runtime and GPU model")
