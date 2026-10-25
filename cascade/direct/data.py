"""Content-pinned deterministic generator shards; restart is independent of rank count."""

from __future__ import annotations

import hashlib
import io
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..shared.direct_moe import digest


def tree_digest(root):
    root = Path(root)
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("generator symlinks are not admitted")
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
            entries.append(
                [path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest()]
            )
    return digest(entries)


class ArraySource:
    """Pinned generated corpus; explicit finite reuse instead of implicit wrapping."""

    def __init__(self, spec, *, objects):
        body = objects.get(spec["object"]["uri"])
        if hashlib.sha256(body).hexdigest() != spec["object"]["sha256"]:
            raise ValueError("corpus digest mismatch")
        self.values = np.load(io.BytesIO(body), allow_pickle=False)
        if self.values.ndim != 3 or not np.isfinite(self.values).all():
            raise ValueError("corpus must be finite [examples,channels,context]")
        self.epochs = spec["epochs"]
        if type(self.epochs) is not int or self.epochs < 1:
            raise ValueError("pin positive finite corpus reuse")
        self.fingerprint = digest(spec)

    def get(self, index):
        if index < 0 or index >= len(self.values) * self.epochs:
            raise ValueError("declared generated corpus exhausted")
        return self.values[index % len(self.values)]


class SandboxSource:
    """Generate fixed-size seed shards with the existing strict miner sandbox.

    Every shard uses a contract-derived seed and a fixed n_series. Regeneration
    restores the exact generator RNG/data position by construction; no arbitrary
    Python pickle or miner code executes inside the trainer process. One shard is
    retained in memory per rank. CPU generation time is independently observable.
    """

    def __init__(self, spec, *, objects):
        from ..shared.config import GeneratorConfig

        self.spec = spec
        if "archive" in spec:
            from ..funding.store import extract_zip_safely

            body = objects.get(spec["archive"]["uri"])
            if hashlib.sha256(body).hexdigest() != spec["archive"]["sha256"]:
                raise ValueError("generator archive checksum mismatch")
            self._temporary = tempfile.TemporaryDirectory(prefix="direct-generator-")
            self.root = extract_zip_safely(body, self._temporary.name)
        else:
            self.root = Path(spec["repository"])
        if tree_digest(self.root) != spec["source_sha256"]:
            raise ValueError("generator source changed")
        self.cfg = GeneratorConfig(**spec["generator_config"])
        if not self.cfg.sandbox_strict:
            raise ValueError("production generator requires strict sandbox isolation")
        self.shard_size = spec["shard_examples"]
        if self.shard_size < 1:
            raise ValueError("invalid generator shard size")
        self.fingerprint = digest({k: v for k, v in spec.items() if k != "repository"})
        self.shard, self.arrays = None, None

    def get(self, index):
        from ..trainer.sandbox import run_in_sandbox

        shard, offset = divmod(index, self.shard_size)
        if shard != self.shard:
            seed = int(digest({"seed": self.spec["seed"], "shard": shard})[:15], 16)
            cfg = replace(self.cfg, corpus_n_series=self.shard_size, corpus_target_points=0)
            result = run_in_sandbox(
                self.root, seed, cfg, blocked=tuple(self.spec["blocked_imports"])
            )
            self.arrays = result.series
            self.shard = shard
        arr = self.arrays[offset]
        return arr["values"] if isinstance(arr, dict) else arr


def open_source(spec, objects):
    if spec["kind"] == "array":
        return ArraySource(spec, objects=objects)
    if spec["kind"] == "sandbox":
        return SandboxSource(spec, objects=objects)
    raise ValueError("unknown direct data source")
