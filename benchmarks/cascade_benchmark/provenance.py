"""Verify local FEV task and data pins before any worker is started."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_VERIFIED_SNAPSHOTS: dict[str, dict] = {}
_DATASET_IDENTITIES: dict[str, str] = {}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def task_id(task) -> str:
    """The established FEV report key, also checked when pinning tasks."""
    for attr in ("task_name", "name"):
        value = getattr(task, attr, None)
        if isinstance(value, str) and value:
            for root, sha in _DATASET_IDENTITIES.items():
                value = value.replace(root, f"sha256:{sha}")
            return value
    dataset = str(getattr(task, "dataset_path", ""))
    dataset = _DATASET_IDENTITIES.get(str(Path(dataset).resolve()), dataset)
    parts = [dataset, str(getattr(task, "dataset_config", ""))]
    return "/".join(p for p in parts if p) + f"/h{getattr(task, 'horizon', '')}"


def tree_hash(path):
    root = Path(path)
    if root.is_symlink():
        raise ValueError("FEV data symlinks are not pinned")
    if root.is_file():
        h = hashlib.sha256()
        with root.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                h.update(chunk)
        return h.hexdigest()
    entries = []
    for file in sorted(root.rglob("*")):
        if file.is_symlink():
            raise ValueError("FEV data symlinks are not pinned")
        if file.is_file():
            entries.append([file.relative_to(root).as_posix(), tree_hash(file)])
    if not entries:
        raise ValueError("empty FEV dataset")
    return hashlib.sha256(canonical(entries).encode()).hexdigest()


def tree_snapshot(path):
    """Cheap change detector around/beyond hashing, including file replacement."""
    root = Path(path)
    files = [root] if root.is_file() else sorted(root.rglob("*"))
    out = []
    for file in files:
        if file.is_symlink():
            raise ValueError("FEV data symlinks are not pinned")
        if file.is_file():
            s = file.stat()
            out.append((str(file), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns))
    return out


def verified_snapshot(src):
    return _VERIFIED_SNAPSHOTS.get(str(Path(src).resolve()))


def verify_fev_manifest(path):
    """Return a content/task identity independent of the local mount paths.

    Tasks must name those same local datasets (absolute dataset_path in YAML).
    No worker is permitted to fetch a mutable Hub dataset implicitly.
    """
    import fev

    path = Path(path).resolve()
    manifest = json.loads(path.read_text())

    def local(name):
        p = Path(name)
        p = p if p.is_absolute() else path.parent / p
        if p.is_symlink():
            raise ValueError("FEV data symlinks are not pinned")
        return p.resolve()

    tasks = local(manifest["tasks"]["path"])
    task_snapshot = tree_snapshot(tasks)
    if (tasks.is_symlink() or tree_hash(tasks) != manifest["tasks"]["sha256"]
            or tree_snapshot(tasks) != task_snapshot):
        raise ValueError("FEV task hash mismatch")
    datasets, snapshots = {}, {}
    for row in manifest["datasets"]:
        root = local(row["path"])
        key = str(root.resolve())
        if key in datasets:
            raise ValueError("duplicate FEV dataset pin")
        before = tree_snapshot(root)
        if tree_hash(root) != row["sha256"] or tree_snapshot(root) != before:
            raise ValueError("FEV dataset hash mismatch")
        datasets[key] = row["sha256"]
        snapshots[key] = before
    _DATASET_IDENTITIES.update(datasets)
    actual_tasks = fev.Benchmark.from_yaml(str(tasks)).tasks
    used, identities = set(), set()
    for task in actual_tasks:
        identity = task_id(task)
        if identity in identities:
            raise ValueError("duplicate FEV task identity; preserve unique task_name values")
        identities.add(identity)
        root = str(Path(task.dataset_path).resolve())
        if task.dataset_config is not None or root not in datasets:
            raise ValueError("FEV tasks must use the pinned local datasets, not mutable Hub refs")
        used.add(root)
    if not used or used != set(datasets):
        raise ValueError("FEV pinned dataset/task coverage mismatch")
    os.environ["CASCADE_BENCH_FEV_TASKS"] = str(tasks.resolve())
    import yaml

    def normalized(value):
        if isinstance(value, dict):
            return {k: normalized(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalized(v) for v in value]
        if isinstance(value, str):
            if value in datasets:
                return {"dataset_sha256": datasets[value]}
            for root, sha in sorted(datasets.items(), key=lambda item: -len(item[0])):
                value = value.replace(root, f"sha256:{sha}")
        return value

    # Paths locate bytes; they do not identify the scientific dataset/tasks.
    semantic = {"tasks": normalized(yaml.safe_load(tasks.read_text())),
                "datasets": sorted(datasets.values())}
    identity = hashlib.sha256(canonical(semantic).encode()).hexdigest()
    _VERIFIED_SNAPSHOTS[str(tasks.resolve())] = {
        "tasks": task_snapshot, "datasets": snapshots,
        "task_sha256": manifest["tasks"]["sha256"],
        "manifest_sha256": identity,
    }
    return identity
