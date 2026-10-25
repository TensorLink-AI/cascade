"""Host and container limits for the isolated benchmark environment."""

from __future__ import annotations

import contextlib
import math
import os
import threading
from pathlib import Path

GIB = 1 << 30
_SPAWN_LOCK = threading.RLock()
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _read(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def _groups(root: Path, proc: Path):
    """Current cgroups and their parents, including a namespaced mount root."""
    bases = [root]
    for line in _read(proc / "self/cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3 or ".." in Path(parts[2]).parts:
            continue
        controllers, relative = parts[1], parts[2].lstrip("/")
        if not controllers:
            bases.append(root / relative)
        else:
            for controller in controllers.split(","):
                bases.extend([root / controller / relative,
                              root / controllers / relative])
    seen = set()
    for base in bases:
        while base == root or root in base.parents:
            if base not in seen:
                seen.add(base)
                yield base
            if base == root:
                break
            base = base.parent


def host_limits(*, cgroup_root: Path = Path("/sys/fs/cgroup"),
                proc_root: Path = Path("/proc")) -> tuple[int, int | None]:
    """Effective CPUs and available RAM; quotas can be tighter than the VM."""
    cpus = os.cpu_count() or 1
    with contextlib.suppress(AttributeError, OSError):
        cpus = min(cpus, len(os.sched_getaffinity(0)))
    available = []
    for line in _read(proc_root / "meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            with contextlib.suppress(ValueError, IndexError):
                available.append(int(line.split()[1]) * 1024)
    for group in _groups(cgroup_root, proc_root):
        quota = _read(group / "cpu.max").split()
        if len(quota) != 2:
            quota = [_read(group / "cpu.cfs_quota_us"),
                     _read(group / "cpu.cfs_period_us")]
        with contextlib.suppress(ValueError, ZeroDivisionError):
            q, p = map(int, quota)
            if q > 0 and p > 0:
                cpus = min(cpus, max(1, math.floor(q / p)))
        for limit_name, usage_name in (("memory.max", "memory.current"),
                                       ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            with contextlib.suppress(ValueError):
                limit, used = int(_read(group / limit_name)), int(_read(group / usage_name))
                if 0 < limit < 1 << 60:
                    available.append(max(0, limit - used))
    return max(1, cpus), min(available) if available else None


def worker_ram_bytes(checkpoints: list[str]) -> int:
    """Conservative model loading + dataset allowance, overridable by operator."""
    size = 0
    for checkpoint in checkpoints:
        root = Path(checkpoint)
        for file in root.glob("*.safetensors"):
            size += file.stat().st_size
    return 2 * GIB + 2 * size


@contextlib.contextmanager
def spawn_environment(device: str | None = None, *, threads: int = 1):
    """Set child limits before Python imports numpy/torch; restore the parent.

    Serialize spawn-time environment changes so simultaneous sweeps cannot
    inherit another worker's CUDA mask. Each worker runs one task at a time.
    """
    changes = {name: str(threads) for name in _THREAD_ENV}
    if device is not None:
        changes["CUDA_VISIBLE_DEVICES"] = device
    with _SPAWN_LOCK:
        saved = {name: os.environ.get(name) for name in changes}
        try:
            os.environ.update(changes)
            yield
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def limit_worker_threads() -> None:
    """Also cover libraries imported before the worker entry point."""
    import torch
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(1)
    # Keep the controller alive for the lifetime of the worker.
    global _thread_controller
    _thread_controller = threadpool_limits(limits=1)
