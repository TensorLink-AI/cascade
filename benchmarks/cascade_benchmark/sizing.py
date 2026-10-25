"""How many sweep workers fit this box — ``--workers auto|N``.

Pure arithmetic over measured numbers, so it is unit-testable without a GPU:

* **by GPU memory** — per visible device, ``floor((free − headroom) /
  per_worker)``, summed over devices. ``per_worker`` is MEASURED by a probe
  process (``parallel.probe``): it loads every checkpoint of the pass on one
  device, runs one warm-up batch at the sweep's ``--batch-size`` and reads
  ``torch.cuda.max_memory_reserved`` (the caching allocator's high-water mark,
  which is ≥ ``max_memory_allocated`` and is what the device actually loses),
  then adds the CUDA context (:data:`CUDA_CONTEXT_BYTES`) and a
  :data:`SAFETY` margin for the variance of real batches (GIFT-Eval's
  univariate-ized batches can overshoot ``batch_size`` by one entry's
  variates; long horizons block-decode). Without a measurement it falls back
  to :data:`ARCH_ESTIMATE_BYTES` (per checkpoint, documented per arch);
* **by CPU** — ``cpu_count − 1`` (one core for the parent: queueing, the
  report, the OS). Dataset building and the gluonts metrics are single-threaded
  CPU work, which is why one GPU feeds several workers;
* **cap** — :data:`DEFAULT_CAP` (``--max-workers``), and never more workers
  than units.

``auto`` = the minimum of the three (≥ 1). A CPU device sizes by CPU only.
Workers are spread round-robin over the devices that still have room, and
each worker sees exactly one GPU (``CUDA_VISIBLE_DEVICES``). An explicit ``N``
is taken as given (still ≤ units), round-robin over every visible device.

Nothing here can change a number in the report: worker count only decides
which process runs which unit (``suites/_common.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1 << 30
MIB = 1 << 20
# Per-device VRAM left free for the driver, fragmentation and anything else on
# the card (the main validator's private-pool eval may share it).
DEFAULT_HEADROOM_BYTES = 2 * GIB
# Hard ceiling on auto workers (process + host-RAM overhead grows per worker:
# each holds its own copy of the dataset it is building).
DEFAULT_CAP = 16
# CUDA context + cuBLAS/cuDNN workspaces, not counted by the torch allocator.
CUDA_CONTEXT_BYTES = 512 * MIB
# Multiplier on the measured warm-up peak.
SAFETY = 1.25
# Per-checkpoint fallback when no measurement is available (probe failed, or a
# plan computed offline): the measured-order peak of the fp32 forward at
# batch 512 + the context, rounded up. ``cascade_moe`` = Cascade-MoE-28M,
# ``toto2`` = the dense toto2 lineage; unknown archs take the larger.
ARCH_ESTIMATE_BYTES = {"cascade_moe": 3 * GIB, "toto2": 2 * GIB}
DEFAULT_ESTIMATE_BYTES = 3 * GIB


@dataclass(frozen=True)
class GpuInfo:
    """One visible GPU: ``ident`` is what goes into a worker's
    ``CUDA_VISIBLE_DEVICES`` (an index or UUID as the parent saw it)."""

    ident: str
    free_bytes: int
    total_bytes: int
    name: str = ""


@dataclass
class WorkerPlan:
    workers: int
    # per worker: the CUDA_VISIBLE_DEVICES value, or None (CPU / in-process)
    devices: list = field(default_factory=list)
    per_worker_bytes: int | None = None
    reason: str = ""

    def describe(self) -> str:
        per = (f"{self.per_worker_bytes / GIB:.2f} GiB/worker"
               if self.per_worker_bytes else "per-worker VRAM n/a")
        devs = ",".join(str(d) for d in self.devices if d is not None) or "-"
        return f"workers={self.workers} devices=[{devs}] {per} ({self.reason})"


def parse_workers(value) -> str | int:
    """``"auto"`` or a positive int; raises ``ValueError`` otherwise."""
    if isinstance(value, bool):
        raise ValueError(f"workers must be 'auto' or an int >= 1; got {value!r}")
    if isinstance(value, int):
        n = value
    else:
        s = str(value).strip().lower()
        if s == "auto":
            return "auto"
        try:
            n = int(s)
        except ValueError:
            raise ValueError(f"workers must be 'auto' or an int >= 1; got {value!r}") from None
    if n < 1:
        raise ValueError(f"workers must be 'auto' or an int >= 1; got {value!r}")
    return n


def per_worker_from_peak(peak_bytes: int) -> int:
    """VRAM one worker needs, from the probe's measured allocator peak."""
    return int(peak_bytes * SAFETY) + CUDA_CONTEXT_BYTES


def estimate_per_worker(archs: list[str]) -> int:
    """Fallback per-worker VRAM: the documented per-arch estimate summed over
    the checkpoints the worker holds (a paired pass holds them all)."""
    return sum(ARCH_ESTIMATE_BYTES.get(a, DEFAULT_ESTIMATE_BYTES) for a in archs) or \
        DEFAULT_ESTIMATE_BYTES


def device_capacity(gpu: GpuInfo, per_worker: int, headroom: int) -> int:
    if per_worker <= 0:
        return 0
    return max(0, (int(gpu.free_bytes) - int(headroom)) // int(per_worker))


def distribute(n: int, idents: list[str], capacities: list[int] | None = None) -> list[str]:
    """Assign ``n`` workers round-robin over ``idents``, skipping a device once
    it holds ``capacities[i]`` workers (``None`` = unbounded). Raises when the
    capacities cannot hold ``n``."""
    if not idents:
        raise ValueError("no devices to distribute workers over")
    caps = list(capacities) if capacities is not None else [None] * len(idents)
    load = [0] * len(idents)
    out: list[str] = []
    i = 0
    while len(out) < n:
        for _ in range(len(idents)):
            k = i % len(idents)
            i += 1
            if caps[k] is None or load[k] < caps[k]:
                load[k] += 1
                out.append(idents[k])
                break
        else:
            raise ValueError(f"capacities {caps} cannot hold {n} workers")
    return out


def plan_workers(requested, *, device: str, gpus: list[GpuInfo], cpu_count: int | None,
                 per_worker_bytes: int | None, n_units: int,
                 headroom_bytes: int = DEFAULT_HEADROOM_BYTES,
                 cap: int = DEFAULT_CAP) -> WorkerPlan:
    """The worker plan for ``requested`` (``"auto"`` or an int) — see the
    module docstring for the rule. ``gpus`` are the devices this sweep may use
    (already narrowed to ``cuda:K`` when one was named)."""
    req = parse_workers(requested)
    units = max(1, int(n_units))
    on_gpu = str(device).startswith("cuda")
    by_cpu = max(1, int(cpu_count or 1) - 1)
    if on_gpu and not gpus:
        return WorkerPlan(1, [None], per_worker_bytes, "cuda requested but no GPU visible")
    idents = [g.ident for g in gpus]
    if req != "auto":
        n = max(1, min(int(req), units))
        devs = distribute(n, idents) if on_gpu else [None] * n
        return WorkerPlan(n, devs, per_worker_bytes,
                          f"explicit {req}" + (f", capped to {units} units" if n < req else ""))
    if not on_gpu:
        n = max(1, min(by_cpu, cap, units))
        return WorkerPlan(n, [None] * n, None,
                          f"cpu device: min(cpu-1={by_cpu}, cap={cap}, units={units})")
    per = int(per_worker_bytes) if per_worker_bytes else None
    if not per:
        return WorkerPlan(1, distribute(1, idents), None, "per-worker VRAM unknown: 1 worker")
    caps = [device_capacity(g, per, headroom_bytes) for g in gpus]
    by_gpu = sum(caps)
    n = min(by_gpu, by_cpu, cap, units)
    why = (f"min(gpu={by_gpu} [" + ", ".join(
        f"{g.ident}: ({g.free_bytes / GIB:.1f} GiB free - {headroom_bytes / GIB:.1f}) // "
        f"{per / GIB:.2f} = {c}" for g, c in zip(gpus, caps, strict=True))
        + f"], cpu-1={by_cpu}, cap={cap}, units={units})")
    if n < 1:
        # nothing fits by the arithmetic: still run (the sequential sweep
        # would try too) on the roomiest device
        best = max(range(len(gpus)), key=lambda k: gpus[k].free_bytes)
        return WorkerPlan(1, [idents[best]], per, why + " -> 0 fit; 1 worker")
    return WorkerPlan(n, distribute(n, idents, caps), per, why)
