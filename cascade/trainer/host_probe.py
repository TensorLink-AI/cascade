"""Host-side run telemetry — what the POD contributed to a run, apart from the
submission.

A heat screens generators on the compute their corpus bought, and the wall is
priced in generator speed deliberately (DEC-CA-0001). That pricing is only a
measurement of the *generator* if the device under every entrant is the same.
The fleet is rented per round, sometimes across several marketplaces, so it is
not: two pods reporting the same ``nvidia-smi`` name can differ by a large
factor in achieved throughput, and the CPU slice a lane actually gets — which
this pipeline leans on hard, since the 4M model moves ~262k tokens per step —
varies with lane fan-out and with whatever else shares the physical box.

Nothing in the run record separated the two. ``data_wait_frac`` says whether
the corpus starved training, so its *absence* rules starvation out; it cannot
say the host was slow. This module supplies the missing half:

* ``host_bench_tokens_per_s`` — a **fixed** calibration workload timed before
  the round's corpus stream opens. It is byte-identical on every host and
  completely independent of the submission, so it is directly comparable pod to
  pod. It is the field that turns "host variance" from an inference into a
  measured number, and the one a per-host ``ref_throughput`` normalisation would
  have to be built on.
* lane geometry, CPU capability, GPU nameplate + PCIe link, and two opaque host
  ids — enough to group runs by pod *and* by physical machine (co-tenancy), so
  realized throughput can be regressed on host facts to say how much of a
  round's spread is host-attributable rather than generator-attributable.

Read the bench as a **host index, not a prediction**: it is a different
workload from a training step, so its absolute value is not comparable to the
run's ``throughput_tokens_per_s`` — only host-to-host ratios are meaningful, and
only within one ``host_bench_spec``.

Scope and safety:

* **Observability only.** These fields ride the existing training-log channel
  (``logs/round-<id>/<role>.jsonl`` plus the optional wandb mirror). They never
  touch the manifest, the receipt, ``contract_digest``, or any scored quantity —
  no wire-format change, no signature break, no consensus input.
* **Never fatal.** Every probe is independently wrapped: a failure omits its own
  keys, logs at debug, and the round proceeds. A pod missing ``nvidia-smi``,
  ``/proc``, or CUDA still trains.
* **Never perturbs the run.** The bench uses its own RNG (it does not touch the
  global torch/numpy seeds the contract pins), frees its buffers, and runs
  *before* the corpus stream opens — and ``max_train_seconds`` anchors at the
  first training batch, so the bench cannot eat the compute budget either.

The workload constants are deliberately **not** configurable: a bench whose size
an operator can change is a bench whose numbers cannot be pooled across pods.
``[telemetry] host_bench`` turns it off; it does not resize it. Any change to
the workload must bump :data:`HOST_BENCH_SPEC` so old and new numbers are never
silently compared.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import socket
import subprocess
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("cascade.trainer.host")

# ── the fixed calibration workload ───────────────────────────────────────────
#
# One GPU "step" is a (D, D) fp32 matmul + tanh + backward, plus a per-step
# host→device copy of the input so the CPU/PCIe leg of the pipeline is in the
# measurement rather than hidden by a device-resident buffer. D is chosen so
# D*D == 262_144 elements == the ~262k tokens a real training step moves, which
# makes the tokens/s unit mean the same thing in both numbers even though the
# work per token differs.
HOST_BENCH_DIM = 512
HOST_BENCH_TOKENS_PER_STEP = HOST_BENCH_DIM * HOST_BENCH_DIM   # 262_144
HOST_BENCH_WARMUP_STEPS = 3      # absorbs CUDA context init + autotune
HOST_BENCH_STEPS = 20
HOST_BENCH_CPU_ITERS = 20
HOST_BENCH_D2D_MIB = 64          # device-to-device copy size for the bandwidth leg
HOST_BENCH_D2D_ITERS = 10
HOST_BENCH_SEED = 0x0CA5CADE     # fixed: the bench must be identical on every host

# Bump on ANY change to the workload above. Emitted with every measurement so a
# regression can group by spec instead of pooling numbers from two workloads.
HOST_BENCH_SPEC = "v1:mm512-fp32-x20+h2d+cputanh262144x20+d2d64Mx10"

# Opaque-id salt. The published ids are keyed hashes, not hostnames or IPs: they
# group runs without disclosing provider inventory. Fixed so ids are stable
# across rounds (the whole point is grouping runs BY host over time).
_HOST_ID_SALT = b"cascade-host-id-v1"
_HOST_ID_HEX = 12

# Where a pod's identity comes from, most trustworthy first. /etc/machine-id is
# per-container on docker-template pods, so it identifies the POD (all lanes of
# one pod share it). boot_id is NOT namespaced, so two pods on one physical
# machine share it — that difference is exactly the co-tenancy signal.
_MACHINE_ID_PATHS = ("/etc/machine-id", "/var/lib/dbus/machine-id")
_BOOT_ID_PATH = "/proc/sys/kernel/random/boot_id"


def _keyed_id(raw: str) -> str:
    """Stable opaque id for ``raw`` — salted sha256, truncated."""
    h = hashlib.sha256(_HOST_ID_SALT + raw.strip().encode("utf-8", "replace"))
    return h.hexdigest()[:_HOST_ID_HEX]


def _read_first_line(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]


def host_ids() -> dict:
    """Opaque ``host_id`` (the pod) and ``host_boot_id`` (the physical machine).

    ``host_id_source`` records which file the pod id came from, because the
    grouping is only as good as its source: ``machine-id`` groups the lanes of
    one pod exactly, while the ``hostname`` fallback is merely "probably the
    same box". ``host_boot_id`` is a different scope, not a fallback — two pods
    sharing a ``host_boot_id`` but not a ``host_id`` are **co-tenants on one
    machine**, which is the case a per-pod id can never show.
    """
    out: dict = {}
    for path in _MACHINE_ID_PATHS:
        try:
            value = _read_first_line(path)
        except OSError:
            continue
        if value:
            out["host_id"] = _keyed_id(value)
            out["host_id_source"] = "machine-id"
            break
    try:
        boot = _read_first_line(_BOOT_ID_PATH)
    except OSError:
        boot = ""
    if boot:
        out["host_boot_id"] = _keyed_id(boot)
    if "host_id" not in out:
        # No machine-id: fall back to boot_id (groups by machine, not pod), then
        # hostname. Both are labelled so a consumer knows the grouping is coarser.
        if boot:
            out["host_id"], out["host_id_source"] = _keyed_id(boot), "boot-id"
        else:
            out["host_id"] = _keyed_id(socket.gethostname())
            out["host_id_source"] = "hostname"
    return out


def lane_geometry() -> dict:
    """This lane's position in its pod's fan-out, from the dispatch-injected env.

    Only the orchestrator knows a pod's lane count — the pod-side view is a
    masked ``CUDA_VISIBLE_DEVICES`` that cannot tell a single-GPU pod from one
    lane of eight — so it stamps the geometry into each lane's env
    (:func:`cascade.trainer.remote.build_remote_command`). An absent env means a
    single lane, which is what a local run or a 1-GPU pod is.

    ``host_gen_cpu_slice`` is the core count the sandboxed generator is actually
    confined to, read from the sandbox's own slicing rule rather than
    re-derived, so the two can never drift apart. It is the number to test
    against when asking whether slowness tracks CPU contention: the trainer
    process keeps the pod's full affinity set, but the generator feeding it does
    not.
    """
    from .sandbox import LANE_COUNT_ENV, LANE_INDEX_ENV, _lane_cpu_slice

    idx_s = os.environ.get(LANE_INDEX_ENV, "")
    cnt_s = os.environ.get(LANE_COUNT_ENV, "")
    out = {
        "host_lane_index": int(idx_s) if idx_s.isdigit() else 0,
        "host_lane_count": int(cnt_s) if cnt_s.isdigit() else 1,
    }
    lane = _lane_cpu_slice()
    if lane is not None:
        out["host_gen_cpu_slice"] = int(lane[1])
    return out


def cpu_facts() -> dict:
    """Cores this process may actually use, the box's total, and the CPU model.

    ``host_cpu_count`` is ``len(os.sched_getaffinity(0))``, not
    ``os.cpu_count()``: on docker-template pods ``cpu_count`` reports the HOST's
    cores even under a cgroup/cpuset limit, so the affinity set is the only one
    that answers "what did this run get". Both are emitted — a large gap between
    them is itself a finding (a cpuset-limited pod on a big machine).
    """
    out: dict = {"host_cpu_count_total": int(os.cpu_count() or 0)}
    try:
        out["host_cpu_count"] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):      # non-Linux: no affinity API
        out["host_cpu_count"] = out["host_cpu_count_total"]
    model = ""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    if model:
        out["host_cpu_model"] = model
    return out


# nvidia-smi query fields, in the order they are parsed out of each CSV row.
# All are stable, long-standing --query-gpu names; `nounits` keeps them numeric.
_SMI_FIELDS = (
    "index", "name", "driver_version", "clocks.max.sm", "clocks.max.memory",
    "pcie.link.gen.current", "pcie.link.width.current",
)


def _run_smi(argv: list[str]) -> str:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=15, check=True
    ).stdout


def gpu_facts(run_fn=None) -> dict:
    """GPU nameplate for THIS lane's device, plus the box's GPU count.

    ``nvidia-smi`` does not honour ``CUDA_VISIBLE_DEVICES``, so it reports every
    GPU on the physical box. That is useful twice over: ``host_gpu_count`` is the
    box's fan-out, and comparing it to ``host_lane_count`` says whether cascade
    holds the whole pod or shares the machine with another tenant. The per-device
    fields are taken from the row matching this lane's mask, and are **omitted**
    rather than guessed when the mask is not a single ordinal and there is more
    than one row — a nameplate attributed to the wrong device is worse than a
    missing one.

    ``pcie.link.*`` is current, not max, on purpose: a link that negotiated down
    (a shared riser, an oversubscribed pod) is a host explanation for a slow run
    that nameplate specs hide. ``run_fn`` is injected by tests; the default
    shells out.
    """
    run = run_fn or _run_smi
    raw = run([
        "nvidia-smi", f"--query-gpu={','.join(_SMI_FIELDS)}",
        "--format=csv,noheader,nounits",
    ])
    rows = [
        [tok.strip() for tok in line.split(",")]
        for line in raw.splitlines() if line.strip()
    ]
    rows = [r for r in rows if len(r) == len(_SMI_FIELDS)]
    if not rows:
        return {}
    out: dict = {"host_gpu_count": len(rows)}

    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    row = None
    if mask.isdigit():
        row = next((r for r in rows if r[0] == mask), None)
    elif len(rows) == 1:
        row = rows[0]
    if row is None:
        return out

    _, name, driver, sm_clock, mem_clock, pcie_gen, pcie_width = row
    out["host_gpu_name"] = name
    if driver:
        out["host_driver_version"] = driver
    for key, value in (
        ("host_gpu_clock_max_mhz", sm_clock),
        ("host_gpu_mem_clock_max_mhz", mem_clock),
        ("host_pcie_gen", pcie_gen),
        ("host_pcie_width", pcie_width),
    ):
        with contextlib.suppress(ValueError):   # "[N/A]" on virtualised GPUs
            out[key] = int(float(value))
    return out


def torch_facts(device: str) -> dict:
    """Stack versions and the device's SM count / memory — the cheap extras that
    distinguish two pods whose ``nvidia-smi`` names match.

    SM count is the one that most often explains a same-name mismatch (a
    partitioned or cut-down card), and the CUDA/torch pair pins the software
    stack a run executed on, which is not otherwise in the log for HEAT runs
    (only finals carry the image-digest pin).
    """
    import torch

    out: dict = {"host_torch_version": str(torch.__version__)}
    if getattr(torch.version, "cuda", None):
        out["host_cuda_version"] = str(torch.version.cuda)
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return out
    props = torch.cuda.get_device_properties(0)
    out["host_gpu_sm_count"] = int(props.multi_processor_count)
    out["host_gpu_mem_total_mb"] = int(props.total_memory // (1024 * 1024))
    out["host_gpu_capability"] = f"{props.major}.{props.minor}"
    return out


def _cpu_bench(iters: int = HOST_BENCH_CPU_ITERS) -> float:
    """Elementwise-tanh passes per second over a fixed 262k-element buffer, in
    tokens/s.

    Single-threaded by construction (numpy's tanh is not parallelised), so this
    indexes the SPEED of the core this lane got, while ``host_cpu_count`` /
    ``host_gen_cpu_slice`` count them. Slow-per-core and few-cores are different
    host faults with different fixes, and a single aggregate would confound them.
    """
    x = np.linspace(-3.0, 3.0, HOST_BENCH_TOKENS_PER_STEP, dtype=np.float64)
    buf = np.empty_like(x)
    np.tanh(x, out=buf)                    # warm the pages
    t0 = time.perf_counter()
    for _ in range(iters):
        np.tanh(x, out=buf)
    elapsed = max(1e-9, time.perf_counter() - t0)
    return iters * HOST_BENCH_TOKENS_PER_STEP / elapsed


def host_bench(device: str) -> dict:
    """Time the fixed calibration workload on ``device``; return its metrics.

    Three legs, because a slow host is not one thing: ``host_bench_tokens_per_s``
    is the composite index (H2D copy + matmul + backward, the shape of a
    training step), ``host_bench_cpu_tokens_per_s`` isolates the CPU leg, and
    ``host_bench_d2d_gb_s`` the device's own memory bandwidth. A pod that is slow
    on the composite but normal on CPU and bandwidth is a compute-throttled card;
    slow on CPU alone is contention for the lane's cores.

    RNG hygiene: the inputs come from a dedicated ``torch.Generator``, so the
    global seeds the contract pins are untouched (the trainer re-seeds them at
    ``train()`` regardless, but a telemetry probe must not depend on that). GPU
    buffers are freed and the caching allocator released before returning, so the
    training run that follows starts from the allocator state it would have had.

    Known caveat, stated rather than hidden: on a multi-lane pod the lanes are
    dispatched together, so they bench CONCURRENTLY. The number therefore includes
    sibling contention — which is the right quantity for "what did this run get",
    but means a lane benched alone (a staggered retry, a single-lane pod) is not
    strictly comparable to one benched alongside seven others. Group by
    ``host_lane_count`` before comparing across pod shapes.
    """
    import torch

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        # A custom backend can name a device this box does not have. Falling back
        # keeps the CPU leg — losing every leg over one bad device string would
        # discard the telemetry exactly on the hosts most worth looking at.
        log.debug("bench device %s unavailable; falling back to cpu", dev)
        dev = torch.device("cpu")
    on_cuda = dev.type == "cuda"

    def sync() -> None:
        if on_cuda:
            torch.cuda.synchronize(dev)

    gen = torch.Generator().manual_seed(HOST_BENCH_SEED)
    x_cpu = torch.randn(
        HOST_BENCH_DIM, HOST_BENCH_DIM, generator=gen, dtype=torch.float32
    )
    w = torch.randn(
        HOST_BENCH_DIM, HOST_BENCH_DIM, generator=gen, dtype=torch.float32
    ).to(dev).requires_grad_(True)

    def step() -> None:
        # The H2D copy stays inside the timed step: a device-resident input would
        # hide exactly the host-side leg this bench exists to expose.
        x = x_cpu.to(dev)
        torch.tanh(x @ w).square().mean().backward()
        w.grad = None

    t_wall = time.perf_counter()
    for _ in range(HOST_BENCH_WARMUP_STEPS):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(HOST_BENCH_STEPS):
        step()
    sync()
    elapsed = max(1e-9, time.perf_counter() - t0)

    out = {
        "host_bench_spec": HOST_BENCH_SPEC,
        "host_bench_device": str(dev),
        "host_bench_tokens_per_s": round(
            HOST_BENCH_STEPS * HOST_BENCH_TOKENS_PER_STEP / elapsed, 1
        ),
        "host_bench_cpu_tokens_per_s": round(_cpu_bench(), 1),
    }
    if on_cuda:
        n = HOST_BENCH_D2D_MIB * 1024 * 1024 // 4
        src = torch.empty(n, dtype=torch.float32, device=dev)
        dst = torch.empty_like(src)
        dst.copy_(src)
        sync()
        t0 = time.perf_counter()
        for _ in range(HOST_BENCH_D2D_ITERS):
            dst.copy_(src)
        sync()
        d2d = max(1e-9, time.perf_counter() - t0)
        # Read + write, hence 2×: a copy touches the bus twice per element.
        out["host_bench_d2d_gb_s"] = round(
            2 * HOST_BENCH_D2D_ITERS * src.numel() * 4 / d2d / 1e9, 1
        )
        del src, dst
    # Drop the bench's own references BEFORE releasing the allocator's cache, so
    # the training run that follows starts from the allocator state it would have
    # had without the bench (rebound rather than `del`ed: `step` closes over them).
    w = x_cpu = None
    if on_cuda:
        torch.cuda.empty_cache()
    out["host_bench_seconds"] = round(time.perf_counter() - t_wall, 2)
    return out


def resolve_device(preferred: str | None = None) -> str:
    """The device string to bench on: the trainer's own if it exposes one.

    ``BaseTrainer`` is a Protocol and only the reference implementation carries a
    ``.device``, so the caller passes what it can and this fills in the rest —
    the bench must work under a custom backend, just possibly on CPU.
    """
    if preferred:
        return str(preferred)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                       # noqa: BLE001 - no torch: CPU legs only
        return "cpu"


def host_snapshot(*, device: str | None = None, run_bench: bool = True) -> dict:
    """Every host fact this pod can report, as one flat dict of JSON scalars.

    Assembled probe by probe with each failure swallowed independently, so a
    partial answer is the normal degraded mode: a pod with no ``nvidia-smi``
    still reports its CPU and lane facts, and a pod where the bench blows up
    still reports its identity. Keys are absent, never null, when a probe could
    not answer — a consumer filters on presence.

    Flat and scalar-only because both sinks require it: the S3 log is one JSON
    object per line and wandb charts only scalars.
    """
    facts: dict = {}
    dev = resolve_device(device)
    probes = (
        ("ids", host_ids),
        ("lane", lane_geometry),
        ("cpu", cpu_facts),
        ("gpu", gpu_facts),
        ("torch", lambda: torch_facts(dev)),
    )
    for name, probe in probes:
        try:
            facts.update(probe())
        except Exception as e:              # noqa: BLE001 - telemetry is never fatal
            log.debug("host probe %s unavailable (continuing): %s", name, e)
    if run_bench:
        try:
            facts.update(host_bench(dev))
        except Exception as e:              # noqa: BLE001
            log.warning("host calibration bench failed (continuing without it): %s", e)
    return facts


def host_summary_line(facts: dict) -> str:
    """The host snapshot as one parseable ``key=value`` line.

    ``TrainResult.metrics`` never crosses the remote boundary (the worker's
    receipt is a metrics-less ``TrainedEntry``), and the S3 log is written by the
    POD — so if a pod's credentials are wrong, or the bucket is unreachable, this
    stderr line is the only channel by which the orchestrator learns what
    hardware the run got. It is deliberately the same shape as the existing
    per-run telemetry line so one log grep collects both.
    """
    keys = (
        "host_id", "host_boot_id", "host_lane_index", "host_lane_count",
        "host_cpu_count", "host_gen_cpu_slice", "host_gpu_count", "host_gpu_name",
        "host_gpu_sm_count", "host_pcie_gen", "host_pcie_width",
        "host_bench_tokens_per_s", "host_bench_cpu_tokens_per_s",
        "host_bench_d2d_gb_s", "host_bench_seconds",
    )
    parts = [f"{k}={facts[k]}" for k in keys if k in facts]
    return " ".join(parts) if parts else "no host facts available"
