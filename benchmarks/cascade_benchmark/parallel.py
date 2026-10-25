"""Concurrent public-benchmark sweep: work units over N worker processes.

Every suite splits into independent units (one GIFT-Eval config / BOOM
dataset / TIME task / FEV task — ``suites/_common.py``). The parent PLANS each
suite (checkpoint-free), then either runs the units in-process (``workers =
1``: the sequential sweep) or hands them to N **spawned** worker processes
(never forked: a forked CUDA context is unusable). Each worker loads every
checkpoint of the pass ONCE (``predictor.load_wrapper_cached``) and pulls one
unit at a time from the parent, so a slow unit never stalls the others. The
parent collects each unit's per-checkpoint outcomes by unit index and
assembles every suite in the FIXED canonical unit order — the report is
byte-identical to the sequential sweep's (same rows, aggregate, skip list),
whatever the worker count, device spread or completion order.

Numerics are untouched: every unit runs the same code on the same inputs with
the same ``batch_size`` (batches never span units), fp32, the same quantile
grid. A unit that fails is recorded exactly as the sequential sweep records it
(``run_unit`` never raises). A worker that DIES (OOM-kill, CUDA fatal) is
replaced and its unit re-run once; a unit that kills two workers ends the
sweep with an error — the sequential sweep would have died on it too.

Paired sweeps: ``checkpoints`` may hold several checkpoints (challenger +
king, king + init); each unit's dataset is built once and forecast by every
checkpoint, and the caller gets one result list per checkpoint, each equal to
sweeping that checkpoint alone.
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import json
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import sizing
from .limits import host_limits, limit_worker_threads, spawn_environment, worker_ram_bytes
from .results import SuiteResult
from .suites import SUITE_MODULES
from .suites._common import SuitePlan, run_plan
from .torch_runtime import apply_runtime, capture_runtime

# A unit whose worker died this many times ends the sweep.
MAX_UNIT_CRASHES = 2
# Warm-up batch the VRAM probe forwards (per checkpoint): ``batch_size``
# series of this length, decoded this far — one full context window and the
# longest single-pass CPM decode block (toto2 STABLE_DECODE_STEPS).
WARMUP_LEN = 8192
WARMUP_HORIZON = 768
PROBE_TIMEOUT_S = 900


@dataclass
class SweepTimings:
    """Wall-clock bookkeeping (report-only; never in the report JSON)."""

    plan: str = ""
    total_s: float = 0.0
    suites: dict = field(default_factory=dict)  # name -> {units, wall_s, unit_s}

    def to_dict(self) -> dict:
        return {"plan": self.plan, "total_s": self.total_s, "suites": self.suites}


def _say(msg: str) -> None:
    print(f"[sweep] {msg}", file=sys.stderr, flush=True)


def load_suite(path: str):
    """The suite implementation at ``path`` — a module (``pkg.mod``) or an
    attribute of one (``pkg.mod:obj``) exposing ``plan`` / ``run_unit`` /
    ``assemble``. Imported by NAME so a spawned worker resolves the same code."""
    mod_name, _, attr = path.partition(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr) if attr else mod


def plan_suites(names: list[str], modules: dict[str, str], max_series: int | None) -> dict:
    """``{suite: SuitePlan | SuiteResult}`` in ``names`` order (parent)."""
    return {n: load_suite(modules[n]).plan(max_series=max_series) for n in names}


def interleaved_jobs(names: list[str], plans: dict) -> list[tuple[str, object]]:
    """Start every suite early; preserve canonical unit order within each.

    A large BOOM sweep must not delay the first TIME/FEV task until its last
    unit starts. Results are still assembled by suite, independent of timing.
    """
    pending = {name: iter(plans[name].units) for name in names
               if isinstance(plans[name], SuitePlan)}
    jobs = []
    while pending:
        for name in list(pending):
            try:
                jobs.append((name, next(pending[name])))
            except StopIteration:
                del pending[name]
    return jobs


# ── GPU discovery + per-worker VRAM probe (spawned: the parent never touches CUDA) ──


def _visible_idents(n: int) -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    ids = [s.strip() for s in raw.split(",") if s.strip()] if raw else []
    return ids[:n] if ids else [str(i) for i in range(n)]


def _arch_of(checkpoint_dir: str) -> str:
    try:
        cfg = json.loads((Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — unknown arch takes the larger estimate
        return ""
    return "cascade_moe" if "moe" in cfg else ("toto2" if "toto2" in cfg else "")


def _probe_main(checkpoints: list[str], device: str, batch_size: int, measure: bool,
                out_q, runtime=None) -> None:  # pragma: no cover — needs a GPU
    try:
        import numpy as np
        import torch

        from .predictor import _load_wrapper

        limit_worker_threads()
        if runtime is not None:
            apply_runtime(runtime)

        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        idents = _visible_idents(n)
        gpus = []
        for i in range(n):
            free, total = torch.cuda.mem_get_info(i)
            gpus.append((idents[i], int(free), int(total), torch.cuda.get_device_name(i)))
        peak = None
        if measure and n:
            ordinal = int(device.split(":", 1)[1]) if ":" in device else 0
            dev = f"cuda:{ordinal}"
            torch.cuda.reset_peak_memory_stats(dev)
            rng = np.random.default_rng(0)
            hist = list(rng.standard_normal((max(1, batch_size), WARMUP_LEN)).cumsum(axis=1))
            wrappers = []
            for ck in checkpoints:
                w = _load_wrapper(Path(ck), dev)
                wrappers.append(w)  # paired workers retain EVERY model, including the king
                if hasattr(w, "forecast_quantiles_batch"):
                    w.forecast_quantiles_batch(hist, WARMUP_HORIZON)
                else:
                    w.forecast(hist[0], WARMUP_HORIZON, 100)
            torch.cuda.synchronize(dev)
            peak = int(max(torch.cuda.max_memory_reserved(dev),
                           torch.cuda.max_memory_allocated(dev)))
        out_q.put({"gpus": gpus, "peak": peak})
    except BaseException as e:  # noqa: BLE001 — reported, the parent falls back
        out_q.put({"error": f"{type(e).__name__}: {e}"})


def probe(checkpoints: list[str], device: str, batch_size: int, *, measure: bool = True,
          timeout_s: float = PROBE_TIMEOUT_S) -> dict:
    """Visible GPUs (``[(ident, free, total, name)]``) and, with ``measure``,
    the allocator peak of loading ``checkpoints`` + one warm-up batch on one
    device — from a short-lived spawned process. ``{"error": …}`` on failure."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_probe_main,
                    args=(list(checkpoints), device, batch_size, measure, q, capture_runtime()),
                    daemon=True)
    with spawn_environment():
        p.start()
    try:
        return q.get(timeout=timeout_s)
    except queue.Empty:
        return {"error": f"probe timed out after {timeout_s:.0f}s"}
    finally:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
        q.close()


def choose_plan(workers, *, checkpoints: list[str], device: str, batch_size: int,
                n_units: int, max_workers: int = sizing.DEFAULT_CAP,
                worker_vram_bytes: int | None = None,
                headroom_bytes: int = sizing.DEFAULT_HEADROOM_BYTES,
                probe_fn: Callable[..., dict] = probe,
                cpu_count: int | None = None,
                available_ram_bytes: int | None = None,
                per_worker_ram_bytes: int | None = None) -> sizing.WorkerPlan:
    """Resolve ``--workers`` into a :class:`~cascade_benchmark.sizing.WorkerPlan`."""
    req = sizing.parse_workers(workers)
    if req == 1 or n_units <= 1:
        return sizing.WorkerPlan(1, [None], None, "sequential" if req == 1 else
                                 f"{n_units} unit(s): sequential")
    effective_cpus, effective_ram = host_limits()
    cpus = cpu_count if cpu_count is not None else effective_cpus
    ram = available_ram_bytes if available_ram_bytes is not None else effective_ram
    gpus: list[sizing.GpuInfo] = []
    per = worker_vram_bytes
    if str(device).startswith("cuda"):
        measure = req == "auto" and not per
        got = probe_fn(checkpoints, device, batch_size, measure=measure)
        if "error" in got:
            return sizing.WorkerPlan(1, [None], None, f"GPU probe failed ({got['error']}): "
                                     "sequential")
        gpus = [sizing.GpuInfo(*g) for g in got.get("gpus") or []]
        if ":" in str(device):
            k = int(str(device).split(":", 1)[1])
            gpus = gpus[k:k + 1]
        if measure:
            if got.get("peak"):
                per = sizing.per_worker_from_peak(int(got["peak"]))
            else:
                per = sizing.estimate_per_worker([_arch_of(c) for c in checkpoints])
    plan = sizing.plan_workers(req, device=device, gpus=gpus, cpu_count=cpus,
                               per_worker_bytes=per, n_units=n_units,
                               headroom_bytes=headroom_bytes, cap=max_workers)
    if req == "auto" and ram is not None:
        per_ram = per_worker_ram_bytes or worker_ram_bytes(checkpoints)
        # Available RAM already excludes the parent; leave additional space
        # for IPC, reports and filesystem buffers. This is a safety estimate,
        # not a guarantee that every real dataset fits the allowance.
        spare = max(512 << 20, ram // 10)
        by_ram = max(1, (ram - spare) // per_ram)
        if plan.workers > by_ram:
            plan.devices = plan.devices[:by_ram]
            plan.workers = by_ram
        plan.reason += f", RAM cap={by_ram} ({per_ram / sizing.GIB:.2f} GiB/worker)"
    return plan


# ── the worker ───────────────────────────────────────────────────────────────


def _worker_main(wid: int, in_q, out_q, modules: dict[str, str], ctxs: dict,
                 checkpoints: list[str], opts: dict) -> None:
    try:
        limit_worker_threads()
        apply_runtime(opts.pop("_runtime"))
        mods = {n: load_suite(p) for n, p in modules.items()}
    except BaseException:  # noqa: BLE001
        out_q.put(("fatal", wid, None, traceback.format_exc()))
        return
    while True:
        item = in_q.get()
        if item is None:
            return
        j, name, unit = item
        t0 = time.monotonic()
        try:
            outs = mods[name].run_unit(ctxs[name], unit, checkpoints, **opts)
        except BaseException:  # noqa: BLE001 — run_unit must never raise: a bug, end the sweep
            out_q.put(("fatal", wid, j, traceback.format_exc()))
            return
        out_q.put(("done", wid, j, (outs, time.monotonic() - t0)))


class WorkerCrashed(RuntimeError):
    """A unit killed its worker :data:`MAX_UNIT_CRASHES` times, or a worker
    hit a bug — the sweep ends without a report, as a sequential crash would."""


@dataclass
class _Worker:
    proc: object
    in_q: object
    device: str | None
    job: int | None = None


def _run_units_parallel(jobs: list[tuple[str, object]], plan: sizing.WorkerPlan,
                        modules: dict[str, str], ctxs: dict, checkpoints: list[str],
                        opts: dict, timings: SweepTimings) -> dict[int, list]:
    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    workers: dict[int, _Worker] = {}
    next_wid = [0]

    def spawn(device: str | None) -> None:
        wid = next_wid[0]
        next_wid[0] += 1
        in_q = ctx.Queue()
        wopts = dict(opts)
        if device is not None:
            wopts["device"] = "cuda"
        with spawn_environment(device):
            p = ctx.Process(target=_worker_main,
                            args=(wid, in_q, out_q, modules, ctxs, checkpoints, wopts),
                            daemon=True)
            p.start()
        workers[wid] = _Worker(p, in_q, device)

    pending = collections.deque(range(len(jobs)))
    results: dict[int, list] = {}
    crashes: collections.Counter = collections.Counter()
    respawns = 0
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    try:
        for dev in plan.devices:
            spawn(dev)
        while len(results) < len(jobs):
            for w in workers.values():
                if w.job is None and pending:
                    j = pending.popleft()
                    w.job = j
                    name = jobs[j][0]
                    first.setdefault(name, time.monotonic())
                    w.in_q.put((j, name, jobs[j][1]))
            try:
                kind, wid, j, payload = out_q.get(timeout=0.5)
            except queue.Empty:
                kind = None
            if kind == "fatal":
                raise WorkerCrashed(f"sweep worker {wid} failed"
                                    + (f" on {jobs[j][0]} unit {jobs[j][1]!r}" if j is not None
                                       else "") + f":\n{payload}")
            if kind == "done":
                w = workers.get(wid)
                if w is not None and w.job == j:
                    w.job = None
                if j not in results:
                    outs, dt = payload
                    results[j] = outs
                    name = jobs[j][0]
                    last[name] = time.monotonic()
                    s = timings.suites.setdefault(name, {"units": 0, "wall_s": 0.0,
                                                         "unit_s": 0.0})
                    s["units"] += 1
                    s["unit_s"] += float(dt)
            for wid, w in list(workers.items()):
                if w.proc.is_alive():
                    continue
                del workers[wid]
                if w.job is not None and w.job not in results:
                    crashes[w.job] += 1
                    name, unit = jobs[w.job]
                    if crashes[w.job] >= MAX_UNIT_CRASHES:
                        raise WorkerCrashed(
                            f"{name} unit {unit!r} killed {MAX_UNIT_CRASHES} sweep workers "
                            f"(last exit code {w.proc.exitcode})")
                    _say(f"worker {wid} died (exit {w.proc.exitcode}) on {name} unit "
                         f"{unit!r}; re-running it on a fresh worker")
                    pending.appendleft(w.job)
                respawns += 1
                if respawns > 2 * max(1, plan.workers) + 2:
                    raise WorkerCrashed(f"sweep workers keep dying ({respawns} respawns)")
                spawn(w.device)
    finally:
        for w in workers.values():
            with contextlib.suppress(Exception):
                w.in_q.put(None)
        for w in workers.values():
            w.proc.join(timeout=10)
            if w.proc.is_alive():
                w.proc.terminate()
                w.proc.join(timeout=5)
            w.in_q.close()
        out_q.close()
    for name, t0 in first.items():
        timings.suites[name]["wall_s"] = last.get(name, t0) - t0
    return results


# ── the sweep ────────────────────────────────────────────────────────────────


def run_sweep(suite_names: list[str], checkpoints: list[str], *, num_samples: int,
              max_series: int | None, device: str, batch_size: int, workers="1",
              modules: dict[str, str] | None = None, max_workers: int = sizing.DEFAULT_CAP,
              worker_vram_bytes: int | None = None,
              headroom_bytes: int = sizing.DEFAULT_HEADROOM_BYTES,
              probe_fn: Callable[..., dict] = probe,
              timings: SweepTimings | None = None,
              per_worker_ram_bytes: int | None = None) -> list[list[SuiteResult]]:
    """Sweep ``suite_names`` for every checkpoint in ``checkpoints`` →
    ``[per checkpoint: [SuiteResult per suite, in suite_names order]]``.
    ``modules`` overrides the suite → module mapping (tests)."""
    limit_worker_threads()  # sequential and parallel must use the same CPU math settings
    mods = {n: (modules or {}).get(n) or SUITE_MODULES[n] for n in suite_names}
    timings = timings if timings is not None else SweepTimings()
    t_start = time.monotonic()
    plans = plan_suites(suite_names, mods, max_series)
    jobs = interleaved_jobs(suite_names, plans)
    plan = choose_plan(workers, checkpoints=checkpoints, device=device, batch_size=batch_size,
                       n_units=len(jobs), max_workers=max_workers,
                       worker_vram_bytes=worker_vram_bytes, headroom_bytes=headroom_bytes,
                       probe_fn=probe_fn, per_worker_ram_bytes=per_worker_ram_bytes)
    timings.plan = plan.describe()
    _say(f"plan: {timings.plan}; {len(jobs)} units over {len(suite_names)} suite(s), "
         f"{len(checkpoints)} checkpoint(s)")
    opts = {"num_samples": num_samples, "device": device, "batch_size": batch_size}
    per_ckpt: list[list[SuiteResult]] = [[] for _ in checkpoints]
    if plan.workers <= 1:
        for n in suite_names:
            t0 = time.monotonic()
            mod = load_suite(mods[n])
            got = run_plan(mod, plans[n], checkpoints, **opts)
            units = len(plans[n].units) if isinstance(plans[n], SuitePlan) else 0
            dt = time.monotonic() - t0
            timings.suites[n] = {"units": units, "wall_s": dt, "unit_s": dt}
            for i, r in enumerate(got):
                per_ckpt[i].append(r)
        timings.total_s = time.monotonic() - t_start
        return per_ckpt
    ctxs = {n: p.ctx for n, p in plans.items() if isinstance(p, SuitePlan)}
    opts["_runtime"] = capture_runtime()
    results = _run_units_parallel(jobs, plan, {n: mods[n] for n in ctxs}, ctxs,
                                  list(checkpoints), opts, timings)
    # canonical order: suite order, then each suite's unit order
    by_suite: dict[str, list[list]] = {n: [] for n in suite_names}
    for j, (n, _u) in enumerate(jobs):
        by_suite[n].append(results[j])
    for n in suite_names:
        p = plans[n]
        timings.suites.setdefault(n, {"units": 0, "wall_s": 0.0, "unit_s": 0.0})
        if not isinstance(p, SuitePlan):
            got = run_plan(None, p, checkpoints, **opts)
        else:
            mod = load_suite(mods[n])
            got = [mod.assemble(p, [u[i] for u in by_suite[n]]) for i in range(len(checkpoints))]
        for i, r in enumerate(got):
            per_ckpt[i].append(r)
    timings.total_s = time.monotonic() - t_start
    return per_ckpt
