"""Materialise one leg's corpus on a CPU VM (DEC-CA-0053).

Runs ON the rented CPU VM, never on the orchestrator. :func:`generate_corpus`
fans the generator out over ``n`` sandboxed child streams — child ``i`` on
:func:`~cascade.generation.shards.child_seed` ``(seed, i)``, each the same
sandbox the GPU pod's ``stream_cpu`` path uses (rlimits, env scrub, netns,
the shared :class:`~cascade.interface.generator.SeriesValidator`) — and writes
each child's series into its own parquet shards. A child stops at its share of
``target_points``, at the generation window, or when the generator ends; the
counts each child reached go into ``corpus.json``, which is all an auditor
needs to regenerate the exact corpus.

Children are pinned one core each through the sandbox's lane slice
(``CASCADE_LANE_INDEX``/``CASCADE_LANE_COUNT``), which also caps BLAS to one
thread per child. The window is checked between series; a child blocked
mid-series is bounded by the sandbox's own per-frame stall window.

Faults: a :class:`~cascade.trainer.corpus.CorpusError` (the generator failed
or its output was rejected) or a ``ValueError`` (the series validator's
rejection) is a ``generator`` fault; anything else is ``infra``. Either stops the other children; the index is still written so the
partial corpus can be inspected, and ``faults`` says why.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import __version__
from ..shared.config import GenerationConfig, GeneratorConfig
from .shards import ShardError, _pa, as_2d, child_seed, finalize_index, write_shard

log = logging.getLogger("cascade.generation.tier")

# (repo, seed, child cfg, child budget points, max wall seconds) → a context
# manager yielding the child's validated series. Injectable for tests.
ChildRunner = Callable[[Path, int, GeneratorConfig, int, int],
                       AbstractContextManager[Iterator[np.ndarray]]]

_spawn_lock = threading.Lock()


def _lane_env(index: int, count: int):
    """Set the sandbox lane env for one child spawn (the parent reads it
    while building the child's affinity and thread caps)."""
    from ..trainer.sandbox import LANE_COUNT_ENV, LANE_INDEX_ENV

    @contextlib.contextmanager
    def cm():
        old = {k: os.environ.get(k) for k in (LANE_INDEX_ENV, LANE_COUNT_ENV)}
        os.environ[LANE_INDEX_ENV] = str(index)
        os.environ[LANE_COUNT_ENV] = str(count)
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    return cm()


def sandbox_runner(repo: Path, seed: int, cfg: GeneratorConfig, budget: int,
                   max_wall: int) -> AbstractContextManager[Iterator[np.ndarray]]:
    """The production child: :func:`cascade.trainer.sandbox.stream_series`
    (container mode included, per ``cfg.sandbox_mode``)."""
    from ..trainer import sandbox

    return sandbox.stream_series(repo, seed, cfg, budget, max_wall_seconds=max_wall)


def inprocess_runner(repo: Path, seed: int, cfg: GeneratorConfig, budget: int,
                     max_wall: int) -> AbstractContextManager[Iterator[np.ndarray]]:
    """No sandbox (offline runs and tests): the in-process stream."""
    from ..trainer.stream import _inprocess_stream

    return contextlib.nullcontext(_inprocess_stream(repo, seed, cfg, budget))


def _values(element) -> np.ndarray:
    if isinstance(element, dict):
        extra = set(element) - {"values", "groups"}
        if extra:
            raise ShardError(f"record fields {sorted(extra)} are not stored by the shard format")
        element = element["values"]
    return as_2d(element)


def _split(total: int, n: int) -> list[int]:
    base, rem = divmod(int(total), n)
    return [base + (rem if i == 0 else 0) for i in range(n)]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def _host_vcpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _cpu_seconds() -> float:
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


class _Child:
    def __init__(self, index: int, seed: int, budget: int) -> None:
        self.index = index
        self.seed = seed
        self.budget = budget
        self.shards: list[dict] = []
        self.points = 0
        self.stop_reason = ""
        self.fault: dict | None = None
        self.lengths: Counter = Counter()      # pow2 length bucket → series
        self.channels: Counter = Counter()


def _run_child(child: _Child, *, repo: Path, cfg: GeneratorConfig, n_children: int,
               out_dir: Path, shard_points: int, deadline: float, max_wall: int,
               runner: ChildRunner, stop: threading.Event, pin: bool,
               now: Callable[[], float]) -> None:
    buf: list[np.ndarray] = []
    buf_points = 0

    def flush() -> None:
        nonlocal buf, buf_points
        if buf:
            name = f"c{child.index}-{len(child.shards):05d}.parquet"
            child.shards.append(write_shard(out_dir / name, buf))
            buf, buf_points = [], 0

    try:
        with _spawn_lock:
            env = _lane_env(child.index, n_children) if pin else contextlib.nullcontext()
            with env:
                cm = runner(repo, child.seed, cfg, child.budget, max_wall)
                frames = cm.__enter__()
        try:
            from ..interface.generator import SeriesValidator

            validator = SeriesValidator.from_config(cfg)
            for index, element in enumerate(frames):
                element = validator.process(element, index)
                a = _values(element)
                child.lengths[1 << max(0, int(a.shape[-1]) - 1).bit_length()] += 1
                child.channels[int(a.shape[0]) if a.ndim == 2 else 1] += 1
                buf.append(element if isinstance(element, dict) else a)
                buf_points += int(a.size)
                child.points += int(a.size)
                if buf_points >= shard_points:
                    flush()
                if child.points >= child.budget:
                    child.stop_reason = "target"
                    break
                if stop.is_set():
                    child.stop_reason = "stopped"
                    break
                if now() >= deadline:
                    child.stop_reason = "window"
                    break
            else:
                child.stop_reason = "exhausted"
        except BaseException as e:
            cm.__exit__(type(e), e, e.__traceback__)
            raise
        else:
            cm.__exit__(None, None, None)
        flush()
    except Exception as e:  # noqa: BLE001 — classified, reported, never raised
        from ..trainer.corpus import CorpusError

        # CorpusError: the sandbox's verdict on the generator; ValueError: the
        # shared SeriesValidator's rejection (the in-process path raises it
        # directly). Everything else is the machine's fault.
        kind = ("generator" if isinstance(e, (CorpusError, ShardError, ValueError))
                else "infra")
        if "sandbox_isolation_unavailable" in str(e):
            kind = "infra"           # the VM cannot make a netns/userns: ours, not theirs
        child.fault = {"kind": kind, "error": f"{type(e).__name__}: {e}"[:500]}
        child.stop_reason = f"{kind}_fault"
        stop.set()
        with contextlib.suppress(Exception):
            flush()


def _merge(counters) -> dict[str, int]:
    total: Counter = Counter()
    for c in counters:
        total.update(c)
    return {str(k): v for k, v in sorted(total.items())}


def generate_corpus(
    repo_dir: Path | str, seed: int, gen_cfg: GeneratorConfig, gcfg: GenerationConfig,
    out_dir: Path | str, *, deadline_s: float | None = None, use_sandbox: bool = True,
    runner: ChildRunner | None = None, generator_ref: str = "",
    now: Callable[[], float] = time.monotonic,
) -> dict:
    """Generate one leg's corpus into ``out_dir`` and return its index
    (``corpus.json`` plus ``faults`` and ``stop_reason``).

    ``deadline_s`` overrides the window (seconds from now; default
    ``gcfg.window_seconds``). ``runner`` overrides the child (default: the
    sandbox, or the in-process stream with ``use_sandbox=False``)."""
    _pa()  # Initialize the allocator before worker threads can import Arrow.
    repo = Path(repo_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = gcfg.n_children
    window = float(deadline_s if deadline_s is not None else gcfg.window_seconds)
    run = runner or (sandbox_runner if use_sandbox else inprocess_runner)
    child_cfg = replace(gen_cfg, max_memory_mb=gcfg.child_memory_mb,
                        max_length=gcfg.max_length)
    children = [_Child(i, child_seed(seed, i), b)
                for i, b in enumerate(_split(gcfg.target_points, n))]
    stop = threading.Event()
    t0, c0 = now(), _cpu_seconds()
    wall0 = time.monotonic()
    deadline = t0 + window
    threads = [threading.Thread(
        target=_run_child, args=(c,), daemon=True, name=f"gen-child-{c.index}",
        kwargs=dict(repo=repo, cfg=child_cfg, n_children=n, out_dir=out,
                    shard_points=gcfg.shard_points, deadline=deadline,
                    max_wall=int(window), runner=run, stop=stop,
                    pin=runner is None and use_sandbox, now=now),
    ) for c in children]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    faults = [{"child": c.index, **c.fault} for c in children if c.fault]
    reasons = {c.stop_reason for c in children}
    if not faults and not any(c.points for c in children):
        kind = "generator" if reasons == {"exhausted"} else "infra"
        faults.append({"child": None, "kind": kind, "error": "generation produced no usable series"})
    if faults:
        stop_reason = ("generator_fault" if any(f["kind"] == "generator" for f in faults)
                       else "infra_fault")
    elif "window" in reasons:
        stop_reason = "window"
    elif reasons == {"exhausted"}:
        stop_reason = "exhausted"
    else:
        stop_reason = "target"
    meta = {
        "generator_ref": generator_ref,
        "vcpus": gcfg.vcpus,
        "host_vcpus": _host_vcpus(),
        "n_children": n,
        "cpu_model": cpu_model(),
        "cpu_seconds": round(_cpu_seconds() - c0, 3),
        "wall_seconds": round(time.monotonic() - wall0, 3),
        "target_points": gcfg.target_points,
        "window_seconds": window,
        "max_length": gcfg.max_length,
        "min_channels": gen_cfg.min_channels,
        "max_channels": gen_cfg.max_channels,
        "stop_reason": stop_reason,
        "faults": faults,
        "length_histogram": _merge(c.lengths for c in children),
        "channels_histogram": _merge(c.channels for c in children),
        "cascade_version": __version__,
    }
    kids = [{"index": c.index, "seed": c.seed, "stop_reason": c.stop_reason,
             "shards": c.shards} for c in children]
    index = finalize_index(out, seed=seed, children=kids, meta=meta)
    log.info("corpus %s: %d series, %d points, stop=%s", index["corpus_digest"][:12],
             index["n_series"], index["points"], stop_reason)
    return index
