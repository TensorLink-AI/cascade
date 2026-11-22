"""T7 measurement kit for lane D context variants (LANE_D_PLAN §4 T7, §1.6).

The T7 pod run times the lane D scorer per context variant on the launch eval
host (4k/8k/16k) or on ``valeval-laned`` (32k–128k, S8 only), checks that a
4k checkpoint and a byte copy declared at 128k score identically on windows
with ≤ 4096 history, and writes ``evidence/t7.json`` (schema ``laned_t7/1``)
from :func:`build_evidence` with ``json.dump``.

A variant is armed only when its projected settlement wall time and its peak
memory fit the budget: ≤ 50% of a settlement and the host GPU's memory
(:func:`settlement_budget`). ``fits``, ``armed``, ``identical`` and
``passed`` are always derived from the measured numbers, never passed in.

A row certifies only what really ran (RV3). The no-pad scorer truncates each
history to ``min(n, context)``, so a variant timed on shorter windows would
measure a smaller model input. Every row therefore records ``input_len`` (the
model input length that ran, which must equal the variant) and the window
history range, and a history shorter than the variant is a
:class:`ShortHistoryError`. Memory is labelled by source: ``cuda_max_allocated``
is the variant's own device peak; ``process_peak_rss`` (CPU ``ru_maxrss``, the
process-lifetime peak every later row inherits) is never accepted as pod
evidence.

:func:`measure_variant` and :func:`short_window_check` need torch (they run
the scorer); everything else is plain Python.
"""

from __future__ import annotations

import json
import math
import operator
import resource
import shutil
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ..shared.config.lane_d import LANE_D_CONTEXTS
from .lane_d_scorer import SCORER_CONTEXTS, batch_plan, forecast_quantiles

SCHEMA = "laned_t7/1"
WP = "W5d"
LAUNCH_VARIANTS: tuple[int, ...] = (4096, 8192, 16384)
S8_HOST = "valeval-laned"          # 96 GB, rented only at S8 for 32k–128k
MAX_LAUNCH_VARIANT = max(LAUNCH_VARIANTS)
TRAINED_CONTEXT = 4096             # 128 patches: the training row length
DECLARED_CONTEXT = max(SCORER_CONTEXTS)
MIN_MODELS = 10                    # up to 8 entries + KING_D + house per settlement
CUDA_PEAK_MEM_SOURCE = "cuda_max_allocated"   # this variant's own device peak
CPU_PEAK_MEM_SOURCE = "process_peak_rss"      # process-lifetime ru_maxrss: never a budget input
PEAK_MEM_SOURCES: tuple[str, ...] = (CUDA_PEAK_MEM_SOURCE, CPU_PEAK_MEM_SOURCE)
POD_SOURCE = "pod"
_GIB = float(2 ** 30)


class ShortHistoryError(ValueError):
    """A variant would be measured on a window with fewer than ``variant``
    points of history: the no-pad scorer would silently run it shorter."""


def settlement_budget(mem_gb, *, settlement_blocks: int = 900, block_seconds: float = 12.0,
                      fraction: float = 0.5) -> dict:
    """The T7 budget of one variant: ``fraction`` of a settlement's wall time
    and the host GPU's memory (§1.6)."""
    wall = float(settlement_blocks) * float(block_seconds) * float(fraction)
    if not (math.isfinite(wall) and wall > 0):
        raise ValueError(f"settlement budget wall time must be positive, got {wall!r}")
    if not _positive(mem_gb):
        raise ValueError(f"mem_gb must be a positive finite number, got {mem_gb!r}")
    return {"wall_s": wall, "mem_gb": mem_gb}


def _positive(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)) and float(v) > 0)


def _int(v, what: str) -> int:
    if isinstance(v, bool):
        raise ValueError(f"{what} must be an int, got {v!r}")
    try:
        return operator.index(v)
    except TypeError:
        raise ValueError(f"{what} must be an int, got {v!r}") from None


def _fits(wall_s: float, peak_mem_gb: float, budget: dict) -> bool:
    return float(wall_s) <= float(budget["wall_s"]) and float(peak_mem_gb) <= float(budget["mem_gb"])


def _check_setup(*, host, gpu, variant, budget, n_models, n_windows) -> int:
    """The row checks that do not need a measurement; returns the variant."""
    v = _int(variant, "variant")
    if v not in LANE_D_CONTEXTS:
        raise ValueError(f"variant {v} is not a [lane_d] context {list(LANE_D_CONTEXTS)}")
    for name, value in (("host", host), ("gpu", gpu)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    if v <= MAX_LAUNCH_VARIANT and host == S8_HOST:
        raise ValueError(f"variant {v} is measured on the launch eval host, never {S8_HOST}")
    if v > MAX_LAUNCH_VARIANT and host != S8_HOST:
        raise ValueError(f"variant {v} is an S8 variant, measured on {S8_HOST}, not {host!r}")
    if _int(n_models, "n_models") < MIN_MODELS:
        raise ValueError(f"n_models must be ≥ {MIN_MODELS} (8 entries + KING_D + house), "
                         f"got {n_models!r}")
    if _int(n_windows, "n_windows") < 1:
        raise ValueError(f"n_windows must be ≥ 1, got {n_windows!r}")
    if (not isinstance(budget, dict) or not _positive(budget.get("wall_s"))
            or not _positive(budget.get("mem_gb"))):
        raise ValueError(f"budget must be {{'wall_s', 'mem_gb'}} positive numbers, got {budget!r}")
    return v


def _check_exercised(variant: int, input_len, history_min, history_max) -> tuple[int, int, int]:
    """RV3: the row ran the variant. ``input_len`` must equal it, and every
    window must have at least ``variant`` points of history."""
    ran = _int(input_len, "input_len")
    if ran != variant:
        raise ValueError(f"input_len {ran} != variant {variant}: the model did not run at the "
                         f"variant's context")
    lo = _int(history_min, "history_min")
    hi = _int(history_max, "history_max")
    if lo < variant:
        raise ShortHistoryError(f"history_min {lo} < variant {variant}: those windows run "
                                f"shorter than the variant under the no-pad scorer")
    if hi < lo:
        raise ValueError(f"history_max {hi} < history_min {lo}")
    return ran, lo, hi


def variant_row(*, host, gpu, variant, input_len, history_min, history_max, wall_s,
                peak_mem_gb, peak_mem_source, budget, n_models, n_windows) -> dict:
    """One measured variant; ``fits`` is derived from the numbers. Refuses a
    row that did not exercise its variant (``input_len != variant``,
    ``history_min < variant``) or whose memory source is unknown."""
    v = _check_setup(host=host, gpu=gpu, variant=variant, budget=budget, n_models=n_models,
                     n_windows=n_windows)
    ran, lo, hi = _check_exercised(v, input_len, history_min, history_max)
    for name, value in (("wall_s", wall_s), ("peak_mem_gb", peak_mem_gb)):
        if not _positive(value):
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    if not isinstance(peak_mem_source, str) or peak_mem_source not in PEAK_MEM_SOURCES:
        raise ValueError(f"peak_mem_source must be one of {list(PEAK_MEM_SOURCES)}, "
                         f"got {peak_mem_source!r}")
    return {"host": host, "gpu": gpu, "variant": v, "input_len": ran, "history_min": lo,
            "history_max": hi, "wall_s": float(wall_s), "peak_mem_gb": float(peak_mem_gb),
            "peak_mem_source": peak_mem_source, "budget": dict(budget),
            "fits": _fits(wall_s, peak_mem_gb, budget), "n_models": _int(n_models, "n_models"),
            "n_windows": _int(n_windows, "n_windows")}


_ROW_INPUTS = ("host", "gpu", "variant", "input_len", "history_min", "history_max", "wall_s",
               "peak_mem_gb", "peak_mem_source", "budget", "n_models", "n_windows")


def _recheck_row(i: int, row, *, source: str) -> dict:
    """Rebuild a handed-in row through :func:`variant_row` (a hand-edited dict
    gets every check), then apply the pod memory rule."""
    if not isinstance(row, dict):
        raise ValueError(f"rows[{i}] must be a dict, got {type(row).__name__}")
    missing = [k for k in _ROW_INPUTS if k not in row]
    if missing:
        raise ValueError(f"rows[{i}] lacks {missing}")
    r = variant_row(**{k: row[k] for k in _ROW_INPUTS})
    if source == POD_SOURCE and r["peak_mem_source"] != CUDA_PEAK_MEM_SOURCE:
        raise ValueError(f"rows[{i}] (variant {r['variant']}): peak_mem_source "
                         f"{r['peak_mem_source']!r} is not a budget input; pod evidence needs "
                         f"{CUDA_PEAK_MEM_SOURCE!r}")
    return r


def build_evidence(rows, *, short_window, tool, torch_version, source) -> dict:
    """The ``laned_t7/1`` evidence object. Every row is re-checked; ``fits``,
    ``armed``, ``short_window.identical`` and ``passed`` are recomputed here.
    With ``source == "pod"`` only ``cuda_max_allocated`` memory is accepted."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"source must be a non-empty string, got {source!r}")
    rows = [_recheck_row(i, r, source=source) for i, r in enumerate(rows)]
    if not rows:
        raise ValueError("T7 evidence needs at least one measured variant")
    rows.sort(key=lambda r: r["variant"])
    sw = dict(short_window)
    missing = [k for k in ("base_context", "declared_context", "n_windows", "max_abs_diff")
               if k not in sw]
    if missing:
        raise ValueError(f"short_window lacks {missing}")
    sw["identical"] = float(sw["max_abs_diff"]) == 0.0
    armed = [r["variant"] for r in rows if r["fits"]]
    measured = {r["variant"] for r in rows}
    passed = (sw["identical"] is True and set(LAUNCH_VARIANTS) <= measured
              and TRAINED_CONTEXT in armed)
    return {"schema": SCHEMA, "wp": WP, "source": source, "torch": str(torch_version),
            "tool": dict(tool), "measured_at": datetime.now(UTC).isoformat(),
            "rows": rows, "short_window": sw, "armed": armed, "passed": passed}


def _peak_rss_gb() -> float:
    """The process's peak resident set size (ru_maxrss: KiB on Linux, bytes on macOS)."""
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak *= 1024.0
    return peak / _GIB


def measure_variant(checkpoint_dir, histories, *, context, horizon, host, gpu, n_models,
                    budget, device) -> dict:
    """Time one lane D scorer run (load + forecast) of ``checkpoint_dir`` on
    ``histories`` at ``context``, project it to ``n_models`` per settlement,
    and record the peak device memory (``cuda_max_allocated``; on CPU the
    process peak RSS, labelled ``process_peak_rss``).

    A history shorter than ``context`` is a :class:`ShortHistoryError` before
    torch is imported or the checkpoint is read: the variant is never timed
    on a silently truncated input."""
    rows = [np.asarray(h, dtype=np.float64).reshape(-1) for h in histories]
    v = _check_setup(host=host, gpu=gpu, variant=context, budget=budget, n_models=n_models,
                     n_windows=len(rows))
    lengths = [r.size for r in rows]
    short = [(i, n) for i, n in enumerate(lengths) if n < v]
    if short:
        i, n = short[0]
        raise ShortHistoryError(f"{len(short)} of {len(lengths)} windows are shorter than the "
                                f"variant {v} (first: window {i}, {n} points); the no-pad "
                                f"scorer would run them below {v}")
    meta = json.loads((Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8"))
    plan = batch_plan(lengths, context=v, patch_size=int(meta["moe"]["patch_size"]))
    if list(plan) != [v]:
        raise ValueError(f"variant {v}: the batch plan runs input lengths {list(plan)}, "
                         f"not the variant alone")
    input_len = next(iter(plan))

    import torch

    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    forecast_quantiles(checkpoint_dir, rows, context=v, horizon=horizon, device=device)
    if on_cuda:
        torch.cuda.synchronize(device)
    per_model = time.perf_counter() - start
    if on_cuda:
        peak, source = torch.cuda.max_memory_allocated(device) / _GIB, CUDA_PEAK_MEM_SOURCE
    else:
        peak, source = _peak_rss_gb(), CPU_PEAK_MEM_SOURCE
    return variant_row(host=host, gpu=gpu, variant=v, input_len=input_len,
                       history_min=min(lengths), history_max=max(lengths),
                       wall_s=per_model * int(n_models), peak_mem_gb=peak,
                       peak_mem_source=source, budget=budget, n_models=n_models,
                       n_windows=len(rows))


def short_window_check(checkpoint_dir, histories, *, horizon, device: str = "cpu") -> dict:
    """The checkpoint at the trained 4096 vs a byte copy of it declared at
    131072, on windows with ≤ 4096 history: the no-pad scorer must give
    identical forecasts. A longer history is a ``ValueError`` before loading."""
    rows = [np.asarray(h, dtype=np.float64).reshape(-1) for h in histories]
    if not rows:
        raise ValueError("short_window_check needs at least one window")
    for i, r in enumerate(rows):
        if r.size > TRAINED_CONTEXT:
            raise ValueError(f"window {i}: {r.size} points of history; the short-window check "
                             f"takes ≤ {TRAINED_CONTEXT}")
    with tempfile.TemporaryDirectory(prefix="laned-t7-") as tmp:
        copy = shutil.copytree(Path(checkpoint_dir), Path(tmp) / f"declared-{DECLARED_CONTEXT}")
        base = forecast_quantiles(checkpoint_dir, rows, context=TRAINED_CONTEXT,
                                  horizon=horizon, device=device)
        other = forecast_quantiles(copy, rows, context=DECLARED_CONTEXT, horizon=horizon,
                                   device=device)
    diff = float(np.max(np.abs(base - other)))
    return {"base_context": TRAINED_CONTEXT, "declared_context": DECLARED_CONTEXT,
            "n_windows": len(rows), "max_abs_diff": diff, "identical": diff == 0.0}


__all__ = ["CPU_PEAK_MEM_SOURCE", "CUDA_PEAK_MEM_SOURCE", "LAUNCH_VARIANTS",
           "PEAK_MEM_SOURCES", "S8_HOST", "SCHEMA", "ShortHistoryError", "build_evidence",
           "measure_variant", "settlement_budget", "short_window_check", "variant_row"]
