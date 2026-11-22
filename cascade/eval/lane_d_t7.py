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

:func:`measure_variant` and :func:`short_window_check` need torch (they run
the scorer); everything else is plain Python.
"""

from __future__ import annotations

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
from .lane_d_scorer import SCORER_CONTEXTS, forecast_quantiles

SCHEMA = "laned_t7/1"
WP = "W5d"
LAUNCH_VARIANTS: tuple[int, ...] = (4096, 8192, 16384)
S8_HOST = "valeval-laned"          # 96 GB, rented only at S8 for 32k–128k
MAX_LAUNCH_VARIANT = max(LAUNCH_VARIANTS)
TRAINED_CONTEXT = 4096             # 128 patches: the training row length
DECLARED_CONTEXT = max(SCORER_CONTEXTS)
MIN_MODELS = 10                    # up to 8 entries + KING_D + house per settlement
_GIB = float(2 ** 30)


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


def variant_row(*, host, gpu, variant, wall_s, peak_mem_gb, budget, n_models,
                n_windows) -> dict:
    """One measured variant; ``fits`` is derived from the numbers."""
    v = _check_setup(host=host, gpu=gpu, variant=variant, budget=budget, n_models=n_models,
                     n_windows=n_windows)
    for name, value in (("wall_s", wall_s), ("peak_mem_gb", peak_mem_gb)):
        if not _positive(value):
            raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return {"host": host, "gpu": gpu, "variant": v, "wall_s": float(wall_s),
            "peak_mem_gb": float(peak_mem_gb), "budget": dict(budget),
            "fits": _fits(wall_s, peak_mem_gb, budget), "n_models": _int(n_models, "n_models"),
            "n_windows": _int(n_windows, "n_windows")}


def build_evidence(rows, *, short_window, tool, torch_version, source) -> dict:
    """The ``laned_t7/1`` evidence object. ``fits``, ``armed``,
    ``short_window.identical`` and ``passed`` are recomputed here."""
    rows = [dict(r) for r in rows]
    if not rows:
        raise ValueError("T7 evidence needs at least one measured variant")
    for r in rows:
        r["fits"] = _fits(r["wall_s"], r["peak_mem_gb"], r["budget"])
    rows.sort(key=lambda r: r["variant"])
    sw = dict(short_window)
    missing = [k for k in ("base_context", "declared_context", "n_windows", "max_abs_diff")
               if k not in sw]
    if missing:
        raise ValueError(f"short_window lacks {missing}")
    sw["identical"] = float(sw["max_abs_diff"]) == 0.0
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"source must be a non-empty string, got {source!r}")
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
    and record the peak device memory (on CPU, the process peak RSS)."""
    import torch

    histories = list(histories)
    _check_setup(host=host, gpu=gpu, variant=context, budget=budget, n_models=n_models,
                 n_windows=len(histories))
    on_cuda = str(device).startswith("cuda")
    if on_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    forecast_quantiles(checkpoint_dir, histories, context=context, horizon=horizon,
                       device=device)
    if on_cuda:
        torch.cuda.synchronize(device)
    per_model = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) / _GIB if on_cuda else _peak_rss_gb()
    return variant_row(host=host, gpu=gpu, variant=context, wall_s=per_model * int(n_models),
                       peak_mem_gb=peak, budget=budget, n_models=n_models,
                       n_windows=len(histories))


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


__all__ = ["LAUNCH_VARIANTS", "S8_HOST", "SCHEMA", "build_evidence", "measure_variant",
           "settlement_budget", "short_window_check", "variant_row"]
