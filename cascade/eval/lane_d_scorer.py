"""Lane D scorer (LANE_D_PLAN §1.1, §1.5; W5d).

* **Quantile head, 9-level grid.** The model's own quantile head is read at
  :data:`SCORING_LEVELS` (picked out of the head's Q21 by exact level match,
  nothing interpolated), as lane A does under ``[v2] quantile_head_after``.
  No copula sampling, no Monte-Carlo noise.
* **Declared, armed and effective contexts (rev6.6).** ``[lane_d]
  contexts`` is the set a miner may DECLARE (every variant from launch).
  :func:`armed_record` computes, per settlement, which variants are ARMED
  (readiness and T7 gates, from the pinned documents' bytes); an entry is
  scored at :func:`effective_context`: its declared context when armed, else
  the highest armed variant. A forecast runs only at a context in
  :data:`SCORER_CONTEXTS` and, when the armed set is given, in it too
  (:func:`declared_context`). Horizons are capped at
  :data:`MAX_HORIZON` (16 CPM patches × 32: longer spans are outside
  training). Both are refused before anything is loaded or torch imported.
* **No padding.** Each history is truncated to the largest patch multiple it
  really has, capped at the declared ``context``; histories of equal length
  are batched together (:func:`batch_plan`, one forward per length). The
  pinned wrapper's edge-pad is never used, so a 4k model and a declared-128k
  copy of it score identically on windows with ≤ 4096 history (T7).
* **Context is scorer-side.** The model is built from the checkpoint's own
  ``config.json`` with ``context_length`` set to the declared context; the
  weights fit any context (non-persistent positional buffers).
* **fp32.** F32 and BF16 checkpoints load alike: floating tensors are upcast
  to fp32 before ``load_state_dict(strict=True)``.
* **Repo code only.** The model class is this release's
  ``cascade.trainer.pyramid_model``; no file from the checkpoint is imported.

:func:`score_window` turns a ``(horizon, 9)`` forecast into the per-window
``crps`` (weighted quantile loss over the grid) and ``mase`` (0.5 quantile)
the lane D report carries. It is numpy only, so the box-side settlement uses
it without torch.
"""

from __future__ import annotations

import hashlib
import json
import operator
import os
from pathlib import Path

import numpy as np

from .crps import DEFAULT_QUANTILE_LEVELS, pinball_components
from .mase import mase as mase_one

SCORING_LEVELS: tuple[float, ...] = tuple(float(q) for q in DEFAULT_QUANTILE_LEVELS)
MEDIAN_INDEX = SCORING_LEVELS.index(0.5)
SCORING_METHOD = "quantile_head"

# Every context the scorer builds a model at: the trained 4096 (128 patches)
# and its 2×…32× extensions. ``[lane_d] contexts`` (rev6.6: all six from
# launch) is the declarable set; each variant arms per settlement on its two
# gates (:func:`armed_record`), never by overlay.
SCORER_CONTEXTS: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536, 131072)
# CPM masks contiguous spans of at most cpm_c_max = 16 patches of 32 points.
MAX_HORIZON = 16 * 32


def _as_int(value, what: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{what} must be an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{what} must be an integer, got {value!r}") from None


def declared_context(context, contexts) -> int:
    """``int(context)`` when it is in ``contexts`` (the set the forecast may
    run at: the settlement's armed set, or the declarable ``[lane_d]
    contexts``) and a scorer context (in :data:`SCORER_CONTEXTS`);
    ``ValueError`` otherwise."""
    c = _as_int(context, "context")
    allowed = {_as_int(x, "allowed context") for x in contexts}
    if c not in allowed:
        raise ValueError(f"context {c} is not in the allowed lane D contexts {sorted(allowed)}")
    if c not in SCORER_CONTEXTS:
        raise ValueError(f"context {c} is not a scorer context {list(SCORER_CONTEXTS)}")
    return c


def _checked_horizon(horizon) -> int:
    h = _as_int(horizon, "horizon")
    if not 1 <= h <= MAX_HORIZON:
        raise ValueError(f"horizon {h} outside [1, {MAX_HORIZON}] (16 CPM patches x 32)")
    return h


# ── per-variant arming (plan rev6.6, §1.1 context bullet; W5d) ──────────────
# Every [lane_d] contexts value is declarable from launch. A variant is ARMED
# at a settlement only when both gates pass: (1) readiness, at least
# ``min_long_series`` series from at least GATE_MIN_SOURCES sources can use it
# (the W1/R16 per-rung counts, a ``laned_readiness/1`` document); (2) T7, its
# wall time and memory fit the settlement budget on the eval host (the
# ``laned_t7/1`` pod document). Pure and numpy-free: the box-side settlement
# (W7a) records the result, with the documents' sha256s, in the signed report.

GATE_MIN_SOURCES = 20
READINESS_SCHEMA = "laned_readiness/1"
T7_SCHEMA = "laned_t7/1"
_T7_POD_SOURCE = "pod"
_T7_PEAK_SOURCE = "cuda_max_allocated"


def _doc(raw, what: str) -> dict:
    """Strict JSON object from the pinned document bytes (NaN refused)."""
    if not isinstance(raw, (bytes, bytearray)):
        raise ValueError(f"{what} must be the document bytes, got {type(raw).__name__}")

    def _no_constant(name):
        raise ValueError(f"{what}: non-finite number {name}")

    try:
        doc = json.loads(bytes(raw).decode("utf-8"), parse_constant=_no_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{what} is not JSON: {exc}") from None
    if not isinstance(doc, dict):
        raise ValueError(f"{what} is not a JSON object")
    return doc


def _count(value, what: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{what} must be a non-negative integer, got {value!r}")
    return value


def _readiness_counts(raw) -> dict[int, tuple[int, int]]:
    """``{context: (series, sources)}`` from a ``laned_readiness/1`` document."""
    doc = _doc(raw, "readiness")
    if doc.get("schema") != READINESS_SCHEMA:
        raise ValueError(f"readiness schema must be {READINESS_SCHEMA!r}, got {doc.get('schema')!r}")
    rungs = doc.get("rungs")
    if not isinstance(rungs, list):
        raise ValueError("readiness has no rungs list")
    out: dict[int, tuple[int, int]] = {}
    for i, rung in enumerate(rungs):
        if not isinstance(rung, dict):
            raise ValueError(f"readiness rung {i} is not an object")
        c = rung.get("context")
        if type(c) is not int or c not in SCORER_CONTEXTS:
            raise ValueError(f"readiness rung {i}: context {c!r} is not a scorer context")
        if c in out:
            raise ValueError(f"readiness rung {i}: duplicate context {c}")
        out[c] = (_count(rung.get("series"), f"rung {c} series"),
                  _count(rung.get("sources"), f"rung {c} sources"))
    return out


def _finite_number(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return float(value)


def _t7_fits(raw) -> dict[int, bool]:
    """``{variant: passes gate 2}`` from a ``laned_t7/1`` document. ``fits``
    is recomputed from each row's numbers; a stated value that disagrees is a
    hand edit and is refused. Only pod evidence measured with each variant's
    own CUDA peak is a budget input."""
    doc = _doc(raw, "t7")
    if doc.get("schema") != T7_SCHEMA:
        raise ValueError(f"t7 schema must be {T7_SCHEMA!r}, got {doc.get('schema')!r}")
    rows = doc.get("rows")
    if not isinstance(rows, list):
        raise ValueError("t7 has no rows list")
    pod = doc.get("source") == _T7_POD_SOURCE
    out: dict[int, bool] = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"t7 row {i} is not an object")
        v = row.get("variant")
        if type(v) is not int or v not in SCORER_CONTEXTS:
            raise ValueError(f"t7 row {i}: variant {v!r} is not a scorer context")
        if v in out:
            raise ValueError(f"t7 row {i}: duplicate variant {v}")
        budget = row.get("budget")
        if not isinstance(budget, dict):
            raise ValueError(f"t7 row {v}: no budget")
        fits = (_finite_number(row.get("wall_s"), f"t7 row {v} wall_s")
                <= _finite_number(budget.get("wall_s"), f"t7 row {v} budget.wall_s")
                and _finite_number(row.get("peak_mem_gb"), f"t7 row {v} peak_mem_gb")
                <= _finite_number(budget.get("mem_gb"), f"t7 row {v} budget.mem_gb"))
        if row.get("fits") is not fits:
            raise ValueError(f"t7 row {v}: stated fits {row.get('fits')!r} disagrees with "
                             f"its numbers ({fits})")
        exercised = row.get("input_len") == v and type(row.get("history_min")) is int \
            and row["history_min"] >= v
        out[v] = bool(fits and pod and exercised
                      and row.get("peak_mem_source") == _T7_PEAK_SOURCE)
    return out


def armed_record(contexts, *, readiness, t7, min_long_series) -> dict:
    """The armed variants of ``contexts`` and the evidence they were armed on.

    ``readiness`` / ``t7`` are the pinned ``laned_readiness/1`` and
    ``laned_t7/1`` document bytes (``None`` when absent: that gate fails for
    every variant). Returns ``{"armed_contexts": [...], "evidence":
    {"min_long_series", "min_sources", "readiness_sha256", "t7_sha256",
    "variants": [{"context", "series", "sources", "t7_fits", "armed"}]}}``,
    ascending by context. Deterministic in its inputs. ``ValueError`` on a
    malformed document or a context outside :data:`SCORER_CONTEXTS`."""
    ctxs = sorted({_as_int(c, "context") for c in contexts})
    if not ctxs or any(c not in SCORER_CONTEXTS for c in ctxs):
        raise ValueError(f"contexts {ctxs} must be a non-empty subset of {list(SCORER_CONTEXTS)}")
    floor = _count(_as_int(min_long_series, "min_long_series"), "min_long_series")
    counts = {} if readiness is None else _readiness_counts(readiness)
    fits = {} if t7 is None else _t7_fits(t7)
    variants = []
    for c in ctxs:
        series, sources = counts.get(c, (0, 0))
        t7_ok = bool(fits.get(c, False))
        variants.append({"context": c, "series": series, "sources": sources, "t7_fits": t7_ok,
                         "armed": series >= floor and sources >= GATE_MIN_SOURCES and t7_ok})
    return {"armed_contexts": [v["context"] for v in variants if v["armed"]],
            "evidence": {
                "min_long_series": floor, "min_sources": GATE_MIN_SOURCES,
                "readiness_sha256": None if readiness is None
                else hashlib.sha256(bytes(readiness)).hexdigest(),
                "t7_sha256": None if t7 is None else hashlib.sha256(bytes(t7)).hexdigest(),
                "variants": variants}}


def effective_context(declared, armed_contexts) -> int | None:
    """The context an entry is scored at: ``declared`` when it is armed, else
    the highest armed variant below ``declared``, else ``None`` (plan rev6.15,
    owner 2026-11-23, "Highest ≤ declared, else skip": no armed variant is ≤
    ``declared``, so the entry is not scored this settlement and the
    settlement skips it ``context_below_armed``). A model never runs past the
    context its miner declared. Rev6.6: an unarmed declaration is valid and is
    scored at its own context from the settlement its variant arms.
    ``ValueError`` when nothing is armed or ``declared`` is not a scorer
    context."""
    d = _as_int(declared, "declared context")
    if d not in SCORER_CONTEXTS:
        raise ValueError(f"declared context {d} is not a scorer context {list(SCORER_CONTEXTS)}")
    armed = {_as_int(c, "armed context") for c in armed_contexts}
    if not armed:
        raise ValueError("no armed lane D context: nothing can be scored")
    if d in armed:
        return d
    below = [c for c in armed if c < d]
    return max(below) if below else None


def level_indices(head_levels, wanted=SCORING_LEVELS, tol: float = 1e-9) -> list[int]:
    """Position of every ``wanted`` level in ``head_levels`` (exact match)."""
    have = [float(v) for v in head_levels]
    out = []
    for q in wanted:
        hits = [i for i, v in enumerate(have) if abs(v - float(q)) <= tol]
        if not hits:
            raise ValueError(f"quantile head lacks scoring level {q}")
        out.append(hits[0])
    return out


def usable_length(n_points: int, context: int, patch_size: int) -> int:
    """The no-pad history length: the largest patch multiple ≤ min(n, context)."""
    return (min(int(n_points), int(context)) // int(patch_size)) * int(patch_size)


def batch_plan(history_lengths, *, context: int, patch_size: int) -> dict[int, list[int]]:
    """The no-pad plan: ``{model input length: [window indices]}``, keys and
    indices ascending. A window's input length is its :func:`usable_length`;
    nothing is padded up. A history shorter than one patch is a ``ValueError``."""
    ps = _as_int(patch_size, "patch_size")
    ctx = _as_int(context, "context")
    if ps < 1 or ctx < ps:
        raise ValueError(f"context {ctx} / patch_size {ps} invalid")
    plan: dict[int, list[int]] = {}
    for i, n in enumerate(history_lengths):
        n = _as_int(n, "history length")
        if n < ps:
            raise ValueError(f"window {i}: history of {n} points is shorter than one patch ({ps})")
        plan.setdefault(usable_length(n, ctx, ps), []).append(i)
    return {length: plan[length] for length in sorted(plan)}


def _load_model(checkpoint_dir: Path, context: int, device: str):
    import torch
    from safetensors.torch import load_file

    from ..trainer import pyramid_model as pm

    meta = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    raw = dict(meta["moe"])
    raw["context_length"] = int(context)
    cfg = pm.CascadeMoEConfig.from_dict(raw)
    model = pm.CascadeMoEModel(cfg).eval()
    state = load_file(str(checkpoint_dir / "weights.safetensors"))
    state = {k: (v.to(torch.float32) if v.is_floating_point() else v) for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return pm, cfg, model.to(device)


def _forward(pm, cfg, model, rows: np.ndarray, horizon: int, device: str) -> np.ndarray:
    """``(B, L)`` float64 histories (L a patch multiple) → ``(B, horizon, Q21)``,
    the pyramid head's quantiles in data units (the wrapper's arithmetic,
    without its pad)."""
    import torch

    ps = int(cfg.patch_size)
    b, length = rows.shape
    count = (int(horizon) + ps - 1) // ps
    context = torch.as_tensor(rows, dtype=torch.float64, device=device)[:, None]
    x = torch.cat([context, context.new_zeros(b, 1, count * ps)], -1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., length:] = True
    flat = x.reshape(b, -1)
    z, loc, scale = pm.causal_standardize(flat, mask.reshape_as(flat))
    z = z.masked_fill(mask.reshape_as(flat), 0).float().reshape(b, 1, -1, ps)
    params = model(z, mask=mask.reshape(b, 1, -1, ps), with_parameters=True)
    start = length // ps - 1
    params = params[:, :, start:start + count]
    loc = loc.reshape(b, 1, -1, ps)[:, :, start:start + count, -1].float()
    scale = scale.reshape(b, 1, -1, ps)[:, :, start:start + count, -1].float()
    q = pm.quantiles(params) * scale[..., None, None] + loc[..., None, None]
    out = q[:, 0].reshape(b, -1, q.shape[-1])[:, :int(horizon)]
    return out.cpu().numpy().astype(np.float64)


def forecast_quantiles(checkpoint_dir: str | os.PathLike, histories, *, context: int,
                       horizon: int, contexts=None, device: str = "cpu") -> np.ndarray:
    """``(n, horizon, 9)`` quantile forecasts of ``histories`` at
    :data:`SCORING_LEVELS`.

    ``context`` is the context to run at (an entry's
    :func:`effective_context`) and ``contexts`` the settlement's armed set:
    with it ``context`` must pass :func:`declared_context`; without it
    ``context`` must still be in
    :data:`SCORER_CONTEXTS`. ``horizon`` must be in ``[1, MAX_HORIZON]``.
    Both checks run before anything is loaded or torch is imported. Each
    window's model input is its last :func:`batch_plan` length points plus
    ``ceil(h/32)`` masked horizon patches, one forward per length. Raises
    ``ValueError`` on a history shorter than one patch, a non-finite history
    or a non-finite forecast."""
    ctx = declared_context(context, SCORER_CONTEXTS if contexts is None else contexts)
    h = _checked_horizon(horizon)
    d = Path(checkpoint_dir)
    rows = []
    for i, hist in enumerate(histories):
        row = np.asarray(hist, dtype=np.float64).reshape(-1)
        if not np.isfinite(row).all():
            raise ValueError(f"window {i}: non-finite history")
        rows.append(row)
    if not rows:
        return np.empty((0, h, len(SCORING_LEVELS)), dtype=np.float64)
    # The patch size is read from the checkpoint's own config (json only), so
    # a too-short history is refused before torch or the weights load.
    meta = json.loads((d / "config.json").read_text(encoding="utf-8"))
    plan = batch_plan([r.size for r in rows], context=ctx, patch_size=int(meta["moe"]["patch_size"]))

    import torch

    pm, cfg, model = _load_model(d, ctx, device)
    picks = np.asarray(level_indices(pm.QUANTILE_LEVELS), dtype=np.int64)
    out = np.empty((len(rows), h, len(picks)), dtype=np.float64)
    with torch.no_grad():
        for length, idx in plan.items():
            q = _forward(pm, cfg, model, np.stack([rows[i][-length:] for i in idx]), h, device)
            out[idx] = q[:, :, picks]
    if not np.isfinite(out).all():
        raise ValueError("non-finite lane D quantiles")
    return out


def score_window(q: np.ndarray, target: np.ndarray, history: np.ndarray,
                 seasonal_period: int) -> tuple[float, float]:
    """``(crps, mase)`` of one ``(horizon, 9)`` forecast: ``crps`` is the
    window's weighted quantile loss over the grid (``2·mean_q pinball /
    Σ|y|``), ``mase`` scores the 0.5 quantile. ``ValueError`` when the
    forecast is malformed or the target has no scale (``Σ|y| = 0``)."""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    if q.shape != (y.size, len(SCORING_LEVELS)) or not np.isfinite(q).all():
        raise ValueError(f"forecast shape {q.shape} / finiteness invalid for horizon {y.size}")
    qloss, abs_t = pinball_components(q.T[:, None, :], y[None, :], SCORING_LEVELS)
    if not float(abs_t[0]) > 0.0:
        raise ValueError("target has no scale")
    crps = float(2.0 * qloss[0].mean() / float(abs_t[0]))
    m = float(mase_one(q[:, MEDIAN_INDEX], y, np.asarray(history, dtype=np.float64).reshape(-1),
                       max(1, int(seasonal_period))))
    return crps, m


__all__ = ["MAX_HORIZON", "MEDIAN_INDEX", "SCORER_CONTEXTS", "SCORING_LEVELS", "SCORING_METHOD",
           "batch_plan", "declared_context", "forecast_quantiles", "level_indices",
           "score_window", "usable_length"]
