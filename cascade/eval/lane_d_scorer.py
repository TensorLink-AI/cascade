"""Lane D scorer (LANE_D_PLAN §1.1, §1.5; W5d).

* **Quantile head, 9-level grid.** The model's own quantile head is read at
  :data:`SCORING_LEVELS` (picked out of the head's Q21 by exact level match,
  nothing interpolated), as lane A does under ``[v2] quantile_head_after``.
  No copula sampling, no Monte-Carlo noise.
* **Declared contexts.** A forecast runs only at a context in
  :data:`SCORER_CONTEXTS` and, when the armed ``[lane_d] contexts`` are
  given, in those too (:func:`declared_context`). Horizons are capped at
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
# and its 2×…32× extensions. ``[lane_d] contexts`` arms a subset of these
# (launch: 4k/8k/16k; 32k–128k by overlay at S8 after T7).
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
    """``int(context)`` when it is an armed context (in ``contexts``) and a
    scorer context (in :data:`SCORER_CONTEXTS`); ``ValueError`` otherwise."""
    c = _as_int(context, "context")
    armed = {_as_int(x, "armed context") for x in contexts}
    if c not in armed:
        raise ValueError(f"context {c} is not an armed [lane_d] context {sorted(armed)}")
    if c not in SCORER_CONTEXTS:
        raise ValueError(f"context {c} is not a scorer context {list(SCORER_CONTEXTS)}")
    return c


def _checked_horizon(horizon) -> int:
    h = _as_int(horizon, "horizon")
    if not 1 <= h <= MAX_HORIZON:
        raise ValueError(f"horizon {h} outside [1, {MAX_HORIZON}] (16 CPM patches x 32)")
    return h


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

    ``contexts`` is the armed ``[lane_d] contexts``: with it ``context`` must
    pass :func:`declared_context`; without it ``context`` must still be in
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
