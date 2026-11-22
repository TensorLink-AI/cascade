"""Lane D scorer (LANE_D_PLAN §1.1, §1.5; W5d deepens it).

* **Quantile head, 9-level grid.** The model's own quantile head is read at
  :data:`SCORING_LEVELS` (picked out of the head's Q21 by exact level match,
  nothing interpolated), as lane A does under ``[v2] quantile_head_after``.
  No copula sampling, no Monte-Carlo noise.
* **No padding.** Each history is truncated to the largest patch multiple it
  really has, capped at the declared ``context``; histories of equal length
  are batched together. The pinned wrapper's edge-pad is never used.
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
import os
from pathlib import Path

import numpy as np

from .crps import DEFAULT_QUANTILE_LEVELS, pinball_components
from .mase import mase as mase_one

SCORING_LEVELS: tuple[float, ...] = tuple(float(q) for q in DEFAULT_QUANTILE_LEVELS)
MEDIAN_INDEX = SCORING_LEVELS.index(0.5)
SCORING_METHOD = "quantile_head"


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


def _load_model(checkpoint_dir: Path, context: int):
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
    return pm, cfg, model


def _forward(pm, cfg, model, rows: np.ndarray, horizon: int) -> np.ndarray:
    """``(B, L)`` float64 histories (L a patch multiple) → ``(B, horizon, Q21)``,
    the pyramid head's quantiles in data units (the wrapper's arithmetic,
    without its pad)."""
    import torch

    ps = int(cfg.patch_size)
    b, length = rows.shape
    count = (int(horizon) + ps - 1) // ps
    context = torch.as_tensor(rows, dtype=torch.float64)[:, None]
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
                       horizon: int) -> np.ndarray:
    """``(n, horizon, 9)`` quantile forecasts of ``histories`` at
    :data:`SCORING_LEVELS`; raises ``ValueError`` on a history shorter than
    one patch, a non-finite history or a non-finite forecast."""
    import torch

    if int(horizon) < 1 or int(context) < 1:
        raise ValueError("horizon and context must be positive")
    d = Path(checkpoint_dir)
    pm, cfg, model = _load_model(d, int(context))
    picks = np.asarray(level_indices(pm.QUANTILE_LEVELS), dtype=np.int64)
    rows = []
    for h in histories:
        row = np.asarray(h, dtype=np.float64).reshape(-1)
        if not np.isfinite(row).all():
            raise ValueError("non-finite history")
        n = usable_length(row.size, int(context), int(cfg.patch_size))
        if n < int(cfg.patch_size):
            raise ValueError(f"history of {row.size} points is shorter than one patch")
        rows.append(row[-n:])
    out = np.empty((len(rows), int(horizon), len(picks)), dtype=np.float64)
    by_len: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_len.setdefault(r.size, []).append(i)
    with torch.no_grad():
        for length in sorted(by_len):
            idx = by_len[length]
            q = _forward(pm, cfg, model, np.stack([rows[i] for i in idx]), int(horizon))
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


__all__ = ["MEDIAN_INDEX", "SCORING_LEVELS", "SCORING_METHOD", "forecast_quantiles",
           "level_indices", "score_window", "usable_length"]
