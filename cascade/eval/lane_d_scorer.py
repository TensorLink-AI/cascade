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
  :data:`MAX_HORIZON` (64 patches × 32 = 2048, the band cap; training's CPM
  spans stop at 16 patches, so the long bands extrapolate). Both are refused before anything is loaded or torch imported.
* **No padding.** Each history is truncated to the largest patch multiple it
  really has, capped at the declared ``context``; histories of equal length
  are batched together (:func:`batch_plan`, one batch per length). A batch
  runs as deterministic micro-batches of at most :func:`forward_batch_size`
  windows (#785: ``windows x patches`` past ~65k overflows the CUDA grid in
  the pyramid's attention, and memory goes first), a pure function of the
  input length, horizon, patch size and :data:`MAX_FORWARD_TOKENS`. The
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
# Owner 2026-11-27: per-window GIFT-Eval horizons up to [lane_d] max_horizon
# (2048 = 64 patches of 32). The masked horizon is ONE forward block of
# ceil(h / 32) patches after the history (no autoregression); the pyramid
# reads no ``max_patches`` (xPos positions are computed for the sequence it
# is given), so a declared-4096 entry runs 128 + 64 = 192 patches at h = 2048
# with its full context. Training masked spans of at most cpm_c_max = 16
# patches: longer horizons are an extrapolation every model faces alike.
MAX_HORIZON = 64 * 32
# #785: patch tokens (windows x channels x patches, horizon patches included)
# one model forward may carry. The pyramid reshapes to (B * P, C, -1) before
# scaled_dot_product_attention; at ~65,536 rows the CUDA launch fails
# ("invalid configuration argument") and larger batches OOM first. Measured
# on the launch 4090 class: 62,400 / 33,000 tokens ran (<= 4.02 GiB peak),
# 66,000 failed. Half the 65,535 grid limit; a release constant, never a
# runtime memory probe, so every validator cuts the same micro-batches.
MAX_FORWARD_TOKENS = 32768


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
        raise ValueError(f"horizon {h} outside [1, {MAX_HORIZON}] (64 patches x 32)")
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


def forward_batch_size(input_len: int, horizon: int, patch_size: int, *, channels: int = 1,
                       max_tokens: int = MAX_FORWARD_TOKENS) -> int:
    """Windows per model forward (#785): ``max_tokens // (channels *
    patches)``, at least 1, where ``patches`` counts the history patches
    plus the ``ceil(horizon / patch_size)`` masked horizon patches.
    Deterministic in its arguments; ``ValueError`` on a non-positive input."""
    length = _as_int(input_len, "input length")
    h = _as_int(horizon, "horizon")
    ps = _as_int(patch_size, "patch_size")
    c = _as_int(channels, "channels")
    cap = _as_int(max_tokens, "max_tokens")
    if min(length, h, ps, c, cap) < 1:
        raise ValueError(f"forward_batch_size: non-positive input "
                         f"(len {length}, h {h}, patch {ps}, channels {c}, max_tokens {cap})")
    patches = -(-length // ps) + -(-h // ps)
    return max(1, cap // (c * patches))


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
    """``(B, L)`` univariate or ``(B, C, L)`` multichannel float64 histories
    (L a patch multiple) → ``(B, horizon, Q21)`` resp. ``(B, C, horizon,
    Q21)``, the pyramid head's quantiles in data units (the wrapper's
    ``_parameters`` arithmetic, without its pad). The C channels of a window
    are one joint forecast (the model's variate axis); each channel is
    standardized on its own, as in training. ``(B, L)`` is exactly the
    ``C = 1`` case, so a univariate batch runs the same arithmetic as before
    F2 (rev6.18 A4: bit-identical)."""
    import torch

    ps = int(cfg.patch_size)
    rows = np.asarray(rows)
    uni = rows.ndim == 2
    b, c, length = (rows.shape[0], 1, rows.shape[1]) if uni else rows.shape
    count = (int(horizon) + ps - 1) // ps
    context = torch.as_tensor(rows, dtype=torch.float64, device=device)
    context = context[:, None] if uni else context
    x = torch.cat([context, context.new_zeros(b, c, count * ps)], -1)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[..., length:] = True
    flat = x.reshape(b * c, -1)
    z, loc, scale = pm.causal_standardize(flat, mask.reshape_as(flat))
    z = z.masked_fill(mask.reshape_as(flat), 0).float().reshape(b, c, -1, ps)
    params = model(z, mask=mask.reshape(b, c, -1, ps), with_parameters=True)
    start = length // ps - 1
    params = params[:, :, start:start + count]
    loc = loc.reshape(b, c, -1, ps)[:, :, start:start + count, -1].float()
    scale = scale.reshape(b, c, -1, ps)[:, :, start:start + count, -1].float()
    q = pm.quantiles(params) * scale[..., None, None] + loc[..., None, None]
    if uni:
        out = q[:, 0].reshape(b, -1, q.shape[-1])[:, :int(horizon)]
    else:
        out = q.reshape(b, c, -1, q.shape[-1])[:, :, :int(horizon)]
    return out.cpu().numpy().astype(np.float64)


def _forward_chunked(pm, cfg, model, rows: np.ndarray, horizon: int, device: str,
                     max_tokens: int) -> np.ndarray:
    """:func:`_forward` over ``rows`` (``(B, ..., L)``, one input length) in
    contiguous micro-batches of :func:`forward_batch_size` windows, in order,
    concatenated along the window axis. In exact arithmetic a window's
    forward is independent of its batch (per-row standardization, per-series
    attention, per-token routing with no capacity drop); in fp32 the GEMM
    blocking depends on the batch rows, so the cut moves the last bits
    (~1e-8 relative, measured on CPU). The cut is therefore part of the
    deterministic recipe: fixed by :data:`MAX_FORWARD_TOKENS`, and each
    micro-batch equals forecasting its windows on their own. An input that
    fits under the cap is not cut (the pre-#785 forward, bit for bit)."""
    rows = np.asarray(rows)
    channels = int(np.prod(rows.shape[1:-1], dtype=np.int64)) if rows.ndim > 2 else 1
    step = forward_batch_size(rows.shape[-1], horizon, int(cfg.patch_size), channels=channels,
                              max_tokens=max_tokens)
    if rows.shape[0] <= step:
        return _forward(pm, cfg, model, rows, horizon, device)
    return np.concatenate([_forward(pm, cfg, model, rows[i:i + step], horizon, device)
                           for i in range(0, rows.shape[0], step)], axis=0)


def _as_window(hist, i: int) -> np.ndarray:
    """One history as a float64 array: 1-D ``(L,)`` (univariate; any 1-D
    shape as before F2) or 2-D ``(C, L)`` (multichannel, C ≥ 2; a ``(1, L)``
    history is the univariate ``(L,)``)."""
    a = np.asarray(hist, dtype=np.float64)
    if a.ndim == 2 and a.shape[0] > 1:
        row = a
    elif a.ndim <= 2:
        row = a.reshape(-1)
    else:
        raise ValueError(f"window {i}: history shape {a.shape} is neither (L,) nor (C, L)")
    if not np.isfinite(row).all():
        raise ValueError(f"window {i}: non-finite history")
    return row


def forecast_rows(histories) -> list[int]:
    """The forecast rows of each window in :func:`forecast_quantiles`'
    output: 1 per univariate window, C per ``(C, L)`` window (C ≥ 2)."""
    out = []
    for h in histories:
        shape = np.shape(h)
        out.append(int(shape[0]) if len(shape) == 2 and int(shape[0]) > 1 else 1)
    return out


def forecast_quantiles(checkpoint_dir: str | os.PathLike, histories, *, context: int,
                       horizon: int, contexts=None, device: str = "cpu",
                       max_tokens: int = MAX_FORWARD_TOKENS) -> np.ndarray:
    """``(n, horizon, 9)`` quantile forecasts of ``histories`` at
    :data:`SCORING_LEVELS`.

    rev6.18 A4 (F2): a history may be a ``(C, L)`` multichannel window
    (C ≥ 2), forecast jointly by one model call. The output then has one row
    per channel, window-major and channel-minor (:func:`forecast_rows`):
    ``(Σ rows, horizon, 9)``. Windows batch by (channels, input length); the
    univariate windows form exactly the pre-F2 per-length batches, so a
    univariate forecast is bit-identical with or without multichannel windows
    in the request. The channel pick (empty columns, the 32 cap) happens
    box-side before the request (:mod:`cascade.eval.lane_d_channels`).

    ``context`` is the context to run at (an entry's
    :func:`effective_context`) and ``contexts`` the settlement's armed set:
    with it ``context`` must pass :func:`declared_context`; without it
    ``context`` must still be in
    :data:`SCORER_CONTEXTS`. ``horizon`` must be in ``[1, MAX_HORIZON]``.
    Both checks run before anything is loaded or torch is imported. Each
    window's model input is its last :func:`batch_plan` length points plus
    ``ceil(h/32)`` masked horizon patches, one batch per length, run in
    micro-batches of :func:`forward_batch_size` windows (``max_tokens``
    patch tokens per forward; #785). Raises
    ``ValueError`` on a history shorter than one patch, a non-finite history
    or a non-finite forecast."""
    ctx = declared_context(context, SCORER_CONTEXTS if contexts is None else contexts)
    h = _checked_horizon(horizon)
    tokens = _as_int(max_tokens, "max_tokens")
    if tokens < 1:
        raise ValueError(f"max_tokens must be positive, got {tokens}")
    d = Path(checkpoint_dir)
    rows = [_as_window(hist, i) for i, hist in enumerate(histories)]
    if not rows:
        return np.empty((0, h, len(SCORING_LEVELS)), dtype=np.float64)
    # The patch size is read from the checkpoint's own config (json only), so
    # a too-short history is refused before torch or the weights load.
    meta = json.loads((d / "config.json").read_text(encoding="utf-8"))
    ps = int(meta["moe"]["patch_size"])
    n_ch = [1 if r.ndim == 1 else int(r.shape[0]) for r in rows]
    # One no-pad plan per channel count: C = 1 is the univariate plan, the
    # same batches as before F2.
    plans: dict[int, dict[int, list[int]]] = {}
    for c in sorted(set(n_ch)):
        members = [i for i, k in enumerate(n_ch) if k == c]
        sub = batch_plan([rows[i].shape[-1] for i in members], context=ctx, patch_size=ps)
        plans[c] = {length: [members[j] for j in idx] for length, idx in sub.items()}
    offsets = np.concatenate([[0], np.cumsum(n_ch)]).astype(np.int64)

    import torch

    pm, cfg, model = _load_model(d, ctx, device)
    picks = np.asarray(level_indices(pm.QUANTILE_LEVELS), dtype=np.int64)
    out = np.empty((int(offsets[-1]), h, len(picks)), dtype=np.float64)
    with torch.no_grad():
        for c, plan in plans.items():
            for length, idx in plan.items():
                q = _forward_chunked(pm, cfg, model,
                                     np.stack([rows[i][..., -length:] for i in idx]), h, device,
                                     tokens)
                if c == 1:
                    out[offsets[idx]] = q[:, :, picks]
                    continue
                for j, i in enumerate(idx):
                    out[offsets[i]:offsets[i + 1]] = q[j][:, :, picks]
    if not np.isfinite(out).all():
        raise ValueError("non-finite lane D quantiles")
    return out


def score_window(q: np.ndarray, target: np.ndarray, history: np.ndarray,
                 seasonal_period: int) -> tuple[float, float]:
    """``(crps, mase)`` of one ``(horizon, 9)`` forecast: ``crps`` is the
    window's weighted quantile loss over the grid (``2·mean_q pinball /
    Σ|y|``), ``mase`` scores the 0.5 quantile. ``ValueError`` when the
    forecast is malformed or the target has no scale (``Σ|y| = 0``).

    rev6.18 A4 (F2, DEC-CA-0041): a ``(C, horizon, 9)`` forecast of a
    ``(C, horizon)`` target with a ``(C, L)`` history (C ≥ 2) is ONE
    per-window score: each channel's ``crps`` and ``mase`` as above, then
    the arithmetic mean over channels; ``crps`` averages only the channels
    with ``Σ|y_c| > 0`` (zero-target channels are masked from the quantile
    half, as lane A's ``wql_per_window`` does), ``mase`` averages every
    channel. ``ValueError`` when no channel has scale. A 2-D call never
    touches the univariate path, which is unchanged."""
    q = np.asarray(q, dtype=np.float64)
    if q.ndim == 3:
        return _score_multichannel(q, target, history, seasonal_period)
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


def _score_multichannel(q: np.ndarray, target, history, seasonal_period: int
                        ) -> tuple[float, float]:
    from .crps import wql_per_window

    y = np.asarray(target, dtype=np.float64)
    hist = np.asarray(history, dtype=np.float64)
    c = q.shape[0]
    if (c < 2 or y.ndim != 2 or y.shape[0] != c or hist.ndim != 2 or hist.shape[0] != c
            or q.shape != (c, y.shape[1], len(SCORING_LEVELS)) or not np.isfinite(q).all()):
        raise ValueError(f"multichannel forecast {q.shape} / target {y.shape} / history "
                         f"{hist.shape} invalid")
    # (num_q, C, H) quantiles against the (C, H) target: one row per channel.
    qloss, abs_t = pinball_components(np.transpose(q, (2, 0, 1)), y, SCORING_LEVELS)
    wql, valid = wql_per_window(qloss, abs_t)
    if not bool(valid.any()):
        raise ValueError("target has no scale")
    crps = float(wql[valid].mean())
    sp = max(1, int(seasonal_period))
    m = float(np.mean([float(mase_one(q[k, :, MEDIAN_INDEX], y[k], hist[k], sp))
                       for k in range(c)]))
    return crps, m


__all__ = ["MAX_FORWARD_TOKENS", "MAX_HORIZON", "MEDIAN_INDEX", "SCORER_CONTEXTS",
           "SCORING_LEVELS", "SCORING_METHOD", "batch_plan", "declared_context",
           "forecast_quantiles", "forecast_rows", "forward_batch_size", "level_indices",
           "score_window", "usable_length"]
