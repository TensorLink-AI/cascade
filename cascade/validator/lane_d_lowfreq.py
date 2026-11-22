"""The §1.3b A10 always-run low-frequency non-inferiority check (W3b5;
LANE_D_PLAN §1.3b, §1.7, §4 T4b).

Every crowning candidate (a verdict that cleared at stage ``confirm``) is
checked at every settlement, independent of the §1.3b gate draw:

- **The slice** is the newest private-pool snapshot's windows whose ``freq``
  is in ``[lane_d] lowfreq_freqs`` (exact string match). It is NOT limited to
  the forward-only frozen set: daily-or-slower series almost never pass the
  §1.3 forward rule (§1.7), so A10 uses the private pool instead. A slice
  window counts only when its target is usable (finite, non-zero, the
  configured horizon) and the candidate, KING_D and the house all have a
  usable forecast that the lane D scorer scores to finite CRPS and MASE; any
  other window is left out of the slice (deterministically, never scored on
  partial inputs). ``n_windows`` is the size of that slice.
- **The test**: per reference (``king_d``, ``house``) and metric (``crps``,
  ``mase``), the paired percentile LCB_α of the candidate's relative
  improvement ``1 − geomean(candidate)/geomean(reference)`` over the slice
  windows (:func:`cascade.eval.expert_lane.paired_geomean_lcb`, windows
  resampled with replacement, the bootstrap seeded by the settlement seed,
  the candidate and the reference digest). ``pass`` iff all four LCBs are
  ``≥ −tau_public``; otherwise ``fail`` (the caller maps it to
  ``lowfreq_regression``). When the house holds KING_D both references are
  the house and ``lcb.house == lcb.king_d`` exactly.
- **No floor beyond one window**: an empty slice is ``no_windows`` and never
  blocks (with no window there is no test); one window or more is evaluated.

Light imports only (numpy + the scorer's numpy half + the generic paired
bootstrap): this module is on the validator package's import path.
"""

from __future__ import annotations

import math

import numpy as np

from ..eval.expert_lane import paired_geomean_lcb
from ..eval.lane_d_scorer import SCORING_LEVELS, score_window

LOWFREQ_REFS = ("king_d", "house")
LOWFREQ_METRICS = ("crps", "mase")


def target_usable(w: dict, horizon: int) -> bool:
    """The stored target is a finite, non-zero ``(horizon,)`` vector."""
    t = w["target"]
    return (t.shape == (horizon,) and bool(np.isfinite(t).all())
            and float(np.abs(t).sum()) > 0.0)


def forecast_ok(fc: dict | None, wid: str, horizon: int) -> bool | None:
    """True = a usable forecast, None = absent, False = malformed (a fault)."""
    if fc is None or wid not in fc:
        return None
    q = np.asarray(fc[wid], dtype=np.float64)
    return bool(q.shape == (horizon, len(SCORING_LEVELS)) and np.isfinite(q).all())


def lowfreq_windows(pool: dict, cfg) -> list[dict]:
    """The candidate slice of one loaded snapshot: its windows whose ``freq``
    is exactly one of ``cfg.lowfreq_freqs``, one per window id (the first
    listed), in window-id order."""
    freqs = {str(f) for f in cfg.lowfreq_freqs}
    out: dict[str, dict] = {}
    for w in pool["windows"]:
        wid = w["window_id"]
        if wid and w["freq"] in freqs and wid not in out:
            out[wid] = w
    return [out[k] for k in sorted(out)]


def _block(result: str, n: int, cfg) -> dict:
    return {"result": result, "freqs": [str(f) for f in cfg.lowfreq_freqs], "n_windows": int(n),
            "tau_public": float(cfg.tau_public)}


def _scored(windows: list[dict], forecasts: dict, digests: list[str], horizon: int
            ) -> tuple[list[dict], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """The slice windows every one of ``digests`` can be scored on, and each
    digest's per-window ``(crps, mase)`` over exactly those windows."""
    kept: list[dict] = []
    per: dict[str, tuple[list[float], list[float]]] = {d: ([], []) for d in digests}
    for w in windows:
        wid = w["window_id"]
        if not target_usable(w, horizon):
            continue
        if any(forecast_ok(forecasts.get(d), wid, horizon) is not True for d in digests):
            continue
        vals = {}
        for d in digests:
            try:
                c, m = score_window(forecasts[d][wid], w["target"], w["history"],
                                    w["seasonal_period"])
            except ValueError:
                break
            if not (math.isfinite(c) and math.isfinite(m)):
                break
            vals[d] = (c, m)
        if len(vals) != len(digests):
            continue
        kept.append(w)
        for d in digests:
            per[d][0].append(vals[d][0])
            per[d][1].append(vals[d][1])
    return kept, {d: (np.asarray(c, dtype=np.float64), np.asarray(m, dtype=np.float64))
                  for d, (c, m) in per.items()}


def lowfreq_check(windows: list[dict], forecasts: dict, *, candidate: str, king_d: str,
                  house: str, cfg, seed: str) -> dict:
    """The verdict's ``lowfreq`` block for one crowning candidate on the slice
    ``windows`` (:func:`lowfreq_windows`): ``no_windows`` (``n_windows`` 0,
    no ``lcb``) when no slice window can be scored for the candidate, KING_D
    and the house; otherwise ``pass``/``fail`` with
    ``lcb = {king_d, house} × {crps, mase}``."""
    refs = {"king_d": str(king_d), "house": str(house)}
    digests = sorted({str(candidate), *refs.values()})
    kept, scores = _scored(windows, forecasts, digests, int(cfg.horizon))
    if not kept:
        return _block("no_windows", 0, cfg)
    cand = scores[str(candidate)]
    by_ref: dict[str, dict[str, float]] = {}
    for ref_digest in sorted(set(refs.values())):
        ref = scores[ref_digest]
        by_ref[ref_digest] = {
            metric: float(paired_geomean_lcb(
                ref[i], cand[i], alpha=float(cfg.alpha), B=int(cfg.bootstrap_B),
                seed=f"{seed}:lowfreq:{candidate}:{ref_digest}:{metric}"))
            for i, metric in enumerate(LOWFREQ_METRICS)}
    lcb = {role: dict(by_ref[refs[role]]) for role in LOWFREQ_REFS}
    tau = float(cfg.tau_public)
    worst = min(lcb[r][m] for r in LOWFREQ_REFS for m in LOWFREQ_METRICS)
    # A NaN LCB (no finite statistic) is never a pass: fail closed.
    result = "pass" if math.isfinite(worst) and worst >= -tau else "fail"
    return {**_block(result, len(kept), cfg), "lcb": lcb}


__all__ = ["LOWFREQ_METRICS", "LOWFREQ_REFS", "forecast_ok", "lowfreq_check",
           "lowfreq_windows", "target_usable"]
