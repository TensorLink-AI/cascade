"""Trained-checkpoint validity gate — a cheap pass/fail screen before the duel.

This is metronome's analogue of teutonic's pre-duel trainability probe: a fast,
binary "is this trained model fit to be scored at all?" check run on a model's
per-window scores *before* the paired-bootstrap KOTH verdict. It does **not**
judge quality (that is :mod:`.koth`); it only rejects models whose forecasts are
numerically broken — NaN/inf blow-ups or impossible negative metrics — so a
degenerate run can neither win a round nor corrupt the bootstrap statistic.

Pure numpy/CPU, so it is unit-tested with the rest of the eval math.
"""

from __future__ import annotations

import numpy as np

from .scoring import WindowScore


def check_scores_valid(scores: list[WindowScore]) -> str | None:
    """Return a rejection reason if ``scores`` are not fit to be scored, else None.

    Failure modes caught (all "the trained model is broken", not "the data is
    weak"):

    * ``no_scores`` — the model produced nothing to score.
    * ``non_finite`` — any MASE / pinball-loss / abs-target value is NaN or inf
      (a blown-up or diverged checkpoint).
    * ``negative_metric`` — any value is negative, which is impossible for these
      non-negative error metrics and signals corruption or a scorer bug.

    Quantity (too few common windows) is intentionally *not* gated here — that is
    the ``min_windows`` "inconclusive" path in :mod:`.koth`.
    """
    if not scores:
        return "no_scores"
    for s in scores:
        mase = float(s.mase)
        abs_target = float(s.abs_target)
        qloss = np.asarray(s.qloss_per_q, dtype=np.float64)
        if not (np.isfinite(mase) and np.isfinite(abs_target) and np.all(np.isfinite(qloss))):
            return "non_finite"
        if mase < 0.0 or abs_target < 0.0 or np.any(qloss < 0.0):
            return "negative_metric"
    return None
