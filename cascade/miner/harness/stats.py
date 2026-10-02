"""The judge's arithmetic: paired improvements, noise margins, paired LCBs.

``rel`` everywhere is the RELATIVE IMPROVEMENT over the reference on the same
windows: ``1 − cand / ref`` of the round geomean. Positive = the candidate is
better (lower geomean). Pairing is by window order, exactly as the duel pairs.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace


def rel_improvement(cand_geo: float, ref_geo: float) -> float:
    if not (ref_geo > 0 and math.isfinite(cand_geo) and math.isfinite(ref_geo)):
        return float("nan")
    return 1.0 - cand_geo / ref_geo


def noise_sigma(rels: list[float]) -> float:
    """σ of one paired comparison, measured on the king against itself.

    ``rels`` are ``rel_improvement(salted king, king)`` per salt. Each is a
    difference of two noisy legs of the SAME generator — exactly the noise a
    candidate-vs-king comparison carries when nothing changed — so σ is their
    RMS (the true mean is 0). Returns NaN with no measurements."""
    vals = [r for r in rels if math.isfinite(r)]
    if not vals:
        return float("nan")
    return math.sqrt(sum(r * r for r in vals) / len(vals))


def margin(sigma: float, *, z: float, floor: float, n: int = 1) -> float:
    """``max(floor, z × σ / √n)`` — σ is a single paired comparison's noise
    (a difference of two legs), so ``n`` independent rounds shrink it by √n."""
    if not math.isfinite(sigma):
        return floor
    return max(floor, z * sigma / math.sqrt(max(n, 1)))


def paired_lcb(king_scores: list, cand_scores: list, koth_params, *, seed: int,
               lcb_margin: float = 0.0):
    """The duel's paired cluster bootstrap on ``(king, cand)`` rows, judged as a
    fresh-king level round against ``lcb_margin``. Returns ``RoundResult``."""
    from ...eval.koth import evaluate_round

    p = replace(koth_params, win_margin_start=lcb_margin, win_margin_end=lcb_margin,
                margin_warmup_rounds=0, margin_mode="level", init_gate_mode="off",
                gift_gate_mode="off", min_windows=min(koth_params.min_windows, 20),
                min_clusters=0, dethrone_cp=1)
    return evaluate_round(king_scores, cand_scores, p, seed=seed, king_tenure_rounds=0)


def mean(xs: list[float]) -> float:
    vals = [x for x in xs if math.isfinite(x)]
    return statistics.fmean(vals) if vals else float("nan")
