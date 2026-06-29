"""King-of-the-hill decision: does the challenger dethrone the king?

A round compares two trained models — the king's and the challenger's — scored
on the *same* eval windows (see :mod:`.scoring`). The decision is a paired
bootstrap LCB on the relative geomean(CRPS, MASE) improvement of challenger
over king. The challenger *wins the round* iff that LCB clears the win margin
and there are enough common windows to make the call.

Dethroning is deliberately sticky: the validator requires ``dethrone_cp``
consecutive round wins before the throne actually changes hands (the
consecutive-win bookkeeping lives in :mod:`cascade.validator.state`). This
module owns the single-round statistical verdict and the margin schedule; it
holds no state.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bootstrap import paired_bootstrap_lcb_aggregated
from .scoring import WindowScore, global_geomean, stack_components


@dataclass(frozen=True)
class KothParams:
    """Decision parameters, loaded from ``chain.toml [scoring]``.

    Attributes:
        win_margin_start / win_margin_end: affine margin warmup. A freshly
            crowned king is easier to challenge (``start``); the margin ramps
            to ``end`` over ``margin_warmup_rounds`` of tenure so an
            entrenched king must be beaten more decisively.
        margin_warmup_rounds: tenure (in won rounds) over which the margin
            ramps from start to end.
        min_windows: below this many common eval windows, no decision is made
            (the round is inconclusive; the king holds).
        bootstrap_B: bootstrap resamples.
        bootstrap_alpha: one-sided LCB level.
        dethrone_cp: consecutive round wins required to dethrone.
    """

    win_margin_start: float
    win_margin_end: float
    margin_warmup_rounds: int
    min_windows: int
    bootstrap_B: int
    bootstrap_alpha: float
    dethrone_cp: int


def margin_for_tenure(params: KothParams, king_tenure_rounds: int) -> float:
    """Affine margin schedule as a function of the king's tenure.

    ``start`` at tenure 0, ramping linearly to ``end`` at
    ``margin_warmup_rounds`` and clamped there after. Mirrors horizon's
    ``win_margin_start``/``win_margin_end`` warmup so an established king is
    harder to displace than a brand-new one.
    """
    if params.margin_warmup_rounds <= 0:
        return params.win_margin_end
    frac = min(max(king_tenure_rounds, 0) / params.margin_warmup_rounds, 1.0)
    return params.win_margin_start + frac * (params.win_margin_end - params.win_margin_start)


@dataclass(frozen=True)
class RoundResult:
    """Outcome of one king-vs-challenger round.

    Attributes:
        challenger_wins_round: LCB cleared the margin on enough windows.
        lcb: paired-bootstrap lower confidence bound on relative improvement.
        margin: the margin this round was judged against (tenure-adjusted).
        n_windows: number of paired eval windows scored.
        king_geomean / chal_geomean: observed (non-bootstrapped) geomeans, for
            logging.
        inconclusive: True when ``n_windows < min_windows`` — the king holds
            and the win counter does not advance.
    """

    challenger_wins_round: bool
    lcb: float
    margin: float
    n_windows: int
    king_geomean: float
    chal_geomean: float
    inconclusive: bool


def evaluate_round(
    king_scores: list[WindowScore],
    chal_scores: list[WindowScore],
    params: KothParams,
    *,
    seed: int | str,
    king_tenure_rounds: int = 0,
) -> RoundResult:
    """Judge one round. ``king_scores`` and ``chal_scores`` must be paired:
    same windows, same order. Raises ``ValueError`` if lengths disagree.
    """
    if len(king_scores) != len(chal_scores):
        raise ValueError(
            f"unpaired scores: king {len(king_scores)} vs challenger {len(chal_scores)}"
        )
    n = len(king_scores)
    margin = margin_for_tenure(params, king_tenure_rounds)

    if n < params.min_windows:
        return RoundResult(
            challenger_wins_round=False,
            lcb=float("nan"),
            margin=margin,
            n_windows=n,
            king_geomean=global_geomean(king_scores),
            chal_geomean=global_geomean(chal_scores),
            inconclusive=True,
        )

    k_qloss, k_abs, k_mase = stack_components(king_scores)
    c_qloss, c_abs, c_mase = stack_components(chal_scores)
    lcb = paired_bootstrap_lcb_aggregated(
        k_qloss, k_abs, k_mase,
        c_qloss, c_abs, c_mase,
        alpha=params.bootstrap_alpha,
        B=params.bootstrap_B,
        seed=seed,
    )
    return RoundResult(
        challenger_wins_round=bool(lcb >= margin),
        lcb=lcb,
        margin=margin,
        n_windows=n,
        king_geomean=global_geomean(king_scores),
        chal_geomean=global_geomean(chal_scores),
        inconclusive=False,
    )
