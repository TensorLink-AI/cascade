"""Heat-cut separation — is the screen's finalist boundary real, or noise?

The heat ranks challengers by the observed ``global_geomean`` and advances the
top ``[round] finalists``. It *must* advance that many whatever the evidence
says, so unlike the throne decision (:mod:`.koth`) there is no
"inconclusive → king holds" branch for a lower confidence bound to gate. What we
can still ask is how much daylight there is **at the cut**: run the same paired
cluster bootstrap the KOTH verdict uses on the last entrant that advanced versus
the first that did not.

The pairing is what makes this worth computing. Every entrant in a round screens
on the *same* windows (``windows_for_round(base_seed, n, block)``), so the two
sides of the cut are paired row-for-row and window difficulty cancels — the same
property that gives the KOTH LCB its power.

This is a DIAGNOSTIC and never gates. The heat still ranks on the point estimate:
ranking by an LCB would reorder the field by a variance penalty, favouring the
lower-variance generator over the better one, and would not give the screen the
option to decline. A cut LCB that sits below the KOTH win margin round after
round means the screen is separating the field by less than the final would call
a win — with ``finalists = 1`` the runner-up never reaches the final for the real
statistic to settle it. That is the signal to raise ``[round] finalists``,
``heat_n_windows`` or ``heat_num_samples``; a different ranking statistic would
not fix it.
"""

from __future__ import annotations

from .bootstrap import paired_bootstrap_quantiles_aggregated
from .scoring import WindowScore, global_geomean, stack_components, window_clusters

# Bootstrap spread reported either side of the LCB. p50 is the point estimate the
# heat actually ranks on; a wide p50-to-LCB gap means the cut rides a heavy tail.
_SPREAD_QUANTILES = (0.5, 0.95)


def measure_heat_cut(
    advanced_scores: list[WindowScore],
    rejected_scores: list[WindowScore],
    *,
    alpha: float,
    B: int,
    seed: int | str,
    margin: float,
):
    """Bootstrap the gap at the heat's finalist boundary.

    ``advanced_scores`` is the *last* entrant that advanced and
    ``rejected_scores`` the *first* that did not — the two either side of the
    cut, both scored on the same windows in the same order. Improvement is
    measured of the advanced entrant over the rejected one, so a positive LCB
    means the screen reliably placed them in that order.

    ``margin`` is the reference bar (the KOTH ``win_margin_start``), recorded so
    a reader can see what the cut is being held against without knowing the
    validator's config. Returns a :class:`~cascade.shared.manifest.HeatCut`.

    Raises ``ValueError`` if the two score lists are not paired (different
    lengths); the bootstrap itself raises if the per-window targets disagree,
    which would mean the two entrants somehow screened on different windows.
    """
    from ..shared.manifest import HeatCut

    if len(advanced_scores) != len(rejected_scores):
        raise ValueError(
            f"unpaired heat scores: advanced {len(advanced_scores)} vs "
            f"rejected {len(rejected_scores)}"
        )

    clusters, n_clusters = window_clusters(advanced_scores)
    a_qloss, a_abs, a_mase = stack_components(advanced_scores)
    r_qloss, r_abs, r_mase = stack_components(rejected_scores)

    # Rejected entrant plays the "king" role: improvement is (rejected − advanced)
    # / rejected, positive when the advanced entrant scores lower (better).
    qs = paired_bootstrap_quantiles_aggregated(
        r_qloss, r_abs, r_mase,
        a_qloss, a_abs, a_mase,
        quantiles=(alpha, *_SPREAD_QUANTILES),
        B=B, seed=seed, clusters=clusters,
    )
    lcb = qs[float(alpha)]

    g_adv = global_geomean(advanced_scores)
    g_rej = global_geomean(rejected_scores)
    observed = (g_rej - g_adv) / g_rej if g_rej > 0 else float("nan")

    return HeatCut(
        lcb=float(lcb),
        p50=float(qs[0.5]),
        p95=float(qs[0.95]),
        observed=float(observed),
        margin=float(margin),
        # NaN compares False, so an uncomputable cut is never "separated".
        separated=bool(lcb >= margin),
        n_windows=len(advanced_scores),
        n_clusters=int(n_clusters),
    )
