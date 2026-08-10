"""Heat-screen selection diagnostics — how decisive was the screen?

The heat trains every eligible challenger cheaply, ranks them on
:func:`~.scoring.global_geomean`, and advances the top ``[round] finalists``.
That is a *selection* (pick k of N), not the duel's hypothesis test (is the
challenger better than the king by at least a margin), so the duel's LCB-vs-margin
rule has no direct analogue here: with ``finalists = 1`` the pick is the argmin
whatever bound you wrap around it.

What the screen genuinely lacks is any sense of whether the pick was decisive.
It screens on a fraction of the windows and samples the final judges on, so the
ranking can be close to a coin flip and nothing downstream would say so. This
module measures that, and *only* measures it — the finalists are chosen by the
observed ranking in :mod:`cascade.trainer.loop`; nothing here feeds that choice.

Both quantities come off one joint cluster bootstrap
(:func:`~.bootstrap.joint_bag_geomeans`) — a single shared resample across every
entrant, which is what makes within-bag comparisons paired:

* ``p_best`` — the fraction of bags in which an entrant has the lowest geomean.
  This is the selection question asked directly. A field where the leader holds
  ``p_best ≈ 1/N`` is being ranked by noise.
* ``leader_lcb`` — the leader-vs-runner-up paired LCB on relative improvement,
  the duel's exact statistic with the runner-up in the king's slot. ``> 0`` means
  the screen separated first from second; ``≤ 0`` means it did not, and the two
  are interchangeable on this evidence.

Deliberately NOT computed: a marginal upper confidence bound on each entrant's
own score, ranked ascending. It reads like the conservative choice, but the
marginal spread is dominated by shared window difficulty that is identical for
every entrant, so it shifts the whole field by nearly the same amount; where it
does bite it penalises per-window dispersion, which is not what the duel scores.
See ``decisions/DEC-CA-0006-heat-lcb-diagnostics-not-selection.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .bootstrap import joint_bag_geomeans
from .scoring import WindowScore, stack_components


@dataclass(frozen=True)
class HeatDiagnostics:
    """Shadow diagnostics for one heat screen. Never gates the finalist pick.

    Attributes:
        p_best: ``{key: P(entrant has the lowest bag geomean)}`` over the joint
            bootstrap. Sums to 1 across entrants (ties broken by lowest index,
            as :meth:`numpy.ndarray.argmin` does).
        leader_lcb: paired LCB on the leader's relative improvement over the
            runner-up, ``(runner - leader) / runner``. Positive means the screen
            separated the top two. ``None`` with fewer than two entrants.
            Equals ``lcb_vs[runner_up_key]``; kept as its own field because it is
            the headline "was the screen decisive" number on the heat block.
        lcb_vs: ``{key: paired LCB of the leader over that entrant}`` for every
            NON-leader entrant, off the same joint bootstrap. The leader
            *separates* from an entrant iff its value is ``> 0``; entrants it does
            not separate from are statistically interchangeable with it on this
            evidence and form the tied set (see :func:`tied_set`). Empty with
            fewer than two entrants.
        leader_key / runner_up_key: the two entrants ``leader_lcb`` compares, by
            observed rank. ``runner_up_key`` is ``None`` with a single entrant.
        n_windows / n_clusters: the evidence behind the screen. ``n_clusters``
            is the honest effective sample size when the pool carries ``source``
            labels; without them it equals ``n_windows``.
    """

    p_best: dict[str, float]
    leader_lcb: float | None
    leader_key: str | None
    runner_up_key: str | None
    n_windows: int
    n_clusters: int
    lcb_vs: dict[str, float] = field(default_factory=dict)


def _clusters(scores: Sequence[WindowScore]) -> tuple[list, int]:
    """Cluster labels (upstream feed id) per row, mirroring the KOTH bootstrap:
    rows without a ``source`` are singleton clusters, degrading to the classic
    per-window bootstrap."""
    labels: list = [s.source if s.source else f"__row{i}" for i, s in enumerate(scores)]
    return labels, len(set(labels))


def screen_diagnostics(
    ranked: Sequence[tuple[str, list[WindowScore]]],
    *,
    seed: int | str,
    B: int = 10_000,
    alpha: float = 0.05,
) -> HeatDiagnostics | None:
    """Measure how decisive a heat screen was.

    ``ranked`` is ``(key, scores)`` per entrant **in observed rank order** (best
    first) — the caller has already sorted, so ``ranked[0]`` is the leader and
    ``ranked[1]`` the runner-up. Every entrant must carry the same windows in the
    same order; the heat scores them all on one slice, so it does by construction.

    Returns ``None`` when there is nothing to measure (no entrants, or no
    windows), or when the entrants turn out not to be paired — a diagnostic must
    never fail a round, so an unpaired field is reported as "no diagnostics"
    rather than raised.
    """
    if not ranked:
        return None
    keys = [k for k, _ in ranked]
    all_scores = [s for _, s in ranked]
    if not all_scores[0]:
        return None

    clusters, n_clusters = _clusters(all_scores[0])
    n_windows = len(all_scores[0])
    try:
        bags = joint_bag_geomeans(
            [stack_components(s) for s in all_scores],
            B=B, seed=seed, clusters=clusters,
        )
    except ValueError:
        return None
    if bags.shape[1] == 0:
        return None

    counts = np.bincount(bags.argmin(axis=0), minlength=len(keys))
    p_best = {k: float(c / bags.shape[1]) for k, c in zip(keys, counts, strict=True)}

    # Paired LCB of the leader over EVERY other entrant, off the one shared
    # resample above — so window difficulty cancels in each comparison. This is
    # the runner-up formula with bags[1] generalised to bags[j]; leader_lcb below
    # is literally lcb_vs[runner_up_key], which is what keeps the DEC-CA-0006
    # pin ("leader_lcb equals the duel statistic on the top two") meaningful.
    lcb_vs: dict[str, float] = {}
    for j in range(1, len(keys)):
        other = bags[j]
        safe_other = np.where(np.abs(other) < 1e-9, 1e-9, other)
        lcb_vs[keys[j]] = float(np.quantile((other - bags[0]) / safe_other, alpha))

    runner_up_key = keys[1] if len(keys) >= 2 else None
    leader_lcb = lcb_vs[runner_up_key] if runner_up_key is not None else None

    return HeatDiagnostics(
        p_best=p_best,
        leader_lcb=leader_lcb,
        leader_key=keys[0],
        runner_up_key=runner_up_key,
        n_windows=n_windows,
        n_clusters=n_clusters,
        lcb_vs=lcb_vs,
    )


def tied_set(
    diagnostics: HeatDiagnostics | None,
    ranked_keys: Sequence[str],
    *,
    cap: int,
) -> list[str]:
    """The leader plus every entrant the screen did NOT separate it from.

    ``ranked_keys`` is every scored entrant in observed rank order (best first);
    the returned list preserves that order and is truncated to ``cap``. The
    criterion is ``lcb_vs[key] <= 0`` — the leader's paired 95% lower bound on
    relative improvement over that entrant fails to clear zero, so on this
    evidence the two are interchangeable.

    The bar is 0, not the duel's win margin: the margin exists because the *king*
    holds ties, and among challengers there is no incumbent and no null. Dropping
    an entrant is safe exactly when we are confident the leader is at least as
    good, which is what a one-sided bound at 0 states.

    Multiplicity is deliberately NOT corrected. Testing the leader against N−1
    entrants at 5% inflates this set (~1 false inclusion on a 20-entrant field),
    but a field-size-dependent bar would let the same two generators separate or
    not depending on how many unrelated entrants showed up, and could be gamed by
    padding the field. ``cap`` is the control instead. See
    ``decisions/DEC-CA-0012-tie-aware-finalists-cohort-duel.md``.

    Degrades safely: no diagnostics (a scalar-only screener, an unpaired field, a
    single entrant) returns just the leader, i.e. exactly the pre-DEC-CA-0012
    behaviour. ``cap <= 0`` returns empty.
    """
    if cap <= 0 or not ranked_keys:
        return []
    if diagnostics is None or not diagnostics.lcb_vs:
        return list(ranked_keys[:1])
    out = [ranked_keys[0]]
    for key in ranked_keys[1:]:
        lcb = diagnostics.lcb_vs.get(key)
        # An entrant with no recorded comparison cannot be shown to be separated,
        # but neither can it be shown to be tied; treat a missing value as
        # separated so a partial diagnostic never inflates GPU spend.
        if lcb is not None and lcb <= 0.0:
            out.append(key)
    return out[:cap]
