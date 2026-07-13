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

from dataclasses import dataclass, replace

import numpy as np

from .bootstrap import (
    paired_bootstrap_lcb_aggregated,
    paired_bootstrap_quantiles_aggregated,
)
from .gift_gate import GiftGateResult
from .scoring import WindowScore, global_geomean, stack_components

# The public-benchmark gate rollout modes (``[scoring] gift_gate_mode``):
#   "off"     — gate never runs (default; pure private-pool KOTH).
#   "shadow"  — gate is computed and logged on a private-pool win, but the
#               verdict is NOT changed (calibrate tolerance against real noise).
#   "enforce" — gate is AND-ed into the dethrone decision.
GIFT_GATE_MODES = ("off", "shadow", "enforce")


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
        min_clusters: below this many distinct window clusters (upstream feeds,
            from pool metadata ``source``), no decision is made. Raw window
            count overstates the evidence when the windows come from a handful
            of correlated feeds; this is the breadth floor. ``0`` disables it
            (and pools without ``source`` metadata are unaffected — every
            window is then its own cluster).
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
    min_clusters: int = 0
    # Public-benchmark no-regression gate (see :mod:`.gift_gate`). Off by
    # default; ``gift_gate_tolerance`` is the relative slack the challenger may
    # be worse by on GIFT-Eval, ``gift_gate_min_configs`` the floor of shared
    # configs below which the gate is uncomputable (→ inconclusive). The gate
    # reuses ``bootstrap_B``/``bootstrap_alpha``. Defaults keep it inert.
    gift_gate_mode: str = "off"
    gift_gate_tolerance: float = 0.03
    gift_gate_min_configs: int = 15


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
        gift_lcb: public-benchmark gate LCB, when the gate ran this round
            (``None`` = gate off / not reached / uncomputable). Diagnostic only.
        gift_gate_passed: whether the gate passed, when it ran (``None``
            otherwise). Under ``enforce`` a False here has already been folded
            into ``challenger_wins_round``; under ``shadow`` it is logged only.
        n_clusters: distinct window clusters (upstream feeds) behind the
            verdict — the honest effective sample size.
        win_rate: fraction of windows where the challenger's per-window
            geomean beats the king's. Diagnostic (shadow) only: 0.5 is noise;
            a significant LCB with win_rate near 0.5 means rare-but-big wins.
        wilcoxon_p: Wilcoxon signed-rank p-value on the paired per-window
            geomean differences (``None`` when scipy is unavailable or the
            test is degenerate). Diagnostic (shadow) only — the LCB-vs-margin
            rule decides; this monitors agreement of a rank-based view.
        per_domain_win_rate: ``{domain: (win_rate, n_windows)}``. A sign flip
            across domains means pool composition is deciding rounds — the
            "stop aggregating" tripwire, logged for observability.
    """

    challenger_wins_round: bool
    lcb: float
    margin: float
    n_windows: int
    king_geomean: float
    chal_geomean: float
    inconclusive: bool
    gift_lcb: float | None = None
    gift_gate_passed: bool | None = None
    n_clusters: int = 0
    win_rate: float | None = None
    wilcoxon_p: float | None = None
    per_domain_win_rate: dict | None = None
    # Diagnostic spread of the same bootstrap the LCB gates on: the median and
    # 95th pct of the relative-improvement distribution (the LCB is its 5th pct).
    # A wide gap between a positive median and a negative LCB = a fragile verdict
    # whose point estimate rides a heavy tail. Display only; never gates.
    boot_p50: float | None = None
    boot_p95: float | None = None


def _window_clusters(scores: list[WindowScore]) -> tuple[list, int]:
    """Cluster labels for the paired bootstrap, one per (window, channel) row.

    The cluster key is the upstream feed id (pool metadata ``source``) when
    present; rows without one are their own singleton cluster, which degrades
    exactly to the classic per-window bootstrap for legacy pools.
    """
    labels: list = []
    for i, s in enumerate(scores):
        labels.append(s.source if s.source else f"__row{i}")
    return labels, len(set(labels))


def _per_window_geomeans(scores: list[WindowScore]) -> np.ndarray:
    """Per-window geomean(WQL, MASE) — the scalar behind the shadow
    diagnostics only; the decision LCB uses the aggregate-then-divide form."""
    g = np.empty(len(scores))
    for i, s in enumerate(scores):
        wql = 2.0 * float(np.mean(s.qloss_per_q)) / max(abs(s.abs_target), 1e-9)
        g[i] = np.sqrt(max(wql, 1e-12) * max(s.mase, 1e-12))
    return g


def _shadow_diagnostics(
    king_scores: list[WindowScore], chal_scores: list[WindowScore]
) -> tuple[float | None, float | None, dict | None]:
    """(win_rate, wilcoxon_p, per_domain_win_rate) — logged, never gating."""
    if not king_scores:
        return None, None, None
    g_king = _per_window_geomeans(king_scores)
    g_chal = _per_window_geomeans(chal_scores)
    wins = g_chal < g_king
    win_rate = float(wins.mean())

    wilcoxon_p: float | None = None
    diffs = g_king - g_chal
    if np.any(diffs != 0.0) and len(diffs) >= 10:
        try:
            from scipy.stats import wilcoxon

            wilcoxon_p = float(wilcoxon(diffs, zero_method="wilcox").pvalue)
        except Exception:  # noqa: BLE001 — a diagnostic must never fail a round
            wilcoxon_p = None

    per_domain: dict[str, tuple[float, int]] = {}
    domains = [s.domain or "unknown" for s in king_scores]
    for dom in sorted(set(domains)):
        mask = np.asarray([d == dom for d in domains])
        per_domain[dom] = (float(wins[mask].mean()), int(mask.sum()))
    return win_rate, wilcoxon_p, per_domain


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
    clusters, n_clusters = _window_clusters(king_scores)

    if n < params.min_windows or (params.min_clusters > 0 and n_clusters < params.min_clusters):
        return RoundResult(
            challenger_wins_round=False,
            lcb=float("nan"),
            margin=margin,
            n_windows=n,
            king_geomean=global_geomean(king_scores),
            chal_geomean=global_geomean(chal_scores),
            inconclusive=True,
            n_clusters=n_clusters,
        )

    k_qloss, k_abs, k_mase = stack_components(king_scores)
    c_qloss, c_abs, c_mase = stack_components(chal_scores)
    lcb = paired_bootstrap_lcb_aggregated(
        k_qloss, k_abs, k_mase,
        c_qloss, c_abs, c_mase,
        alpha=params.bootstrap_alpha,
        B=params.bootstrap_B,
        seed=seed,
        clusters=clusters,
    )
    win_rate, wilcoxon_p, per_domain = _shadow_diagnostics(king_scores, chal_scores)
    boot_p50 = boot_p95 = None
    try:
        # Same B/seed/clusters as the LCB above ⇒ the 5th-pct here == lcb; we keep
        # the median and 95th pct for display. A diagnostic must never fail a round.
        qs = paired_bootstrap_quantiles_aggregated(
            k_qloss, k_abs, k_mase, c_qloss, c_abs, c_mase,
            quantiles=(0.5, 0.95), B=params.bootstrap_B, seed=seed, clusters=clusters,
        )
        boot_p50, boot_p95 = qs.get(0.5), qs.get(0.95)
    except Exception:  # noqa: BLE001 — spread is display-only
        pass
    return RoundResult(
        challenger_wins_round=bool(lcb >= margin),
        lcb=lcb,
        margin=margin,
        n_windows=n,
        king_geomean=global_geomean(king_scores),
        chal_geomean=global_geomean(chal_scores),
        inconclusive=False,
        n_clusters=n_clusters,
        win_rate=win_rate,
        wilcoxon_p=wilcoxon_p,
        per_domain_win_rate=per_domain,
        boot_p50=boot_p50,
        boot_p95=boot_p95,
    )


def apply_gift_gate(
    result: RoundResult, gate: GiftGateResult, *, mode: str
) -> RoundResult:
    """Fold the public-benchmark gate into a round result. Pure — returns a new
    :class:`RoundResult`; the private-pool decision in ``result`` is untouched
    except where ``enforce`` blocks a win.

    Truth table (the gate only matters on a private-pool *win*):

    * win × pass                → win (unchanged)
    * win × fail   (enforce)    → not a win, streak resets (a public regression)
    * win × uncomputable        → inconclusive (king holds, streak untouched)
    * win, mode = shadow        → win (unchanged); gate recorded for logging
    * loss / inconclusive       → unchanged (gate is never consulted)

    ``gift_lcb``/``gift_gate_passed`` are always recorded for observability, so
    a shadow run logs exactly what an enforce run would have decided.
    """
    diagnostic = replace(
        result,
        gift_lcb=(gate.lcb if gate.computed else None),
        gift_gate_passed=(gate.passed if gate.computed else None),
    )
    if mode != "enforce" or not result.challenger_wins_round:
        return diagnostic
    if not gate.computed:
        # Uncomputable gate on an otherwise-winning round: make no decision
        # rather than pass or fail silently — the king holds, streak untouched.
        return replace(diagnostic, challenger_wins_round=False, inconclusive=True)
    if not gate.passed:
        return replace(diagnostic, challenger_wins_round=False)
    return diagnostic
