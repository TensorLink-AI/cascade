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
    increment_bootstrap_rel,
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

# Margin denomination (``[scoring] margin_mode``; DEC-CA-0027):
#   "level"     — LCB of (king − chal) / king vs the margin. The from-scratch
#                 rule, exact historical behaviour (default).
#   "increment" — LCB of (king − chal) / unit, unit = the floored mean
#                 |improvement over the shared warm-start init| — the margin
#                 then prices a fraction of a typical per-round increment, so
#                 dethrones stay clearable as a compounding lineage converges.
#                 Needs the baseline (init) checkpoint scored as a third paired
#                 reference; a random-init round has no baseline and is judged
#                 at "level" regardless (the E2 init-round semantics).
MARGIN_MODES = ("level", "increment")


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
    # Init-baseline floor ([scoring] init_gate_mode): the shared warm-start
    # init, scored on the SAME duel windows, as an absolute floor under the
    # crown — "nobody takes the throne who's worse than doing nothing".
    # Same rollout shape as the gift gate: "off" → "shadow" (floor computed
    # and recorded, never gates) → "enforce" (a challenger whose observed
    # geomean is worse than ``baseline × (1 + init_gate_tolerance)`` cannot
    # win the round). It can only BLOCK a dethrone, never grant one; king
    # retention is untouched. Consensus-relevant: every validator must run
    # the same mode or verdicts fork (release-then-activate, shadow first).
    init_gate_mode: str = "off"
    init_gate_tolerance: float = 0.0
    # Margin denomination (see MARGIN_MODES). "level" is bit-identical to the
    # pre-field behaviour; receipts record these via the params dict, so an
    # increment round replays under the rule that decided it.
    margin_mode: str = "level"
    # Floor on the increment normaliser, as a fraction of the baseline level
    # (scale-free): with both increments ≈ 0 an unfloored unit divides by
    # noise exactly when the evidence is weakest (the DEC-CA-0009 lesson).
    margin_increment_floor: float = 0.01


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
    # Which CRPS-family aggregation judged this round ("geomean" | "pooled").
    # Recorded on the receipt so an auditor replays a round under the rule that
    # actually decided it; see :func:`cascade.eval.scoring.global_geomean`.
    wql_mode: str = "geomean"
    # Which margin denomination judged this round (see MARGIN_MODES), and the
    # shared warm-start init's observed geomean whenever the init was scored
    # this round — for the "increment" margin and/or the init-baseline floor.
    # The baseline is the init checkpoint's SCORED face (its published
    # weights.safetensors), the same artifact form challengers are scored on.
    margin_mode: str = "level"
    baseline_geomean: float | None = None
    # Init-baseline floor verdict (``KothParams.init_gate_mode``): None = gate
    # off or no baseline this round; under "enforce" a False has already been
    # folded into ``challenger_wins_round``; under "shadow" it is logged only.
    init_floor_passed: bool | None = None


def _window_clusters(scores: list[WindowScore]) -> tuple[list, int]:
    """Cluster labels for the paired bootstrap, one per (window, channel) row.

    The cluster key is the upstream feed id (pool metadata ``source``) when
    present; rows without one fall back to their ``series_id`` — i.e. their
    window. For every univariate pool (one row per window, unique series ids)
    that is exactly the classic per-window bootstrap, byte-identical to the
    old per-row fallback. The distinction bites only when a window carries
    several rows: a C-channel window's rows are near-perfectly correlated, and
    a per-ROW fallback would resample them as C independent observations —
    inflating the effective sample size precisely when multivariate windows
    enter the pool (DEC-CA-0026 item 3). Keying on the window keeps a
    12-channel window from voting 12 times even on a pool with no ``source``
    metadata.
    """
    labels: list = []
    for s in scores:
        labels.append(s.source if s.source else f"__series:{s.series_id}")
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
    wql_mode: str = "geomean",
    baseline_scores: list[WindowScore] | None = None,
) -> RoundResult:
    """Judge one round. ``king_scores`` and ``chal_scores`` must be paired:
    same windows, same order. Raises ``ValueError`` if lengths disagree.

    ``wql_mode`` is the CRPS-family aggregation (see
    :func:`cascade.eval.scoring.global_geomean`). Live rounds use the default
    ``"geomean"``; pass ``"pooled"`` only when replaying a receipt written
    before 2026-07-28, which recorded its own mode for exactly this purpose.

    ``baseline_scores`` (the shared warm-start init scored on the same
    windows, in the same order) is REQUIRED when ``params.margin_mode ==
    "increment"`` and forbidden otherwise — the caller (validator loop /
    audit) owns the fall-back-to-level decision for rounds with no baseline
    (random init), so an armed increment mode with no baseline here is a
    wiring bug and raises rather than silently judging under the wrong rule.
    """
    if len(king_scores) != len(chal_scores):
        raise ValueError(
            f"unpaired scores: king {len(king_scores)} vs challenger {len(chal_scores)}"
        )
    if params.margin_mode not in MARGIN_MODES:
        raise ValueError(
            f"margin_mode must be one of {MARGIN_MODES}; got {params.margin_mode!r}"
        )
    increment = params.margin_mode == "increment"
    init_gate = str(params.init_gate_mode or "off") != "off"
    if increment and baseline_scores is None:
        raise ValueError(
            "margin_mode='increment' needs baseline_scores (the shared warm-start "
            "init scored on the same windows); judge a random-init round under "
            "'level' instead"
        )
    if not increment and not init_gate and baseline_scores is not None:
        raise ValueError(
            "baseline_scores given but margin_mode is 'level' and the "
            "init-baseline gate is off"
        )
    if increment and len(baseline_scores) != len(king_scores):
        raise ValueError(
            f"unpaired baseline: {len(baseline_scores)} vs king {len(king_scores)}"
        )
    n = len(king_scores)
    margin = margin_for_tenure(params, king_tenure_rounds)
    clusters, n_clusters = _window_clusters(king_scores)
    base_geo = (
        global_geomean(baseline_scores, wql_mode=wql_mode)
        if baseline_scores is not None else None
    )

    if n < params.min_windows or (params.min_clusters > 0 and n_clusters < params.min_clusters):
        return RoundResult(
            challenger_wins_round=False,
            lcb=float("nan"),
            margin=margin,
            n_windows=n,
            king_geomean=global_geomean(king_scores, wql_mode=wql_mode),
            chal_geomean=global_geomean(chal_scores, wql_mode=wql_mode),
            inconclusive=True,
            n_clusters=n_clusters,
            wql_mode=wql_mode,
            margin_mode=params.margin_mode,
            baseline_geomean=base_geo,
        )

    k_qloss, k_abs, k_mase = stack_components(king_scores)
    c_qloss, c_abs, c_mase = stack_components(chal_scores)
    boot_p50 = boot_p95 = None
    if increment:
        b_qloss, b_abs, b_mase = stack_components(baseline_scores)
        rel = increment_bootstrap_rel(
            (k_qloss, k_abs, k_mase),
            (c_qloss, c_abs, c_mase),
            (b_qloss, b_abs, b_mase),
            B=params.bootstrap_B, seed=seed, clusters=clusters,
            wql_mode=wql_mode, floor_frac=params.margin_increment_floor,
        )
        lcb = (
            float(np.quantile(rel, params.bootstrap_alpha))
            if rel.size else float("nan")
        )
        if rel.size:
            boot_p50 = float(np.quantile(rel, 0.5))
            boot_p95 = float(np.quantile(rel, 0.95))
    else:
        lcb = paired_bootstrap_lcb_aggregated(
            k_qloss, k_abs, k_mase,
            c_qloss, c_abs, c_mase,
            alpha=params.bootstrap_alpha,
            B=params.bootstrap_B,
            seed=seed,
            clusters=clusters,
            wql_mode=wql_mode,
        )
        try:
            # Same B/seed/clusters as the LCB above ⇒ the 5th-pct here == lcb; we
            # keep the median and 95th pct for display. A diagnostic must never
            # fail a round.
            qs = paired_bootstrap_quantiles_aggregated(
                k_qloss, k_abs, k_mase, c_qloss, c_abs, c_mase,
                quantiles=(0.5, 0.95), B=params.bootstrap_B, seed=seed,
                clusters=clusters, wql_mode=wql_mode,
            )
            boot_p50, boot_p95 = qs.get(0.5), qs.get(0.95)
        except Exception:  # noqa: BLE001 — spread is display-only
            pass
    win_rate, wilcoxon_p, per_domain = _shadow_diagnostics(king_scores, chal_scores)
    chal_geo = global_geomean(chal_scores, wql_mode=wql_mode)
    # Init-baseline floor (KothParams.init_gate_mode): the challenger's
    # observed geomean against the shared init's, on these very windows.
    # Enforce AND-s it into the win — the gift-gate shape: it can only BLOCK
    # a dethrone, never grant one. No baseline this round ⇒ the floor cannot
    # run (None), whatever the mode.
    init_floor_passed: bool | None = None
    if init_gate and base_geo is not None:
        init_floor_passed = bool(
            chal_geo <= base_geo * (1.0 + max(0.0, params.init_gate_tolerance))
        )
    wins = bool(lcb >= margin)
    if params.init_gate_mode == "enforce" and init_floor_passed is False:
        wins = False
    return RoundResult(
        challenger_wins_round=wins,
        lcb=lcb,
        margin=margin,
        n_windows=n,
        king_geomean=global_geomean(king_scores, wql_mode=wql_mode),
        chal_geomean=chal_geo,
        inconclusive=False,
        n_clusters=n_clusters,
        wql_mode=wql_mode,
        margin_mode=params.margin_mode,
        baseline_geomean=base_geo,
        init_floor_passed=init_floor_passed,
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
