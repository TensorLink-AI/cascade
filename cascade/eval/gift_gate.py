"""Public-benchmark no-regression gate — the second, AND-ed condition on a
dethrone.

The private rotating pool (see :mod:`.koth`) decides *whether the challenger is
better*; this gate independently checks the challenger has not gotten
*meaningfully worse on broad public data* (GIFT-Eval). It is deliberately **not
winnable**: it can only block a dethrone the private LCB already granted, never
cause one. That asymmetry is the whole point — a generator that games the public
benchmark still has to beat the private pool to take the throne, while a
generator that overfits the private pool's domains and regresses on the broad
public battery is held back.

The statistic mirrors the KOTH decision: a paired bootstrap over the shared
GIFT-Eval configs of the relative ``geomean(CRPS, MASE)`` improvement of
challenger over king, where each metric is the official Seasonal-Naive-normalized
shifted geometric mean (:mod:`benchmarks.cascade_benchmark.aggregate`). Instead
of re-deriving the baseline here, the sidecar hands back per-config
``crps_ratio``/``mase_ratio`` (model ÷ vendored Seasonal-Naive), so this module
is pure numpy over those ratios.

The gate passes when the lower confidence bound on that relative improvement
clears ``-tolerance``: the challenger may be noise-level worse, but not
*statistically significantly* worse. A challenger that is genuinely equal on the
public benchmark passes deterministically; one that regressed clearly fails
deterministically — no coin flip at parity (which is why this is a bootstrap
bound, not a raw ``chal < king`` comparison).

Determinism matches :mod:`.bootstrap`: ``seed`` is the round's block hash, so
every validator draws identical bootstrap indices on the same comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bootstrap import _seed_to_int

# Each GIFT-Eval config's ratio row must carry these keys (produced by the
# sidecar's ``gift-eval`` suite, keyed by the ``full`` = ``name/freq/term`` id).
_KEY = "full"
_RATIOS = ("crps_ratio", "mase_ratio")


def _shifted_gmean(x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """``exp(mean(log(x + eps))) - eps`` along the last axis — the upstream
    ``leaderboard.shifted_gmean`` (see ``benchmarks…/aggregate.py``), vectorized
    so a whole bootstrap stack aggregates at once. ``x`` is ``(..., n)``;
    returns ``(...)``."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    if n == 0:
        return np.full(x.shape[:-1], np.nan)
    return np.exp(np.sum(np.log(x + epsilon), axis=-1) / n) - epsilon


def _clean_ratio_column(values: list[float]) -> np.ndarray | None:
    """``replace_invalid_values``: None/±inf → nan → filled with the finite
    mean (upstream order). Returns ``None`` if nothing is finite."""
    a = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    a[np.isinf(a)] = np.nan
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return None
    a[np.isnan(a)] = finite.mean()
    return a


@dataclass(frozen=True)
class GiftGateResult:
    """Outcome of the public-benchmark no-regression gate for one round.

    Attributes:
        computed: True when the paired bootstrap ran (enough shared configs,
            valid ratios). When False the gate is *uncomputable* — the caller
            treats the round as inconclusive (king holds, streak untouched),
            never as a silent pass or fail.
        passed: True when ``lcb >= -tolerance`` (challenger not meaningfully
            worse on the public battery). ``None`` when ``computed`` is False.
        lcb: paired-bootstrap lower confidence bound on the relative
            geomean(CRPS, MASE) improvement of challenger over king (positive =
            challenger better). ``nan`` when not computed.
        tolerance: the ``-tolerance`` threshold ``lcb`` was judged against.
        n_configs: shared GIFT-Eval configs the bootstrap resampled.
        king_agg / chal_agg: observed (non-bootstrapped) geomeans, for logging.
        reason: short diagnostic when ``computed`` is False.
    """

    computed: bool
    passed: bool | None
    lcb: float
    tolerance: float
    n_configs: int
    king_agg: float
    chal_agg: float
    reason: str = ""


def uncomputable_gate(tolerance: float, reason: str) -> GiftGateResult:
    """A gate that could not run (sidecar unavailable, data-revision mismatch,
    …) — the caller treats the round as inconclusive."""
    return GiftGateResult(
        computed=False, passed=None, lcb=float("nan"), tolerance=tolerance,
        n_configs=0, king_agg=float("nan"), chal_agg=float("nan"), reason=reason,
    )


def _paired_ratio_columns(
    king_rows: list[dict], chal_rows: list[dict]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Inner-join king/challenger rows on ``full`` and return cleaned, aligned
    ``(king_crps, king_mase, chal_crps, chal_mase)`` ratio columns — or ``None``
    if the join is empty or a column has no finite values."""
    king_by = {r[_KEY]: r for r in king_rows if _KEY in r}
    chal_by = {r[_KEY]: r for r in chal_rows if _KEY in r}
    shared = sorted(set(king_by) & set(chal_by))
    if not shared:
        return None
    cols: list[np.ndarray] = []
    for by, key in ((king_by, "crps_ratio"), (king_by, "mase_ratio"),
                    (chal_by, "crps_ratio"), (chal_by, "mase_ratio")):
        col = _clean_ratio_column([by[f].get(key) for f in shared])
        if col is None:
            return None
        cols.append(col)
    return cols[0], cols[1], cols[2], cols[3]


def _geomean(crps_agg: np.ndarray, mase_agg: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(crps_agg, 1e-12) * np.maximum(mase_agg, 1e-12))


def evaluate_gift_gate(
    king_rows: list[dict],
    chal_rows: list[dict],
    *,
    tolerance: float,
    alpha: float,
    B: int,
    seed: int | str,
    min_configs: int,
) -> GiftGateResult:
    """Judge the public-benchmark no-regression gate. ``king_rows`` and
    ``chal_rows`` are the sidecar's per-config ratio rows (keyed by ``full``);
    they need not be pre-aligned — the join handles it.

    Returns a :class:`GiftGateResult`; ``computed=False`` (uncomputable → the
    round is inconclusive) when fewer than ``min_configs`` configs are shared or
    the ratios carry no finite values.
    """
    joined = _paired_ratio_columns(king_rows, chal_rows)
    if joined is None:
        return GiftGateResult(
            computed=False, passed=None, lcb=float("nan"), tolerance=tolerance,
            n_configs=0, king_agg=float("nan"), chal_agg=float("nan"),
            reason="no shared gift-eval configs (or no finite ratios)",
        )
    k_crps, k_mase, c_crps, c_mase = joined
    n = k_crps.shape[0]
    king_agg = float(_geomean(_shifted_gmean(k_crps), _shifted_gmean(k_mase)))
    chal_agg = float(_geomean(_shifted_gmean(c_crps), _shifted_gmean(c_mase)))
    if n < min_configs:
        return GiftGateResult(
            computed=False, passed=None, lcb=float("nan"), tolerance=tolerance,
            n_configs=n, king_agg=king_agg, chal_agg=chal_agg,
            reason=f"only {n} shared configs < min_configs {min_configs}",
        )

    rng = np.random.default_rng(_seed_to_int(seed))
    idx = rng.integers(0, n, size=(B, n))                     # (B, n) paired
    king_geo = _geomean(_shifted_gmean(k_crps[idx]), _shifted_gmean(k_mase[idx]))
    chal_geo = _geomean(_shifted_gmean(c_crps[idx]), _shifted_gmean(c_mase[idx]))
    safe_king = np.where(np.abs(king_geo) < 1e-9, 1e-9, king_geo)
    rel = (king_geo - chal_geo) / safe_king                  # + = challenger better
    lcb = float(np.quantile(rel, alpha))
    return GiftGateResult(
        computed=True,
        passed=bool(lcb >= -tolerance),
        lcb=lcb,
        tolerance=tolerance,
        n_configs=n,
        king_agg=king_agg,
        chal_agg=chal_agg,
    )
