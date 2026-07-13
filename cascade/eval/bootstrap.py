"""Paired bootstrap LCB — the core statistic for the KOTH decision.

Both the king's trained model and the challenger's trained model are scored on
the *same* eval windows, so the comparison is paired: window-level difficulty
cancels in the per-window difference, giving 2-5x tighter CIs than comparing
two independent LCBs. Margins are expressed as RELATIVE fractions of the king's
score, and lower scores are better (CRPS, MASE, their geomean):

    rel_diff = (king - challenger) / king

A positive LCB means the challenger reliably beats the king by at least that
fraction.

Determinism: ``seed`` accepts a string and hashes to an int internally. In
production the seed is the chain block hash at the round's start, so every
validator draws identical bootstrap samples on the same comparison.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _seed_to_int(seed: int | str) -> int:
    if isinstance(seed, int):
        return seed
    h = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "big", signed=False)


def _relative_diffs(king: np.ndarray, challenger: np.ndarray) -> np.ndarray:
    """``(king - challenger) / king`` with a small floor on the denominator."""
    safe_king = np.where(np.abs(king) < 1e-9, 1e-9, king)
    return (king - challenger) / safe_king


def paired_bootstrap_lcb(
    king_scores: np.ndarray,
    challenger_scores: np.ndarray,
    alpha: float = 0.05,
    B: int = 10000,
    seed: int | str = 42,
) -> float:
    """One-sided lower confidence bound on the relative improvement of
    challenger over king, on a per-window scalar metric (lower better).

    Returns the ``alpha``-quantile of the bootstrap distribution of mean
    relative differences. Positive means the challenger reliably beats the king.
    """
    king = np.asarray(king_scores, dtype=np.float64)
    chal = np.asarray(challenger_scores, dtype=np.float64)
    if king.shape != chal.shape:
        raise ValueError(f"shape mismatch: king {king.shape} vs challenger {chal.shape}")
    if king.ndim != 1:
        raise ValueError(f"scores must be 1-D; got {king.shape}")
    rel = _relative_diffs(king, chal)
    n = rel.shape[0]
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(_seed_to_int(seed))
    idx = rng.integers(0, n, size=(B, n))
    boot_means = rel[idx].mean(axis=1)
    return float(np.quantile(boot_means, alpha))


def cluster_codes(clusters: list | np.ndarray | None, n: int) -> np.ndarray:
    """Map per-window cluster labels to dense integer codes ``(n,)``.

    ``None`` means every window is its own cluster (the classic i.i.d.
    bootstrap). Labels are grouped by value in first-appearance order, so the
    coding — and therefore the bootstrap draw for a fixed seed — is
    deterministic in the (already deterministic) window order.
    """
    if clusters is None:
        return np.arange(n)
    if len(clusters) != n:
        raise ValueError(f"clusters length {len(clusters)} != n windows {n}")
    codes: dict = {}
    out = np.empty(n, dtype=np.int64)
    for i, label in enumerate(clusters):
        out[i] = codes.setdefault(label, len(codes))
    return out


def _cluster_sums(
    qloss_per_q: np.ndarray,
    abs_target: np.ndarray,
    mase: np.ndarray,
    codes: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster sufficient statistics for the bag metric.

    The bag metric is built from sums (qloss / abs-target numerators and
    denominators; log-MASE totals and counts), so a cluster resample only needs
    each cluster's sums — no per-window indexing inside the bootstrap loop.
    """
    nq = qloss_per_q.shape[1]
    qloss_c = np.zeros((n_clusters, nq))
    for q in range(nq):
        qloss_c[:, q] = np.bincount(codes, weights=qloss_per_q[:, q], minlength=n_clusters)
    abs_c = np.bincount(codes, weights=abs_target, minlength=n_clusters)
    logmase_c = np.bincount(
        codes, weights=np.log(np.maximum(mase, 1e-9)), minlength=n_clusters
    )
    n_c = np.bincount(codes, minlength=n_clusters).astype(np.float64)
    return qloss_c, abs_c, logmase_c, n_c


def _bag_geomeans(
    qloss_c: np.ndarray,
    abs_c: np.ndarray,
    logmase_c: np.ndarray,
    n_c: np.ndarray,
    idx: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Per-bag geomean(MWSQL, geomean MASE) under cluster resampling.

    Each bag draws clusters with replacement (``idx`` indexes clusters),
    aggregates the MWSQL numerator/denominator separately, and divides once
    (eps floor) — removing the per-window pathology of MWSQL. MASE is
    aggregated as a *geometric* mean (log-space): per-window MASE differences
    are heavy-tailed, and an arithmetic mean lets a single exploding window
    dominate every bag it lands in, inflating the LCB's variance.

    Shapes: qloss_c (G, num_q), abs_c / logmase_c / n_c (G,), idx (B, g).
    Returns (B,) bag geomeans.
    """
    bag_qloss_sum = qloss_c[idx].sum(axis=1)                  # (B, num_q)
    bag_abs_sum = np.maximum(abs_c[idx].sum(axis=1), eps)     # (B,)
    per_q = 2.0 * bag_qloss_sum / bag_abs_sum[:, None]        # (B, num_q)
    bag_mwsql = per_q.mean(axis=1)                            # (B,)
    bag_n = np.maximum(n_c[idx].sum(axis=1), 1.0)             # (B,)
    bag_mase = np.exp(logmase_c[idx].sum(axis=1) / bag_n)     # (B,) geomean MASE
    return np.sqrt(np.maximum(bag_mwsql, 1e-12) * np.maximum(bag_mase, 1e-12))


def _rel_bootstrap_aggregated(
    king_qloss: np.ndarray,
    king_abs_target: np.ndarray,
    king_mase: np.ndarray,
    chal_qloss: np.ndarray,
    chal_abs_target: np.ndarray,
    chal_mase: np.ndarray,
    *,
    B: int,
    seed: int | str,
    clusters: list | np.ndarray | None,
) -> np.ndarray:
    """The ``(B,)`` paired-cluster bootstrap distribution of relative geomean
    improvement ``(king − chal) / king``. Shared core of the decision LCB and the
    diagnostic spread, so both read quantiles off the *same* draws and can never
    disagree. Returns an empty array when there are no windows.
    """
    if king_qloss.shape != chal_qloss.shape:
        raise ValueError(
            f"qloss shape mismatch: king {king_qloss.shape} vs chal {chal_qloss.shape}"
        )
    if king_qloss.ndim != 2:
        raise ValueError(f"qloss must be (N, num_q); got {king_qloss.shape}")
    n = king_qloss.shape[0]
    for name, arr in (
        ("king_abs_target", king_abs_target),
        ("chal_abs_target", chal_abs_target),
        ("king_mase", king_mase),
        ("chal_mase", chal_mase),
    ):
        if arr.shape != (n,):
            raise ValueError(f"{name} shape {arr.shape}; expected ({n},)")
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if not np.allclose(king_abs_target, chal_abs_target):
        raise ValueError(
            "king_abs_target and chal_abs_target must be elementwise equal; "
            "windows are not paired correctly"
        )

    codes = cluster_codes(clusters, n)
    g = int(codes.max()) + 1
    king_c = _cluster_sums(king_qloss, king_abs_target, king_mase, codes, g)
    chal_c = _cluster_sums(chal_qloss, chal_abs_target, chal_mase, codes, g)

    rng = np.random.default_rng(_seed_to_int(seed))
    idx = rng.integers(0, g, size=(B, g))
    king_geo = _bag_geomeans(*king_c, idx)
    chal_geo = _bag_geomeans(*chal_c, idx)
    safe_king = np.where(np.abs(king_geo) < 1e-9, 1e-9, king_geo)
    return (king_geo - chal_geo) / safe_king


def paired_bootstrap_lcb_aggregated(
    king_qloss: np.ndarray,
    king_abs_target: np.ndarray,
    king_mase: np.ndarray,
    chal_qloss: np.ndarray,
    chal_abs_target: np.ndarray,
    chal_mase: np.ndarray,
    alpha: float = 0.05,
    B: int = 10_000,
    seed: int | str = 42,
    clusters: list | np.ndarray | None = None,
) -> float:
    """Paired (cluster) bootstrap LCB on relative geomean improvement.

    The metric is the global geomean of ``MeanWeightedSumQuantileLoss`` and
    geometric-mean MASE. Each bag resamples once (paired across king and
    challenger) and aggregates the MWSQL numerator/denominator separately
    before dividing — which removes the per-window pathology of MWSQL.

    ``clusters`` (optional, one hashable label per window — e.g. the upstream
    feed id from pool metadata ``source``) switches to a **cluster bootstrap**:
    whole clusters are resampled, never individual windows. Windows from one
    feed are correlated in which model they favour, so resampling them
    independently understates the variance and yields an overconfident LCB;
    with clusters the effective sample size is the number of feeds, which is
    the honest one. ``None`` keeps the classic per-window bootstrap.

    Returns the ``alpha``-quantile of the bag relative differences; positive
    means the challenger reliably beat the king.
    """
    rel = _rel_bootstrap_aggregated(
        king_qloss, king_abs_target, king_mase,
        chal_qloss, chal_abs_target, chal_mase,
        B=B, seed=seed, clusters=clusters,
    )
    if rel.size == 0:
        return float("nan")
    return float(np.quantile(rel, alpha))


def paired_bootstrap_quantiles_aggregated(
    king_qloss: np.ndarray,
    king_abs_target: np.ndarray,
    king_mase: np.ndarray,
    chal_qloss: np.ndarray,
    chal_abs_target: np.ndarray,
    chal_mase: np.ndarray,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
    B: int = 10_000,
    seed: int | str = 42,
    clusters: list | np.ndarray | None = None,
) -> dict[float, float]:
    """Diagnostic spread of the *same* bootstrap the LCB gates on.

    Returns ``{q: value}`` for each requested quantile of the relative-improvement
    distribution. With identical ``B``/``seed``/``clusters`` the ``alpha``-quantile
    here equals :func:`paired_bootstrap_lcb_aggregated`'s LCB by construction, so
    reporting ``p5 / p50 / p95`` shows how far the median improvement sits above a
    negative LCB (a wide gap = the point estimate is carried by a fragile tail).
    Never gates — display only. NaN per quantile when there are no windows.
    """
    rel = _rel_bootstrap_aggregated(
        king_qloss, king_abs_target, king_mase,
        chal_qloss, chal_abs_target, chal_mase,
        B=B, seed=seed, clusters=clusters,
    )
    if rel.size == 0:
        return {q: float("nan") for q in quantiles}
    return {float(q): float(np.quantile(rel, q)) for q in quantiles}
