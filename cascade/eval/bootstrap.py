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


def _bag_geomeans(
    qloss_per_q: np.ndarray,
    abs_target: np.ndarray,
    mase: np.ndarray,
    idx: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """Per-bag geomean(MWSQL, mean MASE) under index resampling.

    Each bag aggregates ``qloss_per_q`` and ``abs_target`` separately, divides
    once (eps floor on the denominator), and geomeans with the bag's mean MASE.

    Shapes: qloss_per_q (N, num_q), abs_target (N,), mase (N,), idx (B, n).
    Returns (B,) bag geomeans.
    """
    qloss_bag = qloss_per_q[idx]                              # (B, n, num_q)
    abs_bag = abs_target[idx]                                 # (B, n)
    mase_bag = mase[idx]                                      # (B, n)
    bag_qloss_sum = qloss_bag.sum(axis=1)                    # (B, num_q)
    bag_abs_sum = np.maximum(abs_bag.sum(axis=1), eps)        # (B,)
    per_q = 2.0 * bag_qloss_sum / bag_abs_sum[:, None]        # (B, num_q)
    bag_mwsql = per_q.mean(axis=1)                            # (B,)
    bag_mase = mase_bag.mean(axis=1)                          # (B,)
    return np.sqrt(np.maximum(bag_mwsql, 1e-12) * np.maximum(bag_mase, 1e-12))


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
) -> float:
    """Paired bootstrap LCB on relative geomean improvement.

    The metric is the global geomean of ``MeanWeightedSumQuantileLoss`` and
    mean MASE. Each bag resamples window indices once (paired across king and
    challenger) and aggregates the MWSQL numerator/denominator separately
    before dividing — which removes the per-window pathology of MWSQL.

    Returns the ``alpha``-quantile of the bag relative differences; positive
    means the challenger reliably beat the king.
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
        return float("nan")
    if not np.allclose(king_abs_target, chal_abs_target):
        raise ValueError(
            "king_abs_target and chal_abs_target must be elementwise equal; "
            "windows are not paired correctly"
        )

    rng = np.random.default_rng(_seed_to_int(seed))
    idx = rng.integers(0, n, size=(B, n))
    king_geo = _bag_geomeans(king_qloss, king_abs_target, king_mase, idx)
    chal_geo = _bag_geomeans(chal_qloss, chal_abs_target, chal_mase, idx)
    safe_king = np.where(np.abs(king_geo) < 1e-9, 1e-9, king_geo)
    rel = (king_geo - chal_geo) / safe_king
    return float(np.quantile(rel, alpha))
