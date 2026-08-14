"""Shadow channel-redundancy telemetry (DEC-CA-0022, in the DEC-CA-0010 shape:
MEASURE first, gate later — or never).

The feared multivariate exploit is C near-duplicate channels filling the token
budget with no information: byte-dedup cannot see it (``_series_key`` hashes
whole series; channels within one series are never compared), and while it is
self-financed (the miner pays full freight for redundant tokens), it erodes
corpus quality. Before any gate exists, the honest field's numbers must — per
DEC-CA-0008's threshold lesson — say where honest generators actually sit.
This module computes, per multichannel series:

* ``max_abs_corr`` — the maximum off-diagonal |Pearson| between standardized
  channels: 1.0 −ε is the jitter-duplicate signature;
* ``effective_rank`` — the participation ratio of the channel-covariance
  spectrum, ``(Σλ)² / Σλ²`` ∈ [1, C]: how many "real" channels the series
  carries.

An accumulator aggregates the per-series numbers into one summary record the
trainer folds into its run metrics (→ the public training log). SHADOW ONLY:
nothing reads these values in any scoring or acceptance path, and at ``C = 1``
(today's cap) the accumulator observes nothing and the metrics are unchanged.
The proposed ``max_channel_corr = 0.999`` gate arms only after these logs
clear honest generators (DEC-CA-0022 open question 6).

Cost: O(C²L) per multichannel series, zero for univariate.
"""

from __future__ import annotations

import numpy as np

# Numerical floor for a channel's standard deviation. A constant channel has
# no linear relationship with anything (its centered values are all zero), so
# it contributes ~0 correlation rather than NaN — redundancy of constant
# channels is the reject_constant gate's business, not this diagnostic's.
_STD_EPS = 1e-12


def series_channel_stats(arr: np.ndarray) -> tuple[float, float] | None:
    """``(max_abs_corr, effective_rank)`` for a ``(C, L)`` series; ``None``
    when the series has fewer than 2 channels (nothing to correlate)."""
    a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    n_ch = a.shape[0]
    if n_ch < 2:
        return None
    centered = a - a.mean(axis=1, keepdims=True)
    std = centered.std(axis=1)
    z = centered / np.maximum(std, _STD_EPS)[:, None]
    cov = (z @ z.T) / a.shape[1]          # correlation matrix of the floored z
    off = cov - np.diag(np.diag(cov))
    max_abs_corr = float(np.abs(off).max())
    # Participation ratio on the covariance spectrum of the *standardized*
    # channels — eigenvalues of the correlation matrix, so scale-free.
    eig = np.linalg.eigvalsh(cov)
    eig = np.maximum(eig, 0.0)
    total = float(eig.sum())
    if total <= 0.0:
        return (max_abs_corr, 1.0)
    effective_rank = float(total * total / float((eig * eig).sum()))
    return (max_abs_corr, effective_rank)


def corr_enforce_gate(cfg):
    """Per-series channel-correlation gate for the drain/stream paths, from
    ``[generator] channel_corr_mode`` / ``max_channel_corr`` (DEC-CA-0022).

    Returns ``None`` unless the mode is ``"enforce"`` — "off" does nothing and
    "shadow" is already served by the trainer's always-on accumulator (the
    telemetry IS the shadow log). The enforce bar targets near-identity only
    (the jitter-duplicate exploit); arming it before the shadow distribution
    clears honest generators is a config decision this code cannot stop, but
    the roadmap forbids.
    """
    if getattr(cfg, "channel_corr_mode", "off") != "enforce":
        return None
    bar = float(cfg.max_channel_corr)

    def gate(canon: np.ndarray, index: int | None = None) -> None:
        stats = series_channel_stats(canon)
        if stats is not None and stats[0] > bar:
            where = "" if index is None else f" (series {index})"
            raise ValueError(
                f"max off-diagonal channel |corr| {stats[0]:.6f} exceeds "
                f"max_channel_corr {bar:.6f}{where} (near-duplicate channels)"
            )

    return gate


class ChannelStatsAccumulator:
    """Streaming aggregator: observe every corpus series, summarise once.

    Per-series records are NOT retained (a 16k-series corpus must not grow the
    log by 16k rows); the summary carries the distribution ends that decide
    the gate question — how close honest generators come to the 0.999 bar, and
    how much real rank their channel groups carry.
    """

    def __init__(self) -> None:
        self._corrs: list[float] = []
        self._ranks: list[float] = []
        self._n_channels: list[int] = []

    def observe(self, arr: np.ndarray | dict) -> None:
        if isinstance(arr, dict):             # extended record: stats on values
            arr = arr["values"]
        a = np.asarray(arr)
        if a.ndim < 2 or a.shape[0] < 2:      # univariate: free, and silent
            return
        stats = series_channel_stats(a)
        if stats is None:
            return
        self._corrs.append(stats[0])
        self._ranks.append(stats[1])
        self._n_channels.append(int(a.shape[0]))

    @property
    def n_observed(self) -> int:
        return len(self._corrs)

    def summary(self) -> dict | None:
        """One log-ready record, or ``None`` when no multichannel series was
        seen — the trainer then omits the key entirely, keeping univariate
        run metrics byte-identical to pre-telemetry builds."""
        if not self._corrs:
            return None
        corrs = np.asarray(self._corrs)
        ranks = np.asarray(self._ranks)
        return {
            "n_multichannel_series": int(len(corrs)),
            "max_channels_seen": int(max(self._n_channels)),
            "max_abs_corr_p50": round(float(np.quantile(corrs, 0.50)), 6),
            "max_abs_corr_p90": round(float(np.quantile(corrs, 0.90)), 6),
            "max_abs_corr_p99": round(float(np.quantile(corrs, 0.99)), 6),
            "max_abs_corr_max": round(float(corrs.max()), 6),
            # How much of the corpus would trip the PROPOSED (unarmed) bar —
            # the one number the arming decision reads first.
            "frac_over_0999": round(float((corrs > 0.999).mean()), 6),
            "effective_rank_p50": round(float(np.quantile(ranks, 0.50)), 4),
            "effective_rank_min": round(float(ranks.min()), 4),
        }
