"""Shadow channel-redundancy telemetry (DEC-CA-0026, in the DEC-CA-0010 shape:
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
  carries;
* ``min_partner_z`` — the opposite end of the axis: unrelated series stacked
  into one array. Under ``budget_denomination = "series_points"`` a ``(C, L)``
  series bills ``L`` points but trains ``C×`` channel tokens, so gluing
  independent univariate rows buys tokens without any cross-channel signal —
  it is no longer self-financed the way duplication is. Level correlation
  cannot see it (independent random walks correlate spuriously), so this
  reads the cross-correlation of first differences (shared innovations) at
  every lag within ±``_PARTNER_MAX_LAG``, as a length-normalised z-score
  ``max|r|·sqrt(n−1)``, takes each channel's strongest partner at its best
  lag, and reports the weakest channel: a low value means some channel moves
  independently of every other channel at every nearby lag. Scanning lags
  keeps honest lead/lag coupling (a lagged causal DAG) partnered; a lag-0
  read flags it. Coupling that is purely nonlinear, or lagged beyond the
  window, still reads low — the gate below judges a fraction for that reason.

An accumulator aggregates the per-series numbers into one summary record the
trainer folds into its run metrics (→ the public training log). SHADOW ONLY:
nothing reads these values in any scoring or acceptance path, and at ``C = 1``
(today's cap) the accumulator observes nothing and the metrics are unchanged.
The proposed ``max_channel_corr = 0.999`` gate arms only after these logs
clear honest generators (DEC-CA-0026 open question 6).

Cost: O(C²L) per multichannel series for ``series_channel_stats`` and
O(C² L log L) for ``series_min_partner_z`` (≈0.6 ms at C = 2, ≈4 ms at C = 8,
≈40 ms at C = 32 for L = 4096); zero for univariate. The always-on
accumulator reads the partner statistic on every series up to the eval's
8-channel cap and on a content-hashed 1-in-8 sample above it.
"""

from __future__ import annotations

import numpy as np

# Numerical floor for a channel's standard deviation. A constant channel has
# no linear relationship with anything (its centered values are all zero), so
# it contributes ~0 correlation rather than NaN — redundancy of constant
# channels is the reject_constant gate's business, not this diagnostic's.
_STD_EPS = 1e-12

# Lags scanned for a channel's strongest partner (both directions).
_PARTNER_MAX_LAG = 64

# A channel whose strongest partner reads below this innovation z-score is
# "unpartnered". The score is a maximum over 2·64+1 lags, so the bar sits
# above the Gaussian null of that maximum (median ≈ 3.1) with headroom for
# heavy-tailed innovations (spikes, steps, counts), which inflate chance
# peaks; a real r = 0.1 coupling at L = 4096 still reads z ≈ 6.4.
_UNPARTNERED_Z = 5.0

# Always-on telemetry reads the O(C² L log L) partner statistic on every
# series up to this width (the eval's channel cap) and on a content-hashed
# sample above it, so wide honest generators do not pay for the diagnostic.
_PARTNER_FULL_MAX_C = 8
_PARTNER_WIDE_SAMPLE = 8

# The unpartnered enforce gate decides on a fraction, so it waits for this
# many multichannel series before judging: a generator whose first few
# groups happen to be weakly coupled is not rejected on a handful of rows.
_UNPARTNERED_MIN_SERIES = 64


def _corr_matrix(a: np.ndarray) -> np.ndarray:
    """Correlation matrix of the rows of ``a`` with the std floor applied."""
    centered = a - a.mean(axis=1, keepdims=True)
    std = centered.std(axis=1)
    z = centered / np.maximum(std, _STD_EPS)[:, None]
    return (z @ z.T) / a.shape[1]


def series_channel_stats(arr: np.ndarray) -> tuple[float, float] | None:
    """``(max_abs_corr, effective_rank)`` for a ``(C, L)`` series; ``None``
    when the series has fewer than 2 channels (nothing to correlate)."""
    a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    n_ch = a.shape[0]
    if n_ch < 2:
        return None
    cov = _corr_matrix(a)                 # correlation matrix of the floored z
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


def series_min_partner_z(arr: np.ndarray) -> float | None:
    """Weakest channel's strongest shared-innovation coupling at any lag
    within ±``_PARTNER_MAX_LAG``, as a z-score; ``None`` below 2 channels or
    3 steps. A channel that never moves has no innovations and reads 0."""
    a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if a.shape[0] < 2 or a.shape[1] < 3:
        return None
    d = np.diff(a, axis=1)
    centered = d - d.mean(axis=1, keepdims=True)
    z = centered / np.maximum(centered.std(axis=1), _STD_EPS)[:, None]
    n = z.shape[1]
    lag = min(_PARTNER_MAX_LAG, n - 1)
    m = 1 << int(np.ceil(np.log2(2 * n)))   # zero-pad: linear, not circular
    spec = np.fft.rfft(z, n=m, axis=1)
    xc = np.fft.irfft(spec[:, None, :] * np.conj(spec[None, :, :]), n=m, axis=2) / n
    window = np.concatenate([xc[..., :lag + 1], xc[..., m - lag:]], axis=2)
    peak = np.abs(window).max(axis=2)
    np.fill_diagonal(peak, 0.0)
    return float(peak.max(axis=1).min() * np.sqrt(n - 1))


def corr_enforce_gate(cfg):
    """Channel gate for the drain/stream paths (DEC-CA-0026), covering both
    ends of the channel axis:

    * ``[generator] channel_corr_mode`` / ``max_channel_corr`` — per series,
      near-identical channels (the jitter-duplicate exploit);
    * ``[generator] unpartnered_mode`` / ``max_unpartnered_frac`` — per run,
      the fraction of multichannel series carrying a channel whose
      innovations are uncorrelated with every other channel (unrelated rows
      stacked on the channel axis). Judged on a fraction, never one series,
      because honest lagged or nonlinear coupling can read low at lag 0.

    Returns ``None`` unless at least one mode is ``"enforce"`` — "off" does
    nothing and "shadow" is already served by the trainer's always-on
    accumulator (the telemetry IS the shadow log). Arming either before its
    shadow distribution clears honest generators is a config decision this
    code cannot stop, but the roadmap forbids. The returned gate is stateful
    (it counts the run's multichannel series), so build one per stream.
    """
    corr_on = getattr(cfg, "channel_corr_mode", "off") == "enforce"
    partner_on = getattr(cfg, "unpartnered_mode", "off") == "enforce"
    if not (corr_on or partner_on):
        return None
    corr_bar = float(cfg.max_channel_corr) if corr_on else 0.0
    frac_bar = float(cfg.max_unpartnered_frac) if partner_on else 0.0
    counts = {"multichannel": 0, "unpartnered": 0}

    def gate(canon: np.ndarray, index: int | None = None) -> None:
        where = "" if index is None else f" (series {index})"
        if corr_on:
            stats = series_channel_stats(canon)
            if stats is not None and stats[0] > corr_bar:
                raise ValueError(
                    f"max off-diagonal channel |corr| {stats[0]:.6f} exceeds "
                    f"max_channel_corr {corr_bar:.6f}{where} (near-duplicate channels)"
                )
        if partner_on:
            z = series_min_partner_z(canon)
            if z is None:
                return
            counts["multichannel"] += 1
            if z < _UNPARTNERED_Z:
                counts["unpartnered"] += 1
            n = counts["multichannel"]
            if n >= _UNPARTNERED_MIN_SERIES:
                frac = counts["unpartnered"] / n
                if frac > frac_bar:
                    raise ValueError(
                        f"unpartnered multichannel fraction {frac:.3f} over {n} "
                        f"series exceeds max_unpartnered_frac {frac_bar:.3f}{where} "
                        "(channels whose innovations are uncorrelated with every "
                        "other channel: unrelated rows stacked on the channel axis)"
                    )

    return gate


class ChannelStatsAccumulator:
    """Streaming aggregator: observe every corpus series, summarise once.

    Per-series records are NOT retained (a 16k-series corpus must not grow the
    log by 16k rows); the summary carries the distribution ends that decide
    the gate question — how close honest generators come to the 0.999 bar, how
    much real rank their channel groups carry, and how often a channel shares
    no innovations with any other channel.
    """

    def __init__(self) -> None:
        self._corrs: list[float] = []
        self._ranks: list[float] = []
        self._partner_z: list[float] = []
        self._n_channels: list[int] = []
        self._seen: set[bytes] = set()

    def observe(self, arr: np.ndarray | dict) -> None:
        if isinstance(arr, dict):             # extended record: stats on values
            arr = arr["values"]
        a = np.asarray(arr)
        if a.ndim < 2 or a.shape[0] < 2:      # univariate: free, and silent
            return
        # Dedup by content: in cache_reuse mode the training stream CYCLES the
        # corpus under the token budget, and counting each pass would inflate
        # n_multichannel_series and repeat every stat in the quantiles — the
        # summary must describe the CORPUS, not the schedule. One 16-byte key
        # per unique multichannel series; fresh streams never collide.
        import hashlib

        key = hashlib.blake2b(
            np.ascontiguousarray(a).tobytes(), digest_size=16
        ).digest()
        if key in self._seen:
            return
        self._seen.add(key)
        stats = series_channel_stats(a)
        if stats is None:
            return
        self._corrs.append(stats[0])
        self._ranks.append(stats[1])
        self._n_channels.append(int(a.shape[0]))
        if a.shape[0] <= _PARTNER_FULL_MAX_C or key[0] % _PARTNER_WIDE_SAMPLE == 0:
            partner_z = series_min_partner_z(a)
            if partner_z is not None:
                self._partner_z.append(partner_z)

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
        out = {
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
        if self._partner_z:
            pz = np.asarray(self._partner_z)
            out["min_partner_z_p10"] = round(float(np.quantile(pz, 0.10)), 4)
            out["min_partner_z_p50"] = round(float(np.quantile(pz, 0.50)), 4)
            # Share of multichannel series carrying at least one channel that
            # moves independently of all the others (the glued-rows shape).
            out["frac_unpartnered"] = round(
                float((pz < _UNPARTNERED_Z).mean()), 6)
        return out
