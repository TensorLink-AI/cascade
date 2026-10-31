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
  reads the lag-0 correlation of first differences (shared innovations), as
  a length-normalised z-score ``|r|·sqrt(n−1)`` (≈ |N(0, 1)| under
  independence), takes each channel's strongest partner, and reports the
  weakest channel: a low value means some channel moves independently of
  every other channel. Honest coupling that is purely lagged or nonlinear
  also reads low — which is why this is measured before anything is gated.

An accumulator aggregates the per-series numbers into one summary record the
trainer folds into its run metrics (→ the public training log). SHADOW ONLY:
nothing reads these values in any scoring or acceptance path, and at ``C = 1``
(today's cap) the accumulator observes nothing and the metrics are unchanged.
The proposed ``max_channel_corr = 0.999`` gate arms only after these logs
clear honest generators (DEC-CA-0026 open question 6).

Cost: O(C²L) per multichannel series, zero for univariate.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

# Numerical floor for a channel's standard deviation. A constant channel has
# no linear relationship with anything (its centered values are all zero), so
# it contributes ~0 correlation rather than NaN — redundancy of constant
# channels is the reject_constant gate's business, not this diagnostic's.
_STD_EPS = 1e-12

# A channel whose strongest partner reads below this innovation z-score is
# "unpartnered". Under independence each pairwise z is ≈ |N(0, 1)|, so even
# the best of 31 partners (C = 32) exceeds 4 only ~0.2% of the time, while a
# real r = 0.1 shared-innovation coupling at L = 4096 reads z ≈ 6.4.
_UNPARTNERED_Z = 4.0


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


def series_min_partner_z(arr: np.ndarray, groups=None) -> float | None:
    """Weakest channel's strongest shared-innovation coupling, as a z-score;
    ``None`` below 2 channels or 3 steps. A channel that never moves has no
    innovations and reads 0."""
    a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
    if a.shape[0] < 2 or a.shape[1] < 3:
        return None
    d = np.diff(a, axis=1)
    corr = _corr_matrix(d)
    np.fill_diagonal(corr, 0.0)
    if groups is not None:
        labels = np.asarray(groups)
        corr = np.where(labels[:, None] == labels[None, :], corr, 0.0)
    partner = np.abs(corr).max(axis=1)
    return float(partner.min() * np.sqrt(d.shape[1] - 1))


def corr_enforce_gate(cfg):
    """Per-series channel-correlation gate for the drain/stream paths, from
    ``[generator] channel_corr_mode`` / ``max_channel_corr`` (DEC-CA-0026).

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
        groups = None
        if isinstance(arr, dict):
            groups = arr.get("groups")
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

        payload = np.ascontiguousarray(a).tobytes()
        if groups is not None:
            payload += b"\x00groups" + np.asarray(groups, dtype=np.uint8).tobytes()
        key = hashlib.blake2b(payload, digest_size=16).digest()
        if key in self._seen:
            return
        self._seen.add(key)
        stats = series_channel_stats(a)
        if stats is None:
            return
        self._corrs.append(stats[0])
        self._ranks.append(stats[1])
        self._n_channels.append(int(a.shape[0]))
        partner_z = series_min_partner_z(a, groups=groups)
        if partner_z is not None:
            self._partner_z.append(partner_z)

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


_STOP = object()


class BackgroundObserver:
    """Run ``fn(item)`` for every submitted item on ONE background thread, in
    submission order (a FIFO queue, so a stateful accumulator sees exactly
    the inline sequence and ends in exactly the inline state).

    Telemetry only: it moves :meth:`ChannelStatsAccumulator.observe` (an
    O(C²L) numpy pass + a content hash per multichannel series) off the
    training thread. The queue is bounded (``maxsize`` items held by
    reference — the stream yields a fresh array per series and the batcher
    only reads them), so a slow observer back-pressures the stream instead
    of growing memory. The first exception ``fn`` raises is kept, later items
    are dropped, and it is re-raised on the next :meth:`__call__` or by
    :meth:`close` — an observer failure still fails the run, as it did
    inline. Call :meth:`close` before reading the accumulator.
    """

    def __init__(self, fn, *, maxsize: int = 64, name: str = "channel-stats") -> None:
        self._fn = fn
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(maxsize)))
        self._error: BaseException | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is _STOP:
                    return
                if self._error is None:
                    self._fn(item)
            except BaseException as e:  # noqa: BLE001 — re-raised on the caller's thread
                self._error = e
            finally:
                self._q.task_done()

    def _raise(self) -> None:
        if self._error is not None:
            raise self._error

    def __call__(self, item) -> None:
        if self._closed:
            raise RuntimeError("BackgroundObserver is closed")
        self._raise()
        self._q.put(item)

    def flush(self) -> None:
        """Block until every submitted item has been observed."""
        self._q.join()
        self._raise()

    def close(self, *, raise_errors: bool = True) -> None:
        """Drain the queue, stop the thread and (by default) re-raise the
        observer's failure. Idempotent."""
        if not self._closed:
            self._closed = True
            self._q.put(_STOP)
            self._thread.join()
        if raise_errors:
            self._raise()
