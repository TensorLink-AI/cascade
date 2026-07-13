"""DataGenerator — the miner-facing contract.

Every submitted generator must subclass :class:`DataGenerator`. The generator
is the adapter between the trainer's standard corpus-building protocol and
whatever arbitrary synthetic-data process the miner has designed.

The contract is intentionally narrow:

* construction takes the local repo directory and a deterministic ``seed``
* ``generate(n_series)`` yields exactly ``n_series`` univariate float series
* output is bounded: per-series length in ``[min_length, max_length]`` and a
  global cap on total emitted points, both enforced by the trainer

Determinism is load-bearing. The trainer seeds every generator from the chain
block hash at the training round's start, so two honest trainers (or a
validator re-running the trainer to audit) draw the *same* corpus. A generator
whose output depends on wall-clock, process entropy, or un-seeded RNG breaks
auditability and is rejected by the determinism check in
``cascade verify``.

The on-chain submission is a single pointer string
``metro-v1:gen:hippius:<repo>@<digest>``; the Hippius Hub ``repo@digest``
references the full repo tree — generator code, ``config.json``, and
``requirements.txt`` — together (the OCI digest *is* the content hash, so it
pins them).
A generator is **code-only** (purely algorithmic): it must NOT ship learned
weights of any kind, so the competition is on the data-generating prior, not on
a large pretrained forecaster distilled into a "generator". ``torch``/``gpytorch``
are available as compute libraries for GP/kernel priors. Determinism still
applies — seed every framework RNG (NumPy and, if used, ``torch.manual_seed`` +
``torch.use_deterministic_algorithms(True)``) so the corpus stays reproducible.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np


class DataGenerator(ABC):
    """Standard interface every submitted generator must implement.

    Implementations are loaded from a miner-controlled HF repo at a pinned git
    SHA. The subclass MUST be importable as ``generator.Generator`` from the
    cloned repo root. The trainer rejects generators whose code imports
    anything on the static-guard blocked list (see
    :mod:`cascade.interface.static_guard`) and runs them inside a
    network-isolated sandbox.
    """

    @abstractmethod
    def __init__(self, config_dir: str, *, seed: int) -> None:
        """Construct from a local repo directory and a deterministic seed.

        ``config_dir`` contains the materialised HF repo at the pinned
        revision: ``config.json``, ``generator.py``, and ``requirements.txt``
        (already installed by the trainer before this constructor runs).

        ``seed`` is the only source of randomness the generator is allowed to
        use. Derive every RNG from it (``np.random.default_rng(seed)``); do not
        read the system clock, ``os.urandom``, or any un-seeded global RNG. The
        constructor MUST NOT touch the network.
        """

    @abstractmethod
    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        """Yield exactly ``n_series`` training series.

        Each yielded value is a ``float`` ``np.ndarray`` (finite, no NaN or
        inf), either 1-D ``(L,)`` (univariate) or 2-D ``(C, L)`` (``C`` variates
        of length ``L``). A 1-D series is treated as a single channel ``(1, L)``.
        ``C`` must not exceed the configured ``max_channels`` (1 today), and the
        length ``L`` must fall within ``[min_length, max_length]``; the trainer
        validates every series with :func:`check_series` and aborts the round if
        any fails.

        The sequence MUST be a deterministic function of the ``seed`` passed to
        ``__init__`` and of ``n_series`` only.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable generator name, for operator logs."""


# ─────────────────────────────── output checks ─────────────────────────────

# Peak-|value| ceiling for a single series, defaulting to the float32 max. The
# trainer keeps raw series in float64 through the causal scaler (only the O(1)
# z-scores and asinh targets cast to float32), so a *uniformly* huge series is not
# the hazard — it standardizes to O(1) and trains fine. The residual hazard is the
# standardization ratio ``(x - loc) / scale``: ``scale`` is floored at ``eps=1e-5``
# (see ``cascade.trainer.toto2_model.causal_standardize``), so an extreme raw
# magnitude is what could push that ratio past float64's asinh limit to ``inf`` →
# NaN loss. Capping |value| here keeps the worst-case ratio ~7e43 (asinh ~102,
# finite). This is a conservative magnitude bound, not a hard invariant: it has no
# false positives on realistic data (real series top out ~1e13, far below the cap).
# The complementary in-trainer fix for large-but-finite z is to clamp |z|.
CAST_SAFE_MAX_FLOAT32 = 3.4028234663852886e38


def check_series(
    arr: object,
    *,
    min_length: int,
    max_length: int,
    max_channels: int = 1,
    max_abs: float | None = None,
    reject_constant: bool = False,
    index: int | None = None,
) -> None:
    """Validate a single emitted series. Raises ``ValueError`` on any problem.

    Accepts a 1-D ``(L,)`` series (univariate) or a 2-D ``(C, L)`` series
    (``C`` variates of length ``L``); a 1-D series counts as one channel. The
    length band applies to ``L`` and ``C`` must be in ``[1, max_channels]``.

    Two optional *data-quality* gates (both off by default so direct callers and
    ``cascade verify``'s static path are unchanged; the trainer turns them on
    from ``chain.toml [generator]``):

    * ``max_abs`` — reject a series whose peak magnitude exceeds this ceiling.
      Pass :data:`CAST_SAFE_MAX_FLOAT32` as a conservative bound that keeps the
      trainer's float64 standardization ratio (whose denominator is eps-floored)
      well clear of the asinh overflow that would yield NaN loss.
    * ``reject_constant`` — reject a flat (zero-range) series. The robust causal
      scaler clamps a constant series' scale to ``eps``, so it carries no
      gradient signal; a corpus of them trains nothing.

    Used by the trainer while draining ``generate`` and by ``cascade verify``
    on the miner side so a miner sees the same failure locally. The trainer
    catches the error, marks the generator's training run failed, and the
    challenger simply doesn't qualify this round — a bad generator can never
    poison the king.
    """
    where = "" if index is None else f" (series {index})"
    if not isinstance(arr, np.ndarray):
        raise ValueError(
            f"generate must yield np.ndarray{where}; got {type(arr).__name__}"
        )
    if arr.ndim not in (1, 2):
        raise ValueError(f"series must be 1-D (L,) or 2-D (C, L){where}; got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(f"series dtype must be floating{where}; got {arr.dtype}")
    channels = 1 if arr.ndim == 1 else int(arr.shape[0])
    if channels < 1 or channels > max_channels:
        raise ValueError(
            f"series has {channels} channels outside [1, {max_channels}]{where}"
        )
    n = int(arr.shape[-1])
    if n < min_length or n > max_length:
        raise ValueError(
            f"series length {n} outside [{min_length}, {max_length}]{where}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"series has non-finite values{where}")
    # ── data-quality gates (opt-in; run only after finiteness is established) ──
    if max_abs is not None:
        peak = float(np.abs(arr).max())
        if peak > max_abs:
            raise ValueError(
                f"series peak magnitude {peak:.3e} exceeds cast-safe max "
                f"{max_abs:.3e}{where}"
            )
    if reject_constant and float(np.ptp(arr)) == 0.0:
        raise ValueError(f"series is constant (zero range){where}")


def _series_key(canon: np.ndarray) -> bytes:
    """16-byte content digest of a canonical ``(C, L)`` float64 array.

    Storing the digest (not the bytes) keeps the dedup set flat regardless of
    corpus size; the channel count is folded in so a univariate and a
    single-channel-of-multivariate series never collide.
    """
    return hashlib.blake2b(
        canon.shape[0].to_bytes(4, "big") + canon.tobytes(), digest_size=16
    ).digest()


def drain_generator(
    gen: DataGenerator,
    n_series: int,
    *,
    min_length: int,
    max_length: int,
    max_total_points: int,
    max_channels: int = 1,
    max_abs: float | None = None,
    reject_constant: bool = False,
    max_dup_fraction: float = 1.0,
) -> list[np.ndarray]:
    """Pull ``n_series`` series from ``gen``, validating each one.

    Enforces the per-series length band, the channel cap, and a global cap on
    total emitted points (a memory / time guard against a generator that emits a
    few enormous series). Raises ``ValueError`` if the generator yields the wrong
    count, a malformed series, or blows the point budget.

    ``max_abs`` and ``reject_constant`` are forwarded to :func:`check_series` as
    per-series data-quality gates (see there). ``max_dup_fraction`` is a
    corpus-level gate: reject if the fraction of series that are exact byte-copies
    of an earlier one exceeds it (defence against "emit one series N times"
    spam). It is accumulated *during* the drain — one digest per series, no second
    pass — and ``1.0`` disables it. Byte-exact matching is deliberately the
    zero-false-positive choice: honest seeded continuous-parameter generators
    never byte-collide, so a loose cap only trips lazy duplication (a determined
    adversary can still evade it with 1e-15 jitter — that's a v1 floor, not an
    anti-adversary defence). All three gates default to no-op so existing callers
    are unchanged; the trainer sets them from ``chain.toml [generator]``.

    Each series is canonicalised to a contiguous ``(C, L)`` float64 array (a 1-D
    series becomes ``(1, L)``), so the corpus the base trainer consumes always
    carries a channel axis. The point budget counts every emitted value
    (``C * L``). Returns the list in yield order; the trainer hashes it (see
    :func:`cascade.shared.manifest.corpus_digest`) so the corpus is
    reproducible and auditable.
    """
    if n_series <= 0:
        raise ValueError(f"n_series must be positive; got {n_series}")
    dedup = max_dup_fraction < 1.0
    seen: set[bytes] = set()
    dups = 0
    out: list[np.ndarray] = []
    total = 0
    for i, arr in enumerate(gen.generate(n_series)):
        if i >= n_series:
            raise ValueError(
                f"generate yielded more than n_series={n_series} series"
            )
        check_series(
            arr, min_length=min_length, max_length=max_length,
            max_channels=max_channels, max_abs=max_abs,
            reject_constant=reject_constant, index=i,
        )
        canon = np.ascontiguousarray(np.atleast_2d(np.asarray(arr, dtype=np.float64)))
        total += int(canon.size)
        if total > max_total_points:
            raise ValueError(
                f"total emitted points {total} exceeds cap {max_total_points}"
            )
        if dedup:
            key = _series_key(canon)
            if key in seen:
                dups += 1
            else:
                seen.add(key)
        out.append(canon)
    if len(out) != n_series:
        raise ValueError(
            f"generate yielded {len(out)} series; expected exactly {n_series}"
        )
    if dedup:
        frac = dups / len(out)
        if frac > max_dup_fraction:
            raise ValueError(
                f"duplicate-series fraction {frac:.3f} exceeds cap "
                f"{max_dup_fraction:.3f} ({dups}/{len(out)} exact copies)"
            )
    return out
