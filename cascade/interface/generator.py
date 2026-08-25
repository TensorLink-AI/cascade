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
from collections.abc import Iterator, Mapping

import numpy as np

# ── the record carrier (DEC-CA-0020) ─────────────────────────────────────────

# The interface version this code accepts. A generator repo may declare
# ``"interface_version"`` in its config.json; a declared version NEWER than
# this is rejected before the generator runs (clear early error instead of a
# mid-drain validation failure). Absent ⇒ 1. Bump only when a new payload
# field is ACCEPTED (DEC-CA-0020's refuse-unconsumed rule).
SUPPORTED_INTERFACE_VERSION = 1

# The one field a record yield must carry at interface_version 1.
VALUES_FIELD = "values"

# Reserved payload names: published semantics, refused data. Each is claimed by
# a decision record and waits for the release that CONSUMES it (trainer + eval
# together); until then a yield carrying one is hard-rejected, so no miner can
# squat a name and no accepted-but-ignored field can ever exist. The table is
# the migration path: accepting a field later is additive and breaks nobody.
RESERVED_FIELDS: dict[str, str] = {
    "mask": "per-entry observedness mask parallel to values (DEC-CA-0023)",
    "start": "absolute time anchor of the first step (DEC-CA-0021; no consumer under the calendar-free arch pin)",
    "freq": "sampling frequency / calendar step (DEC-CA-0021)",
    "group_id": "cross-series panel/group label (DEC-CA-0024; panels are variates for this arch)",
    "roles": "per-channel variate roles: target / past-cov / future-known (DEC-CA-0026)",
    "labels": "per-series free-form tags (no consumer scheduled)",
    "quantiles": "per-step distributional payload (no consumer scheduled)",
}


def canonicalize_yield(
    item: object,
    *,
    accepted: tuple[str, ...] | list[str] = (),
    index: int | None = None,
) -> object:
    """Normalise one ``generate()`` yield.

    The carrier accepts two shapes per DEC-CA-0020:

    * a bare ``np.ndarray`` — the documented common case, passed through
      untouched (zero cost, zero behaviour change for every deployed
      generator);
    * a named-field record (any ``Mapping``) carrying ``"values"`` plus, when
      the operator has armed them (``[training] accepted_fields``), any of the
      ``accepted`` payload fields. A reserved name NOT in ``accepted`` is
      hard-rejected with its semantics named; unknown names are hard-rejected
      outright.

    Returns the bare values array when no accepted extra is present — a
    values-only record corpus hashes byte-for-byte the same as its bare-array
    twin — or a ``{"values": …, <field>: …}`` dict when accepted extras ride
    along (their validation is :func:`check_record`, downstream). Raises
    ``ValueError`` on any malformed record.
    """
    if not isinstance(item, Mapping):
        return item
    where = "" if index is None else f" (series {index})"
    extras: dict[str, object] = {}
    for key in item:
        if not isinstance(key, str):
            raise ValueError(
                f"record field names must be strings{where}; got {type(key).__name__}"
            )
        if key == VALUES_FIELD:
            continue
        if key in RESERVED_FIELDS:
            if key in accepted:
                extras[key] = item[key]
                continue
            raise ValueError(
                f"record field {key!r} is reserved but not yet accepted{where}: "
                f"{RESERVED_FIELDS[key]}. interface_version "
                f"{SUPPORTED_INTERFACE_VERSION} accepts {VALUES_FIELD!r}"
                + (f" plus {sorted(accepted)}" if accepted else " only") + "."
            )
        raise ValueError(
            f"unknown record field {key!r}{where}; interface_version "
            f"{SUPPORTED_INTERFACE_VERSION} accepts {VALUES_FIELD!r}"
            + (f" plus {sorted(accepted)}" if accepted else " only")
        )
    if VALUES_FIELD not in item:
        raise ValueError(f"record yield is missing the {VALUES_FIELD!r} field{where}")
    if not extras:
        return item[VALUES_FIELD]
    return {VALUES_FIELD: item[VALUES_FIELD], **extras}


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

        Optional opt-in keyword (DEC-CA-0028, only while ``[training]
        real_corpus_ref`` is armed — never passed today): a constructor that
        declares ``real_corpus_dir=None`` additionally receives the local
        read-only path of the owner-published shared real corpus, identical
        bytes on every machine (content-pinned). Treat the PATH as opaque —
        derive outputs from the corpus *contents* only; a generator that bakes
        the path (which differs per machine) into its output is
        non-deterministic across hosts and fails the audit. Constructors
        without the keyword are unaffected under every config.
        """

    @abstractmethod
    def generate(self, n_series: int) -> Iterator[np.ndarray | Mapping]:
        """Yield exactly ``n_series`` training series.

        Each yield is EITHER a bare array — the documented common case — or a
        named-field record (a ``dict``) whose ``"values"`` key carries that
        same array (DEC-CA-0020). At ``interface_version`` 1 the record form
        may carry the ``"values"`` field ONLY: reserved names
        (:data:`RESERVED_FIELDS` — ``mask``, ``start``, ``freq``, ``group_id``,
        ``roles``, ``labels``, ``quantiles``) have published semantics but are
        hard-rejected until the release that consumes them, and unknown names
        are always rejected. A values-only record corpus is byte-identical
        (same digest) to its bare-array twin, so the two forms are freely
        interchangeable.

        The values array is a ``float`` ``np.ndarray`` (finite, no NaN or
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


# ── extended records: canonical form, validation, digest framing ─────────────
#
# The DEC-CA-0020 sentinel scheme. A values-only series' digest bytes begin
# with an 8-byte (corpus/stream) or 4-byte (series key) big-endian channel
# count — first byte 0x00 for any realistic C — so the extended-record frame
# claims the one lead byte no legacy frame can produce:
#
#   0xFF | n_fields (8B BE) | for each field name in SORTED order:
#         len(name) 8B BE | name utf-8 | ndim 8B BE | each dim 8B BE | raw bytes
#
# Field dtypes are pinned by _CANON_DTYPES (values float64; mask/roles uint8),
# so the raw bytes are unambiguous without a dtype tag. Values-only corpora
# never touch this path and keep their frozen bytes forever.

RECORD_SENTINEL = b"\xff"

_CANON_DTYPES = {"values": np.float64, "mask": np.uint8, "roles": np.uint8}


def canonicalize_record(rec: dict) -> dict[str, np.ndarray]:
    """Canonical arrays for an extended record: ``values`` → contiguous
    ``(C, L)`` float64; ``mask`` → contiguous ``(C, L)`` uint8; ``roles`` →
    contiguous ``(C,)`` uint8. Shape/content validity is :func:`check_record`'s
    job — this only fixes dtype/layout so digests are byte-stable."""
    out: dict[str, np.ndarray] = {}
    for name, arr in rec.items():
        dtype = _CANON_DTYPES.get(name, np.float64)
        a = np.asarray(arr, dtype=dtype)
        if name != "roles":
            a = np.atleast_2d(a)
        out[name] = np.ascontiguousarray(a)
    return out


def record_frame_bytes(canon_rec: dict[str, np.ndarray]) -> bytes:
    """The extended record's digest framing (see the scheme above)."""
    parts = [RECORD_SENTINEL, len(canon_rec).to_bytes(8, "big")]
    for name in sorted(canon_rec):
        a = canon_rec[name]
        nb = name.encode("utf-8")
        parts.append(len(nb).to_bytes(8, "big"))
        parts.append(nb)
        parts.append(a.ndim.to_bytes(8, "big"))
        for dim in a.shape:
            parts.append(int(dim).to_bytes(8, "big"))
        parts.append(a.tobytes())
    return b"".join(parts)


def _record_key(canon_rec: dict[str, np.ndarray]) -> bytes:
    """Dedup key for an extended record — 0xFF-framed, so it can never collide
    with a values-only :func:`_series_key` (whose input starts 0x00…)."""
    return hashlib.blake2b(record_frame_bytes(canon_rec), digest_size=16).digest()


def check_record(
    canon_rec: dict[str, np.ndarray],
    *,
    max_missing_frac: float = 1.0,
    allow_future_known: bool = False,
    index: int | None = None,
) -> None:
    """Validate an extended record's accepted extras against its values.

    ``values`` is assumed already validated by :func:`check_series`. Rules:

    * ``mask`` (DEC-CA-0023): same ``(C, L)`` shape as values; entries in
      {0, 1} (1 = missing/unobserved); every masked entry of ``values``
      pinned EXACTLY 0.0 (one canonical byte encoding per missingness
      pattern — a free-valued filler would be a hidden data channel and
      would randomise the dup gate); overall missing fraction ≤
      ``max_missing_frac``; no fully-masked channel (nothing observed to
      scale from).
    * ``roles`` (DEC-CA-0026): shape ``(C,)``; entries in {0=target,
      1=past-covariate, 2=future-known}; value 2 admitted only when
      ``allow_future_known`` (armed only after the EVAL_POOL exogeneity rule
      exists); at least one target channel (an all-covariate series carries
      no loss signal).
    """
    where = "" if index is None else f" (series {index})"
    values = canon_rec[VALUES_FIELD]
    mask = canon_rec.get("mask")
    if mask is not None:
        if mask.shape != values.shape:
            raise ValueError(
                f"mask shape {mask.shape} != values shape {values.shape}{where}"
            )
        if mask.max(initial=0) > 1:
            raise ValueError(f"mask entries must be 0/1{where}")
        masked = mask.astype(bool)
        if masked.any() and np.any(values[masked] != 0.0):
            raise ValueError(
                f"masked values must be pinned exactly 0.0{where} (the filler is "
                "canonical, not a data channel)"
            )
        frac = float(masked.mean())
        if frac > max_missing_frac:
            raise ValueError(
                f"missing fraction {frac:.3f} exceeds max_missing_frac "
                f"{max_missing_frac:.3f}{where}"
            )
        if masked.all(axis=1).any():
            raise ValueError(f"mask covers an entire channel{where}")
    roles = canon_rec.get("roles")
    if roles is not None:
        n_channels = values.shape[0]
        if roles.shape != (n_channels,):
            raise ValueError(
                f"roles shape {roles.shape} != ({n_channels},){where} "
                "(one role per channel)"
            )
        top = 2 if allow_future_known else 1
        if roles.max(initial=0) > top:
            raise ValueError(
                f"roles entries must be in [0, {top}]{where}"
                + ("" if allow_future_known else
                   " (future-known covariates [2] are not admitted until the "
                   "eval pool's exogeneity curation rule is in force)")
            )
        if not np.any(roles == 0):
            raise ValueError(f"roles must include at least one target channel{where}")


class SeriesValidator:
    """The one per-series validation + canonicalisation pipeline, shared by
    every corpus producer — :func:`drain_generator` (cache_reuse), the
    in-process stream, and the sandbox stream child — so the carrier rules can
    never drift between feed modes.

    ``process(item, index)`` runs the full chain: record carrier
    (:func:`canonicalize_yield` with the accepted-field set) → values
    validation (:func:`check_series`) → extended-record canonicalisation +
    validation (:func:`check_record`) → the caller's ``extra_series_check``
    (e.g. the trainer's channel-correlation enforce gate) → the cumulative
    budgets, POINTS first (``max_total_points`` over values entries; 0
    disables — streaming modes stop at the token budget instead) then BYTES
    (``max_payload_bytes``: values at 8 B/point plus accepted extras at
    1 B/entry; 0 disables). Returns the canonical element — a contiguous
    ``(C, L)`` float64 array, or a canonical record dict when accepted extras
    ride along. Raises ``ValueError`` on any violation.
    """

    def __init__(
        self,
        *,
        min_length: int,
        max_length: int,
        max_channels: int = 1,
        max_abs: float | None = None,
        reject_constant: bool = False,
        accepted_fields: tuple[str, ...] | list[str] = (),
        max_missing_frac: float = 1.0,
        allow_future_known: bool = False,
        max_total_points: int = 0,
        max_payload_bytes: int = 0,
        extra_series_check=None,
    ) -> None:
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.max_channels = int(max_channels)
        self.max_abs = max_abs
        self.reject_constant = bool(reject_constant)
        self.accepted_fields = tuple(accepted_fields)
        self.max_missing_frac = float(max_missing_frac)
        self.allow_future_known = bool(allow_future_known)
        self.max_total_points = int(max_total_points)
        self.max_payload_bytes = int(max_payload_bytes)
        self.extra_series_check = extra_series_check
        self.total_points = 0
        self.payload_bytes = 0

    @classmethod
    def from_config(cls, cfg, extra_series_check=None) -> SeriesValidator:
        """Build from a ``GeneratorConfig`` — the production wiring every
        stream/sandbox path uses, so a new knob added here reaches all of them
        at once. ``max_abs_value = 0.0`` maps to the cast-safe ceiling exactly
        as the drain call sites always did."""
        return cls(
            min_length=cfg.min_length,
            max_length=cfg.max_length,
            max_channels=cfg.max_channels,
            max_abs=cfg.max_abs_value or CAST_SAFE_MAX_FLOAT32,
            reject_constant=cfg.reject_constant,
            accepted_fields=tuple(cfg.accepted_fields),
            max_missing_frac=cfg.max_missing_frac,
            allow_future_known=cfg.allow_future_known,
            # No point budget here: the streaming modes this constructor
            # serves stop at the TOKEN budget; max_total_points binds the
            # materialised drain, which builds its own validator.
            max_payload_bytes=cfg.max_payload_bytes,
            extra_series_check=extra_series_check,
        )

    def process(self, item: object, index: int) -> np.ndarray | dict[str, np.ndarray]:
        rec = canonicalize_yield(item, accepted=self.accepted_fields, index=index)
        arr = rec[VALUES_FIELD] if isinstance(rec, dict) else rec
        check_series(
            arr, min_length=self.min_length, max_length=self.max_length,
            max_channels=self.max_channels, max_abs=self.max_abs,
            reject_constant=self.reject_constant, index=index,
        )
        if isinstance(rec, dict):
            element: np.ndarray | dict[str, np.ndarray] = canonicalize_record(rec)
            check_record(
                element, max_missing_frac=self.max_missing_frac,
                allow_future_known=self.allow_future_known, index=index,
            )
            canon = element[VALUES_FIELD]
            elem_bytes = sum(int(a.nbytes) for a in element.values())
        else:
            canon = np.ascontiguousarray(
                np.atleast_2d(np.asarray(arr, dtype=np.float64))
            )
            element = canon
            elem_bytes = int(canon.nbytes)
        if self.extra_series_check is not None:
            self.extra_series_check(canon, index)
        self.total_points += int(canon.size)
        if self.max_total_points and self.total_points > self.max_total_points:
            raise ValueError(
                f"total emitted points {self.total_points} exceeds cap "
                f"{self.max_total_points}"
            )
        self.payload_bytes += elem_bytes
        if self.max_payload_bytes and self.payload_bytes > self.max_payload_bytes:
            raise ValueError(
                f"total payload bytes {self.payload_bytes} exceeds cap "
                f"{self.max_payload_bytes}"
            )
        return element


def element_dedup_key(element: np.ndarray | dict[str, np.ndarray]) -> bytes:
    """Dedup key for a canonical element from :meth:`SeriesValidator.process` —
    the legacy :func:`_series_key` for arrays, the 0xFF-framed
    :func:`_record_key` for extended records (never collide; see the framing
    note above)."""
    if isinstance(element, dict):
        return _record_key(element)
    return _series_key(element)


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
    max_payload_bytes: int = 0,
    accepted_fields: tuple[str, ...] | list[str] = (),
    max_missing_frac: float = 1.0,
    allow_future_known: bool = False,
    target_points: int = 0,
    extra_series_check=None,
) -> list[np.ndarray | dict[str, np.ndarray]]:
    """Pull ``n_series`` series from ``gen``, validating each one.

    Enforces the per-series length band, the channel cap, and a global cap on
    total emitted points (a memory / time guard against a generator that emits a
    few enormous series). Raises ``ValueError`` if the generator yields the wrong
    count, a malformed series, or blows the point budget.

    ``target_points`` (DEC-CA-0031) switches the drain to a POINTS
    denomination: when > 0, ``n_series`` is only the upper bound handed to
    ``generate`` (callers size it like the stream's prefix bound) and the
    drain stops at the first series that brings cumulative values-points to
    at least the target — the series count is free, so a generator may trade
    length against count under one corpus size. The consumed prefix is
    deterministic in (seed, ``n_series``), exactly like the streaming feed's
    early stop. A generator that exhausts its yields below the target is
    rejected. 0 keeps the legacy exact-count contract.

    ``max_abs`` and ``reject_constant`` are forwarded to :func:`check_series` as
    per-series data-quality gates (see there). ``max_dup_fraction`` is a
    corpus-level gate: reject if the fraction of series that are exact byte-copies
    of an earlier one exceeds it (defence against "emit one series N times"
    spam). It is accumulated *during* the drain — one digest per series, no second
    pass — and ``1.0`` disables it. Byte-exact matching is deliberately the
    zero-false-positive choice: honest seeded continuous-parameter generators
    never byte-collide, so a loose cap only trips lazy duplication (a determined
    adversary can still evade it with 1e-15 jitter — that's a v1 floor, not an
    anti-adversary defence). All gates default to no-op so existing callers
    are unchanged; the trainer sets them from ``chain.toml [generator]``.

    ``max_payload_bytes`` is the CARRIER's budget denomination (DEC-CA-0020 G3):
    a cap on the total canonical payload bytes of the drained corpus — values
    at 8 bytes/point plus any accepted extras (mask/roles at 1 byte/entry). At
    the shipped value (16e9 = ``max_total_points`` × 8) it is numerically
    identical to the point cap for a values-only corpus and can never trip
    first there; extras are priced automatically. ``0`` disables it.

    Record yields (DEC-CA-0020) are normalised through
    :func:`canonicalize_yield`: a values-only record corpus digests
    byte-identically to its bare-array twin, and ``accepted_fields`` extras
    (``mask``, ``roles`` — armed only via ``[training] accepted_fields``) are
    canonicalised, validated by :func:`check_record` (``max_missing_frac`` /
    ``allow_future_known``), and carried as ``{"values", …}`` dict elements
    in the returned list. The POINT budget counts values entries only — a
    mask marks points missing, it does not add points.

    Each series is canonicalised to a contiguous ``(C, L)`` float64 array (a 1-D
    series becomes ``(1, L)``), so the corpus the base trainer consumes always
    carries a channel axis. The point budget counts every emitted value
    (``C * L``). Returns the list in yield order; the trainer hashes it (see
    :func:`cascade.shared.manifest.corpus_digest`) so the corpus is
    reproducible and auditable.
    """
    if n_series <= 0:
        raise ValueError(f"n_series must be positive; got {n_series}")
    validator = SeriesValidator(
        min_length=min_length, max_length=max_length, max_channels=max_channels,
        max_abs=max_abs, reject_constant=reject_constant,
        accepted_fields=accepted_fields, max_missing_frac=max_missing_frac,
        allow_future_known=allow_future_known,
        max_total_points=max_total_points, max_payload_bytes=max_payload_bytes,
        extra_series_check=extra_series_check,
    )
    dedup = max_dup_fraction < 1.0
    seen: set[bytes] = set()
    dups = 0
    out: list[np.ndarray | dict[str, np.ndarray]] = []
    points = 0
    for i, item in enumerate(gen.generate(n_series)):
        if i >= n_series:
            raise ValueError(
                f"generate yielded more than n_series={n_series} series"
            )
        element = validator.process(item, i)
        if dedup:
            key = element_dedup_key(element)
            if key in seen:
                dups += 1
            else:
                seen.add(key)
        out.append(element)
        if target_points > 0:
            values = element["values"] if isinstance(element, dict) else element
            points += int(values.size)
            if points >= target_points:
                break
    if target_points > 0:
        if points < target_points:
            raise ValueError(
                f"generate exhausted after {len(out)} series / {points} points; "
                f"corpus_target_points={target_points} not reached"
            )
    elif len(out) != n_series:
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
