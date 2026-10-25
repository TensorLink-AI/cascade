"""Mode constants and fail-loud validators for individual chain.toml values."""

from __future__ import annotations

import re

SANDBOX_MODES = ("subprocess", "container")


def validate_sandbox_mode(mode: str) -> str:
    """Return ``mode`` if it is a known sandbox mode, else raise ValueError."""
    if mode not in SANDBOX_MODES:
        raise ValueError(f"sandbox_mode={mode!r} invalid; expected one of {SANDBOX_MODES}")
    return mode


# Record-carrier fields with a WIRED consumer (accept = consume, one release;
# DEC-CA-0020's refuse-unconsumed rule made mechanical). [training]
# accepted_fields may only name these; every other reserved name is still
# rejected at load so a typo or a premature arming fails the boot, not a round.
# Batch denominations at C > 1 (DEC-CA-0041) — see
# TrainingContractConfig.batch_denomination.
BATCH_DENOMINATIONS = ("series", "sequences")


def validate_batch_denomination(mode: str) -> str:
    """Fail loud: "sequences" silently degrading to "series" would cut a wide
    miner's step count 32× under an operator who believed it equalised."""
    if mode not in BATCH_DENOMINATIONS:
        raise ValueError(
            f"batch_denomination={mode!r} invalid; expected one of {BATCH_DENOMINATIONS}")
    return mode


# Token-budget denominations (DEC-CA-0042) — see
# TrainingContractConfig.budget_denomination.
BUDGET_DENOMINATIONS = ("points", "series_points")
# "points+mv<PCT>" (DEC-CA-0047): token billing with a width bonus — every
# values entry is a point, and a series with C > 1 is billed at
# 100 / (100 + PCT) of its points, so an all-multichannel corpus trains PCT %
# more tokens than a univariate one on ANY GPU (the budget, not the wall,
# stops every leg). PCT is an integer 1..100 so the rule is exact integer
# arithmetic the audit replays bit-for-bit.
_MV_BONUS_RE = __import__("re").compile(r"^points\+mv([1-9][0-9]?|100)$")


def budget_denomination_parts(mode: str) -> tuple[str, int]:
    """``(base, bonus_pct)`` of a budget denomination: ``("points", 0)``,
    ``("series_points", 0)`` or ``("points", PCT)`` for ``points+mvPCT``."""
    m = _MV_BONUS_RE.match(str(mode))
    if m:
        return "points", int(m.group(1))
    return str(mode), 0


def _validate_packed_sources_mode(mode: str) -> str:
    from ...interface.static_guard import validate_packed_sources_mode
    return validate_packed_sources_mode(mode)


def validate_budget_denomination(mode: str) -> str:
    """Fail loud: "series_points" silently degrading to "points" would bill a
    wide miner C× the budget its trainer, validator, and auditor agreed on."""
    if mode not in BUDGET_DENOMINATIONS and not _MV_BONUS_RE.match(str(mode)):
        raise ValueError(
            f"budget_denomination={mode!r} invalid; expected one of {BUDGET_DENOMINATIONS} "
            "or 'points+mv<1..100>'")
    return mode


CONSUMABLE_FIELDS = ("mask", "roles")


def validate_accepted_fields(value: object) -> tuple[str, ...]:
    """Normalise ``[training] accepted_fields`` (sorted, deduped) and reject
    names without a wired consumer."""
    fields = tuple(sorted({str(v) for v in (value or ())}))
    bad = [f for f in fields if f not in CONSUMABLE_FIELDS]
    if bad:
        raise ValueError(
            f"[training] accepted_fields={bad} have no wired consumer; "
            f"acceptable: {sorted(CONSUMABLE_FIELDS)} (a reserved field arms only "
            "in the release that consumes it — DEC-CA-0020)"
        )
    return fields


# Immutable-ref shape for [training] real_corpus_ref (DEC-CA-0028) — mirrors
# cascade.shared.hippius.DIGEST_RE without importing the storage stack at
# config-load time. Only content-pinned refs are expressible: a branch or tag
# would let the corpus bytes move under a fixed contract_digest.
_REAL_CORPUS_REF_RE = re.compile(r"^\S+@(sha256:[0-9a-f]{64}|hf:[0-9a-f]{40})$")


def validate_real_corpus_ref(value: object) -> str:
    """Normalise ``[training] real_corpus_ref``; "" (unarmed) or an immutable
    ``repo@digest`` ref. A mutable or malformed ref fails the boot, not a
    round."""
    ref = str(value or "")
    if ref and not _REAL_CORPUS_REF_RE.match(ref):
        raise ValueError(
            f"[training] real_corpus_ref={ref!r} is not an immutable ref; expected "
            "\"repo@sha256:<64 hex>\" (or \"repo@hf:<40 hex>\") — the shared real "
            "corpus must be content-pinned (DEC-CA-0028)"
        )
    return ref


DEDUP_MODES = ("off", "shadow", "enforce")


def validate_dedup_mode(mode: str, key: str = "dedup_mode") -> str:
    """Return ``mode`` if it is a known dedup mode, else raise ValueError.

    Typos used to fall through to "off": an anti-spam screen silently not
    running is exactly the failure an operator would not notice.
    """
    if mode not in DEDUP_MODES:
        raise ValueError(f"{key}={mode!r} invalid; expected one of {DEDUP_MODES}")
    return mode


# Miner-funded compute modes (DEC-CA-0036) — see RoundConfig.funded_mode.
FUNDED_MODES = ("off", "shadow", "required")


def validate_funded_mode(mode: str) -> str:
    """Return ``mode`` if it is a known funded mode, else raise ValueError.

    Same rationale as dedup: "required" silently degrading to "off" would put
    unfunded entries back on the operator's GPU bill without anyone noticing.
    """
    if mode not in FUNDED_MODES:
        raise ValueError(f"funded_mode={mode!r} invalid; expected one of {FUNDED_MODES}")
    return mode


FUNDED_POD_MODES = ("off", "rent")


def validate_funded_pods(mode: str) -> str:
    """Return ``mode`` if it is a known funded-pods mode, else raise ValueError.

    "rent" silently degrading to "off" would move every funded challenger leg
    back onto the operator's GPU bill — same fail-loud rationale as
    :func:`validate_funded_mode`.
    """
    if mode not in FUNDED_POD_MODES:
        raise ValueError(
            f"funded_pods={mode!r} invalid; expected one of {FUNDED_POD_MODES}")
    return mode


def validate_funded_field_cap(value: object) -> int:
    """A non-negative int, fail-loud: a negative cap is truthy, so it would
    silently seat NOBODY every round (the elastic-cap arithmetic reads any
    non-zero value as "use me") — review 2026-09-02."""
    cap = int(value)  # type: ignore[arg-type]
    if cap < 0:
        raise ValueError(f"funded_field_cap={cap} invalid; must be >= 0")
    return cap


def validate_funded_pod_skus(value: object) -> tuple[str, ...]:
    """A tuple of SKU strings, fail-loud on a bare string: ``"RTX4090"``
    would iterate as its CHARACTERS ('R', 'T', …) and every probe/rent would
    target one-letter SKUs — review 2026-09-02."""
    if isinstance(value, str):
        raise ValueError(
            f'funded_pod_skus={value!r} invalid; must be a list of SKU '
            f'strings, e.g. ["RTX4090", "A6000"]')
    return tuple(str(x) for x in value)  # type: ignore[union-attr]


def validate_funded_host_bench_floor(value: object) -> tuple[tuple[str, float], ...]:
    """``[round] funded_host_bench_floor``: a TOML table ``{SKU = tokens/s}``
    → sorted ``((sku, floor), …)``. Fail-loud on anything but a mapping of
    non-negative numbers: a list or a bare number would silently arm nothing
    (config knobs need loader parsing, 2026-09-08)."""
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(
            f"funded_host_bench_floor={value!r} invalid; must be a table of "
            f'SKU = tokens/s, e.g. {{ RTX4090 = 450e6 }}')
    out = []
    for sku, floor in value.items():
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or floor < 0:
            raise ValueError(
                f"funded_host_bench_floor[{sku!r}]={floor!r} invalid; must be a "
                f"non-negative number of tokens/s")
        if str(sku).strip():
            out.append((str(sku).strip(), float(floor)))
    return tuple(sorted(out))


def validate_funded_sku_wall_seconds(value: object) -> tuple[tuple[str, int], ...]:
    """``[round] funded_sku_wall_seconds``: a TOML table ``{SKU = seconds}`` —
    the MEASURED wall of one full-budget leg on that GPU type — → sorted
    ``((sku, seconds), …)``. Fail-loud on anything but a mapping of positive
    ints (config knobs need loader parsing, 2026-09-08)."""
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(
            f"funded_sku_wall_seconds={value!r} invalid; must be a table of "
            f"SKU = seconds, e.g. {{ RTX4090 = 12000, L40S = 7200 }}")
    out = []
    for sku, secs in value.items():
        if isinstance(secs, bool) or not isinstance(secs, (int, float)) or secs <= 0:
            raise ValueError(
                f"funded_sku_wall_seconds[{sku!r}]={secs!r} invalid; must be a "
                f"positive number of seconds")
        if str(sku).strip():
            out.append((str(sku).strip(), int(secs)))
    return tuple(sorted(out))


def funded_sku_wall_for(walls: tuple[tuple[str, int], ...], sku: str,
                        default: float) -> float:
    """The measured leg wall (seconds) for ``sku``, case-insensitive;
    ``default`` (the contract's max_train_seconds) when the SKU has none."""
    want = (sku or "").strip().lower()
    for name, secs in walls:
        if name.lower() == want:
            return float(secs)
    return float(default)


def validate_funded_price_cap(name: str, value: object) -> float:
    """A non-negative USD figure; 0 = off."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{name}={value!r} invalid; must be a non-negative number (0 = off)")
    return float(value)


def funded_host_bench_floor_for(floors: tuple[tuple[str, float], ...], sku: str) -> float:
    """The armed floor (tokens/s) for ``sku``, case-insensitive; ``0.0`` when
    the SKU has none (= the check is off for that round)."""
    want = (sku or "").strip().lower()
    for name, floor in floors:
        if name.lower() == want:
            return float(floor)
    return 0.0


# Champion publication policies for direct (vault) submissions (DEC-CA-0036):
# when a vault-ref king's code goes public. See cascade.funding.champion.
CHAMPION_PUBLISH_MODES = ("off", "crown", "delay", "dethrone")


def validate_champion_publish(mode: str) -> str:
    if mode not in CHAMPION_PUBLISH_MODES:
        raise ValueError(
            f"champion_publish={mode!r} invalid; expected one of {CHAMPION_PUBLISH_MODES}")
    return mode


def validate_corpus_target_points(target: object, max_total_points: int) -> int:
    """Normalise ``[generator] corpus_target_points`` (DEC-CA-0031).

    0 = off (the legacy exactly-``corpus_n_series`` drain). A positive target
    above ``max_total_points`` would make every materialised drain abort at
    the point cap before reaching its target — a config error, failed at boot
    rather than at the first round."""
    t = int(target or 0)
    if t < 0:
        raise ValueError(f"[generator] corpus_target_points={t} must be >= 0")
    if t > int(max_total_points):
        raise ValueError(
            f"[generator] corpus_target_points={t} exceeds max_total_points="
            f"{int(max_total_points)} — the drain would hit the point cap before "
            "its target on every run"
        )
    return t


# Corpus feed modes — how a generator's data reaches the trainer. Identical for
# king and challenger (folded into contract_digest via TrainingContractConfig).
#   stream_cpu  — live-stream fresh series from a CPU generator, no reuse.
#   stream_gpu  — live-stream from a GPU-resident generator (torch); high
#                 throughput, relaxed (tolerance/same-hardware) audit.
#   cache_reuse — draw a fixed corpus once, then multi-pass it under the budget.
#   shards      — read a corpus the CPU generation tier materialised into
#                 digest-pinned shards (DEC-CA-0053); the GPU pod runs no
#                 miner code.
CORPUS_MODES = ("stream_cpu", "stream_gpu", "cache_reuse", "shards")

# DEC-CA-0053: a series longer than the context — keep the last window, or
# cut seeded random windows from it.
LONG_SERIES_MODES = ("last", "crop")
# DEC-CA-0053: the longest series / context the contract may name.
MAX_SERIES_LENGTH = 32768


def validate_long_series(mode: str) -> str:
    """``[training] long_series`` (DEC-CA-0053)."""
    if mode not in LONG_SERIES_MODES:
        raise ValueError(f"[training] long_series must be one of {LONG_SERIES_MODES}; got {mode!r}")
    return mode


def validate_context_after(after: int, context_length: int, patch_size: int) -> int:
    """``[training] context_length_after`` (DEC-CA-0053): 0 = no schedule, else
    a larger context, a whole number of patches, at most MAX_SERIES_LENGTH."""
    if after == 0:
        return 0
    if after <= context_length:
        raise ValueError(f"[training] context_length_after={after} must exceed "
                         f"context_length={context_length}")
    if after > MAX_SERIES_LENGTH or (patch_size and after % patch_size):
        raise ValueError(f"[training] context_length_after={after} must be <= "
                         f"{MAX_SERIES_LENGTH} and a multiple of patch_size={patch_size}")
    return after


ANNEAL_SHAPES = ("cosine", "linear", "1-sqrt")


def validate_anneal_shape(shape: str) -> str:
    """``[training] anneal_shape`` (DEC-CA-0050): the fork-anneal decay curve."""
    if shape not in ANNEAL_SHAPES:
        raise ValueError(f"[training] anneal_shape must be one of {ANNEAL_SHAPES}; got {shape!r}")
    return shape


MAX_CORPUS_EPOCHS = 16   # beyond ~16 repeats extra passes are worth ~nothing


def validate_corpus_epochs(epochs: int, corpus_mode: str) -> int:
    """``[training] corpus_epochs`` (DEC-CA-0050 generation surface)."""
    if epochs < 0 or epochs > MAX_CORPUS_EPOCHS:
        raise ValueError(f"[training] corpus_epochs must be in [0, {MAX_CORPUS_EPOCHS}]; got {epochs}")
    if epochs and corpus_mode not in ("cache_reuse", "shards"):
        raise ValueError("[training] corpus_epochs requires corpus_mode = 'cache_reuse' "
                         "or 'shards'")
    return epochs


def validate_corpus_mode(mode: str) -> str:
    """Return ``mode`` if it is a known corpus feed mode, else raise ValueError."""
    if mode not in CORPUS_MODES:
        raise ValueError(f"corpus_mode={mode!r} invalid; expected one of {CORPUS_MODES}")
    return mode
