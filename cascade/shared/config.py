"""Config loader for chain.toml — single source of truth for subnet config.

Miners, the trainer, and validators all load from here. The schema is
versioned; a file newer than this code warns and proceeds (operator-controlled
file, deployed by hand alongside the binaries — the same policy horizon uses).
"""

from __future__ import annotations

import re
import sys
import tomllib  # py311+
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHAIN_TOML = REPO_ROOT / "chain.toml"


@dataclass(frozen=True)
class SubnetConfig:
    netuid: int
    name: str
    description: str


@dataclass(frozen=True)
class GeneratorConfig:
    """Bounds the trainer enforces on every generator's output.

    ``corpus_n_series`` series are drawn per training run; each must have
    length in ``[min_length, max_length]`` and the run aborts if total emitted
    points exceed ``max_total_points`` or generation runs past
    ``max_generate_seconds``.

    ``max_channels`` is the per-series variate cap. cascade trains a Toto2
    backbone from scratch — a multivariate architecture — so the corpus schema
    carries a channel axis. ``max_channels = 1`` keeps submissions univariate
    for now (a generator may still yield 1-D series, promoted to ``(1, L)``);
    raising it later turns on multivariate priors *without* a schema change.

    Sandbox selection (trainer-side; never part of ``contract_digest``):
    ``sandbox_mode = "subprocess"`` is the rlimited, netns-wrapped child
    (:mod:`cascade.trainer.sandbox`); ``"container"`` runs that same child
    inside a locked-down docker/podman container (``--network=none``,
    ``--cap-drop=ALL``, read-only rootfs — :mod:`cascade.trainer.
    sandbox_container`), using ``sandbox_image`` (digest-pin it in production)
    and ``sandbox_python`` (the interpreter inside the image). With
    ``sandbox_strict = true``, subprocess mode REFUSES to run when the host
    cannot provide a network namespace instead of silently downgrading to the
    Python-level socket guard — set it on any production trainer.
    """

    corpus_n_series: int
    min_length: int
    max_length: int
    max_total_points: int
    max_generate_seconds: int
    max_memory_mb: int
    max_repo_mb: int = 128  # cap on fetched submission bytes (code-only; no shipped weights)
    max_channels: int = 1
    # Cheap data-quality gates on generator output (see cascade.interface.generator).
    #   max_abs_value       — reject a series whose peak |value| exceeds this; 0.0
    #                         means "cast-safe default" (the float32 ceiling), which
    #                         is always applied since an over-max value is untrainable.
    #   reject_constant     — reject any flat (zero-range) series.
    #   max_dup_fraction    — cap the fraction of exact byte-duplicate series in a
    #                         materialised (cache_reuse) corpus; 1.0 disables it.
    max_abs_value: float = 0.0
    reject_constant: bool = False
    max_dup_fraction: float = 1.0
    # ── the record carrier (DEC-CA-0020; never part of contract_digest) ──────
    # ``interface_version`` names which yield payloads the trainer accepts: 1 =
    # record-or-array with ``values`` only (every reserved field hard-rejected).
    # A generator repo may declare its own ``interface_version`` in config.json;
    # one newer than this is rejected before the generator runs. Bumps ONLY
    # when a new payload field is accepted+consumed in one release — at that
    # point the accepted field set folds into [training] (one deliberate
    # contract_digest bump, DEC-CA-0020 layer 3).
    # ``max_payload_bytes`` is the carrier cap in canonical payload BYTES —
    # the budget denomination that stays correct when non-values fields arrive.
    # At 16e9 it equals max_total_points × 8 (float64), so it is numerically
    # inert today. 0 disables.
    interface_version: int = 1
    max_payload_bytes: int = 0
    # Runtime mirrors of the DIGEST-BOUND [training] acceptance keys (the
    # loader copies them in; there is no [generator] key for either — one
    # config source). They ride here because this dataclass is what reaches
    # the sandbox children and every drain/stream call site.
    #   accepted_fields    — record fields to accept AND consume ("mask",
    #                        "roles"); () = values-only (deployed default).
    #   allow_future_known — admit roles value 2 (future-known covariates).
    #   real_corpus_ref    — the owner-published shared real corpus pin
    #                        (DEC-CA-0028); "" = none (deployed default).
    # Note: json round-trips turn the tuple into a list; treat as a container.
    accepted_fields: tuple[str, ...] = ()
    allow_future_known: bool = False
    real_corpus_ref: str = ""
    # RUNTIME-resolved local path of the materialised shared real corpus —
    # never read from chain.toml (paths are machine-local). The PARENT side of
    # each sandbox resolves it (fetch + verify against real_corpus_ref, see
    # cascade.trainer.corpus.resolve_real_corpus) before spawning, and it rides
    # into the child via this dataclass's JSON. Empty while real_corpus_ref is
    # armed is a loud error at generator load, never a silent no-corpus run.
    real_corpus_dir: str = ""
    # Per-series cap on the masked fraction when "mask" is accepted
    # (DEC-CA-0023's gate): a mostly-missing series is mostly filler bytes
    # bought at full freight. Meaningless while accepted_fields is empty.
    max_missing_frac: float = 0.5
    # Channel-redundancy gate (DEC-CA-0026; data-quality, not digest-bound).
    #   channel_corr_mode — "off" (default) | "shadow" (telemetry only; the
    #                       trainer's shadow accumulator already logs it) |
    #                       "enforce" (reject a series whose max off-diagonal
    #                       |Pearson| exceeds max_channel_corr).
    #   max_channel_corr  — the near-identity bar; 0.999 targets the
    #                       jitter-duplicate exploit only. Enforce ONLY after
    #                       the shadow logs clear honest generators.
    channel_corr_mode: str = "off"
    max_channel_corr: float = 0.999
    sandbox_mode: str = "subprocess"   # "subprocess" | "container"
    sandbox_image: str = ""            # container image for sandbox_mode="container"
    sandbox_python: str = "python3"    # python inside that image (worker: /root/cascade/.venv/bin/python)
    sandbox_strict: bool = False       # refuse to run without hard network isolation


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


# Corpus feed modes — how a generator's data reaches the trainer. Identical for
# king and challenger (folded into contract_digest via TrainingContractConfig).
#   stream_cpu  — live-stream fresh series from a CPU generator, no reuse.
#   stream_gpu  — live-stream from a GPU-resident generator (torch); high
#                 throughput, relaxed (tolerance/same-hardware) audit.
#   cache_reuse — draw a fixed corpus once, then multi-pass it under the budget.
CORPUS_MODES = ("stream_cpu", "stream_gpu", "cache_reuse")


def validate_corpus_mode(mode: str) -> str:
    """Return ``mode`` if it is a known corpus feed mode, else raise ValueError."""
    if mode not in CORPUS_MODES:
        raise ValueError(f"corpus_mode={mode!r} invalid; expected one of {CORPUS_MODES}")
    return mode


@dataclass(frozen=True)
class SizeSpec:
    """One additional model size trained in the round's final stage.

    cascade's final stage trains the king and the surviving challenger at more
    than one Toto2 size so the throne is decided on a *combined* score across
    sizes (a scaling check, not just the cheapest rung). Each spec overrides only
    the width/depth fields that change with size, plus the per-size frozen-arch
    digest and reference throughput; the fields fixed across the Toto2 family
    (``head_dim``, ``patch_size``, the objective, the optimiser recipe) are
    inherited from the base :class:`TrainingContractConfig` via
    :meth:`TrainingContractConfig.for_size`, so a size cannot silently diverge on
    anything but its shape.
    """

    arch_preset: str
    base_arch_digest: str          # sha256 of THIS size's frozen arch+init
    d_model: int
    num_layers: int
    num_heads: int
    mlp_expansion: int
    ref_throughput_tokens_per_s: int   # measured on the reference GPU for this size
    # Exact FFN hidden width from the released config.json (0 ⇒ derive as
    # d_model × mlp_expansion). Toto-2.0-4m ships d_ff = 688, not 2×256.
    d_ff: int = 0
    # ── size-conditional silicon + budget (DEC-CA-0027) ──────────────────────
    # ``expected_gpu``: this size's pinned duel GPU when it differs from the
    # base [training] expected_gpu ("" = inherit). From-scratch economics die
    # between 22M and ~100M params, and 300M+ trains on H100-class silicon by
    # owner direction — so the SKU pin is a per-size contract term, checked by
    # the validator's gpu_name gate through the per-size contract exactly like
    # the base pin. ``ref_throughput_tokens_per_s`` above MUST be measured on
    # THIS GPU (the token budget derives from it).
    # ``target_train_hours``: this size's budget hours when they differ from
    # the base (0.0 = inherit) — a 313M warm-start increment is ~6h, not the
    # 4M's 3h; the value is chosen with the margin re-unitisation, from the
    # null-LCB noise-floor measurement.
    # Digest note: SizeSpec fields fold into contract_digest via extra_sizes,
    # so ADDING these fields moves the digest only for configs that already
    # carry a [[training.sizes]] block (none deployed — every shipped config
    # has them commented out). For any future size block they are part of that
    # size's identity, which is the point: silicon and budget are terms of the
    # controlled experiment.
    expected_gpu: str = ""
    target_train_hours: float = 0.0


# Screen-stage wall-clock guard (see TrainingContractConfig.for_hours): a heat
# run's hard deadline derives from its own cheap hours budget instead of
# inheriting the final's ``max_train_seconds``. Defaults implement the owner
# policy "the budget hours ARE the wall-clock cap" (factor 1.0): a run stops at
# the token budget or the nominal time, whichever comes first, and a time stop
# is flagged ``deadline_hit`` rather than run long. Raise the factor (via
# ``[round] heat_guard_factor``) when heat pods are a slower SKU than the
# reference-throughput box — at 1.0 a slower SKU turns every heat into a
# time-truncated run, which makes the screen partially reward fast-to-generate
# data over good data. The floor absorbs fixed overheads (sandbox boot, first
# batch) that don't shrink with tiny budgets. Overridable per deployment from
# ``[round]``; the derived heat contract stays trainer-internal (screened,
# discarded, never digest-gated).
HEAT_GUARD_FACTOR = 1.0
HEAT_GUARD_FLOOR_SECONDS = 900


@dataclass(frozen=True)
class TrainingContractConfig:
    """The fixed training contract — identical for king and challenger.

    The central invariant of cascade: the only thing that varies between the
    two trained models is the generator's data. Every field here is held
    constant across the pair (and folded into ``contract_digest``) so the eval
    is a controlled measurement of data quality.

    cascade trains a **Toto2 backbone from random initialisation** on each
    generator's corpus — it does *not* fine-tune a released checkpoint. Training
    from scratch is what makes the corpus the only source of learned signal: a
    fine-tune would confound "good data" with "what the pretrained weights
    already knew". Because the run starts from noise, the contract has to pin
    the *entire* recipe — architecture, objective, masking, optimiser, and the
    compute budget — not just three hyperparameters. ``base_arch_digest`` is the
    sha256 of the frozen architecture + initialisation code; set it before
    launch so a trainer can't silently swap models between rounds.

    The architecture defaults below track Datadog's released ``Toto-2.0-4m``
    (arch family fixes ``head_dim = 64``; Toto 2.0 uses ``patch_size = 32`` and
    a 9-quantile pinball head). Pin the exact integers to that checkpoint's
    ``config.json`` before launch, same convention as ``base_arch_digest``.

    Budget is expressed as **wall-clock hours on the owner's reference GPU**
    (``target_train_hours``, e.g. 3) but *enforced* as a fixed token count —
    ``target_train_hours × 3600 × ref_throughput_tokens_per_s`` (the
    ``train_tokens`` property). Going through a pinned token count rather than a
    raw timer matters: both king and challenger then see **identical compute**
    regardless of data-dependent throughput (a pure timer would let a generator
    win by emitting cheap-to-step data rather than better data), and a re-derived
    audit run reproduces the same step count. ``max_train_seconds`` is the hard
    wall-clock guard. Budgeting by compute (not epochs) also keeps a tiny corpus
    from winning by being trivially memorised in a few passes.
    """

    # identity / architecture (pin to Datadog/Toto-2.0-4m config.json)
    base_arch: str
    arch_preset: str
    base_arch_digest: str
    d_model: int
    num_layers: int
    num_heads: int
    head_dim: int
    patch_size: int
    mlp_expansion: int
    num_quantiles: int
    # objective / masking (Toto 2.0 Contiguous Patch Masking + quantile head)
    masking: str
    cpm_c_max: int
    cpm_p_max: float
    input_transform: str
    # I/O lengths (must match [eval] so the trained model fits the eval windows)
    context_length: int
    horizon: int
    # from-scratch budget — wall-clock hours on the reference GPU, enforced as a
    # fixed (fair, reproducible) token count = hours × reference throughput
    target_train_hours: float
    ref_throughput_tokens_per_s: int
    warmup_fraction: float
    batch_size: int
    # optimiser (u-muP transfer: tune on a small proxy, pin the result here)
    optimizer: str
    base_lr: float
    weight_decay: float
    lr_schedule: str
    umup_base_d_model: int
    train_seed_salt: int
    max_train_seconds: int
    # corpus feed mode (one of CORPUS_MODES); identical for king & challenger
    corpus_mode: str = "stream_cpu"
    # Exact FFN hidden width from the released config.json (0 ⇒ derive as
    # d_model × mlp_expansion). Toto-2.0-4m ships d_ff = 688, not 2×256.
    d_ff: int = 0
    # Pinned GPU model for byte-exact re-derivation. When non-empty, the validator
    # asserts every trained entry's recorded gpu_name == this (king and challenger
    # ran the same SKU); empty ⇒ require only that king and challenger match each
    # other when both report a gpu_name. Folded into contract_digest.
    expected_gpu: str = ""
    # Pinned training-runtime image for byte-exact re-derivation: the digest of
    # the container image (torch/CUDA/cuDNN stack) every FINAL run must execute
    # in. Accepts a full digest-pinned ref (``…@sha256:<64hex>``) or a bare
    # ``sha256:<64hex>``. When non-empty, the trainer/worker refuses a final run
    # unless its runtime reports the same digest (CASCADE_TRAIN_IMAGE_DIGEST,
    # injected at pod launch). Folded into contract_digest, so re-pinning the
    # image is a new contract. Empty ⇒ unpinned (no check).
    train_image_digest: str = ""
    # Extra model sizes trained alongside the base (primary) size in the final
    # stage. Empty ⇒ single-size rounds (the legacy behaviour). Folded into
    # contract_digest, so a validator's contract gate covers every size at once.
    extra_sizes: tuple[SizeSpec, ...] = ()
    # ── accepted record-field set (DEC-CA-0020 layer 3; digest-bound) ────────
    # The payload fields the carrier ACCEPTS and the trainer CONSUMES — one
    # release ships both, and arming is this one [training] key (a deliberate
    # contract_digest bump via the drop-when-default convention in
    # cascade.shared.manifest: empty = omitted from the hash, so deployed
    # digests are untouched until an operator sets it). Valid entries are
    # reserved names with a wired consumer: "mask" (DEC-CA-0023), "roles"
    # (DEC-CA-0026). Order-insensitive (normalised sorted at load).
    accepted_fields: tuple[str, ...] = ()
    # roles value 2 (future-known covariates) admission. Digest-bound and OFF
    # until docs/EVAL_POOL.md carries the covariate exogeneity curation rule —
    # arming it before that rule exists is forbidden (DEC-CA-0026).
    allow_future_known: bool = False
    # ── owner-published shared real corpus (DEC-CA-0028; digest-bound) ───────
    # Immutable ref ("repo@sha256:<64hex>" or "repo@hf:<40hex>") of the frozen,
    # licensed real-data corpus every generator may READ (mounted/passed
    # read-only into the sandbox as the optional ``real_corpus_dir``
    # constructor kwarg). "" = no shared corpus — the deployed default, and via
    # the drop-when-default convention the field is absent from contract_digest
    # until set, so arming is the deliberate digest bump. Arming is FORBIDDEN
    # until docs/EVAL_POOL.md carries the corpus/eval-pool disjointness rule
    # (provenance + time wall) and the DEC-CA-0028 pricing experiment has run.
    real_corpus_ref: str = ""

    def tokens_for_hours(self, hours: float) -> int:
        """Point-pass budget for ``hours`` on the reference GPU at this size's
        throughput. Used for both the full final budget and the cheaper heat
        budget; going through a token count (not a raw timer) keeps king and
        challenger on identical compute and keeps a re-derived run reproducible."""
        return int(round(hours * 3600.0 * self.ref_throughput_tokens_per_s))

    def for_hours(
        self,
        hours: float,
        *,
        guard_factor: float = HEAT_GUARD_FACTOR,
        guard_floor_seconds: int = HEAT_GUARD_FLOOR_SECONDS,
    ) -> TrainingContractConfig:
        """This size's contract at a reduced ``hours`` budget — a heat/screen run.

        Scales BOTH knobs together: ``train_tokens`` (via ``target_train_hours``)
        and the hard wall-clock guard, ``max(guard_factor × hours,
        guard_floor_seconds)`` capped at the pinned ``max_train_seconds``. The
        run stops at the token budget or that deadline, whichever comes first —
        without this a screen run inherits the final's guard, and one
        pathologically slow (or adversarially trickling) corpus can hold a heat
        slot for the final-scale hours at a ~30-min budget. At the default
        ``guard_factor = 1.0`` the budget hours ARE the cap (owner policy): a
        run that can't sustain reference throughput truncates and is flagged
        ``deadline_hit``. Callers wire the knobs from ``[round]``
        (``heat_guard_factor`` / ``heat_guard_floor_seconds``); raise the factor
        when heat pods are a slower SKU than the reference box. Final runs never
        come through here; their guard stays the contract value."""
        if hours <= 0:
            raise ValueError(f"hours must be positive; got {hours}")
        if guard_factor <= 0:
            raise ValueError(f"guard_factor must be positive; got {guard_factor}")
        guard = max(int(round(guard_factor * hours * 3600.0)), int(guard_floor_seconds))
        return replace(
            self,
            target_train_hours=float(hours),
            max_train_seconds=min(guard, self.max_train_seconds),
            extra_sizes=(),
        )

    @property
    def train_tokens(self) -> int:
        """Enforced final-stage budget in point-passes: ``target_train_hours`` of
        the reference GPU at ``ref_throughput_tokens_per_s``. King and challenger
        both train to this exact count — fair (equal compute, not equal
        wall-clock, which data-dependent throughput could skew) and reproducible
        (a re-derived run matches)."""
        return self.tokens_for_hours(self.target_train_hours)

    @property
    def warmup_tokens(self) -> int:
        return int(round(self.train_tokens * self.warmup_fraction))

    def for_size(self, spec: SizeSpec) -> TrainingContractConfig:
        """A per-size training contract: this base recipe with the width/depth,
        frozen-arch digest, and throughput swapped for ``spec``. The result has
        no nested ``extra_sizes`` (it IS a single concrete size), so its
        ``contract_digest`` is the stable identity of that one size.

        Size-conditional overrides (DEC-CA-0027): a spec with a non-empty
        ``expected_gpu`` swaps the GPU pin (the validator's gpu_name gate then
        asserts THIS size's silicon), and a positive ``target_train_hours``
        swaps the budget hours (the token budget follows via ``train_tokens``,
        at this size's measured throughput). Unset ⇒ inherit the base — every
        existing config resolves exactly as before."""
        return replace(
            self,
            arch_preset=spec.arch_preset,
            base_arch_digest=spec.base_arch_digest,
            d_model=spec.d_model,
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            mlp_expansion=spec.mlp_expansion,
            d_ff=spec.d_ff,
            ref_throughput_tokens_per_s=spec.ref_throughput_tokens_per_s,
            expected_gpu=spec.expected_gpu or self.expected_gpu,
            target_train_hours=(
                spec.target_train_hours if spec.target_train_hours > 0
                else self.target_train_hours
            ),
            extra_sizes=(),
        )

    @property
    def primary_size(self) -> TrainingContractConfig:
        """The base (primary) size as a standalone single-size contract."""
        return replace(self, extra_sizes=())

    def all_sizes(self) -> list[TrainingContractConfig]:
        """Every configured size as a standalone single-size contract: the primary
        first, then each :class:`SizeSpec` in ``extra_sizes``. This is the size
        *registry* the round's screen/throne pointers select from — not (in
        general) what a round trains; see :meth:`ChainConfig.throne_contracts`."""
        return [self.primary_size, *(self.for_size(s) for s in self.extra_sizes)]

    @property
    def size_registry(self) -> dict[str, TrainingContractConfig]:
        """``arch_preset`` → single-size contract, over the primary + extra sizes."""
        return {c.arch_preset: c for c in self.all_sizes()}

    def contract_for(self, preset: str) -> TrainingContractConfig:
        """Resolve a size by ``arch_preset`` to its single-size contract.

        Raises ``ValueError`` listing the available presets if ``preset`` is not
        configured (typo, or a `[[training.sizes]]` block still commented out)."""
        registry = self.size_registry
        if preset not in registry:
            raise ValueError(
                f"unknown size {preset!r}; configured sizes are {sorted(registry)} "
                "(add a [[training.sizes]] block or fix the [round] screen/throne size)"
            )
        return registry[preset]


@dataclass(frozen=True)
class RoundConfig:
    """Round cadence and the two-stage (heat → final) selection.

    A round spans ``epoch_blocks`` (≈24h at a 12s block time): the trainer runs
    exactly one round per epoch, so the king is trained once per day. The round's
    base seed is the chain block hash at the *epoch boundary*, so the whole day
    shares one :class:`~cascade.trainer.contract.RoundSeeds` — every heat and
    final training in the round uses identical random init and data-order RNG.

    Only commitments revealed STRICTLY BEFORE the epoch boundary are eligible:
    commit late and you compete in the next round, not this one. That boundary is
    the submission deadline, and it is deterministic so every honest party
    re-derives the same field.

    The field is first screened cheaply — every eligible challenger is trained for
    ``heat_train_hours`` on the ``screen_size`` and scored internally by the
    trainer; the top ``finalists`` then advance to the full
    ``[training] target_train_hours`` final against the king, trained at each of
    ``throne_sizes``.

    ``screen_size`` and ``throne_sizes`` are ``arch_preset`` names selecting from
    the size registry (``[training]`` primary + each ``[[training.sizes]]``).
    Empty ⇒ both default to the primary size (single-size rounds, today's
    behaviour). This is the scaling seam: promoting a rung is just pointing these
    at a bigger size (e.g. ``screen_size = "toto2-4m"``, ``throne_sizes =
    ["toto2-22m"]``) — no code change. The screen size is independent of the
    throne size, so a cheap small screen can feed a larger throne.
    """

    epoch_blocks: int = 7200          # ≈24h at 12s blocks; one round per epoch
    round_hours: float = 24.0         # informational: wall-clock span of an epoch
    # ── Scheduled cadence change (see :func:`effective_epoch_blocks`) ────────
    # A round-length change is CONSENSUS-RELEVANT: every validator derives the
    # epoch from the block grid, so a fleet that switches at different moments
    # selects different eval-pool snapshots for the same round. Restart timing
    # cannot coordinate that across independent operators — so the switch is
    # pinned to a BLOCK instead. Operators pull and restart whenever they like;
    # every node flips at the same block on its own.
    #
    # ``epoch_activation_block`` is the first block on which ``epoch_blocks``
    # applies; before it, ``epoch_blocks_prev`` does. Both 0 ⇒ no scheduled
    # change, ``epoch_blocks`` always applies (the steady state — clear these
    # two once the activation block is safely in the past).
    #
    # The activation block MUST be a boundary of BOTH grids, or a round is cut
    # short or double-counted at the seam; ``load_chain_config`` enforces it.
    epoch_blocks_prev: int = 0
    epoch_activation_block: int = 0
    heat_train_hours: float = 0.5     # cheap screening budget per competitor
    heat_n_windows: int = 256         # eval windows the heat screens on (≤ [eval] n_windows)
    # Sample forecasts per window in the heat screen. The heat only RANKS the
    # field, and CRPS rankings are stable at far fewer samples than the final
    # verdict needs — the screen eval runs sequentially on the orchestrator's
    # CPU, so this is the knob that keeps a large field's screening from
    # rivalling its training time. 0 ⇒ reuse [eval] num_samples.
    heat_num_samples: int = 0
    # Heat wall-clock cap = max(heat_guard_factor × heat_train_hours,
    # heat_guard_floor_seconds), never above [training] max_train_seconds. The
    # run stops at the token budget or this deadline, whichever first; a time
    # stop is flagged deadline_hit. 1.0 = the budget hours ARE the cap; raise it
    # when heat pods are a slower SKU than the reference-throughput box, or the
    # screen starts rewarding fast-to-generate data over good data.
    heat_guard_factor: float = 1.0
    heat_guard_floor_seconds: int = 900
    # In-round re-queues for a heat challenger whose dispatch (and its
    # in-dispatch retry) died on an INFRASTRUCTURE failure — storage layer or
    # SSH transport (rc=255). Challenger-fault failures never re-queue. Each
    # re-queue is one more full heat run in the worst case, so the added
    # wall-clock is bounded by this × the per-dispatch timeout. 0 ⇒ off
    # (the pre-2026-08-19 behaviour: infra casualties drop terminally).
    heat_infra_requeues: int = 2
    finalists: int = 1                # challengers promoted from the heat to the final
    # ── Tie-aware finalists (DEC-CA-0012, trainer half) — inert at defaults ──
    # ``max_finalists > 1`` arms the tie-aware advance rule: a leader the
    # screen's own statistic separated from the runner-up (paired LCB > 0, the
    # ``lcb_vs`` diagnostic in cascade.eval.heat) advances ALONE whatever the
    # cap; a statistically tied top is re-scored on a larger eval slice (the
    # run-off below) and whoever still cannot be separated advances too,
    # capped here. The validator then duels the WHOLE cohort under a
    # family-wise alpha/k. At the default 1 the tie logic never runs — exactly
    # one finalist advances as before and every manifest hashes identically.
    max_finalists: int = 1
    # Total eval windows the tie run-off re-scores the tied set on (CPU-only,
    # on the orchestrator, against heat checkpoints still on local disk).
    # Only the INCREMENTAL windows beyond the heat's slice are scored — the
    # round's window selection is a seeded permutation prefix, so the heat's
    # slice is a strict prefix of this one and pairing holds by construction.
    # Clamped to [eval] n_windows; must exceed the heat's actual window count
    # to add evidence. 0 = no run-off: a tied top advances by the heat
    # ranking, capped at ``max_finalists``.
    tie_runoff_windows: int = 0
    # Wall clock for the WHOLE run-off, mirroring ``dedup_phase_seconds``: on
    # expiry the pre-run-off tied set advances (capped) — a screen that cannot
    # finish must not sink the round it protects. Inert while max_finalists=1.
    tie_runoff_phase_seconds: int = 900
    screen_size: str = ""             # arch_preset the heat screens at ("" ⇒ primary)
    throne_sizes: tuple[str, ...] = ()  # arch_presets the final trains/judges at (() ⇒ [primary])
    # Anti-spam: 1 hotkey = 1 submission (lifetime). When True, a hotkey that has
    # already entered a round's heat is never screened again — it must re-register
    # (a new UID, paying the registration cost) to resubmit, so a miner cannot
    # cheaply re-roll a generator against the throne. The trainer enforces it and
    # persists the burn set at ``submissions_db_path`` (resolved under the
    # trainer's ``work_root`` when relative). Off ⇒ a hotkey re-competes every
    # epoch — handy for testnet iteration; keep ON for mainnet.
    one_submission_per_hotkey: bool = True
    # Only commitments revealed AT/AFTER this block ever enter a round — the
    # official go-live gate. Anything committed to the netuid before launch
    # (squatters, rehearsal commits, migrated-from-testnet leftovers) never
    # competes and never burns its one submission. 0 = no floor (testnet).
    # Set to the announced launch block in mainnet chain.toml AT LAUNCH.
    commit_floor_block: int = 0
    # Genesis baseline king (burn-until-dethroned). A Hippius generator ref
    # (repo@digest): whenever no on-chain champion has a resolvable commitment,
    # the trainer trains THIS generator as the king — a fixed, un-earnable floor
    # — instead of auto-promoting the lowest-UID miner. The king entry carries a
    # sentinel hotkey and uid = -1, which is out-of-range for every metagraph, so
    # the validator's weight routing drops it and BURNS to burn_uid until a real
    # miner dethrones the baseline. Empty ("") = off: legacy behaviour (promote
    # the lowest-UID challenger to interim king). The ref MUST be a PUBLIC Hub
    # repo the trainer can fetch anonymously (same contract as a miner submission).
    genesis_generator_ref: str = ""
    submissions_db_path: str = "trainer_submissions.json"
    # Commit-order witness: {hotkey: {pending, committed}} block numbers, written
    # every poll tick. The chain deletes a commit's block when drand reveals it,
    # so the evidence of who submitted a generator FIRST exists only for whoever
    # watched while it was sealed — this file IS that record, and the duplicate
    # screen's tie-break reads it (falling back to reveal order per hotkey when
    # it has nothing). Relative paths resolve under work_root.
    commit_witness_path: str = "trainer_commit_witness.json"
    # Content-level duplicate screen (cascade.interface.dedup): before the heat,
    # each challenger repo is fingerprinted and compared PAIRWISE against the
    # king and every kept earlier-committed challenger; an EXACTLY identical
    # repo — same tree, same normalized tokens, or same tokens modulo renames
    # — loses its heat slot (the earlier submission keeps it, so copying can
    # never displace the original). Dropped entrants still burn their one
    # lifetime submission — they entered the round; refunding would give free
    # re-rolls against the screen.
    # There is deliberately NO similarity threshold: a ratio bar (originally
    # drop at sim ≥ 0.99, with a sketch tier for oversize streams) is gameable
    # by spacing edits just under it — operators were already laddering at
    # 0.90–0.986 — and it produced the one confirmed false positive (a round's
    # eventual finalist). Enforcement rests on EXACT identity of code (the
    # digest tiers) or of output (the behavioral probe).
    # "off" = no screen; "shadow" = fingerprint + log verdicts, drop nothing;
    # "enforce" = drop. The dataclass default is "off" (behavior-preserving
    # for configs without the key); the shipped mainnet chain.toml enforces.
    dedup_mode: str = "off"
    # config_only tier: identical normalized .py streams, differing functional
    # config/data files. This is BOTH the observed same-round ticket-spam
    # pattern (self-declared A/B/C config sweeps) and the legitimate way to
    # compete by forking a public generator — config IS the product for a data
    # generator. False (default): shadow-log the verdict, never drop, whatever
    # dedup_mode says. Flip only after the shadow log shows the split.
    dedup_config_only_enforce: bool = False
    # Cost caps on the screen itself. Tokenizing is linear but not cheap
    # (~2s/MB) and input size is attacker-chosen up to [generator]
    # max_repo_mb. dedup_max_text_mb bounds the tokenizer's input per repo;
    # dedup_max_tokens bounds the retained token prefix, whose only remaining
    # quadratic consumer is the absolute-delta measurement on config_only
    # labels (0 = uncapped, do not use in production). Field repos run 7–11k
    # tokens / ~100KB, so both defaults are orders of magnitude above honest.
    dedup_max_tokens: int = 50_000
    dedup_max_text_mb: int = 4
    # Wall-clock budget for the WHOLE screen (fetch + fingerprint + compare +
    # probe). On expiry the round proceeds unscreened: a screen that cannot
    # finish must not be able to sink the round it protects.
    dedup_phase_seconds: int = 900
    # The same screen runs inside `cascade-trainer --plan-only`, which the
    # provisioner invokes as a subprocess under a HARD 600s timeout — and a
    # timed-out plan rents no fleet at all, i.e. costs the round its GPU. So
    # the sizing path gets its own, much smaller budget: over-rent (the screen
    # gave up, size off the raw field) is a cost; a lost rental window is a
    # round. Keep this well under the caller's timeout.
    dedup_plan_seconds: int = 120
    # Behavioral probe: sandbox-draw dedup_probe_series series per surviving
    # entrant under the shared round seed, TWICE. Two draws that differ ⇒ the
    # generator violates the determinism contract (the entropy re-roll that
    # lets identical code mint distinct corpora). Identical probe output
    # across two entrants (or vs the king) ⇒ same generative process
    # regardless of the code — behavior_identical. dedup_probe_mode gates
    # probe-derived drops (nondeterministic / probe_failed /
    # behavior_identical) INDEPENDENTLY of dedup_mode, so the static tiers
    # can enforce while the probe observes: within-family heat spreads in the
    # live field suggest nondeterminism is widespread, and probe drops burn
    # hotkeys — ship shadow, enforce only after the log says it's safe.
    # 0 series disables the probe regardless of mode.
    dedup_probe_mode: str = "shadow"   # off | shadow | enforce
    dedup_probe_series: int = 8
    # The probe's OWN wall clock, overriding [generator] max_generate_seconds
    # for probe draws only. The full-corpus budget (1800s for 16384 series)
    # would let one hostile submission stall the orchestrator ~an hour per
    # draw — and probe EXECUTION is paid even in shadow mode. A generator
    # that cannot emit dedup_probe_series series in this window while owing
    # thousands in the full budget is pathological or hostile.
    dedup_probe_generate_seconds: int = 120
    # Total wall clock for the probe stage. The per-draw budget above is
    # derived from it (budget ÷ waves ÷ 2 draws, clamped), so N hostile
    # entrants each burning their full clock cannot extend the stage without
    # bound — worst case is this number, not N × the per-draw budget.
    dedup_probe_budget_seconds: int = 600
    # The probe EXECUTES untrusted generator code on the orchestrator — the box
    # holding the private eval pool and the trainer's wallet — so it demands a
    # kernel-enforced sandbox: [generator] sandbox_mode = "container", or
    # subprocess mode with sandbox_strict = true (netns required, no silent
    # degradation to the in-process socket guard). Without one of those the
    # probe DISABLES itself and logs an error. Set this true to run it anyway
    # on a box you are willing to treat as compromised (testnet).
    dedup_probe_allow_weak_sandbox: bool = False
    # Timed-reveal safety margin: `cascade deploy` targets its timelock reveal at
    # `epoch boundary − reveal_margin_blocks`, so a submission stays hidden for
    # its whole window and is public only for the last few minutes before the
    # field locks. The margin must absorb commit-inclusion + drand reveal jitter
    # (a reveal landing AT/after the boundary misses the round) while staying
    # too short for a copier to fetch + re-commit + land their own reveal.
    # ~25 blocks ≈ 5 min at 12s blocks; tighten after measuring live jitter.
    reveal_margin_blocks: int = 25

    @property
    def finalist_cap(self) -> int:
        """Upper bound on challengers the heat may advance (DEC-CA-0012).

        The legacy constant while the tie logic is off (``max_finalists <= 1``),
        else ``max(finalists, max_finalists)``. The heat's fast paths and the
        provisioner's final-fleet sizing must share this bound so the
        pre-phased fleet and the ``within_budget`` breaker always cover the
        worst case the advance rule can produce.
        """
        if self.max_finalists > 1:
            return max(self.finalists, self.max_finalists)
        return self.finalists


@dataclass(frozen=True)
class EvalConfig:
    """Held-out eval windows scored each round (same set for king and challenger).

    ``eval_dataset`` is the identifier the manifest carries and the validator
    matches on. ``eval_source = "private-rotating"`` means the windows are drawn
    from an owner-controlled private pool and the *slice rotates per round*
    (seeded by the round's block hash) — TIME-style contamination resistance, so
    a generator cannot distribution-match a fixed public benchmark. The concrete
    pool loader (``window_pool``) is a boundary; the seeded rotation/selection
    lives in ``cascade.validator.windows``.
    """

    eval_dataset: str
    eval_source: str
    window_pool: str
    num_samples: int
    n_windows: int
    context_length: int
    horizon: int
    # ── Public-benchmark logging (log-only; never feeds scoring/weights) ──────
    # Off by default. When on, the validator runs the out-of-process sidecar
    # (``benchmarks/``) on a dethroned challenger and logs GIFT-Eval/BOOM/TIME
    # numbers. ``benchmark_suites = ()`` runs all three; ``benchmark_num_samples
    # = 0`` reuses ``num_samples``; ``benchmark_max_series = 0`` runs the full
    # benchmark (use a small cap for a smoke run). Defaults keep old toml loading.
    run_benchmarks: bool = False
    benchmark_project_dir: str = "benchmarks"
    benchmark_suites: tuple[str, ...] = ()
    benchmark_num_samples: int = 0
    benchmark_max_series: int = 0
    # ── Public-benchmark no-regression gate (CONSENSUS; see gift_gate) ────────
    # Consumed only when ``[scoring] gift_gate_mode`` != "off". Runs the
    # gift-eval sidecar on the primary king/challenger checkpoints on a
    # private-pool win. ``gift_gate_datasets`` pins the config subset
    # ("" = full 97 configs); ``gift_gate_num_samples = 0`` reuses
    # ``num_samples``; ``gift_gate_data_dir`` points at pinned benchmark data on
    # the validator ("" = the suite's own env vars). Defaults keep old toml.
    gift_gate_datasets: str = ""
    gift_gate_num_samples: int = 0
    gift_gate_data_dir: str = ""
    gift_gate_timeout_s: int = 3600
    # Cascade duel-eval coverage (see cascade.validator.cascade). Cap on datasets
    # per suite when the trainer scores the duel checkpoints on GIFT-Eval / BOOM /
    # TIME. ``0`` = the FULL battery (all configs) — the default, since Cascade's
    # promotion should see the whole eval. Kept separate from the log-only
    # ``benchmark_max_series`` so tightening telemetry never quietly shrinks the
    # Cascade decision. Set a positive cap only to speed up testnet iteration.
    cascade_bench_max_series: int = 0
    # Cascade post-publish bench hold (see provision.policy.bench_hold_active):
    # how long the provisioner may keep the round's FINAL pod alive past its
    # normal teardown signal while the duel bench runs, in hours. The hold ends
    # the moment the trainer's bench_complete marker lands (report uploaded or
    # bench failed); this cap only bounds a crashed/wedged bench. Inert while
    # ``[scoring] cascade_enabled`` is off — the trainer never writes the
    # pending marker and the provisioner never arms the hold.
    bench_hold_max_hours: float = 2.0
    # ── Jittered round mix (DEC-TB-0003 port; CONSENSUS, block-gated) ────────
    # ``mix_from_block = 0`` keeps the legacy uniform permutation everywhere
    # (byte-identical history). When set to an epoch-boundary block, rounds at
    # or after that block draw a Dirichlet-jittered, class-rotated,
    # bag-filtered mix instead — every validator AND the trainer must run this
    # code (with the same value) before the activation block, or verdicts fork
    # from that block on. Audit replay applies each round's own rule via the
    # receipt's epoch_start_block. ``mix_target_windows`` (0 = n_windows) is
    # the jittered draw size — smaller than the pool is what gives the
    # Dirichlet room (calibration: 1200 of a ~2400 pool rotates the mix hard;
    # 2000 barely moves it).
    mix_from_block: int = 0
    mix_jitter_alpha: float = 4.0
    mix_block_slots: int = 8
    mix_class_keep_frac: float = 0.7
    mix_series_bag_frac: float = 0.7
    mix_target_windows: int = 0


@dataclass(frozen=True)
class ScoringConfig:
    win_margin_start: float
    win_margin_end: float
    margin_warmup_rounds: int
    min_windows: int
    bootstrap_B: int
    bootstrap_alpha: float
    dethrone_cp: int
    # Breadth floor for the verdict: below this many distinct window clusters
    # (upstream feeds, from pool metadata ``source``) the round is inconclusive.
    # 0 disables; pools without ``source`` metadata are unaffected. Default keeps
    # older chain.toml loadable.
    min_clusters: int = 0
    # Reward routing: equal weight is split across the current king plus up to
    # ``reward_prior_kings`` previous distinct kings still registered; with none
    # registered, all weight burns to ``burn_uid``. ``reward_prior_kings = 0``
    # reproduces pure winner-take-all. Defaults keep older chain.toml loadable.
    reward_prior_kings: int = 0
    burn_uid: int = 0
    # Geometric decay of the reward across [current king, prior kings by recency]:
    # the king gets a share ∝ 1, the next ∝ king_decay, the next ∝ king_decay²,
    # … normalised. ``king_decay = 1.0`` is the flat equal split (back-compat);
    # ``< 1`` skews to the current king so it is unambiguously the highest-
    # incentive UID (which is how the trainer identifies the king — a flat split
    # ties it with prior kings). E.g. 0.5 gives a 4-king court shares
    # ≈ 0.53/0.27/0.13/0.07.
    king_decay: float = 1.0
    # King-resync safety valve: max consecutive rounds the validator holds the
    # throne for a champion whose trained king disagrees (incentive lag). Past
    # this many holds it abandons the stuck champion and adopts the trainer's
    # trained king (see cascade.validator.state.demote_to_trained). A champion
    # with no usable commitment can never re-sync, so an unbounded hold would
    # wedge the subnet; this bounds recovery. ``<= 0`` disables the valve (hold
    # indefinitely — the pre-safety-valve behaviour). Normal resync is 1 round,
    # so the default leaves generous slack before tripping.
    king_resync_max_rounds: int = 5
    # Public-benchmark no-regression gate (see cascade.eval.koth / .gift_gate):
    # "off" (default) | "shadow" (compute + log, verdict unchanged) | "enforce"
    # (AND into the dethrone decision). ``gift_gate_tolerance`` is the relative
    # slack the challenger may be worse by on gift-eval; ``gift_gate_min_configs``
    # is the shared-config floor below which the gate is uncomputable
    # (→ inconclusive). Reuses ``bootstrap_B``/``bootstrap_alpha``.
    gift_gate_mode: str = "off"
    gift_gate_tolerance: float = 0.03
    gift_gate_min_configs: int = 15
    # Margin denomination (DEC-CA-0027; see cascade.eval.koth.MARGIN_MODES).
    # "level" (default) = LCB of (king − chal)/king — exact current behaviour.
    # "increment" = the warm-start-era rule: the shared init is scored as a
    # third paired reference and the margin prices a fraction of a typical
    # per-round increment. [scoring] is fleet-consensus (not digest-bound):
    # keep identical across validators or verdicts fork. Arm together with the
    # warm-start era flip; rounds with no warm_start_ckpt (random init) are
    # judged at "level" automatically.
    margin_mode: str = "level"
    margin_increment_floor: float = 0.01
    # Cascade — king-reign promotion / warm-start (see cascade.validator.cascade).
    # ``cascade_enabled`` is the master switch: off (default) ⇒ pure KOTH, no
    # reign clock, no public-benchmark scoring, no warm-start promotion. When on,
    # and the reigning king holds the throne ``cascade_reign_days`` worth of
    # BLOCKS (7200/day at 12 s, anchored to the manifest's epoch block so every
    # validator fires the same round) undethroned, the reign's best checkpoint
    # (lowest geomean of the six GIFT-Eval / BOOM / TIME CRPS+MASE numbers the
    # trainer stamps onto the signed manifest) is installed as the warm-start
    # init; the king persists on the throne with a fresh reign clock
    # (DEC-CA-0004). The clock is persisted and survives restarts.
    cascade_enabled: bool = False
    # Threshold in ROUNDS ("survived N challenges"). Field name kept for
    # call-site compatibility; the config key is ``cascade_reign_rounds``.
    cascade_reign_days: int = 7
    # Promotion ENVELOPE knobs (DEC-CA-0013, propose-and-verify). The trainer
    # selects the promoted warm-start set; validators verify the signed
    # declaration against these bounds — so, like cascade_reign_rounds, they
    # are fleet-consensus values ([scoring] is not in contract_digest; keep
    # them identical across validators or their accept/reject verdicts fork).
    # cascade_top_k caps how many member checkpoints one promotion may declare
    # (the trainer may declare fewer); cascade_quality_epsilon is the quality
    # floor — every member's bench geomean must sit within (1 + epsilon) of
    # the best verifiable score in the reign, so diversity can never be bought
    # with a materially worse init.
    cascade_top_k: int = 3
    cascade_quality_epsilon: float = 0.05
    # Block-gated contract transition (the DEC-CA-0016/0019 release-then-
    # activate pattern applied to contract_digest itself). When BOTH are set,
    # a manifest whose epoch-boundary block precedes ``contract_from_block``
    # is accepted against ``prior_contract_digest`` instead of the digest
    # recomputed from the local [training] — so validators can restart onto a
    # release that changes the training contract WITHOUT sacrificing the
    # round currently in flight under the old contract, and the fleet needs
    # no synchronized restart window. Rounds at/after the block accept only
    # the new digest. [scoring] lives outside contract_digest, so these keys
    # never perturb the digest they gate. "" / 0 = gate off (legacy exact
    # match). Fleet-consensus values like every [scoring] key: keep identical
    # across validators. Retire the pin (back to ""/0) in the release AFTER
    # the transition round is beyond every validator's scoring horizon.
    prior_contract_digest: str = ""
    contract_from_block: int = 0


@dataclass(frozen=True)
class DependencyConfig:
    max_packages: int
    allowed: tuple[str, ...]


@dataclass(frozen=True)
class StaticGuardConfig:
    blocked: tuple[str, ...]


@dataclass(frozen=True)
class StorageConfig:
    """Hippius storage endpoints (credentials come from the environment).

    The **registry** (Hippius Hub OCI, ``hub_registry_url``) stores models/
    checkpoints/generators pinned by ``repo@digest``; **S3** (``s3_endpoint``)
    stores training manifests (``manifest_bucket``) and per-round training logs
    (``logs_bucket``).
    """

    hub_registry_url: str
    hub_namespace: str
    s3_endpoint: str
    s3_region: str
    manifest_bucket: str
    logs_bucket: str
    # Eval-pool bucket (daily snapshots + pool/index.json). When ``pool_bucket``
    # is set the validator pulls the rotating pool from here instead of a static
    # ``[eval] window_pool`` CID. ``pool_s3_endpoint`` / ``pool_s3_region`` default
    # to the Hippius S3 endpoint above; point them at Cloudflare R2 (with
    # ``POOL_S3_ACCESS_KEY`` / ``POOL_S3_SECRET_KEY``) to publish there instead.
    pool_bucket: str = ""
    pool_s3_endpoint: str = ""
    pool_s3_region: str = ""
    # Cloudflare R2 live backup of the eval-pool bucket. When ``pool_backup_bucket``
    # is set, every pool write (snapshot tars + ``pool/index.json``) is mirrored to
    # this bucket and validator reads fall back to it when Hippius S3 is down — so a
    # Hippius 5xx no longer strands the pool-pin gate as ``pool_pin_unverifiable``
    # (see cascade.shared.hippius.S3MirrorStore / pool_s3_store). The endpoint,
    # region, and credentials are REUSED from the ``backup_*`` R2 below
    # (BACKUP_S3_ACCESS_KEY / BACKUP_S3_SECRET_KEY); only the bucket differs. Empty
    # ⇒ no pool backup (plain S3, unchanged). Hippius stays primary — this is a
    # backup, never a hard dependency: an R2-only failure is logged, not raised.
    pool_backup_bucket: str = ""
    # HuggingFace-Hub dataset repo used as a manifest/receipt fallback ONLY when
    # Hippius S3 is down (see cascade.shared.hippius.HFFallbackStore). Empty ⇒
    # no fallback (plain S3). Make it a PUBLIC dataset so receipts stay auditable
    # during an outage; auth via HF_TOKEN.
    hf_backup_repo: str = ""
    # Cloudflare R2 (or any S3-compatible) backup of the manifest/receipt bucket.
    # When ``backup_s3_endpoint`` is set every manifest/receipt write is mirrored
    # here (dual-write) and reads fall back here when Hippius S3 is unavailable —
    # a full off-Hippius backup, not just an outage failover (see
    # cascade.shared.hippius.S3MirrorStore). ``backup_bucket`` defaults to the
    # primary ``manifest_bucket`` name; ``backup_s3_region`` defaults to R2's
    # ``"auto"``. Credentials via BACKUP_S3_ACCESS_KEY / BACKUP_S3_SECRET_KEY.
    backup_bucket: str = ""
    backup_s3_endpoint: str = ""
    backup_s3_region: str = ""
    # Private king archive: a dedicated S3-compatible (Cloudflare R2) bucket where
    # `cascade-scrape-kings` saves every generator that has held the throne — the
    # code, fetched from the Hub by its content-addressed ref and packed to a
    # deterministic tar — under ``kings/`` plus a ``kings/index.json`` "db", and
    # EVERY eligible participant generator under ``generators/<hotkey>/`` (grouped
    # by committing miner) with its own index (see cascade.shared.king_archive).
    # This is a permanent private record,
    # independent of the public Hub repos (which a miner could delete).
    # ``king_archive_s3_endpoint`` / ``king_archive_s3_region``
    # default to the R2 ``backup_*`` values above; ``king_archive_bucket`` defaults
    # to ``cascade-king-archive``. Credentials via KING_ARCHIVE_S3_ACCESS_KEY /
    # KING_ARCHIVE_S3_SECRET_KEY, falling back to the BACKUP_S3_* pair when unset.
    king_archive_bucket: str = ""
    king_archive_s3_endpoint: str = ""
    king_archive_s3_region: str = ""


@dataclass(frozen=True)
class WandbConfig:
    """Optional Weights & Biases logging for the reference trainer.

    Observability only — wandb numbers never feed scoring or weights. When
    ``enabled`` (and the ``wandb`` package + ``WANDB_API_KEY`` are present, for
    ``mode = "online"``), the trainer mirrors the same per-step records it streams
    to the Hippius S3 log into a **live** wandb run, one run per ``(round,
    competitor, size)`` tagged with the miner hotkey/uid. Point
    ``project``/``entity`` at a PUBLIC wandb project so miners can watch their
    generator train as it occurs. Credentials come from the environment
    (``WANDB_API_KEY``), never this committed file. Defaults keep old toml
    loading (the whole ``[wandb]`` section is optional).
    """

    enabled: bool = False
    project: str = "cascade"
    entity: str = ""
    mode: str = "online"   # online | offline | disabled


@dataclass(frozen=True)
class TelemetryConfig:
    """Host-side run telemetry (:mod:`cascade.trainer.host_probe`).

    Observability only, and never part of ``contract_digest``: these fields ride
    the training-log channel next to the per-step metrics and are read by nobody
    in the scoring path. They exist because the heat prices generator speed
    (DEC-CA-0001) on a fleet rented per round — so "was this run slow because of
    the submission or because of the pod?" needs host-side evidence, and
    ``data_wait_frac`` can only rule out the corpus.

    ``host_probe`` gathers the free facts (lane geometry, CPU/GPU capability,
    opaque pod + machine ids). ``host_bench`` additionally times the fixed
    calibration workload before the corpus stream opens — a few seconds of GPU
    per run, and the only field that is comparable ACROSS pods. Both default on;
    turn ``host_bench`` off if a provider's pods are so slow to boot that the
    extra seconds matter. There is deliberately no size knob: a resizable bench
    produces numbers that cannot be pooled across the fleet (see
    ``host_probe.HOST_BENCH_SPEC``).

    ``scratch_shadow_every_rounds`` (DEC-CA-0014 Stage 1) arms the shadow
    scratch control: every M-th warm-started round, the trainer additionally
    trains the reigning king's generator FROM SCRATCH (random init, identical
    contract) after the manifest publishes, benches it on the public suites,
    and publishes the numbers as a clearly-labeled telemetry artifact
    (``cascade.shared.scratch_report``) beside the round's signed bench
    report. The scratch run never enters the manifest, the bench report, the
    promotion candidate pool, or any scoring path — it exists to measure the
    lineage-vs-scratch gap that decides whether DEC-CA-0014's Stage-2 reseed
    valve is ever warranted. ``0`` (the default) = off. Deliberately a
    ``[telemetry]`` key: it must never touch ``contract_digest``.
    """

    host_probe: bool = True
    host_bench: bool = True
    scratch_shadow_every_rounds: int = 0


@dataclass(frozen=True)
class ManifestConfig:
    """Where the trainer publishes training receipts and the validator reads
    them. Manifests live in the ``[storage] manifest_bucket`` S3 bucket;
    ``trainer_hotkey`` is the only hotkey whose manifest a validator trusts.

    ``validator_hotkey`` is the trust anchor for public *round receipts*
    (``cascade.shared.receipt``): when set, ``cascade-audit`` requires receipts
    to be signed by exactly this ss58; empty ⇒ the audit verifies against the
    receipt's self-declared signer and WARNs that the signer is unpinned.
    """

    trainer_hotkey: str
    poll_seconds: int
    validator_hotkey: str = ""


@dataclass(frozen=True)
class ValidatorConfig:
    weight_set_interval_blocks: int
    poll_seconds: int
    hf_cache_seconds: int
    state_db_path: str
    # Cascade persistence (see cascade.validator.cascade). ``cascade_state_db_path``
    # holds the reign clock + reign checkpoint log (JSON) so Cascade survives
    # restarts; ``warm_start_init_path`` is where a fired Cascade writes the
    # promoted checkpoint pointer for the trainer to warm-start every subsequent
    # round from. Defaults keep older chain.toml loadable.
    cascade_state_db_path: str = "cascade_state.json"
    warm_start_init_path: str = "warm_start_init.json"
    # First-boot champion inheritance: with no local state, adopt the throne
    # from the signed public receipt trail (verified against the pinned
    # manifest/receipt hotkey) instead of judging the next manifest blind.
    # A validator joining mid-reign otherwise crowns whichever king it happens
    # to see win first. Default on; existing state always wins over bootstrap.
    bootstrap_from_receipts: bool = True
    # Operator kill-switch: push the burn vector instead of the champion on
    # EVERY weight-set (round votes, resync votes, and re-asserts) while still
    # refreshing last_update — the validator stays Yuma-active but endorses no
    # miner. Champion state keeps evolving on disk untouched, so flipping this
    # back off resumes voting the persisted king immediately. For "something
    # looks wrong, stop endorsing while I investigate" — NOT a shutdown.
    force_burn: bool = False


@dataclass(frozen=True)
class ChainConfig:
    schema_version: int
    subnet: SubnetConfig
    generator: GeneratorConfig
    training: TrainingContractConfig
    round: RoundConfig
    eval: EvalConfig
    scoring: ScoringConfig
    dependencies: DependencyConfig
    static_guard: StaticGuardConfig
    storage: StorageConfig
    manifest: ManifestConfig
    validator: ValidatorConfig
    wandb: WandbConfig = field(default_factory=WandbConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def netuid(self) -> int:
        return self.subnet.netuid

    def screen_contract(self) -> TrainingContractConfig:
        """The single-size contract the heat screens at — ``[round] screen_size``
        resolved against the size registry, defaulting to the primary size."""
        name = self.round.screen_size or self.training.arch_preset
        return self.training.contract_for(name)

    def throne_contracts(self) -> list[TrainingContractConfig]:
        """The single-size contracts the final trains + the throne is judged at —
        ``[round] throne_sizes`` resolved against the registry, defaulting to just
        the primary size. One element ⇒ a single-size throne; several ⇒ the
        combined-score throne pools across them."""
        names = self.round.throne_sizes or (self.training.arch_preset,)
        return [self.training.contract_for(n) for n in names]

    def koth_params(self) -> Any:
        """Build a :class:`cascade.eval.koth.KothParams` from ``[scoring]``.

        Imported lazily so :mod:`cascade.shared.config` stays free of the
        eval package at import time.
        """
        from ..eval.koth import KothParams

        return KothParams(
            win_margin_start=self.scoring.win_margin_start,
            win_margin_end=self.scoring.win_margin_end,
            margin_warmup_rounds=self.scoring.margin_warmup_rounds,
            min_windows=self.scoring.min_windows,
            bootstrap_B=self.scoring.bootstrap_B,
            bootstrap_alpha=self.scoring.bootstrap_alpha,
            dethrone_cp=self.scoring.dethrone_cp,
            min_clusters=self.scoring.min_clusters,
            gift_gate_mode=self.scoring.gift_gate_mode,
            gift_gate_tolerance=self.scoring.gift_gate_tolerance,
            gift_gate_min_configs=self.scoring.gift_gate_min_configs,
            margin_mode=self.scoring.margin_mode,
            margin_increment_floor=self.scoring.margin_increment_floor,
        )


class LaunchConfigError(RuntimeError):
    """chain.toml still carries launch placeholders that must be set."""


_PLACEHOLDER_DIGEST = "0" * 64


def effective_epoch_blocks(round_cfg: RoundConfig, block: int) -> int:
    """Round length in force AT ``block`` — the only correct way to turn a block
    into an epoch.

    A cadence change is consensus-relevant: the epoch is derived from the block
    grid (``block // epoch_blocks``), so two validators using different lengths
    for the same round select different eval-pool snapshots and reach different
    verdicts. Restart timing cannot coordinate that across independent
    operators, so the switch is pinned to ``epoch_activation_block``: every node
    flips at the same block regardless of when it restarted, and an operator who
    pulls early is not punished for it.

    Returns ``epoch_blocks_prev`` for blocks strictly before the activation
    block, ``epoch_blocks`` from it onward. With no scheduled change (both keys
    0, the steady state) this is just ``epoch_blocks`` and the call is free.

    ALWAYS pass the block the epoch is being derived FOR — a manifest's
    ``created_block``, a receipt's ``epoch_start_block``, a reveal target —
    never the current head. Deriving a historical round's epoch from the head
    is what makes an audit disagree with the validator that wrote the receipt.
    ``load_chain_config`` guarantees the activation block is a boundary of both
    grids, so the seam is unambiguous.
    """
    eb = max(1, int(round_cfg.epoch_blocks))
    act = int(round_cfg.epoch_activation_block)
    prev = int(round_cfg.epoch_blocks_prev)
    if act and prev > 0 and int(block) < act:
        return max(1, prev)
    return eb


def assert_launch_ready(cfg: ChainConfig, *, role: str) -> None:
    """Refuse to start a live service while ``chain.toml`` holds placeholders.

    ``role`` is ``"trainer"`` or ``"validator"``; each needs a slightly different
    set. Raises :class:`LaunchConfigError` listing every unset value so the
    operator fixes them in one pass rather than one failed launch at a time.
    """
    problems: list[str] = []
    if cfg.netuid <= 0:
        problems.append("[subnet] netuid is 0 (set the live netuid)")
    if cfg.training.base_arch_digest in ("", _PLACEHOLDER_DIGEST) or len(cfg.training.base_arch_digest) != 64:
        problems.append(
            "[training] base_arch_digest is a placeholder "
            "(run `cascade-trainer --offline` and paste the printed digest)"
        )
    for spec in cfg.training.extra_sizes:
        if spec.base_arch_digest in ("", _PLACEHOLDER_DIGEST) or len(spec.base_arch_digest) != 64:
            problems.append(
                f"[[training.sizes]] base_arch_digest for {spec.arch_preset!r} is a "
                "placeholder (run `cascade-trainer --offline` and paste the printed digest)"
            )
    if not cfg.manifest.trainer_hotkey:
        problems.append("[manifest] trainer_hotkey is empty (set the owner trainer ss58 hotkey)")
    # The round's screen/throne size pointers must name configured sizes.
    registry = cfg.training.size_registry
    for label, name in [("screen_size", cfg.round.screen_size), *(("throne_sizes", n) for n in cfg.round.throne_sizes)]:
        if name and name not in registry:
            problems.append(
                f"[round] {label} = {name!r} is not a configured size {sorted(registry)} "
                "(add the [[training.sizes]] block or fix the name)"
            )
    if role == "validator":
        from .hippius import is_hub_ref

        # Either a daily-published pool bucket OR a static window_pool ref.
        if not cfg.storage.pool_bucket and not is_hub_ref(cfg.eval.window_pool):
            problems.append(
                "no eval pool configured: set [storage] pool_bucket (daily snapshots, "
                "recommended) or pin a [eval] window_pool Hippius Hub ref (repo@digest)"
            )
    if problems:
        raise LaunchConfigError(
            "chain.toml is not launch-ready:\n  - " + "\n  - ".join(problems)
        )


_MARGIN_MODES = ("level", "increment")


def _margin_mode(value: object) -> str:
    """Validate ``[scoring] margin_mode`` at load — a typo here would silently
    judge every round under the wrong denomination."""
    mode = str(value)
    if mode not in _MARGIN_MODES:
        raise ValueError(f"[scoring] margin_mode={mode!r} invalid; one of {_MARGIN_MODES}")
    return mode


_GIFT_GATE_MODES = ("off", "shadow", "enforce")


def _gift_gate_mode(value: object) -> str:
    """Validate ``[scoring] gift_gate_mode`` at load time so a typo fails fast
    rather than silently disabling the consensus gate."""
    mode = str(value)
    if mode not in _GIFT_GATE_MODES:
        raise ValueError(
            f"[scoring] gift_gate_mode={mode!r} invalid; one of {_GIFT_GATE_MODES}"
        )
    return mode


def load_chain_config(path: Path | str | None = None) -> ChainConfig:
    """Load and parse ``chain.toml``. Raises on a missing file or unsupported
    (too-old) schema; warns and proceeds on a newer schema."""
    p = Path(path) if path is not None else DEFAULT_CHAIN_TOML
    if not p.exists():
        raise FileNotFoundError(f"chain.toml not found at {p}")
    with p.open("rb") as fh:
        raw = tomllib.load(fh)

    schema = int(raw.get("schema_version", 0))
    if schema < 1:
        raise ValueError(f"chain.toml schema_version={schema} unsupported; need >=1")
    if schema > 1:
        print(
            f"warning: chain.toml schema_version={schema} is newer than this code (1); "
            "fields may be ignored",
            file=sys.stderr,
        )

    sub = raw.get("subnet", {})
    g = raw["generator"]
    t = raw["training"]
    e = raw["eval"]
    s = raw["scoring"]
    d = raw["dependencies"]
    sg = raw["static_guard"]
    st = raw.get("storage", {})
    m = raw["manifest"]
    v = raw["validator"]
    r = raw.get("round", {})
    wb = raw.get("wandb", {})
    tm = raw.get("telemetry", {})

    # Scheduled cadence change — reject a seam that would cut a round short.
    # Both grids must agree at the activation block, else the epoch containing
    # it has two different lengths depending on which side you ask from.
    _eb = int(r.get("epoch_blocks", 7200))
    _ebp = int(r.get("epoch_blocks_prev", 0))
    _eab = int(r.get("epoch_activation_block", 0))
    if bool(_ebp) != bool(_eab):
        raise ValueError(
            "[round] epoch_blocks_prev and epoch_activation_block must be set "
            f"together (got prev={_ebp}, activation={_eab}); both 0 = no "
            "scheduled cadence change"
        )
    if _eab:
        if _ebp < 1 or _eb < 1:
            raise ValueError(
                f"[round] epoch_blocks={_eb} and epoch_blocks_prev={_ebp} must both be >= 1"
            )
        if _eab % _eb or _eab % _ebp:
            raise ValueError(
                f"[round] epoch_activation_block={_eab} must be a multiple of BOTH "
                f"epoch_blocks={_eb} and epoch_blocks_prev={_ebp} — otherwise the "
                "round spanning the seam has no well-defined length"
            )

    # Extra final-stage sizes ([[training.sizes]] array of tables). The base
    # [training] block is always the primary size; these are trained alongside it.
    extra_sizes = tuple(
        SizeSpec(
            arch_preset=str(z["arch_preset"]),
            base_arch_digest=str(z["base_arch_digest"]),
            d_model=int(z["d_model"]),
            num_layers=int(z["num_layers"]),
            num_heads=int(z["num_heads"]),
            mlp_expansion=int(z["mlp_expansion"]),
            ref_throughput_tokens_per_s=int(z["ref_throughput_tokens_per_s"]),
            d_ff=int(z.get("d_ff", 0)),
            expected_gpu=str(z.get("expected_gpu", "")),
            target_train_hours=float(z.get("target_train_hours", 0.0)),
        )
        for z in t.get("sizes", [])
    )

    return ChainConfig(
        schema_version=schema,
        subnet=SubnetConfig(
            netuid=int(sub.get("netuid", 0)),
            name=str(sub.get("name", "cascade")),
            description=str(sub.get("description", "")),
        ),
        generator=GeneratorConfig(
            corpus_n_series=int(g["corpus_n_series"]),
            min_length=int(g["min_length"]),
            max_length=int(g["max_length"]),
            max_total_points=int(g["max_total_points"]),
            max_generate_seconds=int(g["max_generate_seconds"]),
            max_memory_mb=int(g["max_memory_mb"]),
            max_repo_mb=int(g.get("max_repo_mb", 2048)),
            max_channels=int(g.get("max_channels", 1)),
            max_abs_value=float(g.get("max_abs_value", 0.0)),
            reject_constant=bool(g.get("reject_constant", False)),
            max_dup_fraction=float(g.get("max_dup_fraction", 1.0)),
            interface_version=int(g.get("interface_version", 1)),
            max_payload_bytes=int(g.get("max_payload_bytes", 0)),
            # Mirrored from the digest-bound [training] keys — single source.
            accepted_fields=validate_accepted_fields(t.get("accepted_fields", ())),
            allow_future_known=bool(t.get("allow_future_known", False)),
            real_corpus_ref=validate_real_corpus_ref(t.get("real_corpus_ref", "")),
            max_missing_frac=float(g.get("max_missing_frac", 0.5)),
            channel_corr_mode=validate_dedup_mode(
                str(g.get("channel_corr_mode", "off")), "channel_corr_mode"),
            max_channel_corr=float(g.get("max_channel_corr", 0.999)),
            sandbox_mode=validate_sandbox_mode(str(g.get("sandbox_mode", "subprocess"))),
            sandbox_image=str(g.get("sandbox_image", "")),
            sandbox_python=str(g.get("sandbox_python", "python3")),
            sandbox_strict=bool(g.get("sandbox_strict", False)),
        ),
        training=TrainingContractConfig(
            base_arch=str(t["base_arch"]),
            arch_preset=str(t["arch_preset"]),
            base_arch_digest=str(t["base_arch_digest"]),
            d_model=int(t["d_model"]),
            num_layers=int(t["num_layers"]),
            num_heads=int(t["num_heads"]),
            head_dim=int(t["head_dim"]),
            patch_size=int(t["patch_size"]),
            mlp_expansion=int(t["mlp_expansion"]),
            d_ff=int(t.get("d_ff", 0)),
            num_quantiles=int(t["num_quantiles"]),
            masking=str(t["masking"]),
            cpm_c_max=int(t["cpm_c_max"]),
            cpm_p_max=float(t["cpm_p_max"]),
            input_transform=str(t["input_transform"]),
            context_length=int(t["context_length"]),
            horizon=int(t["horizon"]),
            target_train_hours=float(t["target_train_hours"]),
            ref_throughput_tokens_per_s=int(t["ref_throughput_tokens_per_s"]),
            warmup_fraction=float(t["warmup_fraction"]),
            batch_size=int(t["batch_size"]),
            optimizer=str(t["optimizer"]),
            base_lr=float(t["base_lr"]),
            weight_decay=float(t["weight_decay"]),
            lr_schedule=str(t["lr_schedule"]),
            umup_base_d_model=int(t["umup_base_d_model"]),
            train_seed_salt=int(t["train_seed_salt"]),
            max_train_seconds=int(t["max_train_seconds"]),
            corpus_mode=validate_corpus_mode(str(t.get("corpus_mode", "stream_cpu"))),
            expected_gpu=str(t.get("expected_gpu", "")),
            train_image_digest=str(t.get("train_image_digest", "")),
            extra_sizes=extra_sizes,
            accepted_fields=validate_accepted_fields(t.get("accepted_fields", ())),
            allow_future_known=bool(t.get("allow_future_known", False)),
            real_corpus_ref=validate_real_corpus_ref(t.get("real_corpus_ref", "")),
        ),
        round=RoundConfig(
            epoch_blocks=int(r.get("epoch_blocks", 7200)),
            round_hours=float(r.get("round_hours", 24.0)),
            epoch_blocks_prev=int(r.get("epoch_blocks_prev", 0)),
            epoch_activation_block=int(r.get("epoch_activation_block", 0)),
            heat_train_hours=float(r.get("heat_train_hours", 0.5)),
            heat_n_windows=int(r.get("heat_n_windows", 256)),
            heat_num_samples=int(r.get("heat_num_samples", 0)),
            heat_guard_factor=float(r.get("heat_guard_factor", 1.0)),
            heat_guard_floor_seconds=int(r.get("heat_guard_floor_seconds", 900)),
            finalists=int(r.get("finalists", 1)),
            max_finalists=int(r.get("max_finalists", 1)),
            tie_runoff_windows=int(r.get("tie_runoff_windows", 0)),
            tie_runoff_phase_seconds=int(r.get("tie_runoff_phase_seconds", 900)),
            screen_size=str(r.get("screen_size", "")),
            throne_sizes=tuple(str(x) for x in r.get("throne_sizes", ())),
            one_submission_per_hotkey=bool(r.get("one_submission_per_hotkey", True)),
            commit_floor_block=int(r.get("commit_floor_block", 0)),
            genesis_generator_ref=str(r.get("genesis_generator_ref", "")),
            submissions_db_path=str(r.get("submissions_db_path", "trainer_submissions.json")),
            commit_witness_path=str(r.get("commit_witness_path",
                                          "trainer_commit_witness.json")),
            dedup_mode=validate_dedup_mode(str(r.get("dedup_mode", "off")),
                                           "dedup_mode"),
            dedup_config_only_enforce=bool(r.get("dedup_config_only_enforce", False)),
            dedup_max_tokens=int(r.get("dedup_max_tokens", 50_000)),
            dedup_max_text_mb=int(r.get("dedup_max_text_mb", 4)),
            dedup_phase_seconds=int(r.get("dedup_phase_seconds", 900)),
            dedup_plan_seconds=int(r.get("dedup_plan_seconds", 120)),
            dedup_probe_mode=validate_dedup_mode(
                str(r.get("dedup_probe_mode", "shadow")), "dedup_probe_mode"),
            dedup_probe_series=int(r.get("dedup_probe_series", 8)),
            dedup_probe_generate_seconds=int(r.get("dedup_probe_generate_seconds", 120)),
            dedup_probe_budget_seconds=int(r.get("dedup_probe_budget_seconds", 600)),
            dedup_probe_allow_weak_sandbox=bool(
                r.get("dedup_probe_allow_weak_sandbox", False)),
            reveal_margin_blocks=int(r.get("reveal_margin_blocks", 25)),
        ),
        eval=EvalConfig(
            eval_dataset=str(e["eval_dataset"]),
            eval_source=str(e.get("eval_source", "private-rotating")),
            window_pool=str(e.get("window_pool", "")),
            num_samples=int(e["num_samples"]),
            n_windows=int(e["n_windows"]),
            context_length=int(e["context_length"]),
            horizon=int(e["horizon"]),
            run_benchmarks=bool(e.get("run_benchmarks", False)),
            benchmark_project_dir=str(e.get("benchmark_project_dir", "benchmarks")),
            benchmark_suites=tuple(str(s) for s in e.get("benchmark_suites", ())),
            benchmark_num_samples=int(e.get("benchmark_num_samples", 0)),
            benchmark_max_series=int(e.get("benchmark_max_series", 0)),
            gift_gate_datasets=str(e.get("gift_gate_datasets", "")),
            gift_gate_num_samples=int(e.get("gift_gate_num_samples", 0)),
            gift_gate_data_dir=str(e.get("gift_gate_data_dir", "")),
            gift_gate_timeout_s=int(e.get("gift_gate_timeout_s", 3600)),
            cascade_bench_max_series=int(e.get("cascade_bench_max_series", 0)),
            bench_hold_max_hours=float(e.get("bench_hold_max_hours", 2.0)),
            mix_from_block=int(e.get("mix_from_block", 0)),
            mix_jitter_alpha=float(e.get("mix_jitter_alpha", 4.0)),
            mix_block_slots=int(e.get("mix_block_slots", 8)),
            mix_class_keep_frac=float(e.get("mix_class_keep_frac", 0.7)),
            mix_series_bag_frac=float(e.get("mix_series_bag_frac", 0.7)),
            mix_target_windows=int(e.get("mix_target_windows", 0)),
        ),
        scoring=ScoringConfig(
            win_margin_start=float(s["win_margin_start"]),
            win_margin_end=float(s["win_margin_end"]),
            margin_warmup_rounds=int(s["margin_warmup_rounds"]),
            min_windows=int(s["min_windows"]),
            bootstrap_B=int(s["bootstrap_B"]),
            bootstrap_alpha=float(s["bootstrap_alpha"]),
            dethrone_cp=int(s["dethrone_cp"]),
            min_clusters=int(s.get("min_clusters", 0)),
            reward_prior_kings=int(s.get("reward_prior_kings", 0)),
            burn_uid=int(s.get("burn_uid", 0)),
            king_decay=float(s.get("king_decay", 1.0)),
            king_resync_max_rounds=int(s.get("king_resync_max_rounds", 5)),
            gift_gate_mode=_gift_gate_mode(s.get("gift_gate_mode", "off")),
            gift_gate_tolerance=float(s.get("gift_gate_tolerance", 0.03)),
            gift_gate_min_configs=int(s.get("gift_gate_min_configs", 15)),
            margin_mode=_margin_mode(s.get("margin_mode", "level")),
            margin_increment_floor=float(s.get("margin_increment_floor", 0.01)),
            cascade_enabled=bool(s.get("cascade_enabled", False)),
            # ``cascade_reign_rounds`` is the current name; ``cascade_reign_days``
            # is accepted as a legacy alias so deployed files keep loading. The
            # UNIT changed with the clock (days -> rounds) — at the 24h cadence
            # the two were numerically identical, which is why the old value
            # carries over unchanged.
            cascade_reign_days=int(
                s.get("cascade_reign_rounds", s.get("cascade_reign_days", 7))
            ),
            cascade_top_k=int(s.get("cascade_top_k", 3)),
            cascade_quality_epsilon=float(s.get("cascade_quality_epsilon", 0.05)),
            prior_contract_digest=str(s.get("prior_contract_digest", "") or ""),
            contract_from_block=int(s.get("contract_from_block", 0) or 0),
        ),
        dependencies=DependencyConfig(
            max_packages=int(d["max_packages"]),
            allowed=tuple(str(x) for x in d["allowed"]),
        ),
        static_guard=StaticGuardConfig(
            blocked=tuple(str(x) for x in sg["blocked"]),
        ),
        storage=StorageConfig(
            hub_registry_url=str(st.get("hub_registry_url", "https://registry.hippius.com")),
            hub_namespace=str(st.get("hub_namespace", "cascade")),
            s3_endpoint=str(st.get("s3_endpoint", "https://s3.hippius.com")),
            s3_region=str(st.get("s3_region", "decentralized")),
            manifest_bucket=str(st.get("manifest_bucket", "cascade-manifests")),
            logs_bucket=str(st.get("logs_bucket", "cascade-logs")),
            pool_bucket=str(st.get("pool_bucket", "")),
            pool_s3_endpoint=str(st.get("pool_s3_endpoint", "")),
            pool_s3_region=str(st.get("pool_s3_region", "")),
            pool_backup_bucket=str(st.get("pool_backup_bucket", "")),
            hf_backup_repo=str(st.get("hf_backup_repo", "")),
            backup_bucket=str(st.get("backup_bucket", "")),
            backup_s3_endpoint=str(st.get("backup_s3_endpoint", "")),
            backup_s3_region=str(st.get("backup_s3_region", "")),
            king_archive_bucket=str(st.get("king_archive_bucket", "")),
            king_archive_s3_endpoint=str(st.get("king_archive_s3_endpoint", "")),
            king_archive_s3_region=str(st.get("king_archive_s3_region", "")),
        ),
        manifest=ManifestConfig(
            trainer_hotkey=str(m["trainer_hotkey"]),
            poll_seconds=int(m["poll_seconds"]),
            validator_hotkey=str(m.get("validator_hotkey", "")),
        ),
        validator=ValidatorConfig(
            weight_set_interval_blocks=int(v["weight_set_interval_blocks"]),
            poll_seconds=int(v["poll_seconds"]),
            hf_cache_seconds=int(v["hf_cache_seconds"]),
            state_db_path=str(v["state_db_path"]),
            cascade_state_db_path=str(v.get("cascade_state_db_path", "cascade_state.json")),
            warm_start_init_path=str(v.get("warm_start_init_path", "warm_start_init.json")),
            bootstrap_from_receipts=bool(v.get("bootstrap_from_receipts", True)),
            force_burn=bool(v.get("force_burn", False)),
        ),
        wandb=WandbConfig(
            enabled=bool(wb.get("enabled", False)),
            project=str(wb.get("project", "cascade")),
            entity=str(wb.get("entity", "")),
            mode=str(wb.get("mode", "online")),
        ),
        telemetry=TelemetryConfig(
            host_probe=bool(tm.get("host_probe", True)),
            host_bench=bool(tm.get("host_bench", True)),
            scratch_shadow_every_rounds=int(tm.get("scratch_shadow_every_rounds", 0)),
        ),
        raw=raw,
    )
