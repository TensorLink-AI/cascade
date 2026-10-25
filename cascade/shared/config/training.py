"""The fixed training contract: SizeSpec, TrainingContractConfig and its [training] parser."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .validators import (
    validate_accepted_fields,
    validate_anneal_shape,
    validate_batch_denomination,
    validate_budget_denomination,
    validate_context_after,
    validate_corpus_epochs,
    validate_corpus_mode,
    validate_long_series,
    validate_real_corpus_ref,
)

# Reserved for model definitions that have not been supplied or qualified.
# Release each ID only when its actual backend and full contract are wired.
UNIMPLEMENTED_ARCH_PRESETS = frozenset(f"cascade-moe-v2-{size}m" for size in (2, 7, 28, 56, 120))


@dataclass(frozen=True)
class SizeSpec:
    """Additional model size for a combined score across configured sizes.

    Overrides shape, architecture digest, throughput, GPU and budget through
    :meth:`TrainingContractConfig.for_size`. Other recipe fields are inherited
    from the base contract.
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
    # Per-size GPU/budget overrides (DEC-CA-0027); empty/zero inherits the base.
    # Throughput must be measured on this GPU. Fields are digest-bound through
    # extra_sizes, so changing them changes the size's training identity.
    expected_gpu: str = ""
    target_train_hours: float = 0.0


# Defaults for TrainingContractConfig.for_hours and legacy heat screening.
# The derived wall guard uses that leg's hours, a multiplier and an overhead
# floor; reaching it sets deadline_hit. Callers may override through [round].
HEAT_GUARD_FACTOR = 1.0
HEAT_GUARD_FLOOR_SECONDS = 900


@dataclass(frozen=True)
class TrainingContractConfig:
    """Training recipe shared by comparable king and challenger legs.

    The effective contract selects Toto2 or Cascade-MoE via :meth:`at_block`;
    the era separately pins random initialization or a shared warm start.
    Architecture, objective, masking, optimizer and budget are contract terms.
    Runtime flags and activation schedules are excluded from the digest by
    the projection in ``cascade.shared.manifest``.

    ``train_tokens`` is target_train_hours × 3600 × reference throughput,
    counted under budget_denomination. ``max_train_seconds`` also caps wall
    time, so a slower leg may stop before consuming the full point budget.
    """

    # Identity / architecture of the effective model family.
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
    # Point budget = reference GPU hours × reference throughput.
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
    # ── batch denomination at C > 1 (DEC-CA-0041; digest-bound) ──────────────
    # What ``batch_size`` COUNTS when a series carries C > 1 variates.
    # "series": a (P, C) bucket fills to batch_size series — a C=32 batch is
    # batch_size×32 sequences, so a fixed token budget buys C× fewer optimizer
    # steps (Toto 2's own batch geometry: 64 series × 32 variates).
    # "sequences": a bucket fills to max(1, batch_size // C) series, holding
    # tokens-per-step ~constant across C — every miner gets the same step
    # count from the same budget whatever channel mix they emit, at the price
    # of fewer independent groups per wide step. Bit-identical at C = 1 either
    # way (and C > 1 was impossible before max_channels rose, so no historical
    # round is affected). Drop-when-default: "series" is absent from
    # contract_digest; arming "sequences" is a deliberate contract cut — king
    # and challenger batch identically either way, but every trainer must agree.
    batch_denomination: str = "series"
    # ── budget denomination at C > 1 (DEC-CA-0042; digest-bound) ─────────────
    # What ONE budget point is when a series carries C > 1 variates — the
    # rule the stream's stop, the trainer's token counter, and the audit
    # re-derivation all share (the corpus digest is the rolling digest of the
    # consumed prefix, so the stop rule is consensus-relevant).
    # "points": every values entry is a point — a (C, L) series costs C×L
    # (the legacy rule; C > 1 was unreachable while max_channels was 1).
    # "series_points": a (C, L) series costs L — per time-step positions, so
    # the channel-token budget a corpus earns scales with its width
    # (univariate 1×, C=32 → 32×). Paired with batch_denomination = "series"
    # (64 series per step whatever C) this holds the STEP count equal across
    # widths while a wide corpus trains C× the tokens per step: the
    # multivariate advantage DEC-CA-0041's ablation measured, bought at C×
    # compute per step under the unchanged wall (max_train_seconds is still
    # the law — a wide leg that cannot finish its budget stops at the wall,
    # flagged deadline_hit, like any other). Bit-identical at C = 1.
    # Drop-when-default: "points" is absent from contract_digest; setting
    # "series_points" is a deliberate contract cut — trainer, worker image,
    # and cascade-audit must all read it (an old worker would silently bill
    # the legacy rule and stop C× early).
    # "points+mv<PCT>" (DEC-CA-0047): token billing (C×L) with a width bonus —
    # a C > 1 series is billed 100/(100+PCT) of its points, so an all-
    # multichannel corpus trains PCT % more tokens than a univariate one and
    # the GPU drawn never decides the token count (the budget binds, not the
    # wall). A copied or junk second channel spends real budget. Digest-bound
    # like the other values; the same string travels to the worker, the audit
    # and the miner's local scorer.
    budget_denomination: str = "points"
    # Scheduled switch (DEC-CA-0047): from chain block `budget_denomination_after_block`
    # the effective denomination is `budget_denomination_after` (the two fields are
    # NEVER part of contract_digest — see manifest._NEVER_IN_DIGEST — so pinning a
    # schedule moves no in-flight digest; the EFFECTIVE contract at a block is
    # `at_block(block)`, whose digest differs only past the gate). Every leg of an
    # era carries the era's start block (`--contract-block`), so king pre-train,
    # challengers, the settlement manifest and the audit all agree by construction.
    # "" / 0 = no schedule.
    budget_denomination_after: str = ""
    budget_denomination_after_block: int = 0
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
    # ── fork-anneal: finished-form duel checkpoints (DEC-CA-0029; digest-bound)
    # The deferred "D" of the wsd schedule (DEC-CA-0018's anticipated decay
    # branch). When > 0 (requires lr_schedule = "wsd"), every training run
    # FORKS at the end of its stable phase: the mid-stable weights + optimizer
    # state are retained in the checkpoint as ``weights_stable.safetensors`` +
    # ``optimizer.safetensors`` (the lineage branch — warm-starts resume from
    # THESE, never from a decayed endpoint), then training continues for
    # ``anneal_fraction × train_tokens`` more tokens under a cosine decay of
    # base_lr → 0, and the ANNEALED weights become ``weights.safetensors`` —
    # the artifact every scoring layer (duel eval, benchmark sidecar,
    # promotion BenchScores) loads. Identical for king and challenger, so the
    # controlled experiment is preserved; the wall-clock guard stretches by
    # the same fraction (total wall = max_train_seconds × (1 + fraction)).
    # 0.0 = off — the deployed default, absent from contract_digest via
    # drop-when-default, so shipping this field moves no deployed digest;
    # setting it is the deliberate contract cut (release-then-activate,
    # trainer + validators together, testnet first).
    anneal_fraction: float = 0.0
    # ── measured variance bundle (DEC-CA-0033; digest-bound) ─────────────────
    # Three knobs from the 2026-08-26 seed-variance measurement
    # (recorded in DEC-CA-0033); each ships inert via
    # drop-when-default, and setting any is a deliberate contract cut
    # (release-then-activate, trainer + validators together, testnet first).
    #
    # ema_decay > 0: maintain an exponential moving average of the weights
    # (per optimizer step) and ship the EMA as ``weights.safetensors`` — the
    # artifact every scoring layer loads — while the raw endpoint rides
    # beside it as ``weights_stable.safetensors`` (the lineage branch
    # warm-starts resume from; DEC-CA-0029's file convention, reused so the
    # loader needs no new case). Measured at 0.999 on heat-length runs:
    # shrinks entrant-specific generation-seed noise 4–11× and improves the
    # absolute geomean ~7% (12/12 runs). The cheap alternative to
    # fork-anneal for finished form — never arm both (the trainer refuses).
    ema_decay: float = 0.0
    # gen_seed_mix > 1: invoke the generator N times with N derived
    # generation seeds (each ~1/N of the token budget, series interleaved
    # round-robin), so an entrant's score averages N corpus realizations and
    # the residual seed-noise term drops ~√N. Nothing in the miner interface
    # changes — generators must already be pure functions of the passed seed
    # (sandbox contract). 1 = single invocation (deployed default).
    gen_seed_mix: int = 1
    # rewarmup_fraction > 0: WARM-STARTED wsd runs get a short linear LR
    # warmup over this fraction of train_tokens instead of hitting base_lr
    # on step 1 with fresh optimizer state — the measured first-step kick
    # costs +0.11 geomean at base_lr (the u158 probe, reproduced in-run).
    # The from-scratch run's warmup-once semantics are untouched.
    rewarmup_fraction: float = 0.0
    # ── Toto2-aligned optimizer constants (DEC-CA-0035; digest-bound) ────────
    # Datadog's Toto 2.0 recipe (arXiv 2605.20119) was HP-swept once at 4M
    # under u-µP; its DIMENSIONLESS constants transfer directly, its LR
    # VALUES do not (different parametrization convention — only the
    # matrix:AdamW ratio carries). Defaults below are today's deployed
    # behavior, so every knob is drop-when-default digest-inert. Measured
    # targets (DEC-CA-0035 — the full bundle
    # @ warm LR 5e-4 BEAT the converged warm-start init by 0.0038, the
    # campaign best): muon_momentum 0.96, muon_row_beta2 0.999, grad_clip
    # 7.0, adamw betas (0.91, 0.972), adamw_lr_scale 1/54 ≈ 0.0185, plus
    # weight_decay → 2e-8 (already a field above — no new knob needed).
    muon_momentum: float = 0.95    # Muon Nesterov momentum
    muon_row_beta2: float = 0.95   # NorMuon per-row second-moment EMA decay
    grad_clip: float = 1.0         # global grad-norm clip in the train loop
    adamw_beta1: float = 0.9       # AdamW group (embeddings/heads/biases)
    adamw_beta2: float = 0.999
    adamw_lr_scale: float = 1.0    # AdamW group LR = base_lr × this (Toto2
                                   # runs the matrix:AdamW split at 54:1)
    # warm_lr_scale < 1: WARM-STARTED runs train at base_lr × warm_lr_scale,
    # keyed off the same warm_started signal wsd's warmup-once already uses;
    # from-scratch (generation-start) runs keep full base_lr. Measured
    # target 0.125 (→ 5e-4): a converged init tolerates far less LR than a
    # random one, and no re-warmup trick substitutes (DEC-CA-0033 verdict).
    warm_lr_scale: float = 1.0
    # ── Cascade-MOE-28M recipe (DEC-CA-0050; digest-bound) ───────────────────
    # Only the ``base_arch = "cascade_moe"`` trainer reads these; each ships
    # inert via drop-when-default, so no deployed digest moves. The toto2
    # trainer REFUSES any of them armed (an armed knob it ignores would bump
    # contract_digest while changing no numerics — a contract lie).
    #
    # micro_batches > 1: split each optimizer step's batch into this many
    # sequential forward/backward passes (gradient accumulation) — the
    # spec batch (64 groups × 8 ch × 4096) needs ~56 GiB in one pass, over
    # the L40S's 48 GB. Numerically a mean over micro-batches.
    micro_batches: int = 1
    # µP width correction on the NorMuon LR: lr × (umup_base_d_model /
    # d_model) ** mup_lr_exponent. 0.0 = off; the MoE spec uses 1.42.
    mup_lr_exponent: float = 0.0
    # Power-function EMAs (Karras et al. 2024): relative widths, e.g.
    # (0.05, 0.10). Each is carried across warm-starts and saved as
    # ``ema_<width>.safetensors``; ``power_ema_eval`` picks the one shipped
    # as weights.safetensors (-1 = the raw weights). Routing bias is always
    # taken from the live model. Never arm with ema_decay.
    power_ema_sigma_rels: tuple[float, ...] = ()
    power_ema_eval: int = -1
    # What the power EMAs average over: "lineage" (t carried across warm
    # starts, so the window is σ_rel of the WHOLE lineage — the scored share of
    # the current leg shrinks as the lineage grows) or "leg" (restarted at
    # every warm start: the scored average covers only this leg's training;
    # the raw lineage weights and optimizer state are untouched).
    power_ema_scope: str = "lineage"
    # Release-gated continuation semantics; empty preserves historical digests.
    continuation_protocol: str = ""
    # Fork-anneal decay shape (DEC-CA-0029): "cosine" (deployed) | "linear"
    # | "1-sqrt" (the MoE spec's cooldown). The MoE trainer freezes the
    # routing bias for the second half of the anneal.
    anneal_shape: str = "cosine"
    # Copula energy-score loss: its weight is capped so its gradient at the
    # heads' shared input is <= copula_loss_cap × the marginal (pinball)
    # gradient norm. 0.0 = no copula loss (head untrained).
    copula_loss_cap: float = 0.0
    copula_samples: int = 8        # sample paths per group for the energy score
    copula_patches: int = 2        # trailing predicted patches scored jointly
    # ── generation surface (DEC-CA-0050; digest-bound, both archs) ───────────
    # corpus_epochs > 0 (requires corpus_mode = "cache_reuse"): the unique
    # corpus is exactly ceil(train_tokens / corpus_epochs) points, generated
    # BEFORE step 0 (the wall clock starts at the first batch) with every
    # gen_seed_mix child building concurrently, then cycled at most
    # corpus_epochs times. Lets a slow, expensive prior compete on quality:
    # ~4 epochs of repeated data are worth nearly as much as unique data
    # (Muennighoff et al., NeurIPS 2023). 0 = off (the generator's own
    # corpus_target_points, deployed behaviour).
    corpus_epochs: int = 0
    # ── slow-adapting router (DEC-CA-0050 v2; digest-bound) ──────────────────
    # Lane B hangs off the king's routing, so once the lineage's routing gate
    # has OPENED (the lane-B driver's one-way latch) lane A's router moves
    # slower: the router param group's LR × ``router_lr_scale`` and the
    # loss-free balancing-bias step × ``route_bias_rate_scale``. Read only by
    # the cascade_moe trainer, and only on a leg flagged ``--router-slow``
    # (every leg of an era whose seed block the latch precedes — one rule for
    # king and challengers); before the gate opens the router trains at full
    # rate. Expert legs freeze router + bias regardless. 1.0 = off.
    router_lr_scale: float = 1.0
    route_bias_rate_scale: float = 1.0
    # DEC-CA-0053: what the trainer does with a series longer than
    # context_length. "last" keeps the final window (every deployed config);
    # "crop" cuts it into seeded random context-length windows so a long
    # stored trajectory is not discarded. Digest-bound, drop-when-default.
    long_series: str = "last"
    # DEC-CA-0053 scheduled context growth (the DEC-CA-0047 pattern): from
    # chain block `context_length_after_block` the effective context_length is
    # `context_length_after`. NEVER part of contract_digest; the effective
    # contract is `at_block(block)`. 0 / 0 = no schedule.
    context_length_after: int = 0
    context_length_after_block: int = 0
    # RUNTIME ONLY (never read from chain.toml, never in contract_digest): the
    # per-leg ``--router-slow`` flag the worker sets on the leg's resolved
    # contract — the two knobs above bite only when it is True.
    router_slow: bool = False
    # ── v2 activation (DEC-CA-0050; NEVER part of contract_digest) ───────────
    # ``[v2] from_block`` and the post-v2 contract (``[v2.training]``): an era
    # whose start block is ``>= v2_from_block`` trains under ``v2_contract``
    # (Cascade-MOE-28M); every earlier era under this contract, byte-for-byte.
    # Resolved by :meth:`at_block` at the ERA START (the ``--contract-block``
    # every leg of an era carries), like the DEC-CA-0047 schedule. 0 / None =
    # no v2 (the base contract forever).
    v2_from_block: int = 0
    v2_contract: TrainingContractConfig | None = None

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

    def v2_active(self, block: int | None) -> bool:
        """Whether the era starting at ``block`` (or containing it — the v2
        block is an era boundary) trains under the v2 contract."""
        gate = int(self.v2_from_block or 0)
        return (gate > 0 and self.v2_contract is not None and block is not None
                and int(block) >= gate)

    def at_block(self, block: int | None) -> TrainingContractConfig:
        """The contract EFFECTIVE at chain ``block`` (pass an era's START
        block): from ``v2_from_block`` the v2 contract (DEC-CA-0050), else
        this contract with the DEC-CA-0047 schedule applied —
        ``budget_denomination`` swapped to ``budget_denomination_after`` once
        ``block >= budget_denomination_after_block``; the schedule fields are
        cleared on the result. The DEC-CA-0053 context schedule applies the
        same way (``context_length`` → ``context_length_after``). ``None`` or
        no schedule ⇒ ``self``."""
        if self.v2_active(block):
            return self.v2_contract.at_block(block)  # type: ignore[union-attr]
        out = self
        after = str(self.budget_denomination_after or "")
        gate = int(self.budget_denomination_after_block or 0)
        if after and gate > 0 and block is not None and int(block) >= gate:
            out = replace(out, budget_denomination=after,
                          budget_denomination_after="", budget_denomination_after_block=0)
        ctx = int(self.context_length_after or 0)
        cgate = int(self.context_length_after_block or 0)
        if ctx > 0 and cgate > 0 and block is not None and int(block) >= cgate:
            out = replace(out, context_length=ctx,
                          context_length_after=0, context_length_after_block=0)
        return out

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



def parse_training_contract(t: dict) -> TrainingContractConfig:
    """Parse one ``[training]``-shaped table (the live ``[training]`` or the
    post-v2 ``[v2.training]``) into a :class:`TrainingContractConfig`. Extra
    sizes are the table's own ``[[…sizes]]`` array."""
    for variant in (t, *t.get("sizes", [])):
        if (variant.get("arch_preset") in UNIMPLEMENTED_ARCH_PRESETS
                or "implementation_status" in variant or "target_active_parameters" in variant):
            raise ValueError(
                f"{variant.get('arch_preset', 'model variant')!r} is a model placeholder, "
                "not a training contract. Populate and qualify the model, source digest, "
                "recipe and measured budget, release its reserved architecture ID, "
                "then remove implementation_status and "
                "target_active_parameters before activation.")
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
    return TrainingContractConfig(
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
        batch_denomination=validate_batch_denomination(
            str(t.get("batch_denomination", "series"))),
        budget_denomination=validate_budget_denomination(
            str(t.get("budget_denomination", "points"))),
        budget_denomination_after=(
            validate_budget_denomination(str(t["budget_denomination_after"]))
            if str(t.get("budget_denomination_after", "") or "") else ""),
        budget_denomination_after_block=int(t.get("budget_denomination_after_block", 0) or 0),
        accepted_fields=validate_accepted_fields(t.get("accepted_fields", ())),
        allow_future_known=bool(t.get("allow_future_known", False)),
        real_corpus_ref=validate_real_corpus_ref(t.get("real_corpus_ref", "")),
        anneal_fraction=float(t.get("anneal_fraction", 0.0)),
        # DEC-CA-0033 knobs (parsing added with DEC-CA-0035 — the fields
        # landed with dataclass defaults only, so a TOML arming would
        # have silently no-oped)
        ema_decay=float(t.get("ema_decay", 0.0)),
        gen_seed_mix=int(t.get("gen_seed_mix", 1)),
        rewarmup_fraction=float(t.get("rewarmup_fraction", 0.0)),
        # DEC-CA-0035 Toto2-aligned constants
        muon_momentum=float(t.get("muon_momentum", 0.95)),
        muon_row_beta2=float(t.get("muon_row_beta2", 0.95)),
        grad_clip=float(t.get("grad_clip", 1.0)),
        adamw_beta1=float(t.get("adamw_beta1", 0.9)),
        adamw_beta2=float(t.get("adamw_beta2", 0.999)),
        adamw_lr_scale=float(t.get("adamw_lr_scale", 1.0)),
        warm_lr_scale=float(t.get("warm_lr_scale", 1.0)),
        # DEC-CA-0050 Cascade-MOE-28M recipe
        micro_batches=int(t.get("micro_batches", 1)),
        mup_lr_exponent=float(t.get("mup_lr_exponent", 0.0)),
        power_ema_sigma_rels=tuple(float(v) for v in t.get("power_ema_sigma_rels", ())),
        power_ema_eval=int(t.get("power_ema_eval", -1)),
        power_ema_scope=_power_ema_scope(t.get("power_ema_scope", "lineage")),
        continuation_protocol=_continuation_protocol(t.get("continuation_protocol", "")),
        anneal_shape=validate_anneal_shape(str(t.get("anneal_shape", "cosine"))),
        copula_loss_cap=float(t.get("copula_loss_cap", 0.0)),
        copula_samples=int(t.get("copula_samples", 8)),
        copula_patches=int(t.get("copula_patches", 2)),
        corpus_epochs=validate_corpus_epochs(
            int(t.get("corpus_epochs", 0)), str(t.get("corpus_mode", "stream_cpu"))),
        router_lr_scale=float(t.get("router_lr_scale", 1.0)),
        route_bias_rate_scale=float(t.get("route_bias_rate_scale", 1.0)),
        long_series=validate_long_series(str(t.get("long_series", "last"))),
        context_length_after=validate_context_after(
            int(t.get("context_length_after", 0) or 0), int(t["context_length"]),
            int(t["patch_size"])),
        context_length_after_block=int(t.get("context_length_after_block", 0) or 0),
    )


def _power_ema_scope(value: object) -> str:
    v = str(value)
    if v not in ("lineage", "leg"):
        raise ValueError(f"[training] power_ema_scope must be 'lineage' or 'leg'; got {value!r}")
    return v



def _continuation_protocol(value):
    if value not in {"", "cascade-continuation/1"}:
        raise ValueError("unsupported training continuation protocol")
    return value
