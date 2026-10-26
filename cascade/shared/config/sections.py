"""Small section dataclasses: subnet, generator, deps, guards, storage, telemetry, validator."""

from __future__ import annotations

from dataclasses import dataclass


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

    ``corpus_target_points`` (DEC-CA-0031) re-denominates the MATERIALISED
    drain (``cache_reuse`` / ``cascade verify``) in points instead of series:
    when > 0, the drain stops at the first series that reaches this many
    cumulative values-points and the series COUNT is free — a generator may
    spend the same corpus as 65k×1024 instead of 16k×4096, which is what makes
    compute-heavy priors (GP/kernel draws scale ~cubically in length)
    affordable inside ``max_generate_seconds``. 0 (default) keeps the legacy
    exactly-``corpus_n_series`` drain. Inert for the streaming feed modes,
    which already stop at the round's token budget. Never part of
    ``contract_digest`` (no [generator] key is).

    ``stream_stall_seconds`` is the STREAMING per-frame stall window — how
    long the trainer waits between series before killing the child — split
    out from ``max_generate_seconds`` so the batch drain budget can be raised
    without letting a stalled generator idle a heat slot for the same, larger
    window (the two roles were one knob before DEC-CA-0031). 0 (default)
    falls back to ``max_generate_seconds``, byte-identical to the old
    behaviour for configs without the key.

    ``min_channels`` / ``max_channels`` bound every emitted series. V2 presets
    require 8–32 channels. Missing keys preserve the historical one-channel
    floor; pre-v2 training/replay resolves that legacy floor explicitly.

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
    min_channels: int = 1
    # Points-denominated materialised drain + decoupled streaming stall window
    # (DEC-CA-0031; see the class docstring). Both default inert.
    corpus_target_points: int = 0
    stream_stall_seconds: int = 0
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
    #                        "roles", "groups"); () = values-only (historical default).
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

    def __post_init__(self):
        if not 1 <= self.min_channels <= self.max_channels:
            raise ValueError("[generator] requires 1 <= min_channels <= max_channels")

    @property
    def effective_stall_seconds(self) -> int:
        """The streaming per-frame stall window: ``stream_stall_seconds`` when
        set, else ``max_generate_seconds`` (the pre-DEC-CA-0031 single-knob
        behaviour). Every streaming consumer goes through here so the two
        knobs can never drift apart at a call site."""
        return int(self.stream_stall_seconds) or int(self.max_generate_seconds)



@dataclass(frozen=True)
class DependencyConfig:
    max_packages: int
    allowed: tuple[str, ...]


@dataclass(frozen=True)
class StaticGuardConfig:
    blocked: tuple[str, ...]
    # "scan" (legacy): Python packed into string constants is unpacked and its
    # imports scanned. "reject": such a constant is itself a rejection at
    # ADMISSION (rolling intake) and in `cascade verify` — every module must be
    # a .py file in the tree so the code that runs is the code dedup sees. The
    # per-leg sandbox preflight keeps "scan" whatever this says, so a seated
    # king's own legs are never affected (rules apply at the door only).
    packed_sources: str = "scan"
    # "reject" applies to admissions at chain block >= this (0 = immediately).
    packed_sources_from_block: int = 0

    def packed_sources_at(self, block: int | None) -> str:
        """Effective mode at ``block``: ``"reject"`` only once the gate is reached."""
        if self.packed_sources != "reject":
            return self.packed_sources
        gate = int(self.packed_sources_from_block or 0)
        if gate > 0 and (block is None or int(block) < gate):
            return "scan"
        return "reject"


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
    # Name of an env var holding a SEPARATE wandb key for ISOLATED (payer-paid)
    # pods, forwarded to them as WANDB_API_KEY so funded legs stream live too.
    # The payer has console access to their pod and can read it — so it must be
    # a key scoped to the public project, never the operator's own key (a value
    # of "WANDB_API_KEY" is refused). Empty ⇒ funded legs keep no wandb.
    funded_key_env: str = ""
    # OPTIONAL validator-side settlement summary (lane-A verdict, lane-B/C
    # verdict counts and payouts, emission split, ladder, king check), logged
    # AFTER the receipt publishes. Off by default; independent of ``enabled``
    # (the trainer mirror). Validators never need W&B: no key / no package ⇒
    # a silent no-op, and it can never affect a verdict or the weights.
    validator_summary: bool = False


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

    ``bench_anneal_fraction`` (DEC-CA-0030) arms bench-anneal: before the
    post-round benchmark sweep, each duel checkpoint is resumed on the pod
    that trained it (weights + optimizer state, fresh salted corpus) under a
    pure cosine decay base_lr → 0 for this fraction of the full token budget,
    and the sweep scores the ANNEALED copy — so the signed BenchScores (the
    public bench stream, promotion picks, the DEC-CA-0017 guard) read
    finished-form numbers instead of wsd's mid-stable artifacts. The
    canonical duel checkpoint, the manifest, the bench-report wire format,
    and every validator-verified byte are untouched — validators keep
    consuming the same signed six numbers; only what the trainer benches
    changes. Any anneal-leg failure falls back to benching the raw
    checkpoint (a mid-stable number beats a missing one). ``0.0`` = off.
    Deliberately a ``[telemetry]`` key: it must never touch
    ``contract_digest`` — the digest-bound path to finished-form DUEL
    artifacts is [training] anneal_fraction (DEC-CA-0029), a contract cut.

    ``funded_bench`` (DEC-CA-0036) benches each funded challenger on its
    OWN payer pod after the duel (the pod stays up through publish and is
    torn down when its sweep ends) instead of not at all — a payer-billed
    box the operator never benched on, so its numbers are a FILTER, never a
    published fact: the top ``funded_bench_verify_top`` payer-reported
    challengers are re-benched on the operator's king pod, and only the
    operator's numbers reach the signed report (and so the promotion pool).
    A payer number more than ``funded_bench_verify_tolerance`` (relative,
    on the six-number cascade score) better than the operator's re-bench is
    a forged sweep: the entry is dropped from the report (its submission was
    already spent at the duel settle; the promotion seat is what the drop
    denies). Payer numbers outside the verified top-N are logged only.
    """

    host_probe: bool = True
    host_bench: bool = True
    bench_anneal_fraction: float = 0.0
    funded_bench: bool = True
    funded_bench_verify_top: int = 1
    funded_bench_verify_tolerance: float = 0.02


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
    # The trainer stamps ``prev_manifest_sha256`` (the hash link validators
    # check) on settlements at/after this boundary. It is part of the signed
    # body, so set it only once every validator runs a release that knows the
    # field. 0 = never stamped.
    hash_link_from_block: int = 0


@dataclass(frozen=True)
class ValidatorConfig:
    weight_set_interval_blocks: int
    poll_seconds: int
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
class ActivationConfig:
    """Stake-weighted activation of the DEC-CA-0043 rollover (DEC-CA-0045).

    Each upgraded validator writes a plain on-chain commitment
    ``cascade-ready:1:<feature>:…`` from its hotkey. At every boundary of
    the grid in force, every node adds up the stake of the permit-holding
    validators that have signalled ``feature``; the first boundary where
    the signed share reaches ``threshold`` LOCKS IN, and the rollover is
    the next boundary after it. Lock-in is one-way: a node that has seen it
    persists the block and never re-evaluates. The typed-in DEC-CA-0043
    keys (``rolling_from_block`` etc.) always win over the resolved block
    — the owner override. ``feature = ""`` disables signalling and
    resolution entirely (the typed-in keys are then the only way to arm).
    """

    feature: str = ""
    # Fraction of ELIGIBLE validator stake (permit holders; see
    # ``dormant_after_blocks``) that must have signalled for lock-in.
    threshold: float = 0.51
    # The grid from the resolved rollover (``epoch_blocks`` after it, the
    # loaded ``epoch_blocks`` becoming ``epoch_blocks_prev``). 0 = keep the
    # loaded grid (the rollover still switches, prev == after).
    epoch_blocks_after: int = 0
    # A permit holder whose ``last_update`` (last weight-set) is older than
    # this many blocks at the tally block is left OUT of the eligible stake,
    # so a dead validator's stake cannot hold the number down. 0 = count
    # every permit holder.
    dormant_after_blocks: int = 0
    # RUNTIME ONLY (never read from chain.toml): the rollover block
    # ``apply_activation`` wrote into this config's DEC-CA-0043 keys, so a
    # resolved rollover is distinguishable from a typed-in one (the typed
    # one is the owner override and is never re-resolved).
    resolved_block: int = 0
    # RUNTIME ONLY: the forfeiture block ``apply_forfeit_activation`` wrote
    # into ``[scoring] forfeit_from_block`` (DEC-CA-0048 decided on chain),
    # so a resolved block is distinguishable from a typed-in one.
    resolved_forfeit_block: int = 0
    # RUNTIME ONLY: the margin-v2 block ``apply_margin_v2_activation`` wrote
    # into ``[scoring] margin_v2_from_block`` (DEC-CA-0049 decided on chain),
    # so a resolved block is distinguishable from a typed-in one.
    resolved_margin_v2_block: int = 0
    # RUNTIME ONLY: the pre-launch burn block ``apply_prelaunch_burn_activation``
    # wrote into ``[v2] prelaunch_burn_from_block`` (DEC-CA-0055 decided on
    # chain), so a resolved block is distinguishable from a typed-in one.
    resolved_prelaunch_burn_block: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.feature)


@dataclass(frozen=True)
class NetConfigConfig:
    """The owner-signed network config (DEC-CA-0057, :mod:`cascade.shared.netconfig`).

    ``owner_hotkey`` is the ONE key whose signed overlay a validator (and the
    trainer, read-only) will adopt: the owner's MAIN validator hotkey — the
    same key that signs the results satellites replay (DEC-CA-0058). Empty
    ⇒ completely inert: nothing is fetched, signalled or applied and every
    config, note and receipt is byte-identical to a release without it.
    ``poll_seconds`` is how often ``config/network/latest.json`` is re-read.

    ``seq`` / ``digest`` / ``applied_block`` are RUNTIME ONLY (never read from
    ``chain.toml``): set on a config built by ``netconfig.build_config`` so the
    receipt can stamp the overlay a settlement was judged under (0 / "" on
    the base config — the receipt then omits the fields)."""

    owner_hotkey: str = ""
    network_id: str = "bittensor:finney"
    poll_seconds: int = 300
    seq: int = 0
    digest: str = ""
    applied_block: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.owner_hotkey)


def parse_netconfig(raw: object, *, activation_feature: str = "") -> NetConfigConfig:
    """``[netconfig]`` → :class:`NetConfigConfig`. Absent / empty table = inert.
    Refuses unknown keys (a typo here would silently leave a fleet unable to
    adopt), and an owner key without ``[activation] feature``: adoption is a
    DEC-CA-0045 note segment, and a node that does not signal cannot count."""
    if raw is None:
        return NetConfigConfig()
    if not isinstance(raw, dict):
        raise ValueError("[netconfig] must be a table")
    unknown = set(raw) - {"owner_hotkey", "poll_seconds", "network_id"}
    if unknown:
        raise ValueError(f"[netconfig] unknown keys {sorted(unknown)} (owner_hotkey, poll_seconds)")
    owner = str(raw.get("owner_hotkey", "") or "").strip()
    poll = raw.get("poll_seconds", 300)
    if isinstance(poll, bool) or not isinstance(poll, int) or poll < 10:
        raise ValueError(f"[netconfig] poll_seconds must be an integer >= 10; got {poll!r}")
    if owner and not str(activation_feature or "").strip():
        raise ValueError("[netconfig] owner_hotkey is set but [activation] feature is empty: "
                         "an overlay is adopted by a DEC-CA-0045 note segment, which needs "
                         "the activation note")
    network = str(raw.get("network_id", "bittensor:finney")).strip()
    if not network:
        raise ValueError("[netconfig] network_id must name the chain deployment")
    return NetConfigConfig(owner_hotkey=owner, poll_seconds=int(poll), network_id=network)
