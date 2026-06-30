"""Config loader for chain.toml — single source of truth for subnet config.

Miners, the trainer, and validators all load from here. The schema is
versioned; a file newer than this code warns and proceeds (operator-controlled
file, deployed by hand alongside the binaries — the same policy horizon uses).
"""

from __future__ import annotations

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
    """

    corpus_n_series: int
    min_length: int
    max_length: int
    max_total_points: int
    max_generate_seconds: int
    max_memory_mb: int
    max_repo_mb: int = 128  # cap on fetched submission bytes (code-only; no shipped weights)
    max_channels: int = 1


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
    # Pinned GPU model for byte-exact re-derivation. When non-empty, the validator
    # asserts every trained entry's recorded gpu_name == this (king and challenger
    # ran the same SKU); empty ⇒ require only that king and challenger match each
    # other when both report a gpu_name. Folded into contract_digest.
    expected_gpu: str = ""
    # Extra model sizes trained alongside the base (primary) size in the final
    # stage. Empty ⇒ single-size rounds (the legacy behaviour). Folded into
    # contract_digest, so a validator's contract gate covers every size at once.
    extra_sizes: tuple[SizeSpec, ...] = ()

    def tokens_for_hours(self, hours: float) -> int:
        """Point-pass budget for ``hours`` on the reference GPU at this size's
        throughput. Used for both the full final budget and the cheaper heat
        budget; going through a token count (not a raw timer) keeps king and
        challenger on identical compute and keeps a re-derived run reproducible."""
        return int(round(hours * 3600.0 * self.ref_throughput_tokens_per_s))

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
        ``contract_digest`` is the stable identity of that one size."""
        return replace(
            self,
            arch_preset=spec.arch_preset,
            base_arch_digest=spec.base_arch_digest,
            d_model=spec.d_model,
            num_layers=spec.num_layers,
            num_heads=spec.num_heads,
            mlp_expansion=spec.mlp_expansion,
            ref_throughput_tokens_per_s=spec.ref_throughput_tokens_per_s,
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
    heat_train_hours: float = 0.5     # cheap screening budget per competitor
    heat_n_windows: int = 256         # eval windows the heat screens on (≤ [eval] n_windows)
    finalists: int = 1                # challengers promoted from the heat to the final
    screen_size: str = ""             # arch_preset the heat screens at ("" ⇒ primary)
    throne_sizes: tuple[str, ...] = ()  # arch_presets the final trains/judges at (() ⇒ [primary])


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


@dataclass(frozen=True)
class ScoringConfig:
    win_margin_start: float
    win_margin_end: float
    margin_warmup_rounds: int
    min_windows: int
    bootstrap_B: int
    bootstrap_alpha: float
    dethrone_cp: int
    # Reward routing: equal weight is split across the current king plus up to
    # ``reward_prior_kings`` previous distinct kings still registered; with none
    # registered, all weight burns to ``burn_uid``. ``reward_prior_kings = 0``
    # reproduces pure winner-take-all. Defaults keep older chain.toml loadable.
    reward_prior_kings: int = 0
    burn_uid: int = 0


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


@dataclass(frozen=True)
class ManifestConfig:
    """Where the trainer publishes training receipts and the validator reads
    them. Manifests live in the ``[storage] manifest_bucket`` S3 bucket;
    ``trainer_hotkey`` is the only hotkey whose manifest a validator trusts."""

    trainer_hotkey: str
    poll_seconds: int


@dataclass(frozen=True)
class ValidatorConfig:
    weight_set_interval_blocks: int
    poll_seconds: int
    hf_cache_seconds: int
    state_db_path: str


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
        )


class LaunchConfigError(RuntimeError):
    """chain.toml still carries launch placeholders that must be set."""


_PLACEHOLDER_DIGEST = "0" * 64


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
            extra_sizes=extra_sizes,
        ),
        round=RoundConfig(
            epoch_blocks=int(r.get("epoch_blocks", 7200)),
            round_hours=float(r.get("round_hours", 24.0)),
            heat_train_hours=float(r.get("heat_train_hours", 0.5)),
            heat_n_windows=int(r.get("heat_n_windows", 256)),
            finalists=int(r.get("finalists", 1)),
            screen_size=str(r.get("screen_size", "")),
            throne_sizes=tuple(str(x) for x in r.get("throne_sizes", ())),
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
        ),
        scoring=ScoringConfig(
            win_margin_start=float(s["win_margin_start"]),
            win_margin_end=float(s["win_margin_end"]),
            margin_warmup_rounds=int(s["margin_warmup_rounds"]),
            min_windows=int(s["min_windows"]),
            bootstrap_B=int(s["bootstrap_B"]),
            bootstrap_alpha=float(s["bootstrap_alpha"]),
            dethrone_cp=int(s["dethrone_cp"]),
            reward_prior_kings=int(s.get("reward_prior_kings", 0)),
            burn_uid=int(s.get("burn_uid", 0)),
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
        ),
        manifest=ManifestConfig(
            trainer_hotkey=str(m["trainer_hotkey"]),
            poll_seconds=int(m["poll_seconds"]),
        ),
        validator=ValidatorConfig(
            weight_set_interval_blocks=int(v["weight_set_interval_blocks"]),
            poll_seconds=int(v["poll_seconds"]),
            hf_cache_seconds=int(v["hf_cache_seconds"]),
            state_db_path=str(v["state_db_path"]),
        ),
        raw=raw,
    )
