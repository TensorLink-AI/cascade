"""Config loader for chain.toml — single source of truth for subnet config.

Miners, the trainer, and validators all load from here. The schema is
versioned; a file newer than this code warns and proceeds (operator-controlled
file, deployed by hand alongside the binaries — the same policy horizon uses).
"""

from __future__ import annotations

import sys
import tomllib  # py311+
from dataclasses import dataclass, field
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

    @property
    def train_tokens(self) -> int:
        """Enforced training budget in point-passes: ``target_train_hours`` of the
        reference GPU at ``ref_throughput_tokens_per_s``. King and challenger both
        train to this exact count — fair (equal compute, not equal wall-clock,
        which data-dependent throughput could skew) and reproducible (a re-derived
        run matches)."""
        return int(round(self.target_train_hours * 3600.0 * self.ref_throughput_tokens_per_s))

    @property
    def warmup_tokens(self) -> int:
        return int(round(self.train_tokens * self.warmup_fraction))


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
class QueueConfig:
    """Trainer-side submission queue (see :mod:`cascade.trainer.queue`).

    The trainer enqueues challenger generators discovered on-chain and trains
    them FIFO, skipping any whose generator CID equals the reigning king's (an
    exact copy of the king) or that was already trained this reign.

    Attributes:
        state_db_path: where the backlog + per-reign trained cache is persisted
            so it survives trainer restarts.
        trained_cache_size: ring-buffer cap on already-trained CIDs per reign.
    """

    state_db_path: str
    trained_cache_size: int


@dataclass(frozen=True)
class ChainConfig:
    schema_version: int
    subnet: SubnetConfig
    generator: GeneratorConfig
    training: TrainingContractConfig
    eval: EvalConfig
    scoring: ScoringConfig
    dependencies: DependencyConfig
    static_guard: StaticGuardConfig
    storage: StorageConfig
    manifest: ManifestConfig
    validator: ValidatorConfig
    queue: QueueConfig
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def netuid(self) -> int:
        return self.subnet.netuid

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
    if not cfg.manifest.trainer_hotkey:
        problems.append("[manifest] trainer_hotkey is empty (set the owner trainer ss58 hotkey)")
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
    q = raw.get("queue", {})

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
        queue=QueueConfig(
            state_db_path=str(q.get("state_db_path", "trainer_queue.db")),
            trained_cache_size=int(q.get("trained_cache_size", 256)),
        ),
        raw=raw,
    )
