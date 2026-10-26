"""The ChainConfig aggregate and the launch-readiness checks over it."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any

from .generation import GenerationConfig
from .lane_c import LaneCConfig
from .lane_d import LaneDConfig
from .marketcast import MarketCastConfig
from .rounds import RoundConfig, effective_epoch_blocks
from .scoring import (
    EvalConfig,
    ScoringConfig,
    effective_margin_mode,
    effective_win_margin_start,
    margin_v2_active,
    mv_score_active,
)
from .sections import (
    ActivationConfig,
    DependencyConfig,
    GeneratorConfig,
    ManifestConfig,
    NetConfigConfig,
    StaticGuardConfig,
    StorageConfig,
    SubnetConfig,
    TelemetryConfig,
    ValidatorConfig,
    WandbConfig,
)
from .training import TrainingContractConfig
from .v2 import LaneBConfig, V2Config
from .validators import funded_sku_wall_for


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
    activation: ActivationConfig = field(default_factory=ActivationConfig)
    lane_b: LaneBConfig = field(default_factory=LaneBConfig)
    v2: V2Config = field(default_factory=V2Config)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    # DEC-CA-0056 fine-tuning lane (inert until [lane_c] from_block is typed)
    lane_c: LaneCConfig = field(default_factory=LaneCConfig)
    # DEC-CA-0062 distillation / real-data lane (wired, not implemented)
    lane_d: LaneDConfig = field(default_factory=LaneDConfig)
    # DEC-CA-0064 MarketCast lane-C fine-tune eval (built; off at from_block = 0)
    marketcast: MarketCastConfig = field(default_factory=MarketCastConfig)
    # DEC-CA-0057 owner-signed network config (inert while owner_hotkey is "")
    netconfig: NetConfigConfig = field(default_factory=NetConfigConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def netuid(self) -> int:
        return self.subnet.netuid

    def screen_contract(self) -> TrainingContractConfig:
        """The single-size contract the heat screens at — ``[round] screen_size``
        resolved against the size registry, defaulting to the primary size."""
        name = self.round.screen_size or self.training.arch_preset
        return self.training.contract_for(name)

    def generator_for_contract(self, contract) -> GeneratorConfig:
        """Apply the v2 width floor without rejecting archived v1 corpora.

        Old configs omit min_channels and retain one channel. The checked-in
        v2 policy raises it; the separately pinned pre-v2 model keeps its
        original admission range when replayed or trained during transition.
        """
        gen = replace(self.generator, accepted_fields=contract.accepted_fields)
        if "groups" in contract.accepted_fields:
            gen = replace(gen, interface_version=2)
        if self.training.v2_contract is not None and contract.base_arch == self.training.base_arch:
            gen = replace(gen, min_channels=1)
        if contract.corpus_mode == "shards":
            gen = replace(gen, max_length=self.generation.max_length,
                          max_memory_mb=self.generation.child_memory_mb,
                          max_generate_seconds=self.generation.window_seconds)
        return gen

    def throne_contracts(self) -> list[TrainingContractConfig]:
        """The single-size contracts the final trains + the throne is judged at —
        ``[round] throne_sizes`` resolved against the registry, defaulting to just
        the primary size. One element ⇒ a single-size throne; several ⇒ the
        combined-score throne pools across them."""
        names = self.round.throne_sizes or (self.training.arch_preset,)
        return [self.training.contract_for(n) for n in names]

    def throne_contracts_at(self, block: int | None) -> list[TrainingContractConfig]:
        """:meth:`throne_contracts` under the contract effective at ``block``
        (an era's start block — DEC-CA-0047 scheduled switch)."""
        if self.training.v2_active(block):
            # DEC-CA-0050: the v2 contract is single-size (load-checked)
            return [self.training.at_block(block).primary_size]
        names = self.round.throne_sizes or (self.training.arch_preset,)
        base = self.training.at_block(block)
        return [base.contract_for(n) for n in names]

    def eval_windows_at(self, block: int | None, *, settlement: bool = True) -> int:
        """Private verdict budget under this round's own era, including replay."""
        from ..era import v2_settlement_active

        active = v2_settlement_active(self, block) if settlement else self.v2.active(block)
        return (int(self.v2.eval_n_windows) if active and self.v2.eval_n_windows
                else int(self.eval.n_windows))

    def koth_params(self, block: int | None = None) -> Any:
        """Build a :class:`cascade.eval.koth.KothParams` from ``[scoring]``
        for the round at epoch boundary ``block`` (``None`` = the steady-state
        values; a scheduled ``win_margin_start`` change resolves per round —
        see :func:`effective_win_margin_start`).

        Imported lazily so :mod:`cascade.shared.config` stays free of the
        eval package at import time.
        """
        from ...eval.koth import KothParams
        from ..era import effective_margin_warmup_rounds, v2_settlement_active

        v2 = getattr(self, "v2", None)
        if v2 is not None and v2.margin_set() and v2_settlement_active(self, block):
            # The v2-era bar ([v2] win_margin_*), counted in blocks like
            # DEC-CA-0049's on the grid in force at this round's block.
            start = float(v2.win_margin_start)
            end = float(v2.win_margin_end)
            warmup = max(1, int(v2.margin_warmup_blocks)
                         // max(1, int(effective_epoch_blocks(self.round, int(block)))))
        elif margin_v2_active(self.scoring, block):
            # DEC-CA-0049: the v2 bar, its decay counted in blocks on the grid
            # in force at this round's block.
            start = float(self.scoring.win_margin_start_v2)
            end = float(self.scoring.win_margin_end_v2)
            warmup = max(1, int(self.scoring.margin_warmup_blocks_v2)
                         // max(1, int(effective_epoch_blocks(self.round, int(block)))))
        else:
            start = effective_win_margin_start(self.scoring, block)
            end = self.scoring.win_margin_end
            warmup = effective_margin_warmup_rounds(self.round, self.scoring, block)
        from ..era import bench_gate_mode_at as _bench_gate_mode_at
        from ..era import bench_gate_rule_at as _bench_gate_rule_at

        return KothParams(
            win_margin_start=start,
            win_margin_end=end,
            margin_warmup_rounds=warmup,
            min_windows=self.scoring.min_windows,
            bootstrap_B=self.scoring.bootstrap_B,
            bootstrap_alpha=self.scoring.bootstrap_alpha,
            dethrone_cp=self.scoring.dethrone_cp,
            min_clusters=self.scoring.min_clusters,
            gift_gate_mode=self.scoring.gift_gate_mode,
            gift_gate_tolerance=self.scoring.gift_gate_tolerance,
            gift_gate_min_configs=self.scoring.gift_gate_min_configs,
            # The settlement's EFFECTIVE mode (the v2 go-live can switch it
            # on), so a receipt records the gate that actually judged it.
            bench_gate_mode=(_bench_gate_mode_at(self, block) if block is not None
                             else self.scoring.bench_gate_mode),
            bench_gate_min_items=self.scoring.bench_gate_min_items,
            # DEC-CA-0061: the settlement's effective bench-gate rule and
            # rule v2's thresholds.
            bench_gate_rule=_bench_gate_rule_at(self, block),
            bench_gate_tolerance=self.scoring.bench_gate_tolerance,
            bench_gate_min_win_rate=self.scoring.bench_gate_min_win_rate,
            margin_mode=effective_margin_mode(self.scoring, block),
            margin_increment_floor=self.scoring.margin_increment_floor,
            init_gate_mode=self.scoring.init_gate_mode,
            init_gate_tolerance=self.scoring.init_gate_tolerance,
            mv_score=mv_score_active(self.scoring, block),
        )

    def king_check_params(self, block: int | None) -> Any:
        """The DEC-CA-0055 king check's pool params for the settlement at
        ``block``: :meth:`koth_params` in increment units with the init floor
        off, and — when ``[v2] king_check_margin`` is set — that FIXED bar
        instead of the tenure-decayed dethrone bar. The validator and every
        replay (audit, satellite) call this one definition."""
        from dataclasses import replace as _replace

        p = _replace(self.koth_params(block=block), margin_mode="increment",
                     init_gate_mode="off")
        m = float(getattr(getattr(self, "v2", None), "king_check_margin", 0.0) or 0.0)
        return _replace(p, win_margin_start=m, win_margin_end=m) if m > 0.0 else p

    def judged_level_params(self, params: Any, block: int | None) -> Any:
        """``params`` judged in LEVEL units for the round at ``block``: the
        margin mode set to "level" and the margin schedule set to the pre-v2
        level bar (``effective_win_margin_start`` / ``win_margin_end`` /
        ``effective_margin_warmup_rounds``). DEC-CA-0049's v2 bar is priced in
        INCREMENT units; a round that falls back to level (no init baseline:
        a random-init round or a multi-size duel) must never be judged at a
        0.4 LEVEL bar (40 % of the absolute score = undethroneable). Every
        other field is kept. Bit-identical to ``replace(params,
        margin_mode="level")`` while the v2 bar is not active. The validator's
        in-round fallback and the audit replay both call this."""
        from dataclasses import replace as _replace

        from ..era import effective_margin_warmup_rounds

        return _replace(
            params,
            margin_mode="level",
            win_margin_start=effective_win_margin_start(self.scoring, block),
            win_margin_end=self.scoring.win_margin_end,
            margin_warmup_rounds=effective_margin_warmup_rounds(
                self.round, self.scoring, block),
        )



class LaunchConfigError(RuntimeError):
    """chain.toml still carries launch placeholders that must be set."""


_PLACEHOLDER_DIGEST = "0" * 64



# Bittensor's block time. A FIXED chain property — never derive it from
# ``round_hours / epoch_blocks``: the DEC-CA-0045 rollover moves the grid
# 3600 -> 900 blocks without changing the clock, so that ratio reads 48 s/block.
BLOCK_SECONDS = 12.0


def funded_sku_wall_fit(cfg: ChainConfig) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split the configured funded SKUs (``funded_pod_skus``, else
    ``funded_pod_sku``) into (fitting, excluded) by the per-SKU wall
    (``funded_sku_wall_seconds``, else the contract's ``max_train_seconds``)
    against ONE epoch's wall clock. An excluded SKU can never rent: its latest
    safe start is before the epoch begins. The shipped (mainnet-measured) wall
    table is inherited by any chain.toml that does not override it — on a
    2 h testnet grid that silently narrowed rents to the SKUs WITHOUT an
    entry, read as "works" (2026-09-21 review)."""
    rnd = cfg.round
    skus = tuple(rnd.funded_pod_skus) or ((rnd.funded_pod_sku,) if rnd.funded_pod_sku else ())
    if not skus:
        return (), ()
    cap = max(int(c.max_train_seconds) for c in cfg.throne_contracts())
    epoch_wall = max(1, int(rnd.epoch_blocks)) * BLOCK_SECONDS
    walls = tuple(rnd.funded_sku_wall_seconds or ())
    fit = tuple(s for s in skus if funded_sku_wall_for(walls, s, cap) < epoch_wall)
    excluded = tuple(s for s in skus if s not in fit)
    return fit, excluded


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
    # Per-round GPU-type choice needs an unpinned expected_gpu: a hard pin
    # would fail EVERY leg trained on any other chosen type (validator
    # gpu_name gate) — requeue-with-burn ×3 → terminal fail for every seated
    # miner. Comment-enforced until 2026-09-02; now fail-loud at launch.
    if (role == "trainer" and cfg.round.funded_pods == "rent"
            and cfg.round.funded_sku_per_leg and cfg.training.expected_gpu):
        problems.append(
            "[round] funded_sku_per_leg mixes GPU types inside one round but "
            "[training] expected_gpu pins one; set expected_gpu = \"\" (coordinated "
            "validator change) or turn funded_sku_per_leg off")
    if (role == "trainer" and cfg.round.funded_pods == "rent"
            and len(cfg.round.funded_pod_skus) > 1 and cfg.training.expected_gpu):
        problems.append(
            "[round] funded_pod_skus lists multiple GPU types but [training] "
            f"expected_gpu pins {cfg.training.expected_gpu!r} — per-round SKU "
            'choice needs expected_gpu = "" (a coordinated contract change)')
    if role == "trainer" and cfg.round.funded_pods == "rent":
        fit, excluded = funded_sku_wall_fit(cfg)
        rolling = int(getattr(cfg.round, "rolling_from_block", 0) or 0) > 0
        if excluded and not fit and rolling:
            # Rolling intake (DEC-CA-0043): a leg picks the first boundary
            # its wall clears, so a wall longer than one grid step is not
            # "never starts" — it is "settles a boundary later". The grid
            # bound is a BOUNDARY-mode rule; under rolling the per-leg target
            # (_leg_local.end_wall) bounds each SKU instead.
            logging.getLogger("cascade.config").warning(
                "[round] funded_sku_wall_seconds: every configured SKU (%s) has a "
                "measured wall of at least one epoch (%d blocks); rolling intake is "
                "armed (rolling_from_block=%d) so legs target a later boundary — "
                "make sure the walls were measured for THIS contract, not inherited",
                ", ".join(excluded), int(cfg.round.epoch_blocks),
                int(cfg.round.rolling_from_block))
        elif excluded and not fit:
            problems.append(
                "[round] funded_sku_wall_seconds excludes EVERY configured SKU "
                f"({', '.join(excluded)}): each measured wall is at least one epoch "
                f"({int(cfg.round.epoch_blocks)} blocks) long, so no leg could ever "
                "start — set walls measured for THIS grid (or {} to use the "
                "contract's max_train_seconds)")
        elif excluded:
            logging.getLogger("cascade.config").warning(
                "[round] funded_sku_wall_seconds excludes %s from funded rents (wall >= "
                "one epoch of %d blocks); only %s can rent — set walls measured for "
                "this grid if that is not intended", ", ".join(excluded),
                int(cfg.round.epoch_blocks), ", ".join(fit))
    # The round's screen/throne size pointers must name configured sizes.
    registry = cfg.training.size_registry
    for label, name in [("screen_size", cfg.round.screen_size), *(("throne_sizes", n) for n in cfg.round.throne_sizes)]:
        if name and name not in registry:
            problems.append(
                f"[round] {label} = {name!r} is not a configured size {sorted(registry)} "
                "(add the [[training.sizes]] block or fix the name)"
            )
    if role == "validator":
        from ..hippius import is_hub_ref

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
