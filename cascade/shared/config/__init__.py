"""Config loader for chain.toml — single source of truth for subnet config.

Miners, the trainer, and validators all load from here. The schema is
versioned; a file newer than this code warns and proceeds (operator-controlled
file, deployed by hand alongside the binaries — the same policy horizon uses).
"""

# The implementation is split by concern across the submodules below; this
# package re-exports every name the former single-module config.py defined,
# so ``from cascade.shared.config import X`` keeps working unchanged.
from __future__ import annotations

from .chain_config import _PLACEHOLDER_DIGEST as _PLACEHOLDER_DIGEST
from .chain_config import (
    BLOCK_SECONDS,
    ChainConfig,
    LaunchConfigError,
    assert_launch_ready,
    funded_sku_wall_fit,
)
from .generation import (
    GENERATION_PROVIDERS,
    KING_CORPUS_PUBLISH,
    GenerationConfig,
    parse_generation,
)
from .lane_c import (
    LaneCConfig,
    check_target,
    parse_lane_c,
    parse_target,
)
from .lane_d import LaneDConfig, parse_lane_d
from .loader import (
    DEFAULT_CHAIN_TOML,
    REPO_ROOT,
    RETIRED_KEYS,
    load_chain_config,
    parse_chain_config,
)
from .marketcast import MarketCastConfig, parse_marketcast
from .rounds import (
    RoundConfig,
    check_rollover_alignment,
    effective_epoch_blocks,
)
from .scoring import _GIFT_GATE_MODES as _GIFT_GATE_MODES
from .scoring import _INIT_GATE_MODES as _INIT_GATE_MODES
from .scoring import _MARGIN_MODES as _MARGIN_MODES
from .scoring import (
    EvalConfig,
    ScoringConfig,
    check_margin_v2,
    cohort_maxt_active,
    cohort_maxt_increment_active,
    effective_margin_mode,
    effective_win_margin_start,
    margin_v2_active,
    margin_v2_configured,
    mv_score_active,
)
from .scoring import _gift_gate_mode as _gift_gate_mode
from .scoring import _init_gate_mode as _init_gate_mode
from .scoring import _margin_mode as _margin_mode
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
    parse_netconfig,
)
from .training import (
    HEAT_GUARD_FACTOR,
    HEAT_GUARD_FLOOR_SECONDS,
    SizeSpec,
    TrainingContractConfig,
    parse_training_contract,
)
from .v2 import (
    LANE_B_REMAINDERS,
    LaneBConfig,
    V2Config,
    check_v2_activation,
    lane_b_replay_cover,
    parse_lane_b,
)
from .validators import _MV_BONUS_RE as _MV_BONUS_RE
from .validators import _REAL_CORPUS_REF_RE as _REAL_CORPUS_REF_RE
from .validators import (
    ANNEAL_SHAPES,
    BATCH_DENOMINATIONS,
    BUDGET_DENOMINATIONS,
    CHAMPION_PUBLISH_MODES,
    CONSUMABLE_FIELDS,
    CORPUS_MODES,
    DEDUP_MODES,
    FUNDED_MODES,
    FUNDED_POD_MODES,
    LONG_SERIES_MODES,
    MAX_CORPUS_EPOCHS,
    MAX_SERIES_LENGTH,
    SANDBOX_MODES,
    budget_denomination_parts,
    funded_host_bench_floor_for,
    funded_sku_wall_for,
    validate_accepted_fields,
    validate_anneal_shape,
    validate_batch_denomination,
    validate_budget_denomination,
    validate_champion_publish,
    validate_corpus_epochs,
    validate_corpus_mode,
    validate_corpus_target_points,
    validate_dedup_mode,
    validate_funded_field_cap,
    validate_funded_host_bench_floor,
    validate_funded_mode,
    validate_funded_pod_skus,
    validate_funded_pods,
    validate_funded_price_cap,
    validate_funded_sku_wall_seconds,
    validate_real_corpus_ref,
    validate_sandbox_mode,
)
from .validators import _validate_packed_sources_mode as _validate_packed_sources_mode

__all__ = [
    "ANNEAL_SHAPES",
    "ActivationConfig",
    "BATCH_DENOMINATIONS",
    "BLOCK_SECONDS",
    "BUDGET_DENOMINATIONS",
    "CHAMPION_PUBLISH_MODES",
    "CONSUMABLE_FIELDS",
    "CORPUS_MODES",
    "LONG_SERIES_MODES",
    "MAX_SERIES_LENGTH",
    "ChainConfig",
    "DEDUP_MODES",
    "DEFAULT_CHAIN_TOML",
    "DependencyConfig",
    "EvalConfig",
    "FUNDED_MODES",
    "FUNDED_POD_MODES",
    "GeneratorConfig",
    "HEAT_GUARD_FACTOR",
    "HEAT_GUARD_FLOOR_SECONDS",
    "LANE_B_REMAINDERS",
    "LaneBConfig",
    "LaneCConfig",
    "LaunchConfigError",
    "MAX_CORPUS_EPOCHS",
    "ManifestConfig",
    "NetConfigConfig",
    "REPO_ROOT",
    "RETIRED_KEYS",
    "RoundConfig",
    "SANDBOX_MODES",
    "ScoringConfig",
    "SizeSpec",
    "StaticGuardConfig",
    "StorageConfig",
    "SubnetConfig",
    "TelemetryConfig",
    "TrainingContractConfig",
    "V2Config",
    "GENERATION_PROVIDERS",
    "KING_CORPUS_PUBLISH",
    "GenerationConfig",
    "parse_generation",
    "ValidatorConfig",
    "WandbConfig",
    "assert_launch_ready",
    "budget_denomination_parts",
    "check_margin_v2",
    "check_rollover_alignment",
    "check_v2_activation",
    "cohort_maxt_active",
    "cohort_maxt_increment_active",
    "effective_epoch_blocks",
    "effective_margin_mode",
    "effective_win_margin_start",
    "funded_host_bench_floor_for",
    "funded_sku_wall_fit",
    "funded_sku_wall_for",
    "lane_b_replay_cover",
    "load_chain_config",
    "parse_chain_config",
    "parse_netconfig",
    "margin_v2_active",
    "margin_v2_configured",
    "mv_score_active",
    "parse_lane_b",
    "parse_lane_c",
    "parse_lane_d",
    "LaneDConfig",
    "parse_marketcast",
    "MarketCastConfig",
    "parse_target",
    "check_target",
    "parse_training_contract",
    "validate_accepted_fields",
    "validate_anneal_shape",
    "validate_batch_denomination",
    "validate_budget_denomination",
    "validate_champion_publish",
    "validate_corpus_epochs",
    "validate_corpus_mode",
    "validate_corpus_target_points",
    "validate_dedup_mode",
    "validate_funded_field_cap",
    "validate_funded_host_bench_floor",
    "validate_funded_mode",
    "validate_funded_pod_skus",
    "validate_funded_pods",
    "validate_funded_price_cap",
    "validate_funded_sku_wall_seconds",
    "validate_real_corpus_ref",
    "validate_sandbox_mode",
]
