"""chain.toml path defaults, retired keys and load_chain_config."""

from __future__ import annotations

import re
import sys
import tomllib  # py311+
from dataclasses import replace
from pathlib import Path

from .chain_config import ChainConfig
from .generation import parse_generation
from .lane_c import parse_lane_c
from .lane_d import parse_lane_d
from .marketcast import check_marketcast_arming, parse_marketcast
from .reliability import parse_reliability
from .rounds import RoundConfig, check_rollover_alignment
from .scoring import (
    EvalConfig,
    ScoringConfig,
    _bench_gate_tolerance,
    _bench_gate_win_rate,
    _gift_gate_mode,
    _init_gate_mode,
    _margin_mode,
    check_margin_v2,
    parse_bench_workers,
)
from .sections import (
    ActivationConfig,
    DependencyConfig,
    GeneratorConfig,
    ManifestConfig,
    StaticGuardConfig,
    StorageConfig,
    SubnetConfig,
    TelemetryConfig,
    ValidatorConfig,
    WandbConfig,
    parse_netconfig,
)
from .training import parse_training_contract
from .v2 import (
    V2Config,
    check_bench_source,
    check_v2_activation,
    parse_lane_b,
    parse_v2_economics,
)
from .validators import (
    _validate_packed_sources_mode,
    validate_accepted_fields,
    validate_champion_publish,
    validate_corpus_target_points,
    validate_dedup_mode,
    validate_funded_field_cap,
    validate_funded_host_bench_floor,
    validate_funded_mode,
    validate_funded_pod_skus,
    validate_funded_pods,
    validate_funded_price_cap,
    validate_funded_sku_wall_seconds,
    validate_measured_image,
    validate_nonneg_seconds,
    validate_real_corpus_ref,
    validate_sandbox_mode,
    validate_watchdog_tolerance,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHAIN_TOML = REPO_ROOT / "chain.toml"
if not DEFAULT_CHAIN_TOML.is_file():
    # Wheels have no checkout root. Ship the same reviewed defaults in-package.
    DEFAULT_CHAIN_TOML = Path(__file__).resolve().parents[2] / "data" / "chain.toml"


# Keys of the retired boundary-round trainer (heat screen, tie run-off, duel
# seating and its activation blocks, commit witness, field-wide dedup screen +
# behavioural probe + its retained token prefix, unfunded-boundary skipping,
# scratch shadow). Deployed chain.tomls still carry them; the loader reads only the keys it knows, so these load and are IGNORED.
# Listed so that stays a deliberate, tested property (tests/unit/test_config.py)
# rather than an accident — never re-use one of these names for a new knob.
RETIRED_KEYS: dict[str, frozenset[str]] = {
    "round": frozenset({
        "heat_num_samples", "heat_infra_requeues", "tie_runoff_windows",
        "tie_runoff_phase_seconds", "init_gate_mode", "duel_field_cap",
        "duel_seat_all", "genesis_generator_ref", "skip_unfunded_rounds",
        "max_rounds_per_day", "funded_capacity_probe", "funded_capacity_reserve",
        "commit_witness_path", "dedup_config_only_enforce",
        "dedup_config_only_from_block", "dedup_phase_seconds", "dedup_plan_seconds",
        "dedup_probe_mode", "dedup_probe_series", "dedup_probe_generate_seconds",
        "dedup_probe_budget_seconds", "dedup_probe_allow_weak_sandbox",
        "dedup_max_tokens", "duel_from_block", "funded_activation_block",
    }),
    "telemetry": frozenset({"scratch_shadow_every_rounds"}),
    # Parsed by older releases (hf_cache_seconds was REQUIRED there), read by none.
    "eval": frozenset({"eval_source"}),
    "validator": frozenset({"hf_cache_seconds"}),
}


CKPT_TRANSPORTS = ("ssh", "r2")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def parse_ckpt_transport(r: dict) -> dict:
    """The ``[round]`` checkpoint-transport knobs (r2 direct harvest), parsed
    and validated: ``funded_harvest_transport``, ``ckpt_relay_host``,
    ``min_free_disk_gb``, ``ckpt_r2_bucket``, ``ckpt_r2_prefix``. Unknown
    transports and malformed buckets/prefixes fail loud — a typo silently
    falling back to the ssh path would re-open the 2026-10-31 disk incident."""
    transport = str(r.get("funded_harvest_transport", "ssh")).strip().lower()
    if transport not in CKPT_TRANSPORTS:
        raise ValueError(f"[round] funded_harvest_transport={transport!r} invalid; "
                         f"expected one of {CKPT_TRANSPORTS}")
    relay = r.get("ckpt_relay_host", "")
    if not isinstance(relay, str):
        raise ValueError("[round] ckpt_relay_host must be a hosts-file entry name (string)")
    gb = r.get("min_free_disk_gb", 30.0)
    if isinstance(gb, bool) or not isinstance(gb, (int, float)) or not (0 <= float(gb) < 1e6):
        raise ValueError(f"[round] min_free_disk_gb must be a finite number >= 0; got {gb!r}")
    bucket = str(r.get("ckpt_r2_bucket", "") or "").strip()
    if bucket and not _BUCKET_RE.match(bucket):
        raise ValueError(f"[round] ckpt_r2_bucket={bucket!r} is not a valid bucket name")
    prefix = str(r.get("ckpt_r2_prefix", "ckpt/"))
    if (prefix.startswith("/") or ".." in prefix.split("/")
            or not re.fullmatch(r"[A-Za-z0-9._/-]*", prefix)):
        raise ValueError(f"[round] ckpt_r2_prefix={prefix!r} invalid (relative key prefix, "
                         "letters/digits/._-/ only, no '..')")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return {"funded_harvest_transport": transport, "ckpt_relay_host": relay.strip(),
            "min_free_disk_gb": float(gb), "ckpt_r2_bucket": bucket,
            "ckpt_r2_prefix": prefix}


def parse_chain_config(raw: dict) -> ChainConfig:
    """Parse an already-loaded ``chain.toml`` table (``tomllib`` shape).

    Every load check runs on it, so a table built in memory — ``chain.toml``
    merged with an owner-signed network-config overlay (DEC-CA-0057,
    :mod:`cascade.shared.netconfig`) — passes exactly the validation a file
    on disk does. ``raw`` is not mutated; the result keeps it as ``.raw``."""
    if not isinstance(raw, dict):
        raise TypeError("parse_chain_config needs a chain.toml table (dict)")
    return load_chain_config(raw=raw)


def load_chain_config(path: Path | str | None = None, *, raw: dict | None = None) -> ChainConfig:
    """Load and parse ``chain.toml``. Raises on a missing file or unsupported
    (too-old) schema; warns and proceeds on a newer schema. ``raw`` (an
    already-loaded table, see :func:`parse_chain_config`) skips the file."""
    if raw is None:
        p = Path(path) if path is not None else DEFAULT_CHAIN_TOML
        if not p.exists():
            raise FileNotFoundError(f"chain.toml not found at {p}")
        with p.open("rb") as fh:
            raw = tomllib.load(fh)

    # Legacy configurations remain unchanged; direct activation fails closed.
    from ..direct_moe import parse_direct_config
    parse_direct_config(raw.get("direct_moe", {}))

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

    # Generator code never runs on the orchestrator / a validator without a
    # hardened sandbox (CLAUDE.md operational invariant).
    from ..sandbox_policy import check_generator_sandbox_config
    check_generator_sandbox_config(g, r)

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

    # Rolling intake + era king (DEC-CA-0043): every block key names ONE
    # rollover block, aligned to the grid on both sides of it, and the era
    # king gate never precedes the corrections it stacks on. A misaligned or
    # split rollover would let one node open an era where another still runs
    # boundary rounds — refuse to load rather than fork the fleet. The same
    # check guards a rollover resolved at runtime from validator signals
    # (DEC-CA-0045, ``cascade.shared.activation.apply_activation``).
    # Dethrone bar v2 (DEC-CA-0049): refuse a half-configured or floorless bar.
    _mv2_start = float(s.get("win_margin_start_v2", 0.0) or 0.0)
    _mv2_end = float(s.get("win_margin_end_v2", 0.0) or 0.0)
    _mv2_blocks = int(s.get("margin_warmup_blocks_v2", 0) or 0)
    _mv2_from = int(s.get("margin_v2_from_block", 0) or 0)
    check_margin_v2(_mv2_start, _mv2_end, _mv2_blocks, _mv2_from)
    _rolling = max(0, int(r.get("rolling_from_block", 0) or 0))
    _era_king = max(0, int(s.get("era_king_from_block", 0) or 0))
    _tenure_blocks = max(0, int(s.get("tenure_blocks_from_block", 0) or 0))
    _era_settlements = max(0, int(r.get("era_settlements", 0) or 0))
    check_rollover_alignment(
        rolling_from_block=_rolling,
        era_king_from_block=_era_king,
        tenure_blocks_from_block=_tenure_blocks,
        epoch_blocks=_eb,
        epoch_blocks_prev=_ebp,
        epoch_activation_block=_eab,
        era_settlements=_era_settlements,
        cohort_maxt_from_block=max(0, int(s.get("cohort_maxt_from_block", 0) or 0)),
        cohort_maxt_increment_from_block=max(
            0, int(s.get("cohort_maxt_increment_from_block", 0) or 0)),
        funded_pods=str(r.get("funded_pods", "off") or "off"),
        funded_king_rent=bool(r.get("funded_king_rent", False)),
    )

    # Stake-weighted activation (DEC-CA-0045): validators signal readiness on
    # chain; the rollover block is resolved from those signals at runtime when
    # the DEC-CA-0043 keys are 0. Validated here so a bad threshold or grid
    # never reaches the resolver.
    _forf_succ = str(s.get("forfeit_successor_hotkey", "") or "").strip()
    if _forf_succ and _forf_succ in {str(x) for x in (s.get("forfeit_hotkeys", ()) or ())}:
        raise ValueError(
            f"[scoring] forfeit_successor_hotkey={_forf_succ!r} is itself listed in "
            "forfeit_hotkeys — a forfeited hotkey cannot be crowned")
    ac = raw.get("activation", {})
    _act_feature = str(ac.get("feature", "") or "").strip()
    if _act_feature and not re.fullmatch(r"[A-Za-z0-9._-]+", _act_feature):
        raise ValueError(
            f"[activation] feature={_act_feature!r} must be a token of letters, digits, "
            "'.', '_' or '-' (it is written verbatim into the on-chain note)")
    _act_threshold = float(ac.get("threshold", 0.51))
    if not (0.0 < _act_threshold <= 1.0):
        raise ValueError(
            f"[activation] threshold={_act_threshold} must be in (0, 1] (a fraction "
            "of eligible validator stake)")
    _act_grid_after = max(0, int(ac.get("epoch_blocks_after", 0) or 0))
    if _act_grid_after and _eb % _act_grid_after:
        raise ValueError(
            f"[activation] epoch_blocks_after={_act_grid_after} must divide "
            f"[round] epoch_blocks={_eb}: the resolved rollover is a boundary of "
            "the grid before it and must be one of the grid after it too")
    if _act_feature and not _rolling:
        # The block the fleet resolves is ANY boundary of the loaded grid plus
        # one; ``apply_activation`` must be able to arm it (the DEC-CA-0043
        # rule: it starts an era on the grid after it). Refuse a config whose
        # lock-in could resolve a block the node then cannot apply.
        if _era_settlements < 1:
            raise ValueError(
                "[round] era_settlements must be >= 1 when [activation] feature is set "
                "(the resolved rollover opens an era)")
        _era_after = (_act_grid_after or _eb) * _era_settlements
        if _eb % _era_after:
            raise ValueError(
                f"[round] epoch_blocks={_eb} must be a multiple of the era after the "
                f"rollover ({_act_grid_after or _eb} × era_settlements={_era_settlements} "
                f"= {_era_after}), or a lock-in at some boundary resolves a rollover "
                "that cannot start an era")
        if str(r.get("funded_pods", "off") or "off") != "rent" or not bool(
                r.get("funded_king_rent", False)):
            raise ValueError(
                "[activation] feature is set but [round] funded_pods/funded_king_rent are "
                "not \"rent\"/true — the resolved rollover could not be applied")

    # cascade v2 (DEC-CA-0050): ONE typed activation block. The post-v2
    # contract rides on the live one as a schedule (never in contract_digest);
    # lane B launches separately, leaving the full allocation to lane A first.
    v2raw = raw.get("v2", {}) or {}
    _v2_from = max(0, int(v2raw.get("from_block", 0) or 0))
    training = parse_training_contract(t)
    v2_training = (parse_training_contract(v2raw["training"])
                   if isinstance(v2raw.get("training"), dict) else None)
    check_v2_activation(
        from_block=_v2_from, has_contract=v2_training is not None,
        epoch_blocks=_eb, epoch_blocks_prev=_ebp, epoch_activation_block=_eab,
        era_settlements=_era_settlements, rolling_from_block=_rolling,
        activation_feature=str(ac.get("feature", "") or ""))
    if v2_training is not None:
        if v2_training.train_seed_salt != training.train_seed_salt:
            raise ValueError(
                "[v2.training] train_seed_salt must equal [training] train_seed_salt "
                "(round seeds are derived from it where the era is not known)")
        if v2_training.extra_sizes or v2_training.budget_denomination_after:
            raise ValueError("[v2.training] carries no [[sizes]] and no "
                             "budget_denomination_after schedule")
        training = replace(training, v2_from_block=_v2_from, v2_contract=v2_training)
    generation = parse_generation(raw.get("generation"))
    # DEC-CA-0053: a contract that trains on shards needs the CPU tier.
    for name, c in (("[training]", training), ("[v2.training]", v2_training)):
        if c is not None and c.corpus_mode == "shards" and not generation.enabled:
            raise ValueError(f"{name} corpus_mode = 'shards' needs [generation] enabled = true "
                             "(the CPU tier materialises the shards)")
    lb_raw = raw.get("lane_b", {}) or {}
    _lb_from = lb_raw.get("from_block", 0)
    if type(_lb_from) is not int or _lb_from < 0:
        raise ValueError("[lane_b] from_block must be an integer >= 0 (0 = off)")
    if _lb_from:
        if not _v2_from or _lb_from < _v2_from:
            raise ValueError("[lane_b] from_block needs [v2] from_block set and must be "
                             "at or after it")
        _lb_grid = _ebp if (_eab and _ebp and _lb_from < _eab) else _eb
        _lb_era = max(1, _lb_grid) * max(1, _era_settlements)
        if _lb_from % _lb_era:
            raise ValueError(f"[lane_b] from_block={_lb_from} must be an era boundary "
                             f"(a multiple of {_lb_era})")
    # Public launch blocks name era STARTS; the settlement AT that boundary
    # belongs to the preceding era. Keep LaneBConfig.active's inclusive API.
    lane_b = parse_lane_b(lb_raw, from_block=(_lb_from + 1 if _lb_from else 0),
                          era_settlements=_era_settlements,
                          # DEC-CA-0059 horizon targets = the scored ladder
                          horizons=tuple(int(h) for h in (e.get("scored_horizons", ())
                                                          or ())),
                          v2_from_block=_v2_from)
    _mg = int(lane_b.merge_guard_from_block)
    if _mg and _v2_from:
        # a typed merge-guard block is an era boundary at/after the v2 go-live
        # (the era-init gate keys on era starts; DEC-CA-0050 merge guards)
        _mg_grid = _ebp if (_eab and _ebp and 0 < _mg < _eab) else _eb
        _mg_era = max(1, _mg_grid) * max(1, _era_settlements)
        if _mg < _v2_from or _mg % _mg_era:
            raise ValueError(f"[lane_b] merge_guard_from_block={_mg} must be an era boundary "
                             f"(a multiple of {_mg_era}) at or after [v2] from_block="
                             f"{_v2_from}")
    _ab = int(lane_b.acceptance_from_block)
    if _ab:
        _ab_grid = _ebp if (_eab and _ebp and 0 < _ab < _eab) else _eb
        _ab_era = max(1, _ab_grid) * max(1, _era_settlements)
        if not _v2_from or _ab < _v2_from or _ab % _ab_era:
            raise ValueError("[lane_b] acceptance_from_block must be an era boundary "
                             "at or after [v2] from_block")
    # DEC-CA-0056: lane C's typed block must be an era boundary on the grid
    # in force at it, at or after the v2 go-live (checked inside parse_lane_c).
    # DEC-CA-0064: [marketcast] suites admit the mkt:<suite> lane-C targets
    # whether or not MarketCast is armed (MarketCast-only lane C, owner
    # 2026-10-20: the shipped targets must load with it off; active_targets
    # then counts none of them and lane C's share burns). Its cross-table
    # arming checks run below.
    marketcast = parse_marketcast(raw.get("marketcast", {}))
    lc_raw = raw.get("lane_c", {}) or {}
    _lc_from = int(lc_raw.get("from_block", 0) or 0) if isinstance(lc_raw, dict) else 0
    _lc_grid = _ebp if (_eab and _ebp and 0 < _lc_from < _eab) else _eb
    lane_c = parse_lane_c(
        lc_raw, v2_from_block=_v2_from,
        era_blocks=max(1, _lc_grid) * max(1, _era_settlements),
        era_settlements=_era_settlements,
        lane_b_share=float((raw.get("lane_b", {}) or {}).get("share", 0.5)),
        mkt_suites=tuple(marketcast.suites),
        # concurrent competitions (owner 2026-10-20): MarketCast's start is
        # [marketcast] from_block (one source of truth); a competition block
        # is an era boundary on the grid in force at it
        kind_from_blocks={"mkt": int(marketcast.from_block)},
        era_blocks_at=lambda b: max(1, _ebp if (_eab and _ebp and 0 < b < _eab) else _eb)
        * max(1, _era_settlements))
    # An mkt: win pays through a lane-C slot (lane C's TTL): its tiered
    # full-pay period must not outlive it (checked armed or not, like lane C).
    _lc_ttl = int(lane_c.win_ttl_eras)
    if _lc_ttl and any(n > _lc_ttl for _, n in marketcast.full_pay_tiers):
        raise ValueError(f"[marketcast] full_pay_tiers {list(marketcast.full_pay_tiers)}: "
                         f"full_eras exceeds [lane_c] win_ttl_eras={_lc_ttl} (an mkt: win "
                         "pays through a lane-C slot)")

    cfg = ChainConfig(
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
            min_channels=int(g.get("min_channels", 1)),
            king_min_channels=int(g.get("king_min_channels", 0)),
            king_min_channels_refs=tuple(str(r) for r in g.get("king_min_channels_refs", ())),
            corpus_target_points=validate_corpus_target_points(
                g.get("corpus_target_points", 0), int(g["max_total_points"])),
            stream_stall_seconds=int(g.get("stream_stall_seconds", 0)),
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
        training=training,
        round=RoundConfig(
            **parse_reliability(r),
            epoch_blocks=int(r.get("epoch_blocks", 7200)),
            round_hours=float(r.get("round_hours", 24.0)),
            epoch_blocks_prev=int(r.get("epoch_blocks_prev", 0)),
            epoch_activation_block=int(r.get("epoch_activation_block", 0)),
            rolling_from_block=_rolling,
            era_settlements=_era_settlements,
            heat_train_hours=float(r.get("heat_train_hours", 0.5)),
            heat_n_windows=int(r.get("heat_n_windows", 256)),
            heat_guard_factor=float(r.get("heat_guard_factor", 1.0)),
            heat_guard_floor_seconds=int(r.get("heat_guard_floor_seconds", 900)),
            finalists=int(r.get("finalists", 1)),
            max_finalists=int(r.get("max_finalists", 1)),
            screen_size=str(r.get("screen_size", "")),
            throne_sizes=tuple(str(x) for x in r.get("throne_sizes", ())),
            funded_bench_margin_seconds=max(0, int(r.get("funded_bench_margin_seconds", 0) or 0)),
            blocked_hotkeys=tuple(str(x) for x in r.get("blocked_hotkeys", ()) or ()),
            blocked_from_block=max(0, int(r.get("blocked_from_block", 0) or 0)),
            one_submission_per_hotkey=bool(r.get("one_submission_per_hotkey", True)),
            commit_floor_block=int(r.get("commit_floor_block", 0)),
            submissions_db_path=str(r.get("submissions_db_path", "trainer_submissions.json")),
            funded_mode=validate_funded_mode(str(r.get("funded_mode", "off"))),
            funded_queue_path=str(r.get("funded_queue_path", "funded_queue.json")),
            funded_entry_ttl_hours=float(r.get("funded_entry_ttl_hours", 36.0)),
            funded_pods=validate_funded_pods(str(r.get("funded_pods", "off"))),
            payer_vault_dir=str(r.get("payer_vault_dir", "")),
            funded_pod_sku=str(r.get("funded_pod_sku", "")),
            funded_pod_image=str(r.get("funded_pod_image", "")),
            worker_code_fingerprint=str(r.get("worker_code_fingerprint", "")).strip().lower(),
            funded_ready_timeout_seconds=float(
                r.get("funded_ready_timeout_seconds", 900.0)),
            funded_robot_duration_days=max(1, int(r.get("funded_robot_duration_days", 1))),
            funded_pod_checkpoint=str(r.get("funded_pod_checkpoint", "harvest")),
            **parse_ckpt_transport(r),
            funded_field_cap=validate_funded_field_cap(
                r.get("funded_field_cap", 0)),
            funded_pod_skus=validate_funded_pod_skus(
                r.get("funded_pod_skus", ())),
            funded_cpu_blocklist=tuple(
                str(x).strip() for x in (r.get("funded_cpu_blocklist", ()) or ())
                if str(x).strip()),
            funded_host_bench_floor=validate_funded_host_bench_floor(
                r.get("funded_host_bench_floor", None)),
            funded_sku_per_leg=bool(r.get("funded_sku_per_leg", True)),
            funded_max_price_per_hour=validate_funded_price_cap(
                "funded_max_price_per_hour", r.get("funded_max_price_per_hour", 1.5)),
            funded_max_leg_cost_usd=validate_funded_price_cap(
                "funded_max_leg_cost_usd", r.get("funded_max_leg_cost_usd", 1.6)),
            funded_sku_wall_seconds=(validate_funded_sku_wall_seconds(
                r["funded_sku_wall_seconds"]) if "funded_sku_wall_seconds" in r
                else RoundConfig.funded_sku_wall_seconds),
            funded_sku_wall_measured_image=validate_measured_image(
                r.get("funded_sku_wall_measured_image", "")),
            progress_watchdog_grace_seconds=validate_nonneg_seconds(
                "progress_watchdog_grace_seconds",
                r.get("progress_watchdog_grace_seconds",
                      RoundConfig.progress_watchdog_grace_seconds)),
            progress_watchdog_tolerance=validate_watchdog_tolerance(
                r.get("progress_watchdog_tolerance", RoundConfig.progress_watchdog_tolerance)),
            provider_ttl_margin_seconds=validate_nonneg_seconds(
                "provider_ttl_margin_seconds",
                r.get("provider_ttl_margin_seconds", RoundConfig.provider_ttl_margin_seconds)),
            payer_key_check_seconds=validate_nonneg_seconds(
                "payer_key_check_seconds",
                r.get("payer_key_check_seconds", RoundConfig.payer_key_check_seconds)),
            funded_king_rent=bool(r.get("funded_king_rent", False)),
            detached_dispatch=bool(r.get("detached_dispatch", True)),
            dispatch_poll_seconds=max(5, int(r.get("dispatch_poll_seconds", 30))),
            dispatch_reattach_grace_seconds=max(0, int(r.get("dispatch_reattach_grace_seconds", 900))),
            funded_operator_fallback=bool(r.get("funded_operator_fallback", False)),
            funded_operator_fallback_fresh_window_seconds=max(
                0, int(r.get("funded_operator_fallback_fresh_window_seconds", 3600))),
            funded_require_tensor_digests=bool(r.get("funded_require_tensor_digests", False)),
            submission_vault_dir=str(r.get("submission_vault_dir", "")),
            champion_publish=validate_champion_publish(
                str(r.get("champion_publish", "off"))),
            champion_publish_delay_rounds=int(r.get("champion_publish_delay_rounds", 2)),
            dedup_mode=validate_dedup_mode(str(r.get("dedup_mode", "off")),
                                           "dedup_mode"),
            private_copy_mode=validate_dedup_mode(
                str(r.get("private_copy_mode", "shadow")), "private_copy_mode"),
            private_copy_min_containment=min(1.0, max(0.0, float(
                r.get("private_copy_min_containment", 0.8)))),
            private_copy_min_tokens=max(0, int(r.get("private_copy_min_tokens", 2_000))),
            dedup_embedded_mode=validate_dedup_mode(
                str(r.get("dedup_embedded_mode", "shadow")), "dedup_embedded_mode"),
            dedup_max_text_mb=int(r.get("dedup_max_text_mb", 4)),
            reveal_margin_blocks=int(r.get("reveal_margin_blocks", 25)),
        ),
        eval=EvalConfig(
            eval_dataset=str(e["eval_dataset"]),
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
            calib_horizons=tuple(int(h) for h in e.get("calib_horizons", ())),
            calib_windows=int(e.get("calib_windows", 256)),
            calib_num_samples=int(e.get("calib_num_samples", 32)),
            scored_horizons=tuple(int(h) for h in e.get("scored_horizons", ())),
            scored_from_block=int(e.get("scored_from_block", 0) or 0),
            bench_hold_max_hours=float(e.get("bench_hold_max_hours", 2.0)),
            bench_workers=parse_bench_workers(e.get("bench_workers", "auto")),
            bench_paired=bool(e.get("bench_paired", True)),
            mix_from_block=int(e.get("mix_from_block", 0)),
            mix_jitter_alpha=float(e.get("mix_jitter_alpha", 4.0)),
            mix_block_slots=int(e.get("mix_block_slots", 8)),
            mix_class_keep_frac=float(e.get("mix_class_keep_frac", 0.7)),
            mix_series_bag_frac=float(e.get("mix_series_bag_frac", 0.7)),
            mix_target_windows=int(e.get("mix_target_windows", 0)),
            mix_tier_from_block=int(e.get("mix_tier_from_block", 0)),
            mix_tier_jitter_alpha=float(e.get("mix_tier_jitter_alpha", 75.0)),
        ),
        scoring=ScoringConfig(
            win_margin_start=float(s["win_margin_start"]),
            win_margin_end=float(s["win_margin_end"]),
            margin_warmup_rounds=int(s["margin_warmup_rounds"]),
            win_margin_start_prev=float(s.get("win_margin_start_prev", 0.0) or 0.0),
            margin_activation_block=max(0, int(s.get("margin_activation_block", 0) or 0)),
            win_margin_start_prev2=float(s.get("win_margin_start_prev2", 0.0) or 0.0),
            margin_activation_block2=max(0, int(s.get("margin_activation_block2", 0) or 0)),
            cohort_maxt_from_block=max(0, int(s.get("cohort_maxt_from_block", 0) or 0)),
            forfeit_hotkeys=tuple(str(x) for x in s.get("forfeit_hotkeys", ()) or ()),
            forfeit_from_block=max(0, int(s.get("forfeit_from_block", 0) or 0)),
            forfeit_successor_hotkey=str(s.get("forfeit_successor_hotkey", "") or "").strip(),
            increment_from_block=max(0, int(s.get("increment_from_block", 0) or 0)),
            cohort_maxt_increment_from_block=max(
                0, int(s.get("cohort_maxt_increment_from_block", 0) or 0)),
            mv_score_from_block=max(0, int(s.get("mv_score_from_block", 0) or 0)),
            entry_fault_from_block=max(0, int(s.get("entry_fault_from_block", 0) or 0)),
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
            bench_gate_mode=_gift_gate_mode(s.get("bench_gate_mode", "off")),
            bench_gate_min_items=int(s.get("bench_gate_min_items", 10)),
            # DEC-CA-0061 rule v2 thresholds
            bench_gate_tolerance=_bench_gate_tolerance(s.get("bench_gate_tolerance", 0.01)),
            bench_gate_min_win_rate=_bench_gate_win_rate(s.get("bench_gate_min_win_rate", 0.5)),
            # DEC-CA-0034 (parsing added with DEC-CA-0035's loader sweep —
            # same landed-without-parsing defect as the [training] knobs)
            init_gate_mode=_init_gate_mode(s.get("init_gate_mode", "off")),
            init_gate_tolerance=float(s.get("init_gate_tolerance", 0.0)),
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
            declared_contract_from_block=int(s.get("declared_contract_from_block", 0) or 0),
            era_king_from_block=_era_king,
            tenure_blocks_from_block=_tenure_blocks,
            margin_warmup_blocks=max(0, int(s.get("margin_warmup_blocks", 0) or 0)),
            cascade_reign_blocks=max(0, int(s.get("cascade_reign_blocks", 0) or 0)),
            king_resync_max_blocks=max(0, int(s.get("king_resync_max_blocks", 0) or 0)),
            win_margin_start_v2=_mv2_start,
            win_margin_end_v2=_mv2_end,
            margin_warmup_blocks_v2=_mv2_blocks,
            margin_v2_from_block=_mv2_from,
        ),
        dependencies=DependencyConfig(
            max_packages=int(d["max_packages"]),
            allowed=tuple(str(x) for x in d["allowed"]),
        ),
        static_guard=StaticGuardConfig(
            blocked=tuple(str(x) for x in sg["blocked"]),
            packed_sources=_validate_packed_sources_mode(
                str(sg.get("packed_sources", "scan"))),
            packed_sources_from_block=int(sg.get("packed_sources_from_block", 0) or 0),
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
            hash_link_from_block=int(m.get("hash_link_from_block", 0)),
        ),
        validator=ValidatorConfig(
            weight_set_interval_blocks=int(v["weight_set_interval_blocks"]),
            poll_seconds=int(v["poll_seconds"]),
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
            funded_key_env=str(wb.get("funded_key_env", "")).strip(),
            validator_summary=bool(wb.get("validator_summary", False)),
        ),
        telemetry=TelemetryConfig(
            host_probe=bool(tm.get("host_probe", True)),
            host_bench=bool(tm.get("host_bench", True)),
            bench_anneal_fraction=float(tm.get("bench_anneal_fraction", 0.0)),
            funded_bench=bool(tm.get("funded_bench", True)),
            funded_bench_verify_top=int(tm.get("funded_bench_verify_top", 1)),
            funded_bench_verify_tolerance=float(
                tm.get("funded_bench_verify_tolerance", 0.02)),
        ),
        activation=ActivationConfig(
            feature=str(ac.get("feature", "") or "").strip(),
            threshold=_act_threshold,
            epoch_blocks_after=_act_grid_after,
            dormant_after_blocks=max(0, int(ac.get("dormant_after_blocks", 0) or 0)),
        ),
        lane_b=lane_b,
        v2=V2Config(from_block=_v2_from,
                    bench_gate_mode=_gift_gate_mode(v2raw.get("bench_gate_mode", "off")),
                    **parse_v2_economics(v2raw)),
        generation=generation,
        lane_c=lane_c,
        lane_d=parse_lane_d(raw.get("lane_d", {})),
        marketcast=marketcast,
        netconfig=parse_netconfig(raw.get("netconfig"), activation_feature=_act_feature),
        raw=raw,
    )
    if (cfg.round.funded_harvest_transport == "r2"
            and not (cfg.storage.backup_s3_endpoint or "").strip()):
        raise ValueError("[round] funded_harvest_transport = \"r2\" needs [storage] "
                         "backup_s3_endpoint (the Cloudflare R2 checkpoint bucket)")
    # DEC-CA-0063: a validator bench source with a block is an era-boundary
    # v2 rule that needs the king ladder (k = 1) and a named main validator.
    _bs = int(cfg.v2.bench_source_from_block)
    if _bs:
        _bs_grid = _ebp if (_eab and _ebp and _bs < _eab) else _eb
        check_bench_source(v2=cfg.v2, owner_hotkey=cfg.netconfig.owner_hotkey,
                           era_blocks=max(1, _bs_grid) * max(1, _era_settlements),
                           manifest_validator_hotkey=cfg.manifest.validator_hotkey)
    # An enforced v2 gate pairs complete suites only; a capped sweep is partial
    # by construction and would hold every challenger.
    # Checked while the launch is still dated (rule block unresolved) and
    # whichever rule block an overlay later names: "enforce" is the condition.
    if cfg.v2.bench_gate_mode == "enforce" and int(cfg.eval.cascade_bench_max_series):
        raise ValueError("[eval] cascade_bench_max_series must be 0 while [v2] bench_gate_mode "
                         "= \"enforce\": the v2 gate requires complete benchmark suites")
    # DEC-CA-0064: an armed MarketCast is an era-boundary block at/after the v2
    # go-live and lane C, scored on the main validator's GPU (DEC-CA-0063).
    if (cfg.v2.commit_from_block and cfg.v2.from_block
            and cfg.v2.commit_from_block > cfg.v2.from_block):
        raise ValueError("[v2] commit_from_block must not follow from_block")
    if (cfg.lane_c.from_block and cfg.lane_c.takes_from == "lane_b"
            and (not cfg.lane_b.from_block
                 or int(cfg.lane_c.from_block) + 1 < int(cfg.lane_b.from_block))):
        # Legacy C allocations drawn from B still need that source to be live.
        # C drawn from A can launch independently, including before B.
        raise ValueError(f"[lane_c] from_block={cfg.lane_c.from_block} needs lane B live by "
                         f"its first settlement ([lane_b] from_block={cfg.lane_b.from_block})")
    _mc = int(cfg.marketcast.from_block)
    if _mc:
        _mc_grid = _ebp if (_eab and _ebp and _mc < _eab) else _eb
        check_marketcast_arming(
            cfg.marketcast, era_blocks=max(1, _mc_grid) * max(1, _era_settlements),
            v2_from_block=_v2_from, lane_c_from_block=int(cfg.lane_c.from_block),
            validator_bench_from=(_bs if str(cfg.v2.bench_source) == "validator" else 0),
            owner_hotkey=str(cfg.netconfig.owner_hotkey or ""))
    return cfg
