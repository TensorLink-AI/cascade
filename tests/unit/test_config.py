"""chain.toml loads and exposes the expected schema."""

from __future__ import annotations

from cascade.eval.crps import DEFAULT_QUANTILE_LEVELS
from cascade.eval.koth import KothParams


def test_config_loads(cfg):
    assert cfg.schema_version == 1
    assert cfg.subnet.name == "cascade"
    assert cfg.generator.corpus_n_series > 0
    assert cfg.generator.min_length < cfg.generator.max_length
    assert cfg.generator.max_channels >= 1
    # From-scratch Toto2 contract: budgeted by ~hours on the reference GPU,
    # enforced as a fixed (derived) token count.
    assert cfg.training.base_arch == "toto2"
    assert cfg.training.target_train_hours > 0
    assert cfg.training.train_tokens == round(
        cfg.training.target_train_hours * 3600 * cfg.training.ref_throughput_tokens_per_s
    )
    assert cfg.training.head_dim == 64
    assert cfg.training.num_quantiles == len(DEFAULT_QUANTILE_LEVELS)
    # I/O lengths must line up with the eval windows the model is scored on.
    assert cfg.training.horizon == cfg.eval.horizon
    assert cfg.eval.n_windows > 0
    assert cfg.scoring.dethrone_cp >= 1


def test_train_budget_derives_from_hours(cfg):
    from dataclasses import replace

    # train_tokens = hours × 3600 × throughput; warmup_tokens = fraction of that.
    c = replace(cfg.training, target_train_hours=2.0, ref_throughput_tokens_per_s=1000, warmup_fraction=0.1)
    assert c.train_tokens == 2 * 3600 * 1000
    assert c.warmup_tokens == round(c.train_tokens * 0.1)
    # Doubling the hours doubles the enforced compute the data competes under.
    assert replace(c, target_train_hours=4.0).train_tokens == 2 * c.train_tokens


def test_training_contract_digest_covers_recipe(cfg):
    # Every contract field is folded into the digest, so two recipes that differ
    # in the optimiser, the token budget, or the architecture are not "identical
    # terms". This is the controlled-experiment pin for from-scratch training.
    from dataclasses import replace

    from cascade.shared.manifest import contract_digest

    base = contract_digest(cfg.training)
    # Budget is pinned via the hours × throughput fields (train_tokens is derived).
    assert base != contract_digest(replace(cfg.training, target_train_hours=cfg.training.target_train_hours + 1))
    assert base != contract_digest(replace(cfg.training, ref_throughput_tokens_per_s=1))
    assert base != contract_digest(replace(cfg.training, optimizer="adamw"))
    assert base != contract_digest(replace(cfg.training, d_model=cfg.training.d_model * 2))


def test_round_cadence_loads(cfg):
    # Daily-style cadence + two-stage selection knobs.
    assert cfg.round.epoch_blocks > 0
    assert cfg.round.heat_train_hours > 0
    assert cfg.round.heat_train_hours < cfg.training.target_train_hours  # heat is the cheap screen
    assert cfg.round.finalists >= 1
    assert cfg.round.heat_n_windows <= cfg.eval.n_windows


def test_shipped_config_is_single_size_at_launch(cfg):
    # 20M is disabled in the committed chain.toml at launch — rounds run the 4M
    # primary only. Uncomment [[training.sizes]] + repoint throne_sizes to promote.
    assert cfg.training.extra_sizes == ()
    assert [s.arch_preset for s in cfg.training.all_sizes()] == [cfg.training.arch_preset]
    # screen and throne both resolve to the primary at launch.
    assert cfg.screen_contract().arch_preset == cfg.training.arch_preset
    assert [t.arch_preset for t in cfg.throne_contracts()] == [cfg.training.arch_preset]


def test_all_sizes_primary_plus_extra(two_size_cfg):
    cfg = two_size_cfg
    sizes = cfg.training.all_sizes()
    assert len(sizes) == 1 + len(cfg.training.extra_sizes) == 2
    assert sizes[0].arch_preset == cfg.training.arch_preset      # primary first
    assert sizes[0].extra_sizes == ()                            # each is a single concrete size
    presets = [s.arch_preset for s in sizes]
    assert len(presets) == len(set(presets))                     # distinct sizes


def test_screen_and_throne_pointers_resolve_against_registry(two_size_cfg):
    from dataclasses import replace

    # Sliding window: screen at the small primary, throne at the bigger size only.
    cfg = replace(two_size_cfg, round=replace(two_size_cfg.round,
                  screen_size=two_size_cfg.training.arch_preset,
                  throne_sizes=("toto2-test-xl",)))
    assert cfg.screen_contract().arch_preset == cfg.training.arch_preset
    assert [t.arch_preset for t in cfg.throne_contracts()] == ["toto2-test-xl"]
    # screen budget uses the screen size's throughput; throne is the bigger size.
    assert cfg.screen_contract().d_model == cfg.training.d_model
    assert cfg.throne_contracts()[0].d_model == 512


def test_contract_for_unknown_size_raises(cfg):
    import pytest

    with pytest.raises(ValueError, match="unknown size"):
        cfg.training.contract_for("toto2-does-not-exist")


def test_for_size_overrides_only_shape_and_keeps_family_invariants(two_size_cfg):
    cfg = two_size_cfg
    spec = cfg.training.extra_sizes[0]
    sized = cfg.training.for_size(spec)
    # width/depth + digest + throughput come from the spec …
    assert sized.d_model == spec.d_model and sized.num_layers == spec.num_layers
    assert sized.base_arch_digest == spec.base_arch_digest
    assert sized.ref_throughput_tokens_per_s == spec.ref_throughput_tokens_per_s
    # … the family invariants and the budget hours are inherited from [training].
    assert sized.head_dim == cfg.training.head_dim == 64
    assert sized.patch_size == cfg.training.patch_size
    assert sized.target_train_hours == cfg.training.target_train_hours


def test_extra_sizes_change_contract_digest(two_size_cfg):
    # Every size is folded into the one manifest-level contract digest, so the
    # validator's contract gate covers all sizes at once.
    from dataclasses import replace

    from cascade.shared.manifest import contract_digest

    base = contract_digest(two_size_cfg.training)
    assert base != contract_digest(replace(two_size_cfg.training, extra_sizes=()))


def test_tokens_for_hours_uses_per_size_throughput(two_size_cfg):
    cfg = two_size_cfg
    spec = cfg.training.extra_sizes[0]
    sized = cfg.training.for_size(spec)
    assert sized.tokens_for_hours(cfg.round.heat_train_hours) == round(
        cfg.round.heat_train_hours * 3600 * spec.ref_throughput_tokens_per_s
    )


def test_koth_params_builds_from_scoring(cfg):
    params = cfg.koth_params()
    assert isinstance(params, KothParams)
    assert params.win_margin_start == cfg.scoring.win_margin_start
    assert params.win_margin_end == cfg.scoring.win_margin_end
    # DEC-CA-0016: the margin DECAYS with tenure (end < start), and the floor
    # must stay strictly positive — the guardrail the config review enforces.
    assert params.win_margin_end <= params.win_margin_start
    assert params.win_margin_end > 0
    assert params.dethrone_cp == cfg.scoring.dethrone_cp


def test_static_guard_blocks_internal_modules(cfg):
    blocked = cfg.static_guard.blocked
    assert "cascade.trainer" in blocked
    assert "cascade.shared.chain" in blocked
    assert "socket" in blocked


def test_generator_allowlist_has_torch_as_compute_lib_but_no_weights_format(cfg):
    # torch/gpytorch are allowlisted as COMPUTE libraries for GP/kernel priors,
    # but generators are code-only — safetensors (a weights container) is not
    # allowlisted, and shipped weight files are rejected at repo-layout time.
    allowed = {a.lower() for a in cfg.dependencies.allowed}
    assert "torch" in allowed
    assert "safetensors" not in allowed


def test_corpus_mode_is_a_known_mode(cfg):
    from cascade.shared.config import CORPUS_MODES

    assert cfg.training.corpus_mode in CORPUS_MODES


def test_corpus_mode_folded_into_contract_digest(cfg):
    # The feed mode is part of the controlled contract — king and challenger must
    # use the same one — so changing it must change the digest.
    from dataclasses import replace

    from cascade.shared.manifest import contract_digest

    base = contract_digest(cfg.training)
    alt = "cache_reuse" if cfg.training.corpus_mode != "cache_reuse" else "stream_cpu"
    assert base != contract_digest(replace(cfg.training, corpus_mode=alt))


def test_validate_corpus_mode_rejects_unknown():
    import pytest

    from cascade.shared.config import validate_corpus_mode

    with pytest.raises(ValueError):
        validate_corpus_mode("turbo")


def test_validate_dedup_mode_rejects_unknown():
    # A typo used to fall through to "off": an anti-spam screen that silently
    # is not running is exactly the failure nobody notices.
    import pytest

    from cascade.shared.config import validate_dedup_mode

    for bad in ("enfoce", "Enforce", "on", ""):
        with pytest.raises(ValueError):
            validate_dedup_mode(bad)
    assert validate_dedup_mode("enforce") == "enforce"


def test_shipped_config_backs_the_probe_with_hard_isolation(cfg):
    # The dedup probe executes untrusted generator code on the orchestrator,
    # which holds the private eval pool and the trainer's wallet. Shipping the
    # probe on with a degradable sandbox is the one combination that must not
    # occur (cascade.trainer.loop.TrainerRunner._probe_sandbox_ok enforces it
    # at runtime; this pins the shipped file).
    probe_on = (cfg.round.dedup_probe_mode != "off"
                and cfg.round.dedup_probe_series > 0)
    if probe_on and not cfg.round.dedup_probe_allow_weak_sandbox:
        assert cfg.generator.sandbox_mode == "container" or cfg.generator.sandbox_strict


def test_for_hours_scales_budget_and_wall_clock_guard(cfg):
    # A heat/screen contract scales BOTH the token budget and the hard wall-clock
    # guard to its cheap hours budget — a stalling generator costs minutes of a
    # heat slot, never the final's max_train_seconds — with a floor for the fixed
    # overheads and a cap at the pinned contract guard.
    import pytest

    from cascade.shared.config import HEAT_GUARD_FACTOR, HEAT_GUARD_FLOOR_SECONDS

    c = cfg.training.primary_size
    heat = c.for_hours(0.5)
    assert heat.train_tokens == c.tokens_for_hours(0.5)
    assert heat.max_train_seconds == int(round(HEAT_GUARD_FACTOR * 0.5 * 3600))  # 5400
    assert heat.max_train_seconds < c.max_train_seconds
    # tiny testnet budgets hit the overhead floor …
    assert c.for_hours(0.01).max_train_seconds == HEAT_GUARD_FLOOR_SECONDS
    # … and a big semifinal budget never EXCEEDS the pinned contract guard
    assert c.for_hours(10.0).max_train_seconds == c.max_train_seconds
    with pytest.raises(ValueError):
        c.for_hours(0.0)


def test_heat_num_samples_knob(cfg):
    # chain.toml ships the heat screen at the final's sample count (2026-07-27),
    # so the screen ranks on the same estimator the duel decides on;
    # 0 (the default) means "reuse [eval] num_samples".
    from dataclasses import replace

    assert cfg.round.heat_num_samples == cfg.eval.num_samples == 100
    bare = replace(cfg.round, heat_num_samples=0)
    assert (bare.heat_num_samples or cfg.eval.num_samples) == cfg.eval.num_samples


def test_for_hours_guard_knobs_override_defaults(cfg):
    # [round] heat_guard_factor / heat_guard_floor_seconds feed straight through:
    # 1.0 = the budget hours ARE the cap; a bigger factor buys slower-SKU headroom.
    c = cfg.training.primary_size
    assert c.for_hours(0.5).max_train_seconds == 1800                 # 1.0 default
    assert c.for_hours(0.5, guard_factor=3.0).max_train_seconds == 5400
    assert c.for_hours(0.1, guard_floor_seconds=300).max_train_seconds == 360
    # chain.toml ships the owner policy: cap == budget, final cap == 3h budget
    assert cfg.round.heat_guard_factor == 1.0
    assert cfg.training.max_train_seconds == int(cfg.training.target_train_hours * 3600)


# ── scheduled cadence change ([round] epoch_activation_block) ────────────────


def test_effective_epoch_blocks_switches_at_the_activation_block():
    """The whole point of the gate: the length depends on the BLOCK being
    resolved, not on when the process started. Two nodes that restarted on
    opposite sides of the switch must agree on every round."""
    from cascade.shared.config import RoundConfig, effective_epoch_blocks

    r = RoundConfig(epoch_blocks=3600, epoch_blocks_prev=7200,
                    epoch_activation_block=8726400)
    assert effective_epoch_blocks(r, 8726399) == 7200   # last pre-switch block
    assert effective_epoch_blocks(r, 8726400) == 3600   # activation is inclusive
    assert effective_epoch_blocks(r, 8730000) == 3600
    assert effective_epoch_blocks(r, 0) == 7200


def test_effective_epoch_blocks_is_identity_without_a_schedule():
    from cascade.shared.config import RoundConfig, effective_epoch_blocks

    r = RoundConfig(epoch_blocks=3600)
    for b in (0, 1, 8726399, 8726400, 10**9):
        assert effective_epoch_blocks(r, b) == 3600


def test_epoch_start_agrees_across_the_switch_for_pre_switch_rounds():
    """A round that ran under the old cadence keeps its original boundary
    forever — this is what stops an audit disagreeing with the validator that
    wrote the receipt."""
    from cascade.shared.config import RoundConfig, effective_epoch_blocks

    r = RoundConfig(epoch_blocks=3600, epoch_blocks_prev=7200,
                    epoch_activation_block=8726400)
    created = 8722585                       # mid-round, pre-switch
    eb = effective_epoch_blocks(r, created)
    assert (created // eb) * eb == 8719200  # NOT 8722800, which the 3600 grid gives


def test_activation_block_must_be_on_both_grids(tmp_path):
    """A seam that is not a boundary of both grids leaves the round spanning it
    with two different lengths depending on which side you ask from."""
    import shutil

    import pytest

    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    src = DEFAULT_CHAIN_TOML.read_text()
    p = tmp_path / "chain.toml"

    # 8726400 is a multiple of both 7200 and 3600 — accepted.
    p.write_text(src)
    shutil.copy(DEFAULT_CHAIN_TOML, p)
    load_chain_config(p)

    # 8722800 is on the 3600 grid but NOT the 7200 grid — rejected.
    p.write_text(src.replace("epoch_activation_block = 8726400",
                             "epoch_activation_block = 8722800"))
    with pytest.raises(ValueError, match="multiple of BOTH"):
        load_chain_config(p)


def test_cadence_keys_must_be_set_together(tmp_path):
    import pytest

    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    src = DEFAULT_CHAIN_TOML.read_text()
    p = tmp_path / "chain.toml"
    p.write_text(src.replace("epoch_blocks_prev      = 7200",
                             "epoch_blocks_prev      = 0"))
    with pytest.raises(ValueError, match="must be set together"):
        load_chain_config(p)


# ── DEC-CA-0031: points-denominated drain budget + split stall window ────────


def test_shipped_config_arms_points_denomination():
    """The shipped mainnet template arms corpus_target_points at today's
    corpus size in points (16384 × 4096) and splits the streaming stall
    window off the raised batch drain budget at its pre-split value."""
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    c = load_chain_config(DEFAULT_CHAIN_TOML)
    g = c.generator
    assert g.corpus_target_points == 16384 * 4096
    assert g.corpus_target_points <= g.max_total_points
    assert g.max_generate_seconds == 7200
    # The stall window is pinned at the pre-DEC-CA-0031 value: raising the
    # batch budget must not widen how long a stalled generator holds a lane.
    assert g.stream_stall_seconds == 1800
    assert g.effective_stall_seconds == 1800


def test_effective_stall_seconds_falls_back_to_max_generate(cfg):
    from dataclasses import replace

    g = replace(cfg.generator, stream_stall_seconds=0, max_generate_seconds=900)
    assert g.effective_stall_seconds == 900
    assert replace(g, stream_stall_seconds=300).effective_stall_seconds == 300


def test_corpus_target_points_must_fit_the_point_cap(tmp_path):
    import pytest

    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    src = DEFAULT_CHAIN_TOML.read_text()
    p = tmp_path / "chain.toml"
    p.write_text(src.replace("corpus_target_points = 67_108_864",
                             "corpus_target_points = 3_000_000_000"))
    with pytest.raises(ValueError, match="max_total_points"):
        load_chain_config(p)


def test_generator_budget_keys_stay_out_of_contract_digest():
    """DEC-CA-0031's premise: no [generator] key is in contract_digest, so the
    budget re-denomination never forks manifests or needs a validator restart.
    The digest hashes TrainingContractConfig only — assert the armed template's
    training digest is bit-equal to the same [training] with the drain armed
    or not (i.e. the generator block cannot reach it)."""
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config
    from cascade.shared.manifest import contract_digest

    c = load_chain_config(DEFAULT_CHAIN_TOML)
    assert "corpus_target_points" not in {
        f for f in c.training.__dataclass_fields__
    }
    assert "stream_stall_seconds" not in {
        f for f in c.training.__dataclass_fields__
    }
    # And the digest input really is the training dataclass alone.
    assert len(contract_digest(c.training)) == 64
