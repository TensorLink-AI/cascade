"""chain.toml loads and exposes the expected schema."""

from __future__ import annotations

from metronome.eval.crps import DEFAULT_QUANTILE_LEVELS
from metronome.eval.koth import KothParams


def test_config_loads(cfg):
    assert cfg.schema_version == 1
    assert cfg.subnet.name == "metronome"
    assert cfg.generator.corpus_n_series > 0
    assert cfg.generator.min_length < cfg.generator.max_length
    assert cfg.generator.max_channels >= 1
    # From-scratch Toto2 contract: budgeted by tokens, not epochs.
    assert cfg.training.base_arch == "toto2"
    assert cfg.training.train_tokens > 0
    assert cfg.training.head_dim == 64
    assert cfg.training.num_quantiles == len(DEFAULT_QUANTILE_LEVELS)
    # I/O lengths must line up with the eval windows the model is scored on.
    assert cfg.training.horizon == cfg.eval.horizon
    assert cfg.eval.n_windows > 0
    assert cfg.scoring.dethrone_cp >= 1


def test_training_contract_digest_covers_recipe(cfg):
    # Every contract field is folded into the digest, so two recipes that differ
    # in the optimiser, the token budget, or the architecture are not "identical
    # terms". This is the controlled-experiment pin for from-scratch training.
    from dataclasses import replace

    from metronome.shared.manifest import contract_digest

    base = contract_digest(cfg.training)
    assert base != contract_digest(replace(cfg.training, train_tokens=cfg.training.train_tokens + 1))
    assert base != contract_digest(replace(cfg.training, optimizer="adamw"))
    assert base != contract_digest(replace(cfg.training, d_model=cfg.training.d_model * 2))


def test_koth_params_builds_from_scoring(cfg):
    params = cfg.koth_params()
    assert isinstance(params, KothParams)
    assert params.win_margin_start <= params.win_margin_end
    assert params.dethrone_cp == cfg.scoring.dethrone_cp


def test_static_guard_blocks_internal_modules(cfg):
    blocked = cfg.static_guard.blocked
    assert "metronome.trainer" in blocked
    assert "metronome.shared.chain" in blocked
    assert "socket" in blocked


def test_generator_allowlist_excludes_torch(cfg):
    # Generators emit data; they must not need torch.
    assert "torch" not in {a.lower() for a in cfg.dependencies.allowed}
