"""Make the repo root importable so ``import cascade`` works without an
editable install, and expose shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.shared.config import ChainConfig, load_chain_config  # noqa: E402

# Credentials that turn "fake" code paths real: with these present, code under
# test that falls through to ``open_manifest_store`` (e.g. the receipt-index
# refresh inside ``_publish_round_receipt``) writes to the LIVE buckets. That
# happened on 2026-07-18 — a test's fixture round landed in the production
# ``receipts/index.json`` because the suite ran in a shell with .env sourced.
# Tests that need a credential set it explicitly via monkeypatch.setenv.
_LIVE_CREDENTIAL_ENV = (
    "HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY",
    "HIPPIUS_HUB_TOKEN", "HIPPIUS_TOKEN",
    "HIPPIUS_HUB_USERNAME", "HIPPIUS_REGISTRY_USERNAME",
    "HIPPIUS_HUB_PASSWORD", "HIPPIUS_REGISTRY_PASSWORD",
    "BACKUP_S3_ACCESS_KEY", "BACKUP_S3_SECRET_KEY",
    "HF_TOKEN", "WANDB_API_KEY", "LIUM_API_KEY", "SHADEFORM_API_KEY",
    "VAST_API_KEY", "RUNPOD_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_live_credentials(monkeypatch):
    """Scrub storage/API credentials so no test can touch live services."""
    for name in _LIVE_CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _no_secrets_marker():
    """A test that calls a trainer / validator ``main`` marks the pytest
    process as holding secrets (cascade.shared.sandbox_policy); never let
    that leak into later tests that run generator code."""
    from cascade.shared.sandbox_policy import clear_holds_secrets

    yield
    clear_holds_secrets()


@pytest.fixture(autouse=True)
def _isolate_default_settlement_journal(tmp_path, monkeypatch):
    """Legacy loop fakes stub champion persistence but retain its default path.

    Keep the new undo journal in their temporary directory too. Recovery tests
    with an explicit state bundle path use the real production path unchanged.
    """
    from cascade.validator import settlement_transaction

    original = settlement_transaction.journal_path

    def isolated(runner):
        path = original(runner)
        if Path(runner.cfg.validator.state_db_path) == Path("validator_state.db"):
            return tmp_path / path.name
        return path

    monkeypatch.setattr(settlement_transaction, "journal_path", isolated)


@pytest.fixture(scope="session")
def cfg() -> ChainConfig:
    """The mainnet template with its ENFORCING pins neutralized.

    chain.toml carries the real launch pins (expected_gpu, the worker-image
    digest, the go-live commit floor, the validator_hotkey audit trust
    anchor) — those assert real hardware, a real container env, post-launch
    block heights, and a mainnet signer, none of which exists under pytest.
    Blank them HERE, in one place, so the template can stay production-true
    while every fixture-driven test still runs on fakes. Tests that exercise
    the pins set them explicitly via replace().
    """
    from dataclasses import replace

    c = load_chain_config(REPO_ROOT / "chain.toml")
    # Tests using fake block clocks also reparse raw configuration (satellites,
    # overlays). Neutralize the real calendar there as well as on cfg.v2.
    import copy

    raw = copy.deepcopy(c.raw)
    raw["v2"]["launch_at_utc"] = ""
    raw["netconfig"]["owner_hotkey"] = ""
    return replace(c, training=replace(c.training, expected_gpu="",
                                       train_image_digest=""),
                   raw=raw,
                   netconfig=replace(c.netconfig, owner_hotkey=""),
                   v2=replace(c.v2, launch_at_utc="", prelaunch_burn_readiness=""),
                   # corpus_target_points ships ARMED (DEC-CA-0031): a points-
                   # denominated drain would override every fixture's small
                   # corpus_n_series with a 67M-point draw. Tests that exercise
                   # the points path arm it explicitly via replace().
                   # Generic fixtures replay v1/univariate paths. New-v2
                   # admission tests load the unmodified release policy.
                   generator=replace(c.generator, corpus_target_points=0, min_channels=1,
                                     sandbox_strict=False),  # isolated trusted fixture generators
                   # The scheduled cadence pair and the ARMED funded config
                   # are real-chain state: under pytest's tiny block
                   # numbers the pair resolves every round to the OLD grid, and
                   # funded_mode="required" would route every fixture through
                   # the funded queue. Neutralize like the pins;
                   # funded tests arm their own RoundConfig.
                   round=replace(c.round, commit_floor_block=0, dedup_mode="off",
                                 epoch_blocks_prev=0, epoch_activation_block=0,
                                 funded_mode="off",
                                 funded_pods="off", funded_king_rent=False),
                   # a scheduled fresh-king margin change ships armed: pin the
                   # steady value so fixtures judge every block the same way
                   scoring=replace(c.scoring, win_margin_start_prev=0.0,
                                   margin_activation_block=0),
                   manifest=replace(c.manifest, validator_hotkey=""))


@pytest.fixture(scope="session")
def example_generator_dir() -> Path:
    return REPO_ROOT / "scripts" / "example_generator"


@pytest.fixture()
def two_size_cfg(cfg):
    """A config with a second (synthetic) size in the registry AND a combined
    throne over both sizes, so the multi-size final + combined-throne path stays
    under test even though the shipped chain.toml runs 4M-only at launch."""
    from dataclasses import replace

    from cascade.shared.config import SizeSpec

    spec = SizeSpec(
        arch_preset="toto2-test-xl", base_arch_digest="f" * 64,
        d_model=512, num_layers=8, num_heads=8, mlp_expansion=2,
        ref_throughput_tokens_per_s=90_000,
    )
    training = replace(cfg.training, extra_sizes=(spec,))
    rnd = replace(cfg.round, throne_sizes=(cfg.training.arch_preset, "toto2-test-xl"))
    return replace(cfg, training=training, round=rnd)


@pytest.fixture()
def small_cfg(cfg):
    """Shrink the series count so the example generator's python AR(1) loop runs
    fast under test (the length band stays at the chain.toml values, which the
    example generator's own config.json — 128..1024 — fits inside)."""
    from dataclasses import replace

    gen = replace(cfg.generator, corpus_n_series=6)
    return replace(cfg, generator=gen)
