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
    "HF_TOKEN", "WANDB_API_KEY",
)


@pytest.fixture(autouse=True)
def _no_live_credentials(monkeypatch):
    """Scrub storage/API credentials so no test can touch live services."""
    for name in _LIVE_CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def cfg() -> ChainConfig:
    """The mainnet template with its ENFORCING pins neutralized.

    chain.toml carries the real launch pins (expected_gpu, the worker-image
    digest, the go-live commit floor) — those assert real hardware and a real container env, neither of
    which exists under pytest. Blank them HERE, in one place, so the template
    can stay production-true while every fixture-driven test still runs on
    fakes. Tests that exercise the pins set them explicitly via replace().
    """
    from dataclasses import replace

    c = load_chain_config(REPO_ROOT / "chain.toml")
    return replace(c, training=replace(c.training, expected_gpu="",
                                       train_image_digest=""),
                   round=replace(c.round, commit_floor_block=0, dedup_mode="off"))


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
