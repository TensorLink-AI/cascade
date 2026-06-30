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


@pytest.fixture(scope="session")
def cfg() -> ChainConfig:
    return load_chain_config(REPO_ROOT / "chain.toml")


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
