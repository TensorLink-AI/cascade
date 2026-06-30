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
def small_cfg(cfg):
    """Shrink the series count so the example generator's python AR(1) loop runs
    fast under test (the length band stays at the chain.toml values, which the
    example generator's own config.json — 128..1024 — fits inside)."""
    from dataclasses import replace

    gen = replace(cfg.generator, corpus_n_series=6)
    return replace(cfg, generator=gen)
