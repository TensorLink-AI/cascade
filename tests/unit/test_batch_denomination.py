"""[training] batch_denomination (DEC-CA-0041): step-count parity across C.

Under the legacy "series" fill a fixed token budget buys C× fewer optimizer
steps at width C (a C=32 leg collapses from ~152k to ~4.8k updates for the
same tokens). "sequences" fills a (P, C) bucket to ``max(1, batch_size // C)``
series so tokens-per-step stays ~constant and every channel mix earns the
same step count. The load-bearing properties pinned here:

* C = 1 is bit-identical under both denominations (every historical round).
* the field is digest-inert at its default and a deliberate digest bump when
  armed (the drop-when-default convention).
"""
from __future__ import annotations

import numpy as np
import pytest

from cascade.shared.config import validate_batch_denomination
from cascade.trainer.toto2_trainer import iter_training_batches

PS = 8
MAXP = 16
L = PS * MAXP


def _series(n, c, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(c, L)) for _ in range(n)]


def _batches(stream, bs, denom):
    return list(iter_training_batches(
        iter(stream), patch_size=PS, max_ctx_patches=MAXP,
        batch_size=bs, batch_denomination=denom))


# ── equivalence at C = 1 ─────────────────────────────────────────────────────

def test_c1_is_bit_identical_across_denominations():
    stream = _series(130, 1)
    a = _batches(stream, 64, "series")
    b = _batches(stream, 64, "sequences")
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert np.array_equal(x, y)


# ── sequence fill arithmetic ─────────────────────────────────────────────────

@pytest.mark.parametrize("c,expect_fill", [(1, 64), (2, 32), (4, 16), (8, 8),
                                           (16, 4), (32, 2), (5, 12)])
def test_sequences_bucket_fills_to_batch_over_c(c, expect_fill):
    out = _batches(_series(expect_fill * 3, c), 64, "sequences")
    # full buckets first; every full batch holds batch_size // C series
    assert out[0].shape == (expect_fill, c, L)
    assert all(b.shape[0] == expect_fill for b in out)


def test_sequences_tokens_per_step_stays_flat():
    """The property the whole change buys: tokens/step within one C's batch
    never exceeds batch_size sequences and stays >= half of it (the floor-div
    remainder), so a fixed budget yields ~the same step count at every C."""
    for c in (1, 2, 3, 4, 8, 16, 32):
        out = _batches(_series(96, c), 64, "sequences")
        seqs = out[0].shape[0] * c
        assert seqs <= 64
        assert seqs > 64 - c  # floor-division slack only


def test_series_denomination_is_the_legacy_fill():
    out = _batches(_series(64, 32), 64, "series")
    assert out[0].shape == (64, 32, L)  # 2048 sequences: Toto 2's geometry


def test_fill_never_below_one_series():
    # C larger than batch_size still trains, one group per step
    out = _batches(_series(3, 32), 16, "sequences")
    assert all(b.shape[0] == 1 for b in out[:3])


# ── config validation + digest discipline ────────────────────────────────────

def test_validator_rejects_unknown_denomination():
    assert validate_batch_denomination("series") == "series"
    assert validate_batch_denomination("sequences") == "sequences"
    with pytest.raises(ValueError, match="batch_denomination"):
        validate_batch_denomination("tokens")


def test_digest_inert_at_default_and_bumps_when_armed(cfg):
    from dataclasses import replace

    from cascade.shared.manifest import contract_digest, contract_payload

    base = replace(cfg.training, batch_denomination="series")
    armed = replace(cfg.training, batch_denomination="sequences")
    assert "batch_denomination" not in contract_payload(base), (
        "default must drop from the digest payload or every deployed "
        "fleet's contract_digest moves")
    assert contract_payload(armed)["batch_denomination"] == "sequences"
    assert contract_digest(base) != contract_digest(armed)


def test_chain_toml_arms_sequences():
    """The shipped mainnet config arms step-count parity with the C=32 cap."""
    from pathlib import Path

    from cascade.shared.config import load_chain_config

    root = Path(__file__).resolve().parents[2]
    cfg = load_chain_config(root / "chain.toml")
    assert cfg.training.batch_denomination == "sequences"
    assert cfg.generator.max_channels == 32
