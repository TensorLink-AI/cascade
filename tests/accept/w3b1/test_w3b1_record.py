"""W3b1: the drawn gate suite and its hash are recorded for the lane D report
(§1.3b "Progress board", §11.1) as a ``gate`` dict ``{suite, block,
block_hash}`` built by ``cascade.validator.lane_d_gate.gate_record`` from the
``[lane_d]`` config, so a satellite can re-derive it from the block hash."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from .helpers import ALL_SUITES, BOUNDARY, HASHES, N_HASHES, SHIPPED, lane_d_cfg


def _record(cfg, block, h, sentinel=None):
    from cascade.validator.lane_d_gate import gate_record

    return gate_record(cfg, block, h, sentinel=sentinel)


def _suite_for(cfg, h, sentinel=None):
    from cascade.validator.lane_d_gate import gate_suite_for

    return gate_suite_for(h, gate_suites=cfg.gate_suites, gate_weights=cfg.gate_weights,
                          sentinel=sentinel)


@pytest.mark.parametrize("sentinel", [None, "boom"])
def test_t7b_record_carries_suite_block_and_hash(sentinel):
    cfg = lane_d_cfg()
    for i, h in enumerate(HASHES[:200]):
        block = BOUNDARY + 900 * i
        rec = _record(cfg, block, h, sentinel)
        assert isinstance(rec, dict), f"gate_record returned {type(rec).__name__}"
        assert rec["suite"] == _suite_for(cfg, h, sentinel)
        assert rec["suite"] != sentinel
        assert rec["block"] == block and type(rec["block"]) is int
        assert rec["block_hash"] == h


def test_t7b_record_is_json_round_trippable():
    """The report is JSON: the record survives a canonical dump and load."""
    rec = _record(lane_d_cfg(), BOUNDARY, HASHES[7], "gift")
    text = json.dumps(rec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert json.loads(text) == rec


def test_t7b_record_from_default_config_is_equal_weighted():
    """``LaneDConfig()`` defaults (and the shipped ``[lane_d]``) draw the
    remaining suites equally: weights start equal (§1.3b)."""
    from cascade.shared.config.lane_d import LaneDConfig

    for cfg in (LaneDConfig(), lane_d_cfg()):
        counts = Counter(_record(cfg, BOUNDARY, h, "time")["suite"] for h in HASHES)
        assert counts["time"] == 0
        want = N_HASHES / 3
        for s in ("gift", "boom", "fev"):
            assert abs(counts[s] - want) <= 0.10 * want, dict(counts)


def test_t7b_record_respects_configured_gate_weights():
    """A ``[lane_d] gate_weights`` table that lists two suites (as an overlay
    would set it) restricts the draw to them, in proportion."""
    cfg = lane_d_cfg(gate_weights={"gift": 3.0, "fev": 1.0})
    counts = Counter(_record(cfg, BOUNDARY, h)["suite"] for h in HASHES)
    assert set(counts) == {"gift", "fev"}, dict(counts)
    assert abs(counts["gift"] - 0.75 * N_HASHES) <= 0.10 * 0.75 * N_HASHES, dict(counts)


def test_t7b_record_respects_configured_gate_suites():
    cfg = lane_d_cfg(gate_suites=["boom", "time"], gate_weights={"boom": 1.0, "time": 1.0})
    for h in HASHES[:300]:
        assert _record(cfg, BOUNDARY, h, "boom")["suite"] == "time"


def test_t7b_two_validators_record_the_same_gate():
    """Mainnet and testnet validators (each loading its own shipped file,
    whose ``[lane_d]`` gate keys agree) stamp the same record per hash."""
    cfgs = [lane_d_cfg(name) for name in SHIPPED]
    assert cfgs[0].gate_suites == cfgs[1].gate_suites
    assert cfgs[0].gate_weights == cfgs[1].gate_weights
    for h in HASHES[:200]:
        for sentinel in (None, *ALL_SUITES):
            a, b = (_record(c, BOUNDARY, h, sentinel) for c in cfgs)
            assert a == b, f"hash {h} sentinel {sentinel}: {a!r} vs {b!r}"


@pytest.mark.parametrize("bad", ["", "0x", "0x" + "zz" * 32, "ab" * 32])
def test_t7b_record_refuses_malformed_hash(bad):
    with pytest.raises(ValueError):
        _record(lane_d_cfg(), BOUNDARY, bad)


@pytest.mark.parametrize("bad", [-1, True])
def test_t7b_record_refuses_bad_block(bad):
    with pytest.raises((ValueError, TypeError)):
        _record(lane_d_cfg(), bad, HASHES[0])


def test_t7b_record_refuses_empty_eligible_set():
    cfg = lane_d_cfg(gate_suites=["gift"], gate_weights={"gift": 1.0})
    with pytest.raises(ValueError):
        _record(cfg, BOUNDARY, HASHES[0], "gift")
