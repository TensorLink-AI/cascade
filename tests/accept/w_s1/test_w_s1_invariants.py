"""W-S1 TS3: lane A unarmed ⇒ byte-identical receipts, verdicts and config
digests (invariant 1 applies to lane A here), each paired per invariant 10
with an armed test. The tooling's pairing suffix is
``armed_nonempty_lane_d_effect``; here it names the lane A sentinel effect.

The goldens were computed on main (e614e7b) from the shared receipt fixture
(``tests/unit/receipt_fixture.py``), the committed
``tests/fixtures/round_receipt_v3.json`` and a fixed bench-row fixture."""

from __future__ import annotations

import hashlib
import json

import pytest

from .helpers import (
    EFFECTIVE,
    ERA,
    GOLDEN_FACTORS,
    GOLDEN_GATE_DETAIL_SHA,
    GOLDEN_KING_CHECK_DETAIL_SHA,
    GOLDEN_RECEIPT,
    GOLDEN_REJECTED_RECEIPT_SHA,
    GOLDEN_SCORED_RECEIPT_SHA,
    SHIPPED,
    WS1_DEFAULTS,
    WS1_KEYS,
    body_of,
    body_sha,
    comparable,
    consensus_fingerprint,
    detail_sha,
    field,
    gate,
    king_check,
    lane_a_contract_digests,
    load_ok,
    raw_with,
    rejected_receipt,
    scaled_pair,
    shipped_raw,
    synth_hash,
    validate_ok,
    with_sentinel,
)

# ── config ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_a_config_unchanged_when_unarmed(name):
    """Drop-when-default: shipped, every W-S1 key removed (main's [v2]) and
    every default written out parse equal, with equal contract / numerics /
    koth digests, bench rule and mode at the probe blocks."""
    explicit = {k: (list(v) if isinstance(v, tuple) else v) for k, v in WS1_DEFAULTS.items()}
    shipped = load_ok(shipped_raw(name))
    dropped = load_ok(raw_with(name, drop=WS1_KEYS), " (no W-S1 keys)")
    written = load_ok(raw_with(name, explicit), " (explicit defaults)")
    assert comparable(shipped) == comparable(dropped) == comparable(written)
    fp = consensus_fingerprint(dropped)
    assert consensus_fingerprint(shipped) == fp
    assert consensus_fingerprint(written) == fp


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_a_config_armed_nonempty_lane_d_effect(name):
    """One overlay arms the sentinel with the ratchet and tau set: each key
    takes effect and the sentinel is armed from the block; lane A's training
    contract digests and koth params don't move."""
    from cascade.shared.sentinel import sentinel_armed

    base = load_ok(shipped_raw(name))
    over = {"sentinel_from_block": EFFECTIVE, "sentinel_suites": ["boom", "time", "fev"],
            "sentinel_ratchet": True, "sentinel_tau": 0.02}
    armed = validate_ok(base, {"v2": over})
    for k, v in over.items():
        got = field(armed.v2, k)
        assert (list(got) if isinstance(got, tuple) else got) == v, (k, got)
    assert armed.v2 != base.v2
    assert sentinel_armed(armed, EFFECTIVE) and not sentinel_armed(base, EFFECTIVE)
    assert not sentinel_armed(armed, EFFECTIVE - ERA[name])
    assert lane_a_contract_digests(armed) == lane_a_contract_digests(base)
    assert armed.training == base.training and armed.round == base.round


# ── verdicts ─────────────────────────────────────────────────────────────────


def test_inv1_lane_a_bench_gate_unchanged_when_unarmed():
    """The gate's detail on the golden fixture is byte-identical to main, with
    no sentinel keywords and with ``sentinel=None`` (ratchet and tau then
    have no effect)."""
    king, chal = scaled_pair(GOLDEN_FACTORS)
    assert detail_sha(gate(king, chal)) == GOLDEN_GATE_DETAIL_SHA
    for ratchet, tau in ((False, 0.01), (True, 0.01), (True, 0.3)):
        res = gate(king, chal, sentinel=None, sentinel_ratchet=ratchet, sentinel_tau=tau)
        assert detail_sha(res) == GOLDEN_GATE_DETAIL_SHA, (ratchet, tau)


def test_inv1_lane_a_bench_gate_armed_nonempty_lane_d_effect():
    king, chal = scaled_pair(GOLDEN_FACTORS)
    res = gate(king, chal, sentinel="boom", sentinel_ratchet=True, sentinel_tau=0.01)
    assert res.computed and res.passed is True, res.reason
    d = res.detail()
    assert "boom" not in d["suites"] and d["sentinel"]["suite"] == "boom"
    assert detail_sha(res) != GOLDEN_GATE_DETAIL_SHA


def test_inv1_lane_a_king_check_unchanged_when_unarmed():
    init, king = scaled_pair(GOLDEN_FACTORS)
    assert detail_sha(king_check(king, init)) == GOLDEN_KING_CHECK_DETAIL_SHA
    res = king_check(king, init, sentinel=None, sentinel_ratchet=True, sentinel_tau=0.01)
    assert detail_sha(res) == GOLDEN_KING_CHECK_DETAIL_SHA


def test_inv1_lane_a_king_check_armed_nonempty_lane_d_effect():
    init, king = scaled_pair(GOLDEN_FACTORS)
    res = king_check(king, init, sentinel="gift", sentinel_ratchet=False, sentinel_tau=0.01)
    assert res.computed and res.passed is True, res.reason
    assert "gift" not in res.detail()["suites"]
    assert detail_sha(res) != GOLDEN_KING_CHECK_DETAIL_SHA


# ── receipts ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_a_receipt_unchanged_when_unarmed(name):
    """Unarmed, the validator's stamp is ``None`` and every receipt keeps its
    bytes: the shared scored / rejected fixtures, the committed golden
    receipt (dump and load), and a receipt built with ``sentinel=None``."""
    from cascade.shared.sentinel import sentinel_record

    from cascade.shared.receipt import dump_receipt, load_receipt
    from tests.unit.receipt_fixture import make_rejected_receipt, make_scored_receipt

    rec = sentinel_record(load_ok(shipped_raw(name)), EFFECTIVE, synth_hash(0))
    assert rec is None, f"unarmed config stamped {rec!r}"
    scored = make_scored_receipt(None)[0]
    assert body_sha(scored) == GOLDEN_SCORED_RECEIPT_SHA
    assert body_sha(make_rejected_receipt(None)) == GOLDEN_REJECTED_RECEIPT_SHA
    assert body_sha(rejected_receipt(sentinel=rec)) == GOLDEN_REJECTED_RECEIPT_SHA
    assert body_sha(with_sentinel(scored, rec)) == GOLDEN_SCORED_RECEIPT_SHA
    golden = GOLDEN_RECEIPT.read_text(encoding="utf-8")
    loaded = load_receipt(golden)
    assert dump_receipt(loaded) == golden
    assert getattr(loaded, "sentinel", None) is None
    assert "sentinel" not in json.loads(golden)


@pytest.mark.parametrize("name", SHIPPED)
def test_inv1_lane_a_receipt_armed_nonempty_lane_d_effect(name):
    from cascade.shared.sentinel import sentinel_record

    from tests.unit.receipt_fixture import make_scored_receipt

    from .helpers import armed_cfg

    rec = sentinel_record(armed_cfg(name), EFFECTIVE, synth_hash(0))
    assert rec is not None and rec["suite"] in WS1_DEFAULTS["sentinel_suites"]
    stamped = with_sentinel(make_scored_receipt(None)[0], rec)
    assert body_of(stamped)["sentinel"] == rec
    assert hashlib.sha256(stamped.canonical_body()).hexdigest() != GOLDEN_SCORED_RECEIPT_SHA
