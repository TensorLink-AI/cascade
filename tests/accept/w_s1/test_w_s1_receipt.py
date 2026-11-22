"""W-S1: the receipt field and TS1's satellite half.

* ``cascade.shared.sentinel.sentinel_record(cfg, era_start_block, block_hash)``
  is ``None`` while unarmed, else the structured record
  ``{"suite", "block", "block_hash"}`` (extra keys allowed) that the main
  validator stamps;
* ``RoundReceipt.sentinel`` / ``build_receipt(sentinel=...)`` carry it as the
  canonical body key ``"sentinel"``, present only when set, through dump and
  load;
* ``cascade.validator.satellite.sentinel_replay(receipt, cfg, client)``
  returns the suite the satellite judges with: the one derived from the
  block hash it reads (``client.block_hash(record["block"])``); the recorded
  one when the hash can't be read (``client`` None, or ``ChainError``); and it
  raises ``ValueError`` when a readable hash contradicts the record. ``None``
  when the receipt records no sentinel.
"""

from __future__ import annotations

import pytest

from .helpers import (
    ALL_SUITES,
    EFFECTIVE,
    ERA,
    SHIPPED,
    armed_cfg,
    body_of,
    load_ok,
    rejected_receipt,
    shipped_raw,
    synth_hash,
    with_sentinel,
)

HASH = synth_hash("era-2698")


class FakeChain:
    """A chain client's ``block_hash`` (the satellite's only chain read here)."""

    def __init__(self, hashes: dict[int, str] | None = None, fail: bool = False) -> None:
        self.hashes = dict(hashes or {})
        self.fail = fail
        self.asked: list[int] = []

    def block_hash(self, block=None) -> str:
        from cascade.shared.chain import ChainError

        self.asked.append(int(block))
        if self.fail or int(block) not in self.hashes:
            raise ChainError("get_block_hash_failed: state pruned / node unreachable")
        return self.hashes[int(block)]


def _record(name: str = "chain.toml", block_hash: str = HASH):
    from cascade.shared.sentinel import sentinel_record

    rec = sentinel_record(armed_cfg(name), EFFECTIVE, block_hash)
    assert rec is not None, "armed config produced no sentinel record"
    return rec


@pytest.mark.parametrize("name", SHIPPED)
def test_ws1_record_shape_when_armed(name):
    from cascade.shared.sentinel import sentinel_for_era

    rec = _record(name)
    assert rec["suite"] == sentinel_for_era(HASH, list(ALL_SUITES))
    assert rec["block"] == EFFECTIVE and rec["block_hash"] == HASH


@pytest.mark.parametrize("name", SHIPPED)
def test_ws1_no_record_when_unarmed_or_before_the_block(name):
    from cascade.shared.sentinel import sentinel_record

    assert sentinel_record(load_ok(shipped_raw(name)), EFFECTIVE, HASH) is None
    assert sentinel_record(armed_cfg(name), EFFECTIVE - ERA[name], HASH) is None


def test_ws1_record_uses_the_configured_suites():
    from cascade.shared.sentinel import sentinel_record

    cfg = armed_cfg("chain.toml", sentinel_suites=["boom", "time"])
    got = {sentinel_record(cfg, EFFECTIVE, synth_hash(i))["suite"] for i in range(64)}
    assert got == {"boom", "time"}


def test_ws1_receipt_carries_the_record_through_dump_and_load():
    from cascade.shared.receipt import dump_receipt, load_receipt, receipt_has_unknown_fields
    from tests.unit.receipt_fixture import make_scored_receipt

    rec = _record()
    plain = make_scored_receipt(None)[0]
    stamped = with_sentinel(plain, rec)
    body = body_of(stamped)
    assert body.get("sentinel") == rec, f"canonical body sentinel = {body.get('sentinel')!r}"
    rest = {k: v for k, v in body.items() if k != "sentinel"}
    assert rest == body_of(plain), "stamping the sentinel moved another receipt field"
    back = load_receipt(dump_receipt(stamped))
    assert back.sentinel == rec
    assert back.canonical_body() == stamped.canonical_body()
    assert receipt_has_unknown_fields(back) is False


def test_ws1_build_receipt_takes_the_sentinel_keyword():
    rec = _record()
    stamped = rejected_receipt(sentinel=rec)
    assert body_of(stamped).get("sentinel") == rec
    assert "sentinel" not in body_of(rejected_receipt(sentinel=None))


# ── satellite replay (TS1) ───────────────────────────────────────────────────


def _stamped(rec):
    from tests.unit.receipt_fixture import make_scored_receipt

    return with_sentinel(make_scored_receipt(None)[0], rec)


def _replay(receipt, cfg, client):
    from cascade.validator.satellite import sentinel_replay

    return sentinel_replay(receipt, cfg, client)


def test_ts1_satellite_verifies_the_record_from_the_block_hash():
    rec = _record()
    chain = FakeChain({EFFECTIVE: HASH})
    assert _replay(_stamped(rec), armed_cfg("chain.toml"), chain) == rec["suite"]
    assert chain.asked == [EFFECTIVE], f"satellite read blocks {chain.asked}"


@pytest.mark.parametrize("client", ["none", "chain-error", "missing-block"])
def test_ts1_satellite_replays_the_recorded_value_when_the_hash_is_unreadable(client):
    rec = _record()
    chain = {"none": None, "chain-error": FakeChain(fail=True),
             "missing-block": FakeChain({EFFECTIVE + 1: HASH})}[client]
    assert _replay(_stamped(rec), armed_cfg("chain.toml"), chain) == rec["suite"]


def test_ts1_satellite_refuses_a_record_the_hash_contradicts():
    """A readable hash that draws another suite, or that differs from the
    recorded hash, is a replay failure, never silently accepted."""
    rec = _record()
    other = next(s for s in ALL_SUITES if s != rec["suite"])
    cfg = armed_cfg("chain.toml")
    with pytest.raises(ValueError):
        _replay(_stamped({**rec, "suite": other}), cfg, FakeChain({EFFECTIVE: HASH}))
    alt = next(synth_hash(f"alt-{i}") for i in range(64)
               if _record(block_hash=synth_hash(f"alt-{i}"))["suite"] != rec["suite"])
    with pytest.raises(ValueError):
        _replay(_stamped(rec), cfg, FakeChain({EFFECTIVE: alt}))


def test_ts1_unreadable_hash_still_checks_the_recorded_suite():
    rec = _record()
    with pytest.raises(ValueError):
        _replay(_stamped({**rec, "suite": "gifteval"}), armed_cfg("chain.toml"), None)


def test_ts1_receipt_without_a_sentinel_replays_none():
    from tests.unit.receipt_fixture import make_scored_receipt

    plain = make_scored_receipt(None)[0]
    assert _replay(plain, load_ok(shipped_raw("chain.toml")), FakeChain(fail=True)) is None
