"""#933 (launch-critical): the trainer resolves lane A's king from SIGNED
sources only — the validators' receipt trail, then its own persisted era king
— and HOLDS when neither exists. It never adopts the top-incentive hotkey:
after the 2026-11-25 launch (A 0 / B 35 / C 15 / D 50) that hotkey is a lane
B/C/D payee, and adopting it restarted the king leg for lane D's king on
testnet (era 14199, every round rejected so no receipt named a king).

Runs the real :class:`RollingScheduler` against the unit suite's fake
``LegOps`` (no pods, no buckets, no torch)."""

from __future__ import annotations

# A lane D payee: top on-chain incentive, never lane A's king.
LANE_D_PAYEE = "LANED"


def _setup(tmp_path, *, receipt: str | None, chain: str | None):
    from tests.accept.w_cmp2.helpers import fixture_cfg
    from tests.unit.test_rolling import REF, Clock, FakeClient, _armed, _commit, _sched

    armed = _armed(fixture_cfg(tmp_path / "cfg"))
    clock = Clock()
    sched, ops = _sched(armed, tmp_path / "work", clock)
    ops.king_hk, ops.chain_king = receipt, chain
    q = ops.queue()
    era_start = armed.round.rolling_from_block
    ops.commits.append(_commit("ALFA", REF["ALFA"], era_start - 100))
    q.add("ALFA", REF["ALFA"], reveal_block=era_start - 100)
    return armed, clock, sched, ops, FakeClient(), era_start


def _run_two_settlements(tmp_path, *, silent_after_first: bool):
    """Two settlements with a lane D payee on top of the incentive. The
    receipt names KING at the first; with ``silent_after_first`` every later
    receipt names no king (a rejected round)."""
    from tests.unit.test_rolling import EB, REF, _advance, _commit, _join

    armed, clock, sched, ops, client, era_start = _setup(tmp_path, receipt="KING",
                                                         chain=LANE_D_PAYEE)
    sched.tick(client, era_start + 5)
    _join(sched)
    b1 = era_start + EB
    _advance(clock, ops, sched, client, from_block=era_start + 5, to_block=b1 + 1)
    # a second challenger for the second settlement
    ops.commits.append(_commit("BRAV", REF["BRAV"], b1))
    ops.queue().add("BRAV", REF["BRAV"], reveal_block=b1)
    if silent_after_first:
        ops.king_hk = None
    sched.tick(client, b1 + 2)
    _join(sched)
    _advance(clock, ops, sched, client, from_block=b1 + 2, to_block=era_start + 2 * EB + 1)
    return sched, ops


def test_w_launch_no_receipt_keeps_the_persisted_lane_a_king(tmp_path):
    """A settlement whose receipts name no king (rejected round, validator
    outage) keeps the era's persisted king — through the rest of the era AND
    across the next era roll — while a lane D payee holds the top incentive."""
    from tests.unit.test_rolling import EB, _advance, _join

    armed, clock, sched, ops, client, era_start = _setup(tmp_path, receipt="KING", chain="KING")
    sched.tick(client, era_start + 5)
    _join(sched)
    assert sched.state.current.king_hotkey == "KING"
    # every receipt from here on names no king; lane D's payee tops incentive
    ops.king_hk, ops.chain_king = None, LANE_D_PAYEE
    block = era_start + 5
    era_blocks = EB * int(armed.round.era_settlements)
    for nxt in (era_start + EB + 1, era_start + 2 * EB + 1, era_start + era_blocks + 7):
        _advance(clock, ops, sched, client, from_block=block, to_block=nxt)
        block = nxt
        assert sched.state.current.king_hotkey == "KING", (
            f"block {nxt}: lane A king switched to {sched.state.current.king_hotkey!r}")
    assert sched.state.current.index > 0 and block >= era_start + era_blocks
    assert all(call[0] == "KING" for call in ops.king_calls), ops.king_calls
    assert LANE_D_PAYEE not in {e.miner_hotkey for m in ops.manifests for e in m.entries}
    assert all(m.entry_for_role("king") is None
               or m.entry_for_role("king").miner_hotkey == "KING" for m in ops.manifests)


def test_w_launch_top_incentive_payee_never_becomes_king(tmp_path):
    """No receipt, top incentive on a lane B/C/D payee: the direct resolver
    returns the persisted king, never the payee."""
    from tests.unit.test_rolling import _join

    armed, clock, sched, ops, client, era_start = _setup(tmp_path, receipt="KING", chain="KING")
    sched.tick(client, era_start + 5)
    _join(sched)
    ops.king_hk, ops.chain_king = None, LANE_D_PAYEE
    assert sched._king_hotkey(client) == "KING"
    sched._adopt_dethrone(client)
    assert sched.state.current.king_hotkey == "KING"


def test_w_launch_first_run_without_receipt_or_state_holds(tmp_path):
    """First-ever run (no persisted era, no receipt): HOLD — no king, no king
    leg, an ERROR-level alert — never the top-incentive hotkey."""
    from tests.unit.test_rolling import _join

    armed, clock, sched, ops, client, era_start = _setup(tmp_path, receipt=None,
                                                         chain=LANE_D_PAYEE)
    assert sched.state.current is None
    assert sched._king_hotkey(client) is None
    sched.tick(client, era_start + 5)
    _join(sched)
    assert not sched.state.current.king_hotkey
    assert ops.king_calls == []
    alerts = sched.r.__dict__.get("_money_alerts") or {}
    assert ("lane_a_king_unresolved", "lane_a") in alerts
    assert alerts[("lane_a_king_unresolved", "lane_a")]["severity"] == "critical"
    # a signed receipt naming the king clears the hold and the alert
    ops.king_hk = "KING"
    assert sched._king_hotkey(client) == "KING"
    assert ("lane_a_king_unresolved", "lane_a") not in (sched.r.__dict__.get("_money_alerts")
                                                        or {})


def test_w_launch_manifests_byte_identical_with_and_without_receipt_king(tmp_path):
    """Normal case unchanged: with the receipt naming the king at every
    settlement, the king is KING whatever the incentive says. A run whose
    later receipts name no king publishes byte-identical manifests — the
    persisted king, never the top-incentive lane D payee."""
    _, ops_a = _run_two_settlements(tmp_path / "a", silent_after_first=False)
    _, ops_b = _run_two_settlements(tmp_path / "b", silent_after_first=True)
    assert len(ops_a.manifests) == len(ops_b.manifests) == 2
    assert all(m.entry_for_role("king").miner_hotkey == "KING" for m in ops_a.manifests)
    assert ([m.canonical_body() for m in ops_a.manifests]
            == [m.canonical_body() for m in ops_b.manifests])
