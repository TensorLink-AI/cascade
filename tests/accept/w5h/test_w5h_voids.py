"""W5h (rev5): the `deregistered` void (B8; a court slot burns, with W8) and
`ready_expired` (B10, `ready_ttl_blocks`), both typed in state and report,
each freeing its slots (plan §1.5 "State", T18c, T18d). Contract: README.md
"Voids"."""

from __future__ import annotations

from tests.accept.w5g import helpers as g

from .helpers import (
    ARM_BLOCK,
    T_X,
    Queue,
    assert_judged,
    assert_skip,
    cold,
    early_pool,
    entry,
    lane_d_cfg,
    pool,
    uids,
)


def test_t18c_a_pending_entry_whose_hotkey_deregisters_is_voided(tmp_path):
    x = entry("dereg")
    q = Queue(tmp_path)
    early = early_pool(tmp_path, 1)
    r1 = q.settle(1, [x], early)
    assert_skip(q, r1, x["digest"], "pending_floor", "pending")
    r2 = q.settle(2, [x], early, uid_of={})
    assert_skip(q, r2, x["digest"], "deregistered", "closed")


def test_t18c_the_voided_entry_frees_its_coldkey_slot(tmp_path):
    k = cold("dereg-slot")
    x = entry("dereg-a", reveal_ts=T_X, reveal_block=ARM_BLOCK + 60, coldkey=k)
    y = entry("dereg-b", reveal_ts=T_X + 60, reveal_block=ARM_BLOCK + 65, coldkey=k)
    q = Queue(tmp_path, lane_d_cfg(max_judged_per_coldkey_per_settlement=1))
    r = q.settle(1, [x, y], pool(tmp_path, 1), uid_of=uids(y))
    assert_skip(q, r, x["digest"], "deregistered", "closed")
    assert_judged(r, y["digest"])


def _crowned_at_first_verdict(tmp_path):
    """X against the genesis house H0: boundary 1 lists no entry; X enters at
    boundary 2 and is crowned at its first verdict there (plan rev6.11 (b):
    first clear ∧ A10 ∧ a passing §1.3b gate; no confirmation duel). Built
    from W5g's primitives, not W5g's `Flow`, so W5g's fixture chain can move
    without moving this test. Returns `settle_at(n, uid_of)` for boundary 3
    onwards (re-using snapshot 2) and X."""
    cfg = g.lane_d_cfg()
    h0 = g.house()
    x = g.entry("x", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x")}
    snaps = {1: g.snapshot(tmp_path, 1), 2: g.snapshot(tmp_path, 2)}

    def settle_at(n, listed, uid_of, snap=2):
        pool_dir, recs = snaps[snap]
        hash_, bench, _ = g.gate_inputs("pass", x["digest"], h0["digest"], f"b{n}")
        return g.settle(cfg, g.boundary(n), pool_dir, listed, h0,
                        g.forecasts(recs, models), uid_of, path,
                        hash_=hash_, bench=bench).report

    r1 = settle_at(1, [], g.uids(x), snap=1)
    assert r1["judged"] == [], "fixture: boundary 1 lists no entry"
    r2 = settle_at(2, [x], g.uids(x))
    v = g.verdict_of(r2, x["digest"])
    assert v is not None and v["stage"] == "first" and v["cleared"] is True, \
        f"fixture: X did not clear at its first verdict at boundary 2: {v}"
    assert g.read_state(path)["king"]["digest"] == x["digest"], \
        "fixture: X not crowned at its first verdict"
    return settle_at, x


def test_t18c_a_deregistered_king_slot_burns(tmp_path):
    """X is crowned at boundary 2 (its first verdict, rev6.11 (b)); at
    boundary 3 X's hotkey has no UID: KING_D's court slot burns, typed
    `deregistered` (B8, with W8)."""
    settle_at, x = _crowned_at_first_verdict(tmp_path)
    r = settle_at(3, [x], {})
    (king_slot,) = [s for s in r["pay"]["court"] if s["digest"] == x["digest"]]
    assert king_slot["uid"] is None
    assert king_slot["burn_reason"] == "deregistered"


def test_t18d_an_entry_revealed_after_the_ready_ttl_is_ready_expired(tmp_path):
    x = entry("ttl", reveal_block=ARM_BLOCK + 60, ready_block=ARM_BLOCK + 60 - 101)
    q = Queue(tmp_path, lane_d_cfg(ready_ttl_blocks=100))
    r = q.settle(1, [x], pool(tmp_path, 1))
    assert_skip(q, r, x["digest"], "ready_expired", "closed")


def test_t18d_a_reveal_at_the_ttl_edge_is_inside_it(tmp_path):
    """`reveal_block - ready_block == ready_ttl_blocks` is still in time (the
    intake's `reveal_block <= ready_expires_block`, W3)."""
    x = entry("ttl-edge", reveal_block=ARM_BLOCK + 60, ready_block=ARM_BLOCK + 60 - 100)
    q = Queue(tmp_path, lane_d_cfg(ready_ttl_blocks=100))
    r = q.settle(1, [x], pool(tmp_path, 1))
    assert_judged(r, x["digest"])


def test_t18d_the_expired_entry_frees_its_coldkey_slot(tmp_path):
    k = cold("ttl-slot")
    x = entry("ttl-a", reveal_ts=T_X, reveal_block=ARM_BLOCK + 60,
              ready_block=ARM_BLOCK + 60 - 500, coldkey=k)
    y = entry("ttl-b", reveal_ts=T_X + 60, reveal_block=ARM_BLOCK + 65, coldkey=k)
    q = Queue(tmp_path, lane_d_cfg(ready_ttl_blocks=100,
                                   max_judged_per_coldkey_per_settlement=1))
    r = q.settle(1, [x, y], pool(tmp_path, 1))
    assert_skip(q, r, x["digest"], "ready_expired", "closed")
    assert_judged(r, y["digest"])
