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


def test_t18c_a_deregistered_king_slot_burns(tmp_path):
    """W5g's flow crowns X at boundary 2; at boundary 3 X's hotkey has no UID:
    KING_D's court slot burns, typed `deregistered` (B8, with W8)."""
    f = g.Flow(tmp_path, mode2="pass")
    assert g.read_state(f.path)["king"]["digest"] == f.x["digest"], "fixture: X not crowned"
    f.uid_of = {}
    r = f.next("pass", snap=2)
    (king_slot,) = [s for s in r["pay"]["court"] if s["digest"] == f.x["digest"]]
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
