"""W5h: `[lane_d] max_reveal_delay_days` closes an entry still pending at the
limit, unjudged, logged, its slots freed (plan §1.5 "Maximum reveal delay",
T18b). Contract: README.md "The reveal-delay close"."""

from __future__ import annotations

import logging

from .helpers import (
    ARM_BLOCK,
    BLOCKS_PER_DAY,
    SETTLEMENT,
    T_X,
    Queue,
    assert_judged,
    assert_skip,
    boundary,
    cold,
    early_pool,
    entry,
    lane_d_cfg,
    pool,
)

# x is revealed exactly at boundary(1); with max_reveal_delay_days = 1 its
# deadline is reveal_block + 7200 = boundary(9) (8 settlements of 900).
REVEAL = ARM_BLOCK + SETTLEMENT
DEADLINE_N = 9


def _x(**over):
    return entry("late", reveal_ts=T_X, reveal_block=REVEAL, ready_block=REVEAL - 20, **over)


def test_fixture_deadline_lands_on_a_boundary():
    assert boundary(DEADLINE_N) == REVEAL + 1 * BLOCKS_PER_DAY
    assert boundary(DEADLINE_N - 1) < REVEAL + BLOCKS_PER_DAY


def test_t18b_an_entry_pending_at_the_limit_closes_reveal_delay_exceeded(tmp_path, caplog):
    x = _x()
    q = Queue(tmp_path, lane_d_cfg(max_reveal_delay_days=1))
    early = early_pool(tmp_path, 1)
    r2 = q.settle(2, [x], early)
    assert_skip(q, r2, x["digest"], "pending_floor", "pending")
    r8 = q.settle(DEADLINE_N - 1, [x], early)
    assert_skip(q, r8, x["digest"], "pending_floor", "pending")
    caplog.set_level(logging.INFO)
    r9 = q.settle(DEADLINE_N, [x], early)
    assert_skip(q, r9, x["digest"], "reveal_delay_exceeded", "closed")
    logged = [rec.getMessage() for rec in caplog.records
              if rec.levelno >= logging.WARNING and rec.getMessage().startswith("lane_d_")
              and "reveal_delay_exceeded" in rec.getMessage()
              and x["digest"][:12] in rec.getMessage()]
    assert logged, "the reveal-delay close is not logged (WARNING+, a lane_d_ tag, the " \
                   "reason and the entry's first 12 hex)"


def test_t18b_the_close_comes_before_judging_and_frees_the_coldkey_slot(tmp_path):
    """At the limit the entry closes even if fresh windows would now freeze it,
    and its coldkey slot goes to the coldkey's next entry (cap 1)."""
    k = cold("delay")
    x = _x(coldkey=k)
    y = entry("late-y", reveal_ts=T_X + 60, reveal_block=REVEAL + 100,
              ready_block=REVEAL + 80, coldkey=k)
    q = Queue(tmp_path, lane_d_cfg(max_reveal_delay_days=1,
                                   max_judged_per_coldkey_per_settlement=1))
    q.settle(2, [x], early_pool(tmp_path, 1))
    r = q.settle(DEADLINE_N, [x, y], pool(tmp_path, 2))
    assert_skip(q, r, x["digest"], "reveal_delay_exceeded", "closed")
    assert_judged(r, y["digest"])


def test_an_entry_inside_the_limit_is_judged(tmp_path):
    x = _x()
    q = Queue(tmp_path, lane_d_cfg(max_reveal_delay_days=1))
    q.settle(2, [x], early_pool(tmp_path, 1))
    r = q.settle(DEADLINE_N - 1, [x], pool(tmp_path, 2))
    assert_judged(r, x["digest"])


def test_the_limit_follows_the_key(tmp_path):
    """Two days: boundary(9) is inside the limit, nothing closes."""
    x = _x()
    q = Queue(tmp_path, lane_d_cfg(max_reveal_delay_days=2))
    early = early_pool(tmp_path, 1)
    q.settle(2, [x], early)
    r = q.settle(DEADLINE_N, [x], early)
    assert_skip(q, r, x["digest"], "pending_floor", "pending")


def test_the_delay_key_is_parsed():
    """`[lane_d] max_reveal_delay_days` (W2; plan default 14)."""
    assert lane_d_cfg().max_reveal_delay_days == 14
    assert lane_d_cfg(max_reveal_delay_days=3).max_reveal_delay_days == 3
