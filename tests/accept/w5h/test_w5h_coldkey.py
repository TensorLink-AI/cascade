"""W5h: the coldkey re-check at reveal (`coldkey_changed`) and the bounded
`coldkey_unresolved` (plan §1.2 D22 option b, T6b). Contract: README.md
"The coldkey re-check at reveal" and "The coldkey_unresolved bound"."""

from __future__ import annotations

import logging

import pytest

from .helpers import (
    ARM_BLOCK,
    COLDKEY_TAG,
    T_X,
    Queue,
    assert_judged,
    assert_skip,
    cold,
    entry,
    lane_d_cfg,
    pool,
    tagged,
)

# ── coldkey_changed ─────────────────────────────────────────────────────────


def test_t6b_a_coldkey_changed_between_ready_and_reveal_is_refused(tmp_path):
    x = entry("chg", coldkey_at_reveal=cold("someone-else"))
    q = Queue(tmp_path)
    r = q.settle(1, [x], pool(tmp_path, 1))
    assert_skip(q, r, x["digest"], "coldkey_changed", "closed")


def test_the_report_carries_the_coldkey_recorded_at_ready(tmp_path):
    """The signed `coldkey` is the value recorded at ready (A8), never the
    changed one."""
    x = entry("chg-rec", coldkey_at_reveal=cold("other"))
    q = Queue(tmp_path)
    r = q.settle(1, [x], pool(tmp_path, 1))
    e = [e for e in r["entries"] if e["digest"] == x["digest"]][0]
    assert e["coldkey"] == x["coldkey_at_ready"]


def test_a_changed_coldkey_entry_frees_its_coldkey_slot(tmp_path):
    """The refused entry is never judged and never counts against a cap: with
    a per-coldkey cap of 1, the coldkey's next entry is judged."""
    k = cold("slot")
    x = entry("chg-a", reveal_ts=T_X, reveal_block=ARM_BLOCK + 60, coldkey=k,
              coldkey_at_reveal=cold("thief"))
    y = entry("chg-b", reveal_ts=T_X + 60, reveal_block=ARM_BLOCK + 65, coldkey=k)
    q = Queue(tmp_path, lane_d_cfg(max_judged_per_coldkey_per_settlement=1))
    r = q.settle(1, [x, y], pool(tmp_path, 1))
    assert_skip(q, r, x["digest"], "coldkey_changed", "closed")
    assert_judged(r, y["digest"])


def test_a_matching_recheck_is_judged(tmp_path):
    x = entry("same")
    x["coldkey_at_reveal"] = x["coldkey_at_ready"]
    q = Queue(tmp_path)
    r = q.settle(1, [x], pool(tmp_path, 1))
    assert_judged(r, x["digest"])


def test_an_entry_without_a_recheck_field_is_judged_as_before(tmp_path):
    """Back-compat: an entry record with no `coldkey_at_reveal` key (every
    earlier WP's fixtures) is selected exactly as before."""
    x = entry("legacy")
    assert "coldkey_at_reveal" not in x
    q = Queue(tmp_path)
    r = q.settle(1, [x], pool(tmp_path, 1))
    assert_judged(r, x["digest"])


def test_a_missing_coldkey_at_ready_still_closes_at_registration(tmp_path):
    """`coldkey_at_ready` empty is a registration refusal (W5g's case): closed
    `coldkey_unresolved` at once, whatever the bound."""
    x = entry("noready", coldkey="")
    q = Queue(tmp_path, lane_d_cfg(max_coldkey_unresolved_settlements=4))
    r = q.settle(1, [x], pool(tmp_path, 1))
    assert_skip(q, r, x["digest"], "coldkey_unresolved", "closed")


# ── coldkey_unresolved: open, alerted, bounded ──────────────────────────────


def test_t6b_an_unreadable_recheck_is_coldkey_unresolved_and_alerted(tmp_path, caplog):
    x = entry("unres", coldkey_at_reveal=None)
    q = Queue(tmp_path, lane_d_cfg(max_coldkey_unresolved_settlements=4))
    caplog.set_level(logging.INFO)
    r = q.settle(1, [x], pool(tmp_path, 1))
    assert_skip(q, r, x["digest"], "coldkey_unresolved", "pending")
    assert tagged(caplog.records, COLDKEY_TAG, x["digest"]), \
        f"no WARNING+ log record starting {COLDKEY_TAG!r} naming {x['digest'][:12]}"


def test_an_unresolved_entry_is_not_charged_and_holds_no_cap(tmp_path):
    k = cold("unres-slot")
    x = entry("unres-a", reveal_ts=T_X, reveal_block=ARM_BLOCK + 60, coldkey=k,
              coldkey_at_reveal=None)
    y = entry("unres-b", reveal_ts=T_X + 60, reveal_block=ARM_BLOCK + 65, coldkey=k)
    q = Queue(tmp_path, lane_d_cfg(max_judged_per_coldkey_per_settlement=1,
                                   max_coldkey_unresolved_settlements=4))
    r = q.settle(1, [x, y], pool(tmp_path, 1))
    assert_skip(q, r, x["digest"], "coldkey_unresolved", "pending")
    assert_judged(r, y["digest"])
    rec = q.state_entry(x["digest"])
    assert rec["used_windows"] == []


def test_an_unresolved_entry_resolved_later_is_judged(tmp_path):
    """A later settlement whose listing carries a readable, matching re-check
    resolves the entry: it is selected like any other."""
    x = entry("unres-late", coldkey_at_reveal=None)
    q = Queue(tmp_path, lane_d_cfg(max_coldkey_unresolved_settlements=4))
    p1 = pool(tmp_path, 1)
    r1 = q.settle(1, [x], p1)
    assert_skip(q, r1, x["digest"], "coldkey_unresolved", "pending")
    resolved = dict(x, coldkey_at_reveal=x["coldkey_at_ready"])
    r2 = q.settle(2, [resolved], p1)
    assert_judged(r2, x["digest"])


def test_an_unresolved_entry_that_resolves_to_another_coldkey_is_changed(tmp_path):
    x = entry("unres-chg", coldkey_at_reveal=None)
    q = Queue(tmp_path, lane_d_cfg(max_coldkey_unresolved_settlements=4))
    p1 = pool(tmp_path, 1)
    q.settle(1, [x], p1)
    r2 = q.settle(2, [dict(x, coldkey_at_reveal=cold("swap"))], p1)
    assert_skip(q, r2, x["digest"], "coldkey_changed", "closed")


@pytest.mark.parametrize("bound", [1, 3])
def test_t6b_coldkey_unresolved_closes_after_the_bound(tmp_path, caplog, bound):
    """With `max_coldkey_unresolved_settlements = N` an unresolved entry is
    held open for N - 1 settlements and closed unjudged at its N-th
    unresolved settlement (each one alerted); it is never judged."""
    x = entry(f"bound{bound}", coldkey_at_reveal=None)
    q = Queue(tmp_path, lane_d_cfg(max_coldkey_unresolved_settlements=bound))
    p1 = pool(tmp_path, 1)
    caplog.set_level(logging.INFO)
    for n in range(1, bound + 1):
        caplog.clear()
        r = q.settle(n, [x], p1)
        want = "closed" if n == bound else "pending"
        assert_skip(q, r, x["digest"], "coldkey_unresolved", want)
        assert tagged(caplog.records, COLDKEY_TAG, x["digest"]), \
            f"settlement {n}: no {COLDKEY_TAG} alert for {x['digest'][:12]}"
    r = q.settle(bound + 1, [x], p1)
    assert x["digest"] not in r["judged"]
    assert q.state_entry(x["digest"])["status"] == "closed"
    assert q.state_entry(x["digest"])["reason"] == "coldkey_unresolved"


def test_the_bound_key_is_parsed():
    """`[lane_d] max_coldkey_unresolved_settlements` (W2; plan default 4)."""
    assert lane_d_cfg().max_coldkey_unresolved_settlements == 4
    assert lane_d_cfg(max_coldkey_unresolved_settlements=2).max_coldkey_unresolved_settlements == 2
