"""W5h: FIFO by T under the global and per-coldkey caps; a capped entry is
skipped `coldkey_cap_settlement`, carried and not charged (plan §1.2
"Enforcement", T6b). Contract: README.md "FIFO and the caps"."""

from __future__ import annotations

from .helpers import (
    ARM_BLOCK,
    T_X,
    Queue,
    assert_judged,
    assert_skip,
    cold,
    entry,
    lane_d_cfg,
    pool,
    read_state,
)


def _three_on_one_coldkey():
    k = cold("shared")
    a = entry("a", reveal_ts=T_X, reveal_block=ARM_BLOCK + 60, coldkey=k)
    b = entry("b", reveal_ts=T_X + 60, reveal_block=ARM_BLOCK + 65, coldkey=k)
    c = entry("c", reveal_ts=T_X + 120, reveal_block=ARM_BLOCK + 70, coldkey=k)
    return a, b, c


def test_t6b_three_entries_one_coldkey_two_judged_one_skipped_and_carried(tmp_path):
    """T6b: three entries from one coldkey ready in one settlement, default
    `max_judged_per_coldkey_per_settlement` (2): the two earliest by T are
    judged, the third is `coldkey_cap_settlement`, still open."""
    a, b, c = _three_on_one_coldkey()
    q = Queue(tmp_path)
    assert q.cfg.max_judged_per_coldkey_per_settlement == 2
    # Listed out of T order: selection is by T, never by input order.
    r = q.settle(1, [c, a, b], pool(tmp_path, 1))
    assert_judged(r, a["digest"])
    assert_judged(r, b["digest"])
    assert_skip(q, r, c["digest"], "coldkey_cap_settlement", "pending")


def test_t6b_the_capped_entry_is_not_charged(tmp_path):
    """Skipped and carried, not charged: no frozen-set progress, no used
    windows, no rows, no verdict for the capped entry."""
    a, b, c = _three_on_one_coldkey()
    q = Queue(tmp_path)
    r = q.settle(1, [a, b, c], pool(tmp_path, 1))
    rec = read_state(q.path)["entries"][c["digest"]]
    assert rec["used_windows"] == [], rec["used_windows"]
    fs = rec.get("frozen_set")
    assert fs is None or not fs.get("window_ids"), f"the capped entry advanced: {fs!r}"
    assert not [x for x in r["rows"] if x["digest"] == c["digest"]]


def test_t6b_the_carried_entry_is_judged_once_its_coldkey_has_room(tmp_path):
    """Carried to the next settlement: once the coldkey's two judged entries
    are no longer open (both lose), the carried entry is judged."""
    a, b, c = _three_on_one_coldkey()
    q = Queue(tmp_path)
    worse = {a["digest"]: "bad", b["digest"]: "bad"}
    p1 = pool(tmp_path, 1)
    r1 = q.settle(1, [a, b, c], p1, house_q="good", quality=worse)
    assert_skip(q, r1, c["digest"], "coldkey_cap_settlement", "pending")
    for dg in (a["digest"], b["digest"]):
        assert q.state_entry(dg)["status"] == "lost", \
            f"fixture: {dg[:12]} must lose its duel at boundary 1"
    r2 = q.settle(2, [a, b, c], p1, house_q="good", quality=worse)
    assert_judged(r2, c["digest"])


def test_the_global_cap_selects_fifo_by_t(tmp_path):
    """`max_entries_per_settlement` = 2 over three coldkeys: the two earliest
    reveals are judged whatever the input order and digest order."""
    early = entry("g-early", reveal_ts=T_X, reveal_block=ARM_BLOCK + 60)
    mid = entry("g-mid", reveal_ts=T_X + 60, reveal_block=ARM_BLOCK + 65)
    late = entry("g-late", reveal_ts=T_X + 120, reveal_block=ARM_BLOCK + 70)
    q = Queue(tmp_path, lane_d_cfg(max_entries_per_settlement=2))
    r = q.settle(1, [late, mid, early], pool(tmp_path, 1))
    assert_judged(r, early["digest"])
    assert_judged(r, mid["digest"])
    assert_skip(q, r, late["digest"], "coldkey_cap_settlement", "pending")


def test_the_global_cap_skips_the_latest_even_when_its_digest_sorts_first(tmp_path):
    entries = [entry(f"d{i}", reveal_ts=T_X + 30 * i, reveal_block=ARM_BLOCK + 60 + i)
               for i in range(4)]
    latest = max(entries, key=lambda e: e["reveal_ts"])
    q = Queue(tmp_path, lane_d_cfg(max_entries_per_settlement=3))
    r = q.settle(1, sorted(entries, key=lambda e: e["digest"]), pool(tmp_path, 1))
    for e in entries:
        if e is latest:
            assert_skip(q, r, e["digest"], "coldkey_cap_settlement", "pending")
        else:
            assert_judged(r, e["digest"])


def test_entries_on_distinct_coldkeys_are_all_judged_under_the_caps(tmp_path):
    entries = [entry(f"u{i}", reveal_ts=T_X + 30 * i, reveal_block=ARM_BLOCK + 60 + i)
               for i in range(3)]
    q = Queue(tmp_path)
    r = q.settle(1, entries, pool(tmp_path, 1))
    for e in entries:
        assert_judged(r, e["digest"])
    assert r["skipped"] == []


def test_the_caps_are_in_the_signed_params(tmp_path):
    q = Queue(tmp_path, lane_d_cfg(max_entries_per_settlement=5,
                                   max_judged_per_coldkey_per_settlement=1))
    r = q.settle(1, [entry("p")], pool(tmp_path, 1))
    assert r["params"]["max_entries_per_settlement"] == 5
    assert r["params"]["max_judged_per_coldkey_per_settlement"] == 1
