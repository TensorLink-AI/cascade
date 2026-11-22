"""WB15: row claims + overlap deferral (plan §3.0 lane B track, §12b.2 LB2
"Overlaps", T-rows TB16 and TB21).

The WB15 row, verbatim:

    | WB15 | Row claims + overlap deferral: earliest reveal; both modes coexist
    (declared first, then ESFT excluding held rows) | `status/lane_b.json`
    `deferred: rows_held` | WB14 | ci | `scripts/laned/accept WB15` (TB16) |

TB16, verbatim (§4):

    | TB16 | **(rev4)** Two entries on overlapping rows | The earliest reveal
    holds; the other shows `deferred: rows_held` with the holder and expected
    release |

TB21, verbatim (§4, owned by WB15 per the `| Row | WPs |` table):

    | TB21 | **(rev4)** No declared rows | ESFT selection excluding held rows
    (both modes coexist) |

LB2 "Overlaps" (§12b.2): rows are exclusive against open legs,
finished-but-unsettled legs and pending merges (DEC-CA-0059); claim order is
the earliest reveal; a deferred claim shows `deferred: rows_held` in
`status/lane_b.json` with the holder's entry id and the expected release
(the holder's verdict or merge era). ESFT stays the default when a miner
declares none, and both modes coexist: a declared entry claims its rows
first, then ESFT-selected entries are planned excluding all held rows.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from . import helpers as h
from .helpers import HK_A, HK_B, HK_C, HK_E, HK_F, ROWS_A, ROWS_B, ROWS_C


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


def _held(*sels) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for sel in sels:
        for li, rows in sel.items():
            out.setdefault(int(li), set()).update(rows)
    return out


def _assert_not_charged(sc, hotkey: str) -> None:
    """A deferred claim is neither rejected nor charged: its funded entry
    stays ``queued`` with no attempt, no requeue and no error class, and it
    never went in flight."""
    e = sc.fq.get(hotkey)
    assert e is not None, "the deferred entry was removed from the funding queue"
    assert e.status == "queued", f"deferred entry status {e.status!r} (must stay queued)"
    assert (e.attempts, e.requeues, e.last_error_class) == (0, 0, ""), (
        f"deferred entry was charged/requeued: attempts={e.attempts} requeues={e.requeues} "
        f"class={e.last_error_class!r}")
    assert hotkey not in {x.hotkey for x in sc.fq.in_flight()}
    assert hotkey in {s.hotkey for s in sc.driver.queue.pending}, (
        "a deferred claim stays pending in the driver queue (it proceeds once released)")
    assert not [o for o in sc.driver.st.outcomes if o.get("hotkey") == hotkey], (
        "a deferral is not an outcome (no requeue, no miner fault)")


def _assert_deferred_row(sc, status: dict, hotkey: str, holder: str, *, base_era: int) -> None:
    row = h.deferred_for(status, hotkey)
    assert row.get("deferred") == h.ROWS_HELD, row
    assert row.get("entry_ref") == sc.entry_ref(hotkey), row
    hold = row.get("holder")
    assert isinstance(hold, dict), f"holder must name the holder's entry, got {hold!r}"
    assert hold.get("hotkey") == holder and hold.get("entry_ref") == sc.entry_ref(holder), hold
    rel = row.get("expected_release")
    assert isinstance(rel, dict) and rel.get("kind") in ("verdict", "merge"), (
        f"expected_release is the holder's verdict or merge era, got {rel!r}")
    era = rel.get("era")
    assert isinstance(era, int) and not isinstance(era, bool), rel
    # an open / finished-but-unsettled holder is judged (or expired) within
    # max_base_age_eras of its base era (expert_lane.LaneBQueue.stale)
    assert rel["kind"] == "verdict", rel
    assert base_era <= era <= base_era + int(sc.lb.max_base_age_eras), rel


# ── the plan text these tests implement ──────────────────────────────────────


def test_plan_rows_pinned():
    text = h.PLAN.read_text(encoding="utf-8")
    assert ("| TB16 | **(rev4)** Two entries on overlapping rows | The earliest reveal holds; "
            "the other shows `deferred: rows_held` with the holder and expected release |") in text
    assert ("| TB21 | **(rev4)** No declared rows | ESFT selection excluding held rows "
            "(both modes coexist) |") in text
    assert "| TB16, TB21 | WB15 |" in text
    assert ("| WB15 | Row claims + overlap deferral: earliest reveal; both modes coexist "
            "(declared first, then ESFT excluding held rows) | `status/lane_b.json` "
            "`deferred: rows_held` | WB14 | ci | `scripts/laned/accept WB15` (TB16) |") in text


# ── declared rows are what the leg trains ────────────────────────────────────


def test_declared_entry_trains_exactly_its_rows(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    started = sc.open_gate()
    assert [leg.sub.hotkey for leg in started] == [HK_A]
    (leg,) = started
    assert h.norm(leg.selection) == ROWS_A, (
        f"a declared entry trains its declared rows, not an ESFT pick: {h.norm(leg.selection)}")
    assert leg.funding.get("entry_ref") == sc.entry_ref(HK_A)
    assert sc.fq.get(HK_A).status == "in_flight"
    assert h.deferred_rows(sc.status()) == []


def test_disjoint_declared_entries_both_start(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    sc.fund(HK_C, 30, ROWS_C)
    sc.open_gate()
    legs = sc.open_legs()
    assert set(legs) == {HK_A, HK_C}
    assert h.norm(legs[HK_A].selection) == ROWS_A
    assert h.norm(legs[HK_C].selection) == ROWS_C
    assert h.deferred_rows(sc.status()) == []


# ── TB16: the earliest reveal holds, the other is deferred ───────────────────


def test_tb16_earliest_reveal_holds_and_later_is_deferred(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    sc.fund(HK_B, 20, ROWS_B)
    started = sc.open_gate()
    assert [leg.sub.hotkey for leg in started] == [HK_A], (
        "only the earliest reveal starts on overlapping rows")
    assert h.norm(started[0].selection) == ROWS_A
    _assert_not_charged(sc, HK_B)
    st = sc.status()
    _assert_deferred_row(sc, st, HK_B, HK_A, base_era=1)
    assert [r["hotkey"] for r in h.deferred_rows(st)] == [HK_B], "the holder is not deferred"


def test_tb16_deferred_row_is_in_published_status_lane_b_json(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    sc.fund(HK_B, 20, ROWS_B)
    sc.open_gate()
    doc = sc.published()
    assert doc.get("lane") == "b"
    _assert_deferred_row(sc, doc, HK_B, HK_A, base_era=1)


def test_tb16_funding_and_chain_order_do_not_decide(tmp_path):
    """Seniority is the reveal block, not who funded first or the order the
    chain rows arrive in."""
    sc = h.Scenario(tmp_path)
    sc.fund(HK_B, 20, ROWS_B)               # funds first, reveals later
    sc.fund(HK_A, 10, ROWS_A)
    sc.ops.rows.reverse()
    sc.open_gate()
    assert set(sc.open_legs()) == {HK_A}
    assert h.norm(sc.open_legs()[HK_A].selection) == ROWS_A
    _assert_not_charged(sc, HK_B)
    _assert_deferred_row(sc, sc.status(), HK_B, HK_A, base_era=1)


def test_tb16_reveal_tie_breaks_on_hotkey(tmp_path):
    """Equal reveal blocks: the lower hotkey holds. The plan names only
    "earliest reveal"; the tie rule is the one the codebase already uses for
    reveal seniority (``cascade.funding.queue.select_field``: reveal block,
    then hotkey; ``expert_lane.LaneBQueue.next_dispatchable``: commit block,
    then hotkey)."""
    sc = h.Scenario(tmp_path)
    sc.fund(HK_B, 15, ROWS_B)
    sc.fund(HK_A, 15, ROWS_A)
    sc.open_gate()
    assert set(sc.open_legs()) == {HK_A}
    _assert_deferred_row(sc, sc.status(), HK_B, HK_A, base_era=1)


def test_tb16_open_leg_holds_against_an_earlier_reveal(tmp_path):
    """Rows are exclusive against open legs: a claim revealed earlier but
    funded after the holder started waits for the open leg."""
    sc = h.Scenario(tmp_path)
    sc.fund(HK_B, 20, ROWS_B)
    sc.open_gate()
    assert set(sc.open_legs()) == {HK_B}
    sc.fund(HK_A, 10, ROWS_A)
    assert sc.tick() == []
    _assert_not_charged(sc, HK_A)
    _assert_deferred_row(sc, sc.status(), HK_A, HK_B, base_era=1)


def test_finished_unsettled_leg_still_holds(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    sc.open_gate()
    sc.return_leg(HK_A, pointer=h.KING.replace("king@", "leg-a@"))
    assert not sc.open_legs() and sc.driver.st.finished
    sc.fund(HK_B, 20, ROWS_B)
    assert sc.tick() == []
    _assert_not_charged(sc, HK_B)
    _assert_deferred_row(sc, sc.status(), HK_B, HK_A, base_era=1)


def test_deferral_is_stable_and_uncharged_across_ticks(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    sc.fund(HK_B, 20, ROWS_B)
    sc.open_gate()
    for _ in range(3):
        assert sc.tick() == []
        _assert_not_charged(sc, HK_B)
        _assert_deferred_row(sc, sc.status(), HK_B, HK_A, base_era=1)
    assert len(sc.ops.dispatched) == 1


# ── a released hold lets the deferred claim proceed ─────────────────────────


def test_released_hold_lets_deferred_claim_start_next_tick(tmp_path):
    from cascade.trainer.lane_c_driver import LegFailure

    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    sc.fund(HK_B, 20, ROWS_B)
    sc.open_gate()
    _assert_deferred_row(sc, sc.status(), HK_B, HK_A, base_era=1)
    # the holder's leg ends (a miner fault ends its entry): its rows are free
    sc.return_leg(HK_A, failure=LegFailure("generator crashed", miner_fault=True,
                                           error_class="generator", burn=True))
    assert sc.fq.get(HK_A).status == "failed"
    started = sc.tick()
    assert [leg.sub.hotkey for leg in started] == [HK_B]
    assert h.norm(started[0].selection) == ROWS_B
    assert sc.fq.get(HK_B).status == "in_flight"
    assert h.deferred_rows(sc.status()) == []


# ── TB21: both modes coexist ─────────────────────────────────────────────────


def test_tb21_esft_planned_after_declared_excluding_held_rows(tmp_path):
    """E declares nothing and reveals FIRST; it is still planned after the
    declared entries, excluding the rows they hold (A, C) and nothing else:
    the deferred B holds no rows, so ESFT may take them."""
    sc = h.Scenario(tmp_path)
    sc.fund(HK_E, 5)
    sc.fund(HK_A, 10, ROWS_A)
    sc.fund(HK_B, 20, ROWS_B)
    sc.fund(HK_C, 30, ROWS_C)
    sc.open_gate()
    legs = sc.open_legs()
    assert set(legs) == {HK_A, HK_C, HK_E}
    assert h.norm(legs[HK_A].selection) == ROWS_A
    assert h.norm(legs[HK_C].selection) == ROWS_C
    held = _held(ROWS_A, ROWS_C)
    e = legs[HK_E]
    assert not h.rows_of(e.selection) & h.rows_of(ROWS_A), "ESFT took a declared holder's rows"
    assert not h.rows_of(e.selection) & h.rows_of(ROWS_C), "ESFT took a declared holder's rows"
    assert h.norm(e.selection) == h.norm(h.esft_pick(held)), (
        "ESFT excludes exactly the held rows (declared holders), not a deferred claim's rows")
    assert h.rows_of(e.exclude) >= h.rows_of(ROWS_A) | h.rows_of(ROWS_C), (
        "the ESFT leg records the held rows it was planned under")
    _assert_not_charged(sc, HK_B)
    _assert_deferred_row(sc, sc.status(), HK_B, HK_A, base_era=1)


def test_tb21_esft_after_declared_on_a_later_tick(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.fund(HK_A, 10, ROWS_A)
    sc.open_gate()
    sc.fund(HK_E, 40)
    (e,) = sc.tick()
    assert e.sub.hotkey == HK_E
    assert h.norm(e.selection) == h.norm(h.esft_pick(ROWS_A))
    assert h.rows_of(e.exclude) >= h.rows_of(ROWS_A)


def test_declared_claim_deferred_by_an_esft_leg(tmp_path):
    """An open ESFT leg holds its rows like any other: a later declared claim
    on them is deferred, holder = the ESFT leg."""
    sc = h.Scenario(tmp_path)
    sc.fund(HK_E, 5)
    sc.open_gate()
    (e,) = sc.open_legs().values()
    assert h.norm(e.selection) == h.norm(h.esft_pick({}))
    sc.fund(HK_A, 10, ROWS_A)          # ROWS_A lies inside E's ESFT pick on layer 0
    assert sc.tick() == []
    _assert_not_charged(sc, HK_A)
    _assert_deferred_row(sc, sc.status(), HK_A, HK_E, base_era=1)


# ── invariant 1 (paired, invariant 10) ───────────────────────────────────────


def test_inv1_esft_only_unchanged_when_unarmed(tmp_path):
    """No declared rows anywhere: planning is today's ESFT queue. The first
    entry plans with no exclusions, the second excludes exactly the first's
    open rows, and nothing is deferred."""
    sc = h.Scenario(tmp_path)
    sc.fund(HK_E, 5)
    sc.fund(HK_F, 6)
    sc.open_gate()
    legs = sc.open_legs()
    assert set(legs) == {HK_E, HK_F}
    first = h.esft_pick({})
    assert h.norm(legs[HK_E].selection) == h.norm(first)
    assert legs[HK_E].exclude == {}
    assert h.norm(legs[HK_F].selection) == h.norm(h.esft_pick(first))
    assert h.norm(legs[HK_F].exclude) == h.norm(first)
    assert not (sc.status().get("deferred") or [])


def test_inv1_esft_only_armed_nonempty_lane_d_effect(tmp_path):
    """The pair: with one declared entry the ESFT entries' plan changes (they
    exclude the declared rows) and an overlapping declaration is deferred."""
    sc = h.Scenario(tmp_path)
    sc.fund(HK_E, 5)
    sc.fund(HK_F, 6)
    sc.fund(HK_A, 7, ROWS_A)
    sc.fund(HK_B, 8, ROWS_B)
    sc.open_gate()
    legs = sc.open_legs()
    assert set(legs) == {HK_A, HK_E, HK_F}
    assert h.norm(legs[HK_A].selection) == ROWS_A
    assert h.norm(legs[HK_E].selection) == h.norm(h.esft_pick(ROWS_A))
    assert h.norm(legs[HK_F].selection) == h.norm(
        h.esft_pick(_held(ROWS_A, legs[HK_E].selection)))
    assert h.deferred_rows(sc.status())


# ── determinism ──────────────────────────────────────────────────────────────


def test_plan_is_independent_of_input_order(tmp_path):
    def run(sub, order, reverse_chain):
        sc = h.Scenario(tmp_path / sub)
        spec = {HK_E: (5, None), HK_A: (10, ROWS_A), HK_B: (20, ROWS_B), HK_C: (30, ROWS_C)}
        for hk in order:
            sc.fund(hk, *spec[hk])
        if reverse_chain:
            sc.ops.rows.reverse()
        sc.open_gate()
        st = sc.status()
        return ({hk: h.norm(leg.selection) for hk, leg in sc.open_legs().items()},
                sorted((r.get("hotkey"), str(r.get("holder"))) for r in h.deferred_rows(st)))

    a = run("a", [HK_E, HK_A, HK_B, HK_C], False)
    b = run("b", [HK_C, HK_B, HK_A, HK_E], True)
    c = run("c", [HK_B, HK_E, HK_C, HK_A], False)
    assert a == b == c
    assert a[0][HK_A] == ROWS_A and [d[0] for d in a[1]] == [HK_B]


def test_plan_is_independent_of_hash_seed(tmp_path):
    outs = []
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(h.REPO)}
        p = subprocess.run(
            [sys.executable, "-P", "-m", "tests.accept.wb15.helpers", str(tmp_path / seed)],
            cwd=str(h.REPO), env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-2000:]
        outs.append(p.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1] == outs[2], outs
    import json

    doc = json.loads(outs[0])
    assert doc["open"][HK_A] == {"0": ROWS_A[0]}, doc["open"].get(HK_A)
    assert [r["hotkey"] for r in doc["deferred"]] == [HK_B]
    assert doc["deferred"][0]["deferred"] == h.ROWS_HELD
