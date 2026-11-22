"""WB18: sentinel interaction (plan §3.0 lane B track, §12b.5 LB5, D35;
T-rows TB19, TB20 and TS4).

The WB18 row, verbatim:

    | WB18 | Sentinel interaction (D35/LB5): suite-target deferral,
    `sentinel_max_defer_eras`, no charge on deferral | `status/lane_b.json`
    `deferred: sentinel_target` | W-S1, WB10 | ci | `scripts/laned/accept WB18`
    (TB19–TB20, TS4) |

TB19 / TB20 / TS4, verbatim (§4):

    | TB19 | **(rev4)** Suite-target entry on the current sentinel suite |
    `deferred: sentinel_target`, not charged; judged in a later era |
    | TB20 | **(rev4)** Same entry past `sentinel_max_defer_eras` | Judged on
    its target, logged |
    | TS4 | **(rev4: now in §4)** Lane B entry targeting the sentinel |
    Deferred (D35) |

LB5 (§12b.5): D35 (a) — a suite-target entry whose target is the current
sentinel defers to the next era; a deferred entry shows
`deferred: sentinel_target`; a deferral doesn't consume the miner's per-round
submission: the entry stays queued and uncharged until its target isn't the
sentinel; there's a bound, `sentinel_max_defer_eras` (recommended 4), after
which it's judged on its target regardless, logged.

D35 (§17.4 register): decided — "A sentinel suite is not a lane B target or
guard that era". The sentinel is §16's draw from the era-boundary block hash
(W-S1, ``cascade.shared.sentinel.sentinel_for_era``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from . import helpers as h
from .helpers import BOOM, GIFT, HK_A, HK_B, HK_C, HK_E, ROWS_A, ROWS_B


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _assert_not_charged(sc, hotkey: str) -> None:
    """A deferral doesn't consume the miner's submission: the funded entry
    stays ``queued`` with no attempt, no requeue and no error class, never in
    flight, pending in the driver queue, and no ``outcomes`` row."""
    e = sc.fq.get(hotkey)
    assert e is not None, "the deferred entry was removed from the funding queue"
    assert e.status == "queued", f"deferred entry status {e.status!r} (must stay queued)"
    assert (e.attempts, e.requeues, e.last_error_class) == (0, 0, ""), (
        f"deferred entry was charged/requeued: attempts={e.attempts} requeues={e.requeues} "
        f"class={e.last_error_class!r}")
    assert hotkey not in {x.hotkey for x in sc.fq.in_flight()}
    assert hotkey not in sc.open_legs(), "a sentinel-deferred entry must not start a leg"
    assert hotkey in {s.hotkey for s in sc.driver.queue.pending}, (
        "a deferred entry stays pending in the driver queue (judged in a later era)")
    assert not [o for o in sc.driver.st.outcomes if o.get("hotkey") == hotkey], (
        "a deferral is not an outcome (no requeue, no miner fault)")


def _assert_sentinel_row(sc, status: dict, hotkey: str, target: str) -> None:
    row = h.deferred_for(status, hotkey)
    assert row.get("deferred") == h.SENTINEL_TARGET, (
        f"a suite-target entry on the era's sentinel shows `deferred: sentinel_target`; "
        f"got {row!r}")
    assert row.get("entry_ref") == sc.entry_ref(hotkey, target), row


def _assert_started(sc, hotkey: str, target: str) -> None:
    legs = sc.open_legs()
    assert hotkey in legs, (
        f"{hotkey[:9]} ({target}) should have started; open={sorted(legs)} "
        f"deferred={sc.status().get('deferred')!r}")
    assert tuple(legs[hotkey].sub.declared) == (target,), (
        f"judged on its own target: {legs[hotkey].sub.declared!r}")
    assert legs[hotkey].funding.get("entry_ref") == sc.entry_ref(hotkey, target)
    assert sc.fq.get(hotkey).status == "in_flight"
    assert not [r for r in (sc.status().get("deferred") or []) if r.get("hotkey") == hotkey]


# ── the plan text these tests implement ──────────────────────────────────────


def test_plan_rows_pinned():
    text = h.PLAN.read_text(encoding="utf-8")
    assert ("| WB18 | Sentinel interaction (D35/LB5): suite-target deferral, "
            "`sentinel_max_defer_eras`, no charge on deferral | `status/lane_b.json` "
            "`deferred: sentinel_target` | W-S1, WB10 | ci | `scripts/laned/accept WB18` "
            "(TB19–TB20, TS4) |") in text
    assert ("| TB19 | **(rev4)** Suite-target entry on the current sentinel suite | "
            "`deferred: sentinel_target`, not charged; judged in a later era |") in text
    assert ("| TB20 | **(rev4)** Same entry past `sentinel_max_defer_eras` | "
            "Judged on its target, logged |") in text
    assert ("| TS4 | **(rev4: now in §4)** Lane B entry targeting the sentinel | "
            "Deferred (D35) |") in text
    assert "| TB19, TB20 | WB18 |" in text
    assert "| TS4 | W-S2, WB18 |" in text
    assert ("| D35 | decided | A sentinel suite is not a lane B target or guard that era "
            "| S | owner (default), 2026-11-21 |") in text
    flat = _flat(text)
    for line in (
        "(a) **Accept and document (recommended).** A suite-target entry whose target is "
        "the current sentinel defers to the next era.",
        "A deferred entry shows `deferred: sentinel_target`.",
        "**A deferral doesn't consume the miner's per-round submission:** the entry stays "
        "queued and uncharged until its target isn't the sentinel.",
        "There's a bound, `sentinel_max_defer_eras` (recommended 4), after which it's "
        "judged on its target regardless, logged.",
        "**Lane B:** it's neither a guard nor a target that era (D35).",
        "**TS4:** a lane B entry targeting the sentinel suite is deferred (D35).",
    ):
        assert line in flat, line


def test_shipped_bound_is_four():
    """The recommended bound is the shipped default (WB10's parse)."""
    lb = h.lane_b_cfg(max_defer=None)
    assert lb.sentinel_max_defer_eras == 4
    assert lb.sentinel_target_rule == "defer"


# ── suite targets are lane B slots (DEC-CA-0071 §1) ──────────────────────────


def test_suite_targets_are_slots_with_s_codes():
    """With ``suite_targets = true`` the slot list gains ``suite:gift`` ..
    ``suite:fev`` next to the domains (and horizons), and each commits as
    ``s-<suite>`` and parses back."""
    from cascade.shared.expert_manifest import (
        format_expert_targets,
        parse_expert_targets,
        target_code,
    )

    lb = h.lane_b_cfg()
    targets = tuple(lb.targets)
    for t in h.SUITE_TARGETS:
        assert t in targets, f"{t} missing from the armed slot list {targets!r}"
    assert targets[:2] == ("energy", "web"), targets
    ref = h.ref_of(HK_A)
    for s, t in zip(h.SUITES, h.SUITE_TARGETS, strict=True):
        assert target_code(t) == f"s-{s}", (t, target_code(t))
        payload = format_expert_targets([t], ref, targets, max_targets=1)
        assert f"metro-v1:expert:s-{s}:hippius:" in payload, payload
        assert parse_expert_targets(payload, targets, max_targets=1) == ((t,), ref)


# ── TB19 / TS4: deferral on the era's sentinel ───────────────────────────────


def test_sentinel_target_entry_is_deferred_uncharged(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.set_sentinel(1, "gift")
    sc.fund(HK_A, 10, GIFT)
    started = sc.open_gate()
    assert HK_A not in {leg.sub.hotkey for leg in started}
    _assert_not_charged(sc, HK_A)
    _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)
    pub = sc.published()
    _assert_sentinel_row(sc, pub, HK_A, GIFT)
    assert h.deferred_rows(pub) == h.deferred_rows(sc.status())
    # the draw is W-S1's over the era-start block hash, read from the chain
    assert h.era_start(1) in sc.ops.hash_reads, sc.ops.hash_reads
    assert set(sc.ops.hash_reads) <= {h.era_start(1)}, (
        f"the sentinel is read from the era-start block only: {sc.ops.hash_reads}")
    # repeated ticks in the same era: still deferred, still uncharged
    for _ in range(3):
        assert HK_A not in {leg.sub.hotkey for leg in sc.tick(1)}
    _assert_not_charged(sc, HK_A)
    _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)


def test_non_sentinel_targets_are_unaffected(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.set_sentinel(1, "gift")
    sc.fund(HK_A, 10, GIFT)
    sc.fund(HK_B, 20, BOOM)
    sc.fund(HK_E, 30, "energy")
    sc.open_gate()
    _assert_started(sc, HK_B, BOOM)
    _assert_started(sc, HK_E, "energy")
    _assert_not_charged(sc, HK_A)
    rows = h.deferred_rows(sc.status())
    assert [r.get("hotkey") for r in rows] == [HK_A], rows


def test_rotation_releases_the_deferral(tmp_path):
    """TB19 "judged in a later era": the next era's sentinel is another suite,
    so the entry starts on its own target; a suite-target entry on the NEW
    sentinel is deferred instead."""
    sc = h.Scenario(tmp_path)
    sc.set_sentinel(1, "gift")
    sc.set_sentinel(2, "boom")
    sc.fund(HK_A, 10, GIFT)
    sc.open_gate()
    _assert_not_charged(sc, HK_A)
    sc.fund(HK_B, 20, BOOM)
    sc.tick(2)
    _assert_started(sc, HK_A, GIFT)
    _assert_not_charged(sc, HK_B)
    _assert_sentinel_row(sc, sc.status(), HK_B, BOOM)
    assert [r.get("hotkey") for r in h.deferred_rows(sc.status())] == [HK_B]


def test_sentinel_deferred_claim_holds_no_rows(tmp_path):
    """A declared-rows entry deferred for its sentinel target holds nothing:
    an overlapping later declaration on another target starts with its own
    rows. When the deferral lifts, WB15's row rule then applies to it."""
    sc = h.Scenario(tmp_path, max_defer=4)
    sc.set_sentinel(1, "gift")
    sc.set_sentinel(2, "boom")
    sc.fund(HK_A, 10, GIFT, ROWS_A)
    sc.fund(HK_B, 20, "energy", ROWS_B)
    sc.open_gate()
    _assert_not_charged(sc, HK_A)
    _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)
    _assert_started(sc, HK_B, "energy")
    assert h.norm(sc.open_legs()[HK_B].selection) == ROWS_B
    sc.tick(2)
    row = h.deferred_for(sc.status(), HK_A)
    assert row.get("deferred") == h.ROWS_HELD, (
        f"once its target is no longer the sentinel the claim meets B's held rows: {row!r}")
    _assert_not_charged(sc, HK_A)


# ── TB20: the bound ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("bound", [1, 3])
def test_past_max_defer_eras_judged_on_target_logged(tmp_path, caplog, bound):
    """Deferred in at most ``sentinel_max_defer_eras`` eras (several ticks in
    one era count once); in the next era it starts on its own target although
    that target is still the sentinel, and the trainer logs it."""
    sc = h.Scenario(tmp_path, max_defer=bound)
    for era in range(1, bound + 3):
        sc.set_sentinel(era, "gift")
    sc.fund(HK_A, 10, GIFT)
    sc.open_gate()
    sc.tick(1)
    sc.tick(1)
    for era in range(1, bound + 1):
        if era > 1:
            sc.tick(era)
            sc.tick(era)
        _assert_not_charged(sc, HK_A)
        _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=h.LOGGER):
        sc.tick(bound + 1)
    _assert_started(sc, HK_A, GIFT)
    hits = [r for r in caplog.records
            if r.name == h.LOGGER and r.levelno >= logging.INFO
            and HK_A[:8] in r.getMessage() and "sentinel_max_defer_eras" in r.getMessage()]
    assert hits, ("TB20: the past-bound start is logged (hotkey + sentinel_max_defer_eras); "
                  f"got {[r.getMessage() for r in caplog.records]}")


def test_bound_survives_a_trainer_restart(tmp_path):
    """The deferred-era count is durable driver state: a restart between two
    deferred eras does not reset the bound."""
    sc = h.Scenario(tmp_path, max_defer=2)
    for era in (1, 2, 3):
        sc.set_sentinel(era, "gift")
    sc.fund(HK_A, 10, GIFT)
    sc.open_gate()
    _assert_not_charged(sc, HK_A)
    sc.restart()
    sc.tick(2)
    _assert_not_charged(sc, HK_A)
    _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)
    sc.restart()
    sc.tick(3)
    _assert_started(sc, HK_A, GIFT)


def test_runner_ops_reads_the_chain_block_hash():
    """Production wiring: ``RunnerLaneBOps.block_hash`` is the chain client's
    ``block_hash`` (the read the validators make for the same draw)."""
    from cascade.trainer.lane_b_driver import RunnerLaneBOps

    assert hasattr(RunnerLaneBOps, "block_hash"), "RunnerLaneBOps has no block_hash"
    want = h.hash_drawing("time", 7)
    seen: list = []

    class Client:
        def block_hash(self, block=None):
            seen.append(block)
            return want

    ops = RunnerLaneBOps(SimpleNamespace(cfg=h.chain_cfg(h.lane_b_cfg())), Client(), None)
    assert ops.block_hash(h.era_start(7)) == want
    assert seen == [h.era_start(7)]


# ── determinism ──────────────────────────────────────────────────────────────


def test_plan_is_independent_of_input_order(tmp_path):
    def run(sub, order, reverse_chain):
        sc = h.Scenario(tmp_path / sub, max_defer=1)
        for era, suite in ((1, "gift"), (2, "gift")):
            sc.set_sentinel(era, suite)
        spec = {HK_E: (5, "energy", None), HK_A: (10, GIFT, ROWS_A), HK_B: (20, GIFT, None),
                HK_C: (30, BOOM, None)}
        for hk in order:
            sc.fund(hk, *spec[hk])
        if reverse_chain:
            sc.ops.rows.reverse()
        sc.open_gate()
        first = (sorted(sc.open_legs()),
                 sorted((r.get("hotkey"), r.get("deferred")) for r in h.deferred_rows(sc.status())))
        sc.tick(2)
        second = ({hk: h.norm(leg.selection) for hk, leg in sc.open_legs().items()},
                  sorted((r.get("hotkey"), r.get("deferred"))
                         for r in h.deferred_rows(sc.status())))
        return first, second

    a = run("a", [HK_E, HK_A, HK_B, HK_C], False)
    b = run("b", [HK_C, HK_B, HK_A, HK_E], True)
    c = run("c", [HK_B, HK_E, HK_C, HK_A], False)
    assert a == b == c
    assert a[0][0] == [HK_C, HK_E]
    assert a[0][1] == [(HK_A, h.SENTINEL_TARGET), (HK_B, h.SENTINEL_TARGET)]
    # era 2, past the bound (1): B starts on its target; A's declared rows meet
    # the rows the era-1 ESFT legs hold, so WB15's rows_held rule applies
    assert set(a[1][0]) == {HK_B, HK_C, HK_E}, a[1]
    assert a[1][1] == [(HK_A, h.ROWS_HELD)], a[1]


def test_plan_is_independent_of_hash_seed(tmp_path):
    outs = []
    for seed in ("0", "1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(h.REPO)}
        p = subprocess.run(
            [sys.executable, "-P", "-m", "tests.accept.wb18.helpers", str(tmp_path / seed)],
            cwd=str(h.REPO), env=env, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr[-2000:]
        outs.append(p.stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1] == outs[2], outs
    doc = json.loads(outs[0])
    assert doc["1"]["open"] == sorted([HK_C, HK_E]), doc
    assert [r[:2] for r in doc["1"]["deferred"]] == [[HK_A, h.SENTINEL_TARGET],
                                                     [HK_B, h.SENTINEL_TARGET]], doc
    assert doc["2"]["open"] == sorted([HK_B, HK_C, HK_E]), doc
    assert [r[:2] for r in doc["2"]["deferred"]] == [[HK_A, h.ROWS_HELD]], doc


# ── invariant 1 (paired, invariant 10) ───────────────────────────────────────


def test_inv1_targets_unchanged_when_unarmed():
    """``suite_targets`` at its default: the slot list is the domains, as
    before WB18."""
    lb = h.lane_b_cfg(suite_targets=False, max_defer=None)
    assert tuple(lb.targets) == ("energy", "web")
    assert not [t for t in lb.targets if t.startswith("suite:")]


def test_inv1_targets_armed_nonempty_lane_d_effect():
    lb = h.lane_b_cfg(suite_targets=True)
    assert set(h.SUITE_TARGETS) <= set(lb.targets), lb.targets


def test_inv1_sentinel_unchanged_when_unarmed(tmp_path):
    """Everything at default (no suite targets, ``sentinel_from_block = 0``):
    domain entries plan as today, the chain's block hash is never read and
    nothing is deferred."""
    sc = h.Scenario(tmp_path, suite_targets=False, max_defer=None, sentinel_from_block=0)
    sc.fund(HK_E, 5, "energy")
    sc.fund(HK_C, 6, "web")
    sc.open_gate()
    assert set(sc.open_legs()) == {HK_E, HK_C}
    assert sc.ops.hash_reads == []
    assert not (sc.status().get("deferred") or [])


def test_inv1_sentinel_armed_nonempty_lane_d_effect(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.set_sentinel(1, "gift")
    sc.fund(HK_E, 5, "energy")
    sc.fund(HK_A, 10, GIFT)
    sc.open_gate()
    _assert_started(sc, HK_E, "energy")
    _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)


def test_inv1_suite_target_unchanged_when_unarmed(tmp_path):
    """Suite targets armed, the sentinel not (``sentinel_from_block = 0``): a
    suite-target entry starts on its target, with no block hash read and
    nothing deferred."""
    sc = h.Scenario(tmp_path, sentinel_from_block=0)
    sc.fund(HK_A, 10, GIFT)
    sc.open_gate()
    _assert_started(sc, HK_A, GIFT)
    assert sc.ops.hash_reads == []
    assert sc.status().get("deferred") == []


def test_inv1_suite_target_armed_nonempty_lane_d_effect(tmp_path):
    sc = h.Scenario(tmp_path)
    sc.set_sentinel(1, "gift")
    sc.fund(HK_A, 10, GIFT)
    sc.open_gate()
    _assert_not_charged(sc, HK_A)
    _assert_sentinel_row(sc, sc.status(), HK_A, GIFT)
