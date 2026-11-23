"""W5g: the exact typed reason per case (plan rev6 §3.2 invariant 11:
"`pending`, `hold` and `burn` always carry a typed reason enum, and tests
assert the exact reason"; W5g Accept: "asserts the exact reason enum per
case"). README.md has the contract.

Every case is a real `settle_lane_d` run. For each, the entry's
`(status, reason)` is asserted by exact value in the state file and in the
signed report's `entries`, identically; a reason the report also lists as a
skip appears in `skipped` exactly once with the same value; every report
validates against `lane-d-report/1`.

`test_every_reason_the_settlement_can_emit_has_a_case` reads
`cascade/validator/lane_d.py` and fails on an emitted reason no case here
reaches; `test_every_enum_reason_is_cased_or_owned_elsewhere` accounts for
every value of the enum.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from .helpers import (
    DAY,
    T_X,
    Flow,
    boundary,
    entry,
    entry_of,
    forecasts,
    gate_inputs,
    house,
    lane_d_cfg,
    read_state,
    settle,
    skips_of,
    snapshot,
    uids,
)

REPO = Path(__file__).resolve().parents[3]

# Enum values no settlement path emits yet; their WP pins them (plan §3.0).
OWNED_ELSEWHERE = {
    "coldkey_changed": "W5h (coldkey re-check at reveal, A8, T6b)",
    "reveal_delay_exceeded": "W5h (max_reveal_delay_days close, A9, T18b)",
    "mirror_lag_exceeded": "W5a (max_mirror_lag_hours, A1)",
}


class One:
    """One settlement at boundary(n) on snapshot 1 with a healthy bench."""

    def __init__(self, tmp_path, entries, *, cfg=None, models=None, uid_of=None, fc_edit=None,
                 n=1, armed=None):
        self.cfg = cfg or lane_d_cfg()
        self.h0 = house()
        self.path = tmp_path / "state" / "lane_d_state.json"
        pool1, recs1 = snapshot(tmp_path, 1)
        if models is None:
            models = {self.h0["digest"]: ("bad", "house-0")}
            for e in entries:
                models[e["digest"]] = ("sharp", e["digest"][:8])
        fc = forecasts(recs1, models)
        if fc_edit is not None:
            fc_edit(fc)
        hash_, bench, _ = gate_inputs("pass", entries[0]["digest"], self.h0["digest"], "b1")
        self.report = settle(self.cfg, boundary(n), pool1, entries, self.h0, fc,
                             uids(*entries) if uid_of is None else uid_of, self.path,
                             hash_=hash_, bench=bench, armed=armed).report


def _x(**over):
    return {**entry("x", reveal_ts=T_X), **over}


# Each case returns (state_path, report, digest).


def case_pending_floor(tmp_path):
    x = _x(reveal_ts=T_X + 60 * DAY)          # after every window: never frozen
    r = One(tmp_path, [x])
    return r.path, r.report, x["digest"]


def case_coldkey_cap_settlement(tmp_path):
    a = _x()
    b = {**entry("b", reveal_ts=T_X + 60), "coldkey_at_ready": a["coldkey_at_ready"]}
    r = One(tmp_path, [a, b], cfg=lane_d_cfg(max_judged_per_coldkey_per_settlement=1))
    return r.path, r.report, b["digest"]


def case_coldkey_unresolved(tmp_path):
    x = _x(coldkey_at_ready="")
    r = One(tmp_path, [x])
    return r.path, r.report, x["digest"]


def case_king_unscored(tmp_path):
    x = _x()
    r = One(tmp_path, [x], models={x["digest"]: ("sharp", "x")})   # no house forecasts
    return r.path, r.report, x["digest"]


def case_deregistered(tmp_path):
    x = _x()
    r = One(tmp_path, [x], uid_of={})
    return r.path, r.report, x["digest"]


def case_ready_expired(tmp_path):
    x = _x()
    r = One(tmp_path, [x], cfg=lane_d_cfg(ready_ttl_blocks=10))      # revealed 20 blocks late
    return r.path, r.report, x["digest"]


def case_duplicate_digest(tmp_path):
    x = _x(digest=house()["digest"])
    r = One(tmp_path, [x], models={house()["digest"]: ("bad", "house-0")})
    return r.path, r.report, x["digest"]


def case_hotkey_has_open_entry(tmp_path):
    a = _x()
    b = {**entry("b", reveal_ts=T_X + 60), "hotkey": a["hotkey"]}
    r = One(tmp_path, [a, b])
    return r.path, r.report, b["digest"]


def case_context_not_armed(tmp_path):
    x = _x(context=1234)
    r = One(tmp_path, [x])
    return r.path, r.report, x["digest"]


def _armed_record(armed, contexts=(4096, 8192, 16384), mls=300) -> dict:
    """A consistent W5d armed record (written by hand, no W5d import): armed
    variants pass both gates, the others fail the readiness gate."""
    variants = [{"context": c, "series": 4633 if c in armed else 12,
                 "sources": 42 if c in armed else 3, "t7_fits": True, "armed": c in armed}
                for c in contexts]
    return {"armed_contexts": [c for c in contexts if c in armed],
            "evidence": {"min_long_series": mls, "min_sources": 20,
                         "readiness_sha256": "a1" * 32, "t7_sha256": "b2" * 32,
                         "variants": variants}}


def case_context_below_armed(tmp_path):
    """Plan rev6.15 (owner 2026-11-23, "Highest ≤ declared, else skip"):
    declared 4096 with only 8192 and 16384 armed. The entry is not judged and
    stays pending (W7a owns the producer side; this is the reason's case)."""
    x = _x(context=4096)
    r = One(tmp_path, [x], armed=_armed_record((8192, 16384)))
    return r.path, r.report, x["digest"]


def case_checkpoint_fault_forecasts(tmp_path):
    x = _x()

    def poison(fc):
        wid = sorted(fc[x["digest"]])[0]
        fc[x["digest"]][wid] = np.full_like(fc[x["digest"]][wid], np.nan)

    r = One(tmp_path, [x], fc_edit=poison)
    return r.path, r.report, x["digest"]


def case_lane_closed(tmp_path):
    x = _x()
    # until_block must be an era boundary: boundary(4) is one (4 settlements/era).
    r = One(tmp_path, [x], cfg=lane_d_cfg(until_block=boundary(4)), n=4)
    return r.path, r.report, x["digest"]


def case_lost_duel(tmp_path):
    x = _x()
    r = One(tmp_path, [x], models={house()["digest"]: ("good", "house-0"),
                                   x["digest"]: ("bad", "x")})
    return r.path, r.report, x["digest"]


def _flow(tmp_path, mode2):
    f = Flow(tmp_path, mode2=mode2)
    return f.path, f.reports[2], f.x["digest"]


def case_bench_hold(tmp_path):
    return _flow(tmp_path, "host_down")


def case_bench_hold_regate(tmp_path):
    f = Flow(tmp_path, mode2="no_bench")
    return f.path, f.next("reference_fault"), f.x["digest"]


def case_bench_hold_requeued(tmp_path):
    f = Flow(tmp_path, mode2="no_bench", cfg=lane_d_cfg(max_bench_holds=1))
    return f.path, f.next("host_down"), f.x["digest"]


def case_checkpoint_fault_gate(tmp_path):
    return _flow(tmp_path, "candidate_fault")


def case_checkpoint_fault_regate(tmp_path):
    f = Flow(tmp_path, mode2="no_bench")
    return f.path, f.next("candidate_fault"), f.x["digest"]


def case_gate_regression(tmp_path):
    return _flow(tmp_path, "regress")


def case_crowned(tmp_path):
    return _flow(tmp_path, "pass")


def case_lowfreq_regression(tmp_path):
    """W3b5's T4b fixture (tests/accept/w3b5): boundary(1) lists no entry; X
    enters at boundary(2), clears the genesis house at its first verdict
    (rev6.11 (b)) and regresses 30% on the private pool's daily slice there
    (the always-run A10 check fails). A judged, final loss: never gated,
    never crowned."""
    from tests.accept.w3b5 import helpers as w3

    cfg = w3.lane_d_cfg()
    h0, x = w3.house(), w3.entry("x", reveal_ts=w3.T_FIRST)
    hd, xd = h0["digest"], x["digest"]
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = w3.snapshot(tmp_path, 1, w3.T_FIRST)
    fc1 = {hd: w3.hourly_forecasts(recs1, "bad", "house-0"),
           xd: w3.hourly_forecasts(recs1, "sharp", "x")}
    s1 = w3.settle(cfg, w3.boundary(1), pool1, [], h0, fc1, w3.uids(x), path)
    assert s1.report["judged"] == [], "fixture: boundary(1) lists no entry"
    lf2 = w3.lowfreq_records("lf2", 8, "D")
    pool2, recs2 = w3.snapshot(tmp_path, 2, w3.T_FIRST, lf2)
    fc2 = w3.merge({hd: w3.hourly_forecasts(recs2, "bad", "house-0"),
                    xd: w3.hourly_forecasts(recs2, "sharp", "x")},
                   {hd: w3.lowfreq_forecasts(lf2, w3.flat(1.0)),
                    xd: w3.lowfreq_forecasts(lf2, w3.flat(1.3))})
    report = w3.settle(cfg, w3.boundary(2), pool2, [x], h0, fc2, w3.uids(x), path).report
    v = w3.verdict_of(report, xd)
    assert v is not None and v["stage"] == "first" and v["cleared"] is True, \
        f"fixture: X must clear at its first verdict: {v!r}"
    assert v["lowfreq"]["result"] == "fail", f"fixture: the A10 check must fail: {v['lowfreq']!r}"
    assert v["outcome"] == "lost", v["outcome"]
    assert report["crowned"] is None, "a low-frequency regression must not crown"
    assert report["king_after"] == report["king_before"]
    assert "gate" not in report or xd not in repr(report["gate"]), \
        "a lowfreq_regression candidate is never gated"
    return path, report, xd


# case id → (builder, exact reason, exact status | a tuple of allowed statuses,
#            listed in report.skipped?)
CASES = {
    "pending_floor": (case_pending_floor, "pending_floor", "pending", True),
    "coldkey_cap_settlement": (case_coldkey_cap_settlement, "coldkey_cap_settlement",
                               "pending", True),
    "coldkey_unresolved": (case_coldkey_unresolved, "coldkey_unresolved", "closed", True),
    "king_unscored": (case_king_unscored, "king_unscored", "pending", True),
    "deregistered": (case_deregistered, "deregistered", "closed", True),
    "ready_expired": (case_ready_expired, "ready_expired", "closed", True),
    "duplicate_digest": (case_duplicate_digest, "duplicate_digest", "closed", True),
    "hotkey_has_open_entry": (case_hotkey_has_open_entry, "hotkey_has_open_entry",
                              "closed", True),
    "context_not_armed": (case_context_not_armed, "context_not_armed", "closed", True),
    "context_below_armed": (case_context_below_armed, "context_below_armed", "pending", True),
    "checkpoint_fault_forecasts": (case_checkpoint_fault_forecasts, "checkpoint_fault",
                                   "closed", True),
    "checkpoint_fault_gate": (case_checkpoint_fault_gate, "checkpoint_fault", "closed", False),
    "checkpoint_fault_regate": (case_checkpoint_fault_regate, "checkpoint_fault",
                                "closed", True),
    "lane_closed": (case_lane_closed, "lane_closed", "closed", True),
    "lost_duel": (case_lost_duel, "lost_duel", "lost", False),
    # rev6.11 (b): a held first clear is `hold` (no confirm_pending).
    "bench_hold": (case_bench_hold, "bench_hold", "hold", False),
    "bench_hold_regate": (case_bench_hold_regate, "bench_hold", "hold", True),
    "bench_hold_requeued": (case_bench_hold_requeued, "bench_hold", "pending", True),
    "gate_regression": (case_gate_regression, "gate_regression", "lost", False),
    "crowned": (case_crowned, "crowned", "crowned", False),
    "lowfreq_regression": (case_lowfreq_regression, "lowfreq_regression", "lost", False),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_exact_reason_per_case(tmp_path, case):
    from cascade.validator import lane_d_report, lane_d_state

    build, reason, status, is_skip = CASES[case]
    path, report, dg = build(tmp_path)
    statuses = status if isinstance(status, tuple) else (status,)
    rec = read_state(path)["entries"][dg]
    assert rec["reason"] == reason, f"{case}: state reason {rec['reason']!r} != {reason!r}"
    assert rec["status"] in statuses, f"{case}: state status {rec['status']!r}"
    e = entry_of(report, dg)
    assert (e["status"], e["reason"]) == (rec["status"], rec["reason"]), \
        f"{case}: the report's entry differs from the state's"
    assert reason in lane_d_report.ENTRY_REASONS and reason in lane_d_state.ENTRY_REASONS
    if is_skip:
        assert skips_of(report, dg) == [reason], f"{case}: skipped {skips_of(report, dg)}"
        assert reason in lane_d_report.SKIP_REASONS
        assert dg not in report["judged"]
    else:
        assert skips_of(report, dg) == [], f"{case}: a judged entry is not a skip"
        assert dg in report["judged"]
    lane_d_report.validate_report(report)


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_reason_in_state_and_report_is_typed(tmp_path, case):
    """No free text: every entry of every case, not just the one under test,
    carries an enum reason, and the report's entries and skips agree with
    the state."""
    from cascade.validator import lane_d_report, lane_d_state

    path, report, _ = CASES[case][0](tmp_path)
    state = read_state(path)
    for dg, rec in state["entries"].items():
        assert rec["reason"] in lane_d_state.ENTRY_REASONS, (dg, rec["reason"])
        assert rec["status"] in lane_d_state.ENTRY_STATUSES, (dg, rec["status"])
        e = entry_of(report, dg)
        assert (e["status"], e["reason"]) == (rec["status"], rec["reason"])
    for s in report["skipped"]:
        assert s["reason"] in lane_d_report.SKIP_REASONS, s
        assert state["entries"][s["digest"]]["reason"] == s["reason"], s
    for slot in report["pay"]["court"]:
        assert slot["burn_reason"] is None or slot["burn_reason"] in lane_d_report.BURN_REASONS


def _emitted_reasons() -> set[str]:
    """The reason enum values that appear as string literals (not docstrings)
    in `cascade/validator/lane_d.py`, plus the gate outcomes it maps onto
    entry reasons."""
    from cascade.validator import lane_d_report

    src = (REPO / "cascade" / "validator" / "lane_d.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))
    reasons = set(map(str, lane_d_report.ENTRY_REASONS))
    found = {n.value for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and isinstance(n.value, str)
             and id(n) not in docstrings and n.value in reasons}
    found |= set(map(str, lane_d_report.GATE_OUTCOMES)) & reasons
    return found


def test_no_confirm_stage_reason_or_status_exists():
    """rev6.11 (b): the confirmation duel is dropped. `confirm_pending` and
    `lost_confirmation` are in no reason or status enum, and the settlement
    emits neither (W5e: "Drop `lost_confirmation`; a loss is `lost_duel`")."""
    from cascade.validator import lane_d_report, lane_d_state

    for name in ("confirm_pending", "lost_confirmation"):
        for enum in (lane_d_report.ENTRY_REASONS, lane_d_state.ENTRY_REASONS,
                     lane_d_report.SKIP_REASONS):
            assert name not in tuple(map(str, enum)), name
        assert name not in _emitted_reasons(), name
    for enum in (lane_d_report.ENTRY_STATUSES, lane_d_state.ENTRY_STATUSES):
        assert "confirm_pending" not in tuple(map(str, enum))


def test_every_reason_the_settlement_can_emit_has_a_case():
    covered = {v[1] for v in CASES.values()}
    missing = sorted(_emitted_reasons() - covered)
    assert not missing, f"reasons settle_lane_d can emit with no W5g case: {missing}"


def test_every_enum_reason_is_cased_or_owned_elsewhere():
    from cascade.validator import lane_d_report, lane_d_state

    covered = {v[1] for v in CASES.values()}
    enum = set(map(str, lane_d_report.ENTRY_REASONS)) | set(map(str, lane_d_state.ENTRY_REASONS))
    unaccounted = sorted(enum - covered - set(OWNED_ELSEWHERE))
    assert not unaccounted, f"enum reasons with no case and no owner: {unaccounted}"
    assert set(OWNED_ELSEWHERE) <= enum


def test_the_hold_bound_key_is_parsed():
    """`[lane_d] max_bench_holds` (W2 lists it; plan §10.4 default 4)."""
    assert lane_d_cfg().max_bench_holds == 4
    assert lane_d_cfg(max_bench_holds=0).max_bench_holds == 0
    assert lane_d_cfg(max_bench_holds=7).max_bench_holds == 7
