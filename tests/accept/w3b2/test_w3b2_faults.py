"""W3b2: faults and the one-session rule (plan §1.3b "Faults", §4 T17).

* Partial, NaN / inf or missing-metric rows on the drawn suite are the
  candidate's ``checkpoint_fault`` (the miner's fault: judged, not passed).
* The same in KING_D's or the house's rows is an infra hold, ``bench_hold``
  (typed, never the miner's fault: the candidate is not judged).
* The three row sets must come from ONE host session; otherwise the gate
  refuses with ``mixed_session`` (stored rows from another session are never
  compared against). Contract: README.md."""

from __future__ import annotations

import json

import pytest

from .helpers import GATE_SUITE, OTHER_SESSION, SENTINEL, gate_kwargs, triple


def _gate(cand, king, house, **over):
    from cascade.validator.lane_d_bench import paired_gate

    return paired_gate(cand, king, house, **gate_kwargs(**over))


def _nan_crps(rec, suite=GATE_SUITE):
    rec["suites"][suite]["rows"][7]["crps"] = float("nan")


def _inf_mase(rec, suite=GATE_SUITE):
    rec["suites"][suite]["rows"][3]["mase"] = float("inf")


def _none_mase(rec, suite=GATE_SUITE):
    rec["suites"][suite]["rows"][11]["mase"] = None


def _missing_crps(rec, suite=GATE_SUITE):
    del rec["suites"][suite]["rows"][0]["crps"]


def _partial(rec, suite=GATE_SUITE):
    rec["suites"][suite]["partial"] = True


def _dropped_item(rec, suite=GATE_SUITE):
    del rec["suites"][suite]["rows"][20]


BREAKS = {"nan_crps": _nan_crps, "inf_mase": _inf_mase, "none_mase": _none_mase,
          "missing_crps": _missing_crps, "partial": _partial, "dropped_item": _dropped_item}


@pytest.mark.parametrize("brk", sorted(BREAKS))
def test_t17_candidate_fault_is_checkpoint_fault(brk):
    # The candidate would otherwise pass comfortably: the fault alone decides.
    cand, king, house = triple(king=(1.05, 1.05), house=(1.05, 1.05))
    BREAKS[brk](cand)
    res = _gate(cand, king, house)
    assert res["outcome"] == "checkpoint_fault", res
    assert res["passed"] is False and res["hold"] is False
    assert "candidate" in res["fault_on"]
    assert isinstance(res.get("reason"), str) and res["reason"]
    assert json.loads(json.dumps(res, allow_nan=False)) == res


@pytest.mark.parametrize("ref", ["king_d", "house"])
@pytest.mark.parametrize("brk", sorted(BREAKS))
def test_t17_reference_fault_is_an_infra_hold(ref, brk):
    cand, king, house = triple(king=(0.9, 0.9), house=(0.9, 0.9))
    BREAKS[brk](king if ref == "king_d" else house)
    res = _gate(cand, king, house)
    assert res["outcome"] == "bench_hold", res
    assert res["passed"] is None and res["hold"] is True
    assert ref in res["fault_on"] and "candidate" not in res["fault_on"]
    assert isinstance(res.get("reason"), str) and res["reason"]
    assert json.loads(json.dumps(res, allow_nan=False)) == res


def test_fault_outcomes_are_the_report_reasons():
    from cascade.validator.lane_d_report import SKIP_REASONS

    cand, king, house = triple()
    _nan_crps(cand)
    assert _gate(cand, king, house)["outcome"] in SKIP_REASONS
    cand, king, house = triple()
    _nan_crps(house)
    assert _gate(cand, king, house)["outcome"] in SKIP_REASONS


def test_both_sides_faulty_is_a_hold_never_the_miners_fault():
    cand, king, house = triple()
    _nan_crps(cand)
    _partial(house)
    res = _gate(cand, king, house)
    assert res["outcome"] == "bench_hold" and res["hold"] is True
    assert "house" in res["fault_on"]


def test_faulty_house_identical_to_king_d_holds_on_both():
    cand, king, _ = triple()
    _inf_mase(king)
    res = _gate(cand, king, king)
    assert res["outcome"] == "bench_hold"
    assert {"house", "king_d"} <= set(res["fault_on"])


@pytest.mark.parametrize("brk", sorted(BREAKS))
def test_t7b_sentinel_fault_never_gates(brk):
    cand, king, house = triple(king=(1.03, 1.03), house=(1.03, 1.03))
    BREAKS[brk](cand, SENTINEL)
    BREAKS[brk](house, SENTINEL)
    res = _gate(cand, king, house)
    assert res["outcome"] == "pass" and res["passed"] is True, res
    assert res["sentinel"]["suite"] == SENTINEL


# ── one session ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("which", [0, 1, 2])
def test_mixed_session_is_refused(which):
    recs = list(triple())
    recs[which]["session"] = OTHER_SESSION
    with pytest.raises(ValueError) as e:
        _gate(*recs)
    assert "mixed_session" in str(e.value)
    assert getattr(e.value, "code", None) == "mixed_session"


def test_stored_king_rows_from_an_earlier_session_are_refused_even_if_identical():
    cand, king, house = triple()
    old = json.loads(json.dumps(king))
    old["session"] = OTHER_SESSION
    with pytest.raises(ValueError, match="mixed_session"):
        _gate(cand, old, house)


@pytest.mark.parametrize("bad", [None, "", 7])
def test_missing_session_is_refused(bad):
    cand, king, house = triple()
    for rec in (cand, king, house):
        rec["session"] = bad
    with pytest.raises((ValueError, TypeError)):
        _gate(cand, king, house)
