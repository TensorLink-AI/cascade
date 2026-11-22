"""W3b4 (b): partial classification (plan §1.3b "Faults", §4 T17, §13).

``classify_suite`` decides, for the drawn suite of one host session, whether
the candidate's or a reference's suite result is complete (``ok``), the
miner's ``checkpoint_fault`` or an infra ``bench_hold``, with a typed
sub-reason per role. Every case is asserted on ``classify_suite`` AND through
``paired_gate``, which must use the same classification. Contract: README.md."""

from __future__ import annotations

import copy
import json

import pytest

from .helpers import (
    GATE_SUITE,
    N_ITEMS,
    OTHER_SESSION,
    PATTERNS,
    REFS,
    ROLES,
    SENTINEL,
    broken,
    gate_kwargs,
    pin,
    triple,
)

CANDIDATE_FAULT_CODES = ("absent", "error", "partial", "missing_items", "missing_metric",
                         "null_metric", "nonfinite_metric")


def _classify(records, *, pinned=True, suite=GATE_SUITE):
    from cascade.validator.lane_d_bench import classify_suite

    return classify_suite(*records, suite=suite,
                          pinned_identity=(pin(suite) if pinned is True else pinned))


def _gate(records, *, pinned=True, **over):
    from cascade.validator.lane_d_bench import paired_gate

    kw = gate_kwargs(**over)
    return paired_gate(*records, pinned_identity=(pin(kw["suite"]) if pinned is True
                                                  else pinned), **kw)


def _both(records, *, pinned=True, **over):
    """``(classify_suite result, paired_gate result)``, checked to agree."""
    from cascade.validator.lane_d_bench import FAULT_CODES, HOLD_CODES

    suite = over.get("suite", GATE_SUITE)
    cls = _classify(copy.deepcopy(records), pinned=pinned, suite=suite)
    res = _gate(copy.deepcopy(records), pinned=pinned, **over)
    assert json.loads(json.dumps(cls, allow_nan=False)) == cls
    assert json.loads(json.dumps(res, allow_nan=False)) == res
    assert cls["outcome"] in ("ok", "checkpoint_fault", "bench_hold"), cls
    assert res.get("identity") == cls.get("identity"), (res, cls)
    if cls["outcome"] == "ok":
        assert cls["code"] is None and cls["codes"] == {} and cls["fault_on"] == [], cls
        assert res["outcome"] in ("pass", "gate_regression"), res
        assert res.get("code", "absent-key") is None and res.get("codes") == {}, res
    else:
        assert res["outcome"] == cls["outcome"], (res, cls)
        for key in ("code", "codes", "fault_on"):
            assert res.get(key) == cls[key], (key, res, cls)
        assert isinstance(res["reason"], str) and res["reason"], res
        assert "candidate" not in cls["fault_on"] or cls["outcome"] == "checkpoint_fault"
        if cls["outcome"] == "checkpoint_fault":
            assert cls["code"] in FAULT_CODES, cls
            assert cls["fault_on"] == ["candidate"], cls
            assert set(cls["codes"]) == {"candidate"}, cls
            assert res["passed"] is False and res["hold"] is False, res
        else:
            assert cls["code"] in HOLD_CODES, cls
            assert res["passed"] is None and res["hold"] is True, res
            assert "candidate" not in res["fault_on"], res
        assert set(cls["codes"]) <= set(ROLES), cls
    return cls, res


# ── the code vocabularies ────────────────────────────────────────────────────


def test_code_vocabularies():
    from cascade.validator.lane_d_bench import FAULT_CODES, HOLD_CODES

    assert isinstance(FAULT_CODES, tuple) and isinstance(HOLD_CODES, tuple)
    assert set(CANDIDATE_FAULT_CODES) <= set(FAULT_CODES)
    assert set(FAULT_CODES) <= set(HOLD_CODES), "any defect on a reference holds"
    for infra in ("suite_mismatch", "timeout"):
        assert infra in HOLD_CODES and infra not in FAULT_CODES


# ── complete ⇒ ok ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("pinned", [True, None], ids=["pinned", "unpinned"])
def test_complete_rows_are_ok(pinned):
    cls, res = _both(triple(), pinned=pinned)

    assert cls["outcome"] == "ok"
    assert res["outcome"] == "pass", res


# ── the candidate alone ⇒ checkpoint_fault (the miner's fault, T17) ───────────


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("pinned", [True, None], ids=["pinned", "unpinned"])
def test_candidate_alone_is_a_checkpoint_fault_with_its_code(pattern, pinned):
    code = PATTERNS[pattern][1]
    cls, res = _both(broken(triple(), ["candidate"], pattern), pinned=pinned)

    assert cls["outcome"] == "checkpoint_fault", cls
    assert cls["code"] == code, cls
    assert cls["codes"] == {"candidate": code}, cls
    assert "king_d" not in res["fault_on"] and res["hold"] is False, (
        "T17: a crafted candidate never holds KING_D")


# ── the same pattern on every row set ⇒ bench_hold (infra) ───────────────────


@pytest.mark.parametrize("pattern", sorted(set(PATTERNS) - {"drop_item", "empty_rows"}))
@pytest.mark.parametrize("pinned", [True, None], ids=["pinned", "unpinned"])
def test_the_same_pattern_on_all_rows_is_an_infra_hold(pattern, pinned):
    code = PATTERNS[pattern][1]
    cls, _ = _both(broken(triple(), ROLES, pattern), pinned=pinned)

    assert cls["outcome"] == "bench_hold", cls
    assert cls["code"] == code, cls
    assert cls["codes"] == {r: code for r in ROLES}, cls
    assert cls["fault_on"] == sorted(REFS), cls


def test_an_item_missing_from_every_row_set_is_a_suite_mismatch_hold():
    cls, _ = _both(broken(triple(), ROLES, "drop_item"))

    assert cls["outcome"] == "bench_hold", cls
    assert cls["code"] == "suite_mismatch", cls
    assert cls["codes"] == {r: "suite_mismatch" for r in ROLES}, cls


# ── a reference's pattern ⇒ bench_hold whatever the candidate's rows are ─────


@pytest.mark.parametrize("pattern", sorted(PATTERNS))
@pytest.mark.parametrize("ref", REFS)
def test_reference_alone_is_an_infra_hold_with_its_code(ref, pattern):
    code = PATTERNS[pattern][1]
    cls, _ = _both(broken(triple(), [ref], pattern))

    assert cls["outcome"] == "bench_hold", cls
    assert cls["code"] == code, cls
    assert cls["codes"] == {ref: code}, cls
    assert cls["fault_on"] == [ref], cls


@pytest.mark.parametrize("pattern", ["nan_crps", "partial_flag", "error_status"])
def test_both_references_faulty_hold_and_never_blame_the_candidate(pattern):
    cls, _ = _both(broken(triple(), REFS, pattern))

    assert cls["outcome"] == "bench_hold" and cls["fault_on"] == sorted(REFS), cls
    assert "candidate" not in cls["codes"], cls


def test_candidate_and_one_reference_faulty_is_a_hold():
    recs = broken(broken(triple(), ["candidate"], "nan_crps"), ["house"], "partial_flag")
    cls, _ = _both(recs)

    assert cls["outcome"] == "bench_hold", cls
    assert cls["code"] == "partial", cls
    assert cls["codes"] == {"candidate": "nonfinite_metric", "house": "partial"}, cls
    assert cls["fault_on"] == ["house"], cls


def test_candidate_error_while_references_error_too_is_a_hold():
    cls, _ = _both(broken(triple(), ROLES, "error_status"))

    assert cls["outcome"] == "bench_hold" and cls["code"] == "error", cls


def test_candidate_error_while_one_reference_is_absent_is_a_hold():
    recs = broken(broken(triple(), ["candidate"], "error_status"), ["king_d"], "absent_suite")
    cls, _ = _both(recs)

    assert cls["outcome"] == "bench_hold" and cls["code"] == "absent", cls
    assert cls["codes"] == {"candidate": "error", "king_d": "absent"}, cls


# ── timeouts are always infra ────────────────────────────────────────────────


def _timeout_status(rec):
    rec["suites"][GATE_SUITE] = {"status": "timeout", "rows": []}


def _error_timed_out(rec):
    rec["suites"][GATE_SUITE] = {"status": "error", "rows": [],
                                 "detail": "sidecar timed out after 5400 s"}


def _error_timeout_word(rec):
    rec["suites"][GATE_SUITE] = {"status": "error", "rows": [],
                                 "detail": "TimeoutError: bench sweep exceeded its budget"}


TIMEOUTS = {"status": _timeout_status, "error_timed_out": _error_timed_out,
            "error_timeout_word": _error_timeout_word}


@pytest.mark.parametrize("how", sorted(TIMEOUTS))
@pytest.mark.parametrize("role", ROLES)
def test_a_timeout_is_a_hold_even_on_the_candidate_alone(role, how):
    recs = copy.deepcopy(list(triple()))
    TIMEOUTS[how](recs[ROLES.index(role)])
    cls, _ = _both(tuple(recs))

    assert cls["outcome"] == "bench_hold", cls
    assert cls["code"] == "timeout", cls
    assert cls["codes"] == {role: "timeout"}, cls
    assert cls["fault_on"] == ([] if role == "candidate" else [role]), cls


def test_a_timeout_outranks_a_candidate_fault():
    recs = copy.deepcopy(list(broken(triple(), ["candidate"], "nan_crps")))
    _timeout_status(recs[2])
    cls, _ = _both(tuple(recs))

    assert cls["outcome"] == "bench_hold" and cls["code"] == "timeout", cls


# ── FEV scope-skips are not partial (§13) ────────────────────────────────────


def _coverage(rec, suite, *, skipped, n_expected, n_series, partial=None):
    e = rec["suites"][suite]
    e.update(skipped=skipped, n_expected=n_expected, n_series=n_series)
    if partial is not None:
        e["partial"] = partial


SCOPE_SKIPS = [{"key": f"fev/multi-{i}", "stage": "scope", "reason": "multivariate target"}
               for i in range(3)]


def test_fev_scope_skips_are_not_partial_even_if_flagged():
    recs = triple(suite="fev", sentinel=SENTINEL)
    for rec in recs:
        _coverage(rec, "fev", skipped=SCOPE_SKIPS, n_expected=N_ITEMS + 3,
                  n_series=N_ITEMS, partial=True)
    cls, res = _both(recs, suite="fev")

    assert cls["outcome"] == "ok", cls
    assert res["outcome"] == "pass", res


def test_a_non_scope_skip_on_the_candidate_is_partial_even_unflagged():
    recs = triple(suite="fev", sentinel=SENTINEL)
    for rec in recs:
        _coverage(rec, "fev", skipped=list(SCOPE_SKIPS), n_expected=N_ITEMS + 3,
                  n_series=N_ITEMS)
    _coverage(recs[0], "fev", n_expected=N_ITEMS + 4, n_series=N_ITEMS,
              skipped=SCOPE_SKIPS + [{"key": "fev/uni-9", "stage": "score",
                                      "reason": "forward pass returned NaN"}])
    cls, _ = _both(recs, suite="fev")

    assert cls["outcome"] == "checkpoint_fault", cls
    assert cls["code"] == "partial", cls


def test_scope_stage_skips_exempt_only_fev():
    recs = triple()
    for rec in recs:
        _coverage(rec, GATE_SUITE, skipped=SCOPE_SKIPS[:1], n_expected=N_ITEMS + 1,
                  n_series=N_ITEMS)
    cls, _ = _both(recs)

    assert cls["outcome"] == "bench_hold" and cls["code"] == "partial", cls


def test_a_max_series_cap_is_partial():
    recs = triple()
    _coverage(recs[1], GATE_SUITE, skipped=[], n_expected=N_ITEMS, n_series=N_ITEMS)
    recs[1]["suites"][GATE_SUITE]["max_series"] = 25
    cls, _ = _both(recs)

    assert cls["outcome"] == "bench_hold" and cls["codes"] == {"king_d": "partial"}, cls


# ── one session ──────────────────────────────────────────────────────────────


def test_classify_refuses_mixed_sessions():
    from cascade.validator.lane_d_bench import MixedSessionError

    recs = copy.deepcopy(list(triple()))
    recs[1]["session"] = OTHER_SESSION
    with pytest.raises(MixedSessionError) as exc:
        _classify(tuple(recs))
    assert exc.value.code == "mixed_session"


def test_classify_refuses_an_unknown_suite():
    with pytest.raises(ValueError):
        _classify(triple(), pinned=None, suite="m4")


def test_classify_is_pure():
    recs = broken(triple(), ["candidate"], "nan_crps")
    before = json.dumps(recs, sort_keys=True)
    a = _classify(recs)
    b = _classify(recs)

    assert a == b
    assert json.dumps(recs, sort_keys=True) == before, "classify_suite mutated its input"
