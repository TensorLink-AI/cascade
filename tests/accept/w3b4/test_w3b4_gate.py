"""W3b4: suite identity on the paired gate (plan §1.3b "Suite identity is
verified per drawn suite"; §4 T17).

Rows whose drawn-suite identity (version or item set) differs from the pinned
identity are refused with ``suite_mismatch``: an infra ``bench_hold``, never
the miner's fault. The identities are recorded in every gate result.
Contract: README.md."""

from __future__ import annotations

import copy

import pytest

from .helpers import (
    DIGESTS,
    GATE_SUITE,
    N_ITEMS,
    REFS,
    ROLES,
    SENTINEL,
    SESSION,
    broken,
    entry,
    gate_kwargs,
    pin,
    triple,
)
from .test_w3b4_classify import _both, _gate


def _identity(suite, e):
    from cascade.eval.suite_identity import suite_identity

    return suite_identity(suite, e)


# ── identity recorded in the result ──────────────────────────────────────────


def test_identity_is_recorded_on_a_pass():
    recs = triple()
    _, res = _both(recs)

    assert res["outcome"] == "pass", res
    ident = res["identity"]
    assert ident["pinned"] == pin()
    for role, rec in zip(ROLES, recs, strict=True):
        assert ident[role] == _identity(GATE_SUITE, rec["suites"][GATE_SUITE]), role


def test_identity_is_recorded_on_a_regression():
    _, res = _both(triple(ref_factor=0.9))

    assert res["outcome"] == "gate_regression", res
    assert res["identity"]["king_d"]["items_sha256"] == pin()["items_sha256"]


def test_identity_is_recorded_on_faults_with_none_for_an_unswept_role():
    _, res = _both(broken(triple(), ["candidate"], "absent_suite"))

    assert res["outcome"] == "checkpoint_fault", res
    assert res["identity"]["candidate"] is None
    assert res["identity"]["house"]["version"] == pin()["version"]


def test_identity_is_recorded_without_a_pin():
    _, res = _both(triple(), pinned=None)

    assert res["identity"]["pinned"] is None
    assert res["identity"]["king_d"]["digest"] == pin()["digest"]


def test_a_matching_pin_does_not_change_the_verdict():
    from cascade.validator.lane_d_bench import classify_suite

    recs = triple(ref_factor=1.01)
    assert classify_suite(*copy.deepcopy(recs), suite=GATE_SUITE,
                          pinned_identity=pin())["outcome"] == "ok"
    pinned = _gate(copy.deepcopy(recs))
    free = _gate(copy.deepcopy(recs), pinned=None)

    assert pinned["outcome"] == free["outcome"] == "pass"
    assert pinned["lcb"] == free["lcb"]


# ── suite_mismatch: always infra ─────────────────────────────────────────────


def _set_revision(rec, rev):
    rec["suites"][GATE_SUITE]["data_revision"] = rev


def test_every_row_set_on_another_version_is_a_suite_mismatch_hold():
    recs = triple(revision="boom@9e8d7c6")
    cls, res = _both(recs)

    assert cls["outcome"] == "bench_hold" and cls["code"] == "suite_mismatch", cls
    assert cls["codes"] == {r: "suite_mismatch" for r in ROLES}, cls
    assert res["fault_on"] == sorted(REFS), res


def test_candidate_alone_on_another_version_is_a_hold_not_a_fault():
    recs = copy.deepcopy(list(triple()))
    _set_revision(recs[0], "boom@9e8d7c6")
    cls, res = _both(tuple(recs))

    assert cls["outcome"] == "bench_hold" and cls["code"] == "suite_mismatch", cls
    assert cls["codes"] == {"candidate": "suite_mismatch"}, cls
    assert res["fault_on"] == [], res


@pytest.mark.parametrize("ref", REFS)
def test_a_reference_on_another_version_is_a_suite_mismatch_hold(ref):
    recs = copy.deepcopy(list(triple()))
    _set_revision(recs[ROLES.index(ref)], "boom@9e8d7c6")
    cls, _ = _both(tuple(recs))

    assert cls["outcome"] == "bench_hold" and cls["code"] == "suite_mismatch", cls
    assert cls["codes"] == {ref: "suite_mismatch"} and cls["fault_on"] == [ref], cls


def test_a_missing_revision_against_a_pinned_one_is_a_suite_mismatch():
    recs = copy.deepcopy(list(triple()))
    del recs[2]["suites"][GATE_SUITE]["data_revision"]
    cls, _ = _both(tuple(recs))

    assert cls["code"] == "suite_mismatch" and cls["codes"] == {"house": "suite_mismatch"}


@pytest.mark.parametrize("role", ROLES)
def test_an_item_outside_the_pinned_set_is_a_suite_mismatch(role):
    recs = copy.deepcopy(list(triple()))
    rows = recs[ROLES.index(role)]["suites"][GATE_SUITE]["rows"]
    rows.append({"item": f"{GATE_SUITE}/task-extra", "crps": 0.5, "mase": 1.0})
    cls, _ = _both(tuple(recs))

    assert cls["outcome"] == "bench_hold" and cls["code"] == "suite_mismatch", cls
    assert cls["codes"] == {role: "suite_mismatch"}, cls


def test_unpinned_records_that_disagree_on_the_version_hold():
    recs = copy.deepcopy(list(triple()))
    _set_revision(recs[1], "boom@9e8d7c6")
    cls, _ = _both(tuple(recs), pinned=None)

    assert cls["outcome"] == "bench_hold" and cls["code"] == "suite_mismatch", cls


def test_unpinned_records_that_agree_on_the_version_are_not_a_mismatch():
    cls, res = _both(triple(revision="boom@9e8d7c6"), pinned=None)

    assert cls["outcome"] == "ok" and res["outcome"] == "pass", (cls, res)


def test_a_pin_built_with_suite_identity_matches_its_own_sweep():
    recs = triple()
    own = _identity(GATE_SUITE, recs[1]["suites"][GATE_SUITE])
    cls, _ = _both(recs, pinned=own)

    assert cls["outcome"] == "ok", cls


def test_the_sentinel_is_not_checked_against_the_gate_pin():
    recs = copy.deepcopy(list(triple()))
    for rec in recs:
        rec["suites"][SENTINEL]["data_revision"] = "gift@other"
    _, res = _both(tuple(recs))

    assert res["outcome"] == "pass", res


# ── pin refusals ─────────────────────────────────────────────────────────────


def _bad_pins():
    other = pin("time")
    short = dict(pin(), items_sha256="abc123")
    upper = dict(pin(), items_sha256=pin()["items_sha256"].upper())
    no_sha = {k: v for k, v in pin().items() if k != "items_sha256"}
    bad_version = dict(pin(), version=7)
    return {"other_suite": other, "short_sha": short, "upper_sha": upper,
            "no_sha": no_sha, "bad_version": bad_version, "not_mapping": "boom@3f1c2a9"}


@pytest.mark.parametrize("which", sorted(_bad_pins()))
def test_a_malformed_or_foreign_pin_is_refused(which):
    from cascade.validator.lane_d_bench import classify_suite, paired_gate

    bad = _bad_pins()[which]
    with pytest.raises((TypeError, ValueError)):
        paired_gate(*triple(), pinned_identity=bad, **gate_kwargs())
    with pytest.raises((TypeError, ValueError)):
        classify_suite(*triple(), suite=GATE_SUITE, pinned_identity=bad)


# ── one host session (SweepSession) ──────────────────────────────────────────


class _Host:
    """A stand-in for the bench pod (an external service): per digest, the
    suite entries it returns; a digest it has nothing for gets ``{}``."""

    def __init__(self, table):
        self.table = table

    def __call__(self, digests, suites):
        return {d: {s: copy.deepcopy(self.table[d][s]) for s in suites
                    if s in self.table.get(d, {})} for d in digests}


def _session(table):
    from cascade.validator.lane_d_bench import SweepSession

    s = SweepSession(SESSION, _Host(table))
    return s.paired(DIGESTS["candidate"], DIGESTS["king_d"], DIGESTS["house"],
                    suites=(GATE_SUITE, SENTINEL))


def _table():
    return {DIGESTS[r]: {GATE_SUITE: entry(GATE_SUITE, r),
                         SENTINEL: entry(SENTINEL, r, revision="gift@r1")} for r in ROLES}


def test_host_returning_nothing_for_the_candidate_alone_is_its_fault():
    table = _table()
    table[DIGESTS["candidate"]] = {}
    cls, res = _both(_session(table))

    assert cls["outcome"] == "checkpoint_fault" and cls["code"] == "absent", cls
    assert res["hold"] is False, res


def test_host_timing_out_the_whole_session_holds():
    table = _table()
    for r in ROLES:
        table[DIGESTS[r]][GATE_SUITE] = {"status": "timeout", "rows": []}
    cls, _ = _both(_session(table))

    assert cls["outcome"] == "bench_hold" and cls["code"] == "timeout", cls
    assert cls["codes"] == {r: "timeout" for r in ROLES}, cls


def test_host_rows_on_a_stale_data_revision_hold():
    table = _table()
    for r in ROLES:
        table[DIGESTS[r]][GATE_SUITE]["data_revision"] = "boom@stale00"
    cls, _ = _both(_session(table))

    assert cls["outcome"] == "bench_hold" and cls["code"] == "suite_mismatch", cls


# ── T17 end to end ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("pattern", ["nan_crps", "partial_flag", "none_mase"])
def test_t17_crafted_candidate_is_a_checkpoint_fault_and_king_d_is_not_held(pattern):
    _, res = _both(broken(triple(), ["candidate"], pattern))

    assert res["outcome"] == "checkpoint_fault", res
    assert res["fault_on"] == ["candidate"] and res["hold"] is False, res
    assert res["identity"]["king_d"]["n_items"] == N_ITEMS
