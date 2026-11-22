"""W3b2: one host session, one paired sweep (plan §1.3b: "The house and KING_D
are re-swept in the same host session as the candidate (cached rows are
re-used only within that session)").

``SweepSession(session_id, sweep, cache=None)`` wraps the bench host's sweep
(an external service: here a recording stand-in that returns recorded suite
outputs). ``paired(candidate, king_d, house, *, suites)`` returns the three
sweep records, sweeping every digest this session has not swept yet in ONE
host call. Contract: README.md."""

from __future__ import annotations

import pytest

from .helpers import (
    CAND,
    GATE_SUITE,
    HOUSE,
    KING,
    OTHER_SESSION,
    SENTINEL,
    SESSION,
    RecordingHost,
    digest,
    gate_kwargs,
    host_table,
)

CAND2 = digest("candidate-2")
SUITES2 = (GATE_SUITE, SENTINEL)


def _session(sid, host, cache=None):
    from cascade.validator.lane_d_bench import SweepSession

    return SweepSession(sid, host, cache=cache)


def _host():
    return RecordingHost(host_table(CAND, CAND2, KING, HOUSE))


def test_first_paired_call_sweeps_all_three_in_one_host_call():
    host = _host()
    s = _session(SESSION, host)
    assert s.session_id == SESSION
    recs = s.paired(CAND, KING, HOUSE, suites=SUITES2)
    assert len(host.calls) == 1, host.calls
    digests, suites = host.calls[0]
    assert sorted(digests) == sorted([CAND, KING, HOUSE])
    assert set(suites) >= set(SUITES2)
    for rec, dg in zip(recs, (CAND, KING, HOUSE), strict=True):
        assert rec["session"] == SESSION and rec["digest"] == dg
        assert set(rec["suites"]) >= set(SUITES2)
        assert rec["suites"][GATE_SUITE]["status"] == "ok"
        assert rec["suites"][GATE_SUITE]["rows"] == host.table[dg][GATE_SUITE]


def test_king_d_and_house_are_reused_within_the_session():
    host = _host()
    s = _session(SESSION, host)
    s.paired(CAND, KING, HOUSE, suites=SUITES2)
    recs = s.paired(CAND2, KING, HOUSE, suites=SUITES2)
    assert len(host.calls) == 2
    assert list(host.calls[1][0]) == [CAND2], "cached KING_D/house rows must be reused"
    assert recs[0]["digest"] == CAND2 and all(r["session"] == SESSION for r in recs)


def test_fully_cached_call_does_not_sweep():
    host = _host()
    s = _session(SESSION, host)
    first = s.paired(CAND, KING, HOUSE, suites=SUITES2)
    again = s.paired(CAND, KING, HOUSE, suites=SUITES2)
    assert len(host.calls) == 1
    assert again == first


def test_house_identical_to_king_d_is_swept_once():
    host = _host()
    s = _session(SESSION, host)
    _, k, h = s.paired(CAND, KING, KING, suites=SUITES2)
    assert sorted(host.calls[0][0]) == sorted([CAND, KING])
    assert k == h and k["digest"] == KING


def test_a_new_session_re_sweeps_even_with_a_shared_cache():
    host = _host()
    cache: dict = {}
    _session(OTHER_SESSION, host, cache).paired(CAND, KING, HOUSE, suites=SUITES2)
    recs = _session(SESSION, host, cache).paired(CAND2, KING, HOUSE, suites=SUITES2)
    assert len(host.calls) == 2
    assert sorted(host.calls[1][0]) == sorted([CAND2, KING, HOUSE]), (
        "KING_D and the house must be re-swept in the candidate's session")
    assert all(r["session"] == SESSION for r in recs)


def test_a_suite_not_yet_swept_re_sweeps_the_cached_digest():
    host = _host()
    s = _session(SESSION, host)
    s.paired(CAND, KING, HOUSE, suites=(GATE_SUITE,))
    recs = s.paired(CAND, KING, HOUSE, suites=SUITES2)
    assert len(host.calls) == 2
    assert sorted(host.calls[1][0]) == sorted([CAND, KING, HOUSE])
    assert SENTINEL in host.calls[1][1]
    assert all(SENTINEL in r["suites"] for r in recs)


def test_records_of_two_sessions_never_pair():
    from cascade.validator.lane_d_bench import paired_gate

    host = _host()
    old = _session(OTHER_SESSION, host).paired(CAND, KING, HOUSE, suites=SUITES2)
    new = _session(SESSION, host).paired(CAND, KING, HOUSE, suites=SUITES2)
    with pytest.raises(ValueError, match="mixed_session"):
        paired_gate(new[0], old[1], new[2], **gate_kwargs())


def test_session_records_feed_the_gate():
    from cascade.validator.lane_d_bench import paired_gate

    host = _host()
    recs = _session(SESSION, host).paired(CAND, KING, HOUSE, suites=SUITES2)
    res = paired_gate(*recs, **gate_kwargs())
    assert res["outcome"] in ("pass", "gate_regression")
    assert res["session"] == SESSION
    assert res["digests"] == {"candidate": CAND, "king_d": KING, "house": HOUSE}


@pytest.mark.parametrize("bad", ["", None])
def test_session_id_is_required(bad):
    with pytest.raises((ValueError, TypeError)):
        _session(bad, _host())


# ── a digest the host omits (RV5 non-blocking note 1, kept consistent with W3b4) ──
#
# SweepSession returns a record with no entry for a digest the host left out.
# The gate then classifies it as W3b4's classify_suite does: the candidate
# alone absent while both references are complete is the miner's
# checkpoint_fault (code "absent"); a reference absent is a bench_hold, whatever
# the candidate's rows are. A host adapter must report a per-digest infra
# failure as a status (timeout / error with detail), never by omission.


class _OmittingHost(RecordingHost):
    def __init__(self, table, omit):
        super().__init__(table)
        self.omit = set(omit)

    def __call__(self, digests, suites):
        got = super().__call__(digests, suites)
        return {d: v for d, v in got.items() if d not in self.omit}


def test_host_omitting_the_candidate_alone_is_a_checkpoint_fault():
    from cascade.validator.lane_d_bench import classify_suite, paired_gate

    host = _OmittingHost(host_table(CAND, KING, HOUSE), omit={CAND})
    recs = _session(SESSION, host).paired(CAND, KING, HOUSE, suites=SUITES2)
    assert GATE_SUITE not in recs[0]["suites"], recs[0]
    res = paired_gate(*recs, **gate_kwargs())
    cls = classify_suite(*recs, suite=GATE_SUITE)
    assert res["outcome"] == cls["outcome"] == "checkpoint_fault", (res, cls)
    assert cls["code"] == "absent" and res["fault_on"] == ["candidate"], (res, cls)
    assert res["hold"] is False and res["passed"] is False


@pytest.mark.parametrize("omit", ["king_d", "house"])
def test_host_omitting_a_reference_holds(omit):
    from cascade.validator.lane_d_bench import classify_suite, paired_gate

    dg = {"king_d": KING, "house": HOUSE}[omit]
    host = _OmittingHost(host_table(CAND, KING, HOUSE), omit={dg})
    recs = _session(SESSION, host).paired(CAND, KING, HOUSE, suites=SUITES2)
    res = paired_gate(*recs, **gate_kwargs())
    cls = classify_suite(*recs, suite=GATE_SUITE)
    assert res["outcome"] == cls["outcome"] == "bench_hold", (res, cls)
    assert omit in res["fault_on"] and "candidate" not in res["fault_on"], res
    assert res["hold"] is True and res["passed"] is None
