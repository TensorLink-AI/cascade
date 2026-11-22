"""W3b3: the sentinel never gates. Its sweep runs after the settlement, and a
sentinel regression, fault or error leaves the crown, the verdicts, the pay
(weights) and every later settlement byte-identical (plan §1.3b "swept for
each new KING_D and published only", §16 rule 2, T7b). README.md states the
contract."""

from __future__ import annotations

import copy
import json
import math

import pytest

from . import helpers as h

# Sentinel outcomes for the new king: good, a regression vs both references,
# a NaN fault and a host error.
SCENARIOS = ("good", "regression", "fault", "error")


def _host_for(scenario: str, report: dict):
    r = h.roles_of(report)
    if scenario == "error":
        return h.RecordingHost({}, raises=RuntimeError("bench pod unreachable"))
    king_factor = {"good": 0.9, "regression": 1.6, "fault": 0.9}[scenario]
    host = h.host_for(r["king"], r["previous_king"], r["house"], king_factor=king_factor)
    if scenario == "fault":
        for s in h.SENTINEL_SUITES:
            bad = host.table[r["king"]][s]
            bad[0]["crps"] = math.nan
    return host


def _chain_with_sentinel(root, scenario: str):
    """The real chain with the sentinel step run after every crowning
    settlement, as the validator runs it."""
    published: dict = {}
    store = h.MemoryStore()

    def after(boundary, report, pay):
        if report is None:
            return
        before = (h.canonical(report), copy.deepcopy(pay))
        res = h.run(report, era_hash_=h.hash_drawing("gift"),
                    bench=h.session(_host_for(scenario, report), f"{h.SESSION}/{boundary}"),
                    store=store, era_block=h.ERA_START_1 if boundary < h.BOUNDARY_4
                    else h.ERA_START_2)
        assert (h.canonical(report), pay) == before, "the sentinel step mutated the settlement"
        published[boundary] = res

    chain = h.build_chain(root, after=after)
    return chain, published, store


def _unsigned(report: dict) -> dict:
    """The report without its sr25519 signature (randomised per signing)."""
    return {k: v for k, v in report.items() if k != "signature"}


def _unsigned_state(path) -> str:
    state = json.loads(path.read_text())
    state.get("last_report", {}).pop("signature", None)
    return h.canonical(state)


_RUNS: dict = {}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Lazy: the chains run (and the module under test is imported) only
    when a test body reads a scenario, so on main the failure is in the test,
    never in the fixture."""
    class Runs:
        def __getitem__(self, scenario):
            if scenario not in _RUNS:
                _RUNS[scenario] = _chain_with_sentinel(
                    tmp_path_factory.mktemp(f"w3b3-{scenario}"), scenario)
            return _RUNS[scenario]

        def items(self):
            return [(s, self[s]) for s in SCENARIOS]

    return Runs()


def test_the_sentinel_ran_for_every_crowning_and_only_those(runs):
    for scenario, (_, published, store) in runs.items():
        assert sorted(b for b, r in published.items() if r is not None) == \
            [h.BOUNDARY_2, h.BOUNDARY_4], scenario
        assert published[h.BOUNDARY_1] is None and published[h.BOUNDARY_3] is None
        assert len(store.objects) == 2, scenario


def test_the_sentinel_saw_the_numbers(runs):
    """The four scenarios really differ in what was published."""
    texts = {s: runs[s][1][h.BOUNDARY_4]["text"] for s in SCENARIOS}
    assert len(set(texts.values())) == len(SCENARIOS)
    assert runs["good"][1][h.BOUNDARY_4]["record"]["status"] == "ok"
    assert runs["regression"][1][h.BOUNDARY_4]["record"]["status"] == "ok"
    assert runs["fault"][1][h.BOUNDARY_4]["record"]["status"] == "fault"
    assert runs["error"][1][h.BOUNDARY_4]["record"]["status"] == "error"
    good = runs["good"][1][h.BOUNDARY_4]["record"]["models"]["king"]["crps"]
    worse = runs["regression"][1][h.BOUNDARY_4]["record"]["models"]["king"]["crps"]
    house = runs["regression"][1][h.BOUNDARY_4]["record"]["models"]["house"]["crps"]
    assert worse > house > good


def test_settlements_are_identical_whatever_the_sentinel_numbers(runs):
    """Crown, verdicts, pay (the weight input), state and every later
    settlement are byte-identical with sentinel numbers good, bad, faulty or
    missing."""
    ref_chain = runs["good"][0]
    for scenario in SCENARIOS[1:]:
        chain = runs[scenario][0]
        for b in (h.BOUNDARY_1, h.BOUNDARY_2, h.BOUNDARY_3, h.BOUNDARY_4):
            rep, pay = chain[b]
            ref_rep, ref_pay = ref_chain[b]
            assert h.canonical(_unsigned(rep)) == h.canonical(_unsigned(ref_rep)), (scenario, b)
            assert pay == ref_pay, (scenario, b)
        assert _unsigned_state(chain["state"]) == _unsigned_state(ref_chain["state"]), scenario
    assert ref_chain[h.BOUNDARY_4][0]["crowned"] == h.Y["digest"]


def test_the_record_declares_it_never_gates(runs):
    for scenario in SCENARIOS:
        for b in (h.BOUNDARY_2, h.BOUNDARY_4):
            assert runs[scenario][1][b]["record"]["gating"] is False


def test_the_sentinel_step_writes_nothing_but_its_record(tmp_path):
    """The state file and the report are untouched by a regression sweep run
    directly on a finished chain."""
    chain = h.build_chain(tmp_path / "chain")
    state_bytes = chain["state"].read_bytes()
    report = chain[h.BOUNDARY_4][0]
    snap = h.canonical(report)
    store = h.MemoryStore()
    res = h.run(report, era_hash_=h.hash_drawing("boom"),
                bench=h.session(_host_for("regression", report)), store=store)
    assert res["record"]["status"] == "ok"
    assert h.canonical(report) == snap
    assert chain["state"].read_bytes() == state_bytes
    assert list(store.objects) == [res["key"]]
