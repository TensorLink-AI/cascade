"""W7d fixture checks: the honest chains are what README "Fixture chains"
says, and the chain rules README pins hold on them. These pass on main: they
check the fixtures (real `settle_lane_d` runs), not the replay.

The audit (`check_lane_d`) PASSes every honest report, and the W7c receipt
chain links, so a replay that disagrees with an honest chain is wrong, not
the fixture.
"""

from __future__ import annotations

import pytest

from .helpers import (
    ALLOWED_PRIMITIVES,
    body_sha,
    build_all,
    forbidden,
    genesis_digest,
    sha,
    static_closure,
)


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-fixture"))


def _reports(c):
    reps = c.pub.reports()
    return [reps[b] for b in c.pub.boundaries]


def test_fixture_every_honest_report_passes_the_audit(chains):
    from cascade.audit.lane_d import check_lane_d

    for name, c in chains.items():
        for r in _reports(c):
            res = check_lane_d(r, c.cfg, signer_hotkey=c.pub.signer_ss58)
            assert res.status == "PASS", (name, r["boundary"], res)


def test_fixture_receipts_link_from_genesis_and_bind_bodies_and_states(chains):
    for name, c in chains.items():
        reps = _reports(c)
        prev = genesis_digest(reps[0]["house"])
        for r, rec, b in zip(reps, c.pub.receipts, c.pub.boundaries, strict=True):
            assert rec is not None, (name, b)
            assert rec["boundary"] == b == r["boundary"]
            assert rec["prior_state_sha256"] == prev, (name, b)
            assert rec["report_sha256"] == body_sha(r), (name, b)
            assert rec["posterior_state_sha256"] == sha(c.pub.states[b]), (name, b)
            prev = rec["posterior_state_sha256"]


def _outcomes(report: dict) -> list:
    return [v.get("outcome") for v in report.get("verdicts") or []]


def _gate_digests(report: dict) -> list:
    return [(g.get("digest"), g.get("outcome")) for g in
            (report.get("gate") or {}).get("results") or []]


def _status(report: dict, dg: str) -> list:
    return [e.get("status") for e in report.get("entries") or [] if e.get("digest") == dg]


def test_fixture_chain_shapes(chains):
    """rev6.11 (b): the crown at the first verdict. Each assertion names the
    README "Fixture chains" fact it checks; on main the old code first-clears
    X at b2 (`confirm_pending`) and never crowns (assertion failures)."""
    x, y = chains["gated"].x["digest"], chains["gated"].y["digest"]
    g = _reports(chains["gated"])
    assert [r["crowned"] for r in g] == [None, x, None], "gated: X crowned at b2"
    assert g[0]["verdicts"] == [] and _status(g[0], x) == ["pending"], "gated: b1 floor wait"
    assert _outcomes(g[1]) == ["crowned"], "gated: b2 is X's first and only verdict"
    assert _gate_digests(g[1]) == [(x, "pass")]
    assert g[2]["verdicts"] == [] and "gate" not in g[2]

    h = _reports(chains["held"])
    assert [r["crowned"] for r in h] == [None, None, x], "held: X crowned at b3"
    # A held first clear is status `hold`, reason `bench_hold`; its verdict
    # outcome is `cleared` (never crowned), as W5g, W7d's house chains (C
    # in `open_copy`) and W17 pin it.
    assert "gate" not in h[1] and _outcomes(h[1]) == ["cleared"], \
        "held: b2's ungated first clear is held"
    assert _status(h[1], x) == ["hold"], "held: a held candidate stays open as hold"
    assert [e.get("reason") for e in h[1].get("entries") or [] if e.get("digest") == x] \
        == ["bench_hold"], "held: the hold's reason is bench_hold"
    assert h[2]["verdicts"] == [] and _gate_digests(h[2]) == [(x, "pass")]

    hb = _reports(chains["house_bar"])
    assert [r["crowned"] for r in hb] == [None, x, y], "house_bar: X at b2, Y at b3"
    assert hb[2]["cohorts"] and "house_digest" in hb[2]["cohorts"][0]
    assert hb[2]["verdicts"] and "house" in hb[2]["verdicts"][0]
    assert [p["digest"] for p in hb[2]["prior_kings"]] == [x]
    assert [s["uid"] for s in hb[2]["pay"]["court"]] == [12, 11]

    rf = _reports(chains["refresh"])
    h1 = chains["refresh"].h1["digest"]
    assert rf[1]["king_before"]["digest"] == h1, "the house refresh moves a house-held throne"
    assert rf[1]["cohorts"] and rf[1]["cohorts"][0]["king_digest"] == h1
    assert [r["crowned"] for r in rf] == [None, x, None], "refresh: X crowned at b2 vs H1"


def test_fixture_no_report_carries_a_confirmation_stage(chains):
    """rev6.11 (b) / W7a: no cohort or verdict at a stage other than `first`
    (or none), no `confirmation` block, no `confirm_pending` / `confirmed` /
    `lost_confirmation` anywhere."""
    gone = {"confirm_pending", "confirmed", "lost_confirmation"}
    for name, c in chains.items():
        for r in _reports(c):
            for co in r["cohorts"]:
                assert co.get("stage", "first") == "first", (name, r["boundary"], co.get("stage"))
            for v in r["verdicts"]:
                assert v.get("stage", "first") == "first", (name, r["boundary"])
                assert "confirmation" not in v, (name, r["boundary"])
                assert v.get("outcome") not in gone, (name, r["boundary"], v.get("outcome"))
            for e in r["entries"]:
                assert e.get("status") not in gone and e.get("reason") not in gone, \
                    (name, r["boundary"], e)
            assert "confirm_on_king_change" not in r["params"], (name, r["boundary"])


def test_fixture_chain_rules_hold_on_every_honest_chain(chains):
    """README "Chain rules" C2-C5, checked here on the fixtures."""
    for name, c in chains.items():
        reps = _reports(c)
        rpk = int(c.cfg.reward_prior_kings)
        judged_on: dict[str, set] = {}
        for i, r in enumerate(reps):
            if i:
                p = reps[i - 1]
                want_before = p["king_after"]
                if p["king_after"]["house"] and r["house"]["digest"] != p["house"]["digest"]:
                    want_before = dict(r["house"])
                assert r["king_before"] == want_before, (name, i)
                if r["crowned"] and not r["king_before"]["house"]:
                    want_prior = ([r["king_before"]] + p["prior_kings"])[:rpk]
                else:
                    want_prior = p["prior_kings"]
                assert r["prior_kings"] == want_prior, (name, i)
            else:
                assert r["king_before"] == {**r["house"]} and r["prior_kings"] == []
            if r["crowned"] is None:
                assert r["king_after"] == r["king_before"], (name, i)
            else:
                (e,) = [e for e in r["entries"] if e["digest"] == r["crowned"]]
                assert r["king_after"] == {"digest": e["digest"], "hotkey": e["hotkey"],
                                           "house": False, "reveal_ts": e["reveal_ts"]}
            cohorts = {co["index"]: co for co in r["cohorts"]}
            for v in r["verdicts"]:
                co = cohorts[v["cohort"]]
                dg = v["digest"]
                # C4 (rev6.11): every verdict is a first verdict against
                # king_before, and an entry is judged at most once.
                assert co["king_digest"] == r["king_before"]["digest"], (name, i)
                assert dg not in judged_on, (name, i, "judged twice")
                judged_on.setdefault(dg, set()).update(co["window_ids"])


def test_fixture_the_allowed_primitives_reach_no_forbidden_module():
    """The invariant 12 allowlist is satisfiable on main: the shared pure
    primitives' own static closure holds no forbidden module."""
    closure = static_closure(ALLOWED_PRIMITIVES)
    assert set(ALLOWED_PRIMITIVES) <= closure
    bad = sorted(m for m in closure if forbidden(m))
    assert not bad, bad


def test_fixture_forbidden_names_cover_the_producer_modules():
    for m in ("cascade.validator.lane_d", "cascade.validator.lane_d_frozen",
              "cascade.validator.lane_d_forward", "cascade.validator.lane_d_bench",
              "cascade.validator.lane_d_gate", "cascade.validator.lane_d_report",
              "cascade.validator.lane_d_state", "cascade.validator.weights",
              "cascade.shared.emission", "cascade.shared.chain", "cascade.audit.lane_d",
              "cascade.eval.lane_d_scorer", "torch"):
        assert forbidden(m), m
    for m in ALLOWED_PRIMITIVES + ("cascade.satellite.lane_d_replay", "numpy", "hashlib"):
        assert not forbidden(m), m
