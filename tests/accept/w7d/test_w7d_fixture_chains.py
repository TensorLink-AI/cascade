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


def test_fixture_chain_shapes(chains):
    x, y = chains["gated"].x["digest"], chains["gated"].y["digest"]
    g = _reports(chains["gated"])
    assert [r["crowned"] for r in g] == [None, x, None]
    assert [v["outcome"] for v in g[0]["verdicts"]] == ["confirm_pending"]
    assert g[1]["gate"]["results"][0]["outcome"] == "pass"
    assert g[2]["verdicts"] == [] and "gate" not in g[2]

    h = _reports(chains["held"])
    assert [r["crowned"] for r in h] == [None, None, x]
    assert "gate" not in h[1] and h[1]["verdicts"][0]["outcome"] == "confirmed"
    assert h[2]["verdicts"] == [] and h[2]["gate"]["results"][0]["digest"] == x
    (held_entry,) = [e for e in h[1]["entries"] if e["digest"] == x]
    assert held_entry["status"] == "confirm_pending"

    hb = _reports(chains["house_bar"])
    assert [r["crowned"] for r in hb] == [None, x, None, y]
    assert "house_digest" in hb[2]["cohorts"][0] and "house" in hb[2]["verdicts"][0]
    assert [p["digest"] for p in hb[3]["prior_kings"]] == [x]
    assert [s["uid"] for s in hb[3]["pay"]["court"]] == [12, 11]

    rs = _reports(chains["d20_restart"])
    h1 = chains["d20_restart"].h1["digest"]
    assert rs[1]["verdicts"][0]["stage"] == "first" and rs[1]["verdicts"][0]["king_digest"] == h1
    assert rs[1]["king_before"]["digest"] == h1, "the house refresh moves a house-held throne"
    assert rs[2]["crowned"] == x

    og = _reports(chains["d20_original"])
    h0 = chains["d20_original"].h0["digest"]
    v = og[1]["verdicts"][0]
    assert v["stage"] == "confirm" and v["king_digest"] == h0
    assert v["confirmation"]["first_king_digest"] == h0
    assert og[1]["crowned"] == x


def test_fixture_chain_rules_hold_on_every_honest_chain(chains):
    """README "Chain rules" C2-C5, checked here on the fixtures."""
    for name, c in chains.items():
        reps = _reports(c)
        rpk = int(c.cfg.reward_prior_kings)
        firsts: dict[str, dict] = {}
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
                if v["stage"] == "first":
                    assert co["king_digest"] == r["king_before"]["digest"], (name, i)
                else:
                    first = firsts[dg]
                    assert v["confirmation"]["first_set_sha256"] == first["set_sha256"]
                    assert v["confirmation"]["first_king_digest"] == first["king_digest"]
                    if c.cfg.confirm_on_king_change == "restart":
                        assert co["king_digest"] == r["king_before"]["digest"]
                    else:
                        assert co["king_digest"] == first["king_digest"]
                    assert not set(co["window_ids"]) & judged_on.get(dg, set()), (name, i)
                judged_on.setdefault(dg, set()).update(co["window_ids"])
                if v["stage"] == "first" and v["outcome"] == "confirm_pending":
                    firsts[dg] = co


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
