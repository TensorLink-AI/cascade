"""W5e: the W3b rotating gate wired into `settle_lane_d` (plan rev6 §1.3b,
T3, T4, T17; W3b1 `lane_d_gate`, W3b2 `lane_d_bench`). README.md has the
contract.

rev6.11 (b): a crown is decided at the FIRST verdict, so a first clear is
the crowning candidate and is gated at the settlement it clears. The W5g
follow-up lives here (the W5g row is not edited): a `bench_hold` is a
first-clear candidate the gate could not decide, and the hold sweep re-gates
it on its first clear (no second duel). The bench pod is the one external
service: `helpers.BenchHost` returns recorded suite outputs through a real
`lane_d_bench.SweepSession`.

On main a first clear is not gated (it goes `confirm_pending`), so these
tests fail on an AssertionError.
"""

from __future__ import annotations

from .helpers import (
    GATE_SUITES,
    T_X,
    T_Y,
    BenchHost,
    bench_session,
    bench_table,
    block_hash,
    boundary,
    drawn_suite,
    entry,
    entry_of,
    expected_gate_record,
    house,
    lane_d_cfg,
    model_forecasts,
    read_state,
    rows_of,
    settle_gated,
    snapshot,
    uids,
    verdict_of,
)


def _fc(recs, models: dict) -> dict:
    return {dg: model_forecasts(recs, q, name) for dg, (q, name) in models.items()}


def _gate_view(report) -> dict | None:
    return report.get("gate")


def _assert_gate_record(report, cfg, b, hash_, sentinel=None):
    exp = expected_gate_record(cfg, b, hash_, sentinel=sentinel)
    got = _gate_view(report)
    assert isinstance(got, dict), f"the report carries no gate record: {got!r}"
    assert {k: got.get(k) for k in exp} == exp


class Run:
    """X first-clears against the genesis house H0 at boundary(1); that clear
    is the crowning candidate and is gated on bench `spec`."""

    def __init__(self, tmp_path, *, spec, sentinel=None, hash1=None, cfg=None,
                 x_quality="sharp"):
        self.cfg = cfg or lane_d_cfg()
        self.h0 = house()
        self.x = entry("x", reveal_ts=T_X)
        self.path = tmp_path / "state" / "lane_d_state.json"
        self.sentinel = sentinel
        self.models = {self.h0["digest"]: ("bad", "house-0"), self.x["digest"]: (x_quality, "x")}
        self.pool1, self.recs1 = snapshot(tmp_path, 1)
        self.host1 = BenchHost(bench_table(spec(self)))
        self.hash1 = hash1 or block_hash("b1")
        self.s1 = settle_gated(self.cfg, boundary(1), self.pool1, [self.x], self.h0,
                               _fc(self.recs1, self.models), uids(self.x), self.path,
                               hash_=self.hash1, bench=bench_session(self.host1, "sess-1"),
                               sentinel=sentinel)
        v = verdict_of(self.s1.report, self.x["digest"])
        assert v is not None and v.get("stage", "first") == "first" and v["cleared"] is True, \
            f"fixture: X did not clear at its first verdict: {v}"

    @property
    def drawn(self) -> str:
        return drawn_suite(self.cfg, self.hash1, self.sentinel)

    def state_entry(self) -> dict:
        return read_state(self.path)["entries"][self.x["digest"]]


def test_gate_pass_crowns_at_the_first_verdict_and_records_the_draw(tmp_path):
    r = Run(tmp_path, spec=lambda r: {r.x["digest"]: 0.9, r.h0["digest"]: 1.0})
    assert r.s1.report["crowned"] == r.x["digest"]
    assert read_state(r.path)["king"]["digest"] == r.x["digest"]
    assert verdict_of(r.s1.report, r.x["digest"])["outcome"] == "crowned"
    assert {r.x["digest"], r.h0["digest"]} <= r.host1.swept_digests(), \
        "the candidate, KING_D and the house must be swept in the settlement's session"
    assert r.drawn in r.host1.swept_suites()
    _assert_gate_record(r.s1.report, r.cfg, boundary(1), r.hash1)


def test_a_first_clear_is_gated_at_the_settlement_it_clears(tmp_path):
    r = Run(tmp_path, spec=lambda r: {r.x["digest"]: 0.9, r.h0["digest"]: 1.0})
    assert r.host1.calls, "a first clear is the crowning candidate (rev6.11): it must be gated"
    assert r.x["digest"] in r.host1.swept_digests()
    assert verdict_of(r.s1.report, r.x["digest"])["outcome"] != "confirm_pending"


def test_t4_gate_regression_is_lost_with_a_typed_reason(tmp_path):
    r = Run(tmp_path, spec=lambda r: {r.x["digest"]: 1.10, r.h0["digest"]: 1.0})
    assert r.s1.report["crowned"] is None
    assert read_state(r.path)["king"]["digest"] == r.h0["digest"]
    rec = r.state_entry()
    assert (rec["status"], rec["reason"]) == ("lost", "gate_regression")
    e = entry_of(r.s1.report, r.x["digest"])
    assert (e["status"], e["reason"]) == ("lost", "gate_regression")
    assert verdict_of(r.s1.report, r.x["digest"])["outcome"] == "lost"
    _assert_gate_record(r.s1.report, r.cfg, boundary(1), r.hash1)


def test_t17_candidate_fault_on_the_drawn_suite_is_checkpoint_fault(tmp_path):
    r = Run(tmp_path, spec=lambda r: {r.x["digest"]: {s: "nan" for s in GATE_SUITES},
                                      r.h0["digest"]: 1.0})
    assert r.s1.report["crowned"] is None
    assert read_state(r.path)["king"]["digest"] == r.h0["digest"]
    rec = r.state_entry()
    assert (rec["status"], rec["reason"]) == ("closed", "checkpoint_fault")
    e = entry_of(r.s1.report, r.x["digest"])
    assert (e["status"], e["reason"]) == ("closed", "checkpoint_fault")
    assert verdict_of(r.s1.report, r.x["digest"])["outcome"] != "crowned"


def test_reference_fault_is_bench_hold_and_the_first_clear_survives(tmp_path):
    """W5g follow-up (in W5e's row): `bench_hold` = a first-clear candidate
    the gate could not decide. At the next settlement, with no fresh window,
    the hold sweep re-gates that first clear on the new draw and crowns it,
    without a new duel."""
    r = Run(tmp_path, spec=lambda r: {r.x["digest"]: 0.9,
                                      r.h0["digest"]: {s: "nan" for s in GATE_SUITES}})
    assert r.s1.report["crowned"] is None
    assert read_state(r.path)["king"]["digest"] == r.h0["digest"]
    rec = r.state_entry()
    assert rec["reason"] == "bench_hold"
    assert rec["status"] == "hold", \
        f"a bench hold is open and not the miner's fault (no confirm_pending): {rec['status']}"
    e = entry_of(r.s1.report, r.x["digest"])
    assert e["reason"] == "bench_hold" and e["status"] == rec["status"]
    assert verdict_of(r.s1.report, r.x["digest"])["outcome"] not in ("crowned", "lost")
    host2 = BenchHost(bench_table({r.x["digest"]: 0.9, r.h0["digest"]: 1.0}))
    hash2 = block_hash("b2")
    s2 = settle_gated(r.cfg, boundary(2), r.pool1, [r.x], r.h0, _fc(r.recs1, r.models),
                      uids(r.x), r.path, hash_=hash2, bench=bench_session(host2, "sess-2"))
    assert s2.report["crowned"] == r.x["digest"], "a bench hold burned the candidate's chance"
    assert rows_of(s2.report, r.x["digest"]) == set(), \
        "a held first clear was re-duelled instead of re-gated"
    assert read_state(r.path)["king"]["digest"] == r.x["digest"]
    assert r.x["digest"] in host2.swept_digests()
    _assert_gate_record(s2.report, r.cfg, boundary(2), hash2)


def test_the_sentinel_is_swept_and_never_gates(tmp_path):
    sentinel = "boom"

    def spec(r):
        return {r.x["digest"]: {**{s: 0.9 for s in GATE_SUITES}, sentinel: 1.5},
                r.h0["digest"]: 1.0}

    r = Run(tmp_path, spec=spec, sentinel=sentinel)
    assert r.drawn != sentinel
    assert sentinel in r.host1.swept_suites(), "the era's sentinel must be swept and recorded"
    assert r.s1.report["crowned"] == r.x["digest"], "a sentinel regression blocked a crowning"
    _assert_gate_record(r.s1.report, r.cfg, boundary(1), r.hash1, sentinel=sentinel)
    assert _gate_view(r.s1.report)["suite"] != sentinel


def _hashes_with_different_draws(cfg):
    first = block_hash("draw-0")
    a = drawn_suite(cfg, first)
    for i in range(1, 64):
        h = block_hash(f"draw-{i}")
        if drawn_suite(cfg, h) != a:
            return first, h
    raise AssertionError("fixture: no two block hashes draw different suites")


def test_the_draw_follows_the_boundary_block_hash(tmp_path):
    cfg = lane_d_cfg()
    ha, hb = _hashes_with_different_draws(cfg)
    bad_on = drawn_suite(cfg, ha)

    def spec(r):
        return {r.x["digest"]: {**{s: 0.9 for s in GATE_SUITES}, bad_on: 1.10},
                r.h0["digest"]: 1.0}

    ra = Run(tmp_path / "a", spec=spec, hash1=ha, cfg=cfg)
    rb = Run(tmp_path / "b", spec=spec, hash1=hb, cfg=cfg)
    assert _gate_view(ra.s1.report)["suite"] == bad_on
    assert _gate_view(rb.s1.report)["suite"] == drawn_suite(cfg, hb) != bad_on
    assert ra.state_entry()["reason"] == "gate_regression"
    assert rb.s1.report["crowned"] == rb.x["digest"]


def test_losers_are_never_swept(tmp_path):
    cfg = lane_d_cfg()
    h0 = house()
    x = entry("x", reveal_ts=T_X)
    z = entry("z", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1)
    models = {h0["digest"]: ("bad", "house-0"), x["digest"]: ("sharp", "x"),
              z["digest"]: ("bad", "z")}
    host1 = BenchHost(bench_table({x["digest"]: 0.9, z["digest"]: 0.9, h0["digest"]: 1.0}))
    s1 = settle_gated(cfg, boundary(1), pool1, [x, z], h0, _fc(recs1, models), uids(x, z), path,
                      hash_=block_hash("b1"), bench=bench_session(host1, "sess-1"))
    assert read_state(path)["entries"][z["digest"]]["status"] == "lost"
    assert s1.report["crowned"] == x["digest"]
    swept = host1.swept_digests()
    assert x["digest"] in swept
    assert z["digest"] not in swept, swept


def test_no_clear_means_no_sweep_and_no_gate_record(tmp_path):
    cfg = lane_d_cfg()
    h0 = house()
    z = entry("z", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    pool1, recs1 = snapshot(tmp_path, 1)
    models = {h0["digest"]: ("fair", "house-0"), z["digest"]: ("bad", "z")}
    host1 = BenchHost(bench_table({z["digest"]: 0.9, h0["digest"]: 1.0}))
    s1 = settle_gated(cfg, boundary(1), pool1, [z], h0, _fc(recs1, models), uids(z), path,
                      hash_=block_hash("b1"), bench=bench_session(host1, "sess-1"))
    assert host1.calls == [] and _gate_view(s1.report) is None


def test_every_first_clear_is_gated_and_the_best_passing_one_crowns(tmp_path):
    cfg = lane_d_cfg()
    h0 = house()
    x1 = entry("x1", reveal_ts=T_X)
    x2 = entry("x2", reveal_ts=T_X)
    path = tmp_path / "state" / "lane_d_state.json"
    models = {h0["digest"]: ("bad", "house-0"), x1["digest"]: ("sharp", "x1"),
              x2["digest"]: ("good", "x2")}
    pool1, recs1 = snapshot(tmp_path, 1)
    host1 = BenchHost(bench_table({x1["digest"]: 1.10, x2["digest"]: 0.9, h0["digest"]: 1.0}))
    s1 = settle_gated(cfg, boundary(1), pool1, [x1, x2], h0, _fc(recs1, models), uids(x1, x2),
                      path, hash_=block_hash("b1"), bench=bench_session(host1, "sess-1"))
    v1, v2 = verdict_of(s1.report, x1["digest"]), verdict_of(s1.report, x2["digest"])
    assert v1["cleared"] and v2["cleared"], f"fixture: {v1} {v2}"
    assert v1["lcb"] > v2["lcb"], "fixture: x1 must be the max-T best"
    state = read_state(path)
    assert (state["entries"][x1["digest"]]["status"],
            state["entries"][x1["digest"]]["reason"]) == ("lost", "gate_regression")
    assert s1.report["crowned"] == x2["digest"], \
        "the gate result of the best clear must not block a passing runner-up"
    assert state["king"]["digest"] == x2["digest"]
    assert {x1["digest"], x2["digest"]} <= host1.swept_digests()


def test_t3_with_a_miner_king_the_house_still_bars_the_gate(tmp_path):
    cfg = lane_d_cfg()
    h0 = house()
    k = entry("k", reveal_ts=T_X)
    y = entry("y", reveal_ts=T_Y, after=1)
    path = tmp_path / "state" / "lane_d_state.json"
    models = {h0["digest"]: ("bad", "house-0"), k["digest"]: ("fair", "k"),
              y["digest"]: ("sharp", "y")}
    pool, recs = snapshot(tmp_path, 1)
    settle_gated(cfg, boundary(1), pool, [k], h0, _fc(recs, models), uids(k), path,
                 hash_=block_hash("b1"),
                 bench=bench_session(BenchHost(bench_table({k["digest"]: 0.9,
                                                            h0["digest"]: 1.0})), "sess-1"))
    assert read_state(path)["king"]["digest"] == k["digest"], "fixture: K was not crowned"
    # Y beats K on the drawn suite but regresses against the house.
    pool, recs = snapshot(tmp_path, 2, t=T_Y)
    host = BenchHost(bench_table({y["digest"]: 0.95, k["digest"]: 1.0, h0["digest"]: 0.8}))
    s = settle_gated(cfg, boundary(2), pool, [k, y], h0, _fc(recs, models), uids(k, y),
                     path, hash_=block_hash("b2"), bench=bench_session(host, "sess-2"))
    v = verdict_of(s.report, y["digest"])
    assert v is not None and v.get("stage", "first") == "first" and v["cleared"], f"fixture: {v}"
    assert v["king_digest"] == k["digest"]
    assert {y["digest"], k["digest"], h0["digest"]} <= host.swept_digests()
    rec = read_state(path)["entries"][y["digest"]]
    assert (rec["status"], rec["reason"]) == ("lost", "gate_regression")
    assert s.report["crowned"] is None
    assert read_state(path)["king"]["digest"] == k["digest"]
