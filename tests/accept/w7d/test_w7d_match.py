"""W7d: the replay MATCHes every honest chain and re-derives its verdicts,
crown, pay shares and receipt digests (plan rev6 §3.0 W7d, §14.3 items 1-2,
§1.5 "re-derive the lane D weights from the receipt chain", invariant 7).

Every chain is real `settle_lane_d` runs (helpers.CHAINS): gated, held
(bench_hold then a re-gated crown), house bar (a miner king, the house as a
second bar, a prior king in the court) and a house refresh (rev6.11: no
confirmation duel, every crown at a first verdict).
On main the module is absent: `ModuleNotFoundError: cascade.satellite`.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import re
import subprocess
import sys

import pytest

from .helpers import (
    CHAIN_TIMEOUT,
    REPLAY_SCRIPT,
    REPO,
    UNVERIFIABLE_REASONS,
    body_sha,
    build_all,
    court_shares,
    describe,
    doctored,
    replay,
    replay_module,
    save_publication,
    sha,
)

PINNED_FIELDS = ("boundary", "status", "mismatches", "reason", "crowned", "king_after",
                 "outcomes", "gate", "pay", "report_sha256", "posterior_state_sha256")
PATH_RE = re.compile(r"^[a-z][a-z0-9_]*(\[[0-9]+\]|\.[a-z0-9_]+)*$")


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-match"))


def _reports(c):
    reps = c.pub.reports()
    return [reps[b] for b in c.pub.boundaries]


def _all_match(results, pub):
    mod = replay_module()
    assert [r.boundary for r in results] == pub.boundaries, describe(results)
    assert all(r.status == mod.MATCH for r in results), describe(results)
    assert all(tuple(r.mismatches) == () and r.reason is None for r in results), \
        describe(results)


def test_result_type_and_status_constants():
    mod = replay_module()
    assert len({mod.MATCH, mod.MISMATCH, mod.UNVERIFIABLE}) == 3
    assert all(isinstance(s, str) for s in (mod.MATCH, mod.MISMATCH, mod.UNVERIFIABLE))
    assert set(UNVERIFIABLE_REASONS) <= set(mod.UNVERIFIABLE_REASONS)
    assert dataclasses.is_dataclass(mod.SettlementReplay)
    names = {f.name for f in dataclasses.fields(mod.SettlementReplay)}
    assert set(PINNED_FIELDS) <= names, sorted(set(PINNED_FIELDS) - names)
    assert mod.SettlementReplay.__dataclass_params__.frozen


@pytest.mark.parametrize("name", ["gated", "held", "house_bar", "refresh"])
def test_every_honest_chain_matches_from_the_published_report_files(chains, name):
    pub = chains[name].pub
    _all_match(replay(pub), pub)


@pytest.mark.parametrize("name", ["gated", "house_bar", "refresh"])
def test_an_honest_chain_matches_from_a_report_mapping(chains, name):
    pub = chains[name].pub
    _all_match(replay(pub, reports=pub.reports()), pub)


@pytest.mark.parametrize("name", ["gated", "held", "house_bar", "refresh"])
def test_an_honest_chain_matches_with_state_snapshots_and_metagraphs(chains, name):
    pub = chains[name].pub
    res = replay(pub, states=dict(pub.states), uid_of=dict(pub.uid_of))
    _all_match(res, pub)
    for r, b in zip(res, pub.boundaries, strict=True):
        assert r.posterior_state_sha256 == sha(pub.states[b])


@pytest.mark.parametrize("name", ["gated", "held", "house_bar", "refresh"])
def test_the_replay_rederives_crown_king_verdicts_and_gate(chains, name):
    c = chains[name]
    res = replay(c.pub)
    _all_match(res, c.pub)
    for r, rep in zip(res, _reports(c), strict=True):
        assert r.crowned == rep["crowned"], (rep["boundary"], r.crowned)
        assert r.king_after == rep["king_after"]["digest"]
        assert dict(r.outcomes) == {v["digest"]: v["outcome"] for v in rep["verdicts"]}
        want_gate = {g["digest"]: g["outcome"] for g in (rep.get("gate") or {}).get("results", [])}
        assert dict(r.gate) == want_gate
        assert r.report_sha256 == body_sha(rep)


def test_the_held_candidate_is_regated_and_crowned_without_a_verdict(chains):
    c = chains["held"]
    res = replay(c.pub)
    _all_match(res, c.pub)
    x = c.x["digest"]
    # The held first clear's verdict outcome is `cleared` (status `hold`,
    # reason `bench_hold`; test_w7d_fixture_chains, W5g, W17).
    assert res[1].crowned is None and dict(res[1].outcomes) == {x: "cleared"}
    assert dict(res[1].gate) == {}, "no gate ran at the held settlement"
    assert res[2].crowned == x and dict(res[2].outcomes) == {} and dict(res[2].gate) == {x: "pass"}


def test_every_crown_is_at_the_first_verdict_where_the_producer_put_it(chains):
    """rev6.11 (b): crowned = first clear ∧ A10 ∧ gate, at the entry's first
    (and only) verdict."""
    for name, want in (("gated", [None, "x", None]), ("house_bar", [None, "x", "y"]),
                       ("refresh", [None, "x", None])):
        c = chains[name]
        res = replay(c.pub)
        _all_match(res, c.pub)
        assert [r.crowned for r in res] == [None if w is None else getattr(c, w)["digest"]
                                            for w in want], name
        for r in res:
            if r.crowned is not None:
                assert dict(r.outcomes).get(r.crowned) == "crowned", (name, r.boundary)


@pytest.mark.parametrize("name", ["gated", "held", "house_bar"])
def test_the_replay_rederives_pay_shares(chains, name):
    c = chains[name]
    res = replay(c.pub, uid_of=dict(c.pub.uid_of))
    _all_match(res, c.pub)
    for r, rep in zip(res, _reports(c), strict=True):
        want = court_shares(c.cfg, rep["pay"]["court"])
        got = [tuple(s) for s in r.pay]
        assert [(d, u) for d, u, _ in got] == [(d, u) for d, u, _ in want], rep["boundary"]
        for (_, _, g), (_, _, w) in zip(got, want, strict=True):
            assert math.isclose(g, w, rel_tol=1e-12, abs_tol=1e-15)
        assert math.isclose(sum(s for _, _, s in got), float(c.cfg.share), rel_tol=1e-12)


def test_the_house_bar_court_pays_the_new_king_and_the_prior_king_by_court_weights(chains):
    c = chains["house_bar"]
    last = replay(c.pub, uid_of=dict(c.pub.uid_of))[-1]
    assert len(last.pay) == 2, f"fixture: Y crowned over X, two paid slots; got {last.pay}"
    # rev6.23: a two-slot court renormalises court_weights[:2] (80 / 15).
    assert hasattr(c.cfg, "court_weights"), "[lane_d] court_weights is not parsed (rev6.23)"
    share, (w0, w1) = float(c.cfg.share), [float(w) for w in c.cfg.court_weights[:2]]
    (y_d, y_uid, y_s), (x_d, x_uid, x_s) = [tuple(s) for s in last.pay]
    assert (y_d, y_uid, x_d, x_uid) == (c.y["digest"], 12, c.x["digest"], 11)
    assert math.isclose(y_s, share * w0 / (w0 + w1)) and math.isclose(x_s, share * w1 / (w0 + w1))
    assert math.isclose(y_s, share * 0.80 / 0.95) and math.isclose(x_s, share * 0.15 / 0.95)
    # The weights the producer's emission derives from the same court agree
    # (the test may import it; the replay may not).
    from cascade.shared.emission import lane_d_vector

    vec = lane_d_vector(16, share=share, court_uids=[12, 11],
                        court_weights=c.cfg.court_weights, burn_uid=0)
    assert math.isclose(vec[12], y_s) and math.isclose(vec[11], x_s)


def test_a_house_held_slot_burns_in_the_rederived_pay(chains):
    c = chains["gated"]
    first = replay(c.pub)[0]
    ((d, uid, s),) = [tuple(x) for x in first.pay]
    assert d == c.h0["digest"] and uid is None and math.isclose(s, float(c.cfg.share))


def test_the_replay_is_pure(chains):
    pub = chains["house_bar"].pub
    files = {p.name: p.read_bytes() for p in sorted(pub.reports_dir.iterdir())}
    receipts, states, uids = (copy.deepcopy(pub.receipts), dict(pub.states),
                              copy.deepcopy(pub.uid_of))
    reports = pub.reports()
    before = copy.deepcopy(reports)
    replay(pub, reports=reports, states=states, uid_of=uids)
    replay(pub)
    assert reports == before and pub.receipts == receipts and uids == pub.uid_of
    assert states == pub.states
    assert {p.name: p.read_bytes() for p in sorted(pub.reports_dir.iterdir())} == files


def test_inv7_extensions_never_change_a_rederived_verdict(chains):
    """§14.3 item 4 / invariant 7: the optional `extensions` object is never
    consensus. A re-signed report whose `extensions` claims another crown and
    carries free text still MATCHes, with the same re-derived values."""
    c = chains["gated"]
    honest = replay(c.pub)

    def note(rep):
        rep["extensions"] = {"crowned": None, "note": "the house kept the throne",
                             "outcome": "lost"}

    reports, receipts = doctored(c.pub, 2, note)
    res = replay(c.pub, reports=reports, receipts=receipts)
    _all_match(res, c.pub)
    assert res[1].crowned == honest[1].crowned == c.x["digest"]
    assert dict(res[1].outcomes) == dict(honest[1].outcomes)
    assert [tuple(s) for s in res[1].pay] == [tuple(s) for s in honest[1].pay]


def test_inv7_a_mismatch_names_field_paths_never_free_text(chains):
    mod = replay_module()
    c = chains["gated"]

    def lie(rep):
        rep["verdicts"][0]["outcome"] = "lost"
        rep["params"]["alpha"] = 0.1

    reports, receipts = doctored(c.pub, 2, lie)
    res = replay(c.pub, reports=reports, receipts=receipts)
    assert res[1].status == mod.MISMATCH, describe(res)
    assert res[1].mismatches, "a MISMATCH without a field path"
    assert len(set(res[1].mismatches)) == len(res[1].mismatches)
    bad = [p for p in res[1].mismatches if not PATH_RE.fullmatch(p)]
    assert not bad, f"mismatches that are not field paths: {bad}"


def _run(saved, seed: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONHASHSEED"] = seed
    out = subprocess.run([sys.executable, "-P", "-c", REPLAY_SCRIPT, str(saved)],
                         cwd=str(REPO), env=env, capture_output=True, text=True,
                         timeout=CHAIN_TIMEOUT)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("name", ["house_bar", "refresh"])
def test_replay_results_are_independent_of_the_hash_seed(chains, tmp_path, name):
    saved = save_publication(chains[name].pub, tmp_path / "pub")
    runs = [_run(saved, seed) for seed in ("0", "1", "4242")]
    assert runs[0]["results"], "no result"
    assert runs[0]["results"] == runs[1]["results"] == runs[2]["results"]
    mod = replay_module()
    want = [mod.MATCH] * len(chains[name].pub.boundaries)
    assert [r["status"] for r in runs[0]["results"]] == want
