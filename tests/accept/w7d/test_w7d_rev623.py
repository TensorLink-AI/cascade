"""W7d / W8b, plan rev6.23 (owner 2026-11-24): the chain replay
(`cascade.audit.lane_d_chain`) carries KING_D's tenure between settlements
(0 at genesis, 1 after a crown, else + 1), checks each report's
``crown_bar`` against it and the config's bar, judges every verdict against
that bar, and pays the court by ``[lane_d] court_weights``.

On main the reports carry no ``crown_bar`` and `court_shares` decays by the
retired ``king_decay``."""

from __future__ import annotations

import json
import math

import pytest

from .helpers import build_all, describe, doctored, has_path, replay, replay_module


@pytest.fixture(scope="module")
def chains(tmp_path_factory):
    return build_all(tmp_path_factory.mktemp("w7d-rev623"))


def _tenures(c) -> list[int]:
    reps = c.pub.reports()
    return [reps[b]["crown_bar"]["tenure"] for b in c.pub.boundaries]


def _want_tenures(c) -> list[int]:
    """0 at genesis; 1 after a settlement that crowned (a miner, or a house
    retake), else the previous tenure + 1."""
    reps = c.pub.reports()
    out, t = [], 0
    for b in c.pub.boundaries:
        out.append(t)
        r = reps[b]
        retake = r["king_after"]["house"] and not r["king_before"]["house"]
        t = 1 if (r["crowned"] is not None or retake) else t + 1
    return out


@pytest.mark.parametrize("name", ["gated", "held", "house_bar", "refresh"])
def test_every_honest_chain_states_the_chain_tenure_and_bar(chains, name):
    c = chains[name]
    assert _tenures(c) == _want_tenures(c)
    reps = c.pub.reports()
    for b in c.pub.boundaries:
        cb = reps[b]["crown_bar"]
        assert cb["bar"] == c.cfg.crown_bar(cb["tenure"])
    mod = replay_module()
    res = replay(c.pub)
    assert all(r.status == mod.MATCH for r in res), describe(res)


@pytest.mark.parametrize("name", ["gated", "house_bar"])
def test_a_tenure_off_the_chain_is_a_mismatch(chains, name):
    c = chains[name]
    n = len(c.pub.boundaries)

    def bump(rep):
        t = int(rep["crown_bar"]["tenure"]) + 3
        rep["crown_bar"] = {"tenure": t, "bar": c.cfg.crown_bar(t)}

    reports, receipts = doctored(c.pub, n, bump)
    res = replay(c.pub, reports=reports, receipts=receipts)
    mod = replay_module()
    assert res[-1].status == mod.MISMATCH and has_path(res[-1], "crown_bar"), describe(res)


def test_a_bar_off_the_config_is_a_mismatch(chains):
    c = chains["gated"]

    def lower(rep):
        rep["crown_bar"]["bar"] = float(c.cfg.d_margin) / 2

    reports, receipts = doctored(c.pub, 1, lower)
    res = replay(c.pub, reports=reports, receipts=receipts)
    assert has_path(res[0], "crown_bar"), describe(res)


def test_the_state_snapshot_carries_the_next_tenure(chains):
    c = chains["house_bar"]
    want = _want_tenures(c)[1:]
    reps = c.pub.reports()
    for i, b in enumerate(c.pub.boundaries[:-1]):
        raw = c.pub.states.get(b)
        if raw is None:
            continue
        assert json.loads(raw)["king_tenure"] == want[i], b
    last = c.pub.boundaries[-1]
    r = reps[last]
    retake = r["king_after"]["house"] and not r["king_before"]["house"]
    nxt = 1 if (r["crowned"] is not None or retake) else r["crown_bar"]["tenure"] + 1
    if c.pub.states.get(last) is not None:
        assert json.loads(c.pub.states[last])["king_tenure"] == nxt


def test_the_replay_pays_court_weights():
    """`court_shares`: KING_D, prior 1, prior 2 at 0.80 / 0.15 / 0.05 of the
    share; a short court renormalises over its slots."""
    from cascade.audit.lane_d_chain import court_shares, crown_bar_at

    from .helpers import lane_d_cfg

    cfg = lane_d_cfg()
    s = float(cfg.share)
    assert court_shares(cfg, 3) == pytest.approx([s * 0.80, s * 0.15, s * 0.05])
    two = court_shares(cfg, 2)
    assert math.isclose(two[0], s * 0.80 / 0.95) and math.isclose(two[1], s * 0.15 / 0.95)
    assert court_shares(cfg, 1) == pytest.approx([s])
    assert crown_bar_at(cfg, 0) == cfg.d_margin_start
    assert crown_bar_at(cfg, cfg.d_margin_warmup_settlements) == cfg.d_margin
    assert crown_bar_at(cfg, 10 ** 6) == cfg.d_margin
