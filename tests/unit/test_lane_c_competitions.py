"""Concurrent lane-C competitions (owner 2026-10-20, DEC-CA-0056 §Concurrent
competitions): lane C's share is split by competition WEIGHT (normalised over
every configured competition), then equally among that competition's targets;
each competition has its own start / end; after its end nothing new is
admitted or judged and held wins wind down (``finish_full_pay`` capped at
``wind_down_max_eras``, or ``stop``). Consensus-deterministic: the audit
re-derives the per-competition split and rejects a forged one."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.audit.checks import _lane_c_paid_wins
from cascade.shared.config import load_chain_config
from cascade.shared.config.lane_c import (
    COMPETITION_KINDS,
    LaneCCompetition,
    parse_lane_c,
    register_competition_kind,
)
from cascade.shared.config.marketcast import parse_marketcast
from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts
from cascade.shared.era import era_length_blocks
from cascade.shared.lane_c import (
    LaneCState,
    TargetWin,
    active_targets,
    competition_closed,
    competition_state,
    lane_c_pay_record,
    lane_c_split,
    lane_c_vector,
    pay_wins,
)

ROOT = Path(__file__).resolve().parents[2]
ERA = 7200
FAKE, FAKE2 = "fakecomp", "fakecomp2"
for _k in (FAKE, FAKE2):
    # a second (and third) forward-only competition kind, test-only
    if _k not in COMPETITION_KINDS:
        register_competition_kind(_k, eval="test forward-only eval", forward_only=True,
                                  parse_value=lambda v: v)
M, V = "mkt:returns", "mkt:vol"
FA, FB, FC = f"{FAKE}:a", f"{FAKE}:b", f"{FAKE}:c"
MC = parse_marketcast({"from_block": 2 * ERA})          # MarketCast starts at 2 eras


def _lc(comps: dict, targets=(M, V, FA, FB, FC), **kw) -> object:
    raw = {"targets": list(targets), "competitions": comps, "from_block": 2 * ERA, **kw}
    return parse_lane_c(raw, v2_from_block=ERA, era_blocks=ERA, era_settlements=4,
                        lane_b_share=0.5, kind_from_blocks={"mkt": MC.from_block})


def _win(t, hk, block, uid, full_eras=5):
    return TargetWin(target=t, hotkey=hk, uid=uid, round_id=f"r{block}", win_block=block,
                     gain_lcb=0.03, tier=0.02, full_eras=full_eras)


def _vec(rec, lc, n=16):
    return lane_c_vector(n, cfg=lc, wins=pay_wins(rec), burn_uid=0,
                         slot_frac=rec["slot_frac"])


# ── weight first, then per target ────────────────────────────────────────────


@pytest.mark.parametrize("w_mkt, w_fake", [(1.0, 1.0), (2.0, 1.0)])
def test_share_splits_by_weight_then_per_target(w_mkt, w_fake):
    lc = _lc({"mkt": {"weight": w_mkt}, FAKE: {"weight": w_fake, "from_block": 2 * ERA}})
    b = 3 * ERA
    comps, slot = lane_c_split(lc, b, ERA, MC)
    tot = w_mkt + w_fake
    assert [c["kind"] for c in comps] == [FAKE, "mkt"]
    assert {c["kind"]: c["state"] for c in comps} == {FAKE: "active", "mkt": "active"}
    assert slot[M] == slot[V] == pytest.approx(w_mkt / tot / 2)        # 2 mkt targets
    for t in (FA, FB, FC):
        assert slot[t] == pytest.approx(w_fake / tot / 3)              # 3 fake targets
    st = LaneCState(slots={t: _win(t, f"h{i}", 2 * ERA + 1800, uid=i + 1)
                           for i, t in enumerate((M, V, FA, FB, FC))})
    rec = lane_c_pay_record(st, block=b, era_blocks=ERA, cfg=lc, n_uids=16, mc=MC)
    v = _vec(rec, lc)
    assert sum(v) == pytest.approx(lc.share) and v[0] == pytest.approx(0.0, abs=1e-15)
    assert v[1] == pytest.approx(lc.share * w_mkt / tot / 2)
    assert v[3] == pytest.approx(lc.share * w_fake / tot / 3)


def test_pending_competition_burns_its_part_and_never_moves_another():
    lc = _lc({"mkt": {"weight": 2.0}, FAKE: {"weight": 1.0, "from_block": 4 * ERA}})
    st = LaneCState(slots={M: _win(M, "alice", 2 * ERA + 1800, 1),
                           FA: _win(FA, "bob", 2 * ERA + 1800, 2)})
    pend = lane_c_pay_record(st, block=3 * ERA, era_blocks=ERA, cfg=lc, n_uids=8, mc=MC)
    live = lane_c_pay_record(st, block=4 * ERA, era_blocks=ERA, cfg=lc, n_uids=8, mc=MC)
    by = {c["kind"]: c for c in pend["competitions"]}
    assert by[FAKE]["state"] == "pending" and by[FAKE]["slot_frac"] == 0.0
    assert by["mkt"]["state"] == "active"
    assert active_targets(lc, 3 * ERA, MC) == (M, V)
    assert [w["target"] for w in pend["wins"]] == [M]          # bob's slot is not paid yet
    vp, vl = _vec(pend, lc, 8), _vec(live, lc, 8)
    assert vp[1] == vl[1] == pytest.approx(lc.share * 2 / 3 / 2)   # alice: unchanged
    # the pending competition's whole third burns (plus mkt:vol's empty slot)
    assert vp[0] == pytest.approx(lc.share * (1 / 3 + 1 / 3))
    assert vl[2] == pytest.approx(lc.share / 3 / 3)
    for v in (vp, vl):
        assert sum(v) == pytest.approx(lc.share)


def test_configured_competition_without_targets_burns_its_weight():
    lc = _lc({"mkt": {}, FAKE2: {"from_block": 2 * ERA}}, targets=(M, V))
    comps, slot = lane_c_split(lc, 3 * ERA, ERA, MC)
    assert {c["kind"]: c["frac"] for c in comps} == {"mkt": 0.5, FAKE2: 0.5}
    assert set(slot) == {M, V} and sum(slot.values()) == pytest.approx(0.5)


# ── the end: admission, dispatch, verdicts, wind-down ────────────────────────


def _ending(**fake):
    return _lc({"mkt": {"weight": 1.0},
                FAKE: {"weight": 1.0, "from_block": 2 * ERA, "until_block": 6 * ERA, **fake}})


def test_states_and_admission_around_the_end():
    lc = _ending()
    comp = lc.competition(FAKE)
    assert competition_state(comp, 6 * ERA, ERA) == "active"       # the settlement AT it
    assert competition_state(comp, 6 * ERA + 1800, ERA) == "winding_down"
    assert competition_state(comp, 11 * ERA + 5400, ERA) == "winding_down"
    assert competition_state(comp, 12 * ERA, ERA) == "ended"       # 6 eras after the end
    assert FA in active_targets(lc, 6 * ERA, MC) and FA not in active_targets(lc, 6 * ERA + 1, MC)
    assert competition_closed(lc, FA, 6 * ERA - 1) is None
    assert "ended at block" in competition_closed(lc, FA, 6 * ERA)
    assert competition_closed(lc, M, 10**9) is None                # mkt has no end
    stop = _ending(wind_down="stop")
    assert competition_state(stop.competition(FAKE), 6 * ERA + 1800, ERA) == "ended"


def test_wind_down_pays_full_through_full_pay_capped_then_stops():
    lc = _ending()
    st = LaneCState(slots={FA: _win(FA, "short", 5 * ERA, 1, full_eras=5),     # to 10 eras
                           FB: _win(FB, "long", 5 * ERA, 2, full_eras=10),     # capped at 12
                           M: _win(M, "mk", 5 * ERA, 3, full_eras=5)})

    def paid(block):
        rec = lane_c_pay_record(st, block=block, era_blocks=ERA, cfg=lc, n_uids=8, mc=MC)
        v = _vec(rec, lc, 8)
        assert sum(v) == pytest.approx(lc.share)
        return {w["target"]: (w["multiplier"], w.get("wind_down", False))
                for w in rec["wins"]}, rec

    p, rec = paid(7 * ERA)
    assert p[FA] == (1.0, True) and p[FB] == (1.0, True)
    assert {c["kind"]: c["state"] for c in rec["competitions"]}[FAKE] == "winding_down"
    p, _ = paid(9 * ERA + 5400)
    assert FA in p and FB in p
    p, _ = paid(10 * ERA)                     # "short" leaves its full-pay period: no decay tail
    assert FA not in p and p[FB] == (1.0, True)
    assert p[M] == (0.5, False)               # an active competition still decays normally
    p, _ = paid(11 * ERA + 5400)
    assert FB in p
    p, rec = paid(12 * ERA)                   # wind_down_max_eras = 6 eras after the end
    assert FA not in p and FB not in p
    assert {c["kind"]: c["state"] for c in rec["competitions"]}[FAKE] == "ended"


def test_wind_down_stop_ends_pay_at_the_end():
    lc = _ending(wind_down="stop")
    st = LaneCState(slots={FA: _win(FA, "a", 5 * ERA, 1, full_eras=10)})
    at = lane_c_pay_record(st, block=6 * ERA, era_blocks=ERA, cfg=lc, n_uids=8, mc=MC)
    after = lane_c_pay_record(st, block=6 * ERA + 1800, era_blocks=ERA, cfg=lc, n_uids=8, mc=MC)
    assert [w["target"] for w in at["wins"]] == [FA]
    assert after["wins"] == [] and FA not in after["slot_frac"]
    v = _vec(after, lc, 8)
    assert v[0] == pytest.approx(lc.share) and sum(v) == pytest.approx(lc.share)


def test_vector_sums_to_one_in_every_state(cfg):
    lb = replace(cfg.lane_b, share=0.5)
    lc = _ending()
    st = LaneCState(slots={FA: _win(FA, "a", 5 * ERA, 1), M: _win(M, "m", 5 * ERA, 2)})
    for block in (ERA + 1, 3 * ERA, 6 * ERA, 7 * ERA, 12 * ERA, 30 * ERA):
        rec = lane_c_pay_record(st, block=block, era_blocks=ERA, cfg=lc, n_uids=8, mc=MC)
        a, b = lane_parts([3], 8, king_decay=0.5, burn_uid=0, lane_b_cfg=lb,
                          lane_c_cfg=lc, c_wins=pay_wins(rec), c_slot_frac=rec["slot_frac"])
        v = emission_vector(a, b, EmissionPolicy(), burn_uid=0, lane_a_uids=[3])
        assert sum(v) == pytest.approx(1.0), block


def test_queued_entry_of_an_ended_competition_fails_uncharged(tmp_path):
    from cascade.funding.queue import COMPETITION_ENDED_CLASS
    from cascade.trainer.lane_c_driver import LaneCDriver

    from .test_lane_c_trainer_driver import FROM, KING, Clock, E, FakeOps, _cfg, _fund

    base = _cfg()
    until = FROM + 2 * ERA
    lc = replace(base.lane_c, competitions=(LaneCCompetition("mkt", until_block=until),))
    c = SimpleNamespace(lane_c=lc, marketcast=base.marketcast)
    ops = FakeOps(tmp_path, Clock())
    drv = LaneCDriver(c, ops, tmp_path / "d.json")
    _fund(ops, "ALFA", E)
    # a leg (leg_hours = 1h = 300 blocks) started now could not finish before the end
    assert drv.tick(until - 100, 10, KING, FROM) == []
    e = ops.q.get("ALFA")
    assert (e.status, e.last_error_class, e.attempts) == ("failed", COMPETITION_ENDED_CLASS, 0)
    assert "nothing charged" in e.last_error and ops.trained == []
    # before that it dispatches as usual
    ops2 = FakeOps(tmp_path / "b", Clock())
    drv2 = LaneCDriver(c, ops2, tmp_path / "d2.json")
    _fund(ops2, "BRAV", E)
    assert [x["hotkey"] for x in drv2.tick(until - 400, 10, KING, FROM)] == ["BRAV"]


def test_intake_refuses_a_fund_after_the_end(tmp_path):
    from .test_lane_c_trainer_intake import _h, _intake

    intake, _, lane_c, _ = _intake(tmp_path)
    lc = replace(parse_lane_c({}), competitions=(LaneCCompetition("mkt", until_block=6 * ERA),))
    intake.lane_c_closed = lambda t: competition_closed(lc, t, 6 * ERA)
    status, body = intake.lane_c_fund(_h())
    assert (status, body["code"]) == (409, "competition_ended") and lane_c.entries() == []
    intake.lane_c_closed = lambda t: competition_closed(lc, t, 6 * ERA - 1)
    assert intake.lane_c_fund(_h())[0] == 202


def test_validator_judges_nothing_after_the_end(cfg, tmp_path):
    """MarketCast past its until_block: a new entry and the backlog candidate
    registered before the end get ``ended`` rows — no verdict, no slot — and
    the audit accepts exactly that."""
    from cascade.audit.checks import check_lane_c

    from .test_marketcast_validator import (
        BLOCK,
        EB,
        _entry,
        _history,
        _rc,
        _rows,
        _runner,
        _settle,
        _snapshot,
        _Store,
    )

    s1 = _snapshot(tmp_path, 5000, "s1")
    store = _Store()
    es = [_entry("alice"), _entry("dave", train_block=BLOCK + 100)]
    r = _runner(cfg, tmp_path / "v", store, lambda b: s1, history=_history(es))
    lc = replace(r.cfg.lane_c, competitions=(LaneCCompetition("mkt", until_block=BLOCK),))
    r.cfg = replace(r.cfg, lane_c=lc)
    rows1 = _rows(r, _settle(store, "1001", es[:1]))
    assert len(r._lane_c().mkt_backlog) == 1                   # registered AT the end
    rows2 = _rows(r, _settle(store, "1002", es[1:], prev="1001", block=BLOCK + EB))
    got = {(x["hotkey"], x["stage"], x["outcome"]) for x in rows2 if "error" not in x}
    assert got == {("alice", "ended", ""), ("dave", "ended", "")}
    assert r._lane_c().mkt_backlog == [] and r._lane_c().slots == {}
    c = r.cfg
    assert check_lane_c(_rc("1001", BLOCK, rows1), c).status != "FAIL"
    ok = check_lane_c(_rc("1002", BLOCK + EB, rows2), c)
    assert ok.status != "FAIL", ok
    forged = [dict(x, stage="verdict") for x in rows2]
    assert check_lane_c(_rc("1002", BLOCK + EB, forged), c).status == "FAIL"


# ── the audit replay ─────────────────────────────────────────────────────────


def _audit_cfg(cfg, lc):
    return replace(cfg, lane_c=lc, marketcast=MC)


def test_audit_replays_the_split_and_rejects_a_forged_share(cfg):
    lc = _ending()
    c = _audit_cfg(cfg, lc)
    st = LaneCState(slots={FA: _win(FA, "a", 5 * ERA, 1), M: _win(M, "m", 5 * ERA, 2)})
    for blk in (3 * ERA, 7 * ERA, 12 * ERA):
        eb = era_length_blocks(c.round, blk)
        rec = lane_c_pay_record(st, block=blk, era_blocks=eb, cfg=lc, n_uids=8, mc=MC)
        rec = json.loads(json.dumps(rec))            # as a receipt carries it
        assert _lane_c_paid_wins(SimpleNamespace(epoch_start_block=blk, lane_c_pay=rec),
                                 c, 8)[1] is None, blk
    blk = 7 * ERA
    eb = era_length_blocks(c.round, blk)
    rec = json.loads(json.dumps(lane_c_pay_record(st, block=blk, era_blocks=eb, cfg=lc,
                                                  n_uids=8, mc=MC)))

    def why(r):
        return _lane_c_paid_wins(SimpleNamespace(epoch_start_block=blk, lane_c_pay=r), c, 8)[1]

    comps = [dict(x) for x in rec["competitions"]]
    comps[0]["frac"], comps[1]["frac"] = 0.75, 0.25
    assert "per-competition" in why({**rec, "competitions": comps})
    comps = [dict(x, state="active") for x in rec["competitions"]]
    assert "per-competition" in why({**rec, "competitions": comps})
    assert "slot fractions" in why({**rec, "slot_frac": {**rec["slot_frac"], FA: 0.4}})
    wins = [dict(w, wind_down=False) if w["target"] == FA else w for w in rec["wins"]]
    assert "wind_down" in why({**rec, "wins": wins})
    # a win paid in the wind-down past its own full-pay period
    late = json.loads(json.dumps(lane_c_pay_record(
        LaneCState(slots={FA: _win(FA, "a", 5 * ERA, 1, full_eras=10)}), block=blk,
        era_blocks=eb, cfg=lc, n_uids=8, mc=MC)))
    bad = [dict(w, full_eras=1, tier=0.01) for w in late["wins"]]
    assert why({**late, "wins": bad}) is not None


# ── config validation ────────────────────────────────────────────────────────


@pytest.mark.parametrize("comps, msg", [
    ({"mkt": {}, FAKE: {"from_block": 2 * ERA + 1800}}, "era boundary"),
    ({"mkt": {"until_block": 3 * ERA + 1800}}, "era boundary"),
    ({"mkt": {}, FAKE: {"from_block": 4 * ERA, "until_block": 4 * ERA}}, "must be after"),
    ({"mkt": {}, FAKE: {"from_block": 4 * ERA, "until_block": 3 * ERA}}, "must be after"),
    ({"mkt": {"until_block": ERA}}, "must be after"),            # before [marketcast] from
    ({"mkt": {}, FAKE: {"until_block": 4 * ERA}}, "must be after"),   # no start
    ({"mkt": {"weight": -1.0}, FAKE: {}}, "weight must be"),
    ({"mkt": {}, FAKE: {}, "nosuchcomp": {}}, "unknown competition"),
    ({"mkt": {}, FAKE: {}, "domain": {"weight": 1.0}}, "forward-only eval"),
    ({"mkt": {"from_block": 2 * ERA}, FAKE: {}}, "from_block is \\[marketcast\\]"),
    ({"mkt": {}}, "has no \\[lane_c.competitions.fakecomp\\]"),
    ({"mkt": {"weight": 0.0}, FAKE: {"weight": 0.0}}, "sum to > 0"),
    ({"mkt": {"wind_down": "decay"}, FAKE: {}}, "wind_down must be"),
    ({"mkt": {"budget": 1}, FAKE: {}}, "unknown keys"),
    ({"mkt": {}, FAKE: {"from_block": ERA}}, ">= \\[lane_c\\] from_block"),
])
def test_competition_config_validation(comps, msg):
    with pytest.raises(ValueError, match=msg):
        _lc(comps)


def test_loader_refuses_a_non_boundary_until_block(tmp_path):
    text = (ROOT / "chain.toml").read_text(encoding="utf-8")
    p = tmp_path / "chain.toml"
    p.write_text(re.sub(r"(?m)^until_block\s*=\s*0", "until_block        = 7201", text, count=1),
                 encoding="utf-8")
    with pytest.raises(ValueError, match="until_block"):
        load_chain_config(p)


# ── the shipped tomls: MarketCast only, weight 1, exactly as before ──────────


def test_shipped_tomls_are_marketcast_only_weight_one_and_unchanged():
    for f in ("chain.toml", "chain.testnet.toml"):
        c = load_chain_config(ROOT / f)
        lc = c.lane_c
        assert lc.competitions == (LaneCCompetition("mkt", weight=1.0),)
        assert lc.competition_list == replace(lc, competitions=()).competition_list
        # armed, the single competition is exactly the old equal split
        mc = replace(c.marketcast, from_block=4 * ERA)
        armed = replace(lc, from_block=2 * ERA)
        st = LaneCState(slots={M: _win(M, "a", 4 * ERA, 1)})
        for block in (0, 3 * ERA, 4 * ERA - 1, 4 * ERA, 5 * ERA, 40 * ERA, 10**9):
            rec = lane_c_pay_record(st, block=block, era_blocks=ERA, cfg=armed, n_uids=8,
                                    mc=mc)
            live = mc.active(block)
            assert rec["slots"] == (2 if live else 0)
            assert rec["slot_frac"] == ({M: 0.5, V: 0.5} if live else {})
            assert [x["state"] for x in rec["competitions"]] == [
                "active" if live else "pending"]
            v = _vec(rec, armed, 8)
            want = armed.share / 2 * (rec["wins"][0]["multiplier"] if rec["wins"] else 0.0)
            assert bool(rec["wins"]) == (live and block < 4 * ERA + 28 * ERA)   # TTL
            assert v[1] == pytest.approx(want) and sum(v) == pytest.approx(armed.share)
        # MarketCast off (the shipped state): nothing active at any block
        for block in (0, 7201, 10**9):
            assert lane_c_split(lc, block, ERA, c.marketcast)[1] == {}


# ── overlay (DEC-CA-0057) ────────────────────────────────────────────────────


def test_overlay_may_change_weight_and_until_only(cfg):
    from cascade.shared.netconfig import lane_c_competition_changes, retroactive_changes

    base = replace(cfg, lane_c=_lc({"mkt": {}, FAKE: {"from_block": 2 * ERA}}))

    def with_(**kw):
        comps = tuple(replace(x, **kw) if x.kind == FAKE else x
                      for x in base.lane_c.competition_list)
        return replace(base, lane_c=replace(base.lane_c, competitions=comps))

    assert lane_c_competition_changes(base, with_(weight=3.0)) == []
    assert lane_c_competition_changes(base, with_(until_block=10 * ERA)) == []
    assert "release-only" in lane_c_competition_changes(base, with_(from_block=3 * ERA))[0]
    assert "release-only" in lane_c_competition_changes(base, with_(wind_down="stop"))[0]
    more = replace(base, lane_c=replace(base.lane_c, competitions=base.lane_c.competitions
                                        + (LaneCCompetition(FAKE2, from_block=2 * ERA),)))
    assert "adding or removing" in lane_c_competition_changes(base, more)[0]
    # an end is never moved to (or from) a block already passed
    assert any("until_block" in x for x in retroactive_changes(base, with_(until_block=4 * ERA),
                                                                 8 * ERA))
    assert not any("until_block" in x for x in retroactive_changes(
        base, with_(until_block=10 * ERA), 8 * ERA))


# ── display ──────────────────────────────────────────────────────────────────


def test_status_and_dashboard_show_competitions():
    from cascade.miner.dashboard import lane_lines
    from cascade.shared.v2_status import emission_split, lane_c_view

    lc = _ending()
    view = lane_c_view(lc, None, block=7 * ERA, mc=MC, era_blocks=ERA)
    by = {c["kind"]: c for c in view["competitions"]}
    assert by[FAKE]["state"] == "winding_down" and by[FAKE]["until_block"] == 6 * ERA
    assert by["mkt"]["frac"] == 0.5 and by["mkt"]["from_block"] == 2 * ERA
    assert view["per_target"][FA]["competition"] == FAKE
    assert view["per_target"][M]["slot_frac"] == pytest.approx(0.25)
    st = LaneCState(slots={FA: _win(FA, "a", 5 * ERA, 1)})
    rec = lane_c_pay_record(st, block=7 * ERA, era_blocks=ERA, cfg=lc, n_uids=4, mc=MC)
    lines = lane_lines({"lane_c_verdicts": [], "lane_c_pay": rec})
    assert any("fakecomp: 50% winding_down" in x and "until block 43,200" in x for x in lines)
    v = _vec(rec, lc, 4)
    split = emission_split(v, [], {"wins": []}, lane_c_pay=rec)
    assert split["lane_c"] == pytest.approx(lc.share / 2 / 3, abs=1e-6)
