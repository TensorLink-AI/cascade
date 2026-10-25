"""DEC-CA-0056 receipt stamps (drop-when-default, round trip) and the
published v2 status / receipt-index columns."""

from __future__ import annotations

import json
from dataclasses import replace

from cascade.shared.config.lane_c import parse_lane_c
from cascade.shared.lane_c import LaneCState, TargetWin
from cascade.shared.receipt import dump_receipt, load_receipt
from cascade.shared.v2_status import build_v2_status, emission_split, v2_index_fields

from .receipt_fixture import make_scored_receipt


def test_pre_lane_c_receipt_bytes_unchanged_and_stamps_round_trip():
    rc, _, _ = make_scored_receipt()
    assert b"lane_c" not in rc.canonical_body()
    rows = ({"hotkey": "alice", "target": "mkt:returns", "stage": "verdict", "passed": True,
             "reason": "win", "gain_lcb": 0.02, "gain": 0.05, "n": 300, "public_lcbs": {}},)
    pay = {"block": 7200, "era_blocks": 7200, "share": 0.25, "slots": 9,
           "wins": [{"target": "mkt:returns", "hotkey": "alice", "uid": 7,
                     "win_block": 7200, "age": 0, "multiplier": 1.0}]}
    stamped = replace(rc, lane_c=rows, lane_c_pay=pay)
    body = json.loads(stamped.canonical_body())
    assert body["lane_c"][0]["target"] == "mkt:returns" and body["lane_c_pay"] == pay
    back = load_receipt(dump_receipt(stamped))
    assert back.lane_c == rows and back.lane_c_pay == pay
    assert back.canonical_body() == stamped.canonical_body()


def test_status_document_gains_lane_c_only_once_armed(cfg):
    assert "lane_c" not in build_v2_status(cfg, block=100, as_of="t")
    lc = replace(parse_lane_c({}), from_block=7200)
    st = LaneCState()
    st.record(TargetWin(target="mkt:vol", hotkey="alice", uid=7, round_id="r",
                        win_block=14400, gain_lcb=0.03))
    trainer = {"queue": {"queued": 2}, "targets": {"mkt:returns": {
        "queued": 2, "open": 1, "finished": 0, "slots": 2}}, "open": [], "finished": 0,
        "legs_started": 3, "settlements": [], "outcomes": []}
    doc = build_v2_status(replace(cfg, lane_c=lc), block=20000, as_of="t", lane_c_state=st,
                          lane_c_trainer=trainer)
    v = doc["lane_c"]
    assert v["active"] and v["state"] == "live" and v["share"] == 0.25
    assert v["per_target"]["mkt:vol"]["holder"]["hotkey"] == "alice"
    assert v["per_target"]["mkt:returns"]["holder"] is None
    assert v["per_target"]["mkt:returns"]["queued"] == 2
    assert v["per_target"]["mkt:returns"]["open"] == 1 and v["payout"]["slots"] == 0   # MarketCast off: no active slot
    assert v["trainer"]["legs_started"] == 3


def test_trainer_hook_reads_the_driver_status_not_its_leg_state(cfg):
    from types import SimpleNamespace

    from cascade.shared import v2_status

    lc = replace(parse_lane_c({}), from_block=7200)
    published = []

    class _Drv:
        st = SimpleNamespace(open=[], finished=[], settled={})     # LaneCDriverState-shaped

        def status(self):
            return {"targets": {"mkt:vol": {"queued": 1, "open": 0, "finished": 1,
                                                    "slots": 2}}, "legs_started": 1}

    sched = SimpleNamespace(cfg=replace(cfg, lane_c=lc), lane_b=None, lane_c=_Drv(),
                            state=SimpleNamespace(current=None),
                            r=SimpleNamespace(manifest_store=lambda: "store"))
    orig = v2_status.publish_v2_status
    v2_status.publish_v2_status = lambda store, doc: published.append(doc)
    try:
        assert v2_status.publish_trainer_v2_status(sched, 20000, min_interval_s=0)
    finally:
        v2_status.publish_v2_status = orig
    per = published[0]["lane_c"]["per_target"]
    assert per["mkt:vol"]["finished"] == 1 and per["mkt:vol"]["holder"] is None


def test_index_columns_and_emission_split():
    w = [0.25, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.125, 0.0, 0.125]
    b_pay = {"block": 1, "era_blocks": 1, "share": 0.25, "slots": 2,
             "wins": [{"domain": "energy", "uid": 9, "multiplier": 1.0}]}
    c_pay = {"block": 1, "era_blocks": 1, "share": 0.25, "slots": 2,
             "wins": [{"target": "horizon:64", "uid": 7, "multiplier": 1.0}]}
    split = emission_split(w, [3], b_pay, lane_c_pay=c_pay)
    assert split == {"lane_a": 0.5, "lane_b": 0.125, "lane_c": 0.125, "burn": 0.25}
    assert "lane_c" not in emission_split(w, [3], b_pay)

    class _R:
        weights, reward_uids, lane_b, lane_b_pay = w, (3,), None, b_pay
        lane_c = ({"hotkey": "alice", "target": "horizon:64", "stage": "verdict",
                   "passed": True, "reason": "win", "gain_lcb": 0.02, "gain": 0.04},
                  {"error": "bad_signature"})
        lane_c_pay = c_pay

    cols = v2_index_fields(_R())
    assert cols["lane_c_verdicts"][0]["target"] == "horizon:64"
    assert len(cols["lane_c_verdicts"]) == 1 and cols["lane_c_pay"] == c_pay
    assert cols["emission_split"]["lane_c"] == 0.125
