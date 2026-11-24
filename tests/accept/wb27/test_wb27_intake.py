"""WB27 intake (plan rev6.21, owner 2026-11-24): ``X-Lane-B-Levers``, 12 h
legs, ``X-Lane-B-Train-Tokens`` (stop at the token OR hour limit) and
own-risk GPU types on ``POST /v1/lane-b/fund``. Every refusal queues and
vaults nothing (nothing charged)."""

from __future__ import annotations

import json

import pytest

from . import helpers as h


def test_fund_without_any_new_header_is_todays_fund(tmp_path):
    h.require_built()
    it, q = h.intake(tmp_path, h.chain_cfg())
    status, body = it.lane_b_fund(h.fund_headers())
    assert status == 202, body
    assert not {"levers", "train_tokens", "gpu_measured", "train_hours"} & set(body)
    e = q.get(h.HK)
    assert e.levers is None and e.train_tokens == 0 and e.gpu_measured is True


def test_levers_admitted_and_recorded(tmp_path):
    h.require_built()
    cfg, _ = h.armed_cfg()
    it, q = h.intake(tmp_path, cfg)
    lv = {"lr_scale": 1.5, "curriculum": [["easy", 0.4], ["hard", 0.6]], "ema": 0.05}
    status, body = it.lane_b_fund(h.fund_headers(**{h.LEVERS_HEADER: json.dumps(lv)}))
    assert status == 202, body
    want = {"lr_scale": 1.5, "curriculum": [["easy", "0.4"], ["hard", "0.6"]], "ema": 0.05}
    assert body["levers"] == want and q.get(h.HK).levers == want


@pytest.mark.parametrize("lv,code", [
    ({"lr_scale": 2.5}, "lr_scale_out_of_range"),
    ({"curriculum": [["a", 0.5]]}, "bad_curriculum"),
    ({"context_length": 16384}, "context_length_unfit"),      # 40 GB x 4 > 0.9 x 80 GB
    ({"context_length": 3000}, "context_length_out_of_range"),
    ({"ema": 0.25}, "ema_out_of_range"),
    ({"bogus": 1}, "bad_levers"),
])
def test_out_of_range_levers_refused_nothing_charged(tmp_path, lv, code):
    h.require_built()
    cfg, _ = h.armed_cfg()
    it, q = h.intake(tmp_path, cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{h.LEVERS_HEADER: json.dumps(lv)}))
    assert status == 400 and body["code"] == code, body
    h.nothing_charged(it, q)


def test_longer_context_admitted_and_its_steps_scale(tmp_path):
    h.require_built()
    cfg = h.chain_cfg()
    c = h.base_contract(cfg)
    it, q = h.intake(tmp_path, cfg)
    # 8192: 40 GB x 2 = 80 GB > 0.9 x 80 GB -> unfit on the fake row
    status, body = it.lane_b_fund(h.fund_headers(**{h.LEVERS_HEADER: '{"context_length":8192}'}))
    assert status == 400 and body["code"] == "context_length_unfit", body
    h.nothing_charged(it, q)
    # an own-risk GPU runs it at the miner's risk; the token limit's steps use
    # tokens per step at that context
    status, body = it.lane_b_fund(h.fund_headers(**{
        h.LEVERS_HEADER: '{"context_length":8192}', h.TOKENS_HEADER: "20e9",
        "X-Gpu-Type": h.UNMEASURED_SKU}))
    assert status == 202, body
    tps = c.batch_size * 8192
    assert body["steps"] == min(20 * 10**9 // tps, h.want_reference_steps(c, 12.0))
    assert q.get(h.HK).levers == {"context_length": 8192}


def test_twelve_hours_admitted(tmp_path):
    h.require_built()
    cfg = h.chain_cfg()
    it, q = h.intake(tmp_path, cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{h.HOURS_HEADER: "12"}))
    assert status == 202, body
    c = h.base_contract(cfg)
    assert body["train_hours"] == 12.0
    assert body["steps"] == min(int(12 * 3600 * h.RATE), h.want_reference_steps(c, 12.0))
    assert body["guard_seconds"] <= 54_000
    status, body = it.lane_b_fund(h.fund_headers(**{h.HOURS_HEADER: "12.5"}))
    assert status == 400 and body["code"] == "bad_train_hours", body


def test_token_limit_admission(tmp_path):
    h.require_built()
    cfg, _ = h.armed_cfg()
    c = h.base_contract(cfg)
    tps = h.want_tokens_per_step(c)
    assert tps == c.batch_size * c.context_length              # never a hard-coded 2.1M
    it, q = h.intake(tmp_path, cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{h.TOKENS_HEADER: "20e9"}))
    assert status == 202, body
    # tokens only: the hour limit is max_leg_hours
    assert body["train_tokens"] == 20 * 10**9 and body["train_hours"] == 12.0
    assert body["steps"] == min(20 * 10**9 // tps, h.want_reference_steps(c, 12.0))
    assert body["guard_seconds"] == 12 * 3600
    e = q.get(h.HK)
    assert (e.train_tokens, e.leg_steps, e.train_hours) == (20 * 10**9, body["steps"], 12.0)


def test_token_and_hour_limits_together(tmp_path):
    h.require_built()
    cfg, _ = h.armed_cfg()
    c = h.base_contract(cfg)
    it, _q = h.intake(tmp_path, cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{h.TOKENS_HEADER: "20000000000",
                                                    h.HOURS_HEADER: "1"}))
    assert status == 202, body
    # stops at whichever limit comes first: the steps cap of 1 h binds here
    assert body["train_hours"] == 1.0 and body["guard_seconds"] == 3600
    assert body["steps"] == min(20 * 10**9 // h.want_tokens_per_step(c),
                                h.want_reference_steps(c, 1.0))


@pytest.mark.parametrize("raw,code", [("1.5", "bad_train_tokens"), ("-5", "bad_train_tokens"),
                                      ("abc", "bad_train_tokens"),
                                      ("1000", "train_tokens_out_of_range")])
def test_bad_token_limits_refused(tmp_path, raw, code):
    h.require_built()
    cfg, _ = h.armed_cfg()
    it, q = h.intake(tmp_path, cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{h.TOKENS_HEADER: raw}))
    assert status == 400 and body["code"] == code, body
    h.nothing_charged(it, q)


def test_own_risk_gpu(tmp_path):
    h.require_built()
    cfg, _ = h.armed_cfg()
    it, q = h.intake(tmp_path, cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{h.TOKENS_HEADER: "20e9",
                                                    "X-Gpu-Type": h.UNMEASURED_SKU}))
    assert status == 202, body
    assert body["gpu_measured"] is False and q.get(h.HK).gpu_measured is False
    # hours only on an unmeasured type: the hour limit stops it (owner correction)
    it2, q2 = h.intake(tmp_path / "b", cfg)
    status, body = it2.lane_b_fund(h.fund_headers(**{h.HOURS_HEADER: "4",
                                                     "X-Gpu-Type": h.UNMEASURED_SKU}))
    assert status == 202 and body["gpu_measured"] is False, body
    assert body["steps"] == h.want_reference_steps(h.base_contract(cfg), 4.0)


def test_own_risk_gpu_refused_without_a_limit_or_off_lium(tmp_path):
    h.require_built()
    cfg = h.chain_cfg()
    it, q = h.intake(tmp_path / "f", cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{h.TOKENS_HEADER: "20e9",
                                                    "X-Gpu-Type": h.UNMEASURED_SKU,
                                                    "X-Gpu-Allow-Fallback": "true"}))
    assert status == 400, body
    h.nothing_charged(it, q)
    it, q = h.intake(tmp_path / "n", cfg)
    status, body = it.lane_b_fund(h.fund_headers(**{"X-Gpu-Type": h.UNMEASURED_SKU}))
    assert status == 400, body
    h.nothing_charged(it, q)
