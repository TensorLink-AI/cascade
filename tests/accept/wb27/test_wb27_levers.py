"""WB27 (plan rev6.21, owner 2026-11-24): the four miner levers — ranges,
defaults, typed refusals, canonical form, signature tag, curriculum budget
split and stage seeds. Contract: README.md "Levers"."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from . import helpers as h


def _armed_lb():
    cfg, _ = h.armed_cfg()
    return cfg.lane_b


def _code(raw, lb=None):
    lv = h.levers()
    with pytest.raises(lv.LeversError) as e:
        lv.parse_levers(raw, lb or _armed_lb())
    assert e.value.code in h.CODES, e.value.code
    return e.value.code


def test_refusal_codes_are_exported():
    lv = h.levers()
    assert set(lv.LEVER_CODES) == set(h.CODES)
    assert lv.LEVERS_HEADER == h.LEVERS_HEADER


def test_no_header_is_todays_leg():
    lv = h.levers()
    assert lv.parse_levers("", _armed_lb()) is None
    assert lv.parse_levers(None, h.chain_cfg().lane_b) is None
    d = lv.LegLevers()
    assert d.is_default and d.as_dict() == {} and d.lr_scale == 1.0
    assert d.curriculum == () and d.context_length == 4096 and d.ema == 0.10


def test_levers_are_the_default_lane_b_behaviour():
    """owner 2026-11-24: no activation block — the shipped config takes them."""
    lv, lb = h.levers(), h.chain_cfg().lane_b
    assert not hasattr(lb, "miner_levers_from_block")
    assert lv.parse_levers('{"lr_scale": 1.5}', lb).lr_scale == 1.5
    assert lv.parse_levers('{"curriculum": [["a", 0.5], ["b", 0.5]]}', lb).curriculum
    # every lane B leg is an EMA leg (#845): the EMA choice needs no key
    assert lv.parse_levers('{"ema": 0.05}', lb).ema == 0.05
    # every lever at its default stays accepted (a no-op header)
    assert lv.parse_levers('{"lr_scale": 1.0, "context_length": 4096}', lb).is_default


def test_lr_scale_range():
    lv, lb = h.levers(), _armed_lb()
    assert (lb.lr_scale_min, lb.lr_scale_max) == (0.5, 2.0)
    for ok, want in (("0.5", 0.5), (2, 2.0), ("1.25", 1.25), (0.125 * 5, 0.625)):
        assert lv.parse_levers(json.dumps({"lr_scale": ok}), lb).lr_scale == want
    for bad in (0.49, 2.01, 0, -1, "x", True, None, 1.2345, "nan"):
        assert _code(json.dumps({"lr_scale": bad})) == "lr_scale_out_of_range", bad


def test_curriculum_rules():
    lv, lb = h.levers(), _armed_lb()
    got = lv.parse_levers('{"curriculum": [["easy", 0.3], ["mid", "0.3"], ["hard", 0.4]]}', lb)
    assert got.curriculum == (("easy", 30), ("mid", 30), ("hard", 40))
    assert got.as_dict()["curriculum"] == [["easy", "0.3"], ["mid", "0.3"], ["hard", "0.4"]]
    # one stage is the default (today)
    assert lv.parse_levers('{"curriculum": [["all", 1]]}', lb).is_default
    for bad in ([["a", 0.5], ["b", 0.4]],                  # sums to 0.9
                [["a", 0.25], ["b", 0.25], ["c", 0.25], ["d", 0.25]],   # 4 stages
                [["a", 0.5], ["a", 0.5]],                  # duplicate
                [["A", 0.5], ["b", 0.5]],                  # bad name
                [["a", 0.555], ["b", 0.445]],              # 3 decimals
                [["a", 0], ["b", 1]], [], "easy", [["a"]]):
        assert _code(json.dumps({"curriculum": bad})) == "bad_curriculum", bad


def test_context_length_wire_format_and_refusals():
    lv, lb = h.levers(), _armed_lb()
    assert lv.parse_levers('{"context_length": 4096}', lb).context_length == 4096
    for c in (8192, 16384, 32768):
        assert lv.parse_levers(json.dumps({"context_length": c}), lb).context_length == c
    capped = replace(lb, max_context_length=8192)       # the [lane_b] ceiling
    assert _code('{"context_length": 16384}', capped) == "context_length_out_of_range"
    for c in (2048, 5000, 65536, "8192", True):
        assert _code(json.dumps({"context_length": c})) == "context_length_out_of_range", c


def test_context_memory_fit_rule():
    lv = h.levers()
    from cascade.funding.step_rates import StepRate

    r = StepRate(sku=h.SKU, gpu_name="x", gpu_count=8, steps_per_s=1.0, tokens_per_s=1.0,
                 overhead_seconds=0.0, peak_memory_bytes=10, memory_total_bytes=100)
    assert lv.context_fits(r, 4096)
    assert lv.context_fits(r, 32768)                       # 80 <= 90
    assert not lv.context_fits(replace(r, peak_memory_bytes=12), 32768)   # 96 > 90
    assert not lv.context_fits(replace(r, peak_memory_bytes=0), 8192)


def test_ema_choice():
    lv = h.levers()
    lb = _armed_lb()
    assert lb.sigma_rel_choices == (0.03, 0.05, 0.10, 0.15, 0.20)
    for w in lb.sigma_rel_choices:
        assert lv.parse_levers(json.dumps({"ema": w}), lb).ema == w
    assert lv.parse_levers('{"ema": 0.1}', lb).is_default
    narrow = replace(lb, sigma_rel_choices=(0.05, 0.10))
    assert _code('{"ema": 0.2}', narrow) == "ema_out_of_range"   # overlay-settable set
    for bad in (0.25, 0.5, 0, "0.05", None):
        assert _code(json.dumps({"ema": bad})) == "ema_out_of_range", bad


def test_malformed_headers():
    for raw in ("nope", "[1]", '{"x": 1}', '"lr_scale"', "{"):
        assert _code(raw) == "bad_levers", raw


def test_signed_tag_binds_the_header_as_sent():
    lv = h.levers()
    raw = ' {"lr_scale":1.5} '
    assert lv.levers_tag(raw) == "levers=" + hashlib.sha256(raw.strip().encode()).hexdigest()
    assert lv.levers_tag("") == "" and lv.levers_tag(None) == ""
    from cascade.funding.intake import lane_b_signed_ref

    base = lane_b_signed_ref(("energy",), h.REF)
    assert base == f"energy|{h.REF}"                       # today's bytes
    signed = lane_b_signed_ref(("energy",), h.REF, levers=raw)
    assert signed == f"{base}|{lv.levers_tag(raw)}"
    assert lane_b_signed_ref(("energy",), h.REF, train_hours="2.0", train_tokens="20e9",
                             levers=raw) == f"{base}|hours=2.0|tokens=20e9|{lv.levers_tag(raw)}"


def test_curriculum_points_and_stage_seeds():
    lv = h.levers()
    assert lv.curriculum_points(1001, (("a", 30), ("b", 70))) == [300, 701]
    assert lv.curriculum_points(1000, ()) == [1000]
    pts = lv.curriculum_points(10**9 + 7, (("a", 33), ("b", 33), ("c", 34)))
    assert sum(pts) == 10**9 + 7 and pts[0] == pts[1]
    seeds = [lv.curriculum_stage_seed(123, i) for i in range(3)]
    assert len(set(seeds)) == 3 and 123 not in seeds
    assert seeds == [lv.curriculum_stage_seed(123, i) for i in range(3)]   # deterministic
    want = int.from_bytes(hashlib.sha256(b"cascade-curriculum/1:123:0").digest()[:8], "big")
    assert seeds[0] == want & ((1 << 63) - 1)


def test_entry_levers_check():
    lv = h.levers()
    cfg, b = h.armed_cfg()
    good = {"lr_scale": 1.5, "ema": 0.05}
    assert lv.check_entry_levers({}, lb_cfg=cfg.lane_b, train_block=0) is None
    assert lv.check_entry_levers(good, lb_cfg=cfg.lane_b, train_block=b) is None
    assert lv.check_entry_levers({"lr_scale": 3.0}, lb_cfg=cfg.lane_b, train_block=b) == \
        "lr_scale_out_of_range"
    assert lv.check_entry_levers({"lr_scale": 1.0}, lb_cfg=cfg.lane_b, train_block=b) == \
        "bad_levers"                                       # not the canonical form
    assert lv.lane_b_scored_weights_for(cfg.lane_b, b, lv.LegLevers(ema=0.05)) == "ema005"
    assert lv.lane_b_scored_weights_for(cfg.lane_b, b, None) == "ema010"
    assert lv.lane_b_scored_weights_for(cfg.lane_b, 0, lv.LegLevers(ema=0.05)) == "ema005"
