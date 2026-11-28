"""W-OFFBOX: the offloaded settlement (pool phase on the lane D eval host)
publishes the same report bytes, state bytes and receipts as the in-process
path for the same inputs, so validators following our signed report see no
change. Contract: README.md "Same report bytes"."""

from __future__ import annotations

import json

import pytest

from . import helpers as h


def _pair(tmp_path):
    fx = h.w8b.fixture(tmp_path / "fx")
    ref = h.w8c.Rig(tmp_path / "ref", fx=fx)
    off = h.w8c.Rig(tmp_path / "off", fx=fx)
    host = h.offloaded(off, h.copy_pins(fx, tmp_path / "host_pool"))
    return ref, off, host


def _run_both(ref, off, boundaries):
    s_ref, s_off = h.w8b.Store(), h.w8b.Store()
    step_ref, step_off = h.step(ref, s_ref), h.step(off, s_off)
    for b in boundaries:
        step_ref(b)
        step_off(b)
        assert h.report_bytes(off, b) == h.report_bytes(ref, b), \
            f"boundary {b}: the offloaded report is not byte-identical"
        assert off.state_path.read_bytes() == ref.state_path.read_bytes(), \
            f"boundary {b}: the offloaded posterior state differs"
    assert s_off.objects == s_ref.objects, "the published objects differ"
    return s_ref, s_off


def test_offloaded_reports_are_byte_identical_across_a_crowning(tmp_path):
    """B1 crowns X through the bench gate; B2 and B3 crown nothing."""
    ref, off, host = _pair(tmp_path)
    s_ref, _ = _run_both(ref, off, (h.B1, h.B2, h.B3))
    from cascade.validator.lane_d_wiring import report_key

    assert json.loads(s_ref.objects[report_key(h.B1)])["crowned"] == ref.fx["x"]
    assert [r["judge"]["boundary"] for r in host.requests] == [h.B1, h.B2, h.B3], \
        "one settle request per settlement"
    assert off.eval_host.calls, "the eval host (not the box) forecast every digest"


def test_offloaded_is_byte_identical_with_prior_pins(tmp_path):
    """Two pins (the earlier one a prior snapshot): the judge order and the
    window dedup match the in-process path."""
    ref, off, _ = _pair(tmp_path)
    for rig in (ref, off):
        rig.pool.override = lambda b, rig=rig: ([rig.pool.pins(1)] if b <= h.B1
                                                 else [rig.pool.pins(1), rig.pool.pins(2)])
    _run_both(ref, off, (h.B1, h.B2))


def test_offloaded_is_byte_identical_when_an_entry_faults(tmp_path):
    """A miner's own checkpoint fault on the eval host closes the entry
    (``checkpoint_fault``) in both paths, byte for byte."""
    from cascade.eval.lane_d_forecasts import LaneDForecastError

    ref, off, _ = _pair(tmp_path)
    for rig in (ref, off):
        rig.eval_host.faults[rig.fx["x"]] = LaneDForecastError("forecast_malformed", "w-offbox")
    s_ref, _ = _run_both(ref, off, (h.B1,))
    from cascade.validator.lane_d_wiring import report_key

    rep = json.loads(s_ref.objects[report_key(h.B1)])
    assert {"digest": ref.fx["x"], "reason": "checkpoint_fault"} in rep["skipped"]


def test_judge_pools_record_survives_the_json_transport(tmp_path):
    """The judge request and the judged record are plain JSON: the round trip
    changes nothing (no tuple, non-str key, set or array survives silently)."""
    h.require_offbox()
    from cascade.validator import lane_d

    cfg = h.w8b.chain_cfg(tmp_path / "v").lane_d
    rec = lane_d.lane_d_config_record(cfg)
    assert lane_d.lane_d_config_from_record(json.loads(json.dumps(rec))) == cfg
    with pytest.raises(TypeError):
        lane_d.plain_json({1: "x"})
    with pytest.raises(TypeError):
        lane_d.plain_json({"a": {1, 2}})
