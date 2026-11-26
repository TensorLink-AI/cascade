"""W8c fail closed: each missing or unverifiable input is its own typed
LaneDInputsError, nothing is written, and W8b's step turns it into
``inputs_unavailable`` (no settlement this tick, retried). A miner's own
checkpoint fault is not an infrastructure failure: it reaches settle_lane_d as
malformed forecasts and closes that entry (``checkpoint_fault``). Contract:
README.md."""

from __future__ import annotations

import json

import numpy as np
import pytest

from . import helpers as h


def _refused(rig, code: str, boundary: int = h.B1):
    before = rig.files()
    with pytest.raises(Exception) as ei:
        rig.source(boundary)
    h.assert_inputs_error(ei.value, code)
    assert rig.files() == before, f"{code}: a refusal writes nothing"
    return ei.value


def test_error_codes_are_exported():
    m = h.inputs_mod()
    assert set(h.INPUT_CODES) <= set(m.INPUTS_ERROR_CODES)
    with pytest.raises(ValueError):
        m.LaneDInputsError("not_a_code", "x")
    e = m.LaneDInputsError("pool_unavailable", "detail")
    assert isinstance(e, RuntimeError) and e.code == "pool_unavailable"


def test_chain_reader_unavailable(tmp_path):
    rig = h.Rig(tmp_path)
    rig.factory.fail = True
    _refused(rig, "chain_unavailable")


@pytest.mark.parametrize("text", ["{not json", json.dumps({"schema": "other/1", "entries": []}),
                                  json.dumps({"schema": "lane-d-intake-queue/1",
                                              "entries": [{"hotkey": 1}]})])
def test_intake_queue_unreadable(tmp_path, text):
    from cascade.lane_d.service import QUEUE_FILENAME

    rig = h.Rig(tmp_path)
    (rig.intake_dir / QUEUE_FILENAME).write_text(text, encoding="utf-8")
    _refused(rig, "intake_unreadable")


def test_reveal_timestamp_unreadable(tmp_path):
    rig = h.Rig(tmp_path)
    rig.reader.fail_timestamp = True
    _refused(rig, "reveal_unreadable")


def test_metagraph_unreadable(tmp_path):
    rig = h.Rig(tmp_path)
    rig.reader.fail_hotkeys = True
    _refused(rig, "metagraph_unreadable")


def test_lane_d_state_unreadable(tmp_path):
    rig = h.Rig(tmp_path)
    rig.state_path.write_text("{corrupt", encoding="utf-8")
    _refused(rig, "state_unreadable")


@pytest.mark.parametrize("damage", ["missing", "forged", "wrong_signer"])
def test_house_pointer_unavailable(tmp_path, damage):
    rig = h.Rig(tmp_path)
    if damage == "missing":
        rig.house_path.unlink()
    elif damage == "forged":
        doc = json.loads(rig.house_path.read_text())
        doc["reveal_ts"] = int(doc["reveal_ts"]) + 1
        rig.house_path.write_text(json.dumps(doc, sort_keys=True))
    else:
        h.write_house_pointer(rig.house_path, rig.fx["house"], key=h.w7a.fixture_key("not-me"))
    e = _refused(rig, "house_unavailable")
    from cascade.validator.lane_d_house import LaneDHouseError

    assert isinstance(e.__cause__, LaneDHouseError), "the W5f refusal is chained"
    assert rig.eval_host.calls == [], "no forecast without a verified house"


@pytest.mark.parametrize("how", ["raises", "empty", "missing_dir"])
def test_pool_unavailable(tmp_path, how):
    rig = h.Rig(tmp_path)
    if how == "raises":
        rig.pool.fail = True
    elif how == "empty":
        rig.pool.override = lambda b: []
    else:
        rig.pool.override = lambda b: [{"dir": tmp_path / "nope", "sha256": "0" * 64}]
    _refused(rig, "pool_unavailable")
    assert rig.eval_host.calls == []


def test_pool_digest_mismatch(tmp_path):
    """A snapshot whose bytes do not hash to its W6d pin is refused."""
    rig = h.Rig(tmp_path)
    pin = rig.pool.pins(1)
    rig.pool.override = lambda b: [{"dir": pin["dir"], "sha256": "f" * 64}]
    _refused(rig, "pool_digest_mismatch")
    assert rig.eval_host.calls == [], "no window of an unpinned snapshot leaves the box"


@pytest.mark.parametrize("field", ["git_head", "package_hash", "schema"])
def test_handshake_mismatch_scores_nothing(tmp_path, field):
    rig = h.Rig(tmp_path)
    rig.eval_host.hello_doc = h.hello(**{field: "stale"})
    _refused(rig, "handshake_mismatch")
    assert rig.eval_host.calls == [], "a mismatched eval host gets no forecast request (T19)"


@pytest.mark.parametrize("who", ["house", "entry"])
def test_eval_host_unreachable(tmp_path, who):
    rig = h.Rig(tmp_path)
    dg = rig.fx["house"]["digest"] if who == "house" else rig.fx["x"]
    rig.eval_host.faults[dg] = ConnectionError("w8c: eval host lost")
    _refused(rig, "forecasts_unavailable")


@pytest.mark.parametrize("code", ["checkpoint_unavailable", "forecast_failed"])
def test_eval_host_infrastructure_codes_fail_closed_even_for_an_entry(tmp_path, code):
    """Not the miner's fault (the vault or the host): no settlement this tick."""
    rig = h.Rig(tmp_path)
    rig.eval_host.faults[rig.fx["x"]] = h.forecasts_mod().LaneDForecastError(code, "w8c")
    _refused(rig, "forecasts_unavailable")


@pytest.mark.parametrize("code", h.MINER_FAULT_CODES)
def test_bars_fault_fails_closed(tmp_path, code):
    """KING_D or the house cannot be scored: never a guessed bar."""
    rig = h.Rig(tmp_path)
    rig.eval_host.faults[rig.fx["house"]["digest"]] = \
        h.forecasts_mod().LaneDForecastError(code, "w8c")
    _refused(rig, "forecasts_unavailable")


def test_bars_malformed_array_fails_closed(tmp_path):
    rig = h.Rig(tmp_path)
    rig.eval_host.shapes[rig.fx["house"]["digest"]] = np.zeros((1, 2, 3))
    _refused(rig, "forecast_malformed")


def _fault_inputs(rig):
    # rev6.18 A7 (F4): lane_d_lowfreq is deleted; forecast_ok lives in lane_d.
    from cascade.validator.lane_d import _forecast_ok as forecast_ok

    got = rig.source(h.B1)
    x = rig.fx["x"]
    wids = h.window_ids(rig.fx["chain"]["pools"][1])
    assert x in got.forecasts, "a faulted entry still has a forecasts record"
    assert all(forecast_ok(got.forecasts[x], wid, h.HORIZON) is False for wid in wids), \
        "every window of a faulted entry is malformed (settle_lane_d: checkpoint_fault)"
    house = rig.fx["house"]["digest"]
    assert all(forecast_ok(got.forecasts[house], wid, h.HORIZON) is True for wid in wids)
    return got


@pytest.mark.parametrize("code", h.MINER_FAULT_CODES)
def test_entry_fault_reaches_settlement_as_checkpoint_fault(tmp_path, code):
    rig = h.Rig(tmp_path)
    rig.eval_host.faults[rig.fx["x"]] = h.forecasts_mod().LaneDForecastError(code, "w8c")
    _fault_inputs(rig)
    store = h.w8b.Store()
    step = rig.step(store=store)
    step(h.B1)
    from cascade.validator.lane_d_wiring import report_key

    report = json.loads(store.objects[report_key(h.B1)])
    assert {"digest": rig.fx["x"], "reason": "checkpoint_fault"} in report["skipped"]


def test_entry_malformed_array_is_a_checkpoint_fault(tmp_path):
    rig = h.Rig(tmp_path)
    n = len(h.window_ids(rig.fx["chain"]["pools"][1]))
    rig.eval_host.shapes[rig.fx["x"]] = np.full((n, h.HORIZON, 9), np.nan)
    _fault_inputs(rig)


def test_step_reports_a_refusal_as_inputs_unavailable_with_the_cause(tmp_path):
    """Through W8b: the tick does not settle, the W8c code is the cause, and
    the next tick with the input back settles normally."""
    rig = h.Rig(tmp_path)
    store = h.w8b.Store()
    step = rig.step(store=store)
    rig.pool.fail = True
    with pytest.raises(Exception) as ei:
        step(h.B1)
    h.w8b.assert_wiring_error(ei.value, "inputs_unavailable")
    h.assert_inputs_error(ei.value.__cause__, "pool_unavailable")
    assert store.puts == [] and not rig.state_path.exists()
    rig.pool.fail = False
    assert step(h.B1) is not None
