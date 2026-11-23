"""WB24: ``cascade lane-b status`` shows the lane B deferral and refusal
reasons from ``status/lane_b.json`` (WB15 ``rows_held``, WB18
``sentinel_target``, the WB17 ``gate_closed_reason`` values) beside the
intake queue it shows today.

The public document is served by the loopback recorder at
``<[storage] s3_endpoint>/<manifest_bucket>/status/lane_b.json`` of the
default chain.toml (a fixture copy whose ``[storage]`` points at the
recorder); the intake queue is the recorder's ``GET /v1/lane-b/queue``. Its
rows carry none of the reason codes, so a code in the output came from the
status document. Without a status document the command is today's."""

from __future__ import annotations

import pytest

from . import helpers as h
from .helpers import cli_env, intake, net  # noqa: F401 — pytest fixtures

OTHER = "5OtherHotkeyWB24BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
ENTRY_REF = "energy|" + h.REF
ALL_REASONS = (h.ROWS_HELD, h.SENTINEL_TARGET, *h.GATE_REASONS)


def _queue_row(hotkey=h.HK, targets=("energy",)):
    return {"hotkey": hotkey, "targets": list(targets), "ref": ",".join(targets) + "|" + h.REF,
            "status": "queued", "phase": "queued", "reveal_block": 700, "attempts": 0,
            "requeues": 0, "gpu": ""}


def _rows_held_row():
    from cascade.trainer import row_claims as rc

    holder = rc.leg_holder(OTHER, "energy|other@sha256:" + "e" * 64, {0: [3, 4, 5, 6]},
                           base_era=40, era_index=41, max_base_age_eras=2)
    return rc.deferred_row(h.HK, ENTRY_REF, holder, {0: [4, 5, 6, 7]})


def _sentinel_row():
    from cascade.trainer import row_claims as rc

    return rc.sentinel_row(h.HK, "suite:gift|" + h.REF, "suite:gift", [40], 2, 41)


def _status_doc(*, gate_closed_reason="", deferred=()):
    return {"lane": "b", "block": 123456, "as_of": h.now_iso(), "leg_funding": "miner",
            "gate_closed_reason": gate_closed_reason, "gate_rule": "",
            "gate_opened_block": 0, "open_rule": "tokens_king_check",
            "queue": {"entries": [_queue_row()], "queued_depth": 1},
            "pending": 0, "open": [], "finished": 0, "pending_merges": 0, "outcomes": [],
            "deferred": [dict(r) for r in deferred]}


@pytest.fixture
def served(tmp_path, cli_env, intake):  # noqa: F811
    toml = h.write_toml(tmp_path / "status-chain.toml", s3_endpoint=intake.base)
    cli_env.monkeypatch.setattr("cascade.shared.config.loader.DEFAULT_CHAIN_TOML", toml)
    intake.set_queue([_queue_row()])
    return intake


def _status(capsys, base, *, json_mode=True, hotkey=h.HK):
    argv = ["lane-b", "status", base]
    if hotkey:
        argv += ["--hotkey", hotkey]
    if json_mode:
        argv.append("--json")
    return h.run_cli(capsys, argv)


def _no_key(run):
    assert h.FAKE_KEY not in run.text(), "lane-b status printed the provider key"


@pytest.mark.parametrize("reason", [h.ROWS_HELD, h.SENTINEL_TARGET])
def test_status_shows_deferral_reason(reason, served, capsys):
    row = _rows_held_row() if reason == h.ROWS_HELD else _sentinel_row()
    assert row["deferred"] == reason, "fixture error: row_claims changed its reason code"
    served.set_status_doc(_status_doc(deferred=[row]))
    run = _status(capsys, served.base)
    assert run.rc == h.OK, f"exit {run.rc}: {run.err[-1500:]}"
    shown = [d for d in h.dicts_in(run.doc().get("data")) if d.get("deferred") == reason
             and d.get("hotkey") == h.HK]
    assert shown, (f"the JSON data must carry the status/lane_b.json deferral row "
                   f"({reason}) of {h.HK}; data: {run.doc().get('data')}")
    human = _status(capsys, served.base, json_mode=False)
    assert human.rc == h.OK
    assert reason in human.out, f"human output must name the deferral reason {reason!r}"
    _no_key(run)
    _no_key(human)
    assert any(r.path == f"/{h.BUCKET}/{h.STATUS_KEY}" for r in served.requests), (
        "status/lane_b.json was not read from the configured public bucket")


@pytest.mark.parametrize("reason", h.GATE_REASONS)
def test_status_shows_gate_closed_reason(reason, served, capsys):
    served.set_status_doc(_status_doc(gate_closed_reason=reason))
    run = _status(capsys, served.base)
    assert run.rc == h.OK, f"exit {run.rc}: {run.err[-1500:]}"
    data = run.doc().get("data")
    assert any(d.get("gate_closed_reason") == reason for d in h.dicts_in(data)), (
        f"the JSON data must carry gate_closed_reason = {reason!r}; data: {data}")
    human = _status(capsys, served.base, json_mode=False)
    assert reason in human.out, f"human output must name gate_closed_reason {reason!r}"
    _no_key(human)


def test_status_shows_reasons_without_hotkey_filter(served, capsys):
    served.set_status_doc(_status_doc(gate_closed_reason="tokens_short",
                                      deferred=[_rows_held_row()]))
    human = _status(capsys, served.base, json_mode=False, hotkey=None)
    assert human.rc == h.OK
    for reason in ("tokens_short", h.ROWS_HELD):
        assert reason in human.out, f"`lane-b status` (all entries) must name {reason!r}"


# ── unarmed: no status document -> today's output ───────────────────────────

TODAY_KEYS = {"queued_depth", "entries", "as_of", "submissions", "freshness"}


def test_status_output_unchanged_when_unarmed(served, capsys):
    """No ``status/lane_b.json`` (404): the command is today's: exit 0, the
    same data keys, no reason code invented."""
    run = _status(capsys, served.base)
    assert run.rc == h.OK, f"exit {run.rc}: {run.err[-1500:]}"
    data = run.doc().get("data") or {}
    assert set(data) >= TODAY_KEYS, f"missing today's keys {TODAY_KEYS - set(data)}"
    human = _status(capsys, served.base, json_mode=False)
    assert human.rc == h.OK
    for reason in ALL_REASONS:
        assert reason not in human.text(), f"{reason!r} shown without a status document"
    _no_key(run)
    _no_key(human)


def test_status_output_armed_nonempty_lane_d_effect(served, capsys):
    served.set_status_doc(_status_doc(gate_closed_reason="king_check_not_passed",
                                      deferred=[_rows_held_row()]))
    run = _status(capsys, served.base)
    assert run.rc == h.OK
    data = run.doc().get("data") or {}
    assert set(data) >= TODAY_KEYS, "today's keys must stay with a status document"
    human = _status(capsys, served.base, json_mode=False)
    assert "king_check_not_passed" in human.out and h.ROWS_HELD in human.out


def test_status_open_gate_shows_no_gate_reason(served, capsys):
    served.set_status_doc(_status_doc(gate_closed_reason=""))
    human = _status(capsys, served.base, json_mode=False)
    assert human.rc == h.OK
    for reason in h.GATE_REASONS:
        assert reason not in human.text()


# ── sanity (pass on main): the codes are the producers' ─────────────────────


def test_reason_codes_match_producers():
    from cascade.trainer import row_claims as rc

    assert (rc.ROWS_HELD, rc.SENTINEL_TARGET) == (h.ROWS_HELD, h.SENTINEL_TARGET)
    consts = h.string_constants(h.EXPERT_LANE)
    for reason in h.GATE_REASONS:
        assert reason in consts, f"{reason} is not a gate_closed_reason in expert_lane.py"
    assert "gate_closed_reason" in h.string_constants(h.DRIVER)
    assert h.STATUS_KEY in h.string_constants(h.DRIVER)
