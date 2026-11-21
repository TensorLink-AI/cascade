"""R5 (§3.0 R5 Accept: "refuses during a scoring or settlement window fixture";
§10.5 S5; owner rules): every restart script refuses, `--dry-run` included,
in a settlement, scoring, harvest or freeze window, with legs in flight, or
when the window can't be placed. State and time come only from the fixture
root and `LANED_NOW`."""

from __future__ import annotations

import pytest

from .helpers import BOUNDARY, QUIET_NOW, SERVICES, build_box, log_line, output, plan_lines

ALL = tuple(SERVICES)


@pytest.fixture
def box(tmp_path):
    return build_box(tmp_path)


def assert_refused(box, proc, rule):
    text = output(proc)
    assert proc.returncode == 1, f"expected exit 1 (refused), got {proc.returncode}:\n{text}"
    assert f"REFUSE [{rule}]" in text, f"no `REFUSE [{rule}]`:\n{text}"
    assert box.shim_calls() == [], box.shim_calls()
    assert not (box.root / "_state_backups").exists()
    assert not box.release_dir.exists()


@pytest.mark.parametrize("svc", ALL)
def test_refuses_just_before_a_settlement(box, svc):
    box.write_logs(block=BOUNDARY - 50)  # 10 min before the boundary
    assert_refused(box, box.run(svc), "settlement")


@pytest.mark.parametrize("svc", ALL)
def test_refuses_just_after_a_settlement(box, svc):
    box.write_logs(block=BOUNDARY + 25)  # 5 min after the boundary
    assert_refused(box, box.run(svc), "settlement")


@pytest.mark.parametrize("svc", ALL)
def test_refuses_when_window_cannot_be_placed(box, svc):
    box.write_logs(block=None)
    assert_refused(box, box.run(svc), "clock")


@pytest.mark.parametrize("svc", ALL)
def test_refuses_while_validator_is_scoring(box, svc):
    box.write_logs(validator_extra=log_line(QUIET_NOW, "new manifest round=777"))
    assert_refused(box, box.run(svc), "scoring")


@pytest.mark.parametrize("svc", ALL)
def test_refuses_with_legs_in_flight(box, svc):
    box.write_queue({"x": {"status": "in_flight", "phase": "training",
                           "hotkey": "5InFlightHotkey"}})
    assert_refused(box, box.run(svc), "in_flight")


@pytest.mark.parametrize("svc", ALL)
def test_refuses_during_harvest(box, svc):
    box.write_era_state({"verify_queue": ["harvest-1"]})
    assert_refused(box, box.run(svc), "harvest")


@pytest.mark.parametrize("now", ["2026-11-18T23:00:00Z", "2026-11-19T07:59:00Z"])
@pytest.mark.parametrize("svc", ALL)
def test_refuses_in_wed_thu_freeze(box, svc, now):
    box.write_logs(now=now)
    assert_refused(box, box.run(svc, now=now), "freeze")


@pytest.mark.parametrize("svc", ALL)
def test_owner_ack_lets_freeze_through_and_is_forwarded(box, svc):
    now = "2026-11-18T23:00:00Z"
    box.write_logs(now=now)
    proc = box.run(svc, "--ack", "freeze", now=now)
    assert proc.returncode == 0, output(proc)
    checks = plan_lines(proc.stdout, "check")
    assert any("--ack freeze" in ln for ln in checks), "\n".join(checks) or proc.stdout


def test_quiet_window_just_outside_freeze_is_go(box):
    now = "2026-11-19T08:00:00Z"
    box.write_logs(now=now)
    proc = box.run("validator", now=now)
    assert proc.returncode == 0, output(proc)


@pytest.mark.parametrize("svc", ALL)
def test_missing_required_flag_is_a_usage_error(box, svc):
    args = box.args(svc)
    i = args.index("--release-ref")
    del args[i:i + 2]
    proc = box.run(svc, args=args)
    assert proc.returncode != 0, output(proc)
    assert plan_lines(proc.stdout, "restart") == [], proc.stdout
