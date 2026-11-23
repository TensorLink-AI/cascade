"""WB23 intake (TB25): ``X-Lane-B-Train-Hours`` on ``POST /v1/lane-b/fund``.
6.0 h is admitted with steps from the measured per-GPU rate and a guard at
or under ``max_leg_seconds``; 6.5 h is refused at funding with nothing
queued or vaulted (nothing charged). The intake is built with the production
catalogs (``main.gpu_catalogs``); only the chain reads are stand-ins."""

from __future__ import annotations

import inspect

import pytest

from .helpers import (
    BAD_TRAIN_HOURS,
    NEEDS_GPU,
    RATE,
    SKU,
    TRAIN_HOURS_HEADER,
    UNAVAILABLE,
    WITH_FALLBACK,
    base_contract,
    chain_cfg,
    lane_b_cfg,
    rates_env,
    want_guard,
    want_reference_steps,
    want_steps,
)

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "org/gen@sha256:" + "a" * 64


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _intake(tmp_path, *, wired=True, landing=None):
    from cascade.funding.intake import FundingIntake

    params = inspect.signature(FundingIntake).parameters
    assert "lane_b_landing" in params, (
        "FundingIntake takes no lane_b_landing (WB23: the expected landing boundary)")
    from cascade.funding.main import gpu_catalogs
    from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
    from cascade.funding.vault import PayerKeyVault

    cfg = chain_cfg()
    cat_a, gpus_b = gpu_catalogs(cfg, env=rates_env(tmp_path))
    clock = Clock()
    lane_a = FundedQueue(tmp_path / "queue.json", clock=clock)
    lane_b = LaneBQueue(lane_b_queue_path(tmp_path / "queue.json"), clock=clock)
    intake = FundingIntake(
        lane_a, PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, ref: 700, require_signature=False,
        verify=lambda hk, msg, s: False, clock=clock, gpu_catalog=cat_a,
        lane_b_queue=lane_b, resolve_lane_b_reveal=lambda hk, t, r: 700,
        lane_b_targets=None, lane_b_max_targets=2,
        lane_b_gpus=gpus_b if wired else None, lane_b_cfg=cfg.lane_b,
        lane_b_landing=landing)
    return intake, lane_b


def _h(**kw):
    from cascade.funding.intake import LANE_B_TARGETS_HEADER

    h = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, LANE_B_TARGETS_HEADER: "energy",
         "X-Lium-Api-Key": "sk-live", "X-Gpu-Type": SKU}
    h.update(kw)
    return {k: v for k, v in h.items() if v is not None}


def _nothing_charged(intake, lane_b):
    assert lane_b.get(HK) is None, "a refused fund must queue nothing"
    assert intake.vault.get(HK) is None, "a refused fund must vault no key"


def test_tb25_six_hours_admitted(tmp_path):
    intake, lane_b = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "6.0"}))
    assert status == 202, body
    lb, c = lane_b_cfg(), base_contract()
    steps = want_steps(6.0, RATE, want_reference_steps(c, 6.0))
    assert body["train_hours"] == 6.0 and body["steps"] == steps
    assert body["guard_seconds"] == want_guard(6.0, lb) <= int(lb.max_leg_seconds)
    assert body["gpu"] == SKU and body["steps_per_s"] == pytest.approx(RATE)
    e = lane_b.get(HK)
    assert (e.train_hours, e.leg_steps, e.leg_guard_seconds) == (
        6.0, steps, body["guard_seconds"])
    assert intake.vault.get(HK) == "sk-live"


def test_tb25_six_and_a_half_hours_refused_nothing_charged(tmp_path):
    intake, lane_b = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "6.5"}))
    assert status == 400 and body["code"] == BAD_TRAIN_HOURS, body
    _nothing_charged(intake, lane_b)


@pytest.mark.parametrize("raw", ["0", "0.25", "1.75", "abc", "nan"])
def test_off_grid_hours_refused(tmp_path, raw):
    intake, lane_b = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: raw}))
    assert status == 400 and body["code"] == BAD_TRAIN_HOURS, body
    _nothing_charged(intake, lane_b)


def test_train_hours_needs_a_chosen_gpu(tmp_path):
    intake, lane_b = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "2.0", "X-Gpu-Type": None}))
    assert status == 400 and body["code"] == NEEDS_GPU, body
    _nothing_charged(intake, lane_b)


def test_train_hours_with_a_fallback_refused(tmp_path):
    intake, lane_b = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "2.0",
                                            "X-Gpu-Allow-Fallback": "true"}))
    assert status == 400 and body["code"] == WITH_FALLBACK, body
    _nothing_charged(intake, lane_b)


def test_train_hours_without_lane_b_gpus_unavailable(tmp_path):
    intake, lane_b = _intake(tmp_path, wired=False)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "2.0"}))
    assert status == 400 and body["code"] == UNAVAILABLE, body
    _nothing_charged(intake, lane_b)


def test_fund_without_train_hours_unchanged_when_unarmed(tmp_path):
    """No header: today's fund — no leg length on the entry or in the reply."""
    intake, lane_b = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h())
    assert status == 202, body
    assert not {"train_hours", "steps", "guard_seconds", "landing"} & set(body)
    e = lane_b.get(HK)
    assert (e.train_hours, e.leg_steps, e.leg_guard_seconds) == (0.0, 0, 0)


def test_fund_without_train_hours_armed_nonempty_lane_d_effect(tmp_path):
    intake, lane_b = _intake(tmp_path)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "0.5"}))
    assert status == 202, body
    assert body["train_hours"] == 0.5 and lane_b.get(HK).leg_steps == body["steps"] > 0


def test_tb26_landing_boundary_returned_to_the_miner(tmp_path):
    seen = []

    def landing(guard_seconds):
        seen.append(guard_seconds)
        return {"target_boundary": 9_600_000, "era_index": 12, "cross_era": True}

    intake, _ = _intake(tmp_path, landing=landing)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "6.0"}))
    assert status == 202, body
    assert seen == [body["guard_seconds"]]
    assert body["landing"] == {"target_boundary": 9_600_000, "era_index": 12,
                               "cross_era": True}


def test_landing_failure_never_fails_the_fund(tmp_path):
    def landing(guard_seconds):
        raise RuntimeError("no chain")

    intake, lane_b = _intake(tmp_path, landing=landing)
    status, body = intake.lane_b_fund(_h(**{TRAIN_HOURS_HEADER: "3.0"}))
    assert status == 202 and "landing" not in body, body
    assert lane_b.get(HK).train_hours == 3.0


# ── the signed field ──────────────────────────────────────────────────────────

def test_signed_ref_binds_train_hours():
    from cascade.funding.intake import lane_b_signed_ref

    assert "train_hours" in inspect.signature(lane_b_signed_ref).parameters, (
        "lane_b_signed_ref takes no train_hours (WB23: the signature binds the leg length)")
    today = lane_b_signed_ref(("energy",), REF)
    assert lane_b_signed_ref(("energy",), REF, train_hours="") == today
    assert lane_b_signed_ref(("energy",), REF, train_hours=" 6.0 ") == today + "|hours=6.0"
    rows = {0: [1, 2]}
    with_rows = lane_b_signed_ref(("energy",), REF, rows)
    assert lane_b_signed_ref(("energy",), REF, rows, train_hours="2") == with_rows + "|hours=2"


def test_queue_entry_round_trips_and_old_files_load(tmp_path):
    import dataclasses

    from cascade.funding.queue import FundedEntry, LaneBQueue, lane_b_queue_path

    names = {f.name: f.default for f in dataclasses.fields(FundedEntry)}
    for k, d in (("train_hours", 0.0), ("leg_steps", 0), ("leg_guard_seconds", 0)):
        assert k in names and names[k] == d, f"FundedEntry.{k} missing or default != {d}"
    assert "leg_length" in inspect.signature(LaneBQueue.add_lane_b).parameters
    from .helpers import leg

    m = leg()
    path = lane_b_queue_path(tmp_path / "queue.json")
    q = LaneBQueue(path, clock=Clock())
    q.add_lane_b(HK, ("energy",), REF, 700, leg_length=m.LegLength(2.0, 900, 9600))
    again = LaneBQueue(path, clock=Clock()).get(HK)
    assert (again.train_hours, again.leg_steps, again.leg_guard_seconds) == (2.0, 900, 9600)
    row = next(r for r in q.public_view()["entries"] if r.get("hotkey") == HK)
    assert row.get("train_hours") == 2.0 and row.get("leg_steps") == 900
