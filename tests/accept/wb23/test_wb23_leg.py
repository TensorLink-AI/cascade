"""WB23 module ``cascade.shared.lane_b_leg``: the train_hours grid, steps fixed
from the measured per-GPU rate (D42b (a)), the guard capped at ``[lane_b]
max_leg_seconds``, the lane B leg contract, and the consensus check of a
recorded leg (TB25, TB27)."""

from __future__ import annotations

import dataclasses
import math

import pytest

from .helpers import (
    BAD_TRAIN_HOURS,
    NEEDS_GPU,
    OVERHEAD,
    RATE,
    REPO,
    TRAIN_HOURS_HEADER,
    UNAVAILABLE,
    WITH_FALLBACK,
    base_contract,
    lane_b_cfg,
    leg,
    step_rate,
    want_guard,
    want_guard_floor,
    want_reference_steps,
    want_steps,
)


def test_constants():
    m = leg()
    assert m.TRAIN_HOURS_HEADER == TRAIN_HOURS_HEADER
    assert (m.HOURS_STEP, m.MIN_TRAIN_HOURS) == (0.5, 0.5)
    assert (m.BAD_TRAIN_HOURS, m.NEEDS_GPU, m.WITH_FALLBACK, m.UNAVAILABLE) == (
        BAD_TRAIN_HOURS, NEEDS_GPU, WITH_FALLBACK, UNAVAILABLE)
    assert issubclass(m.LegLengthError, ValueError)


def test_module_is_torch_free():
    path = REPO / "cascade" / "shared" / "lane_b_leg.py"
    assert path.is_file(), "cascade/shared/lane_b_leg.py is missing (WB23 builds it)"
    src = path.read_text(encoding="utf-8")
    assert "import torch" not in src and "from torch" not in src


def test_shipped_keys_are_the_d42_values():
    """WB10's keys as D42 decided them: 6 h, 6 h x 1.25 s, 1.25."""
    lb = lane_b_cfg()
    assert float(lb.max_leg_hours) == 6.0
    assert int(lb.max_leg_seconds) == 27_000
    assert float(lb.guard_factor) == 1.25


@pytest.mark.parametrize("raw,want", [("0.5", 0.5), ("1", 1.0), ("3.0", 3.0), (" 6.0 ", 6.0),
                                      ("5.5", 5.5)])
def test_parse_train_hours_accepts_the_grid(raw, want):
    assert leg().parse_train_hours(raw, max_leg_hours=6.0) == want


@pytest.mark.parametrize("raw", ["", None, "   "])
def test_parse_train_hours_absent_is_todays_leg(raw):
    assert leg().parse_train_hours(raw, max_leg_hours=6.0) is None


@pytest.mark.parametrize("raw", ["0", "0.25", "0.75", "6.5", "7", "nan", "inf", "-1", "abc",
                                 "1e400", "1.5h"])
def test_parse_train_hours_refuses_off_grid(raw):
    m = leg()
    with pytest.raises(m.LegLengthError) as ei:
        m.parse_train_hours(raw, max_leg_hours=6.0)
    assert ei.value.code == BAD_TRAIN_HOURS


def test_parse_train_hours_follows_the_cap():
    m = leg()
    assert m.parse_train_hours("6.5", max_leg_hours=8.0) == 6.5
    with pytest.raises(m.LegLengthError):
        m.parse_train_hours("4.5", max_leg_hours=4.0)


def test_train_hours_tag():
    m = leg()
    assert m.train_hours_tag(" 6.0 ") == "hours=6.0"
    assert m.train_hours_tag("") == "" and m.train_hours_tag(None) == ""


def test_reference_steps_is_for_hours_formula():
    m = leg()
    c = base_contract()
    for h in (0.5, 3.0, 6.0):
        assert m.reference_steps(c, h) == want_reference_steps(c, h)
        assert m.reference_steps(c, h) == c.for_hours(h).training_steps
    assert m.reference_steps(c, 3.0) == c.training_steps


def test_leg_steps_real_hours_on_the_gpu():
    m = leg()
    assert m.leg_steps(6.0, RATE, cap=10 ** 9) == want_steps(6.0, RATE, 10 ** 9)
    assert m.leg_steps(0.5, 0.1366, cap=10 ** 9) == math.floor(1800 * 0.1366)
    # a cheaper (slower) GPU buys fewer steps per hour
    assert m.leg_steps(3.0, 0.1, cap=10 ** 9) < m.leg_steps(3.0, 0.4, cap=10 ** 9)


def test_leg_steps_clamped_and_never_zero():
    m = leg()
    assert m.leg_steps(6.0, 100.0, cap=500) == 500
    with pytest.raises(ValueError):
        m.leg_steps(0.5, 1e-6, cap=10 ** 9)


def test_leg_guard_seconds():
    m = leg()
    lb = lane_b_cfg()
    kw = dict(guard_factor=float(lb.guard_factor), max_leg_seconds=int(lb.max_leg_seconds))
    assert m.leg_guard_seconds(0.5, overhead_seconds=OVERHEAD, **kw) == want_guard(0.5, lb)
    assert m.leg_guard_seconds(3.0, overhead_seconds=OVERHEAD, **kw) == want_guard(3.0, lb)
    # 6 h x 1.25 + overhead > 27,000 s: capped at max_leg_seconds
    assert m.leg_guard_seconds(6.0, overhead_seconds=OVERHEAD, **kw) == int(lb.max_leg_seconds)


def test_tb25_six_hours_admitted_with_guard_under_cap():
    m = leg()
    lb, c = lane_b_cfg(), base_contract()
    got = m.admit_leg_length(lb, step_rate(), 6.0, contract=c)
    assert dataclasses.is_dataclass(got) and got.train_hours == 6.0
    assert got.steps == want_steps(6.0, RATE, want_reference_steps(c, 6.0))
    assert got.guard_seconds == want_guard(6.0, lb) <= int(lb.max_leg_seconds)
    assert got.as_dict() == {"train_hours": 6.0, "steps": got.steps,
                             "guard_seconds": got.guard_seconds}
    with pytest.raises(dataclasses.FrozenInstanceError):
        got.steps = 1


def test_admission_clamps_to_the_reference_steps():
    m = leg()
    c = base_contract()
    got = m.admit_leg_length(lane_b_cfg(), step_rate(rate=50.0), 1.0, contract=c)
    assert got.steps == want_reference_steps(c, 1.0)


def test_leg_contract_uses_the_lane_b_cap_not_max_train_seconds():
    m = leg()
    lb, c = lane_b_cfg(), base_contract()
    length = m.admit_leg_length(lb, step_rate(), 6.0, contract=c)
    before = dataclasses.asdict(c)
    lc = m.leg_contract(c, length)
    assert lc.training_steps == length.steps
    assert lc.target_train_hours == 6.0
    assert lc.max_train_seconds == length.guard_seconds
    # 27,000 s is above the pinned [v2.training] max_train_seconds: the lane B
    # cap replaces it for this leg only, the base contract is untouched
    assert length.guard_seconds > c.max_train_seconds
    assert lc.extra_sizes == ()
    assert dataclasses.asdict(c) == before


# ── the consensus check of a recorded leg (TB27) ─────────────────────────────

def _check(**kw):
    return leg().check_leg_record(lb_cfg=lane_b_cfg(), contract=base_contract(), **kw)


def test_check_leg_record_unchanged_when_unarmed():
    """A leg funded without train_hours records nothing: no check."""
    assert _check(train_hours=0.0, steps=0, guard_seconds=0) is None


def test_check_leg_record_armed_nonempty_lane_d_effect():
    lb, c = lane_b_cfg(), base_contract()
    good = leg().admit_leg_length(lb, step_rate(), 6.0, contract=c)
    assert _check(train_hours=6.0, steps=good.steps, guard_seconds=good.guard_seconds) is None
    assert _check(train_hours=6.0, steps=good.steps + want_reference_steps(c, 6.0),
                  guard_seconds=good.guard_seconds) == "steps_out_of_range"


@pytest.mark.parametrize("hours", [0.25, 0.75, 6.5, 0.0, -1.0, float("nan")])
def test_check_leg_record_refuses_hours_off_grid(hours):
    assert _check(train_hours=hours, steps=10, guard_seconds=3000) == "train_hours_out_of_range"


def test_check_leg_record_steps_bounds():
    c = base_contract()
    cap = want_reference_steps(c, 2.0)
    g = want_guard_floor(2.0, lane_b_cfg())
    assert _check(train_hours=2.0, steps=cap, guard_seconds=g) is None
    assert _check(train_hours=2.0, steps=cap + 1, guard_seconds=g) == "steps_out_of_range"
    assert _check(train_hours=2.0, steps=0, guard_seconds=g) == "steps_out_of_range"


def test_tb27_guard_over_the_cap_is_refused():
    lb = lane_b_cfg()
    mls = int(lb.max_leg_seconds)
    assert _check(train_hours=6.0, steps=100, guard_seconds=mls) is None
    assert _check(train_hours=6.0, steps=100, guard_seconds=mls + 1) == "guard_out_of_range"
    floor = want_guard_floor(1.0, lb)
    assert _check(train_hours=1.0, steps=100, guard_seconds=floor) is None
    assert _check(train_hours=1.0, steps=100, guard_seconds=floor - 1) == "guard_out_of_range"
