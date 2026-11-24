"""Miner-chosen lane B leg length (WB23; LANE_D_PLAN §12b.6 D42 item 2, D42b).

A lane B fund may carry ``train_hours`` (0.5 .. ``[lane_b] max_leg_hours``,
0.5 h steps). D42b (a): the hours are wall hours on the chosen GPU, so the
leg's steps are fixed at admission from that GPU's measured expert-only step
rate (WB19) and carried in the signed leg record. The wall guard is
``train_hours x guard_factor`` plus the measured overhead, capped at
``[lane_b] max_leg_seconds``. ``[training] max_train_seconds`` is never
edited: the lane B leg contract carries the lane B cap instead.

The consensus check (:func:`check_leg_record`) needs no step-rate table:
validators and satellites check the recorded steps and guard against the
``[lane_b]`` keys and the contract alone.

Stdlib only: importing it never imports torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

__all__ = [
    "BAD_TRAIN_HOURS",
    "HOURS_STEP",
    "MIN_TRAIN_HOURS",
    "NEEDS_GPU",
    "TRAIN_HOURS_HEADER",
    "UNAVAILABLE",
    "WITH_FALLBACK",
    "LegLength",
    "LegLengthError",
    "admit_leg_length",
    "check_leg_record",
    "leg_contract",
    "leg_guard_seconds",
    "leg_steps",
    "parse_train_hours",
    "reference_steps",
    "train_hours_tag",
]

TRAIN_HOURS_HEADER = "X-Lane-B-Train-Hours"
HOURS_STEP = 0.5
MIN_TRAIN_HOURS = 0.5

BAD_TRAIN_HOURS = "bad_train_hours"
NEEDS_GPU = "train_hours_needs_gpu"
WITH_FALLBACK = "train_hours_with_fallback"
UNAVAILABLE = "train_hours_unavailable"


class LegLengthError(ValueError):
    """A refused ``train_hours``; ``code`` is the typed refusal code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _on_grid(hours: float, max_leg_hours: float) -> bool:
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        return False
    if not math.isfinite(hours) or hours < MIN_TRAIN_HOURS or hours > float(max_leg_hours):
        return False
    return float(hours * 2).is_integer()


def parse_train_hours(raw, *, max_leg_hours: float) -> float | None:
    """The header value as hours, None when absent (today's leg). Raises
    :class:`LegLengthError` (``BAD_TRAIN_HOURS``) for anything off the grid."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        d = Decimal(text)
    except InvalidOperation:
        d = None
    if d is None or not d.is_finite() or (d * 2) != (d * 2).to_integral_value():
        raise LegLengthError(BAD_TRAIN_HOURS, _grid_message(text, max_leg_hours))
    hours = float(d)
    if not _on_grid(hours, max_leg_hours):
        raise LegLengthError(BAD_TRAIN_HOURS, _grid_message(text, max_leg_hours))
    return hours


def _grid_message(text: str, max_leg_hours: float) -> str:
    return (f"{TRAIN_HOURS_HEADER} {text!r} must be {MIN_TRAIN_HOURS:g} .. "
            f"{float(max_leg_hours):g} hours in {HOURS_STEP:g} h steps")


def train_hours_tag(raw) -> str:
    """The signed-ref tag of the header value as sent (stripped); ``""`` absent."""
    text = str(raw or "").strip()
    return f"hours={text}" if text else ""


def reference_steps(contract, train_hours: float) -> int:
    """The steps a lane A leg of ``train_hours`` runs (``for_hours``'s formula):
    the consensus ceiling of a lane B leg's steps."""
    return max(1, round(int(contract.training_steps) * float(train_hours)
                        / float(contract.target_train_hours)))


def leg_steps(train_hours: float, steps_per_s: float, *, cap: int) -> int:
    steps = min(math.floor(float(train_hours) * 3600.0 * float(steps_per_s)), int(cap))
    if steps < 1:
        raise ValueError(f"{train_hours:g} h at {steps_per_s:g} steps/s trains no step")
    return int(steps)


def leg_guard_seconds(train_hours: float, *, guard_factor: float, overhead_seconds: float,
                      max_leg_seconds: int) -> int:
    return int(min(math.ceil(float(train_hours) * 3600.0 * float(guard_factor)
                             + float(overhead_seconds)), int(max_leg_seconds)))


@dataclass(frozen=True)
class LegLength:
    train_hours: float
    steps: int
    guard_seconds: int

    def as_dict(self) -> dict:
        return {"train_hours": float(self.train_hours), "steps": int(self.steps),
                "guard_seconds": int(self.guard_seconds)}


def admit_leg_length(lb_cfg, rate, train_hours: float, *, contract) -> LegLength:
    """The leg length fixed at admission on the measured ``rate`` (WB19 StepRate)."""
    h = float(train_hours)
    steps = leg_steps(h, rate.steps_per_s, cap=reference_steps(contract, h))
    guard = leg_guard_seconds(h, guard_factor=float(lb_cfg.guard_factor),
                              overhead_seconds=float(rate.overhead_seconds),
                              max_leg_seconds=int(lb_cfg.max_leg_seconds))
    return LegLength(train_hours=h, steps=steps, guard_seconds=guard)


def leg_contract(base, length: LegLength):
    """The lane B leg contract: steps fixed at admission, the lane B guard."""
    return replace(base, training_steps=int(length.steps),
                   target_train_hours=float(length.train_hours),
                   max_train_seconds=int(length.guard_seconds), extra_sizes=())


def check_leg_record(*, train_hours, steps, guard_seconds, lb_cfg, contract) -> str | None:
    """Consensus check of a recorded leg length; None = valid (or not recorded)."""
    if not train_hours and not steps and not guard_seconds:
        return None
    if not _on_grid(train_hours, float(lb_cfg.max_leg_hours)):
        return "train_hours_out_of_range"
    h = float(train_hours)
    if isinstance(steps, bool) or not isinstance(steps, int) or not (
            1 <= steps <= reference_steps(contract, h)):
        return "steps_out_of_range"
    mls = int(lb_cfg.max_leg_seconds)
    floor = min(math.ceil(h * 3600.0 * float(lb_cfg.guard_factor)), mls)
    if isinstance(guard_seconds, bool) or not isinstance(guard_seconds, int) or not (
            floor <= guard_seconds <= mls):
        return "guard_out_of_range"
    return None
