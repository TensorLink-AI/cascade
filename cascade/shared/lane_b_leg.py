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
    "BAD_TRAIN_TOKENS",
    "TOKENS_OUT_OF_RANGE",
    "TOKENS_WITH_HOURS",
    "TRAIN_TOKENS_HEADER",
    "HOURS_STEP",
    "MIN_TRAIN_HOURS",
    "NEEDS_GPU",
    "TRAIN_HOURS_HEADER",
    "UNAVAILABLE",
    "WITH_FALLBACK",
    "LegLength",
    "LegLengthError",
    "admit_leg_length",
    "admit_leg_limits",
    "limit_wall_seconds",
    "check_leg_record",
    "LIMIT_STOPS",
    "leg_caps",
    "leg_contract",
    "leg_guard_seconds",
    "leg_steps",
    "parse_train_hours",
    "parse_train_tokens",
    "reference_steps",
    "tokens_per_step",
    "train_hours_tag",
    "train_tokens_tag",
]

TRAIN_HOURS_HEADER = "X-Lane-B-Train-Hours"
# owner 2026-11-24 (rev6.21): a token budget instead of hours
TRAIN_TOKENS_HEADER = "X-Lane-B-Train-Tokens"
HOURS_STEP = 0.5
MIN_TRAIN_HOURS = 0.5

BAD_TRAIN_HOURS = "bad_train_hours"
NEEDS_GPU = "train_hours_needs_gpu"
WITH_FALLBACK = "train_hours_with_fallback"
UNAVAILABLE = "train_hours_unavailable"
BAD_TRAIN_TOKENS = "bad_train_tokens"
TOKENS_WITH_HOURS = "train_tokens_with_hours"
TOKENS_OUT_OF_RANGE = "train_tokens_out_of_range"
# a token budget is a positive integer of at most this many digits
_MAX_TOKEN_DIGITS = 16


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
    # owner 2026-11-24 (rev6.21): the requested token budget (0 = an hours
    # fund); the steps are then derived from it and the hours from the steps
    train_tokens: int = 0
    # owner 2026-11-24: False = a GPU type with no WB19 rate, at the miner's
    # own risk (no refund if it OOMs, cannot boot or crashes)
    gpu_measured: bool = True

    @property
    def limits(self) -> bool:
        """A token/hour-limits leg (stops at whichever limit comes first)."""
        return bool(self.train_tokens) or not self.gpu_measured

    def as_dict(self) -> dict:
        d = {"train_hours": float(self.train_hours), "steps": int(self.steps),
             "guard_seconds": int(self.guard_seconds)}
        if self.train_tokens:
            d["train_tokens"] = int(self.train_tokens)
        if not self.gpu_measured:
            d["gpu_measured"] = False
        return d


def leg_caps(lb_cfg, block: int | None = None) -> tuple[float, int]:
    """``(max_leg_hours, max_leg_seconds)`` binding a leg at ``block`` (owner
    2026-11-24: 12 h / 54000 s shipped, no activation block)."""
    from .config.lane_b import lane_b_leg_caps

    return lane_b_leg_caps(lb_cfg, block)


def admit_leg_length(lb_cfg, rate, train_hours: float, *, contract) -> LegLength:
    """The leg length fixed at admission on the measured ``rate`` (WB19 StepRate)."""
    h = float(train_hours)
    steps = leg_steps(h, rate.steps_per_s, cap=reference_steps(contract, h))
    guard = leg_guard_seconds(h, guard_factor=float(lb_cfg.guard_factor),
                              overhead_seconds=float(rate.overhead_seconds),
                              max_leg_seconds=leg_caps(lb_cfg)[1])
    return LegLength(train_hours=h, steps=steps, guard_seconds=guard)


def tokens_per_step(contract) -> int:
    """Points one optimizer step of ``contract`` consumes: ``batch_size``
    sequences of ``context_length`` points, i.e. ``context_length //
    patch_size`` patches of ``patch_size`` each (``batch_denomination =
    "sequences"``, ``budget_denomination = "points"``). Derived from the
    contract, never a constant (2,097,152 on the shipped v2 contract)."""
    batch, ctx, patch = (int(contract.batch_size), int(contract.context_length),
                         int(getattr(contract, "patch_size", 0) or 0) or 1)
    if batch < 1 or ctx < 1 or ctx % patch:
        raise ValueError(f"contract batch/context/patch {batch}/{ctx}/{patch} gives no step size")
    return batch * (ctx // patch) * patch


def parse_train_tokens(raw) -> int | None:
    """The ``X-Lane-B-Train-Tokens`` value as an int (``20e9`` and
    ``20000000000`` both accepted, must be a whole number > 0), None when
    absent. Raises :class:`LegLengthError` (``BAD_TRAIN_TOKENS``)."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        d = Decimal(text)
    except InvalidOperation:
        d = None
    if (d is None or not d.is_finite() or d <= 0 or d != d.to_integral_value()
            or len(str(int(d))) > _MAX_TOKEN_DIGITS):
        raise LegLengthError(BAD_TRAIN_TOKENS, f"{TRAIN_TOKENS_HEADER} {text!r} must be a "
                                               "whole number of tokens > 0 (e.g. 20e9)")
    return int(d)


def train_tokens_tag(raw) -> str:
    """The signed-ref tag of the tokens header as sent; ``""`` absent."""
    text = str(raw or "").strip()
    return f"tokens={text}" if text else ""


def admit_leg_limits(lb_cfg, rate, *, train_hours: float | None, train_tokens: int | None,
                     contract) -> LegLength:
    """owner 2026-11-24 (rev6.21): a leg that stops at its token limit OR its
    hour limit, whichever comes first, on any GPU (``rate`` None =
    unmeasured, at the miner's own risk). The hour limit is ``train_hours``,
    else the ``max_leg_hours`` in force; the steps are the token limit's
    (``tokens // tokens_per_step``) capped at the WB23 steps cap (a lane A
    leg of the same hours), or that cap with no token limit. The wall is the
    hour limit itself (hitting it is a normal stop, not a failure). Raises
    :class:`LegLengthError` (``TOKENS_OUT_OF_RANGE``)."""
    max_hours, max_secs = leg_caps(lb_cfg)
    h = float(train_hours) if train_hours else float(max_hours)
    if not _on_grid(h, max_hours):
        raise LegLengthError(BAD_TRAIN_HOURS, _grid_message(f"{h:g}", max_hours))
    cap = reference_steps(contract, h)
    if train_tokens:
        steps = int(train_tokens) // tokens_per_step(contract)
        if steps < 1:
            raise LegLengthError(TOKENS_OUT_OF_RANGE, f"{int(train_tokens)} tokens is less "
                                                      f"than one step ({tokens_per_step(contract)}"
                                                      " tokens)")
        steps = min(steps, cap)
    else:
        steps = cap
    del rate                         # the wall, not the rate, bounds these legs
    return LegLength(train_hours=h, steps=steps, guard_seconds=limit_wall_seconds(h, max_secs),
                     train_tokens=int(train_tokens or 0))


def limit_wall_seconds(hours: float, max_leg_seconds: int) -> int:
    """The hour limit of a limits leg as its training wall (capped)."""
    return int(min(math.ceil(float(hours) * 3600.0), int(max_leg_seconds)))


def leg_contract(base, length: LegLength):
    """The lane B leg contract: steps fixed at admission, the lane B guard. A
    limits leg (owner 2026-11-24) also records its actual steps / tokens /
    stop in the checkpoint (``lane_b_stop_record``)."""
    return replace(base, training_steps=int(length.steps),
                   target_train_hours=float(length.train_hours),
                   max_train_seconds=int(length.guard_seconds), extra_sizes=(),
                   **({"lane_b_stop_record": True} if length.limits else {}))


LIMIT_STOPS = ("steps", "tokens", "hours", "corpus")


def check_leg_record(*, train_hours, steps, guard_seconds, lb_cfg, contract,
                     block: int | None = None, train_tokens: int = 0,
                     gpu_measured: bool = True, trained_steps: int = 0,
                     trained_tokens: int = 0, stop: str = "") -> str | None:
    """Consensus check of a recorded leg length; None = valid (or not recorded).
    ``block`` is the entry's ``train_block`` (caps: owner 2026-11-24, 12 h).

    A *limits* leg (owner 2026-11-24: a token limit, or an unmeasured GPU):
    its steps are re-derived
    from the token limit and the hour limit's steps cap, its wall is the hour
    limit, and the recorded actuals (``trained_steps`` / ``trained_tokens`` /
    ``stop``, from the leg checkpoint) must be consistent with both limits —
    judges re-derive from what the leg trained, never an admission estimate."""
    limits = bool(train_tokens) or not gpu_measured
    if not limits and not train_hours and not steps and not guard_seconds:
        return None
    max_hours, mls = leg_caps(lb_cfg, block)
    if not _on_grid(train_hours, max_hours):
        return "train_hours_out_of_range"
    h = float(train_hours)
    cap = reference_steps(contract, h)
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= cap:
        return "steps_out_of_range"
    if not limits:
        floor = min(math.ceil(h * 3600.0 * float(lb_cfg.guard_factor)), mls)
        if isinstance(guard_seconds, bool) or not isinstance(guard_seconds, int) or not (
                floor <= guard_seconds <= mls):
            return "guard_out_of_range"
        return None
    if train_tokens:
        if isinstance(train_tokens, bool) or not isinstance(train_tokens, int) or (
                steps != min(int(train_tokens) // tokens_per_step(contract), cap)):
            return "steps_not_tokens"
    elif steps != cap:
        return "steps_out_of_range"
    if guard_seconds != limit_wall_seconds(h, mls):
        return "guard_out_of_range"
    for v in (trained_steps, trained_tokens):
        if isinstance(v, bool) or not isinstance(v, int) or v < 0:
            return "actuals_out_of_range"
    if stop not in LIMIT_STOPS or not 1 <= trained_steps <= steps:
        return "actuals_out_of_range"
    if stop == "steps" and trained_steps != steps:
        return "actuals_out_of_range"
    if trained_tokens > trained_steps * tokens_per_step(contract):
        return "actuals_out_of_range"
    return None
