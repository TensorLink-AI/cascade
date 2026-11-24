"""Miner-chosen lane B leg levers (WB27; LANE_D_PLAN rev6.21, owner 2026-11-24).

A lane B fund may carry one header, ``X-Lane-B-Levers``, a JSON object with
any of four keys (an absent key is its default; no header = every lever at
its default = today's leg):

* ``lr_scale``       multiplies the leg's base LR, ``[lane_b] lr_scale_min`` ..
                     ``lr_scale_max`` (shipped 0.5 .. 2.0), at most 3 decimals;
                     default 1.0.
* ``curriculum``     1..3 ordered stages ``[[name, fraction], ...]``: names
                     ``[a-z][a-z0-9_-]{0,23}``, unique; fractions > 0 with at
                     most 2 decimals, summing to exactly 1. Stage *i* is its
                     own corpus (seed :func:`curriculum_stage_seed`, generator
                     keyword ``curriculum_stage=<name>``), read in order.
                     Default: one stage (today).
* ``context_length`` the leg's training window, 4096 .. ``[lane_b]
                     max_context_length`` (32768), powers of two. The model
                     has no length bound (xPos per forward pass; the only
                     4096 lock is the contract-field check in
                     ``CascadeMoEConfig.from_contract``, which the lane B leg
                     builds at 4096): the batch window and the tokens per
                     step change for that leg only; the checkpoint's
                     ``moe.context_length`` stays 4096 and judging is
                     unchanged. A context above 4096 must fit the chosen
                     GPU's measured memory (:func:`context_fits`).
* ``ema``            the scored power-EMA width from ``[lane_b]
                     sigma_rel_choices`` (shipped 0.03 / 0.05 / 0.10 / 0.15 /
                     0.20; default 0.10, rev6.20's); only that EMA is
                     computed, scored and merged (every lane B leg is a
                     rev6.20 EMA / no-cooldown leg).

No activation block (owner 2026-11-24: lane B is not rolled out, the levers
are its default behaviour); the bounds, the signed record and the refusals
stay. The expert manifest entry records the
canonical non-default levers (``levers``); judges re-derive with
:func:`check_entry_levers`. The trainer and the worker derive the leg
contract with ONE helper, :func:`lane_b_leg_contract`.

Stdlib only: importing it never imports torch.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation

__all__ = [
    "leg_context_contract",
    "BAD_CURRICULUM",
    "BAD_LEVERS",
    "CONTEXT_OUT_OF_RANGE",
    "CONTEXT_UNFIT",
    "DEFAULT_CONTEXT",
    "EMA_OUT_OF_RANGE",
    "EMA_WIDTHS",
    "LEVERS_HEADER",
    "LEVER_CODES",
    "LR_OUT_OF_RANGE",
    "MAX_STAGES",
    "LegLevers",
    "LeversError",
    "check_entry_levers",
    "context_fits",
    "curriculum_points",
    "curriculum_stage_seed",
    "effective_levers",
    "apply_spec_levers",
    "lane_b_leg_contract",
    "leg_length_contract",
    "lane_b_scored_weights_for",
    "levers_from_dict",
    "levers_tag",
    "parse_levers",
    "valid_stage_name",
]

LEVERS_HEADER = "X-Lane-B-Levers"

BAD_LEVERS = "bad_levers"
LR_OUT_OF_RANGE = "lr_scale_out_of_range"
BAD_CURRICULUM = "bad_curriculum"
CONTEXT_OUT_OF_RANGE = "context_length_out_of_range"
CONTEXT_UNFIT = "context_length_unfit"
EMA_OUT_OF_RANGE = "ema_out_of_range"
LEVER_CODES = (BAD_LEVERS, LR_OUT_OF_RANGE, BAD_CURRICULUM,
               CONTEXT_OUT_OF_RANGE, CONTEXT_UNFIT,
               EMA_OUT_OF_RANGE)

MAX_STAGES = 3
DEFAULT_CONTEXT = 4096
MAX_CONTEXT = 32768
# memory-fit rule of a context > 4096 (rev6.21 (3); used once it is opened)
MEMORY_HEADROOM = 0.90
# the image's POWER_EMA_SIGMA_RELS (pyramid_trainer.py); index = power_ema_eval
EMA_WIDTHS = (0.03, 0.05, 0.10, 0.15, 0.20)   # the shipped [lane_b] sigma_rel_choices
DEFAULT_EMA = 0.10
_KEYS = ("lr_scale", "curriculum", "context_length", "ema")
_STAGE_NAME = re.compile(r"[a-z][a-z0-9_-]{0,23}")


class LeversError(ValueError):
    """A refused lever; ``code`` is the typed refusal code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LegLevers:
    """The levers of one lane B leg. ``curriculum`` is ``((name, hundredths),
    ...)``; ``()`` = one stage (today)."""

    lr_scale: float = 1.0
    curriculum: tuple[tuple[str, int], ...] = ()
    context_length: int = DEFAULT_CONTEXT
    ema: float = DEFAULT_EMA

    @property
    def is_default(self) -> bool:
        return self == LegLevers()

    def as_dict(self) -> dict:
        """The canonical non-default levers (``{}`` = every default): what the
        funded entry and the expert manifest entry record."""
        out: dict = {}
        if self.lr_scale != 1.0:
            out["lr_scale"] = float(self.lr_scale)
        if self.curriculum:
            out["curriculum"] = [[str(n), _frac_text(h)] for n, h in self.curriculum]
        if self.context_length != DEFAULT_CONTEXT:
            out["context_length"] = int(self.context_length)
        if self.ema != DEFAULT_EMA:
            out["ema"] = float(self.ema)
        return out

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))


def _frac_text(hundredths: int) -> str:
    return str(Decimal(int(hundredths)) / Decimal(100))


def valid_stage_name(name) -> bool:
    """A curriculum stage name: ``[a-z][a-z0-9_-]{0,23}``."""
    return isinstance(name, str) and _STAGE_NAME.fullmatch(name) is not None


def _decimal(v, *, places: int, code: str, what: str) -> Decimal:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise LeversError(code, f"{what} must be a number; got {v!r}")
    try:
        d = Decimal(str(v).strip())
    except InvalidOperation:
        raise LeversError(code, f"{what} must be a number; got {v!r}") from None
    if not d.is_finite():
        raise LeversError(code, f"{what} must be finite; got {v!r}")
    q = d.scaleb(places)
    if q != q.to_integral_value():
        raise LeversError(code, f"{what} allows at most {places} decimals; got {v!r}")
    return d


def _lr_scale(v, lb_cfg) -> float:
    d = _decimal(v, places=3, code=LR_OUT_OF_RANGE, what="lr_scale")
    lo, hi = float(lb_cfg.lr_scale_min), float(lb_cfg.lr_scale_max)
    if not Decimal(str(lo)) <= d <= Decimal(str(hi)):
        raise LeversError(LR_OUT_OF_RANGE, f"lr_scale {v!r} must be in [{lo:g}, {hi:g}]")
    return float(d)


def _curriculum(v) -> tuple[tuple[str, int], ...]:
    if not isinstance(v, (list, tuple)) or not 1 <= len(v) <= MAX_STAGES:
        raise LeversError(BAD_CURRICULUM, f"curriculum must be 1..{MAX_STAGES} stages "
                                          f"[[name, fraction], ...]; got {v!r}")
    stages: list[tuple[str, int]] = []
    for st in v:
        if not isinstance(st, (list, tuple)) or len(st) != 2:
            raise LeversError(BAD_CURRICULUM, f"a curriculum stage is [name, fraction]; got {st!r}")
        name, frac = st
        if not isinstance(name, str) or not _STAGE_NAME.fullmatch(name):
            raise LeversError(BAD_CURRICULUM, f"stage name {name!r} must match "
                                              "[a-z][a-z0-9_-]{0,23}")
        d = _decimal(frac, places=2, code=BAD_CURRICULUM, what=f"stage {name} fraction")
        if not Decimal(0) < d <= Decimal(1):
            raise LeversError(BAD_CURRICULUM, f"stage {name} fraction must be in (0, 1]")
        stages.append((name, int(d * 100)))
    if len({n for n, _ in stages}) != len(stages):
        raise LeversError(BAD_CURRICULUM, "curriculum stage names must be unique")
    if sum(h for _, h in stages) != 100:
        raise LeversError(BAD_CURRICULUM, "curriculum fractions must sum to exactly 1")
    return () if len(stages) == 1 else tuple(stages)


def _context(v, lb_cfg=None) -> int:
    ceiling = min(MAX_CONTEXT, int(getattr(lb_cfg, "max_context_length", MAX_CONTEXT)
                                   or MAX_CONTEXT))
    if isinstance(v, bool) or not isinstance(v, int) or not (
            DEFAULT_CONTEXT <= v <= ceiling) or v & (v - 1):
        raise LeversError(CONTEXT_OUT_OF_RANGE, f"context_length must be a power of two in "
                                                f"[{DEFAULT_CONTEXT}, {ceiling}]; got {v!r}")
    return int(v)


def _ema(v, lb_cfg) -> float:
    widths = tuple(float(x) for x in getattr(lb_cfg, "sigma_rel_choices", EMA_WIDTHS))
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) not in widths:
        raise LeversError(EMA_OUT_OF_RANGE, f"ema must be one of {widths}; got {v!r}")
    return float(v)


def levers_from_dict(d, lb_cfg) -> LegLevers:
    """Validate a levers object (the header's, a funded entry's, an expert
    entry's) against ``lb_cfg``. Raises :class:`LeversError`."""
    if not isinstance(d, dict):
        raise LeversError(BAD_LEVERS, f"{LEVERS_HEADER} must be a JSON object")
    unknown = sorted(set(d) - set(_KEYS))
    if unknown:
        raise LeversError(BAD_LEVERS, f"unknown lever(s) {unknown}; known: {list(_KEYS)}")
    return LegLevers(
        lr_scale=_lr_scale(d["lr_scale"], lb_cfg) if "lr_scale" in d else 1.0,
        curriculum=_curriculum(d["curriculum"]) if "curriculum" in d else (),
        context_length=_context(d["context_length"], lb_cfg) if "context_length" in d
        else DEFAULT_CONTEXT,
        ema=_ema(d["ema"], lb_cfg) if "ema" in d else DEFAULT_EMA)


def parse_levers(raw, lb_cfg) -> LegLevers | None:
    """The header value as levers, ``None`` when absent (today's leg). Every
    refusal is a :class:`LeversError` with a typed code."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        d = json.loads(text)
    except ValueError:
        raise LeversError(BAD_LEVERS, f"{LEVERS_HEADER} is not JSON") from None
    return levers_from_dict(d, lb_cfg)


def levers_tag(raw) -> str:
    """The signed-ref tag: ``levers=<sha256 of the header value as sent,
    stripped>``; ``""`` when absent (the signed field is today's)."""
    text = str(raw or "").strip()
    return f"levers={hashlib.sha256(text.encode('utf-8')).hexdigest()}" if text else ""


def effective_levers(lb_cfg, levers: LegLevers | None, block: int | None) -> LegLevers:
    """The levers a leg trains: the funded ones (every default when none).
    ``lb_cfg`` / ``block`` are accepted for call-site symmetry (no activation
    block, owner 2026-11-24)."""
    del lb_cfg, block
    return LegLevers() if levers is None else levers


def lane_b_scored_weights_for(lb_cfg, block: int | None, levers: LegLevers | None) -> str:
    """``scored_weights`` a lane B leg must carry: the label of its EMA width
    (``"ema010"`` by default, rev6.20's; every lane B leg is an EMA leg)."""
    from .config.lane_b import lane_b_ema_label

    return lane_b_ema_label(effective_levers(lb_cfg, levers, block).ema)


def check_entry_levers(levers: dict, *, lb_cfg, train_block: int | None) -> str | None:
    """Consensus check of an expert entry's recorded ``levers``; None = valid.
    Non-default levers must re-validate against ``[lane_b]``."""
    del train_block
    if not levers:
        return None
    try:
        parsed = levers_from_dict(dict(levers), lb_cfg)
    except LeversError as e:
        return e.code
    if parsed.as_dict() != dict(levers):
        return BAD_LEVERS            # not the canonical form the trainer writes
    return None


def curriculum_stage_seed(seed: int, index: int) -> int:
    """Stage ``index``'s generation seed: a stdlib sha256 derivation of the
    leg's generation seed (63-bit, never the leg seed itself)."""
    h = hashlib.sha256(f"cascade-curriculum/1:{int(seed)}:{int(index)}".encode()).digest()
    return int.from_bytes(h[:8], "big") & ((1 << 63) - 1)


def curriculum_points(token_budget: int, curriculum) -> list[int]:
    """Each stage's point budget: ``budget × h_i // 100``, the last stage the
    remainder (so they sum to the budget exactly)."""
    stages = list(curriculum or ())
    if not stages:
        return [int(token_budget)]
    b = int(token_budget)
    out = [b * int(h) // 100 for _, h in stages[:-1]]
    return [*out, b - sum(out)]


def leg_context_contract(contract, levers: dict | None):
    """``contract`` at the context a recorded lane B leg trained (its
    ``levers["context_length"]``, default 4096): what the token limit and the
    tokens-per-step of a leg are re-derived against."""
    c = int((levers or {}).get("context_length", DEFAULT_CONTEXT) or DEFAULT_CONTEXT)
    return contract if c == int(contract.context_length) else replace(contract, context_length=c)


def context_fits(rate, context_length: int) -> bool:
    """The memory-fit rule of a context (rev6.21 (3)): the WB19 measured peak
    at 4096 scaled linearly must stay under ``MEMORY_HEADROOM`` of the type's
    memory. A rate with no memory figures never fits a context > 4096."""
    c = int(context_length)
    if c == DEFAULT_CONTEXT:
        return True
    peak, total = int(getattr(rate, "peak_memory_bytes", 0) or 0), int(
        getattr(rate, "memory_total_bytes", 0) or 0)
    if peak <= 0 or total <= 0:
        return False
    return peak * c / DEFAULT_CONTEXT <= MEMORY_HEADROOM * total


def apply_levers(contract, levers: LegLevers):
    """The leg contract with the levers folded in: ``lane_b_lr_scale``,
    ``curriculum`` and, on a rev6.20 EMA leg, the ONE chosen EMA width
    (``power_ema_sigma_rels = (ema,)``, ``power_ema_eval = 0``: only that
    EMA is computed, scored and merged; owner 2026-11-24). A leg with nothing
    to fold returns ``contract`` itself."""
    from .config.lane_b import LANE_B_EMA_SCHEDULE

    kw: dict = {}
    if levers.lr_scale != 1.0:
        kw["lane_b_lr_scale"] = float(levers.lr_scale)
    if levers.curriculum:
        kw["curriculum"] = tuple((str(n), int(h)) for n, h in levers.curriculum)
    if str(getattr(contract, "lr_schedule", "")) == LANE_B_EMA_SCHEDULE:
        # every lane B leg is a rev6.20 EMA leg: ONE chosen width (default 0.10)
        kw["power_ema_sigma_rels"] = (float(levers.ema),)
        kw["power_ema_eval"] = 0
    if levers.context_length != DEFAULT_CONTEXT:
        # the leg's training window (batch math + tokens per step change for
        # this leg only; the model config stays at 4096, see pyramid_trainer)
        kw["context_length"] = int(levers.context_length)
    if kw.get("curriculum") and str(getattr(contract, "corpus_mode", "")) != "shards":
        raise ValueError("a curriculum needs a corpus_mode='shards' leg contract")
    return replace(contract, **kw) if kw else contract


def leg_length_contract(cfg, base, spec: dict | None):
    """Step 1 of the lane B leg contract: the WB23 length of ``spec``
    (``train_hours`` / ``leg_steps`` / ``leg_guard_seconds``, steps clamped to
    the reference cap) or ``[lane_b] leg_hours`` (today's leg)."""
    from .lane_b_leg import LegLength, leg_contract, reference_steps

    spec = dict(spec or {})
    hours = float(spec.get("train_hours", 0.0) or 0.0)
    if hours > 0:
        steps = min(int(spec.get("leg_steps", 0) or 0), reference_steps(base, hours))
        return leg_contract(base, LegLength(
            train_hours=hours, steps=max(1, steps),
            guard_seconds=int(spec.get("leg_guard_seconds", 0) or 0),
            train_tokens=int(spec.get("train_tokens", 0) or 0),
            gpu_measured=spec.get("gpu_measured", True) is not False))
    return base.for_hours(
        float(cfg.lane_b.leg_hours),
        guard_factor=cfg.round.heat_guard_factor,
        guard_floor_seconds=cfg.round.heat_guard_floor_seconds)


def apply_spec_levers(lb_cfg, contract, spec: dict | None, block: int):
    """Step 3 (after rev6.20's ``lane_b_leg_contract``): the levers of
    ``spec`` effective at ``block`` folded into the contract."""
    raw = dict((spec or {}).get("levers") or {})
    levers = levers_from_dict(raw, lb_cfg) if raw else None
    return apply_levers(contract, effective_levers(lb_cfg, levers, int(block)))


def lane_b_leg_contract(cfg, base, *, block: int, spec: dict | None = None):
    """The lane B leg contract both the trainer (the receipt digest it
    expects) and the worker (what it trains) derive, in this order: the WB23
    length (:func:`leg_length_contract`), rev6.20's ``lane_b_leg_contract``
    (EMA, no cooldown), the WB27 levers (:func:`apply_spec_levers`). The
    trainer and the worker make the same three calls."""
    from .config.lane_b import lane_b_leg_contract as ema_leg

    contract = leg_length_contract(cfg, base, spec)
    contract = ema_leg(contract)
    return apply_spec_levers(cfg.lane_b, contract, spec, int(block))
