"""``[lane_b]`` keys of the lane B track (LANE_D_PLAN §12b, §14.2; DEC-CA-0071).

Suite targets (§12b.1), miner-chosen experts (D39, D40), the opening rule
(§12b.4, D41), the sentinel interaction (§12b.5, D35) and the D42 leg cost
(§12b.6). :func:`parse_lane_b_track` reads and checks them; ``parse_lane_b``
(``v2.py``) folds the result into :class:`~.v2.LaneBConfig`.

Every key is overlay-settable (``ALLOWED["lane_b"] = None``) except
``contamination_hashset``, a release-only data pin (``netconfig``). At its
default no key is read by any consensus surface, so the shipped table moves
no contract, numerics, koth, leg-contract or verdict digest (invariant 1):
the defaults are the unarmed behaviour (``suite_targets = false``, no hash
set pinned, ``open_rule = "cogate"``).
"""

from __future__ import annotations

import math
import re

# §12b.4 / D41: the opening rule. "cogate" (today's routing co-gate) is the
# shipped default; "tokens_king_check" (D41 a) and "tokens" (D41 b) arrive with
# the capability release and are switched on by overlay.
LANE_B_OPEN_RULES = ("tokens_king_check", "tokens", "cogate")
# D40 (a): router rows stay frozen in an expert-only leg.
LANE_B_ROUTER_ROWS = ("frozen",)
# D35 (a): a suite-target entry whose target is the era's sentinel defers.
LANE_B_SENTINEL_TARGET_RULES = ("defer",)
# §12b.1: the sub-window hit counts that refuse a suite-target corpus.
LANE_B_CONTAMINATION_HIT_KINDS = ("exact", "normalised")

LANE_B_TRACK_DEFAULTS: dict[str, object] = {
    "suite_targets": False,
    "contamination_hashset": "",
    "contamination_min_hits": (("exact", 1), ("normalised", 3)),
    "suite_target_max_repo_mb": 8,
    "min_experts_frac": 0.03125,       # 8 / 256
    "max_experts_frac": 0.1484375,     # 38 / 256
    "router_rows": "frozen",
    "traffic_steal_bound": 0.02,
    "open_rule": "cogate",
    "sentinel_target_rule": "defer",
    "sentinel_max_defer_eras": 4,
    "funded_pod_skus": (),             # loader: the same file's [round] funded_pod_skus
    "max_leg_hours": 6.0,
    "max_leg_seconds": 27_000,         # 6 h x 1.25
    "guard_factor": 1.25,
    "ema_no_cooldown_from_block": 0,   # 0 = never (rev6.20; loader: era boundary)
}

# Lane B EMA, no cooldown (plan rev6.20, owner 2026-11-24: "let's just do EMA,
# no cooldown"). From ``[lane_b] ema_no_cooldown_from_block`` (0 = never) a
# lane B leg's contract carries this ``lr_schedule`` (warmup as WSD, then
# constant LR to the last step, no decay tail) and its scored + merged
# ``weights.safetensors`` is the image's sigma_rel 0.10 power EMA (the owner's
# standing scored width, ``pyramid_trainer.POWER_EMA_SIGMA_RELS[POWER_EMA_EVAL]``).
# The expert manifest entry records ``scored_weights``; judges re-derive it
# from the entry's ``train_block``. Lane A never takes this schedule.
LANE_B_EMA_SCHEDULE = "warmup_constant_ema"
LANE_B_EMA_SIGMA_REL = 0.10
LANE_B_SCORED_EMA = "ema010"
# "" = the raw cooled endpoint (W-EMA; dropped from the manifest body)
LANE_B_SCORED_WEIGHTS = ("", LANE_B_SCORED_EMA)


def lane_b_scored_weights(lane_b_cfg, block: int | None) -> str:
    """The ``scored_weights`` a lane B leg dispatched at ``block`` must carry:
    ``"ema010"`` from ``ema_no_cooldown_from_block`` (> 0), else ``""``."""
    n = int(getattr(lane_b_cfg, "ema_no_cooldown_from_block", 0) or 0)
    return LANE_B_SCORED_EMA if n > 0 and block is not None and int(block) >= n else ""


def lane_b_leg_schedule(lane_b_cfg, contract, block: int | None):
    """The lane B leg contract at dispatch ``block``: ``contract`` unchanged
    before the key (or while unset), else the same contract with
    ``lr_schedule = LANE_B_EMA_SCHEDULE``. The ONE rule the trainer (expected
    receipt digest) and the worker (the contract it trains) both apply."""
    if not lane_b_scored_weights(lane_b_cfg, block):
        return contract
    from dataclasses import replace

    if str(getattr(contract, "base_arch", "") or "") != "cascade_pyramid":
        raise ValueError("[lane_b] ema_no_cooldown_from_block: the lane B EMA schedule "
                         "needs a cascade_pyramid leg contract; got "
                         f"{getattr(contract, 'base_arch', '')!r}")
    return replace(contract, lr_schedule=LANE_B_EMA_SCHEDULE)


_HASHSET_RE = re.compile(r"sha256:[0-9a-f]{64}")
# D42: train_hours moves in 0.5 h steps, so the cap sits on the same grid.
_LEG_HOURS_STEP = 0.5


def _bool(key: str, v: object) -> bool:
    if not isinstance(v, bool):
        raise ValueError(f"[lane_b] {key} must be true or false; got {v!r}")
    return v


def _int(key: str, v: object, *, lo: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"[lane_b] {key} must be an integer; got {v!r}")
    if v < lo:
        raise ValueError(f"[lane_b] {key} must be >= {lo}; got {v}")
    return int(v)


def _float(key: str, v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"[lane_b] {key} must be a number; got {v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"[lane_b] {key} must be finite; got {v!r}")
    return f


def _choice(key: str, v: object, allowed: tuple[str, ...]) -> str:
    if not isinstance(v, str) or v not in allowed:
        raise ValueError(f"[lane_b] {key} must be one of {allowed}; got {v!r}")
    return v


def _hashset(v: object) -> str:
    if not isinstance(v, str) or (v != "" and not _HASHSET_RE.fullmatch(v)):
        raise ValueError('[lane_b] contamination_hashset must be "" (none pinned) or '
                         f"sha256:<64 lowercase hex>; got {v!r}")
    return v


def _min_hits(v: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(v, dict) or set(v) != set(LANE_B_CONTAMINATION_HIT_KINDS):
        raise ValueError("[lane_b] contamination_min_hits must be a table "
                         f"{{exact = <int>, normalised = <int>}}; got {v!r}")
    return tuple((k, _int(f"contamination_min_hits.{k}", v[k], lo=1))
                 for k in LANE_B_CONTAMINATION_HIT_KINDS)


def _skus(v: object) -> tuple[str, ...]:
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f"[lane_b] funded_pod_skus must be a non-empty list of SKU "
                         f"strings; got {v!r}")
    if any(not isinstance(s, str) or not s.strip() for s in v):
        raise ValueError(f"[lane_b] funded_pod_skus entries must be non-empty strings; "
                         f"got {v!r}")
    if len(set(v)) != len(v):
        raise ValueError(f"[lane_b] funded_pod_skus must be unique; got {v!r}")
    return tuple(v)


def parse_lane_b_track(lb: dict, *,
                       round_funded_pod_skus: tuple[str, ...] | None = None,
                       generator_max_repo_mb: int | None = None) -> dict[str, object]:
    """The lane B track keys of ``lb`` as ``LaneBConfig`` field values,
    type- and range-checked (``ValueError`` on a refusal).

    ``round_funded_pod_skus`` is the default of ``funded_pod_skus`` (D42: lane
    B's list starts as today's lane A list); ``None`` leaves it empty.
    ``generator_max_repo_mb`` bounds ``suite_target_max_repo_mb`` (D38: a
    stricter cap, never above ``[generator] max_repo_mb``)."""
    d = LANE_B_TRACK_DEFAULTS
    out: dict[str, object] = {
        "suite_targets": _bool("suite_targets", lb.get("suite_targets", d["suite_targets"])),
        "contamination_hashset": _hashset(
            lb.get("contamination_hashset", d["contamination_hashset"])),
        "contamination_min_hits": (_min_hits(lb["contamination_min_hits"])
                                   if "contamination_min_hits" in lb
                                   else d["contamination_min_hits"]),
        "suite_target_max_repo_mb": _int(
            "suite_target_max_repo_mb",
            lb.get("suite_target_max_repo_mb", d["suite_target_max_repo_mb"]), lo=1),
        "min_experts_frac": _float("min_experts_frac",
                                   lb.get("min_experts_frac", d["min_experts_frac"])),
        "max_experts_frac": _float("max_experts_frac",
                                   lb.get("max_experts_frac", d["max_experts_frac"])),
        "router_rows": _choice("router_rows", lb.get("router_rows", d["router_rows"]),
                               LANE_B_ROUTER_ROWS),
        "traffic_steal_bound": _float("traffic_steal_bound",
                                      lb.get("traffic_steal_bound", d["traffic_steal_bound"])),
        "open_rule": _choice("open_rule", lb.get("open_rule", d["open_rule"]),
                             LANE_B_OPEN_RULES),
        "sentinel_target_rule": _choice(
            "sentinel_target_rule", lb.get("sentinel_target_rule", d["sentinel_target_rule"]),
            LANE_B_SENTINEL_TARGET_RULES),
        "sentinel_max_defer_eras": _int(
            "sentinel_max_defer_eras",
            lb.get("sentinel_max_defer_eras", d["sentinel_max_defer_eras"]), lo=1),
        "funded_pod_skus": (_skus(lb["funded_pod_skus"]) if "funded_pod_skus" in lb
                            else tuple(round_funded_pod_skus or ())),
        "max_leg_hours": _float("max_leg_hours", lb.get("max_leg_hours", d["max_leg_hours"])),
        "max_leg_seconds": _int("max_leg_seconds",
                                lb.get("max_leg_seconds", d["max_leg_seconds"]), lo=1),
        "guard_factor": _float("guard_factor", lb.get("guard_factor", d["guard_factor"])),
        "ema_no_cooldown_from_block": _int(
            "ema_no_cooldown_from_block",
            lb.get("ema_no_cooldown_from_block", d["ema_no_cooldown_from_block"]), lo=0),
    }
    if out["suite_targets"] and not out["contamination_hashset"]:
        raise ValueError("[lane_b] suite_targets = true needs contamination_hashset pinned "
                         "(the contamination check gates every suite-target entry)")
    cap = out["suite_target_max_repo_mb"]
    if generator_max_repo_mb is not None and cap > int(generator_max_repo_mb):
        raise ValueError(f"[lane_b] suite_target_max_repo_mb={cap} must be <= [generator] "
                         f"max_repo_mb={int(generator_max_repo_mb)}")
    lo, hi = out["min_experts_frac"], out["max_experts_frac"]
    for key, f in (("min_experts_frac", lo), ("max_experts_frac", hi)):
        if not 0.0 < f <= 1.0:
            raise ValueError(f"[lane_b] {key} must be in (0, 1]; got {f}")
    if lo > hi:
        raise ValueError(f"[lane_b] min_experts_frac={lo} must be <= max_experts_frac={hi}")
    if not 0.0 <= out["traffic_steal_bound"] <= 1.0:
        raise ValueError(f"[lane_b] traffic_steal_bound must be in [0, 1]; "
                         f"got {out['traffic_steal_bound']}")
    mlh = out["max_leg_hours"]
    steps = mlh / _LEG_HOURS_STEP
    if mlh < _LEG_HOURS_STEP or steps != math.floor(steps):
        raise ValueError(f"[lane_b] max_leg_hours must be >= {_LEG_HOURS_STEP} on the "
                         f"{_LEG_HOURS_STEP} h grid; got {mlh}")
    if out["guard_factor"] < 1.0:
        raise ValueError(f"[lane_b] guard_factor must be >= 1.0 (a factor below 1 truncates "
                         f"every leg); got {out['guard_factor']}")
    return out
