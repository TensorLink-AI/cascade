"""``[round]`` money-safety knobs (2026-11-02): parsed and validated in one place.

Trainer-only policy — nothing here reaches a manifest, a receipt or anything a
validator computes. Every knob has a loader parse here AND a round-trip test
(tests/unit/test_money_safety.py): a dataclass field without the parse makes a
TOML arming silently no-op (the defect class that bit this repo three times).
"""

from __future__ import annotations

import math

# name -> (default, kind). kind: "count" (int >= 0), "seconds" (int >= 0),
# "usd" (finite float >= 0), "factor" (finite float >= 1), "gb" (finite >= 0).
MONEY_KNOBS: dict[str, tuple[object, str]] = {
    "operator_fault_cap": (3, "count"),
    "operator_breaker_cooldown_seconds": (3600, "seconds"),
    "operator_cost_safety_factor": (1.25, "factor"),
    "operator_balance_reserve_usd": (50.0, "usd"),
    "operator_daily_spend_cap_usd": (200.0, "usd"),
    "payer_projected_rate_per_hour": (0.0, "usd"),
    "expected_checkpoint_gb": (15.0, "gb"),
    "harvest_disk_retries": (3, "count"),
    "harvest_disk_retry_seconds": (60, "seconds"),
}


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"[round] {name}={value!r} invalid; must be a finite number")
    return float(value)


def parse_money_safety(r: dict) -> dict:
    """The ``[round]`` money-safety knobs from a raw table; bad values fail loud."""
    out: dict = {}
    for name, (default, kind) in MONEY_KNOBS.items():
        value = r.get(name, default)
        if kind in ("count", "seconds"):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"[round] {name}={value!r} invalid; must be a non-negative "
                                 "integer (0 = off)")
            out[name] = int(value)
            continue
        v = _number(name, value)
        if kind == "factor" and v < 1.0:
            raise ValueError(f"[round] {name}={value!r} invalid; must be >= 1.0")
        if v < 0:
            raise ValueError(f"[round] {name}={value!r} invalid; must be >= 0")
        out[name] = v
    return out


# ``[validator]`` eval checkpoint cache retention (the 401 GB cache of 2026-11-02).
VALIDATOR_CACHE_KNOBS: dict[str, tuple[object, str]] = {
    "eval_ckpt_cache_dir": ("_eval_ckpts", "path"),
    "eval_ckpt_cache_max_gb": (120.0, "gb"),
    "eval_ckpt_cache_min_free_gb": (60.0, "gb"),
    "eval_ckpt_cache_grace_seconds": (3600, "seconds"),
}


def parse_validator_cache(v: dict) -> dict:
    """The ``[validator]`` eval-cache retention knobs; bad values fail loud.
    ``eval_ckpt_cache_max_gb = 0`` disables the byte budget (the free-space
    floor still applies); ``eval_ckpt_cache_min_free_gb = 0`` disables the floor."""
    out: dict = {}
    for name, (default, kind) in VALIDATOR_CACHE_KNOBS.items():
        value = v.get(name, default)
        if kind == "path":
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"[validator] {name} must be a non-empty path string")
            out[name] = value.strip()
        elif kind == "seconds":
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"[validator] {name}={value!r} invalid; must be a "
                                 "non-negative integer")
            out[name] = int(value)
        else:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"[validator] {name}={value!r} invalid; must be a finite "
                                 "number >= 0")
            out[name] = float(value)
    return out
