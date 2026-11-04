"""``[round]`` money-safety knobs (2026-11-02): parsed and validated in one place.

Trainer-only policy — nothing here reaches a manifest, a receipt or anything a
validator computes. Every knob has a loader parse here AND a round-trip test
(tests/unit/test_money_safety.py): a dataclass field without the parse makes a
TOML arming silently no-op (the defect class that bit this repo three times).
"""

from __future__ import annotations

import math

# name -> (default, kind). kind: "count" (int >= 0), "seconds" (int >= 0),
# "usd" (finite float >= 0), "factor" (finite float >= 1), "gb" (finite >= 0),
# "hours" (finite >= 0), "fraction" (finite in [0, 1]), "rates" (a TOML table
# ``{name = usd_per_hour}`` → a sorted tuple of (name, rate) pairs), "scope"
# (one of :data:`DAILY_CAP_SCOPES`), "paths" (a list of file paths).
MONEY_KNOBS: dict[str, tuple[object, str]] = {
    "operator_fault_cap": (3, "count"),
    "operator_breaker_cooldown_seconds": (3600, "seconds"),
    "operator_cost_safety_factor": (1.25, "factor"),
    "operator_balance_reserve_usd": (50.0, "usd"),
    "operator_daily_spend_cap_usd": (200.0, "usd"),
    "payer_projected_rate_per_hour": (0.0, "usd"),
    "expected_checkpoint_gb": (15.0, "gb"),
    "harvest_disk_retries": (2, "count"),
    "harvest_disk_retry_seconds": (30, "seconds"),
    # money safety II (2026-11-03 audit): a kept pod's hard cap (3 h); the
    # priced keep (harvest_keep_cost_fraction) usually ends it sooner.
    "harvest_keep_max_seconds": (10800, "seconds"),
    "harvest_keep_cost_fraction": (0.5, "fraction"),
    "miner_waste_cap_usd": (30.0, "usd"),
    "operator_pause_resume_seconds": (21600, "seconds"),
    "box_fault_window_seconds": (7200, "seconds"),
    # ── money safety II (2026-11-03 money-leak audit) ──
    "operator_runway_alert_hours": (6.0, "hours"),
    "operator_standing_pods_usd_per_hour": ((), "rates"),
    "operator_daily_cap_scope": ("miner_legs", "scope"),
    "miner_waste_alert_usd": (15.0, "usd"),
    "king_pod_ttl_margin_seconds": (0, "seconds"),
    "king_idle_reap_checks": (2, "count"),
    "king_idle_check_seconds": (900, "seconds"),
    "operator_inventory_interval_seconds": (3600, "seconds"),
    "operator_inventory_allow": ((), "names"),
    "operator_inventory_host_files": ((), "paths"),
    # second audit (B): an ssh checkpoint harvest is bounded at
    # expected_checkpoint_gb x this (0 = unbounded)
    "harvest_max_bytes_factor": (1.2, "gb"),
    # review M2: the expected checkpoint per model size (arch preset → GB);
    # expected_checkpoint_gb is the floor
    "expected_checkpoint_gb_by_preset": ((), "rates"),
    # audit M: the operator pool is switched on explicitly, never by a cap
    "ops_operator_pool_enabled": (False, "bool"),
    # audit D: our own validator's hotkey; its rejected receipts of OUR
    # settlements raise one critical alert ("" = off)
    "receipt_watch_validator_hotkey": ("", "text"),
    # owner 2026-11-04: an our-side lemon loss below this reruns on the
    # miner's own key instead of an approval-gated operator make-good (0 = off)
    "make_good_rerun_below_usd": (20.0, "usd"),
}

# ``operator_daily_cap_scope``: which booked operator spend counts toward the
# daily cap. "miner_legs" (default) = operator-billed miner legs only (the #417
# behaviour); "all" = + CPU VMs, lemons and other operator pods, but NOT the
# king pod / backups; "all_incl_king" = everything (review H2: a king era's
# ~$240-300 booking would otherwise refuse every make-good until midnight).
# The runway guard covers total spend either way.
DAILY_CAP_SCOPES = ("miner_legs", "all", "all_incl_king")


def _whole(value: object) -> int | None:
    """An integral number as int (``3600`` and ``3600.0`` alike — a float
    spelling must never fail the whole config load; review F12), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    return None


def _number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"[round] {name}={value!r} invalid; must be a finite number")
    return float(value)


def _rates(name: str, value: object) -> tuple[tuple[str, float], ...]:
    """``{name = usd_per_hour}`` (or an already-parsed pair tuple) → sorted pairs."""
    if isinstance(value, (tuple, list)) and all(
            isinstance(p, (tuple, list)) and len(p) == 2 for p in value):
        value = dict(value)
    if not isinstance(value, dict):
        raise ValueError(f"[round] {name} must be a table of name = USD per hour")
    out = []
    for k, v in value.items():
        if not isinstance(k, str) or not k.strip():
            raise ValueError(f"[round] {name}: pod names must be non-empty strings")
        rate = _number(f"{name}.{k}", v)
        if rate < 0:
            raise ValueError(f"[round] {name}.{k}={v!r} invalid; must be >= 0")
        out.append((k.strip(), rate))
    return tuple(sorted(out))


def _strings(name: str, value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)) or not all(
            isinstance(x, str) and x.strip() for x in value):
        raise ValueError(f"[round] {name} must be a list of non-empty strings")
    return tuple(x.strip() for x in value)


def parse_money_safety(r: dict) -> dict:
    """The ``[round]`` money-safety knobs from a raw table; bad values fail loud."""
    out: dict = {}
    for name, (default, kind) in MONEY_KNOBS.items():
        value = r.get(name, default)
        if kind == "rates":
            out[name] = _rates(name, value)
            continue
        if kind == "scope":
            if value not in DAILY_CAP_SCOPES:
                raise ValueError(f"[round] {name}={value!r} invalid; must be one of "
                                 f"{', '.join(DAILY_CAP_SCOPES)}")
            out[name] = value
            continue
        if kind == "text":
            if not isinstance(value, str):
                raise ValueError(f"[round] {name}={value!r} invalid; must be a string")
            out[name] = value.strip()
            continue
        if kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"[round] {name}={value!r} invalid; must be true or false")
            out[name] = value
            continue
        if kind in ("names", "paths"):
            out[name] = _strings(name, value)
            continue
        if kind in ("count", "seconds"):
            whole = _whole(value)
            if whole is None or whole < 0:
                raise ValueError(f"[round] {name}={value!r} invalid; must be a non-negative "
                                 "integer (0 = off)")
            out[name] = whole
            continue
        v = _number(name, value)
        if kind == "fraction" and not 0.0 <= v <= 1.0:
            raise ValueError(f"[round] {name}={value!r} invalid; must be in [0, 1]")
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
            whole = _whole(value)
            if whole is None or whole < 0:
                raise ValueError(f"[validator] {name}={value!r} invalid; must be a "
                                 "non-negative integer")
            out[name] = whole
        else:
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"[validator] {name}={value!r} invalid; must be a finite "
                                 "number >= 0")
            out[name] = float(value)
    return out
