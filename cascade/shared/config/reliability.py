"""Trainer-only recovery policy; never changes model, emission or scoring rules."""

from __future__ import annotations

import math
import re

# Flat [round] keys keep these controls compatible with signed policy overlays.
DEFAULTS = {
    "ops_max_active_per_coldkey": 0,
    "ops_gpu_failure_threshold": 0,
    "ops_gpu_failure_window_seconds": 3600,
    "ops_gpu_cooldown_seconds": 10800,
    "ops_alert_interval_seconds": 60,
    "ops_low_balance_usd": 10.0,
    "ops_oldest_wait_seconds": 7200,
    "ops_settlement_hold_seconds": 900,
    "ops_acceptance_timeout_seconds": 1200,
    "ops_clock_drift_seconds": 120,
    "ops_operator_providers": ("lium",),
    "ops_operator_daily_cap_usd": 0.0,
    "ops_operator_leg_cap_usd": 0.0,
    "ops_paid_fallback_after_seconds": 0,
    "ops_king_backup_after_seconds": 0,
    "ops_king_backup_cap_usd": 0.0,
    "ops_alert_webhook_env": "",
    "ops_continuation_interval_seconds": 0,
    "ops_continuation_max_bytes": 17179869184,
    "ops_continuation_image_digest": "",
}


def parse_reliability(raw: dict) -> dict:
    result = {}
    for name, default in DEFAULTS.items():
        value = raw.get(name, default)
        if isinstance(default, tuple):
            if (
                not isinstance(value, (tuple, list))
                or not value
                or not all(isinstance(p, str) for p in value)
                or len(set(value)) != len(value)
            ):
                raise ValueError(f"{name} must be a nonempty provider list without duplicates")
            value = tuple(value)
            if any(p not in {"lium", "shadeform"} for p in value):
                raise ValueError(f"{name}: supported providers are lium and shadeform")
        elif isinstance(default, str):
            if name == "ops_continuation_image_digest":
                if (
                    not isinstance(value, str)
                    or value
                    and not re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                ):
                    raise ValueError(
                        "continuation qualification requires an immutable image digest"
                    )
            elif not isinstance(value, str) or value and not value.isidentifier():
                raise ValueError(f"{name} must be an environment variable name")
        else:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number")
            if isinstance(default, int):
                if value != int(value):
                    raise ValueError(f"{name} must be an integer")
                value = int(value)
            else:
                value = float(value)
        result[name] = value
    if result["ops_gpu_failure_threshold"] and not all(
        result[k] > 0 for k in ("ops_gpu_failure_window_seconds", "ops_gpu_cooldown_seconds")
    ):
        raise ValueError("GPU outage detection requires positive window and cooldown")
    if (
        result["ops_paid_fallback_after_seconds"]
        or result["ops_king_backup_after_seconds"]
        or result["ops_operator_providers"] != ("lium",)
    ) and not all(
        result[k] > 0 for k in ("ops_operator_daily_cap_usd", "ops_operator_leg_cap_usd")
    ):
        raise ValueError(
            "operator fallback and replicas require explicit per-leg and daily spending caps"
        )
    pooled = result["ops_operator_providers"] != ("lium",) or any(
        result[k]
        for k in (
            "ops_operator_leg_cap_usd",
            "ops_operator_daily_cap_usd",
            "ops_paid_fallback_after_seconds",
            "ops_king_backup_after_seconds",
        )
    )
    if pooled and (
        result["ops_alert_interval_seconds"] <= 0
        or not all(
            result[k] > 0 for k in ("ops_operator_leg_cap_usd", "ops_operator_daily_cap_usd")
        )
    ):
        raise ValueError(
            "operator pool requires per-leg and daily spending caps and an enabled watchdog"
        )
    if result["ops_king_backup_after_seconds"] and result["ops_king_backup_cap_usd"] <= 0:
        raise ValueError("king backups require an explicit positive spending cap")
    if result["ops_continuation_interval_seconds"] and (
        not result["ops_continuation_image_digest"] or result["ops_continuation_max_bytes"] <= 0
    ):
        raise ValueError("continuation requires a qualified image and a positive storage bound")
    if (
        result["ops_king_backup_after_seconds"] or result["ops_continuation_interval_seconds"]
    ) and not raw.get("detached_dispatch", True):
        raise ValueError("king replicas and continuation require detached_dispatch")
    return result
