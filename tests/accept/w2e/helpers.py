"""Shared spec and helpers for the W2e acceptance tests (the rev6.10 /
rev6.11 / rev6.18 ``[lane_d]`` keys).

Copied from docs/plans/LANE_D_PLAN.md: the W2e row, rev6.10 (a) / (b),
rev6.11 (a) / (e) and rev6.18 A3 / D24 (F4: ``public_weight`` and
``window_ratio_floor`` added; the rev6.9 backtest keys and their loader
inequality deleted, owner 2026-11-24). README.md states the
contract. Every check that can fail is an ``assert`` so a missing behaviour on
main fails as an ``AssertionError``.
"""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
import tomllib
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SHIPPED = ("chain.toml", "chain.testnet.toml")

ERA_MAIN = 3600
ARM_BLOCK = 2700 * ERA_MAIN
EFFECTIVE = 2698 * ERA_MAIN
ARM = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a"}

W2E_DEFAULTS: dict[str, object] = {
    "embargo_hours": 48.0,
    "max_windows_per_settlement": 5000,
    # Owner 2026-11-23 ("first verdict ~6 h"): 0.2, was 0.5 (tests-correction).
    "min_fresh_window_share": 0.2,
    # rev6.18 A3 (owner 2026-11-23): 50% live + 50% public; ratio clamp [0.5, 2].
    "public_weight": 0.5,
    "window_ratio_floor": 0.5,
    # rev6.14 (a): the judging start j = T + eval_delay_hours.
    "eval_delay_hours": 6.0,
}
# rev6.18 D24 / A7 (owner 2026-11-24 / 2026-11-23): retired, ignored by the loader.
RETIRED = ("backtest_weight_cap", "backtest_ratio_floor", "lowfreq_freqs")
W2E_KEYS = tuple(W2E_DEFAULTS)

# (value A, value B, co-keys). Each pair differs and is legal on its own.
MUTATIONS: dict[str, tuple[object, object, dict]] = {
    "embargo_hours": (24.0, 12.0, {}),
    "max_windows_per_settlement": (2000, 8000, {}),
    "min_fresh_window_share": (0.75, 0.25, {}),
    "public_weight": (0.3, 0.7, {}),
    "window_ratio_floor": (0.25, 0.75, {}),
    "eval_delay_hours": (3.0, 12.0, {}),
}

REFUSED: dict[str, list[object]] = {
    "embargo_hours": [-1.0, -0.5, True, "48"],
    "max_windows_per_settlement": [0, -1, 999, True, 5000.5, "5000"],
    "min_fresh_window_share": [-0.1, 1.5, True, "0.5"],
    "public_weight": [-0.01, 1.0, 1.5, True, "0.5"],
    "window_ratio_floor": [0.0, 1.0, -0.1, 1.5, True, "0.5"],
    "eval_delay_hours": [-1.0, -0.5, True, "6"],
}
ACCEPTED: dict[str, list[object]] = {
    "embargo_hours": [0.0, 6.0, 48.0, 72.0],
    "max_windows_per_settlement": [1000, 5000, 20000],
    "min_fresh_window_share": [0.0, 0.5, 1.0],
    "public_weight": [0.0, 0.5, 0.99],
    "window_ratio_floor": [0.01, 0.5, 0.99],
    "eval_delay_hours": [0.0, 6.0, 24.0],
}

# Overlay values that differ from the default and from ARM.
OVERLAY_VALUE: dict[str, object] = {
    "embargo_hours": 24.0,
    "max_windows_per_settlement": 3000,
    "min_fresh_window_share": 0.75,
    "public_weight": 0.3,
    "window_ratio_floor": 0.6,
    "eval_delay_hours": 12.0,
}


@cache
def _raw_text(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def shipped_raw(name: str) -> dict:
    return tomllib.loads(_raw_text(name))


def raw_with(name: str, lane_d: dict | None = None, *, drop: tuple[str, ...] = ()) -> dict:
    """The shipped table with ``[lane_d]`` keys overwritten (a ``None`` value
    removes the key) or removed (``drop``)."""
    raw = shipped_raw(name)
    ld = dict(raw.get("lane_d") or {})
    for k in drop:
        ld.pop(k, None)
    for k, v in (lane_d or {}).items():
        if v is None:
            ld.pop(k, None)
        else:
            ld[k] = copy.deepcopy(v)
    raw["lane_d"] = ld
    return raw


def parse(raw: dict):
    from cascade.shared.config import parse_chain_config

    return parse_chain_config(raw)


def load_ok(raw: dict, why: str = ""):
    try:
        return parse(raw)
    except Exception as e:     # noqa: BLE001 - every refusal is the failure under test
        raise AssertionError(f"loader refused a valid [lane_d] table{why}: "
                             f"{type(e).__name__}: {e}") from None


def refused(raw: dict) -> bool:
    try:
        parse(raw)
    except (ValueError, TypeError, KeyError):
        return True
    return False


def field(ld, key: str):
    assert hasattr(ld, key), f"LaneDConfig has no field {key!r} (the [lane_d] key is not parsed)"
    return getattr(ld, key)


def same(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) == float(b)
    return a == b


# ── overlays ─────────────────────────────────────────────────────────────────


def overlay_body(cfg, overrides: dict, effective_block: int = EFFECTIVE):
    from cascade.shared import netconfig as nc

    try:
        return nc.next_overlay(overrides, effective_block, None,
                               network=nc.network_identity(cfg))
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"overlay {overrides!r} refused at build: "
                             f"{type(e).__name__}: {e}") from None


def validate_ok(cfg, overrides: dict, effective_block: int = EFFECTIVE):
    from cascade.shared import netconfig as nc

    body = overlay_body(cfg, overrides, effective_block)
    try:
        return nc.validate_overlay(cfg, body)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"overlay {overrides!r} refused by validate_overlay: "
                             f"{type(e).__name__}: {e}") from None


def validate_refused(cfg, overrides: dict, effective_block: int = EFFECTIVE) -> bool:
    """True when the overlay is refused at build or at adoption."""
    from cascade.shared import netconfig as nc

    try:
        body = nc.next_overlay(overrides, effective_block, None,
                               network=nc.network_identity(cfg))
        nc.validate_overlay(cfg, body)
    except Exception:     # noqa: BLE001 - any typed refusal is the outcome under test
        return True
    return False


def overrides_refusal(overrides: dict) -> str | None:
    from cascade.shared import netconfig as nc

    try:
        nc.check_overrides(overrides)
    except nc.NetConfigError as e:
        return str(e)
    return None


# ── TOML round-trip ──────────────────────────────────────────────────────────


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    raise AssertionError(f"[lane_d] value {v!r} ({type(v).__name__}) has no TOML form here")


_TABLE_HEADER = re.compile(r"^\[")


def splice_keys(name: str, values: dict) -> str:
    """The shipped file's text with ``values`` set inside ``[lane_d]`` (an
    existing line for the key is replaced, else the key is appended)."""
    lines = _raw_text(name).splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "[lane_d]")
    end = next((i for i in range(start + 1, len(lines)) if _TABLE_HEADER.match(lines[i])),
               len(lines))
    body = [ln for ln in lines[start + 1:end]
            if not any(re.match(rf"^\s*{re.escape(k)}\s*=", ln) for k in values)]
    added = [f"{k} = {_toml_value(v)}\n" for k, v in values.items()]
    return "".join(lines[:start + 1] + added + body + lines[end:])


def load_text(text: str, tmp_path: Path, name: str = "chain.toml"):
    from cascade.shared.config import load_chain_config

    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    try:
        return load_chain_config(p)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"loader refused the spliced [lane_d] table: "
                             f"{type(e).__name__}: {e}") from None


def run_unread_keys(*configs: str | Path) -> subprocess.CompletedProcess:
    args = [sys.executable, str(REPO / "scripts" / "laned" / "unread_keys.py"),
            "--root", str(REPO), "--source", "cascade/shared/config"]
    for c in configs:
        args += ["--config", str(c)]
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
