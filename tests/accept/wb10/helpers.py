"""Shared spec and helpers for the WB10 acceptance tests (``[lane_b]`` config).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB10 row and the
lane B track preamble (§3.0), §12b.1-§12b.6, §14.2 and the §17.4 register
(D35, D38-D42, D42b), plus the accepted DEC-CA-0071 amendment rev4. Every
check that can fail is an ``assert``, so a missing behaviour on main fails
with an ``AssertionError`` (never an attribute error on a dataclass
instance). Nothing from the PATHS modules is imported at module level.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SHIPPED = ("chain.toml", "chain.testnet.toml")

# Era length is 3600 blocks on mainnet (900 x 4) and 600 on testnet (150 x 4).
ERA_MAIN = 3600
EFFECTIVE = 2698 * ERA_MAIN            # 9_712_800: overlay effective_block (era start)
SETTLEMENT = 900
PROBE_BLOCKS = (0, 8_190_000 + SETTLEMENT, EFFECTIVE, EFFECTIVE + SETTLEMENT)


def hashset_digest(tag: str) -> str:
    """A well-formed ``sha256:<64 hex>`` pin for ``contamination_hashset``."""
    return "sha256:" + hashlib.sha256(f"cascade wb10 contamination {tag}".encode()).hexdigest()


HASHSET_A = hashset_digest("hashset-v1")
HASHSET_B = hashset_digest("hashset-v2")

# Defaults (§12b, §17.4, DEC-CA-0071 amendment rev4). ``None`` = derived;
# funded_pod_skus defaults to the same file's [round] funded_pod_skus (D42:
# "separate from [round]", starting at today's lane A list).
WB10_DEFAULTS: dict[str, object] = {
    "suite_targets": False,                                # DEC-CA-0071 §1: defaults to false
    "contamination_hashset": "",                           # no hash set pinned until WB11
    "contamination_min_hits": {"exact": 1, "normalised": 3},  # §12b.1 recommended
    "suite_target_max_repo_mb": 8,                         # D38 (a)
    "min_experts_frac": 0.03125,                           # D39
    "max_experts_frac": 0.1484375,                         # D39
    "router_rows": "frozen",                               # D40 (a)
    "traffic_steal_bound": 0.02,                           # §12b.2 recommended
    "open_rule": "cogate",                                 # §12b.4: default "cogate" (unarmed digests)
    "sentinel_target_rule": "defer",                       # D35 (a)
    "sentinel_max_defer_eras": 4,                          # §12b.5 recommended
    "funded_pod_skus": None,                               # = [round] funded_pod_skus
    # owner 2026-11-24 (plan rev6.21; tests-correction): 12 h legs, live from
    # [lane_b] miner_levers_from_block (6 h / 27000 s still bind before it)
    "max_leg_hours": 12.0,                                 # D42, rev6.21
    "max_leg_seconds": 54_000,                             # §12b.6: 12 h x 1.25
    "guard_factor": 1.25,                                  # §12b.6 recommended
}
WB10_KEYS = tuple(WB10_DEFAULTS)
RELEASE_ONLY = ("contamination_hashset",)
# WB22 (D42; owner 2026-11-23) owns the SHIPPED VALUE of these keys: lane B's
# own GPU list may differ from [round] funded_pod_skus in a shipped file (that
# is its purpose, "separate from [round]"). The PARSER default (key absent)
# stays the same file's [round] list, pinned by test_wb10_key_default and
# test_wb10_funded_pod_skus_default_is_lane_a_list_and_separate. The key is
# trainer/intake-side (no consensus digest reads it), so the drop-when-default
# fingerprints still hold with any shipped value.
SHIPPED_VALUE_OWNED_ELSEWHERE = ("funded_pod_skus",)
# One legal value under the decided options: the mutation is the refusal of
# other values (test_wb10_validation.py).
SINGLE_VALUED = ("router_rows", "sentinel_target_rule")

# suite_targets = true is set with a pinned hash set (LB1: the contamination
# check is a precondition for arming suite targets).
SUITE_CO = {"contamination_hashset": HASHSET_A}

# Per key: (value A, value B, co-keys). A != B, so an unread key fails.
MUTATIONS: dict[str, tuple[object, object, dict]] = {
    "suite_targets": (True, False, SUITE_CO),
    "contamination_hashset": (HASHSET_A, HASHSET_B, {}),
    "contamination_min_hits": ({"exact": 1, "normalised": 3},
                               {"exact": 2, "normalised": 5}, {}),
    "suite_target_max_repo_mb": (4, 16, {}),
    "min_experts_frac": (0.0625, 0.015625, {}),
    "max_experts_frac": (0.25, 0.125, {}),
    "traffic_steal_bound": (0.01, 0.05, {}),
    "open_rule": ("tokens_king_check", "tokens", {}),
    "sentinel_max_defer_eras": (2, 6, {}),
    "funded_pod_skus": (["8xH100"], ["8xH100", "8xH200"], {}),
    "max_leg_hours": (3.0, 4.5, {}),
    "max_leg_seconds": (14_400, 21_600, {}),
    "guard_factor": (1.5, 1.1, {}),
}

# Refused values per key (type and range).
REFUSED: dict[str, list[object]] = {
    "suite_targets": [1, 0, "true"],
    "contamination_hashset": ["sha256:" + "a1" * 31 + "a",        # 63 hex
                              "sha256:" + "g" * 64,               # not hex
                              "md5:" + HASHSET_A.split(":", 1)[1],
                              123, True],
    "contamination_min_hits": [{"exact": 0, "normalised": 3},
                               {"exact": 1, "normalised": 0},
                               {"exact": -1, "normalised": 3},
                               {"exact": 1.5, "normalised": 3},
                               {"exact": True, "normalised": 3},
                               {"exact": 1, "normalised": 3, "fuzzy": 2},
                               {}, 3, "1"],
    "suite_target_max_repo_mb": [0, -1, True, "8", 8.5],
    "min_experts_frac": [0.0, -0.1, 1.5, True, "0.03125"],
    "max_experts_frac": [0.0, -0.1, 1.5, True, "0.1484375"],
    "router_rows": ["", "router", 0, True, ["frozen"]],
    "traffic_steal_bound": [-0.01, 1.5, True, "0.02"],
    "open_rule": ["", "king_check", "tokens+king_check", 1, True],
    "sentinel_target_rule": ["", "skip", 1, True],
    "sentinel_max_defer_eras": [-1, True, 2.5, "4"],
    "funded_pod_skus": [[], "8xH100", [""], ["8xH100", "8xH100"], [1]],
    "max_leg_hours": [0.0, 0.25, -1.0, True, "6", 6.2],
    "max_leg_seconds": [0, -1, True, "27000", 27_000.5],
    "guard_factor": [0.0, -1.0, 0.9, True, "1.25"],
}
# Values that must load and be carried as written (range edges included).
ACCEPTED: dict[str, list[object]] = {
    "suite_targets": [True, False],
    "contamination_hashset": [HASHSET_A],
    "suite_target_max_repo_mb": [1, 8],
    "min_experts_frac": [0.03125, 0.1484375],
    "max_experts_frac": [0.1484375, 1.0],
    "router_rows": ["frozen"],
    "traffic_steal_bound": [0.02, 1.0],
    "open_rule": ["tokens_king_check", "tokens", "cogate"],
    "sentinel_target_rule": ["defer"],
    "sentinel_max_defer_eras": [1, 4, 12],
    "funded_pod_skus": [["8xH100"], ["8xH200", "8xH100", "4xH200", "4xH100"]],
    "max_leg_hours": [0.5, 3.0, 6.0],
    "max_leg_seconds": [1800, 27_000],
    "guard_factor": [1.0, 1.25, 2.0],
}


# ── loading ──────────────────────────────────────────────────────────────────


@cache
def _raw_text(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def shipped_raw(name: str) -> dict:
    """A fresh copy of a shipped config's tomllib table."""
    return tomllib.loads(_raw_text(name))


def round_skus(name: str) -> tuple[str, ...]:
    """The shipped file's [round] funded_pod_skus (lane A's GPU list)."""
    return tuple(shipped_raw(name).get("round", {}).get("funded_pod_skus", ()))


def generator_max_repo_mb(name: str) -> int:
    return int(shipped_raw(name).get("generator", {}).get("max_repo_mb"))


def default_of(name: str, key: str):
    """The plan default of ``key`` for shipped file ``name``."""
    if key == "funded_pod_skus":
        return round_skus(name)
    return WB10_DEFAULTS[key]


def raw_with(name: str, lane_b: dict | None = None, *, drop: tuple[str, ...] = (),
             base: dict | None = None) -> dict:
    """The shipped table (or ``base``) with ``[lane_b]`` keys overwritten
    (``lane_b``; a ``None`` value removes the key) or removed (``drop``)."""
    raw = copy.deepcopy(base) if base is not None else shipped_raw(name)
    lb = dict(raw.get("lane_b") or {})
    for k in drop:
        lb.pop(k, None)
    for k, v in (lane_b or {}).items():
        if v is None:
            lb.pop(k, None)
        else:
            lb[k] = copy.deepcopy(v)
    raw["lane_b"] = lb
    return raw


def parse(raw: dict):
    from cascade.shared.config import parse_chain_config

    return parse_chain_config(raw)


def load_ok(raw: dict, why: str = ""):
    """Parse ``raw``; a refusal is an assertion failure."""
    try:
        return parse(raw)
    except Exception as e:     # noqa: BLE001 - every refusal is the failure under test
        raise AssertionError(f"loader refused a valid [lane_b] table{why}: "
                             f"{type(e).__name__}: {e}") from None


def refused(raw: dict) -> bool:
    try:
        parse(raw)
    except (ValueError, TypeError, KeyError):
        return True
    return False


def field(lb, key: str):
    assert hasattr(lb, key), f"LaneBConfig has no field {key!r} (the [lane_b] key is not parsed)"
    return getattr(lb, key)


def norm(value):
    """Lists and tuples compare equal; a mapping (or a sequence of pairs, for
    a table-valued key) becomes a dict of normalised values."""
    if isinstance(value, dict):
        return {str(k): norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(norm(v) for v in value)
    return value


def as_table(value) -> dict:
    """A table-valued key (``contamination_min_hits``) as a plain dict."""
    try:
        return {str(k): v for k, v in dict(value).items()}
    except (TypeError, ValueError) as e:
        raise AssertionError(f"not a table / mapping: {value!r}") from e


def same_value(key: str, got, want) -> bool:
    if key == "contamination_min_hits":
        return as_table(got) == as_table(want)
    return norm(got) == norm(want)


def comparable(cfg):
    """The parsed config without its raw table."""
    return replace(cfg, raw={})


# ── consensus fingerprints (invariant 1, lane B) ─────────────────────────────


def _digest(obj) -> str:
    from cascade.shared.direct_moe import digest

    return digest(json.loads(json.dumps(obj, sort_keys=True, default=str)))


def lane_a_contract_digests(cfg) -> dict:
    from cascade.shared.manifest import contract_digest, numerics_contract_digest

    out = {
        "contract_digest": contract_digest(cfg.training),
        "numerics_contract_digest": numerics_contract_digest(cfg.training),
    }
    v2c = getattr(cfg.training, "v2_contract", None)
    if v2c is not None:
        out["v2_contract_digest"] = contract_digest(v2c)
        out["v2_numerics_contract_digest"] = numerics_contract_digest(v2c)
    return out


def consensus_fingerprint(cfg) -> dict:
    """Every contract / numerics / koth / lane B consensus digest a default
    [lane_b] key must never move (drop-when-default)."""
    from cascade.shared.manifest import contract_digest

    out = lane_a_contract_digests(cfg)
    # today's lane B leg contract derivation (worker.py: for_hours(leg_hours)
    # with the [round] heat guard knobs)
    leg_base = getattr(cfg.training, "v2_contract", None) or cfg.training
    leg = leg_base.for_hours(float(cfg.lane_b.leg_hours),
                             guard_factor=cfg.round.heat_guard_factor,
                             guard_floor_seconds=cfg.round.heat_guard_floor_seconds)
    out["lane_b_leg_contract_digest"] = contract_digest(leg)
    out["lane_b_verdict_params"] = _digest(asdict(cfg.lane_b.verdict_params()))
    out["lane_b_targets"] = list(cfg.lane_b.targets)
    for block in PROBE_BLOCKS:
        out[f"koth_params@{block}"] = _digest(asdict(cfg.koth_params(block=block)))
        out[f"lane_b_active@{block}"] = cfg.lane_b.active(block)
    return out


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
    """validate_overlay on ``cfg``; a refusal is an assertion failure."""
    from cascade.shared import netconfig as nc

    body = overlay_body(cfg, overrides, effective_block)
    try:
        return nc.validate_overlay(cfg, body)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"overlay {overrides!r} refused by validate_overlay: "
                             f"{type(e).__name__}: {e}") from None


def overrides_refusal(overrides: dict) -> str | None:
    """The NetConfigError text check_overrides raises, or None if accepted."""
    from cascade.shared import netconfig as nc

    try:
        nc.check_overrides(overrides)
    except nc.NetConfigError as e:
        return str(e)
    return None


# ── TOML round-trip ──────────────────────────────────────────────────────────


def toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(str(k))} = {toml_value(x)}"
                                for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise AssertionError(f"[lane_b] value {v!r} ({type(v).__name__}) has no TOML form")


def dump_wb10(lb, keys: tuple[str, ...] = WB10_KEYS) -> dict[str, str]:
    """``key -> TOML value text`` read back off a parsed LaneBConfig."""
    out = {}
    for k in keys:
        v = field(lb, k)
        if k == "contamination_min_hits":
            v = as_table(v)
        out[k] = toml_value(v)
    return out


_TABLE_HEADER = re.compile(r"^\[")


def splice_lane_b(name: str, lines_by_key: dict[str, str]) -> str:
    """The shipped file's text with each ``key = ...`` line of ``[lane_b]``
    replaced (or appended) by ``key = <value text>``."""
    lines = _raw_text(name).splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "[lane_b]")
    end = next((i for i in range(start + 1, len(lines)) if _TABLE_HEADER.match(lines[i])),
               len(lines))
    key_re = re.compile(r"^\s*(" + "|".join(map(re.escape, lines_by_key)) + r")\s*=")
    body = [ln for ln in lines[start + 1:end] if not key_re.match(ln)]
    new = [f"{k} = {v}\n" for k, v in lines_by_key.items()]
    return "".join(lines[:start + 1] + new + body + lines[end:])


def load_text(text: str, tmp_path: Path, name: str = "chain.toml"):
    from cascade.shared.config import load_chain_config

    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    try:
        return load_chain_config(p)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"loader refused the spliced [lane_b] table: "
                             f"{type(e).__name__}: {e}") from None


# ── scripts/laned tools ──────────────────────────────────────────────────────


def run_unread_keys(*configs: str | Path) -> subprocess.CompletedProcess:
    args = [sys.executable, "-P", str(REPO / "scripts" / "laned" / "unread_keys.py"),
            "--root", str(REPO), "--source", "cascade/shared/config"]
    for c in configs:
        args += ["--config", str(c)]
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
