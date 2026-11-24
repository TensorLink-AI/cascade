"""Shared spec and helpers for the W2 acceptance tests (``[lane_d]`` config).

Everything here is copied from docs/plans/LANE_D_PLAN.md (rev6): the W2 row,
§1, §6 / §17.4 defaults and §14.2. Every check that can fail is an ``assert``
so a missing behaviour on main fails as an ``AssertionError`` (never as a
``ValueError`` or attribute error on a dataclass instance).
"""

from __future__ import annotations

import copy
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

# Era length is 3600 blocks on mainnet (900 x 4) and 600 on testnet (150 x 4):
# every block below is an era boundary on both grids.
ERA_MAIN = 3600
ARM_BLOCK = 2700 * ERA_MAIN            # 9_720_000: lane D from_block when armed
ARM_BLOCK_2 = 2701 * ERA_MAIN          # 9_723_600
UNTIL_BLOCK = 2724 * ERA_MAIN          # 9_806_400: kill switch end
UNTIL_BLOCK_2 = 2730 * ERA_MAIN        # 9_828_000
EFFECTIVE = 2698 * ERA_MAIN            # 9_712_800: overlay effective_block (era start)
SETTLEMENT = 900                       # one mainnet settlement (>= one testnet one)

# The activation overlay of D10 / D1: from_block, share 0.25, takes_from lane_a.
ARM = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a"}

# The new W2 keys and their plan defaults. ``None`` = the plan states no
# default; the test pins the type only (see README "defaults assumed").
W2_DEFAULTS: dict[str, object] = {
    "from_block": 0,
    "until_block": 0,
    "share": 0.0,
    "takes_from": "lane_b",
    "remainder": "burn",
    # rev6.6 (owner 2026-11-23): every variant is declarable from launch;
    # arming is computed per settlement (W5d), never configured.
    "contexts": (4096, 8192, 16384, 32768, 65536, 131072),
    "horizon": 64,
    "forward_margin_hours": 0.0,       # rev6.14 (a): 6 -> 0 (b = T)
    "min_windows": 1000,
    "min_clusters": 20,
    "min_long_series": 300,
    "d_margin": 0.01,
    # rev6.23 (owner 2026-11-24): the tenure-decaying crown bar (d_margin is its end).
    "d_margin_start": 0.03,
    "d_margin_warmup_settlements": 32,
    "alpha": 0.05,
    "bootstrap_B": 10_000,
    "gate_suites": ("gift", "boom", "time", "fev"),
    "gate_weights": None,              # equal over gate_suites (checked separately)
    "tau_public": 0.01,
    "reward_prior_kings": 2,
    # rev6.23 (owner 2026-11-24): explicit court weights replace king_decay.
    "court_weights": (0.80, 0.15, 0.05),
    "max_entries_per_settlement": 8,
    "max_open_per_coldkey": 2,
    "max_bench_holds": 4,
    "loser_ttl_eras": 4,
    "dtypes": ("BF16",),
    "max_mirror_lag_hours": 6.0,
    "max_judged_per_coldkey_per_settlement": 2,
    "max_coldkey_unresolved_settlements": 4,
    "max_reveal_delay_days": 14,
    "lowfreq_freqs": None,             # "daily and slower" (checked separately)
    "ready_ttl_blocks": None,          # no plan default: a positive integer
    "max_source_share": 0.25,
}
W2_KEYS = tuple(W2_DEFAULTS)

# rev6.18 B11 (owner brief 2026-11-23): the DEC-CA-0062 placeholder keys and
# confirm_on_king_change are RETIRED. The deployed file still carries them, so
# each must load and change nothing (listed in the loader's RETIRED_KEYS);
# the shipped files carry none of them.
RETIRED_VALUES: dict[str, object] = {
    "confirm_on_king_change": "original",
    # rev6.23 (owner 2026-11-24): court_weights replaces it.
    "king_decay": 0.25,
    "targets": ("energy",),
    "teacher_ref": "ephemeris:panel@1",
    "teacher_api": "ephemeris",
    "teacher_mode": "explicit",
    "teacher_models": ("chronos-2",),
    "real_corpus_ref": "repo@sha256:" + "0123456789abcdef" * 4,
    "data_sources": ("real",),
    "augmentations": ("scale",),
    "expert_choice": "affinity",
    "max_context_length": 16384,
    "pool_prefix": "pool_d2",
    "gain_margin": 0.02,
    "full_rounds": 4,
    "half_life_rounds": 2.0,
    "floor": 0.2,
    "win_ttl_eras": 20,
}
RETIRED_LANE_D = tuple(RETIRED_VALUES)

# Per key: (value A, value B) for the mutation test, plus co-keys the value
# needs to load (until_block needs an armed from_block; a co-key set to None is
# removed so its default follows, e.g. gate_weights over a changed
# gate_suites). Each pair differs, so a parser that ignores the key yields
# equal configs and fails.
MUTATIONS: dict[str, tuple[object, object, dict]] = {
    "from_block": (ARM_BLOCK, ARM_BLOCK_2, {}),
    "until_block": (UNTIL_BLOCK, UNTIL_BLOCK_2, {"from_block": ARM_BLOCK}),
    "share": (0.25, 0.1, {"takes_from": "lane_a"}),
    "takes_from": ("lane_a", "lane_c", {}),
    # rev6.6: B is a non-default subset (the default is all six variants).
    "contexts": ([4096, 32768], [4096, 8192, 16384], {}),
    "horizon": (32, 128, {}),
    "forward_margin_hours": (3.0, 6.0, {}),   # rev6.14 (a): <= eval_delay_hours (6)
    "min_windows": (500, 2000, {}),
    "min_clusters": (10, 30, {}),
    "min_long_series": (150, 600, {}),
    "d_margin": (0.02, 0.03, {}),
    "d_margin_start": (0.04, 0.05, {}),
    "d_margin_warmup_settlements": (16, 64, {}),
    "alpha": (0.1, 0.01, {}),
    "bootstrap_B": (2000, 20_000, {}),
    "gate_suites": (["gift", "boom"], ["gift", "boom", "time"], {"gate_weights": None}),
    "gate_weights": ({"gift": 2.0, "boom": 1.0, "time": 1.0, "fev": 1.0},
                     {"gift": 1.0, "boom": 1.0, "time": 1.0, "fev": 3.0},
                     {"gate_suites": ["gift", "boom", "time", "fev"]}),
    "tau_public": (0.02, 0.005, {}),
    # rev6.23: court_weights moves with it (co_for: one weight per slot).
    "reward_prior_kings": (3, 1, {}),
    "court_weights": ([0.7, 0.2, 0.1], [0.6, 0.3, 0.1], {}),
    "max_entries_per_settlement": (4, 16, {}),
    "max_open_per_coldkey": (3, 1, {}),
    "max_bench_holds": (6, 2, {}),
    "loser_ttl_eras": (8, 2, {}),
    "max_mirror_lag_hours": (3.0, 12.0, {}),
    "max_judged_per_coldkey_per_settlement": (1, 3, {}),
    "max_coldkey_unresolved_settlements": (6, 2, {}),
    "max_reveal_delay_days": (7, 21, {}),
    "lowfreq_freqs": (["W"], ["D", "W"], {}),
    "ready_ttl_blocks": (1800, 2700, {}),
    "max_source_share": (0.5, 0.1, {}),
}
# remainder has one legal value ("burn") and dtypes one ("BF16"): their
# mutation is the refusal of every other value (test_w2_validation.py).
SINGLE_VALUED = ("remainder", "dtypes")

# Refused values per key (type and range). Each must be refused by the loader.
REFUSED: dict[str, list[object]] = {
    "from_block": [-1, True, "9720000", ARM_BLOCK + 1],
    "until_block": [-1, True, ARM_BLOCK - ERA_MAIN, ARM_BLOCK, UNTIL_BLOCK + 1],
    "share": [-0.1, 1.5, True, "0.25"],
    "takes_from": ["lane_z", "lane_d", ""],
    "remainder": ["lane_a", "king", ""],
    "contexts": [[], [262144], [4000], [0], [-4096], ["4096"], [True], 4096],
    "horizon": [0, -64, 513, 720, True, 64.5, "64"],
    "forward_margin_hours": [-1.0, True, "6"],
    "min_windows": [0, -1, True, 1.5],
    "min_clusters": [0, -1, True, 2.5],
    "min_long_series": [-1, True, 1.5],
    "d_margin": [0.0, -0.01, 1.0, True, 0.05],     # 0.05 > d_margin_start (rev6.23)
    "d_margin_start": [0.0, -0.03, 1.0, True, 0.005],  # 0.005 < d_margin
    "d_margin_warmup_settlements": [0, -1, True, 1.5, "32"],
    "alpha": [0.0, 1.0, -0.05, True],
    "bootstrap_B": [0, -1, True, 1000.5],
    "gate_suites": [[], [""], ["gift", "gift"], "gift", [1]],
    "gate_weights": [{"gift": -1.0, "boom": 1.0, "time": 1.0, "fev": 1.0},
                     {"gift": 0.0, "boom": 0.0, "time": 0.0, "fev": 0.0},
                     {"gift": 1.0, "xyz": 1.0},
                     {"gift": "1"}],
    "tau_public": [-0.01, 1.0, True],
    "reward_prior_kings": [-1, True, 1.5],
    "court_weights": [[0.8, 0.2], [0.8, 0.15, 0.05, 0.0], [0.9, 0.15, -0.05],
                      [0.8, 0.15, 0.1], [0.8, 0.15, 0.05 + 1e-8], [True, False, False],
                      ["0.8", 0.15, 0.05], 0.8, [float("nan"), 0.5, 0.5],
                      [float("inf"), 0.0, 0.0], []],
    "max_entries_per_settlement": [0, -1, True, 2.5],
    "max_open_per_coldkey": [0, -1, True],
    "max_bench_holds": [-1, True, 1.5],
    "loser_ttl_eras": [-1, True, 1.5],
    "dtypes": [[], ["F32"], ["BF16", "F32"], ["bf16"], "BF16"],
    "max_mirror_lag_hours": [0.0, -1.0, True],
    "max_judged_per_coldkey_per_settlement": [0, -1, True],
    "max_coldkey_unresolved_settlements": [0, -1, True],
    "max_reveal_delay_days": [0, -1, True],
    "lowfreq_freqs": [[], [""], [1], "D"],
    "ready_ttl_blocks": [0, -1, True, 1.5],
    "max_source_share": [0.0, -0.1, 1.5, True],
}
# Values that must load (range edges included).
ACCEPTED: dict[str, list[object]] = {
    "share": [0.0, 0.1, 0.25],
    "contexts": [[4096], [131072], [4096, 8192, 16384, 32768, 65536, 131072]],
    "horizon": [1, 64, 512],
    "takes_from": ["lane_a", "lane_b", "lane_c"],
    "dtypes": [["BF16"]],
    "max_source_share": [1.0, 0.25],
    "court_weights": [[1.0, 0.0, 0.0], [1 / 3, 1 / 3, 1 / 3], [0.80, 0.15, 0.05 + 1e-10]],
    "d_margin_start": [0.01, 0.5],        # = d_margin (no decay) and wide
    "d_margin_warmup_settlements": [1, 1000],
}


def co_for(key: str, value) -> dict:
    """The co-keys ``key = value`` needs to load: MUTATIONS' co-keys, and
    (rev6.23) for ``reward_prior_kings = n`` a ``court_weights`` of
    ``1 + n`` slots (the loader refuses a length mismatch)."""
    co = dict(MUTATIONS[key][2]) if key in MUTATIONS else {}
    if key == "reward_prior_kings" and type(value) is int and value >= 0:
        co["court_weights"] = [1.0] + [0.0] * value
    return co


# ── loading ──────────────────────────────────────────────────────────────────


@cache
def _raw_text(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


# TESTNET-FAST profile (owner 2026-11-24: lanes B/C/D end-to-end on testnet in
# hours): chain.testnet.toml scales these [lane_d] waits / caps to its 30-min
# settlement grid, each line commented `# TESTNET-FAST (2026-11-24): <mainnet>`.
# key -> (release default, testnet-fast value). The default tests above run on
# the testnet file with these restored (shipped_raw); test_w2_testnet_fast.py
# pins the profile values themselves and that chain.toml carries none of them.
TESTNET = "chain.testnet.toml"
TESTNET_FAST_LANE_D: dict[str, tuple[object, object]] = {
    "eval_delay_hours": (6.0, 0.5),
    "min_windows": (1000, 200),
    "min_clusters": (20, 5),
    "min_long_series": (300, 30),
    "d_margin_warmup_settlements": (32, 4),
    "embargo_hours": (48.0, 1.0),
    "max_windows_per_settlement": (5000, 1000),
}


def file_raw(name: str) -> dict:
    """The shipped file exactly as written (testnet-fast profile included)."""
    return tomllib.loads(_raw_text(name))


def shipped_raw(name: str) -> dict:
    """A fresh deep copy of a shipped config's tomllib table; on the testnet
    file the TESTNET-FAST [lane_d] keys are restored to their release default."""
    raw = file_raw(name)
    if name == TESTNET:
        ld = raw.setdefault("lane_d", {})
        for k, (default, _fast) in TESTNET_FAST_LANE_D.items():
            ld[k] = default
    return raw


def raw_with(name: str, lane_d: dict | None = None, *, drop: tuple[str, ...] = (),
             table: dict | None = None) -> dict:
    """The shipped table with ``[lane_d]`` keys overwritten (``lane_d``; a
    ``None`` value removes the key), removed (``drop``) or the whole table
    replaced (``table``)."""
    raw = shipped_raw(name)
    ld = copy.deepcopy(table) if table is not None else dict(raw.get("lane_d") or {})
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
    """Parse ``raw``; a refusal is an assertion failure."""
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


def norm(value):
    """Lists and tuples compare equal; mappings (or pair sequences) become dicts."""
    if isinstance(value, dict):
        return {k: norm(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(norm(v) for v in value)
    return value


def weights(value) -> dict[str, float]:
    """gate_weights as {suite: share of the total} (any positive scale)."""
    try:
        d = dict(value)
    except (TypeError, ValueError) as e:
        raise AssertionError(f"gate_weights is not a suite -> weight mapping: {value!r}") from e
    total = sum(float(v) for v in d.values())
    assert total > 0, f"gate_weights sum to {total}"
    return {str(k): round(float(v) / total, 12) for k, v in d.items()}


def comparable(cfg):
    """The parsed config without its raw table."""
    return replace(cfg, raw={})


# ── consensus fingerprints (invariant 1) ─────────────────────────────────────


def consensus_fingerprint(cfg) -> dict:
    """The contract / consensus digests a [lane_d] default must never move."""
    from cascade.shared.direct_moe import digest
    from cascade.shared.manifest import contract_digest, numerics_contract_digest

    out = {
        "contract_digest": contract_digest(cfg.training),
        "numerics_contract_digest": numerics_contract_digest(cfg.training),
    }
    v2c = getattr(cfg.training, "v2_contract", None)
    if v2c is not None:
        out["v2_contract_digest"] = contract_digest(v2c)
        out["v2_numerics_contract_digest"] = numerics_contract_digest(v2c)
    for block in (0, ARM_BLOCK - SETTLEMENT, ARM_BLOCK + SETTLEMENT):
        out[f"koth_params@{block}"] = digest(json.loads(json.dumps(
            asdict(cfg.koth_params(block=block)), sort_keys=True, default=str)))
    return out


# ── overlays ─────────────────────────────────────────────────────────────────


def overlay_body(cfg, overrides: dict, effective_block: int = EFFECTIVE):
    """Build a seq-1 overlay body; a refusal is an assertion failure."""
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


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{json.dumps(str(k))} = {_toml_value(x)}"
                                for k, x in v.items()) + " }"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise AssertionError(f"[lane_d] value {v!r} ({type(v).__name__}) has no TOML form")


def dump_lane_d(ld, keys: tuple[str, ...]) -> str:
    """The ``[lane_d]`` table text for ``keys`` read back off a parsed config."""
    lines = ["[lane_d]"]
    for k in keys:
        v = field(ld, k)
        if k == "gate_weights":
            v = dict(v)          # a suite -> weight table in TOML, however it is held
        lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines) + "\n"


_TABLE_HEADER = re.compile(r"^\[")


def splice_lane_d(name: str, table_text: str) -> str:
    """The shipped file's text with its ``[lane_d]`` table replaced."""
    lines = _raw_text(name).splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "[lane_d]")
    end = next((i for i in range(start + 1, len(lines)) if _TABLE_HEADER.match(lines[i])),
               len(lines))
    return "".join(lines[:start]) + table_text + "\n" + "".join(lines[end:])


def load_text(text: str, tmp_path: Path, name: str = "chain.toml"):
    from cascade.shared.config import load_chain_config

    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    try:
        return load_chain_config(p)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"loader refused the dumped [lane_d] table: "
                             f"{type(e).__name__}: {e}") from None


# ── scripts/laned tools ──────────────────────────────────────────────────────


def run_unread_keys(*configs: str | Path) -> subprocess.CompletedProcess:
    args = [sys.executable, str(REPO / "scripts" / "laned" / "unread_keys.py"),
            "--root", str(REPO), "--source", "cascade/shared/config"]
    for c in configs:
        args += ["--config", str(c)]
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
