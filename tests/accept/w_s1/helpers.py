"""Shared spec and helpers for the W-S1 acceptance tests (§16 sentinel, lane A).

Everything here comes from docs/plans/LANE_D_PLAN.md: the W-S1 row (§3.0),
§16 (the rule, consensus and rollout, TS1-TS5), §3.2 invariants 1 and 10,
§14.2 and the §17.4 register (D33 adopt, D34 ratchet on, D45 no shadow).

Every check that can fail is an ``assert``. A missing behaviour on main fails
with an ``AssertionError`` (an unknown keyword or dataclass field is turned
into one here), or with an import error for a PATHS module. Nothing from a
PATHS module is imported at module level or in a fixture.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import tomllib
from dataclasses import replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SHIPPED = ("chain.toml", "chain.testnet.toml")
GOLDEN_RECEIPT = REPO / "tests" / "fixtures" / "round_receipt_v3.json"

# Era length: 3600 blocks on mainnet (900 x 4), 600 on testnet (150 x 4).
ERA = {"chain.toml": 3600, "chain.testnet.toml": 600}
EFFECTIVE = 2698 * 3600            # 9_712_800: an era boundary on both grids
SETTLEMENT = 900
PROBE_BLOCKS = (0, 8_190_000 + SETTLEMENT, EFFECTIVE, EFFECTIVE + SETTLEMENT)

ALL_SUITES = ("gift", "boom", "time", "fev")

# Defaults (§16, §17.4 D33/D34). sentinel_from_block = 0 is "off".
WS1_DEFAULTS: dict[str, object] = {
    "sentinel_from_block": 0,
    "sentinel_suites": ALL_SUITES,
    "sentinel_ratchet": True,      # D34: recommended on, decided on
    "sentinel_tau": 0.01,          # D34: "by more than sentinel_tau (0.01)"
}
WS1_KEYS = tuple(WS1_DEFAULTS)

# Per key: (value A, value B). A != B, so an unread key fails.
MUTATIONS: dict[str, tuple[object, object]] = {
    "sentinel_from_block": (EFFECTIVE, EFFECTIVE + 3600),
    "sentinel_suites": (list(ALL_SUITES), ["boom", "time", "fev"]),
    "sentinel_ratchet": (True, False),
    "sentinel_tau": (0.02, 0.05),
}

REFUSED: dict[str, list[object]] = {
    "sentinel_from_block": [-1, True, "9712800", 9_712_800.0, EFFECTIVE + 1],
    "sentinel_suites": [[], ["gift", "gift"], ["gifteval"], ["GIFT"], "gift", [1], [""]],
    "sentinel_ratchet": [1, 0, "true"],
    "sentinel_tau": [-0.01, True, "0.01"],
}
ACCEPTED: dict[str, list[object]] = {
    "sentinel_from_block": [0, EFFECTIVE, EFFECTIVE + 3600],
    "sentinel_suites": [list(ALL_SUITES), ["boom", "time", "fev"], ["boom", "time"]],
    "sentinel_ratchet": [True, False],
    "sentinel_tau": [0.01, 0.05, 0.5],
}

# Unarmed goldens, computed on main @ e614e7b (byte-identity, invariant 1).
GOLDEN_GATE_DETAIL_SHA = "e78aad390a8a9f29024f1bc6440b39642aeaf920e1298c800a65f56ec6cd7010"
GOLDEN_KING_CHECK_DETAIL_SHA = "d9cacd307c9e3073dd506665d502e39495010b37597e5b38eadc0e80c11414ab"
GOLDEN_SCORED_RECEIPT_SHA = "fe4d5a71be1cac8e26089109ee679fe9f92126842549a6c5c4ec0287958033be"
GOLDEN_REJECTED_RECEIPT_SHA = "4e4151ab26ce0395a5938f0759110a453d41b7ec8e5e3b03b29dc8249be10f62"
# The per-suite factors of the golden gate fixture (every suite better).
GOLDEN_FACTORS = {"gift": 0.9, "boom": 0.95, "time": 0.92, "fev": 0.97}


def synth_hash(tag: str | int) -> str:
    """A block-hash-shaped value (``0x`` + 64 lowercase hex), as the chain
    client returns it."""
    return "0x" + hashlib.sha256(f"cascade ws1 block {tag}".encode()).hexdigest()


# ── loading ──────────────────────────────────────────────────────────────────


@cache
def _raw_text(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def shipped_raw(name: str) -> dict:
    return tomllib.loads(_raw_text(name))


def raw_with(name: str, v2: dict | None = None, *, drop: tuple[str, ...] = ()) -> dict:
    """The shipped table with ``[v2]`` keys overwritten or removed."""
    raw = shipped_raw(name)
    t = dict(raw.get("v2") or {})
    for k in drop:
        t.pop(k, None)
    for k, v in (v2 or {}).items():
        t[k] = copy.deepcopy(v)
    raw["v2"] = t
    return raw


def parse(raw: dict):
    from cascade.shared.config import parse_chain_config

    return parse_chain_config(raw)


def load_ok(raw: dict, why: str = ""):
    try:
        return parse(raw)
    except Exception as e:     # noqa: BLE001 - every refusal is the failure under test
        raise AssertionError(f"loader refused a valid [v2] table{why}: "
                             f"{type(e).__name__}: {e}") from None


def refused(raw: dict) -> bool:
    try:
        parse(raw)
    except (ValueError, TypeError, KeyError):
        return True
    return False


def field(v2, key: str):
    assert hasattr(v2, key), f"V2Config has no field {key!r} (the [v2] key is not parsed)"
    return getattr(v2, key)


def norm(value):
    if isinstance(value, (list, tuple)):
        return tuple(norm(v) for v in value)
    return value


def comparable(cfg):
    return replace(cfg, raw={})


def armed_cfg(name: str, **extra):
    """The shipped config armed at EFFECTIVE through the chain.toml table."""
    return load_ok(raw_with(name, {"sentinel_from_block": EFFECTIVE, **extra}), " (armed)")


# ── consensus fingerprint (invariant 1) ──────────────────────────────────────


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def lane_a_contract_digests(cfg) -> dict:
    from cascade.shared.manifest import contract_digest, numerics_contract_digest

    out = {"contract_digest": contract_digest(cfg.training),
           "numerics_contract_digest": numerics_contract_digest(cfg.training)}
    v2c = getattr(cfg.training, "v2_contract", None)
    if v2c is not None:
        out["v2_contract_digest"] = contract_digest(v2c)
        out["v2_numerics_contract_digest"] = numerics_contract_digest(v2c)
    return out


def consensus_fingerprint(cfg) -> dict:
    from dataclasses import asdict

    from cascade.shared.era import bench_gate_mode_at, bench_gate_rule_at

    out = lane_a_contract_digests(cfg)
    for block in PROBE_BLOCKS:
        out[f"koth_params@{block}"] = _digest(asdict(cfg.koth_params(block=block)))
        out[f"bench_rule@{block}"] = bench_gate_rule_at(cfg, block)
        out[f"bench_mode@{block}"] = bench_gate_mode_at(cfg, block)
        out[f"v2_active@{block}"] = cfg.v2.active(block)
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
    from cascade.shared import netconfig as nc

    body = overlay_body(cfg, overrides, effective_block)
    try:
        return nc.validate_overlay(cfg, body)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"overlay {overrides!r} refused by validate_overlay: "
                             f"{type(e).__name__}: {e}") from None


def overrides_refusal(overrides: dict) -> str | None:
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
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    raise AssertionError(f"[v2] value {v!r} ({type(v).__name__}) has no TOML form")


def splice_v2(name: str, lines_by_key: dict[str, str]) -> str:
    """The shipped text with each ``key = ...`` line of ``[v2]`` replaced (or
    added) by ``key = <value text>``."""
    lines = _raw_text(name).splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "[v2]")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")),
               len(lines))
    key_re = re.compile(r"^\s*(" + "|".join(map(re.escape, lines_by_key)) + r")\s*=")
    body = [ln for ln in lines[start + 1:end] if not key_re.match(ln)]
    new = [f"{k} = {v}\n" for k, v in lines_by_key.items()]
    return "".join(lines[:start + 1] + new + body + lines[end:])


def load_text(text: str, tmp_path: Path, name: str):
    from cascade.shared.config import load_chain_config

    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    try:
        return load_chain_config(p)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"loader refused the spliced [v2] table: "
                             f"{type(e).__name__}: {e}") from None


# ── bench rows (the DEC-CA-0061 rule v2 inputs) ──────────────────────────────


def bench_rows(n: int = 40, seed: int = 0, suites=ALL_SUITES) -> dict:
    """Signed-row-shaped king rows: per suite ``n`` tasks with CRPS / MASE."""
    rng = np.random.default_rng(seed)
    out = {}
    for suite in ALL_SUITES:
        base = rng.uniform(0.5, 1.5, size=(n, 2))
        if suite in suites:
            out[suite] = {"status": "ok", "rows": [
                {"item": f"{suite}/{i}", "crps": float(c), "mase": float(m)}
                for i, (c, m) in enumerate(base)]}
    return out


def scaled_pair(factors: dict[str, float], *, drop: tuple[str, ...] = ()) -> tuple[dict, dict]:
    """(king rows, challenger rows): the challenger's task is the king's times
    ``factors[suite]`` times a per-task jitter in [0.97, 1.03]. Suites in
    ``drop`` are absent from both sides."""
    king = bench_rows()
    chal = json.loads(json.dumps(king))
    rng = np.random.default_rng(7)
    for suite, f in factors.items():
        for r in chal[suite]["rows"]:
            x = f * float(rng.uniform(0.97, 1.03))
            r["crps"] *= x
            r["mase"] *= x
    for s in drop:
        king.pop(s, None)
        chal.pop(s, None)
    return king, chal


GATE_KW = {"alpha": 0.05, "B": 200, "seed": "ws1"}
KC_PARAMS = SimpleNamespace(bootstrap_alpha=0.05, bootstrap_B=200, bench_gate_min_items=10,
                            bench_gate_tolerance=0.01, bench_gate_min_win_rate=0.5)


def _no_kwarg(e: TypeError, what: str) -> AssertionError:
    return AssertionError(f"{what} does not take the W-S1 sentinel keywords: {e}")


def gate(king_rows, chal_rows, **kw):
    """``evaluate_bench_gate`` under rule v2 (an unknown keyword on main is
    an assertion failure)."""
    from cascade.eval.bench_gate import RULE_V2, evaluate_bench_gate

    try:
        return evaluate_bench_gate(king_rows, chal_rows, None, None, rule=RULE_V2,
                                   **GATE_KW, **kw)
    except TypeError as e:
        if "sentinel" in str(e):
            raise _no_kwarg(e, "evaluate_bench_gate") from None
        raise


def king_check(king_rows, init_rows, **kw):
    from cascade.validator.king_check import king_check_gate

    try:
        return king_check_gate(king_rows, init_rows, None, None, KC_PARAMS, seed="ws1", **kw)
    except TypeError as e:
        if "sentinel" in str(e):
            raise _no_kwarg(e, "king_check_gate") from None
        raise


def detail_sha(result) -> str:
    return hashlib.sha256(json.dumps(result.detail(), sort_keys=True).encode()).hexdigest()


def verdict_core(result) -> dict:
    """The parts of a gate result a verdict reads."""
    d = result.detail() or {}
    return {"computed": result.computed, "passed": result.passed, "reason": result.reason,
            "suites": d.get("suites"), "avg_ratio": d.get("avg_ratio"),
            "worst_lcb": d.get("worst_lcb"), "win_rate": d.get("win_rate")}


def geomean(xs) -> float:
    xs = list(xs)
    return math.exp(math.fsum(math.log(x) for x in xs) / len(xs))


# ── receipts ─────────────────────────────────────────────────────────────────


def with_sentinel(receipt, record):
    """``receipt`` with its ``sentinel`` field set (a missing field on main is
    an assertion failure)."""
    try:
        return replace(receipt, sentinel=record)
    except TypeError as e:
        raise AssertionError(f"RoundReceipt has no 'sentinel' field: {e}") from None


def rejected_receipt(**kw):
    """``build_receipt`` for a rejected round (the receipt_fixture recipe), with
    extra keywords passed through."""
    from cascade.shared.chain import seed_from_block_hash
    from cascade.shared.receipt import build_receipt
    from cascade.trainer.contract import RoundSeeds
    from tests.unit.receipt_fixture import BLOCK_HASH, EPOCH_START, make_manifest

    class _Salt:
        train_seed_salt = 1337

    base_seed = seed_from_block_hash(BLOCK_HASH)
    manifest = make_manifest(None, base_seed=base_seed)
    try:
        return build_receipt(round_id=manifest.round_id, status="rejected",
                             epoch_start_block=EPOCH_START, epoch_block_hash=BLOCK_HASH,
                             base_seed=base_seed, seeds=RoundSeeds.derive(base_seed, _Salt()),
                             manifest=manifest, reject_reason="signature_invalid", **kw)
    except TypeError as e:
        if "sentinel" in str(e):
            raise AssertionError(f"build_receipt has no 'sentinel' keyword: {e}") from None
        raise


def body_of(receipt) -> dict:
    return json.loads(receipt.canonical_body().decode("utf-8"))


def body_sha(receipt) -> str:
    return hashlib.sha256(receipt.canonical_body()).hexdigest()


# ── processes and tools ──────────────────────────────────────────────────────


DRAW_SCRIPT = (
    "import json, sys\n"
    "from cascade.shared.sentinel import sentinel_for_era\n"
    "hashes = json.loads(sys.argv[1]); suites = json.loads(sys.argv[2])\n"
    "print(json.dumps([sentinel_for_era(h, suites) for h in hashes]))\n"
)


def draw_in_process(hashes: list[str], suites, hash_seed: str) -> list[str]:
    """The draws computed by a separate interpreter (another validator) with
    its own ``PYTHONHASHSEED``."""
    env = {"PYTHONPATH": str(REPO), "PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin"}
    proc = subprocess.run([sys.executable, "-P", "-c", DRAW_SCRIPT, json.dumps(hashes),
                           json.dumps(list(suites))], capture_output=True, text=True,
                          timeout=120, check=False, env=env, cwd=str(REPO))
    assert proc.returncode == 0, f"draw subprocess failed:\n{proc.stderr}"
    return json.loads(proc.stdout)


def run_unread_keys(*configs: str | Path) -> subprocess.CompletedProcess:
    args = [sys.executable, "-P", str(REPO / "scripts" / "laned" / "unread_keys.py"),
            "--root", str(REPO), "--source", "cascade/shared/config"]
    for c in configs:
        args += ["--config", str(c)]
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
