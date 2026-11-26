"""Shared spec and helpers for the W2b acceptance tests (``[lane_d_ops]``).

Copied from docs/plans/LANE_D_PLAN.md (rev6): the W2b row, §1.2 (the
dedicated private vault bucket ``cascade-lane-d-vault``, object paths
``lane-d/<hotkey>/<digest>/``, never the shared manifest bucket), §1.7 /
§10.2 (``pool_d`` lives in a private bucket, testnet under its own prefix),
D11 (the launch eval host) and §14.2 (a non-consensus table read only by the
main validator and the intake; satellites never read it).

No PATHS module is imported at module level: every import of
``cascade.*`` happens inside a function a test body calls, so a missing
module fails inside the test (the W0c expect-fail convention).
"""

from __future__ import annotations

import ast
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
OPS_MODULE = "cascade.shared.config.lane_d_ops"
SATELLITE_MODULE = "cascade.validator.satellite"

# Blocks on the mainnet (3600) and testnet (600) era grids (as W2).
ERA_MAIN = 3600
ARM_BLOCK = 2700 * ERA_MAIN
SETTLEMENT = 900
ARM = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a"}

# Every [lane_d_ops] key. ``None`` = no plan default: the test pins validity
# (README "defaults assumed"); the builder picks and lists the value.
OPS_DEFAULTS: dict[str, object] = {
    "vault_bucket": "cascade-lane-d-vault",    # §1.2
    "vault_prefix": "lane-d",                  # §11.1 lane-d/<hotkey>/<digest>/
    "pool_d_bucket": None,                     # a private bucket (§1.7)
    "pool_d_prefix": None,
    "eval_host": "",                           # unset until the S4 bring-up (D11)
    "queue_max_inflight": None,                # positive int
    "queue_max_pending": None,                 # positive int
    "queue_poll_seconds": None,                # positive int
}
OPS_KEYS = tuple(OPS_DEFAULTS)
STR_KEYS = ("vault_bucket", "vault_prefix", "pool_d_bucket", "pool_d_prefix", "eval_host")
INT_KEYS = ("queue_max_inflight", "queue_max_pending", "queue_poll_seconds")

# Per key: two different legal values (mutation test). Column A and column B
# are each a legal table as a whole (the buckets differ from each other).
MUTATIONS: dict[str, tuple[object, object]] = {
    "vault_bucket": ("cascade-lane-d-vault-a", "cascade-lane-d-vault-b"),
    "vault_prefix": ("lane-d-a", "ops/lane-d"),
    "pool_d_bucket": ("cascade-pool-d-a", "cascade-pool-d-b"),
    "pool_d_prefix": ("pool_d_a", "ops/pool_d"),
    "eval_host": ("laned-eval", "valpriv-4090"),
    "queue_max_inflight": (2, 5),
    "queue_max_pending": (32, 128),
    "queue_poll_seconds": (30, 90),
}
TABLE_A = {k: a for k, (a, _b) in MUTATIONS.items()}
TABLE_B = {k: b for k, (_a, b) in MUTATIONS.items()}

_BAD_BUCKETS = ["", "a", "Cascade-Vault", "-vault", "vault-", "vault_bucket", "cascade lane",
                "x" * 64, 5, True]
_BAD_PREFIXES = ["", "/lane-d", "lane-d/../x", "..", "lane d", 5, True]
_BAD_INTS = [0, -1, True, 1.5, "4"]
REFUSED: dict[str, list[object]] = {
    "vault_bucket": _BAD_BUCKETS,
    "vault_prefix": _BAD_PREFIXES,
    "pool_d_bucket": _BAD_BUCKETS,
    "pool_d_prefix": _BAD_PREFIXES,
    "eval_host": ["user@laned-eval", "user:pw@laned-eval", "laned eval", "-laned-eval",
                  "http://laned-eval", "laned-eval/x", 7, True],
    "queue_max_inflight": _BAD_INTS,
    "queue_max_pending": _BAD_INTS,
    "queue_poll_seconds": _BAD_INTS,
}
ACCEPTED: dict[str, list[object]] = {
    "vault_bucket": ["cascade-lane-d-vault", "abc", "lane.d-vault-2"],
    "vault_prefix": ["lane-d", "testnet/lane-d"],
    "pool_d_prefix": ["pool_d", "testnet/pool_d"],
    "eval_host": ["", "laned-eval", "valpriv-4090", "eval-1.laned.internal"],
    "queue_max_inflight": [1, 64],
    "queue_poll_seconds": [1, 3600],
}

# Secret-shaped key names: never accepted (the table carries names and
# locations only; credentials are env-only).
SECRET_KEYS = ("vault_secret", "hippius_sub_token", "vault_access_key", "s3_secret_key",
               "hf_token", "eval_host_password", "secret", "token")

_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def valid_bucket(v) -> bool:
    return isinstance(v, str) and bool(_BUCKET_RE.match(v))


def valid_prefix(v) -> bool:
    return (isinstance(v, str) and bool(v) and not v.startswith("/")
            and ".." not in v.split("/") and not any(c.isspace() for c in v))


# ── loading ──────────────────────────────────────────────────────────────────


@cache
def _raw_text(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def shipped_raw(name: str) -> dict:
    return tomllib.loads(_raw_text(name))


def raw_with(name: str, ops: dict | None = None, *, drop: tuple[str, ...] = (),
             table: dict | None = None, absent: bool = False,
             lane_d: dict | None = None) -> dict:
    """The shipped table with ``[lane_d_ops]`` keys overwritten (``ops``),
    removed (``drop``), replaced whole (``table``) or the table removed
    (``absent``); ``lane_d`` overwrites ``[lane_d]`` keys (arming)."""
    raw = shipped_raw(name)
    if lane_d:
        raw.setdefault("lane_d", {}).update(copy.deepcopy(lane_d))
    if absent:
        raw.pop("lane_d_ops", None)
        return raw
    t = copy.deepcopy(table) if table is not None else dict(raw.get("lane_d_ops") or {})
    for k in drop:
        t.pop(k, None)
    t.update(copy.deepcopy(ops or {}))
    raw["lane_d_ops"] = t
    return raw


def parse(raw: dict):
    from cascade.shared.config import parse_chain_config

    return parse_chain_config(raw)


def load_ok(raw: dict, why: str = ""):
    try:
        return parse(raw)
    except Exception as e:     # noqa: BLE001 - every refusal is the failure under test
        raise AssertionError(f"chain loader refused the config{why}: "
                             f"{type(e).__name__}: {e}") from None


def ops_of(cfg):
    """The ``[lane_d_ops]`` view of a loaded config (the main-validator /
    intake entry point)."""
    from cascade.shared.config.lane_d_ops import lane_d_ops_config

    return lane_d_ops_config(cfg)


def ops_ok(raw: dict, why: str = ""):
    cfg = load_ok(raw, why)
    try:
        return ops_of(cfg)
    except (ValueError, TypeError, KeyError) as e:
        raise AssertionError(f"[lane_d_ops] refused{why}: {type(e).__name__}: {e}") from None


def refusal(raw: dict) -> str | None:
    """The refusal text for ``raw`` (from the chain loader or from
    ``lane_d_ops_config``), or None when both accept it."""
    try:
        cfg = parse(raw)
    except (ValueError, TypeError, KeyError) as e:
        return str(e)
    from cascade.shared.config.lane_d_ops import lane_d_ops_config

    try:
        lane_d_ops_config(cfg)
    except (ValueError, TypeError, KeyError) as e:
        return str(e)
    return None


def field(ops, key: str):
    assert hasattr(ops, key), f"LaneDOpsConfig has no field {key!r} (the key is not parsed)"
    return getattr(ops, key)


def comparable(cfg):
    return replace(cfg, raw={})


# ── consensus fingerprints ───────────────────────────────────────────────────


def consensus_fingerprint(cfg) -> dict:
    """Contract digests (lane A and v2) and koth_params digests around the
    lane D arming block."""
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


def receipt_bytes(cfg) -> tuple[str, str]:
    """A scored and a rejected receipt built from ``cfg`` (contract digests,
    seeds, koth params, king_decay all read off the config), serialised."""
    from cascade.shared.receipt import dump_receipt
    from tests.unit.receipt_fixture import make_rejected_receipt, make_scored_receipt

    scored, _k, _c = make_scored_receipt(cfg)
    return dump_receipt(scored), dump_receipt(make_rejected_receipt(cfg))


# ── TOML round-trip ──────────────────────────────────────────────────────────


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    raise AssertionError(f"[lane_d_ops] value {v!r} ({type(v).__name__}) is not a str or int")


def dump_ops(ops) -> str:
    lines = ["[lane_d_ops]"]
    lines += [f"{k} = {_toml_value(field(ops, k))}" for k in OPS_KEYS]
    return "\n".join(lines) + "\n"


_TABLE_HEADER = re.compile(r"^\[")


def splice_ops(name: str, table_text: str) -> str:
    """The shipped file's text with its ``[lane_d_ops]`` table replaced (or
    appended when the file has none)."""
    lines = _raw_text(name).splitlines(keepends=True)
    starts = [i for i, ln in enumerate(lines) if ln.strip() == "[lane_d_ops]"]
    if not starts:
        return "".join(lines) + "\n" + table_text
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if _TABLE_HEADER.match(lines[i])),
               len(lines))
    return "".join(lines[:start]) + table_text + "\n" + "".join(lines[end:])


def load_text(text: str, tmp_path: Path, name: str):
    from cascade.shared.config import load_chain_config

    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    try:
        return load_chain_config(p)
    except Exception as e:     # noqa: BLE001
        raise AssertionError(f"loader refused the dumped file: {type(e).__name__}: {e}") from None


# ── static import graph ──────────────────────────────────────────────────────


def module_file(name: str) -> tuple[Path | None, bool]:
    p = REPO.joinpath(*name.split("."))
    if (p / "__init__.py").is_file():
        return p / "__init__.py", True
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py"), False
    return None, False


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts) + 1)]


def _is_type_checking(test: ast.AST) -> bool:
    return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def _load_time(body: list[ast.stmt]):
    """Statements that run when the module is imported: function bodies are
    skipped, class bodies, if/try/with blocks are entered, ``if
    TYPE_CHECKING:`` bodies are skipped."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.ClassDef):
            yield from _load_time(node.body)
        elif isinstance(node, ast.If):
            if not _is_type_checking(node.test):
                yield from _load_time(node.body)
            yield from _load_time(node.orelse)
        elif isinstance(node, ast.Try):
            for b in (node.body, node.orelse, node.finalbody):
                yield from _load_time(b)
            for h in node.handlers:
                yield from _load_time(h.body)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            yield from _load_time(node.body)
        else:
            yield node


def imported_modules(name: str, *, everywhere: bool) -> set[str]:
    """The in-repo ``cascade.*`` modules ``name`` imports: at load time, or
    (``everywhere``) anywhere in the file, function bodies included. A
    ``from pkg import x`` where ``pkg/x`` is a module counts as importing it;
    every parent package of an imported module counts too."""
    path, is_pkg = module_file(name)
    assert path is not None, f"module {name} not found under {REPO}"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = list(ast.walk(tree)) if everywhere else list(_load_time(tree.body))
    pkg = name if is_pkg else name.rpartition(".")[0]
    out: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            for a in node.names:
                out.update(_parents(a.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.update(_parents(mod))
            out.update(f"{mod}.{a.name}" for a in node.names)
    return {m for m in out if m.split(".")[0] == "cascade" and module_file(m)[0] is not None}


def satellite_code_path() -> set[str]:
    """Every module a satellite process can load: satellite.py's own imports
    (module level AND inside its functions), its parent packages, and the
    load-time import closure of all of them."""
    todo = list(imported_modules(SATELLITE_MODULE, everywhere=True)
                | set(_parents(SATELLITE_MODULE)))
    seen: set[str] = set()
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        todo.extend(imported_modules(m, everywhere=False) - seen)
    return seen


def string_constants(name: str) -> set[str]:
    path, _ = module_file(name)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


# ── scripts/laned tools ──────────────────────────────────────────────────────


def run_unread_keys(*configs: str | Path) -> subprocess.CompletedProcess:
    args = [sys.executable, "-P", str(REPO / "scripts" / "laned" / "unread_keys.py"),
            "--root", str(REPO), "--source", "cascade/shared/config"]
    for c in configs:
        args += ["--config", str(c)]
    return subprocess.run(args, capture_output=True, text=True, timeout=120, check=False)
