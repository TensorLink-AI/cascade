"""The W3b2 pod-evidence verifier and its offline fakes (RV5 BLOCKING item 6).

RV5 showed that synthetic rows labelled ``source: "pod"``, with any block
hash and a ``gate`` recomputed from those rows, passed the old verifier. The
evidence is now bound to facts an offline forger can't choose:

* **the suite** (README "Pod evidence (a)"): the drawn suite must have a
  repo pin: its data revision from ``benchmarks/cascade_benchmark/datasets.py``
  (``DATASETS[<sidecar>].revision``) and its item set from the repo's
  Seasonal-Naive baseline table for that suite
  (``benchmarks/cascade_benchmark/data/<gifteval|boom>_seasonal_naive.json``).
  Every role's drawn-suite identity must equal that pin, and ``gate`` is
  recomputed with ``pinned_identity`` = the pin;
* **the checkpoints** (b): each role carries a registry ref
  ``<repo>@sha256:<64 hex>`` whose hex is its sweep record's digest; the
  owner's live verify resolves every ref in the registry and checks KING_D
  and the house against the pointers in force at ``draw.block``;
* **the block** (c): offline only its format, network and netuid are
  checked; the ``verified_live`` record written by the owner's
  ``python -P -m cascade.laned.bench_evidence verify --evidence FILE`` re-read
  the hash from the chain and re-derived the suite with ``gate_suite_for``.
  The record is hash-bound to the evidence body (``body_sha256``, W3a's
  canonical form), so an edit after verification fails;
* **the params** are ``[lane_d] alpha`` / ``tau_public`` / ``bootstrap_B``
  of the shipped ``chain.toml``.

Nothing here imports a PATHS module at module level. ``lane_d_bench`` and
``bench_evidence`` are imported inside function bodies.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCHEMA = "laned_w3b2/2"
VERIFY_SCHEMA = "laned_w3b2_verify/1"
MODULE = "cascade.laned.bench_evidence"
MODULE_PATH = "cascade/laned/bench_evidence.py"
GATE_MODULE_PATH = "cascade/validator/lane_d_bench.py"
NETWORK = "finney"
ROLES = ("candidate", "king_d", "house")
REFS = ("king_d", "house")
MIN_ITEMS = 10
HEX64 = re.compile(r"[0-9a-f]{64}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HASH = re.compile(r"0x[0-9a-f]{64}")
REF = re.compile(r"[^\s@]+@sha256:([0-9a-f]{64})")
DATASETS_PY = REPO / "benchmarks" / "cascade_benchmark" / "datasets.py"
ITEM_TABLES = {"gift": REPO / "benchmarks" / "cascade_benchmark" / "data" / "gifteval_seasonal_naive.json",
               "boom": REPO / "benchmarks" / "cascade_benchmark" / "data" / "boom_seasonal_naive.json"}
SIDECAR = {"gift": "gift-eval", "boom": "boom", "time": "time", "fev": "fev"}
VERIFY_KEYS = ("schema", "verified", "body_sha256", "network", "netuid", "block", "block_hash",
               "finalized_block", "suite", "sentinel", "refs", "pointers", "resolved",
               "verified_at", "tool")
REGISTRY_REPO = "hub.hippius.com/cascade/checkpoints"
SEED = "lane-d/1:w3b2:pod"


# ── the shipped config and the repo pins ─────────────────────────────────────


def chain_raw() -> dict:
    return tomllib.loads((REPO / "chain.toml").read_text(encoding="utf-8"))


def lane_d_cfg():
    from cascade.shared.config import parse_chain_config

    return parse_chain_config(chain_raw()).lane_d


def netuid() -> int:
    return int(chain_raw()["subnet"]["netuid"])


def draw(block_hash: str, sentinel):
    """W3b1's gate suite for ``block_hash`` under the shipped chain.toml."""
    from cascade.validator.lane_d_gate import gate_suite_for

    cfg = lane_d_cfg()
    return gate_suite_for(block_hash, gate_suites=cfg.gate_suites,
                          gate_weights=cfg.gate_weights, sentinel=sentinel)


def repo_revisions() -> dict[str, str]:
    """``{sidecar suite: revision}`` read from the ``DatasetSpec(...)`` calls
    in ``benchmarks/cascade_benchmark/datasets.py`` (parsed, not imported)."""
    tree = ast.parse(DATASETS_PY.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DatasetSpec"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        rev = [k.value for k in node.keywords if k.arg == "revision"]
        if rev and isinstance(rev[0], ast.Constant) and isinstance(rev[0].value, str):
            out[str(node.args[0].value)] = rev[0].value
    return out


def _sha(obj, *, sort_keys: bool = False) -> str:
    canon = json.dumps(obj, sort_keys=sort_keys, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def pinned_items(suite: str) -> list[str] | None:
    p = ITEM_TABLES.get(suite)
    if p is None or not p.is_file():
        return None
    return sorted(json.loads(p.read_text(encoding="utf-8")))


def repo_pin(suite: str) -> dict | None:
    """The suite identity the repo pins for ``suite``, computed here
    independently of ``cascade.eval.suite_identity`` (W3b4's definition):
    ``None`` when the repo pins no revision or no item set for it (TIME and
    FEV today)."""
    rev = repo_revisions().get(SIDECAR.get(suite, ""))
    items = pinned_items(suite)
    if not rev or not items:
        return None
    items_sha = _sha(items)
    return {"suite": suite, "version": rev, "items_sha256": items_sha, "n_items": len(items),
            "digest": _sha({"items_sha256": items_sha, "suite": suite, "version": rev},
                           sort_keys=True)}


def canonical_body(evidence: dict) -> bytes:
    """W3a's form: the evidence WITHOUT ``verified_live``, sorted keys,
    compact separators, ASCII, UTF-8."""
    body = {k: v for k, v in evidence.items() if k != "verified_live"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def body_sha256(evidence: dict) -> str:
    return hashlib.sha256(canonical_body(evidence)).hexdigest()


def ref_for(digest: str, repo: str = REGISTRY_REPO) -> str:
    return f"{repo}@sha256:{digest}"


# ── the verifier ─────────────────────────────────────────────────────────────


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=False)


def _tool_problems(tool, *, module_path: str, check_git: bool, module: str | None = None
                   ) -> list[str]:
    if not isinstance(tool, dict):
        return ["tool is not an object"]
    out = []
    if module is not None and tool.get("module") != module:
        out.append(f"tool.module: want {module!r}, got {tool.get('module')!r}")
    sha = tool.get("git_sha")
    if not isinstance(sha, str) or not HEX40.fullmatch(sha):
        out.append("tool.git_sha is not 40 hex")
    elif check_git:
        if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            out.append(f"tool.git_sha {sha} is not a commit in this repository")
        elif _git("cat-file", "-e", f"{sha}:{module_path}").returncode != 0:
            out.append(f"tool.git_sha {sha} has no {module_path}")
    if check_git and tool.get("git_dirty") is not False:
        out.append("tool.git_dirty is not false (the run must be from a clean checkout)")
    if not check_git and not isinstance(tool.get("git_dirty"), bool):
        out.append("tool.git_dirty is not a bool")
    return out


def _entry_items(entry) -> list:
    return [r.get("item") for r in (entry.get("rows") or []) if isinstance(r, dict)]


def evidence_problems(ev, *, check_git: bool = True, require_live: bool = True) -> list[str]:
    """Every reason ``ev`` is not valid W3b2 pod evidence (empty = valid).
    ``require_live=False`` checks the body alone (never enough for L4)."""
    from cascade.validator.lane_d_bench import paired_gate

    if not isinstance(ev, dict):
        return ["evidence is not a JSON object"]
    p: list[str] = []
    if ev.get("schema") != SCHEMA:
        p.append(f"schema != {SCHEMA}")
    if ev.get("wp") != "W3b2":
        p.append("wp != W3b2")
    if ev.get("source") != "pod":
        p.append("source != pod (fixture evidence is never accepted)")
    if ev.get("passed") is not True:
        p.append("passed is not true")
    if not isinstance(ev.get("host"), str) or not ev.get("host"):
        p.append("host missing")
    p += _tool_problems(ev.get("tool"), module_path=GATE_MODULE_PATH, check_git=check_git)
    session = ev.get("session")
    if not isinstance(session, str) or not session:
        p.append("session missing")

    cfg = lane_d_cfg()
    suite, sentinel = ev.get("suite"), ev.get("sentinel")
    if suite not in tuple(cfg.gate_suites):
        return p + [f"suite {suite!r} is not one of [lane_d] gate_suites"]
    if sentinel is not None and (not isinstance(sentinel, str) or sentinel == suite):
        p.append("sentinel must be None or a suite other than the drawn one")

    draw_rec = ev.get("draw") or {}
    if draw_rec.get("network") != NETWORK:
        p.append(f"draw.network != {NETWORK!r} (the gate draws from the mainnet boundary)")
    if draw_rec.get("netuid") != netuid():
        p.append(f"draw.netuid != chain.toml netuid {netuid()}")
    blk = draw_rec.get("block")
    if isinstance(blk, bool) or not isinstance(blk, int) or blk <= 0:
        p.append("draw.block is not a positive int")
    bh = draw_rec.get("block_hash")
    if not isinstance(bh, str) or not HASH.fullmatch(bh):
        p.append("draw.block_hash is not 0x + 64 hex")
    else:
        want = draw(bh, sentinel)
        if suite != want:
            p.append(f"suite {suite!r} is not the W3b1 draw {want!r} for draw.block_hash")

    pin = repo_pin(suite)
    if pin is None:
        return p + [f"the drawn suite {suite!r} has no repo pin (a pinned revision in "
                    "benchmarks/cascade_benchmark/datasets.py AND a pinned item table): its "
                    "identity can't be checked, so the evidence is refused; run on a block "
                    "whose draw is a pinned suite, or land a pin first"]
    if ev.get("pin") != pin:
        p.append(f"pin {ev.get('pin')!r} != the repo pin for {suite!r} {pin!r}")

    params = ev.get("params") or {}
    if params.get("alpha") != cfg.alpha:
        p.append(f"params.alpha != [lane_d] alpha {cfg.alpha}")
    if params.get("tau_public") != cfg.tau_public:
        p.append(f"params.tau_public != [lane_d] tau_public {cfg.tau_public}")
    if params.get("B") != cfg.bootstrap_B:
        p.append(f"params.B != [lane_d] bootstrap_B {cfg.bootstrap_B}")
    if not isinstance(params.get("seed"), str) or not params.get("seed"):
        p.append("params.seed missing")

    rows = ev.get("rows") or {}
    if not isinstance(rows, dict) or sorted(rows) != sorted(ROLES):
        return p + [f"rows must be exactly {list(ROLES)}"]
    for role in ROLES:
        rec = rows[role]
        if not isinstance(rec, dict):
            return p + [f"rows.{role} is not an object"]
        if rec.get("session") != session:
            p.append(f"rows.{role}.session != session (one host session only)")
        if not isinstance(rec.get("digest"), str) or not HEX64.fullmatch(rec["digest"]):
            p.append(f"rows.{role}.digest is not 64 hex")
        entry = (rec.get("suites") or {}).get(suite) or {}
        items = _entry_items(entry)
        if entry.get("status") != "ok" or len(items) < MIN_ITEMS:
            p.append(f"rows.{role}: drawn suite {suite!r} not ok with >= {MIN_ITEMS} items")
        if entry.get("data_revision") != pin["version"]:
            p.append(f"rows.{role}: drawn suite data_revision {entry.get('data_revision')!r} "
                     f"!= the pinned revision {pin['version']!r}")
        if len(items) != len(set(items)) or not all(isinstance(i, str) for i in items) \
                or _sha(sorted(set(items))) != pin["items_sha256"]:
            p.append(f"rows.{role}: drawn suite items are not the repo-pinned item set "
                     f"({len(set(map(str, items)))} items vs {pin['n_items']} pinned)")
        if sentinel is not None and sentinel not in (rec.get("suites") or {}):
            p.append(f"rows.{role}: sentinel {sentinel!r} not swept")
    digs = [rows[r].get("digest") for r in ROLES]
    if digs[0] in digs[1:]:
        p.append("candidate digest equals a reference digest")

    refs = ev.get("refs")
    if not isinstance(refs, dict) or sorted(refs) != sorted(ROLES):
        p.append(f"refs must be exactly {list(ROLES)} (registry refs repo@sha256:<64 hex>)")
    else:
        for role in ROLES:
            m = REF.fullmatch(refs[role]) if isinstance(refs[role], str) else None
            if m is None:
                p.append(f"refs.{role} {refs[role]!r} is not <repo>@sha256:<64 hex>")
            elif m.group(1) != rows[role].get("digest"):
                p.append(f"refs.{role} does not pin rows.{role}.digest")

    try:
        gate = paired_gate(rows["candidate"], rows["king_d"], rows["house"], suite=suite,
                           alpha=params.get("alpha"), tau_public=params.get("tau_public"),
                           B=params.get("B"), seed=params.get("seed"), sentinel=sentinel,
                           pinned_identity=pin)
    except (ValueError, TypeError, KeyError) as e:
        return p + [f"paired_gate refused the recorded rows: {e}"]
    if json.loads(json.dumps(gate)) != ev.get("gate"):
        p.append("gate != paired_gate recomputed from the recorded rows with the repo pin "
                 "(hand-edited)")
    if gate.get("outcome") not in ("pass", "gate_regression"):
        p.append(f"gate outcome {gate.get('outcome')!r}: the pod run must compute a verdict")
    ident = gate.get("identity") or {}
    for role in ROLES:
        if (ident.get(role) or {}).get("digest") != pin["digest"]:
            p.append(f"gate.identity.{role} is not the pinned identity")
    if sentinel is not None and not (gate.get("sentinel") or {}).get("lcb"):
        p.append("the sentinel suite was not swept to numbers")
    if require_live:
        p += verified_live_problems(ev, check_git=check_git)
    return p


def verified_live_problems(ev: dict, *, check_git: bool = True) -> list[str]:
    """The ``verified_live`` record (README "The live verify"), offline."""
    rec = ev.get("verified_live")
    if rec is None:
        return ["verified_live: missing: the evidence was never re-checked against the chain "
                "and the registry (run `python -P -m cascade.laned.bench_evidence verify "
                "--evidence evidence/w3b2.json`); synthetic evidence never gets one"]
    if not isinstance(rec, dict):
        return ["verified_live is not an object"]
    miss = [k for k in VERIFY_KEYS if k not in rec]
    if miss:
        return [f"verified_live: missing keys {miss}"]
    out: list[str] = []
    if rec["schema"] != VERIFY_SCHEMA:
        out.append(f"verified_live.schema != {VERIFY_SCHEMA}")
    if rec["verified"] is not True:
        out.append("verified_live.verified is not true")
    if rec["body_sha256"] != body_sha256(ev):
        out.append("verified_live.body_sha256 is not the sha256 of the evidence body: the "
                   "evidence changed after verification")
    d = ev.get("draw") or {}
    for k in ("network", "netuid", "block", "block_hash"):
        if rec[k] != d.get(k):
            out.append(f"verified_live.{k} {rec[k]!r} != draw.{k} {d.get(k)!r}")
    fin = rec["finalized_block"]
    if isinstance(fin, bool) or not isinstance(fin, int) or not isinstance(d.get("block"), int) \
            or fin < d["block"]:
        out.append("verified_live.finalized_block is not >= draw.block (unfinalized draw)")
    if rec["suite"] != ev.get("suite") or rec["sentinel"] != ev.get("sentinel"):
        out.append("verified_live.suite / sentinel != the evidence's")
    elif (isinstance(rec["block_hash"], str) and HASH.fullmatch(rec["block_hash"])
          and draw(rec["block_hash"], rec["sentinel"]) != rec["suite"]):
        out.append("verified_live.suite is not gate_suite_for(verified_live.block_hash)")
    refs = ev.get("refs") or {}
    if rec["refs"] != refs:
        out.append("verified_live.refs != the evidence refs")
    ptr = rec["pointers"]
    if not isinstance(ptr, dict) or any(ptr.get(r) != refs.get(r) for r in REFS):
        out.append("verified_live.pointers (KING_D / the house at draw.block) != refs.king_d / "
                   "refs.house")
    rows = ev.get("rows") or {}
    res = rec["resolved"]
    for role in ROLES:
        dg = (rows.get(role) or {}).get("digest")
        if not isinstance(res, dict) or res.get(role) != f"sha256:{dg}":
            out.append(f"verified_live.resolved.{role} != sha256:<rows.{role}.digest>")
    try:
        at = datetime.fromisoformat(rec["verified_at"])
        if at.tzinfo is None:
            out.append("verified_live.verified_at has no timezone")
        elif at > datetime.now(UTC):
            out.append("verified_live.verified_at is in the future")
    except (TypeError, ValueError):
        out.append("verified_live.verified_at is not ISO 8601")
    out += [f"verified_live.{x}" for x in _tool_problems(rec["tool"], module_path=MODULE_PATH,
                                                          check_git=check_git, module=MODULE)]
    return out


# ── synthetic evidence (the forgery the verifier must refuse) ────────────────


def _block_hash_for(suite: str, sentinel) -> str:
    i = 0
    while True:
        h = "0x" + hashlib.sha256(f"w3b2 block {i}".encode()).hexdigest()
        if draw(h, sentinel) == suite:
            return h
        i += 1


def _rows(items: list[str], tag: str, factor: tuple[float, float], noise: float = 0.01
          ) -> list[dict]:
    base = random.Random("w3b2 rv5 base")
    rng = random.Random(f"w3b2 rv5 noise {tag}")
    out = []
    for item in items:
        c, m = math.exp(base.gauss(-0.7, 0.6)), math.exp(base.gauss(0.1, 0.4))
        out.append({"item": item, "crps": c * factor[0] * math.exp(rng.gauss(0.0, noise)),
                    "mase": m * factor[1] * math.exp(rng.gauss(0.0, noise))})
    return out


def digest(name: str) -> str:
    return hashlib.sha256(f"cascade w3b2 rv5 checkpoint {name}".encode()).hexdigest()


def synthetic_body(*, suite: str = "gift", sentinel: str = "time",
                   king: tuple[float, float] = (1.03, 1.03),
                   house: tuple[float, float] | None = None,
                   house_is_king: bool = True) -> dict:
    """A body that is self-consistent by construction: real pinned item ids
    and revision, well-formed refs, a block hash whose draw is ``suite``, and
    ``gate`` recomputed from the made-up numbers. It is exactly what an
    offline forger can write, so with no ``verified_live`` it must be
    REFUSED. (``house_is_king``: while lane D is unarmed the house holds
    KING_D, so the two references are one checkpoint.)"""
    from cascade.validator.lane_d_bench import paired_gate

    cfg = lane_d_cfg()
    pin = repo_pin(suite)
    assert pin is not None, f"no repo pin for {suite}"
    items = pinned_items(suite)
    session = "valpriv-4090/2026-11-22T03:00:00Z/1"
    dg = {"candidate": digest("candidate"), "king_d": digest("king_d"),
          "house": digest("king_d") if house_is_king else digest("house")}
    factors = {"candidate": (1.0, 1.0), "king_d": king,
               "house": king if house_is_king else (house or king)}
    sen_items = [f"{sentinel}/task-{i:03d}" for i in range(24)]
    recs = {}
    for role in ROLES:
        tag = "king_d" if (house_is_king and role == "house") else role
        recs[role] = {"session": session, "digest": dg[role], "suites": {
            suite: {"status": "ok", "rows": _rows(items, f"{suite} {tag}", factors[role]),
                    "data_revision": pin["version"]},
            sentinel: {"status": "ok", "rows": _rows(sen_items, f"{sentinel} {tag}", (1.0, 1.0))}}}
    params = {"alpha": cfg.alpha, "tau_public": cfg.tau_public, "B": cfg.bootstrap_B,
              "seed": SEED}
    gate = paired_gate(recs["candidate"], recs["king_d"], recs["house"], suite=suite,
                       sentinel=sentinel, pinned_identity=pin, **params)
    return {"schema": SCHEMA, "wp": "W3b2", "source": "pod", "host": "valpriv-4090",
            "tool": {"git_sha": "1" * 39 + "2", "git_dirty": False}, "session": session,
            "draw": {"network": NETWORK, "netuid": netuid(), "block": 9612000,
                     "block_hash": _block_hash_for(suite, sentinel)},
            "suite": suite, "sentinel": sentinel, "pin": pin, "params": params,
            "refs": {r: ref_for(dg[r]) for r in ROLES},
            "rows": recs, "gate": json.loads(json.dumps(gate)), "passed": True}


def record_for(ev: dict, *, finalized: int | None = None) -> dict:
    """A ``verified_live`` record as the owner's verify would write it for
    ``ev`` (used to test the offline record check; the real one comes from
    ``bench_evidence.verify_live``)."""
    d = ev["draw"]
    refs = ev.get("refs") or {}
    blk = d["block"] if isinstance(d["block"], int) else 0
    return {"schema": VERIFY_SCHEMA, "verified": True, "body_sha256": body_sha256(ev),
            "network": d["network"], "netuid": d["netuid"], "block": d["block"],
            "block_hash": d["block_hash"],
            "finalized_block": blk + 3 if finalized is None else finalized,
            "suite": ev["suite"], "sentinel": ev["sentinel"], "refs": copy.deepcopy(refs),
            "pointers": {r: refs.get(r) for r in REFS},
            "resolved": {r: f"sha256:{ev['rows'][r]['digest']}" for r in ROLES},
            "verified_at": "2026-11-22T04:00:00+00:00",
            "tool": {"module": MODULE, "git_sha": "3" * 40, "git_dirty": False}}


# ── read-only fakes for the live verify (external services) ──────────────────


class _ReadOnly:
    """Logs every call; any attribute outside ``_allowed`` is a forbidden use
    (a write, a reconnect, a stake call…) and is recorded in ``forbidden``."""

    _allowed: tuple[str, ...] = ()

    def __init__(self):
        object.__setattr__(self, "calls", [])
        object.__setattr__(self, "forbidden", [])

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self.forbidden.append(name)
        raise AttributeError(f"{type(self).__name__}.{name} is not a read the verify may use")


class FakeChain(_ReadOnly):
    """``ChainClient``-shaped: ``block_hash(block)`` (raises for an unknown
    block, like ``ChainError``) and ``finalized_block()``."""

    def __init__(self, hashes: dict[int, str], finalized: int):
        super().__init__()
        self.hashes, self.finalized = dict(hashes), finalized

    def block_hash(self, block=None):
        self.calls.append(("block_hash", block))
        if block not in self.hashes:
            raise RuntimeError(f"get_block_hash_failed: no block {block}")
        return self.hashes[block]

    def finalized_block(self):
        self.calls.append(("finalized_block",))
        return self.finalized


class FakeRegistry(_ReadOnly):
    """``resolve(ref)`` -> ``"sha256:<hex>"`` of the manifest the ref pins, or
    ``None`` when the registry doesn't hold it."""

    def __init__(self, published):
        super().__init__()
        self.published = set(published)

    def resolve(self, ref):
        self.calls.append(("resolve", ref))
        if ref not in self.published:
            return None
        return "sha256:" + ref.rsplit("@sha256:", 1)[1]


class FakePointers(_ReadOnly):
    """``at_block(block)`` -> ``{"house": ref, "king_d": ref}``: lane A's king
    checkpoint (the house) and KING_D in force at ``block``."""

    def __init__(self, table: dict[int, dict]):
        super().__init__()
        self.table = {k: dict(v) for k, v in table.items()}

    def at_block(self, block):
        self.calls.append(("at_block", block))
        return dict(self.table[block])


def live_world(ev: dict):
    """The chain, registry and pointers that really hold ``ev``'s facts."""
    d = ev["draw"]
    chain = FakeChain({d["block"]: d["block_hash"]}, d["block"] + 5)
    registry = FakeRegistry(ev["refs"].values())
    pointers = FakePointers({d["block"]: {r: ev["refs"][r] for r in REFS}})
    return chain, registry, pointers


# ── the module, offline ──────────────────────────────────────────────────────


def bench_evidence_module(monkeypatch):
    """``cascade.laned.bench_evidence`` imported in-process with every socket
    connect refused and ``bittensor`` / ``boto3`` unimportable: the verify
    uses only the adapters it is handed."""
    import importlib

    def _blocked(*a, **k):
        raise OSError("W3b2 test: network is blocked in-process")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)
    for pkg in ("bittensor", "boto3", "botocore"):
        monkeypatch.setitem(sys.modules, pkg, None)
    return importlib.import_module(MODULE)


_SITECUSTOMIZE = '''\
import os, socket
def _blocked(*a, **k):
    raise OSError("W3b2 test: network is blocked; a refusal must not open a connection")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
_m = os.environ.get("W3B2_NET_BLOCK_MARKER")
if _m:
    open(_m, "w").write("blocked")
'''


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """``python -P -m cascade.laned.bench_evidence ARGS`` with bittensor,
    boto3 and botocore unimportable and every socket connect refused."""
    sandbox = tmp_path / "w3b2_sandbox"
    for pkg in ("bittensor", "boto3", "botocore"):
        (sandbox / pkg).mkdir(parents=True, exist_ok=True)
        (sandbox / pkg / "__init__.py").write_text(
            f"raise ImportError('W3b2 test: {pkg} is blocked; a refusal comes first')\n")
    (sandbox / "sitecustomize.py").write_text(_SITECUSTOMIZE)
    env = {k: v for k, v in os.environ.items()
           if not (k.startswith("HIPPIUS_") or k.startswith("AWS_") or k.startswith("BT_"))}
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        env.pop(k, None)
    env["PYTHONPATH"] = os.pathsep.join([str(sandbox), str(REPO)])
    marker = tmp_path / "w3b2_net_block_marker"
    env["W3B2_NET_BLOCK_MARKER"] = str(marker)
    proc = subprocess.run([sys.executable, "-P", "-m", MODULE, *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=180, check=False)
    assert marker.is_file(), "the network block (sitecustomize) did not load in the CLI run"
    return proc


def write_json(path: Path, data) -> Path:
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path
