"""Shared spec and fixture builders for the W17 acceptance tests
(`status/lane_d.json` + the W18 vault ledger `laned_vault_ledger/1`).

Copied from docs/plans/LANE_D_PLAN.md (rev6; rev6.2 item (e), PR #636):

- W17 row: "`status/lane_d.json` producer (queue, pending progress, king,
  board; rev3: per-entry reason enum including `coldkey_cap_settlement`,
  `confirm_pending` and reveal deadline)". rev6.2 (e), owner 2026-11-22:
  "The W18 vault ledger `laned_vault_ledger/1` (current era, entries with
  statuses / `closed_era`, open disputes) is produced by W17."
- §11.3 Support: "`status/lane_d.json` shows each pending entry's
  eligible-window progress against the floors."
- rev6.10 (a) / rev6.12 (a): the fixture feeds (W7c's chain) pin
  `embargo_hours = 48`, so progress counts embargo windows (data time in
  `(b - e, b]`, released after `b`) as well as fresh ones.
- §1.3b Progress board: "every king's suite numbers (gate and the era's
  sentinel, suite named) go into the signed lane D index (§11.1) and
  `status/lane_d.json`." §16 rule 4 / W-S3: "W17 should embed
  `build_sentinel_board`'s output there."
- §1.2: "Credentials and URLs are never logged, printed or put on a command
  line"; §3.2 invariant 1: unarmed, nothing changes.

Every state and report comes from REAL `settle_lane_d` runs on the W7c
synthetic chain (its pool snapshots and forecasts, numpy only, no torch).
The bench pod and the public bucket are the only stand-ins. Nothing here
imports a module in PATHS at import time; every call into the module under
test happens inside a test body.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from functools import cache
from pathlib import Path

from tests.accept.w7c import helpers as w7c

REPO = Path(__file__).resolve().parents[3]
MODULE = "cascade.validator.lane_d_status"

STATUS_KEY = "status/lane_d.json"
LEDGER_SCHEMA = "laned_vault_ledger/1"

ERA = 3600                      # mainnet era: 900-block settlements x 4
SETTLEMENT = 900
ARM_BLOCK = w7c.ARM_BLOCK       # 9_720_000, an era boundary
ARM_ERA = ARM_BLOCK // ERA      # 2700
BLOCKS_PER_DAY = 7200           # 86400 s / 12 s blocks (cascade BLOCK_SECONDS)

# Settlement boundaries used by the fixtures. B1 and B2 belong to the era
# that starts at ARM_BLOCK; B2_NEXT_ERA to the next one (ARM_BLOCK + ERA).
B1 = ARM_BLOCK + SETTLEMENT
B2 = ARM_BLOCK + 2 * SETTLEMENT
B2_NEXT_ERA = ARM_BLOCK + ERA + SETTLEMENT

OPEN_STATE_STATUSES = ("pending", "confirm_pending", "hold")
QUEUE_ROW_KEYS = {"digest", "hotkey", "coldkey", "context", "status", "reason",
                  "ready_block", "reveal_block", "reveal_ts", "reveal_deadline_block",
                  "confirm_stage", "progress"}
PROGRESS_KEYS = {"windows", "clusters", "min_windows", "min_clusters", "frozen"}
KING_KEYS = {"digest", "hotkey", "house", "since_boundary", "tenure_blocks"}
DOC_KEYS = {"schema", "lane", "block", "era", "boundary", "reason_enum", "floors", "king",
            "house", "prior_kings", "queue", "board", "sentinel"}
BOARD_KEYS = {"boundary", "crowned", "judged", "skipped", "verdicts", "gate"}
VERDICT_KEYS = {"digest", "king_digest", "stage", "outcome", "lcb", "gain"}
WALL_CLOCK_KEYS = ("as_of", "generated_at", "timestamp", "updated_at", "published_at")
LEDGER_KEYS = {"schema", "era", "entries", "disputes"}
LEDGER_ENTRY_KEYS = {"hotkey", "digest", "status", "closed_era"}

# (state status, state reason) -> ledger status (README "Ledger statuses").
CLOSED_TO_LEDGER = {
    "deregistered": "voided",
    "ready_expired": "ready_expired",
    "checkpoint_fault": "closed_unjudged",
    "lane_closed": "closed_unjudged",
    "coldkey_unresolved": "closed_unjudged",
    "coldkey_changed": "closed_unjudged",
    "reveal_delay_exceeded": "closed_unjudged",
    "duplicate_digest": "closed_unjudged",
    "hotkey_has_open_entry": "closed_unjudged",
    "context_not_armed": "closed_unjudged",
}

SECRET_ENV = {"HIPPIUS_S3_ACCESS_KEY": "w17-access-marker-7Q2",
              "HIPPIUS_S3_SECRET_KEY": "w17-secret-marker-9Z4",
              "HF_TOKEN": "w17-hf-marker-3K8"}


def strict_loads(text: str):
    """``json.loads`` that refuses NaN / Infinity and duplicate keys."""
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")

    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError(f"duplicate key {k!r}")
            out[k] = v
        return out
    return json.loads(text, parse_constant=bad, object_pairs_hook=pairs)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def era_hash(tag: str) -> str:
    return "0x" + hashlib.sha256(f"w17 era {tag}".encode()).hexdigest()


def body_sha(report: dict) -> str:
    return w7c.body_sha(report)


# ── configuration: the shipped chain.toml, [lane_d] armed ──────────────────


@cache
def _shipped() -> str:
    return (REPO / "chain.toml").read_text(encoding="utf-8")


ARMED_LANE_D = {"from_block": ARM_BLOCK, "share": 0.25, "takes_from": "lane_a",
                "remainder": "burn", "contexts": [4096, 8192, 16384],
                "horizon": w7c.HORIZON, "forward_margin_hours": w7c.MARGIN_H,
                "max_mirror_lag_hours": 6.0, "min_windows": 24, "min_clusters": 4,
                "bootstrap_B": 200}


def chain_cfg(*, armed: bool = True, sentinel: bool = False, **lane_over):
    """The shipped mainnet config (real loader parse). ``armed`` arms
    ``[lane_d]`` at :data:`ARM_BLOCK` with the W7c floors (24 windows / 4
    sources); ``sentinel`` sets ``[v2] sentinel_from_block = ARM_BLOCK``
    (W-S1). ``lane_over`` overrides ``[lane_d]`` keys."""
    from cascade.shared.config import parse_chain_config

    raw = tomllib.loads(_shipped())
    ld = dict(raw.get("lane_d") or {})
    if armed:
        ld.update(ARMED_LANE_D)
    ld.update(lane_over)
    raw["lane_d"] = ld
    v2 = dict(raw.get("v2") or {})
    if sentinel:
        v2["sentinel_from_block"] = ARM_BLOCK
    raw["v2"] = v2
    return parse_chain_config(raw)


# ── the external services ───────────────────────────────────────────────────


class MemoryStore:
    """The public bucket: the ``put_text`` / ``get_text`` duck type of
    ``cascade.shared.hippius.S3Store`` (an absent key raises
    ``ObjectNotFound``). Every call is logged."""

    def __init__(self):
        self.objects: dict[str, str] = {}
        self.puts: list[tuple[str, str, dict]] = []
        self.gets: list[str] = []

    def put_text(self, key: str, text: str, **kw) -> None:
        self.puts.append((key, text, dict(kw)))
        self.objects[key] = text

    def get_text(self, key: str) -> str:
        from cascade.shared.hippius import ObjectNotFound

        self.gets.append(key)
        if key not in self.objects:
            raise ObjectNotFound(f"s3_get_missing: {key}")
        return self.objects[key]

    def calls(self) -> int:
        return len(self.puts) + len(self.gets)


class DownHost:
    """A bench pod that can't be reached (every sweep raises)."""

    def __call__(self, digests, suites):
        raise ConnectionError("w17 fixture: bench pod unreachable")


# ── the chain (W7c pools + forecasts; settle_lane_d is real) ────────────────


def _entry(name: str, cold: str, reveal_ts: int, ready_block: int, reveal_block: int) -> dict:
    return {"hotkey": w7c.fixture_key(f"w17-{name}").ss58_address,
            "digest": w7c.digest(f"w17 entry {name}"), "context": 4096,
            "coldkey_at_ready": w7c.fixture_key(f"w17-{cold}").ss58_address,
            "ready_block": ready_block, "reveal_block": reveal_block, "reveal_ts": reveal_ts}


class Fixture:
    """One synthetic lane D chain under ``root``.

    Entries (reveal order X < L < Y < E < D < Z):

    - X (coldkey C1): sharp forecasts; clears.
    - Y (coldkey C1, after X): over ``max_judged_per_coldkey_per_settlement``
      = 1 at B1, so ``coldkey_cap_settlement`` and still pending.
    - Z (coldkey C2): revealed too late for any window of the snapshots:
      ``pending_floor`` with zero progress.
    - L (coldkey C3): the house's quality: ``lost_duel`` at B1.
    - E: readied more than ``ready_ttl_blocks`` before its reveal:
      ``ready_expired`` (closed at registration).
    - D: hotkey not registered: ``deregistered`` (voided).
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.base = w7c.Chain(self.root / "w7c")
        self.house = dict(self.base.house)
        a, t = ARM_BLOCK, w7c.T_X
        self.x = dict(self.base.x)
        self.y = _entry("y", "x-cold-shared", t + 60, a + 41, a + 61)
        self.y["coldkey_at_ready"] = self.x["coldkey_at_ready"]
        self.z = _entry("z", "z-cold", t + 5 * 3600, a + 42, a + 62)
        self.l = _entry("l", "l-cold", t + 30, a + 43, a + 63)
        self.e = _entry("e", "e-cold", t + 90, a - 8000, a + 64)
        self.d = _entry("d", "d-cold", t + 120, a + 45, a + 65)
        every = [r for recs in self.base.records for r in recs]
        self.forecasts = dict(self.base.forecasts)
        self.forecasts[self.l["digest"]] = w7c._model_forecasts(every, "bad", "w17-l")
        self.uid_of = dict(self.base.uid_of)
        for n, e in enumerate((self.y, self.z, self.l, self.e), start=20):
            self.uid_of[e["hotkey"]] = n
        self.signer = self.base.signer
        self.pools = self.base.pools

    def names(self) -> dict[str, str]:
        return {e["digest"]: n for n, e in (("x", self.x), ("y", self.y), ("z", self.z),
                                            ("l", self.l), ("e", self.e), ("d", self.d))}

    def state_path(self, run: str) -> Path:
        return self.base.state_path(run)

    def settle(self, *, cfg, boundary: int, snapshot: int, entries: list[dict],
               state_path: Path, host=None, tag: str = ""):
        """``settle_lane_d`` at ``boundary`` on pool snapshot ``snapshot``
        (1-based; earlier snapshots retained), with the W5e gate inputs and a
        bench session on ``host`` (a passing pod by default)."""
        from cascade.validator.lane_d import settle_lane_d
        from cascade.validator.lane_d_bench import SweepSession

        if host is None:
            host = w7c.BenchHost({self.house["digest"]: 1.0, self.x["digest"]: 0.9})
        kw = {}
        if snapshot > 1:
            kw["prior_pool_dirs"] = list(self.pools[:snapshot - 1])
        return settle_lane_d(boundary_ts=2_000_000_000 + 12 * int(boundary), cfg=cfg.lane_d, boundary=boundary, pool_dir=self.pools[snapshot - 1],
                             entries=[dict(e) for e in entries], house=dict(self.house),
                             forecasts=self.forecasts, uid_of=dict(self.uid_of),
                             state_path=state_path, signer=self.signer,
                             boundary_block_hash=w7c.block_hash(f"w17 {tag} {boundary}"),
                             sentinel=None, bench=SweepSession(f"w17-{tag}-{boundary}", host),
                             **kw)


def read_state(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_report(state_path: Path, boundary: int) -> dict:
    from cascade.validator.lane_d_state import load_report_file, report_path

    return load_report_file(report_path(state_path, boundary))


def rich_b1(root: Path) -> dict:
    """The queue scenario: every fixture entry revealed before B1, settled
    once at B1 with ``max_judged_per_coldkey_per_settlement = 1``. The bench
    pod is down, so X's first clear (the crowning candidate, rev6.11 (b)) is
    held open: ``hold`` / ``bench_hold``."""
    fx = Fixture(root)
    cfg = chain_cfg(max_judged_per_coldkey_per_settlement=1)
    sp = fx.state_path("rich")
    s = fx.settle(cfg=cfg, boundary=B1, snapshot=1, state_path=sp, tag="rich",
                  entries=[fx.x, fx.y, fx.z, fx.l, fx.e, fx.d], host=DownHost())
    return {"fx": fx, "cfg": cfg, "state_path": sp, "state": read_state(sp),
            "report": s.report, "reports": [read_report(sp, B1)]}


def crowned(root: Path) -> dict:
    """The king scenario: L at B1 (lost), then X listed and crowned at its
    first verdict at B2_NEXT_ERA (the next era) on snapshot 2 (rev6.11 (b))."""
    fx = Fixture(root)
    cfg = chain_cfg()
    sp = fx.state_path("crowned")
    s1 = fx.settle(cfg=cfg, boundary=B1, snapshot=1, state_path=sp, tag="crowned",
                   entries=[fx.l])
    st1 = read_state(sp)
    s2 = fx.settle(cfg=cfg, boundary=B2_NEXT_ERA, snapshot=2, state_path=sp, tag="crowned",
                   entries=[fx.x, fx.l])
    return {"fx": fx, "cfg": cfg, "state_path": sp, "state_b1": st1, "report_b1": s1.report,
            "state": read_state(sp), "report": s2.report,
            "reports": [read_report(sp, B1), read_report(sp, B2_NEXT_ERA)]}


def held(root: Path) -> dict:
    """The bench-hold scenario: B1 lists no entry; X's first clear at B2
    with the bench pod down is held: ``hold`` / ``bench_hold`` (rev6.11 (b))."""
    fx = Fixture(root)
    cfg = chain_cfg()
    sp = fx.state_path("held")
    fx.settle(cfg=cfg, boundary=B1, snapshot=1, state_path=sp, tag="held", entries=[])
    s2 = fx.settle(cfg=cfg, boundary=B2, snapshot=2, state_path=sp, tag="held",
                   entries=[fx.x], host=DownHost())
    return {"fx": fx, "cfg": cfg, "state_path": sp, "state": read_state(sp),
            "report": s2.report, "reports": [read_report(sp, B1), read_report(sp, B2)]}


def progress_b1(root: Path) -> dict:
    """The progress scenario: floors of 30 windows / 4 sources, so X (24
    eligible windows over 4 sources in snapshot 1) stays ``pending_floor``."""
    fx = Fixture(root)
    cfg = chain_cfg(min_windows=30)
    sp = fx.state_path("progress")
    s = fx.settle(cfg=cfg, boundary=B1, snapshot=1, state_path=sp, tag="progress",
                  entries=[fx.x, fx.z])
    return {"fx": fx, "cfg": cfg, "state_path": sp, "state": read_state(sp),
            "report": s.report, "reports": [read_report(sp, B1)]}


# ── calls into the module under test (imported at call time) ────────────────


def mod():
    import importlib

    return importlib.import_module(MODULE)


def status(cfg, *, state, report, block, era_block=None, era_block_hash=None, store=None):
    return mod().build_lane_d_status(cfg, state=copy.deepcopy(state),
                                     report=copy.deepcopy(report), block=block,
                                     era_block=era_block, era_block_hash=era_block_hash,
                                     store=MemoryStore() if store is None else store)


def ledger(cfg, *, state, reports, block, disputes=()):
    return mod().build_vault_ledger(cfg, state=copy.deepcopy(state),
                                    reports=copy.deepcopy(list(reports)), block=block,
                                    disputes=list(disputes))


def row_of(doc: dict, digest: str) -> dict:
    rows = [r for r in doc["queue"] if r["digest"] == digest]
    assert len(rows) == 1, f"{digest[:8]} appears {len(rows)} times in the queue"
    return rows[0]


def ledger_entry(led: dict, digest: str) -> dict:
    rows = [r for r in led["entries"] if r["digest"] == digest]
    assert len(rows) == 1, f"{digest[:8]} appears {len(rows)} times in the ledger"
    return rows[0]


def all_keys(obj) -> set[str]:
    out: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            out |= all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= all_keys(v)
    return out
