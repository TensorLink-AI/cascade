"""Shared spec and helpers for the WB18 acceptance tests (sentinel interaction,
D35 / LB5).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB18 row (§3.0
lane B track), §12b.5 LB5, D35 (§6 and the §17.4 register), the T-rows
TB19 / TB20 / TS4 (§4), §16 (the sentinel draw, W-S1) and the build rules
(§3.1); and from DEC-CA-0071 §1 (suite targets ``suite:<suite>``, codes
``s-<suite>``).

The scenarios run the REAL lane B machinery end to end, without torch, the
same way WB15's do:

- funding: the real ``FundingIntake.lane_b_fund`` into the real
  ``cascade.funding.queue.LaneBQueue`` file, with the target in
  ``X-Lane-B-Targets`` and the commitment built by the real
  ``format_expert_targets``;
- planning and dispatch: the real ``LaneBDriver.tick`` over the real
  ``expert_lane.LaneBQueue``;
- the era's sentinel: the real W-S1 draw ``cascade.shared.sentinel.
  sentinel_for_era`` over the era-start block hash the ops return;
- status: the real ``LaneBDriver.status`` and ``publish_lane_b_status``
  (``status/lane_b.json``).

Stand-ins, all for external services or the GPU boundary (build rule 2):

- ``Ops``: chain commitments, the chain's block hashes (``block_hash``, the
  read the validators make as ``client.block_hash(era_start)``), storage and
  pod dispatch. A dispatched leg stays open;
- ``StubProbe``: the king forward (``expert_lane.probe_target_routing`` loads
  a checkpoint with torch);
- the reveal oracle and the signature verifier of the intake;
- an in-memory object store for ``publish_public_json``.

No module in WB18's PATHS is imported at module level, so a missing name is
reported inside the test that needs it.
"""

from __future__ import annotations

import hashlib
import importlib.abc
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PLAN = REPO / "docs" / "plans" / "LANE_D_PLAN.md"

SENTINEL_TARGET = "sentinel_target"
ROWS_HELD = "rows_held"
ROWS_HEADER = "X-Lane-B-Declared-Rows"
LOGGER = "cascade.trainer.lane_b"

# The four bench suites (§16 rule 1 default ``sentinel_suites``; DEC-CA-0071 §1
# suite targets ``suite:gift`` .. ``suite:fev``, codes ``s-gift`` .. ``s-fev``).
SUITES = ("gift", "boom", "time", "fev")
SUITE_TARGETS = tuple(f"suite:{s}" for s in SUITES)
GIFT, BOOM = "suite:gift", "suite:boom"

# Era grid of the fixture: 1000 blocks per era (one settlement per era). The
# sentinel arms from era 1 (era 0 is the gate-opening era with PREV_KING).
ERA_LEN = 1000
SENTINEL_FROM = ERA_LEN

PREV_KING = "cascade/king-prev@sha256:" + hashlib.sha256(b"wb18 prev king").hexdigest()
KING = "cascade/king@sha256:" + hashlib.sha256(b"wb18 king").hexdigest()
HASHSET = "sha256:" + hashlib.sha256(b"wb18 contamination hash set").hexdigest()

ESFT_K = 6

# Hotkeys sort A < B < C < E < F.
HK_A = "5HotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
HK_B = "5HotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
HK_C = "5HotkeyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
HK_E = "5HotkeyEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
HK_F = "5HotkeyFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"

ROWS_A = {0: [0, 1, 2, 3, 4, 5]}
ROWS_B = {0: [4, 5, 6, 7, 8, 9]}            # overlaps A on layer 0 rows 4, 5


def ref_of(hotkey: str) -> str:
    return f"miner/gen-{hotkey[7:9].lower()}@sha256:" + hashlib.sha256(
        hotkey.encode()).hexdigest()


def era_start(era: int) -> int:
    return int(era) * ERA_LEN


def hash_drawing(suite: str, era: int) -> str:
    """A ``0x`` + 64-hex block hash whose W-S1 draw over ``SUITES`` is
    ``suite`` (searched deterministically from the era number)."""
    from cascade.shared.sentinel import sentinel_for_era

    for i in range(10_000):
        h = "0x" + hashlib.sha256(f"wb18 era {era} try {i}".encode()).hexdigest()
        if sentinel_for_era(h, SUITES) == suite:
            return h
    raise AssertionError(f"no block hash draws {suite!r}")


# ── torch guard ──────────────────────────────────────────────────────────────


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(
                f"WB18: '{name}' must not be imported (the sentinel deferral is planning, "
                "not model code; the king forward is the stub probe)", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


# ── the king-forward stand-in ────────────────────────────────────────────────


def experts_per_layer() -> tuple[int, ...]:
    from cascade.trainer.declared_rows import model_experts_per_layer

    return tuple(model_experts_per_layer())


def _routing_probe_class():
    from cascade.trainer.expert_lane import RoutingProbe

    class StubProbe(RoutingProbe):
        """Identical routing for every king (the gate saturates) and an ESFT
        ``select`` taking the ``ESFT_K`` lowest unexcluded rows per layer."""

        def __init__(self, target: str) -> None:
            epl = experts_per_layer()
            super().__init__(topk=[np.zeros((8, 2), dtype=np.int64) for _ in epl],
                             affinity=np.zeros((len(epl), max(epl)), dtype=np.int64))
            self.target = target

        def select(self, cfg, exclude=None):
            ex = {int(k): {int(r) for r in v} for k, v in (exclude or {}).items()}
            return {li: [r for r in range(n) if r not in ex.get(li, set())][:ESFT_K]
                    for li, n in enumerate(experts_per_layer())}

    return StubProbe


@contextmanager
def stub_king_forward():
    from cascade.trainer import expert_lane

    probe_cls = _routing_probe_class()
    real = expert_lane.probe_target_routing

    def fake(king_dir, contexts, cfg, target):
        return probe_cls(target)

    expert_lane.probe_target_routing = fake
    try:
        yield
    finally:
        expert_lane.probe_target_routing = real


# ── external services ────────────────────────────────────────────────────────


class Ops:
    """The driver's injected I/O: chain commitments and block hashes,
    storage, pod dispatch."""

    miner_funded = True

    def __init__(self, tmp: Path, fq) -> None:
        self.tmp = tmp
        self.fq = fq
        self.rows: list = []
        self.dispatched: list = []
        self.hashes: dict[int, str] = {}      # era-start block -> block hash
        self.hash_reads: list[int] = []

    def funded_queue(self):
        return self.fq

    def hotkey_busy(self, hotkey: str) -> bool:
        return False

    def commitments(self) -> list:
        return list(self.rows)

    def block_hash(self, block: int) -> str:
        self.hash_reads.append(int(block))
        if int(block) not in self.hashes:
            raise RuntimeError(f"fixture chain has no hash for block {block}")
        return self.hashes[int(block)]

    def contexts(self, domain: str, block: int) -> list:
        rng = np.random.default_rng(len(domain))
        return [np.cumsum(rng.normal(size=32)) for _ in range(16)]

    def fetch(self, pointer: str) -> Path:
        return self.tmp / "ckpt"

    def dispatch_leg(self, leg, era_index: int, block: int, done) -> None:
        self.dispatched.append((leg, done))


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def put_text(self, key: str, text: str, **kw) -> None:
        self.objects[key] = text


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ── the scenario ─────────────────────────────────────────────────────────────


def lane_b_cfg(*, suite_targets: bool = True, max_defer: int | None = 2):
    """The lane B config through the real parser: two domains, room for many
    legs per target, a long base age (no leg expires in a test), suite
    targets armed (with the hash set pinned, as the parser requires) and
    ``sentinel_max_defer_eras`` (``None`` = the shipped default)."""
    from cascade.shared.config import parse_lane_b

    raw = {"from_block": 1, "domains": ["energy", "web"], "open_tokens": 100,
           "routing_jaccard": 0.7, "legs_per_domain": 4, "max_base_age_eras": 20}
    if suite_targets:
        raw.update({"suite_targets": True, "contamination_hashset": HASHSET})
    if max_defer is not None:
        raw["sentinel_max_defer_eras"] = int(max_defer)
    return parse_lane_b(raw)


def chain_cfg(lb, *, sentinel_from_block: int = SENTINEL_FROM):
    """The driver's config: ``[lane_b]``, the ``[v2]`` sentinel keys (W-S1)
    and the round grid (``ERA_LEN`` blocks per era)."""
    from cascade.shared.config.rounds import RoundConfig

    v2 = SimpleNamespace(sentinel_from_block=int(sentinel_from_block),
                         sentinel_suites=SUITES, sentinel_ratchet=True, sentinel_tau=0.01)
    return SimpleNamespace(lane_b=lb, v2=v2,
                           round=RoundConfig(epoch_blocks=ERA_LEN, era_settlements=1))


class Scenario:
    """One lane B deployment: intake, funding queue, driver, chain."""

    def __init__(self, tmp: Path, *, suite_targets: bool = True, max_defer: int | None = 2,
                 sentinel_from_block: int = SENTINEL_FROM) -> None:
        from cascade.funding.intake import FundingIntake
        from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
        from cascade.funding.vault import PayerKeyVault

        self.tmp = Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.lb = lane_b_cfg(suite_targets=suite_targets, max_defer=max_defer)
        self.cfg = chain_cfg(self.lb, sentinel_from_block=sentinel_from_block)
        self.clock = Clock()
        self.reveals: dict[str, int] = {}
        lane_a = FundedQueue(self.tmp / "queue.json", clock=self.clock)
        self.fq = LaneBQueue(lane_b_queue_path(self.tmp / "queue.json"), clock=self.clock)
        self.intake = FundingIntake(
            lane_a, PayerKeyVault(dir=None, clock=self.clock),
            resolve_reveal=lambda hk, ref: None, require_signature=False,
            verify=lambda hk, msg, s: False, clock=self.clock,
            lane_b_queue=self.fq, resolve_lane_b_reveal=lambda hk, t, r: self.reveals.get(hk),
            lane_b_targets=tuple(self.lb.targets), lane_b_max_targets=1, lane_b_cfg=self.lb)
        self.ops = Ops(self.tmp, self.fq)
        self.state_path = self.tmp / "lane_b_driver.json"
        self.driver = self.new_driver()
        self.block = 0

    def new_driver(self):
        """A driver over the same state file (a trainer restart)."""
        from cascade.trainer.lane_b_driver import LaneBDriver

        return LaneBDriver(self.cfg, self.ops, self.state_path)

    def restart(self) -> None:
        self.driver = self.new_driver()

    # chain + intake

    def set_sentinel(self, era: int, suite: str) -> None:
        """The chain's era-start block hash for ``era`` draws ``suite``."""
        self.ops.hashes[era_start(era)] = hash_drawing(suite, era)

    def fund(self, hotkey: str, reveal_block: int, target: str, rows=None) -> None:
        from cascade.shared.expert_manifest import format_expert_targets

        assert target in tuple(self.lb.targets), (
            f"{target!r} is not a lane B slot: with [lane_b] suite_targets = true the slot "
            f"list gains {list(SUITE_TARGETS)} (DEC-CA-0071 §1); got {self.lb.targets!r}")
        ref = ref_of(hotkey)
        try:
            payload = format_expert_targets([target], ref, tuple(self.lb.targets),
                                            max_targets=1)
        except ValueError as e:
            raise AssertionError(f"no lane B commitment for target {target!r}: {e}") from e
        self.ops.rows = [r for r in self.ops.rows if r.hotkey != hotkey]
        self.ops.rows.append(SimpleNamespace(hotkey=hotkey, uid=len(self.ops.rows) + 10,
                                             commit_block=int(reveal_block), payload=payload))
        self.reveals[hotkey] = int(reveal_block)
        headers = {"X-Miner-Hotkey": hotkey, "X-Commit-Ref": ref,
                   "X-Lane-B-Targets": target, "X-Lium-Api-Key": "sk-" + hotkey[7:12]}
        if rows is not None:
            headers[ROWS_HEADER] = json.dumps({str(k): v for k, v in rows.items()})
        self.clock.t += 1.0
        status, body = self.intake.lane_b_fund(headers)
        assert status == 202, (status, body)

    def entry_ref(self, hotkey: str, target: str) -> str:
        from cascade.funding.queue import lane_b_entry_ref

        return lane_b_entry_ref((target,), ref_of(hotkey))

    # trainer

    def tick(self, era: int, king: str = KING) -> list:
        """One scheduler tick inside ``era`` (blocks keep increasing)."""
        self.block = max(self.block + 1, era_start(era) + 1)
        assert self.block < era_start(era + 1), "fixture ran out of blocks in the era"
        with torch_blocked(), stub_king_forward():
            return self.driver.tick(self.block, int(era), king)

    def open_gate(self) -> list:
        """Era 0 sees PREV_KING, era 1 sees KING: the era-1 tick opens the gate
        and plans every funded entry."""
        self.driver.add_lineage_tokens(1000)
        assert self.tick(0, PREV_KING) == []
        started = self.tick(1, KING)
        assert self.driver.st.gate_opened_block > 0, self.driver.st.routing
        return started

    def open_legs(self) -> dict:
        return {leg.sub.hotkey: leg for leg in self.driver.queue.open}

    # status

    def status(self) -> dict:
        with torch_blocked():
            return self.driver.status()

    def published(self) -> dict:
        from cascade.trainer.lane_b_driver import LANE_B_STATUS_KEY, publish_lane_b_status

        store = MemoryStore()
        sched = SimpleNamespace(lane_b=self.driver,
                                r=SimpleNamespace(manifest_store=lambda: store))
        with torch_blocked():
            ok = publish_lane_b_status(sched, self.block, min_interval_s=0.0,
                                       clock=lambda: 0.0)
        assert ok and LANE_B_STATUS_KEY == "status/lane_b.json", "status/lane_b.json not published"
        return json.loads(store.objects[LANE_B_STATUS_KEY])


def deferred_rows(status: dict) -> list[dict]:
    rows = status.get("deferred")
    assert isinstance(rows, list), (
        f"status/lane_b.json carries a 'deferred' list; got {rows!r}")
    return rows


def deferred_for(status: dict, hotkey: str) -> dict:
    hits = [r for r in deferred_rows(status) if r.get("hotkey") == hotkey]
    assert len(hits) == 1, f"expected one deferred row for {hotkey[:9]}, got {hits!r}"
    return hits[0]


def norm(sel: dict) -> dict[int, list[int]]:
    return {int(k): sorted(int(r) for r in v) for k, v in sel.items() if v}


def summary(tmp: str) -> dict:
    """A fixed TB19 / TS4 / TB20 scenario reduced to its planning outcome per
    era (used by the hash-seed test in fresh interpreters)."""
    sc = Scenario(Path(tmp), max_defer=1)
    for era, suite in ((1, "gift"), (2, "gift"), (3, "boom")):
        sc.set_sentinel(era, suite)
    sc.fund(HK_C, 30, BOOM)
    sc.fund(HK_A, 10, GIFT, ROWS_A)
    sc.fund(HK_E, 5, "energy")
    sc.fund(HK_B, 20, GIFT)
    out: dict = {}
    sc.open_gate()
    for era in (1, 2):
        if era == 2:
            sc.tick(2)
        st = sc.status()
        out[str(era)] = {
            "open": sorted(sc.open_legs()),
            "deferred": sorted([[r.get("hotkey"), r.get("deferred"), r.get("entry_ref")]
                                for r in (st.get("deferred") or [])]),
        }
    return out


if __name__ == "__main__":
    print(json.dumps(summary(sys.argv[1]), sort_keys=True))
