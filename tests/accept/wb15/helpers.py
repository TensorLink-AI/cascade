"""Shared spec and helpers for the WB15 acceptance tests (row claims and
overlap deferral, LB2 "Overlaps").

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB15 row (§3.0
lane B track), §12b.2 LB2 ("Overlaps", "Both modes coexist"), the T-rows
TB16 / TB21 (§4), the build rules (§3.1) and the RV4 note 9 (WB14 stores
declared rows, nothing reads them until WB15).

The scenarios run the REAL lane B machinery end to end, without torch:

- funding: the real ``FundingIntake.lane_b_fund`` (WB14: the
  ``X-Lane-B-Declared-Rows`` header, parsed and stored on the real
  ``cascade.funding.queue.LaneBQueue`` file);
- planning and dispatch: the real ``LaneBDriver.tick`` over the real
  ``expert_lane.LaneBQueue``;
- status: the real ``LaneBDriver.status`` and ``publish_lane_b_status``
  (``status/lane_b.json``).

Stand-ins, all for external services or the GPU boundary (build rule 2):

- ``Ops``: the chain's commitments, storage and pod dispatch (the driver's
  injected ``LaneBOps``; a dispatched leg stays open until the test returns
  it through the driver's own ``done`` callback);
- ``StubProbe``: the king forward over the affinity slice.
  ``expert_lane.probe_target_routing`` loads a checkpoint with torch, so it
  is swapped for a torch-free probe whose ESFT ``select`` ranks experts by a
  fixed affinity (row 0 highest on every layer) and records the exclusions
  it was planned under;
- the reveal oracle and the signature verifier of the intake;
- an in-memory object store for ``publish_public_json``.

No module in WB15's PATHS is imported at module level, so a missing name is
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

ROWS_HELD = "rows_held"
HEADER = "X-Lane-B-Declared-Rows"

# The chain's two consecutive era kings. Their routing is identical (the
# stub probe ignores the checkpoint), so the opening gate's saturation test
# passes once the lineage clock has ``open_tokens``.
PREV_KING = "cascade/king-prev@sha256:" + hashlib.sha256(b"wb15 prev king").hexdigest()
KING = "cascade/king@sha256:" + hashlib.sha256(b"wb15 king").hexdigest()

# ESFT width of the stub probe: rows per layer it selects.
ESFT_K = 6

# Hotkeys sort A < B < C < E (the tie rule is reveal block, then hotkey).
HK_A = "5HotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
HK_B = "5HotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
HK_C = "5HotkeyCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
HK_E = "5HotkeyEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE"
HK_F = "5HotkeyFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"

# Declared rows, keyed by MoE ordinal (WB14). Layer 0 has 128 experts
# (clamp 4..19 at the live fractions), layer 1 has 256 (8..38).
ROWS_A = {0: [0, 1, 2, 3, 4, 5]}
ROWS_B = {0: [4, 5, 6, 7, 8, 9]}            # overlaps A on layer 0 rows 4, 5
ROWS_C = {1: [20, 21, 22, 23, 24, 25, 26, 27]}  # disjoint from A and B


def ref_of(hotkey: str) -> str:
    return f"miner/gen-{hotkey[7:9].lower()}@sha256:" + hashlib.sha256(
        hotkey.encode()).hexdigest()


# ── torch guard ──────────────────────────────────────────────────────────────


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(
                f"WB15: '{name}' must not be imported (row claims are planning, not "
                "model code; the king forward is the stub probe)", name=name)
        return None


@contextmanager
def torch_blocked():
    blocker = _TorchBlocker()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)


# ── model geometry (WB14's real reader) ──────────────────────────────────────


def experts_per_layer() -> tuple[int, ...]:
    from cascade.trainer.declared_rows import model_experts_per_layer

    return tuple(model_experts_per_layer())


# ── the king-forward stand-in ────────────────────────────────────────────────


def _routing_probe_class():
    from cascade.trainer.expert_lane import RoutingProbe

    class StubProbe(RoutingProbe):
        """A king's routing without torch: identical top-k for every king
        (so the gate saturates) and an ESFT ``select`` that takes the
        ``ESFT_K`` highest-affinity rows per layer (row 0 highest) that are
        not excluded. Every call's exclusions are appended to ``calls``."""

        def __init__(self, target: str, calls: list) -> None:
            epl = experts_per_layer()
            super().__init__(topk=[np.zeros((8, 2), dtype=np.int64) for _ in epl],
                             affinity=np.zeros((len(epl), max(epl)), dtype=np.int64))
            self.target = target
            self.calls = calls

        def select(self, cfg, exclude=None):
            ex = {int(k): {int(r) for r in v} for k, v in (exclude or {}).items()}
            self.calls.append({"target": self.target,
                               "exclude": {k: sorted(v) for k, v in ex.items() if v}})
            out = {}
            for li, n in enumerate(experts_per_layer()):
                out[li] = [r for r in range(n) if r not in ex.get(li, set())][:ESFT_K]
            return out

    return StubProbe


def esft_pick(exclude: dict) -> dict[int, list[int]]:
    """What the stub's ESFT selects under ``exclude`` (the expected plan)."""
    ex = {int(k): {int(r) for r in v} for k, v in (exclude or {}).items()}
    return {li: [r for r in range(n) if r not in ex.get(li, set())][:ESFT_K]
            for li, n in enumerate(experts_per_layer())}


@contextmanager
def stub_king_forward(calls: list):
    """Swap ``expert_lane.probe_target_routing`` (the torch king forward) for
    the stub for the duration of the block, then restore it."""
    from cascade.trainer import expert_lane

    probe_cls = _routing_probe_class()
    real = expert_lane.probe_target_routing

    def fake(king_dir, contexts, cfg, target):
        return probe_cls(target, calls)

    expert_lane.probe_target_routing = fake
    try:
        yield
    finally:
        expert_lane.probe_target_routing = real


# ── external services ────────────────────────────────────────────────────────


class Ops:
    """The driver's injected I/O: chain commitments, storage, pod dispatch."""

    miner_funded = True

    def __init__(self, tmp: Path, fq) -> None:
        self.tmp = tmp
        self.fq = fq
        self.rows: list = []
        self.dispatched: list = []          # (leg, done) — legs stay open

    def funded_queue(self):
        return self.fq

    def hotkey_busy(self, hotkey: str) -> bool:
        return False

    def commitments(self) -> list:
        return list(self.rows)

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


def lane_b_cfg():
    """The lane B config through the real parser: two domains, room for
    several legs per domain (so capacity never decides a test), the live
    clamp fractions."""
    from cascade.shared.config import parse_lane_b

    return parse_lane_b({"from_block": 1, "domains": ["energy", "web"], "open_tokens": 100,
                         "routing_jaccard": 0.7, "legs_per_domain": 4,
                         "max_base_age_eras": 2})


class Scenario:
    """One lane B deployment: intake, funding queue, driver, chain rows."""

    def __init__(self, tmp: Path) -> None:
        from cascade.funding.intake import FundingIntake
        from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
        from cascade.funding.vault import PayerKeyVault
        from cascade.trainer.lane_b_driver import LaneBDriver

        self.tmp = Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.lb = lane_b_cfg()
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
        self.cfg = SimpleNamespace(lane_b=self.lb)
        self.driver = LaneBDriver(self.cfg, self.ops, self.tmp / "lane_b_driver.json")
        self.calls: list = []
        self.block = 0

    # chain + intake

    def fund(self, hotkey: str, reveal_block: int, rows=None, *, domain: str = "energy",
             uid: int | None = None) -> None:
        """The miner commits (``metro-v1:expert:<domain>``), the commit reveals
        at ``reveal_block``, then the miner funds through the real intake,
        with ``rows`` as the declared-rows header (None = no header: ESFT)."""
        from cascade.shared.expert_manifest import format_expert_commit

        ref = ref_of(hotkey)
        self.ops.rows = [r for r in self.ops.rows if r.hotkey != hotkey]
        self.ops.rows.append(SimpleNamespace(
            hotkey=hotkey, uid=len(self.ops.rows) + 10 if uid is None else uid,
            commit_block=int(reveal_block), payload=format_expert_commit(domain, ref)))
        self.reveals[hotkey] = int(reveal_block)
        headers = {"X-Miner-Hotkey": hotkey, "X-Commit-Ref": ref,
                   "X-Lane-B-Targets": domain, "X-Lium-Api-Key": "sk-" + hotkey[7:12]}
        if rows is not None:
            headers[HEADER] = json.dumps({str(k): v for k, v in rows.items()})
        self.clock.t += 1.0
        status, body = self.intake.lane_b_fund(headers)
        assert status == 202, (status, body)

    def entry_ref(self, hotkey: str, domain: str = "energy") -> str:
        from cascade.funding.queue import lane_b_entry_ref

        return lane_b_entry_ref((domain,), ref_of(hotkey))

    # trainer

    def tick(self, era_index: int = 1, king: str = KING) -> list:
        self.block += 1
        with torch_blocked(), stub_king_forward(self.calls):
            return self.driver.tick(self.block, era_index, king)

    def open_gate(self) -> list:
        """Era 0 sees PREV_KING, era 1 sees KING: the second tick opens the
        gate and plans every funded entry (one evaluation)."""
        self.driver.add_lineage_tokens(1000)
        assert self.tick(0, PREV_KING) == []
        started = self.tick(1, KING)
        assert self.driver.st.gate_opened_block > 0, self.driver.st.routing
        return started

    def open_legs(self) -> dict:
        return {leg.sub.hotkey: leg for leg in self.driver.queue.open}

    def return_leg(self, hotkey: str, *, pointer: str | None = None, failure=None) -> None:
        """The pod hands the leg back through the driver's own callback."""
        for leg, done in self.ops.dispatched:
            if leg.sub.hotkey == hotkey and leg in self.driver.queue.open:
                if pointer is not None:
                    done(leg, pointer, "corpus", self.block, "cpu")
                else:
                    done(leg, None, failure=failure)
                return
        raise AssertionError(f"no open dispatched leg for {hotkey[:9]}")

    def esft_calls(self, target: str = "energy") -> list[dict]:
        return [c for c in self.calls if c["target"] == target and c["exclude"]]

    # status

    def status(self) -> dict:
        with torch_blocked():
            return self.driver.status()

    def published(self) -> dict:
        """``status/lane_b.json`` as the real publisher writes it."""
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
        "status/lane_b.json carries a 'deferred' list (WB15 output column: "
        f"`deferred: rows_held`); got {rows!r}")
    return rows


def deferred_for(status: dict, hotkey: str) -> dict:
    hits = [r for r in deferred_rows(status) if r.get("hotkey") == hotkey]
    assert len(hits) == 1, f"expected one deferred row for {hotkey[:9]}, got {hits!r}"
    return hits[0]


def norm(sel: dict) -> dict[int, list[int]]:
    """A selection with int keys, sorted rows and empty layers dropped."""
    return {int(k): sorted(int(r) for r in v) for k, v in sel.items() if v}


def rows_of(sel: dict) -> set[tuple[int, int]]:
    return {(int(k), int(r)) for k, v in sel.items() for r in v}


def summary(tmp: str) -> dict:
    """A fixed TB16 + TB21 scenario, reduced to its planning outcome (used by
    the hash-seed test, which runs it in fresh interpreters)."""
    sc = Scenario(Path(tmp))
    sc.fund(HK_E, 5)
    sc.fund(HK_B, 20, ROWS_B)
    sc.fund(HK_C, 30, ROWS_C)
    sc.fund(HK_A, 10, ROWS_A)
    sc.open_gate()
    st = sc.status()
    return {
        "open": {hk: {str(k): v for k, v in norm(leg.selection).items()}
                 for hk, leg in sorted(sc.open_legs().items())},
        "deferred": sorted(
            [{"hotkey": r.get("hotkey"), "deferred": r.get("deferred"),
              "holder": r.get("holder"), "expected_release": r.get("expected_release")}
             for r in (st.get("deferred") or [])], key=lambda r: str(r["hotkey"])),
    }


if __name__ == "__main__":
    print(json.dumps(summary(sys.argv[1]), sort_keys=True))
