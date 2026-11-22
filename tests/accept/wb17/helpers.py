"""Shared spec and helpers for the WB17 acceptance tests (the lane B opening
rule, D41 / LB4).

Everything here comes from docs/plans/LANE_D_PLAN.md: the WB17 row (§3.0
lane B track), §12b.4 LB4, D41 (§6 and the §17.4 register), the T-rows
TB17 / TB18 (§4), the build rules (§3.1) and the L4 rule (§17.1).

The scenarios run the REAL lane B machinery without torch, the same way
WB15's and WB18's do:

- funding: the real ``FundingIntake.lane_b_fund`` into the real
  ``cascade.funding.queue.LaneBQueue`` file, with the commitment built by the
  real ``format_expert_targets``;
- the config: the real ``parse_lane_b`` (``[lane_b] open_rule``, WB10);
- the gate and dispatch: the real ``LaneBDriver.tick`` over its real state
  file.

Stand-ins, all for external services or the GPU boundary (build rule 2):

- ``Ops``: chain commitments, storage, pod dispatch and **``king_check()``**,
  the latest lane A king check the validators stamped on their signed
  receipt (DEC-CA-0055 ``king_check`` record; production reads it through the
  runner's verified receipt read). A dispatched leg stays open;
- ``StubProbe``: the king forward (``expert_lane.probe_target_routing`` loads
  a checkpoint with torch). Every king routes identically, so the routing
  co-gate saturates whenever it is measured;
- the reveal oracle and the signature verifier of the intake.

No module in WB17's PATHS is imported at module level, so a missing name is
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

# §12b.4 LB4: the three rule forms; "cogate" is the shipped default.
RULE_TOKENS_KING_CHECK = "tokens_king_check"
RULE_TOKENS = "tokens"
RULE_COGATE = "cogate"
RULES = (RULE_TOKENS_KING_CHECK, RULE_TOKENS, RULE_COGATE)

# The typed reasons a closed gate records (``gate_closed_reason``).
TOKENS_SHORT = "tokens_short"
KING_CHECK_NOT_PASSED = "king_check_not_passed"     # result "fail" / "deferred"
KING_CHECK_UNAVAILABLE = "king_check_unavailable"   # no readable king check
CLOSED_REASONS = (TOKENS_SHORT, KING_CHECK_NOT_PASSED, KING_CHECK_UNAVAILABLE)

OPEN_TOKENS = 100
ERA_LEN = 1000

PREV_KING = "cascade/king-prev@sha256:" + hashlib.sha256(b"wb17 prev king").hexdigest()
KING = "cascade/king@sha256:" + hashlib.sha256(b"wb17 king").hexdigest()
KING_HOTKEY = "5KingKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK"

ESFT_K = 6

HK_A = "5HotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
HK_B = "5HotkeyBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"

EVIDENCE_SCHEMA = "laned_wb17_gate/1"
TESTNET_NETUID = 259


def ref_of(hotkey: str) -> str:
    return f"miner/gen-{hotkey[7:9].lower()}@sha256:" + hashlib.sha256(
        hotkey.encode()).hexdigest()


def era_start(era: int) -> int:
    return int(era) * ERA_LEN


def king_check(result: str, *, round_id: str = "v2-r0100", king: str = KING,
               block: int = 900) -> dict:
    """A DEC-CA-0055 ``king_check`` record as the signed receipt carries it."""
    return {"result": result, "round_id": round_id, "block": int(block),
            "king_pointer": king, "king_hotkey": KING_HOTKEY, "lcb": 0.0125,
            "margin": 0.0, "reasons": [] if result == "pass" else ["lcb_below_margin"]}


def strict_loads(text: str):
    """``json.loads`` that refuses NaN / Infinity (strict JSON)."""
    def bad(token):
        raise ValueError(f"non-strict JSON constant {token}")

    return json.loads(text, parse_constant=bad)


def is_strict_json(value) -> bool:
    """``value`` is plain JSON: it survives a strict dump + load unchanged
    (no tuples, sets, numpy scalars, NaN or non-string keys)."""
    try:
        text = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return strict_loads(text) == value


# ── torch guard ──────────────────────────────────────────────────────────────


class _TorchBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError(
                f"WB17: '{name}' must not be imported (the opening rule is planning, "
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
        """Identical routing for every king (the co-gate saturates) and an
        ESFT ``select`` taking the ``ESFT_K`` lowest unexcluded rows."""

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
    """The driver's injected I/O: chain commitments, the validators' latest
    king check (``king_check()``), storage, pod dispatch."""

    miner_funded = True

    def __init__(self, tmp: Path, fq) -> None:
        self.tmp = tmp
        self.fq = fq
        self.rows: list = []
        self.dispatched: list = []
        self.check: dict | None = None
        self.check_reads = 0

    def funded_queue(self):
        return self.fq

    def hotkey_busy(self, hotkey: str) -> bool:
        return False

    def commitments(self) -> list:
        return list(self.rows)

    def king_check(self) -> dict | None:
        self.check_reads += 1
        return None if self.check is None else json.loads(json.dumps(self.check))

    def block_hash(self, block: int) -> str:
        raise AssertionError("WB17 arms no sentinel: the driver must not read a block hash")

    def contexts(self, domain: str, block: int) -> list:
        rng = np.random.default_rng(len(domain))
        return [np.cumsum(rng.normal(size=32)) for _ in range(16)]

    def fetch(self, pointer: str) -> Path:
        return self.tmp / "ckpt"

    def dispatch_leg(self, leg, era_index: int, block: int, done) -> None:
        self.dispatched.append((leg, done))


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


# ── the scenario ─────────────────────────────────────────────────────────────


def lane_b_raw(rule: str | None) -> dict:
    raw = {"from_block": 1, "domains": ["energy", "web"], "open_tokens": OPEN_TOKENS,
           "routing_jaccard": 0.7, "legs_per_domain": 4, "max_base_age_eras": 20}
    if rule is not None:
        raw["open_rule"] = rule
    return raw


def lane_b_cfg(rule: str | None):
    """The lane B config through the real parser (``None`` = the key absent,
    i.e. the shipped default)."""
    from cascade.shared.config import parse_lane_b

    return parse_lane_b(lane_b_raw(rule))


def chain_cfg(lb):
    from cascade.shared.config.rounds import RoundConfig

    v2 = SimpleNamespace(sentinel_from_block=0, sentinel_suites=("gift", "boom", "time", "fev"),
                         sentinel_ratchet=True, sentinel_tau=0.01)
    return SimpleNamespace(lane_b=lb, v2=v2,
                           round=RoundConfig(epoch_blocks=ERA_LEN, era_settlements=1))


class Scenario:
    """One lane B deployment: intake, funding queue, driver, chain."""

    def __init__(self, tmp: Path, rule: str | None, *, clock_t: float = 1_000_000.0) -> None:
        from cascade.funding.intake import FundingIntake
        from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
        from cascade.funding.vault import PayerKeyVault

        self.tmp = Path(tmp)
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.lb = lane_b_cfg(rule)
        self.cfg = chain_cfg(self.lb)
        self.clock = Clock(clock_t)
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

    def fund(self, hotkey: str, reveal_block: int, target: str = "energy") -> None:
        from cascade.shared.expert_manifest import format_expert_targets

        ref = ref_of(hotkey)
        payload = format_expert_targets([target], ref, tuple(self.lb.targets), max_targets=1)
        self.ops.rows = [r for r in self.ops.rows if r.hotkey != hotkey]
        self.ops.rows.append(SimpleNamespace(hotkey=hotkey, uid=len(self.ops.rows) + 10,
                                             commit_block=int(reveal_block), payload=payload))
        self.reveals[hotkey] = int(reveal_block)
        headers = {"X-Miner-Hotkey": hotkey, "X-Commit-Ref": ref,
                   "X-Lane-B-Targets": target, "X-Lium-Api-Key": "sk-" + hotkey[7:12]}
        self.clock.t += 1.0
        status, body = self.intake.lane_b_fund(headers)
        assert status == 202, (status, body)

    def tokens(self, n: int) -> None:
        self.driver.add_lineage_tokens(int(n))

    def tick(self, era: int, king: str = KING) -> list:
        """One scheduler tick inside ``era`` (blocks keep increasing)."""
        self.block = max(self.block + 1, era_start(era) + 1)
        assert self.block < era_start(era + 1), "fixture ran out of blocks in the era"
        with torch_blocked(), stub_king_forward():
            return self.driver.tick(self.block, int(era), king)

    def open_legs(self) -> dict:
        return {leg.sub.hotkey: leg for leg in self.driver.queue.open}

    def entry(self, hotkey: str):
        e = self.fq.get(hotkey)
        assert e is not None, f"no funded lane B entry for {hotkey[:9]}"
        return e

    def on_disk(self) -> dict:
        """The persisted driver state, read as strict JSON."""
        return strict_loads(self.state_path.read_text(encoding="utf-8"))


def state_field(st, name: str):
    """``LaneBDriverState.<name>``, as an assertion when the field is absent."""
    from dataclasses import fields

    names = {f.name for f in fields(type(st))}
    assert name in names, (
        f"LaneBDriverState has no '{name}' field (WB17: the driver state records "
        f"gate_rule, gate_evidence and a typed gate_closed_reason); fields: {sorted(names)}")
    return getattr(st, name)


# ── the testnet evidence verifier (TB17 / TB18 on netuid 259) ────────────────


def evidence_problems(ev) -> list[str]:
    """Every reason the testnet evidence file is not acceptable ([] = ok).

    Schema ``laned_wb17_gate/1``::

        {"schema": "laned_wb17_gate/1", "source": "testnet", "netuid": 259,
         "open_rule": "tokens_king_check", "open_tokens": <int>,
         "king_check": <the receipt king_check record the gate read>,
         "closed": <lane_b_driver.json while the gate was closed (TB18)>,
         "opened": <lane_b_driver.json after the gate opened (TB17)>}
    """
    p: list[str] = []
    if not isinstance(ev, dict):
        return ["evidence is not a JSON object"]
    if not is_strict_json(ev):
        p.append("evidence is not strict JSON")
    if ev.get("schema") != EVIDENCE_SCHEMA:
        p.append(f"schema must be {EVIDENCE_SCHEMA!r}; got {ev.get('schema')!r}")
    if ev.get("source") != "testnet":
        p.append(f"source must be 'testnet' (a live testnet run); got {ev.get('source')!r}")
    if ev.get("netuid") != TESTNET_NETUID:
        p.append(f"netuid must be {TESTNET_NETUID}; got {ev.get('netuid')!r}")
    if ev.get("open_rule") != RULE_TOKENS_KING_CHECK:
        p.append(f"open_rule must be {RULE_TOKENS_KING_CHECK!r} (D41 a, decided); "
                 f"got {ev.get('open_rule')!r}")
    ot = ev.get("open_tokens")
    if not isinstance(ot, int) or isinstance(ot, bool) or ot <= 0:
        p.append(f"open_tokens must be a positive int; got {ot!r}")
        ot = None
    kc = ev.get("king_check")
    if not isinstance(kc, dict) or kc.get("result") != "pass" or not kc.get("round_id"):
        p.append(f"king_check must be a passing receipt record with a round_id; got {kc!r}")
        kc = None

    closed = ev.get("closed")
    if not isinstance(closed, dict):
        p.append("closed: the driver state while the gate was closed is missing")
    else:
        if closed.get("gate_opened_block") != 0:
            p.append(f"closed.gate_opened_block must be 0; got {closed.get('gate_opened_block')!r}")
        if closed.get("gate_closed_reason") not in CLOSED_REASONS:
            p.append(f"closed.gate_closed_reason must be one of {list(CLOSED_REASONS)}; "
                     f"got {closed.get('gate_closed_reason')!r}")
        if closed.get("gate_rule", "") != "" or closed.get("gate_evidence", {}) != {}:
            p.append("closed: gate_rule / gate_evidence must be empty while closed")

    opened = ev.get("opened")
    if not isinstance(opened, dict):
        p.append("opened: the driver state after the gate opened is missing")
        return p
    ob = opened.get("gate_opened_block")
    if not isinstance(ob, int) or isinstance(ob, bool) or ob <= 0:
        p.append(f"opened.gate_opened_block must be a positive block; got {ob!r}")
    if opened.get("gate_rule") != ev.get("open_rule"):
        p.append(f"opened.gate_rule {opened.get('gate_rule')!r} != the configured "
                 f"open_rule {ev.get('open_rule')!r}")
    if opened.get("gate_closed_reason", None) != "":
        p.append(f"opened.gate_closed_reason must be ''; got {opened.get('gate_closed_reason')!r}")
    gev = opened.get("gate_evidence")
    if not isinstance(gev, dict) or not gev:
        p.append(f"opened.gate_evidence must be a non-empty object; got {gev!r}")
        return p
    lt = gev.get("lineage_tokens")
    if not isinstance(lt, int) or isinstance(lt, bool):
        p.append(f"gate_evidence.lineage_tokens must be an int; got {lt!r}")
    elif ot is not None and lt < ot:
        p.append(f"gate_evidence.lineage_tokens {lt} < open_tokens {ot}: the rule was not met")
    if ot is not None and gev.get("open_tokens") != ot:
        p.append(f"gate_evidence.open_tokens {gev.get('open_tokens')!r} != open_tokens {ot}")
    gkc = gev.get("king_check")
    if not isinstance(gkc, dict) or gkc.get("result") != "pass":
        p.append(f"gate_evidence.king_check must record the passing check; got {gkc!r}")
    elif kc is not None:
        for key in ("round_id", "king_pointer"):
            if gkc.get(key) != kc.get(key):
                p.append(f"gate_evidence.king_check.{key} {gkc.get(key)!r} != the receipt "
                         f"record's {kc.get(key)!r}")
    if isinstance(opened.get("lineage_tokens"), int) and isinstance(lt, int) \
            and opened["lineage_tokens"] < lt:
        p.append("opened.lineage_tokens is below the evidence's lineage_tokens (edited file)")
    return p


def good_evidence() -> dict:
    """An evidence file the verifier must accept (built here, offline; the
    real one comes from the testnet trainer's lane_b_driver.json)."""
    kc = king_check("pass", round_id="v2-r0042", block=41_000)
    return {
        "schema": EVIDENCE_SCHEMA, "source": "testnet", "netuid": TESTNET_NETUID,
        "open_rule": RULE_TOKENS_KING_CHECK, "open_tokens": OPEN_TOKENS,
        "king_check": kc,
        "closed": {"gate_opened_block": 0, "gate_rule": "", "gate_evidence": {},
                   "gate_closed_reason": KING_CHECK_NOT_PASSED, "lineage_tokens": 150},
        "opened": {"gate_opened_block": 41_012, "gate_rule": RULE_TOKENS_KING_CHECK,
                   "gate_closed_reason": "", "lineage_tokens": 150,
                   "gate_evidence": {"lineage_tokens": 150, "open_tokens": OPEN_TOKENS,
                                     "king_check": dict(kc)}},
    }
