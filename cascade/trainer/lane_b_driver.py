"""Lane B (DEC-CA-0050) orchestrator driver — the trainer's side of the
domain-expert competition, hooked into the rolling scheduler at three points:

* ``tick``     — every scheduler tick: read lane-B commitments
                 (``metro-v1:expert:<domain>:hippius:<ref>``) into the queue,
                 measure routing saturation (previous era's king vs the
                 current one), check the opening gate (a one-way latch), plan
                 the declared-row entries first (WB15: their own rows, in
                 reveal order; an overlap with held rows is deferred
                 uncharged, ``deferred: rows_held``) — a suite-target entry
                 on the era's §16 sentinel is deferred uncharged instead
                 (WB18, D35: ``deferred: sentinel_target``, at most
                 ``sentinel_max_defer_eras`` eras) — then an ESFT selection
                 on the CURRENT era king excluding every held row, and
                 dispatch at most ``legs_per_domain`` per domain;
* ``settle``   — after each settlement manifest is published: drop legs that
                 are era-stale or ``routing_stale`` (the current king routes
                 their domain to different experts than their base king),
                 carry every other finished leg onto that settlement's king
                 (async delta rebase), upload it, sign + publish the expert
                 manifest, judge it with the validators' own code, and queue
                 the passing legs for the next era's init;
* ``init_for`` — when an era rolls: the era init with every queued winning
                 delta applied (disjoint rows, so order-free) — every leg of
                 the new era, king and challengers, starts from the merged
                 weights.

Funding (DEC-CA-0059 item 8, ``[lane_b] leg_funding = "miner"``, the
default): a leg rents on the MINER's deposited key through the same intake,
vault and queue as lanes A and C — ``POST /v1/lane-b/fund`` queues a
:class:`~cascade.funding.queue.LaneBQueue` entry naming the declared targets
and the ref; a commitment dispatches only while its entry is funded, its
hotkey has no other leg in flight (lanes A/B/C, one per hotkey) and the
hotkey's latest commitment is still that entry. The leg runs the funded
challenger path (``_run_funded_leg`` with ``role="expert"``), never an
operator lane: a capacity miss requeues ``no_capacity`` unburned.

Expert rows are exclusive while a leg is open, finished-but-unsettled, or
queued as a pending merge until :meth:`LaneBDriver.init_for` consumes it
(DEC-CA-0050 "state at merge").

All I/O is injected (``LaneBOps``) so the policy is unit-testable and the
real wiring (``RunnerLaneBOps``) stays a thin adapter.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
import time
import types
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from ..shared.atomic import read_text_with_backup, write_text_atomic
from ..shared.era import era_for_block, image_record_at
from . import expert_lane as _tel
from . import leg_reattach as _reattach
from .expert_lane import (
    GATE_TOKENS_SHORT,
    OPEN_RULE_COGATE,
    LaneBQueue,
    LaneBSubmission,
    OpenLeg,
    RoutingProbe,
    gate_open,
    needs_king_check,
    open_rule_verdict,
    routing_metrics,
    routing_stale_union,
    settle_leg,  # noqa: F401 — the model's rebase calls it through this module
    submission_key,
    union_selection,
)
from .row_claims import (
    SENTINEL_HOLD,
    claim_key,
    deferred_row,
    first_overlap,
    has_capacity,
    leg_holder,
    merge_holder,
    normalise_rows,
    target_load,
)
from .runner_generation import leg_corpus

log = logging.getLogger("cascade.trainer.lane_b")

# outcomes of a funded leg that ended without a manifest entry (status rows)
FUNDED_REQUEUED, FUNDED_MINER_FAULT = "requeued", "miner_fault"


class LaneBOps(Protocol):
    def commitments(self) -> list:
        """The chain's Commitment rows."""

    def contexts(self, domain: str, block: int) -> list:
        """The domain's affinity slice."""

    def fetch(self, pointer: str) -> Path:
        """A local directory holding the checkpoint at ``pointer``."""

    def upload(self, ckpt_dir: Path, repo: str) -> str:
        """Upload ``ckpt_dir``; returns the trained pointer."""

    def dispatch_leg(self, leg: OpenLeg, era_index: int, block: int, done) -> None:
        """Run ``leg`` on a pod; ``done`` is called when it returns."""

    def bench_rows(self, ckpt_dir: Path) -> dict | None:
        """The checkpoint's public-benchmark rows (None = unavailable)."""

    def publish_expert_manifest(self, manifest) -> None:
        """Sign and publish the expert manifest."""

    def judge(self, manifest, round_id: str, block: int, sentinel: str | None = None,
              anchor_pointer: str | None = None) -> list:
        """The manifest's ExpertJudgement list; ``sentinel`` (W-S2) is the
        settlement era's §16 suite, passed only while armed; ``anchor_pointer``
        (WB26) the era's anchor king, passed only when the settling legs
        carry anchor rows (suite slots configured)."""

    def merge_scores(self, pointer: str, block: int) -> list:
        """The held-out acceptance scores of ``pointer``."""

    def block_hash(self, block: int) -> str:
        """The chain's ``0x`` hash of ``block`` (read only while the §16
        sentinel and suite targets are armed: the era-start draw, WB18)."""

    # DEC-CA-0059 item 8 (optional: an ops without them dispatches unfunded,
    # the pre-funding behaviour the unit fakes keep):
    #   miner_funded: bool                       — legs need a funded entry
    #   def funded_queue(self): ...              — funding LaneBQueue | None
    #   def hotkey_busy(self, hotkey) -> bool: ...  — a lane-A / lane-C leg runs
    # WB17 (D41 a; optional: absent reads as no readable check):
    #   def king_check(self) -> dict | None: ...  — the latest lane A king check
    #     of the validators' signed receipt (DEC-CA-0055), read only


@dataclass
class FinishedExpertLeg:
    leg: dict                  # asdict(OpenLeg)
    leg_pointer: str
    corpus_digest: str
    train_block: int
    gpu_name: str = ""
    # WB22 / WB25: GPUs the leg trained on (the rented layout); 0 = not recorded
    gpu_n: int = 0
    # W-IMG: the worker image digest the leg ran ("" = not recorded)
    train_image_digest: str = ""


@dataclass
class _PreparedInit:
    pointer: str
    merges: list
    dropped: list
    outcomes: list


@dataclass
class LaneBDriverState:
    queue: str = ""                                   # LaneBQueue JSON
    finished: list[dict] = field(default_factory=list)
    submission_results: list[dict] = field(default_factory=list)  # bounded qualification telemetry
    pending_merges: list[dict] = field(default_factory=list)  # {leg_pointer, base, selection}
    lineage_tokens: int = 0
    # Immutable seed-epoch bindings make the cross-file rollover replay safe.
    lineage_epochs: dict[str, dict] = field(default_factory=dict)
    lineage_meter_version: int = 1
    legacy_budget_tokens: int = 0
    # domain -> the WORST layer's ESFT-selection Jaccard, previous era's king
    # vs the current one (the full per-layer numbers live in ``routing``)
    jaccard: dict[str, float] = field(default_factory=dict)
    merged_inits: dict[str, dict] = field(default_factory=dict)     # era -> {pointer, base, merges}
    # settlements already settled -> {"digest": expert manifest digest | ""}:
    # a re-called settle for the same round (a retried publish) is a no-op
    # that returns the same binding
    settled: dict[str, dict] = field(default_factory=dict)
    # era index -> the latest king pointer seen in that era: the routing gate
    # compares the current king against the PREVIOUS era's king, never
    # against itself on an earlier tick
    kings: dict[str, str] = field(default_factory=dict)
    # the gate's latest saturation numbers: {era, block, king, prev_king,
    # lineage_tokens, domains: {d: {overlap: [per layer], jaccard: [per
    # layer], tokens}}, open, reason}
    routing: dict = field(default_factory=dict)
    # one row per (previous king, king) comparison — the lineage's per-layer
    # router-saturation curve (plot it against lineage_tokens)
    routing_history: list[dict] = field(default_factory=list)
    # block the gate latched open at (0 = closed). One-way: lane B never
    # re-closes — in-flight legs anchor on the king's routing, and later
    # drift is caught per leg by the rebase ``routing_stale`` check
    gate_opened_block: int = 0
    # WB17 (D41 / LB4): the ``[lane_b] open_rule`` the gate opened under and
    # the facts that satisfied it (strict JSON: lineage_tokens, open_tokens,
    # and the passing receipt king_check under ``tokens_king_check``). Written
    # once, at opening, and never rewritten (the latch above); "" / {} while
    # closed and for a state file that latched before WB17
    gate_rule: str = ""
    gate_evidence: dict = field(default_factory=dict)
    # while closed under a token rule: "tokens_short" | "king_check_not_passed"
    # | "king_check_unavailable" (the co-gate records "tokens_short" when the
    # lineage is short, else "" with its free-text reason in ``routing``)
    gate_closed_reason: str = ""
    # legs dropped at settlement without a verdict: {round_id, hotkey, uid,
    # domain, gen_ref, leg_pointer, base_king_ref, king_pointer, outcome
    # ("stale" | "routing_stale" | "merge_refused"), detail}
    outcomes: list[dict] = field(default_factory=list)
    # WB18 (D35 (a), TB20): submission key -> the distinct eras the entry was
    # deferred in because its suite target was that era's sentinel. Durable,
    # so a restart never resets the ``sentinel_max_defer_eras`` bound; pruned
    # to the submissions still pending or open
    sentinel_deferrals: dict[str, list[int]] = field(default_factory=dict)
    # 2026-11-26 re-attach (parity with lane A): leg key (:func:`_attach_key`)
    # -> what a restarted process needs to re-attach a miner-funded open leg
    # to its still-running pod worker instead of requeueing it: {hotkey,
    # entry_ref, attempt_id, dispatch_block, era_index, dispatched_at,
    # seen_at (the live process's heartbeat), base_seed, pod_name, tag,
    # leg_pointer_repo}. The pod is re-found by its ledgered name and the run
    # by its detached tag (both functions of the seed epoch and the hotkey),
    # exactly as a lane A re-attach. Dropped when the leg returns.
    attached: dict[str, dict] = field(default_factory=dict)


SETTLED_KEEP = 256
SENTINELS_KEEP = 8
# a sentinel draw the chain read could not produce this tick: every
# suite-target entry waits (uncounted) until the era's block hash is readable
SENTINEL_UNREADABLE = "?"
KINGS_KEEP = 16
HISTORY_KEEP = 512
OUTCOMES_KEEP = 256


LEG_LENGTH_HOLD = "leg_length"
# Wall the manifest needs after a leg ends (runner_capacity's
# FUNDED_PUBLISH_MARGIN_SECONDS); the default when no runner is wired.
_LEG_LENGTH_MARGIN_SECONDS = 1800.0


def _era_seconds(round_cfg, block: int) -> float:
    """Wall seconds of the era containing ``block`` (12 s blocks); 0 unknown."""
    try:
        from ..shared.era import era_length_blocks

        return float(era_length_blocks(round_cfg, int(block))) * 12.0
    except Exception:  # noqa: BLE001 — a stub round config: no long-leg rule
        return 0.0


def leg_length_admission(round_cfg, *, block_now: int, now: float, guard_seconds: int,
                         margin_seconds: float, current_era: int):
    """WB23 (TB26): the cross-era admission (``rolling.admit``) of a lane B leg
    whose wall is its ``train_hours`` guard. ``cross_era`` = it cannot land in
    the current era and waits for the next era's pre-train window."""
    from .rolling import admit

    return admit(round_cfg, block_now=int(block_now), now=float(now),
                 wall_seconds=float(guard_seconds), margin_seconds=float(margin_seconds),
                 current_era=int(current_era))


def _leg_spec(lb, funding: dict, block: int) -> dict:
    """WB27: the lane B leg spec the trainer and the worker derive the leg
    contract from (``lane_b_levers.lane_b_leg_contract``): the WB23 length
    and the levers effective at the dispatch ``block``. Sets
    ``funding["levers"]`` (what the expert entry records; absent = every
    default). ``{}`` = today's leg (no ``--recipe-json``)."""
    from ..shared.lane_b_levers import effective_levers, levers_from_dict

    spec: dict = {}
    if float(funding.get("train_hours", 0.0) or 0.0) > 0:
        spec.update(train_hours=float(funding["train_hours"]),
                    leg_steps=int(funding.get("leg_steps", 0) or 0),
                    leg_guard_seconds=int(funding.get("leg_guard_seconds", 0) or 0))
        if int(funding.get("train_tokens", 0) or 0) > 0:
            spec["train_tokens"] = int(funding["train_tokens"])
        if funding.get("gpu_measured") is False:
            spec["gpu_measured"] = False
    raw = funding.get("funded_levers") or {}
    levers = effective_levers(lb, levers_from_dict(dict(raw), lb) if raw else None, block)
    if levers.is_default:
        funding.pop("levers", None)
    else:
        funding["levers"] = levers.as_dict()
        spec["levers"] = levers.as_dict()
    return spec


def _funded_leg_length(funding: dict, base):
    """WB23: the funding record's ``LegLength`` (steps clamped to the base
    contract's reference steps, the consensus cap), or None for a leg funded
    without ``train_hours`` (today's ``for_hours`` contract)."""
    from ..shared.lane_b_leg import LegLength, reference_steps

    hours = float((funding or {}).get("train_hours", 0.0) or 0.0)
    if hours <= 0:
        return None
    steps = min(int(funding.get("leg_steps", 0) or 0), reference_steps(base, hours))
    return LegLength(train_hours=hours, steps=max(1, steps),
                     guard_seconds=int(funding.get("leg_guard_seconds", 0) or 0))


def _measured_rate_of(gpus, contract):
    """``sku -> steps/s`` at ``contract``'s context from lane B's WB19 table
    (measured at 4096; the WB27 context lever scales it like the intake
    does), ``None`` for an unmeasured type."""
    scale = 4096 / max(1, int(getattr(contract, "context_length", 4096) or 4096))

    def rate_of(sku: str):
        rate = gpus.rate(sku) if sku else None
        return None if rate is None else float(rate.steps_per_s) * scale

    return rate_of

def _dispatch_era(cfg, current, block: int, rec: dict | None):
    """The era a re-attached leg was DISPATCHED in (its ``dispatch_block``),
    as an era object the dispatch reads ``start_block`` / ``index`` from;
    ``current`` when that is the same era or cannot be derived. A leg that
    trains across an era boundary keeps its dispatch era's contract block,
    contract and GPU table — the identity its detached run is tagged by."""
    if block <= 0:
        return current
    try:
        spec = era_for_block(cfg.round, int(block))
        start = int(spec.start_block)
    except Exception:  # noqa: BLE001 — no era grid: the current era, as before
        return current
    if start == int(getattr(current, "start_block", start)):
        return current
    try:
        index = int((rec or {}).get("era_index", spec.index))
    except (TypeError, ValueError):
        index = int(spec.index)
    if is_dataclass(current) and not isinstance(current, type):
        return replace(current, index=index, start_block=start)
    return types.SimpleNamespace(**{**vars(current), "index": index, "start_block": start})


def _attach_key(leg: OpenLeg) -> str:
    """One open leg's identity in ``LaneBDriverState.attached``: the chain
    commitment it trains and the era it was planned on."""
    return f"{submission_key(leg.sub)}|{int(leg.base_era)}"


def _slice_digest(contexts: list) -> str:
    h = hashlib.sha256()
    for c in contexts:
        a = np.ascontiguousarray(np.asarray(c, dtype=np.float64).reshape(-1))
        h.update(len(a).to_bytes(8, "little"))
        h.update(a.tobytes())
    return h.hexdigest()


def _pauses_prepare(fn):
    """The prepare-ahead worker pauses while ``fn`` (the settlement) runs."""
    import functools

    @functools.wraps(fn)
    def wrapper(self, *a, **kw):
        self._settling.set()
        try:
            return fn(self, *a, **kw)
        finally:
            self._settling.clear()

    return wrapper


class LaneBDriver:
    def __init__(self, cfg, ops: LaneBOps, state_path: Path):
        self.cfg = cfg                 # ChainConfig
        self.ops = ops
        self.path = Path(state_path)
        self._lock = threading.Lock()
        # WB26: the anchor king's public bench rows, swept once per anchor
        self._anchor_rows: dict[str, dict] = {}
        self.st = self._load()
        self.queue = LaneBQueue.from_json(self.st.queue)
        # Routing probes (one king forward over one domain's affinity slice:
        # per-token top-k + ESFT counts), keyed (king, domain, slice digest),
        # and the slice drawn per (current king, domain). A probe loads a
        # checkpoint and runs the slice, so it runs once per king change and
        # NEVER under ``_lock``; ``_cache_lock`` guards only the dict ops.
        self._cache_lock = threading.Lock()
        self._slices: OrderedDict[tuple[str, str], tuple[str, list]] = OrderedDict()
        self._probes: OrderedDict[tuple[str, str, str], RoutingProbe] = OrderedDict()
        # Open legs persisted by an EARLIER process have no thread here (it died
        # with that process): requeued on this process's first tick.
        self._recovered = False
        # rounds the scheduler abandoned (settlement over its bound): settle
        # commits nothing for them any more (see :meth:`abandon`)
        self._abandoned: set[str] = set()
        # In-window settlement retries (owner 2026-11-28): the inputs the FIRST
        # attempt of a round read — {rid: {"legs": [leg_pointer...],
        # "lineage_tokens": n}} — so a retry of that round settles exactly the
        # legs (and lineage clock) an on-time success would have: a leg that
        # finishes during the retry wait settles with the next round
        self._settle_pins: dict[str, dict] = {}
        self._acceptance_jobs: dict = {}
        # Scientific expiry is not physical termination: a stale worker can
        # still own the hotkey's pod and harvest directory until its callback.
        self._running: dict[int, OpenLeg] = {}
        # WB15: the declared claims the latest tick deferred (``deferred:
        # rows_held`` rows of ``status/lane_b.json``); re-derived every tick
        self._deferred: list[dict] = []
        # WB18: era-start block -> the era's sentinel suite (one chain read per
        # era), and the (submission, era) pairs whose past-bound start is logged
        self._sentinels: OrderedDict[int, str] = OrderedDict()
        self._bound_logged: set[tuple[str, int]] = set()
        # 2026-11-26 re-attach: open legs the first tick keeps for a re-attach
        # (dispatched right after its first locked section), and the wall
        # clock heartbeats are stamped with (a test seam)
        self._to_reattach: list[tuple[OpenLeg, dict]] = []
        self.clock = _reattach._wall
        # 2026-11-27 prepare-ahead: set while a settlement runs (the
        # prepare-ahead worker pauses; the settlement's pod steps go first)
        self._settling = threading.Event()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> LaneBDriverState:
        raw, src = read_text_with_backup(self.path, json.loads)
        if not isinstance(raw, dict):     # absent/torn state (and no good backup) starts empty
            if src == "unreadable":
                log.error("lane B: driver state %s and its backup unreadable; starting empty",
                          self.path)
            return LaneBDriverState()
        if src == "backup":
            log.warning("lane B: driver state %s unreadable — restored from its .bak", self.path)
        known = {f.name for f in fields(LaneBDriverState)}
        extra = sorted(k for k in raw if k not in known)
        if extra:
            log.warning("lane B: driver state carries unknown key(s) %s — dropped", extra)
        # tolerate keys an older/newer build wrote (never drop the legs over it)
        state = LaneBDriverState(**{k: v for k, v in raw.items() if k in known})
        if "lineage_meter_version" not in raw:
            # Budget estimates cannot open a gate requiring actual processed
            # points. Keep the old number visible for operator reconciliation.
            state.legacy_budget_tokens = state.lineage_tokens
            state.lineage_tokens = 0
        return state

    def save(self) -> None:
        self.st.queue = self.queue.to_json()
        write_text_atomic(self.path, json.dumps(asdict(self.st), sort_keys=True), backup=True)

    def _recover_orphans(self, fq=None) -> None:
        """First tick of this process: every open leg on file belongs to a
        process that is gone (a leg's thread never outlives it), so it would
        hold its ``legs_per_domain`` slot and its expert rows forever.
        Requeue each one at its FIFO place, uncharged — a funded leg's queue
        entry too (``no_capacity``, never a burn). Caller holds the lock."""
        if self._recovered:
            return
        self._recovered = True
        orphans = list(self.queue.open)
        reattach = self._reattach_candidates()
        for leg in orphans:
            key = _attach_key(leg)
            if key in reattach:
                # 2026-11-26: re-attach (lane A parity) — the leg stays open,
                # its funded entry stays in flight (same attempt, no new
                # charge); the funded path adopts the live ledgered pod and
                # attaches to the running worker, or refuses (requeued then)
                log.warning("lane B: open leg %s/%s (base era %d) has no thread in this process "
                            "— re-attaching to its pod %s (heartbeat %.0fs ago)",
                            leg.sub.hotkey[:8], leg.sub.domain, leg.base_era,
                            reattach[key].get("pod_name", "?"),
                            self.clock() - float(reattach[key].get("seen_at", 0.0)))
                self._to_reattach.append((leg, dict(reattach[key])))
                continue
            rec = self.st.attached.pop(key, None)
            log.warning("lane B: open leg %s/%s (base era %d) has no thread in this process "
                        "— requeued uncharged%s", leg.sub.hotkey[:8], leg.sub.domain,
                        leg.base_era,
                        "" if rec is None else " (re-attach grace expired or unsupported)")
            self.queue.requeue(leg)
            if leg.funding and fq is not None:
                fq.requeue(leg.sub.hotkey, error="trainer restarted mid-leg — requeued",
                           error_class="no_capacity", burn_attempt=False, operator_billed=True)
        if orphans:
            self.save()

    def _reattach_candidates(self) -> dict[str, dict]:
        """Persisted open legs a restarted process may re-attach: miner-funded,
        with a re-attach record whose heartbeat is within ``[round]
        dispatch_reattach_grace_seconds``, and ops that can re-attach."""
        if not callable(getattr(self.ops, "reattach_leg", None)):
            return {}
        grace = _reattach.grace_seconds(self.cfg)
        now = self.clock()
        out = {}
        for leg in self.queue.open:
            key = _attach_key(leg)
            rec = self.st.attached.get(key)
            if leg.funding and _reattach.fresh(rec, now, grace):
                out[key] = rec
        return out

    def _heartbeat(self) -> None:
        """Stamp ``seen_at`` on every leg this process runs (at most every
        :data:`~cascade.trainer.leg_reattach.HEARTBEAT_SECONDS`) and drop
        records of legs no longer open. Caller holds the lock and saves."""
        if not self._recovered:
            return
        now = self.clock()
        live = {_attach_key(leg) for leg in self._running.values()}
        open_keys = {_attach_key(leg) for leg in self.queue.open}
        for key in [k for k in self.st.attached if k not in open_keys]:
            del self.st.attached[key]
        for key in live:
            rec = self.st.attached.get(key)
            if rec is not None and now - float(rec.get("seen_at", 0.0) or 0.0) \
                    >= _reattach.HEARTBEAT_SECONDS:
                rec["seen_at"] = float(now)

    def _record_attach(self, leg: OpenLeg, era_index: int, block: int) -> None:
        """Persist what a restart needs to re-attach ``leg`` (miner-funded
        legs only; the operator pool path keeps today's requeue)."""
        if not leg.funding or not callable(getattr(self.ops, "reattach_leg", None)):
            return
        ident: dict = {}
        fn = getattr(self.ops, "leg_identity", None)
        if callable(fn):
            try:
                ident = dict(fn(leg) or {})
            except Exception as e:  # noqa: BLE001 — a missing identity only loses re-attach
                log.warning("lane B: leg %s identity unavailable (%s) — re-attach not "
                            "recorded", leg.sub.hotkey[:8], e)
                return
        now = float(self.clock())
        self.st.attached[_attach_key(leg)] = {
            "hotkey": leg.sub.hotkey, "entry_ref": str(leg.funding.get("entry_ref", "") or ""),
            "attempt_id": str(leg.funding.get("attempt_id", "") or ""),
            "dispatch_block": int(block), "era_index": int(era_index),
            "dispatched_at": now, "seen_at": now, **ident}

    def _dispatch_reattach(self) -> list[OpenLeg]:
        """Re-attach the legs :meth:`_recover_orphans` kept open. Outside the
        lock (a synchronous failure calls :meth:`_leg_done`)."""
        with self._lock:
            todo, self._to_reattach = self._to_reattach, []
            # a leg the same tick expired or requeued is not re-attached
            todo = [(leg, rec) for leg, rec in todo if leg in self.queue.open]
            for leg, _ in todo:
                self._running[id(leg)] = leg
        for leg, rec in todo:
            try:
                self.ops.reattach_leg(leg, int(rec.get("era_index", leg.base_era)),
                                      int(rec.get("dispatch_block", 0) or 0), self._leg_done, rec)
            except Exception as e:  # noqa: BLE001 — a failed re-attach requeues like before
                self._leg_done(leg, None, failure=_reattach.refusal_failure(
                    f"re-attach dispatch failed: {e}"))
        return [leg for leg, _ in todo]

    # ── funding (DEC-CA-0059 item 8) ─────────────────────────────────────────

    def _miner_funded(self) -> bool:
        return bool(getattr(self.ops, "miner_funded", False))

    def _funded_queue(self):
        fq = getattr(self.ops, "funded_queue", None)
        return fq() if callable(fq) else None

    def _hotkey_busy(self, hotkey: str) -> bool:
        busy = getattr(self.ops, "hotkey_busy", None)
        return bool(busy(hotkey)) if callable(busy) else False

    def live_hotkeys(self) -> set[str]:
        """Hotkeys with a miner-funded lane-B leg running in THIS process: one
        live leg per hotkey across lanes A/B/C (pod names, the harvest dir and
        the payer key are per hotkey), and their payer pods survive the
        lane-A pod sweep."""
        with self._lock:
            if not self._recovered:
                # before the first tick: the legs it will try to re-attach keep
                # their pods out of the restart sweep (2026-11-26)
                keys = self._reattach_candidates()
                return {leg.sub.hotkey for leg in self.queue.open
                        if _attach_key(leg) in keys}
            return {leg.sub.hotkey for leg in [*self.queue.open, *self._running.values()]
                    if leg.funding}

    def _reconcile_funded(self, fq) -> int:
        """Requeue, uncharged, every funded entry ``in_flight`` that no leg of
        this process runs or holds finished (a crash between the queue's
        ``mark_in_flight`` and the driver's save). Caller holds the lock."""
        mine = ({leg.sub.hotkey for leg in self.queue.open if leg.funding}
                | {f["leg"]["sub"]["hotkey"] for f in self.st.finished
                   if f["leg"].get("funding")})
        n = 0
        for e in fq.in_flight():
            if e.hotkey in mine:
                continue
            log.warning("lane B: funded entry %s is in flight but no leg runs it — requeued "
                        "uncharged", e.hotkey[:8])
            fq.requeue(e.hotkey, error="no lane-B leg runs this in-flight entry — requeued",
                       error_class="no_capacity", burn_attempt=False)
            n += 1
        return n

    def _funded_entries(self, fq, rows: list) -> dict:
        """hotkey -> its startable funded entry (``queued``, not backing off)
        whose targets and ref are the hotkey's LATEST chain commitment. An
        entry the chain has moved past (``superseded``), one naming targets
        the network does not configure, or one whose commitment already
        trained a leg (``spent`` — re-commit to train again) ends here.
        Caller holds the lock."""
        from ..funding.queue import split_lane_b_ref
        from ..shared.expert_manifest import parse_expert_targets
        from .expert_lane import submission_key

        if fq is None:
            return {}
        lb = self.cfg.lane_b
        slots = tuple(getattr(lb, "targets", None) or lb.domains)
        cap = int(getattr(lb, "max_domains_per_leg", 1) or 1)
        latest = _latest_commitments(rows)
        out = {}
        for e in fq.queued():
            try:
                declared, ref = split_lane_b_ref(e.ref)
            except ValueError:
                fq.fail(e.hotkey, error="malformed lane-B queue entry", error_class="target",
                        expect_ref=e.ref)
                continue
            if any(t not in slots for t in declared) or len(declared) > cap:
                fq.fail(e.hotkey, error=f"{list(declared)} are not 1..{cap} [lane_b] targets",
                        error_class="target", expect_ref=e.ref)
                continue
            c = latest.get(e.hotkey)
            if c is None:
                continue                     # no commitment visible (yet): wait (TTL bounds it)
            parsed = parse_expert_targets(getattr(c, "payload", "") or "", slots,
                                          max_targets=cap)
            if parsed is None or parsed[1] != ref or set(parsed[0]) != set(declared):
                fq.fail(e.hotkey, error="the hotkey's latest commitment is no longer this "
                        "lane-B entry — re-commit and fund again", error_class="superseded",
                        expect_ref=e.ref)
                continue
            sub = LaneBSubmission(hotkey=e.hotkey, uid=int(getattr(c, "uid", -1)), gen_ref=ref,
                                  domain=parsed[0][0],
                                  commit_block=int(getattr(c, "commit_block", 0) or 0))
            if submission_key(sub) in self.queue.done:
                fq.fail(e.hotkey, error="this commitment already trained a lane-B leg — "
                        "re-commit (a new reveal) to train again", error_class="spent",
                        expect_ref=e.ref)
                continue
            out[e.hotkey] = e
        return out

    def _eligible(self, miner: bool, funded: dict):
        """The dispatch filter: unfunded (operator-pool) mode admits every
        submission; miner-funded mode only one with a matching funded entry
        whose hotkey runs no other leg. Caller holds the lock."""
        if not miner:
            return None
        from ..funding.queue import lane_b_entry_ref

        def ok(sub: LaneBSubmission) -> bool:
            e = funded.get(sub.hotkey)
            return (e is not None and e.ref == lane_b_entry_ref(sub.declared, sub.gen_ref)
                    and not any(leg.sub.hotkey == sub.hotkey for leg in self._running.values())
                    and not self._hotkey_busy(sub.hotkey))
        return ok

    def _held_rows(self) -> dict[int, set[int]]:
        """Expert rows exclusive beyond the open legs: finished legs awaiting
        their settlement and pending merges until :meth:`init_for` consumes
        them. Caller holds the lock."""
        out: dict[int, set[int]] = {}
        sels = [f["leg"]["selection"] for f in self.st.finished]
        sels += [m.get("selection") or {} for m in self.st.pending_merges]
        for sel in sels:
            for li, rows in sel.items():
                out.setdefault(int(li), set()).update(int(r) for r in rows)
        return out

    # ── row claims (WB15, LB2 "Overlaps") ────────────────────────────────────

    def _declared_of(self, funded: dict) -> dict[str, dict[int, list[int]]]:
        """hotkey -> the non-empty declared rows of its startable funded entry
        (WB14 ``declared_rows``). Caller holds the lock."""
        out = {}
        for hk, e in funded.items():
            rows = normalise_rows(getattr(e, "declared_rows", None))
            if rows:
                out[hk] = rows
        return out

    def _row_holders(self, era_index: int) -> list:
        """Every current holder of expert rows, in a fixed order: open legs
        (start order), finished-but-unsettled legs, pending merges. Caller
        holds the lock."""
        from ..funding.queue import lane_b_entry_ref

        lb = self.cfg.lane_b
        age = int(lb.max_base_age_eras)
        out = []
        for leg in self.queue.open:
            ref = str(leg.funding.get("entry_ref") or "") or lane_b_entry_ref(
                leg.sub.declared, leg.sub.gen_ref)
            out.append(leg_holder(leg.sub.hotkey, ref, leg.selection, leg.base_era,
                                  era_index, age))
        for f in self.st.finished:
            leg = f["leg"]
            sub = leg["sub"]
            declared = tuple(sub.get("targets") or (sub["domain"],))
            ref = str((leg.get("funding") or {}).get("entry_ref") or "") or lane_b_entry_ref(
                declared, sub["gen_ref"])
            out.append(leg_holder(sub["hotkey"], ref, leg["selection"], int(leg["base_era"]),
                                  era_index, age))
        by_pointer = {r.get("leg_pointer"): r for r in self.st.submission_results
                      if r.get("leg_pointer")}
        for m in self.st.pending_merges:
            src = by_pointer.get(m.get("rebased_pointer"), {})
            hk, ref = str(src.get("hotkey") or ""), ""
            if src.get("gen_ref"):
                ref = lane_b_entry_ref(tuple(src.get("targets") or (src.get("domain"),)),
                                       src["gen_ref"])
            out.append(merge_holder(hk, ref, m.get("selection") or {}, era_index))
        return out

    # ── sentinel targets (WB18; §12b.5 LB5, D35) ─────────────────────────────

    def _sentinel_target(self, block: int) -> str | None:
        """The slot name of the era's sentinel suite (``suite:<suite>``), read
        only while suite targets are armed and the era starting at or before
        ``block`` holds a sentinel (``sentinel_armed``): W-S1's draw over the
        era-start block hash (``ops.block_hash``, one chain read per era).
        ``None`` = no sentinel this era; :data:`SENTINEL_UNREADABLE` = the
        hash could not be read this tick. Never under ``_lock``."""
        from ..shared.config.v2 import SUITE_TARGET_PREFIX
        from ..shared.era import era_start_block
        from ..shared.sentinel import sentinel_armed, sentinel_from_block

        if not getattr(self.cfg.lane_b, "suite_targets", False) or sentinel_from_block(
                self.cfg) <= 0:
            return None
        start = int(era_start_block(self.cfg.round, int(block)))
        if not sentinel_armed(self.cfg, start):
            return None
        try:
            suite = self._era_sentinel(start)
        except Exception as e:  # noqa: BLE001 — a chain read failure holds, never raises
            log.warning("lane B: era-start block %d hash unreadable (%s) — suite-target "
                        "entries wait for the era's sentinel draw", start, e)
            return SENTINEL_UNREADABLE
        return SUITE_TARGET_PREFIX + suite

    def _era_sentinel(self, start: int) -> str:
        """W-S1's draw for the era starting at ``start`` (cached; one
        ``ops.block_hash`` read per era). Raises on a chain read failure or a
        malformed hash. Never under ``_lock``."""
        from ..shared.sentinel import sentinel_for_era

        with self._cache_lock:
            suite = self._sentinels.get(int(start))
        if suite is None:
            suite = sentinel_for_era(self.ops.block_hash(int(start)),
                                     self.cfg.v2.sentinel_suites)
            with self._cache_lock:
                self._sentinels[int(start)] = suite
                while len(self._sentinels) > SENTINELS_KEEP:
                    self._sentinels.popitem(last=False)
        return suite

    def _settlement_sentinel(self, block: int) -> str | None:
        """W-S2 (§16 rule 2, D35): the §16 sentinel of the era the settlement
        at boundary ``block`` belongs to (``settlement_era_start``) — the
        same draw the validators record for that settlement — or ``None``
        when ``[v2] sentinel_from_block`` does not arm that era (no chain
        read). Read whenever armed, with or without ``[lane_b]
        suite_targets``. An unreadable hash raises ``OSError``: the
        settlement holds (its legs stay finished), never judged with a
        guessed sentinel. Never under ``_lock``."""
        from ..shared.sentinel import sentinel_armed, sentinel_from_block, settlement_era_start

        if sentinel_from_block(self.cfg) <= 0:
            return None
        start = settlement_era_start(self.cfg, int(block))
        if not sentinel_armed(self.cfg, start):
            return None
        try:
            return self._era_sentinel(start)
        except Exception as e:  # noqa: BLE001 — any read failure holds the settlement
            raise OSError(f"lane-B settlement sentinel unreadable (era-start block {start}): "
                          f"{type(e).__name__}: {e}") from e

    def _leg_length_holds(self, funded: dict, block: int, era_index: int) -> list[dict]:
        """WB23 (TB26): the public rows of funded ``train_hours`` entries whose
        guard cannot land in this era (``deferred: leg_length``, with the
        expected ``target_boundary``). They stay queued and uncharged; an entry
        without train_hours is never held here. Caller holds the lock."""
        rows: list[dict] = []
        if not funded:
            return rows
        margin = float(getattr(getattr(self.ops, "r", None), "FUNDED_PUBLISH_MARGIN_SECONDS",
                               _LEG_LENGTH_MARGIN_SECONDS) or _LEG_LENGTH_MARGIN_SECONDS)
        now = time.time()
        era_seconds = _era_seconds(getattr(self.cfg, "round", None), int(block))
        for hk, e in sorted(funded.items()):
            guard = int(getattr(e, "leg_guard_seconds", 0) or 0)
            hours = float(getattr(e, "train_hours", 0.0) or 0.0)
            if hours <= 0 or guard <= 0:
                continue
            try:
                adm = leg_length_admission(self.cfg.round, block_now=block, now=now,
                                           guard_seconds=guard, margin_seconds=margin,
                                           current_era=era_index)
            except Exception as ex:  # noqa: BLE001 — an unreadable timing never starts it
                log.warning("lane B: leg-length admission for %s failed (%s) — held",
                            hk[:8], ex)
                adm = None
            if adm is not None and not adm.cross_era:
                continue
            if adm is not None and era_seconds and guard + margin > era_seconds:
                # owner 2026-11-24: a leg longer than an era (12 h legs: guard
                # 15 h + overhead) can never land inside one — waiting would
                # hold it forever. It starts now and settles at the boundary
                # after its wall, rebased onto that king (max_base_age_eras)
                continue
            rows.append({"hotkey": str(hk), "entry_ref": str(getattr(e, "ref", "")),
                         "deferred": LEG_LENGTH_HOLD,
                         "train_hours": float(e.train_hours), "guard_seconds": guard,
                         "target_boundary": int(adm.target_boundary) if adm else 0,
                         "era_index": int(adm.era_index) if adm else 0})
        return rows

    def _sentinel_holds(self, sentinel: str | None, era_index: int,
                        eligible) -> dict[str, tuple[LaneBSubmission, str]]:
        """hotkey -> (pending submission, D35 decision) for every eligible
        pending entry on the era's sentinel target: ``hold`` (deferred,
        uncharged) or ``past_bound`` (deferred in ``sentinel_max_defer_eras``
        eras already: judged on its target). An unreadable draw holds every
        suite-target entry. Pure over the driver state; caller holds the lock."""
        from ..shared.config.v2 import SUITE_TARGET_PREFIX
        from .expert_lane import submission_key
        from .row_claims import sentinel_decision, sentinel_hit

        if sentinel is None:
            return {}
        bound = int(self.cfg.lane_b.sentinel_max_defer_eras)
        out: dict[str, tuple[LaneBSubmission, str]] = {}
        for sub in sorted(self.queue.pending, key=lambda s: (s.commit_block, s.hotkey)):
            if eligible is not None and not eligible(sub):
                continue
            if sentinel == SENTINEL_UNREADABLE:
                if any(str(t).startswith(SUITE_TARGET_PREFIX) for t in sub.declared):
                    out[sub.hotkey] = (sub, SENTINEL_HOLD)
                continue
            if sentinel_hit(sub.declared, sentinel):
                eras = self.st.sentinel_deferrals.get(submission_key(sub), [])
                out[sub.hotkey] = (sub, sentinel_decision(eras, era_index, bound))
        return out

    def _record_sentinel(self, holds: dict, sentinel: str | None, funded: dict,
                         era_index: int) -> list[dict]:
        """Count this era for every held entry (once per distinct era), log
        each past-bound start (TB20), prune the counts to live submissions and
        return the ``deferred: sentinel_target`` status rows. Caller holds the
        lock."""
        from ..funding.queue import lane_b_entry_ref
        from ..shared.config.v2 import SUITE_TARGET_PREFIX
        from .expert_lane import submission_key
        from .row_claims import sentinel_row

        bound = int(self.cfg.lane_b.sentinel_max_defer_eras)
        suite = "" if sentinel in (None, SENTINEL_UNREADABLE) else str(sentinel)[
            len(SUITE_TARGET_PREFIX):]
        rows: list[dict] = []
        for hk, (sub, decision) in holds.items():
            key = submission_key(sub)
            eras = self.st.sentinel_deferrals.get(key, [])
            if decision != SENTINEL_HOLD:
                if (key, int(era_index)) not in self._bound_logged:
                    self._bound_logged.add((key, int(era_index)))
                    log.info("lane B: %s/%s deferred in %d era(s) %s = sentinel_max_defer_eras "
                             "— judged on its target %s although it is era %d's sentinel",
                             hk[:8], sub.domain, len(eras), sorted(eras), sentinel,
                             int(era_index))
                continue
            if sentinel != SENTINEL_UNREADABLE and int(era_index) not in eras:
                eras = sorted({*eras, int(era_index)})
                self.st.sentinel_deferrals[key] = eras
                log.info("lane B: %s/%s targets era %d's sentinel %s — deferred uncharged "
                         "(%d of sentinel_max_defer_eras=%d)", hk[:8], sub.domain,
                         int(era_index), sentinel, len(eras), bound)
            e = funded.get(hk)
            ref = e.ref if e is not None else lane_b_entry_ref(sub.declared, sub.gen_ref)
            rows.append(sentinel_row(hk, ref, suite, eras, bound, int(era_index)))
        live = {submission_key(s) for s in self.queue.pending}
        live |= {submission_key(leg.sub) for leg in self.queue.open}
        for key in [k for k in self.st.sentinel_deferrals if k not in live]:
            del self.st.sentinel_deferrals[key]
        self._bound_logged = {p for p in self._bound_logged if p[0] in live}
        return rows

    def _mark_funded(self, fq, funded: dict, sub: LaneBSubmission, era_index: int,
                     block: int) -> dict | None:
        """Move ``sub``'s funded entry in flight; the leg's ``funding`` record,
        or None when the queue refused. Caller holds the lock."""
        e = funded[sub.hotkey]
        if not fq.mark_in_flight(sub.hotkey, e.ref, target_boundary=0,
                                 era_index=int(era_index), started_block=int(block)):
            return None
        from .progress import funded_attempt_id

        return {"entry_ref": e.ref, "started_block": int(block),
                "attempt_id": funded_attempt_id(fq, sub.hotkey, e.ref, int(block)),
                **({"gpu": str(e.gpu),
                    "gpu_allow_fallback": bool(e.gpu_allow_fallback),
                    "gpu_max_price_per_hour": float(e.gpu_max_price_per_hour)}
                   if getattr(e, "gpu", "") else {}),
                # WB23: the leg length fixed at the fund (absent = today's leg)
                **({"train_hours": float(e.train_hours), "leg_steps": int(e.leg_steps),
                    "leg_guard_seconds": int(e.leg_guard_seconds)}
                   if float(getattr(e, "train_hours", 0.0) or 0.0) > 0 else {}),
                # owner 2026-11-24: the token limit and an own-risk GPU
                **({"train_tokens": int(e.train_tokens)}
                   if int(getattr(e, "train_tokens", 0) or 0) > 0 else {}),
                **({"gpu_measured": False}
                   if getattr(e, "gpu_measured", True) is False else {}),
                # WB27: the levers funded (absent = every default); the levers
                # the leg trains are fixed at dispatch ("levers")
                **({"funded_levers": dict(e.levers)} if getattr(e, "levers", None) else {})}

    def _start_declared(self, fq, funded: dict, declared: dict, eligible, *, block: int,
                        era_index: int, king_pointer: str) -> tuple[list[OpenLeg], list[dict]]:
        """Plan the declared claims FIRST, in claim order (funded entry's
        reveal block, then hotkey): each trains exactly its declared rows
        unless a holder (open leg, finished-but-unsettled leg, pending merge —
        either mode) owns one of them, in which case it is deferred: left
        pending and queued, uncharged, and reported ``rows_held``. A deferred
        claim holds nothing. Returns ``(started, deferred rows)``. Caller
        holds the lock."""
        lb = self.cfg.lane_b
        subs = sorted((s for s in self.queue.pending
                       if s.hotkey in declared and (eligible is None or eligible(s))),
                      key=lambda s: claim_key(funded[s.hotkey].reveal_block, s.hotkey))
        busy = target_load(self.queue.open)
        started: list[OpenLeg] = []
        deferred: list[dict] = []
        for sub in subs:
            if not has_capacity(sub.declared, busy, lb.legs_per_domain):
                continue
            rows = declared[sub.hotkey]
            holder = first_overlap(rows, self._row_holders(era_index))
            if holder is not None:
                deferred.append(deferred_row(sub.hotkey, funded[sub.hotkey].ref, holder, rows))
                log.info("lane B: %s/%s declared rows held by %s (%s) — deferred until its "
                         "%s (era %d)", sub.hotkey[:8], sub.domain, holder.hotkey[:8],
                         holder.entry_ref, holder.kind, holder.era)
                continue
            held = self._held_rows()
            exclude = self.queue.owned_rows()
            for li, r in held.items():
                exclude.setdefault(li, set()).update(r)
            funding = self._mark_funded(fq, funded, sub, era_index, block)
            if funding is None:
                continue
            try:
                leg = self.queue.start(sub, rows, king_pointer, era_index, exclude=exclude,
                                       held=held, funding=funding)
            except ValueError as err:
                fq.requeue(sub.hotkey, error=f"lane-B rows unavailable ({err})",
                           error_class="no_capacity", burn_attempt=False)
                log.warning("lane B: %s/%s not started: %s", sub.hotkey[:8], sub.domain, err)
                continue
            for t in sub.declared:
                busy[t] = busy.get(t, 0) + 1
            started.append(leg)
        return started, deferred

    def abandon(self, round_id: str) -> str:
        """The scheduler stopped waiting for ``round_id``'s settlement: from
        now on :meth:`settle` commits nothing for it (its legs stay finished
        for the next settlement). Returns the digest ALREADY committed for it
        ("" = none) — atomic with the commit, so the lane-A manifest binds
        exactly what the driver recorded."""
        with self._lock:
            rid = str(round_id)
            if rid not in self.st.settled:
                self._abandoned.add(rid)
            self._settle_pins.pop(rid, None)
            return str(self.st.settled.get(rid, {}).get("digest", "") or "")

    # ── lineage clock ────────────────────────────────────────────────────────

    def add_lineage_tokens(self, n: int) -> None:
        """Explicit local/manual accounting; rolling production uses epochs."""
        with self._lock:
            self.st.lineage_tokens += int(n)
            self.save()

    def record_lineage_epoch(self, epoch: str, binding: dict) -> None:
        """Durably enqueue once BEFORE rolling retires the completed epoch."""
        with self._lock:
            old = self.st.lineage_epochs.get(epoch)
            if old is not None:
                if {k: v for k, v in old.items() if k != "tokens_seen"} != binding:
                    raise ValueError("lineage epoch replay changed its checkpoint binding")
                return
            self.st.lineage_epochs[epoch] = dict(binding)
            try:
                self.save()
            except Exception:
                del self.st.lineage_epochs[epoch]
                raise

    def reconcile_lineage(self) -> None:
        """Fetch checkpoint-bound measurements; apply each once in one write."""
        from ..shared.hippius import StorageError
        from .lineage import read_measurement

        with self._lock:
            pending = [(k, dict(v)) for k, v in self.st.lineage_epochs.items()
                       if "tokens_seen" not in v]
        for epoch, binding in pending:
            try:
                measure = getattr(self.ops, "lineage_measurement", None)
                n = (measure(binding["pointer"], binding) if callable(measure)
                     else read_measurement(self.ops.fetch(binding["pointer"]), binding))
            except (OSError, StorageError, ValueError, KeyError, TypeError) as e:
                log.warning("lane B: lineage epoch %s unmeasured: %s", epoch, e)
                continue
            with self._lock:
                row = self.st.lineage_epochs[epoch]
                if "tokens_seen" in row:
                    continue
                row["tokens_seen"] = n
                self.st.lineage_tokens += n
                try:
                    self.save()
                except Exception:
                    del row["tokens_seen"]
                    self.st.lineage_tokens -= n
                    raise

    # ── tick ─────────────────────────────────────────────────────────────────

    def tick(self, block: int, era_index: int, king_pointer: str | None) -> list[OpenLeg]:
        from ..shared.expert_manifest import parse_expert_targets

        self.reconcile_lineage()
        lb = self.cfg.lane_b
        if not lb.active(block) or not king_pointer:
            self._deferred = []
            return []
        # #909: a finished leg settles against the king's public rows — queue
        # the king's eval-pod bench now, not at the settlement
        pre = getattr(self.ops, "prebench", None)
        if self.st.finished and lb.require_public and callable(pre):
            try:
                pre(king_pointer)
            except Exception as e:  # noqa: BLE001 — a prebench never fails the tick
                log.warning("lane B: king prebench not queued: %s", e)
        # 2026-11-27 (owner): the verdict prepared AHEAD — each finished leg's
        # rebase, candidate bench and merge check against this king run on a
        # background worker now, not inside the next settlement
        prep = getattr(self.ops, "prepare_ahead", None)
        if callable(prep):
            try:
                prep(self.prepare_specs(int(block), int(era_index), king_pointer),
                     paused=self._settling.is_set)
            except Exception as e:  # noqa: BLE001 — prepare-ahead never fails the tick
                log.warning("lane B: prepare-ahead not submitted: %s", e)
        miner = self._miner_funded()
        fq = self._funded_queue() if miner else None
        if fq is not None:
            fq.expire_stale()
        # WB18 (D35): the era's sentinel suite target (None unless armed)
        sentinel = self._sentinel_target(int(block))
        with self._lock:
            self._recover_orphans(fq)
            self._heartbeat()
            for leg in [x for x in self.queue.open
                        if x.funding and self.queue.stale(x, int(era_index), lb)]:
                # a funded leg the trainer could not settle in time: the miner
                # paid for it, so it retrains (uncharged) on the current king
                log.warning("lane B: funded leg %s/%s expired — base era %d is more than %d "
                            "era(s) behind era %d; requeued uncharged", leg.sub.hotkey[:8],
                            leg.sub.domain, leg.base_era, lb.max_base_age_eras, era_index)
                self.queue.requeue(leg)
                if fq is not None:
                    fq.requeue(leg.sub.hotkey, error="lane-B base aged out before the leg "
                               "finished — requeued", error_class="no_capacity",
                               burn_attempt=False, operator_billed=True)
            for leg in self.queue.expire_stale(int(era_index), lb):
                # the rule settlement drops a leg by (``max_base_age_eras``):
                # a leg this old can never be merged, so its slot is freed
                log.warning("lane B: open leg %s/%s expired — base era %d is more than %d "
                            "era(s) behind era %d (counted as a failed attempt)",
                            leg.sub.hotkey[:8], leg.sub.domain, leg.base_era,
                            lb.max_base_age_eras, era_index)
            if fq is not None:
                self._reconcile_funded(fq)
            targets = tuple(getattr(lb, "targets", None) or lb.domains)
            rows = list(_latest_commitments(list(self.ops.commitments())).values())
            active = {}
            for row in rows:
                parsed = parse_expert_targets(getattr(row, "payload", "") or "", targets,
                                              max_targets=int(lb.max_domains_per_leg or 1))
                if parsed is not None:
                    active[row.hotkey] = (tuple(parsed[0]), parsed[1], int(row.commit_block))
            self.queue.pending = [s for s in self.queue.pending
                                  if active.get(s.hotkey) == (tuple(s.declared), s.gen_ref, s.commit_block)]
            for c in rows:
                # DEC-CA-0059: 1 .. max_domains_per_leg declared targets; a
                # single-domain commitment parses exactly as before
                parsed = parse_expert_targets(
                    getattr(c, "payload", "") or "", targets,
                    max_targets=int(getattr(lb, "max_domains_per_leg", 1) or 1))
                if parsed is None:
                    continue
                declared, ref = parsed
                self.queue.submit(LaneBSubmission(
                    hotkey=c.hotkey, uid=int(c.uid), gen_ref=ref, domain=declared[0],
                    commit_block=int(c.commit_block),
                    targets=list(declared) if len(declared) > 1 else []), lb)
            self.st.kings[str(int(era_index))] = king_pointer
            for old in sorted(self.st.kings, key=int)[:-KINGS_KEEP]:
                del self.st.kings[old]
            prev = self._prev_king(era_index, king_pointer)
            latched = self.st.gate_opened_block > 0
            funded = self._funded_entries(fq, rows) if miner else {}
            eligible = self._eligible(miner, funded)
            # WB18 (D35, TB19/TS4): an entry on the era's sentinel target is
            # held back from both passes (uncharged) until the bound lifts it
            holds = self._sentinel_holds(sentinel, int(era_index), eligible)
            held_back = {hk for hk, (_, d) in holds.items() if d == SENTINEL_HOLD}
            if held_back:
                base_eligible = eligible

                def eligible(sub: LaneBSubmission) -> bool:
                    return sub.hotkey not in held_back and (
                        base_eligible is None or base_eligible(sub))
            # WB23 (TB26): a train_hours leg that cannot land in this era waits
            # (queued, uncharged) for the next era's pre-train window
            length_rows = self._leg_length_holds(funded, int(block), int(era_index))
            length_held = {r["hotkey"] for r in length_rows}
            if length_held:
                inner_eligible = eligible

                def eligible(sub: LaneBSubmission) -> bool:
                    return sub.hotkey not in length_held and (
                        inner_eligible is None or inner_eligible(sub))
            # WB15: entries with declared rows are planned first (their own
            # rows, never ESFT); the ESFT pass sees only the rest. With none
            # declared the ESFT pass is exactly the pre-WB15 queue.
            declared = self._declared_of(funded)
            esft_eligible = eligible
            if declared:
                def esft_eligible(sub: LaneBSubmission) -> bool:
                    return sub.hotkey not in declared and eligible(sub)
            want = {t for s in self.queue.next_dispatchable(lb, esft_eligible)
                    for t in s.declared}
            self.save()
        self._dispatch_reattach()
        # WB17 (D41 / LB4): the configured opening rule; the co-gate is the
        # default and runs exactly as before
        rule = str(getattr(lb, "open_rule", OPEN_RULE_COGATE) or OPEN_RULE_COGATE)
        cogate = rule == OPEN_RULE_COGATE
        # ── routing saturation + current-king probes: OUTSIDE the lock ──
        routing: dict[str, dict] = {}
        if prev:
            try:
                for d in lb.domains:
                    routing[d] = routing_metrics(self._probe(prev, king_pointer, d, block),
                                                 self._probe(king_pointer, king_pointer, d, block),
                                                 lb)
            except Exception as e:  # noqa: BLE001
                if not latched and cogate:
                    raise                # closed co-gate: surface it (the tick retries)
                # a latched gate, or a token rule (routing is telemetry there)
                log.warning("lane B: routing-saturation measurement failed: %s", e)
                routing = {}
        # the lane A king check is a store read: OUTSIDE the lock, only while
        # a closed tokens_king_check gate has its tokens met
        check = None
        if not latched and needs_king_check(rule) \
                and self.st.lineage_tokens >= lb.open_tokens:
            check = self._read_king_check()
        if cogate:
            may_open = bool(prev and self.st.lineage_tokens >= lb.open_tokens)
        else:
            may_open = open_rule_verdict(rule, self.st.lineage_tokens, lb, check)[0]
        probes: dict[str, RoutingProbe] = {}
        if want and (latched or may_open):
            from .lane_b_offload import LaneBOffloadUnavailable

            for d in sorted(want):
                try:
                    probes[d] = self._probe(king_pointer, king_pointer, d, block)
                except LaneBOffloadUnavailable as e:
                    # no eval pod: its entries stay queued, uncharged (next tick)
                    log.warning("lane B: %s routing probe deferred: %s", d, e)
        with self._lock:
            if routing:
                self._record_routing(era_index, block, prev, king_pointer, routing)
            if not latched:
                if cogate:
                    ok, why = gate_open(self.st.lineage_tokens, routing, lb)
                    self.st.routing["open"], self.st.routing["reason"] = ok, why
                    reason = (GATE_TOKENS_SHORT if self.st.lineage_tokens < lb.open_tokens
                              else "")
                    evidence = {"lineage_tokens": int(self.st.lineage_tokens),
                                "open_tokens": int(lb.open_tokens),
                                "king": king_pointer, "prev_king": prev} if ok else {}
                else:
                    ok, reason, evidence = open_rule_verdict(
                        rule, self.st.lineage_tokens, lb, check)
                    why = reason or "open"
                if not ok:
                    self.st.gate_closed_reason = reason
                    self._deferred = []
                    self.save()
                    log.info("lane B: gate closed under open_rule=%s (%s)", rule, why)
                    return []
                self.st.gate_opened_block = int(block)
                self.st.gate_rule = rule
                self.st.gate_evidence = {**evidence, "block": int(block),
                                         "era": int(era_index)}
                self.st.gate_closed_reason = ""
                log.info("lane B: gate OPEN at block %d under open_rule=%s (era %d, king %s "
                         "vs previous %s; lineage %.3g tokens; evidence %s) — latched", block,
                         rule, era_index, king_pointer, prev, self.st.lineage_tokens,
                         json.dumps(self.st.gate_evidence, sort_keys=True))
            sentinel_rows = self._record_sentinel(holds, sentinel, funded, int(era_index))
            started, deferred_rows = [], []
            if declared:
                started, deferred_rows = self._start_declared(
                    fq, funded, declared, eligible, block=int(block), era_index=int(era_index),
                    king_pointer=king_pointer)
            self._deferred = sentinel_rows + deferred_rows + length_rows
            if fq is not None:
                # RV6 B4 (§12b.5 LB5): a deferral (``sentinel_target`` or
                # ``rows_held``) is uncharged, so it must not run the funded
                # entry's TTL either. Each deferring tick re-stamps the entry's
                # liveness, as lane A's ``touch`` does for admission-held
                # entries; once released it has a full TTL from this tick.
                # A never-deferred entry is not in these rows: unchanged.
                deferred_hks = sorted({str(r.get("hotkey")) for r in self._deferred
                                       if r.get("hotkey") in funded})
                if deferred_hks:
                    fq.touch(deferred_hks)
            for sub in self.queue.next_dispatchable(lb, esft_eligible):
                tprobes = [probes.get(t) for t in sub.declared]
                if any(pr is None for pr in tprobes):   # dispatchable mid-tick: next tick
                    continue
                # rows of open legs, finished-but-unsettled legs and pending
                # merges are exclusive (two deltas summed onto one row were
                # never judged together)
                held = self._held_rows()
                exclude = self.queue.owned_rows()
                for li, r in held.items():
                    exclude.setdefault(li, set()).update(r)
                # DEC-CA-0059: the union of every declared target's selection,
                # each planned under the same exclusions (one target ⇒ its own)
                sel = union_selection(pr.select(lb, exclude=exclude or None) for pr in tprobes)
                if not any(sel.values()):
                    log.info("lane B: %s/%s has no unowned routed expert row — waits",
                             sub.hotkey[:8], sub.domain)
                    continue
                funding: dict = {}
                if miner:
                    funding = self._mark_funded(fq, funded, sub, era_index, block)
                    if funding is None:
                        continue
                try:
                    leg = self.queue.start(sub, sel, king_pointer, era_index, exclude=exclude,
                                           held=held, funding=funding)
                except ValueError as err:
                    if funding:
                        fq.requeue(sub.hotkey, error=f"lane-B rows unavailable ({err})",
                                   error_class="no_capacity", burn_attempt=False)
                    log.warning("lane B: %s/%s not started: %s", sub.hotkey[:8], sub.domain, err)
                    continue
                started.append(leg)
            self.save()
        for leg in started:
            log.info("lane B: dispatch %s domain=%s experts/layer=%s", leg.sub.hotkey[:8],
                     leg.sub.domain, {k: len(v) for k, v in leg.selection.items()})
            with self._lock:
                self._running[id(leg)] = leg
                self._record_attach(leg, int(era_index), int(block))
                self.save()
            try:
                self.ops.dispatch_leg(leg, era_index, block, self._leg_done)
            except Exception as e:
                from .lane_c_driver import LegFailure

                self._leg_done(leg, None, failure=LegFailure(
                    f"expert dispatch failed: {e}", miner_fault=False,
                    error_class="operator_transient", burn=False))
        return started

    def _read_king_check(self) -> dict | None:
        """The ops' latest lane A king check (WB17, D41 a), or ``None`` when
        the ops has no such read, it returns no record, or the read fails (a
        closed gate then records ``king_check_unavailable`` and retries next
        tick)."""
        read = getattr(self.ops, "king_check", None)
        if not callable(read):
            return None
        try:
            rec = read()
        except Exception as e:  # noqa: BLE001 — an unreadable check keeps the gate closed
            log.warning("lane B: lane A king check unreadable: %s", e)
            return None
        return dict(rec) if isinstance(rec, dict) else None

    def _prev_king(self, era_index: int, king_pointer: str) -> str | None:
        """The gate's reference: the latest king of an EARLIER era whose
        pointer differs from the current king (caller holds the lock). None
        until the lineage has produced a second king — the gate never
        compares a king against itself."""
        for e in sorted((int(k) for k in self.st.kings), reverse=True):
            if e < int(era_index) and self.st.kings[str(e)] != king_pointer:
                return self.st.kings[str(e)]
        return None

    def _record_routing(self, era_index: int, block: int, prev: str, king: str,
                        routing: dict[str, dict]) -> None:
        """Persist + log the gate's per-layer numbers (caller holds the lock).
        A history row is appended once per (previous king, king) pair."""
        self.st.jaccard = {d: min(m["jaccard"], default=0.0) for d, m in routing.items()}
        row = {"era": int(era_index), "block": int(block), "king": king, "prev_king": prev,
               "lineage_tokens": int(self.st.lineage_tokens),
               "domains": {d: dict(m) for d, m in routing.items()}}
        fresh = (self.st.routing.get("king"), self.st.routing.get("prev_king")) != (king, prev)
        self.st.routing = {**row, "open": self.st.gate_opened_block > 0,
                           "reason": self.st.routing.get("reason", "")}
        if fresh:
            self.st.routing_history.append(row)
            del self.st.routing_history[:-HISTORY_KEEP]
            for d, m in routing.items():
                log.info("lane B: routing saturation era=%d %s (%d tokens) — top-k overlap "
                         "per layer %s, ESFT Jaccard per layer %s", era_index, d,
                         int(m.get("tokens", 0)), m["overlap"], m["jaccard"])

    # ── routing probes (cached; never under ``_lock``) ───────────────────────

    def _slice(self, slice_king: str, domain: str, block: int) -> tuple[str, list]:
        """The domain's affinity slice, drawn once per (current king, domain):
        every probe compared on it (this king, the previous king, a leg's
        base king) routes the SAME tokens."""
        key = (slice_king, domain)
        with self._cache_lock:
            hit = self._slices.get(key)
            if hit is not None:
                self._slices.move_to_end(key)
                return hit
        ctx = list(self.ops.contexts(domain, block))[: self.cfg.lane_b.affinity_windows]
        hit = (_slice_digest(ctx), ctx)
        with self._cache_lock:
            self._slices[key] = hit
            while len(self._slices) > self._cache_cap():
                self._slices.popitem(last=False)
        return hit

    def _probe(self, pointer: str, slice_king: str, domain: str, block: int) -> RoutingProbe:
        """``pointer``'s routing of the slice drawn for ``slice_king``."""
        digest, ctx = self._slice(slice_king, domain, block)
        key = (pointer, domain, digest)
        with self._cache_lock:
            hit = self._probes.get(key)
            if hit is not None:
                self._probes.move_to_end(key)
                return hit
        remote = getattr(self.ops, "probe_routing", None)
        if callable(remote):        # production: on the eval pod (never the trainer box)
            probe = remote(pointer, ctx, domain)
        else:
            probe = _tel.probe_target_routing(self.ops.fetch(pointer), ctx, self.cfg.lane_b,
                                              domain)
        with self._cache_lock:
            self._probes[key] = probe
            while len(self._probes) > 2 * self._cache_cap():
                self._probes.popitem(last=False)
        return probe

    def _cache_cap(self) -> int:
        lb = self.cfg.lane_b
        return max(8, 3 * len(getattr(lb, "targets", None) or lb.domains))

    def _leg_done(self, leg: OpenLeg, leg_pointer: str | None, corpus_digest: str = "",
                  train_block: int = 0, gpu_name: str = "", failure=None,
                  gpu_n: int = 0, train_image_digest: str = "",
                  selection: dict | None = None, exclude: dict | None = None) -> None:
        """A leg returned. ``failure`` (a funded leg's
        :class:`~cascade.trainer.lane_c_driver.LegFailure`) decides its queue
        entry: a miner fault ends it; anything else requeues it — a capacity
        miss or an operator fault unburned, an infra fault on one of the
        entry's bounded attempts."""
        fq = self._funded_queue() if leg.funding else None
        with self._lock:
            self._running.pop(id(leg), None)
            self.st.attached.pop(_attach_key(leg), None)
            if leg not in self.queue.open:
                # expired (or requeued) while it ran: its slot is gone and a
                # leg that old is dropped at settlement anyway
                log.warning("lane B: leg %s/%s finished after it was expired — discarded",
                            leg.sub.hotkey[:8], leg.sub.domain)
                return
            if leg_pointer:
                if selection is not None:
                    # a re-uploaded harvest: the leg is recorded with the rows
                    # (and exclusions) its checkpoint was trained on
                    leg.selection = {int(k): sorted(int(x) for x in v)
                                     for k, v in selection.items()}
                    leg.exclude = {int(k): sorted(int(x) for x in v)
                                   for k, v in (exclude or {}).items()}
                self.queue.finish(leg, ok=True)
                self.st.finished.append(asdict(FinishedExpertLeg(
                    leg=asdict(leg), leg_pointer=leg_pointer, corpus_digest=corpus_digest,
                    train_block=int(train_block), gpu_name=gpu_name, gpu_n=int(gpu_n),
                    train_image_digest=str(train_image_digest or ""))))
            elif leg.funding and not getattr(failure, "miner_fault", False):
                msg = str(getattr(failure, "msg", "") or "leg produced no checkpoint")
                cls = str(getattr(failure, "error_class", "") or "operator")
                self.queue.requeue(leg)          # the funded entry bounds the retries
                if fq is not None:
                    fq.requeue(leg.sub.hotkey, error=msg, error_class=cls,
                               burn_attempt=bool(getattr(failure, "burn", False)),
                               # a refused re-attach: exactly the restart requeue
                               **({"operator_billed": True}
                                  if getattr(failure, "operator_billed", False) else {}))
                self._funded_outcome(leg.sub, FUNDED_REQUEUED, {"class": cls, "error": msg[:300]})
                log.warning("lane B: funded leg %s/%s failed [%s] — requeued: %s",
                            leg.sub.hotkey[:8], leg.sub.domain, cls, msg[:200])
            else:
                self.queue.finish(leg, ok=False)
                if leg.funding and fq is not None:
                    fq.fail(leg.sub.hotkey, error=str(failure.msg), error_class=str(
                        failure.error_class), expect_ref=leg.funding.get("entry_ref"))
                    self._funded_outcome(leg.sub, FUNDED_MINER_FAULT,
                                         {"class": failure.error_class,
                                          "error": str(failure.msg)[:300]})
                log.warning("lane B: leg %s/%s failed%s", leg.sub.hotkey[:8], leg.sub.domain,
                            " — entry ended (miner fault)" if leg.funding else "")
            self.save()

    def _model(self):
        """The model-work object (:class:`~cascade.trainer.lane_b_model.LaneBModel`
        interface): ``ops.lane_b_model()`` when the ops offer one (the trainer's
        :class:`RunnerLaneBOps`: the eval pod), else a local one over
        ``ops.fetch`` / ``ops.upload`` / ``ops.bench_rows`` (tests, dev)."""
        make = getattr(self.ops, "lane_b_model", None)
        if callable(make):
            return make()
        from .lane_b_model import LaneBModel

        return LaneBModel(self.ops.fetch, self.ops.upload,
                          getattr(self.ops, "bench_rows", None))

    def _anchor_bench_rows(self, anchor_pointer: str, require: tuple[str, ...],
                           model=None) -> dict:
        """WB26: the anchor king's public bench rows, swept once per anchor.
        Unavailable or incomplete rows raise OSError (the settlement retries,
        as for the king's and the candidates' rows)."""
        from ..eval.public_guards import MissingPublicGuard, pair_public_rows

        rows = self._anchor_rows.get(anchor_pointer)
        if rows is None:
            rows = (model or self._model()).rows(anchor_pointer) or {}
            try:
                pair_public_rows(rows, rows, require=require,
                                 min_items=self.cfg.lane_b.min_public_items)
            except MissingPublicGuard as e:
                raise OSError(f"lane-B anchor public benchmarks unavailable: {e}") from e
            self._anchor_rows = {anchor_pointer: rows}     # one anchor at a time
        return rows

    def _funded_outcome(self, sub: LaneBSubmission, outcome: str, detail) -> None:
        self.st.outcomes.append({"round_id": "", "hotkey": sub.hotkey, "uid": int(sub.uid),
                                 "domain": sub.domain, "gen_ref": sub.gen_ref,
                                 "leg_pointer": "", "base_king_ref": "", "king_pointer": "",
                                 "outcome": outcome, "detail": detail,
                                 **({"targets": list(sub.targets)} if sub.targets else {})})
        del self.st.outcomes[:-OUTCOMES_KEEP]

    # ── settlement ───────────────────────────────────────────────────────────

    def rebase_inputs(self, f: dict, block: int, king_pointer: str) -> dict:
        """The settlement rebase's model inputs of finished leg ``f`` — exactly
        what :meth:`settle` passes ``model.rebase`` (minus the round-named
        out dir / repo) — shared with the prepare-ahead worker so a prepared
        rebase has the settlement's bytes."""
        lb = self.cfg.lane_b
        leg = f["leg"]
        fund = leg.get("funding") or {}
        return {"king_pointer": king_pointer, "base_pointer": leg["base_king_ref"],
                "leg_pointer": f["leg_pointer"],
                "selection": {int(k): v for k, v in leg["selection"].items()},
                "refuse_nonfinite": lb.merge_guard_at_settlement(block),
                "limits": (int(fund.get("train_tokens", 0) or 0) > 0
                           or fund.get("gpu_measured") is False)}

    def prepare_specs(self, block: int, era_index: int, king_pointer: str) -> list[dict]:
        """One prepare-ahead spec per finished leg the next settlement would
        rebase onto ``king_pointer`` (a leg it drops as stale is skipped);
        ``merge_strict`` = the self-judge's strictness. Block-gated flags are
        read at ``block`` — a flip before the settlement only makes the
        prepared result a cache miss (the settlement recomputes)."""
        lb = self.cfg.lane_b
        if not lb.require_public or not king_pointer:
            return []
        with self._lock:
            todo = [dict(f) for f in self.st.finished]
        out = []
        for f in todo:
            leg = f["leg"]
            if era_index - int(leg["base_era"]) > lb.max_base_age_eras:
                continue
            sub = leg["sub"]
            spec = self.rebase_inputs(f, int(block), king_pointer)
            spec["merge_strict"] = bool(lb.merge_guard_at_settlement(int(block)))
            spec["label"] = f"{sub['hotkey'][:8]}/{sub['domain']}"
            out.append(spec)
        return out

    @_pauses_prepare
    def settle(self, *, round_id: str, block: int, era_index: int, king_pointer: str,
               work_dir: Path, anchor_pointer: str | None = None) -> list:
        """Rebase every finished leg onto this settlement's king, publish the
        expert manifest, self-judge it, and queue the winners for merge.

        Called BEFORE the lane-A manifest is published: the lane-A manifest
        binds :meth:`expert_manifest_digest`, so no validator can see the
        settlement without its expert legs. Finished legs leave the queue only
        once the whole settlement succeeded — an I/O failure anywhere keeps
        them for the next settlement (and binds nothing this round).
        Idempotent per ``round_id``.

        ``anchor_pointer`` (WB26, DEC-CA-0071 §3 guard 5): the era's anchor
        king — the promotion member the validator derives from the signed
        training manifest; the caller passes it only while suite slots are
        configured. Every leg is tested on every suite slot (rev6.22), so
        every settling leg carries the anchor's public rows as
        ``public_rows["anchor"]`` (benched once per anchor), and the
        self-judge is guarded against the same anchor."""
        from ..shared.expert_manifest import ExpertEntry, ExpertManifest, leg_label
        from .lane_b_model import MergeRefused

        lb = self.cfg.lane_b
        if not lb.active(block):
            return []
        rid = str(round_id)
        with self._lock:
            if rid in self.st.settled or rid in self._abandoned:
                return []
            pin = self._settle_pins.get(rid)
            if pin is None:
                todo = [dict(f) for f in self.st.finished]
                # one round settles at a time: only the latest round's pin
                self._settle_pins = {rid: {"legs": [f["leg_pointer"] for f in todo],
                                           "lineage_tokens": int(self.st.lineage_tokens)}}
            else:
                # an in-window retry of this round: the first attempt's legs
                keep = set(pin["legs"])
                todo = [dict(f) for f in self.st.finished if f["leg_pointer"] in keep]
            lineage_tokens = int(self._settle_pins[rid]["lineage_tokens"])
        entries, used, dropped = [], [], []
        # W-S2 (D35): the settlement era's §16 sentinel is no public guard nor
        # requirement of lane B; None (unarmed) leaves every call as before
        sentinel = self._settlement_sentinel(int(block)) if todo else None
        require = (tuple(lb.require_public) if sentinel is None
                   else tuple(b for b in lb.require_public if b != sentinel))
        # model work (rebase, bench rows) through the model object: on the
        # eval pod for the trainer (never a checkpoint on the box)
        model = self._model() if todo else None
        king_rows = (model.rows(king_pointer)
                     if todo and lb.require_public else None)
        if todo and lb.require_public:
            from ..eval.public_guards import MissingPublicGuard, pair_public_rows

            try:
                pair_public_rows(king_rows, king_rows, require=require,
                                 min_items=lb.min_public_items)
            except MissingPublicGuard as e:
                raise OSError(f"lane-B king public benchmarks unavailable: {e}") from e
        # #909: rows come from the eval pod's background bench (None = queued,
        # the king's at tick time — RunnerLaneBOps.prebench); an unavailable
        # candidate (or anchor) no longer stops the loop, so every candidate's
        # bench is queued in the same settlement — the first such failure is
        # raised after the loop, before anything is published or committed
        bench_err: OSError | None = None

        def _drop(f: dict, outcome: str, detail) -> None:
            sub = f["leg"]["sub"]
            dropped.append({"round_id": rid, "hotkey": sub["hotkey"], "uid": int(sub["uid"]),
                            "domain": sub["domain"], "gen_ref": sub["gen_ref"],
                            "leg_pointer": f["leg_pointer"],
                            "base_king_ref": f["leg"]["base_king_ref"],
                            "king_pointer": king_pointer, "outcome": outcome,
                            "detail": detail,
                            **({"targets": list(sub["targets"])} if sub.get("targets") else {})})

        for f in todo:
            leg = f["leg"]
            sub = leg["sub"]
            if era_index - int(leg["base_era"]) > lb.max_base_age_eras:
                log.info("lane B: %s/%s stale (base era %s), dropped", sub["hotkey"][:8],
                         sub["domain"], leg["base_era"])
                _drop(f, "stale", {"base_era": int(leg["base_era"]), "era": int(era_index)})
                continue
            rin = self.rebase_inputs(f, int(block), king_pointer)
            sel = rin["selection"]
            if leg["base_king_ref"] != king_pointer:
                # computed outside ``_lock`` (settle never holds it here)
                stale, detail = self._routing_stale(leg, sel, king_pointer, int(block))
                if stale:
                    log.info("lane B: %s/%s routing_stale (current king routes the domain "
                             "elsewhere: ESFT Jaccard per layer %s, top-k overlap per layer "
                             "%s), dropped", sub["hotkey"][:8], sub["domain"],
                             detail["jaccard"], detail["overlap"])
                    _drop(f, "routing_stale", detail)
                    continue
            declared = tuple(sub.get("targets") or (sub["domain"],))
            label = leg_label(declared)          # == the domain for a single target
            out = Path(work_dir) / f"rebased-{rid}-{sub['hotkey'][:12]}-{label}"
            # WB27 (plan rev6.21): the levers the leg trained (recorded at
            # dispatch; {} = every default) — the EMA choice picks the width
            levers = dict((leg.get("funding") or {}).get("levers") or {})
            fund = leg.get("funding") or {}
            limits = rin["limits"]
            # plan rev6.20: the receipt check (receipt_contract_ok) bound the pod
            # to the lane B EMA leg contract, so it scored its (chosen) EMA
            scored = _scored_weights_for(lb, levers)
            # the limits record read, the rebase and its upload: one model
            # step (on the eval pod, pod → Hub; never bytes through the box)
            try:
                res = model.rebase(king_pointer=rin["king_pointer"],
                                   base_pointer=rin["base_pointer"],
                                   leg_pointer=rin["leg_pointer"], selection=sel, out_dir=out,
                                   repo=f"ckpt-r{rid}-expert-{label}-u{sub['uid']}",
                                   refuse_nonfinite=rin["refuse_nonfinite"],
                                   limits=limits)
            except MergeRefused as e:
                log.warning("lane B: %s/%s refused at settlement: %s", sub["hotkey"][:8],
                            sub["domain"], e)
                _drop(f, "merge_refused", str(e))
                continue
            if res.pointer is None:
                log.warning("lane B: %s/%s limits leg without its lane_b_leg record — dropped",
                            sub["hotkey"][:8], sub["domain"])
                _drop(f, "leg_record_missing", {"train_block": int(f["train_block"])})
                continue
            actual, ptr = dict(res.actuals or {}), res.pointer
            rows = {}
            if king_rows is not None:
                rows = {"king": king_rows, "candidate": model.bench(res.bench) or {}}
                try:
                    pair_public_rows(rows["king"], rows["candidate"], require=require,
                                     min_items=lb.min_public_items)
                except MissingPublicGuard as e:
                    bench_err = bench_err or OSError(
                        f"lane-B candidate public benchmarks unavailable: {e}")
                if anchor_pointer:
                    try:
                        rows["anchor"] = self._anchor_bench_rows(anchor_pointer, require,
                                                                 model)
                    except OSError as e:
                        bench_err = bench_err or e
            entries.append(ExpertEntry(
                miner_hotkey=sub["hotkey"], miner_uid=int(sub["uid"]), gen_ref=sub["gen_ref"],
                domain=sub["domain"], selection=sel,
                # the rows other legs held when this one was planned (its
                # selection skipped them) — recorded, not re-derived
                exclude={int(k): list(v) for k, v in (leg.get("exclude") or {}).items() if v},
                base_king_pointer=leg["base_king_ref"], rebased_pointer=ptr,
                corpus_digest=f["corpus_digest"], train_block=int(f["train_block"]),
                commit_block=int(sub["commit_block"]), gpu_name=f.get("gpu_name", ""),
                gpu_n=int(f.get("gpu_n", 0) or 0), public_rows=rows, targets=declared if len(declared) > 1 else (),
                scored_weights=scored,
                # W-IMG: logged, never enforced; absent before the activation
                train_image_digest=(str(f.get("train_image_digest", "") or "")
                                    if image_record_at(self.cfg, int(block)) else ""),
                # WB23: the signed leg length (0s = no train_hours: today's bytes)
                train_hours=float((leg.get("funding") or {}).get("train_hours", 0.0) or 0.0),
                leg_steps=int((leg.get("funding") or {}).get("leg_steps", 0) or 0),
                leg_guard_seconds=int((leg.get("funding") or {}).get("leg_guard_seconds", 0)
                                      or 0),
                levers=levers,
                train_tokens=int(fund.get("train_tokens", 0) or 0),
                gpu_measured=fund.get("gpu_measured") is not False,
                trained_steps=int(actual.get("steps", 0) or 0),
                trained_tokens=int(actual.get("tokens", 0) or 0),
                stop=str(actual.get("stop", "") or "")))
            used.append((ptr, f, sel))
        if bench_err is not None:
            raise bench_err
        judgements: list = []
        digest = ""
        if entries:
            manifest = ExpertManifest(round_id=rid, created_block=int(block),
                                      king_pointer=king_pointer,
                                      lineage_tokens=lineage_tokens,
                                      entries=tuple(entries))
            self.ops.publish_expert_manifest(manifest)
            judge_kw = {} if sentinel is None else {"sentinel": sentinel}
            if anchor_pointer and any("anchor" in (e.public_rows or {}) for e in entries):
                judge_kw["anchor_pointer"] = anchor_pointer     # WB26
            judgements = self.ops.judge(manifest, rid, int(block), **judge_kw)
            digest = manifest.digest()
        # DEC-CA-0059: a leg that took at least one slot merges ONCE, however
        # many of its targets won (keyed by its judged checkpoint)
        passed = _winning_legs(judgements)
        checks_passed = {j.entry.rebased_pointer for j in judgements if j.passed}
        qualified = {}
        if lb.acceptance_at_settlement(block):
            from ..shared.lane_b import LaneBState
            from ..validator.expert_lane import apply_judgements

            for w in apply_judgements(LaneBState(), judgements, round_id=rid,
                                      settle_block=block, qualify_only=True):
                qualified.setdefault(w.rebased_pointer, []).append(asdict(w))
        with self._lock:
            if rid in self._abandoned:
                # the lane-A manifest of this round went out without us
                # (rolling.LANE_B_SETTLE_MAX_SECONDS): commit nothing — the
                # legs stay finished and settle with a later round
                log.warning("lane B: settlement %s finished after it was abandoned — "
                            "nothing committed; its legs wait for the next settlement", rid)
                return []
            progress_outcomes = []
            # owner 2026-11-28: the judge's per-target LCB / bar / gain per
            # judged leg, kept in the driver state for the public queue
            # verdict (display telemetry: never signed, never re-read by any
            # scoring path; the signed manifest/receipts are unchanged)
            from ..funding.public_reason import judgement_margins

            margins = {}
            for j in judgements:
                try:
                    margins[j.entry.rebased_pointer] = judgement_margins(j)
                except Exception as e:  # noqa: BLE001 — telemetry never blocks a settlement
                    log.debug("lane B: verdict margins unavailable: %s", e)
            for ptr, _f, sel in used:
                sub = _f["leg"]["sub"]
                m = margins.get(ptr) or {}
                progress_outcomes.append({"round_id": rid, "hotkey": sub["hotkey"],
                                "gen_ref": sub["gen_ref"], "domain": sub["domain"],
                                "targets": list(sub.get("targets") or [sub["domain"]]),
                                "commit_block": sub["commit_block"],
                                "started_block": (_f["leg"].get("funding") or {}).get("started_block"),
                                "attempt_id": (_f["leg"].get("funding") or {}).get("attempt_id"),
                                "leg_pointer": ptr,
                                # W-E2E: the funded leg's own (trained) pointer —
                                # what _settle_funding finds this verdict by
                                "trained_leg_pointer": _f["leg_pointer"],
                                "outcome": ("qualified" if ptr in passed else
                                            "qualification_not_selected" if ptr in checks_passed else
                                            "qualification_rejected"),
                                **({"judge_stage": m["stage"]} if m.get("stage") else {}),
                                **({"margins": m["targets"]} if m.get("targets") else {})})
                rec = {"rebased_pointer": ptr, "king_pointer": king_pointer,
                       "selection": {str(k): v for k, v in sel.items()}}
                if ptr in qualified:
                    rec["qualified_wins"] = qualified[ptr]
                if ptr in passed and not any(
                        {k: m.get(k) for k in rec} == rec for m in self.st.pending_merges):
                    # the validator-reproducible delta: (rebased − this king);
                    # ``round_id`` locates the settlement that judged it, so a
                    # validator that missed it can judge it on demand
                    self.st.pending_merges.append({**rec, "round_id": rid})
            self.st.finished = [f for f in self.st.finished if f not in todo]
            self._settle_funding(todo, dropped, progress_outcomes)
            self.st.outcomes.extend(dropped)
            self.st.submission_results.extend(progress_outcomes)
            del self.st.submission_results[:-OUTCOMES_KEEP]
            del self.st.outcomes[:-OUTCOMES_KEEP]
            self.st.settled[rid] = {"digest": digest}
            self._settle_pins.pop(rid, None)
            for old in list(self.st.settled)[:-SETTLED_KEEP]:
                del self.st.settled[old]
            self.save()
        return judgements

    def _settle_funding(self, todo: list[dict], dropped: list[dict],
                        judged: list[dict] | None = None) -> None:
        """The funded queue entries of the legs this settlement consumed
        (caller holds the lock): a leg that reached the manifest (judged) or
        was refused at the merge byte-check is spent (``mark_done``); a leg
        dropped unjudged through no fault of the miner (``stale``,
        ``routing_stale``) requeues uncharged and its commitment is admitted
        again — it retrains on the current king."""
        funded = [f for f in todo if f["leg"].get("funding")]
        if not funded:
            return
        fq = self._funded_queue()
        why = {d["leg_pointer"]: d["outcome"] for d in dropped}
        # owner 2026-11-28: the public verdict each spent entry carries
        # a judged row's ``leg_pointer`` is the REBASED checkpoint; the funded
        # leg is found by its trained pointer (W-E2E 2026-11-28: keyed by the
        # rebased one, no settled entry ever stored its verdict — only the
        # driver-state fallback of the public view showed it)
        verdicts = {(r.get("trained_leg_pointer") or r["leg_pointer"]): {**r, "lane": "b"}
                    for r in (judged or []) if r.get("leg_pointer")}
        verdicts.update({d["leg_pointer"]: {"lane": "b", "outcome": d["outcome"],
                                            "round_id": d.get("round_id"),
                                            "gen_ref": d.get("gen_ref")} for d in dropped})
        for f in funded:
            hk = f["leg"]["sub"]["hotkey"]
            if why.get(f["leg_pointer"]) in ("stale", "routing_stale"):
                self.queue.readmit(LaneBSubmission(**f["leg"]["sub"]))
                if fq is not None:
                    fq.requeue(hk, error=f"lane-B leg dropped unjudged at settlement "
                               f"({why[f['leg_pointer']]}) — retrains on the current king",
                               error_class="no_capacity", burn_attempt=False, operator_billed=True)
            elif fq is not None:
                funding = f["leg"]["funding"]
                verdict = verdicts.get(f["leg_pointer"])
                fq.mark_done(hk, expect_ref=funding["entry_ref"],
                             expect_started_block=funding.get("started_block"),
                             **({"verdict": verdict} if verdict else {}))

    def _routing_stale(self, leg: dict, sel: dict[int, list[int]], king_pointer: str,
                       block: int) -> tuple[bool, dict]:
        """Has the current king moved ``leg``'s domain to different experts
        than the leg's base king? Both kings route the slice drawn for the
        current king; the domain is re-planned on the current king under the
        leg's own exclusions. An empty slice is no evidence (kept)."""
        pairs = []
        for d in (leg["sub"].get("targets") or (leg["sub"]["domain"],)):
            cur = self._probe(king_pointer, king_pointer, d, block)
            base = self._probe(leg["base_king_ref"], king_pointer, d, block)
            if not cur.tokens or not base.tokens:
                log.warning("lane B: empty affinity slice for %s — routing-staleness check "
                            "skipped", d)
                return False, {"jaccard": [], "overlap": []}
            pairs.append((base, cur))
        exclude = {int(k): set(v) for k, v in (leg.get("exclude") or {}).items() if v}
        # one target ⇒ exactly routing_stale; several ⇒ the union re-planned
        return routing_stale_union(pairs, sel, self.cfg.lane_b, exclude=exclude or None)

    def status(self) -> dict:
        """The lane-B block of the public status document: funding mode, the
        funded queue (counts per status + the public rows), open legs per
        target, finished legs awaiting a settlement, pending merges and the
        recent outcomes. Never raises on a queue read."""
        from ..shared.submission_progress import lane_b_progress

        lb = self.cfg.lane_b
        fq = self._funded_queue() if self._miner_funded() else None
        queue: dict = {}
        if fq is not None:
            try:
                queue = fq.public_view()
            except Exception as e:  # noqa: BLE001 — a status read never raises
                log.debug("lane B status: queue read failed: %s", e)
        with self._lock:
            return {
                "leg_funding": str(getattr(lb, "leg_funding", "miner")),
                "leg_hours": float(lb.leg_hours),
                "legs_per_domain": int(lb.legs_per_domain),
                "gate_opened_block": int(self.st.gate_opened_block),
                "open_rule": str(getattr(lb, "open_rule", OPEN_RULE_COGATE)),
                "gate_rule": str(self.st.gate_rule),
                "gate_closed_reason": str(self.st.gate_closed_reason),
                "queue": queue,
                "pending": len(self.queue.pending),
                "open": [{"hotkey": leg.sub.hotkey, "uid": int(leg.sub.uid),
                          "targets": list(leg.sub.declared), "ref": leg.sub.gen_ref,
                          "base_era": int(leg.base_era),
                          "experts": sum(len(v) for v in leg.selection.values()),
                          "gpu": str(leg.funding.get("gpu", "") or ""),
                          # WB23: the leg length fixed at the fund (0 = not set)
                          "train_hours": float(leg.funding.get("train_hours", 0.0) or 0.0),
                          "steps": int(leg.funding.get("leg_steps", 0) or 0),
                          "guard_seconds": int(leg.funding.get("leg_guard_seconds", 0) or 0),
                          "funded": bool(leg.funding)} for leg in self.queue.open],
                "finished": len(self.st.finished),
                "pending_merges": len(self.st.pending_merges),
                "submission_progress": lane_b_progress(self.st),
                "outcomes": list(self.st.outcomes[-16:]),
                # WB15: declared claims waiting on held rows (``rows_held``)
                "deferred": [dict(r) for r in self._deferred],
                # 2026-11-27: the verdict prepared ahead, per finished leg
                **({"prepare": self.prepare_status()}
                   if callable(getattr(self.ops, "prepare_status", None)) else {}),
            }

    def prepare_status(self) -> dict:
        """The prepare-ahead worker's per-leg step states ({} = none)."""
        fn = getattr(self.ops, "prepare_status", None)
        if not callable(fn):
            return {}
        try:
            return dict(fn() or {})
        except Exception as e:  # noqa: BLE001 — a status read never raises
            log.debug("lane B prepare status unavailable: %s", e)
            return {}

    def expert_manifest_digest(self, round_id: str) -> str:
        """The digest the lane-A manifest of ``round_id`` binds ("" = none)."""
        return str(self.st.settled.get(str(round_id), {}).get("digest", "") or "")

    # ── era init with merged winners ─────────────────────────────────────────

    def prepare_init_for(self, era_index, init_pointer, work_dir, *, era_start=None):
        """Nonblocking scheduler entry. Publish staged results only when the
        scheduler adopts this exact era/base; obsolete jobs consume no deltas.
        """
        from .merge_acceptance import MergeAcceptancePending

        if not self.cfg.lane_b.acceptance_at_era(era_start) or init_pointer is None:
            return self.init_for(era_index, init_pointer, work_dir, era_start=era_start)
        key = (int(era_index), str(init_pointer))
        with self._lock:
            def deferred_init():
                self.st.merged_inits[str(era_index)] = {"pointer": init_pointer,
                                                       "base": init_pointer, "merges": []}
                self.save()
                log.error("lane B acceptance exceeded its time budget; era %s keeps the "
                          "unmerged init, qualifications remain unpaid and pending", era_index)
                return init_pointer, []

            cached = self.st.merged_inits.get(str(era_index))
            if cached is not None:
                if cached.get("base") != init_pointer:
                    raise ValueError("prepared init belongs to a different base")
                return cached["pointer"], cached["merges"]
            job = self._acceptance_jobs.get(key)
            timeout = max(1, getattr(getattr(self.cfg, "round", None), "ops_acceptance_timeout_seconds", 1200))
            if any(not item["done"].is_set() and time.monotonic() - item["started"] >= timeout
                   for item in self._acceptance_jobs.values()):
                return deferred_init()
            if job is None:
                if any(not item["done"].is_set() for item in self._acceptance_jobs.values()):
                    raise MergeAcceptancePending("previous acceptance job is finishing")
                self._acceptance_jobs.clear()  # obsolete finished jobs consumed nothing
                if not self.st.pending_merges:
                    return init_pointer, []
                job = {"done": threading.Event(), "started": time.monotonic()}
                self._acceptance_jobs[key] = job

                def run():
                    from ..shared.rss_budget import step_rss

                    try:
                        with step_rss(f"lane B acceptance era={era_index}", logger=log,
                                      hint="merge / judge model work belongs on the lane B "
                                      "model host, never in the trainer (#952-#955)"):
                            job["result"] = self.init_for(
                                era_index, init_pointer, work_dir, era_start=era_start,
                                _stage_only=True)
                    except Exception as e:
                        job["error"] = e
                    finally:
                        job["done"].set()

                threading.Thread(target=run, name=f"lane-b-acceptance-{era_index}",
                                 daemon=True).start()
                raise MergeAcceptancePending(f"acceptance for era {era_index} started")
            if not job["done"].is_set():
                raise MergeAcceptancePending(f"acceptance for era {era_index} in progress")
            if "error" in job:
                del self._acceptance_jobs[key]
                raise job["error"]
            prepared = job["result"]
            if not isinstance(prepared, _PreparedInit):
                return prepared
            from ..shared.lane_b import merge_identities, merge_identity

            consumed = merge_identities(prepared.merges + prepared.dropped)
            if not consumed <= merge_identities(self.st.pending_merges):
                raise ValueError("pending merges changed while acceptance was running")
            import copy

            previous = copy.deepcopy(self.st)
            try:
                self.st.merged_inits[str(era_index)] = {
                    "pointer": prepared.pointer, "base": init_pointer, "merges": prepared.merges}
                self.st.pending_merges = [m for m in self.st.pending_merges
                                          if merge_identity(m) not in consumed]
                self.st.outcomes.extend(prepared.outcomes)
                self.st.outcomes = self.st.outcomes[-OUTCOMES_KEEP:]
                self.save()
            except Exception:
                self.st = previous
                raise
            del self._acceptance_jobs[key]
            return prepared.pointer, prepared.merges

    def init_for(self, era_index: int, init_pointer: str | None, work_dir: Path,
                 *, apply_pending: bool = True,
                 era_start: int | None = None, _stage_only: bool = False):
        """``(pointer, merges)``: the era's init with every pending winning
        delta applied and uploaded, plus the merge records the manifest
        declares (validators rebuild it bit-exactly). ``(init_pointer, [])``
        when nothing is pending. Idempotent per era.

        "State at merge" (DEC-CA-0050): the scored ``weights.safetensors``
        gets exactly :func:`expert_lane.apply_merges`'s bytes (what validators
        rebuild); the same delta goes onto ``weights_stable.safetensors`` (the
        raw lineage a lane-A leg warm-starts from) and every power-EMA shadow,
        and the merged rows' optimizer moments are reset. A merge that would
        overlap an earlier one's rows waits for the next era init (never two
        unjudged-together deltas on one row); one with a non-finite delta is
        dropped. ``era_start`` arms the validators' merge guard
        (``[lane_b] merge_guard_from_block``) on the rebuild.

        ``apply_pending=False`` (DEC-CA-0055 seed epochs: merges change the
        init only at a warm start) only READS the merged init already built
        for ``era_index`` — the epoch's first era — and never applies a merge
        that became pending since; ``(init_pointer, [])`` when none was built."""
        if init_pointer is None:
            return None, []
        key = str(era_index)
        with self._lock:
            if key in self.st.merged_inits:
                rec = self.st.merged_inits[key]
                if not apply_pending and rec.get("base") not in (None, init_pointer):
                    return init_pointer, []          # built over another member
                return rec["pointer"], rec["merges"]
            if not apply_pending:
                return init_pointer, []
            # read, never pop: a fetch/upload failure below keeps every
            # pending merge for the retry (the next era roll)
            pending = [dict(m) for m in self.st.pending_merges]
        if not pending:
            return init_pointer, []
        # every checkpoint read, merge and upload through the model object:
        # on the eval pod for the trainer (pod → Hub), never on the box
        from .expert_lane import merge_rows

        model = self._model()
        chosen, dropped, taken = [], [], set()
        acceptance_outcomes = []
        acceptance = self.cfg.lane_b.acceptance_at_era(era_start)
        if not acceptance and any(m.get("qualified_wins") for m in pending):
            raise ValueError("qualified merges require acceptance; refusing to bypass the gate")
        if acceptance:
            if any(not m.get("qualified_wins") for m in pending):
                raise ValueError("acceptance requires qualification evidence; drain legacy "
                                 "pending merges before activating acceptance")
            pending.sort(key=lambda m: (
                int(m["qualified_wins"][0]["commit_block"]),
                str(m["qualified_wins"][0]["hotkey"]), str(m["rebased_pointer"])))
            from .merge_acceptance import accept_candidates

            chosen, dropped, acceptance_outcomes = accept_candidates(
                pending, init_pointer=init_pointer, work_dir=Path(work_dir),
                block=int(era_start), ops=self.ops, cfg=self.cfg.lane_b, model=model)
            pending = []  # accepted/rejected decisions replace the legacy loop
        # per merge (independent of the others): its delta is finite
        finite = model.finite(init_pointer, pending) if pending else []
        for m, ok in zip(pending, finite, strict=True):
            cells = merge_rows(m)
            if cells & taken:
                log.warning("lane B: merge %s overlaps an earlier merge's rows — waits for the "
                            "next era init", m.get("rebased_pointer"))
                continue
            if not ok:
                log.error("lane B: merge %s carries a non-finite delta — dropped",
                          m.get("rebased_pointer"))
                dropped.append(m)
                continue
            chosen.append(m)
            taken |= cells
        if not chosen:
            if _stage_only:
                return _PreparedInit(init_pointer, [], dropped, acceptance_outcomes)
            if dropped:
                with self._lock:
                    self.st.pending_merges = [x for x in self.st.pending_merges
                                              if x not in dropped]
                    self.st.outcomes.extend(acceptance_outcomes)
                    self.st.outcomes = self.st.outcomes[-OUTCOMES_KEEP:]
                    self.save()
            return init_pointer, []
        ptr = model.build(init_pointer, chosen,
                          out_dir=Path(work_dir) / f"merged-init-era{era_index}",
                          repo=f"ckpt-era{era_index}-merged-init",
                          strict=self.cfg.lane_b.merge_guard_at_era(era_start),
                          era_index=int(era_index))
        if _stage_only:
            return _PreparedInit(ptr, chosen, dropped, acceptance_outcomes)
        with self._lock:
            from ..shared.lane_b import merge_identities, merge_identity

            consumed = merge_identities(chosen + dropped)
            self.st.merged_inits[key] = {"pointer": ptr, "base": init_pointer,
                                         "merges": chosen}
            self.st.pending_merges = [m for m in self.st.pending_merges
                                      if merge_identity(m) not in consumed]
            self.st.outcomes.extend(acceptance_outcomes)
            self.st.outcomes = self.st.outcomes[-OUTCOMES_KEEP:]
            self.save()
        log.info("lane B: era %d init carries %d merged domain leg(s) → %s",
                 era_index, len(chosen), ptr)
        return ptr, chosen


def _scored_weights_for(lb, levers: dict) -> str:
    """The ``scored_weights`` of a lane B leg: rev6.20's EMA at the WB27 chosen
    width (``"ema010"`` by default); ``"?"`` for levers that no longer parse."""
    from ..shared.lane_b_levers import LeversError, lane_b_scored_weights_for, levers_from_dict

    try:
        parsed = levers_from_dict(levers, lb) if levers else None
    except LeversError:
        return "?"
    return lane_b_scored_weights_for(lb, None, parsed)


def leg_actuals(leg_dir) -> dict:
    """owner 2026-11-24: a limits leg's ``config.json`` ``lane_b_leg`` record
    ({steps, tokens, stop}: what it trained and the limit that stopped it),
    ``{}`` when absent or unreadable."""
    try:
        meta = json.loads((Path(leg_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    rec = meta.get("lane_b_leg") if isinstance(meta, dict) else None
    if not isinstance(rec, dict):
        return {}
    try:
        return {"steps": int(rec["steps"]), "tokens": int(rec["tokens"]),
                "stop": str(rec["stop"])}
    except (KeyError, TypeError, ValueError):
        return {}
def _latest_commitments(rows) -> dict:
    """hotkey -> its LATEST commitment row (highest ``commit_block``)."""
    out: dict = {}
    for c in rows or ():
        hk = getattr(c, "hotkey", None)
        if not hk:
            continue
        cur = out.get(hk)
        if cur is None or int(getattr(c, "commit_block", 0) or 0) >= int(
                getattr(cur, "commit_block", 0) or 0):
            out[hk] = c
    return out


def _winning_legs(judgements: list) -> set[str]:
    """The judged checkpoints (``rebased_pointer``) that took at least one
    slot under ``apply_judgements`` (one win per target) — each is merged
    exactly once (DEC-CA-0059)."""
    from ..shared.lane_b import LaneBState
    from ..validator.expert_lane import apply_judgements

    st = LaneBState()
    return {w.rebased_pointer for w in apply_judgements(st, judgements, round_id="_",
                                                         settle_block=0)}


class RunnerLaneBOps:
    """The production :class:`LaneBOps`: a thin adapter over the trainer's own
    machinery — chain commitments, the private eval pool (affinity slice +
    the validators' verdict-window rule), the Hub registry, the bench
    sidecar, and the rolling dispatcher (or a local train when the trainer
    runs without remote hosts)."""

    def __init__(self, runner, client, window_source, *, device: str = "cpu", bench=None,
                 offload=None):
        self.r, self.client, self.source, self.device = runner, client, window_source, device
        # #909: the eval-pod lane-B bench (cascade.trainer.lane_b_bench.LaneBBench);
        # None = no --lane-b-bench-hosts: lane B fails closed, never benches here
        self.bench = bench
        # model work (routing probe, lineage read) on the eval pod
        # (cascade.trainer.lane_b_offload.LaneBPodRunner); None = refused
        # unless CASCADE_LANE_B_LOCAL_EVAL=1 — never a local fallback
        self.offload = offload

    # ── funding (DEC-CA-0059 item 8) ─────────────────────────────────────────

    @property
    def miner_funded(self) -> bool:
        """``[lane_b] leg_funding = "miner"`` (the default): every leg needs a
        funded :class:`~cascade.funding.queue.LaneBQueue` entry and rents on
        the miner's key; with funding off nothing dispatches (there is no
        operator fallback)."""
        return str(getattr(self.r.cfg.lane_b, "leg_funding", "miner")) == "miner"

    def funded_queue(self):
        return lane_b_queue_of(self.r) if self.miner_funded else None

    def hotkey_busy(self, hotkey: str) -> bool:
        """A lane-A leg of ``hotkey`` in flight (its funded entry, or a leg
        thread of the rolling scheduler) or a lane-C leg running: one live leg
        per hotkey across lanes (shared pod names, harvest dir, payer key)."""
        from .admission import owner_busy

        if owner_busy(self.r, hotkey):
            return True
        q = self.r._funded_queue()
        e = q.get(hotkey) if q is not None else None
        if e is not None and e.status == "in_flight":
            return True
        sched = self.r.__dict__.get("_rolling_sched")
        if sched is not None and hotkey in getattr(sched, "_threads", {}):
            return True
        lc = getattr(self.r, "lane_c_driver", None)
        try:
            return bool(lc is not None and hotkey in lc.live_hotkeys())
        except Exception:  # noqa: BLE001
            return False

    def commitments(self) -> list:
        from .admission import remember_owners

        r = self.r
        wd = getattr(r, "_with_deadline", None)
        rows = (list(wd(self.client.poll_commitments,
                        float(getattr(r, "chain_read_timeout_s", 60.0) or 60.0)))
                if callable(wd) else list(self.client.poll_commitments()))
        return remember_owners(r, rows)

    def king_check(self) -> dict | None:
        """The latest lane A king check (DEC-CA-0055 ``king_check`` record)
        of the validators' signed receipt, through the runner's verified
        receipt read (WB17, D41 a); ``None`` when none is readable. Read only."""
        got = self.r._receipt_king_check()
        return None if got is None else dict(got[0])

    def block_hash(self, block: int) -> str:
        """The chain's hash of ``block`` (read-only): the era-start hash the
        validators draw the same §16 sentinel from (WB18, D35)."""
        return str(self.client.block_hash(int(block)))

    def contexts(self, domain: str, block: int) -> list:
        from ..validator.expert_lane import affinity_contexts

        return affinity_contexts(self.r.cfg, self.source, domain, block=block)

    def fetch(self, pointer: str) -> Path:
        """A checkpoint on THIS machine: refused on the trainer box (owner rule:
        no checkpoint downloads, no torch there) unless CASCADE_LANE_B_LOCAL_EVAL=1."""
        from .lane_b_offload import refuse_local

        refuse_local(f"checkpoint download ({str(pointer)[-24:]})")
        return self.r._fetch_checkpoint_dir(pointer)

    def _pod(self):
        from .lane_b_offload import LaneBOffloadUnavailable

        if self.offload is None:
            raise LaneBOffloadUnavailable(
                "lane B: no eval pod runner (--lane-b-bench-hosts) — model work never runs "
                "on the trainer box; retried next tick")
        return self.offload

    def probe_routing(self, pointer: str, contexts: list, domain: str):
        """The routing probe: on the eval pod; locally only under the opt-in."""
        from .lane_b_offload import local_eval_allowed

        lb = self.r.cfg.lane_b
        if local_eval_allowed():
            return _tel.probe_target_routing(self.fetch(pointer), contexts, lb, domain)
        return self._pod().probe(pointer, contexts, int(lb.affinity_windows), domain)

    def lineage_measurement(self, pointer: str, binding: dict) -> int:
        from .lane_b_offload import local_eval_allowed
        from .lineage import read_measurement

        if local_eval_allowed():
            return read_measurement(self.fetch(pointer), binding)
        return self._pod().lineage(pointer, binding)

    def push_target(self, repo: str) -> dict:
        """Where a checkpoint named ``repo`` is pushed — :meth:`upload`'s Hub
        repo and HF fallback, for a push made on the eval pod."""
        hub = self.r.hub()
        full = f"{hub.namespace}/{repo}"
        return {"full": full, "hf_repo": self.r._hf_ckpt_repo(full),
                "registry_url": hub.registry_url, "namespace": hub.namespace}

    def lane_b_model(self):
        """The driver's model object: the eval pod
        (:class:`~cascade.trainer.lane_b_offload.PodLaneBModel` — settlement
        rebase, era-init merges and their pushes run on the pod, pod → Hub);
        the box-local :class:`~cascade.trainer.lane_b_model.LaneBModel` only
        under ``CASCADE_LANE_B_LOCAL_EVAL=1``. No pod runner ⇒
        ``LaneBOffloadUnavailable`` (transient: the settlement / era init
        retries; nothing falls back to the box)."""
        from .lane_b_offload import PodLaneBModel, local_eval_allowed

        if local_eval_allowed():
            from .lane_b_model import LaneBModel

            return LaneBModel(self.fetch, self.upload, self.bench_rows)
        return PodLaneBModel(self._pod(), self.bench, self.push_target)

    def upload(self, ckpt_dir: Path, repo: str) -> str:
        from ..shared.hippius import upload_dir_to_hub_or_hf
        from ..shared.manifest import format_trained_pointer

        full = f"{self.r.hub().namespace}/{repo}"
        up = upload_dir_to_hub_or_hf(Path(ckpt_dir), full, self.r.hub(),
                                     hf_repo=self.r._hf_ckpt_repo(full))
        return format_trained_pointer(up.ref.immutable_ref)

    def _seeds(self):
        sched = self.r.__dict__.get("_rolling_sched")
        era = sched.state.current if sched is not None else None
        return self.r._rolling_seeds(era) if era is not None else None

    def dispatch_leg(self, leg: OpenLeg, era_index: int, block: int, done) -> None:
        """Run the leg. A miner-funded leg (``leg.funding``, the default
        ``[lane_b] leg_funding = "miner"``) rents on the MINER's key —
        :meth:`_dispatch_funded`. Only ``leg_funding = "operator"`` (dev /
        emergency) reaches the retired operator lane pool below."""
        if leg.funding:
            self._dispatch_funded(leg, block, done)
            return
        if self.miner_funded:
            # never on the operator's bill: a leg without a funded entry here
            # is a bug upstream — fail it closed (requeued uncharged)
            log.error("lane B: leg %s/%s has no funded entry under leg_funding='miner' — "
                      "not dispatched", leg.sub.hotkey[:8], leg.sub.domain)
            done(leg, None)
            return
        self._dispatch_operator(leg, era_index, block, done)

    def leg_identity(self, leg: OpenLeg) -> dict:
        """What a restart re-finds ``leg``'s run by (2026-11-26 re-attach): the
        seed epoch, the pod's ledgered name and the detached run tag — the
        same names lane A adopts by (``funded_pod_name`` / the dispatcher's
        ``<role>-<hotkey12>-<seed>`` tag)."""
        from ..provision.funded import funded_pod_name

        seeds = self._seeds()
        if seeds is None:
            return {}
        seed = str(seeds.base_seed)
        hk = leg.sub.hotkey
        return {"base_seed": seed,
                "pod_name": funded_pod_name(seed, hk, self.r.cfg.subnet.netuid) + "-0",
                "tag": f"expert-{hk[:12]}-{seed}"}

    def leg_pod_ledgered(self, leg: OpenLeg) -> bool:
        """A ledgered pod (payer or operator-rented) of ``leg`` in the current
        seed epoch: a re-attach then runs attach-only. False only when the
        ledger was read and holds neither name; anything unknown (no seeds
        yet, an unreadable ledger) is True — never risk a fresh rent of a pod
        name that might be live."""
        from .runner_funded_legs import _leg_allocated

        try:
            seeds = self._seeds()
        except Exception:  # noqa: BLE001 — unknown: attach-only decides
            return True
        if seeds is None:
            return True
        return bool(_leg_allocated(self.r, str(seeds.base_seed), leg.sub.hotkey))

    def reattach_leg(self, leg: OpenLeg, era_index: int, block: int, done, rec: dict) -> None:
        """Re-attach a persisted open funded leg after a trainer restart: the
        leg's own funded path at its ORIGINAL dispatch block, attach-only
        (:mod:`cascade.trainer.leg_reattach`): its live ledgered pod is
        adopted and the running worker attached to — no rent, no launch, no
        queue mark (same attempt, no new charge). Anything else is a refusal
        the driver requeues exactly as a restart requeue."""
        if not leg.funding:
            done(leg, None, failure=_reattach.refusal_failure("not a miner-funded leg"))
            return
        self._dispatch_funded(leg, int(block or leg.funding.get("started_block", 0) or 0),
                              done, reattach=dict(rec or {}))

    def _dispatch_funded(self, leg: OpenLeg, block: int, done, *,
                         reattach: dict | None = None) -> None:
        """One lane-B leg on the MINER's key: the funded challenger path
        (``_run_funded_leg``: rent on the payer key → identity / runtime pins →
        the corpus (shards tier on a ``shards`` contract) → worker ``--role
        expert`` with the leg's rows → credential-free harvest → ingest verify
        → teardown), never benched. Configured operator fallback uses the
        shared operator pool; absent capacity requeues without charging the
        miner again. Other failures are classified like a lane-A funded leg."""
        from ..funding.gpu_choice import GpuChoice
        from ..funding.lane_b_gpus import lane_b_gpus_from_env, rent_scope
        from ..provision.layout import layout_gpus
        from ..shared.expert_manifest import leg_label
        from .faults import _FundedOperatorFallback
        from .lane_c_driver import LegFailure
        from .loop import ResolvedGenerator

        if reattach is not None:
            outer = done

            def done(leg, ptr, *a, failure=None, **kw):     # noqa: F811 — the re-attach verdict
                if failure is not None and _reattach.is_refusal(failure):
                    failure = _reattach.refusal_failure(str(failure.msg))
                outer(leg, ptr, *a, failure=failure, **kw)

        def _run():
            if reattach is None:
                return _body()
            if not self.leg_pod_ledgered(leg):
                # 2026-11-28 (5H63pHtyHgkf): no ledgered pod (payer or
                # operator) of this leg exists, so there is nothing to attach
                # to — the old process died before it rented. Attach-only
                # would generate the whole CPU corpus (~1 h) and only then
                # refuse the rent. Run it as a NORMAL dispatch instead: same
                # leg, same in-flight attempt (no new charge), every pre-rent
                # gate applies. The ledger is write-ahead, so no row means no
                # pod that a fresh rent could double.
                log.warning("lane B: re-attach of %s: no ledgered pod for this leg — "
                            "dispatching it normally (generation + fresh rent)",
                            leg.sub.hotkey[:12])
                return _body()
            with _reattach.reattach_only():
                return _body()

        def _body():
            try:
                seeds = self._seeds()
                sched = self.r.__dict__.get("_rolling_sched")
                era = sched.state.current if sched is not None else None
                if seeds is None or era is None:
                    raise LegFailure("no era state yet", miner_fault=False,
                                     error_class="no_capacity", burn=False)
                if reattach is not None and str(reattach.get("base_seed", "")) != str(
                        seeds.base_seed):
                    # another seed epoch: another pod name and run tag
                    raise LegFailure(f"{_reattach.REATTACH_REFUSED}: the seed epoch moved "
                                     f"({reattach.get('base_seed')} -> {seeds.base_seed})",
                                     miner_fault=False, error_class="no_capacity", burn=False)
                if reattach is not None:
                    # 2026-11-27: a re-attach re-derives the leg in its DISPATCH
                    # era, never the current one — the era rolled (2665 -> 2666)
                    # under a finished 8xH200 run, its --contract-block moved
                    # 9594000 -> 9597600, the run tag's identity hash with it,
                    # and the finished run was "not found" and requeued
                    era = _dispatch_era(self.r.cfg, era, int(block), reattach)
                base = self.r.cfg.throne_contracts_at(int(era.start_block))[0]
                f = leg.funding
                length = _funded_leg_length(f, base)
                if length is not None:
                    # WB23 (D42b): steps fixed at the fund, the lane B guard;
                    # the record carries what this leg trains (clamped steps)
                    f["leg_steps"] = int(length.steps)
                # WB27 (plan rev6.21): the levers this leg trains, fixed at its
                # dispatch block
                spec = _leg_spec(self.r.cfg.lane_b, f, int(block))
                # the same three steps the worker makes on the same spec:
                # WB23 length, rev6.20 schedule, WB27 levers
                from ..shared.lane_b_levers import apply_spec_levers, leg_length_contract

                contract = leg_length_contract(self.r.cfg, base, spec)
                # plan rev6.20: the worker derives the same leg contract
                # (every lane B leg: EMA, no cooldown), then the WB27 levers
                from ..shared.config.lane_b import lane_b_leg_contract

                contract = lane_b_leg_contract(contract)
                contract = apply_spec_levers(self.r.cfg.lane_b, contract, spec, int(block))
                label = leg_label(leg.sub.declared)   # the domain for a single target
                gen = ResolvedGenerator(hotkey=leg.sub.hotkey, uid=leg.sub.uid,
                                        ref=leg.sub.gen_ref)
                # WB22 (D42): lane B rents only its own measured GPU types
                gpus = lane_b_gpus_from_env(self.r.cfg, block=int(era.start_block))
                # owner 2026-11-24: an own-risk GPU (no WB19 rate) rents exactly
                # the type the miner chose, nothing else
                own_risk = ((str(f.get("gpu", "") or ""),)
                            if f.get("gpu_measured") is False and f.get("gpu") else ())
                # 2026-11-26 audit: a WB23 fixed-steps leg (train_hours on a
                # measured type, no limits) fixed its steps on the CHOSEN type's
                # rate — the operator make-good path ignores gpu_choice and would
                # rent any measured type (8xRTX4090 0.137 vs 8xH200 0.055 steps/s:
                # a 2.5x slower pod wall-caps it or trips THROUGHPUT ABORT)
                # (_funded_leg_length carries no limits terms: read the record)
                limits = (int(f.get("train_tokens", 0) or 0) > 0
                          or f.get("gpu_measured") is False)
                fixed = (str(f.get("gpu", "") or ""),) if (
                    length is not None and not limits and f.get("gpu")) else ()
                # owner 2026-11-26: an OPERATOR-billed leg whose approval pins
                # GPU types (`approve-operator-leg --sku 8xH100,8xH200`) rents
                # exactly those, cheapest available first, measured or not
                from .money_guard import operator_pinned_skus

                pinned = operator_pinned_skus(self.r, gen.hotkey, queue=self.funded_queue())
                only = pinned or own_risk or fixed
                if pinned:
                    log.warning("lane B: operator-billed leg %s rents only the approved GPU "
                                "types %s", gen.hotkey[:12], ",".join(pinned))
                if not gpus.rent_skus() and not own_risk and not pinned:
                    raise LegFailure("no [lane_b] funded_pod_skus type has a measured "
                                     "step rate (WB19) on the pinned image — not rented",
                                     miner_fault=False, error_class="no_capacity", burn=False)
                self.r._funded_leg_failures.pop(gen.hotkey, None)
                loc = self.r.__dict__.setdefault("_leg_local_obj", threading.local())
                loc.funded_pod_sku = ""
                # DEC-CA-0060: the lane-B entry's own choice, never lane A's
                choice = GpuChoice(
                    gpu=str(f.get("gpu", "") or ""),
                    allow_fallback=bool(f.get("gpu_allow_fallback", False)),
                    max_price_per_hour=float(f.get("gpu_max_price_per_hour", 0.0) or 0.0))
                from .cost_guards import wall_bounded_leg

                # 2026-11-26: a leg funded without train_hours is bounded by
                # its wall (for_hours keeps lane A's step count): its
                # watchdog judges the wall's worth of steps at the WB19 rate
                # of the type it rents, never the contract's
                # (audit 2026-11-26: a LIMITS leg — token limit or own-risk
                # unmeasured GPU — also keeps lane A's reference steps behind its
                # hour wall: wall-bounded by design, judged the same way)
                # (a pinned make-good may rent a type its steps were not fixed
                # on, or an unmeasured one: wall-bounded, judge off unmeasured)
                wall_bounded = (wall_bounded_leg(self.r, _measured_rate_of(gpus, contract))
                                if length is None or limits or pinned
                                else contextlib.nullcontext())
                # the dispatch's rows / contract block / exclude: a failed
                # upload's pending record carries them, and a re-upload of it
                # records the leg with THOSE (never this tick's selection)
                from .runner_common import lane_b_dispatch_fields, lane_b_record_rows

                contract_block = self.r._era_contract_block(era)
                loc.lane_b_dispatch = lane_b_dispatch_fields(leg.selection, contract_block,
                                                             leg.exclude)
                loc.lane_b_reused = None
                try:
                    with rent_scope(self.r, gpus, only=only), wall_bounded:
                        entry = self.r._run_funded_leg(
                            self.r._dispatcher(), gen, seeds, int(block), contract,
                            f"-x{label}-u{leg.sub.uid}", warm_start_ref=leg.base_king_ref,
                            contract_block=contract_block,
                            era_index=era.index, role="expert", bench=False, domain=label,
                            expert_selection=leg.selection,
                            funded_queue=self.funded_queue(), gpu_choice=choice,
                            # the leg spec the worker derives its contract from
                            # (None = today's argv, byte-identical)
                            **({"finetune_recipe": json.dumps(spec, sort_keys=True,
                                                              separators=(",", ":"))}
                               if spec else {}))
                except _FundedOperatorFallback as e:   # incl. _FundedOperatorRent
                    q = self.funded_queue()
                    recovery = q.get(gen.hotkey) if q is not None else None
                    if recovery is not None and recovery.operator_billed:
                        raise LegFailure("operator-funded lane-B recovery is waiting for "
                                         "operator capacity; the miner is not charged again",
                                         miner_fault=False, error_class="operator_transient",
                                         burn=False) from e
                    raise LegFailure(f"no marketplace capacity on the payer key ({e})",
                                     miner_fault=False, error_class="no_capacity",
                                     burn=False) from e
                except Exception as e:
                    verdict = self.r._funded_leg_failures.pop(gen.hotkey, None)
                    breaker = lane_breaker_failure(e)
                    if verdict is None and breaker:
                        raise breaker["failure"] from e
                    if verdict is None and own_risk:
                        # owner 2026-11-24: an own-risk GPU (no WB19 rate) that
                        # OOMs, cannot boot or crashes — the miner's shot, spent,
                        # no requeue / refund (its key paid the rented hours)
                        raise LegFailure(f"lane-B leg failed on own-risk GPU {own_risk[0]}: {e}",
                                         miner_fault=True, error_class="own_risk_gpu_failed",
                                         burn=False) from e
                    if verdict is None:
                        raise LegFailure(f"lane-B leg failed (operator side): {e}",
                                         miner_fault=False, error_class="operator_transient",
                                         burn=False) from e
                    msg, miner_fault, error_class, burn = verdict
                    raise LegFailure(msg or str(e), miner_fault=miner_fault,
                                     error_class=error_class, burn=burn) from e
                finally:
                    loc.lane_b_dispatch = None
                reused = getattr(loc, "lane_b_reused", None)
                loc.lane_b_reused = None
                trained = {}
                if reused:
                    # a re-uploaded earlier harvest: the rows it was TRAINED on
                    trained = {"selection": lane_b_record_rows(reused["expert_selection"]),
                               "exclude": lane_b_record_rows(reused["expert_exclude"])}
                    if trained["selection"] != {int(k): sorted(v)
                                                for k, v in leg.selection.items()}:
                        log.warning("lane B: %s re-uploaded harvest recorded with its trained "
                                    "selection %s (this tick planned %s)", gen.hotkey[:12],
                                    {k: len(v) for k, v in trained["selection"].items()},
                                    {k: len(v) for k, v in leg.selection.items()})
                # WB22 / WB25: the leg's GPU count is the rented layout's (every
                # GPU of it trains); 0 = not known (no rent recorded)
                sku = str(getattr(loc, "funded_pod_sku", "") or "")
                from ..shared.config import image_digest

                done(leg, entry.trained_pointer, entry.corpus_digest, entry.train_block,
                     getattr(entry, "gpu_name", ""), gpu_n=layout_gpus(sku) if sku else 0,
                     # W-IMG: the image the leg ran (its contract's pin, which
                     # the worker's FINAL image check enforced on the pod)
                     train_image_digest=image_digest(str(
                         getattr(contract, "train_image_digest", "") or "")),
                     **trained)
            except LegFailure as f:
                done(leg, None, failure=f)
            except Exception as e:  # noqa: BLE001 — an unclassified failure is ours
                done(leg, None, failure=LegFailure(
                    f"lane-B leg failed without a verdict (operator side): {e}",
                    miner_fault=False, error_class="operator_transient", burn=False))

        threading.Thread(target=_run, name=f"laneb-{leg.sub.hotkey[:8]}"
                         + ("-reattach" if reattach is not None else ""), daemon=True).start()

    def _dispatch_operator(self, leg: OpenLeg, era_index: int, block: int, done) -> None:
        """``[lane_b] leg_funding = "operator"`` only (retired; dev /
        emergency): run the leg on the SAME operator lane pool the rolling
        scheduler's fallback challenger legs draw from
        (``_rolling_lane_pool`` + ``_dispatch_on_free_lane``): a lane is
        checked out while the leg runs, so an expert leg never lands on a GPU
        another leg holds (and vice versa). No operator lane on file ⇒ trains
        locally. The budget is ``[lane_b] leg_hours`` — resolved by the worker
        for ``--role expert``, never passed as ``--train-hours`` (that flag
        tags a reduced-budget HEAT leg)."""
        from .loop import ResolvedGenerator

        seeds = self._seeds()
        gen = ResolvedGenerator(hotkey=leg.sub.hotkey, uid=leg.sub.uid, ref=leg.sub.gen_ref)
        sched = self.r.__dict__.get("_rolling_sched")
        era = sched.state.current if sched is not None else None
        contract = (self.r.cfg.throne_contracts_at(int(era.start_block))[0]
                    if era is not None else self.r.cfg.training.at_block(int(block)).primary_size)
        from ..shared.expert_manifest import leg_label

        label = leg_label(leg.sub.declared)   # the domain for a single target
        suffix = f"-x{label}-u{leg.sub.uid}"
        contract_block = self.r._era_contract_block(era) if era is not None else None

        def _run():
            try:
                hosts = self.r._rolling_final_hosts()
                if hosts:
                    # DEC-CA-0053: a v2 contract trains lane B on shards too
                    corpus = leg_corpus(
                        self.r, gen, seeds, contract,
                        era_index=era.index if era is not None else None)
                    pool = self.r._rolling_lane_pool()
                    entry = self.r._dispatch_on_free_lane(
                        self.r._dispatcher(), pool, pool.known_hosts(),
                        describe=f"lane-B expert leg {leg.sub.hotkey[:12]}/{leg.sub.domain}",
                        gen_ref=gen.ref, uid=gen.uid, hotkey=gen.hotkey, role="expert",
                        base_seed=seeds.base_seed, block=block,
                        arch_preset=contract.arch_preset, repo_suffix=suffix,
                        warm_start_ref=leg.base_king_ref, contract_block=contract_block,
                        domain=label, expert_selection=leg.selection,
                        **corpus)
                else:
                    if getattr(contract, "corpus_mode", "") == "shards":
                        # The local path has no CPU-tier corpus (DEC-CA-0053).
                        raise RuntimeError("lane B: a shards contract needs an operator "
                                           "lane; no local fallback")
                    lc = contract.for_hours(
                        self.r.cfg.lane_b.leg_hours,
                        guard_factor=self.r.cfg.round.heat_guard_factor,
                        guard_floor_seconds=self.r.cfg.round.heat_guard_floor_seconds)
                    from ..shared.config.lane_b import lane_b_leg_contract

                    lc = lane_b_leg_contract(lc)
                    entry = self.r.train_one(
                        gen, "challenger", seeds, block, contract=lc,
                        token_budget=lc.train_tokens, repo_suffix=suffix,
                        warm_start_ref=leg.base_king_ref, expert_selection=leg.selection,
                        lane_info={"lane": "B", "target": ",".join(leg.sub.declared)})
                done(leg, entry.trained_pointer, entry.corpus_digest, entry.train_block,
                     getattr(entry, "gpu_name", ""))
            except Exception as e:  # noqa: BLE001 — a failed leg frees its domain
                log.warning("lane B: leg %s/%s failed: %s", leg.sub.hotkey[:8], leg.sub.domain, e)
                done(leg, None, **lane_breaker_failure(e))

        threading.Thread(target=_run, name=f"laneb-{label}", daemon=True).start()

    def bench_rows(self, ckpt_dir: Path) -> dict | None:
        """#909: the rows of ckpt_dir benched on the eval pod (cached by
        content), or None (queued / no bench host) — NEVER the trainer box."""
        if self.bench is None:
            log.error("lane B bench: no --lane-b-bench-hosts configured — lane-B public rows "
                      "are never benched on the trainer box; settlement fails closed")
            return None
        return self.bench.rows(Path(ckpt_dir))

    def prebench(self, pointer: str) -> None:
        """#909: queue pointer's eval-pod bench ahead of the settlement."""
        if self.bench is not None:
            self.bench.prebench(pointer)

    # ── prepare-ahead (2026-11-27, cascade.trainer.lane_b_prepare) ───────────

    def preparer(self):
        """The prepare-ahead worker over the bench pod (pin and sweep on the
        same host); None without a bench or its pod runner."""
        prep = getattr(self, "_preparer", None)
        if prep is None:
            pod = getattr(self.bench, "_pod", None) if self.bench is not None else None
            if pod is None or not callable(getattr(pod, "rebase_stage", None)):
                return None
            from .lane_b_prepare import LaneBPrepare

            prep = self._preparer = LaneBPrepare(pod, self.bench)
        return prep

    def prepare_ahead(self, specs: list[dict], paused=None) -> None:
        prep = self.preparer()
        if prep is not None:
            prep.submit(specs, paused=paused)

    def prepare_status(self) -> dict:
        prep = self.preparer()
        return prep.status() if prep is not None else {}

    def publish_expert_manifest(self, manifest) -> None:
        from ..shared.expert_manifest import expert_manifest_key, sign_expert_manifest

        if self.r.wallet is not None:
            manifest = sign_expert_manifest(manifest, self.r.wallet)
        else:
            log.warning("lane B: publishing an UNSIGNED expert manifest (no wallet)")
        # W-EVALPUB (owner 2026-11-28): signed, read by every validator: public-read
        self.r.manifest_store().put_text(expert_manifest_key(manifest.round_id),
                                         manifest.to_json(), content_type="application/json",
                                         acl="public-read")

    def judge(self, manifest, round_id: str, block: int, sentinel: str | None = None,
              anchor_pointer: str | None = None) -> list:
        from .lane_b_offload import local_eval_allowed

        local = local_eval_allowed()
        pod = None if local else self._pod()     # refuses before any window draw
        from ..validator.evaluator import evaluate_checkpoint
        from ..validator.expert_lane import judge_manifest, verdict_windows

        cfg = self.r.cfg
        windows = verdict_windows(cfg, self.source, int(round_id), block=int(block))
        # lane B judges v2 eras: the settlement's era contract (the MoE)
        from ..shared.era import settlement_contract_block

        contract = cfg.training.at_block(
            settlement_contract_block(cfg.round, int(block))).primary_size
        # DEC-CA-0067 quantile_head_after: the validator's rule at the same
        # boundary (ValidatorRunner._evaluate(block=)) — the self-judge must
        # score exactly as the fleet will.
        quantile_head = bool(cfg.quantile_head_at(int(block)))
        # DEC-CA-0068: the validator's decode batch at the same boundary
        from ..shared.config import shared_ladder_quantile_batch

        qbatch = shared_ladder_quantile_batch(cfg.eval, int(block)) if quantile_head else 1

        def _eval(pointer: str):
            if pod is not None:                  # on the eval pod, never the box
                return pod.evaluate(pointer, windows, num_samples=cfg.eval.num_samples,
                                    contract=contract, quantile_head=quantile_head,
                                    quantile_batch=qbatch)
            return evaluate_checkpoint(self.fetch(pointer), windows,
                                       num_samples=cfg.eval.num_samples, device=self.device,
                                       contract=contract, quantile_head=quantile_head,
                                       quantile_batch=qbatch)

        return judge_manifest(manifest, king_pointer=manifest.king_pointer,
                              king_scores=_eval(manifest.king_pointer), evaluate=_eval,
                              fetch_dir=self.fetch, cfg=cfg.lane_b, seed=str(round_id),
                              **({"merge_check": self._merge_check(pod)}
                                 if pod is not None else {}),
                              strict=cfg.lane_b.merge_guard_at_settlement(int(block)),
                              sentinel=sentinel,
                              **({"anchor_pointer": anchor_pointer} if anchor_pointer else {}))

    def _merge_check(self, pod):
        """The pod's merge check, reusing a prepared "mergeable" answer."""
        from .lane_b_prepare import merge_check_with

        return merge_check_with(getattr(self, "_preparer", None), self.bench, pod.merge_check)

    def merge_scores(self, pointer: str, block: int) -> list:
        from .lane_b_offload import local_eval_allowed

        pod = None if local_eval_allowed() else self._pod()
        from ..validator.evaluator import evaluate_checkpoint
        from ..validator.expert_lane import verdict_windows

        cfg = self.r.cfg
        windows = verdict_windows(cfg, self.source, int(block), block=int(block))
        from ..shared.config import shared_ladder_quantile_batch

        quantile_head = bool(cfg.quantile_head_at(int(block)))
        contract = cfg.training.at_block(int(block)).primary_size
        qbatch = shared_ladder_quantile_batch(cfg.eval, int(block)) if quantile_head else 1
        if pod is not None:                      # on the eval pod, never the box
            return pod.evaluate(pointer, windows, num_samples=cfg.eval.num_samples,
                                contract=contract, quantile_head=quantile_head,
                                quantile_batch=qbatch)
        return evaluate_checkpoint(
            self.fetch(pointer), windows, num_samples=cfg.eval.num_samples,
            device=self.device, contract=contract,
            quantile_head=quantile_head, quantile_batch=qbatch)


def lane_b_queue_of(runner):
    """The lane-B funding queue the trainer shares with ``cascade-intake``:
    ``lane_b_queue.json`` beside the resolved ``[round] funded_queue_path``
    (None while funding is off)."""
    from .runner_common import funded_queue_of

    return funded_queue_of(runner, lane="b")


LANE_B_STATUS_KEY = "status/lane_b.json"
STATUS_MIN_INTERVAL_S = 60.0
_STATUS_LAST: dict[int, tuple[float, str]] = {}


def publish_lane_b_status(sched, block: int, *, min_interval_s: float = STATUS_MIN_INTERVAL_S,
                          clock=None) -> bool:
    """Publish ``status/lane_b.json`` (public, pull-only) from a rolling
    scheduler's lane-B driver: the funded queue (depth per status, entries
    with their targets), open legs, finished legs awaiting a settlement and
    the recent outcomes. Rate-limited; each publication carries a fresh
    timestamp so an unchanged queue can remain a current observation. Never raises; True when a document was uploaded."""
    from .lane_status import publish_lane_status

    return publish_lane_status(sched, block, lane="b", status_key=LANE_B_STATUS_KEY,
                               last_published=_STATUS_LAST, min_interval_s=min_interval_s,
                               clock=clock, logger=log)


def lane_breaker_failure(exc: BaseException) -> dict:
    """``{"failure": LegFailure(no_capacity)}`` for a leg that failed on an
    operator lane the circuit breaker just took out (``exc.lane_breaker``),
    else ``{}`` — the same verdict the rolling path records
    (:func:`cascade.trainer.runner_rolling.record_lane_breaker_victim`): the
    LANE failed, so the leg requeues unburned without operator-fault backoff."""
    from .faults import _NoOperatorLanes
    from .lane_c_driver import LegFailure

    if isinstance(exc, _NoOperatorLanes):
        return {"failure": LegFailure(
            f"no operator lanes left — requeued without backoff ({exc})",
            miner_fault=False, error_class="no_capacity", burn=False)}
    if not getattr(exc, "lane_breaker", False):
        return {}

    return {"failure": LegFailure(
        f"LANE BREAKER: the operator lane this leg ran on failed several different legs "
        f"in a row and left the rotation — requeued without backoff ({str(exc)[-300:]})",
        miner_fault=False, error_class="no_capacity", burn=False)}
