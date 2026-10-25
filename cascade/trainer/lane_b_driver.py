"""Lane B (DEC-CA-0050) orchestrator driver — the trainer's side of the
domain-expert competition, hooked into the rolling scheduler at three points:

* ``tick``     — every scheduler tick: read lane-B commitments
                 (``metro-v1:expert:<domain>:hippius:<ref>``) into the queue,
                 measure routing saturation (previous era's king vs the
                 current one), check the opening gate (a one-way latch), plan
                 an ESFT selection on the CURRENT era king and dispatch at
                 most ``legs_per_domain`` per domain;
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

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol

import numpy as np

from ..shared.atomic import read_text_with_backup, write_text_atomic
from . import expert_lane as _tel
from .expert_lane import (
    LaneBQueue,
    LaneBSubmission,
    OpenLeg,
    RoutingProbe,
    gate_open,
    routing_metrics,
    routing_stale_union,
    settle_leg,
    union_selection,
)
from .runner_generation import leg_corpus

log = logging.getLogger("cascade.trainer.lane_b")

# outcomes of a funded leg that ended without a manifest entry (status rows)
FUNDED_REQUEUED, FUNDED_MINER_FAULT = "requeued", "miner_fault"


class LaneBOps(Protocol):
    def commitments(self) -> list: ...                       # chain Commitment rows
    def contexts(self, domain: str, block: int) -> list: ...  # affinity slice
    def fetch(self, pointer: str) -> Path: ...
    def upload(self, ckpt_dir: Path, repo: str) -> str: ...  # -> trained pointer
    def dispatch_leg(self, leg: OpenLeg, era_index: int, block: int, done) -> None: ...
    def bench_rows(self, ckpt_dir: Path) -> dict | None: ...
    def publish_expert_manifest(self, manifest) -> None: ...
    def judge(self, manifest, round_id: str, block: int) -> list: ...  # ExpertJudgement list
    def merge_scores(self, pointer: str, block: int) -> list: ...  # held-out acceptance scores
    # DEC-CA-0059 item 8 (optional: an ops without them dispatches unfunded,
    # the pre-funding behaviour the unit fakes keep):
    #   miner_funded: bool                       — legs need a funded entry
    #   def funded_queue(self): ...              — funding LaneBQueue | None
    #   def hotkey_busy(self, hotkey) -> bool: ...  — a lane-A / lane-C leg runs


@dataclass
class FinishedExpertLeg:
    leg: dict                  # asdict(OpenLeg)
    leg_pointer: str
    corpus_digest: str
    train_block: int
    gpu_name: str = ""


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
    # legs dropped at settlement without a verdict: {round_id, hotkey, uid,
    # domain, gen_ref, leg_pointer, base_king_ref, king_pointer, outcome
    # ("stale" | "routing_stale" | "merge_refused"), detail}
    outcomes: list[dict] = field(default_factory=list)


SETTLED_KEEP = 256
KINGS_KEEP = 16
HISTORY_KEEP = 512
OUTCOMES_KEEP = 256


def _slice_digest(contexts: list) -> str:
    h = hashlib.sha256()
    for c in contexts:
        a = np.ascontiguousarray(np.asarray(c, dtype=np.float64).reshape(-1))
        h.update(len(a).to_bytes(8, "little"))
        h.update(a.tobytes())
    return h.hexdigest()


class LaneBDriver:
    def __init__(self, cfg, ops: LaneBOps, state_path: Path):
        self.cfg = cfg                 # ChainConfig
        self.ops = ops
        self.path = Path(state_path)
        self._lock = threading.Lock()
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
        self._acceptance_jobs: dict = {}
        # Scientific expiry is not physical termination: a stale worker can
        # still own the hotkey's pod and harvest directory until its callback.
        self._running: dict[int, OpenLeg] = {}

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
        for leg in orphans:
            log.warning("lane B: open leg %s/%s (base era %d) has no thread in this process "
                        "— requeued uncharged", leg.sub.hotkey[:8], leg.sub.domain,
                        leg.base_era)
            self.queue.requeue(leg)
            if leg.funding and fq is not None:
                fq.requeue(leg.sub.hotkey, error="trainer restarted mid-leg — requeued",
                           error_class="no_capacity", burn_attempt=False, operator_billed=True)
        if orphans:
            self.save()

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
                return set()
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
                n = read_measurement(self.ops.fetch(binding["pointer"]), binding)
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
            return []
        miner = self._miner_funded()
        fq = self._funded_queue() if miner else None
        if fq is not None:
            fq.expire_stale()
        with self._lock:
            self._recover_orphans(fq)
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
            want = {t for s in self.queue.next_dispatchable(lb, eligible) for t in s.declared}
            self.save()
        # ── routing saturation + current-king probes: OUTSIDE the lock ──
        routing: dict[str, dict] = {}
        if prev:
            try:
                for d in lb.domains:
                    routing[d] = routing_metrics(self._probe(prev, king_pointer, d, block),
                                                 self._probe(king_pointer, king_pointer, d, block),
                                                 lb)
            except Exception as e:  # noqa: BLE001
                if not latched:
                    raise                # closed gate: surface it (the tick retries)
                log.warning("lane B: routing-saturation measurement failed: %s", e)
                routing = {}
        probes: dict[str, RoutingProbe] = {}
        if want and (latched or (prev and self.st.lineage_tokens >= lb.open_tokens)):
            for d in sorted(want):
                probes[d] = self._probe(king_pointer, king_pointer, d, block)
        with self._lock:
            if routing:
                self._record_routing(era_index, block, prev, king_pointer, routing)
            if not latched:
                ok, why = gate_open(self.st.lineage_tokens, routing, lb)
                self.st.routing["open"], self.st.routing["reason"] = ok, why
                if not ok:
                    self.save()
                    log.info("lane B: gate closed (%s)", why)
                    return []
                self.st.gate_opened_block = int(block)
                log.info("lane B: gate OPEN at block %d (era %d, king %s vs previous %s; "
                         "lineage %.3g tokens) — latched", block, era_index, king_pointer,
                         prev, self.st.lineage_tokens)
            started = []
            for sub in self.queue.next_dispatchable(lb, eligible):
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
                    e = funded[sub.hotkey]
                    if not fq.mark_in_flight(sub.hotkey, e.ref, target_boundary=0,
                                             era_index=int(era_index), started_block=int(block)):
                        continue
                    from .progress import funded_attempt_id

                    funding = {"entry_ref": e.ref, "started_block": int(block),
                               "attempt_id": funded_attempt_id(fq, sub.hotkey, e.ref, int(block)),
                               **({"gpu": str(e.gpu),
                                   "gpu_allow_fallback": bool(e.gpu_allow_fallback),
                                   "gpu_max_price_per_hour": float(e.gpu_max_price_per_hour)}
                                  if getattr(e, "gpu", "") else {})}
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
            try:
                self.ops.dispatch_leg(leg, era_index, block, self._leg_done)
            except Exception as e:
                from .lane_c_driver import LegFailure

                self._leg_done(leg, None, failure=LegFailure(
                    f"expert dispatch failed: {e}", miner_fault=False,
                    error_class="operator", burn=False))
        return started

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
        probe = _tel.probe_target_routing(self.ops.fetch(pointer), ctx, self.cfg.lane_b, domain)
        with self._cache_lock:
            self._probes[key] = probe
            while len(self._probes) > 2 * self._cache_cap():
                self._probes.popitem(last=False)
        return probe

    def _cache_cap(self) -> int:
        lb = self.cfg.lane_b
        return max(8, 3 * len(getattr(lb, "targets", None) or lb.domains))

    def _leg_done(self, leg: OpenLeg, leg_pointer: str | None, corpus_digest: str = "",
                  train_block: int = 0, gpu_name: str = "", failure=None) -> None:
        """A leg returned. ``failure`` (a funded leg's
        :class:`~cascade.trainer.lane_c_driver.LegFailure`) decides its queue
        entry: a miner fault ends it; anything else requeues it — a capacity
        miss or an operator fault unburned, an infra fault on one of the
        entry's bounded attempts."""
        fq = self._funded_queue() if leg.funding else None
        with self._lock:
            self._running.pop(id(leg), None)
            if leg not in self.queue.open:
                # expired (or requeued) while it ran: its slot is gone and a
                # leg that old is dropped at settlement anyway
                log.warning("lane B: leg %s/%s finished after it was expired — discarded",
                            leg.sub.hotkey[:8], leg.sub.domain)
                return
            if leg_pointer:
                self.queue.finish(leg, ok=True)
                self.st.finished.append(asdict(FinishedExpertLeg(
                    leg=asdict(leg), leg_pointer=leg_pointer, corpus_digest=corpus_digest,
                    train_block=int(train_block), gpu_name=gpu_name)))
            elif leg.funding and not getattr(failure, "miner_fault", False):
                msg = str(getattr(failure, "msg", "") or "leg produced no checkpoint")
                cls = str(getattr(failure, "error_class", "") or "operator")
                self.queue.requeue(leg)          # the funded entry bounds the retries
                if fq is not None:
                    fq.requeue(leg.sub.hotkey, error=msg, error_class=cls,
                               burn_attempt=bool(getattr(failure, "burn", False)))
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

    def _funded_outcome(self, sub: LaneBSubmission, outcome: str, detail) -> None:
        self.st.outcomes.append({"round_id": "", "hotkey": sub.hotkey, "uid": int(sub.uid),
                                 "domain": sub.domain, "gen_ref": sub.gen_ref,
                                 "leg_pointer": "", "base_king_ref": "", "king_pointer": "",
                                 "outcome": outcome, "detail": detail,
                                 **({"targets": list(sub.targets)} if sub.targets else {})})
        del self.st.outcomes[:-OUTCOMES_KEEP]

    # ── settlement ───────────────────────────────────────────────────────────

    def settle(self, *, round_id: str, block: int, era_index: int, king_pointer: str,
               work_dir: Path) -> list:
        """Rebase every finished leg onto this settlement's king, publish the
        expert manifest, self-judge it, and queue the winners for merge.

        Called BEFORE the lane-A manifest is published: the lane-A manifest
        binds :meth:`expert_manifest_digest`, so no validator can see the
        settlement without its expert legs. Finished legs leave the queue only
        once the whole settlement succeeded — an I/O failure anywhere keeps
        them for the next settlement (and binds nothing this round).
        Idempotent per ``round_id``."""
        from ..shared.expert_manifest import ExpertEntry, ExpertManifest, leg_label
        from .moe_model import MergeError

        lb = self.cfg.lane_b
        if not lb.active(block):
            return []
        rid = str(round_id)
        with self._lock:
            if rid in self.st.settled or rid in self._abandoned:
                return []
            todo = [dict(f) for f in self.st.finished]
        entries, used, dropped = [], [], []
        king_dir = self.ops.fetch(king_pointer) if todo else None
        king_rows = (self.ops.bench_rows(king_dir)
                     if todo and lb.require_public else None)
        if todo and lb.require_public:
            from ..eval.public_guards import MissingPublicGuard, pair_public_rows

            try:
                pair_public_rows(king_rows, king_rows, require=lb.require_public,
                                 min_items=lb.min_public_items)
            except MissingPublicGuard as e:
                raise OSError(f"lane-B king public benchmarks unavailable: {e}") from e

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
            sel = {int(k): v for k, v in leg["selection"].items()}
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
            try:
                rebased = settle_leg(king_dir, self.ops.fetch(leg["base_king_ref"]),
                                     self.ops.fetch(f["leg_pointer"]), sel, out,
                                     refuse_nonfinite=lb.merge_guard_at_settlement(block))
            except MergeError as e:
                log.warning("lane B: %s/%s refused at settlement: %s", sub["hotkey"][:8],
                            sub["domain"], e)
                _drop(f, "merge_refused", str(e))
                continue
            ptr = self.ops.upload(rebased, f"ckpt-r{rid}-expert-{label}"
                                           f"-u{sub['uid']}")
            rows = {}
            if king_rows is not None:
                rows = {"king": king_rows, "candidate": self.ops.bench_rows(rebased) or {}}
                try:
                    pair_public_rows(rows["king"], rows["candidate"], require=lb.require_public,
                                     min_items=lb.min_public_items)
                except MissingPublicGuard as e:
                    raise OSError(f"lane-B candidate public benchmarks unavailable: {e}") from e
            entries.append(ExpertEntry(
                miner_hotkey=sub["hotkey"], miner_uid=int(sub["uid"]), gen_ref=sub["gen_ref"],
                domain=sub["domain"], selection=sel,
                # the rows other legs held when this one was planned (its
                # selection skipped them) — recorded, not re-derived
                exclude={int(k): list(v) for k, v in (leg.get("exclude") or {}).items() if v},
                base_king_pointer=leg["base_king_ref"], rebased_pointer=ptr,
                corpus_digest=f["corpus_digest"], train_block=int(f["train_block"]),
                commit_block=int(sub["commit_block"]), gpu_name=f.get("gpu_name", ""),
                public_rows=rows, targets=declared if len(declared) > 1 else ()))
            used.append((ptr, f, sel))
        judgements: list = []
        digest = ""
        if entries:
            manifest = ExpertManifest(round_id=rid, created_block=int(block),
                                      king_pointer=king_pointer,
                                      lineage_tokens=self.st.lineage_tokens,
                                      entries=tuple(entries))
            self.ops.publish_expert_manifest(manifest)
            judgements = self.ops.judge(manifest, rid, int(block))
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
                                      settle_block=block, tiers=lb.full_pay_tiers,
                                      qualify_only=True):
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
            for ptr, _f, sel in used:
                sub = _f["leg"]["sub"]
                progress_outcomes.append({"round_id": rid, "hotkey": sub["hotkey"],
                                "gen_ref": sub["gen_ref"], "domain": sub["domain"],
                                "targets": list(sub.get("targets") or [sub["domain"]]),
                                "commit_block": sub["commit_block"],
                                "started_block": (_f["leg"].get("funding") or {}).get("started_block"),
                                "attempt_id": (_f["leg"].get("funding") or {}).get("attempt_id"),
                                "leg_pointer": ptr,
                                "outcome": ("qualified" if ptr in passed else
                                            "qualification_not_selected" if ptr in checks_passed else
                                            "qualification_rejected")})
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
            self._settle_funding(todo, dropped)
            self.st.outcomes.extend(dropped)
            self.st.submission_results.extend(progress_outcomes)
            del self.st.submission_results[:-OUTCOMES_KEEP]
            del self.st.outcomes[:-OUTCOMES_KEEP]
            self.st.settled[rid] = {"digest": digest}
            for old in list(self.st.settled)[:-SETTLED_KEEP]:
                del self.st.settled[old]
            self.save()
        return judgements

    def _settle_funding(self, todo: list[dict], dropped: list[dict]) -> None:
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
                fq.mark_done(hk, expect_ref=funding["entry_ref"],
                             expect_started_block=funding.get("started_block"))

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
                "queue": queue,
                "pending": len(self.queue.pending),
                "open": [{"hotkey": leg.sub.hotkey, "uid": int(leg.sub.uid),
                          "targets": list(leg.sub.declared), "ref": leg.sub.gen_ref,
                          "base_era": int(leg.base_era),
                          "experts": sum(len(v) for v in leg.selection.values()),
                          "gpu": str(leg.funding.get("gpu", "") or ""),
                          "funded": bool(leg.funding)} for leg in self.queue.open],
                "finished": len(self.st.finished),
                "pending_merges": len(self.st.pending_merges),
                "submission_progress": lane_b_progress(self.st),
                "outcomes": list(self.st.outcomes[-16:]),
            }

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
                    try:
                        job["result"] = self.init_for(
                            era_index, init_pointer, work_dir, era_start=era_start, _stage_only=True)
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
        import torch
        from safetensors.torch import load_file

        from .ckpt_digest import save_tensors_hashed
        from .expert_lane import (
            _moe_names,
            apply_merge_deltas,
            apply_merges,
            merge_deltas,
            merge_power_ema,
            merge_rows,
            reset_optimizer_rows,
        )
        from .moe_trainer import POWER_EMA_FILE
        from .toto2_trainer import OPTIM_STATE_FILE, STABLE_WEIGHTS_FILE

        init_dir = self.ops.fetch(init_pointer)
        names = _moe_names(init_dir)
        cache: dict[str, dict] = {}

        def _load(p: str) -> dict:
            if p not in cache:
                cache[p] = load_file(str(self.ops.fetch(p) / "weights.safetensors"))
            return cache[p]

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
                pending, init_pointer=init_pointer, init_dir=init_dir,
                work_dir=Path(work_dir), block=int(era_start), ops=self.ops,
                cfg=self.cfg.lane_b, names=names, load=_load)
            pending = []  # accepted/rejected decisions replace the legacy loop
        for m in pending:
            cells = merge_rows(m)
            if cells & taken:
                log.warning("lane B: merge %s overlaps an earlier merge's rows — waits for the "
                            "next era init", m.get("rebased_pointer"))
                continue
            if any(not bool(torch.isfinite(d).all()) for _, _, d in merge_deltas([m], _load, names)):
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
        state = apply_merges(
            load_file(str(init_dir / "weights.safetensors")), chosen, _load, names,
            strict=self.cfg.lane_b.merge_guard_at_era(era_start))
        deltas = merge_deltas(chosen, _load, names)
        out = Path(work_dir) / f"merged-init-era{era_index}"
        out.mkdir(parents=True, exist_ok=True)
        import shutil

        for f in ("config.json", "model.py", "forecast_wrapper.py"):
            shutil.copyfile(init_dir / f, out / f)
        save_tensors_hashed(state, out / "weights.safetensors")
        # the lineage files follow the same merge (see expert_lane "state at merge")
        if (init_dir / STABLE_WEIGHTS_FILE).is_file():
            save_tensors_hashed(apply_merge_deltas(
                load_file(str(init_dir / STABLE_WEIGHTS_FILE)), deltas),
                out / STABLE_WEIGHTS_FILE)
        if (init_dir / OPTIM_STATE_FILE).is_file():
            flat, n = reset_optimizer_rows(load_file(str(init_dir / OPTIM_STATE_FILE)),
                                           chosen, names)
            save_tensors_hashed(flat, out / OPTIM_STATE_FILE)
            log.info("lane B: era %d init — optimizer moments reset on %d merged expert "
                     "tensor(s)", era_index, n)
        if (init_dir / POWER_EMA_FILE).is_file():
            save_tensors_hashed(merge_power_ema(load_file(str(init_dir / POWER_EMA_FILE)),
                                                deltas), out / POWER_EMA_FILE)
        ptr = self.ops.upload(out, f"ckpt-era{era_index}-merged-init")
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

    def __init__(self, runner, client, window_source, *, device: str = "cpu"):
        self.r, self.client, self.source, self.device = runner, client, window_source, device

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

    def contexts(self, domain: str, block: int) -> list:
        from ..validator.expert_lane import affinity_contexts

        return affinity_contexts(self.r.cfg, self.source, domain, block=block)

    def fetch(self, pointer: str) -> Path:
        return self.r._fetch_checkpoint_dir(pointer)

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

    def _dispatch_funded(self, leg: OpenLeg, block: int, done) -> None:
        """One lane-B leg on the MINER's key: the funded challenger path
        (``_run_funded_leg``: rent on the payer key → identity / runtime pins →
        the corpus (shards tier on a ``shards`` contract) → worker ``--role
        expert`` with the leg's rows → credential-free harvest → ingest verify
        → teardown), never benched. Configured operator fallback uses the
        shared operator pool; absent capacity requeues without charging the
        miner again. Other failures are classified like a lane-A funded leg."""
        from ..funding.gpu_choice import GpuChoice
        from ..shared.expert_manifest import leg_label
        from .faults import _FundedOperatorFallback
        from .lane_c_driver import LegFailure
        from .loop import ResolvedGenerator

        def _run():
            try:
                seeds = self._seeds()
                sched = self.r.__dict__.get("_rolling_sched")
                era = sched.state.current if sched is not None else None
                if seeds is None or era is None:
                    raise LegFailure("no era state yet", miner_fault=False,
                                     error_class="no_capacity", burn=False)
                contract = self.r.cfg.throne_contracts_at(int(era.start_block))[0].for_hours(
                    self.r.cfg.lane_b.leg_hours,
                    guard_factor=self.r.cfg.round.heat_guard_factor,
                    guard_floor_seconds=self.r.cfg.round.heat_guard_floor_seconds)
                label = leg_label(leg.sub.declared)   # the domain for a single target
                gen = ResolvedGenerator(hotkey=leg.sub.hotkey, uid=leg.sub.uid,
                                        ref=leg.sub.gen_ref)
                f = leg.funding
                self.r._funded_leg_failures.pop(gen.hotkey, None)
                try:
                    entry = self.r._run_funded_leg(
                        self.r._dispatcher(), gen, seeds, int(block), contract,
                        f"-x{label}-u{leg.sub.uid}", warm_start_ref=leg.base_king_ref,
                        contract_block=self.r._era_contract_block(era), era_index=era.index,
                        role="expert", bench=False, domain=label,
                        expert_selection=leg.selection,
                        funded_queue=self.funded_queue(),
                        # DEC-CA-0060: the lane-B entry's own choice, never lane A's
                        gpu_choice=GpuChoice(
                            gpu=str(f.get("gpu", "") or ""),
                            allow_fallback=bool(f.get("gpu_allow_fallback", False)),
                            max_price_per_hour=float(f.get("gpu_max_price_per_hour", 0.0)
                                                     or 0.0)))
                except _FundedOperatorFallback as e:   # incl. _FundedOperatorRent
                    q = self.funded_queue()
                    recovery = q.get(gen.hotkey) if q is not None else None
                    if recovery is not None and recovery.operator_billed:
                        raise LegFailure("operator-funded lane-B recovery is waiting for "
                                         "operator capacity; the miner is not charged again",
                                         miner_fault=False, error_class="operator",
                                         burn=False) from e
                    raise LegFailure(f"no marketplace capacity on the payer key ({e})",
                                     miner_fault=False, error_class="no_capacity",
                                     burn=False) from e
                except Exception as e:
                    verdict = self.r._funded_leg_failures.pop(gen.hotkey, None)
                    if verdict is None:
                        raise LegFailure(f"lane-B leg failed (operator side): {e}",
                                         miner_fault=False, error_class="operator",
                                         burn=False) from e
                    msg, miner_fault, error_class, burn = verdict
                    raise LegFailure(msg or str(e), miner_fault=miner_fault,
                                     error_class=error_class, burn=burn) from e
                done(leg, entry.trained_pointer, entry.corpus_digest, entry.train_block,
                     getattr(entry, "gpu_name", ""))
            except LegFailure as f:
                done(leg, None, failure=f)
            except Exception as e:  # noqa: BLE001 — an unclassified failure is ours
                done(leg, None, failure=LegFailure(
                    f"lane-B leg failed without a verdict (operator side): {e}",
                    miner_fault=False, error_class="operator", burn=False))

        threading.Thread(target=_run, name=f"laneb-{leg.sub.hotkey[:8]}", daemon=True).start()

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
                    entry = self.r.train_one(
                        gen, "challenger", seeds, block, contract=lc,
                        token_budget=lc.train_tokens, repo_suffix=suffix,
                        warm_start_ref=leg.base_king_ref, expert_selection=leg.selection,
                        lane_info={"lane": "B", "target": ",".join(leg.sub.declared)})
                done(leg, entry.trained_pointer, entry.corpus_digest, entry.train_block,
                     getattr(entry, "gpu_name", ""))
            except Exception as e:  # noqa: BLE001 — a failed leg frees its domain
                log.warning("lane B: leg %s/%s failed: %s", leg.sub.hotkey[:8], leg.sub.domain, e)
                done(leg, None)

        threading.Thread(target=_run, name=f"laneb-{label}", daemon=True).start()

    def bench_rows(self, ckpt_dir: Path) -> dict | None:
        from ..eval.benchmarks import run_item_rows

        ec = self.r.cfg.eval
        return run_item_rows(ckpt_dir, project_dir=ec.benchmark_project_dir,
                             max_series=ec.cascade_bench_max_series, device=self.device)

    def publish_expert_manifest(self, manifest) -> None:
        from ..shared.expert_manifest import expert_manifest_key, sign_expert_manifest

        if self.r.wallet is not None:
            manifest = sign_expert_manifest(manifest, self.r.wallet)
        else:
            log.warning("lane B: publishing an UNSIGNED expert manifest (no wallet)")
        self.r.manifest_store().put_text(expert_manifest_key(manifest.round_id),
                                         manifest.to_json(), content_type="application/json")

    def judge(self, manifest, round_id: str, block: int) -> list:
        from ..validator.evaluator import evaluate_checkpoint
        from ..validator.expert_lane import judge_manifest, verdict_windows

        cfg = self.r.cfg
        windows = verdict_windows(cfg, self.source, int(round_id), block=int(block))
        # lane B judges v2 eras: the settlement's era contract (the MoE)
        from ..shared.era import settlement_contract_block

        contract = cfg.training.at_block(
            settlement_contract_block(cfg.round, int(block))).primary_size

        def _eval(pointer: str):
            return evaluate_checkpoint(self.fetch(pointer), windows,
                                       num_samples=cfg.eval.num_samples, device=self.device,
                                       contract=contract)

        return judge_manifest(manifest, king_pointer=manifest.king_pointer,
                              king_scores=_eval(manifest.king_pointer), evaluate=_eval,
                              fetch_dir=self.fetch, cfg=cfg.lane_b, seed=str(round_id),
                              strict=cfg.lane_b.merge_guard_at_settlement(int(block)))

    def merge_scores(self, pointer: str, block: int) -> list:
        from ..validator.evaluator import evaluate_checkpoint
        from ..validator.expert_lane import verdict_windows

        cfg = self.r.cfg
        windows = verdict_windows(cfg, self.source, int(block), block=int(block))
        return evaluate_checkpoint(
            self.fetch(pointer), windows, num_samples=cfg.eval.num_samples,
            device=self.device, contract=cfg.training.at_block(int(block)).primary_size)


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
