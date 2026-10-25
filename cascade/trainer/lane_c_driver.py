"""Lane C (DEC-CA-0056) orchestrator driver — the trainer's side of the
fine-tuning lane, hooked into the rolling scheduler like lane B's driver
(:mod:`cascade.trainer.lane_b_driver`). MarketCast-only lane C (owner
2026-10-20): every target is a forward-only competition's — today only
MarketCast ``mkt:<suite>`` (below); the DEC-CA-0056 generic king fine-tune
legs (``recipe.json`` on the era king, judged on ``pool_c``) are retired.

* ``tick``   — every scheduler tick of a lane-C era: requeue orphans (legs a
               dead process left open), then dispatch at most
               ``legs_per_target`` legs per target from the LANE-C queue
               (:class:`~cascade.funding.queue.LaneCQueue`, funded through
               ``POST /v1/lane-c/fund``) while ``[marketcast]`` is active. A
               leg needs the hotkey's latest chain commitment to still be
               the funded ``metro-v1:ft:<target>:hippius:<ref>``; it runs on
               the MINER's key, fetches the ref and validates its recipe
               (invalid ⇒ miner fault, the entry ends).
* ``settle`` — at each settlement where lane C is active, BEFORE the lane-A
               manifest: the finished legs are signed into a :class:`~cascade.shared.lane_c_manifest.
               LaneCManifest` and published at ``lane_c_manifest_key``; the
               sha256 of the EXACT published text is returned for the
               lane-A manifest's ``lane_c_digest``. Write-ahead (the T2
               ``pending_publish`` rule): the signed text is persisted before
               the first byte goes out, so a crash republishes the SAME
               bytes under the same round, never a re-derived manifest.
               Idempotent per round id.

MarketCast (DEC-CA-0064): an ``mkt:<suite>`` target (dispatched only while
``[marketcast]`` is active)
fine-tunes the PINNED Timestral-28M base (``[marketcast] base_ref`` /
``base_sha256``), not the era king, from a ``marketcast.json`` recipe
(:func:`cascade.shared.marketcast_recipe.parse_marketcast_recipe`) plus an
optional sandboxed miner generator mixed at ``synthetic_fraction``
(:mod:`cascade.trainer.marketcast_runner` / :mod:`.marketcast_leg`). Its leg
never ages out with the king (no ``max_base_age_eras`` expiry, no
``base_moved`` drop) and its manifest entry carries the ``marketcast``
record (pinned base, checkpoint / recipe / data digests, and ``train_time``
— the unix time the trainer DISPATCHED the leg, the forward-only clock the
validators and the audit cut windows from). The W&B tag is ``lane:C/mkt``
(:mod:`cascade.trainer.wandb_sink`).

MarketCast report arbiter (DEC-CA-0064 review fixes 2026-10-19): while
MarketCast is active every settlement's lane-C manifest records, for the
earlier settlements still awaited, either the sha256 of the main validator's
``marketcast/round-<id>.json`` the trainer read, or ``"absent"`` once
``[marketcast] report_timeout_settlements`` settlements passed without it
(``LaneCManifest.mkt_reports``). That signed record — never a clock — is
what tells every validator when to apply a MarketCast round. A report entry
the main validator dropped after ``max_infra_deferrals`` infra deferrals is
requeued here, uncharged (the operator error class).

Lane C never touches lane A's init, king or lineage (a separate product),
and never consumes a lane-A lifetime submission. All I/O is injected
(:class:`LaneCOps`) so the policy is unit-testable; the production adapter is
:class:`RunnerLaneCOps`.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Protocol

from ..funding.queue import COMPETITION_ENDED_CLASS, OPERATOR_FAULT_CLASS, split_lane_c_ref
from ..shared.atomic import read_text_with_backup, write_text_atomic
from ..shared.config.lane_c import is_mkt_target
from .finetune import LegRecipe, RecipeError

log = logging.getLogger("cascade.trainer.lane_c")

SETTLED_KEEP = 256
SPENT_KEEP = 4096
BLOCK_SECONDS = 12.0
OUTCOMES_KEEP = 256
# the error class of an invalid recipe (miner fault: the entry ends)
RECIPE_CLASS = "recipe"


def competition_ends_before_leg(lc, target: str, block: int) -> str:
    """Trainer policy (concurrent competitions, owner 2026-10-20): why a
    queued ``target`` entry must not be dispatched at ``block`` ("" = it may)
    — its competition's ``until_block`` falls before a leg started now could
    finish (``[lane_c] leg_hours``), so it could never be judged."""
    import math

    comp = lc.competition_of(target)
    until = int(getattr(comp, "until_block", 0) or 0)
    if not until:
        return ""
    leg_blocks = int(math.ceil(float(lc.leg_hours) * 3600.0 / BLOCK_SECONDS))
    if int(block) + leg_blocks <= until:
        return ""
    return (f"lane-C competition {comp.kind!r} ends at block {until}: a leg dispatched at "
            f"block {block} could not finish before it — not trained, nothing charged "
            "(not the miner's fault)")


class LegFailure(Exception):
    """A leg's verdict: ``miner_fault`` ⇒ the lane-C entry ends (queue
    ``fail``); otherwise it requeues, ``burn`` consuming one infra attempt."""

    def __init__(self, msg: str, *, miner_fault: bool, error_class: str, burn: bool) -> None:
        super().__init__(msg)
        self.msg, self.miner_fault, self.error_class, self.burn = (
            str(msg), bool(miner_fault), str(error_class), bool(burn))


class LaneCOps(Protocol):
    def commitments(self) -> list: ...                 # chain Commitment rows (latest per hotkey)
    def queue(self): ...                               # LaneCQueue | None
    def lane_a_busy(self, hotkey: str) -> bool: ...    # a lane-A leg of the hotkey in flight
    def fetch_recipe(self, ref: str, target: str, *, submission_time: float,
                     train_time: float) -> LegRecipe: ...             # RecipeError = miner
    def train_leg(self, leg: dict, recipe: LegRecipe) -> dict: ...    # LegFailure on failure
    def sign(self, manifest): ...
    def publish(self, key: str, text: str) -> None: ...
    def spawn(self, fn, name: str) -> None: ...


@dataclass
class LaneCDriverState:
    submission_verdicts: list[dict] = field(default_factory=list)  # bounded receipt observations
    # legs dispatched and not yet finished: {hotkey, uid, target, ref,
    # entry_ref, commit_block, base_pointer, base_era, started_block}
    open: list[dict] = field(default_factory=list)
    # legs trained, waiting for a settlement: {leg, trained_pointer,
    # corpus_digest, train_block, gpu_name, recipe, marketcast}
    finished: list[dict] = field(default_factory=list)
    # round id -> {"digest", "key", "entries"} (digest "" = nothing to settle)
    settled: dict[str, dict] = field(default_factory=dict)
    # write-ahead of ONE lane-C publish: {rid, key, text, digest, pointers}
    pending: dict | None = None
    # legs that ended without a manifest entry: {hotkey, target, ref, outcome,
    # detail, block}
    outcomes: list[dict] = field(default_factory=list)
    legs_started: int = 0
    # DEC-CA-0064: settlements whose MarketCast report is still awaited, oldest
    # first: {"rid", "age"} (age = settlements since; persisted only when
    # non-empty, so a state without MarketCast is byte-identical)
    mkt_await: list[dict] = field(default_factory=list)
    # commitments that already trained a published lane-C leg
    # ("<hotkey>|<target>|<ref>|<commit_block>"): a re-fund of the SAME
    # commitment is refused (``spent``) unless the queue requeued it for an
    # operator fault (persisted only when non-empty)
    spent: list[str] = field(default_factory=list)


class LaneCDriver:
    def __init__(self, cfg, ops: LaneCOps, state_path: Path):
        self.cfg = cfg                 # ChainConfig
        self.ops = ops
        self.path = Path(state_path)
        self._lock = threading.RLock()
        self.st = self._load()
        # open legs persisted by an EARLIER process have no thread here: they
        # are requeued (uncharged) on this process's first tick
        self._recovered = False

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> LaneCDriverState:
        raw, src = read_text_with_backup(self.path, json.loads)
        if not isinstance(raw, dict):
            if src == "unreadable":
                log.error("lane C: driver state %s and its backup unreadable; starting empty",
                          self.path)
            return LaneCDriverState()
        if src == "backup":
            log.warning("lane C: driver state %s unreadable — restored from its .bak", self.path)
        known = {f.name for f in fields(LaneCDriverState)}
        extra = sorted(k for k in raw if k not in known)
        if extra:
            log.warning("lane C: driver state carries unknown key(s) %s — dropped", extra)
        return LaneCDriverState(**{k: v for k, v in raw.items() if k in known})

    def save(self) -> None:
        d = asdict(self.st)
        for k in ("mkt_await", "spent"):
            if not d.get(k):
                d.pop(k, None)
        write_text_atomic(self.path, json.dumps(d, sort_keys=True), backup=True)

    def _outcome(self, leg: dict, outcome: str, detail, block: int = 0) -> None:
        self.st.outcomes.append({"hotkey": leg.get("hotkey", ""), "target": leg.get("target", ""),
                                 "ref": leg.get("ref", ""), "outcome": outcome,
                                 "detail": detail, "block": int(block)})
        del self.st.outcomes[:-OUTCOMES_KEEP]

    # ── live hotkeys (the lane-A pod sweep keeps their payer pods) ───────────

    def live_hotkeys(self) -> set[str]:
        """Hotkeys with a lane-C leg running in THIS process (their payer pod
        must survive the lane-A pod sweep)."""
        with self._lock:
            return {leg["hotkey"] for leg in self.st.open} if self._recovered else set()

    # ── orphans / expiry ─────────────────────────────────────────────────────

    def reconcile(self, queue, *, startup: bool = False) -> int:
        """Requeue, uncharged, every leg no thread of this process runs:
        on the first tick every persisted open leg (its thread died with the
        old process), and at any time a queue entry ``in_flight`` that the
        driver neither runs nor holds finished (a crash between the queue's
        ``mark_in_flight`` and the driver's save). Caller holds the lock."""
        n = 0
        if startup:
            for leg in list(self.st.open):
                log.warning("lane C: open leg %s/%s (base era %d) has no thread in this "
                            "process — requeued uncharged", leg["hotkey"][:8], leg["target"],
                            int(leg["base_era"]))
                if queue is not None:
                    queue.requeue(leg["hotkey"], error="trainer restarted mid-leg — requeued",
                                  error_class="no_capacity", burn_attempt=False,
                                  operator_billed=True)
                self._outcome(leg, "orphaned", "trainer restarted mid-leg")
                n += 1
            self.st.open = []
        if queue is not None:
            mine = ({leg["hotkey"] for leg in self.st.open}
                    | {f["leg"]["hotkey"] for f in self.st.finished})
            for e in queue.in_flight():
                if e.hotkey in mine:
                    continue
                log.warning("lane C: queue entry %s is in flight but no leg runs it — "
                            "requeued uncharged", e.hotkey[:8])
                queue.requeue(e.hotkey, error="no leg runs this in-flight entry — requeued",
                              error_class="no_capacity", burn_attempt=False)
                n += 1
        if n:
            self.save()
        return n

    # ── tick ─────────────────────────────────────────────────────────────────

    def tick(self, block: int, era_index: int, king_pointer: str | None,
             era_start: int | None) -> list[dict]:
        lc = self.cfg.lane_c
        if not lc.era_active(era_start) or not king_pointer:
            return []
        queue = self.ops.queue()
        if queue is None:
            return []
        with self._lock:
            startup = not self._recovered
            self._recovered = True
            self.reconcile(queue, startup=startup)
            self.save()
        queue.expire_stale()
        latest = _latest_commitments(self.ops.commitments())
        started: list[dict] = []
        with self._lock:
            open_per = {t: 0 for t in lc.targets}
            for leg in self.st.open:
                open_per[leg["target"]] = open_per.get(leg["target"], 0) + 1
            busy = {leg["hotkey"] for leg in self.st.open} | {
                f["leg"]["hotkey"] for f in self.st.finished}
            for e in queue.queued():
                try:
                    target, ref = split_lane_c_ref(e.ref)
                except ValueError:
                    queue.fail(e.hotkey, error="malformed lane-C queue entry",
                               error_class="target", expect_ref=e.ref)
                    continue
                if target not in lc.targets:
                    queue.fail(e.hotkey, error=f"{target!r} is not a [lane_c] target",
                               error_class="target", expect_ref=e.ref)
                    continue
                why = competition_ends_before_leg(lc, target, block)
                if why:
                    # concurrent competitions: nothing is rented, nothing is
                    # charged — and it is not the miner's fault
                    queue.fail(e.hotkey, error=why, error_class=COMPETITION_ENDED_CLASS,
                               expect_ref=e.ref)
                    continue
                if open_per.get(target, 0) >= int(lc.legs_per_target) or e.hotkey in busy:
                    continue
                mc = getattr(self.cfg, "marketcast", None)
                if not is_mkt_target(target) or mc is None or not mc.active(block):
                    continue                 # DEC-CA-0064: only while MarketCast is live
                c = latest.get(e.hotkey)
                if c is None:
                    continue                 # no commitment visible (yet): wait (TTL bounds it)
                from ..shared.lane_c_manifest import parse_finetune_commit

                if parse_finetune_commit(getattr(c, "payload", "") or "",
                                         mkt_suites=tuple(getattr(mc, "target_suites", ()))
                                         ) != (target, ref):
                    queue.fail(e.hotkey, error="the hotkey's latest commitment is no longer "
                               f"metro-v1:ft:{target}:hippius:{ref} — re-commit and fund again",
                               error_class="superseded", expect_ref=e.ref)
                    continue
                cblock = int(getattr(c, "commit_block", 0) or 0)
                if (spent_key(e.hotkey, target, ref, cblock) in self.st.spent
                        and getattr(e, "last_error_class", "") != OPERATOR_FAULT_CLASS):
                    # the lane-B rule: one leg per commitment; an operator-fault
                    # requeue (the leg was lost on our side) still retrains
                    queue.fail(e.hotkey, error="this commitment already trained a lane-C leg "
                               "— re-commit (a new reveal) to train again",
                               error_class="spent", expect_ref=e.ref)
                    continue
                if self.ops.lane_a_busy(e.hotkey):
                    continue                 # one leg per hotkey across lanes (shared pod names)
                ck = str(getattr(c, "coldkey", "") or "")
                if ck and sum(1 for x in self.st.open if x.get("coldkey") == ck) >= int(
                            getattr(mc, "max_open_legs_per_coldkey", 1)):
                    continue                 # red-team F6: one owner's sybils share its cap
                if not queue.mark_in_flight(e.hotkey, e.ref, target_boundary=0,
                                            era_index=int(era_index), started_block=int(block)):
                    continue
                from .progress import funded_attempt_id

                leg = {"hotkey": e.hotkey, "uid": int(getattr(c, "uid", -1)), "target": target,
                       "ref": ref, "entry_ref": e.ref,
                       "commit_block": int(getattr(c, "commit_block", 0) or 0),
                       "base_pointer": str(king_pointer), "base_era": int(era_index),
                       "started_block": int(block),
                       "attempt_id": funded_attempt_id(queue, e.hotkey, e.ref, int(block))}
                # DEC-CA-0064: the dispatch time is the forward-only clock
                # (windows start after it + start_margin_seconds)
                import time

                leg["train_time"] = float(getattr(self.ops, "now", time.time)())
                if ck:
                    # red-team F6: the owner the validators' family counts
                    leg["coldkey"] = ck
                if getattr(e, "gpu", ""):
                    # DEC-CA-0060: the miner's GPU choice rides the leg (only
                    # when set — a default leg's state stays byte-identical)
                    leg.update({"gpu": str(e.gpu),
                                "gpu_allow_fallback": bool(e.gpu_allow_fallback),
                                "gpu_max_price_per_hour": float(e.gpu_max_price_per_hour)})
                self.st.open.append(leg)
                self.st.legs_started += 1
                open_per[target] = open_per.get(target, 0) + 1
                busy.add(e.hotkey)
                started.append(leg)
            if started:
                self.save()
        for leg in started:
            log.info("lane C: dispatch %s target=%s ref=%s on base %s (era %d)",
                     leg["hotkey"][:8], leg["target"], leg["ref"][:48], king_pointer, era_index)
            self.ops.spawn(lambda leg=leg: self._run_leg(leg),
                           f"lanec-{leg['target']}-{leg['hotkey'][:8]}")
        return started

    # ── one leg ──────────────────────────────────────────────────────────────

    def _run_leg(self, leg: dict) -> None:
        try:
            # M12: the embargo is checked against the COMMIT time (the
            # commitment block on the dispatch clock) before any rent
            st = float(leg["train_time"]) - (int(leg["started_block"]) - int(
                leg.get("commit_block", leg["started_block"]))) * BLOCK_SECONDS
            recipe = self.ops.fetch_recipe(leg["ref"], leg["target"], submission_time=st,
                                           train_time=float(leg["train_time"]))
            td = getattr(self.ops, "tree_digest", None)
            if callable(td):
                leg["tree_digest"] = str(td(leg["ref"]) or "")
        except RecipeError as e:
            self._leg_done(leg, failure=LegFailure(str(e), miner_fault=True,
                                                   error_class=RECIPE_CLASS, burn=True))
            return
        except LegFailure as f:
            self._leg_done(leg, failure=f)
            return
        except Exception as e:  # noqa: BLE001 — fetching the ref is our side
            self._leg_done(leg, failure=LegFailure(
                f"recipe fetch failed (operator side): {e}", miner_fault=False,
                error_class=OPERATOR_FAULT_CLASS, burn=False))
            return
        try:
            out = self.ops.train_leg(leg, recipe)
        except LegFailure as f:
            self._leg_done(leg, failure=f)
            return
        except Exception as e:  # noqa: BLE001 — an unclassified failure is ours
            self._leg_done(leg, failure=LegFailure(
                f"fine-tune leg failed without a verdict (operator side): {e}",
                miner_fault=False, error_class=OPERATOR_FAULT_CLASS, burn=False))
            return
        self._leg_done(leg, result=out, recipe=recipe)

    def _leg_done(self, leg: dict, *, result: dict | None = None,
                  recipe: LegRecipe | None = None,
                  failure: LegFailure | None = None) -> None:
        queue = self.ops.queue()
        with self._lock:
            if leg not in self.st.open:
                # expired (or requeued at a restart) while it ran: its slot is
                # gone and its entry already requeued
                log.warning("lane C: leg %s/%s finished after it was expired — discarded",
                            leg["hotkey"][:8], leg["target"])
                return
            self.st.open.remove(leg)
            if failure is None and result is not None:
                self.st.finished.append({
                    "leg": dict(leg), "trained_pointer": str(result["trained_pointer"]),
                    "corpus_digest": str(result.get("corpus_digest", "") or ""),
                    "train_block": int(result.get("train_block", leg["started_block"])),
                    "gpu_name": str(result.get("gpu_name", "") or ""),
                    "recipe": asdict(recipe) if recipe is not None else {},
                    **({"tree_digest": str(leg["tree_digest"])} if leg.get("tree_digest")
                       else {}),
                    **({"marketcast": {**dict(result["marketcast"]),
                                       **({"train_time": float(leg["train_time"])}
                                          if "train_time" in leg else {}),
                                       **({"coldkey": str(leg["coldkey"])}
                                          if leg.get("coldkey") else {})}}
                       if result.get("marketcast") else {})})
                log.info("lane C: leg %s/%s finished → %s (settles at the next boundary)",
                         leg["hotkey"][:8], leg["target"], result["trained_pointer"])
            else:
                f = failure or LegFailure("leg produced no checkpoint", miner_fault=False,
                                          error_class=OPERATOR_FAULT_CLASS, burn=False)
                if queue is not None:
                    if f.miner_fault:
                        queue.fail(leg["hotkey"], error=f.msg, error_class=f.error_class,
                                   expect_ref=leg["entry_ref"])
                    else:
                        queue.requeue(leg["hotkey"], error=f.msg, error_class=f.error_class,
                                      burn_attempt=f.burn)
                self._outcome(leg, "miner_fault" if f.miner_fault else "requeued",
                              {"class": f.error_class, "error": f.msg[:300]})
                log.warning("lane C: leg %s/%s failed [%s]%s: %s", leg["hotkey"][:8],
                            leg["target"], f.error_class,
                            " — entry ended (miner fault)" if f.miner_fault else " — requeued",
                            f.msg[:200])
            self.save()

    # ── settlement ───────────────────────────────────────────────────────────

    def settle(self, *, round_id: str, block: int, era_index: int, base_pointer: str,
               created_block: int | None = None, base_round_id: str = "",
               pool_block: int = 0) -> str:
        """Publish the lane-C manifest of this settlement (BEFORE the lane-A
        manifest) and return the sha256 of its exact text ("" = nothing to
        settle). Every finished leg settles (a MarketCast leg's pinned base
        never moves, so nothing ages out with the king). A publish failure raises with the write-ahead kept: the SAME bytes go
        out when this round is retried; a different round discards it (the
        lane-A manifest of that round went out without it) and re-settles its
        legs. Idempotent per ``round_id``."""
        from ..shared.lane_c_manifest import (
            LaneCEntry,
            LaneCManifest,
            dump_lane_c_manifest,
            lane_c_manifest_key,
            lane_c_text_digest,
        )

        lc = self.cfg.lane_c
        if not lc.active(block):
            return ""
        rid = str(round_id)
        queue = self.ops.queue()
        with self._lock:
            if rid in self.st.settled:
                return str(self.st.settled[rid].get("digest", "") or "")
            pend = self.st.pending
            if pend is not None and pend.get("rid") != rid:
                log.warning("lane C: discarding the unpublished/unbound lane-C manifest of "
                            "round %s — its lane-A settlement went out without it; its legs "
                            "settle now", pend.get("rid"))
                self.st.pending = pend = None
                self.save()
            if pend is None:
                todo = list(self.st.finished)   # DEC-CA-0064: the pinned base never moves
                mkt_reports, mkt_await = self._mkt_arbiters(rid, block)
                if not todo and not mkt_reports:
                    self.st.settled[rid] = {"digest": "", "key": "", "entries": 0}
                    self.st.mkt_await = mkt_await
                    self._trim_settled()
                    self.save()
                    return ""
                todo.sort(key=lambda f: (int(f["train_block"]), f["leg"]["hotkey"]))
                entries = tuple(LaneCEntry(
                    miner_hotkey=f["leg"]["hotkey"], miner_uid=int(f["leg"]["uid"]),
                    target=f["leg"]["target"], recipe_ref=f["leg"]["ref"],
                    trained_pointer=f["trained_pointer"], train_block=int(f["train_block"]),
                    funded_by="miner",
                    marketcast=dict(f.get("marketcast") or {}),
                    tree_digest=str(f.get("tree_digest", "") or ""))
                    for f in todo)
                manifest = LaneCManifest(
                    round_id=rid, era_index=int(era_index),
                    created_block=int(block if created_block is None else created_block),
                    base_pointer=str(base_pointer), base_round_id=str(base_round_id or rid),
                    pool_block=int(pool_block), entries=entries, mkt_reports=mkt_reports)
                manifest = self.ops.sign(manifest)
                text = dump_lane_c_manifest(manifest)
                pend = {"rid": rid, "key": lane_c_manifest_key(rid), "text": text,
                        "digest": lane_c_text_digest(text),
                        "pointers": [f["trained_pointer"] for f in todo]}
                if mkt_reports or mkt_await != self.st.mkt_await:
                    pend["mkt_await"] = mkt_await
                self.st.pending = pend
                self.save()            # write-ahead BEFORE the first byte goes out
            else:
                log.warning("lane C: settlement %s was being published — republishing the "
                            "same manifest", rid)
        self.ops.publish(str(pend["key"]), str(pend["text"]))
        with self._lock:
            pointers = set(pend.get("pointers") or ())
            done = [f for f in self.st.finished if f["trained_pointer"] in pointers]
            for f in done:
                if queue is not None:
                    queue.mark_done(f["leg"]["hotkey"], expect_ref=f["leg"]["entry_ref"],
                                    expect_started_block=f["leg"]["started_block"])
                leg = f["leg"]
                k = spent_key(leg["hotkey"], leg["target"], leg["ref"],
                              int(leg.get("commit_block", 0) or 0))
                if k not in self.st.spent:
                    self.st.spent.append(k)
            del self.st.spent[:-SPENT_KEEP]
            self.st.finished = [f for f in self.st.finished if f not in done]
            self.st.settled[rid] = {"digest": str(pend["digest"]), "key": str(pend["key"]),
                                    "entries": len(pointers), "text": str(pend["text"]),
                                    "submission_identities": [
                                        {"hotkey": f["leg"]["hotkey"], "ref": f["leg"]["ref"],
                                         "targets": [f["leg"]["target"]],
                                         "commit_block": f["leg"].get("commit_block"),
                                         "started_block": f["leg"]["started_block"],
                                         "attempt_id": f["leg"].get("attempt_id"),
                                         "trained_pointer": f["trained_pointer"]} for f in done]}
            if "mkt_await" in pend:
                self.st.mkt_await = list(pend["mkt_await"])
            self._trim_settled()
            self.st.pending = None
            self.save()
        log.info("lane C: settlement %s — %d fine-tune leg(s) published at %s (sha256 %s)",
                 rid, len(pointers), pend["key"], str(pend["digest"])[:16])
        return str(pend["digest"])

    def _mkt_arbiters(self, rid: str, block: int) -> tuple[dict, list]:
        """``(mkt_reports, next mkt_await)`` for settlement ``rid``: every
        earlier awaited settlement whose report the trainer can now read is
        recorded by the sha256 of the exact text, every one awaited for
        ``report_timeout_settlements`` settlements as ``"absent"``; ``rid``
        itself joins the awaited list (its report is published while the
        validators judge it). Inert ⇒ ``({}, [])`` and nothing is read.
        Caller holds the lock."""
        from ..shared.marketcast_report import REPORT_ABSENT, mkt_text_digest

        mc = getattr(self.cfg, "marketcast", None)
        if mc is None or not mc.active(block):
            return {}, list(self.st.mkt_await)
        read = getattr(self.ops, "read_mkt_report", None)
        out: dict = {}
        keep: list[dict] = []
        for a in self.st.mkt_await:
            r, age = str(a["rid"]), int(a.get("age", 0)) + 1
            text = None
            if read is not None:
                try:
                    text = read(r)
                except Exception as e:  # noqa: BLE001 — unreadable now: still awaited
                    log.warning("lane C: MarketCast report of %s unreadable (%s)", r, e)
            if text is not None:
                out[r] = mkt_text_digest(text)
                self._mkt_infra_requeue(text)
            elif age >= int(mc.report_timeout_settlements):
                out[r] = REPORT_ABSENT
                log.warning("lane C: MarketCast report of %s absent after %d settlement(s) — "
                            "recorded absent (its entries defer as infra)", r, age)
            else:
                keep.append({"rid": r, "age": age})
        if not any(a["rid"] == str(rid) for a in keep):
            keep.append({"rid": str(rid), "age": 0})
        return out, keep

    def _mkt_infra_requeue(self, text: str) -> None:
        """Requeue, uncharged (the operator error class), every entry a
        report says the main validator gave up on after ``max_infra_
        deferrals`` (its ``infra_reason`` names the drop): the miner's
        training was fine; only our scoring was not. Best effort."""
        from ..shared.marketcast_report import load_mkt_report

        queue = self.ops.queue()
        if queue is None:
            return
        try:
            rep = load_mkt_report(text)
        except (ValueError, KeyError, TypeError):
            return
        for e in rep.entries:
            if not (e.deferred and "max_infra_deferrals" in e.infra_reason):
                continue
            q = queue.get(e.hotkey)
            try:
                t, _ref = split_lane_c_ref(q.ref) if q is not None else ("", "")
            except ValueError:
                continue
            if q is not None and q.status == "done" and t == e.target:
                queue.requeue(e.hotkey, error="MarketCast scoring failed on the main validator "
                              "past max_infra_deferrals — requeued uncharged",
                              error_class=OPERATOR_FAULT_CLASS, burn_attempt=False,
                              operator_billed=True)

    def _trim_settled(self) -> None:
        for old in list(self.st.settled)[:-SETTLED_KEEP]:
            del self.st.settled[old]

    def lane_c_digest(self, round_id: str) -> str:
        """The digest the lane-A manifest of ``round_id`` binds ("" = none)."""
        with self._lock:
            return str(self.st.settled.get(str(round_id), {}).get("digest", "") or "")

    def published(self, round_id: str) -> dict | None:
        """``{key, text}`` of ``round_id``'s published lane-C manifest (None =
        none): the lane-A write-ahead carries it, so a lane-A replay
        republishes the identical bytes BEFORE the manifest that binds them."""
        with self._lock:
            rec = self.st.settled.get(str(round_id))
        if not rec or not rec.get("digest") or not rec.get("text"):
            return None
        return {"key": str(rec["key"]), "text": str(rec["text"])}

    # ── status (dashboards) ──────────────────────────────────────────────────

    def observe_receipt(self, receipt: dict) -> None:
        """Retain exact-entry observations when later receipts omit that entry.

        Telemetry only: this does not set slots, pay rewards or change funding.
        An older receipt cannot roll back an entry's last reported verdict.
        """
        from ..shared.submission_progress import lane_c_progress, receipt_progress

        observed = receipt_progress(receipt)
        if not observed or observed.get("status") != "scored" or type(observed.get("boundary")) is not int:
            return
        with self._lock:
            published = lane_c_progress(self.st)
            previous = list(self.st.submission_verdicts)
            changed = False
            for row in observed["lane_c"]:
                match = next((r for r in reversed(published)
                              if r.get("hotkey") == row.get("hotkey")
                              and r.get("targets") == [row.get("target")]
                              and r.get("trained_pointer")
                              and r["trained_pointer"] == row.get("trained_pointer")
                              and r.get("train_block") is not None
                              and r["train_block"] == row.get("train_block")), None)
                if not match or not match.get("attempt_id"):
                    continue
                key = (match["hotkey"], match["attempt_id"])
                old = next((r for r in self.st.submission_verdicts
                            if (r.get("hotkey"), r.get("attempt_id")) == key), None)
                if old and old["receipt_boundary"] >= observed["boundary"]:
                    continue
                result = {**match, "stage": row.get("outcome") or (
                              "not_passed" if row.get("passed") is False else "unknown"),
                          "receipt_boundary": observed["boundary"],
                          "receipt_round_id": observed.get("round_id")}
                self.st.submission_verdicts = [r for r in self.st.submission_verdicts
                                               if (r.get("hotkey"), r.get("attempt_id")) != key]
                self.st.submission_verdicts.append(result)
                changed = True
            if changed:
                del self.st.submission_verdicts[:-OUTCOMES_KEEP]
                try:
                    self.save()
                except Exception:
                    self.st.submission_verdicts = previous
                    raise

    def status(self, *, queue=None) -> dict:
        """The lane-C block of the trainer's status document: queue depth per
        status and target, open legs (slots) per target, finished legs
        waiting for a settlement, the last settlements and outcomes."""
        from ..shared.submission_progress import lane_c_progress

        lc = self.cfg.lane_c
        q = queue if queue is not None else self.ops.queue()
        counts: dict[str, int] = {}
        funded_view = None
        per_target: dict[str, dict] = {t: {"queued": 0, "open": 0, "finished": 0,
                                           "slots": int(lc.legs_per_target)}
                                       for t in lc.targets}
        if q is not None:
            try:
                funded_view = q.public_view()
                for e in q.entries():
                    counts[e.status] = counts.get(e.status, 0) + 1
                    if e.status == "queued":
                        try:
                            t, _ = split_lane_c_ref(e.ref)
                        except ValueError:
                            continue
                        if t in per_target:
                            per_target[t]["queued"] += 1
            except Exception as e:  # noqa: BLE001 — a status read never raises
                log.debug("lane C status: queue read failed: %s", e)
        with self._lock:
            for leg in self.st.open:
                if leg["target"] in per_target:
                    per_target[leg["target"]]["open"] += 1
            for f in self.st.finished:
                if f["leg"]["target"] in per_target:
                    per_target[f["leg"]["target"]]["finished"] += 1
            last = list(self.st.settled.items())[-8:]
            return {
                "from_block": int(lc.from_block), "leg_hours": float(lc.leg_hours),
                "legs_per_target": int(lc.legs_per_target),
                "queue": counts, "targets": per_target,
                "funded_queue": funded_view,
                "open": [{**{k: leg[k] for k in ("hotkey", "uid", "target", "ref", "base_era",
                                                 "started_block")},
                          "gpu": str(leg.get("gpu", "") or "")} for leg in self.st.open],
                "finished": len(self.st.finished), "legs_started": int(self.st.legs_started),
                "submission_progress": lane_c_progress(self.st),
                "settlements": [{"round_id": r, **{k: v for k, v in rec.items()
                                                   if k != "text"}} for r, rec in last],
                "outcomes": list(self.st.outcomes[-16:]),
            }


def spent_key(hotkey: str, target: str, ref: str, commit_block: int) -> str:
    """The identity of one lane-C commitment (a re-commit — a new reveal —
    is a new one, even of the same ref)."""
    return f"{hotkey}|{target}|{ref}|{int(commit_block)}"


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


class RunnerLaneCOps:
    """The production :class:`LaneCOps`: a thin adapter over the trainer's
    own machinery — chain commitments, the lane-C queue file, the Hub /
    vault fetch, the funded challenger leg path on the miner's key, the
    bench sidecar and the manifest bucket."""

    def __init__(self, runner, client, *, device: str = "cpu"):
        self.r, self.client, self.device = runner, client, device

    def commitments(self) -> list:
        from .admission import remember_owners

        r = self.r
        wd = getattr(r, "_with_deadline", None)
        rows = (list(wd(self.client.poll_commitments,
                        float(getattr(r, "chain_read_timeout_s", 60.0) or 60.0)))
                if callable(wd) else list(self.client.poll_commitments()))
        return remember_owners(r, rows)

    def queue(self):
        return lane_c_queue_of(self.r)

    def lane_a_busy(self, hotkey: str) -> bool:
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
        # DEC-CA-0059 item 8: a miner-funded lane-B leg of the hotkey runs
        lb = getattr(self.r, "lane_b_driver", None)
        live = getattr(lb, "live_hotkeys", None)
        try:
            return bool(callable(live) and hotkey in live())
        except Exception:  # noqa: BLE001
            return False

    def fetch_recipe(self, ref: str, target: str, *, submission_time: float | None = None,
                     train_time: float | None = None) -> LegRecipe:
        import hashlib
        import shutil

        from ..shared.config.lane_c import FORWARD_ONLY_REQUIRED
        from ..shared.hippius import StorageError, fetch_from_hub
        from ..shared.lane_c_manifest import tree_digest

        if not is_mkt_target(target):
            raise RecipeError(f"lane-C target {target!r}: {FORWARD_ONLY_REQUIRED}")
        d = Path(self.r.work_root) / "_lane_c_recipes" / hashlib.sha256(
            ref.encode()).hexdigest()[:16]
        try:
            shutil.rmtree(d, ignore_errors=True)
            try:
                fetch_from_hub(ref, d, self.r.hub())
            except StorageError as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status in (401, 403, 404):
                    raise RecipeError(f"recipe tree unreachable: HTTP {status} for {ref}") from e
                raise
            self._tree_digests = {**getattr(self, "_tree_digests", {}), ref: tree_digest(d)}
            # DEC-CA-0064: marketcast.json (+ an optional generator tree);
            # the embargo against the commit time, the pod clipped to
            # train_time (review M12)
            from .marketcast_runner import read_mkt_recipe

            return read_mkt_recipe(d, self.r.cfg, target=target,
                                   submission_time=submission_time, train_time=train_time)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def tree_digest(self, ref: str) -> str:
        """The :func:`~cascade.shared.lane_c_manifest.tree_digest` of the
        recipe tree last fetched for ``ref`` ("" = none)."""
        return str(getattr(self, "_tree_digests", {}).get(ref, ""))

    def _seeds(self):
        sched = self.r.__dict__.get("_rolling_sched")
        era = sched.state.current if sched is not None else None
        return (self.r._rolling_seeds(era) if era is not None else None), era

    def train_leg(self, leg: dict, recipe: LegRecipe) -> dict:
        """One MarketCast fine-tune leg on the MINER's key
        (:func:`cascade.trainer.marketcast_runner.train_mkt_leg`), never
        benched, never on an operator lane."""
        from ..shared.config.lane_c import FORWARD_ONLY_REQUIRED
        from .marketcast_runner import train_mkt_leg

        if not is_mkt_target(leg["target"]):
            raise LegFailure(f"lane-C target {leg['target']!r}: {FORWARD_ONLY_REQUIRED}",
                             miner_fault=True, error_class="target", burn=True)
        return train_mkt_leg(self, leg, recipe)

    def sign(self, manifest):
        from ..shared.lane_c_manifest import sign_lane_c_manifest

        if self.r.wallet is None:
            log.warning("lane C: publishing an UNSIGNED lane-C manifest (no wallet)")
            return manifest
        return sign_lane_c_manifest(manifest, self.r.wallet)

    def publish(self, key: str, text: str) -> None:
        self.r.manifest_store().put_text(key, text, content_type="application/json")

    def read_mkt_report(self, round_id: str) -> str | None:
        """The main validator's MarketCast report text of ``round_id`` (None
        = not published yet; a read failure raises)."""
        from ..shared.hippius import ObjectNotFound
        from ..shared.marketcast_report import mkt_report_key

        try:
            return self.r.manifest_store().get_text(mkt_report_key(round_id))
        except ObjectNotFound:
            return None

    def now(self) -> float:
        import time

        return time.time()

    def spawn(self, fn, name: str) -> None:
        threading.Thread(target=fn, name=name, daemon=True).start()


def lane_c_queue_of(runner):
    """The lane-C queue the trainer shares with ``cascade-intake``:
    ``lane_c_queue.json`` beside the resolved ``[round] funded_queue_path``
    (None while funding is off)."""
    from .runner_common import funded_queue_of

    return funded_queue_of(runner, lane="c")


LANE_C_STATUS_KEY = "status/lane_c.json"
STATUS_MIN_INTERVAL_S = 60.0
_STATUS_LAST: dict[int, tuple[float, str]] = {}


def publish_lane_c_status(sched, block: int, *, min_interval_s: float = STATUS_MIN_INTERVAL_S,
                          clock=None) -> bool:
    """Publish ``status/lane_c.json`` (public, pull-only) from a rolling
    scheduler's lane-C driver: queue depth, per-target slots (open /
    ``legs_per_target``), finished legs awaiting a settlement, the recent
    settlements and outcomes. Rate-limited; each publication carries a fresh
    timestamp so an unchanged queue can remain a current observation. Never raises; True when a document was uploaded."""
    from .lane_status import publish_lane_status

    return publish_lane_status(sched, block, lane="c", status_key=LANE_C_STATUS_KEY,
                               last_published=_STATUS_LAST, min_interval_s=min_interval_s,
                               clock=clock, logger=log)


__all__ = ["LANE_C_STATUS_KEY", "LaneCDriver", "LaneCDriverState", "LaneCOps", "LegFailure", "RunnerLaneCOps",
           "lane_c_queue_of", "publish_lane_c_status"]
