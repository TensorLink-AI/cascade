"""Lane C (DEC-CA-0056): fine-tune judging and folding for the validator.

Mirrors lane B (:mod:`cascade.validator.lane_b` / :mod:`.expert_lane`). Lane C
is a container for fine-tuning COMPETITIONS, and only a competition with a
FORWARD-ONLY eval is admitted (MarketCast-only lane C, owner 2026-10-20;
:data:`cascade.shared.config.lane_c.COMPETITION_KINDS`). Today the only one
is MarketCast: ``mkt:<suite>`` entries (DEC-CA-0064) take their own pipeline
(:mod:`cascade.validator.marketcast`): registered into a MarketCast round,
judged when a later trainer-signed lane-C manifest records that round's
report (``mkt_reports``), against the PINNED base — never against the king,
never needing one, and a missing report never holds this settlement. The
DEC-CA-0056 generic verdict (king vs fine-tune on the public-feed ``pool_c``
snapshot) is retired: a signed entry or carried pending entry on any other
target is rejected at the ``target`` stage.

What this module keeps for every competition: the signed lane-C manifest
read (bound by ``lane_c_digest``, pinned trainer signature), the
pending / confirm outcome labels (:func:`resolve_outcomes`), the slot fold
(:func:`apply_lane_c`, :func:`~cascade.shared.lane_c.win_beats`) and the
receipt rows (:func:`judgements_record`).

Read failures (manifest, chain history, a fetch) are TRANSIENT:
:class:`LaneCTransient` — the loop holds the settlement (no latch, no
receipt) until they read. State (slots AND pending) folds before the handled
marker persists and rides the ``pre_round`` snapshot, so a re-judgement
replaces it; prepared rounds are judged against a scratch copy folded in
chain order, so a catch-up sees each round's pending set exactly as a live
validator did.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..shared.config.lane_c import is_mkt_target
from ..shared.lane_c import LaneCState, PendingEntry, TargetWin
from ..shared.lane_c_manifest import LaneCEntry
from ..shared.manifest import TrainingManifest, load_manifest, verify_signature

log = logging.getLogger("cascade.validator")

class LaneCTransient(Exception):
    """Lane-C evidence could not be READ or scored (a lane-C manifest, a
    chained lane-A manifest, the commitment history, a checkpoint fetch, the
    king's scoring): the settlement HOLDS and is retried next poll."""


# ── pure judging (replayable) ────────────────────────────────────────────────


OUTCOME_PENDING = "pending"            # judge phase: the target's best passer
OUTCOME_DEFERRED = "deferred"          # judge phase: too few clusters, carried once
OUTCOME_DROPPED = "dropped"            # a carried entry that did not make it
OUTCOME_CONFIRMED = "confirmed"        # confirm phase: took the slot
OUTCOME_BELOW_HOLDER = "below holder LCB"   # confirm phase: passed, slot kept
OUTCOME_NOT_SELECTED = "not_selected"  # passed judge phase; another entry advances


@dataclass
class LaneCJudgement:
    entry: LaneCEntry
    passed: bool
    stage: str          # "target" | "stale" | "binding" | "fault" | "verdict"
    #                     (+ MarketCast only: "infra" | "duplicate")
    reason: str
    gain_lcb: float | None = None
    gain: float | None = None        # point estimate 1 − geomean(ft)/geomean(king)
    n: int = 0
    public_lcbs: dict = field(default_factory=dict)
    base_pointer: str = ""
    phase: str = "judge"             # "judge" (first stage) | "confirm"
    k: int = 1                       # Bonferroni family size (judge phase)
    alpha: float | None = None       # the level the verdict ran at
    clusters: int = 0                # distinct clusters among the target windows
    inconclusive: str = ""           # "" | "windows" | "clusters"
    carried: bool = False            # a deferred entry re-judged
    origin_round: str = ""           # the settlement that first judged it (carried)
    outcome: str = ""
    holder: dict | None = None       # confirm phase: the slot holder offered against
    # DEC-CA-0064 MarketCast entries only: the receipt row's "mkt" record
    # (report binding, after, origin_block, within_wait, coverage, …)
    mkt: dict = field(default_factory=dict)
    # a PASSING judgement's gain-tiered full-pay period (owner 2026-10-20):
    # :meth:`cascade.eval.expert_lane.FullPay.record` of ``gain_lcb`` under
    # the target's tiers (:func:`cascade.shared.lane_c.target_full_pay`),
    # computed at judge time; a confirmed win records it
    full_pay: dict | None = None


def _best_key(j: LaneCJudgement) -> tuple:
    return (-float(j.gain_lcb if j.gain_lcb is not None else float("-inf")),
            int(j.entry.train_block), j.entry.miner_hotkey)


def best_per_target(judgements: list[LaneCJudgement]) -> dict[str, LaneCJudgement]:
    """The best JUDGE-phase passer per target: highest gain LCB, ties to the
    earliest ``train_block``, then hotkey (never UID)."""
    best: dict[str, LaneCJudgement] = {}
    for j in judgements:
        if not j.passed or j.phase != "judge":
            continue
        cur = best.get(j.entry.target)
        if cur is None or _best_key(j) < _best_key(cur):
            best[j.entry.target] = j
    return best


def _holder_dict(w: TargetWin | None) -> dict | None:
    if w is None:
        return None
    return {"hotkey": w.hotkey, "gain_lcb": float(w.gain_lcb), "base_pointer": w.base_pointer,
            "win_block": int(w.win_block), "round_id": str(w.round_id)}


def _confirmed_win(j: LaneCJudgement, *, round_id: str, settle_block: int,
                   era_index: int) -> TargetWin:
    """The win a passing confirm judgement records — its full-pay tier is the
    one computed at judge time (``j.full_pay``), fixed here for good."""
    fp = j.full_pay
    if not fp:
        raise ValueError(f"lane-C confirm of {j.entry.miner_hotkey!r} on {j.entry.target!r} "
                         "carries no full-pay tier")
    return TargetWin(target=j.entry.target, hotkey=j.entry.miner_hotkey,
                     uid=int(j.entry.miner_uid), round_id=str(round_id),
                     win_block=int(settle_block), win_era=int(era_index),
                     trained_pointer=j.entry.trained_pointer, recipe_ref=j.entry.recipe_ref,
                     base_pointer=j.base_pointer, gain_lcb=float(j.gain_lcb or 0.0),
                     tier=float(fp["tier"]), full_eras=int(fp["full_eras"]),
                     below_floor=bool(fp.get("below_floor", False)))


def resolve_outcomes(state: LaneCState, judgements: list[LaneCJudgement], *, round_id: str,
                     settle_block: int, era_index: int = 0) -> None:
    """Label every judgement's ``outcome`` (in place) against ``state`` as it
    stands BEFORE this round folds: confirm passers ``confirmed`` /
    ``below holder LCB`` (:func:`win_beats` vs the holder, recorded in
    ``holder``), confirm failures ``dropped``; the target's best judge-phase
    passer ``pending``; a first-time cluster-inconclusive entry
    ``deferred``; any other carried entry ``dropped``."""
    best = best_per_target(judgements)
    shadow = LaneCState(slots=dict(state.slots))
    for j in sorted((j for j in judgements if j.phase == "confirm" and j.passed),
                    key=lambda j: (j.entry.target, _best_key(j))):
        j.holder = _holder_dict(shadow.slots.get(j.entry.target))
        w = _confirmed_win(j, round_id=round_id, settle_block=settle_block, era_index=era_index)
        j.outcome = OUTCOME_CONFIRMED if shadow.record(w) else OUTCOME_BELOW_HOLDER
    for j in judgements:
        # DEC-CA-0064: a MarketCast entry inconclusive for lack of windows /
        # clusters waits while within [marketcast] verdict_wait_blocks; an
        # infra deferral (the main validator could not compute it) waits
        # until max_infra_deferrals, never consuming that wait
        waits = (bool(j.mkt.get("within_wait")) and bool(j.inconclusive) and (
            not j.passed and j.stage == "verdict")) or (
            j.stage == "infra" and not j.mkt.get("error_class"))
        if j.phase == "confirm":
            if not j.passed:
                j.outcome = OUTCOME_PENDING if waits else OUTCOME_DROPPED
                continue
        elif best.get(j.entry.target) is j:
            j.outcome = OUTCOME_PENDING
        elif waits:
            j.outcome = OUTCOME_DEFERRED
        elif j.carried:
            j.outcome = OUTCOME_DROPPED
        else:
            j.outcome = OUTCOME_NOT_SELECTED if j.passed else OUTCOME_DROPPED


def apply_lane_c(state: LaneCState, judgements: list[LaneCJudgement], *, round_id: str,
                 settle_block: int, era_index: int = 0,
                 consume_pending: bool = True) -> list[TargetWin]:
    """Fold one judged round: every confirmed passer is offered to its slot
    (:meth:`LaneCState.record`), and — when the round resolved the pending
    set (``consume_pending``; False only for a round with no king) — the
    pending set becomes this round's ``pending`` / ``deferred`` entries.
    Returns the wins that took a slot."""
    resolve_outcomes(state, judgements, round_id=round_id, settle_block=settle_block,
                     era_index=era_index)
    wins = []
    for j in sorted((j for j in judgements if j.phase == "confirm" and j.passed),
                    key=lambda j: (j.entry.target, _best_key(j))):
        w = _confirmed_win(j, round_id=round_id, settle_block=settle_block, era_index=era_index)
        if state.record(w):
            wins.append(w)
    if consume_pending:
        state.set_pending([_pending_of(j, round_id=round_id, settle_block=settle_block)
                           for j in judgements
                           if j.outcome in (OUTCOME_PENDING, OUTCOME_DEFERRED)])
    return wins


def _pending_of(j: LaneCJudgement, *, round_id: str, settle_block: int) -> PendingEntry:
    """The pending-set record of a judgement left ``pending`` / ``deferred``.
    A lane-C entry is re-stamped with this round. A MarketCast entry carries
    its wait: a judge-phase pass starts the confirm wait here (fresh windows
    must start after this snapshot's end); a re-pended confirm keeps its
    origin; a re-deferred entry keeps its ORIGIN round (the chain audit
    follows it) and origin block."""
    kind = "confirm" if j.outcome == OUTCOME_PENDING else "deferred"
    rid, mkt = str(round_id), {}
    if is_mkt_target(j.entry.target):
        # (``round_id`` / ``settle_block`` are the MarketCast ROUND's)
        m = j.mkt
        if j.phase == "judge" and j.outcome == OUTCOME_PENDING:
            mkt = {"origin_block": int(settle_block),
                   "fresh_after": float(m.get("snapshot_end", 0.0) or 0.0)}
            if m.get("entry_round"):
                mkt["entry_round"] = str(m["entry_round"])
        else:
            mkt = {"origin_block": int(m.get("origin_block", settle_block)),
                   "fresh_after": float(m.get("fresh_after", 0.0) or 0.0)}
            if m.get("entry_round"):
                mkt["entry_round"] = str(m["entry_round"])
            if m.get("infra_deferrals"):
                mkt["infra_deferrals"] = int(m["infra_deferrals"])
            if j.stage == "infra":
                mkt["infra_at"] = int(settle_block)   # the next attempt shifts the origin
            if kind == "deferred" and j.carried and j.origin_round:
                rid = str(j.origin_round)
    entry = asdict(j.entry)
    if not entry.get("marketcast"):
        entry.pop("marketcast", None)
    return PendingEntry(target=j.entry.target, kind=kind, entry=entry,
                        base_pointer=j.base_pointer, round_id=rid, block=int(settle_block),
                        gain=j.gain, gain_lcb=j.gain_lcb, mkt=mkt)


def judgements_record(judgements: list[LaneCJudgement]) -> list[dict]:
    """The receipt's ``lane_c`` verdict rows (``phase`` / ``k`` / ``alpha`` /
    ``clusters`` / ``outcome`` let ``cascade-audit`` re-derive the
    correction, the pending set and the slot rule)."""
    out = []
    for j in judgements:
        row = {"hotkey": j.entry.miner_hotkey, "uid": int(j.entry.miner_uid),
               "target": j.entry.target, "stage": j.stage, "passed": bool(j.passed),
               "reason": j.reason, "gain_lcb": j.gain_lcb, "gain": j.gain, "n": int(j.n),
               "public_lcbs": dict(j.public_lcbs), "trained_pointer": j.entry.trained_pointer,
               "train_block": int(j.entry.train_block), "phase": j.phase, "k": int(j.k),
               "alpha": j.alpha, "clusters": int(j.clusters), "outcome": j.outcome,
               "base_pointer": j.base_pointer}
        if j.carried or j.phase == "confirm":
            row["origin_round"] = j.origin_round
        if j.phase == "confirm" and j.passed:
            row["holder"] = j.holder
            # the tier a confirmed win records (re-derived by the audit)
            row["full_pay"] = dict(j.full_pay or {})
        if j.mkt:
            row["mkt"] = dict(j.mkt)
        out.append(row)
    return out


# ── the mixin ────────────────────────────────────────────────────────────────


@dataclass
class _LaneCRound:
    round_id: str
    block: int
    rows: list
    judgements: list
    era_index: int = 0
    apply: bool = True
    consume_pending: bool = True     # False: no king — the pending set carries over
    mkt: object = None               # DEC-CA-0064 MktPlan (None = no MarketCast change)


class LaneCMixin:
    """Lane-C state, judging, catch-up and the payout record."""

    def _lane_c_state_path(self) -> Path:
        return Path(str(self.cfg.validator.state_db_path) + ".lane_c.json")

    def _lane_c(self) -> LaneCState:
        """The lane-C state, loaded once. Absent = fresh; TORN = discarded
        loudly and rebuilt by catch-up (``last_round`` None)."""
        from ..shared.lane_c import load_state

        st = getattr(self, "_lane_c_state", None)
        if st is None:
            st = load_state(self._lane_c_state_path())
            if st is None:
                log.error("lane C: state %s is unreadable — discarded; rebuilding from "
                          "activation by catch-up", self._lane_c_state_path())
                st = LaneCState()
            self._lane_c_state = st
        return st

    def _persist_lane_c(self) -> None:
        from ..shared.lane_c import save_state_atomic

        try:
            save_state_atomic(self._lane_c(), self._lane_c_state_path())
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist lane C state: %s", e)
            raise

    # ── payouts ──

    def _lane_c_pay(self, client: object, n_uids: int, block: int) -> dict:
        """The ``lane_c_pay`` record for a vector paid at settlement ``block``
        (holders resolved to their CURRENT uid; deregistered ones unpaid)."""
        from ..shared.era import era_length_blocks
        from ..shared.lane_c import lane_c_pay_record

        st = self._lane_c()
        uid_of = {}
        for w in st.slots.values():
            uid = client.uid_for_hotkey(w.hotkey)  # type: ignore[attr-defined]
            if uid is not None:
                uid_of[w.hotkey] = int(uid)
        eb = int(era_length_blocks(self.cfg.round, int(block)))
        return lane_c_pay_record(st, block=int(block), era_blocks=eb, cfg=self.cfg.lane_c,
                                 n_uids=n_uids, uid_of=uid_of,
                                 mc=getattr(self.cfg, "marketcast", None))

    def _lane_c_receipt_kw(self, rows: list[dict] | None) -> dict:
        pay = getattr(self, "_last_lane_c_pay", None)
        if rows is None and pay is not None:
            rows = []
        return {"lane_c": rows, "lane_c_pay": pay}

    def _lane_c_settle_block(self, client: object) -> int | None:
        """The block a lane-C vector is paid at: this session's last judged
        settlement, else the PERSISTED last settled lane-C boundary, else the
        current settlement boundary (every validator floors alike)."""
        import contextlib

        if not self.cfg.lane_c.from_block:
            return None
        b = getattr(self, "_lane_c_block", None)
        if b is None:
            b = self._lane_c().settled_block
        if b is None:
            with contextlib.suppress(Exception):
                cur = int(client.current_block())  # type: ignore[attr-defined]
                grid = self._lane_b_epoch_blocks(cur)
                b = cur // grid * grid
        return b

    # ── reading ──

    def _lane_c_read(self, key: str) -> str:
        store = getattr(self, "_lane_c_store", None) or getattr(self, "_lane_b_store", None)
        if store is None:
            raise LaneCTransient("no manifest store wired for lane C")
        try:
            return store.get_text(key)
        except Exception as e:  # noqa: BLE001 — any read failure is a transient
            raise LaneCTransient(f"{key} unreadable: {type(e).__name__}: {e}") from e

    def _lane_c_load_settlement(self, round_id: str) -> TrainingManifest:
        from ..shared.hippius import manifest_round_key

        return load_manifest(self._lane_c_read(manifest_round_key(round_id)))

    def _lane_c_commitments(self) -> list | None:
        fn = getattr(self, "commitment_history_fn", None)
        if fn is None:
            return None
        try:
            return list(fn())
        except Exception as e:  # noqa: BLE001 — chain RPC failure: a transient
            raise LaneCTransient(f"commitment history unavailable: {type(e).__name__}: "
                                 f"{e}") from e

    def _lane_c_judge_round(self, manifest: TrainingManifest,
                            state: LaneCState | None = None) -> _LaneCRound:
        """Fetch + verify ``manifest``'s bound lane-C manifest and judge its
        MarketCast part against ``state`` (default: the live state; a prepare
        passes its chain-ordered scratch copy). Deterministic given the bytes
        and the state; READ failures raise :class:`LaneCTransient`."""
        from ..shared.era import settlement_era
        from ..shared.lane_c_manifest import (
            lane_c_manifest_key,
            lane_c_text_digest,
            load_lane_c_manifest,
            verify_lane_c_manifest,
        )

        st = self._lane_c() if state is None else state
        rid, block = str(manifest.round_id), self._epoch_start_block(manifest)
        try:
            era_index = int(settlement_era(self.cfg.round, block).index)
        except Exception:  # noqa: BLE001 — no era grid ⇒ era 0
            era_index = 0
        if rid in st.applied and st.last_round == rid and st.last_rows is not None:
            # a crash-replay of the round already folded: its rows as judged
            return _LaneCRound(rid, block, list(st.last_rows), [], era_index)
        rows: list[dict] = []
        cm = None
        sha = str(getattr(manifest, "lane_c_digest", "") or "")
        if sha:
            text = self._lane_c_read(lane_c_manifest_key(rid))
            if lane_c_text_digest(text) != sha:
                log.warning("lane C: manifest for %s does not match the signed binding; "
                            "lane-C part rejected, lane A proceeds", rid)
                rows.append({"error": "lane_c_digest_mismatch"})
            else:
                try:
                    cm = load_lane_c_manifest(text)
                except Exception as e:  # noqa: BLE001 — the bound bytes are malformed: a verdict
                    rows.append({"error": f"lane_c_manifest_malformed: {e}"})
                if cm is not None and str(cm.round_id) != rid:
                    rows.append({"error": "lane_c_round_mismatch"})
                    cm = None
                if cm is not None and self.verify_signatures and not verify_lane_c_manifest(
                        cm, self.cfg.manifest.trainer_hotkey):
                    log.warning("lane C: manifest for %s fails signature; ignored", rid)
                    rows.append({"error": "bad_signature"})
                    cm = None
        # DEC-CA-0064: MarketCast first — its own pipeline, judged against the
        # pinned base from reports a trainer-signed manifest arbitrates; it
        # never holds this settlement and never needs a king
        mplan, mjs = self._mkt_settle(manifest, cm, rid=rid, block=block, era_index=era_index,
                                      st=st, commitments_fn=self._lane_c_commitments)
        # MarketCast-only lane C (owner 2026-10-20): MarketCast is the only
        # competition. A non-mkt entry cannot reach here — LaneCEntry refuses
        # any target outside a forward-only competition, so such a manifest
        # is ``lane_c_manifest_malformed`` above (the generic pool_c verdict
        # is retired).
        return _LaneCRound(rid, block, rows + judgements_record(mjs), mjs, era_index,
                           mkt=mplan)

    @staticmethod
    def _lane_c_fold(st: LaneCState, r: _LaneCRound) -> list[TargetWin]:
        """Fold one prepared round into ``st`` (idempotent per round id)."""
        wins: list[TargetWin] = []
        if not r.apply:
            return wins
        if r.round_id not in st.applied:
            from .marketcast import fold_mkt_plan

            js = [j for j in r.judgements if not is_mkt_target(j.entry.target)]
            wins = apply_lane_c(st, js, round_id=r.round_id, settle_block=r.block,
                                era_index=r.era_index, consume_pending=r.consume_pending)
            wins += fold_mkt_plan(st, r.mkt)   # DEC-CA-0064 (labels computed at judge time)
            st.last_rows = list(r.rows)
        st.mark_applied(r.round_id, r.block)
        return wins

    def _lane_c_walk_back(self, head: str | None,
                          scratch: LaneCState | None = None) -> list[_LaneCRound]:
        """Judge the chained settlements not yet folded, from ``head`` back to
        the last folded one (or lane C's activation), OLDEST FIRST — each
        against ``scratch`` (a copy of the live state) folded in chain order,
        so a round sees the pending
        set its predecessor left. Payout TTL cannot bound this walk: expired
        holders and pending report identities still affect future verdicts."""
        st = self._lane_c()
        if scratch is None:
            scratch = LaneCState.from_json(st.to_json())
        found: list[TrainingManifest] = []
        rid = str(head or "")
        seen: set[str] = set()
        while rid:
            if rid == st.last_round:
                break
            if rid in seen:
                raise LaneCTransient(f"cyclic settlement history at {rid}")
            seen.add(rid)
            m = self._lane_c_load_settlement(rid)
            with self._settlement_cfg(self._epoch_start_block(m)):
                if not self.cfg.lane_c.active(self._epoch_start_block(m)):
                    break
                signed = not self.verify_signatures or verify_signature(
                    m, self.cfg.manifest.trainer_hotkey)
            if signed and rid not in st.applied:
                found.append(m)
            rid = str(m.prev_round_id or "")
        if found:
            log.warning("lane C: catching up %d unfolded settlement(s) (%s … %s)", len(found),
                        found[-1].round_id, found[0].round_id)
        rounds = []
        for m in reversed(found):
            with self._settlement_cfg(self._epoch_start_block(m)):
                r = self._lane_c_judge_round(m, scratch)
            self._lane_c_fold(scratch, r)
            rounds.append(r)
        return rounds

    def _lane_c_prepare(self, manifest: TrainingManifest) -> list[_LaneCRound] | None:
        """Everything lane C needs for this settlement, READ and JUDGED before
        lane A mutates anything (catch-up rounds first, against a scratch
        copy folded in order). ``None`` when lane C is not active at its
        block. Raises :class:`LaneCTransient`."""
        block = self._epoch_start_block(manifest)
        if not self.cfg.lane_c.active(block):
            return None
        st = self._lane_c()
        scratch = LaneCState.from_json(st.to_json())
        rounds: list[_LaneCRound] = []
        prev = str(manifest.prev_round_id or "")
        if prev and st.last_round != prev:
            rounds += self._lane_c_walk_back(prev, scratch)
        rounds.append(self._lane_c_judge_round(manifest, scratch))
        return rounds

    def _lane_c_commit(self, rounds: list[_LaneCRound] | None) -> list[dict] | None:
        """Fold prepared rounds (idempotent per round id), persist atomically,
        and return the CURRENT round's receipt rows."""
        if rounds is None:
            return None
        st = self._lane_c()
        for r in rounds:
            for w in self._lane_c_fold(st, r):
                log.info("lane C: %s takes target %s (round %s, gain LCB %.4f)",
                         w.hotkey[:8], w.target, r.round_id, w.gain_lcb)
        self._persist_lane_c()
        return list(rounds[-1].rows)

    def _lane_c_catch_up_startup(self) -> None:  # pragma: no cover — live-loop glue
        """Rebuild a missing / torn lane-C state at startup (best-effort)."""
        head = self.state.last_handled_round_id
        st = self._lane_c()
        if not head or st.last_round == head:
            return
        try:
            m = self._lane_c_load_settlement(head)
            with self._settlement_cfg(self._epoch_start_block(m)):
                if not self.cfg.lane_c.active(self._epoch_start_block(m)):
                    return
            rounds = self._lane_c_walk_back(head)
        except LaneCTransient as e:
            log.warning("lane C: startup catch-up deferred (%s)", e)
            return
        if rounds:
            self._lane_c_commit(rounds)

    # ── pre-round snapshot (re-judgement replaces the fold) ──

    def _lane_c_snapshot(self) -> str | None:
        if not self.cfg.lane_c.from_block:
            return None
        return self._lane_c().to_json()

    def _lane_c_restore(self, text: str | None) -> None:
        if text is None or not self.cfg.lane_c.from_block:
            return
        self._lane_c_state = LaneCState.from_json(text)


__all__ = [
    "LaneCJudgement", "LaneCMixin", "LaneCTransient", "OUTCOME_BELOW_HOLDER",
    "OUTCOME_CONFIRMED", "OUTCOME_DEFERRED", "OUTCOME_DROPPED", "OUTCOME_PENDING",
    "apply_lane_c", "best_per_target", "judgements_record", "resolve_outcomes",
]
