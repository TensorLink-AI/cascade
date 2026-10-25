"""Reign tracking, promotion and the bench-anneal leg (TrainerRunner mixin)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..shared.era import (
    era_king_active,
    min_effective_era,
    v2_check_due,
    v2_first_warm_era,
    v2_reign_floor_block,
)
from ..shared.hippius import manifest_round_key
from ..shared.manifest import (
    BenchScores,
    TrainedEntry,
)
from .wandb_sink import log_wandb_event, open_wandb_run

log = logging.getLogger("cascade.trainer")


# Corpus-seed salt for the bench-anneal leg (DEC-CA-0030): the anneal resumes
# a canonical checkpoint on FRESH data (base_seed ^ salt) so the decay pass
# never re-fits the leg's exact training draw, and the salted seed keys the
# leg's _train_work dir + checkpoint repo away from every canonical name.
# The value matches the offline calibration runs, so their published
# -anneal-u<uid> artifacts stay reproducible under the production leg.
BENCH_ANNEAL_SALT = 0xA11EA1A11EA1


def _ladder_share(emission: dict | None) -> dict:
    """The lane-A ladder fields of a receipt's ``emission`` record: the king's
    share (basis points → fraction, else the policy's ``king_share``), the
    ladder event, the court and the key slash. ``{}`` when absent."""
    if not isinstance(emission, dict):
        return {}
    pol = emission.get("policy") or {}
    lad = emission.get("ladder") or {}
    bp = lad.get("bp")
    share = (float(bp) / 10000.0 if bp is not None
             else (None if pol.get("king_share") is None else float(pol["king_share"])))
    return {"ladder_king_share": share, "ladder_event": lad.get("event"),
            "ladder_court_hotkey": lad.get("court_hotkey"),
            "ladder_slashed": lad.get("slashed")}


def king_check_event(check: dict, boundary: int, emission: dict | None = None) -> dict:
    """The W&B record of one DEC-CA-0055 king check (pure)."""
    return {"result": str(check.get("result", "") or ""),
            "check_round": str(check.get("round_id", "") or ""),
            "block": int(boundary or 0),
            "lcb": check.get("lcb"), "margin": check.get("margin"),
            "king_pointer": str(check.get("king_pointer", "") or ""),
            "king_hotkey": str(check.get("king_hotkey", "") or ""),
            "reasons": "; ".join(str(r) for r in (check.get("reasons") or ())),
            **_ladder_share(emission)}


def promotion_event(record: object) -> dict:
    """The W&B record of one fired promotion (pure)."""
    members = tuple(getattr(record, "members", ()) or ())
    return {"generation": int(getattr(record, "generation", 0) or 0),
            "fired_round": str(getattr(record, "fired_round", "") or ""),
            "fired_block": int(getattr(record, "fired_block", 0) or 0),
            "effective_era": int(getattr(record, "effective_era", 0) or 0),
            "king_hotkey": str(getattr(record, "king_hotkey", "") or ""),
            "n_members": len(members),
            "members": ",".join(str(getattr(m, "checkpoint_id", "")) for m in members),
            "signed": getattr(record, "signature", None) is not None}


class _PromotionMixin:
    """Reign tracking + promotion (DEC-CA-0013) and bench dispatch."""

    # ── reign tracking + promotion (DEC-CA-0013) ─────────────────────────────

    def _receipt_king(self) -> str | None:
        """The current king per the validators' signed receipt trail, or the
        last king a receipt named when none is readable right now (``None``
        only before any receipt was ever read). This is the prompt dethrone
        signal the reign clock needs: validators reset their clocks at the
        dethrone verdict, while the on-chain incentive (the caller's fallback
        king source) lags it 1-2 epochs. The last-known value is STICKY across
        transient fetch failures — during the lag window a blip that fell back
        to the stale incentive king would flap the engine's king view and
        reset the reign clock twice. Best-effort; never raises.

        Never SILENT: a receipt that is unreadable or fails its signature is
        logged at ERROR (once per distinct problem) — the sticky king is then
        possibly stale, and a silent stale king hides a dethrone. A receipt
        carrying fields this code does not know is logged at WARNING: restart
        on current code."""
        try:
            from ..shared.hippius import RECEIPT_LATEST_KEY, receipt_latest_key
            from ..shared.receipt import (
                load_receipt,
                receipt_has_unknown_fields,
                verify_receipt_signature,
            )

            anchor = self.cfg.manifest.validator_hotkey
            if not anchor:
                self._note_receipt_problem("validator receipt trust anchor is unset; ignoring unsigned receipts")
                return getattr(self, "_last_receipt_king", None)
            store = self.manifest_store()
            problems: list[str] = []
            for key in (receipt_latest_key(anchor), RECEIPT_LATEST_KEY):
                try:
                    receipt = load_receipt(store.get_text(key))
                except Exception as e:  # noqa: BLE001 — absent/unreadable ⇒ next candidate
                    problems.append(f"{key}: unreadable ({type(e).__name__}: {str(e)[:120]})")
                    continue
                if anchor and not verify_receipt_signature(receipt, anchor):
                    problems.append(f"{key}: round {receipt.round_id} fails the anchor "
                                    "signature check")
                    continue
                v = receipt.verdict
                if receipt.status == "scored" and v is not None and v.king_hotkey:
                    if receipt_has_unknown_fields(receipt):
                        self._note_receipt_problem(
                            f"receipt round {receipt.round_id} carries fields this trainer's "
                            "code does not know — it was written by newer code; restart the "
                            "trainer on current main", level=logging.WARNING)
                    else:
                        self._note_receipt_problem("")
                    self._last_receipt_king = str(v.king_hotkey)
                    return self._last_receipt_king
                problems.append(f"{key}: round {receipt.round_id} names no king "
                                f"(status {receipt.status})")
            self._note_receipt_problem(
                "no verifiable receipt names a king — keeping the last known king "
                f"{(self._last_receipt_king or '<none>')[:12]}, which may be STALE: "
                + "; ".join(problems))
        except Exception as e:  # noqa: BLE001 — the sticky/incentive fallbacks cover a miss
            self._note_receipt_problem(f"receipt king read failed: {type(e).__name__}: {e}")
        return self._last_receipt_king

    def _note_receipt_problem(self, msg: str, *, level: int = logging.ERROR) -> None:
        """Log a receipt-read problem once per distinct message; "" clears it."""
        if msg == self._receipt_problem:
            return
        self._receipt_problem = msg
        if msg:
            log.log(level, "receipt king: %s", msg)

    def _seed_promotion_reign(self) -> None:
        """Deploy-time backfill (DEC-CA-0013): count the rounds the current
        king has ALREADY reigned before this engine existed.

        Runs once, at the first boundary where the engine has never seen a
        king — after that the engine's own persisted clock owns the count.
        Validators anchored their clocks at the dethrone verdict, possibly
        rounds ago; an engine anchoring "now" would fire the promotion
        ``reign_threshold`` rounds later than the fleet's ripeness check
        requires it to be able to fire. The reign is reconstructed from the
        same signed artifacts validators used: the receipt index's unbroken
        tail of scored rounds held by the current king (``reign_tail``) seeds
        the clock, and each reign round's published trainer-signed bench
        report replays through :meth:`TrainerPromotion.record_bench` to
        rebuild the candidate log — exactly the state the engine would hold
        had it been running since the dethrone. Rounds whose bench report is
        unpublished contribute no candidates (the same gap the validator's
        reign log has). Best-effort: any miss falls back to ``note_round``'s
        anchor-at-this-boundary behaviour."""
        if self.promotion is None or getattr(self.promotion, "king_hotkey", None) is not None:
            return
        if getattr(self.promotion, "reign_start_block", None) is not None:
            return
        try:
            from ..shared.bench_report import (
                bench_report_key,
                load_bench_report,
                verify_bench_report_signature,
            )
            from ..shared.hippius import RECEIPT_INDEX_KEY
            from ..shared.manifest import load_manifest, verify_signature
            from .promotion import reign_tail

            store = self.manifest_store()
            idx = json.loads(store.get_text(RECEIPT_INDEX_KEY))
            tail = reign_tail(idx.get("rounds") or [],
                              self.cfg.manifest.validator_hotkey)
            if tail is None:
                return
            king, start_block, round_ids = tail
            if not self.promotion.seed_reign(king, start_block):
                return
            added = 0
            for rid in round_ids:  # oldest → newest
                try:
                    manifest = load_manifest(
                        store.get_text(manifest_round_key(rid)))
                    report = load_bench_report(
                        store.get_text(bench_report_key(rid)))
                except Exception:  # noqa: BLE001 — no report yet ⇒ no candidates
                    continue
                trainer_hotkey = self.cfg.manifest.trainer_hotkey
                if trainer_hotkey and not verify_bench_report_signature(
                        report, trainer_hotkey):
                    log.warning("promotion backfill: bench report for round=%s "
                                "fails signature vs pinned trainer hotkey; skipped", rid)
                    continue
                if trainer_hotkey and not verify_signature(manifest, trainer_hotkey):
                    log.warning("promotion backfill: manifest for round=%s fails "
                                "signature vs pinned trainer hotkey; skipped", rid)
                    continue
                added += self.promotion.record_bench(manifest, report)
            log.info("promotion: deploy backfill — king %s reign counted from "
                     "block %d (%d round(s), %d candidate(s) recovered)",
                     king[:12], start_block, len(round_ids), added)
        except Exception as e:  # noqa: BLE001 — never sink a round on backfill
            log.warning("promotion: deploy backfill failed (engine anchors at "
                        "this boundary instead): %s", e)

    def _replay_reign_bench_reports(self) -> None:
        """Re-ingest the CURRENT reign's published bench reports at the
        boundary. The candidate pool otherwise only sees reports the
        in-process bench thread publishes — an out-of-band report (a mop-up
        after a failed leg, a report that landed while the trainer was down)
        never yields candidates, and a promotion would then fire off an
        incomplete reign.

        Cheap and idempotent by construction: ``record_bench`` dedupes on
        ``trained_pointer`` and gates on the live generation, so replaying
        every reign round each boundary is a handful of small S3 reads a day
        that add nothing when nothing is missing. Runs BEFORE
        ``maybe_promote`` so a fire-time selection sees the full reign.
        Same signature discipline as the deploy backfill: both the report
        and the manifest must verify against the pinned trainer hotkey."""
        if self.promotion is None or getattr(self.promotion, "king_hotkey", None) is None:
            return
        try:
            from ..shared.bench_report import (
                bench_report_key,
                load_bench_report,
                verify_bench_report_signature,
            )
            from ..shared.hippius import RECEIPT_INDEX_KEY
            from ..shared.manifest import load_manifest, verify_signature
            from .promotion import reign_tail

            store = self.manifest_store()
            idx = json.loads(store.get_text(RECEIPT_INDEX_KEY))
            tail = reign_tail(idx.get("rounds") or [],
                              self.cfg.manifest.validator_hotkey)
            if tail is None:
                return
            _king, _start_block, round_ids = tail
            trainer_hotkey = self.cfg.manifest.trainer_hotkey
            added = 0
            for rid in round_ids:
                try:
                    manifest = load_manifest(
                        store.get_text(manifest_round_key(rid)))
                    report = load_bench_report(
                        store.get_text(bench_report_key(rid)))
                except Exception:  # noqa: BLE001 — no report yet ⇒ nothing to replay
                    continue
                if trainer_hotkey and not verify_bench_report_signature(
                        report, trainer_hotkey):
                    log.warning("promotion replay: bench report for round=%s "
                                "fails signature vs pinned trainer hotkey; skipped", rid)
                    continue
                if trainer_hotkey and not verify_signature(manifest, trainer_hotkey):
                    log.warning("promotion replay: manifest for round=%s fails "
                                "signature vs pinned trainer hotkey; skipped", rid)
                    continue
                added += self.promotion.record_bench(manifest, report)
            if added:
                log.info("promotion: boundary replay recovered %d candidate(s) "
                         "from published bench report(s)", added)
        except Exception as e:  # noqa: BLE001 — replay must never sink a round
            log.warning("promotion: bench-report replay failed (ignored): %s", e)

    def _promotion_boundary_step(self, king_hotkey: str | None, epoch_start: int,
                                 round_id: str, *, effective_era: int = 0) -> None:
        """Cascade promotion (DEC-CA-0013) at a boundary: track the reign and
        fire a promotion BEFORE anything trains, so the signed record is
        published (and fetchable by validators) before any manifest pins a
        new-generation member. The reign clock keys off the signed receipt
        trail's verdict king when available — the on-chain incentive lags a
        dethrone by 1-2 epochs — falling back to the incentive king.
        Guarded: promotion must never sink a round. ``effective_era``
        (DEC-CA-0043) stamps a fired record with the era it takes effect."""
        if self.promotion is None:
            return
        try:
            # First boundary of a never-anchored engine: count the rounds the
            # king already reigned (signed receipt tail + published bench
            # reports) before the clock ticks — see _seed_promotion_reign.
            self._seed_promotion_reign()
            # DEC-CA-0050: from the v2 block the MoE lineage starts afresh —
            # no toto2 member or candidate seeds it.
            v2_block = int(getattr(getattr(self.cfg, "v2", None), "from_block", 0) or 0)
            if v2_block and int(epoch_start) >= v2_block:
                self.promotion.start_lineage(v2_block)
            self.promotion.note_round(self._receipt_king() or king_hotkey,
                                      epoch_block=epoch_start)
            # Out-of-band bench reports (mop-ups, trainer-downtime publishes)
            # enter the pool here — before maybe_promote, so a promotion that
            # fires NOW selects from the full reign.
            self._replay_reign_bench_reports()
            # [v2] random_init_eras: no promotion takes effect inside the
            # random-init window — candidates keep accumulating, and the first
            # fire whose effective era clears it selects from the whole window.
            warm = v2_first_warm_era(self.cfg)
            v2_lineage = bool(v2_block and int(epoch_start) >= v2_block)
            hold = bool(v2_lineage and warm is not None and int(effective_era) < warm)
            if v2_lineage and bool(getattr(self.cfg.v2, "king_ladder", False)):
                # DEC-CA-0055 (``[v2] king_ladder``): the v2 lineage promotes ONLY the king, ONLY on
                # the validators' passing king check (never the trainer's own
                # cascade_reign_* ripeness, never challenger candidates).
                self._v2_king_check_step(epoch_start, round_id)
            elif hold:
                log.info("promotion: era %d is inside the v2 random-init window "
                         "(warm starts from era %d); not firing", int(effective_era), warm)
            else:
                # Rolling top-k (DEC-CA-0044) is consensus-gated on the rollover
                # (era_king_from_block): validators reject carried members before it.
                self.promotion.maybe_promote(
                    epoch_block=epoch_start, round_id=round_id, effective_era=effective_era,
                    rolling_topk=era_king_active(self.cfg.scoring, int(epoch_start)),
                    clock_floor_block=v2_reign_floor_block(self.cfg))
        except Exception as e:  # noqa: BLE001
            log.warning("promotion step failed for round=%s: %s", round_id, e)
        # Publish-with-retry: the record survives (persisted) as pending until
        # the publish lands, so a store outage never orphans a generation the
        # pointer file already rotates over.
        self._flush_pending_promotion(round_id)

    def _receipt_king_check(self) -> tuple[dict, int, str] | None:
        """The latest signed receipt's DEC-CA-0055 ``king_check`` record, its
        settlement boundary and the receipt's round id — read and verified the
        way :meth:`_receipt_king` reads the trail (anchor signature first).
        ``None`` when no verifiable receipt carries one. Never raises."""
        try:
            from ..shared.hippius import (
                RECEIPT_LATEST_KEY,
                read_receipt_index,
                receipt_latest_key,
                receipt_round_key,
            )
            from ..shared.receipt import load_receipt, verify_receipt_signature

            anchor = self.cfg.manifest.validator_hotkey
            if not anchor:
                log.error("promotion: validator receipt trust anchor is unset; refusing king checks")
                return None
            store = self.manifest_store()
            keys = [receipt_latest_key(anchor), RECEIPT_LATEST_KEY]
            try:
                rows = read_receipt_index(store).get("rounds", [])
                rows = sorted(rows, key=lambda r: int(r.get("epoch_start_block", r.get("block", 0)) or 0), reverse=True)
                keys.extend(receipt_round_key(str(r["round_id"]), anchor) for r in rows[:64]
                            if r.get("validator_hotkey", anchor) == anchor)
            except Exception:  # missing index still permits a valid latest receipt
                pass
            for key in dict.fromkeys(keys):
                try:
                    text = store.get_text(key)
                    receipt = load_receipt(text)
                except Exception:  # noqa: BLE001 — absent/unreadable ⇒ next candidate
                    continue
                if anchor and not verify_receipt_signature(receipt, anchor):
                    continue
                check = getattr(receipt, "king_check", None)
                if check is None:
                    # a build whose RoundReceipt predates the field: the signed
                    # body still carries it — read it from the raw JSON
                    try:
                        check = json.loads(text).get("king_check")
                    except Exception:  # noqa: BLE001
                        check = None
                if isinstance(check, dict) and check:
                    emission = getattr(receipt, "emission", None)
                    if emission is None:
                        try:
                            emission = json.loads(text).get("emission")
                        except Exception:  # noqa: BLE001
                            emission = None
                    # observability only (the W&B king-check event's ladder share)
                    self.__dict__["_king_check_emission"] = (
                        dict(emission) if isinstance(emission, dict) else None)
                    return (dict(check), int(receipt.epoch_start_block or 0),
                            str(receipt.round_id))
        except Exception as e:  # noqa: BLE001
            log.warning("promotion: receipt king_check read failed: %s", e)
        return None

    def _v2_king_check_step(self, epoch_start: int, round_id: str) -> None:
        """Fire the v2 king-only promotion on a PASSING king check stamped on
        the validators' signed receipt (DEC-CA-0055 item 4). ``fail`` /
        ``deferred`` fire nothing; a check fires once (the engine dedupes on
        its round id). The record's ``fired_block`` is the HEAD block at the
        boundary the trainer actually fires (never backdated to the check's
        boundary); its ``effective_era`` is ``min_effective_era`` of the
        CHECK's settlement boundary — the block the validators judged
        ripeness at (:func:`v2_check_due`) — kept only while the fire block
        still leaves that era its full notice (validators require
        ``effective_era == min_effective_era(fired_block)``), else the check
        is not fired; and never inside the random-init window
        (``v2_first_warm_era``). The record must be PUBLISHED at this
        boundary so this settlement's manifest carries its notice: a publish
        that fails is retracted (:meth:`TrainerPromotion.retract_fire`) and
        the check re-read next boundary, rather than announced a settlement
        late — a late notice is a dead generation for every node. A dethrone
        during the notice era does not cancel a fired promotion."""
        got = self._receipt_king_check()
        if got is None:
            return
        check, receipt_boundary, receipt_round = got
        result = str(check.get("result", "") or "")
        check_round = str(check.get("round_id", "") or "")
        pointer = str(check.get("king_pointer", "") or "")
        boundary = int(check.get("block", 0) or check.get("boundary", 0) or 0)
        if not boundary:
            boundary = receipt_boundary if receipt_round == check_round else int(epoch_start)
        self._wandb_king_check_event(check, boundary)
        if result != "pass":
            # the validators' reign clock restarts at a judged check (pass or
            # fail, never deferred): mirror it, once per check
            if (result == "fail" and check_round
                    and self.promotion.note_king_check(check_round, boundary)):
                log.info("promotion: king check %s FAILED (%s) — no promotion; reign "
                         "clock restarts at %d, the king is re-checked at the next due "
                         "point", check_round, "; ".join(
                             str(r) for r in (check.get("reasons") or ())) or "no reason",
                         boundary)
            return
        if not check_round or not pointer:
            log.warning("promotion: passing king check without round_id/king_pointer — "
                        "ignored: %s", check)
            return
        if check_round == getattr(self.promotion, "last_king_check", ""):
            return
        reign = getattr(self.promotion, "reign_start_block", None)
        if not v2_check_due(self.cfg, boundary=boundary, reign_start_block=reign):
            # The validators decided; the trainer's own view of the reign may
            # lag a dethrone (incentive vs receipt). Never second-guess a pass.
            log.warning("promotion: king check %s passed at boundary %d although this "
                        "trainer's reign clock (start %s) does not see it due — firing "
                        "on the validators' verdict", check_round, boundary, reign)
        # Preserve the check's proposed era only while the actual fire block
        # still gives it the full notice. Never backdate a delayed fire.
        warm = v2_first_warm_era(self.cfg)
        effective = min_effective_era(self.cfg.round, int(boundary))
        now_block = max(int(epoch_start), int(getattr(self, "_last_block", 0) or 0))
        if boundary > now_block:
            log.error("promotion: check %s is from future block %d; not fired", check_round, boundary)
            return
        if effective < min_effective_era(self.cfg.round, now_block):
            log.error("promotion: check %s arrived too late for era %d at block %d; not fired",
                      check_round, effective, now_block)
            return
        if warm is not None and effective < int(warm):
            log.error("promotion: king check %s passed at block %d but its warm start (era "
                      "%d) falls inside the random-init window (warm from era %d) — not "
                      "fired", check_round, boundary, effective, warm)
            return
        size, score = self._king_check_member_meta(check_round, pointer, int(boundary))
        record = self.promotion.promote_king(
            check_round=check_round, king_pointer=pointer, size=size, score=score,
            epoch_block=now_block, round_id=str(round_id), effective_era=effective)
        if record is None:
            return
        # Publish NOW: the notice rides this boundary's manifest only if the
        # record is readable before it builds (_promotion_notices).
        self._flush_pending_promotion(round_id)
        if self.promotion.unpublished_record() is not None:
            self._retract_unpublished_fire(record, check_round)

    def _retract_unpublished_fire(self, record: object, check_round: str) -> None:
        """The record fired at this boundary did not publish. Absent from the
        store ⇒ un-fire it (:meth:`TrainerPromotion.retract_fire`): the
        check is re-read at the next boundary and fires there with a fresh
        fire block if its era still has full notice, or is refused as too
        late — never published a settlement late. Present with the SAME body
        ⇒ the put landed and only its acknowledgement was lost: mark it
        published. Present with another body, or unreadable ⇒ keep it pending
        (the retry and the notice rule decide; logged)."""
        from ..shared.hippius import ObjectNotFound
        from ..shared.promotion import load_promotion_record, promotion_record_key

        key = promotion_record_key(int(record.generation))
        try:
            published = load_promotion_record(self.manifest_store().get_text(key))
        except ObjectNotFound:
            published = None
        except Exception as e:  # noqa: BLE001 — unknown: the flush retries next boundary
            log.error("promotion: record gen=%d unpublished and %s unreadable (%s); kept "
                      "pending — its notice may now be late", record.generation, key, e)
            return
        if published is not None:
            if published.canonical_body() == record.canonical_body():
                self.promotion.mark_record_published()
                return
            log.error("promotion: %s already holds a different record; generation %d stays "
                      "pending for operator reconciliation", key, record.generation)
            return
        if self.promotion.retract_fire(record):
            log.error("promotion: king check %s fired generation %d but its record could not "
                      "be published at this boundary — retracted; re-read at the next "
                      "boundary (fires there only if era %d still has full notice)",
                      check_round, record.generation, int(record.effective_era))

    def _promotion_notices(self, block: int) -> dict:
        """Bind a published future generation to the settlement chain."""
        import hashlib

        from ..shared.era import era_for_block
        from ..shared.promotion import load_promotion_record, promotion_record_key

        if self.promotion is None or not self.cfg.v2.active(block):
            return {}
        generation = int(self.promotion.generation)
        if not generation:
            return {}
        if self.promotion.unpublished_record() is not None:
            return {}
        try:
            record = load_promotion_record(self.manifest_store().get_text(promotion_record_key(generation)))
        except Exception as e:  # noqa: BLE001 — omission cannot authorize a new init
            # the copy this process published (same signed bytes) stands in
            # for a read-back miss, so the fire boundary's manifest still
            # carries the notice; nothing else does
            record = self.__dict__.get("_last_published_promotion")
            if record is None or int(record.generation) != generation:
                log.warning("promotion %d notice unavailable; omitted: %s", generation, e)
                return {}
            log.warning("promotion %d record unreadable (%s); notice from the copy this "
                        "trainer published", generation, e)
        if int(record.effective_era) <= era_for_block(self.cfg.round, block).index:
            return {}
        return {str(generation): hashlib.sha256(record.canonical_body()).hexdigest()}

    def _wandb_king_check_event(self, check: dict, boundary: int) -> None:
        """One W&B event per king check the trainer reads (DEC-CA-0055):
        round, result, LCB, margin, king pointer and the lane-A ladder share
        stamped on the same receipt. Observability only — deduped in memory,
        never raises (``log_wandb_event``)."""
        try:
            check_round = str(check.get("round_id", "") or "")
            seen = self.__dict__.setdefault("_wandb_king_checks", set())
            if not check_round or check_round in seen:
                return
            seen.add(check_round)
            rec = king_check_event(check, boundary, self.__dict__.get("_king_check_emission"))
            log_wandb_event(getattr(self.cfg, "wandb", None), kind="king_check",
                            round_id=check_round, record=rec)
        except Exception as e:  # noqa: BLE001 — telemetry only
            log.debug("king-check wandb event skipped: %s", e)

    def _king_check_member_meta(self, check_round: str, pointer: str,
                                boundary: int = 0) -> tuple[str, float]:
        """``(size, score)`` for the promoted king checkpoint, from the check
        round's trainer-signed bench report (the king's entry), else the
        contract's arch preset and 0.0 (the score is observability only —
        validators re-read the signed report)."""
        import math

        size, score = "", 0.0
        try:
            from ..shared.bench_report import (
                bench_report_key,
                load_bench_report,
                validator_bench_report_key,
            )
            from ..shared.era import validator_bench_configured
            from ..validator.cascade import cascade_score

            e = None
            keys = [bench_report_key(check_round)]
            if validator_bench_configured(self.cfg):
                # DEC-CA-0063: the king check's sweep is the main validator's
                keys.insert(0, validator_bench_report_key(check_round))
            for key in keys:
                try:
                    rep = load_bench_report(self.manifest_store().get_text(key))
                except Exception:  # noqa: BLE001 — absent: try the next source
                    continue
                e = rep.entry_for_pointer(pointer)
                if e is not None:
                    break
            if e is not None:
                s = e.scores
                size = str(e.size or "")
                score = float(cascade_score(s.gifteval_crps, s.gifteval_mase, s.boom_crps,
                                            s.boom_mase, s.time_crps, s.time_mase))
        except Exception as e:  # noqa: BLE001 — metadata only
            log.info("promotion: bench report for king check %s unreadable (%s); "
                     "member size/score from the contract", check_round, e)
        if not size:
            try:
                size = str(self.cfg.throne_contracts_at(int(boundary))[0].arch_preset)
            except Exception:  # noqa: BLE001
                size = ""
        if not math.isfinite(score):
            score = 0.0
        return size, score

    def _flush_pending_promotion(self, round_id: str) -> None:
        """Publish a fired-but-unpublished promotion record, if one is pending.
        Guarded and idempotent — called at the round boundary AND right before
        the manifest publishes, so a transient store outage at fire time heals
        within the same round instead of costing the fleet the whole round."""
        if self.promotion is None:
            return
        try:
            pending = self.promotion.unpublished_record()
            if pending is not None:
                self._publish_promotion_record(pending)
                self.promotion.mark_record_published()
        except Exception as e:  # noqa: BLE001 — stays pending, retried next flush
            from ..shared.promotion import PromotionRecordConflict

            if isinstance(e, PromotionRecordConflict):
                log.error("promotion record publish REFUSED for round=%s (stays pending; "
                          "operator reconciliation): %s", round_id, e)
            else:
                log.warning("promotion record publish failed for round=%s "
                            "(stays pending): %s", round_id, e)

    def _publish_promotion_record(self, record: object) -> None:
        """Sign and publish a fired promotion's record to the manifest bucket
        (``promotions/gen-<n>.json`` + locator index) — the declaration
        validators verify the new generation against. Unsigned publishes are
        possible (no wallet: offline runs) but rejected by verifying
        validators, exactly like an unsigned bench report."""
        from ..shared.promotion import (
            dump_promotion_record,
            publish_promotion_record,
            sign_promotion_record,
        )

        if self.wallet is not None:
            record = sign_promotion_record(record, self.wallet)
        else:
            log.warning("publishing an UNSIGNED promotion record (no wallet); "
                        "validators will reject the new generation")
        key = publish_promotion_record(
            self.manifest_store(), dump_promotion_record(record), record.generation)
        self.__dict__["_last_published_promotion"] = record
        log.info("published promotion record generation=%d members=%d signed=%s → s3://%s/%s",
                 record.generation, len(record.members),
                 record.signature is not None, self.cfg.storage.manifest_bucket, key)
        # W&B promotion event (observability only; never raises)
        try:
            log_wandb_event(getattr(self.cfg, "wandb", None), kind="promotion",
                            round_id=str(record.fired_round), record=promotion_event(record),
                            key=f"gen{int(record.generation)}")
        except Exception as e:  # noqa: BLE001 — telemetry only
            log.debug("promotion wandb event skipped: %s", e)

    def _remote_bench_scores(
        self, host: object, entry: TrainedEntry, round_id: str, primary: str,
        *, role_dir: str | None = None, require_fev: bool = False,
    ) -> BenchScores | None:
        """One remote sweep → six numbers, or ``None`` on any miss.

        ``require_fev`` (DEC-CA-0061, the era judges under the four-suite
        rule): an OPERATOR sweep without FEV numbers is a miss, held and
        re-benched like a missing GIFT-Eval/BOOM/TIME. A payer pod's sweep
        never needs FEV — it only feeds the forged-sweep check; the operator
        re-bench supplies the published numbers.

        With ``[telemetry] bench_anneal_fraction`` armed (DEC-CA-0030) the
        sweep scores an ANNEALED copy of the checkpoint — finished-form
        numbers for the signed BenchScores — falling back to the raw
        checkpoint on any anneal-leg failure."""
        from ..eval.benchmarks import extract_bench_scores
        from .bench_hook import run_post_round_benchmark

        bench_round, bench_role = round_id, (role_dir or entry.role)
        frac = float(getattr(self.cfg.telemetry, "bench_anneal_fraction", 0.0) or 0.0)
        if frac > 0.0:
            annealed = self._dispatch_bench_anneal(host, entry, round_id, primary, frac)
            if annealed is not None:
                bench_round, bench_role = annealed
        report = run_post_round_benchmark(
            host, bench_round, entry.size or primary, self.cascade_bench_plan,
            work_root=self.work_root, role=bench_role,
        )
        operator = not getattr(host, "isolated", False)
        scores = (extract_bench_scores(report, require_fev=require_fev and operator)
                  if report is not None else None)
        if scores is not None and operator:
            self._stash_bench_rows(entry.trained_pointer, report)   # operator numbers only
        return BenchScores(**scores) if scores is not None else None

    def _stash_bench_rows(self, trained_pointer: str, report: dict) -> None:
        """Publish an OPERATOR sweep's per-item rows for the three-suite bench
        gate (DEC-CA-0054) and remember ``(key, sha256)`` so the settlement's
        signed bench report can pin them. A payer pod's rows are never
        stashed — only numbers the operator ran. Inert while
        the bench gate is configured at no block (``bench_gate_configured``); best-effort (a miss leaves the
        gate uncomputable for that checkpoint, never crashes a bench)."""
        from ..shared.era import bench_gate_configured

        if not bench_gate_configured(self.cfg):
            return
        import hashlib

        from ..eval.benchmarks import item_rows_from_report
        from ..shared.bench_report import bench_rows_key, dump_bench_rows

        try:
            rows = item_rows_from_report(report)
            if rows is None:
                return
            text = dump_bench_rows(rows)
            key = bench_rows_key(trained_pointer)
            self.manifest_store().put_text(key, text, content_type="application/json",
                                           acl="public-read")
            reg = self._bench_rows_registry()
            reg[trained_pointer] = {"key": key,
                                    "sha256": hashlib.sha256(text.encode()).hexdigest()}
            path = Path(self.work_root) / "bench_rows.json"
            path.write_text(json.dumps(reg, sort_keys=True), encoding="utf-8")
        except Exception as e:  # noqa: BLE001 — telemetry path; the gate degrades to uncomputable
            log.warning("bench rows for %s not published: %s", trained_pointer[:64], e)

    def _bench_rows_registry(self) -> dict:
        reg = self.__dict__.get("_bench_rows_reg")
        if reg is None:
            path = Path(self.work_root) / "bench_rows.json"
            try:
                reg = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                reg = {}
            self.__dict__["_bench_rows_reg"] = reg
        return reg

    def _dispatch_bench_anneal(
        self, host: object, entry: TrainedEntry, round_id: str, primary: str,
        frac: float,
    ) -> tuple[str, str] | None:
        """Run the bench-anneal leg (DEC-CA-0030) for one duel checkpoint on
        the pod that trained it, returning the ``(round_dir, role_dir)`` pair
        naming the annealed checkpoint's ``_train_work`` location for the
        bench sweep — or ``None`` on any failure (the caller benches the raw
        checkpoint instead; log-only telemetry must never lose the sweep).

        The leg is a stock worker run: resume ``entry.trained_pointer``
        (weights + optimizer state) on a FRESH salted corpus for
        ``frac × target_train_hours`` under the pure-decay recipe
        (``--anneal``). The salt keys the work dir + checkpoint repo away
        from every canonical name, so nothing this leg writes can collide
        with a consensus artifact."""
        from .remote import pod_lane_count

        if not self.trainer_spec:
            log.warning("round=%s: bench-anneal skipped (no trainer_spec — "
                        "benching the raw checkpoint)", round_id)
            return None
        size = entry.size or primary
        contract = self.cfg.training.primary_size
        if size != contract.arch_preset:
            spec = next((sp for sp in self.cfg.training.extra_sizes
                         if sp.arch_preset == size), None)
            if spec is None:
                log.warning("round=%s: bench-anneal skipped for unknown size %r",
                            round_id, size)
                return None
            contract = self.cfg.training.for_size(spec)
        salted = (int(round_id) ^ BENCH_ANNEAL_SALT) & ((1 << 63) - 1)
        suffix = f"-anneal-u{entry.miner_uid}"
        try:
            disp = self._dispatcher()
            # warm_start_ref makes the leg warm_started=True in the trainer, so
            # under an armed [training] warm_lr_scale (DEC-CA-0035) its cosine
            # decays from base_lr × scale. Exactly right for warm-started duel
            # checkpoints (matches their stable-phase LR); for a generation-
            # start checkpoint (stable phase ran full base_lr) the leg starts
            # one notch low — accepted: telemetry-only, errs conservative, and
            # all legs stay mutually comparable at the same start LR.
            disp.dispatch(
                host, gen_ref=entry.gen_ref, uid=entry.miner_uid,
                hotkey=entry.miner_hotkey, role=entry.role,
                base_seed=salted, block=entry.train_block,
                arch_preset=size, train_hours=contract.target_train_hours * frac,
                repo_suffix=suffix, warm_start_ref=entry.trained_pointer,
                lane_count=pod_lane_count(host, [host]), anneal=True,
            )
        except Exception as e:  # noqa: BLE001 — telemetry: fall back, never raise
            log.warning("round=%s: bench-anneal leg failed for %s %s (benching "
                        "the raw checkpoint instead): %s",
                        round_id, entry.role, entry.miner_hotkey, e)
            return None
        log.info("round=%s: bench-anneal leg done for %s %s (frac=%.2f) — "
                 "benching the annealed copy", round_id, entry.role,
                 entry.miner_hotkey, frac)
        # The worker wrote _train_work/<salted>/<size>/<role><suffix>/checkpoint
        # (dir = role + repo suffix, the layout bench_hook.role_paths reads).
        return str(salted), f"{entry.role}{suffix}"

    def _log_bench_pair_wandb(self, report: object) -> None:
        """Mirror the round's king/challenger bench pair to wandb when
        ``[wandb] enabled`` — observability only, same swallow-everything
        contract as the per-step training mirror."""
        try:
            if not getattr(self.cfg.wandb, "enabled", False):
                return
            entries = list(getattr(report, "entries", ()) or ())
            if not entries:
                return
            anchor = next((e for e in entries if e.role == "king"), entries[0])
            sink = open_wandb_run(
                self.cfg.wandb, round_id=str(report.round_id),
                role=f"bench-{anchor.size}", hotkey=anchor.miner_hotkey,
                uid=anchor.miner_uid, size=anchor.size,
            )
            if sink is None:
                return
            for e in entries:
                s = e.scores
                sink.emit({
                    "event": "cascade_bench", "role": e.role,
                    "miner_hotkey": e.miner_hotkey, "miner_uid": e.miner_uid,
                    "gifteval_crps": s.gifteval_crps, "gifteval_mase": s.gifteval_mase,
                    "boom_crps": s.boom_crps, "boom_mase": s.boom_mase,
                    "time_crps": s.time_crps, "time_mase": s.time_mase,
                })
            sink.finish()
        except Exception as e:  # noqa: BLE001 — wandb must never disturb the bench
            log.debug("wandb bench-pair log failed (continuing): %s", e)
