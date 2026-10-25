"""Cascade reign log: bench-report reads and the per-round Cascade step."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..shared.era import settlement_contract_block, validator_bench_configured
from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
)

if TYPE_CHECKING:
    from .loop import RoundOutcome

log = logging.getLogger("cascade.validator")


# How many subsequent rounds the Cascade step keeps re-probing for a round's
# missing bench report before writing that round off. The trainer publishes the
# report ~30-60 min after the manifest (the bench runs post-publish), so it is
# usually there by the time the NEXT round's cascade step runs — one retry
# catches nearly everything. The budget is deliberately generous anyway (each
# retry is one GET against an R2-backed store): a validator that misses a
# report its peers recorded ends the reign with a DIFFERENT log, can select a
# different promotion winner, and then rejects rounds at the warm-start pin
# gate — so convergence is worth days of cheap re-probing, and 5 retries at
# 12h rounds spans most of a 7-round reign.
BENCH_REPORT_RETRY_ROUNDS = 5


class CascadeBenchMixin:
    """Bench numbers into the reign log, with re-probing of late reports."""

    # ── Cascade: king-reign promotion ────────────────────────────────────────

    @staticmethod
    def _bench_scores_dict(entry: TrainedEntry) -> dict | None:
        """The six Cascade numbers off a manifest entry's trainer-signed
        ``bench_scores`` (GIFT-Eval / BOOM / TIME CRPS+MASE), or ``None`` when the
        entry carries none. The FALLBACK source, for rounds published before the
        trainer moved the bench post-publish (those manifests carry the numbers
        in-entry; new ones never do)."""
        bs = entry.bench_scores
        if bs is None:
            return None
        return {
            "gifteval_crps": bs.gifteval_crps, "gifteval_mase": bs.gifteval_mase,
            "boom_crps": bs.boom_crps, "boom_mase": bs.boom_mase,
            "time_crps": bs.time_crps, "time_mase": bs.time_mase,
        }

    def _bench_store(self) -> object:
        if self.bench_report_store is None:
            from ..shared.hippius import open_manifest_store

            self.bench_report_store = open_manifest_store(self.cfg.storage)
        return self.bench_report_store

    def _fetch_bench_report(self, round_id: str, *, require: str | None = None,
                            raise_unavailable: bool = False) -> object | None:
        """The round's trainer-signed bench report
        (:mod:`cascade.shared.bench_report`), or ``None`` — absent (the bench
        runs post-publish, so "not there yet" is a normal state), malformed,
        or failing signature verification.

        ``require`` names a trained pointer the caller needs: a CACHED report
        lacking it is discarded and re-fetched once, so a copy cached before
        the trainer (re)published the entry cannot pin a stale miss for the
        life of the process. ``raise_unavailable`` turns a store READ failure
        (network/auth/5xx — the object's existence is unknown) into a raised
        :class:`StorageError` instead of ``None``: a member check must treat
        "could not read" as a transient the loop retries next poll, never as
        "has no bench numbers". Absence (``ObjectNotFound``) is still ``None``
        — that IS a verdict. Read failures are logged either way; the old
        silent ``None`` left every such miss undiagnosable."""
        from ..shared.bench_report import (
            bench_report_key,
            load_bench_report,
            verify_bench_report_signature,
        )
        from ..shared.hippius import ObjectNotFound, StorageError

        rid = str(round_id)
        cached = self._bench_report_cache.get(rid)
        if cached is not None:
            if require is None or cached.entry_for_pointer(require) is not None:
                return cached
            log.info("cascade: cached bench report for round=%s lacks %s; re-fetching",
                     rid, require[:64])
            self._bench_report_cache.pop(rid, None)
        try:
            text = self._bench_store().get_text(bench_report_key(rid))
        except ObjectNotFound:
            log.info("cascade: bench report for round=%s not published (yet)", rid)
            return None
        except Exception as e:  # noqa: BLE001 — store down / auth / network
            log.warning("cascade: bench report for round=%s could not be READ (%s: %s)",
                        rid, type(e).__name__, str(e)[-300:])
            if raise_unavailable:
                raise StorageError(f"bench_report_unavailable: round={rid}: {e}") from e
            return None
        try:
            report = load_bench_report(text)
        except (ValueError, KeyError, TypeError) as e:
            log.warning("cascade: bench report for round=%s unparseable (%s); ignoring",
                        round_id, e)
            return None
        if self.verify_signatures and not verify_bench_report_signature(
            report, self.cfg.manifest.trainer_hotkey
        ):
            log.warning("cascade: bench report for round=%s failed signature "
                        "verification; ignoring", round_id)
            return None
        self._bench_report_cache[rid] = report
        while len(self._bench_report_cache) > 16:  # small FIFO, insertion-ordered
            self._bench_report_cache.pop(next(iter(self._bench_report_cache)))
        return report

    def _report_bench_scores(self, round_id: str, trained_pointer: str) -> dict | None:
        """The six Cascade numbers for one checkpoint out of the round's signed
        bench report — the AUTHORITATIVE source (every validator reads the same
        signed set). Joined on the exact ``trained_pointer``, never role/UID.
        ``require`` makes a re-probe (:meth:`_drain_pending_bench`) a real GET
        when the cached copy lacks the pointer — the report is republished as
        legs finish, and a partial copy cached earlier must not answer every
        retry from memory."""
        report = self._fetch_bench_report(round_id, require=trained_pointer)
        be = report.entry_for_pointer(trained_pointer) if report is not None else None
        if be is None and validator_bench_configured(self.cfg):
            # DEC-CA-0063: a validator-sourced settlement's numbers (winners,
            # the king, the init) live in the main validator's signed report
            vrep = self._fetch_validator_bench_report(round_id)
            be = vrep.entry_for_pointer(trained_pointer) if vrep is not None else None
        if be is None:
            return None
        s = be.scores
        return {
            "gifteval_crps": s.gifteval_crps, "gifteval_mase": s.gifteval_mase,
            "boom_crps": s.boom_crps, "boom_mase": s.boom_mase,
            "time_crps": s.time_crps, "time_mase": s.time_mase,
        }

    def _record_duel_checkpoints(self, manifest: TrainingManifest, now: float) -> None:
        """Add the round's benched duel checkpoints — the king's AND every
        challenger's at the primary throne size — to the reign log. Both roles
        are promotion candidates (DEC-CA-0013): challenger checkpoints carry
        genuinely different generator data, the deepest diversity the promoted
        set can draw on, and the trainer benches both sides already. The log is
        the envelope's provenance/quality evidence, so a later verification is
        a lookup, not a re-eval.

        Source order per checkpoint: the round's signed bench report
        (authoritative; published after the manifest), then the manifest
        entry's in-entry ``bench_scores`` (king-only, rounds from before the
        bench moved post-publish). Neither yet ⇒ the checkpoint is queued and
        re-probed at the next rounds' cascade steps (the report typically lands
        ~an hour after the manifest); a checkpoint whose report never appears
        simply contributes no bench numbers — the trainer selects (and this
        validator verifies) over the reign checkpoints that have them."""
        if self.cascade is None or self.state.king_hotkey is None:
            return
        if int(getattr(self.cfg.training, "v2_from_block", 0) or 0):
            primary = self.cfg.throne_contracts_at(settlement_contract_block(
                self.cfg.round, self._epoch_start_block(manifest)))[0].arch_preset
        else:
            primary = self.cfg.throne_contracts()[0].arch_preset
        for entry in manifest.entries:
            if (entry.size or primary) != primary:
                continue
            metrics = (self._report_bench_scores(manifest.round_id, entry.trained_pointer)
                       or self._bench_scores_dict(entry))
            if metrics is None:
                self._queue_pending_bench(manifest.round_id, entry)
                continue
            self.cascade.record_checkpoint(
                entry.trained_pointer, now=now, size=entry.size, role=entry.role, **metrics)

    def _queue_pending_bench(self, round_id: str, entry: TrainedEntry) -> None:
        if any(p["round_id"] == str(round_id)
               and p["trained_pointer"] == entry.trained_pointer
               for p in self._pending_bench):
            return
        self._pending_bench.append({
            "round_id": str(round_id),
            "trained_pointer": entry.trained_pointer,
            "size": entry.size,
            "role": entry.role,
            "king_hotkey": self.state.king_hotkey,
            # The reign identity at queue time: an accepted promotion re-crowns
            # the SAME king with a new start block, so the king alone cannot
            # tell "this reign" from "the reign the promotion just closed".
            "reign_start": (None if self.cascade is None
                            else self.cascade.state.reign_start_block),
            "tries": BENCH_REPORT_RETRY_ROUNDS,
        })
        log.info("cascade: no bench numbers yet for round=%s %s checkpoint %s; "
                 "will re-probe the bench report for up to %d round(s)",
                 round_id, entry.role, entry.trained_pointer, BENCH_REPORT_RETRY_ROUNDS)

    def _drain_pending_bench(self, now: float) -> None:
        """Re-probe queued checkpoints' bench reports and log any that landed.

        A queued checkpoint whose reign ended is dropped outright — whether by
        a dethrone (new king) or by an accepted PROMOTION (same king, fresh
        reign): the reign log was cleared at the re-crown, and a closed reign's
        checkpoint leaking into the fresh log would corrupt the quality floor
        the next promotion is verified against."""
        if self.cascade is None or not self._pending_bench:
            return
        keep: list = []
        for p in self._pending_bench:
            if (p["king_hotkey"] != self.cascade.state.king_hotkey
                    or p.get("reign_start") != self.cascade.state.reign_start_block):
                log.info("cascade: dropping pending bench for round=%s (reign ended)",
                         p["round_id"])
                continue
            metrics = self._report_bench_scores(p["round_id"], p["trained_pointer"])
            if metrics is not None:
                self.cascade.record_checkpoint(
                    p["trained_pointer"], now=now, size=p["size"],
                    role=p.get("role", ""), **metrics)
                continue
            p["tries"] -= 1
            if p["tries"] > 0:
                keep.append(p)
            else:
                log.info("cascade: giving up on bench report for round=%s; the "
                         "checkpoint contributes no bench numbers", p["round_id"])
        self._pending_bench = keep

    def _cascade_round(
        self, manifest: TrainingManifest, outcome: RoundOutcome | None
    ) -> None:  # pragma: no cover — live-loop glue; the controller is unit-tested
        """One Cascade step, run at the end of a round (after weights/receipts).
        Resets the reign clock on a dethrone, re-anchors a legacy clock, and
        records the round's benched duel checkpoints into the reign log. The
        validator never fires a promotion (DEC-CA-0013): the trainer declares,
        and the warm-start gate verifies+adopts at manifest time — this step
        only maintains the clock and the evidence the gate verifies against.
        Fully guarded: Cascade never disturbs KOTH."""
        if self.cascade is None:
            return
        import time

        now = time.time()
        try:
            # The reign clock runs on the round's epoch block — identical for every
            # validator (from the signed manifest), so ripeness verdicts agree.
            block = self._epoch_start_block(manifest)
            # Reuse KOTH's dethrone signal to reset the clock (never reimplement it);
            # on adoption (no cascade king but a live champion — a validator
            # joining or restarting mid-reign), crown WITHOUT attestation: an
            # adopted clock must not judge reign timing.
            dethroned = (outcome is not None and outcome.transition.dethroned
                         and outcome.transition.new_king_hotkey)
            if dethroned:
                # DEC-CA-0055: a staged v2 king-only promotion survives the
                # dethrone (accepted is accepted; the notice era rides it out)
                self.cascade.note_dethrone(outcome.transition.new_king_hotkey, block=block,
                                           keep_pending=self._v2_pending_kept())
            elif self.cascade.state.king_hotkey is None and self.state.king_hotkey is not None:
                self.cascade.note_dethrone(self.state.king_hotkey, block=block,
                                           observed=False)
            self.cascade.observe_round(block=block)
            # Earlier rounds first: reports published after those rounds'
            # cascade steps (the bench runs post-publish) land in the reign log
            # now, before this round's checkpoints are considered.
            self._drain_pending_bench(now)
            # A DETHRONE round's checkpoints belong to the reign that just
            # ended and are recorded NOWHERE: the trainer's candidate pool
            # wipes them at its own re-crown, and logging them here (into the
            # freshly cleared log) would give this validator evidence the
            # trainer never selects against — a stricter quality floor that
            # rejects honest promotions (the one-sidedness invariant of
            # _verify_members).
            if not dethroned:
                self._record_duel_checkpoints(manifest, now)
        except Exception as e:  # noqa: BLE001 — Cascade must never disturb a round
            log.warning("cascade step failed for round=%s: %s", manifest.round_id, e)
