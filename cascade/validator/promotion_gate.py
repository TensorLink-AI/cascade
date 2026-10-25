"""Warm-start envelope (DEC-CA-0013): verify declared inits against promotion records."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ..shared.era import validator_bench_configured
from ..shared.manifest import TrainingManifest

log = logging.getLogger("cascade.validator")


class PromotionGateMixin:
    """Warm-start gate, promotion-record and member verification."""

    def _check_warm_start(self, manifest: TrainingManifest) -> str | None:
        """Warm-start envelope gate (Cascade, DEC-CA-0013, propose-and-verify).

        The trainer SELECTS the promoted warm-start set and declares each
        round's init in the signed manifest (``warm_start_ckpt``); this gate
        VERIFIES the declaration instead of re-deriving the selection:

        * a declared init in the accepted generation's member set passes (any
          member — rotation/allocation across members is trainer policy);
        * an unseen init must be justified by a trainer-signed
          :class:`~cascade.shared.promotion.PromotionRecord` for a NEWER
          generation that survives the envelope (:meth:`_verify_promotion`):
          provenance + quality floor against the reign log's signed bench
          numbers, reign-clock ripeness, and the ``cascade_top_k`` set cap;
        * random init ("") passes only in the random-init era (generation 0) —
          once a promotion is live a round must never silently train unpinned.

        Only enforced when Cascade is wired (off ⇒ pure KOTH, field ignored)."""
        if self.cascade is None:
            return None
        legacy_reason = self._adopt_legacy_warm_start()
        if legacy_reason is not None:
            return legacy_reason
        if self._era_active(manifest):
            # DEC-CA-0043: the era gate names the init from the ledger
            # (generation + era rotation) and rejects anything else.
            return None
        declared = manifest.warm_start_ckpt
        state = self.cascade.state
        if not declared:
            if state.generation == 0:
                return None
            return ("warm_start_missing: manifest trained from <random init> but "
                    f"promotion generation {state.generation} is live")
        if declared in state.members:
            return None
        return self._verify_promotion(manifest, declared)

    def _adopt_legacy_warm_start(self) -> str | None:
        """Migration shim (pre-DEC-CA-0013 states): grandfather the single
        pointer this validator's own Cascade installed under the old
        select-and-install mechanism — or, on the owner box, the member set the
        co-hosted trainer's engine wrote — as the accepted generation, once,
        from the random-init era. Returns a rejection reason only for a
        present-but-unreadable file (fail LOUD, never open).

        Known bootstrap gap, accepted: the grandfathered generation has no
        published PromotionRecord (the old winner has no ``source_round``
        provenance to verify one against), so a validator with NO local state
        at all — a fresh joiner during the grandfather window — rejects rounds
        until the first post-upgrade promotion publishes a real record (≤ one
        reign)."""
        if self.cascade is None or self.cascade.state.generation != 0:
            return None
        p = Path(self.cfg.validator.warm_start_init_path)
        if not p.is_file():
            return None
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 — unreadable pin must fail LOUD, not open
            return f"warm_start_state_unreadable: {p}: {e}"
        members = [str(m.get("checkpoint_id") or "") for m in (obj.get("members") or ())]
        members = [m for m in members if m]
        if members:
            self.cascade.adopt_member_set(
                generation=int(obj.get("generation", 1) or 1), members=tuple(members))
        else:
            self.cascade.adopt_legacy_pointer(str(obj.get("checkpoint_id") or ""))
        return None

    def _verify_promotion(self, manifest: TrainingManifest, declared: str) -> str | None:
        """Verify a declared-but-unseen init against the latest published
        promotion record. Accepts (installs the new generation via
        :meth:`CascadeController.note_promotion`) and returns ``None``, or
        returns the rejection reason.

        The envelope, in order: the record exists for a generation NEWER than
        the accepted one and carries the declared init; it is trainer-signed;
        its member set respects ``cascade_top_k``; the reign clock was ripe
        (checked only when this validator's clock can attest timing at all —
        an unanchored or catching-up clock only measures its own uptime, so it
        skips the check rather than rejecting a whole reign's rounds); and
        every member's provenance + quality holds against signed bench numbers
        (:meth:`_verify_members`)."""
        from ..shared.promotion import verify_promotion_record_signature

        assert self.cascade is not None
        state = self.cascade.state
        record = self._fetch_latest_promotion_record()
        if record is None:
            return (f"warm_start_mismatch: manifest trained from {declared!r} but no "
                    f"promotion record justifies it (accepted generation "
                    f"{state.generation})")
        if record.generation <= state.generation:
            return (f"warm_start_mismatch: manifest trained from {declared!r} but the "
                    f"latest promotion record is generation {record.generation} ≤ "
                    f"accepted {state.generation} (stale or replayed)")
        if self.verify_signatures and not verify_promotion_record_signature(
            record, self.cfg.manifest.trainer_hotkey
        ):
            return (f"warm_start_promotion_unsigned: record generation "
                    f"{record.generation} failed signature verification")
        member_ids = record.member_ids()
        if declared not in member_ids:
            return (f"warm_start_mismatch: manifest trained from {declared!r}, not a "
                    f"member of promotion generation {record.generation}")
        k_max = max(1, int(self.cfg.scoring.cascade_top_k))
        if not member_ids or len(member_ids) > k_max:
            return (f"warm_start_promotion_invalid: generation {record.generation} "
                    f"declares {len(member_ids)} member(s); cap is {k_max}")
        block = self._epoch_start_block(manifest)
        # The timing-and-scope checks apply when this validator's clock can
        # attest at all (anchored, generation ≥ 1) and the record is the +1
        # transition it watched — a bootstrap/catch-up validator's clock only
        # measures its own uptime, so it verifies what its evidence supports.
        attesting = (record.generation == state.generation + 1
                     and self.cascade.can_verify_ripeness())
        if attesting and not self.cascade.is_ripe(block=block):
            return (f"warm_start_promotion_early: generation {record.generation} "
                    f"declared before the reign clock ripened")
        member_reason = self._verify_members(
            record, enforce_reign_scope=attesting, carried_exempt=self._era_active(manifest))
        if member_reason is not None:
            return member_reason
        self.cascade.note_promotion(
            generation=record.generation, members=member_ids, block=block,
            effective_era=int(getattr(record, "effective_era", 0) or 0))
        return None

    def _v2_lineage_size(self, record: object) -> str | None:
        """DEC-CA-0050: the arch preset every member of a v2-lineage promotion
        record must carry (the v2 contract's), or None for a record that does
        not take effect as a v2 generation (``effective_era <``
        :func:`v2_first_warm_era`, the :func:`v2_generation_usable` boundary —
        ``v2_first_era + max(2, random_init_eras)``)."""
        from ..shared.era import v2_first_warm_era

        warm = v2_first_warm_era(self.cfg)
        v2c = getattr(self.cfg.training, "v2_contract", None)
        if warm is None or v2c is None:
            return None
        if int(getattr(record, "effective_era", 0) or 0) < warm:
            return None
        return str(v2c.arch_preset)

    def _verify_members(self, record: object, *, enforce_reign_scope: bool,
                        carried_exempt: bool = False,
                        lineage_size: str | None = None) -> str | None:
        """Provenance + quality floor for every member of a promotion record.

        Each member must have trainer-signed bench numbers: from this
        validator's reign log when it recorded that round, else fetched on
        demand from the member's ``source_round`` bench report — so
        verification does not depend on this validator's log being complete. A
        member with no verifiable score fails CLOSED (an init nobody can score
        must not become the field's floor) — but only when the report is
        readable and lacks it, or is definitively absent. A report that cannot
        be READ raises :class:`StorageError` (a transient: the round is
        retried next poll, no receipt) — a swallowed read miss would publish
        a permanent rejection for a report that exists.

        ``enforce_reign_scope`` additionally pins the fetched fallback to the
        CURRENT reign: the report's ``created_block`` must not predate this
        validator's reign anchor, so a checkpoint from a dead reign (stale
        trainer state, fabricated ``source_round``) can't pass provenance just
        because some old signed report once scored it. Only enforced when this
        validator's clock watched the whole reign (the same condition as
        ripeness) — a bootstrap clock would mis-scope honest early-reign
        members. The reign-log path is reign-scoped by construction (the log
        clears on every re-crown). **Carried-over members are exempt**
        (DEC-CA-0044 rolling top-k): a member of the generation this
        validator already accepted (live or staged) was verified when it
        entered and keeps its original ``source_round``, which by
        construction predates the reign it now carries into — scope applies
        only to NEW entrants. Quality is still re-checked for every member.
        ``carried_exempt`` is CONSENSUS-GATED on ``era_king_from_block`` by
        the caller: before the rollover every member is scoped (as the
        previous release does), so an early-upgraded validator never accepts
        a record its peers reject.

        The floor is the best score across the reign log and the members
        themselves; every member must sit within ``cascade_quality_epsilon``
        of it. One-sided by construction: a validator's log is a subset of the
        trainer's candidates (same signed reports), so a missing report can
        only loosen this validator's floor — an honest trainer passes every
        validator; only a cheating trainer splits the fleet, and splitting
        loudly is the correct outcome there."""
        from .cascade import best_score, cascade_score, log_record_for

        assert self.cascade is not None
        state = self.cascade.state
        reign_start = state.reign_start_block
        if lineage_size is not None:
            # DEC-CA-0050: a v2-lineage generation initialises the MoE — a
            # member of any other arch (a toto2 checkpoint) can never do that
            for m in getattr(record, "members", ()):
                if str(getattr(m, "size", "")) != lineage_size:
                    return (f"warm_start_member_wrong_lineage: {m.checkpoint_id} is "
                            f"{getattr(m, 'size', '')!r}, the v2 lineage is "
                            f"{lineage_size!r}")
        carried = (set(state.members) | set(state.pending_members)) if carried_exempt else set()
        scores: dict[str, float] = {}
        for m in getattr(record, "members", ()):
            rec = log_record_for(state, m.checkpoint_id)
            if rec is not None:
                scores[m.checkpoint_id] = rec.score
                continue
            report = self._fetch_bench_report(
                m.source_round, require=m.checkpoint_id, raise_unavailable=True)
            be = None if report is None else report.entry_for_pointer(m.checkpoint_id)
            if be is None and validator_bench_configured(self.cfg):
                # DEC-CA-0063: the king check's sweep is in the main
                # validator's signed report of the check settlement (the
                # member's source_round); published before its receipt, so
                # before the trainer could fire on it. Absent / unreadable
                # HOLDS (raises StorageError), exactly like the trainer path
                # above (raise_unavailable) — never "unverifiable" on one
                # validator while its peers, reading fine, accept.
                vrep = self._fetch_validator_bench_report(m.source_round,
                                                          raise_unavailable=True)
                be = None if vrep is None else vrep.entry_for_pointer(m.checkpoint_id)
                if be is not None:
                    report = vrep
            if be is None:
                return (f"warm_start_member_unverifiable: {m.checkpoint_id} has no "
                        f"signed bench numbers (source_round={m.source_round!r})")
            if (enforce_reign_scope and reign_start is not None
                    and m.checkpoint_id not in carried
                    and int(getattr(report, "created_block", 0)) < int(reign_start)):
                return (f"warm_start_member_out_of_reign: {m.checkpoint_id} was "
                        f"benched in round {m.source_round!r} (block "
                        f"{getattr(report, 'created_block', '?')}), before the "
                        f"current reign anchored at block {reign_start}")
            s = be.scores
            scores[m.checkpoint_id] = cascade_score(
                s.gifteval_crps, s.gifteval_mase, s.boom_crps,
                s.boom_mase, s.time_crps, s.time_mase)
        floor_candidates = list(scores.values())
        if lineage_size is not None:
            # the v2 floor is the v2 lineage's own best — a toto2 checkpoint
            # logged earlier in the same reign is no reference for the MoE
            log_best = min((r.score for r in self.cascade.state.checkpoints
                            if r.size == lineage_size), default=None)
        else:
            log_best = best_score(self.cascade.state)
        if log_best is not None:
            floor_candidates.append(log_best)
        floor = min(floor_candidates)
        epsilon = float(self.cfg.scoring.cascade_quality_epsilon)
        for cid, score in scores.items():
            if score > floor * (1.0 + epsilon):
                return (f"warm_start_member_below_floor: {cid} score {score:.5f} "
                        f"exceeds quality floor {floor:.5f} × (1 + {epsilon})")
        return None

    def _fetch_latest_promotion_record(self) -> object | None:
        """The latest published promotion record, located via the unsigned
        ``promotions/index.json`` (trust comes from the record's signature,
        checked by the caller — the index is a locator only).

        ``None`` when the index is absent (no promotion ever published) or
        the located record is malformed — both verdicts. A store READ failure
        (auth/network/5xx), or an index naming a record that cannot be read,
        raises :class:`StorageError`: the caller's gate must retry next poll,
        never reject, or a transient read failure latches a permanent
        ``warm_start_mismatch`` / ``era_init_mismatch`` receipt."""
        latest = self._latest_promotion_generation()
        if latest <= 0:
            return None
        return self._fetch_promotion_record(latest)

    def _latest_promotion_generation(self) -> int:
        """The generation ``promotions/index.json`` names (0 when absent or
        malformed). A store READ failure raises :class:`StorageError`."""
        from ..shared.hippius import ObjectNotFound, StorageError
        from ..shared.promotion import load_promotion_index, promotion_index_key

        try:
            index_text = self._bench_store().get_text(promotion_index_key())
        except ObjectNotFound:
            return 0  # no promotions published yet
        except Exception as e:  # noqa: BLE001 — store down / auth / network
            log.warning("cascade: promotion index could not be READ (%s: %s)",
                        type(e).__name__, str(e)[-300:])
            raise StorageError(f"promotion_index_unavailable: {e}") from e
        return load_promotion_index(index_text)

    def _fetch_promotion_record(self, generation: int, *,
                                missing_ok: bool = False) -> object | None:
        """The record published at ``promotions/gen-<generation>.json``.

        ``None`` when it is malformed (a verdict) — or, with ``missing_ok``,
        definitively absent (an older generation the catch-up walk needs but
        the store never held). Any other READ failure — and a miss on the
        generation the index names (the record is published BEFORE the index
        names it, so that miss is a read failure or a store inconsistency) —
        raises :class:`StorageError`."""
        from ..shared.hippius import ObjectNotFound, StorageError
        from ..shared.promotion import load_promotion_record, promotion_record_key

        try:
            text = self._bench_store().get_text(promotion_record_key(generation))
        except ObjectNotFound as e:
            if missing_ok:
                log.warning("cascade: promotion record gen=%d is absent", generation)
                return None
            log.warning("cascade: promotion record gen=%d could not be READ (%s)",
                        generation, str(e)[-300:])
            raise StorageError(
                f"promotion_record_unavailable: gen={generation}: {e}") from e
        except Exception as e:  # noqa: BLE001 — store down / auth / network
            log.warning("cascade: promotion record gen=%d could not be READ (%s: %s)",
                        generation, type(e).__name__, str(e)[-300:])
            raise StorageError(
                f"promotion_record_unavailable: gen={generation}: {e}") from e
        try:
            return load_promotion_record(text)
        except Exception as e:  # noqa: BLE001 — malformed content is a verdict
            log.warning("cascade: promotion record gen=%d malformed (%s); ignored",
                        generation, e)
            return None
