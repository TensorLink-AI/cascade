"""Era-king envelope (DEC-CA-0043) checks and era ledger bookkeeping."""

from __future__ import annotations

import logging
from dataclasses import replace

from ..shared.era import (
    EraSpec,
    era_king_active,
    era_length_blocks,
    member_index_for_era,
    min_effective_era,
    settlement_era,
    v2_era_active,
    v2_generation_usable,
)
from ..shared.lane_b import merge_identities
from ..shared.manifest import TrainingManifest
from .king_check import NOTICE_LATE
from .lane_b import LaneBTransient
from .manifest_chain import ref_as_of

log = logging.getLogger("cascade.validator")


def _merge_error_type() -> type:
    """``moe_model.MergeError`` (lazy: torch-free validators never import it
    unless a merged init is actually declared)."""
    try:
        from ..trainer.moe_model import MergeError
    except Exception:  # noqa: BLE001 — no torch: nothing can raise it
        return _NoMergeError
    return MergeError


class _NoMergeError(Exception):
    pass


class EraGateMixin:
    """Era stamp, init, window, chain, king pointer and ref checks."""

    # ── Era envelope (DEC-CA-0043) ───────────────────────────────────────────

    def _era_active(self, manifest: TrainingManifest) -> bool:
        return era_king_active(self.cfg.scoring, self._epoch_start_block(manifest))

    def _check_era(self, manifest: TrainingManifest) -> str | None:
        """Era-king envelope, fail closed, only from ``era_king_from_block``.

        In order: the era stamp exists and parses; it is the era every node
        derives from the settlement's epoch boundary (index / start on the
        era grid / seeds from the previous era's start); the init it names is
        ``members_gen(g)[era % k]`` for the latest verified promotion whose
        ``effective_era <= era`` (a pending record is installed at its first
        effective settlement, never earlier) and ``warm_start_ckpt`` is that
        member; every entry's ``train_block`` lies inside the era's training
        window ``[seed_block, era_end)``; the manifest chains to this
        validator's last handled settlement; the king entry, when it is the
        champion, carries the pointer this validator judged (or adopts the
        era's first king leg); every entry's ref is the hotkey's revealed
        commitment as of its ``train_block``.

        Pre-gate manifests are untouched (bit-identical to the ungated
        validator) — a stray ``era`` stamp before the block is ignored.
        """
        if not self._era_active(manifest):
            return None
        if manifest.era is None:
            return "era_missing: settlement past era_king_from_block carries no era stamp"
        try:
            era = EraSpec.from_json(manifest.era)
        except ValueError as e:
            return f"era_malformed: {e}"
        block = self._epoch_start_block(manifest)
        expected = settlement_era(self.cfg.round, block)
        # DEC-CA-0055: a v2 era inside a seed epoch trains from (and stamps)
        # the epoch's first era's seed block — so its entry window below spans
        # the whole epoch and the epoch's one king leg stays in it.
        expected = replace(expected, seed_block=self._era_seed_block(expected))
        if (era.index, era.start_block, era.seed_block) != (
                expected.index, expected.start_block, expected.seed_block):
            return (f"era_mismatch: manifest era {era.index}@{era.start_block} "
                    f"(seed {era.seed_block}) but the boundary {block} is in era "
                    f"{expected.index}@{expected.start_block} (seed {expected.seed_block})")
        init_reason = self._check_era_init(manifest, era)
        if init_reason is not None:
            return init_reason
        if v2_era_active(self.cfg, era.start_block) and era.generation and not manifest.prev_manifest_sha256:
            return "manifest_chain_broken: a promoted v2 era requires a predecessor hash"
        window_end = era.start_block + era_length_blocks(self.cfg.round, era.start_block)
        for e in manifest.entries:
            if v2_era_active(self.cfg, era.start_block) and (
                    e.init_pointer != manifest.warm_start_ckpt
                    or e.training_contract_digest != manifest.contract_digest):
                return "era_entry_init_mismatch: per-leg init or contract differs from the era"
            if not (era.seed_block <= int(e.train_block) < window_end):
                return (f"era_entry_out_of_window: {e.role} uid{e.miner_uid} trained at "
                        f"block {e.train_block}, outside era {era.index}'s window "
                        f"[{era.seed_block}, {window_end})")
        last = self.state.last_handled_round_id
        if (last is not None and manifest.prev_round_id != last
                and str(manifest.round_id) != last):
            return (f"manifest_chain_broken: prev_round_id {manifest.prev_round_id!r} "
                    f"but this validator last handled {last!r} — walk the chain")
        last_sha = getattr(self.state, "last_handled_manifest_sha", None)
        if (manifest.prev_manifest_sha256 and last is not None and last_sha
                and manifest.prev_round_id == last
                and manifest.prev_manifest_sha256 != last_sha):
            # Hash link: the previous settlement this validator handled is
            # not the one the trainer chained to (a re-published or forked
            # round) — the round id alone cannot tell them apart.
            return (f"manifest_chain_broken: prev_manifest_sha256 "
                    f"{manifest.prev_manifest_sha256[:16]}… but this validator handled "
                    f"round {last!r} at sha {last_sha[:16]}…")
        king_reason = self._check_era_king(manifest, era)
        if king_reason is not None:
            return king_reason
        return self._check_era_refs(manifest)

    def _check_era_init(self, manifest: TrainingManifest, era: EraSpec) -> str | None:
        """The init an era manifest declares must be the one the promotion
        ledger names for that era (zero trainer discretion)."""
        if self.cascade is None:
            # Pure KOTH: no promotions, so an era is random-init by definition.
            if era.generation or era.member_index or manifest.warm_start_ckpt:
                return ("era_init_mismatch: cascade is off, era must train from "
                        "random init")
            return None
        self._promotion_check_manifest = manifest
        try:
            self._refresh_pending_promotion(era)
        finally:
            self._promotion_check_manifest = None
        # Gating is PURE: the ledger in force for this era is derived here and
        # installed only once the settlement is handled (_install_era_promotion)
        # — a transient after the gate leaves the state untouched for the retry.
        generation, members = self._era_ledger(era)
        if generation == 0 or not members:
            expected_member, expected_index = "", 0
        else:
            expected_index = member_index_for_era(era.index, len(members))
            expected_member = members[expected_index]
        if era.generation != generation or era.member_index != expected_index:
            return (f"era_init_mismatch: era {era.index} declares generation "
                    f"{era.generation} member {era.member_index}, the ledger names "
                    f"generation {generation} member {expected_index}")
        epoch = self._epoch_start(era)
        if epoch is not None and epoch != int(era.index):
            # DEC-CA-0055: one init per seed epoch — a merge (or any other
            # init change) lands only at a warm start, never mid-epoch
            if manifest.warm_start_merges and (
                    not self.state.epoch_init
                    or int(self.state.epoch_init.get("epoch", -1)) != epoch
                    or self.state.epoch_init.get("init") != manifest.warm_start_ckpt):
                return (f"era_init_mismatch: era {era.index} is inside the seed epoch from "
                        f"era {epoch}; lane-B merges change the init only at a warm start")
            ei = self.state.epoch_init
            if (ei and int(ei.get("epoch", -1)) == epoch
                    and ei.get("init") != manifest.warm_start_ckpt):
                return (f"era_init_mismatch: era {era.index} must train from its seed "
                        f"epoch's init {ei.get('init') or '<random init>'!r}, manifest "
                        f"trained from {manifest.warm_start_ckpt or '<random init>'!r}")
        if manifest.warm_start_merges:
            # DEC-CA-0050: the member + recorded lane-B wins, rebuilt bit-exactly
            if not self.cfg.lane_b.active(era.start_block):
                return "era_init_mismatch: merged init declared while lane B is inactive"
            if manifest.warm_start_base != expected_member:
                return (f"era_init_mismatch: merged init built on "
                        f"{manifest.warm_start_base!r}, the ledger names {expected_member!r}")
            return self._verify_merged_init(
                manifest, strict=self.cfg.lane_b.merge_guard_at_era(era.start_block))
        if manifest.warm_start_ckpt != expected_member:
            return (f"era_init_mismatch: era {era.index} init is "
                    f"{expected_member or '<random init>'!r}, manifest trained from "
                    f"{manifest.warm_start_ckpt or '<random init>'!r}")
        return None

    def _verify_merged_init(self, manifest: TrainingManifest, *,
                            strict: bool = False) -> str | None:
        """Every merge must be a domain win THIS validator recorded, and the
        declared init must equal member + Σ(rebased − king) on the merged rows,
        tensor for tensor. Cached per init pointer (an era's settlements share
        one init).

        ``strict`` (``[lane_b] merge_guard_from_block`` at the era start):
        merges overlapping on any (layer, row) are refused — two deltas summed
        onto one row were never judged together — and so is any non-finite
        delta.

        A merge not yet recorded whose record names its settlement
        (``round_id``) is judged ON DEMAND from that settlement's bound expert
        manifest first — a validator that missed it converges instead of
        rejecting the era. A READ failure (an expert manifest, a checkpoint)
        raises :class:`StorageError`: the caller's gate retries next poll
        within the read grace, never latching ``era_init_unverifiable``."""
        import hashlib
        import json

        from ..shared.hippius import StorageError
        from .merge_acceptance import acceptance_block

        acceptance_required = (manifest.era is not None and self.cfg.lane_b.acceptance_at_era(
            acceptance_block(self.cfg, manifest)))
        if acceptance_required:
            from .merge_acceptance import verify_acceptance

            try:
                verify_acceptance(self, manifest)
            except ValueError as e:
                return f"era_init_mismatch: {e}"
            except Exception as e:  # evidence/scoring failures retry the gate
                raise StorageError(f"merge acceptance unavailable: {e}") from e
        cache_key = (manifest.warm_start_ckpt, manifest.warm_start_base,
                     hashlib.sha256(json.dumps(manifest.warm_start_merges,
                                               sort_keys=True).encode()).hexdigest(), bool(strict))
        if cache_key in self._merged_init_ok:
            return None
        st = self._lane_b()

        def _recorded(m) -> bool:
            return (acceptance_required and bool(st.qualified_for(m))) or st.has_merge(
                m.get("rebased_pointer", ""), m.get("king_pointer", ""), m.get("selection", {}))

        # DEC-CA-0059 merge-once: a multi-target leg holds one win per target
        # it took, all naming ONE judged checkpoint — its delta enters the
        # init exactly once. A repeated merge (same checkpoint, king, rows)
        # would be applied twice by the rebuild AND match a doubled init.
        if len(merge_identities(manifest.warm_start_merges)) != len(
                manifest.warm_start_merges):
            return "era_init_mismatch: a lane-B merge is declared more than once"
        if strict:
            from ..trainer.expert_lane import merge_row_overlap

            hit = merge_row_overlap(list(manifest.warm_start_merges))
            if hit is not None:
                return (f"era_init_mismatch: lane-B merges overlap on MoE layer {hit[0]} "
                        f"expert row {hit[1]}")
        for m in manifest.warm_start_merges:
            if not _recorded(m) and m.get("round_id"):
                try:
                    self._lane_b_fold_on_demand(str(m["round_id"]))
                except LaneBTransient as e:
                    raise StorageError(f"era_init_unverifiable: {e}") from e
            if not _recorded(m):
                return (f"era_init_mismatch: merge {m.get('rebased_pointer')!r} is not a "
                        "lane-B win this validator recorded")
        try:
            import torch
            from safetensors.torch import load_file

            from ..trainer.expert_lane import _moe_names, apply_merges

            king = next(e for e in manifest.entries if e.role == "king")

            def _load_dir(p):
                return self._fetch_checkpoint_dir(replace(king, trained_pointer=p))

            base_dir = _load_dir(manifest.warm_start_base)
            want = apply_merges(
                load_file(str(base_dir / "weights.safetensors")), list(manifest.warm_start_merges),
                lambda p: load_file(str(_load_dir(p) / "weights.safetensors")),
                _moe_names(base_dir), strict=strict)
            got = load_file(str(_load_dir(manifest.warm_start_ckpt) / "weights.safetensors"))
        except StopIteration:
            return "era_init_mismatch: merged init declared without a king entry"
        except _merge_error_type() as e:
            # deterministic: a merge touching rows outside its selection
            return f"era_init_mismatch: {e}"
        except Exception as e:  # noqa: BLE001 — unreadable ⇒ transient (retried in grace)
            raise StorageError(f"era_init_unverifiable: {type(e).__name__}: {e}") from e
        if want.keys() != got.keys() or any(not torch.equal(want[k], got[k]) for k in want):
            return "era_init_mismatch: merged init bytes differ from member + recorded merges"
        self._merged_init_ok.add(cache_key)
        return None

    def _era_ledger(self, era: EraSpec) -> tuple[int, tuple[str, ...]]:
        """``(generation, members)`` in force for ``era``: the staged
        promotion once its effective era is reached, else the live one.
        DEC-CA-0050 lineage reset: a generation that may not initialise a v2
        era (:func:`v2_generation_usable` — a toto2 generation) reads as
        ``(0, ())``, random init — the rule the trainer's ledger applies."""
        assert self.cascade is not None
        state = self.cascade.state
        if (self._staged_generation(state)
                and int(era.index) >= int(state.pending_effective_era)):
            gen, members = int(state.pending_generation), tuple(state.pending_members)
            eff = int(state.pending_effective_era)
        else:
            gen, members = int(state.generation), tuple(state.members)
            eff = int(getattr(state, "generation_effective_era", 0) or 0)
        if gen and not v2_generation_usable(self.cfg, int(era.start_block), eff):
            return 0, ()
        return gen, members

    @staticmethod
    def _staged_generation(state) -> int:
        """The generation STAGED for a future era (0 = none). A pending record
        without members is not one — see :meth:`_skipped_generation`."""
        if state.pending_generation > state.generation and state.pending_members:
            return int(state.pending_generation)
        return 0

    @staticmethod
    def _skipped_generation(state) -> int:
        """The skip marker: ``pending_generation`` above the live generation
        with NO members records that every generation up to it is dead — its
        record was judged never effective (:data:`~cascade.validator.
        king_check.NOTICE_LATE`), so the init stays the live generation's and
        the catch-up walk continues at the next one. Persisted through the
        existing pending fields (drop-when-zero, carried across a dethrone by
        the DEC-CA-0055 keep rule), cleared when a real record is staged above
        it or the next generation installs. 0 = no marker."""
        if state.pending_generation > state.generation and not state.pending_members:
            return int(state.pending_generation)
        return 0

    def _install_era_promotion(self, manifest: TrainingManifest) -> None:
        """Install the staged promotion at its first effective settlement —
        called when a gated era manifest is HANDLED (scored, resync-held or
        rejected), never from a gate. Idempotent."""
        if self.cascade is None or not self._era_active(manifest) or manifest.era is None:
            return
        try:
            era = EraSpec.from_json(manifest.era)
        except ValueError:
            return
        state = self.cascade.state
        if (self._staged_generation(state)
                and int(era.index) >= int(state.pending_effective_era)):
            # DEC-CA-0055: a v2 king-only promotion does NOT restart the reign
            # clock at its install — the clock already restarted at the
            # passing check's settlement (and keeps whatever it saw since)
            v2_keep = (self._v2_pending_kept() and state.reign_start_block is not None)
            self.cascade.note_promotion(
                generation=state.pending_generation, members=state.pending_members,
                block=(int(state.reign_start_block) if v2_keep else era.start_block),
                effective_era=state.pending_effective_era,
                observed=(state.clock_observed if v2_keep else True))
            self.cascade.stage_promotion(generation=0, members=(), effective_era=0)
            log.info("cascade: promotion generation=%d installed at era %d",
                     self.cascade.state.generation, era.index)

    def _adopt_era_king(self, manifest: TrainingManifest) -> None:
        """Record the pointer the era's king is judged at — the era's first
        king leg, or (already set by apply_round / demote_to_trained) the
        crowned winner's — AFTER the settlement is handled. Adoption inside
        the gate left a transient with an adopted pointer, and a legitimately
        re-published manifest carrying a different first-king-leg pointer
        was then rejected forever."""
        if not self._era_active(manifest) or manifest.era is None:
            return
        try:
            era = EraSpec.from_json(manifest.era)
        except ValueError:
            return
        king = manifest.entry_for_role("king")
        pointer = self.state.king_pointer
        key = self._pointer_era(era)
        if (king is not None and king.miner_hotkey == self.state.king_hotkey
                and (self.state.era_index != key or not pointer)):
            pointer = str(king.trained_pointer)
        if self.state.era_index != key or pointer != self.state.king_pointer:
            self.state = replace(self.state, era_index=int(key), king_pointer=pointer)
        epoch = self._epoch_start(era)
        if epoch is not None and (not self.state.epoch_init
                                  or int(self.state.epoch_init.get("epoch", -1)) != epoch):
            # the seed epoch's init, recorded at its first handled settlement
            self.state = replace(self.state, epoch_init={
                "epoch": int(epoch), "init": str(manifest.warm_start_ckpt or "")})

    def _pointer_era(self, era: EraSpec) -> int:
        """The era the king pointer is held for: the seed epoch's first era
        inside a v2 seed epoch (DEC-CA-0055 — the king trains once per epoch;
        a dethroner's checkpoint is the king leg for the rest of it), else
        the era itself (DEC-CA-0043)."""
        epoch = self._epoch_start(era)
        return int(era.index) if epoch is None else int(epoch)

    def process_settlement(self, manifest: TrainingManifest, windows, base_seed: int):
        """:meth:`process_round` for a settlement: the era ledger is installed
        first (idempotent), the era king pointer adopted after success. A
        transient raised by process_round leaves the pointer untouched."""
        self._install_era_promotion(manifest)
        outcome = self.process_round(manifest, windows, base_seed)
        self._adopt_era_king(manifest)
        return outcome

    def _refresh_pending_promotion(self, era: EraSpec) -> None:
        """Learn of NEWER promotion records ahead of their effective eras.

        Each record must survive the DEC-CA-0013 envelope (signature, set
        cap, ripeness when attestable, member provenance) AND claim an
        ``effective_era`` no earlier than one full era after it fired
        (:meth:`_stage_promotion_record`); a record failing either is ignored
        (logged) — the ledger keeps naming the live generation and a manifest
        trained from the new set fails ``era_init_mismatch`` until a valid
        record exists. A record or bench report that cannot be READ raises
        :class:`StorageError` instead (a transient the poll loop retries): an
        unreadable ledger is not evidence of a bad init.

        CATCH-UP: a validator that missed promotions (or joined at generation
        0) walks ``promotions/gen-<n>.json`` for ``generation + 1 … latest`` in
        order, never skipping one unjudged: each record is verified and staged
        exactly as the +1 record is, and one whose effective era this
        settlement has already reached while a NEWER record exists is
        installed here (the state every peer holds by this era — idempotent
        on a retry). A record whose notice is FINALLY late
        (:data:`~cascade.validator.king_check.NOTICE_LATE`) is never
        effective anywhere: it is recorded as skipped (:meth:`_skipped_generation`,
        the init stays the live generation's — the trainer's ledger applies
        the same rule) and the walk continues at the next generation, so one
        dead record cannot wedge every validator behind it. The walk stops at
        the first absent, malformed or otherwise failing record, and at the
        first record still ahead of this era (staged, installed at its first
        settlement by :meth:`_install_era_promotion` — the one-step lag is
        unchanged). A raise mid-walk restores the state it started from, so
        the gate stays pure for the retry."""
        assert self.cascade is not None
        state = self.cascade.state
        latest = self._fetch_latest_promotion_record()
        if latest is None:
            return
        if (self._staged_generation(state) == int(latest.generation)
                and tuple(latest.member_ids()) != tuple(state.pending_members)):
            # Two bodies for one generation: this validator staged one member
            # set, the published record now names another. Never silently
            # re-judge (peers that staged the first keep it) — hold loudly.
            from ..shared.hippius import StorageError

            log.error("cascade: promotion record gen=%d now names members %s but this "
                      "validator staged %s; holding", latest.generation,
                      tuple(latest.member_ids()), tuple(state.pending_members))
            raise StorageError(f"promotion_record_conflict: gen={latest.generation} "
                               "republished with a different member set")
        if latest.generation <= max(state.generation, state.pending_generation):
            return
        snapshot = state
        try:
            self._walk_promotions(era, latest)
        except BaseException:
            if self.cascade.state is not snapshot:
                self.cascade.state = snapshot
                self.cascade._persist()
            raise

    def _walk_promotions(self, era: EraSpec, latest) -> None:
        assert self.cascade is not None
        while True:
            state = self.cascade.state
            staged, skipped = self._staged_generation(state), self._skipped_generation(state)
            # the next record to judge: past the staged one (already judged),
            # past the skip marker (dead), else the live generation's successor
            target = staged or max(state.generation, skipped) + 1
            if target > latest.generation:
                return
            if target != staged:
                record = (latest if target == latest.generation
                          else self._fetch_promotion_record(target, missing_ok=True))
                if record is None or int(record.generation) != target:
                    log.warning("cascade: promotion record gen=%d missing or malformed; "
                                "catch-up toward gen=%d stops at accepted gen=%d",
                                target, latest.generation, state.generation)
                    return
                if not self._stage_promotion_record(record):
                    if self._skipped_generation(self.cascade.state) >= target:
                        continue    # never effective: judge the next generation
                    return
                state = self.cascade.state
            if target == latest.generation or int(era.index) < int(state.pending_effective_era):
                return
            self._install_caught_up_promotion(era)

    def _install_caught_up_promotion(self, era: EraSpec) -> None:
        """Install the staged generation during a catch-up walk: a NEWER record
        exists and this settlement's era is at/after the staged effective era.
        The reign clock restarts at the effective era's start (where every
        peer restarted it) but is marked UNOBSERVED: this validator did not
        watch the reign that followed (dethrones included), so — as for any
        catching-up clock — the next record's timing/scope checks are skipped
        rather than judged on a reconstructed clock."""
        assert self.cascade is not None
        state = self.cascade.state
        eff = int(state.pending_effective_era)
        if eff and eff < int(era.index):
            block = min(int(era.start_block),
                        eff * era_length_blocks(self.cfg.round, int(era.start_block)))
        else:
            block = int(era.start_block)
        self.cascade.note_promotion(
            generation=state.pending_generation, members=state.pending_members,
            block=block, effective_era=eff, observed=False)
        self.cascade.stage_promotion(generation=0, members=(), effective_era=0)
        log.info("cascade: catch-up installed promotion generation=%d (effective era %d) "
                 "while judging era %d", self.cascade.state.generation, eff, era.index)

    def _stage_promotion_record(self, record) -> bool:
        """Verify one promotion record against the envelope and stage it;
        False (logged) when it is rejected."""
        from ..shared.promotion import verify_promotion_record_signature

        assert self.cascade is not None
        if self.verify_signatures and not verify_promotion_record_signature(
                record, self.cfg.manifest.trainer_hotkey):
            log.warning("cascade: promotion record gen=%d unsigned/invalid; ignored",
                        record.generation)
            return False
        member_ids = record.member_ids()
        attesting = self.cascade.can_verify_ripeness()
        if self._v2_record(record):
            # DEC-CA-0055: a v2 promotion is the king alone, backed by a
            # passing king check (which also replaced the ripeness predicate)
            why = self._v2_promotion_reason(record, attesting=attesting)
            if why == NOTICE_LATE:
                # Final for every node (the chain never gains an earlier
                # announcement): the record is never effective. Record the
                # skip so the walk proceeds to the next generation instead of
                # re-judging this one at every gate; the init stays the live
                # generation's, exactly as the trainer's ledger keeps it.
                self.cascade.stage_promotion(generation=record.generation, members=(),
                                             effective_era=int(record.effective_era))
                log.warning("cascade: promotion record gen=%d is never effective (%s); "
                            "skipped — the init stays generation %d", record.generation,
                            why, self.cascade.state.generation)
                return False
            if why is not None:
                log.warning("cascade: promotion record gen=%d rejected: %s",
                            record.generation, why)
                return False
            member_reason = self._verify_members(record, enforce_reign_scope=False,
                                                 carried_exempt=True,
                                                 lineage_size=self._v2_lineage_size(record))
            if member_reason is not None:
                log.warning("cascade: promotion record gen=%d rejected: %s",
                            record.generation, member_reason)
                return False
            self.cascade.stage_promotion(generation=record.generation, members=member_ids,
                                         effective_era=int(record.effective_era))
            log.info("cascade: v2 king promotion generation=%d staged for era %d (%s)",
                     record.generation, record.effective_era, member_ids[0][:64])
            return True
        k_max = max(1, int(self.cfg.scoring.cascade_top_k))
        if not member_ids or len(member_ids) > k_max:
            log.warning("cascade: promotion record gen=%d declares %d member(s), cap %d; "
                        "ignored", record.generation, len(member_ids), k_max)
            return False
        floor = min_effective_era(self.cfg.round, int(record.fired_block))
        if int(record.effective_era) < floor:
            log.warning("cascade: promotion record gen=%d claims effective_era %d, "
                        "earliest allowed %d (fired at block %d); ignored",
                        record.generation, record.effective_era, floor, record.fired_block)
            return False
        if attesting and not self.cascade.is_ripe(block=int(record.fired_block)):
            log.warning("cascade: promotion record gen=%d fired before the reign clock "
                        "ripened (block %d); ignored", record.generation, record.fired_block)
            return False
        # A StorageError out of the record fetch above or the member check
        # here propagates: the era gate cannot judge this manifest without
        # the provenance, and the poll loop retries the round (never a
        # receipt) — see _gate_read_failed.
        member_reason = self._verify_members(record, enforce_reign_scope=attesting,
                                             carried_exempt=True,
                                             lineage_size=self._v2_lineage_size(record))
        if member_reason is not None:
            log.warning("cascade: promotion record gen=%d rejected: %s",
                        record.generation, member_reason)
            return False
        self.cascade.stage_promotion(generation=record.generation, members=member_ids,
                                     effective_era=int(record.effective_era))
        log.info("cascade: promotion generation=%d staged for era %d (%d member(s))",
                 record.generation, record.effective_era, len(member_ids))
        return True

    def _check_era_king(self, manifest: TrainingManifest, era: EraSpec) -> str | None:
        """The king entry is judged at ONE pointer per era: the pointer this
        validator crowned (a dethrone's winner) or the era's first king leg.
        A new era adopts the manifest's king pointer; a stale-king manifest
        (hotkey ≠ champion) is left to the resync valve, not rejected here."""
        king = manifest.entry_for_role("king")
        if king is None:
            return None
        champ = self.state.king_hotkey
        if champ is not None and king.miner_hotkey != champ:
            return None
        if (self.state.era_index == self._pointer_era(era) and self.state.king_pointer
                and king.trained_pointer != self.state.king_pointer):
            return (f"era_king_pointer_mismatch: era {era.index} king pointer is "
                    f"{self.state.king_pointer!r}, manifest carries "
                    f"{king.trained_pointer!r}")
        # A new era's first king leg is adopted when the settlement is
        # handled (_adopt_era_king), never here: gates leave state untouched.
        return None

    def _entry_fault_active(self, manifest: TrainingManifest) -> bool:
        """CONSENSUS gate: from ``[scoring] entry_fault_from_block`` a bad ref
        invalidates only its own entry, not the settlement (0 = inert). Read
        defensively — a config predating the key is inert."""
        gate = int(getattr(self.cfg.scoring, "entry_fault_from_block", 0) or 0)
        return gate > 0 and self._epoch_start_block(manifest) >= gate

    def _check_era_refs(self, manifest: TrainingManifest) -> str | None:
        """Each entry's ref is the hotkey's revealed commitment AS OF its
        ``train_block`` (not the latest reveal at the boundary), so a
        re-commit between leg start and settlement cannot change what is
        judged. FAIL CLOSED without a history provider (the live loop wires
        ``client.poll_commitments(include_history=True)`` itself); a provider
        failure is a transient (raised — the loop retries next poll).

        Before ``entry_fault_from_block`` one bad entry rejects the whole
        settlement, judged on registered hotkeys' history only (a provider
        that also returns deregistered hotkeys, ``uid=None``, is filtered back
        to exactly what the head metagraph showed). From it, a CHALLENGER
        with an unbound / mismatched ref is excluded from judging and
        recorded (:meth:`era_entry_faults`, :meth:`judged_manifest`), the
        deregistered hotkeys' history counts, and only a bad KING ref still
        rejects the settlement."""
        if self.commitment_history_fn is None:
            return ("era_ref_unverifiable: no commitment-history provider — refs are "
                    "judged as of train_block from era_king_from_block")
        history = self.commitment_history_fn()
        isolate = self._entry_fault_active(manifest)
        if not isolate:
            history = [c for c in history if getattr(c, "uid", 0) is not None]
        faults: dict[str, str] = {}
        for e in manifest.entries:
            bound = ref_as_of(history, e.miner_hotkey, int(e.train_block))
            if bound is None:
                reason = (f"era_ref_unbound: {e.role} uid{e.miner_uid} had no revealed "
                          f"commitment as of block {e.train_block}")
            elif bound != e.gen_ref:
                reason = (f"era_ref_mismatch: {e.role} uid{e.miner_uid} ref {e.gen_ref!r} "
                          f"is not the commitment revealed as of block {e.train_block} "
                          f"({bound!r})")
            else:
                continue
            if not isolate or e.role == "king":
                return reason
            faults.setdefault(e.miner_hotkey, reason)
        cache = self._entry_fault_cache()
        cache.pop(str(manifest.round_id), None)
        if faults:
            cache[str(manifest.round_id)] = (manifest.canonical_body(), dict(faults))
            for hk, reason in faults.items():
                log.warning("round=%s entry %s excluded from judging: %s",
                            manifest.round_id, hk[:12], reason)
        return None

    def _entry_fault_cache(self) -> dict:
        cache = self.__dict__.get("_era_entry_faults")
        if cache is None:
            cache = self.__dict__["_era_entry_faults"] = {}
        return cache

    def era_entry_faults(self, manifest: TrainingManifest) -> dict[str, str]:
        """``{miner_hotkey: reason}`` for the entries the last era gate of THIS
        manifest (same signed body) excluded from judging; empty before
        ``entry_fault_from_block``, when nothing was excluded, or when the gate
        has not run on it. For the receipt (and the audit's replay)."""
        hit = self._entry_fault_cache().get(str(manifest.round_id))
        if hit is None or hit[0] != manifest.canonical_body():
            return {}
        return dict(hit[1])

    def judged_manifest(self, manifest: TrainingManifest) -> TrainingManifest:
        """The manifest the duel judges: ``manifest`` minus every entry of a
        hotkey :meth:`era_entry_faults` excluded (the king is never excluded —
        a bad king ref rejects the settlement). Its ``canonical_body`` no
        longer matches the signature: gate the ORIGINAL, judge this."""
        faults = self.era_entry_faults(manifest)
        if not faults:
            return manifest
        return replace(manifest, entries=[e for e in manifest.entries
                                          if e.miner_hotkey not in faults])
