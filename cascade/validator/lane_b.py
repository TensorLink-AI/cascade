"""Lane-B (DEC-CA-0050) domain-expert judging and folding for the validator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..shared.manifest import (
    TrainedEntry,
    TrainingManifest,
    load_manifest,
    verify_signature,
)
from ..shared.receipt import WindowScoreRecord

log = logging.getLogger("cascade.validator")


class LaneBTransient(Exception):
    """Lane-B evidence (an expert manifest, a lane-A manifest of the chain, a
    checkpoint) could not be READ or scored — a transient: the settlement is
    retried next poll (no latch, no receipt) within the read grace."""


@dataclass
class _LaneBRound:
    """One settlement's lane-B material, judged and ready to fold."""

    round_id: str
    block: int
    rows: list            # the receipt's lane-B verdict rows
    judgements: list      # ExpertJudgement list ([] = nothing to record)
    apply: bool = True    # False = left unapplied (catch-up re-judges it later)
    acceptances: list = field(default_factory=list)


class LaneBMixin:
    """Lane-B state, judging, catch-up and the combined weight vector."""

    # ── lane B (DEC-CA-0050) ──────────────────────────────────────────────────

    def _window_records(self, scores, manifest: TrainingManifest
                        ) -> tuple[WindowScoreRecord, ...]:
        """The receipt's signed per-window records. The pool ``domain`` label
        is stamped ONLY while lane B is active at this settlement's block —
        before that every receipt is byte-identical to the pre-v2 validator
        (the deployed pool is domain-labeled, so an unconditional stamp would
        change every receipt the day v2 ships)."""
        keep = self.cfg.lane_b.active(self._epoch_start_block(manifest))
        return tuple(WindowScoreRecord.from_score(s, keep_domain=keep) for s in scores)

    def _lane_b_state_path(self) -> Path:
        return Path(str(self.cfg.validator.state_db_path) + ".lane_b.json")

    def _lane_b(self):
        """The lane-B state, loaded once. An absent file is a fresh state; a
        TORN one is never trusted as empty — it is discarded loudly and the
        next settlement (or startup) rebuilds it by catch-up (``last_round``
        is None, so the walk replays the recent chain)."""
        from ..shared.lane_b import LaneBState

        st = self._lane_b_state
        if st is None:
            p = self._lane_b_state_path()
            if not p.is_file():
                st = LaneBState()
            else:
                try:
                    st = LaneBState.from_json(p.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001
                    log.error("lane B: state %s is unreadable (%s) — discarded; rebuilding "
                              "from the last %d settlements by catch-up", p, e,
                              self.cfg.lane_b.replay_settlements)
                    st = LaneBState()
            self._lane_b_state = st
        return st

    def _persist_lane_b(self) -> None:
        from ..shared.lane_b import save_state_atomic

        try:
            save_state_atomic(self._lane_b(), self._lane_b_state_path())
        except Exception as e:  # noqa: BLE001
            log.warning("failed to persist lane B state: %s", e)
            raise

    def _lane_b_snapshot(self) -> str | None:
        return self._lane_b().to_json() if self.cfg.lane_b.from_block else None

    def _lane_b_restore(self, text: str | None) -> None:
        if text is not None:
            from ..shared.lane_b import LaneBState

            self._lane_b_state = LaneBState.from_json(text)

    def _lane_b_vector(self, client: object, reward_uids: list[int], n_uids: int,
                       block: int) -> tuple[tuple[float, ...], dict]:
        """``(vector, payout record)`` — the record is what the receipt's
        ``lane_b_pay`` carries so the audit rebuilds the vector exactly."""
        from ..shared.lane_b import combined_weights, lane_b_payouts

        st = self._lane_b()
        uid_of = {}
        for w in st.wins.values():
            uid = client.uid_for_hotkey(w.hotkey)  # type: ignore[attr-defined]
            if uid is not None:
                uid_of[w.hotkey] = int(uid)
        eb = self._lane_b_era_blocks(block)
        vec = tuple(combined_weights(
            reward_uids, n_uids, king_decay=self.cfg.scoring.king_decay,
            burn_uid=self.cfg.scoring.burn_uid, state=st, block=block,
            era_blocks=eb, cfg=self.cfg.lane_b, uid_of=uid_of))
        pay = {"block": int(block), "era_blocks": int(eb),
               "wins": lane_b_payouts(st, block=block, era_blocks=eb, cfg=self.cfg.lane_b,
                                      n_uids=n_uids, uid_of=uid_of)}
        lc = getattr(self.cfg, "lane_c", None)
        if lc is not None and lc.active(block):
            # Record B's actual allocation under C's configured funding
            # source (absent before lane C: bytes unchanged).
            from ..shared.lane_c import lane_c_shares

            pay["share"] = lane_c_shares(self.cfg.lane_b, lc, block)[0]
            from ..shared.lane_b import lane_b_slots

            pay["slots"] = len(lane_b_slots(self.cfg.lane_b))
        return vec, pay

    def _lane_b_receipt_kw(self, rows: list[dict] | None) -> dict:
        """The receipt's lane-B fields for the vector just set: whenever the
        lane-B vector was set the receipt carries rows (``[]`` when nothing
        was judged) AND the payout record, so the audit never falls back to
        strict lane-A equality on a lane-B vector (the resync path did)."""
        pay = self._last_lane_b_pay
        if rows is None and pay is not None:
            rows = []
        return {"lane_b": rows, "lane_b_pay": pay}

    def _lane_b_epoch_blocks(self, block: int) -> int:
        """Settlement grid in force at ``block`` (flooring a block to the
        settlement boundary the lane-B vector is paid at)."""
        from ..shared.config import effective_epoch_blocks

        return int(effective_epoch_blocks(self.cfg.round, block))

    def _lane_b_era_blocks(self, block: int) -> int:
        """Era length in force at ``block`` — the lane-B payout clock counts
        ERAS (the grid in force × ``era_settlements``)."""
        from ..shared.era import era_length_blocks

        return int(era_length_blocks(self.cfg.round, block))

    # Lane-B state is a deterministic FOLD over the chained settlements'
    # bound expert manifests (oldest first), whichever path each arrives by:
    # the direct path (every settlement that passed the lane-A gate, scored or
    # resync), the catch-up walk (rounds this validator missed: downtime, a
    # lost/torn state file, a gate-rejected round), or an on-demand judge
    # when a merged era init names a merge not yet folded. Each is judged at
    # ITS OWN block / windows / king pointer, so every validator converges on
    # the same state whatever it saw live.

    def _lane_b_read(self, key: str) -> str:
        store = self._lane_b_store
        if store is None:
            raise LaneBTransient("no manifest store wired for lane B")
        try:
            return store.get_text(key)
        except Exception as e:  # noqa: BLE001 — any read failure is a transient
            raise LaneBTransient(f"{key} unreadable: {type(e).__name__}: {e}") from e

    def _lane_b_load_settlement(self, round_id: str) -> TrainingManifest:
        from ..shared.hippius import manifest_round_key

        return load_manifest(self._lane_b_read(manifest_round_key(round_id)))

    def _lane_b_judge_round(self, manifest: TrainingManifest) -> _LaneBRound:
        from .merge_acceptance import acceptance_block

        result = self._lane_b_judge_experts(manifest)
        if (manifest.warm_start_merges and manifest.era is not None
                and self.cfg.lane_b.acceptance_at_era(acceptance_block(self.cfg, manifest))):
            from ..shared.hippius import StorageError
            from .merge_acceptance import verify_acceptance

            try:
                reason = self._verify_merged_init(manifest, strict=True)
                if reason:
                    result.rows.append({"error": reason})
                else:
                    result.acceptances = verify_acceptance(self, manifest)
            except StorageError as e:
                raise LaneBTransient(str(e)) from e
        return result

    def _lane_b_judge_experts(self, manifest: TrainingManifest) -> _LaneBRound:
        """Fetch + verify ``manifest``'s bound expert manifest and judge it.
        Deterministic given the bytes; any READ / scoring failure raises
        :class:`LaneBTransient` (the caller retries — never a verdict)."""
        from ..shared.expert_manifest import (
            expert_manifest_key,
            load_expert_manifest,
            verify_expert_manifest,
        )
        from .expert_lane import judge_manifest

        rid, block = str(manifest.round_id), self._epoch_start_block(manifest)
        sha = str(getattr(manifest, "expert_manifest_sha256", "") or "")
        if not sha:
            # no expert manifest named ⇒ no lane-B legs this round (for every node)
            return _LaneBRound(rid, block, [], [])
        text = self._lane_b_read(expert_manifest_key(rid))
        try:
            em = load_expert_manifest(text)
        except Exception as e:  # noqa: BLE001 — the named bytes are malformed: a verdict
            return _LaneBRound(rid, block, [{"error": f"expert_manifest_malformed: {e}"}], [])
        if em.digest() != sha or str(em.round_id) != rid:
            log.warning("lane B: expert manifest for %s does not match the signed binding "
                        "(sha %s… vs %s…); lane-B part rejected, lane A proceeds",
                        rid, em.digest()[:12], sha[:12])
            return _LaneBRound(rid, block, [{"error": "expert_manifest_sha_mismatch"}], [])
        if self.verify_signatures and not verify_expert_manifest(
                em, self.cfg.manifest.trainer_hotkey):
            log.warning("lane B: expert manifest for %s fails signature; ignored", rid)
            return _LaneBRound(rid, block, [{"error": "bad_signature"}], [])
        kings = [e for e in manifest.entries if e.role == "king"]
        if not kings:
            return _LaneBRound(rid, block, [{"error": "no_king"}], [])
        king = kings[0]
        if self._lane_b_window_source is None:
            raise LaneBTransient("no window source wired for lane B")
        base_seed = int(manifest.round_id)
        windows = self._verdict_windows(self._lane_b_window_source, base_seed, block=block)

        def _as_entry(pointer: str) -> TrainedEntry:
            return replace(king, role="challenger", trained_pointer=pointer)

        from .expert_lane import judgements_record

        training = self._training_for(manifest)
        try:
            king_scores = self._evaluate(king, windows, training)
            js = judge_manifest(
                em, king_pointer=king.trained_pointer, king_scores=king_scores,
                evaluate=lambda p: self._evaluate(_as_entry(p), windows, training),
                fetch_dir=lambda p: self._fetch_checkpoint_dir(_as_entry(p)),
                cfg=self.cfg.lane_b, seed=str(base_seed),
                # DEC-CA-0050 merge guards: refuse a non-finite expert delta
                strict=self.cfg.lane_b.merge_guard_at_settlement(block),
            )
        except Exception as e:  # noqa: BLE001 — fetch/eval failure: retried, never a verdict
            raise LaneBTransient(f"judging {rid} failed: {type(e).__name__}: {e}") from e
        return _LaneBRound(rid, block, judgements_record(js), js)

    def _lane_b_walk_back(self, head: str | None) -> list[_LaneBRound]:
        """Judge the chained settlements this validator has NOT folded, from
        ``head`` back to its last folded one (bounded by ``[lane_b]
        replay_settlements``; stops before lane B's activation), OLDEST
        FIRST. A manifest whose signature does not verify is skipped (it was
        never a settlement)."""
        st = self._lane_b()
        found: list[TrainingManifest] = []
        rid, n = str(head or ""), 0
        seen: set[str] = set()
        complete_ledger = False
        while rid and (complete_ledger or n < int(self.cfg.lane_b.replay_settlements)):
            if n >= 10000:
                raise LaneBTransient("merge-acceptance rebuild exceeds 10000 settlements; restore an authenticated state snapshot")
            if rid == st.last_round:
                break
            if rid in seen:
                if complete_ledger:
                    raise LaneBTransient("cycle in the merge-acceptance settlement ledger")
                break
            seen.add(rid)
            m = self._lane_b_load_settlement(rid)
            with self._settlement_cfg(self._epoch_start_block(m)):
                if not self.cfg.lane_b.active(self._epoch_start_block(m)):
                    break
                if self.cfg.lane_b.acceptance_at_settlement(self._epoch_start_block(m)):
                    # Accepted deltas remain in the model after payout expiry.
                    # Their merge-once identities must survive a fresh rebuild.
                    complete_ledger = True
                signed = not self.verify_signatures or verify_signature(
                    m, self.cfg.manifest.trainer_hotkey)
            if signed and rid not in st.applied:
                found.append(m)
            rid, n = str(m.prev_round_id or ""), n + 1
        if found:
            log.warning("lane B: catching up %d unfolded settlement(s) (%s … %s)", len(found),
                        found[-1].round_id, found[0].round_id)
        rounds = []
        for m in reversed(found):
            with self._settlement_cfg(self._epoch_start_block(m)):
                result = self._lane_b_judge_round(m)
                if complete_ledger:
                    # Each later init must see earlier acceptance identities,
                    # including when rebuilding a validator from no state.
                    self._lane_b_commit([result])
                else:
                    rounds.append(result)
        return rounds

    def _lane_b_prepare(self, manifest: TrainingManifest) -> list[_LaneBRound] | None:
        """Everything lane B needs for this settlement, READ and JUDGED before
        lane A mutates anything: the catch-up rounds (when the chain position
        is not this settlement's predecessor) then this round. ``None`` when
        lane B is not active at its block. Raises :class:`LaneBTransient`."""
        block = self._epoch_start_block(manifest)
        if not self.cfg.lane_b.active(block):
            return None
        st = self._lane_b()
        rounds: list[_LaneBRound] = []
        prev = str(manifest.prev_round_id or "")
        if prev and st.last_round != prev:
            rounds += self._lane_b_walk_back(prev)
        rounds.append(self._lane_b_judge_round(manifest))
        return rounds

    def _lane_b_commit(self, rounds: list[_LaneBRound] | None) -> list[dict] | None:
        """Fold prepared rounds into the state (idempotent per round id),
        persist atomically, and return the CURRENT round's receipt rows."""
        if rounds is None:
            return None
        from .expert_lane import apply_judgements

        st = self._lane_b()
        for r in rounds:
            if not r.apply:
                continue
            if r.round_id not in st.applied:
                with self._settlement_cfg(r.block):
                    qualify_only = self.cfg.lane_b.acceptance_at_settlement(r.block)
                    wins = apply_judgements(st, r.judgements, round_id=r.round_id,
                                            settle_block=r.block,
                                            tiers=self.cfg.lane_b.full_pay_tiers,
                                            qualify_only=qualify_only)
                for merge, accepted, block in r.acceptances:
                    st.accept(merge, accepted, block)
                for w in wins:
                    log.info("lane B: %s %s for domain %s (round %s, LCB %.4f, tier %s "
                             "=> %d full eras%s)", w.hotkey[:8],
                             "qualified" if qualify_only else "merged", w.domain, r.round_id,
                             w.target_lcb, w.tier, w.full_eras,
                             ", BELOW the lowest tier" if w.below_floor else "")
            st.mark_applied(r.round_id, r.block)
        self._persist_lane_b()
        return list(rounds[-1].rows)

    def _lane_b_catch_up_startup(self) -> None:  # pragma: no cover — live-loop glue
        """Rebuild a missing / torn lane-B state at startup (best-effort; a
        failure is retried by the next settlement's walk)."""
        head = self.state.last_handled_round_id
        st = self._lane_b()
        if not head or st.last_round == head:
            return
        try:
            m = self._lane_b_load_settlement(head)
            with self._settlement_cfg(self._epoch_start_block(m)):
                if not self.cfg.lane_b.active(self._epoch_start_block(m)):
                    return
            rounds = self._lane_b_walk_back(head)
        except LaneBTransient as e:
            log.warning("lane B: startup catch-up deferred (%s)", e)
            return
        if rounds:
            self._lane_b_commit(rounds)

    def _lane_b_fold_on_demand(self, round_id: str) -> None:
        """Judge and fold one settlement out of chain order (a merged era init
        names a merge this validator has not recorded). Raises
        :class:`LaneBTransient` on a read/scoring failure."""
        st = self._lane_b()
        if round_id in st.applied:
            return
        m = self._lane_b_load_settlement(round_id)
        from .expert_lane import apply_judgements

        with self._settlement_cfg(self._epoch_start_block(m)):
            if self.verify_signatures and not verify_signature(
                    m, self.cfg.manifest.trainer_hotkey):
                return
            if not self.cfg.lane_b.active(self._epoch_start_block(m)):
                return
            r = self._lane_b_judge_round(m)
            apply_judgements(st, r.judgements, round_id=r.round_id, settle_block=r.block,
                             tiers=self.cfg.lane_b.full_pay_tiers,
                             qualify_only=self.cfg.lane_b.acceptance_at_settlement(r.block))
            for merge, accepted, block in r.acceptances:
                st.accept(merge, accepted, block)
        st.mark_applied(r.round_id, r.block, in_chain=False)
        self._persist_lane_b()
