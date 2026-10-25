"""Weight vector, king vote, resync valve and settlement forfeiture."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

from ..shared.era import effective_resync_cap_rounds
from ..shared.manifest import TrainingManifest
from . import state as state_mod
from .state import ChampionState

if TYPE_CHECKING:
    from .loop import RoundOutcome

log = logging.getLogger("cascade.validator")

# A failed re-assert retries after this many blocks (~5 min), not every poll;
# well inside the 5000-block activity cutoff and above weights_rate_limit / 4.
WEIGHT_RETRY_BLOCKS = 25


# Subtensor ``weights_rate_limit`` on netuid 91: a set_weights inside this many
# blocks of the previous one is silently dropped by the chain. A forced early
# re-assert (a forfeiture crowning) waits at least this long.
WEIGHTS_RATE_LIMIT_BLOCKS = 100


class WeightsMixin:
    """Who gets this round's weight, and pushing it on chain."""

    def _maybe_reassert_weights(self, client: object) -> None:
        """Re-push the standing weight vector every ``weight_set_interval_blocks``.

        Subtensor treats a validator whose ``last_update`` is older than the
        subnet's ``activity_cutoff`` (5000 blocks ≈ 16.7 h on netuid 91) as
        inactive in Yuma consensus. A mainnet round (3600 blocks, 12 h) lands
        a manifest inside the cutoff — but voting only on manifests
        still goes inactive the moment the trainer stalls, and the margin is
        one missed round. This re-assert is what keeps that from silencing the
        validator. The vector needs no
        manifest: it is recomputed from the persisted champion state (king +
        registered prior kings; no champion burns to ``burn_uid``), so this is
        a pure freshness signal that can never move the throne. The interval
        must stay ≥ the subnet's ``weights_rate_limit`` (100 blocks on netuid
        91) or the chain silently no-ops the extrinsic; ≤ 0 disables.
        """
        interval = int(self.cfg.validator.weight_set_interval_blocks)
        if interval <= 0:
            return
        cur = int(client.current_block())  # type: ignore[attr-defined]
        if self._last_weight_block is not None and cur - self._last_weight_block < interval:
            return
        failed_at = getattr(self, "_weight_retry_from", None)
        if failed_at is not None and cur - failed_at < WEIGHT_RETRY_BLOCKS:
            return
        uids: list[int] = []
        if self.state.king_hotkey is not None:
            uid = client.uid_for_hotkey(self.state.king_hotkey)  # type: ignore[attr-defined]
            if uid is not None:
                uids.append(uid)
        boundary = self._reassert_boundary(cur)
        if self._ladder_regime(boundary):
            # DEC-CA-0055: under the ladder lane A pays the king and the
            # immediate former king only (the court list is not paid)
            uids = uids + self._court_uid(client, boundary, uids)
        else:
            for hk in self.state.former_kings:
                uid = client.uid_for_hotkey(hk)  # type: ignore[attr-defined]
                if uid is not None and uid not in uids:
                    uids.append(uid)
        log.info("re-asserting weights (last set %s, block %d): reward_uids=%s",
                 "never" if self._last_weight_block is None else
                 f"{cur - self._last_weight_block} blocks ago", cur,
                 uids or [self.cfg.scoring.burn_uid])
        # _apply_weights stamps _last_weight_block only on a successful push:
        # a failed one is retried after WEIGHT_RETRY_BLOCKS (not one interval
        # later — a missed refresh is what lets last_update age past the cutoff).
        before = self._last_weight_block
        self._apply_weights(client, "weight-reassert", uids, boundary=boundary)
        # a failed push retries after WEIGHT_RETRY_BLOCKS, not every poll tick
        self._weight_retry_from = cur if self._last_weight_block == before else None

    @staticmethod
    def _manifest_king_hotkey(manifest: TrainingManifest) -> str | None:
        e = manifest.entry_for_role("king")
        return e.miner_hotkey if e is not None else None

    @staticmethod
    def _manifest_king_uid(manifest: TrainingManifest) -> int | None:
        e = manifest.entry_for_role("king")
        return e.miner_uid if e is not None else None

    def king_synced(self, manifest: TrainingManifest) -> bool:
        """Whether the round's *trained* king matches the validator's champion.

        The trainer picks the king it trains from on-chain incentive, which lags
        the validator's dethrone verdicts. Until the champion
        the validator crowned actually becomes the highest-incentive UID — and so
        the king the trainer trains — the two disagree, and a round trained
        against the *old* king must not have its verdict applied to the *new*
        champion. Synced when the champion is unset (bootstrap) or the trained
        king is the champion.
        """
        if self.state.king_hotkey is None:
            return True
        return self._manifest_king_hotkey(manifest) == self.state.king_hotkey

    def _resync_step(self, manifest: TrainingManifest) -> tuple[ChampionState, str]:
        """Next champion state + receipt reason for a king-resync round.

        Called when the trained king != champion (``king_synced`` is False). By
        default it holds the throne, bumping the consecutive-hold counter, and the
        caller keeps voting the champion so incentive migrates and the trainer
        re-syncs. SAFETY VALVE: once the champion has stayed un-synced for
        ``scoring.king_resync_max_rounds`` consecutive rounds it can never be the
        king the trainer trains (e.g. it has no usable commitment), so holding
        forever would wedge the subnet — the valve abandons it and adopts the
        trainer's trained king (:func:`state.demote_to_trained`), and normal
        scoring resumes next round. ``king_resync_max_rounds <= 0`` disables the
        valve (hold indefinitely). The counter advances once per DISTINCT
        un-synced round: a restart re-gates the same stale manifest, and
        counting those re-gates would let repeated restarts during a pause trip
        the valve and demote a healthy champion. Pure: returns the
        new state; the caller persists, votes, and publishes.
        """
        champ = self.state.king_hotkey
        trained = self._manifest_king_hotkey(manifest)
        round_id = str(manifest.round_id)
        same_round = self.state.last_resync_round_id == round_id
        holds = self.state.resync_holds if same_round else self.state.resync_holds + 1
        cap = effective_resync_cap_rounds(self.cfg.round, self.cfg.scoring,
                                          self._epoch_start_block(manifest))
        from ..shared.era import forfeited_hotkeys
        if trained is not None and trained in forfeited_hotkeys(
                self.cfg.scoring, self._epoch_start_block(manifest)):
            # DEC-CA-0048: the trainer still names a FORFEITED king. The valve
            # must not crown it back — hold (and keep voting the successor)
            # until the trainer follows the successor rule.
            log.warning("round=%s trainer king %s is FORFEITED; the resync valve will not "
                        "adopt it — holding for the successor (%d/%s)",
                        manifest.round_id, (trained or "?")[:12], holds, cap)
            trained = None
        if 0 < cap <= holds and trained is not None:
            log.warning(
                "round=%s king_resync SAFETY VALVE: champion %s un-synced %d rounds "
                "(cap=%d) — demoting to trained king %s, resuming normal scoring next round",
                manifest.round_id, (champ or "?")[:12], holds, cap, (trained or "?")[:12],
            )
            return (
                state_mod.demote_to_trained(
                    self.state, trained_hotkey=trained,
                    trained_uid=self._manifest_king_uid(manifest),
                    trained_block=self._epoch_start_block(manifest),
                    trained_pointer=str(getattr(
                        manifest.entry_for_role("king"), "trained_pointer", "") or ""),
                ),
                f"king_resync_demoted: champion {champ} un-synced {holds} rounds "
                f"(cap={cap}); adopted trained king {trained}",
            )
        log.warning(
            "round=%s trainer king %s != champion %s; voting champion to re-sync "
            "incentive, KOTH state held (%d/%s)",
            manifest.round_id, (trained or "?")[:12], (champ or "?")[:12],
            holds, cap if cap > 0 else "∞",
        )
        return (
            replace(self.state, resync_holds=holds, last_resync_round_id=round_id),
            f"king_resyncing: champion {champ} != trained king {trained}",
        )

    def _king_uid_to_vote(self, manifest: TrainingManifest, *, client: object | None = None) -> int | None:
        """The UID to put the king's weight on this round.

        The **validator's champion state** is the authority on who holds the
        throne, so vote *that* king every round — not the (lagging) king the
        trainer happened to train. This is what makes a dethrone STICK: the new
        champion keeps the weight, incentive migrates to it, and next round the
        trainer trains it as king (they re-sync). Voting the trained/manifest
        king instead — the old behaviour — reverted a dethrone the moment the
        trainer lagged one round, orphaning the champion. The champion hotkey is
        resolved to its current UID via the metagraph (robust to re-registration);
        the manifest king is used only to bootstrap when there is no champion yet.
        """
        if self.state.king_hotkey is not None:
            if client is not None:
                resolved = client.uid_for_hotkey(self.state.king_hotkey)  # type: ignore[attr-defined]
                # UIDs are reusable. An unregistered hotkey has no payable UID.
                return resolved
            return self.state.king_uid
        king_entry = manifest.entry_for_role("king")
        if king_entry is not None and client is not None:
            return client.uid_for_hotkey(king_entry.miner_hotkey)
        return king_entry.miner_uid if king_entry is not None else None

    def _apply_forfeiture(self, manifest: TrainingManifest) -> None:
        """DEC-CA-0048: at a settlement whose boundary has reached
        ``[scoring] forfeit_from_block``, strip the listed hotkeys from the
        throne and the court BEFORE judging. The manifest's king entry (still the
        forfeited hotkey until the trainer follows the receipts) then reads as a
        stale-king manifest and takes the resync path — no duel is judged against
        a forfeited king's checkpoint. Idempotent: nothing changes once applied."""
        from ..shared.era import forfeit_successor, forfeited_hotkeys

        block = self._epoch_start_block(manifest)
        forfeited = forfeited_hotkeys(self.cfg.scoring, block)
        if not forfeited:
            return
        t = state_mod.apply_forfeit(self.state, forfeited=forfeited,
                                    keep_former_kings=self.cfg.scoring.reward_prior_kings,
                                    successor=forfeit_successor(self.cfg.scoring, block) or None,
                                    crowned_block=block)
        if t is None:
            return
        log.warning("round=%s FORFEITURE at block %s: %s — king now %s, court %s",
                    manifest.round_id, block, t.note,
                    (t.state.king_hotkey or "vacant")[:12],
                    [hk[:12] for hk in t.state.former_kings])
        self.state = t.state
        try:
            self._persist_state()
        except Exception as e:  # noqa: BLE001 — the round path persists again
            log.warning("forfeiture: state persist deferred (%s)", e)

    def _reward_uids(
        self, manifest: TrainingManifest, outcome: RoundOutcome | None, client: object
    ) -> list[int]:
        """UIDs that share this round's weight: the current king plus any
        ``former_kings`` still registered (teutonic-style equal-share payout).

        Returns an empty list when there is no king to vote for at all (no
        champion and no manifest king); the loop hands that to
        ``set_equal_share_weights``, which burns to ``burn_uid`` rather than
        reverting. The list is otherwise deduped/range-checked there too.

        ``[validator] force_burn`` empties the list HERE — not just at the
        weight push — so the published receipt's ``reward_uids`` agree with the
        burn vector actually set (``cascade-audit`` recomputes one from the
        other and fails on a mismatch).
        """
        if self.cfg.validator.force_burn:
            return []
        from ..shared.era import forfeited_hotkeys

        try:
            forfeited = forfeited_hotkeys(self.cfg.scoring, self._epoch_start_block(manifest))
        except Exception:  # noqa: BLE001 — no block context ⇒ the gate is simply not reached
            forfeited = frozenset()
        uids: list[int] = []
        if not (self.state.king_hotkey and self.state.king_hotkey in forfeited):
            king_uid = self._king_uid_to_vote(manifest, client=client)
            if king_uid is not None:
                uids.append(king_uid)
        try:
            boundary = self._epoch_start_block(manifest)
        except Exception:  # noqa: BLE001 — no block context ⇒ no v2 ladder
            boundary = None
        if boundary is not None and self._ladder_regime(boundary):
            # DEC-CA-0055: king + the immediate former king (while its 5 days
            # run) — the rest of lane A burns under the ladder
            if not (self.state.court_hotkey and self.state.court_hotkey in forfeited):
                uids = uids + self._court_uid(client, boundary, uids)
            return uids
        for hk in self.state.former_kings:
            if hk in forfeited:
                continue
            uid = client.uid_for_hotkey(hk)  # type: ignore[attr-defined]
            if uid is not None:
                uids.append(uid)
        return uids

    # ── DEC-CA-0055 emission policy ──────────────────────────────────────────

    def _ladder_state(self):
        from ..shared.emission import LadderState

        st = self.state
        return LadderState(king_share_bp=st.king_share_bp, court_hotkey=st.court_hotkey,
                           court_until_block=int(st.court_until_block or 0),
                           slashed=bool(st.king_slashed))

    def _emission_policy(self, boundary: int):
        """The DEC-CA-0055 policy for a vector set at settlement ``boundary``
        (pure over the config + this validator's champion state)."""
        from ..shared.emission import emission_policy

        return emission_policy(self.cfg, boundary=int(boundary),
                               king_since_block=self.state.king_since_block,
                               ladder=self._ladder_state())

    def _ladder_regime(self, boundary: int) -> bool:
        return self._emission_policy(boundary).king_share is not None

    def _court_uid(self, client: object, boundary: int, uids: list[int]) -> list[int]:
        """``[uid]`` of the immediate former king while its court runs (and it
        is registered and not the king), else ``[]``."""
        from ..shared.emission import court_active

        if not court_active(self._ladder_state(), boundary):
            return []
        uid = client.uid_for_hotkey(self.state.court_hotkey)  # type: ignore[attr-defined]
        if uid is None or uid in uids:
            return []
        return [int(uid)]

    def _reassert_boundary(self, cur: int) -> int:
        """The settlement boundary a re-assert's vector is shaped at: the
        current block floored to the grid in force (every validator floors
        alike; the policy is a function of block arithmetic + state)."""
        from ..shared.config import effective_epoch_blocks

        eb = int(effective_epoch_blocks(self.cfg.round, int(cur)))
        return int(cur) // eb * eb

    def _apply_weights(self, client: object, round_id: str, reward_uids: list[int], *,
                       boundary: int | None = None) -> tuple[float, ...]:
        """Set the equal-share weight vector on chain; return it (empty on failure).

        Shared by the scored path and the king-resync path. Always sets weights —
        an empty ``reward_uids`` burns to ``burn_uid`` so emission still leaves the
        network. A failed extrinsic — raised, or reported unsuccessful by the
        client (``False``) — is logged, NOT stamped as the last weight block
        (the re-assert retries it next poll), and the empty vector is recorded
        truthfully in the receipt. A client returning ``None`` (no outcome) is
        treated as submitted.

        ``[validator] force_burn`` overrides the vector to a burn HERE — the
        single choke point every push flows through (scored, resync, re-assert) —
        so the receipt records the burn that was actually set. Champion state is
        never touched by this override."""
        from ..shared.chain import decayed_share_vector

        if self.cfg.validator.force_burn:
            log.warning(
                "FORCE-BURN active (round=%s): %sburning to uid %d (champion state "
                "untouched — unset [validator] force_burn and restart to resume voting)",
                round_id,
                f"dropping reward_uids={reward_uids}; " if reward_uids else "",
                self.cfg.scoring.burn_uid,
            )
            reward_uids = []

        decay = self.cfg.scoring.king_decay
        self._last_lane_b_pay = None
        self._last_lane_c_pay = None
        self._last_emission = None
        try:
            n_uids = client.n_uids()  # type: ignore[attr-defined]
            if boundary is None:
                with contextlib.suppress(Exception):
                    boundary = self._reassert_boundary(
                        int(client.current_block()))  # type: ignore[attr-defined]
            policy = None
            if boundary is not None and not self.cfg.validator.force_burn:
                policy = self._emission_policy(int(boundary))
                if policy.inert:
                    policy = None
            lb_block = self._lane_b_block
            if lb_block is None and self.cfg.lane_b.from_block:
                # restart / re-assert before any settlement this session: the
                # PERSISTED last settled lane-B boundary — the block every peer
                # paid from — never the head floored (that ages a win early)
                lb_block = self._lane_b().settled_block
            if lb_block is None and self.cfg.lane_b.from_block:
                # no lane-B history at all (fresh validator): the current
                # settlement boundary (every validator floors alike)
                with contextlib.suppress(Exception):
                    cur = int(client.current_block())  # type: ignore[attr-defined]
                    grid = self._lane_b_epoch_blocks(cur)
                    lb_block = cur // grid * grid
            lane_b_on = (lb_block is not None and self.cfg.lane_b.active(lb_block)
                         and not self.cfg.validator.force_burn)
            # DEC-CA-0056: lane C live at its settlement block ⇒ the lanes are
            # kept apart (cascade.shared.emission.lane_parts with lane C) even
            # when no DEC-CA-0055 rule shapes lane A (an inert policy).
            lc_block = (self._lane_c_settle_block(client)
                        if self.cfg.lane_c.from_block else None)
            lane_c_on = (lc_block is not None and self.cfg.lane_c.active(lc_block)
                         and not self.cfg.validator.force_burn)
            if lane_c_on:
                from ..shared.emission import EmissionPolicy

                if boundary is None:
                    boundary = int(lc_block)
                return self._set_policy_vector(client, round_id, reward_uids, n_uids,
                                               policy or EmissionPolicy(), int(boundary),
                                               lb_block if lane_b_on else None,
                                               lc_block=int(lc_block))
            if policy is not None:
                # DEC-CA-0055: the lanes kept apart, the policy shapes lane A
                # only, lane B rides through untouched (cascade.shared.emission)
                return self._set_policy_vector(client, round_id, reward_uids, n_uids,
                                               policy, int(boundary),
                                               lb_block if lane_b_on else None)
            if lane_b_on:
                vec, pay = self._lane_b_vector(client, reward_uids, n_uids, lb_block)
                if client.set_weight_vector(list(vec)) is False:  # type: ignore[attr-defined]
                    return self._weights_not_set(round_id)
                self._last_lane_b_pay = pay
                log.info("round=%s weights set (lane A + lane B): %s", round_id,
                         {u: round(w, 4) for u, w in enumerate(vec) if w > 0})
                with contextlib.suppress(Exception):
                    self._last_weight_block = int(client.current_block())  # type: ignore[attr-defined]
                return vec
            if client.set_equal_share_weights(  # type: ignore[attr-defined]
                    reward_uids, n_uids, decay=decay, burn_uid=self.cfg.scoring.burn_uid,
            ) is False:
                return self._weights_not_set(round_id)
            vec = tuple(decayed_share_vector(
                reward_uids, n_uids, decay=decay, burn_uid=self.cfg.scoring.burn_uid))
            log.info("round=%s weights set: reward_uids=%s (n_uids=%d, burn_uid=%d)",
                     round_id, reward_uids or [self.cfg.scoring.burn_uid], n_uids,
                     self.cfg.scoring.burn_uid)
            # Reset the re-assert timer (fake test clients lack current_block).
            with contextlib.suppress(Exception):
                self._last_weight_block = int(client.current_block())  # type: ignore[attr-defined]
            return vec
        except Exception as e:  # noqa: BLE001 — retried next round
            log.warning("weight set failed for round=%s (king holds, retried next round): %s",
                        round_id, e)
            return ()

    def _set_policy_vector(self, client: object, round_id: str, reward_uids: list[int],
                           n_uids: int, policy, boundary: int,
                           lb_block: int | None, *, lc_block: int | None = None
                           ) -> tuple[float, ...]:
        """Build, push and record a DEC-CA-0055 vector. The receipt's
        ``emission`` record (+ ``lane_b_pay`` when lane B is live, +
        ``lane_c_pay`` when lane C is — taken from its configured source's
        share) is enough for the audit to rebuild it exactly. An inert policy
        (lane C live, no DEC-CA-0055 rule) stamps no emission record."""
        from ..eval.expert_lane import LaneBWin
        from ..shared.emission import emission_record, emission_vector, lane_parts

        burn = self.cfg.scoring.burn_uid
        wins: list = []
        pay = None
        if lb_block is not None:
            _, pay = self._lane_b_vector(client, reward_uids, n_uids, lb_block)
            wins = [LaneBWin(uid=int(w["uid"]), domain=str(w["domain"]),
                             multiplier=float(w["multiplier"])) for w in pay["wins"]]
        pay_c = None
        if lc_block is not None:
            from ..shared.lane_c import pay_wins

            pay_c = self._lane_c_pay(client, n_uids, lc_block)
        a, b = lane_parts(reward_uids, n_uids, king_decay=self.cfg.scoring.king_decay,
                          burn_uid=burn, lane_b_cfg=(self.cfg.lane_b if pay is not None else None),
                          wins=wins,
                          lane_c_cfg=(self.cfg.lane_c if pay_c is not None else None),
                          c_wins=(pay_wins(pay_c) if pay_c is not None else ()),
                          c_slot_frac=(dict(pay_c["slot_frac"]) if pay_c is not None else None))
        vec = tuple(emission_vector(a, b if (pay is not None or pay_c is not None) else None,
                                    policy, burn_uid=burn, lane_a_uids=reward_uids))
        if client.set_weight_vector(list(vec)) is False:  # type: ignore[attr-defined]
            return self._weights_not_set(round_id)
        self._last_lane_b_pay = pay
        self._last_lane_c_pay = pay_c
        ladder = getattr(self, "_settlement_ladder", None)
        if not policy.inert:
            self._last_emission = emission_record(
                policy, boundary=boundary, king_since_block=self.state.king_since_block,
                ladder=(ladder[1] if ladder is not None and ladder[0] == str(round_id)
                        else None))
        log.info("round=%s weights set under the DEC-CA-0055 policy %s: %s", round_id,
                 policy.as_record(), {u: round(w, 4) for u, w in enumerate(vec) if w > 0})
        with contextlib.suppress(Exception):
            self._last_weight_block = int(client.current_block())  # type: ignore[attr-defined]
        return vec

    @staticmethod
    def _weights_not_set(round_id: str) -> tuple[float, ...]:
        """The chain rejected / did not accept the push (``_set_weights``
        returned False): nothing is stamped, so the re-assert retries it at
        the next poll, and the receipt records the empty vector."""
        log.warning("weight set NOT accepted for round=%s (rejected or unsuccessful "
                    "extrinsic); retried at the next poll", round_id)
        return ()
