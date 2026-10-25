"""Stake-weighted activation (DEC-CA-0045) and block-clock forfeiture for the validator."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

from ..shared.activation import (
    ActivationStore,
    apply_activation,
    apply_forfeit_activation,
    apply_margin_v2_activation,
    apply_prelaunch_burn_activation,
    ensure_signal,
    forfeit_feature,
    margin_v2_feature,
    prelaunch_burn_feature,
    record_for,
    resolve_activation,
    resolved_margin_v2_block,
    resolved_prelaunch_burn_block,
    resolved_rollover,
)
from .weights import WEIGHTS_RATE_LIMIT_BLOCKS

log = logging.getLogger("cascade.validator")


class ActivationMixin:
    """Rollover, forfeiture, margin-v2 and pre-launch burn lock-ins decided on chain."""

    # ── stake-weighted activation (DEC-CA-0045) ─────────────────────────────

    @property
    def activation_block(self) -> int:
        """The rollover block resolved FROM VALIDATOR SIGNALS and APPLIED to
        the live config (0 = none, or typed into chain.toml — the receipt
        then records nothing). Read off the config, not the record: a
        receipt must never claim a rollover this validator did not run."""
        return resolved_rollover(self.cfg)

    def apply_activation_block(self, block: int) -> bool:
        """Rewrite the live config for a rollover at ``block`` (see
        :func:`cascade.shared.activation.apply_activation`) and refresh every
        object that captured a config slice — the Cascade controller's grid
        and reign threshold. Returns True when the config changed."""
        from ..shared.era import effective_reign_threshold_rounds

        new_cfg = apply_activation(self.cfg, block)
        if new_cfg is self.cfg:
            return False
        self.cfg = new_cfg
        if self.cascade is not None:
            cfg = new_cfg
            self.cascade.round_cfg = cfg.round
            self.cascade.threshold_fn = (
                lambda b: effective_reign_threshold_rounds(cfg.round, cfg.scoring, b))
        log.warning("activation: DEC-CA-0043 rollover ARMED at block %d — rolling intake, "
                    "era king, tenure in blocks and the grid switch (%d → %d) all flip "
                    "there", block, new_cfg.round.epoch_blocks_prev, new_cfg.round.epoch_blocks)
        return True

    def _activation_step(self, client: object, *, now_block: int) -> None:
        """One resolver pass: persist any advance, apply a lock-in, keep this
        validator's on-chain note current. Never raises."""
        if not self.cfg.activation.enabled:
            return
        self._netconfig_now = int(now_block)
        try:
            # DEC-CA-0057: re-read the owner's overlay first, so a newly
            # verified one rides in the note written just below.
            if self.netconfig is not None and self.netconfig.enabled:
                self.netconfig.refresh(self._base_cfg())
            # Say "ready" BEFORE counting, so this validator's own stake is on
            # chain for the first boundary it tallies.
            self._ensure_own_signal(client)
            res = resolve_activation(self.cfg, client, now_block=int(now_block),
                                     record=self._activation)
            if res.tally is not None:
                self._activation_tally = res.tally
            if res.changed:
                self._activation = res.record
                if self.activation_store is not None:
                    self.activation_store.save(res.record)
            if self._activation.locked and self._activation.source != "config":
                self.apply_activation_block(self._activation.activation_block)
                # The note now carries the agreed block for late joiners.
                self._ensure_own_signal(client)
            self._forfeit_step(client, now_block=now_block)
            self._margin_v2_step(client, now_block=now_block)
            self._prelaunch_step(client, now_block=now_block)
            self._netconfig_step(client, now_block=now_block)
        except Exception as e:  # noqa: BLE001 — activation must never disturb a round
            log.warning("activation step failed (%s); retrying next poll", e)

    def _forfeit_store(self) -> ActivationStore | None:
        store = self.activation_store
        if store is None or store.path is None:
            return None
        return ActivationStore(store.path.with_name("activation_forfeit_state.json"))

    def apply_forfeit_block(self, block: int) -> bool:
        """Write the fleet-decided forfeiture block into the live config
        (``[scoring] forfeit_from_block``). Returns True when it changed."""
        new_cfg = apply_forfeit_activation(self.cfg, block)
        if new_cfg is self.cfg:
            return False
        self.cfg = new_cfg
        log.warning("activation: DEC-CA-0048 forfeiture ARMED at block %d — %d listed hotkey(s) "
                    "leave the throne and the court at the first settlement whose boundary "
                    "reaches it (successor %s)", new_cfg.scoring.forfeit_from_block,
                    len(new_cfg.scoring.forfeit_hotkeys),
                    (new_cfg.scoring.forfeit_successor_hotkey or "court")[:12])
        return True

    def _forfeit_step(self, client: object, *, now_block: int) -> None:
        """The forfeiture's own resolver pass (same tally rule, its own
        record). No-op without a forfeit list or under a typed block."""
        spec = forfeit_feature(self.cfg)
        if spec is None or not spec.decided_on_chain:
            return
        res = resolve_activation(self.cfg, client, now_block=int(now_block),
                                 record=self._forfeit, feature=spec)
        if res.tally is not None:
            self._forfeit_tally = res.tally
        if res.changed:
            self._forfeit = res.record
            store = self._forfeit_store()
            if store is not None:
                store.save(res.record)
        if (self._forfeit.locked and self._forfeit.source != "config"
                and self.apply_forfeit_block(self._forfeit.activation_block)):
            self._ensure_own_signal(client)


    @property
    def margin_v2_block(self) -> int:
        """The margin-v2 block resolved FROM VALIDATOR SIGNALS and applied to
        the live config (0 = none, or typed into chain.toml) — stamped on
        receipts so the audit replays the bar the fleet decided."""
        return resolved_margin_v2_block(self.cfg)

    def _margin_v2_store(self) -> ActivationStore | None:
        store = self.activation_store
        if store is None or store.path is None:
            return None
        return ActivationStore(store.path.with_name("activation_margin_v2_state.json"))

    def apply_margin_v2_block(self, block: int) -> bool:
        """Write the fleet-decided margin-v2 block into the live config
        (``[scoring] margin_v2_from_block``). Returns True when it changed."""
        new_cfg = apply_margin_v2_activation(self.cfg, block)
        if new_cfg is self.cfg:
            return False
        self.cfg = new_cfg
        sc = new_cfg.scoring
        log.warning("activation: DEC-CA-0049 dethrone bar v2 ARMED at block %d — rounds from "
                    "it on are judged at %.4f decaying to %.4f over %d blocks of tenure",
                    sc.margin_v2_from_block, sc.win_margin_start_v2, sc.win_margin_end_v2,
                    sc.margin_warmup_blocks_v2)
        return True

    def _margin_v2_step(self, client: object, *, now_block: int) -> None:
        """The margin-v2 bar's own resolver pass (same tally rule, its own
        record). No-op without a v2 bar or under a typed block."""
        spec = margin_v2_feature(self.cfg)
        if spec is None or not spec.decided_on_chain:
            return
        res = resolve_activation(self.cfg, client, now_block=int(now_block),
                                 record=self._margin_v2, feature=spec)
        if res.tally is not None:
            self._margin_v2_tally = res.tally
        if res.changed:
            self._margin_v2 = res.record
            store = self._margin_v2_store()
            if store is not None:
                store.save(res.record)
        if (self._margin_v2.locked and self._margin_v2.source != "config"
                and self.apply_margin_v2_block(self._margin_v2.activation_block)):
            self._ensure_own_signal(client)

    @property
    def prelaunch_burn_block(self) -> int:
        """The DEC-CA-0055 pre-launch burn block resolved FROM VALIDATOR
        SIGNALS and applied to the live config (0 = none, or typed into
        chain.toml) — stamped on receipts so the audit replays the burn."""
        return resolved_prelaunch_burn_block(self.cfg)

    def _prelaunch_store(self) -> ActivationStore | None:
        store = self.activation_store
        if store is None or store.path is None:
            return None
        return ActivationStore(store.path.with_name("activation_prelaunch_burn_state.json"))

    def apply_prelaunch_burn_block(self, block: int) -> bool:
        """Write the fleet-decided pre-launch burn block into the live config
        (``[v2] prelaunch_burn_from_block``). Returns True when it changed."""
        new_cfg = apply_prelaunch_burn_activation(self.cfg, block)
        if new_cfg is self.cfg:
            return False
        self.cfg = new_cfg
        log.warning("activation: DEC-CA-0055 pre-launch burn ARMED at block %d — every "
                    "pre-v2 weight vector sends %.0f%% to burn_uid %d from it until the "
                    "first v2 settlement", new_cfg.v2.prelaunch_burn_from_block,
                    100.0 * new_cfg.v2.prelaunch_burn, new_cfg.scoring.burn_uid)
        return True

    def _prelaunch_step(self, client: object, *, now_block: int) -> None:
        """The pre-launch burn's own resolver pass (same tally rule, its own
        record). No-op without a burn or under a typed block."""
        spec = prelaunch_burn_feature(self.cfg)
        if spec is None or not spec.decided_on_chain:
            return
        res = resolve_activation(self.cfg, client, now_block=int(now_block),
                                 record=self._prelaunch, feature=spec)
        if res.tally is not None:
            self._prelaunch_tally = res.tally
        if res.changed:
            self._prelaunch = res.record
            store = self._prelaunch_store()
            if store is not None:
                store.save(res.record)
        if (self._prelaunch.locked and self._prelaunch.source != "config"
                and self.apply_prelaunch_burn_block(self._prelaunch.activation_block)):
            self._ensure_own_signal(client)

    def _ensure_own_signal(self, client: object) -> None:
        """Write this validator's note when it is not the one already on
        chain. One chain read per CHANGE of note, not per poll: the payload
        last confirmed on chain is remembered in-process."""
        from ..shared.activation import own_signal_payload

        if not self._signal_hotkey:
            return
        seg, in_force = self._netconfig_note_parts()
        want = own_signal_payload(self._base_cfg(), self._activation, self._forfeit,
                                  self._margin_v2, self._prelaunch, seg, in_force)
        if want is None or want == self._signal_sent:
            return
        try:
            have = dict(client.read_plain_commitments())  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001 — retry next poll
            log.debug("activation: note read failed (%s)", e)
            return
        if have.get(self._signal_hotkey) == want:
            self._signal_sent = want                 # confirmed on chain
            return
        # Written now, confirmed by the read on the next poll.
        ensure_signal(client, self._base_cfg(), self._activation, hotkey=self._signal_hotkey,
                      forfeit_record=self._forfeit,
                      margin_v2_record=self._margin_v2,
                      prelaunch_record=self._prelaunch,
                      netconfig_segment=seg, in_force=in_force,
                      current=have)

    def _activation_startup(self, client: object) -> None:
        """Restore the persisted decision, apply it, and run one pass."""
        if not self.cfg.activation.enabled:
            return
        if self.activation_store is not None:
            self._activation = record_for(self.cfg, self.activation_store.load())
        # Apply a restored rollover FIRST: it switches the settlement grid
        # (3600 -> 900), and the forfeiture / margin-v2 blocks below are
        # validated against the grid in force — restored before it, a forfeit
        # block on the 900 grid reads as "not a settlement boundary" and the
        # validator dies at startup.
        if self._activation.locked and self._activation.source != "config":
            self.apply_activation_block(self._activation.activation_block)
        fstore, fspec = self._forfeit_store(), forfeit_feature(self.cfg)
        if fstore is not None and fspec is not None:
            self._forfeit = record_for(self.cfg, fstore.load(), fspec)
            if self._forfeit.locked and self._forfeit.source != "config":
                log.info("activation: restored forfeiture lock-in (block %d, forfeit at %d, via %s)",
                         self._forfeit.lock_block, self._forfeit.activation_block,
                         self._forfeit.source)
                self.apply_forfeit_block(self._forfeit.activation_block)
        mstore, mspec = self._margin_v2_store(), margin_v2_feature(self.cfg)
        if mstore is not None and mspec is not None:
            self._margin_v2 = record_for(self.cfg, mstore.load(), mspec)
            if self._margin_v2.locked and self._margin_v2.source != "config":
                log.info("activation: restored margin-v2 lock-in (block %d, bar from %d, via %s)",
                         self._margin_v2.lock_block, self._margin_v2.activation_block,
                         self._margin_v2.source)
                self.apply_margin_v2_block(self._margin_v2.activation_block)
        pstore, pspec = self._prelaunch_store(), prelaunch_burn_feature(self.cfg)
        if pstore is not None and pspec is not None:
            self._prelaunch = record_for(self.cfg, pstore.load(), pspec)
            if self._prelaunch.locked and self._prelaunch.source != "config":
                log.info("activation: restored pre-launch burn lock-in (block %d, burn from "
                         "%d, via %s)", self._prelaunch.lock_block,
                         self._prelaunch.activation_block, self._prelaunch.source)
                self.apply_prelaunch_burn_block(self._prelaunch.activation_block)
        self._signal_hotkey = str(getattr(client, "hotkey_ss58", lambda: "")() or "")
        if self._activation.locked and self._activation.source != "config":
            log.info("activation: restored lock-in (block %d, rollover %d, via %s)",
                     self._activation.lock_block, self._activation.activation_block,
                     self._activation.source)
        try:
            now_block = int(client.current_block())  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log.warning("activation: chain unavailable at startup (%s); first poll retries", e)
            return
        self._activation_step(client, now_block=now_block)

    def _activation_tick(self, client: object) -> None:
        if not self.cfg.activation.enabled:
            return
        try:
            now_block = int(client.current_block())  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log.debug("activation tick skipped (no block): %s", e)
            return
        self._activation_step(client, now_block=now_block)

    def _forfeit_tick(self, client: object) -> None:
        """Block-clock pass of DEC-CA-0048 (typed or resolved block alike;
        independent of ``[activation] enabled``). Never raises."""
        if not (getattr(self.cfg.scoring, "forfeit_hotkeys", ()) or ()):
            return
        try:
            now_block = int(client.current_block())  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            log.debug("forfeiture tick skipped (no block): %s", e)
            return
        try:
            self._forfeit_on_block(now_block=now_block)
        except Exception as e:  # noqa: BLE001 — the settlement path applies it again
            log.warning("forfeiture tick failed (%s); retrying next poll", e)

    def _forfeit_on_block(self, *, now_block: int) -> bool:
        """DEC-CA-0048 on the BLOCK clock: the poll ``now_block`` reaches
        ``[scoring] forfeit_from_block`` the listed hotkeys leave the throne and
        the court and the named successor is crowned — no manifest needed, so
        the weights move at the block itself rather than at the trainer's next
        manifest (the settlement path, ``_apply_forfeiture``, stays for a
        validator that first meets the block through a manifest; both are
        idempotent). The crowning is stamped with the GATE block, not the poll
        block, so every validator records the same ``king_since_block``
        whatever its poll timing. The standing weight vector is re-pushed as
        soon as the chain's rate limit allows. Returns True when the state
        changed."""
        from ..shared.era import forfeit_successor, forfeited_hotkeys
        from . import state as state_mod

        forfeited = forfeited_hotkeys(self.cfg.scoring, int(now_block))
        if not forfeited:
            return False
        gate = int(getattr(self.cfg.scoring, "forfeit_from_block", 0) or 0)
        t = state_mod.apply_forfeit(self.state, forfeited=forfeited,
                                    keep_former_kings=self.cfg.scoring.reward_prior_kings,
                                    successor=forfeit_successor(self.cfg.scoring, int(now_block)) or None,
                                    crowned_block=gate)
        if t is None:
            return False
        log.warning("FORFEITURE at block %d (gate %d, block clock): %s — king now %s, court %s",
                    int(now_block), gate, t.note, (t.state.king_hotkey or "vacant")[:12],
                    [hk[:12] for hk in t.state.former_kings])
        self.state = t.state
        try:
            self._persist_state()
        except Exception as e:  # noqa: BLE001 — the round path persists again
            log.warning("forfeiture: state persist deferred (%s)", e)
        # Pull the next re-assert forward to the earliest block the chain's
        # weights_rate_limit allows (an earlier push is a silent no-op).
        interval = int(self.cfg.validator.weight_set_interval_blocks)
        if self._last_weight_block is not None and interval > WEIGHTS_RATE_LIMIT_BLOCKS:
            self._last_weight_block -= interval - WEIGHTS_RATE_LIMIT_BLOCKS
        return True

    # ── owner-signed network config (DEC-CA-0057) ───────────────────────────

    def _base_cfg(self):
        """The live config WITHOUT a settlement's overlay (``self.cfg`` outside
        :meth:`_settlement_cfg`). Activation lock-ins are applied to it."""
        return self._cfg_base if self._cfg_base is not None else self.cfg

    def cfg_at(self, block: int | None):
        """The config a settlement whose boundary is ``block`` is judged under
        (:meth:`cascade.shared.netconfig.NetConfigResolver.config_at`): the
        base config itself — same object — when no adopted overlay is in
        force there, or ``[netconfig] owner_hotkey`` is empty."""
        base = self._base_cfg()
        if self.netconfig is None or not self.netconfig.enabled:
            return base
        return self.netconfig.config_at(base, block)

    @contextlib.contextmanager
    def _settlement_cfg(self, block: int | None) -> Iterator[None]:
        """Run a settlement (or a re-assert) with ``self.cfg`` = :meth:`cfg_at`
        ``(block)`` and the Cascade controller's captured grid / reign gate
        following it; restored on exit. A no-op (nothing touched) when no
        overlay is in force — the inert path is bit-identical. Nested use
        resolves its own block and restores the outer view on exit."""
        if (self.netconfig is None or not self.netconfig.enabled
                or block is None):
            yield
            return
        base = self._base_cfg()
        view = self.netconfig.config_at(base, int(block))
        if view is self.cfg:
            yield
            return
        cascade = self.cascade
        saved = (cascade.round_cfg, cascade.threshold_fn) if cascade is not None else None
        previous = (self.cfg, self._cfg_base, self._settlement_block)
        self._cfg_base, self.cfg, self._settlement_block = base, view, int(block)
        if cascade is not None:
            self._point_cascade(view)
        try:
            yield
        finally:
            if self.cfg is not view:
                log.error("netconfig: self.cfg was replaced inside a settlement view; the "
                          "replacement is dropped (the base config is restored)")
            self.cfg, self._cfg_base, self._settlement_block = previous
            if cascade is not None and saved is not None:
                cascade.round_cfg, cascade.threshold_fn = saved

    def _point_cascade(self, cfg) -> None:
        from ..shared.era import effective_reign_threshold_rounds

        self.cascade.round_cfg = cfg.round
        self.cascade.threshold_fn = (
            lambda b: effective_reign_threshold_rounds(cfg.round, cfg.scoring, b))

    def _netconfig_receipt_kw(self) -> dict:
        """Receipt stamp of the overlay this settlement is judged under
        (``{}`` with none — the receipt bytes are then unchanged)."""
        nc = getattr(self.cfg, "netconfig", None)
        if nc is None or not int(nc.seq or 0):
            return {}
        return {"netconfig_seq": int(nc.seq), "netconfig_digest": str(nc.digest)}

    def _netconfig_note_parts(self):
        """``(cfg segment, config in force now)`` for the note — ``(None,
        None)`` while inert."""
        nc = self.netconfig
        if nc is None or not nc.enabled:
            return None, None
        now = self._netconfig_now
        in_force = None
        if now is not None and nc.in_force(now) is not None:
            in_force = nc.config_at(self._base_cfg(), now)
        return nc.segment(), in_force

    def _netconfig_step(self, client: object, *, now_block: int) -> None:
        """Resolve the verified overlay's lock-in (its ``cfg-`` segment) and
        adopt it at its apply block. No-op while inert."""
        nc = self.netconfig
        if nc is None or not nc.enabled:
            return
        before = nc.segment()
        changed = nc.resolve(self._base_cfg(), client, now_block=int(now_block))
        if changed or nc.segment() != before:
            self._ensure_own_signal(client)
