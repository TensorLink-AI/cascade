"""``[validator] force_burn``: the operator kill-switch that burns every
weight-set (round votes, resync votes, re-asserts) while keeping the validator
Yuma-active. The override lives in ``_apply_weights`` — the single choke point
every push flows through — so receipts record the burn actually set. Champion
state must never be touched by it.
"""

from __future__ import annotations

from dataclasses import replace

from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis


class _Client:
    def __init__(self, block=1000, uids=None):
        self.block = block
        self.uids = uids or {}
        self.set_calls: list[list[int]] = []

    def current_block(self):
        return self.block

    def n_uids(self):
        return 256

    def uid_for_hotkey(self, hotkey):
        return self.uids.get(hotkey)

    def set_equal_share_weights(self, reward_uids, n_uids, *, decay, burn_uid):
        self.set_calls.append(list(reward_uids))


def _runner(cfg, state, *, force_burn):
    v = replace(cfg.validator, force_burn=force_burn)
    return ValidatorRunner(cfg=replace(cfg, validator=v), state=state)


KING = "5KingHotkey"


def test_off_by_default(cfg):
    assert cfg.validator.force_burn is False


def test_apply_weights_burns_when_forced(cfg):
    client = _Client(uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7), force_burn=True)
    vec = runner._apply_weights(client, "round-x", [7])
    assert client.set_calls == [[]]  # empty vector = burn to burn_uid
    assert vec[cfg.scoring.burn_uid] == 1.0
    assert sum(vec) == 1.0


def test_apply_weights_normal_when_off(cfg):
    client = _Client(uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7), force_burn=False)
    runner._apply_weights(client, "round-x", [7])
    assert client.set_calls == [[7]]


def test_reassert_burns_but_stays_active(cfg):
    # The freshness push still happens (last_update refresh = stays Yuma-active),
    # but the vector is a burn, not the champion.
    client = _Client(uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7), force_burn=True)
    runner._maybe_reassert_weights(client)
    assert client.set_calls == [[]]
    assert runner._last_weight_block == client.block


class _NoKingManifest:
    @staticmethod
    def entry_for_role(role):
        return None


def test_reward_uids_empty_when_forced(cfg):
    # The override must live in _reward_uids too, so the receipt's reward_uids
    # agree with the burn vector actually set (cascade-audit recomputes one
    # from the other).
    client = _Client(uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7), force_burn=True)
    assert runner._reward_uids(_NoKingManifest(), None, client) == []
    off = _runner(cfg, genesis(KING, 7), force_burn=False)
    assert off._reward_uids(_NoKingManifest(), None, client) == [7]


def test_champion_state_untouched(cfg):
    client = _Client(uids={KING: 7})
    state = genesis(KING, 7)
    runner = _runner(cfg, state, force_burn=True)
    runner._apply_weights(client, "round-x", [7])
    assert runner.state == state  # burn is a vote override, not a dethrone
