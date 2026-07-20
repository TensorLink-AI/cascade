"""Between-rounds weight re-assert: the validator must refresh its on-chain
``last_update`` every ``weight_set_interval_blocks`` from persisted champion
state alone — no manifest required.

Why: subtensor's ``activity_cutoff`` (5000 blocks ≈ 16.7 h on netuid 91) marks
a validator inactive in Yuma consensus once its last weight-set is older than
the cutoff. Mainnet rounds are 7200 blocks, so a vote-per-manifest validator
goes inactive every round, and a stalled trainer silences it entirely
(realized 2026-07-16→20: ~90 h stale during the round pause).
"""

from __future__ import annotations

from dataclasses import replace

from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import ChampionState, genesis


class _Client:
    """Minimal client surface for _maybe_reassert_weights/_apply_weights."""

    def __init__(self, block=1000, uids=None, fail=False):
        self.block = block
        self.uids = uids or {}          # hotkey -> uid
        self.fail = fail
        self.set_calls: list[list[int]] = []

    def current_block(self):
        return self.block

    def n_uids(self):
        return 256

    def uid_for_hotkey(self, hotkey):
        return self.uids.get(hotkey)

    def set_equal_share_weights(self, reward_uids, n_uids, *, decay, burn_uid):
        if self.fail:
            raise RuntimeError("extrinsic dropped")
        self.set_calls.append(list(reward_uids))


def _runner(cfg, state, interval=180):
    v = replace(cfg.validator, weight_set_interval_blocks=interval)
    return ValidatorRunner(cfg=replace(cfg, validator=v), state=state)


KING = "5KingHotkey"
PRIOR = "5PriorKingHotkey"


def test_first_tick_reasserts_immediately(cfg):
    client = _Client(uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7))
    runner._maybe_reassert_weights(client)
    assert client.set_calls == [[7]]
    assert runner._last_weight_block == client.block


def test_within_interval_is_quiet(cfg):
    client = _Client(block=1000, uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7))
    runner._last_weight_block = 900  # 100 < 180
    runner._maybe_reassert_weights(client)
    assert client.set_calls == []
    assert runner._last_weight_block == 900


def test_after_interval_repushes_champion_vector(cfg):
    client = _Client(block=1200, uids={KING: 7, PRIOR: 3})
    state = replace(genesis(KING, 7), former_kings=(PRIOR,))
    runner = _runner(cfg, state)
    runner._last_weight_block = 1000  # 200 >= 180
    runner._maybe_reassert_weights(client)
    assert client.set_calls == [[7, 3]]
    assert runner._last_weight_block == 1200


def test_no_champion_burns(cfg):
    # An empty vector still refreshes last_update: set_equal_share_weights
    # burns to burn_uid on [], which is the whole point of the freshness push.
    client = _Client(uids={})
    runner = _runner(cfg, ChampionState())
    runner._maybe_reassert_weights(client)
    assert client.set_calls == [[]]


def test_unregistered_hotkeys_skipped_and_deduped(cfg):
    client = _Client(uids={KING: 7})  # prior king deregistered
    state = replace(genesis(KING, 7), former_kings=(PRIOR, KING))
    runner = _runner(cfg, state)
    runner._maybe_reassert_weights(client)
    assert client.set_calls == [[7]]


def test_interval_zero_disables(cfg):
    client = _Client(uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7), interval=0)
    runner._maybe_reassert_weights(client)
    assert client.set_calls == []
    assert runner._last_weight_block is None


def test_failed_push_still_stamps_no_hammer(cfg):
    # A dropped extrinsic must NOT retry every poll tick — next attempt comes
    # after one interval (~27 attempts remain before the activity cutoff).
    client = _Client(block=1200, uids={KING: 7}, fail=True)
    runner = _runner(cfg, genesis(KING, 7))
    runner._maybe_reassert_weights(client)  # _apply_weights swallows the error
    assert runner._last_weight_block == 1200


def test_round_weight_set_resets_timer(cfg):
    # A real per-round _apply_weights success stamps the block, so the
    # re-assert never fires right after a round vote.
    client = _Client(block=5000, uids={KING: 7})
    runner = _runner(cfg, genesis(KING, 7))
    runner._apply_weights(client, "round-x", [7])
    assert runner._last_weight_block == 5000
    runner._maybe_reassert_weights(client)  # same block ⇒ within interval
    assert client.set_calls == [[7]]  # only the round vote, no re-assert
