"""Reward routing: equal-share weight vector across the king + prior kings."""

from __future__ import annotations

import pytest

from cascade.shared.chain import ChainError, equal_share_vector


def test_winner_take_all_single_uid():
    w = equal_share_vector([2], 4)
    assert w == [0.0, 0.0, 1.0, 0.0]


def test_equal_split_across_kings():
    w = equal_share_vector([0, 3], 4)
    assert w == [0.5, 0.0, 0.0, 0.5]
    assert sum(w) == pytest.approx(1.0)


def test_dedupes_and_drops_out_of_range():
    # Duplicate UIDs collapse; UIDs >= n_uids (deregistered slot) are dropped.
    w = equal_share_vector([1, 1, 9], 3)
    assert w == [0.0, 1.0, 0.0]


def test_empty_burns_to_burn_uid():
    w = equal_share_vector([], 4, burn_uid=0)
    assert w == [1.0, 0.0, 0.0, 0.0]


def test_all_deregistered_burns():
    # Every rewarded king has left the metagraph ⇒ emission burns, not reverts.
    w = equal_share_vector([7, 8], 4, burn_uid=2)
    assert w == [0.0, 0.0, 1.0, 0.0]


def test_burn_uid_out_of_range_raises():
    with pytest.raises(ChainError):
        equal_share_vector([], 4, burn_uid=9)


def test_nonpositive_n_uids_raises():
    with pytest.raises(ChainError):
        equal_share_vector([0], 0)


# ── ChainClient.weights_for_hotkey (the audit's on-chain cross-check) ─────────


class _FakeMeta:
    n = 3
    hotkeys = ["hk0", "hk1", "hk2"]
    W = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]


class _FakeSubtensor:
    def metagraph(self, netuid, lite=True):
        assert lite is False  # the weight matrix needs the full metagraph
        return _FakeMeta()


def _client():
    from cascade.shared.chain import ChainClient

    c = ChainClient(netuid=1)
    c._subtensor = _FakeSubtensor()
    return c


def test_weights_for_hotkey_returns_row():
    assert _client().weights_for_hotkey("hk1") == [0.0, 0.0, 1.0]


def test_weights_for_hotkey_none_when_unregistered():
    assert _client().weights_for_hotkey("ghost") is None


def test_weights_for_hotkey_falls_back_to_weights_attr():
    class _NoW:
        n = 2
        hotkeys = ["a", "b"]
        weights = [[1.0, 0.0], [0.5, 0.5]]

    class _Sub:
        def metagraph(self, netuid, lite=True):
            return _NoW()

    from cascade.shared.chain import ChainClient

    c = ChainClient(netuid=1)
    c._subtensor = _Sub()
    assert c.weights_for_hotkey("b") == [0.5, 0.5]
