"""Reward routing: equal-share weight vector across the king + prior kings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cascade.shared.chain import ChainClient, ChainError, decayed_share_vector


def test_winner_take_all_single_uid():
    w = decayed_share_vector([2], 4, decay=1.0)
    assert w == [0.0, 0.0, 1.0, 0.0]


def test_equal_split_across_kings():
    w = decayed_share_vector([0, 3], 4, decay=1.0)
    assert w == [0.5, 0.0, 0.0, 0.5]
    assert sum(w) == pytest.approx(1.0)


def test_dedupes_and_drops_out_of_range():
    # Duplicate UIDs collapse; UIDs >= n_uids (deregistered slot) are dropped.
    w = decayed_share_vector([1, 1, 9], 3, decay=1.0)
    assert w == [0.0, 1.0, 0.0]


def test_empty_burns_to_burn_uid():
    w = decayed_share_vector([], 4, burn_uid=0, decay=1.0)
    assert w == [1.0, 0.0, 0.0, 0.0]


def test_all_deregistered_burns():
    # Every rewarded king has left the metagraph ⇒ emission burns, not reverts.
    w = decayed_share_vector([7, 8], 4, burn_uid=2, decay=1.0)
    assert w == [0.0, 0.0, 1.0, 0.0]


def test_negative_uid_burns_genesis_baseline_king():
    # The genesis baseline king (cascade.trainer.loop.GENESIS_KING_UID = -1) is a
    # non-registered sentinel; a negative uid is out of range, so it is dropped and
    # emission burns to burn_uid — the baseline reigns without drawing reward until
    # a real miner dethrones it.
    w = decayed_share_vector([-1], 4, burn_uid=0, decay=1.0)
    assert w == [1.0, 0.0, 0.0, 0.0]


def test_burn_uid_out_of_range_raises():
    with pytest.raises(ChainError):
        decayed_share_vector([], 4, burn_uid=9, decay=1.0)


def test_nonpositive_n_uids_raises():
    with pytest.raises(ChainError):
        decayed_share_vector([0], 0, decay=1.0)


# ── ChainClient.weights_for_hotkey (the audit's on-chain cross-check) ─────────


class _FakeSubtensor:
    """bittensor 11.x shape: typed metagraph for membership + the raw u16
    ``SubtensorModule::Weights`` row, normalised client-side like 10.5's W."""

    def __init__(self, rows):
        self.rows = rows
        self.subnets = SimpleNamespace(metagraph=lambda netuid, block=None, **kw: SimpleNamespace(
            block=77, neurons=[SimpleNamespace(hotkey=h) for h in ("hk0", "hk1", "hk2")]))

    def query(self, item, params=None, *, block=None):
        assert tuple(item)[:2] == ("SubtensorModule", "Weights") and block == 77
        return self.rows.get(params[1], [])


def _client(rows=None):
    c = ChainClient(netuid=1)
    c._subtensor = _FakeSubtensor(rows if rows is not None else {1: [(2, 65535)]})
    return c


def test_weights_for_hotkey_returns_row():
    assert _client().weights_for_hotkey("hk1") == [0.0, 0.0, 1.0]


def test_weights_for_hotkey_none_when_unregistered():
    assert _client().weights_for_hotkey("ghost") is None


def test_weights_for_hotkey_normalises_and_handles_empty_rows():
    c = _client({2: [(0, 100), (1, 100)]})
    assert c.weights_for_hotkey("hk2") == [0.5, 0.5, 0.0]
    assert c.weights_for_hotkey("hk0") == [0.0, 0.0, 0.0]   # never set weights


# ── geometric decay across the king + prior kings ──────────────────────────────


def test_decayed_share_king_gets_most():
    from cascade.shared.chain import decayed_share_vector

    v = decayed_share_vector([13, 3, 5], 20, decay=0.5)   # king first, then former by recency
    assert v[13] > v[3] > v[5] > 0                        # strictly decreasing
    assert round(sum(v), 9) == 1.0
    # shares are 1 : .5 : .25 normalised
    assert abs(v[13] - 4/7) < 1e-9 and abs(v[3] - 2/7) < 1e-9 and abs(v[5] - 1/7) < 1e-9


def test_decay_one_is_equal_share():
    from cascade.shared.chain import decayed_share_vector

    assert decayed_share_vector([2, 5, 7], 10, decay=1.0) == [
        0.0, 0.0, 1 / 3, 0.0, 0.0, 1 / 3, 0.0, 1 / 3, 0.0, 0.0]


def test_decay_preserves_order_and_dedups_first_wins():
    from cascade.shared.chain import decayed_share_vector

    # the king (first) keeps the top share even if it also appears later
    v = decayed_share_vector([9, 4, 9], 12, decay=0.5)
    assert v[9] > v[4] and round(sum(v), 9) == 1.0
    assert abs(v[9] - 2/3) < 1e-9 and abs(v[4] - 1/3) < 1e-9   # only two distinct


def test_decay_out_of_range_dropped_and_empty_burns():
    from cascade.shared.chain import decayed_share_vector

    assert decayed_share_vector([99, 1], 5, decay=0.5)[1] == 1.0    # 99 dropped ⇒ 1 alone
    v = decayed_share_vector([99, 100], 5, decay=0.5, burn_uid=2)   # none valid ⇒ burn
    assert v[2] == 1.0 and sum(v) == 1.0


def test_decay_out_of_bounds_rejected():
    import pytest

    from cascade.shared.chain import ChainError, decayed_share_vector

    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ChainError, match="decay"):
            decayed_share_vector([1], 5, decay=bad)


def test_king_decay_loads_from_toml(cfg):
    # shipped chain.toml keeps the flat split (back-compat)
    assert cfg.scoring.king_decay == 0.5   # geometric decay shipped (2026-07-14)
