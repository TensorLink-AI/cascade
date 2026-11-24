"""F2 (plan rev6.18 A4, DEC-CA-0041): the lane D scorer on ``(C, L)``
multichannel windows, the channel pick and the eval-host transport.
Contract: README.md "Multichannel windows (rev6.18 F2)".

Numpy tests never import torch. Tests marked ``torch`` build the tiny W5d
pyramid checkpoints (CI only). Every call the pre-F2 code refuses with a
non-assertion error is wrapped so it surfaces as an ``AssertionError``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from .helpers import HORIZON, SCORING_LEVELS, SEASONAL_PERIOD, series, sorted_quantiles

torch_mark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)
SEED_TAG = "lane_d_channels"
BLOCK_HASH = "0x" + "ab" * 32


def _scorer():
    import cascade.eval.lane_d_scorer as scorer

    return scorer


def _channels():
    import cascade.eval.lane_d_channels as ch

    return ch


def _spec_pick(block_hash, series_id, candidates, cap):
    """The plan's rule, re-stated (README)."""
    cand = sorted(set(candidates))
    if len(cand) <= cap:
        return cand
    if block_hash is None:
        return []
    seed = hashlib.sha256(f"{block_hash}{series_id}{SEED_TAG}".encode()).hexdigest()
    return sorted(sorted(cand, key=lambda c: hashlib.sha256(f"{seed}:{c}".encode()).hexdigest())
                  [:cap])


def _mv_score(q, target, history):
    try:
        return _scorer().score_window(q, target, history, SEASONAL_PERIOD)
    except ValueError as e:
        raise AssertionError(f"F2: score_window refuses a (C, h, 9) window: {e}") from e


def _mv_window(c: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    hist = np.stack([series(300, seed=seed + k, level=10.0 + 7 * k) for k in range(c)])
    tgt = np.stack([series(HORIZON, seed=50 + seed + k, level=float(hist[k, -1]))
                    for k in range(c)])
    q = np.stack([sorted_quantiles(rng, HORIZON, float(hist[k, -24:].mean())) for k in range(c)])
    return q, tgt, hist


# ── the per-window score ─────────────────────────────────────────────────────


def test_mv_score_is_the_arithmetic_channel_mean_of_univariate_scores():
    s = _scorer()
    q, tgt, hist = _mv_window(4)
    crps, mase = _mv_score(q, tgt, hist)
    each = [s.score_window(q[k], tgt[k], hist[k], SEASONAL_PERIOD) for k in range(4)]
    assert crps == pytest.approx(float(np.mean([e[0] for e in each])), rel=1e-12)
    assert mase == pytest.approx(float(np.mean([e[1] for e in each])), rel=1e-12)


def test_mv_score_masks_zero_target_channels_from_the_quantile_half_only():
    from cascade.eval.mase import mase as mase_one

    s = _scorer()
    q, tgt, hist = _mv_window(3, seed=4)
    tgt[1] = 0.0
    crps, mase = _mv_score(q, tgt, hist)
    kept = [s.score_window(q[k], tgt[k], hist[k], SEASONAL_PERIOD)[0] for k in (0, 2)]
    assert crps == pytest.approx(float(np.mean(kept)), rel=1e-12)
    m = [float(mase_one(q[k][:, s.MEDIAN_INDEX], tgt[k], hist[k], SEASONAL_PERIOD))
         for k in range(3)]
    assert mase == pytest.approx(float(np.mean(m)), rel=1e-12), "MASE averages every channel"


def test_mv_score_refuses_a_window_without_scale_and_a_bad_shape():
    q, tgt, hist = _mv_window(3, seed=9)
    _mv_score(q, tgt, hist)
    s = _scorer()
    with pytest.raises(ValueError):
        s.score_window(q, np.zeros_like(tgt), hist, SEASONAL_PERIOD)
    with pytest.raises(ValueError):
        s.score_window(q[:2], tgt, hist, SEASONAL_PERIOD)
    with pytest.raises(ValueError):
        s.score_window(q[:, :, :8], tgt, hist, SEASONAL_PERIOD)


def test_forecast_rows_count_channels_window_major():
    s = _scorer()
    assert hasattr(s, "forecast_rows"), "F2: lane_d_scorer.forecast_rows(histories)"
    hists = [np.ones(100), np.ones((3, 100)), np.ones((1, 100)), np.ones((32, 50))]
    assert s.forecast_rows(hists) == [1, 3, 1, 32]


# ── the channel pick ─────────────────────────────────────────────────────────


def test_pick_keeps_every_channel_up_to_the_cap():
    ch = _channels()
    assert ch.pick_channels(BLOCK_HASH, "s", [4, 0, 2], 32) == (0, 2, 4)
    assert ch.pick_channels(None, "s", range(32), 32) == tuple(range(32))


def test_pick_of_a_wider_series_is_the_block_hash_rule_and_the_same_for_every_order():
    ch = _channels()
    want = _spec_pick(BLOCK_HASH, "rmisc/x:7", range(50), 32)
    assert len(want) == 32
    assert list(ch.pick_channels(BLOCK_HASH, "rmisc/x:7", range(50), 32)) == want
    assert list(ch.pick_channels(BLOCK_HASH, "rmisc/x:7", reversed(range(50)), 32)) == want
    other = ch.pick_channels("0x" + "cd" * 32, "rmisc/x:7", range(50), 32)
    assert list(other) != want, "the pick depends on the boundary block hash"
    assert ch.pick_channels(None, "rmisc/x:7", range(50), 32) == (), (
        "an over-cap series without a block hash is not scorable")


def test_empty_channels_drop_before_the_cap():
    ch = _channels()
    hist = np.ones((40, 64))
    empty = (1, 3, 9, 10, 20, 21, 39, 0)
    hist[list(empty)] = np.nan
    hist[5, :10] = np.nan                       # a partly-missing channel is not empty
    w = {"window_id": "w", "series_id": "s", "history": hist, "target": np.ones((40, HORIZON))}
    p = ch.window_pick(w, BLOCK_HASH, 32)
    assert p.empty == tuple(sorted(empty))
    assert list(p.channels) == [c for c in range(40) if c not in empty]   # 32 left: no cut
    p = ch.window_pick(w, BLOCK_HASH, 30)
    assert list(p.channels) == _spec_pick(BLOCK_HASH, "s",
                                          [c for c in range(40) if c not in empty], 30)


def test_apply_pick_slices_and_one_channel_is_univariate():
    ch = _channels()
    hist, tgt = np.arange(5 * 70.0).reshape(5, 70), np.arange(5 * 8.0).reshape(5, 8)
    hist[2] = np.nan
    w = {"window_id": "w", "series_id": "s", "history": hist, "target": tgt}
    p = ch.window_pick(w, BLOCK_HASH, 32)
    got = ch.apply_pick(w, p)
    assert got["history"].shape == (4, 70) and np.array_equal(got["target"], tgt[[0, 1, 3, 4]])
    one = ch.ChannelPick("s", 5, (2,), (3,))
    got = ch.apply_pick(w, one)
    assert got["history"].shape == (70,) and np.array_equal(got["target"], tgt[3])
    flat = {"window_id": "u", "history": np.ones((1, 70)), "target": np.ones((1, 8))}
    got = ch.apply_pick(flat, None)
    assert got["history"].shape == (70,) and got["target"].shape == (8,)
    uni = {"window_id": "u", "history": np.ones(70), "target": np.ones(8)}
    assert ch.apply_pick(uni, None) is uni, "a univariate window is untouched"
    assert ch.apply_pick({**w, "history": np.ones((6, 70))}, p) is None


def test_channel_record_names_the_pick():
    ch = _channels()
    p = ch.ChannelPick("s", 40, (1, 2), tuple(range(3, 35)))
    rec = ch.channel_record(p)
    body = json.dumps(list(range(3, 35)), separators=(",", ":")).encode()
    assert rec == {"series_id": "s", "n_channels_raw": 40, "empty_channels": [1, 2],
                   "n_channels": 32, "channels": list(range(3, 35)),
                   "channels_sha256": hashlib.sha256(body).hexdigest()}


# ── the eval-host transport ──────────────────────────────────────────────────


def test_request_keeps_a_multichannel_history_2d_and_univariate_flat():
    from cascade.eval.lane_d_forecasts import REQUEST_SCHEMA, decode_request, encode_request

    hists = [np.arange(10.0), np.arange(30.0).reshape(3, 10), np.arange(10.0).reshape(1, 10)]
    meta, got = decode_request(encode_request({"schema": REQUEST_SCHEMA,
                                               "window_ids": ["a", "b", "c"]}, hists))
    assert [g.shape for g in got] == [(10,), (3, 10), (10,)]
    assert np.array_equal(got[1], hists[1])


def test_forecast_entry_wants_one_row_per_channel(tmp_path, monkeypatch):
    import cascade.eval.lane_d_forecasts as fmod

    wins = [{"window_id": "a", "history": np.ones(64)},
            {"window_id": "b", "history": np.ones((3, 64))}]
    monkeypatch.setattr(fmod, "_checkpoint", lambda *_a, **_k: tmp_path)

    def call(rows: int):
        out = np.zeros((rows, HORIZON, len(SCORING_LEVELS)))
        return fmod.forecast_entry(None, "f" * 64, wins, context=4096, horizon=HORIZON,
                                   contexts=(4096,), forecast=lambda *_a, **_k: out)

    try:
        got = call(4)
    except fmod.LaneDForecastError as e:
        raise AssertionError(f"F2: a (sum of rows, h, 9) reply is refused: {e}") from e
    assert got.shape == (4, HORIZON, len(SCORING_LEVELS))
    with pytest.raises(fmod.LaneDForecastError):
        call(2)


def test_ssh_cut_cuts_every_channel_along_time():
    from cascade.validator.lane_d_inputs import SSHEvalHost

    h = np.arange(3 * 5000.0).reshape(3, 5000)
    got = SSHEvalHost._cut(h, 4096)
    assert got.shape == (3, 4096) and np.array_equal(got, h[:, -4096:])
    assert SSHEvalHost._cut(np.arange(5000.0), 4096).shape == (4096,)


# ── torch: one joint forecast per window, univariate rows unchanged ─────────


@pytest.fixture
def ckpts(tmp_path_factory):
    """The W5d tiny checkpoints (test_w5d_scorer's, built once per session)."""
    import torch

    from .test_w5d_scorer import _checkpoints

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield _checkpoints(str(tmp_path_factory.getbasetemp() / "w5d_ckpts"))
    finally:
        torch.set_num_threads(threads)


@torch_mark
def test_mixed_request_leaves_univariate_rows_bit_identical(ckpts):
    fq = _scorer().forecast_quantiles
    u1, u2 = series(700, seed=1), series(333, seed=2)
    mv = np.stack([series(700, seed=10 + k, level=5.0 + 9 * k) for k in range(3)])
    alone = np.asarray(fq(ckpts["entry"], [u1, u2], context=4096, horizon=HORIZON))
    mixed = np.asarray(fq(ckpts["entry"], [u1, mv, u2], context=4096, horizon=HORIZON))
    assert mixed.shape == (5, HORIZON, len(SCORING_LEVELS)) and np.isfinite(mixed).all()
    assert np.array_equal(mixed[0], alone[0]) and np.array_equal(mixed[4], alone[1])


@torch_mark
def test_multichannel_windows_batch_by_channels_and_length(ckpts):
    fq = _scorer().forecast_quantiles
    a = np.stack([series(400, seed=20 + k) for k in range(3)])
    b = np.stack([series(700, seed=30 + k) for k in range(3)])
    c = np.stack([series(400, seed=40 + k) for k in range(2)])
    got = np.asarray(fq(ckpts["entry"], [a, b, c], context=4096, horizon=HORIZON))
    assert got.shape == (8, HORIZON, len(SCORING_LEVELS))
    for rows, w in ((slice(0, 3), a), (slice(3, 6), b), (slice(6, 8), c)):
        one = np.asarray(fq(ckpts["entry"], [w], context=4096, horizon=HORIZON))
        np.testing.assert_allclose(got[rows], one, rtol=1e-5, atol=1e-6)
