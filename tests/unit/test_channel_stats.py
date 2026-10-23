"""Shadow channel-redundancy telemetry (DEC-CA-0026; measure-first)."""

from __future__ import annotations

import numpy as np

from cascade.trainer.channel_stats import (
    _UNPARTNERED_Z,
    ChannelStatsAccumulator,
    series_channel_stats,
    series_min_partner_z,
)


def test_univariate_series_yield_nothing():
    assert series_channel_stats(np.arange(100.0)) is None
    assert series_channel_stats(np.arange(100.0)[None, :]) is None


def test_jitter_duplicate_signature():
    # The exploit shape: a channel plus a 1e-9-jittered copy → |corr| ≈ 1,
    # effective rank ≈ 1.
    rng = np.random.default_rng(0)
    base = rng.standard_normal(500)
    arr = np.stack([base, base + 1e-9 * rng.standard_normal(500)])
    corr, rank = series_channel_stats(arr)
    assert corr > 0.999
    assert rank < 1.01


def test_independent_channels_read_full_rank():
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((4, 2000))
    corr, rank = series_channel_stats(arr)
    assert corr < 0.2
    assert rank > 3.5


def test_constant_channel_contributes_zero_corr_not_nan():
    arr = np.stack([np.arange(100.0), np.full(100, 5.0)])
    corr, rank = series_channel_stats(arr)
    assert np.isfinite(corr) and np.isfinite(rank)
    assert corr < 1e-6


def test_min_partner_z_needs_two_channels():
    assert series_min_partner_z(np.arange(100.0)) is None
    assert series_min_partner_z(np.arange(100.0)[None, :]) is None


def test_glued_independent_walks_read_unpartnered():
    # The glued-rows shape: two unrelated random walks stacked as (2, L).
    # Their levels can correlate spuriously; their innovations do not.
    rng = np.random.default_rng(3)
    arr = rng.standard_normal((2, 4096)).cumsum(axis=1)
    assert series_min_partner_z(arr) < _UNPARTNERED_Z


def test_lagged_coupling_reads_partnered():
    # A lead/lag pair (lagged causal edge): a lag-0 read would call it
    # unpartnered; the lag scan must not.
    rng = np.random.default_rng(6)
    drive = rng.standard_normal(4096)
    lagged = np.concatenate([rng.standard_normal(30), drive[:-30]])
    arr = np.stack([drive.cumsum(), (lagged + 0.8 * rng.standard_normal(4096)).cumsum()])
    d = np.diff(arr, axis=1)
    assert abs(np.corrcoef(d[0], d[1])[0, 1]) * np.sqrt(d.shape[1] - 1) < 4.0
    assert series_min_partner_z(arr) > 10.0


def test_shared_driver_reads_partnered():
    rng = np.random.default_rng(4)
    driver = rng.standard_normal(4096)
    own = rng.standard_normal((3, 4096))
    arr = (0.5 * driver + own).cumsum(axis=1)
    assert series_min_partner_z(arr) > 10.0


def test_one_glued_channel_is_enough_to_flag():
    # Channels 0 and 1 share a driver; channel 2 is an unrelated row.
    rng = np.random.default_rng(5)
    driver = rng.standard_normal(4096)
    arr = np.stack([
        (driver + 0.3 * rng.standard_normal(4096)).cumsum(),
        (driver + 0.3 * rng.standard_normal(4096)).cumsum(),
        rng.standard_normal(4096).cumsum(),
    ])
    assert series_channel_stats(arr)[0] > 0.9
    assert series_min_partner_z(arr) < _UNPARTNERED_Z


def test_constant_channel_reads_zero_partner_z_not_nan():
    arr = np.stack([np.arange(100.0) ** 1.5, np.full(100, 5.0)])
    z = series_min_partner_z(arr)
    assert np.isfinite(z) and z == 0.0


def test_accumulator_skips_univariate_and_summarises_mv():
    acc = ChannelStatsAccumulator()
    rng = np.random.default_rng(2)
    for _ in range(10):
        acc.observe(rng.standard_normal(100))          # 1-D: ignored
        acc.observe(rng.standard_normal((1, 100)))     # (1, L): ignored
    assert acc.n_observed == 0
    assert acc.summary() is None

    base = rng.standard_normal(300)
    acc.observe(np.stack([base, base + 1e-9]))          # near-duplicate pair
    acc.observe(rng.standard_normal((3, 300)))          # honest independent
    s = acc.summary()
    assert s["n_multichannel_series"] == 2
    assert s["max_channels_seen"] == 3
    assert s["max_abs_corr_max"] > 0.999
    assert s["frac_over_0999"] == 0.5
    assert 1.0 <= s["effective_rank_min"] <= s["effective_rank_p50"]
    # The duplicate pair shares every innovation; the independent triple
    # shares none.
    assert s["frac_unpartnered"] == 0.5
    assert s["min_partner_z_p10"] <= s["min_partner_z_p50"]


def test_trainer_metrics_carry_summary_only_for_mv_corpora():
    # Integration: a univariate run's metrics are unchanged; a multichannel
    # run gains the shadow record. Requires torch (skips without).
    import pytest

    torch = pytest.importorskip("torch")  # noqa: F841
    from types import SimpleNamespace

    from cascade.trainer.toto2_trainer import Toto2Trainer

    contract = SimpleNamespace(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=2, max_train_seconds=30, base_lr=1e-3, weight_decay=0.0,
        optimizer="adamw", warmup_tokens=0, input_transform="arcsinh_causal",
    )
    rng = np.random.default_rng(0)
    trainer = Toto2Trainer(device="cpu", deterministic=False)

    uni = trainer.train(
        iter([rng.normal(size=32).cumsum() for _ in range(4)]), contract,
        training_seed=1, token_budget=10**6, out_dir=__import__("tempfile").mkdtemp(),
    )
    assert "channel_telemetry" not in uni.metrics

    mv = trainer.train(
        iter([rng.normal(size=(2, 32)).cumsum(axis=-1) for _ in range(4)]),
        contract, training_seed=1, token_budget=10**6,
        out_dir=__import__("tempfile").mkdtemp(),
    )
    tele = mv.metrics["channel_telemetry"]
    assert tele["n_multichannel_series"] == 4
    assert tele["max_channels_seen"] == 2
