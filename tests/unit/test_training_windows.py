"""Trainer windowing — the bucketed batch carver (pure numpy, no torch).

Regression for the config-mismatch where full-context windows (context_length)
exceeded the generator's max series length, yielding zero training batches."""

from __future__ import annotations

import numpy as np

from cascade.trainer.toto2_trainer import iter_training_batches


def test_short_series_still_produce_batches():
    rng = np.random.default_rng(0)
    # 2048-long series (< context_length 4096) plus tiny 64-long ones.
    series = [rng.standard_normal(2048) for _ in range(5)] + [rng.standard_normal(64) for _ in range(3)]
    batches = list(iter_training_batches(iter(series), patch_size=32, max_ctx_patches=128, batch_size=4))

    assert batches, "windowing must produce batches from sub-context series"
    for b in batches:
        assert b.ndim == 2
        assert b.shape[1] % 32 == 0
        assert b.shape[1] // 32 >= 2  # at least input+target patch
    widths = {b.shape[1] for b in batches}
    assert 2048 in widths  # 64 patches × 32 (capped by data, not max_ctx_patches)
    assert 64 in widths    # 2 patches × 32


def test_series_below_two_patches_skipped():
    series = [np.ones(32), np.ones(10)]  # 1 patch / 0 patches → both skipped
    out = list(iter_training_batches(iter(series), patch_size=32, max_ctx_patches=128, batch_size=4))
    assert out == []


def test_multivariate_reduced_to_channel_zero():
    s = np.ones((1, 128))
    out = list(iter_training_batches(iter([s]), patch_size=32, max_ctx_patches=128, batch_size=1))
    assert len(out) == 1 and out[0].shape == (1, 128)


def test_patch_count_capped_by_max_ctx_patches():
    # a long series is truncated to max_ctx_patches patches
    s = np.ones(32 * 200)
    out = list(iter_training_batches(iter([s]), patch_size=32, max_ctx_patches=128, batch_size=1))
    assert out[0].shape[1] == 128 * 32
