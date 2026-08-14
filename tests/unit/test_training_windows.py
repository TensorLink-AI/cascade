"""Trainer windowing — the bucketed batch carver (pure numpy, no torch).

Regression for the config-mismatch where full-context windows (context_length)
exceeded the generator's max series length, yielding zero training batches —
and for the DEC-CA-0022 channel-drop trap, where a (C, L) series was billed
C× against the stream budget but reduced to channel 0 before training."""

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
        assert b.ndim == 3           # (B, C, P*patch_size)
        assert b.shape[1] == 1       # univariate stream → single channel
        assert b.shape[2] % 32 == 0
        assert b.shape[2] // 32 >= 2  # at least input+target patch
    widths = {b.shape[2] for b in batches}
    assert 2048 in widths  # 64 patches × 32 (capped by data, not max_ctx_patches)
    assert 64 in widths    # 2 patches × 32


def test_series_below_two_patches_skipped():
    series = [np.ones(32), np.ones(10)]  # 1 patch / 0 patches → both skipped
    out = list(iter_training_batches(iter(series), patch_size=32, max_ctx_patches=128, batch_size=4))
    assert out == []


def test_single_channel_series_keeps_its_bytes():
    s = np.arange(128, dtype=np.float64)[None, :]        # (1, 128)
    out = list(iter_training_batches(iter([s]), patch_size=32, max_ctx_patches=128, batch_size=1))
    assert len(out) == 1 and out[0].shape == (1, 1, 128)
    assert np.array_equal(out[0][0], s)


def test_all_channels_consumed_not_just_channel_zero():
    # The channel-drop regression: every channel of a (C, L) series must reach
    # the batch, byte-for-byte.
    s = np.stack([np.arange(128.0), np.arange(128.0) * -2.0, np.ones(128)])
    out = list(iter_training_batches(iter([s]), patch_size=32, max_ctx_patches=128, batch_size=1))
    assert len(out) == 1 and out[0].shape == (1, 3, 128)
    assert np.array_equal(out[0][0], s)


def test_buckets_key_on_patch_count_and_channels():
    # Same length, different C → different buckets; same (P, C) → same batch.
    rng = np.random.default_rng(1)
    series = [
        rng.standard_normal((1, 128)), rng.standard_normal((3, 128)),
        rng.standard_normal((1, 128)), rng.standard_normal((3, 128)),
    ]
    out = list(iter_training_batches(iter(series), patch_size=32, max_ctx_patches=128, batch_size=2))
    shapes = sorted(b.shape for b in out)
    assert shapes == [(2, 1, 128), (2, 3, 128)]


def test_mixed_channel_corpus_flushes_partial_buckets():
    series = [np.ones((2, 64)), np.ones((1, 64)), np.ones((4, 128))]
    out = list(iter_training_batches(iter(series), patch_size=32, max_ctx_patches=128, batch_size=8))
    shapes = sorted(b.shape for b in out)
    assert shapes == [(1, 1, 64), (1, 2, 64), (1, 4, 128)]


def test_patch_count_capped_by_max_ctx_patches():
    # a long series is truncated to max_ctx_patches patches
    s = np.ones(32 * 200)
    out = list(iter_training_batches(iter([s]), patch_size=32, max_ctx_patches=128, batch_size=1))
    assert out[0].shape[2] == 128 * 32
