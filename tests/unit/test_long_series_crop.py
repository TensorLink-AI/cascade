"""DEC-CA-0053 ``long_series``: "last" keeps today's batches byte-identical;
"crop" cuts long series into seeded, stratified context windows."""

from __future__ import annotations

import dataclasses

import numpy as np

from cascade.shared.config import load_chain_config
from cascade.shared.manifest import contract_digest
from cascade.trainer.toto2_trainer import (
    batch_source,
    crop_long_series,
    iter_training_batches,
)

PS, MAXP = 32, 4          # window = 128 steps
WIN = PS * MAXP


def _stream():
    rng = np.random.default_rng(1)
    return [rng.standard_normal((1, n)) for n in (100, 128, 300, 1000, 129)]


def _batches(stream, contract, seed=5):
    src = batch_source(iter(stream), contract, patch_size=PS, max_ctx_patches=MAXP, seed=seed)
    return list(iter_training_batches(src, patch_size=PS, max_ctx_patches=MAXP, batch_size=4))


def test_last_is_the_unwrapped_path():
    c = load_chain_config().training
    assert c.long_series == "last"
    s = iter(_stream())
    assert batch_source(s, c, patch_size=PS, max_ctx_patches=MAXP, seed=0) is s
    ref = list(iter_training_batches(iter(_stream()), patch_size=PS, max_ctx_patches=MAXP,
                                     batch_size=4))
    got = _batches(_stream(), c)
    assert len(ref) == len(got)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got, strict=True))


def test_crop_window_count_and_shape():
    out = list(crop_long_series(iter(_stream()), window=WIN, seed=3))
    # 100 and 128 pass through; 300 → 3, 1000 → 8, 129 → 2 windows
    assert [a.shape[-1] for a in out[:2]] == [100, 128]
    assert len(out) == 2 + 3 + 8 + 2
    assert all(a.shape[-1] == WIN for a in out[2:])


def test_crop_windows_come_from_the_series_and_cover_it():
    s = np.arange(1000, dtype=np.float64)[None, :]
    wins = list(crop_long_series(iter([s]), window=WIN, seed=11))
    starts = [int(w[0, 0]) for w in wins]
    assert starts == sorted(starts)
    assert starts[0] < (1000 - WIN) / 8 + 1 and starts[-1] >= (1000 - WIN) * 7 / 8 - 1
    for w, o in zip(wins, starts, strict=True):
        assert np.array_equal(w, s[:, o:o + WIN])


def test_crop_is_deterministic_and_seeded():
    a = list(crop_long_series(iter(_stream()), window=WIN, seed=3))
    b = list(crop_long_series(iter(_stream()), window=WIN, seed=3))
    c = list(crop_long_series(iter(_stream()), window=WIN, seed=4))
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert any(not np.array_equal(x, y) for x, y in zip(a, c, strict=True))


def test_crop_keeps_mask_aligned():
    vals = np.arange(600, dtype=np.float64)[None, :]
    mask = (np.arange(600) % 2).astype(np.uint8)[None, :]
    rec = {"values": vals, "mask": mask, "roles": np.zeros(1, np.uint8)}
    for w in crop_long_series(iter([rec]), window=WIN, seed=2):
        o = int(w["values"][0, 0])
        assert np.array_equal(w["mask"], mask[:, o:o + WIN])
        assert w["roles"] is rec["roles"]


def test_crop_contract_routes_through_crop_and_bumps_digest():
    c = load_chain_config().training
    crop = dataclasses.replace(c, long_series="crop")
    assert contract_digest(crop.primary_size) != contract_digest(c.primary_size)
    got = _batches(_stream(), crop)
    n_rows = sum(b.shape[0] for b in got)
    assert n_rows == 1 + 1 + 3 + 8 + 2      # the 100-step series keeps 3 patches
