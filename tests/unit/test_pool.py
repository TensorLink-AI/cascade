"""Eval-window pool loader — fetch (mocked registry) + slice into windows, and
the ref/empty-pool guards. The seeded rotation itself is covered by test_windows."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from metronome.validator import pool as pool_mod
from metronome.validator.pool import PoolError, load_pool

REF = "metronome/eval-pool@sha256:" + "a" * 64


def _cfg_with_pool(cfg, window_pool):
    return replace(cfg, eval=replace(cfg.eval, window_pool=window_pool))


def test_load_pool_rejects_non_ref(cfg):
    with pytest.raises(PoolError):
        load_pool(_cfg_with_pool(cfg, "tensorlink-ai/not-a-ref"))


def test_load_pool_fetches_slices_and_builds_source(cfg, tmp_path, monkeypatch):
    horizon = cfg.eval.horizon
    n_series = 5
    length = cfg.eval.context_length + horizon + 10

    def fake_fetch(ref, dest, hub=None):
        from pathlib import Path

        d = Path(dest)
        d.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for i in range(n_series):
            np.save(d / f"s{i}.npy", rng.standard_normal(length))
        return d

    monkeypatch.setattr(pool_mod, "fetch_from_hub", fake_fetch)

    cfg2 = _cfg_with_pool(cfg, REF)
    source = load_pool(cfg2, cache_dir=tmp_path)
    assert len(source.pool) == n_series
    w = source.pool[0]
    assert w.history.shape[-1] == cfg.eval.context_length
    assert w.target.shape[-1] == horizon

    # deterministic rotating slice (same as RotatingWindowSource contract)
    a = source.windows_for_round(123, n_windows=3)
    b = source.windows_for_round(123, n_windows=3)
    assert [x.series_id for x in a] == [x.series_id for x in b]
    assert len(a) == 3


def test_load_pool_raises_when_no_series(cfg, tmp_path, monkeypatch):
    def fake_fetch(ref, dest, hub=None):
        from pathlib import Path

        Path(dest).mkdir(parents=True, exist_ok=True)  # empty
        return Path(dest)

    monkeypatch.setattr(pool_mod, "fetch_from_hub", fake_fetch)
    with pytest.raises(PoolError):
        load_pool(_cfg_with_pool(cfg, REF), cache_dir=tmp_path)
