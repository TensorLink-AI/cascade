"""Eval-pool builder — cleaning/validation rules, determinism, and a round-trip
through the *actual* validator loader path (no IPFS, no network)."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from metronome.eval.window import EvalWindow
from metronome.pool.builder import (
    PoolBuildConfig,
    build_pool,
    collect_records,
    prepare_series,
    write_pool,
)
from metronome.pool.source import HarvestContext, HarvestedSeries
from metronome.validator.pool import _load_series_dir
from metronome.validator.windows import build_windows_from_series

CFG = PoolBuildConfig(context_length=512, horizon=16, min_context=64)
CTX = HarvestContext(as_of=dt.date(2026, 6, 1), context_length=512, horizon=16, max_series=1000)


def _series(series_id="s", n=600, freq="H", domain="weather", seasonal=24, base=None):
    if base is None:
        base = 10 + np.sin(np.arange(n) / 5.0) + np.linspace(0, 1, n)
    return HarvestedSeries(series_id, np.asarray(base, dtype=float), freq, domain, seasonal)


class _ListSource:
    name = "list"

    def __init__(self, items):
        self.items = items

    def harvest(self, fetch, ctx):
        yield from self.items


# ── cleaning / validation ───────────────────────────────────────────────────


def test_prepare_interpolates_gaps_and_keeps_tail():
    vals = 10 + np.sin(np.arange(600) / 5.0)
    vals[5] = np.nan
    vals[100] = np.inf
    rec, reason = prepare_series(_series(base=vals), CFG)
    assert reason is None and rec is not None
    assert np.isfinite(rec.values).all()
    # truncated to the freshest context_length + horizon points
    assert rec.values.shape[-1] == CFG.keep_length
    assert rec.values.dtype == np.float32
    assert rec.metadata == {"freq": "H", "seasonal_period": 24, "domain": "weather"}


def test_prepare_drops_too_short():
    rec, reason = prepare_series(_series(n=40), CFG)  # < horizon + min_context
    assert rec is None and reason == "too_short"


def test_prepare_drops_too_much_missing():
    vals = 10 + np.sin(np.arange(600) / 5.0)
    vals[: int(0.5 * 600)] = np.nan
    rec, reason = prepare_series(_series(base=vals), CFG)
    assert rec is None and reason == "too_much_missing"


def test_prepare_drops_constant():
    rec, reason = prepare_series(_series(base=np.full(600, 7.0)), CFG)
    assert rec is None and reason == "degenerate"


def test_prepare_drops_too_many_channels():
    twoch = np.stack([np.arange(600.0), np.arange(600.0)])
    rec, reason = prepare_series(_series(base=twoch), CFG)
    assert rec is None and reason == "too_many_channels"


def test_seasonal_period_derived_from_freq_when_absent():
    # seasonal_period=None ⇒ derived from freq via gluonts mapping (H → 24).
    hs = HarvestedSeries("h", 10 + np.sin(np.arange(600) / 5.0), "H", "weather", None)
    rec, reason = prepare_series(hs, CFG)
    assert reason is None and rec.metadata["seasonal_period"] == 24


# ── collection: dedup, caps, id-uniqueness ──────────────────────────────────


def test_collect_dedups_identical_series():
    s = _series("a")
    dup = HarvestedSeries("b", s.values.copy(), "H", "weather", 24)  # same bytes, different id
    records, drops = collect_records([_ListSource([s, dup])], CTX, CFG, fetch=None)
    assert len(records) == 1 and drops["duplicate"] == 1


def test_collect_disambiguates_colliding_ids():
    a = _series("dup", base=10 + np.sin(np.arange(600) / 5.0))
    b = _series("dup", base=10 + np.cos(np.arange(600) / 5.0))  # distinct content
    records, _ = collect_records([_ListSource([a, b])], CTX, CFG, fetch=None)
    ids = sorted(r.series_id for r in records)
    assert ids == ["dup", "dup-2"]


def test_collect_per_domain_cap():
    items = [_series(f"s{i}", base=10 + np.sin(np.arange(600) / (5.0 + i))) for i in range(5)]
    cfg = PoolBuildConfig(context_length=512, horizon=16, min_context=64, max_series_per_domain=2)
    records, drops = collect_records([_ListSource(items)], CTX, cfg, fetch=None)
    assert len(records) == 2 and drops["domain_cap"] == 3


# ── write + round-trip through the validator loader ─────────────────────────


def test_build_pool_round_trips_through_validator_loader(tmp_path):
    items = [
        _series(f"openmeteo__city{i}__temp", base=10 + i + np.sin(np.arange(600) / (5.0 + i)))
        for i in range(6)
    ]
    out = tmp_path / "pool"
    summary = build_pool([_ListSource(items)], out, CTX, CFG, fetch=None)
    assert summary.n_series == 6

    # The exact path metronome.validator.pool.load_pool runs after fetching a CID:
    series, ids = _load_series_dir(out)
    assert len(series) == 6
    md_map = json.loads((out / "metadata.json").read_text())
    # every loaded id has metadata (ids are the .npy stems)
    assert all(sid in md_map for sid in ids)
    metadata = [md_map[sid] for sid in ids]
    windows = build_windows_from_series(
        series, context_length=CFG.context_length, horizon=CFG.horizon, metadata=metadata, id_prefix=""
    )
    assert len(windows) == 6
    w = windows[0]
    assert isinstance(w, EvalWindow)
    assert w.history.shape[-1] == CFG.context_length and w.target.shape[-1] == CFG.horizon
    assert w.metadata["seasonal_period"] == 24


def test_build_is_deterministic(tmp_path):
    items = [_series(f"s{i}", base=10 + np.sin(np.arange(600) / (5.0 + i))) for i in range(4)]
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_pool([_ListSource(items)], a, CTX, CFG, fetch=None)
    build_pool([_ListSource(list(items))], b, CTX, CFG, fetch=None)
    names_a = sorted(p.name for p in a.glob("*.npy"))
    names_b = sorted(p.name for p in b.glob("*.npy"))
    assert names_a == names_b
    for name in names_a:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_write_refuses_empty_and_existing(tmp_path):
    with pytest.raises(ValueError):
        write_pool([], tmp_path / "empty", as_of="2026-06-01", cfg=CFG)

    items = [_series("s0")]
    out = tmp_path / "p"
    build_pool([_ListSource(items)], out, CTX, CFG, fetch=None)
    with pytest.raises(FileExistsError):
        build_pool([_ListSource(items)], out, CTX, CFG, fetch=None)
    # overwrite succeeds
    build_pool([_ListSource(items)], out, CTX, CFG, fetch=None, overwrite=True)
