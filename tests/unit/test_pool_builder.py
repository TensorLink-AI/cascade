"""Eval-pool builder — cleaning/validation rules, determinism, and a round-trip
through the *actual* validator loader path (no IPFS, no network)."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest

from cascade.eval.window import EvalWindow
from cascade.pool.builder import (
    PoolBuildConfig,
    build_pool,
    collect_records,
    prepare_series,
    write_pool,
)
from cascade.pool.source import HarvestContext, HarvestedSeries
from cascade.validator.pool import _load_series_dir
from cascade.validator.windows import build_windows_from_series

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


def test_collect_per_domain_freq_cell_cap():
    """The cell cap is keyed on (domain, freq): an hourly flood hits the cap
    while the same domain's daily series pass untouched."""
    hourly = [
        _series(f"h{i}", base=10 + np.sin(np.arange(600) / (5.0 + i))) for i in range(4)
    ]
    daily = [
        _series(f"d{i}", freq="D", seasonal=7, base=20 + np.cos(np.arange(600) / (7.0 + i)))
        for i in range(2)
    ]
    cfg = PoolBuildConfig(
        context_length=512, horizon=16, min_context=64, max_series_per_domain_freq=2
    )
    records, drops = collect_records([_ListSource(hourly + daily)], CTX, cfg, fetch=None)
    kept = sorted(r.series_id for r in records)
    assert kept == ["d0", "d1", "h0", "h1"]  # 2 per cell; daily cell unaffected
    assert drops["domain_freq_cap"] == 2


# ── write + round-trip through the validator loader ─────────────────────────


def test_build_pool_round_trips_through_validator_loader(tmp_path):
    items = [
        _series(f"openmeteo__city{i}__temp", base=10 + i + np.sin(np.arange(600) / (5.0 + i)))
        for i in range(6)
    ]
    out = tmp_path / "pool"
    summary = build_pool([_ListSource(items)], out, CTX, CFG, fetch=None)
    assert summary.n_series == 6

    # The exact path cascade.validator.pool.load_pool runs after fetching a CID:
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


def test_source_label_lands_in_metadata():
    cfg = PoolBuildConfig(context_length=512, horizon=16)
    labeled = HarvestedSeries(
        "s1", 10 + np.sin(np.arange(600) / 5.0), "H", "energy", 24, source="grid_load"
    )
    rec, reason = prepare_series(labeled, cfg)
    assert reason is None and rec.metadata["source"] == "grid_load"
    # Legacy sources without a label stay unchanged: no source key at all.
    rec2, _ = prepare_series(_series("s2"), cfg)
    assert "source" not in rec2.metadata


def test_require_source_rejects_unlabeled_series():
    # DEC-CA-0026 item 3: the strict knob for pools that will carry
    # multichannel windows — every kept series must name its upstream feed.
    cfg = PoolBuildConfig(context_length=512, horizon=16, require_source=True)
    labeled = HarvestedSeries(
        "s1", 10 + np.sin(np.arange(600) / 5.0), "H", "energy", 24, source="grid_load"
    )
    rec, reason = prepare_series(labeled, cfg)
    assert reason is None and rec.metadata["source"] == "grid_load"
    rec2, reason2 = prepare_series(_series("s2"), cfg)
    assert rec2 is None and reason2 == "missing_source"
    # Default stays permissive: same unlabeled series builds fine.
    rec3, reason3 = prepare_series(_series("s3"), PoolBuildConfig(context_length=512, horizon=16))
    assert reason3 is None and "source" not in rec3.metadata


# ── balanced selection (seeded round-robin across sources) ──────────────────


def _src_series(src, i, domain="transport", freq="H"):
    """A distinct series labelled with upstream feed ``src``."""
    base = 10 + np.sin(np.arange(600) / (5.0 + i)) + (hash(src) % 97) * 0.01
    return HarvestedSeries(f"{src}__{i}", base, freq, domain, 24, source=src)


def _starved_catalog():
    """Two big feeds listed first (20 series each), then 10 one-series feeds —
    the shape that let catalog order starve a capped domain."""
    items = [_src_series("big_a", i) for i in range(20)]
    items += [_src_series("big_b", 100 + i) for i in range(20)]
    items += [_src_series(f"small_{k}", 200 + k) for k in range(10)]
    return items


def _bal(**kw):
    return PoolBuildConfig(context_length=512, horizon=16, min_context=64,
                           selection="balanced", selection_seed="block:1", **kw)


def test_first_come_starves_late_sources_under_a_domain_cap():
    cfg = PoolBuildConfig(context_length=512, horizon=16, min_context=64,
                          max_series_per_domain=12)
    records, drops = collect_records([_ListSource(_starved_catalog())], CTX, cfg, fetch=None)
    assert {r.metadata["source"] for r in records} == {"big_a"}   # the historical failure
    assert drops["domain_cap"] == 38


def test_balanced_spreads_a_capped_domain_across_sources():
    records, drops = collect_records([_ListSource(_starved_catalog())], CTX,
                                     _bal(max_series_per_domain=12), fetch=None)
    srcs = [r.metadata["source"] for r in records]
    assert len(records) == 12 and drops["domain_cap"] == 38
    # 12 distinct feeds available ⇒ one series from each, none doubled up.
    assert len(set(srcs)) == 12


def test_balanced_second_pass_only_after_every_source_had_one():
    records, _ = collect_records([_ListSource(_starved_catalog())], CTX,
                                 _bal(max_series_per_domain=16), fetch=None)
    from collections import Counter as C
    per = C(r.metadata["source"] for r in records)
    assert set(per) == {"big_a", "big_b", *(f"small_{k}" for k in range(10))}
    # 12 in pass one, the remaining 4 can only come from the two big feeds.
    assert per["big_a"] + per["big_b"] == 6 and max(per.values()) <= 3


def test_balanced_respects_the_cell_cap_per_domain_freq():
    hourly = [_src_series(f"h{k}", k) for k in range(6)]
    daily = [_src_series(f"d{k}", 50 + k, freq="D") for k in range(3)]
    records, drops = collect_records([_ListSource(hourly + daily)], CTX,
                                     _bal(max_series_per_domain_freq=2), fetch=None)
    freqs = sorted(r.metadata["freq"] for r in records)
    assert freqs == ["D", "D", "H", "H"] and drops["domain_freq_cap"] == 5


def test_balanced_is_deterministic_and_ignores_harvest_order():
    items = _starved_catalog()
    cfg = _bal(max_series_per_domain=12)
    a, _ = collect_records([_ListSource(items)], CTX, cfg, fetch=None)
    b, _ = collect_records([_ListSource(list(reversed(items)))], CTX, cfg, fetch=None)
    assert [r.series_id for r in a] == [r.series_id for r in b]


def test_balanced_seed_rotates_the_sample():
    items = _starved_catalog()
    picks = set()
    for seed in ("block:1", "block:2", "block:3", "block:4"):
        cfg = PoolBuildConfig(context_length=512, horizon=16, min_context=64,
                              max_series_per_domain=12, selection="balanced",
                              selection_seed=seed)
        recs, _ = collect_records([_ListSource(items)], CTX, cfg, fetch=None)
        picks.add(tuple(r.series_id for r in recs))
    assert len(picks) > 1


def test_balanced_without_caps_keeps_everything():
    items = _starved_catalog()
    first, _ = collect_records([_ListSource(items)], CTX,
                               PoolBuildConfig(context_length=512, horizon=16, min_context=64),
                               fetch=None)
    bal, _ = collect_records([_ListSource(items)], CTX, _bal(), fetch=None)
    assert [r.series_id for r in first] == [r.series_id for r in bal]


def test_balanced_still_dedups_and_disambiguates():
    a = HarvestedSeries("dup", 10 + np.sin(np.arange(600) / 5.0), "H", "x", 24, source="f1")
    b = HarvestedSeries("dup", 10 + np.cos(np.arange(600) / 5.0), "H", "x", 24, source="f2")
    c = HarvestedSeries("copy", 10 + np.sin(np.arange(600) / 5.0), "H", "x", 24, source="f3")
    records, drops = collect_records([_ListSource([a, b, c])], CTX, _bal(), fetch=None)
    assert sorted(r.series_id for r in records) == ["dup", "dup-2"]
    assert drops["duplicate"] == 1


def test_unknown_selection_mode_is_rejected():
    cfg = PoolBuildConfig(context_length=512, horizon=16, selection="random")
    with pytest.raises(ValueError, match="selection"):
        collect_records([_ListSource([])], CTX, cfg, fetch=None)


def test_balanced_seed_is_recorded_in_provenance(tmp_path):
    cfg = _bal(max_series_per_domain=12)
    build_pool([_ListSource(_starved_catalog())], tmp_path / "p", CTX, cfg, fetch=None)
    prov = json.loads((tmp_path / "p" / "provenance.json").read_text())
    assert prov["config"]["selection"] == "balanced"
    assert prov["config"]["selection_seed"] == "block:1"
