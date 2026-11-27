"""W-HZ: the band ``pool_d/1`` region, its loading at each window's own
horizon and the forward rule on every target point (owner 2026-11-27,
incl. the embargo correction: a target may be part embargo-held, part
fresh). Contract: README.md."""

from __future__ import annotations

import json

import numpy as np

from .helpers import (
    FIXED_H,
    HOUR,
    MINUTE,
    T0,
    band_cfg,
    band_series,
    feeds_for,
    minute_fleet,
    need,
    write_band_pool,
)

BOUND = T0 + 6 * HOUR


def _load(d, cfg):
    lane_d = need("cascade.validator.lane_d", "load_pool_for")
    return lane_d.load_pool_for(d, cfg)


def test_band_pool_validates_and_keeps_the_pre_band_digest(tmp_path):
    need("cascade.pool.pool_d_build", "CLOCKS_FILE", "region_points")
    from cascade.pool.pool_d_build import snapshot_sha256, validate_pool_d
    from cascade.validator.lane_d import load_pool

    fleet = minute_fleet(bound=BOUND, sources=2, per_source=2)
    d = write_band_pool(tmp_path / "p", fleet, feeds=feeds_for(fleet))
    assert validate_pool_d(d) == []
    fixed = load_pool(d, FIXED_H)
    banded = _load(d, band_cfg())
    # the clocks are bound by metadata, not by the snapshot digest: a pre-band
    # reader pins the same sha256 and reads the same 64-point windows
    assert fixed["sha256"] == banded["sha256"] == snapshot_sha256(d)
    assert [w["window_id"] for w in fixed["windows"]] == [
        f"{s['rec']['series_id']}:{s['rec']['target_ts'][0]}:h64" for s in fleet]
    # a tampered clocks file is refused
    raw = bytearray((d / "clk.npz").read_bytes())
    raw[-9] ^= 1
    (d / "clk.npz").write_bytes(bytes(raw))
    assert validate_pool_d(d)


def test_each_band_window_is_a_region_slice_at_its_band_horizon(tmp_path):
    fleet = minute_fleet(bound=BOUND, sources=1, per_source=1)
    s = fleet[0]
    d = write_band_pool(tmp_path / "p", fleet, feeds=feeds_for(fleet))
    ws = {w["band"]: w for w in _load(d, band_cfg())["windows"]}
    assert {b: w["h"] for b, w in ws.items()} == {"short": 64, "medium": 480, "long": 736}
    full = np.concatenate([s["hist"], s["tgt"]]).astype(np.float64)
    ts = s["clk"][0]
    for w in ws.values():
        h = w["h"]
        assert w["window_id"] == f"{s['rec']['series_id']}:{int(ts[-h])}:h{h}"
        assert np.array_equal(w["target"], full[-h:])
        assert np.array_equal(w["history"], full[:-h])  # history = every point before the cut
        assert list(w["target_ts"]) == list(ts[-h:]) and w["end_ts"] == int(ts[-1])


def test_more_slots_tile_non_overlapping_targets(tmp_path):
    fleet = minute_fleet(bound=BOUND, sources=1, per_source=1, region=736)
    d = write_band_pool(tmp_path / "p", fleet, feeds=feeds_for(fleet))
    ws = [w for w in _load(d, band_cfg(band_slots=4))["windows"] if w["band"] == "short"]
    assert len(ws) == 4
    spans = sorted((int(w["target_ts"][0]), int(w["target_ts"][-1])) for w in ws)
    assert all(a[1] < b[0] for a, b in zip(spans, spans[1:], strict=False))


def test_a_pre_band_snapshot_loads_under_the_band_rule(tmp_path):
    fleet = minute_fleet(bound=BOUND, sources=1, per_source=2)
    d = write_band_pool(tmp_path / "p", fleet, feeds=feeds_for(fleet), clocks=False)
    ws = _load(d, band_cfg())["windows"]
    # its region is its stored 64-point target: only the 64-point short band fits
    assert {(w["band"], w["h"]) for w in ws} == {("short", 64)}


def test_every_target_point_must_be_unseen_part_embargo_part_fresh(tmp_path):
    """Owner rule: every target point is either embargo-held at the entry
    (arrived before it, still held back) or arrived after it; the history may
    hold anything."""
    need("cascade.validator.lane_d_forward", "window_horizon")
    from cascade.validator.lane_d_forward import eligible_windows

    t_entry = T0
    # a minutely series whose region starts 10 h before the entry: the long
    # target (736 min ~ 12.3 h) is part embargo-held (24 h embargo), part fresh
    s = band_series("hz-mix", "srcA", freq="min", step=MINUTE, region=736,
                    start=t_entry - 10 * HOUR, bound=t_entry)
    emb = write_band_pool(tmp_path / "emb", [s], feeds=feeds_for([s], embargo=24.0))
    pub = write_band_pool(tmp_path / "pub", [s], feeds=feeds_for([s], embargo=0.0))
    for cfg_emb in (24.0, 48.0):
        cfg = band_cfg(embargo_hours=cfg_emb)
        pool = _load(emb, cfg)
        got = {w["band"]: v for w, v in zip(pool["windows"], eligible_windows(
            pool, t_entry=t_entry, t_king=t_entry, cfg=cfg), strict=True)}
        assert got["long"].eligible and got["long"].embargo_window, cfg_emb
        assert got["short"].eligible
    # a feed public at the source (embargo 0) gets no embargo credit
    cfg = band_cfg(embargo_hours=24.0)
    pool = _load(pub, cfg)
    got = {w["band"]: v for w, v in zip(pool["windows"], eligible_windows(
        pool, t_entry=t_entry, t_king=t_entry, cfg=cfg), strict=True)}
    assert not got["long"].eligible
    assert got["long"].reason == "before_margin"


def test_a_target_point_seen_before_the_embargo_is_refused(tmp_path):
    from cascade.validator.lane_d_forward import eligible_windows

    t_entry = T0
    s = band_series("hz-old", "srcA", freq="min", step=MINUTE, region=736,
                    start=t_entry - 30 * HOUR, bound=t_entry)
    d = write_band_pool(tmp_path / "p", [s], feeds=feeds_for([s], embargo=24.0))
    cfg = band_cfg(embargo_hours=24.0)
    pool = _load(d, cfg)
    got = {w["band"]: v for w, v in zip(pool["windows"], eligible_windows(
        pool, t_entry=t_entry, t_king=t_entry, cfg=cfg), strict=True)}
    assert not got["long"].eligible and not got["medium"].eligible


def test_builder_region_is_the_longest_band_inside_the_finite_tail():
    b = need("cascade.pool.pool_d_build", "region_points")
    cfg = b.PoolDBuildConfig(horizon_bands=("short", "medium", "long"))
    raw = np.ones(5000)
    assert b.region_points("H", raw, 5000, cfg) == 736
    assert b.region_points("10S", raw, 5000, cfg) == 928
    assert b.region_points("D", raw, 5000, cfg) == 480
    raw[-100] = np.nan                       # never a gap inside a target
    assert b.region_points("H", raw, 5000, cfg) == 99
    assert b.region_points("H", np.ones(600), 600, cfg) == 600 - cfg.min_context
    assert b.PoolDBuildConfig.from_lane_d(band_cfg()).horizon_bands == (
        "short", "medium", "long")


def test_split_and_join_carry_the_clocks(tmp_path):
    need("cascade.pool.pool_d_split", "CLOCKS_FILE")
    from cascade.pool.pool_d_split import join_split, split_pool_d

    fleet = minute_fleet(bound=BOUND, sources=1, per_source=2)
    d = write_band_pool(tmp_path / "p", fleet, feeds=feeds_for(fleet))
    res = split_pool_d(d, tmp_path / "out")
    joined = join_split(res.hist_tar, res.tgt_tar, tmp_path / "joined")
    assert (joined / "clk.npz").read_bytes() == (d / "clk.npz").read_bytes()
    a, b = _load(d, band_cfg()), _load(joined, band_cfg())
    assert [w["window_id"] for w in a["windows"]] == [w["window_id"] for w in b["windows"]]


def test_retention_keeps_a_snapshot_a_frozen_band_window_needs(tmp_path):
    pub = need("cascade.pool.pool_d_publish", "_window_starts")
    fleet = minute_fleet(bound=BOUND, sources=1, per_source=1)
    d = write_band_pool(tmp_path / "pins" / "old", fleet, feeds=feeds_for(fleet))
    w = [x for x in _load(d, band_cfg())["windows"] if x["band"] == "long"][0]
    assert (w["series_id"], int(w["target_ts"][0])) in pub._window_starts(d)
    assert not {w["window_id"]} & pub._window_ids(d)   # the pre-band match misses it
    pins = {"schema": pub.PINS_SCHEMA,
            "pins": [{"snapshot": "old", "dir": "old", "as_of_ts": 1, "sha256": "0" * 64},
                     {"snapshot": "new", "dir": "new", "as_of_ts": 2, "sha256": "1" * 64}]}
    (tmp_path / "pins" / "pins.json").write_text(json.dumps(pins))
    # an open entry revealed long after the old snapshot, its frozen set
    # holding the old snapshot's long-band window
    state = {"entries": {"e": {"status": "pending", "reveal_ts": 10 ** 10,
                               "frozen_set": {"window_ids": [w["window_id"]]}}}}
    plan = pub.retention_plan(tmp_path / "pins" / "pins.json", state, embargo_hours=0)
    assert "old" in plan["keep"] and "old" not in plan["expire"]


def test_inputs_forecast_each_horizon_group_once():
    """The eval host is asked once per (digest, h), each window at its h,
    and only the window id and history leave the box."""
    from types import SimpleNamespace

    inputs = need("cascade.validator.lane_d_inputs")
    assert hasattr(inputs.LaneDInputsSource, "_forecast_group"), \
        "the inputs source has no per-horizon forecast group"
    calls = []

    class Host:
        def forecast(self, dg, *, context, horizon, windows):
            calls.append((dg, horizon, [sorted(w) for w in windows]))
            return np.zeros((len(windows), horizon, 9))

    src = object.__new__(inputs.LaneDInputsSource)
    src.cfg = SimpleNamespace(lane_d=band_cfg())
    src._eval_host, src._score_cache = Host(), None
    src._hello = lambda b: (SimpleNamespace(git_head="", package_hash=""), None)
    wins = [{"window_id": f"w{i}:{i}:h{h}", "history": np.ones(100), "h": h}
            for i, h in enumerate((64, 480, 64, 736))]
    out, _host = src._forecasts({"dg": 4096}, frozenset({"dg"}), wins, {"king": {}},
                                {"digest": "x"}, (4096,), 1)
    assert sorted((c[1], len(c[2])) for c in calls) == [(64, 2), (480, 1), (736, 1)]
    assert all(keys == ["history", "window_id"] for c in calls for keys in c[2])
    assert {k: v.shape for k, v in out["dg"].items()} == {
        "w0:0:h64": (64, 9), "w1:1:h480": (480, 9), "w2:2:h64": (64, 9),
        "w3:3:h736": (736, 9)}
