"""W6b: the `pool_d/1` snapshot (history to 16384 + 64, per-series
timestamps, `frontier`, `feeds` with the effective `embargo_hours`), built
from the W6a selection and the W6fs first-seen index (plan §3.0 W6b row).

On main each test fails at its first import of `cascade.pool.pool_d_build`
(in PATHS), which `helpers.mods()` performs before any W6a / W6fs import.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import types
from pathlib import Path

import numpy as np
import pytest

from .helpers import (
    AS1,
    AS2,
    CTX_A,
    FIVE_MIN,
    GRID_EXTRA,
    LANE_A_CFG,
    N_AIR,
    N_GRID,
    T0,
    build_world,
    by_id,
    canonical,
    forge_source,
    grid_ts,
    grid_values,
    meta,
    mods,
    no_fetch,
    tree_sha,
)

GRID_R1 = "tsforge__grid_load__region_r1~grid_load.c0"
GRID_R2 = "tsforge__grid_load__region_r2~grid_load.c0"
AIR = "tsforge__air_temp~air_temp.c0"


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    return build_world(tmp_path_factory.mktemp("w6b"))


def test_constants():
    b, _sel, _fs = mods()
    assert b.POOL_D_SCHEMA == "pool_d/1"
    assert b.BUILDER_VERSION == "cascade-pool-d/1"
    assert b.HISTORY_LENGTH == 16384
    assert b.DEFAULT_EMBARGO_HOURS == 48.0
    cfg = b.PoolDBuildConfig()
    assert (cfg.history_length, cfg.horizon, cfg.embargo_hours) == (16384, 64, 48.0)


def test_history_extended_to_16384_plus_64(world):
    b, _sel, _fs = mods()
    m = meta(world["d1"])
    assert m["schema"] == "pool_d/1" and m["horizon"] == 64 and m["history_length"] == 16384
    with np.load(world["d1"] / "hist.npz") as hist, np.load(world["d1"] / "tgt.npz") as tgt:
        for sid in (GRID_R1, GRID_R2):
            assert hist[sid].shape == (16384,) and tgt[sid].shape == (64,)
            assert hist[sid].dtype == np.float32 and tgt[sid].dtype == np.float32
        raw = grid_values("r1")[:N_GRID]
        np.testing.assert_array_equal(tgt[GRID_R1], raw[-64:].astype(np.float32))
        np.testing.assert_array_equal(hist[GRID_R1], raw[-64 - 16384:-64].astype(np.float32))
    assert by_id(m)[GRID_R1]["history_len"] == 16384
    # lane A's pool keeps only context_length + horizon of the same series
    lane_a = np.load(world["pool"] / "tsforge__grid_load__region_r1.npy")
    assert lane_a.shape == (LANE_A_CFG.keep_length,)


def test_short_series_keeps_what_it_has(world):
    m = meta(world["d1"])
    assert by_id(m)[AIR]["history_len"] == N_AIR - 64
    with np.load(world["d1"] / "hist.npz") as hist:
        assert hist[AIR].shape == (N_AIR - 64,)


def test_target_timestamps_and_first_seen(world):
    m = meta(world["d1"])
    s = by_id(m)[GRID_R1]
    assert s["target_ts"] == grid_ts()[-64:]
    assert s["first_target_ts"] == s["target_ts"][0]
    assert s["end_ts"] == s["target_ts"][-1] == T0 - FIVE_MIN
    assert s["first_seen_ts"] == [T0 + 600] * 64
    assert s["feed"] == "grid_load" and s["panel"] == {"region": "r1"}
    assert s["channel"] == 0 and s["column"] == "mw"
    assert "fixture:grid" in s["source"]                    # lane A cluster key
    assert s["pool_series_id"] == "tsforge__grid_load__region_r1"
    assert s["freq"] == "5min" and s["seasonal_period"] == 288 and s["domain"] == "energy"
    for s in m["series"]:
        assert len(s["target_ts"]) == len(s["first_seen_ts"]) == 64
        assert s["target_ts"] == sorted(set(s["target_ts"]))


def test_points_after_as_of_are_not_used(world):
    """First-seen never later than the snapshot: the AS1 rebuild after the
    second scrape is byte-identical to the AS1 build (those points were
    first seen after AS1); the AS2 build uses them."""
    assert tree_sha(world["d1"]) == tree_sha(world["d1b"])
    for d, as_of in ((world["d1"], AS1), (world["d2"], AS2)):
        m = meta(d)
        assert m["as_of_ts"] == as_of
        for s in m["series"]:
            assert max(s["first_seen_ts"]) < as_of
    s2 = by_id(meta(world["d2"]))[GRID_R1]
    assert s2["end_ts"] == T0 + FIVE_MIN * (GRID_EXTRA - 1)
    assert s2["first_seen_ts"][-1] == AS1 + 600


def test_frontier_every_series_and_prior_snapshots(world):
    m1, m2 = meta(world["d1"]), meta(world["d2"])
    for s in m1["series"]:
        assert s["frontier"] == []                           # present, possibly empty
    prior = by_id(m1)
    for s in m2["series"]:
        assert isinstance(s["frontier"], list)
        if s["series_id"] in prior:
            assert s["frontier"] == [[AS1, prior[s["series_id"]]["end_ts"]]]
    assert by_id(m2)[GRID_R1]["frontier"] == [[AS1, T0 - FIVE_MIN]]


def test_feeds_entry_for_every_feed(world):
    for d in (world["d1"], world["d2"]):
        m = meta(d)
        feeds = m["feeds"]
        assert {s["feed"] for s in m["series"]} <= set(feeds)
        for name, f in feeds.items():
            lag = f["mirror_lag_hours"]
            assert lag is None or (isinstance(lag, (int, float)) and not isinstance(lag, bool))
            assert isinstance(f["forecast_product"], bool), name
            assert isinstance(f["publish_ahead"], bool), name
            assert isinstance(f["embargo_hours"], (int, float)), name
        assert feeds["fx_fwd"]["forecast_product"] is True
        assert feeds["grid_load"]["forecast_product"] is False


def test_validate_refuses_missing_frontier_and_feed(world, tmp_path):
    """RV3 BLOCKING 1: a pool_d in which any series lacks `frontier` (absent
    or null) or any feed lacks a `feeds` entry is invalid."""
    b, _sel, _fs = mods()
    assert b.validate_pool_d(world["d1"]) == []
    assert b.validate_pool_d(world["d2"]) == []

    def mutated(name, edit):
        d = tmp_path / name
        d.mkdir()
        for f in ("hist.npz", "tgt.npz"):
            (d / f).write_bytes((world["d1"] / f).read_bytes())
        m = meta(world["d1"])
        edit(m)
        (d / "metadata.json").write_text(json.dumps(m))
        return b.validate_pool_d(d)

    assert mutated("no_frontier", lambda m: m["series"][0].pop("frontier"))
    assert mutated("null_frontier", lambda m: m["series"][0].__setitem__("frontier", None))
    assert mutated("no_feed", lambda m: m["feeds"].pop("grid_load"))
    assert mutated("no_embargo", lambda m: m["feeds"]["grid_load"].pop("embargo_hours"))
    assert mutated("no_lag_key", lambda m: m["feeds"]["grid_load"].pop("mirror_lag_hours"))
    assert mutated("above_global", lambda m: m["feeds"]["grid_load"].__setitem__(
        "embargo_hours", 72.0))
    assert mutated("ahead_nonzero", lambda m: m["feeds"]["ahead"].__setitem__(
        "embargo_hours", 6.0))
    assert mutated("short_target", lambda m: m["series"][0].__setitem__(
        "target_ts", m["series"][0]["target_ts"][:-1]))
    assert mutated("bad_end", lambda m: m["series"][0].__setitem__("end_ts", 1))
    assert mutated("seen_late", lambda m: m["series"][0].__setitem__(
        "first_seen_ts", [AS1 + 1] * 64))


def test_effective_embargo_hours_rule():
    b, _sel, _fs = mods()
    eff = b.effective_embargo_hours
    assert eff({"embargo_hours": 12}, global_hours=48.0, publish_ahead=False) == 12.0
    assert eff({"embargo_hours": 100}, global_hours=48.0, publish_ahead=False) == 48.0
    assert eff({}, global_hours=48.0, publish_ahead=False) == 48.0
    assert eff({"embargo_hours": 0}, global_hours=48.0, publish_ahead=False) == 0.0
    assert eff({"embargo_hours": 12}, global_hours=48.0, publish_ahead=True) == 0.0
    assert eff({}, global_hours=24.0, publish_ahead=False) == 24.0
    for bad in (-1, "abc", True, float("nan"), float("inf"), None, [6]):
        assert eff({"embargo_hours": bad}, global_hours=48.0, publish_ahead=False) is None, bad


def test_feed_embargo_hours_pinned(world):
    feeds = meta(world["d1"])["feeds"]
    assert feeds["air_temp"]["embargo_hours"] == 12.0       # catalog override (shortens)
    assert feeds["capped"]["embargo_hours"] == 48.0         # clamped to [lane_d]
    assert feeds["fx_fwd"]["embargo_hours"] == 48.0         # no catalog value: global
    assert feeds["ahead"]["publish_ahead"] is True
    assert feeds["ahead"]["embargo_hours"] == 0.0           # W6fs publish-ahead
    assert feeds["ahead"]["max_lead_hours"] > 0
    assert meta(world["d1"])["embargo_hours_global"] == 48.0


def test_bad_catalog_embargo_fails_closed(world):
    m = meta(world["d1"])
    assert "bad_emb" not in m["feeds"]
    assert not [s for s in m["series"] if s["feed"] == "bad_emb"]
    assert dict(map(tuple, m["refused"]))["tsforge__bad_emb~bad_emb.c0"] == "bad_embargo"
    assert world["s1"].refused["tsforge__bad_emb~bad_emb.c0"] == "bad_embargo"


def test_no_sub_hourly_zero(world):
    """rev6.12: the pinned value applies to every series of the feed, 5-min
    included; no freq-based zero is written."""
    m = meta(world["d1"])
    assert m["feeds"]["grid_load"]["embargo_hours"] == 48.0
    assert by_id(m)[GRID_R1]["freq"] == "5min"
    assert "embargo_hours" not in by_id(m)[GRID_R1]


def test_global_embargo_from_lane_d_loader():
    b, _sel, _fs = mods()
    from cascade.shared.config.lane_d import parse_lane_d

    parsed = parse_lane_d({})
    assert b.lane_d_embargo_hours(parsed) == getattr(parsed, "embargo_hours", 48.0) == 48.0
    assert b.lane_d_embargo_hours(types.SimpleNamespace(embargo_hours=24.0)) == 24.0
    with pytest.raises(ValueError):
        b.lane_d_embargo_hours(types.SimpleNamespace(embargo_hours=-1.0))
    cfg = b.PoolDBuildConfig.from_lane_d(types.SimpleNamespace(embargo_hours=24.0, horizon=32))
    assert (cfg.embargo_hours, cfg.horizon, cfg.history_length) == (24.0, 32, 16384)


def test_lane_a_pool_bytes_unchanged(world, tmp_path):
    """`pool` bytes unchanged: lane A built alone equals lane A built in the
    pool_d run, and the pool_d build writes nothing outside its out dir."""
    from cascade.pool import builder

    builder.build_pool([forge_source(world["root"])], tmp_path / "alone", CTX_A, LANE_A_CFG,
                       fetch=no_fetch)
    assert tree_sha(tmp_path / "alone") == tree_sha(world["pool"])
    b, _sel, _fs = mods()
    before = tree_sha(world["root"])
    b.build_pool_d(forge_dir=world["root"], selection=world["selection"],
                   first_seen_dir=world["root"] / "first_seen", as_of_ts=AS2,
                   out_dir=tmp_path / "again", prior_dirs=[world["d1"]])
    assert tree_sha(world["root"]) == before
    assert tree_sha(tmp_path / "again") == tree_sha(world["d2"])


def test_build_is_deterministic(world, tmp_path, monkeypatch):
    """Same inputs, same bytes, whatever the wall clock (content-addressed)."""
    b, _sel, _fs = mods()
    monkeypatch.setattr(time, "time", lambda: 2_000_000_000.0)
    b.build_pool_d(forge_dir=world["root"], selection=world["selection"],
                   first_seen_dir=world["root"] / "first_seen", as_of_ts=AS1,
                   out_dir=tmp_path / "later")
    assert sorted(p.name for p in (tmp_path / "later").iterdir()) == [
        "hist.npz", "metadata.json", "tgt.npz"]
    assert tree_sha(tmp_path / "later") == tree_sha(world["d1"])
    assert b.snapshot_sha256(tmp_path / "later") == world["s1"].sha256


def test_loads_with_lane_d_loader(world):
    b, _sel, _fs = mods()
    from cascade.validator.lane_d import load_pool

    pool = load_pool(world["d1"], 64)
    assert pool["sha256"] == world["s1"].sha256 == b.snapshot_sha256(world["d1"])
    assert pool["snapshot"] == f"pool_d-{AS1}" == world["s1"].snapshot
    ids = {w["series_id"] for w in pool["windows"]}
    assert ids == set(by_id(meta(world["d1"])))
    assert world["s1"].n_series == len(ids)
    assert world["s1"].n_feeds == len(meta(world["d1"])["feeds"])
    w = next(w for w in pool["windows"] if w["series_id"] == GRID_R1)
    assert w["window_id"] == f"{GRID_R1}:{grid_ts()[-64]}:h64"
    assert w["frontier"] == [] and w["feed"] == "grid_load"
    assert set(pool["first_seen_roots"]) == set(meta(world["d1"])["first_seen_roots"])
    m = meta(world["d1"])
    cat = (world["root"] / "sources.yaml").read_bytes()
    assert m["catalog_sha256"] == hashlib.sha256(cat).hexdigest()
    assert m["selection_sha256"] == world["selection"].sha256()
    assert m["builder_version"] == "cascade-pool-d/1"


def test_first_seen_roots_and_verify(world):
    _b, _sel, fs = mods()
    for d in (world["d1"], world["d2"]):
        m = meta(d)
        assert m["first_seen_roots"] == sorted(set(m["first_seen_roots"]))
        committed = {e["root"] for e in fs.FirstSeenLog(world["root"] / "first_seen")
                     .roots().values() if e["committed_ts"] <= m["as_of_ts"]}
        assert set(m["first_seen_roots"]) <= committed and m["first_seen_roots"]
        assert fs.verify_snapshot(m, world["root"] / "first_seen") == []
    assert len(meta(world["d2"])["first_seen_roots"]) == 2      # both periods used


def test_rollup_and_target_gap_refused(world):
    m = meta(world["d1"])
    refused = dict(map(tuple, m["refused"]))
    assert refused["tsforge__roll.rollup_D~roll.c0"] == "rollup_not_supported"
    assert refused["tsforge__gappy~gappy.c0"] == "target_gap"
    assert m["refused"] == sorted(m["refused"])
    assert "tsforge__roll~roll.c0" in by_id(m)
    for sid in refused:
        assert sid not in by_id(m)
    with np.load(world["d1"] / "tgt.npz") as tgt:
        assert set(tgt.files) == set(by_id(m))
        for sid in tgt.files:
            assert np.isfinite(tgt[sid]).all()


# ── CLI: the owner's builder run ────────────────────────────────────────────

def _lane_a_snapshot(world, tmp_path):
    from cascade.pool import builder

    out = tmp_path / "lane_a"
    builder.build_pool([forge_source(world["root"])], out, CTX_A, LANE_A_CFG, fetch=no_fetch)
    return out


def _cli(world, tmp_path, lane_a, out_name="cli"):
    b, _sel, _fs = mods()
    ev = tmp_path / f"{out_name}.json"
    rc = b.main(["--forge-dir", str(world["root"]),
                 "--first-seen-dir", str(world["root"] / "first_seen"),
                 "--lane-a-pool", str(lane_a), "--out", str(tmp_path / out_name),
                 "--as-of-ts", str(AS2), "--prior-dir", str(world["d1"]),
                 "--max-panel-series-per-feed", "5", "--evidence-out", str(ev)])
    return rc, ev


def test_cli_builds_with_lane_a_replay(world, tmp_path):
    rc, ev = _cli(world, tmp_path, _lane_a_snapshot(world, tmp_path))
    assert rc == 0
    assert tree_sha(tmp_path / "cli") == tree_sha(world["d2"])
    doc = json.loads(ev.read_text())
    _check_evidence(doc)
    assert doc["sha256"] == world["s2"].sha256
    lines = sorted(f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}"
                   for p in sorted((tmp_path / "lane_a").iterdir())
                   if p.name == "metadata.json" or p.suffix == ".npy")
    assert doc["lane_a_pool_sha256"] == hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_cli_refuses_lane_a_mismatch(world, tmp_path, capsys):
    lane_a = _lane_a_snapshot(world, tmp_path)
    victim = lane_a / "tsforge__air_temp.npy"
    arr = np.load(victim)
    arr[-1] += 1.0
    np.save(victim, arr, allow_pickle=False)
    rc, ev = _cli(world, tmp_path, lane_a, out_name="refused")
    assert rc == 3
    captured = capsys.readouterr()
    assert "lane_a_replay_mismatch" in captured.out + captured.err
    assert not (tmp_path / "refused" / "metadata.json").exists()
    assert not ev.exists()


# ── evidence (the owner's builder run; skipped without $LANED_EVIDENCE) ──────

def _check_evidence(doc):
    assert doc["schema"] == "laned-evidence/w6b/1"
    assert doc["wp"] == "W6b" and doc["source"] == "builder"
    assert doc["generated_by"] == "cascade.pool.pool_d_build"
    body = {k: v for k, v in doc.items() if k != "digest"}
    assert doc["digest"] == hashlib.sha256(canonical(body)).hexdigest(), \
        "evidence edited after it was generated"
    assert doc["problems"] == []
    assert doc["lane_a_unchanged"] is True
    assert doc["history_length"] == 16384
    assert 0 < doc["max_history_len"] <= 16384
    assert isinstance(doc["n_series"], int) and doc["n_series"] > 0
    assert isinstance(doc["n_feeds"], int) and doc["n_feeds"] > 0
    for key in ("sha256", "lane_a_pool_sha256", "catalog_sha256"):
        assert len(doc[key]) == 64 and set(doc[key]) <= set("0123456789abcdef"), key
    assert doc["snapshot"] == f"pool_d-{doc['as_of_ts']}"
    assert isinstance(doc["refused"], dict)
    assert all(isinstance(v, int) for v in doc["refused"].values())
    assert not math.isnan(float(doc["as_of_ts"]))


def _evidence_doc():
    path = os.environ.get("LANED_EVIDENCE")
    if not path:
        pytest.skip("LANED_EVIDENCE unset: the owner's builder run writes evidence/w6b.json")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_evidence_from_builder_run():
    mods()
    _check_evidence(_evidence_doc())


def test_evidence_snapshot_on_3h_boundary():
    mods()
    doc = _evidence_doc()
    assert doc["as_of_ts"] % 10800 == 0
    assert doc["max_history_len"] >= 64
