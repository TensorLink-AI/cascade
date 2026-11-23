"""scripts/laned_w1_measure.py — lane D W1 snapshot-diff bound, on tiny
synthetic snapshot directories (CPU only, no real data)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "laned_w1_measure", Path(__file__).resolve().parents[2] / "scripts" / "laned_w1_measure.py")
w1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = w1
_SPEC.loader.exec_module(w1)

T0 = w1.parse_time("2026-11-01T02:00:00Z")
DAY = 86400.0


def _walk(n: int, seed: int) -> np.ndarray:
    return np.cumsum(np.random.default_rng(seed).normal(size=n)).astype(np.float32)


def _write(d: Path, series: dict[str, np.ndarray], md: dict[str, dict]) -> Path:
    d.mkdir(parents=True)
    for sid, arr in series.items():
        np.save(d / f"{sid}.npy", arr, allow_pickle=False)
    (d / "metadata.json").write_text(json.dumps(md), encoding="utf-8")
    (d / "provenance.json").write_text(json.dumps(
        {"config": {"context_length": 256, "horizon": 64, "keep_horizon": 64},
         "summary": {"as_of": "2026-11-01"}}), encoding="utf-8")
    return d


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def test_appended_points_detected():
    full = _f64(_walk(400, 1))
    a, b = full[:300], full[60:340]          # head re-truncated by 60, 40 new points
    al = w1.align(a, b)
    assert al["k"] == 40 and al["ov"] == 240 and al["eq"].all()
    assert w1.align(a, a)["k"] == 0
    assert w1.align(a, _f64(_walk(300, 99))) is None      # an unrelated array: unaligned
    st, tr = w1.advance(None, {"s": a}, T0)
    st, tr = w1.advance(st, {"s": b}, T0 + DAY)
    assert tr["appended_series"] == 1 and tr["appended_points"] == 40
    assert tr["retruncated_series"] == 1 and tr["dropped_head_points"] == 60
    _arr, fs, kind = st["s"]
    assert np.all(fs[-40:] == T0 + DAY) and np.all(fs[:-40] == T0)
    assert np.all(kind[-40:] == w1.APPENDED)


def test_revised_values_counted_separately():
    full = _f64(_walk(400, 2))
    a = full[:300].copy()
    b = full[:370].copy()
    b[[100, 150, 299]] += 5.0                 # three revised points inside the overlap
    st, _ = w1.advance(None, {"s": a}, T0)
    st, tr = w1.advance(st, {"s": b}, T0 + DAY)
    assert tr["revised_series"] == 1 and tr["revised_points"] == 3
    assert tr["appended_points"] == 70
    _arr, fs, kind = st["s"]
    assert kind[100] == w1.REVISED and fs[100] == T0 + DAY
    assert kind[101] == w1.CARRIED and fs[101] == T0


def test_series_missing_in_second_snapshot_and_new_series():
    st, _ = w1.advance(None, {"gone": _f64(_walk(300, 3)), "kept": _f64(_walk(300, 4))}, T0)
    st, tr = w1.advance(st, {"kept": _f64(_walk(300, 4)), "new": _f64(_walk(300, 5))}, T0 + DAY)
    assert tr["disappeared"] == 1 and tr["appeared"] == 1 and tr["common"] == 1
    assert "gone" not in st
    assert np.all(st["new"][2] == w1.APPEARED)


def _eligible(st, md, commit, **kw):
    kw = dict(dict(margin_s=6 * 3600.0, horizon=64, min_context=64, min_windows=2,
                   min_clusters=2), **kw)
    return w1.eligibility(st, md, commit, **kw)


def test_eligibility_respects_commit_plus_margin():
    full = _f64(_walk(500, 6))
    short = _f64(_walk(500, 7))
    st, _ = w1.advance(None, {"long": full[:300], "short": short[:300]}, T0)
    t1 = T0 + DAY
    # "long" gains 100 new points (>= 64); "short" only 30 — its target mixes old points
    st, _ = w1.advance(st, {"long": full[:400], "short": short[:330]}, t1)
    md = {"long": {"freq": "H", "source": "x"}, "short": {"freq": "H", "source": "y"}}
    ok = _eligible(st, md, t1 - 7 * 3600)      # cutoff t1 - 1 h < t1 ⇒ eligible
    assert ok["total"]["windows"] == 1 and ok["total"]["by_kind"] == {"appended": 1}
    late = _eligible(st, md, t1 - 5 * 3600)    # cutoff t1 + 1 h ⇒ nothing first appears after
    assert late["total"]["windows"] == 0
    assert ok["label"] == w1.BOUND_LABEL and "measured" not in ok["label"]


def test_floors_reported_per_freq():
    series, md = {}, {}
    for i in range(6):
        series[f"s{i}"] = _f64(_walk(500, 10 + i))
        md[f"s{i}"] = {"freq": "5min" if i < 4 else "H", "source": f"src{i % 3}"}
    st, _ = w1.advance(None, {k: v[:300] for k, v in series.items()}, T0)
    st, _ = w1.advance(st, {k: v[:400] for k, v in series.items()}, T0 + DAY)
    e = _eligible(st, md, T0, min_windows=5, min_clusters=3)
    assert e["total"]["windows"] == 6 and e["total"]["sources"] == 3
    assert e["total"]["floors"]["met"] is True
    assert e["per_freq"]["5min"]["windows"] == 4
    assert e["per_freq"]["5min"]["floors"]["met"] is False
    assert e["per_freq"]["H"]["sources"] == 2
    assert _eligible(st, md, T0)["total"]["floors"]["min_windows"] == 2


def test_long_history_flags_truncation():
    lh = w1.long_history({"a": np.zeros(4816), "b": np.zeros(1000)}, {"a": {"source": "s"}},
                         4096 + 720, 64)
    assert lh[">=4096+64"] == {"series": 1, "sources": 1, "label": "lower bound (pool-truncated)"}
    for L in (8192, 16384, 32768, 65536, 131072):
        assert lh[f">={L}+64"] == w1.NOT_ANSWERABLE
    assert "LOWER bound" in lh["note"]


def test_cli_end_to_end(tmp_path):
    full = {f"s{i}": _walk(600, 20 + i) for i in range(3)}
    md = {k: {"freq": "H", "source": f"src{i}", "domain": "d"} for i, k in enumerate(full)}
    root = tmp_path / "snaps"
    _write(root / "100", {k: v[:300] for k, v in full.items()}, md)
    _write(root / "200", {k: v[:400] for k, v in full.items() if k != "s2"}, md)
    out = tmp_path / "w1.json"
    argv = ["--snapshots-dir", str(root), "--out", str(out),
            "--build-time", "100=2026-11-01T02:00:00Z", "--build-time", "200=2026-11-02T02:00:00Z",
            "--snapshot-sha256", "100=" + "ab" * 32, "--t-step-hours", "12"]
    assert w1.main(argv) == 0
    doc = json.loads(out.read_text())
    for k in ("command_line", "host", "timestamp_utc", "snapshot_sha256", "seeds"):
        assert doc[k]
    assert doc["label"] == w1.BOUND_LABEL
    assert doc["snapshot_sha256"]["100"]["tar"] == "ab" * 32
    assert doc["snapshot_sha256"]["200"]["tar"].startswith("not run:")
    assert doc["snapshot_sha256"]["100"]["tree"] == w1.tree_sha256(root / "100")
    (pair,) = doc["pairs"]
    assert pair["transition"]["disappeared"] == 1 and pair["transition"]["appended_points"] == 200
    first = pair["eligibility"][0]                   # T = first build time
    assert first["total"]["windows"] == 2 and first["total"]["floors"]["met"] is False
    assert doc["snapshots"][0]["long_history"][">=8192+64"] == w1.NOT_ANSWERABLE
    assert doc["snapshots"][0]["keep_length"] == 320


def test_needs_two_snapshots_and_build_times(tmp_path):
    root = tmp_path / "one"
    _write(root / "100", {"s": _walk(300, 1)}, {"s": {}})
    with pytest.raises(SystemExit):
        w1.main(["--snapshots-dir", str(root), "--out", str(tmp_path / "x.json")])
    with pytest.raises(ValueError, match=">= 2"):
        w1.measure([{"build_ts": T0}], commit_times=None, step_h=6, margin_h=6, horizon=64,
                   min_context=64, min_windows=1, min_clusters=1)


# ── forge mirror (measured) ─────────────────────────────────────────────────


def test_freq_buckets():
    assert w1.freq_bucket(10) == "10s" and w1.freq_bucket(30) == "10s"
    assert w1.freq_bucket(60) == "1min" and w1.freq_bucket(150) == "1min"
    assert w1.freq_bucket(300) == "5min" and w1.freq_bucket(600) == "10min"
    assert w1.freq_bucket(900) == "15min" and w1.freq_bucket(1800) == "30min"
    assert w1.freq_bucket(3600) == "hourly" and w1.freq_bucket(8 * 3600) == "hourly"
    assert w1.freq_bucket(86400) == "daily" and w1.freq_bucket(7 * 86400) == "slower"
    assert w1.freq_bucket(None) == "unparseable"


def _series(n, *, t0=T0, step=60.0, lag=600.0):
    ts = t0 + step * np.arange(n, dtype=np.float64)
    return {"data_ts": ts, "fs_lo": ts.copy(), "fs_hi": ts + lag, "length": n,
            "baseline": np.zeros(n, bool)}


def test_windows_fresh_and_embargo_split():
    s = _series(200)
    b = T0 + 60 * 100                         # boundary at point 100
    e = 60 * 20                               # embargo reaches back to point 80
    # clock b + 30 min: available up to point 120 (10-min lag) → eligible 81..120;
    # fresh 101..120 (5 windows of 4), embargo 81..100 (5 windows)
    assert w1._windows(s, b=b, e=e, clock_ts=b + 1800, horizon=4, min_context=2) == (5, 5)
    # clock = b: only 81..90 available → no fresh, 2 embargo windows
    assert w1._windows(s, b=b, e=e, clock_ts=b, horizon=4, min_context=2) == (0, 2)
    # e = 0: only post-boundary points count, all fresh
    assert w1._windows(s, b=b, e=0, clock_ts=b + 1800, horizon=4, min_context=2) == (5, 0)
    # a point seen (lower bound) before b - e is never an embargo point
    s2 = _series(200)
    s2["fs_lo"][:] = T0                        # everything "seen" at the start
    assert w1._windows(s2, b=b, e=e, clock_ts=b + 1800, horizon=4, min_context=2) == (0, 0)
    # a tail-trimmed copy (offset = points cut) counts the same
    cut = {k: s[k][70:] for k in ("data_ts", "fs_lo", "fs_hi")} | {"offset": 70}
    assert w1._windows(cut, b=b, e=e, clock_ts=b + 1800, horizon=4, min_context=90) == \
        w1._windows(s, b=b, e=e, clock_ts=b + 1800, horizon=4, min_context=90) == (5, 2)
    # min_context: the first points of a series are never targets
    assert w1._windows(_series(30), b=T0, e=e, clock_ts=T0 + 1e6, horizon=4,
                       min_context=29) == (0, 0)


def test_set_rule_fresh_share_trim_cap_and_floors():
    r = w1._set_rule(600, 1000, 15, 25, min_share=0.5, max_windows=5000, min_windows=1000,
                     min_clusters=20)
    assert (r["fresh_kept"], r["embargo_kept"], r["windows"]) == (600, 600, 1200)
    assert r["fresh_share"] == 0.5 and r["sources"] == 25 and r["met"] is True
    short = w1._set_rule(400, 5000, 30, 30, min_share=0.5, max_windows=5000,
                         min_windows=1000, min_clusters=20)
    assert short["windows"] == 800 and short["fresh_share_met"] and not short["floors_met"]
    assert short["met"] is False                # the share is never relaxed to meet floors
    cap = w1._set_rule(6000, 3000, 40, 40, min_share=0.5, max_windows=5000, min_windows=1000,
                       min_clusters=20)
    assert (cap["fresh_kept"], cap["embargo_kept"]) == (5000, 0)
    s75 = w1._set_rule(300, 1000, 1, 1, min_share=0.75, max_windows=5000, min_windows=1,
                       min_clusters=1)
    assert s75["embargo_kept"] == 100 and s75["fresh_share"] == 0.75
    none = w1._set_rule(0, 900, 0, 20, min_share=0.5, max_windows=5000, min_windows=1,
                        min_clusters=1)
    assert none["windows"] == 0 and none["met"] is False   # embargo can't start a set
    only_fresh = w1._set_rule(10, 10, 1, 2, min_share=1.0, max_windows=5000, min_windows=1,
                              min_clusters=1)
    assert only_fresh["embargo_kept"] == 0 and only_fresh["sources"] == 1


def test_sweep_times_end_inside_mirror():
    end = w1.parse_time("2026-11-20T23:30:00Z")
    ts = w1.forge_sweep_times(end, sweep_days=3, step_h=6)
    assert len(ts) == 13 and ts == sorted(ts)
    assert ts[-1] + 24 * 3600 <= end and ts[-1] == w1.parse_time("2026-11-19T18:00:00Z")
    assert ts[-1] - ts[0] == 3 * 86400


DAY0 = w1.parse_time("2026-11-17T00:00:00Z")


def _mirror(root: Path, *, mtime_hour: float = 23.5) -> Path:
    """A tiny forge mirror: a 1-min feed written one partition per day (mtime
    at ``mtime_hour``), a 2-panel hourly feed with one history-rich partition,
    a weekly feed (the reader skips it) and a disabled feed."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pytest.importorskip("yaml")
    import os

    root.mkdir(parents=True)
    (root / "sources.yaml").write_text(
        "- {id: m1, frequency: PT1M, domain: energy}\n"
        "- {id: hr, frequency: PT1H, domain: weather}\n"
        "- {id: wk, frequency: P1W, domain: econ}\n"
        "- {id: dis, frequency: PT1H, domain: econ, disabled: true}\n", encoding="utf-8")

    def put(feed, day, frame, mtime):
        d = root / "data" / feed
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{day}.parquet"
        frame.to_parquet(p, index=False)
        os.utime(p, (mtime, mtime))

    rng = np.random.default_rng(0)
    for k in range(4):
        t = DAY0 + k * 86400 + 60 * np.arange(1440)
        day = w1.iso(DAY0 + k * 86400)[:10]
        put("m1", day, pd.DataFrame({"timestamp": pd.to_datetime(t, unit="s", utc=True),
                                     "value": rng.normal(size=t.size)}),
            DAY0 + k * 86400 + mtime_hour * 3600)
    n = 5000
    t = DAY0 + 3 * 86400 - 3600 * n + 3600 * np.arange(n)
    frames = [pd.DataFrame({"timestamp": pd.to_datetime(t, unit="s", utc=True),
                            "_panel_station": st, "temp": rng.normal(size=n)})
              for st in ("a", "b")]
    put("hr", "2026-11-20", pd.concat(frames, ignore_index=True), DAY0 + 3 * 86400 + 3600)
    tw = DAY0 - 604800 * np.arange(60)[::-1]
    put("wk", "2026-11-20", pd.DataFrame({"timestamp": pd.to_datetime(tw, unit="s", utc=True),
                                          "v": rng.normal(size=tw.size)}), DAY0 + 3 * 86400)
    put("dis", "2026-11-20", pd.DataFrame({"timestamp": pd.to_datetime(t, unit="s", utc=True),
                                           "v": rng.normal(size=n)}), DAY0 + 3 * 86400)
    return root


def test_forge_first_seen_proxy_bracket(tmp_path):
    mirror = w1.load_forge(_mirror(tmp_path / "f"), horizon=64)
    (s,) = mirror["feeds"]["m1"]["series"]
    assert s["length"] == 4 * 1440
    day2 = (s["data_ts"] >= DAY0 + 2 * 86400) & (s["data_ts"] < DAY0 + 3 * 86400)
    assert np.all(s["fs_hi"][day2] == DAY0 + 2 * 86400 + 23.5 * 3600)   # mtime < day end
    assert np.all(s["fs_lo"][day2] == s["data_ts"][day2])               # >= day start
    assert s["baseline"].sum() == 1440                                  # first partition
    part = w1.load_forge(tmp_path / "f", horizon=64, clock="partition")
    (sp,) = part["feeds"]["m1"]["series"]
    assert np.all(sp["fs_hi"][day2] == DAY0 + 3 * 86400)
    assert mirror["feeds"]["wk"]["skipped"].startswith("slower than daily")
    assert mirror["feeds"]["dis"]["skipped"] == "disabled"
    assert len(mirror["feeds"]["hr"]["series"]) == 2
    assert mirror["mirror_end_ts"] == DAY0 + 3 * 86400 + 23.5 * 3600


def test_forge_cli_report(tmp_path):
    forge = _mirror(tmp_path / "f")
    out = tmp_path / "w1.json"
    argv = ["--forge-dir", str(forge), "--out", str(out), "--embargo-hours", "48",
            "--min-windows", "10", "--min-clusters", "1"]
    assert w1.main(argv) == 0
    doc = json.loads(out.read_text())
    assert doc["labels"]["forge_mirror"] == w1.FORGE_LABEL == "measured from forge mirror"
    assert doc["labels"]["pairs"] == w1.BOUND_LABEL and doc["pairs"].startswith("not run:")
    fm = doc["forge_mirror"]
    assert fm["label"] == w1.FORGE_LABEL
    assert set(fm["per_freq"]) == {n for n, *_ in w1.FREQ_BUCKETS}
    assert fm["per_freq"]["1min"]["sources"] == 1 and fm["per_freq"]["1min"]["series"] == 1
    lat = fm["per_freq"]["1min"]["first_seen_latency_hours"]
    assert 11.0 < lat["median_over_sources_upper"] < 12.5    # mean wait to a 23:30 write
    assert fm["per_freq"]["hourly"]["series"] == 2 and fm["per_freq"]["hourly"]["sources"] == 1
    assert fm["per_freq"]["slower"]["reader_emits"] is False
    assert fm["sources"]["dis"]["skipped"] == "disabled"
    sweep = fm["windows"]["sweep"]
    assert len(sweep) == 13
    for row in sweep:
        for c in row["clocks"].values():
            assert set(c["set"]) >= {"fresh_kept", "embargo_kept", "windows", "met"}
            assert c["set"]["windows"] <= 5000
    summ = fm["windows"]["summary"]
    assert set(summ) == {"T+6h", "T+9h", "T+12h", "T+24h"}
    assert summ["T+24h"]["fresh"]["median"] >= summ["T+6h"]["fresh"]["median"]
    assert summ["T+24h"]["embargo"]["median"] > 0            # 1-min data inside the embargo
    rungs = {r["context"]: r for r in fm["long_history"]["rungs"]}
    assert rungs[4096]["series"] == 3 and rungs[4096]["sources"] == 2  # hr x2 + m1
    assert rungs[8192]["series"] == 0 and rungs[131072]["need_points"] == 131072 + 64
    assert rungs[4096]["gate_met"] is False
    from cascade.shared.config.lane_d import LaneDConfig
    assert fm["long_history"]["gate"]["min_long_series"] == LaneDConfig().min_long_series
    assert doc["forge_params"]["embargo_hours"] == 48 and doc["forge_params"]["max_windows"] == 5000
    out2 = tmp_path / "w1b.json"
    assert w1.main([*argv[:3], str(out2), *argv[4:]]) == 0      # deterministic
    assert json.loads(out2.read_text())["forge_mirror"] == fm


def test_forge_and_snapshots_together(tmp_path):
    forge = _mirror(tmp_path / "f")
    full = {f"s{i}": _walk(600, 40 + i) for i in range(2)}
    md = {k: {"freq": "H", "source": f"src{i}"} for i, k in enumerate(full)}
    root = tmp_path / "snaps"
    _write(root / "100", {k: v[:300] for k, v in full.items()}, md)
    _write(root / "200", {k: v[:400] for k, v in full.items()}, md)
    out = tmp_path / "w1.json"
    assert w1.main(["--snapshots-dir", str(root), "--forge-dir", str(forge), "--out", str(out),
                    "--build-time", "100=2026-11-01T02:00:00Z",
                    "--build-time", "200=2026-11-02T02:00:00Z"]) == 0
    doc = json.loads(out.read_text())
    assert doc["label"] == w1.BOUND_LABEL and isinstance(doc["pairs"], list)
    assert doc["forge_mirror"]["label"] == w1.FORGE_LABEL


def test_cli_needs_a_source_and_valid_share(tmp_path):
    with pytest.raises(SystemExit):
        w1.main(["--out", str(tmp_path / "x.json")])
    forge = _mirror(tmp_path / "f")
    with pytest.raises(SystemExit):
        w1.main(["--forge-dir", str(forge), "--out", str(tmp_path / "x.json"),
                 "--min-fresh-share", "1.5"])
