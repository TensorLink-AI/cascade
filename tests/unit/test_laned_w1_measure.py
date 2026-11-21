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
