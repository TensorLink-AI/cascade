"""scripts/null_fpr.py — lane D null false-positive-rate tool, on tiny
synthetic per-window rows (CPU only, no torch, no real data)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "null_fpr", Path(__file__).resolve().parents[2] / "scripts" / "null_fpr.py")
nf = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = nf
_SPEC.loader.exec_module(nf)

LEVELS = tuple(round(0.1 * i, 1) for i in range(1, 10))


def _base(n_windows: int = 240, n_sources: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    q = rng.uniform(0.5, 2.0, size=(n_windows, len(LEVELS)))
    a = rng.uniform(20.0, 40.0, size=n_windows)
    m = rng.uniform(0.5, 1.5, size=n_windows)
    src = [f"src{i % n_sources}" for i in range(n_windows)]
    return q, a, m, src


def _rows(models: dict, src, *, freq=None) -> nf.Rows:
    n = len(src)
    return nf.Rows(series_id=[f"w{i}" for i in range(n)], cluster=src, source=src,
                   domain=["d"] * n, freq=freq or ["H"] * n, models=models,
                   quantile_levels=LEVELS)


def _scaled(comp, factor: float, *, jitter: float = 0.0, seed: int = 1):
    q, a, m = comp
    rng = np.random.default_rng(seed)
    e = np.exp(rng.normal(0.0, jitter, size=m.shape)) if jitter else np.ones_like(m)
    return q * factor * e[:, None], a.copy(), m * factor * e


KW = dict(n=60, R=8, floors=[20], margins=[0.0, 0.01, 0.03], targets=[0.05],
          era_entries=4, alpha=0.05, B=200, seed="t", subhourly_only=False)


def test_identical_rows_give_zero_fpr():
    q, a, m, src = _base()
    rows = _rows({"a": (q, a, m), "b": (q.copy(), a.copy(), m.copy())}, src)
    out = nf.run_stats(rows, pairs=[("a", "b"), ("b", "a")], **KW)
    for p in out["pairs"].values():
        r = p["floors"]["min_clusters=20"]
        assert all(g["fpr"] == 0.0 for g in r["resampled"]["fpr"])
        assert r["resampled"]["null_lcb_quantiles"]["p99"] == pytest.approx(0.0)
        assert all(g["p_conf"] == 0.0 for g in r["confirmation"]["fpr"])


def test_planted_five_percent_improvement_is_detected():
    q, a, m, src = _base()
    rows = _rows({"king": (q, a, m), "chal": _scaled((q, a, m), 0.95, jitter=0.01)}, src)
    out = nf.run_stats(rows, pairs=[("king", "chal")], **KW)
    r = out["pairs"]["king>chal"]["floors"]["min_clusters=20"]
    by_m = {g["margin"]: g for g in r["resampled"]["fpr"]}
    assert by_m[0.01]["fpr"] == 1.0 and by_m[0.03]["fpr"] == 1.0
    conf = {g["margin"]: g for g in r["confirmation"]["fpr"]}
    assert conf[0.01]["p_conf"] == 1.0
    assert r["resampled"]["null_lcb_quantiles"]["p50"] > 0.03
    # the reverse direction (the worse model as challenger) never clears
    out2 = nf.run_stats(rows, pairs=[("chal", "king")], **KW)
    r2 = out2["pairs"]["chal>king"]["floors"]["min_clusters=20"]
    assert all(g["fpr"] == 0.0 for g in r2["resampled"]["fpr"])


def test_row_pairing_and_order_preserved():
    q, a, m, src = _base()
    b = _scaled((q, a, m), 0.97, jitter=0.05)
    rows = _rows({"a": (q, a, m), "b": b}, src)
    seen = []

    def spy(kq, ka, km, cq, ca, cm, **kw):
        seen.append((kq, km, cq, cm, kw["clusters"]))
        return 0.0

    sc = nf.PairScorer(rows, "a", "b", alpha=0.05, B=10, lcb_fn=spy)
    rng = np.random.default_rng(3)
    idx, _ = nf.draw_set(rng, np.arange(len(rows)), rows.cluster, 50, 20)
    assert np.all(np.diff(idx) > 0)                       # sorted ⇒ file row order
    sc.score(idx, 1)
    kq, km, cq, cm, cl = seen[0]
    np.testing.assert_array_equal(kq, q[idx])
    np.testing.assert_array_equal(km, m[idx])
    np.testing.assert_array_equal(cq, b[0][idx])
    np.testing.assert_array_equal(cm, b[2][idx])
    assert cl == [src[i] for i in idx]
    ws = rows.window_scores("b")
    assert [w.series_id for w in ws] == [f"w{i}" for i in range(len(rows))]
    assert ws[7].mase == b[2][7]


def test_rows_round_trip_and_unpaired_scores_refused(tmp_path):
    from cascade.eval.scoring import WindowScore
    from cascade.eval.window import EvalWindow

    q, a, m, src = _base(40, 20)
    rows = _rows({"a": (q, a, m), "b": _scaled((q, a, m), 0.9)}, src)
    npz, _js = nf.save_rows(tmp_path / "r", rows, {"snapshot_tree_sha256": "x"})
    back, man, sha = nf.load_rows(npz)
    assert man["models"] == ["a", "b"] and sha == nf.file_sha256(npz)
    assert back.series_id.tolist() == rows.series_id.tolist()
    np.testing.assert_array_equal(back.models["b"][2], rows.models["b"][2])
    npz.write_bytes(npz.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        nf.load_rows(npz)

    wins = [EvalWindow(f"w{i}", np.ones(8), np.ones(4), {"source": f"s{i}", "freq": "5min"})
            for i in range(3)]

    def ws(i):
        return WindowScore(series_id=f"w{i}", mase=1.0, qloss_per_q=np.ones(9),
                           abs_target=4.0, source=f"s{i}")

    built = nf.rows_from_scores(wins, {"a": [ws(0), ws(1), ws(2)], "b": [ws(0), ws(1), ws(2)]})
    assert built.freq.tolist() == ["5min"] * 3 and built.cluster.tolist() == ["s0", "s1", "s2"]
    with pytest.raises(ValueError, match="unpaired"):
        nf.rows_from_scores(wins, {"a": [ws(0), ws(1), ws(2)], "b": [ws(1), ws(0), ws(2)]})


def test_disjoint_partition_never_reuses_a_window():
    rng = np.random.default_rng(0)
    pool = np.arange(1000, 1347)
    parts = nf.disjoint_partition(rng, pool, 50)
    assert len(parts) == 347 // 50
    flat = np.concatenate(parts)
    assert len(flat) == len(set(flat.tolist())) == 50 * len(parts)
    assert set(flat.tolist()) <= set(pool.tolist())
    pair, _ = nf.draw_disjoint_pair(rng, pool, np.asarray([str(i % 30) for i in pool]), 50, 20)
    assert not set(pair[0].tolist()) & set(pair[1].tolist())


def test_below_floor_reported_never_scored():
    q, a, m, _src = _base(240)
    few = [f"src{i % 5}" for i in range(240)]                 # only 5 sources
    rows = _rows({"a": (q, a, m), "b": _scaled((q, a, m), 0.9)}, few)
    calls = []

    def spy(*args, **kw):
        calls.append(1)
        return 0.0

    out = nf.run_stats(rows, pairs=[("a", "b")], lcb_fn=spy, **KW)
    r = out["pairs"]["a>b"]["floors"]["min_clusters=20"]
    for key in ("resampled", "disjoint", "confirmation", "spread_ratio"):
        assert isinstance(r[key], str) and r[key].split(":")[0] in ("below floor", "not run")
    assert "5 sources" in r["resampled"]
    assert not calls and out["pairs"]["a>b"]["scored_sets"] == 0
    # A draw that can never reach the floor is rejected, never returned.
    idx, rej = nf.draw_set(np.random.default_rng(0), np.arange(240), rows.cluster, 60, 6,
                           max_tries=5)
    assert idx is None and rej == 5


def test_subhourly_filter_applies_the_same_floor():
    q, a, m, src = _base()
    freq = ["5min" if i % 2 else "H" for i in range(240)]
    rows = _rows({"a": (q, a, m), "b": _scaled((q, a, m), 1.0, jitter=0.02)}, src, freq=freq)
    assert nf.is_subhourly("15T") and nf.is_subhourly("10s") and not nf.is_subhourly("H")
    assert not nf.is_subhourly("")
    kw = dict(KW, subhourly_only=True)
    out = nf.run_stats(rows, pairs=[("a", "b")], **kw)
    assert out["pool_filter"]["kept"] == 120
    r = out["pairs"]["a>b"]["floors"]["min_clusters=20"]
    # odd rows only ⇒ sources src1, src3, … (15 of them) — under a 20 floor
    assert r["pool"]["sources"] == 15 and r["resampled"].startswith("below floor")


def test_self_pair_refused():
    q, a, m, src = _base(40, 20)
    rows = _rows({"a": (q, a, m), "b": (q, a, m)}, src)
    with pytest.raises(ValueError, match="self-pair"):
        nf.ordered_null_pairs(["a", "b"], ["a:a"])
    with pytest.raises(ValueError, match="self-pair"):
        nf.PairScorer(rows, "a", "a", alpha=0.05, B=10)
    with pytest.raises(ValueError, match="same checkpoint bytes"):
        nf.ordered_null_pairs(["a", "b"], None, {"a": "d1", "b": "d1"})
    assert nf.ordered_null_pairs(["a", "b"], None) == [("a", "b"), ("b", "a")]


def test_margin_for_rate_and_wilson():
    s = np.asarray([0.05, 0.04, 0.03, -np.inf, -np.inf] + [-0.01] * 95)
    m = nf.margin_for_rate(s, 0.02)              # 2 of 100 allowed ⇒ just above 0.03
    assert m["allowed_false_wins"] == 2 and 0.03 < m["d_margin"] < 0.0300001
    assert np.mean(s >= m["d_margin"]) <= 0.02
    assert nf.margin_for_rate(s, 0.001)["resolution"]
    assert nf.margin_for_rate(np.full(10, -np.inf), 0.01)["d_margin"] == 0.0
    lo, hi = nf.wilson(0, 50)
    assert lo == 0.0 and 0.0 < hi < 0.1


def test_stats_cli_writes_provenance(tmp_path):
    q, a, m, src = _base()
    rows = _rows({"a": (q, a, m), "b": _scaled((q, a, m), 1.0, jitter=0.02)}, src)
    npz, _ = nf.save_rows(tmp_path / "rows", rows,
                          {"snapshot_tree_sha256": "abc", "seeds": {"draw": 0},
                           "checkpoints": {"a": {"tree_sha256": "d1"},
                                           "b": {"tree_sha256": "d2"}}})
    out = tmp_path / "s.json"
    argv = ["stats", "--rows", str(npz), "--out", str(out), "--era-entries", "6",
            "--n", "60", "--R", "6", "--B", "100", "--min-clusters", "20", "31"]
    assert nf.main(argv) == 0
    doc = json.loads(out.read_text())
    for k in ("command_line", "host", "timestamp_utc", "rows_sha256", "snapshot_tree_sha256",
              "seeds"):
        assert doc[k]
    assert doc["rows_sha256"] == nf.file_sha256(npz)
    r = doc["pairs"]["a>b"]["floors"]
    assert set(r) == {"min_clusters=20", "min_clusters=31"}
    assert r["min_clusters=31"]["resampled"].startswith("below floor")   # only 30 sources
    conf = r["min_clusters=20"]["confirmation"]
    assert set(conf["fpr"][0]["per_era_fpr"]) == {"k=1", "k=6", "k=8"}
