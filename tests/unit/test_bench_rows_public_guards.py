"""Lane-B public guards (DEC-CA-0050 P5): per-item rows out of a sidecar report,
king/candidate pairing, and the hand-off into judge_expert_leg. The sidecar
subprocess is mocked; FEV's pure metrics are tested without gluonts/fev."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from cascade.eval import benchmarks as bench
from cascade.eval.expert_lane import ExpertLaneParams, judge_expert_leg
from cascade.eval.public_guards import MissingPublicGuard, pair_public_rows

REPO = Path(__file__).resolve().parents[2]


def _report(*suites):
    return {"checkpoint": "/x", "suites": list(suites)}


def _suite(name, rows, status="ok"):
    return {"suite": name, "status": status, "metrics": {}, "rows": rows}


def _rows(items, crps_scale=1.0, mase_scale=1.0):
    return [{"full": i, "CRPS": 0.5 * (k + 1) * crps_scale, "MASE": 1.0 + k * mase_scale * 0.1}
            for k, i in enumerate(items)]


# ── bridge: item rows ────────────────────────────────────────────────────────


def test_item_rows_every_suite_raw_scores():
    rep = _report(
        _suite("gift-eval", [{"full": "m4/D/short", "CRPS": 0.4, "MASE": 1.1,
                              "crps_ratio": 0.8, "mase_ratio": 0.9}]),
        _suite("boom", [{"full": "b/5T/short", "CRPS": 0.3, "MASE": float("nan")}]),
        _suite("time", [{"full": "t/short", "CRPS": "x", "MASE": 2.0}]),
        _suite("fev", [], status="skipped"),
        _suite("unknown", [{"full": "z", "CRPS": 1.0, "MASE": 1.0}]),
    )
    got = bench.item_rows_from_report(rep)
    assert set(got) == {"gift", "boom", "time", "fev"}
    assert got["gift"]["rows"] == [{"item": "m4/D/short", "crps": 0.4, "mase": 1.1}]
    assert got["boom"]["rows"][0]["mase"] is None          # NaN → None
    assert got["time"]["rows"][0]["crps"] is None          # non-numeric → None
    assert got["fev"] == {"status": "skipped", "rows": []}
    assert bench.item_rows_from_report(None) is None


def test_run_item_rows_argv_and_parse(monkeypatch, tmp_path):
    seen = {}

    def fake(ckpt, project, tail, *, timeout_s, uv_bin):
        seen["tail"] = tail
        return _report(_suite("fev", _rows(["a", "b"])))

    monkeypatch.setattr(bench, "_invoke_sidecar", fake)
    got = bench.run_item_rows(tmp_path, project_dir=tmp_path, data_dir="/d", max_series=3)
    assert seen["tail"][:2] == ["--suites", "gift-eval,boom,time,fev"]
    assert "--data-dir" in seen["tail"] and "--max-series" in seen["tail"]
    assert [r["item"] for r in got["fev"]["rows"]] == ["a", "b"]


def test_run_item_rows_none_when_sidecar_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(bench, "_invoke_sidecar", lambda *a, **k: None)
    assert bench.run_item_rows(tmp_path, project_dir=tmp_path) is None


def test_run_item_rows_refuses_foreign_checkpoint(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "forecast_wrapper.py").write_text("class Wrapper: ...\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    assert bench.run_item_rows(ckpt, project_dir=tmp_path) is None


# ── pairing ──────────────────────────────────────────────────────────────────


def _side(rows_by_bench, status="ok"):
    return bench.item_rows_from_report(
        _report(*(_suite(s, r, status) for s, r in rows_by_bench.items())))


def test_pairing_inner_join_sorted_and_drop_counts():
    king = _side({"gift-eval": _rows(["c", "a", "b", "only-king"])})
    chal_rows = _rows(["b", "a", "c", "only-chal"])
    chal_rows[0]["MASE"] = -1.0                  # b: unusable MASE on one side
    chal = _side({"gift-eval": chal_rows})
    p = pair_public_rows(king, chal)
    assert p.items["gift_crps"] == ["a", "b", "c"]
    assert p.items["gift_mase"] == ["a", "c"]
    assert p.dropped == {"gift_crps": 2, "gift_mase": 3}
    k, c = p.public["gift_crps"]
    assert len(k) == len(c) == 3
    assert p.missing == []


def test_pairing_duplicates_dropped_and_missing_reported():
    king = _side({"gift-eval": _rows(["a", "a", "b"]), "boom": _rows(["x"])})
    chal = _side({"gift-eval": _rows(["a", "b"]), "boom": _rows(["x"])})
    chal["boom"]["status"] = "error"
    p = pair_public_rows(king, chal)
    assert p.items["gift_crps"] == ["b"]
    assert "boom" in p.missing and "boom_crps" not in p.public
    with pytest.raises(MissingPublicGuard):
        pair_public_rows(king, chal, require=["boom"])
    with pytest.raises(MissingPublicGuard):
        pair_public_rows(king, chal, require=["fev"])      # never reported at all


def test_pairing_is_deterministic():
    king = _side({"gift-eval": _rows(list("edcba")), "fev": _rows(list("zyx"))})
    chal = _side({"gift-eval": _rows(list("abcde")), "fev": _rows(list("xyz"))})
    assert pair_public_rows(king, chal) == pair_public_rows(king, chal)


def test_guard_rows_preserve_data_revision_and_refuse_different_data():
    raw = _report(_suite("gift-eval", _rows(["a", "b"])))
    king = bench.item_rows_from_report({**raw, "data_revisions": {"gift-eval": "revision-a"}})
    chal = bench.item_rows_from_report({**raw, "data_revisions": {"gift-eval": "revision-b"}})
    assert king["gift"]["data_revision"] == "revision-a"
    with pytest.raises(MissingPublicGuard):
        pair_public_rows(king, chal, require=["gift"])
    assert "gift_crps" not in pair_public_rows(king, chal).public
    assert pair_public_rows(king, king, require=["gift"]).items["gift_crps"] == ["a", "b"]


def test_pinned_guard_cannot_pair_against_unknown_revision():
    king = _side({"gift-eval": _rows(["a", "b"])})
    chal = _side({"gift-eval": _rows(["a", "b"])})
    king["gift"]["data_revision"] = "revision-a"
    with pytest.raises(MissingPublicGuard):
        pair_public_rows(king, chal, require=["gift"])


def test_paired_rows_drive_the_lane_b_guard():
    rng = np.random.default_rng(0)
    n = 120
    doms = ["energy"] * n + ["web"] * n
    kq = rng.lognormal(0, 0.3, size=(2 * n, 9))
    km = rng.lognormal(0, 0.3, size=2 * n)
    f = np.array([0.9] * n + [1.0] * n)
    a = np.full(2 * n, 50.0)
    items = [f"cfg{i:03d}" for i in range(97)]
    base = rng.lognormal(-0.3, 0.2, size=97)
    king = _side({"gift-eval": [{"full": i, "CRPS": v, "MASE": v} for i, v in zip(items, base, strict=True)]})
    worse = _side({"gift-eval": [{"full": i, "CRPS": v * 1.05, "MASE": v} for i, v in zip(items, base, strict=True)]})
    same = _side({"gift-eval": [{"full": i, "CRPS": v, "MASE": v} for i, v in zip(items, base, strict=True)]})
    params = ExpertLaneParams(B=1000, min_target_windows=40, min_guard_windows=30)
    kw = dict(target_domain="energy", domains=doms, king_qloss=kq, king_abs_target=a,
              king_mase=km, chal_qloss=kq * f[:, None], chal_abs_target=a, chal_mase=km * f,
              params=params, seed="s")
    assert judge_expert_leg(public=pair_public_rows(king, same).public, **kw).passed
    v = judge_expert_leg(public=pair_public_rows(king, worse).public, **kw)
    assert not v.passed and "gift_crps" in v.reason


# ── FEV pure metrics (sidecar module, no gluonts/fev needed) ─────────────────


def _load_fev_metrics():
    path = REPO / "benchmarks" / "cascade_benchmark" / "fev_metrics.py"
    spec = importlib.util.spec_from_file_location("_fev_metrics_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fev_metrics_perfect_and_naive():
    fm = _load_fev_metrics()
    hist = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 10)
    y = np.array([1.0, 2.0, 3.0, 4.0])
    perfect = fm.TaskAccumulator()
    perfect.add(hist, y, fm.point_as_quantiles(y, 9), m=4)
    crps, mase = perfect.result()
    assert crps == 0.0 and mase is None          # periodic history: zero seasonal scale
    assert np.allclose(fm.seasonal_naive_forecast(hist, 6, 4), [1, 2, 3, 4, 1, 2])
    off = fm.TaskAccumulator()
    off.add(hist, y, fm.point_as_quantiles(y + 1.0, 9), m=1)
    c, m = off.result()
    # point forecast off by 1 everywhere: WQL = 2·Σ|e|·mean(q-weights)/Σ|y|
    assert c == pytest.approx(2 * np.mean([max(q * -1, (q - 1) * -1) for q in fm.QUANTILE_LEVELS])
                              * 4 / 10)
    assert m == pytest.approx(1.0 / fm.seasonal_scale(hist, 1))


def test_fev_metrics_scale_and_geomean_edges():
    fm = _load_fev_metrics()
    assert fm.seasonal_scale(np.ones(20), 4) is None           # zero scale
    assert fm.seasonal_scale(np.array([1.0]), 4) is None        # too short
    assert fm.seasonal_scale(np.array([1.0, 3.0, 2.0]), 7) == pytest.approx(1.5)  # lag-1 fallback
    assert fm.normalised_geomean([0.5, 2.0, None], [1.0, 1.0, 1.0]) == pytest.approx(1.0)
    assert math.isnan(fm.normalised_geomean([None], [1.0]))
    acc = fm.TaskAccumulator()
    acc.add(np.arange(10.0), np.array([np.nan, np.nan]), np.zeros((2, 9)), m=1)
    assert acc.result() == (None, None) and acc.n_series == 0
