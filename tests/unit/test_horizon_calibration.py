"""Multi-horizon calibration telemetry (docs/notes/2026-08-31-multi-horizon-eval.md).

Log-only measurement for the planned scored 16/64/256/720 ladder: seeded
even-by-domain eligibility-filtered draws, production-scorer geomeans, paired
deltas vs the king. Inert by default (``[eval] calib_horizons = []``).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cascade.validator.calibration import (
    MIN_CONTEXT,
    calib_draw,
    load_pool_series,
    run_horizon_calibration,
)

# ── config round-trip (every knob needs the loader parse) ────────────────────


def test_calib_knobs_parse_and_default_off(tmp_path):
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    src = DEFAULT_CHAIN_TOML.read_text()
    p = tmp_path / "chain.toml"
    p.write_text(src)
    cfg = load_chain_config(p)
    assert cfg.eval.calib_horizons == ()          # inert default
    assert cfg.eval.calib_windows == 256
    assert cfg.eval.calib_num_samples == 32

    p.write_text(src.replace(
        "\n[eval]\n",
        "\n[eval]\ncalib_horizons = [16, 256, 720]\n"
        "calib_windows = 128\ncalib_num_samples = 8\n", 1))
    cfg = load_chain_config(p)
    assert cfg.eval.calib_horizons == (16, 256, 720)
    assert (cfg.eval.calib_windows, cfg.eval.calib_num_samples) == (128, 8)


# ── the draw: eligibility, evenness, determinism ─────────────────────────────


def _series_set():
    rng = np.random.default_rng(7)
    series, metadata = [], []
    # domain "a": 6 long series; domain "b": 6 long; domain "c": 4 SHORT ones
    # (eligible at h=64, not at h=720).
    for dom, n, length in (("a", 6, 1500), ("b", 6, 1500), ("c", 4, 300)):
        for _ in range(n):
            series.append(rng.normal(size=length).cumsum())
            metadata.append({"domain": dom})
    return series, metadata


def test_calib_draw_filters_eligibility_and_splits_evenly():
    series, metadata = _series_set()
    w64 = calib_draw(series, metadata, horizon=64, n_windows=12,
                     context_length=4096, seed=1)
    doms64 = sorted(w.metadata["domain"] for w in w64)
    assert len(w64) == 12 and set(doms64) == {"a", "b", "c"}

    w720 = calib_draw(series, metadata, horizon=720, n_windows=12,
                      context_length=4096, seed=1)
    doms720 = {w.metadata["domain"] for w in w720}
    assert "c" not in doms720                       # too short for 720+context
    assert doms720 == {"a", "b"}
    assert all(w.target.shape[-1] == 720 for w in w720)
    # a scarce rung caps at eligible stock, never pads
    assert len(w720) == 12

    # short series need horizon + MIN_CONTEXT
    assert 720 + MIN_CONTEXT > 300


def test_calib_draw_deterministic_in_seed():
    series, metadata = _series_set()
    a = calib_draw(series, metadata, horizon=256, n_windows=8,
                   context_length=4096, seed=42)
    b = calib_draw(series, metadata, horizon=256, n_windows=8,
                   context_length=4096, seed=42)
    c = calib_draw(series, metadata, horizon=256, n_windows=8,
                   context_length=4096, seed=43)
    assert [w.series_id for w in a] == [w.series_id for w in b]
    assert np.array_equal(a[0].target, b[0].target)
    assert [w.series_id for w in a] != [w.series_id for w in c]


def test_calib_draw_empty_when_nothing_eligible():
    series = [np.ones(100)]
    assert calib_draw(series, [{"domain": "x"}], horizon=720, n_windows=4,
                      context_length=4096, seed=0) == []


# ── the run: per-rung geomeans + paired deltas, failures isolated ────────────


def _fake_scores(windows, *, mase):
    from cascade.eval.scoring import WindowScore

    return [WindowScore(series_id=w.series_id, mase=mase,
                        qloss_per_q=np.full(9, 0.5), abs_target=10.0,
                        domain=w.metadata.get("domain", ""))
            for w in windows]


def _pool_dir(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    series, metadata = _series_set()
    md = {}
    for i, (s, m) in enumerate(zip(series, metadata, strict=True)):
        np.save(d / f"s{i}.npy", s)
        md[f"s{i}"] = m
    (d / "metadata.json").write_text(json.dumps(md))
    return d


def test_run_horizon_calibration_reports_rungs_and_deltas(tmp_path):
    d = _pool_dir(tmp_path)

    def fake_eval(ckpt, windows, ns, dev):
        mase = 1.0 if "king" in str(ckpt) else 0.9      # challenger better
        return _fake_scores(windows, mase=mase)

    doc = run_horizon_calibration(
        {"king": tmp_path / "king", "challenger-u9": tmp_path / "chal"},
        d, horizons=(64, 720), n_windows=8, num_samples=4,
        context_length=4096, seed=5, evaluate_fn=fake_eval)

    assert set(doc["horizons"]) == {"64", "720"}
    rung = doc["horizons"]["720"]
    assert rung["entries"]["king"]["geomean"] is not None
    delta = rung["entries"]["challenger-u9"]["mase_delta_vs_king"]
    assert delta["mean"] == pytest.approx(-0.1)
    assert delta["win_frac"] == 1.0
    assert "a" in rung["entries"]["king"]["domains"]


def test_run_horizon_calibration_survives_a_failing_leg(tmp_path):
    d = _pool_dir(tmp_path)

    def flaky_eval(ckpt, windows, ns, dev):
        if "chal" in str(ckpt):
            raise RuntimeError("wrapper exploded")
        return _fake_scores(windows, mase=1.0)

    doc = run_horizon_calibration(
        {"king": tmp_path / "king", "challenger-u9": tmp_path / "chal"},
        d, horizons=(64,), n_windows=6, num_samples=2,
        context_length=4096, seed=5, evaluate_fn=flaky_eval)
    rung = doc["horizons"]["64"]
    assert "king" in rung["entries"] and "challenger-u9" not in rung["entries"]


def test_load_pool_series_reads_snapshot_layout(tmp_path):
    d = _pool_dir(tmp_path)
    series, ids, metadata = load_pool_series(d)
    assert len(series) == len(ids) == len(metadata) == 16
    assert {m.get("domain") for m in metadata} == {"a", "b", "c"}


# ── source-dir plumbing ──────────────────────────────────────────────────────


def test_window_source_exposes_its_series_dir(tmp_path, cfg):
    from cascade.validator.pool import window_source_from_dir

    d = _pool_dir(tmp_path)
    src = window_source_from_dir(d, cfg, label="test")
    assert src.snapshot_dir_for_round(block=None) == d
