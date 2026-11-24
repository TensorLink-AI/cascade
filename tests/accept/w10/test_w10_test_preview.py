"""W10: `cascade lane-d test`, the offline lane D preview (owner 2026-11-24:
"iteration tools for each lane so miners can optimise offline").

The preview must run the validators' own code (draw, score_window,
judge_cohort, crown_bar), never a copy, so it cannot drift. The only fake is
the model forward (``cascade.eval.lane_d_scorer.forecast_quantiles``): no
torch on this box; the torch path is the scorer's own and CI covers it."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from .helpers import CHAIN_TOML

H = 64
PERIOD = 24
SOURCES = 25
PER_SOURCE = 2
PER_SERIES = 6
LENGTH = H * PER_SERIES + 8 * 32
# z-offsets of the 9 scoring levels (0.1 .. 0.9), in units of the spread
_Z = np.array([-1.2816, -0.8416, -0.5244, -0.2533, 0.0, 0.2533, 0.5244, 0.8416, 1.2816])


def _series(tmp_path, *, sources=SOURCES, per_source=PER_SOURCE, name="series.npz"):
    rng = np.random.default_rng(7)
    t = np.arange(LENGTH, dtype=np.float64)
    arrays = {}
    for s in range(sources):
        for k in range(per_source):
            phase = rng.uniform(0, 2 * np.pi)
            arrays[f"src{s:02d}/s{k}"] = (50.0 + 10.0 * np.sin(2 * np.pi * t / PERIOD + phase)
                                          + rng.normal(0, 0.3, t.size))
    path = tmp_path / name
    np.savez(path, **arrays)
    return path


def _ckpt(tmp_path, name: str, kind: str):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"moe": {"patch_size": 32}, "fake": kind}))
    (d / "weights.safetensors").write_bytes(b"\0" * 8)
    return d


def _fake_forecast(ckpt, histories, *, context, horizon, contexts=None, device="cpu", **_):
    """seasonal = seasonal naive (good on these series); flat = last value
    (poor); both with the same quantile spread."""
    kind = json.loads((ckpt / "config.json").read_text())["fake"]
    out = []
    for hist in histories:
        hist = np.asarray(hist, dtype=np.float64)
        if kind == "seasonal":
            med = np.tile(hist[-PERIOD:], horizon // PERIOD + 1)[:horizon]
        else:
            med = np.full(horizon, hist[-1])
        out.append(med[:, None] + 0.5 * _Z[None, :])
    return np.stack(out)


LIGHT_MIN_WINDOWS = 250
LIGHT_B = 1000


@pytest.fixture
def chain_toml(tmp_path):
    """The repo chain.toml with only [lane_d] min_windows / bootstrap_B made
    lighter (CI time); every other lane D rule is the released one."""
    lines = CHAIN_TOML.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "[lane_d]")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    for i in range(start, end):
        key = lines[i].split("=", 1)[0].strip()
        if key == "min_windows":
            lines[i] = f"min_windows = {LIGHT_MIN_WINDOWS}\n"
        elif key == "bootstrap_B":
            lines[i] = f"bootstrap_B = {LIGHT_B}\n"
    path = tmp_path / "chain.toml"
    path.write_text("".join(lines), encoding="utf-8")
    return path


@pytest.fixture
def run(monkeypatch, capsys, chain_toml):
    import cascade.eval.lane_d_scorer as scorer

    monkeypatch.setattr(scorer, "forecast_quantiles", _fake_forecast)

    def _run(*argv):
        from cascade.miner.cli import main

        rc = main(["lane-d", "test", *map(str, argv), "--chain-toml", str(chain_toml), "--json"])
        doc = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        return rc, doc
    return _run


def test_lane_d_test_is_registered_with_json():
    import argparse

    from cascade.miner.cli import build_parser

    root = build_parser()
    (sub,) = [a for a in root._actions if isinstance(a, argparse._SubParsersAction)]
    (lsub,) = [a for a in sub.choices["lane-d"]._actions
               if isinstance(a, argparse._SubParsersAction)]
    assert "test" in lsub.choices
    flags = {s for a in lsub.choices["test"]._actions for s in a.option_strings}
    for f in ("--king", "--context", "--pool-dir", "--series", "--public-weight", "--tenure",
              "--json", "--as-of"):
        assert f in flags, f


def test_a_better_model_clears_the_tenure_zero_bar_live_only(tmp_path, run):
    rc, doc = run(_ckpt(tmp_path, "cand", "seasonal"), "--king", _ckpt(tmp_path, "king", "flat"),
                  "--context", 4096, "--series", _series(tmp_path),
                  "--windows-per-series", PER_SERIES)
    assert rc == 0, doc
    d = doc["data"]
    assert d["public_weight"] == 0.0
    assert d["crown_bar"]["bar"] == pytest.approx(0.03)
    assert d["floors"]["met"] and d["floors"]["sources"] == SOURCES
    assert d["verdict"]["cleared"] and d["would_clear"]
    assert set(d["verdict"]["halves"]) == {"live"}
    assert d["verdict"]["lcb"] >= 0.03
    assert len(d["per_window"]) == d["windows"]["judged"] >= LIGHT_MIN_WINDOWS


def test_the_verdict_is_the_validators_judge_cohort_on_the_same_rows(tmp_path, run, chain_toml):
    from cascade.shared.config import load_chain_config
    from cascade.validator.lane_d import judge_cohort

    rc, doc = run(_ckpt(tmp_path, "cand", "seasonal"), "--king", _ckpt(tmp_path, "king", "flat"),
                  "--context", 4096, "--series", _series(tmp_path),
                  "--windows-per-series", PER_SERIES, "--tenure", 32, "--seed", 3)
    assert rc == 0, doc
    d = doc["data"]
    rows = d["per_window"]
    ld = load_chain_config(chain_toml).lane_d
    ld =dataclasses.replace(ld, public_weight=0.0)
    king = (np.array([r["king_crps"] for r in rows]), np.array([r["king_mase"] for r in rows]))
    cand = (np.array([r["crps"] for r in rows]), np.array([r["mase"] for r in rows]))
    assert d["crown_bar"]["bar"] == pytest.approx(ld.crown_bar(32)) == pytest.approx(0.01)
    newest = hashlib.sha256((tmp_path / "series.npz").read_bytes()).hexdigest()
    (want,) = judge_cohort(king, [cand], clusters=[r["source"] for r in rows], cfg=ld,
                           seed=f"lane-d/1:local:3:{newest}", bar=ld.crown_bar(32))
    assert d["verdict"]["lcb"] == pytest.approx(want["lcb"])
    assert d["verdict"]["cleared"] == want["cleared"]


def test_a_worse_model_does_not_clear(tmp_path, run):
    rc, doc = run(_ckpt(tmp_path, "cand", "flat"), "--king", _ckpt(tmp_path, "king", "seasonal"),
                  "--context", 4096, "--series", _series(tmp_path),
                  "--windows-per-series", PER_SERIES)
    assert rc == 0, doc
    d = doc["data"]
    assert not d["verdict"]["cleared"] and not d["would_clear"]
    assert d["verdict"]["lcb"] < 0.0


def test_a_copy_of_the_king_does_not_clear(tmp_path, run):
    rc, doc = run(_ckpt(tmp_path, "cand", "flat"), "--king", _ckpt(tmp_path, "king", "flat"),
                  "--context", 4096, "--series", _series(tmp_path),
                  "--windows-per-series", PER_SERIES)
    assert rc == 0, doc
    assert not doc["data"]["verdict"]["cleared"]
    assert doc["data"]["verdict"]["gain"] == pytest.approx(0.0)


def test_below_the_floors_never_reports_a_clear(tmp_path, run):
    rc, doc = run(_ckpt(tmp_path, "cand", "seasonal"), "--king", _ckpt(tmp_path, "king", "flat"),
                  "--context", 4096, "--series", _series(tmp_path, sources=4, per_source=1),
                  "--windows-per-series", 3)
    assert rc == 0, doc
    d = doc["data"]
    assert not d["floors"]["met"] and not d["would_clear"]


def test_public_weight_needs_a_public_set_and_then_scores_both_halves(tmp_path, run):
    cand, king = _ckpt(tmp_path, "cand", "seasonal"), _ckpt(tmp_path, "king", "flat")
    series = _series(tmp_path)
    rc, doc = run(cand, "--king", king, "--context", 4096, "--series", series,
                  "--windows-per-series", PER_SERIES, "--public-weight", 0.5)
    assert rc != 0 and doc["error"]["code"] == "windows_missing"
    rc, doc = run(cand, "--king", king, "--context", 4096, "--series", series,
                  "--windows-per-series", PER_SERIES, "--public-weight", 0.5,
                  "--public-series", _series(tmp_path, sources=5, name="pub.npz"))
    assert rc == 0, doc
    assert set(doc["data"]["verdict"]["halves"]) == {"live", "public"}
    assert doc["data"]["public_weight"] == 0.5


def test_an_undeclarable_context_is_refused(tmp_path, run):
    rc, doc = run(_ckpt(tmp_path, "cand", "seasonal"), "--king", _ckpt(tmp_path, "king", "flat"),
                  "--context", 5000, "--series", _series(tmp_path, sources=2))
    assert rc != 0 and doc["error"]["code"] == "bad_context"


def test_context_below_the_armed_set_is_the_validators_skip(tmp_path, run):
    rc, doc = run(_ckpt(tmp_path, "cand", "seasonal"), "--king", _ckpt(tmp_path, "king", "flat"),
                  "--context", 4096, "--armed-contexts", "8192",
                  "--series", _series(tmp_path, sources=2))
    assert rc != 0 and doc["error"]["code"] == "context_below_armed"
