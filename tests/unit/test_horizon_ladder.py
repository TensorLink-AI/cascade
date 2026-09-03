"""Scored horizon ladder (``[eval] scored_horizons`` / ``scored_from_block``).

Block-gated multi-horizon verdict draw: each horizon gets its own seeded
even-by-domain draw, rung sizes are equalised, and the pooled rows flow
through the unchanged round statistic. Inert by default — with the gate at 0
every path is byte-identical to the single-horizon draw.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.validator.windows import (
    ladder_windows,
    ladder_windows_for_round,
    scored_ladder,
)

# ── config round-trip (every knob needs the loader parse) ────────────────────


def test_scored_knobs_parse_and_default_off(tmp_path):
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config

    src = DEFAULT_CHAIN_TOML.read_text()
    p = tmp_path / "chain.toml"
    p.write_text(src)
    cfg = load_chain_config(p)
    assert cfg.eval.scored_horizons == ()         # inert default
    assert cfg.eval.scored_from_block == 0

    p.write_text(src.replace(
        "\n[eval]\n",
        "\n[eval]\nscored_horizons = [64, 256, 720]\n"
        "scored_from_block = 9000000\n", 1))
    cfg = load_chain_config(p)
    assert cfg.eval.scored_horizons == (64, 256, 720)
    assert cfg.eval.scored_from_block == 9000000


# ── the activation gate ──────────────────────────────────────────────────────


def _eval_cfg(horizons=(), from_block=0):
    return SimpleNamespace(scored_horizons=horizons, scored_from_block=from_block)


def test_scored_ladder_block_gate():
    armed = _eval_cfg((64, 720), 1000)
    assert scored_ladder(armed, 1000) == (64, 720)
    assert scored_ladder(armed, 5000) == (64, 720)
    assert scored_ladder(armed, 999) == ()        # pre-gate rounds: legacy rule
    assert scored_ladder(armed, None) == ()
    assert scored_ladder(_eval_cfg((64, 720), 0), 5000) == ()   # no gate block
    assert scored_ladder(_eval_cfg((), 1000), 5000) == ()       # no horizons


# ── the ladder draw: equal rungs, prefixes, determinism ──────────────────────


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


def test_ladder_draws_equal_rungs_with_horizon_prefixes():
    series, metadata = _series_set()
    windows = ladder_windows(series, metadata, horizons=(64, 720),
                             n_windows=24, context_length=4096, seed=3)
    by_h = {64: [w for w in windows if w.series_id.startswith("h64-")],
            720: [w for w in windows if w.series_id.startswith("h720-")]}
    assert len(by_h[64]) == len(by_h[720]) == 12
    assert all(w.target.shape[-1] == 64 for w in by_h[64])
    assert all(w.target.shape[-1] == 720 for w in by_h[720])
    # deterministic in seed
    again = ladder_windows(series, metadata, horizons=(64, 720),
                           n_windows=24, context_length=4096, seed=3)
    assert [w.series_id for w in windows] == [w.series_id for w in again]
    other = ladder_windows(series, metadata, horizons=(64, 720),
                           n_windows=24, context_length=4096, seed=4)
    assert [w.series_id for w in windows] != [w.series_id for w in other]


def test_ladder_equalises_to_the_scarcest_rung():
    series, metadata = _series_set()
    # per-rung ask is 14, but only 12 series are long enough for h=720:
    # both rungs truncate to 12 so the pooled geomean weights them equally.
    windows = ladder_windows(series, metadata, horizons=(64, 720),
                             n_windows=28, context_length=4096, seed=3)
    n64 = sum(w.series_id.startswith("h64-") for w in windows)
    n720 = sum(w.series_id.startswith("h720-") for w in windows)
    assert n64 == n720 == 12


def test_ladder_equalisation_keeps_each_rung_even_by_domain():
    # The scarce rung sets the size; the OTHER rungs must be drawn at that
    # size even-by-domain — not drawn at the full ask and truncated by pool
    # index, which keeps filename order and starves whichever domain sorts
    # last (review 2026-09-02). Build a pool where domain "b" sorts after
    # "a" and only some series are long enough for h=720.
    import numpy as np

    rng = np.random.default_rng(0)
    series, metadata = [], []
    for i in range(20):        # domain a: all long (eligible at 720)
        series.append(rng.normal(size=(1, 1200))); metadata.append({"domain": "a", "source": f"a{i}"})
    for i in range(20):        # domain b: half long, half short
        L = 1200 if i % 2 == 0 else 400
        series.append(rng.normal(size=(1, L))); metadata.append({"domain": "b", "source": f"b{i}"})
    windows = ladder_windows(series, metadata, horizons=(64, 720),
                             n_windows=40, context_length=256, seed=7)
    by_rung = {}
    for w in windows:
        rung, idx = w.series_id.split("-s")
        by_rung.setdefault(rung, []).append(metadata[int(idx)]["domain"])
    # 720 eligible: a=20, b=10 → 30; per-rung ask 20 → m = 20 on both rungs.
    assert len(by_rung["h64"]) == len(by_rung["h720"]) == 20
    # Even-by-domain at size 20 on the h64 rung: 10 a + 10 b (all 20 b are
    # eligible at h=64). Index-order truncation would have given 20 a + 0 b.
    assert by_rung["h64"].count("b") == 10
    assert by_rung["h720"].count("b") == 10


def test_ladder_raises_when_a_rung_has_no_eligible_series():
    series, metadata = _series_set()
    with pytest.raises(ValueError, match="5000"):
        ladder_windows(series, metadata, horizons=(64, 5000),
                       n_windows=8, context_length=4096, seed=1)
    with pytest.raises(ValueError, match="non-empty"):
        ladder_windows(series, metadata, horizons=(),
                       n_windows=8, context_length=4096, seed=1)


# ── equal rungs ⇒ pooled statistic == blend of per-rung statistics ───────────


def test_pooled_geomean_equals_blend_of_rung_geomeans():
    from cascade.eval.scoring import WindowScore, global_geomean

    rng = np.random.default_rng(11)

    def rung(prefix, n):
        return [WindowScore(series_id=f"{prefix}s{i}",
                            mase=float(rng.uniform(0.5, 2.0)),
                            qloss_per_q=rng.uniform(0.1, 1.0, size=9),
                            abs_target=float(rng.uniform(5.0, 50.0)))
                for i in range(n)]

    r64, r720 = rung("h64-", 10), rung("h720-", 10)
    pooled = global_geomean(r64 + r720)
    blend = float(np.sqrt(global_geomean(r64) * global_geomean(r720)))
    assert pooled == pytest.approx(blend, rel=1e-12)


# ── source plumbing + the validator branch ───────────────────────────────────


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


def test_ladder_windows_for_round_draws_from_the_snapshot_dir(tmp_path, cfg):
    from cascade.validator.pool import window_source_from_dir

    src = window_source_from_dir(_pool_dir(tmp_path), cfg, label="test")
    windows = ladder_windows_for_round(
        src, horizons=(64, 720), n_windows=16, context_length=4096,
        round_seed=12345, block=None)
    assert sum(w.series_id.startswith("h64-") for w in windows) == \
        sum(w.series_id.startswith("h720-") for w in windows) == 8


def test_ladder_windows_for_round_errors_without_a_series_dir():
    # A source with no raw-series directory cannot serve a ladder round; a
    # silent fall-back to the single-horizon draw would fork the fleet.
    with pytest.raises(ValueError, match="series directory"):
        ladder_windows_for_round(
            object(), horizons=(64,), n_windows=8, context_length=4096,
            round_seed=1, block=None)


def test_verdict_windows_branches_on_the_gate(tmp_path, cfg):
    from dataclasses import replace

    from cascade.validator.loop import ValidatorRunner
    from cascade.validator.pool import window_source_from_dir

    src = window_source_from_dir(_pool_dir(tmp_path), cfg, label="test")

    stub = SimpleNamespace(cfg=cfg)
    legacy = ValidatorRunner._verdict_windows(stub, src, 12345, block=5000)
    assert not any(w.series_id.startswith("h") for w in legacy)

    armed = replace(cfg, eval=replace(
        cfg.eval, scored_horizons=(64, 720), scored_from_block=1000,
        n_windows=16))
    stub = SimpleNamespace(cfg=armed)
    ladder = ValidatorRunner._verdict_windows(stub, src, 12345, block=5000)
    assert sum(w.series_id.startswith("h64-") for w in ladder) == \
        sum(w.series_id.startswith("h720-") for w in ladder) == 8
    # pre-gate rounds keep the legacy draw even when the knobs are set
    pre = ValidatorRunner._verdict_windows(stub, src, 12345, block=999)
    assert [w.series_id for w in pre] == [w.series_id for w in legacy][:len(pre)]
