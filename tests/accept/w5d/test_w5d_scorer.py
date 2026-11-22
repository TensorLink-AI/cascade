"""W5d: the no-pad lane D scorer, its context variants and its quantile-head
scoring (plan rev6 §3.0 W5d row, §1.1, §4 T7 CI half). Contract: README.md.

Tests without the ``torch`` mark never import torch: the ones that call
``forecast_quantiles`` make ``import torch`` fail (``sys.modules["torch"] =
None``), so the scorer must refuse a bad context or horizon before it loads
anything. Tests marked ``torch`` build tiny 2m pyramid checkpoints with the
repo's model code; CI runs them (``uv sync --all-extras``), the production
box never does (``-m 'not torch'``).
"""

from __future__ import annotations

import inspect
import shutil
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from .helpers import (
    ALL_CONTEXTS,
    DECLARED_128K,
    HEAD_LEVELS,
    HORIZON,
    LAUNCH_CONTEXTS,
    MAX_HORIZON,
    PATCH,
    SCORING_LEVELS,
    SEASONAL_PERIOD,
    TRAINED_CONTEXT,
    ModelInputs,
    lane_d_cfg,
    pyramid_contract,
    series,
    sorted_quantiles,
    tiny_checkpoint,
)

torch_mark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)


@pytest.fixture
def no_torch(monkeypatch):
    """``import torch`` raises inside the test: nothing may load a model."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)


def _scorer():
    import cascade.eval.lane_d_scorer as scorer

    return scorer


def _w5d_forecast():
    """``forecast_quantiles`` with the W5d keyword ``contexts`` (asserted, so
    the W1.5 version fails here on an assertion, not on a TypeError)."""
    fq = _scorer().forecast_quantiles
    params = inspect.signature(fq).parameters
    assert "contexts" in params, (
        "W5d: forecast_quantiles(checkpoint_dir, histories, *, context, horizon, "
        "contexts=None) takes the armed [lane_d] contexts")
    assert params["contexts"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["contexts"].default is None
    return fq


# ── quantile head at the 9-level grid, CRPS as lane A ────────────────────────


def test_scoring_levels_are_lane_a_quantile_head_grid():
    from cascade.eval.crps import DEFAULT_QUANTILE_LEVELS

    s = _scorer()
    assert tuple(s.SCORING_LEVELS) == SCORING_LEVELS == tuple(DEFAULT_QUANTILE_LEVELS)
    assert s.SCORING_METHOD == "quantile_head"
    assert s.SCORING_LEVELS[s.MEDIAN_INDEX] == 0.5


def test_level_indices_pick_exactly_the_nine_levels_from_the_head():
    s = _scorer()
    picks = s.level_indices(HEAD_LEVELS)
    assert picks == [2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert [HEAD_LEVELS[i] for i in picks] == pytest.approx(list(SCORING_LEVELS), abs=0)
    with pytest.raises(ValueError):
        s.level_indices([q for q in HEAD_LEVELS if q != 0.5])


def test_score_window_crps_and_mase_equal_lane_a_per_window():
    """The lane D per-window CRPS is lane A's under quantile_head_after:
    ``score_joint_quantile_forecaster_on_windows`` + ``wql_per_window``."""
    from cascade.eval.crps import wql_per_window
    from cascade.eval.scoring import score_joint_quantile_forecaster_on_windows, stack_components
    from cascade.eval.window import EvalWindow

    s = _scorer()
    rng = np.random.default_rng(5)
    windows, qs = [], []
    for i in range(16):
        hist = series(700 + 13 * i, seed=i)
        tgt = series(HORIZON, seed=100 + i, level=float(hist[-1]))
        q = sorted_quantiles(rng, HORIZON, float(hist[-24:].mean()))
        windows.append(EvalWindow(series_id=f"s{i}", history=hist, target=tgt,
                                  metadata={"seasonal_period": SEASONAL_PERIOD}))
        qs.append(q)
    for w, q in zip(windows, qs, strict=True):
        lane_a = score_joint_quantile_forecaster_on_windows(lambda *_a, q=q: q[None], [w],
                                                            SCORING_LEVELS)
        qloss, abs_t, mase_a = stack_components(lane_a)
        wql, valid = wql_per_window(qloss, abs_t)
        assert bool(valid[0])
        crps, m = s.score_window(q, w.target[0], w.history[0], SEASONAL_PERIOD)
        assert crps == pytest.approx(float(wql[0]), rel=1e-12)
        assert m == pytest.approx(float(mase_a[0]), rel=1e-12)


def test_score_window_refuses_a_grid_other_than_nine_levels():
    s = _scorer()
    rng = np.random.default_rng(1)
    q = sorted_quantiles(rng, HORIZON, 20.0)
    hist, tgt = series(256, 1), series(HORIZON, 2)
    with pytest.raises(ValueError):
        s.score_window(q[:, :8], tgt, hist, SEASONAL_PERIOD)
    with pytest.raises(ValueError):
        s.score_window(np.concatenate([q, q[:, :1]], axis=1), tgt, hist, SEASONAL_PERIOD)


# ── context variants ─────────────────────────────────────────────────────────


def test_scorer_contexts_cover_every_lane_d_context():
    from cascade.eval.lane_d_scorer import SCORER_CONTEXTS

    from cascade.shared.config.lane_d import LANE_D_CONTEXTS

    assert tuple(SCORER_CONTEXTS) == tuple(sorted(SCORER_CONTEXTS))
    assert set(LANE_D_CONTEXTS) <= set(SCORER_CONTEXTS)
    for c in SCORER_CONTEXTS:
        assert type(c) is int and c % PATCH == 0 and TRAINED_CONTEXT <= c <= DECLARED_128K, c


def test_each_lane_d_context_is_a_declared_scorer_context():
    from cascade.eval.lane_d_scorer import declared_context

    cfg = lane_d_cfg()
    assert tuple(cfg.contexts) == LAUNCH_CONTEXTS
    for c in cfg.contexts:
        assert declared_context(c, cfg.contexts) == c
    s8 = lane_d_cfg(contexts=list(ALL_CONTEXTS))
    for c in s8.contexts:
        assert declared_context(c, s8.contexts) == c


def test_undeclared_context_is_refused():
    from cascade.eval.lane_d_scorer import declared_context

    launch = lane_d_cfg().contexts
    for c in (32768, 65536, 131072):          # armed only at S8, by overlay
        with pytest.raises(ValueError):
            declared_context(c, launch)
    for c in (2048, 4095, 4097, 12288, 0, -4096, 262144):
        with pytest.raises(ValueError):
            declared_context(c, ALL_CONTEXTS)
    with pytest.raises(ValueError):
        declared_context(8192, (4096,))


def test_forecast_quantiles_keeps_the_w1_5_call_and_adds_contexts():
    params = inspect.signature(_w5d_forecast()).parameters
    assert list(params)[:2] == ["checkpoint_dir", "histories"]
    for k in ("context", "horizon"):
        assert params[k].kind is inspect.Parameter.KEYWORD_ONLY


def test_forecast_refuses_undeclared_context_before_loading(no_torch, tmp_path):
    fq = _w5d_forecast()
    hist = [series(256, 0)]
    absent = tmp_path / "no-checkpoint-here"
    with pytest.raises(ValueError):
        fq(absent, hist, context=12288, horizon=HORIZON)
    with pytest.raises(ValueError):
        fq(absent, hist, context=32768, horizon=HORIZON, contexts=LAUNCH_CONTEXTS)
    with pytest.raises(ValueError):
        fq(absent, hist, context=8192, horizon=HORIZON, contexts=(4096,))


def test_forecast_refuses_horizon_over_512_before_loading(no_torch, tmp_path):
    s = _scorer()
    assert getattr(s, "MAX_HORIZON", None) == MAX_HORIZON, (
        "W5d: lane_d_scorer.MAX_HORIZON = 512 (16 CPM patches x 32, §1.1)")
    fq = _w5d_forecast()
    absent = tmp_path / "no-checkpoint-here"
    for h in (MAX_HORIZON + 1, 720, 0):
        with pytest.raises(ValueError):
            fq(absent, [series(256, 0)], context=TRAINED_CONTEXT, horizon=h)


# ── no padding: the batch plan ───────────────────────────────────────────────


def test_batch_plan_is_real_history_capped_at_context():
    from cascade.eval.lane_d_scorer import batch_plan

    lengths = [100, 4096, 5000, 64, 4100, 32, 320]
    plan = batch_plan(lengths, context=4096, patch_size=PATCH)
    assert plan == {32: [5], 64: [3], 96: [0], 320: [6], 4096: [1, 2, 4]}
    assert list(plan) == sorted(plan)
    plan16 = batch_plan(lengths, context=16384, patch_size=PATCH)
    assert plan16 == {32: [5], 64: [3], 96: [0], 320: [6], 4096: [1, 4], 4992: [2]}


def test_batch_plan_never_pads_a_short_history_up_to_the_context():
    from cascade.eval.lane_d_scorer import batch_plan

    lengths = [33, 640, 1000, 4096]
    for c in ALL_CONTEXTS:
        plan = batch_plan(lengths, context=c, patch_size=PATCH)
        seen = sorted(i for idx in plan.values() for i in idx)
        assert seen == list(range(len(lengths)))
        for length, idx in plan.items():
            assert length % PATCH == 0
            for i in idx:
                assert length <= min(lengths[i], c)
                assert length == (min(lengths[i], c) // PATCH) * PATCH
        assert max(plan) == 4096 < c or c == 4096


def test_batch_plan_refuses_a_history_shorter_than_one_patch():
    from cascade.eval.lane_d_scorer import batch_plan

    with pytest.raises(ValueError):
        batch_plan([64, 31], context=4096, patch_size=PATCH)
    with pytest.raises(ValueError):
        batch_plan([0], context=4096, patch_size=PATCH)


# ── torch: the real scorer on tiny models ────────────────────────────────────


@lru_cache(maxsize=1)
def _checkpoints(root: str) -> dict[str, Path]:
    """The tiny F32 house (+3 median shift, like a lane A king) and a BF16
    entry, built once per session under ``root``."""
    contract = pyramid_contract()
    base = Path(root)
    return {
        "house": tiny_checkpoint(base / "house", contract, seed=12, median_shift=3.0, bf16=False),
        "entry": tiny_checkpoint(base / "entry", contract, seed=11, median_shift=0.0, bf16=True),
    }


@pytest.fixture
def ckpts(tmp_path_factory):
    import torch

    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield _checkpoints(str(tmp_path_factory.getbasetemp() / "w5d_ckpts"))
    finally:
        torch.set_num_threads(threads)


@torch_mark
def test_no_pad_model_input_is_real_history_capped_at_context(ckpts):
    from cascade.eval.lane_d_scorer import batch_plan  # noqa: F401  (W5d surface)

    fq = _w5d_forecast()
    hists = [series(n, seed=n) for n in (96, 320, 4096, 5000)]
    extra = (HORIZON + PATCH - 1) // PATCH
    for context, want in ((4096, [(1, 1, 3 + extra, PATCH), (1, 1, 10 + extra, PATCH),
                                  (2, 1, 128 + extra, PATCH)]),
                          (16384, [(1, 1, 3 + extra, PATCH), (1, 1, 10 + extra, PATCH),
                                   (1, 1, 128 + extra, PATCH), (1, 1, 156 + extra, PATCH)])):
        with ModelInputs() as seen:
            q = fq(ckpts["entry"], hists, context=context, horizon=HORIZON,
                   contexts=LAUNCH_CONTEXTS)
        assert np.asarray(q).shape == (4, HORIZON, len(SCORING_LEVELS))
        assert sorted(seen.shapes) == sorted(want), (context, seen.shapes)


@torch_mark
def test_no_pad_truncates_to_the_largest_patch_multiple(ckpts):
    fq = _w5d_forecast()
    long = series(330, seed=3)
    a = fq(ckpts["entry"], [long], context=4096, horizon=HORIZON)
    b = fq(ckpts["entry"], [long[-320:]], context=4096, horizon=HORIZON)
    assert np.array_equal(np.asarray(a), np.asarray(b))
    padded = np.concatenate([np.full(4096 - 320, long[-320]), long[-320:]])
    c = fq(ckpts["entry"], [padded], context=4096, horizon=HORIZON)
    assert not np.array_equal(np.asarray(b), np.asarray(c)), (
        "a 320-point window scored like its edge-padded 4096-point copy: it was padded")


@torch_mark
def test_t7_4k_checkpoint_and_declared_128k_copy_identical_on_short_windows(ckpts, tmp_path):
    fq = _w5d_forecast()
    copy = Path(shutil.copytree(ckpts["entry"], tmp_path / "declared-128k"))
    hists = [series(n, seed=n) for n in (32, 100, 1000, 2055, 4096, 4096)]
    base = np.asarray(fq(ckpts["entry"], hists, context=TRAINED_CONTEXT, horizon=HORIZON,
                         contexts=ALL_CONTEXTS))
    assert base.shape == (len(hists), HORIZON, len(SCORING_LEVELS)) and np.isfinite(base).all()
    for c in (*lane_d_cfg().contexts, DECLARED_128K):
        other = np.asarray(fq(copy, hists, context=c, horizon=HORIZON, contexts=ALL_CONTEXTS))
        assert np.array_equal(base, other), f"context {c} differs on ≤ 4096-history windows"


@torch_mark
def test_house_at_4096_equals_its_lane_a_per_window_crps(ckpts):
    """(rev5) The house scored by the lane D scorer at 4096 equals its lane A
    per-window CRPS (``[v2] quantile_head_after``) on the same windows."""
    from cascade.eval.crps import wql_per_window
    from cascade.eval.scoring import stack_components
    from cascade.eval.window import EvalWindow
    from cascade.validator.evaluator import evaluate_checkpoint

    fq = _w5d_forecast()
    s = _scorer()
    windows = []
    for i in range(12):
        hist = series(TRAINED_CONTEXT + 37 * i, seed=40 + i)
        tgt = series(HORIZON, seed=80 + i, level=float(hist[-1]))
        windows.append(EvalWindow(series_id=f"w{i}", history=hist, target=tgt,
                                  metadata={"seasonal_period": SEASONAL_PERIOD}))
    lane_a = evaluate_checkpoint(ckpts["house"], windows, num_samples=1, quantile_head=True,
                                 trust_checkpoint_code=True)
    qloss, abs_t, mase_a = stack_components(lane_a)
    wql, valid = wql_per_window(qloss, abs_t)
    assert valid.all() and len(lane_a) == len(windows)
    q = np.asarray(fq(ckpts["house"], [w.history[0] for w in windows], context=TRAINED_CONTEXT,
                      horizon=HORIZON, contexts=LAUNCH_CONTEXTS))
    assert q.shape == (len(windows), HORIZON, len(SCORING_LEVELS))
    for i, w in enumerate(windows):
        crps, m = s.score_window(q[i], w.target[0], w.history[0], SEASONAL_PERIOD)
        assert crps == pytest.approx(float(wql[i]), rel=1e-5), (i, crps, float(wql[i]))
        assert m == pytest.approx(float(mase_a[i]), rel=1e-5), (i, m, float(mase_a[i]))
