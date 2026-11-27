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
    from cascade.eval.lane_d_scorer import SCORER_CONTEXTS  # noqa: I001
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
    """Named for the pre-band cap; the cap is now 2048 (owner 2026-11-27)."""
    s = _scorer()
    assert getattr(s, "MAX_HORIZON", None) == MAX_HORIZON, (
        "lane_d_scorer.MAX_HORIZON = 2048 (64 patches x 32, the band cap; owner 2026-11-27)")
    fq = _w5d_forecast()
    absent = tmp_path / "no-checkpoint-here"
    for h in (MAX_HORIZON + 1, 4096, 0):
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


# ── #785: settlement-size micro-batching ─────────────────────────────────────
# A settlement draws 1000-5000 windows; one forward over every window of a
# length put windows x patches past the CUDA grid limit (~65,536 rows into
# the pyramid's SDPA: "invalid configuration argument") or out of memory on
# the launch 4090 (testnet E2E, issue #785). Each length's batch now runs in
# deterministic micro-batches of forward_batch_size windows.

CUDA_GRID_ROWS = 65_535          # the measured failure threshold (#785 table)


def _microbatch_forecast():
    """``forecast_quantiles`` with the #785 keyword ``max_tokens`` (asserted,
    so the pre-#785 version fails here on an assertion, not a TypeError)."""
    fq = _w5d_forecast()
    params = inspect.signature(fq).parameters
    assert "max_tokens" in params, (
        "#785: forecast_quantiles(..., max_tokens=MAX_FORWARD_TOKENS) micro-batches each "
        "length's windows")
    assert params["max_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["max_tokens"].default == _scorer().MAX_FORWARD_TOKENS
    return fq


def test_forward_batch_size_is_tokens_over_patches():
    s = _scorer()
    assert s.MAX_FORWARD_TOKENS == 32768
    extra = (HORIZON + PATCH - 1) // PATCH
    # 4096 history + 2 horizon patches = 130 tokens a window: 252 windows.
    assert s.forward_batch_size(4096, HORIZON, PATCH) == 32768 // (128 + extra) == 252
    assert s.forward_batch_size(16384, HORIZON, PATCH) == 32768 // (512 + extra)
    # A partial last patch counts as a patch; channels multiply the tokens.
    assert s.forward_batch_size(4096, 33, PATCH) == 32768 // 130
    assert s.forward_batch_size(4096, HORIZON, PATCH, channels=4) == 32768 // (4 * 130)
    assert s.forward_batch_size(4096, HORIZON, PATCH, max_tokens=260) == 2
    # Never zero: a window bigger than the cap still runs, alone.
    assert s.forward_batch_size(131072, MAX_HORIZON, PATCH, channels=32) == 1
    for bad in ({"input_len": 0}, {"horizon": 0}, {"patch_size": 0}, {"channels": 0},
                {"max_tokens": 0}, {"max_tokens": True}, {"input_len": 4096.0}):
        kw = {"input_len": 4096, "horizon": HORIZON, "patch_size": PATCH, **bad}
        with pytest.raises(ValueError):
            s.forward_batch_size(kw.pop("input_len"), kw.pop("horizon"), kw.pop("patch_size"),
                                 **kw)


def test_forward_batch_stays_under_the_grid_limit_at_every_context():
    """Every scorer context, the launch horizon and the cap: a micro-batch's
    rows (windows x patches) stay under half the measured CUDA limit, and the
    #785 failures (1000 windows at 4096 = 130,000 rows) are cut below it."""
    s = _scorer()
    for c in ALL_CONTEXTS:
        for h in (1, HORIZON, MAX_HORIZON):
            for ch in (1, 2, 32):
                patches = c // PATCH + -(-h // PATCH)
                step = s.forward_batch_size(c, h, PATCH, channels=ch)
                assert step >= 1
                assert step * ch * patches <= s.MAX_FORWARD_TOKENS or step == 1, (c, h, ch)
                assert step * patches < CUDA_GRID_ROWS, (c, h, ch)
    assert CUDA_GRID_ROWS < 1000 * 130                     # the #785 failure, unchunked
    assert s.forward_batch_size(4096, HORIZON, PATCH) * 130 <= 32768


def test_forecast_refuses_a_non_positive_max_tokens_before_loading(no_torch, tmp_path):
    fq = _microbatch_forecast()
    absent = tmp_path / "no-checkpoint-here"
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            fq(absent, [series(256, 0)], context=TRAINED_CONTEXT, horizon=HORIZON,
               max_tokens=bad)


@torch_mark
def test_micro_batched_forecast_is_deterministic_and_re_derivable(ckpts):
    """#785: seven 4096-point windows and three 320-point windows.

    * A cap that cuts nothing (the default, or exactly the batch's tokens)
      is byte-identical to the pre-#785 one forward per length: small inputs
      are unchanged.
    * A cut into 1, 2 or 3 windows a forward is byte-identical to forecasting
      each micro-batch's windows on their own: chunks are contiguous, in
      window order, with no state across them, so the forecast is
      re-derivable from the cap alone.
    * Against the uncut forecast it agrees to float rounding only (CPU fp32
      GEMM blocking depends on the batch rows; measured in CI: equal to ~8
      digits, not bit-equal). That is why the cap is the release constant
      MAX_FORWARD_TOKENS and never a runtime memory probe."""
    fq = _microbatch_forecast()
    s = _scorer()
    hists = [series(TRAINED_CONTEXT + 17 * i, seed=200 + i) for i in range(7)]
    hists[3:3] = [series(320 + i, seed=300 + i) for i in range(3)]
    long_idx, short_idx = [0, 1, 2, 6, 7, 8, 9], [3, 4, 5]
    extra = (HORIZON + PATCH - 1) // PATCH
    kw = {"context": TRAINED_CONTEXT, "horizon": HORIZON}
    with ModelInputs() as seen:
        whole = np.asarray(fq(ckpts["entry"], hists, **kw))
    assert sorted(seen.shapes) == sorted([(3, 1, 10 + extra, PATCH),
                                          (7, 1, 128 + extra, PATCH)])
    exact = np.asarray(fq(ckpts["entry"], hists, max_tokens=7 * (128 + extra), **kw))
    assert np.array_equal(exact, whole), "a cap that cuts nothing changed the forecast"
    for per in (1, 2, 3):
        cap = per * (128 + extra)
        with ModelInputs() as seen:
            q = np.asarray(fq(ckpts["entry"], hists, max_tokens=cap, **kw))
        assert q.shape == whole.shape == (10, HORIZON, len(SCORING_LEVELS))
        long_batches = [b for b, _, p, _ in seen.shapes if p == 128 + extra]
        assert sum(long_batches) == 7 and max(long_batches) == per
        short_step = s.forward_batch_size(320, HORIZON, PATCH, max_tokens=cap)
        short_batches = [b for b, _, p, _ in seen.shapes if p == 10 + extra]
        assert sum(short_batches) == 3 and max(short_batches) == min(3, short_step)
        # Re-derivation: each micro-batch forecast alone, in order.
        want = np.empty_like(q)
        for idx, step in ((long_idx, per), (short_idx, short_step)):
            for j in range(0, len(idx), step):
                part = idx[j:j + step]
                want[part] = np.asarray(fq(ckpts["entry"], [hists[i] for i in part], **kw))
        assert np.array_equal(q, want), f"{per} a forward is not its micro-batches alone"
        np.testing.assert_allclose(q, whole, rtol=1e-5, atol=1e-6,
                                   err_msg=f"{per} a forward moved past float rounding")
