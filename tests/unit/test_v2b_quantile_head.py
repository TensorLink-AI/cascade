"""DEC-CA-0067 bundle item 3 (``[v2] quantile_head_after``): from the v2b
switch era a v2 settlement scores each checkpoint's QUANTILE HEAD directly —
pinball loss on the wrapper's own quantiles at the 9-level grid, no copula
sampling — exactly as the benchmark path already does. The knob is part of the
v2b feature hash; without it everything is byte-identical to before.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cascade.shared.activation as A
from cascade.eval.crps import DEFAULT_QUANTILE_LEVELS, mwsql_components, pinball_components
from cascade.eval.scoring import (
    score_joint_forecaster_on_windows,
    score_joint_quantile_forecaster_on_windows,
)
from cascade.eval.window import EvalWindow
from cascade.shared.config import effective_epoch_blocks, load_chain_config
from cascade.shared.era import era_length_blocks, v2_eval_windows_at, v2_quantile_head_at

ROOT = Path(__file__).resolve().parents[2]
Q21 = (0.01, *(i / 20 for i in range(1, 20)), 0.99)


def _windows(n=5, h=6, seed=0, channels=(1, 1, 2, 1, 3)):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        c = channels[i % len(channels)]
        hist = np.cumsum(rng.standard_normal((c, 40)), axis=-1)
        tgt = hist[:, -1:] + np.cumsum(rng.standard_normal((c, h)), axis=-1)
        out.append(EvalWindow(f"w{i}", hist, tgt, {"source": "private", "seasonal_period": 2}))
    return out


def _same_rows(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert (x.series_id, x.channel, x.source, x.domain) == (y.series_id, y.channel, y.source, y.domain)
        assert x.quantile_levels == y.quantile_levels
        np.testing.assert_allclose(x.qloss_per_q, y.qloss_per_q, rtol=0, atol=1e-12)
        assert x.abs_target == y.abs_target
        np.testing.assert_allclose(x.mase, y.mase, rtol=0, atol=1e-12)


# ── the scorer: the same pinball arithmetic, minus the Monte-Carlo step ──────


def test_pinball_components_is_mwsql_without_the_sample_quantile_step():
    rng = np.random.default_rng(3)
    samples, obs = rng.normal(size=(4, 50, 7)), rng.normal(size=(4, 7))
    want = mwsql_components(samples, obs)
    got = pinball_components(np.quantile(samples, DEFAULT_QUANTILE_LEVELS, axis=1), obs)
    np.testing.assert_array_equal(got[0], want[0])
    np.testing.assert_array_equal(got[1], want[1])
    with pytest.raises(ValueError, match="num_q"):
        pinball_components(np.zeros((3, 4, 7)), obs)
    with pytest.raises(ValueError, match="obs shape"):
        pinball_components(np.zeros((9, 4, 6)), obs)


def test_quantile_scorer_reproduces_the_sample_scorer_on_the_samples_own_quantiles():
    """Feed the quantile scorer exactly the empirical quantiles the sample
    scorer computes internally: every row (qloss, |y|, MASE) must match, so a
    checkpoint whose head equals its sample marginals scores the same."""
    windows = _windows()
    drawn = {}

    def joint(history, horizon, num_samples):
        rng = np.random.default_rng(int(abs(history[0, -1] * 1e6)) % (2**32))
        s = history[:, -1:, None] + 0.3 * rng.standard_normal((history.shape[0], num_samples, horizon))
        drawn[history.shape] = s
        return s

    def quantiles(history, horizon):
        s = joint(history, horizon, 64)
        return np.transpose(np.quantile(s, DEFAULT_QUANTILE_LEVELS, axis=1), (1, 2, 0))

    by_samples = score_joint_forecaster_on_windows(joint, windows, 64)
    by_head = score_joint_quantile_forecaster_on_windows(quantiles, windows)
    _same_rows(by_samples, by_head)
    assert len(by_head) == sum(w.n_channels for w in windows)


def test_quantile_scorer_refuses_bad_output_and_a_grid_without_the_median():
    windows = _windows(n=1, channels=(2,))
    with pytest.raises(ValueError, match="returned shape"):
        score_joint_quantile_forecaster_on_windows(lambda h, hz: np.zeros((1, hz, 9)), windows)
    with pytest.raises(ValueError, match="returned shape"):
        score_joint_quantile_forecaster_on_windows(lambda h, hz: np.zeros((2, hz, 5)), windows)
    bad = np.zeros((2, 6, 9))
    bad[1, 2, 3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        score_joint_quantile_forecaster_on_windows(lambda h, hz: bad, windows)
    with pytest.raises(ValueError, match="0.5"):
        score_joint_quantile_forecaster_on_windows(lambda h, hz: np.zeros((2, hz, 4)), windows,
                                                   quantile_levels=(0.1, 0.3, 0.7, 0.9))


# ── the evaluator: picks the grid out of the head, same ingest path ──────────


def _fake_checkpoint(tmp_path, levels, *, with_method=True):
    d = tmp_path / "ckpt"
    d.mkdir()
    method = """
    def forecast_quantiles_batch(self, histories, horizon):
        out = []
        for h in histories:
            h = np.asarray(h, dtype=np.float64)
            out.append(np.tile(h[-1] + np.asarray(self.quantile_levels), (int(horizon), 1)))
        return np.stack(out)
""" if with_method else ""
    (d / "forecast_wrapper.py").write_text(f"""
import numpy as np

class Wrapper:
    quantile_levels = {list(levels)!r}

    def __init__(self, checkpoint_dir, device="cpu"):
        pass

    def forecast(self, history, horizon, num_samples):
        h = np.asarray(history, dtype=np.float64).reshape(-1)
        return np.full((1, num_samples, horizon), h[-1])
{method}""")
    return d


def test_load_quantile_forecaster_picks_the_grid_from_a_21_level_head(tmp_path):
    from cascade.validator.evaluator import load_quantile_forecaster, quantile_indices

    assert quantile_indices(Q21, DEFAULT_QUANTILE_LEVELS) == [2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert quantile_indices((0.1, 0.5, 0.9), DEFAULT_QUANTILE_LEVELS) is None
    assert quantile_indices(None, DEFAULT_QUANTILE_LEVELS) is None
    fn = load_quantile_forecaster(_fake_checkpoint(tmp_path, Q21), trust_checkpoint_code=True)
    hist = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    q = fn(hist, 4)
    assert q.shape == (2, 4, 9)
    np.testing.assert_allclose(q[0, 0], 3.0 + np.asarray(DEFAULT_QUANTILE_LEVELS))
    np.testing.assert_allclose(q[1, 3], 30.0 + np.asarray(DEFAULT_QUANTILE_LEVELS))
    # a 1-D history is one channel
    assert fn(np.array([1.0, 2.0]), 3).shape == (1, 3, 9)


@pytest.mark.parametrize("levels,with_method", [((0.1, 0.3, 0.7, 0.9), True), (Q21, False)])
def test_load_quantile_forecaster_refuses_a_head_without_the_grid(tmp_path, levels, with_method):
    from cascade.validator.evaluator import EvaluatorError, load_quantile_forecaster

    d = _fake_checkpoint(tmp_path, levels, with_method=with_method)
    with pytest.raises(EvaluatorError, match="quantile_head_unavailable"):
        load_quantile_forecaster(d, trust_checkpoint_code=True)


def test_evaluate_checkpoint_quantile_head_scores_the_head_with_and_without_workers(tmp_path):
    from cascade.validator.evaluator import evaluate_checkpoint

    d = _fake_checkpoint(tmp_path, Q21)
    windows = _windows()
    solo = evaluate_checkpoint(d, windows, num_samples=8, trust_checkpoint_code=True,
                               quantile_head=True)
    par = evaluate_checkpoint(d, windows, num_samples=8, trust_checkpoint_code=True,
                              quantile_head=True, workers=2)
    _same_rows(solo, par)
    # the head is the last value plus the level, so the 0.5 quantile is last + 0.5
    w0 = windows[0]
    assert solo[0].mase > 0 and solo[0].abs_target == float(np.abs(w0.target[0]).sum())
    # and the sample path is still the sample path (num_samples drives it)
    samp = evaluate_checkpoint(d, windows, num_samples=8, trust_checkpoint_code=True)
    assert len(samp) == len(solo)


# ── the rule: from the switch era, v2 settlements only, part of the bundle ──


@pytest.fixture
def switched():
    cfg = load_chain_config(ROOT / "chain.toml")
    assert cfg.v2.quantile_head_after
    era = era_length_blocks(cfg.round, 0)
    epoch = int(effective_epoch_blocks(cfg.round, 0))
    gate = 10_000 * era
    switch = gate + 3 * era
    cfg = replace(cfg, v2=replace(cfg.v2, from_block=gate, image_decoupled_from_block=switch))
    return cfg, gate, switch, epoch


def test_quantile_head_follows_the_switch_era_exactly_like_the_window_budget(switched):
    cfg, gate, switch, epoch = switched
    for boundary in (gate - epoch, gate, gate + epoch, switch - epoch, switch):
        assert not v2_quantile_head_at(cfg, boundary), boundary
        assert not cfg.quantile_head_at(boundary)
    for boundary in (switch + epoch, switch + 5 * epoch, switch + 40 * epoch):
        assert v2_quantile_head_at(cfg, boundary), boundary
        assert cfg.quantile_head_at(boundary)
        assert v2_eval_windows_at(cfg, boundary) == cfg.v2.eval_n_windows_after
    assert not v2_quantile_head_at(cfg, None) and not cfg.quantile_head_at(None)
    off = replace(cfg, v2=replace(cfg.v2, quantile_head_after=False))
    assert not any(v2_quantile_head_at(off, b) for b in (switch + epoch, switch + 40 * epoch))
    unswitched = replace(cfg, v2=replace(cfg.v2, image_decoupled_from_block=0))
    assert not v2_quantile_head_at(unswitched, switch + 40 * epoch)


def test_evaluate_reads_the_rule_at_the_settlement_boundary(switched, monkeypatch, tmp_path):
    import cascade.validator.evaluator as ev
    from cascade.validator.evaluation import EvaluationMixin

    cfg, gate, switch, epoch = switched
    seen = []

    def fake_evaluate(dest, windows, *, num_samples, device, contract, workers, quantile_head):
        seen.append(quantile_head)
        return []

    monkeypatch.setattr(ev, "evaluate_checkpoint", fake_evaluate)
    runner = SimpleNamespace(evaluate_fn=None, _fetch_checkpoint_dir=lambda _: tmp_path,
                             cfg=cfg, device="cpu")
    entry = SimpleNamespace(size=cfg.training.arch_preset)
    training = cfg.training
    EvaluationMixin._evaluate(runner, entry, [], training, block=switch)
    EvaluationMixin._evaluate(runner, entry, [], training, block=switch + epoch)
    EvaluationMixin._evaluate(runner, entry, [], training)          # no boundary: samples
    EvaluationMixin._evaluate(runner, entry, [], training, block=gate + epoch)
    assert seen == [False, True, False, False]


def test_every_validator_scoring_site_passes_its_boundary():
    """Each ``_evaluate`` call in the validator names ``block=`` — a site that
    forgot would silently score an era under the sample path."""
    import inspect

    import cascade.validator.evaluation as evaluation
    import cascade.validator.king_check as king_check
    import cascade.validator.lane_b as lane_b
    import cascade.validator.loop as loop
    import cascade.validator.merge_acceptance as merge_acceptance

    for mod in (loop, evaluation, lane_b, merge_acceptance, king_check):
        src = inspect.getsource(mod)
        calls = [m for m in re.finditer(r"\._evaluate\((?!self\b)", src)]
        assert calls, mod.__name__
        for m in calls:
            tail = src[m.end():m.end() + 300]
            assert "block=" in tail, (mod.__name__, tail[:120])


def test_knob_is_in_the_bundle_hash_and_round_trips_the_loader(tmp_path):
    cfg = load_chain_config(ROOT / "chain.toml")
    assert cfg.v2.quantile_head_after is True
    assert cfg.v2.image_decoupled_feature == A.v2b_feature_name_for(cfg.v2) == "v2b-b10591aa"
    spec = A.image_decoupled_feature(replace(cfg, v2=replace(cfg.v2, from_block=9_000_000)))
    assert spec is not None and spec.name == "v2b-b10591aa"
    # off ⇒ the pre-knob name, so a non-opting deployment is unchanged
    assert A.v2b_feature_name_for(replace(cfg.v2, quantile_head_after=False)) == "v2b-90bc5237"
    text = (ROOT / "chain.toml").read_text()
    p = tmp_path / "chain.toml"
    p.write_text(re.sub(r"(?m)^quantile_head_after = true$", 'quantile_head_after = "yes"', text))
    with pytest.raises(ValueError, match="quantile_head_after must be true/false"):
        load_chain_config(p)
    p.write_text(re.sub(r"(?m)^quantile_head_after = true$", "quantile_head_after = false",
                        text).replace('"v2b-b10591aa"', '"v2b-90bc5237"'))
    off = load_chain_config(p)
    assert off.v2.quantile_head_after is False and off.v2.image_decoupled_feature == "v2b-90bc5237"
    assert A.summary(cfg, A.ActivationRecord(), None, None, None, None)["image_decoupled"][
        "quantile_head_after"] is True


def test_flipping_the_knob_without_repinning_refuses_on_a_voting_node(tmp_path):
    from cascade.shared.config.loader import require_vote_pin

    require_vote_pin()          # reset by tests/conftest.py
    text = (ROOT / "chain.toml").read_text()
    p = tmp_path / "chain.toml"
    p.write_text(re.sub(r"(?m)^quantile_head_after = true$", "quantile_head_after = false", text))
    with pytest.raises(ValueError, match="quantile_head_after=false.*refusing to start"):
        load_chain_config(p)


# ── the private GPU transport: quantiles in, nothing else accepted ───────────


def test_private_worker_quantile_request_round_trips_on_the_real_pyramid_model(tmp_path):
    import torch

    from cascade.trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.pyramid_trainer import save_checkpoint
    from cascade.validator.evaluator import evaluate_checkpoint
    from cascade.validator.private_worker import (
        SCHEMA_QUANTILES,
        read_forecasts,
        run_request,
        write_request,
    )
    from scripts.pyramid_config import local_config

    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        from cascade.shared.config import parse_chain_config

        contract = parse_chain_config(local_config("2m")).training
        config = CascadeMoEConfig.from_contract(contract)
        with torch.random.fork_rng():
            torch.manual_seed(123)
            model = CascadeMoEModel(config)
        directory = tmp_path / "checkpoint"
        directory.mkdir()
        save_checkpoint(directory, model.state_dict(), config, contract)
        rng = np.random.default_rng(42)
        windows = [EvalWindow(str(i), rng.normal(size=(c, 64)), rng.normal(size=(c, 8)),
                             {"source": "private-source"}) for i, c in enumerate((1, 2))]
        levels = DEFAULT_QUANTILE_LEVELS
        request, output = tmp_path / "request.npz", tmp_path / "quantiles.npz"
        identity = write_request(request, directory, windows, contract=contract, num_samples=4,
                                 quantile_levels=levels)
        with np.load(request, allow_pickle=False) as payload:
            meta = payload["metadata"].tobytes().decode()
            assert SCHEMA_QUANTILES in meta and "private-source" not in meta
        report = run_request(directory, request, output, device="cpu", workers=2, qualify=True)
        assert report["qualified"] and report["schema"] == SCHEMA_QUANTILES
        arrays = read_forecasts(output, identity, windows, 4, levels)
        assert [x.shape for x in arrays] == [(1, 8, 9), (2, 8, 9)]
        assert all(np.isfinite(x).all() for x in arrays)
        # monotone in the level: a real quantile head, not samples
        assert all((np.diff(x, axis=-1) >= 0).all() for x in arrays)
        # a sample-shaped reply is never accepted as quantiles (and vice versa)
        with pytest.raises(ValueError, match="shape/dtype"):
            read_forecasts(output, identity, windows, 4)
        it = iter(arrays)
        remote = score_joint_quantile_forecaster_on_windows(lambda *_: next(it), windows, levels)
        local = evaluate_checkpoint(directory, windows, num_samples=4, contract=contract,
                                    quantile_head=True)
        _same_rows(local, remote)
        _same_rows(local, evaluate_checkpoint(directory, windows, num_samples=4, contract=contract,
                                              quantile_head=True, workers=2))
        # the sample path still answers the sample request on the same checkpoint
        sample_req = tmp_path / "samples_request.npz"
        write_request(sample_req, directory, windows, contract=contract, num_samples=4)
        run_request(directory, sample_req, tmp_path / "samples.npz", device="cpu")
    finally:
        torch.set_num_threads(old)


def test_private_worker_refuses_an_unknown_request_schema(tmp_path):
    import json

    from cascade.validator.private_worker import run_request

    meta = {"schema": "cascade-private-eval/9", "horizons": [], "contract": {}, "runtime": {},
            "num_samples": 1, "checkpoint": {}}
    np.savez(tmp_path / "r.npz", metadata=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8))
    with pytest.raises(ValueError, match="incompatible"):
        run_request(tmp_path, tmp_path / "r.npz", tmp_path / "o.npz", device="cpu")
