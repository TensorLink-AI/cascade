"""`cascade score` — pool resolution (synthetic / local dir) and the
train→eval wiring, with the heavy train/eval steps mocked."""

from __future__ import annotations

import numpy as np

from cascade.miner import score as score_mod


def test_synthetic_pool_is_offline_and_labelled(cfg):
    windows, label = score_mod._load_pool_windows(
        cfg, pool_dir=None, pool_ref="", n_windows=8, seed=0, cache_dir="/tmp"
    )
    assert 0 < len(windows) <= 8
    assert "synthetic-sample" in label and "directional" in label
    # windows carry the configured geometry
    w = windows[0]
    assert w.target.shape[-1] == cfg.eval.horizon


def test_local_pool_dir_takes_precedence(cfg, tmp_path):
    # write a few held-out series; the dir path must be used (not synthetic)
    n = cfg.eval.context_length + cfg.eval.horizon
    for i in range(5):
        np.save(tmp_path / f"series{i}.npy",
                np.sin(np.arange(n) / 7.0) + 0.1 * np.random.default_rng(i).standard_normal(n))
    windows, label = score_mod._load_pool_windows(
        cfg, pool_dir=tmp_path, pool_ref="", n_windows=4, seed=1, cache_dir=tmp_path
    )
    assert label.startswith("dir:")
    assert 0 < len(windows) <= 4


class _FakeStream:
    digest = "deadbeef" * 8
    n_series = 12

    def __enter__(self): return self
    def __exit__(self, *a): return False
    def series(self):
        for _ in range(3):
            yield np.ones((1, 64), dtype=np.float64)


class _FakeTrainer:
    def train(self, stream, contract, *, training_seed, token_budget, out_dir, logger=None):
        from cascade.trainer.contract import TrainResult
        for _ in stream:  # drain so the digest/n_series finalise
            pass
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model.safetensors").write_bytes(b"x")
        return TrainResult(local_dir=out_dir, param_count=1, train_seconds=4.2, metrics={})


def test_score_generator_wiring(cfg, tmp_path, monkeypatch):
    from cascade.eval.scoring import WindowScore

    # stub the three heavy seams
    monkeypatch.setattr("cascade.trainer.main._load_trainer", lambda spec: _FakeTrainer())
    monkeypatch.setattr("cascade.trainer.stream.open_round_stream",
                        lambda *a, **k: _FakeStream())

    def fake_eval(ckpt, windows, *, num_samples, device):
        assert (ckpt / "model.safetensors").exists()   # trained checkpoint reached the evaluator
        rng = np.random.default_rng(0)
        return [WindowScore(series_id=str(i), mase=1.0,
                            qloss_per_q=rng.uniform(0.1, 1.0, 9), abs_target=5.0)
                for i in range(len(windows))]
    monkeypatch.setattr("cascade.validator.evaluator.evaluate_checkpoint", fake_eval)

    r = score_mod.score_generator(
        "scripts/example_generator", cfg, n_windows=5, seed=0, cache_dir=tmp_path,
    )
    assert r.geomean > 0 and np.isfinite(r.geomean)
    assert r.n_windows == 5
    assert r.n_series == 12 and r.corpus_digest.startswith("deadbeef")
    assert r.train_seconds == 4.2
    assert "synthetic-sample" in r.pool_label
