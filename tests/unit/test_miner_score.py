"""`cascade score` — pool resolution (synthetic / local dir) and the
train→eval wiring, with the heavy train/eval steps mocked."""

from __future__ import annotations

from pathlib import Path

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
    def __init__(self):
        self.calls = []

    def train(self, stream, contract, *, training_seed, token_budget, out_dir, logger=None,
              **kw):
        from cascade.trainer.contract import TrainResult
        self.calls.append(kw)
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


def _stub_train_eval(monkeypatch):
    from cascade.eval.scoring import WindowScore

    trainer = _FakeTrainer()
    monkeypatch.setattr("cascade.trainer.main._load_trainer", lambda spec: trainer)
    monkeypatch.setattr("cascade.trainer.stream.open_round_stream",
                        lambda *a, **k: _FakeStream())
    monkeypatch.setattr(
        "cascade.validator.evaluator.evaluate_checkpoint",
        lambda ckpt, windows, *, num_samples, device: [
            WindowScore(series_id=str(i), mase=1.0, qloss_per_q=np.full(9, 0.5), abs_target=5.0)
            for i in range(len(windows))],
    )
    return trainer


def test_default_is_random_init(cfg, tmp_path, monkeypatch):
    trainer = _stub_train_eval(monkeypatch)
    r = score_mod.score_generator("scripts/example_generator", cfg, n_windows=3, cache_dir=tmp_path)
    assert r.init_label == "random init"
    assert trainer.calls == [{}]                    # no warm_start_dir handed to the trainer


def test_warm_start_local_dir_reaches_trainer(cfg, tmp_path, monkeypatch):
    trainer = _stub_train_eval(monkeypatch)
    init = tmp_path / "promoted"
    init.mkdir()
    r = score_mod.score_generator("scripts/example_generator", cfg, n_windows=3,
                                  cache_dir=tmp_path, warm_start=init)
    assert r.init_label == f"warm-start dir:{init}"
    assert trainer.calls == [{"warm_start_dir": init}]


def test_warm_start_hub_ref_and_pointer_fetch(cfg, tmp_path, monkeypatch):
    fetched = []

    def fake_fetch(ref, dest, hub=None):
        fetched.append(ref)
        Path(dest).mkdir(parents=True, exist_ok=True)
        return Path(dest)
    monkeypatch.setattr("cascade.shared.hippius.fetch_from_hub", fake_fetch)
    ref = "cascade/ckpt-r1-challenger-toto2-4m-u5@sha256:" + "ab" * 32

    d, label = score_mod._resolve_warm_start(cfg, ref, cache_dir=tmp_path)
    d2, _ = score_mod._resolve_warm_start(cfg, f"metro-v1:trained:hippius:{ref}", cache_dir=tmp_path)
    assert fetched == [ref, ref]                     # pointer prefix stripped, bare ref as-is
    assert d == d2 and d.is_dir() and str(d).startswith(str(tmp_path / "warm-start"))
    assert label == f"warm-start {ref}"


def test_warm_start_live_reads_round_status(cfg, tmp_path, monkeypatch):
    ref = "cascade/ckpt-r1-challenger-toto2-4m-u5@sha256:" + "cd" * 32
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_round_status",
                        lambda storage, **k: {"warm_start": {
                            "init_checkpoint": f"metro-v1:trained:hippius:{ref}", "size": "toto2-4m"}})
    seen = []
    monkeypatch.setattr("cascade.shared.hippius.fetch_from_hub",
                        lambda r, dest, hub=None: (seen.append(r), Path(dest).mkdir(parents=True), Path(dest))[-1])
    d, label = score_mod._resolve_warm_start(cfg, "live", cache_dir=tmp_path)
    assert seen == [ref] and d.is_dir() and ref in label

    # a random-init round (no warm_start block) resolves to random init, no fetch
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_round_status",
                        lambda storage, **k: {"round_id": "1"})
    d, label = score_mod._resolve_warm_start(cfg, "live", cache_dir=tmp_path)
    assert d is None and label.startswith("random init")
    assert seen == [ref]


def test_warm_start_garbage_is_rejected(cfg, tmp_path):
    import pytest

    with pytest.raises(ValueError, match="--warm-start"):
        score_mod._resolve_warm_start(cfg, str(tmp_path / "nope"), cache_dir=tmp_path)
