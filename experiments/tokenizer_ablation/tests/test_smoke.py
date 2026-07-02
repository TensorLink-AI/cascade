"""CPU smoke tests — no GIFT-Eval, no network, no GPU required.

Exercises the whole train/infer path on a tiny config so the components are
proven to fit together. GIFT-Eval and Hub push are covered by import only
(their heavy/pinned deps are optional extras).
"""
from __future__ import annotations

import numpy as np

from tsfm_ablation import Cfg, make_arm
from tsfm_ablation.config import default_runs


def _tiny_cfg(**over):
    """A minimal fixed-tokenizer config that trains in a couple of steps."""
    cfg = Cfg(run_name="tiny", tokenizer="fixed", ctx_span=256, fine_span=128,
              d_model=32, n_layers=2, n_heads=2, train_kmin=1, train_kmax=2,
              cpm_cmax=2, cpm_span_max=1, steps=3, batch=2, n_variates=2,
              eval_every=2, ckpt_every=100, gift_ckpts=(), pool="short")
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def test_config_matrix_shapes():
    runs = default_runs()
    assert len(runs) == 12
    assert make_arm("T0", 0).n_ctx_tokens == 128
    assert make_arm("T1", 0).n_ctx_tokens == 44
    assert make_arm("T2", 0).n_ctx_tokens == 128
    # every arm keeps the same fine-band decode budget
    assert all(c.zone_tokens == 32 for c in runs)


def test_generator_deterministic_and_finite():
    from tsfm_ablation.generators import sample_ensemble
    a = sample_ensemble(8, 512, np.random.default_rng(0))
    b = sample_ensemble(8, 512, np.random.default_rng(0))
    assert a.shape == (8, 512) and a.dtype == np.float32
    assert np.isfinite(a).all()
    assert np.array_equal(a, b)  # seed-reproducible


def test_paired_corpus_batches_identical(tmp_path):
    from tsfm_ablation.corpus import CorpusSampler, build_pool
    cdir = str(tmp_path / "corpus")
    build_pool(cdir, "short", 32, 512, seed0=1)
    s1 = CorpusSampler(cdir, "short", data_seed=7)
    s2 = CorpusSampler(cdir, "short", data_seed=7)
    b1, b2 = s1.batch(3, 4, 256), s2.batch(3, 4, 256)
    assert np.array_equal(b1, b2)  # paired: same (data_seed, step) -> same batch


def _build_tiny_corpus(tmp_path):
    from tsfm_ablation.corpus import build_pool
    cdir = str(tmp_path / "corpus")
    build_pool(cdir, "short", 32, 512, seed0=1)
    return cdir


def test_forward_train_and_predict():
    import torch

    from tsfm_ablation.model import MiniTSFM2, count_params, sample_cpm_mask
    cfg = _tiny_cfg()
    model = MiniTSFM2(cfg)
    assert count_params(model) > 0
    rng = np.random.default_rng(0)
    L = cfg.ctx_span + cfg.train_kmax * cfg.patch
    x = torch.randn(cfg.batch, cfg.n_variates, L)
    k, mp = sample_cpm_mask(cfg, cfg.batch, cfg.n_variates, rng)
    loss, *_ = model.forward_train(x, k, mp)
    assert torch.isfinite(loss)
    q = model.predict(x[..., :cfg.ctx_span], k=2)
    assert q.shape == (cfg.batch, cfg.n_variates, 2 * cfg.patch, 9)
    # quantiles are sorted (de-crossed) along the last axis
    assert bool((q[..., 1:] >= q[..., :-1] - 1e-4).all())


def test_tokenizers_same_param_count():
    """The three tokenizers must yield identical parameter counts (no capacity
    confound) at a shared width."""
    from tsfm_ablation.model import MiniTSFM2, count_params
    shared = dict(d_model=32, n_layers=2, n_heads=2, fine_span=128)
    t0 = MiniTSFM2(Cfg(tokenizer="fixed", ctx_span=256, **shared))
    t1 = MiniTSFM2(Cfg(tokenizer="pyramid", ctx_span=256, fine_span=128,
                       pyramid_levels=((128, 32), (128, 32)), d_model=32,
                       n_layers=2, n_heads=2))
    t3 = MiniTSFM2(Cfg(tokenizer="adaptive", ctx_span=256, adaptive_hist_tokens=4,
                       **shared))
    assert count_params(t0) == count_params(t1) == count_params(t3)


def test_train_one_and_probe(tmp_path):
    from tsfm_ablation.probes import long_horizon_probe
    from tsfm_ablation.train import train_one
    cdir = _build_tiny_corpus(tmp_path)
    ckpt = str(tmp_path / "ck")
    cfg = _tiny_cfg()
    model, log = train_one(cfg, ckpt, cdir, verbose=False)
    assert len(log["val_crps"]) >= 1
    assert all(np.isfinite(v) for v in log["val_crps"])
    probe = long_horizon_probe(model, n_series=2, total_h=128, n_buckets=2)
    assert len(probe["pearson"]) == 2


def test_optional_eval_modules_import():
    # These import lazily and must not require the pinned gift stack to import.
    import tsfm_ablation.gift_eval  # noqa: F401
    import tsfm_ablation.hub  # noqa: F401
    import tsfm_ablation.runner  # noqa: F401
