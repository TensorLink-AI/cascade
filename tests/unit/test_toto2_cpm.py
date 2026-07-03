"""Toto2 CPM (contiguous patch masking) — mask sampling, no-leakage through the
mask channel, and the checkpoint wrapper's single-pass quantile decoding.

These run the real (tiny) torch model on CPU; the wrapper tests exercise the
exact ``forecast_wrapper.py`` text that ships inside every checkpoint.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cascade.trainer.toto2_model import (
    QUANTILE_LEVELS,
    Toto2Config,
    Toto2Model,
    causal_standardize,
    patch_anchors,
)
from cascade.trainer.toto2_trainer import Toto2Trainer, sample_cpm_masks

TINY = Toto2Config(
    d_model=16, num_layers=1, num_heads=1, head_dim=16, patch_size=4,
    mlp_expansion=2, num_quantiles=9, context_length=16, horizon=8, max_patches=10,
)


# ── mask sampling ─────────────────────────────────────────────────────────────


def test_cpm_masks_shape_and_budget():
    rng = np.random.default_rng(0)
    masks = sample_cpm_masks(64, 32, c_max=4, p_max=0.4, rng=rng)
    assert masks.shape == (64, 32) and masks.dtype == bool
    # each row's masked fraction stays near its p ~ U(0, p_max) draw: at most
    # p_max·P plus one span of overshoot
    assert (masks.sum(axis=1) <= 0.4 * 32 + 4).all()
    assert masks.any(), "expected at least some masked patches over 64 rows"


def test_cpm_masks_deterministic_given_seed():
    a = sample_cpm_masks(8, 16, c_max=4, p_max=0.4, rng=np.random.default_rng(7))
    b = sample_cpm_masks(8, 16, c_max=4, p_max=0.4, rng=np.random.default_rng(7))
    assert (a == b).all()


def test_cpm_masks_degenerate_inputs_mask_nothing():
    rng = np.random.default_rng(0)
    assert not sample_cpm_masks(4, 1, c_max=4, p_max=0.4, rng=rng).any()
    assert not sample_cpm_masks(4, 16, c_max=4, p_max=0.0, rng=rng).any()


# ── model: mask channel semantics ─────────────────────────────────────────────


def test_forward_shape_with_and_without_mask():
    torch.manual_seed(0)
    model = Toto2Model(TINY).eval()
    patches = torch.randn(3, 5, TINY.patch_size)
    with torch.no_grad():
        out = model(patches)
        assert out.shape == (3, 5, TINY.patch_size, TINY.num_quantiles)
        mask = torch.zeros(3, 5, dtype=torch.bool)
        assert torch.allclose(model(patches, mask=mask), out)


def test_masked_entries_cannot_leak():
    """Changing input values under masked entries must not change the output —
    the model may only see the mask bits and position there."""
    torch.manual_seed(0)
    model = Toto2Model(TINY).eval()
    patches = torch.randn(2, 5, TINY.patch_size)
    mask = torch.zeros(2, 5, dtype=torch.bool)
    mask[:, 2:4] = True
    corrupted = patches.clone()
    corrupted[:, 2:4] = 1e6  # garbage under the mask
    with torch.no_grad():
        assert torch.allclose(model(patches, mask=mask), model(corrupted, mask=mask))


# ── causal scaler ─────────────────────────────────────────────────────────────


def test_causal_scaler_is_causal():
    """Stats at step t must not move when future values change."""
    torch.manual_seed(0)
    x = torch.randn(2, 64).cumsum(-1)
    _, loc, scale = causal_standardize(x)
    y = x.clone()
    y[:, 40:] += 1e3
    _, loc2, scale2 = causal_standardize(y)
    assert torch.allclose(loc[:, :40], loc2[:, :40])
    assert torch.allclose(scale[:, :40], scale2[:, :40])


def test_causal_scaler_masked_entries_carry_stats_forward():
    torch.manual_seed(0)
    x = torch.randn(1, 64).cumsum(-1)
    mask = torch.zeros(1, 64)
    mask[:, 32:48] = 1.0
    corrupted = x.clone()
    corrupted[:, 32:48] = 1e6  # garbage under the mask must not touch the stats
    _, loc_a, scale_a = causal_standardize(x, mask=mask)
    _, loc_b, scale_b = causal_standardize(corrupted, mask=mask)
    assert torch.allclose(loc_a, loc_b) and torch.allclose(scale_a, scale_b)
    # stats are frozen across the span: anchor at span end == anchor at span start
    assert torch.allclose(loc_a[:, 47], loc_a[:, 31])
    assert torch.allclose(scale_a[:, 47], scale_a[:, 31])


def test_causal_scaler_backfills_leading_steps():
    x = torch.arange(32, dtype=torch.float32)[None, :] * 3.0
    _, loc, scale = causal_standardize(x, min_obs=8)
    # steps 0..6 reuse the stats of step 7 (first with 8 observations)
    assert torch.allclose(loc[:, :7], loc[:, 7:8].expand(-1, 7))
    assert (scale > 0).all()


def test_causal_scaler_survives_large_offsets():
    """Regression: the cumulative E[x²]−E[x]² form used to cancel catastrophically
    in float32, collapsing scale to the eps floor for any series whose mean ≫ std
    (counter/gauge-style benchmark data at 1e7+ levels) and unscaling forecasts by
    up to 7 orders of magnitude."""
    rng = np.random.default_rng(0)
    series = 1e7 + rng.normal(0.0, 100.0, size=2048)
    for dtype in (torch.float32, torch.float64):
        x = torch.as_tensor(series[None, :], dtype=dtype)
        _, _, scale = causal_standardize(x)
        tail = scale[0, 64:]
        assert (tail > 50.0).all() and (tail < 200.0).all(), f"scale broken for {dtype}"


def test_normuon_restores_orthogonalized_norm():
    """Regression: the per-row second-moment normalization must redistribute the
    orthogonalized update, not rescale it — without the norm restore, the
    zero-init EMA inflated step 1 by ~1/sqrt(1-β₂) and steady-state elements to
    RMS 1 instead of the ~1/sqrt(cols) the Muon-convention base_lr assumes."""
    from cascade.trainer.toto2_trainer import _MuonAdamW, _polar_express

    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(48, 64))
    p.grad = torch.randn(48, 64)
    opt = _MuonAdamW([p], [], lr=1.0, weight_decay=0.0)
    expected = _polar_express(p.grad * (1.0 + opt.momentum))  # Nesterov, step 1
    before = p.data.clone()
    opt.step()
    # lr=1 and scale=max(1, sqrt(48/64))=1 → |Δp|_F must equal |orthogonalized|_F
    step_norm = (p.data - before).norm()
    assert torch.isclose(step_norm, expected.norm(), rtol=0.05)


def test_patch_anchors_pick_patch_ends():
    loc = torch.arange(16, dtype=torch.float32)[None, :]
    scale = loc + 100
    a_loc, a_scale = patch_anchors(loc, scale, patch_size=4)
    assert a_loc.tolist() == [[3.0, 7.0, 11.0, 15.0]]
    assert a_scale.tolist() == [[103.0, 107.0, 111.0, 115.0]]


# ── variate-axis attention ────────────────────────────────────────────────────


def test_multivariate_forward_shape_and_univariate_equivalence():
    torch.manual_seed(0)
    model = Toto2Model(TINY).eval()
    patches = torch.randn(2, 3, 5, TINY.patch_size)  # (B, C=3, P, ps)
    with torch.no_grad():
        out = model(patches)
        assert out.shape == (2, 3, 5, TINY.patch_size, TINY.num_quantiles)
        # univariate (B, P, ps) is exactly the C=1 slice of the 4-D path
        uni = model(patches[:, 0])
        assert torch.allclose(uni, model(patches[:, :1])[:, 0])


def test_variate_layer_closes_each_group_of_four():
    # Toto-2.0-4m config.json: layer_group_size=4, num_variate_layers_per_group=1,
    # variate_layer_first=false → three time layers then one variate layer.
    model = Toto2Model(Toto2Config(num_layers=4))
    assert [blk.axis for blk in model.blocks] == ["time", "time", "time", "variate"]
    model8 = Toto2Model(Toto2Config(num_layers=8))
    assert [blk.axis for blk in model8.blocks] == ["time", "time", "time", "variate"] * 2


def test_ffn_hidden_pins_d_ff():
    assert Toto2Config(d_ff=688).ffn_hidden == 688
    assert Toto2Config(d_ff=0, d_model=256, mlp_expansion=2).ffn_hidden == 512


def test_umupp_residual_weights_match_release_constants():
    """a² + b² = 1 per branch (unit-scale stream), and the attention/FFN scale
    ratio matches the released residual_attn_ratio ≈ 5.136 at S = 128."""
    import math

    cfg = Toto2Config(context_length=4096, patch_size=32)  # S = 128
    model = Toto2Model(cfg)
    for blk in model.blocks:
        assert abs(blk.attn_a**2 + blk.attn_b**2 - 1.0) < 1e-9
        assert abs(blk.mlp_a**2 + blk.mlp_b**2 - 1.0) < 1e-9
    ratio2 = 128 / math.log(128)
    assert abs(math.sqrt(ratio2) - 5.136215466577748) < 1e-9  # config.json residual_attn_ratio
    # exact u-μP eq. 25–31 values for block 0 (branch indices l=1 attn, l=2 mlp)
    af2 = 2.0 * 0.75**2 / (ratio2 + 1.0)
    aa2 = ratio2 * af2
    tau2_attn = aa2 / 4.0                 # L/2 = 4 at num_layers=4
    tau2_mlp = af2 / (4.0 + aa2)
    b0 = model.blocks[0]
    assert abs(b0.attn_a - math.sqrt(tau2_attn / (tau2_attn + 1))) < 1e-9
    assert abs(b0.mlp_a - math.sqrt(tau2_mlp / (tau2_mlp + 1))) < 1e-9


def test_xpos_finite_and_scales_q_up_k_down():
    torch.manual_seed(0)
    from cascade.trainer.toto2_model import _xpos

    hd = 16
    idx = torch.arange(hd // 2).float() / (hd // 2)
    inv_freq = 1.0 / (10000.0**idx)
    zeta = (idx + 0.4) / 1.4
    q = torch.ones(1, 1, 134, hd)  # full decode window length
    k = torch.ones(1, 1, 134, hd)
    q2, k2 = _xpos(q, k, inv_freq, zeta)
    assert torch.isfinite(q2).all() and torch.isfinite(k2).all()
    # relative-position property survives the decay: q·k at equal positions is
    # position-independent (scales cancel), so dot(q_m, k_m) is constant
    dots = (q2 * k2).sum(-1).squeeze()
    assert torch.allclose(dots, dots[0].expand_as(dots), atol=1e-4)


def test_polar_express_orthogonalizes():
    torch.manual_seed(0)
    from cascade.trainer.toto2_trainer import _polar_express

    G = torch.randn(48, 64)
    X = _polar_express(G)
    eye = X @ X.mT
    assert torch.allclose(eye, torch.eye(48), atol=5e-2)
    # sign alignment: X should correlate with G's polar factor direction
    assert (X * G).sum() > 0


# ── checkpoint wrapper: single-pass CPM decode ────────────────────────────────


@pytest.fixture()
def tiny_checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    trainer = Toto2Trainer(device="cpu")
    model = Toto2Model(TINY)
    contract = SimpleNamespace(input_transform="arcsinh_causal")
    trainer._save_checkpoint(tmp_path, model, TINY, QUANTILE_LEVELS, contract)
    return tmp_path


def _load_wrapper(d: Path):
    spec = importlib.util.spec_from_file_location("test_ckpt_wrapper", d / "forecast_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Wrapper(str(d), device="cpu")


def test_wrapper_forecast_contract_and_determinism(tiny_checkpoint: Path):
    w = _load_wrapper(tiny_checkpoint)
    hist = np.sin(np.arange(40) / 3.0)
    out = w.forecast(hist, horizon=8, num_samples=5)
    assert out.shape == (1, 5, 8)
    assert np.isfinite(out).all()
    # seeded per window: identical across calls and across fresh instances
    again = _load_wrapper(tiny_checkpoint).forecast(hist, horizon=8, num_samples=5)
    assert np.array_equal(out, again)


def test_wrapper_quantiles_shape_monotone_and_batched(tiny_checkpoint: Path):
    w = _load_wrapper(tiny_checkpoint)
    assert np.allclose(w.quantile_levels, QUANTILE_LEVELS)
    h1 = np.sin(np.arange(40) / 3.0)
    h2 = np.cos(np.arange(25) / 5.0) * 10 + 3
    q = w.forecast_quantiles_batch([h1, h2], horizon=8)
    assert q.shape == (2, 8, len(QUANTILE_LEVELS))
    assert np.isfinite(q).all()
    # quantiles are sorted at decode → non-crossing after the monotone unscale
    assert (np.diff(q, axis=-1) >= 0).all()
    # batch rows equal single-series calls (same window prep per row)
    assert np.allclose(q[0], w.forecast_quantiles(h1, 8)[0], atol=1e-5)
    assert np.allclose(q[1], w.forecast_quantiles(h2, 8)[0], atol=1e-5)


def test_wrapper_block_decodes_past_positional_capacity(tiny_checkpoint: Path):
    # max_patches=10, patch_size=4 → stable single-pass span is 8 patches (32
    # steps); a 50-step horizon forces ≥2 block-decode rounds.
    w = _load_wrapper(tiny_checkpoint)
    q = w.forecast_quantiles_batch([np.arange(30, dtype=float)], horizon=50)
    assert q.shape == (1, 50, len(QUANTILE_LEVELS))
    assert np.isfinite(q).all()
    out = w.forecast(np.arange(30, dtype=float), horizon=50, num_samples=3)
    assert out.shape == (1, 3, 50) and np.isfinite(out).all()


def test_wrapper_forecast_tracks_context_level(tiny_checkpoint: Path):
    """Regression: sample forecasts must live near the context's level, and the
    sample path must agree with the quantile head it draws from. (A variable
    shadow once clamped samples into the quantile-INDEX range 1..8.)"""
    w = _load_wrapper(tiny_checkpoint)
    hist = 500.0 + np.sin(np.arange(64) / 5.0)  # level ~500, tight spread
    out = w.forecast(hist, horizon=8, num_samples=64)[0]  # (64, 8)
    q = w.forecast_quantiles(hist, 8)[0]                   # (8, nq)
    # samples are inverse-CDF draws from the decoded quantiles → bounded by them
    assert (out >= q[:, 0] - 1e-4).all() and (out <= q[:, -1] + 1e-4).all()
    # and nowhere near the 1..8 index range for a level-500 series
    assert abs(np.median(out) - 500.0) < 100.0


def test_wrapper_block_decode_advances_anchor(tiny_checkpoint: Path):
    """Regression: blocks ≥ 2 must be unscaled with causal stats advanced
    through the committed medians (training semantics: stats move through every
    observed patch), not the pre-horizon anchor. A stub model that always emits
    z = 1 decodes every value to ``sinh(1)·scale + loc`` at the *current*
    anchor — so block 2's output must use the stats recomputed after
    committing block 1 (which its above-the-mean commits have moved)."""
    w = _load_wrapper(tiny_checkpoint)

    class _OneModel:
        def __call__(self, patches, mask=None):
            B, P, ps = patches.shape
            return torch.ones(B, P, ps, len(QUANTILE_LEVELS))

    w.model = _OneModel()
    hist = np.linspace(0.0, 100.0, 40)
    # max_patches=10, ps=4 → stable span is 8 patches (32 steps); horizon 40 → 2 blocks
    q = w.forecast_quantiles_batch([hist], horizon=40)[0]     # (40, nq)
    b1, b2 = q[:32, 4], q[32:, 4]
    assert np.allclose(b1, b1[0])                             # one anchor per block
    ctx = w._prep([hist])[0].cpu().numpy()
    x2 = torch.as_tensor(np.concatenate([ctx, b1]), dtype=torch.float64)[None]
    _, loc2, scale2 = w.m.causal_standardize(x2)
    expected = np.sinh(1.0) * float(scale2[0, -1]) + float(loc2[0, -1])
    assert np.allclose(b2, expected, atol=1e-9)               # advanced anchor
    assert not np.isclose(b2[0], b1[0])                       # …and it moved


def test_wrapper_quantiles_survive_large_offset_level(tiny_checkpoint: Path):
    """Regression (end-to-end for the scaler fix): at a 1e7 level with std 100,
    decoded quantiles must sit at the context level with a non-degenerate
    spread — the float32 collapse pinned scale to 1e-5, flattening every
    quantile onto loc."""
    w = _load_wrapper(tiny_checkpoint)
    rng = np.random.default_rng(0)
    hist = 1e7 + rng.normal(0.0, 100.0, size=64)
    q = w.forecast_quantiles_batch([hist], horizon=8)[0]      # (8, nq)
    assert abs(np.median(q) - 1e7) < 1e6
    spread = q[:, -1] - q[:, 0]
    assert (spread > 1.0).all()                               # not collapsed onto loc


def test_trainer_end_to_end_tiny_run(tmp_path: Path):
    """A few CPM training steps on CPU produce a loadable checkpoint whose
    wrapper satisfies both the validator contract and the quantile API."""
    contract = SimpleNamespace(
        context_length=16, horizon=8, patch_size=4, d_model=16, num_layers=1,
        num_heads=1, head_dim=16, mlp_expansion=2, num_quantiles=9,
        batch_size=4, max_train_seconds=30, base_lr=1e-3, weight_decay=0.0,
        optimizer="adamw", warmup_tokens=0, input_transform="arcsinh_causal",
    )
    rng = np.random.default_rng(0)
    stream = (rng.normal(size=32).cumsum() for _ in range(16))
    trainer = Toto2Trainer(device="cpu", deterministic=False)
    result = trainer.train(
        stream, contract, training_seed=1, token_budget=1024, out_dir=tmp_path / "ckpt"
    )
    assert result.param_count > 0 and result.metrics["steps"] > 0
    w = _load_wrapper(tmp_path / "ckpt")
    assert w.forecast(np.arange(20, dtype=float), 8, 4).shape == (1, 4, 8)
    assert w.forecast_quantiles_batch([np.arange(20, dtype=float)], 8).shape == (1, 8, 9)
