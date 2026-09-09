"""The multivariate inference contract (DEC-CA-0041).

``cascade.validator.evaluator.load_forecaster`` prefers ``forecast_joint`` when a
checkpoint exposes it and otherwise lifts the 1-D ``forecast`` through the
per-channel adapter. Until this method existed, NO checkpoint the trainer
produced could take the joint path, so a multivariate window was always scored
one channel at a time and cross-channel structure was unusable — arming
``mv_score_from_block`` would have rewarded nothing.

The load-bearing property is the equality at ``C = 1``: joint decoding must
reproduce the univariate path exactly, or turning it on would silently reprice
every univariate round.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from cascade.trainer.toto2_model import Toto2Config, Toto2Model  # noqa: E402
from cascade.trainer.toto2_trainer import _FORECAST_WRAPPER_PY  # noqa: E402

MODEL_SRC = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "cascade" / "trainer" / "toto2_model.py"
)


def _checkpoint(tmp_path):
    """Materialise a real (tiny) checkpoint dir: the wrapper imports model.py
    from it and rebuilds the architecture from config.json, exactly as the
    validator does."""
    from safetensors.torch import save_file

    cfg = Toto2Config(d_model=32, num_layers=4, num_heads=2, head_dim=16,
                      patch_size=8, num_quantiles=9, context_length=128,
                      max_patches=40)
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "model.py").write_text(MODEL_SRC.read_text(encoding="utf-8"), encoding="utf-8")
    (d / "forecast_wrapper.py").write_text(_FORECAST_WRAPPER_PY, encoding="utf-8")
    (d / "config.json").write_text(json.dumps({
        "arch": "test",
        "toto2": cfg.to_dict(),
        "quantile_levels": [round(0.1 * i, 1) for i in range(1, 10)],
    }), encoding="utf-8")
    torch.manual_seed(0)
    model = Toto2Model(cfg)
    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
              str(d / "weights.safetensors"))
    return d


def _wrapper(d):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("t_fj_wrapper", d / "forecast_wrapper.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.Wrapper(str(d), device="cpu")


def test_forecast_joint_matches_forecast_at_one_channel(tmp_path):
    """The whole rollout rests on this: arming the joint path must not move a
    single univariate score."""
    w = _wrapper(_checkpoint(tmp_path))
    rng = np.random.default_rng(0)
    hist = rng.normal(0.0, 1.0, size=256).cumsum()

    uni = w.forecast(hist, 16, 32)                      # (1, 32, 16)
    joint = w.forecast_joint(hist[None, :], 16, 32)     # (1, 32, 16)
    assert uni.shape == joint.shape == (1, 32, 16)
    assert np.array_equal(uni, joint), (
        f"C=1 joint decode diverged from the univariate path "
        f"(max |delta| {np.abs(uni - joint).max():.3e})"
    )


def test_forecast_joint_returns_every_channel(tmp_path):
    w = _wrapper(_checkpoint(tmp_path))
    rng = np.random.default_rng(1)
    hist = rng.normal(0.0, 1.0, size=(5, 256)).cumsum(axis=1)
    out = w.forecast_joint(hist, 12, 8)
    assert out.shape == (5, 8, 12)
    assert np.isfinite(out).all()


def test_channels_condition_on_each_other(tmp_path):
    """Joint decoding must actually route information across the variate axis:
    perturbing a SIBLING channel has to move this channel's forecast. If it
    does not, the joint path is per-channel decoding wearing a different shape
    and multivariate scoring would still reward nothing."""
    w = _wrapper(_checkpoint(tmp_path))
    rng = np.random.default_rng(2)
    hist = rng.normal(0.0, 1.0, size=(3, 256)).cumsum(axis=1)

    base = w.forecast_joint(hist, 12, 4)
    other = hist.copy()
    other[1] = other[1] + rng.normal(0.0, 5.0, size=other.shape[1]).cumsum()
    moved = w.forecast_joint(other, 12, 4)

    # channel 0's own history is untouched; only a sibling changed
    assert np.array_equal(hist[0], other[0])
    assert not np.allclose(base[0], moved[0]), (
        "channel 0 ignored a sibling's change — variate attention is inert"
    )


def test_validator_prefers_the_joint_path(tmp_path):
    """load_forecaster should now dispatch to forecast_joint rather than the
    per-channel adapter."""
    ev = pytest.importorskip("cascade.validator.evaluator")
    d = _checkpoint(tmp_path)
    fn = ev.load_forecaster(d, device="cpu")
    rng = np.random.default_rng(3)
    hist = rng.normal(0.0, 1.0, size=(4, 256)).cumsum(axis=1)
    out = fn(hist, 12, 6)
    assert np.asarray(out).shape == (4, 6, 12)
