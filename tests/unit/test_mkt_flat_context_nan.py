"""MarketCast flat-context NaN: our adapter's gradient, and an operator fault
(2026-11-28 00:45).

A miner-funded MarketCast leg (5Cu2sxndK8Dw, worker-v2.1.0-mkt-rc2) died with
``non-finite training loss at step 1050`` and was recorded as "miner
submission rejected". The cause was OURS: ``forward_block`` ran the library's
``_refine_block`` -> ``advancing_block_frames`` (timestral @ a534bdbd,
``scaler.py:199-200``, ``var.sqrt().clamp(min=minimum_scale)``). On an EXACTLY
flat context (the miner's generator clips ``exp(cumsum)`` to +-3, ~3.6e-4 of
1024-step contexts) the median rounds to the context mean in float32, the
second patch's variance is exactly 0, and ``sqrt``'s backward is ``0/0 = NaN``
behind the scale floor; ``clip_grad_norm_`` spread it into every LoRA weight
and the next forward's loss was NaN. These pin the fix:

* the training refine frame has finite gradients on a flat context, and is
  BIT-IDENTICAL (values and gradients) to the library's on non-flat ones;
* without autograd ``forward_block`` still calls the library's own method,
  the library is never patched, and scoring (``forecast()``) is untouched;
* a non-finite training loss or gradient is an OPERATOR fault: the fixed
  worker exits 1 (not the rc 3 rejection), and the trainer classes the
  message as ``operator`` (unburned, requeued) for the fixed worker AND for
  the deployed mkt-rc2 image's rc 3 line; a genuine recipe rejection stays
  the miner's.

The classifier / worker halves are torch-free; the numeric ones need torch +
timestral (CI installs every extra).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]  # tests/unit/ (rc3 backport; was tests/accept/w_cmp/)
# The failed leg's own worker line (the trainer's _failed_legs evidence).
RC2_LINE = ("2026-11-28 00:45:33,576 ERROR cascade.trainer.worker: miner submission "
            "rejected: marketcast leg: non-finite training loss at step 1050")
JOB_LINE = ("[1]+  Exit 3                  CUDA_VISIBLE_DEVICES=0 /root/cascade/.venv/bin/python "
            "-m cascade.trainer.worker --role mkt_finetune")


def _verdict(rc, text):
    # rc3 backport: the rc2 base's classifier predates the ``role`` kwarg
    # (miner-sized OOM, not in this release); the NaN rule is role-independent.
    from cascade.trainer.faults import classify_funded_worker_failure

    return classify_funded_worker_failure(rc, text, stalled_before=False)


# ── blame (torch-free) ───────────────────────────────────────────────────────


def test_deployed_rc2_nonfinite_rejection_is_an_operator_fault():
    """The live image's rc 3 line, exactly as the dispatcher relays it."""
    from cascade.funding.queue import OPERATOR_FAULT_CLASS
    from cascade.trainer.remote import rejection_reason

    reason = rejection_reason(RC2_LINE + "\n" + JOB_LINE + "\n")
    text = f"remote mkt_finetune on funded-x: miner submission rejected: {reason}"
    assert _verdict(3, text) == (False, OPERATOR_FAULT_CLASS, False)


@pytest.mark.parametrize("what", ["loss", "gradient"])
def test_fixed_worker_nonfinite_exit_is_an_operator_fault(what):
    from cascade.funding.queue import OPERATOR_FAULT_CLASS
    from cascade.trainer.remote import summarize_failure

    stderr = ("2026-11-28 00:45:33,576 ERROR cascade.trainer.worker: marketcast leg failed "
              f"(operator fault): non-finite training {what} at step 7\n"
              + JOB_LINE.replace("Exit 3", "Exit 1") + "\n")
    cause, detail = summarize_failure(stderr, rc=1)
    text = f"remote mkt_finetune on funded-x failed (rc=1) {cause}\n{detail}"
    # an unclassified rc 1 would be "infra" and burn one of the miner's attempts
    assert _verdict(1, text) == (False, OPERATOR_FAULT_CLASS, False)


def test_a_genuine_recipe_rejection_stays_the_miners():
    assert _verdict(3, "miner submission rejected: marketcast leg: no training data for "
                       "the target suite 'range'") == (True, "generator", False)
    assert _verdict(1, "remote mkt_finetune failed (rc=1) exit_code=1: boom") == (
        False, "infra", True)


def test_nonfinite_training_is_not_a_miner_leg_error():
    from cascade.trainer import marketcast_leg as L

    cls = getattr(L, "MktNonFiniteTraining", None)
    assert cls is not None, "MktNonFiniteTraining missing"
    assert not issubclass(cls, L.MktLegError)


MC_TABLE = {
    "suites": ["returns", "vol"], "assets": {"crypto": ["BTC", "ETH"]},
    "frequencies": {"returns": ["1m", "5m"], "vol": ["5m"]},
    "horizons": {"returns": ["1m", "5m"], "vol": ["5m"]}, "max_steps": 1000,
}
RECIPE = {"target": "mkt:returns",
          "data": {"assets": {"crypto": ["BTC", "ETH"]}, "class_weights": {"crypto": 1.0},
                   "frequencies": ["1m", "5m"], "start": "2025-01-01", "end": "2025-01-02"},
          "context_length": 128, "horizon_mix": {"1m": 1.0, "5m": 1.0},
          "covariates": ["volume", "calendar"], "augmentations": [],
          "objective": {"suite_weights": {"returns": 1.0, "vol": 0.5}, "coverage_penalty": 0.1},
          "optimizer": {"lr": 1e-3, "schedule": "cosine", "warmup_fraction": 0.1, "steps": 6,
                        "batch": 4, "weight_decay": 0.0, "clip": 1.0, "ema_decay": 0.9},
          "trainable": {"scope": "lora", "lora_rank": 2}, "synthetic_fraction": 0.0}


def test_worker_exits_1_not_3_on_nonfinite_training(tmp_path, monkeypatch):
    from cascade.pool import marketcast as P
    from cascade.shared.config import load_chain_config
    from cascade.shared.config.marketcast import parse_marketcast
    from cascade.trainer import marketcast_leg as L
    from cascade.trainer.worker import run_mkt_worker

    cls = getattr(L, "MktNonFiniteTraining", None)
    assert cls is not None, "MktNonFiniteTraining missing"
    mc = replace(parse_marketcast(MC_TABLE), from_block=7200)
    cfg = replace(load_chain_config(str(ROOT / "chain.toml")), marketcast=mc)
    monkeypatch.setattr(L, "build_training_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(P, "load_snapshot", lambda *a, **k: SimpleNamespace(digest="d"))

    def boom(**kw):
        raise cls("non-finite training loss at step 1050")

    monkeypatch.setattr(L, "run_mkt_leg", boom)
    errors = []
    log = SimpleNamespace(error=lambda *a, **k: errors.append(a[0] % a[1:]),
                          info=lambda *a, **k: None, exception=lambda *a, **k: None,
                          warning=lambda *a, **k: None)
    args = SimpleNamespace(local_only=True, warm_start_ref=mc.base_ref,
                           recipe_json=json.dumps({"marketcast": RECIPE, "has_generator": False}),
                           work_root=tmp_path, hotkey="5Cu2sxndK8Dw", uid=171, block=1,
                           base_seed=9, gen_ref="g")
    rc = run_mkt_worker(args, cfg, log, post=lambda *a: [], get=lambda *a: [],
                        base_loader=lambda: (SimpleNamespace(model=None), tmp_path))
    assert rc == 1, errors
    assert not any("miner submission rejected" in e for e in errors)
    assert _verdict(rc, "\n".join(errors))[1] == "operator"


# ── numerics (torch + timestral) ─────────────────────────────────────────────


def _lib():
    torch = pytest.importorskip("torch")
    pytest.importorskip("timestral")
    from timestral.config import TimestralConfig
    from timestral.forecaster import TimestralForecaster
    from timestral.model import TimestralModel

    torch.manual_seed(0)
    config = TimestralConfig(patch_size=4, d_model=16, num_heads=2, mlp_hidden_dim=32,
        block_schedule=("time", "time", "variate"), quantile_knots=(0.1, 0.5, 0.9),
        block_size=4, context_length=64)
    return torch, TimestralForecaster(TimestralModel(config))


def _grads(torch, fc):
    return [p.grad for p in fc.model.parameters() if p.grad is not None]


def test_flat_context_trains_with_finite_gradients():
    torch, fc = _lib()
    from timestral import scaler as S

    from cascade.eval import timestral_adapter as TA

    # A flat level large enough that every median rounds to the context mean
    # in float32 (minimum_scale 1e-6): the second patch's variance is exactly 0.
    contexts = np.full((2, 64), 1.0e5)
    contexts[1] = np.random.default_rng(0).normal(size=64).cumsum() + 50.0
    # control: the LIBRARY's frame NaNs the gradient on exactly this input
    sm = torch.zeros(1, 3, 4, dtype=torch.float64, requires_grad=True)
    flat = torch.full((1, 64), 1.0e5, dtype=torch.float32)
    _, scale = S.advancing_block_frames(flat, torch.ones_like(flat, dtype=torch.bool), sm, 1e-6)
    scale.sum().backward()
    assert torch.isnan(sm.grad).any(), "control: library sqrt(0) backward should be NaN"

    out = TA.forward_block(fc, contexts, 12)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    grads = _grads(torch, fc)
    assert grads and all(torch.isfinite(g).all() for g in grads), "NaN gradient on a flat context"
    assert any(g.abs().sum() > 0 for g in grads)


def test_non_flat_refine_is_bit_identical_to_the_library():
    torch, fc = _lib()
    from cascade.eval import timestral_adapter as TA

    refine = getattr(TA, "_refine_block", None)
    assert refine is not None, "gradient-safe refine missing"
    rng = np.random.default_rng(1)
    ctx = torch.as_tensor(rng.normal(size=(3, 64)).cumsum(-1) + 10.0, dtype=torch.float32)
    mask = torch.ones_like(ctx, dtype=torch.bool)
    mask[0, :9] = False
    base = torch.as_tensor(rng.normal(size=(3, 12, 3)), dtype=torch.float32).sort(-1).values
    base = base * 2.0 + 10.0
    qa = base.clone().requires_grad_(True)
    qb = base.clone().requires_grad_(True)
    ya = fc._refine_block(qa, ctx, mask, None)
    yb = refine(fc, qb, ctx, mask, None)
    assert torch.equal(ya, yb)
    w = torch.as_tensor(rng.normal(size=ya.shape), dtype=ya.dtype)
    (ya * w).sum().backward()
    (yb * w).sum().backward()
    assert torch.equal(qa.grad, qb.grad)

    # whole-model forward and parameter gradients: library frame vs ours
    contexts = rng.normal(size=(2, 64)).cumsum(-1)
    lib = fc._refine_block
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TA, "_refine_block", lambda f, *a: lib(*a))
        y0 = TA.forward_block(fc, contexts, 12)
        y0.square().mean().backward()
    g0 = [g.clone() for g in _grads(torch, fc)]
    fc.model.zero_grad(set_to_none=True)
    y1 = TA.forward_block(fc, contexts, 12)
    y1.square().mean().backward()
    assert torch.equal(y0, y1)
    assert all(torch.equal(a, b) for a, b in zip(g0, _grads(torch, fc), strict=True))


def test_eval_forward_is_untouched():
    torch, fc = _lib()
    import timestral.forecaster as F
    import timestral.scaler as S

    from cascade.eval import timestral_adapter as TA

    lib_refine, lib_frames = F.TimestralForecaster._refine_block, S.advancing_block_frames
    calls = []
    orig = fc._refine_block
    fc._refine_block = lambda *a: (calls.append(1), orig(*a))[1]
    contexts = np.random.default_rng(2).normal(size=(2, 64)).cumsum(-1)
    with torch.no_grad():
        nograd = TA.forward_block(fc, contexts, 12)
    assert calls, "without autograd forward_block must call the library's own refine"
    del fc._refine_block
    trained = TA.forward_block(fc, contexts, 12)
    assert torch.equal(nograd, trained.detach())
    assert F.TimestralForecaster._refine_block is lib_refine
    assert S.advancing_block_frames is lib_frames and F.advancing_block_frames is lib_frames
    evaluated = fc.forecast(contexts, 12, symmetric_averaging=False,
                            clip_nonnegative=False).quantiles[:, 0]
    torch.testing.assert_close(trained.detach(), evaluated)
