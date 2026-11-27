"""2026-11-27: the first live rc4 lane B leg died in its first backward with
"expected scalar type Float but found BFloat16" (expert_training.py,
_ExpertMatmul.backward): training runs under bf16 autocast, the forward's bmm
is bf16 while the selected/frozen leaves stay fp32, and the custom backward
mixed the two. The fast-path parity tests ran in fp32 without autocast, so they
never reached this. Here the Function runs under bf16 autocast (CPU, so CI can
run it) the way the pyramid trainer drives it."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _fn():
    from cascade.trainer.expert_training import _ExpertMatmul
    return _ExpertMatmul


def _inputs(dtype=torch.float32):
    g = torch.Generator().manual_seed(0)
    e, t, d, f = 4, 6, 8, 5
    x = torch.randn(e, t, d, generator=g, dtype=dtype, requires_grad=True)
    frozen = torch.randn(e, d, f, generator=g)                       # fp32 buffer
    selected = torch.randn(2, d, f, generator=g, requires_grad=True)  # fp32 leaf
    rows = torch.tensor([1, 3])
    return x, frozen, selected, rows


def test_backward_under_bf16_autocast_returns_fp32_leaf_grads():
    x, frozen, selected, rows = _inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = _fn().apply(x, frozen, selected, rows)
        loss = out.float().square().mean()
    try:
        loss.backward()       # raised RuntimeError (Float vs BFloat16) on base
    except RuntimeError as e:
        raise AssertionError(f"expert matmul backward fails under bf16 autocast: {e}") from e
    assert selected.grad is not None and selected.grad.dtype == torch.float32
    assert torch.isfinite(selected.grad).all()
    assert x.grad is not None and x.grad.dtype == x.dtype and torch.isfinite(x.grad).all()


def test_backward_with_bf16_activations_and_fp32_leaves():
    # the shape the fast path hands it: bf16 activations, fp32 leaves, no autocast
    x, frozen, selected, rows = _inputs(torch.bfloat16)
    x2 = x.detach().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = _fn().apply(x2, frozen, selected, rows)
    try:
        out.float().sum().backward()
    except RuntimeError as e:
        raise AssertionError(f"expert matmul backward fails under bf16 autocast: {e}") from e
    assert selected.grad.dtype == torch.float32 and torch.isfinite(selected.grad).all()


def test_fp32_path_unchanged():
    # no autocast: the gradient equals the dense reference exactly as before
    x, frozen, selected, rows = _inputs()
    out = _fn().apply(x, frozen, selected, rows)
    out.square().sum().backward()
    xs = x.detach()[rows]
    ref = (xs.mT @ (2 * (xs @ selected.detach())))
    assert torch.allclose(selected.grad, ref, rtol=1e-5, atol=1e-6)
