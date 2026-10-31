"""The training-time fast kernels are the shipped operators (2026-10-31 speed fix).

``pyramid_fast`` re-implements ``MoEFFN.forward`` (load-binned expert dispatch,
deferred balance term) and ``DilatedConvEmbed.forward`` (im2col) for training;
``pyramid_model.py`` itself — the LOCKED architecture source validators
evaluate with — is unchanged. These pin the equivalences on CPU.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from cascade.trainer import pyramid_fast as pf
from cascade.trainer.pyramid_model import (
    CascadeMoEConfig,
    CascadeMoEModel,
    DilatedConvEmbed,
    MoEFFN,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config(**kw):
    base = dict(d_model=32, num_heads=4, head_dim=8, num_layers=3, dense_ffn=48, expert_hidden=4,
                shared_hidden=16, num_experts=16, experts_per_layer=(16,), moe_blocks=(1,),
                variate_blocks=(2,), context_length=128)
    return CascadeMoEConfig(**{**base, **kw})


def _routed(layer, x):
    idx, _, _, _ = layer.route(x)
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=layer.num_experts)
    return order // layer.cfg.top_k, torch.cumsum(counts, 0) - counts, counts


@pytest.mark.parametrize("skew", [0.0, 3.0, 50.0])     # balanced, skewed, collapsed router
def test_binned_dispatch_is_the_per_expert_reference(skew):
    torch.manual_seed(0)
    cfg = tiny_config(num_experts=32, experts_per_layer=(32,), top_k=4)
    layer = MoEFFN(cfg, 32).double()
    with torch.no_grad():
        layer.route_bias[0], layer.route_bias[1] = skew, skew / 2
    x = torch.randn(256, 32, dtype=torch.float64, requires_grad=True)
    tok, starts, counts = _routed(layer, x)
    got = pf.binned_experts(layer, x, tok, starts, counts)
    ref, at = [], 0
    for e, n in enumerate(counts.tolist()):
        if n:
            ref.append(F.gelu(x[tok[at:at + n]] @ layer.w_in[e]) @ layer.w_out[e])
            at += n
    ref = torch.cat(ref)
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-12)       # bit-exact on CUDA
    params = (x, layer.w_in, layer.w_out)
    for a, b in zip(torch.autograd.grad(got.square().sum(), params),
                    torch.autograd.grad(ref.square().sum(), params), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-11)


def test_binned_padding_stays_under_twice_the_rows_for_a_collapsed_router():
    torch.manual_seed(1)
    cfg = tiny_config(num_experts=32, experts_per_layer=(32,), top_k=4)
    layer = MoEFFN(cfg, 32)
    with torch.no_grad():
        layer.route_bias[0] = 1e3
    x = torch.randn(512, 32)
    tok, starts, counts = _routed(layer, x)
    seen = []
    real = layer.expert_mm
    layer.expert_mm = lambda buf, w, kind: (seen.append(buf.shape[0] * buf.shape[1]) if kind == "w_in" else None) or real(buf, w, kind)
    pf.binned_experts(layer, x, tok, starts, counts)
    assert sum(seen) < 2 * int(counts.sum())


def test_im2col_conv_embed_is_the_conv1d_embed():
    torch.manual_seed(0)
    emb = DilatedConvEmbed(tiny_config()).double()
    x2 = torch.randn(4, 4, 64, dtype=torch.float64)
    torch.testing.assert_close(pf.conv_embed_forward(emb, x2), emb(x2), rtol=0, atol=1e-12)


def test_deferred_balance_is_the_override_aux_loss():
    torch.manual_seed(0)
    layer = MoEFFN(tiny_config(), 16).train()
    x = torch.randn(64, 32)
    load = torch.randint(0, 50, (16,))
    layer.balance_load_override = load
    pf.moe_forward(layer, x, 16)
    want = layer.last_aux
    layer.balance_load_override, layer.defer_balance = None, True
    pf.moe_forward(layer, x, 16)
    assert layer.last_aux is None
    assert torch.equal(pf.finish_balance(layer, layer.pending_balance, load), want)


def test_accelerated_model_matches_the_shipped_model():
    torch.manual_seed(0)
    ref = CascadeMoEModel(tiny_config()).eval()
    fast = CascadeMoEModel(tiny_config()).eval()
    fast.load_state_dict(ref.state_dict())
    pf.accelerate(fast)
    assert fast.fast_training is True
    z = torch.randn(2, 3, 4, 32)
    torch.testing.assert_close(fast(z, with_parameters=True), ref(z, with_parameters=True),
                               rtol=1e-5, atol=1e-5)
    assert sorted(fast.state_dict()) == sorted(ref.state_dict())  # checkpoint unchanged


def test_accelerate_leaves_other_classes_alone():
    class Compact(MoEFFN):
        pass

    model = CascadeMoEModel(tiny_config())
    model.blocks[1].ffn.__class__ = Compact
    pf.accelerate(model)
    assert "forward" not in model.blocks[1].ffn.__dict__


def test_muon_owners_are_deterministic_and_balanced():
    from cascade.trainer.pyramid_optimizer import assign_muon_owners

    params = [torch.empty(n) for n in (100, 90, 80, 10, 10, 5, 5, 1)]
    owners = assign_muon_owners(params, 3)
    assert owners == assign_muon_owners(params, 3)
    load = [sum(p.numel() for p, r in zip(params, owners, strict=True) if r == k) for k in range(3)]
    assert max(load) - min(load) <= 100
    assert assign_muon_owners(params, 1) == [0] * len(params)


def test_state_sources_send_each_muon_param_from_its_owner():
    from cascade.trainer.pyramid_dp import state_sources
    from cascade.trainer.pyramid_optimizer import assign_muon_owners, build_optimizer

    model = CascadeMoEModel(tiny_config())
    opt = build_optimizer(model, 32)
    opt.muon_owner = assign_muon_owners(opt.muon_params, 4)
    src = dict(zip(model.state_dict(), state_sources(model, opt), strict=True))
    muon = {id(p): r for p, r in zip(opt.muon_params, opt.muon_owner, strict=True)}
    for name, p in model.named_parameters():
        assert src[name] == muon.get(id(p), 0)
    params = dict(model.named_parameters())
    assert all(src[name] == 0 for name in src if name not in params)   # buffers
    assert set(muon.values()) - {0}                               # work actually spread


def test_one_gpu_step_is_unchanged_by_the_single_forward():
    """micro_batch_pass with no gang: the deferred-balance path (accelerated
    model) and the pre-pass path (plain model) give the same losses/grads."""
    import numpy as np

    from cascade.trainer.pyramid_dp import micro_batch_pass
    from cascade.trainer.toto2_trainer import sample_cpm_masks

    torch.manual_seed(0)
    plain = CascadeMoEModel(tiny_config()).train()
    fast = CascadeMoEModel(tiny_config()).train()
    fast.load_state_dict(plain.state_dict())
    pf.accelerate(fast)
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, 3, 128)).cumsum(-1)
    masks = sample_cpm_masks(12, 4, c_max=16, p_max=0.4, rng=rng).reshape(4, 3, -1)
    splits = np.array_split(np.arange(4), 4)
    out = []
    for model in (plain, fast):
        params = list(model.parameters())
        gen = torch.Generator().manual_seed(3)
        ref, loss_m, loss_j, load = micro_batch_pass(model, params, values, masks, None, splits,
                                                     set(range(4)), gen, "cpu", step=0, tokens=0, gang=None)
        out.append((ref, [p.grad for p in params], loss_m, loss_j, [x.clone() for x in load]))
    (r1, g1, m1, j1, l1), (r2, g2, m2, j2, l2) = out
    assert all(torch.equal(a, b) for a, b in zip(l1, l2, strict=True))
    assert m1 == pytest.approx(m2, rel=1e-5) and j1 == pytest.approx(j2, rel=1e-5)
    for a, b in zip(r1 + g1, r2 + g2, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
