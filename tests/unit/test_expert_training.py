"""Compact expert state preserves masked-training gradients and merge rules."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from cascade.trainer.expert_training import ExpertSlices, _ExpertMatmul  # noqa: E402
from cascade.trainer.moe_model import (  # noqa: E402
    CascadeMoEConfig,
    CascadeMoEModel,
    expert_grad_mask_hooks,
    merge_expert_slices,
    moe_param_names,
    param_counts,
)
from cascade.trainer.moe_trainer import (  # noqa: E402
    PowerEMA,
    build_moe_optimizer,
    restore_frozen,
)


def _model():
    return CascadeMoEModel(CascadeMoEConfig(
        d_model=8, num_layers=3, num_heads=2, head_dim=4, patch_size=4,
        dense_ffn=12, num_experts=8, top_k=2, expert_hidden=5, shared_hidden=7,
        conv_channels=4, moe_blocks=(1, 2), variate_blocks=(),
    ))


def _optimizer(model):
    return build_moe_optimizer(model, SimpleNamespace(
        optimizer="normuon_adamw", base_lr=0.002, weight_decay=0.03,
    ))


def test_selected_matmul_gradcheck_with_stale_canonical_rows():
    torch.manual_seed(1)
    x = torch.randn(5, 3, 4, dtype=torch.double, requires_grad=True)
    frozen = torch.randn(5, 4, 2, dtype=torch.double)
    selected = torch.randn(2, 4, 2, dtype=torch.double, requires_grad=True)
    rows = torch.tensor([1, 4])
    assert torch.autograd.gradcheck(
        lambda a, b: _ExpertMatmul.apply(a, frozen, b, rows), (x, selected))


@pytest.mark.parametrize("selection", [{0: [1, 3]}, {1: [2]}, {0: [1, 3], 1: [2]}])
def test_compact_matches_masked_training_and_ema(selection):
    torch.manual_seed(7)
    dense = _model()
    compact = copy.deepcopy(dense)
    base = {n: p.clone() for n, p in dense.state_dict().items()}
    expert_grad_mask_hooks(dense, selection)
    slices = ExpertSlices(compact, selection)
    old_opt, new_opt = _optimizer(dense), _optimizer(slices)
    old_ema, new_ema = PowerEMA(dense, 0.1), PowerEMA(compact, 0.1, expert_slices=slices)
    names = moe_param_names(dense)
    assert param_counts(dense) == param_counts(compact)
    assert all(not p.requires_grad for p in compact.parameters())
    for _ in range(3):
        old_opt.zero_grad()
        new_opt.zero_grad()
        for _ in range(2):  # accumulation must sum compact gradients too
            a = torch.randn(2, 2, 4, 4, requires_grad=True)
            b = a.detach().clone().requires_grad_()
            ya, yb = dense(a), compact(b)
            torch.testing.assert_close(ya, yb, rtol=2e-5, atol=2e-6)
            ((ya.square().mean() + dense.aux_loss) / 2).backward()
            ((yb.square().mean() + compact.aux_loss) / 2).backward()
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=2e-6)
        for name, leaf in slices.named_parameters():
            expected = dict(dense.named_parameters())[name].grad[slices.rows[name]]
            torch.testing.assert_close(leaf.grad, expected, rtol=5e-4, atol=2e-6)
        torch.nn.utils.clip_grad_norm_([p for p in dense.parameters() if p.requires_grad], 0.1)
        torch.nn.utils.clip_grad_norm_(list(slices.parameters()), 0.1)
        old_opt.step()
        new_opt.step()
        slices.sync()
        old_ema.update(dense)
        new_ema.update(compact)
        assert all(p.grad is None for p in compact.parameters())
        # Checks exact frozen bytes, not just a numerical tolerance.
        merge_expert_slices(base, compact.state_dict(), selection, names)
        for name, tensor in dense.state_dict().items():
            torch.testing.assert_close(tensor, compact.state_dict()[name], rtol=2e-4, atol=2e-6)
    endpoint = {n: p.clone() for n, p in compact.state_dict().items()}
    scored = new_ema.export(compact)
    assert all(torch.equal(v, compact.state_dict()[n]) for n, v in endpoint.items())
    merge_expert_slices(base, scored, selection, names)
    expected = restore_frozen(old_ema.export(dense), base, selection, names)
    for name in scored:
        torch.testing.assert_close(scored[name], expected[name], rtol=2e-4, atol=2e-6)
    for name, leaf in slices.named_parameters():
        assert new_opt._bufs[leaf].shape == leaf.mT.shape
        assert new_opt._row_v[leaf].shape == (*leaf.mT.shape[:-1], 1)
        assert new_ema.shadow[name].shape == leaf.shape


@pytest.mark.parametrize("count", [8, 16, 38])
def test_mainnet_allocates_only_selected_rows(count):
    # Full production geometry without allocating 191M real CPU values.
    with torch.device("meta"):
        model = CascadeMoEModel(CascadeMoEConfig())
        slices = ExpertSlices(model, {li: list(range(count)) for li in range(7)})
        emas = [PowerEMA(model, s, expert_slices=slices) for s in (0.05, 0.1)]
        opt = _optimizer(slices)
        for p in slices.parameters():
            p.grad = torch.empty_like(p)
        opt.step()
    assert param_counts(model)["total"] == 191_496_256
    expected = 7 * count * 2 * 512 * 92
    assert sum(p.numel() for p in slices.parameters()) == expected
    assert all(sum(p.numel() for p in ema.shadow.values()) == expected for ema in emas)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    tensors = list(model.parameters()) + list(model.buffers()) + list(slices.parameters())
    tensors += [p.grad for p in slices.parameters()]
    tensors += list(opt._bufs.values()) + list(opt._row_v.values())
    tensors += [p for ema in emas for p in ema.shadow.values()]
    state_bytes = sum(p.numel() * p.element_size() for p in tensors)
    assert state_bytes < {8: 0.82, 16: 0.92, 38: 1.19}[count] * 2**30


@pytest.mark.parametrize("selection", [{}, {0: [-1]}, {0: [8]}, {0: [1, 1]}])
def test_invalid_or_empty_rows_refused(selection):
    with pytest.raises(ValueError, match="expert"):
        ExpertSlices(_model(), selection)
