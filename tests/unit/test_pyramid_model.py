"""Executable checks of the supplied ladder, independent of the historical MoE."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch
import torch.nn.functional as F

from cascade.trainer.pyramid_model import (
    CascadeMoEConfig,
    CascadeMoEModel,
    MoEFFN,
    causal_standardize,
    energy_score,
    param_counts,
    quantiles,
    sample_copula_paths,
)
from cascade.trainer.pyramid_optimizer import build_optimizer, orthogonalise, schedule_factor
from cascade.trainer.pyramid_trainer import combine_gradients, losses

ROOT = Path(__file__).resolve().parents[2]


def tiny_config():
    return CascadeMoEConfig(d_model=32, num_heads=4, head_dim=8, num_layers=3,
        dense_ffn=48, expert_hidden=4, shared_hidden=16, num_experts=16,
        experts_per_layer=(16,), moe_blocks=(1,), variate_blocks=(2,), context_length=128)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("size", ["2m", "7m", "28m", "120m"])
def test_counts_reproduce_the_supplied_reference(size):
    from cascade.shared.config.training import parse_training_contract
    from scripts.pyramid_config import training_table

    row = json.loads((ROOT / "deploy/presets/pyramid-ladder.json").read_text())["sizes"][size]
    contract = parse_training_contract(training_table(size))
    cfg = CascadeMoEConfig.from_contract(contract)
    with torch.device("meta"):
        model = CascadeMoEModel(cfg)
    assert param_counts(model) == {"total": row["total_params"], "active": row["active_params"]}
    assert [layer.w_in.shape[0] for layer in model.moe_layers()] == row["expert_counts"]
    assert all(model.blocks[i].ffn is None and model.blocks[i].norm2 is None for i in (4, 9))
    assert all(model.blocks[i].attn.gate is None for i in (4, 9))
    assert model.head.out_features == 32 * 11
    assert float(model.blocks[0].attn_a) == pytest.approx(0.31272, abs=1e-5)


def test_time_causality_and_masked_values_are_hidden():
    torch.manual_seed(7)
    model = CascadeMoEModel(tiny_config()).eval()
    x = torch.randn(1, 2, 4, 32)
    mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    mask[:, :, 1] = True
    hidden = x.clone()
    hidden[:, :, 1] += 1e6
    assert torch.equal(model(x, mask), model(hidden, mask))
    changed = x.clone()
    changed[:, :, 3] += 100
    torch.testing.assert_close(model(x)[:, :, :3], model(changed)[:, :, :3])
    block = model.blocks[2]
    h = torch.randn(1, 2, 32)
    torch.testing.assert_close(block(h), block.attn_b * h + block.attn_a * block.attn(block.norm1(h)))


def test_stem_is_a_sequential_causal_conv_on_observed_flags():
    model = CascadeMoEModel(tiny_config())
    stem = model.conv_embed
    x = torch.randn(2, 3, 64)
    x[..., 32:] = torch.randint(0, 2, (2, 3, 32)).float()
    h = torch.stack([x[..., :32], 1 - x[..., 32:]], dim=1).reshape(2, 2, 96)
    h = F.gelu(stem.convs[0](F.pad(h, (6, 0))))
    h = F.gelu(stem.convs[1](F.pad(h, (24, 0))))
    expected = stem.proj(F.avg_pool1d(h, 32).transpose(1, 2))
    torch.testing.assert_close(stem(x), expected)
    assert torch.count_nonzero(model.role_embed.weight) == 0
    assert torch.count_nonzero(model.blocks[0].attn.gate.weight) == 0


def test_group_selection_uses_bias_but_gates_use_unbiased_affinities():
    layer = MoEFFN(tiny_config(), 16)
    x = torch.randn(11, 32)
    layer.route_bias.copy_(torch.linspace(-0.4, 0.4, 16))
    idx, weights, affinities, logits = layer.route(x)
    torch.testing.assert_close(affinities, F.softplus(logits).sqrt())
    expected = affinities.gather(-1, idx)
    torch.testing.assert_close(weights, expected / expected.sum(-1, keepdim=True))
    assert all(len(set((ids // 2).tolist())) <= 4 for ids in idx)
    counts = torch.arange(16)
    before = layer.route_bias.clone()
    mean = counts.float().mean()
    delta = 0.003 * ((mean - counts) / mean).clamp(-1, 1)
    target = before + delta
    layer.update_bias(counts)
    torch.testing.assert_close(layer.route_bias, target - target.mean())


@pytest.mark.parametrize("experts", [16, 128])
def test_dense_mask_reference_matches_sparse_dispatch_and_router_gradients(experts):
    layer = MoEFFN(tiny_config(), experts)
    other = copy.deepcopy(layer)
    x = torch.randn(9, 32, requires_grad=True)
    idx, weight, _, _ = other.route(x)
    reference = other.shared(x)
    for expert in range(experts):
        gate = ((idx == expert) * weight).sum(-1, keepdim=True)
        reference = reference + gate * (F.gelu(x @ other.w_in[expert]) @ other.w_out[expert])
    result = layer(x, seq_len=3)
    torch.testing.assert_close(result, reference, atol=1e-6, rtol=1e-5)
    result.square().mean().backward()
    reference.square().mean().backward()
    torch.testing.assert_close(layer.router.weight.grad, other.router.weight.grad, atol=1e-6, rtol=1e-4)


def test_scaler_excludes_hidden_values_and_preserves_binary_rows():
    x = torch.arange(64.0)[None]
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 20:30] = True
    a = causal_standardize(x, mask)
    y = x.clone()
    y[mask] = -1e9
    b = causal_standardize(y, mask)
    torch.testing.assert_close(a[1], b[1])
    torch.testing.assert_close(a[2], b[2])
    _, loc, scale = causal_standardize((x % 2), mask)
    assert torch.count_nonzero(loc) == 0 and torch.equal(scale, torch.ones_like(scale))


def test_unified_head_and_energy_use_the_specified_coordinates_and_tails():
    model = CascadeMoEModel(tiny_config())
    params = model.head.bias.detach().reshape(1, 1, 32, 11)
    q = quantiles(params)
    expected = 0.5 * torch.special.ndtri(torch.tensor([i / 10 for i in range(1, 10)]))
    torch.testing.assert_close(q[0, 0, 0], expected)
    paths = sample_copula_paths(params, 10000, generator=torch.Generator().manual_seed(5))
    assert paths.shape == (1, 1, 10000, 32)
    assert float(paths.abs().max()) > float(q.abs().max())  # unbounded normal-score tails
    assert float(paths.std()) == pytest.approx(0.5, abs=0.02)
    pair = torch.stack([torch.zeros(32), torch.ones(32)])
    assert float(energy_score(pair, torch.zeros(32))) == pytest.approx(0.0)


def test_marginal_loss_is_dense_sum_of_nine_asinh_pinballs():
    model = CascadeMoEModel(tiny_config())
    x = torch.randn(1, 2, 128)
    mask = torch.zeros(1, 2, 4, dtype=torch.bool)
    marginal, joint = losses(model, x, mask, generator=torch.Generator().manual_seed(4))
    assert torch.isfinite(marginal + joint)
    assert marginal > 0 and joint > 0
    (marginal + joint).backward()
    assert model.head.weight.grad is not None
    assert all(layer.router.weight.grad is not None for layer in model.moe_layers())


def test_optimizer_partition_and_exact_width_rule():
    model = CascadeMoEModel(tiny_config())
    opt = build_optimizer(model, 128)
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    muon = {id(p) for p in opt.muon_params}
    assert id(model.role_embed.weight) in muon
    assert id(model.conv_embed.proj.weight) in muon
    assert id(model.blocks[1].ffn.w_in) in muon
    assert id(model.patch_embed.weight) not in muon
    assert id(model.head.weight) not in muon
    assert opt.param_groups[0]["base_lr"] == pytest.approx(0.002125 * (128/512)**0.42)
    for group in opt.adamw.param_groups:
        if any(p is model.head.weight for p in group["params"]):
            assert group["base_lr"] == 0.0017
        if any(p is model.blocks[1].ffn.router.weight for p in group["params"]):
            assert group["base_lr"] == 0.000425 and group["weight_decay"] == 0


def test_batched_normuon_equals_independent_matrices_and_zero_stays_zero():
    gradient = torch.randn(3, 6, 9)
    batch = orthogonalise(gradient)
    for i in range(3):
        torch.testing.assert_close(batch[i], orthogonalise(gradient[i]), atol=0.02, rtol=0.02)
    assert torch.count_nonzero(orthogonalise(torch.zeros(4, 6))) == 0
    assert [schedule_factor(s) for s in (0, 40, 80, 24999, 25000, 27499)] == [0, .5, 1, 1, 1, 0]


def test_energy_cap_is_measured_over_the_entire_parameter_gradient():
    params = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    params[0].grad = torch.tensor([30., 40.])
    params[1].grad = torch.zeros(1)
    reference = [torch.tensor([3., 4.]), torch.zeros(1)]
    weight = combine_gradients(params, reference)
    assert weight == pytest.approx(.01)
    torch.testing.assert_close(params[0].grad, torch.tensor([3.3, 4.4]))
    params[0].grad.fill_(float('nan'))
    with pytest.raises(FloatingPointError):
        combine_gradients(params, reference)


def test_compact_experts_preserve_router_and_variable_layer_bounds():
    from cascade.trainer.expert_training import ExpertSlices
    from cascade.trainer.pyramid_model import merge_expert_slices, moe_param_names

    model = CascadeMoEModel(tiny_config())
    base = {k: v.clone() for k, v in model.state_dict().items()}
    selection = {0: [0, 1, 2, 3]}
    leaves = ExpertSlices(model, selection)
    optimizer = build_optimizer(leaves, 32)
    # Force traffic through selected experts, while the selection bias remains frozen.
    model(torch.randn(1, 2, 4, 32)).square().mean().backward()
    optimizer.step()
    leaves.sync()
    merge_expert_slices(base, model.state_dict(), selection, moe_param_names(model))
    assert all(not p.requires_grad for n, p in model.named_parameters() if 'w_in' not in n and 'w_out' not in n)
    invalid = CascadeMoEModel(replace(tiny_config(), num_experts=32))
    with pytest.raises(ValueError, match="invalid expert rows"):
        ExpertSlices(invalid, {0: [17]})


def test_collapsed_routing_has_bounded_dispatch_and_unused_compact_rows_are_valid():
    from cascade.trainer.expert_training import ExpertSlices

    cfg = replace(tiny_config(), num_experts=128, experts_per_layer=(128,))
    model = CascadeMoEModel(cfg)
    layer = model.moe_layers()[0]
    layer.route_bias.copy_(torch.arange(128.0))  # chosen groups exclude rows 0..3
    slices = ExpertSlices(model, {0: [0, 1, 2, 3]})
    output = model(torch.ones(1, 2, 4, 32))
    output.square().mean().backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in slices.parameters())
    current = model.moe_layers()[0]
    assert int(current.last_load.sum()) == 2 * 4 * 8
    assert int(current.last_topk.min()) > 3


def test_microbatch_balance_uses_global_counts_instead_of_local_loads():
    layer = MoEFFN(tiny_config(), 16)
    x = torch.randn(12, 32)
    idx, _, affinity, logits = layer.route(x)
    full = layer._aux_loss(idx, affinity, logits, 4)
    layer.balance_load_override = torch.bincount(idx.reshape(-1), minlength=16)
    terms = []
    for rows in (slice(0, 4), slice(4, 12)):
        local_idx, _, s, z = layer.route(x[rows])
        weight = x[rows].shape[0] / len(x)
        terms.append(weight * layer._aux_loss(local_idx, s, z, 4))
    torch.testing.assert_close(sum(terms), full)


def test_dashboard_does_not_count_padding_as_dead_experts():
    from cascade.shared.v2_status import king_routing_health

    load = torch.zeros(2, 512)
    load[0, :128] = 1
    load[1, :256] = 1
    health = king_routing_health(load.numpy(), expert_counts=[128, 256])
    assert all(row['dead'] == 0 for row in health['layers'])
    assert all(row['max_over_mean'] == 1.0 for row in health['layers'])
