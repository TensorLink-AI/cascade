"""Cascade-MOE-28M (DEC-CA-0050): shape/param accounting at the pinned size,
routing determinism + loss-free balancing, copula sampling, and the lane-B
expert freeze / merge hooks. Tiny configs on CPU except the size check, which
only builds the model (no forward)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from cascade.trainer.moe_model import (  # noqa: E402
    CascadeMoEConfig,
    CascadeMoEModel,
    MergeError,
    domain_expert_affinity,
    energy_score,
    expert_grad_mask_hooks,
    merge_expert_slices,
    moe_param_names,
    param_counts,
    rebase_expert_delta,
    sample_copula_paths,
    select_domain_experts,
)

TINY = CascadeMoEConfig(
    d_model=32, num_layers=4, num_heads=2, head_dim=16, patch_size=4,
    context_length=32, horizon=8, variate_blocks=(3,), moe_blocks=(1, 2),
    dense_ffn=40, num_experts=8, top_k=2, expert_hidden=8, shared_hidden=24,
    conv_channels=8,
)


def _tiny(seed: int = 0) -> CascadeMoEModel:
    torch.manual_seed(seed)
    return CascadeMoEModel(TINY)


def test_pinned_size_matches_spec():
    m = CascadeMoEModel(CascadeMoEConfig())
    c = param_counts(m)
    # spec: 27.5M active / 191M total
    assert c["active"] == pytest.approx(27.5e6, rel=0.03), c
    assert c["total"] == pytest.approx(191e6, rel=0.03), c
    cfg = m.cfg
    assert cfg.top_k * cfg.expert_hidden + cfg.shared_hidden == cfg.dense_ffn
    assert [b.axis for b in m.blocks].count("variate") == 2
    assert [i for i, b in enumerate(m.blocks) if b.moe] == [1, 2, 3, 5, 6, 7, 8]


def test_forward_shapes_uni_and_multivariate():
    m = _tiny()
    q, cop = m(torch.randn(3, 5, 4), with_copula=True)
    assert q.shape == (3, 5, 4, 9) and cop.shape == (3, 5, 4, 5)
    assert m(torch.randn(3, 5, 4)).shape == (3, 5, 4, 9)   # quantiles-only default
    q, cop = m(torch.randn(2, 3, 5, 4), roles=torch.tensor([[0, 1, 1], [0, 0, 1]]),
               with_copula=True)
    assert q.shape == (2, 3, 5, 4, 9) and cop.dtype == torch.float32
    assert m.aux_loss is not None and m.aux_loss.ndim == 0 and m.aux_loss > 0


def test_time_causality():
    m = _tiny().eval()
    x = torch.randn(1, 6, 4)
    q1 = m(x)
    x2 = x.clone()
    x2[:, 4:] += 10.0
    q2 = m(x2)
    torch.testing.assert_close(q1[:, :4], q2[:, :4])


def test_dispatch_matches_naive_loop():
    m = _tiny().eval()
    layer = m.moe_layers()[0]
    x = torch.randn(30, TINY.d_model)
    y = layer(x, seq_len=10)
    idx, w, _, _ = layer.route(x)
    ref = layer.shared(x).clone()
    for n in range(30):
        for j in range(TINY.top_k):
            e = idx[n, j]
            h = torch.nn.functional.gelu(x[n] @ layer.w_in[e])
            ref[n] += w[n, j] * (h @ layer.w_out[e])
    torch.testing.assert_close(y, ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(w.sum(-1), torch.ones(30))


def test_forward_is_deterministic():
    m = _tiny().eval()
    x = torch.randn(2, 5, 4)
    assert torch.equal(m(x), m(x))


def test_bias_moves_toward_balance_and_is_checkpointed():
    m = _tiny().train()
    m(torch.randn(4, 5, 4))
    layer = m.moe_layers()[0]
    load = layer.last_load.float()
    m.update_routing_bias()
    b = layer.route_bias
    over, under = load > load.mean(), load < load.mean()
    assert (b[over] < 0).all() and (b[under] > 0).all()
    assert "blocks.1.ffn.route_bias" in m.state_dict()
    assert not any("route_bias" in n for n, _ in m.named_parameters())


def test_copula_samples_and_energy_score():
    g = torch.Generator().manual_seed(0)
    q = torch.linspace(-1, 1, 9).expand(2, 6, 9).contiguous()
    cop = torch.zeros(2, 6, 5)
    cop[..., 0] = 3.0   # strong shared factor ⇒ points move together
    s = sample_copula_paths(q, cop, 256, generator=g)
    assert s.shape == (256, 2, 6)
    assert s.min() >= -1 and s.max() <= 1
    corr = torch.corrcoef(s[:, 0, :2].T)[0, 1]
    assert corr > 0.8
    assert energy_score(s, torch.zeros(2, 6)) < energy_score(s, torch.full((2, 6), 5.0))


def test_affinity_and_selection_are_deterministic():
    m = _tiny()
    batches = [(torch.randn(2, 5, 4), None) for _ in range(3)]
    a = domain_expert_affinity(m, batches)
    assert a.shape == (2, TINY.num_experts)
    assert int(a[0].sum()) == 3 * 2 * 5 * TINY.top_k
    sel = select_domain_experts(a, p=0.2, min_n=3, max_n=3)
    assert sel == select_domain_experts(a, p=0.2, min_n=3, max_n=3)
    assert all(len(v) == 3 for v in sel.values())
    ex = select_domain_experts(a, p=0.2, min_n=3, max_n=3, exclude={0: set(sel[0])})
    assert not set(ex[0]) & set(sel[0])


def test_esft_cumulative_threshold():
    # concentrated layer: 2 experts carry 60% ⇒ min_n binds; flat layer: p binds
    conc = torch.tensor([[30, 30, 5, 5, 5, 5, 10, 10]])
    assert select_domain_experts(conc, p=0.5, min_n=1, max_n=8) == {0: [0, 1]}
    flat = torch.full((1, 8), 10)
    assert len(select_domain_experts(flat, p=0.5, min_n=1, max_n=8)[0]) == 4
    assert len(select_domain_experts(flat, p=0.5, min_n=1, max_n=3)[0]) == 3
    assert len(select_domain_experts(conc, p=0.1, min_n=5, max_n=8)[0]) == 5


def _train_selected(m, sel, steps=3):
    expert_grad_mask_hooks(m, sel)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.1)
    for _ in range(steps):
        q = m(torch.randn(2, 5, 4))
        q.square().mean().backward()
        opt.step()
        opt.zero_grad()


def test_frozen_leg_changes_only_selected_rows_and_merges():
    king = _tiny()
    king_sd = {k: v.clone() for k, v in king.state_dict().items()}
    sel = {0: [1, 5], 1: [2]}
    cand = _tiny()
    cand.load_state_dict(king_sd)
    _train_selected(cand, sel)
    cand_sd = cand.state_dict()
    names = moe_param_names(king)
    merged = merge_expert_slices(king_sd, cand_sd, sel, names)
    for k in king_sd:
        assert torch.equal(merged[k], cand_sd[k]), k
    w_in = names[0][0]
    assert not torch.equal(cand_sd[w_in][[1, 5]], king_sd[w_in][[1, 5]])
    assert torch.equal(cand_sd[w_in][[0, 2, 3, 4, 6, 7]], king_sd[w_in][[0, 2, 3, 4, 6, 7]])


def test_disjoint_merges_commute():
    """Two domains' legs trained off the same king on disjoint selections:
    merging A then B equals merging B then A (each merge only writes its own
    rows, so the second merge is checked against the first's output)."""
    king = _tiny()
    king_sd = {k: v.clone() for k, v in king.state_dict().items()}
    names = moe_param_names(king)
    sels = [{0: [0, 1]}, {0: [4], 1: [6]}]
    cands = []
    for i, sel in enumerate(sels):
        c = _tiny()
        c.load_state_dict(king_sd)
        torch.manual_seed(100 + i)
        _train_selected(c, sel)
        cands.append(c.state_dict())

    def rebase(base, cand, sel):
        # the production merge re-bases a leg onto the current king: take the
        # leg's selected rows, everything else from the current king
        out = {k: v.clone() for k, v in base.items()}
        for li, pair in enumerate(names):
            for n in pair:
                out[n][sel.get(li, [])] = cand[n][sel.get(li, [])]
        return out

    a_then_b = merge_expert_slices(
        merge_expert_slices(king_sd, cands[0], sels[0], names), rebase(
            merge_expert_slices(king_sd, cands[0], sels[0], names), cands[1], sels[1]),
        sels[1], names)
    b_then_a = merge_expert_slices(
        merge_expert_slices(king_sd, cands[1], sels[1], names), rebase(
            merge_expert_slices(king_sd, cands[1], sels[1], names), cands[0], sels[0]),
        sels[0], names)
    for k in king_sd:
        assert torch.equal(a_then_b[k], b_then_a[k]), k


def test_merge_refuses_smuggled_changes():
    king = _tiny()
    king_sd = {k: v.clone() for k, v in king.state_dict().items()}
    names = moe_param_names(king)
    bad = {k: v.clone() for k, v in king_sd.items()}
    bad["head.weight"] += 1e-3
    with pytest.raises(MergeError, match="outside the expert selection"):
        merge_expert_slices(king_sd, bad, {0: [1]}, names)
    bad = {k: v.clone() for k, v in king_sd.items()}
    bad[names[0][0]][3] += 1e-3
    with pytest.raises(MergeError, match="unselected expert rows"):
        merge_expert_slices(king_sd, bad, {0: [1]}, names)
    bad = {k: v.clone() for k, v in king_sd.items()}
    bad["blocks.1.ffn.route_bias"] += 1.0
    with pytest.raises(MergeError):
        merge_expert_slices(king_sd, bad, {0: [1]}, names)


def test_rebase_carries_delta_onto_newer_king():
    base = _tiny()
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    names = moe_param_names(base)
    sel = {0: [1, 5]}
    cand = _tiny()
    cand.load_state_dict(base_sd)
    _train_selected(cand, sel)
    cand_sd = cand.state_dict()
    # lane A moved the king on since (every weight, incl. the selected rows)
    now = {k: (v + 0.01 if v.is_floating_point() else v.clone()) for k, v in base_sd.items()}
    out = rebase_expert_delta(now, base_sd, cand_sd, sel, names)
    w = names[0][0]
    torch.testing.assert_close(out[w][[1, 5]], now[w][[1, 5]] + cand_sd[w][[1, 5]] - base_sd[w][[1, 5]])
    assert torch.equal(out[w][[0, 2]], now[w][[0, 2]])
    assert torch.equal(out["head.weight"], now["head.weight"])
    # no drift ⇒ identical to the plain merge
    plain = merge_expert_slices(base_sd, cand_sd, sel, names)
    same = rebase_expert_delta(base_sd, base_sd, cand_sd, sel, names)
    for k in plain:
        torch.testing.assert_close(same[k], plain[k])
    bad = {k: v.clone() for k, v in cand_sd.items()}
    bad["head.weight"] += 1.0
    with pytest.raises(MergeError):
        rebase_expert_delta(now, base_sd, bad, sel, names)
