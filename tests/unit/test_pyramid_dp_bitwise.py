"""Phase-B host-sync removals in the pyramid DP step are bit-identical to main.

``main_micro_batch_pass`` below is ``cascade/trainer/pyramid_dp.py``'s
``micro_batch_pass`` VERBATIM at main 5e3fade (only its relative imports made
absolute): the oracle. Two gloo ranks run it and the current function on the
same model, batch and generator, fast (lane A) and reference (lane B) paths,
and every gradient, loss, routing load and generator state must be
``torch.equal`` — not merely close.
"""

from __future__ import annotations

import socket
import types

import numpy as np
import pytest
import torch

from cascade.trainer.pyramid_dp import (
    broadcast_from_sources,
    micro_batch_pass,
    owned_splits,
    skip_copula_draws,
    sum_int_bucket,
)
from cascade.trainer.pyramid_model import CascadeMoEConfig, CascadeMoEModel

CFG = dict(d_model=8, num_layers=3, num_heads=2, head_dim=4, patch_size=4, dense_ffn=12,
           num_experts=8, top_k=2, expert_hidden=5, shared_hidden=7, conv_channels=4,
           moe_blocks=(1, 2), variate_blocks=())
B, C, L, SPLITS = 4, 2, 16, 4


# fmt: off
# ruff: noqa: E501
def main_micro_batch_pass(model, params, values, masks, groups, splits, mine, cop_gen, device, *,
                     step: int, tokens: int, gang):
    """The step's micro-batch work on this rank, summed across the gang.

    Leaves the summed copula gradient in ``p.grad`` and returns ``(reference,
    loss_m, loss_j, load)``: the summed marginal gradient, both batch-weighted
    losses and the summed routing loads. With ``gang=None`` and every split
    owned this is the one-GPU step, operation for operation."""
    from cascade.trainer.pyramid_trainer import losses, prediction
    from cascade.trainer.toto2_trainer import check_loss_finite

    b, c, length = values.shape
    ps = model.cfg.patch_size
    layers = list(model.moe_layers())
    values_t = torch.as_tensor(values, device=device, dtype=torch.float64) if gang else None

    def batch(ids):
        x = (values_t[ids] if values_t is not None
             else torch.as_tensor(values[ids], device=device, dtype=torch.float64))
        mask = torch.as_tensor(masks[ids], device=device)
        g = None if groups is None else torch.as_tensor(groups[ids], device=device)
        return x, mask, g

    if getattr(model, "fast_training", False):
        from cascade.trainer.pyramid_fast import finish_aux

        load = [torch.zeros(layer.num_experts, device=device, dtype=torch.int64) for layer in layers]
        # One forward per owned micro-batch, graphs kept: the balance term needs
        # the FULL step's routing counts, which are exactly these forwards' counts
        # summed over every split and rank (the forward is deterministic) — so it
        # is finished here, before any backward, with no counting pre-pass.
        for layer in layers:
            layer.balance_load_override = None
            layer.defer_balance = True
        kept = []
        try:
            for k, ids in enumerate(splits):
                if k not in mine:
                    skip_copula_draws(cop_gen, len(ids), c, length // ps, ps,
                                      None if groups is None else torch.as_tensor(groups[ids]), device)
                    continue
                x, mask, g = batch(ids)
                marginal, joint = losses(model, x, mask, generator=cop_gen, groups=g)
                kept.append((len(ids) / b, marginal, joint, [layer.pending_balance for layer in layers]))
                for total, layer in zip(load, layers, strict=True):
                    total += layer.last_load
                    layer.pending_balance = None
        finally:
            for layer in layers:
                layer.defer_balance = False
        if gang:
            gang.sum_(load)
        reference = [None] * len(params)
        loss_m = loss_j = 0.0
        bad = False
        # Accumulate BOTH gradients over the complete batch before applying its cap.
        for weight, marginal, joint, pending in kept:
            marginal = marginal + finish_aux(model, pending, load)
            if gang is None:
                check_loss_finite(float(marginal.detach()), step=step, tokens=tokens)
                check_loss_finite(float(joint.detach()), step=step, tokens=tokens)
            elif not (np.isfinite(float(marginal.detach())) and np.isfinite(float(joint.detach()))):
                bad = True
                continue
            gm = torch.autograd.grad(marginal * weight, params, retain_graph=True, allow_unused=True)
            gj = torch.autograd.grad(joint * weight, params, allow_unused=True)
            for i, (p, m, j) in enumerate(zip(params, gm, gj, strict=True)):
                if m is not None:
                    reference[i] = m.detach() if reference[i] is None else reference[i] + m.detach()
                if j is not None:
                    p.grad = j.detach() if p.grad is None else p.grad + j.detach()
            loss_m += float(marginal.detach()) * weight
            loss_j += float(joint.detach()) * weight
        del kept
    else:
        load = [torch.zeros(layer.num_experts, device=device, dtype=torch.int64) for layer in layers]
        for layer in layers:
            layer.balance_load_override = None
        if len(splits) > 1:
            # Counts are non-differentiable, but balance probabilities need the
            # FULL batch's counts. A bounded no-grad pass avoids retaining all
            # micro-batch activations or changing the auxiliary objective.
            with torch.no_grad():
                for k, ids in enumerate(splits):
                    if k not in mine:
                        continue
                    x, mask, g = batch(ids)
                    prediction(model, x, mask, groups=g)
                    for total, layer in zip(load, layers, strict=True):
                        total += layer.last_load
            if gang:
                gang.sum_(load)
            for total, layer in zip(load, layers, strict=True):
                layer.balance_load_override = total.clone()
            for total in load:
                total.zero_()
        reference = [None] * len(params)
        loss_m = loss_j = 0.0
        bad = False
        # Accumulate BOTH gradients over the complete batch before applying its cap.
        for k, ids in enumerate(splits):
            if k not in mine:
                skip_copula_draws(cop_gen, len(ids), c, length // ps, ps,
                                  None if groups is None else torch.as_tensor(groups[ids]), device)
                continue
            weight = len(ids) / b
            x, mask, g = batch(ids)
            marginal, joint = losses(model, x, mask, generator=cop_gen, groups=g)
            marginal = marginal + model.aux_loss
            if gang is None:
                check_loss_finite(float(marginal.detach()), step=step, tokens=tokens)
                check_loss_finite(float(joint.detach()), step=step, tokens=tokens)
            elif not (np.isfinite(float(marginal.detach())) and np.isfinite(float(joint.detach()))):
                bad = True
                continue
            gm = torch.autograd.grad(marginal * weight, params, retain_graph=True, allow_unused=True)
            gj = torch.autograd.grad(joint * weight, params, allow_unused=True)
            for i, (p, m, j) in enumerate(zip(params, gm, gj, strict=True)):
                if m is not None:
                    reference[i] = m.detach() if reference[i] is None else reference[i] + m.detach()
                if j is not None:
                    p.grad = j.detach() if p.grad is None else p.grad + j.detach()
            loss_m += float(marginal.detach()) * weight
            loss_j += float(joint.detach()) * weight
            for total, layer in zip(load, layers, strict=True):
                total += layer.last_load
    if gang is None:
        return reference, loss_m, loss_j, load
    if gang.any_(bad):
        if gang.rank == 0:
            check_loss_finite(float("nan"), step=step, tokens=tokens)
        raise FloatingPointError("non-finite loss on another rank")
    # A gradient absent on every rank stays absent (combine_gradients skips it).
    present = torch.tensor([[reference[i] is not None, p.grad is not None]
                            for i, p in enumerate(params)], device=device, dtype=torch.int32)
    gang.dist.all_reduce(present, op=gang.dist.ReduceOp.MAX)
    for i, p in enumerate(params):
        if present[i, 0]:
            if reference[i] is None:
                reference[i] = torch.zeros_like(p)
            gang.dist.all_reduce(reference[i], op=gang.dist.ReduceOp.SUM)
        if present[i, 1]:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            gang.dist.all_reduce(p.grad, op=gang.dist.ReduceOp.SUM)
    losses_t = torch.tensor([loss_m, loss_j], device=device, dtype=torch.float64)
    if getattr(model, "fast_training", False):
        gang.sum_([losses_t])       # the loads were summed before the backwards
    else:
        gang.sum_([losses_t, *load])
    return reference, float(losses_t[0]), float(losses_t[1]), load

# fmt: on


def _model(fast):
    torch.manual_seed(7)
    model = CascadeMoEModel(CascadeMoEConfig(**CFG))
    model.train()
    if fast:
        from cascade.trainer.pyramid_fast import accelerate

        accelerate(model)
    return model


def _batch(grouped):
    rng = np.random.default_rng(3)
    values = np.cumsum(rng.standard_normal((B, C, L)), axis=-1)
    masks = np.zeros((B, C, L // 4), dtype=bool)
    masks[:, :, -1] = True
    groups = np.array([[0, 1]] * B, dtype=np.int64) if grouped else None
    return values, masks, groups


def _run(fn, fast, grouped, gang, mine, **kw):
    model = _model(fast)
    values, masks, groups = _batch(grouped)
    params = list(model.parameters())
    splits = np.array_split(np.arange(B), SPLITS)
    gen = torch.Generator().manual_seed(11)
    ref, lm, lj, load = fn(model, params, values, masks, groups, splits, mine, gen,
                           torch.device("cpu"), step=0, tokens=0, gang=gang, **kw)
    return {"ref": [None if r is None else r.clone() for r in ref],
            "grad": [None if p.grad is None else p.grad.clone() for p in params],
            "loss": (lm, lj), "load": [t.clone() for t in load], "gen": gen.get_state()}


def _assert_identical(a, b):
    assert a["loss"] == b["loss"]                      # exact floats
    assert torch.equal(a["gen"], b["gen"])
    for x, y in zip(a["load"], b["load"], strict=True):
        assert x.dtype == y.dtype == torch.int64 and torch.equal(x, y)
    for key in ("ref", "grad"):
        for x, y in zip(a[key], b[key], strict=True):
            assert (x is None) == (y is None)
            if x is not None:
                assert x.dtype == y.dtype and torch.equal(x, y)


def _gang(rank, dist):
    gang = types.SimpleNamespace(rank=rank, dist=dist, device=torch.device("cpu"))
    gang.sum_ = lambda ts: [dist.all_reduce(t) for t in ts]

    def any_(flag):
        t = torch.tensor([int(flag)])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    gang.any_ = any_
    return gang


def _rank(rank, world, port, out):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    gang, mine = _gang(rank, dist), owned_splits(SPLITS, rank, world)
    results = {}
    for fast in (True, False):
        for grouped in (False, True):
            old = _run(main_micro_batch_pass, fast, grouped, gang, mine)
            new = _run(micro_batch_pass, fast, grouped, gang, mine)
            values_t = torch.as_tensor(_batch(grouped)[0], dtype=torch.float64)
            reused = _run(micro_batch_pass, fast, grouped, gang, mine, values_t=values_t)
            results[(fast, grouped)] = (old, new, reused)
    # the integer bucket equals one all-reduce per tensor
    loads = [torch.arange(n, dtype=torch.int64) * (rank + 1) for n in (3, 8, 1)]
    per = [t.clone() for t in loads]
    for t in per:
        dist.all_reduce(t)
    sum_int_bucket(gang, loads)
    results["bucket"] = (per, loads)
    # the broadcast helper (gloo: one by one) delivers each tensor from its source
    ts = [torch.full((2,), float(rank * 10 + i)) for i in range(4)]
    broadcast_from_sources(dist, ts, [0, 1, 1, 0], torch.device("cpu"))
    results["bcast"] = ts
    torch.save(results, f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


def test_one_process_step_is_bit_identical_to_main():
    everything = set(range(SPLITS))
    for fast in (True, False):
        for grouped in (False, True):
            _assert_identical(_run(main_micro_batch_pass, fast, grouped, None, everything),
                              _run(micro_batch_pass, fast, grouped, None, everything))


def test_two_rank_gang_step_is_bit_identical_to_main(tmp_path):
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_rank, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    for rank in (0, 1):
        got = torch.load(tmp_path / f"rank{rank}.pt")
        for fast in (True, False):
            for grouped in (False, True):
                old, new, reused = got[(fast, grouped)]
                _assert_identical(old, new)
                _assert_identical(old, reused)
        per, bucket = got["bucket"]
        assert all(torch.equal(a, b) for a, b in zip(per, bucket, strict=True))
        assert [t[0].item() for t in got["bcast"]] == [0.0, 11.0, 12.0, 3.0]


def test_int_bucket_refuses_float_tensors():
    gang = types.SimpleNamespace(sum_=lambda ts: None)
    with pytest.raises(TypeError, match="int64"):
        sum_int_bucket(gang, [torch.zeros(2)])
    sum_int_bucket(gang, [])                            # nothing to send


def test_values_t_must_be_the_float64_copy():
    gang = types.SimpleNamespace(rank=0)
    values, masks, _ = _batch(False)
    model = _model(True)
    with pytest.raises(ValueError, match="float64 device copy"):
        micro_batch_pass(model, list(model.parameters()), values, masks, None,
                         np.array_split(np.arange(B), SPLITS), {0}, torch.Generator(),
                         torch.device("cpu"), step=0, tokens=0, gang=gang,
                         values_t=torch.as_tensor(values, dtype=torch.float32))
