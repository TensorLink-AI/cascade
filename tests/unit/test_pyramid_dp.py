"""Multi-GPU (data-parallel) pyramid step: same step as one GPU.

Run on CPU with the gloo backend: two ranks deal the micro-batches between
them and must reproduce the one-process gradients, losses, routing loads and
copula random stream.
"""

from __future__ import annotations

import socket
import types

import numpy as np
import pytest
import torch

from cascade.trainer.pyramid_dp import micro_batch_pass, owned_splits, skip_copula_draws
from cascade.trainer.pyramid_model import (
    NUM_QUANTILES,
    CascadeMoEConfig,
    CascadeMoEModel,
    sample_copula_paths,
)

CFG = dict(d_model=8, num_layers=3, num_heads=2, head_dim=4, patch_size=4, dense_ffn=12,
           num_experts=8, top_k=2, expert_hidden=5, shared_hidden=7, conv_channels=4,
           moe_blocks=(1, 2), variate_blocks=())
B, C, L, SPLITS = 4, 2, 16, 4


def _model():
    torch.manual_seed(7)
    model = CascadeMoEModel(CascadeMoEConfig(**CFG))
    model.train()
    return model


def _batch():
    rng = np.random.default_rng(3)
    values = np.cumsum(rng.standard_normal((B, C, L)), axis=-1)
    masks = np.zeros((B, C, L // 4), dtype=bool)
    masks[:, :, -1] = True
    return values, masks


def _step(model, gang, mine):
    values, masks = _batch()
    params = list(model.parameters())
    splits = np.array_split(np.arange(B), SPLITS)
    gen = torch.Generator().manual_seed(11)
    ref, lm, lj, load = micro_batch_pass(model, params, values, masks, None, splits, mine, gen,
                                         torch.device("cpu"), step=0, tokens=0, gang=gang)
    return {"ref": [None if r is None else r.clone() for r in ref],
            "grad": [None if p.grad is None else p.grad.clone() for p in params],
            "loss": (lm, lj), "load": [t.clone() for t in load], "gen": gen.get_state()}


def _rank(rank, world, port, out):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    gang = types.SimpleNamespace(rank=rank, dist=dist, device=torch.device("cpu"))
    gang.sum_ = lambda ts: [dist.all_reduce(t) for t in ts]

    def any_(flag):
        t = torch.tensor([int(flag)])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    gang.any_ = any_
    result = _step(_model(), gang, owned_splits(SPLITS, rank, world))
    torch.save(result, f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


@pytest.mark.parametrize("grouped", [False, True])
def test_skip_advances_the_generator_like_the_sampler(grouped):
    n, c, p, ps = 2, 3, 5, 4
    params = torch.randn(n, c, p - 1, ps, NUM_QUANTILES + 2)
    groups = torch.tensor([[0, 1, 1], [0, 0, 2]]) if grouped else None
    a, b = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    sample_copula_paths(params, 4, generator=a, groups=groups)
    skip_copula_draws(b, n, c, p, ps, groups, torch.device("cpu"))
    assert torch.equal(a.get_state(), b.get_state())


def test_two_ranks_reproduce_the_one_gpu_step(tmp_path):
    import torch.multiprocessing as mp

    one = _step(_model(), None, set(range(SPLITS)))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.get_context("spawn")
    mp.spawn(_rank, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    for rank in (0, 1):
        got = torch.load(tmp_path / f"rank{rank}.pt")
        assert got["loss"] == pytest.approx(one["loss"], rel=1e-6)
        for a, b in zip(got["load"], one["load"], strict=True):
            assert torch.equal(a, b)
        for key in ("ref", "grad"):
            for a, b in zip(got[key], one[key], strict=True):
                assert (a is None) == (b is None) or (b is None and not a.any())
                if b is not None:
                    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7)
        assert torch.equal(got["gen"], one["gen"])          # same random stream after the step


def test_owned_splits_deal_round_robin():
    assert owned_splits(8, 0, 4) == {0, 4} and owned_splits(8, 3, 4) == {3, 7}
    assert set().union(*(owned_splits(8, r, 8) for r in range(8))) == set(range(8))
