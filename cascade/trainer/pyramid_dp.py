"""Multi-GPU data parallelism for the Cascade MoE (pyramid) training step.

A Lane A leg on a 4×/8× pod trains the SAME step as a one-GPU leg: one batch,
split into the contract's ``micro_batches``, gradients summed over the whole
batch, one optimizer step. With ``N`` GPUs the micro-batches are dealt out
round-robin (rank ``r`` takes splits ``r, r+N, …``) and everything the step
sums is summed across ranks:

* the no-grad routing-load pre-pass (the balance loss sees full-batch counts);
* both gradient sets (the marginal "reference" and the copula gradient);
* the loss values and the routing loads that drive the bias update.

Rank 0 is the training process itself: it reads the corpus, draws the masks,
sends each batch, takes the optimizer step and broadcasts the updated weights
and buffers. Ranks 1..N-1 are spawned helpers that hold only the model: no
optimizer state, no corpus, no checkpoint I/O.

The copula sampler's random stream stays the single-GPU stream: every rank
starts the step from rank 0's generator state and, for a split it does not
own, advances the generator by exactly the draws that split would have made
(:func:`skip_copula_draws`). The only difference from one GPU is the order
of floating-point sums in the all-reduce.
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import os
import socket

import numpy as np
import torch

log = logging.getLogger(__name__)

GANG_ENV = "CASCADE_TRAIN_GPUS"
_TIMEOUT = datetime.timedelta(seconds=600)
_STEP, _STOP, _GATHER, _SCATTER = 1, 0, 2, 3


def gang_size(device) -> int:
    """GPUs this leg trains on: every visible CUDA device for a plain
    ``"cuda"`` device (``CASCADE_TRAIN_GPUS`` may lower it), else 1."""
    if str(device) != "cuda" or not torch.cuda.is_available():
        return 1
    n = torch.cuda.device_count()
    with contextlib.suppress(ValueError):
        n = min(n, int(os.environ.get(GANG_ENV, n)))
    return max(1, n)


def bound_cpu_threads(world: int) -> int:
    """Give each rank its share of the host's cores: 8 ranks at torch's
    default (every core each) oversubscribed a 240-core pod (~34 busy
    cores per rank = ~270 threads, 2026-10-31). The GPU step does not depend on it."""
    n = max(1, (os.cpu_count() or 1) // max(1, int(world)))
    torch.set_num_threads(n)
    return n


def owned_splits(n_splits: int, rank: int, world: int) -> set[int]:
    return set(range(rank, n_splits, world))


def skip_copula_draws(gen, n: int, c: int, p: int, ps: int, groups, device) -> None:
    """Advance ``gen`` exactly as :func:`~.pyramid_model.sample_copula_paths`
    does for one split of ``n`` rows (``p`` patches, so ``p - 1`` targets):
    the path noise, then the shared or per-group latent."""
    torch.randn((n, c, p - 1, 4, ps), device=device, dtype=torch.float32, generator=gen)
    width = 1
    if groups is not None and not bool((groups == 0).all()):
        width = int(groups.max()) + 1
    torch.randn((n, width, p - 1, 4, 1), device=device, dtype=torch.float32, generator=gen)


class Gang:
    """Rank 0's handle on the helper ranks (see the module doc)."""

    def __init__(self, world: int, cfg, *, deterministic: bool, seed: int) -> None:
        import torch.distributed as dist
        import torch.multiprocessing as mp

        self.world, self.rank = int(world), 0
        self.device = torch.device("cuda", 0)
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ctx = mp.get_context("spawn")
        self.errors = ctx.Queue()
        self.procs = [ctx.Process(target=_follower, daemon=True,
                                  args=(r, self.world, port, cfg.to_dict(), deterministic,
                                        seed, self.errors))
                      for r in range(1, self.world)]
        for p in self.procs:
            p.start()
        torch.cuda.set_device(0)
        bound_cpu_threads(self.world)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=0,
                                world_size=self.world, timeout=_TIMEOUT)
        self.dist = dist
        self.synced = False
        log.info("pyramid: training on %d GPUs (micro-batches dealt round-robin)", self.world)

    # ── collectives ──────────────────────────────────────────────────────────

    def sum_(self, tensors) -> None:
        for t in tensors:
            self.dist.all_reduce(t, op=self.dist.ReduceOp.SUM)

    def any_(self, flag: bool) -> bool:
        t = torch.tensor([1 if flag else 0], device=self.device)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX)
        return bool(t.item())

    def broadcast_state(self, model) -> None:
        for t in model.state_dict().values():
            self.dist.broadcast(t, src=0)
        self.synced = True

    # ── sharded muon (every rank steps its own share) ───────────────────────

    optimizer = None
    scatter_pending = False

    def shard_optimizer(self, model, optimizer) -> None:
        """Split the muon params over the gang; rank 0 keeps AdamW, buffers
        and the routing-bias update. Call before the first step; set
        ``scatter_pending`` after loading optimizer state into rank 0."""
        from .pyramid_optimizer import assign_muon_owners

        optimizer.muon_owner = assign_muon_owners(optimizer.muon_params, self.world)
        optimizer.rank = 0
        self.optimizer = optimizer
        self._sources = state_sources(model, optimizer)

    def sync_after_step(self, model) -> None:
        """Every rank's weights after a sharded step: each muon param from its
        owner, everything else (AdamW params, buffers) from rank 0."""
        for t, src in zip(model.state_dict().values(), self._sources, strict=True):
            self.dist.broadcast(t, src=src)

    def gather_muon(self) -> None:
        """Bring every muon param's state to rank 0 (before any read of the
        optimizer state: continuation, stable snapshot, checkpoint)."""
        if self.optimizer is None:
            return
        self.dist.broadcast(torch.tensor([_GATHER, 0, 0, 0, 0, 0], device=self.device,
                                         dtype=torch.int64), src=0)
        exchange_muon_state(self.dist, self.optimizer, self.device, to_root=True)

    def _scatter_muon(self) -> None:
        self.dist.broadcast(torch.tensor([_SCATTER, 0, 0, 0, 0, 0], device=self.device,
                                         dtype=torch.int64), src=0)
        exchange_muon_state(self.dist, self.optimizer, self.device, to_root=False)
        self.scatter_pending = False

    def send_step(self, values, masks, groups, n_splits: int, gen, lr: float = 0.0) -> None:
        if not self.synced:
            raise RuntimeError("gang weights not synchronised before the first step")
        if self.scatter_pending:
            self._scatter_muon()
        b, c, length = values.shape
        header = torch.tensor([_STEP, b, c, length, int(groups is not None), n_splits],
                              device=self.device, dtype=torch.int64)
        self.dist.broadcast(header, src=0)
        _send_batch(self.dist, self.device, values, masks, groups)
        state = gen.get_state().to(self.device)
        self.dist.broadcast(state, src=0)
        self.dist.broadcast(torch.tensor([float(lr)], device=self.device, dtype=torch.float64), src=0)

    def stop(self, *, clean: bool) -> None:
        """Release the helpers. ``clean`` = they are waiting for a step header;
        otherwise (a failed step) a broadcast could block, so terminate them."""
        if clean and self.synced:
            with contextlib.suppress(Exception):
                header = torch.zeros(6, device=self.device, dtype=torch.int64)
                header[0] = _STOP
                self.dist.broadcast(header, src=0)
        for p in self.procs:
            if clean:
                p.join(timeout=60)
            if p.is_alive():
                p.terminate()
        with contextlib.suppress(Exception):
            self.dist.destroy_process_group()

    def raise_follower_error(self, exc: BaseException) -> None:
        """Re-raise ``exc``, unless a helper ran out of GPU memory: then raise
        that OOM, so the failure is classified as the sizing fault it is."""
        try:
            name, msg = self.errors.get(timeout=5)
        except Exception:  # noqa: BLE001 — no helper report
            raise exc from None
        if name == "OutOfMemoryError":
            raise torch.OutOfMemoryError(f"rank helper: {msg}") from exc
        raise exc


def state_sources(model, optimizer) -> list[int]:
    """The broadcast source rank of each ``model.state_dict()`` entry, in
    order: a muon param's owner, else rank 0. Identical on every rank."""
    index = {id(p): i for i, p in enumerate(optimizer.muon_params)}
    owner = {name: optimizer.muon_owner[index[id(p)]]
             for name, p in model.named_parameters() if id(p) in index}
    return [owner.get(name, 0) for name in model.state_dict()]


def exchange_muon_state(dist, optimizer, device, *, to_root: bool) -> None:
    """Move each non-root-owned muon param's momentum and row variance
    between its owner and rank 0 (``to_root``: owner -> 0, else 0 -> owner).
    Every rank walks the same params in the same order; a presence flag goes
    first, since a param never stepped has no state yet."""
    me = int(optimizer.rank)
    for i, p in enumerate(optimizer.muon_params):
        owner = optimizer.muon_owner[i]
        if owner == 0:
            continue
        src = owner if to_root else 0
        gshape = tuple(p.mT.shape) if p.ndim == 3 else tuple(p.shape)
        flag = torch.tensor([int(p in optimizer._bufs and p in optimizer._row_v)],
                            device=device, dtype=torch.int64)
        dist.broadcast(flag, src=src)
        if not int(flag.item()):
            continue
        keep = me == (0 if to_root else owner)
        for store, shape in ((optimizer._bufs, gshape), (optimizer._row_v, gshape[:-1] + (1,))):
            buf = (store[p].contiguous() if me == src
                   else torch.empty(shape, device=device, dtype=p.dtype))
            dist.broadcast(buf, src=src)
            if keep:
                store[p] = buf
        if not to_root and me == 0:
            # rank 0's copy is now the owner's: drop it so a read without a
            # gather fails loudly instead of using a stale state
            optimizer._bufs.pop(p, None)
            optimizer._row_v.pop(p, None)


def _send_batch(dist, device, values, masks, groups) -> None:
    dist.broadcast(torch.as_tensor(values, device=device, dtype=torch.float64), src=0)
    dist.broadcast(torch.as_tensor(np.ascontiguousarray(masks), device=device), src=0)
    if groups is not None:
        dist.broadcast(torch.as_tensor(groups, device=device, dtype=torch.int64), src=0)


def micro_batch_pass(model, params, values, masks, groups, splits, mine, cop_gen, device, *,
                     step: int, tokens: int, gang: Gang | None):
    """The step's micro-batch work on this rank, summed across the gang.

    Leaves the summed copula gradient in ``p.grad`` and returns ``(reference,
    loss_m, loss_j, load)``: the summed marginal gradient, both batch-weighted
    losses and the summed routing loads. With ``gang=None`` and every split
    owned this is the one-GPU step, operation for operation."""
    from .pyramid_trainer import losses, prediction
    from .toto2_trainer import check_loss_finite

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
        from .pyramid_fast import finish_aux

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


def _follower(rank, world, port, cfg_dict, deterministic, seed, errors) -> None:
    """Helper rank: hold the model, run owned micro-batches, take weights."""
    import types

    import torch.distributed as dist

    from .pyramid_model import CascadeMoEConfig, CascadeMoEModel
    from .toto2_trainer import Toto2Trainer

    try:
        torch.cuda.set_device(rank)
        bound_cpu_threads(world)
        device = torch.device("cuda", rank)
        dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                                world_size=world, timeout=_TIMEOUT)
        torch.manual_seed(seed)
        Toto2Trainer._enable_determinism(types.SimpleNamespace(deterministic=deterministic),
                                         torch, seed)
        model = CascadeMoEModel(CascadeMoEConfig.from_dict(cfg_dict)).to(device)
        model.train()
        from .pyramid_fast import accelerate

        accelerate(model)               # the gang only trains lane A (no slices)
        params = list(model.parameters())
        gen = torch.Generator(device=device)
        gang = types.SimpleNamespace(rank=rank, dist=dist, device=device)
        gang.sum_ = lambda ts: [dist.all_reduce(t, op=dist.ReduceOp.SUM) for t in ts]

        def any_(flag):
            t = torch.tensor([1 if flag else 0], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return bool(t.item())

        gang.any_ = any_
        from .pyramid_optimizer import assign_muon_owners, build_optimizer
        from .pyramid_trainer import combine_gradients

        optimizer = build_optimizer(model, model.cfg.d_model)
        optimizer.muon_owner = assign_muon_owners(optimizer.muon_params, world)
        optimizer.rank = rank
        sources = state_sources(model, optimizer)
        for t in model.state_dict().values():
            dist.broadcast(t, src=0)
        while True:
            header = torch.zeros(6, device=device, dtype=torch.int64)
            dist.broadcast(header, src=0)
            if int(header[0]) == _STOP:
                break
            if int(header[0]) in (_GATHER, _SCATTER):
                exchange_muon_state(dist, optimizer, device, to_root=int(header[0]) == _GATHER)
                continue
            b, c, length, has_groups, n_splits = (int(v) for v in header[1:].tolist())
            values = torch.empty((b, c, length), device=device, dtype=torch.float64)
            dist.broadcast(values, src=0)
            masks = torch.empty((b, c, length // model.cfg.patch_size), device=device,
                                dtype=torch.bool)
            dist.broadcast(masks, src=0)
            groups = None
            if has_groups:
                groups = torch.empty((b, c), device=device, dtype=torch.int64)
                dist.broadcast(groups, src=0)
            state = gen.get_state().to(device)
            dist.broadcast(state, src=0)
            gen.set_state(state.cpu())
            lr = torch.zeros(1, device=device, dtype=torch.float64)
            dist.broadcast(lr, src=0)
            splits = np.array_split(np.arange(b), n_splits)
            for p in params:
                p.grad = None
            reference, *_ = micro_batch_pass(
                model, params, values, masks.cpu().numpy(),
                None if groups is None else groups.cpu().numpy(), splits,
                owned_splits(len(splits), rank, world), gen, device,
                step=0, tokens=0, gang=gang)
            # The same gradients as rank 0 (all-reduced), so the same combine
            # and clip, then this rank's share of the muon step.
            combine_gradients(params, reference)
            torch.nn.utils.clip_grad_norm_(params, 7.0)
            optimizer.step_muon(float(lr.item()))
            del reference
            for p in params:
                p.grad = None
            for t, src in zip(model.state_dict().values(), sources, strict=True):
                dist.broadcast(t, src=src)
    except BaseException as e:  # noqa: BLE001 — rank 0 re-raises it by type
        with contextlib.suppress(Exception):
            errors.put((type(e).__name__, str(e)[:500]))
        raise
    finally:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()
