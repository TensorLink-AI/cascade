"""Power EMA on the v2 pyramid trainer — IMAGE-ONLY (owner 2026-11-01): the
settings are fixed worker constants (``pyramid_trainer.POWER_EMA_*``), never
contract terms, so ``contract_digest`` is untouched. The scored
``weights.safetensors`` is the sigma_rel=0.10 per-leg power EMA
(``moe_trainer.PowerEMA`` math), the raw lineage (``weights_stable`` +
``optimizer``) is untouched, continuations carry the averages bitwise, lane-B
frozen bytes survive and the routing bias stays live. CPU only; the
multi-rank case runs on the gloo backend. "plain" runs switch the EMA off by
monkeypatching the module constants (tests only).
"""
from __future__ import annotations

import socket
import types
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from safetensors.torch import load_file

from cascade.eval.checkpoint_guard import verify_checkpoint
from cascade.shared.config import parse_chain_config
from cascade.shared.manifest import contract_digest, contract_payload
from cascade.trainer import pyramid_trainer
from cascade.trainer.continuation import Continuation, SnapshotStore
from cascade.trainer.moe_trainer import POWER_EMA_FILE, PowerEMA, gamma_for_sigma_rel
from cascade.trainer.pyramid_model import (
    CascadeMoEConfig,
    CascadeMoEModel,
    merge_expert_slices,
    moe_param_names,
)
from cascade.trainer.pyramid_trainer import power_ema_settings
from cascade.trainer.pyramid_wrapper import Wrapper
from cascade.trainer.toto2_trainer import OPTIM_STATE_FILE, STABLE_WEIGHTS_FILE, Toto2Trainer
from cascade.validator.private_worker import CHECKPOINT_FILES, checkpoint_hashes
from scripts.pyramid_config import local_config

SIGMAS = (0.05, 0.10)
EVAL = 1


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def contract():
    c = parse_chain_config(local_config("2m")).training
    return replace(c, training_steps=3, batch_size=8, max_train_seconds=90)


@pytest.fixture
def plain(monkeypatch):
    """Returns a context manager in which the image's power EMA is off (tests only)."""
    import contextlib

    @contextlib.contextmanager
    def ctx():
        with monkeypatch.context() as m:
            m.setattr(pyramid_trainer, "POWER_EMA_SIGMA_RELS", ())
            m.setattr(pyramid_trainer, "POWER_EMA_EVAL", -1)
            yield
    return ctx


def data():
    rng = np.random.default_rng(84)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def train(path, contract, **kwargs):
    return Toto2Trainer(device="cpu").train(iter(data()), contract,
        training_seed=1, token_budget=contract.train_tokens, out_dir=path, **kwargs)


def train_plain(plain, path, contract, **kwargs):
    with plain():
        return train(path, contract, **kwargs)


class Recorder:
    """A continuation that is due after every step and only records: the raw
    model state and EMA state after each optimizer step (no numerics change)."""

    def __init__(self):
        self.store = types.SimpleNamespace(load=lambda: None)
        self.states = []

    def due(self):
        return True

    def save(self, state):
        self.states.append({"model": {k: v.detach().clone() for k, v in state["model"].items()},
                            "emas": {k: v.clone() for k, v in state["emas"].items()}})


def _is_buffer(name):
    return name.endswith("route_bias")


def _files(path):
    return {name: load_file(str(path / name))
            for name in ("weights.safetensors", STABLE_WEIGHTS_FILE, OPTIM_STATE_FILE)}


def _equal(left, right):
    assert left.keys() == right.keys()
    for k in left:
        assert left[k].dtype == right[k].dtype and torch.equal(left[k], right[k]), k


# ── (b) the formula, on a toy sequence ────────────────────────────────────────


def test_power_ema_matches_the_reference_formula_on_a_toy_sequence():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    seq = [torch.randn(2, 3) for _ in range(9)]
    for sigma in SIGMAS:
        ema = PowerEMA(model, sigma)
        gamma = gamma_for_sigma_rel(sigma)
        ref = np.zeros((2, 3), dtype=np.float64)           # independent numpy reference
        for t, x in enumerate(seq, start=1):
            with torch.no_grad():
                model.weight.copy_(x)
            ema.update(model)
            beta = (1 - 1 / t) ** (gamma + 1)            # average = β·previous + (1−β)·current
            ref = beta * ref + (1 - beta) * x.numpy().astype(np.float64)
        assert ema.t == len(seq)
        np.testing.assert_allclose(ema.shadow["weight"].double().numpy(), ref, rtol=1e-6, atol=1e-7)
        assert ema.export(model)["bias"].dtype == model.bias.dtype


# ── (b)(c)(f)(g) a real leg ───────────────────────────────────────────────────


def test_scored_file_is_the_selected_ema_and_training_is_unperturbed(tmp_path, plain, contract):
    plain = train_plain(plain, tmp_path / "plain", contract)
    rec = Recorder()
    armed = train(tmp_path / "armed", contract, continuation=rec)
    assert len(rec.states) == 3 and armed.metrics["power_ema_steps"] == 3
    assert armed.metrics["power_ema_eval_sigma_rel"] == SIGMAS[EVAL]

    # (c) the raw endpoint, stable snapshot and optimizer are bit-identical
    p, a = _files(plain.local_dir), _files(armed.local_dir)
    _equal(p[STABLE_WEIGHTS_FILE], a[STABLE_WEIGHTS_FILE])
    _equal(p[OPTIM_STATE_FILE], a[OPTIM_STATE_FILE])
    _equal(p["weights.safetensors"], {k: v.contiguous() for k, v in rec.states[-1]["model"].items()})

    # (b) weights.safetensors = the selected EMA, replayed from the per-step raw weights
    scored = a["weights.safetensors"]
    gamma = gamma_for_sigma_rel(SIGMAS[EVAL])
    shadow = None
    for t, state in enumerate(rec.states, start=1):
        beta = (1 - 1 / t) ** (gamma + 1)
        cur = {k: v.float() for k, v in state["model"].items() if not _is_buffer(k)}
        if shadow is None:
            shadow = {k: v.clone() for k, v in cur.items()}
        for k, v in cur.items():
            shadow[k].mul_(beta).add_(v, alpha=1 - beta)
    moved = 0
    for k, v in shadow.items():
        assert torch.equal(scored[k], v), k
        moved += not torch.equal(scored[k], p["weights.safetensors"][k])
    assert moved                                           # the EMA is not the raw endpoint
    key = f"ema{SIGMAS[EVAL]:g}"
    assert int(rec.states[-1]["emas"][f"{key}.__t__"].item()) == 3
    for k, v in shadow.items():
        assert torch.equal(rec.states[-1]["emas"][f"{key}.{k}"], v)

    # (f) the routing bias is the live model's, never averaged
    bias = [k for k in scored if _is_buffer(k)]
    assert bias
    for k in bias:
        assert torch.equal(scored[k], p["weights.safetensors"][k])
        assert k not in shadow

    # (g) every checkpoint consumer still validates the same file layout
    report = verify_checkpoint(armed.local_dir, contract)
    assert report.n_params == verify_checkpoint(plain.local_dir, contract).n_params
    assert set(checkpoint_hashes(armed.local_dir)) == set(CHECKPOINT_FILES)
    assert not (armed.local_dir / POWER_EMA_FILE).exists()
    assert sorted(f.name for f in armed.local_dir.iterdir()) == sorted(
        f.name for f in plain.local_dir.iterdir())
    q = Wrapper(armed.local_dir).forecast_quantiles_batch([np.arange(4096.0)], 64)
    assert q.shape == (1, 64, 21) and np.isfinite(q).all()


def test_warm_start_resumes_the_raw_lineage_and_restarts_the_ema(tmp_path, plain, contract):
    """Future A: a warm start reads weights_stable + optimizer (raw), never the
    EMA; with and without the EMA the next leg's lineage is bit-identical, and
    the warm-started leg's average starts fresh (scope = per leg)."""
    base = train(tmp_path / "base", contract)
    plain = train_plain(plain, tmp_path / "plain", contract, warm_start_dir=base.local_dir)
    armed = train(tmp_path / "armed", contract, warm_start_dir=base.local_dir)
    assert armed.metrics["optim_state_resumed"] and armed.metrics["power_ema_steps"] == 3
    p, a = _files(plain.local_dir), _files(armed.local_dir)
    _equal(p[STABLE_WEIGHTS_FILE], a[STABLE_WEIGHTS_FILE])
    _equal(p[OPTIM_STATE_FILE], a[OPTIM_STATE_FILE])


# ── (d) crash + resume ────────────────────────────────────────────────────────


def test_restart_reproduces_the_uninterrupted_ema(tmp_path, contract):
    class Recording(Continuation):
        """Saves like the real continuation and records each step's EMA state."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.emas = {}

        def save(self, state):
            self.emas[state["step"]] = {k: v.clone() for k, v in state["emas"].items()}
            super().save(state)

    full = Recording(SnapshotStore(tmp_path / "full", {"contract": "y", "seed": 1}), interval=0)
    expected = train(tmp_path / "expected", contract, continuation=full)
    store = SnapshotStore(tmp_path / "snapshots", {"contract": contract_digest(contract), "seed": 1})

    class CrashAfterSecond(Continuation):
        def save(self, state):
            self.store.save(state["step"], state)
            if state["step"] == 2:
                raise RuntimeError("simulated pod loss")

    with pytest.raises(RuntimeError, match="simulated pod loss"):
        train(tmp_path / "failed", contract, continuation=CrashAfterSecond(store, interval=0))
    snap = store.load()
    assert snap["emas"]                                    # the snapshot carries the averages
    _equal(snap["emas"], full.emas[2])                     # ... bitwise, as of the crash step
    resumed = Recording(store, interval=0)
    result = train(tmp_path / "recovered", contract, continuation=resumed)
    assert result.metrics["power_ema_steps"] == 3
    assert sorted(resumed.emas) == [3]
    _equal(resumed.emas[3], full.emas[3])                  # restored EMA states continue bitwise
    left, right = _files(expected.local_dir), _files(result.local_dir)
    for name in left:
        _equal(left[name], right[name])


def test_a_snapshot_without_ema_state_is_refused(tmp_path, plain, contract):
    store = SnapshotStore(tmp_path / "snapshots", {"contract": "x", "seed": 1})

    class CrashAfterFirst(Continuation):
        def save(self, state):
            self.store.save(state["step"], state)
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        train_plain(plain, tmp_path / "failed", contract, continuation=CrashAfterFirst(store, interval=0))
    with pytest.raises(ValueError, match="power EMA state"):
        train(tmp_path / "resumed", contract, continuation=Continuation(store, interval=0))


# ── (e) lane B ────────────────────────────────────────────────────────────────


def test_expert_leg_averages_only_its_rows_and_keeps_frozen_bytes(tmp_path, plain, contract):
    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_trainer import losses

    base = train(tmp_path / "base", contract)
    model = load_king(base.local_dir)
    model.train()
    losses(model, torch.as_tensor(data()[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    selection = {i: layer.last_load.topk(4).indices.tolist() for i, layer in enumerate(model.moe_layers())}
    names = moe_param_names(model)
    raw = train_plain(plain, tmp_path / "raw", contract, warm_start_dir=base.local_dir,
                      expert_selection=selection)
    leg = train(tmp_path / "leg", contract, warm_start_dir=base.local_dir, expert_selection=selection)
    king = load_file(str(base.local_dir / "weights.safetensors"))
    cand = load_file(str(leg.local_dir / "weights.safetensors"))
    raw_cand = load_file(str(raw.local_dir / "weights.safetensors"))
    _equal(merge_expert_slices(king, cand, selection, names), cand)
    owned = {n: li for li, pair in enumerate(names) for n in pair}
    changed = False
    for k, v in king.items():
        if k not in owned:
            assert torch.equal(cand[k], v), k                 # whole frozen tensor, byte for byte
            continue
        rows = sorted(selection.get(owned[k], []))
        keep = torch.ones(v.shape[0], dtype=torch.bool)
        keep[rows] = False
        assert torch.equal(cand[k][keep], v[keep]), k        # unselected expert rows
        changed |= not torch.equal(cand[k][rows], raw_cand[k][rows])
    assert changed                                           # selected rows carry the leg's EMA
    for f in (STABLE_WEIGHTS_FILE, OPTIM_STATE_FILE, POWER_EMA_FILE):
        assert not (leg.local_dir / f).exists()


# ── (a) one rank vs a two-rank (gloo) gang ────────────────────────────────────

CFG = dict(d_model=8, num_layers=3, num_heads=2, head_dim=4, patch_size=4, dense_ffn=12,
           num_experts=8, top_k=2, expert_hidden=5, shared_hidden=7, conv_channels=4,
           moe_blocks=(1, 2), variate_blocks=())
B, C, L, SPLITS, STEPS = 4, 2, 16, 4, 3


def _batch(step):
    rng = np.random.default_rng(100 + step)
    values = np.cumsum(rng.standard_normal((B, C, L)), axis=-1)
    masks = np.zeros((B, C, L // 4), dtype=bool)
    masks[:, :, -1] = True
    return values, masks


def _run(gang, rank, world):
    """STEPS pyramid steps exactly as train_pyramid / the gang follower take
    them: shared micro-batch pass, combine, clip, each rank's optimizer share,
    Gang.sync_after_step, then the power-EMA update."""
    from cascade.trainer.pyramid_dp import Gang, micro_batch_pass, owned_splits, state_sources
    from cascade.trainer.pyramid_optimizer import assign_muon_owners, build_optimizer
    from cascade.trainer.pyramid_trainer import combine_gradients

    torch.manual_seed(7)
    model = CascadeMoEModel(CascadeMoEConfig(**CFG))
    model.train()
    params = list(model.parameters())
    optimizer = build_optimizer(model, model.cfg.d_model)
    sync = None
    if gang is not None:
        optimizer.muon_owner = assign_muon_owners(optimizer.muon_params, world)
        optimizer.rank = rank
        sync = object.__new__(Gang)
        sync.dist, sync._sources = gang.dist, state_sources(model, optimizer)
        # #399 (pyramid phase B): sync_after_step passes the gang's device to
        # broadcast_from_sources (NCCL-grouped on cuda, plain on gloo/cpu)
        sync.device = gang.device
    emas = [PowerEMA(model, s) for s in SIGMAS]
    gen = torch.Generator().manual_seed(11)
    traj = []                                   # raw parameters after every step
    for step in range(STEPS):
        values, masks = _batch(step)
        for p in params:
            p.grad = None
        splits = np.array_split(np.arange(B), SPLITS)
        mine = owned_splits(SPLITS, rank, world)
        ref, _, _, load = micro_batch_pass(model, params, values, masks, None, splits, mine, gen,
                                           torch.device("cpu"), step=step, tokens=0, gang=gang)
        combine_gradients(params, ref)
        torch.nn.utils.clip_grad_norm_(params, 7.0)
        if rank == 0:
            optimizer.step()
            for layer, total in zip(model.moe_layers(), load, strict=True):
                layer.update_bias(total)
        else:
            optimizer.step_muon(optimizer.param_groups[0]["lr"])
        if sync is not None:
            sync.sync_after_step(model)
        for ema in emas:
            ema.update(model)
        traj.append({n: p.detach().clone() for n, p in model.named_parameters()})
    flat = {}
    for ema in emas:
        flat.update(ema.state_tensors())
    return {"emas": {k: v.clone() for k, v in flat.items()}, "traj": traj,
            "model": {k: v.detach().clone() for k, v in model.state_dict().items()}}


def _rank(rank, world, port, out):
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank,
                            world_size=world)
    gang = types.SimpleNamespace(rank=rank, dist=dist, device=torch.device("cpu"))
    gang.sum_ = lambda ts: [dist.all_reduce(t) for t in ts]

    def any_(flag):
        t = torch.tensor([int(flag)])
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t.item())

    gang.any_ = any_
    torch.save(_run(gang, rank, world), f"{out}/rank{rank}.pt")
    dist.destroy_process_group()


def test_two_rank_gang_ema_matches_one_rank(tmp_path):
    import torch.multiprocessing as mp

    one = _run(None, 0, 1)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    mp.spawn(_rank, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt") for r in (0, 1))
    # after sync_after_step every rank holds rank 0's bytes, so the average
    # (kept on rank 0 in train_pyramid) is rank-independent, bit for bit
    _equal(r0["model"], r1["model"])
    _equal(r0["emas"], r1["emas"])
    # vs one process: only the all-reduce's summation order differs, and
    # NorMuon amplifies that rounding in the RAW weights (CI saw up to 3.4e-5
    # after 3 steps), which is the existing DP behaviour, not the EMA's. The
    # EMA is a convex combination of the raw iterates, so it may drift from the
    # one-rank average by no more than the raw trajectory itself drifts (plus
    # fp32 rounding of the update): it adds no rank dependence of its own.
    assert r0["emas"].keys() == one["emas"].keys()
    assert len(r0["traj"]) == len(one["traj"]) == STEPS
    for k, v in one["emas"].items():
        if k.endswith(".__t__"):
            assert torch.equal(r0["emas"][k], v)
            continue
        prefix = [f"ema{s:g}." for s in SIGMAS if k.startswith(f"ema{s:g}.")][0]
        name = k[len(prefix):]
        drift = 0.0
        for many, single in zip(r0["traj"], one["traj"], strict=True):
            drift = max(drift, (many[name] - single[name]).abs().max().item())
        slack = 1e-6 * v.abs().max().item() + 1e-9
        assert (r0["emas"][k] - v).abs().max().item() <= drift + slack, k
    assert int(r0["emas"][f"ema{SIGMAS[0]:g}.__t__"].item()) == STEPS


# ── image-only: the EMA is NOT a contract term ────────────────────────────────


def test_fixed_settings_are_the_owner_values():
    assert pyramid_trainer.POWER_EMA_SIGMA_RELS == (0.05, 0.10)
    assert pyramid_trainer.POWER_EMA_EVAL == 1
    assert power_ema_settings() == ((0.05, 0.10), 1)        # the 0.10 average is scored


def test_bad_fixed_settings_are_refused(monkeypatch):
    for sigmas, index in (((0.05, 0.10), 2), ((0.05, 0.10), -1), ((), 0), ((0.05, 0.3), 0)):
        with monkeypatch.context() as m:
            m.setattr(pyramid_trainer, "POWER_EMA_SIGMA_RELS", sigmas)
            m.setattr(pyramid_trainer, "POWER_EMA_EVAL", index)
            with pytest.raises(ValueError, match="power EMA"):
                power_ema_settings()


def test_the_contract_cannot_arm_the_pyramid_ema(tmp_path, contract):
    """The contract's own power_ema_* fields stay pinned at their defaults for
    the pyramid, so the average can only come from the image."""
    with pytest.raises(ValueError, match="power_ema_sigma_rels"):
        train(tmp_path, replace(contract, power_ema_sigma_rels=SIGMAS, power_ema_eval=EVAL))


@pytest.mark.parametrize("name", ["chain.toml", "chain.testnet.toml"])
def test_deployed_contract_digest_is_independent_of_the_image_ema(monkeypatch, name):
    """contract_digest of the repo configs is byte-identical to main's: the
    loader reads no EMA setting for the pyramid and the image constants never
    enter the hashed payload."""
    import tomllib

    from scripts.pyramid_config import ROOT

    def digests():
        cfg = parse_chain_config(tomllib.loads((ROOT / name).read_text())).training
        return cfg, contract_digest(cfg), contract_digest(cfg.v2_contract)

    cfg, legacy, v2 = digests()
    for c in (cfg, cfg.v2_contract):
        assert c.power_ema_sigma_rels == () and c.power_ema_eval == -1
        payload = contract_payload(c)
        assert "power_ema_sigma_rels" not in payload and "power_ema_eval" not in payload
    for sigmas, index in (((), -1), ((0.2,), 0), ((0.05, 0.10), 0)):
        with monkeypatch.context() as m:
            m.setattr(pyramid_trainer, "POWER_EMA_SIGMA_RELS", sigmas)
            m.setattr(pyramid_trainer, "POWER_EMA_EVAL", index)
            assert digests()[1:] == (legacy, v2)
