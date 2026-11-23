"""Power EMA on the v2 pyramid trainer — IMAGE-ONLY (owner 2026-11-01): the
settings are fixed worker constants (``pyramid_trainer.POWER_EMA_*``), never
contract terms, so ``contract_digest`` is untouched. Every leg computes the
sigma_rel 0.05 / 0.10 per-leg power EMAs (``moe_trainer.PowerEMA`` math), the
raw lineage (``weights_stable`` + ``optimizer``) is untouched, continuations
carry the averages bitwise, lane-B frozen bytes survive and the routing bias
stays live. CPU only; the multi-rank case runs on the gloo backend. "plain"
runs switch the EMA off by monkeypatching the module constants (tests only).

Activation (lane D W-EMA, plan rev5 §8): the EMA is SCORED on NO leg. Every
leg of lanes A and B — random init, a warm-start king / challenger (also from
an EMA-marked init), a lane-B expert on a raw or an EMA-marked king — scores
the raw cooled endpoint, byte-identical to the pre-EMA trainer, and writes
both averages unscored as weights_ema005 / weights_ema010 (no weights_raw, no
config.json marker).
"""
from __future__ import annotations

import contextlib
import json
import socket
import types
from dataclasses import replace
from pathlib import Path

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
from cascade.trainer.pyramid_trainer import (
    RAW_WEIGHTS_FILE,
    ema_weights_file,
    power_ema_active,
    power_ema_settings,
)
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
    @contextlib.contextmanager
    def ctx():
        with monkeypatch.context() as m:
            m.setattr(pyramid_trainer, "POWER_EMA_SIGMA_RELS", ())
            m.setattr(pyramid_trainer, "POWER_EMA_EVAL", -1)
            yield
    return ctx


def data(rng_seed=84):
    rng = np.random.default_rng(rng_seed)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def train(path, contract, *, seed=1, rng_seed=84, **kwargs):
    return Toto2Trainer(device="cpu").train(iter(data(rng_seed)), contract,
        training_seed=seed, token_budget=contract.train_tokens, out_dir=path, **kwargs)


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
        emas = state.get("emas")                           # absent on a raw leg
        self.states.append({"model": {k: v.detach().clone() for k, v in state["model"].items()},
                            "emas": None if emas is None else {k: v.clone() for k, v in emas.items()}})


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


# ── activation: no leg scores an average (W-EMA: raw endpoint everywhere) ────

MARKER = {"sigma_rels": [0.05, 0.10], "scored_sigma_rel": 0.10}
EMA005, EMA010, RAW = ema_weights_file(0.05), ema_weights_file(0.10), RAW_WEIGHTS_FILE


def _bytes(path):
    return {f.name: f.read_bytes() for f in sorted(Path(path).iterdir())}


def _config(path):
    return json.loads((Path(path) / "config.json").read_text())


def _weights(path, name="weights.safetensors"):
    return load_file(str(Path(path) / name))


def _replay(states, sigma):
    """The reference average of width ``sigma``, replayed from the raw
    per-step weights the continuation recorded (buffers stay live)."""
    gamma = gamma_for_sigma_rel(sigma)
    shadow = None
    for t, state in enumerate(states, start=1):
        beta = (1 - 1 / t) ** (gamma + 1)
        cur = {k: v.float() for k, v in state["model"].items() if not _is_buffer(k)}
        if shadow is None:
            shadow = {k: v.clone() for k, v in cur.items()}
        for k, v in cur.items():
            shadow[k].mul_(beta).add_(v, alpha=1 - beta)
    return shadow


def _assert_average(saved, shadow, raw):
    """``saved`` (an exported average) = the replayed shadow for every
    parameter and the raw endpoint for every buffer (the routing bias)."""
    assert saved.keys() == raw.keys()
    for k, v in saved.items():
        assert torch.equal(v, shadow[k] if k in shadow else raw[k]), k
    assert any(_is_buffer(k) for k in saved) and not any(_is_buffer(k) for k in shadow)


def test_extra_file_names():
    assert (EMA005, EMA010, RAW) == ("weights_ema005.safetensors", "weights_ema010.safetensors",
                                     "weights_raw.safetensors")


def test_activation_rule_scores_raw_for_every_init(tmp_path):
    """One rule for every leg (lanes A and B, any init): never "best of both"
    per model; switching the scored file is a later owner activation (D45)."""
    raw, marked, broken = (tmp_path / n for n in ("raw", "marked", "broken"))
    for d in (raw, marked, broken):
        d.mkdir()
    (raw / "config.json").write_text(json.dumps({"moe": {}}))
    (marked / "config.json").write_text(json.dumps({"moe": {}, "power_ema": MARKER}))
    (broken / "config.json").write_text("{")
    sel = {0: [1]}
    assert power_ema_active(None, None) is False           # random init
    assert power_ema_active(raw, None) is False            # promoted init: king/challenger
    assert power_ema_active(marked, None) is False         # init from an EMA-scored image
    assert power_ema_active(str(marked), None) is False
    assert power_ema_active(raw, sel) is False             # expert on a raw king
    assert power_ema_active(marked, sel) is False          # expert on an EMA-marked king
    assert power_ema_active(broken, sel) is False
    assert power_ema_active(tmp_path / "missing", sel) is False


def test_shadow_epoch_scores_raw_bitwise_and_writes_both_averages(tmp_path, plain, contract):
    """A random-init leg: weights.safetensors and every
    pre-existing file are byte-identical to the trainer without the feature;
    the two averages ride along as weights_ema005 / weights_ema010 and equal
    the reference replay; no marker, no weights_raw."""
    ref = train_plain(plain, tmp_path / "ref", contract)
    rec = Recorder()
    leg = train(tmp_path / "leg", contract, continuation=rec)
    got, want = _bytes(leg.local_dir), _bytes(ref.local_dir)
    assert set(got) == set(want) | {EMA005, EMA010}
    assert {k: got[k] for k in want} == want
    assert "power_ema" not in _config(leg.local_dir)
    assert leg.metrics["power_ema_scored"] is False and leg.metrics["power_ema_steps"] == 3
    assert "power_ema_eval_sigma_rel" not in leg.metrics
    raw = _weights(leg.local_dir)
    _equal(raw, {k: v.contiguous() for k, v in rec.states[-1]["model"].items()})
    for sigma, name in ((0.05, EMA005), (0.10, EMA010)):
        _assert_average(_weights(leg.local_dir, name), _replay(rec.states, sigma), raw)
    assert all(s["emas"] for s in rec.states)              # snapshots carry the shadows
    # every checkpoint consumer reads only the four checkpoint files
    verify_checkpoint(leg.local_dir, contract)
    assert checkpoint_hashes(leg.local_dir) == checkpoint_hashes(ref.local_dir)


def test_king_and_challenger_of_a_warm_start_epoch_both_score_raw(tmp_path, contract):
    """The king's retrain and a challenger of a warm-start epoch, with
    different seeds and corpora, both score the raw endpoint (W-EMA)."""
    init = train(tmp_path / "init", contract)                 # last epoch's raw king
    king = train(tmp_path / "king", contract, warm_start_dir=init.local_dir)
    challenger = train(tmp_path / "challenger", contract, warm_start_dir=init.local_dir,
                       seed=7, rng_seed=91)
    for leg in (king, challenger):
        assert leg.metrics["power_ema_scored"] is False and leg.metrics["power_ema_steps"] == 3
        assert "power_ema_eval_sigma_rel" not in leg.metrics
        assert "power_ema" not in _config(leg.local_dir)
        assert not (leg.local_dir / RAW).exists()
        assert (leg.local_dir / EMA005).is_file() and (leg.local_dir / EMA010).is_file()
        verify_checkpoint(leg.local_dir, contract)


# ── (b)(c)(f)(g) a warm-start leg: raw scored, both averages unscored ────────


def test_warm_start_scores_the_raw_endpoint_and_training_is_unperturbed(tmp_path, plain, contract):
    init = train(tmp_path / "init", contract)
    plain = train_plain(plain, tmp_path / "plain", contract, warm_start_dir=init.local_dir)
    rec = Recorder()
    armed = train(tmp_path / "armed", contract, warm_start_dir=init.local_dir, continuation=rec)
    assert len(rec.states) == 3 and armed.metrics["power_ema_steps"] == 3
    assert armed.metrics["power_ema_scored"] is False

    # (c) every file of the EMA-off leg (the raw endpoint, config.json without
    # a marker, the stable snapshot and optimizer) is byte-identical
    got, want = _bytes(armed.local_dir), _bytes(plain.local_dir)
    assert set(got) == set(want) | {EMA005, EMA010}
    assert {k: got[k] for k in want} == want
    p, a = _files(plain.local_dir), _files(armed.local_dir)
    raw = a["weights.safetensors"]
    _equal(raw, {k: v.contiguous() for k, v in rec.states[-1]["model"].items()})

    # (b) weights_ema005 / weights_ema010 = the replayed 0.05 / 0.10 averages;
    # (f) the routing bias is live; they are real averages, not the raw endpoint
    for sigma, name in ((0.05, EMA005), (0.10, EMA010)):
        shadow = _replay(rec.states, sigma)
        avg = _weights(armed.local_dir, name)
        _assert_average(avg, shadow, raw)
        assert any(not torch.equal(avg[k], raw[k]) for k in shadow), name
        key = f"ema{sigma:g}"
        assert int(rec.states[-1]["emas"][f"{key}.__t__"].item()) == 3
        for k, v in shadow.items():
            assert torch.equal(rec.states[-1]["emas"][f"{key}.{k}"], v)

    # (g) every checkpoint consumer still validates and hashes only the four
    # checkpoint files, equal to the EMA-off leg's
    report = verify_checkpoint(armed.local_dir, contract)
    assert report.n_params == verify_checkpoint(plain.local_dir, contract).n_params
    assert set(checkpoint_hashes(armed.local_dir)) == set(CHECKPOINT_FILES)
    assert checkpoint_hashes(armed.local_dir) == checkpoint_hashes(plain.local_dir)
    assert not (armed.local_dir / POWER_EMA_FILE).exists()
    _equal(p[STABLE_WEIGHTS_FILE], a[STABLE_WEIGHTS_FILE])
    q = Wrapper(armed.local_dir).forecast_quantiles_batch([np.arange(4096.0)], 64)
    assert q.shape == (1, 64, 21) and np.isfinite(q).all()


def test_warm_start_resumes_the_raw_lineage_and_restarts_the_ema(tmp_path, plain, contract):
    """Future A: a warm start reads weights_stable + optimizer (raw), never the
    EMA; with and without the EMA the next leg's lineage is bit-identical, and
    each warm-started leg's average starts fresh (scope = per leg) — also one
    epoch further on, from a warm-started king."""
    init = train(tmp_path / "init", contract)
    base = train(tmp_path / "base", contract, warm_start_dir=init.local_dir)
    plain = train_plain(plain, tmp_path / "plain", contract, warm_start_dir=base.local_dir)
    armed = train(tmp_path / "armed", contract, warm_start_dir=base.local_dir)
    assert armed.metrics["optim_state_resumed"] and armed.metrics["power_ema_steps"] == 3
    p, a = _files(plain.local_dir), _files(armed.local_dir)
    _equal(p[STABLE_WEIGHTS_FILE], a[STABLE_WEIGHTS_FILE])
    _equal(p[OPTIM_STATE_FILE], a[OPTIM_STATE_FILE])


# ── (d) crash + resume keeps the leg's decision ───────────────────────────────


class Recording(Continuation):
    """Saves like the real continuation and records each step's EMA state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emas = {}

    def save(self, state):
        self.emas[state["step"]] = {k: v.clone() for k, v in state["emas"].items()}
        super().save(state)


class CrashAt(Continuation):
    at = 2

    def save(self, state):
        self.store.save(state["step"], state)
        if state["step"] == self.at:
            raise RuntimeError("simulated pod loss")


@pytest.mark.parametrize("warm", [True, False])
def test_restart_reproduces_the_uninterrupted_leg_and_keeps_its_decision(tmp_path, contract, warm):
    init = train(tmp_path / "init", contract).local_dir if warm else None
    kw = {"warm_start_dir": init} if warm else {}
    full = Recording(SnapshotStore(tmp_path / "full", {"contract": "y", "seed": 1}), interval=0)
    expected = train(tmp_path / "expected", contract, continuation=full, **kw)
    store = SnapshotStore(tmp_path / "snapshots", {"contract": contract_digest(contract), "seed": 1})
    with pytest.raises(RuntimeError, match="simulated pod loss"):
        train(tmp_path / "failed", contract, continuation=CrashAt(store, interval=0), **kw)
    snap = store.load()
    assert snap["power_ema_scored"] is False               # raw scored on every leg (W-EMA)
    _equal(snap["emas"], full.emas[2])                     # the snapshot carries the averages bitwise
    resumed = Recording(store, interval=0)
    result = train(tmp_path / "recovered", contract, continuation=resumed, **kw)
    assert sorted(resumed.emas) == [3]
    _equal(resumed.emas[3], full.emas[3])                  # restored EMA states continue bitwise
    assert result.metrics["power_ema_scored"] is False and result.metrics["power_ema_steps"] == 3
    assert _bytes(expected.local_dir) == _bytes(result.local_dir)   # incl. the extra files


def test_a_snapshot_from_the_other_decision_or_without_ema_state_is_refused(
        tmp_path, plain, monkeypatch, contract):
    """A continuation never mixes scored files: a snapshot an earlier image
    made with the EMA scored (simulated by forcing the gate on) is refused by
    this image, which scores raw on every leg, and vice versa."""
    init = train(tmp_path / "init", contract).local_dir

    @contextlib.contextmanager
    def gate_on():
        with monkeypatch.context() as m:
            m.setattr(pyramid_trainer, "power_ema_active", lambda *_: True)
            yield

    stores = {}
    for name, kw, ctx in (("shadow", {"warm_start_dir": init}, contextlib.nullcontext),
                          ("scored", {"warm_start_dir": init}, gate_on),
                          ("bare", {}, plain)):
        stores[name] = SnapshotStore(tmp_path / name, {"contract": "x", "seed": 1})
        crash = CrashAt(stores[name], interval=0)
        crash.at = 1
        with ctx(), pytest.raises(RuntimeError, match="simulated pod loss"):
            train(tmp_path / f"failed-{name}", contract, continuation=crash, **kw)
    with gate_on(), pytest.raises(ValueError, match="other power EMA decision"):
        train(tmp_path / "a", contract, warm_start_dir=init,
              continuation=Continuation(stores["shadow"], interval=0))
    with pytest.raises(ValueError, match="other power EMA decision"):
        train(tmp_path / "b", contract, warm_start_dir=init,
              continuation=Continuation(stores["scored"], interval=0))
    with pytest.raises(ValueError, match="lacks its power EMA state"):
        train(tmp_path / "c", contract, continuation=Continuation(stores["bare"], interval=0))


# ── (e) lane B ────────────────────────────────────────────────────────────────


def _selection(king_dir):
    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_trainer import losses

    model = load_king(king_dir)
    model.train()
    losses(model, torch.as_tensor(data()[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    return ({i: layer.last_load.topk(4).indices.tolist() for i, layer in enumerate(model.moe_layers())},
            moe_param_names(model))


def _assert_expert_rows(king, cand, selection, names):
    """Frozen tensors and unselected rows are the king's bytes."""
    _equal(merge_expert_slices(king, cand, selection, names), cand)
    owned = {n: li for li, pair in enumerate(names) for n in pair}
    for k, v in king.items():
        if k not in owned:
            assert torch.equal(cand[k], v), k                 # whole frozen tensor, byte for byte
            continue
        rows = sorted(selection.get(owned[k], []))
        keep = torch.ones(v.shape[0], dtype=torch.bool)
        keep[rows] = False
        assert torch.equal(cand[k][keep], v[keep]), k        # unselected expert rows


def test_expert_leg_on_a_random_init_epoch_king_scores_raw(tmp_path, plain, contract):
    king = train(tmp_path / "king", contract)                 # the current epoch: raw king
    selection, names = _selection(king.local_dir)
    ref = train_plain(plain, tmp_path / "ref", contract, warm_start_dir=king.local_dir,
                      expert_selection=selection)
    leg = train(tmp_path / "leg", contract, warm_start_dir=king.local_dir, expert_selection=selection)
    got, want = _bytes(leg.local_dir), _bytes(ref.local_dir)
    assert set(got) == set(want) | {EMA005, EMA010}
    assert {k: got[k] for k in want} == want                 # scored file bitwise raw
    assert "power_ema" not in _config(leg.local_dir) and leg.metrics["power_ema_scored"] is False
    king_w = _weights(king.local_dir)
    for name in (EMA005, EMA010):                            # shadows keep frozen bytes too
        _assert_expert_rows(king_w, _weights(leg.local_dir, name), selection, names)


def test_expert_leg_on_an_ema_marked_king_still_scores_raw(tmp_path, plain, contract):
    """§8: "a lane B leg from an EMA-marked king still scores raw". The king
    carries the marker an EMA-scored king of an earlier image would; the leg
    is byte-identical to the EMA-off leg and its averages average only the
    selected rows."""
    init = train(tmp_path / "init", contract)
    base = train(tmp_path / "base", contract, warm_start_dir=init.local_dir)
    assert "power_ema" not in _config(base.local_dir)          # this image never marks a king
    marked = _config(base.local_dir)
    marked["power_ema"] = MARKER
    (base.local_dir / "config.json").write_text(json.dumps(marked, sort_keys=True))
    selection, names = _selection(base.local_dir)
    ref = train_plain(plain, tmp_path / "ref", contract, warm_start_dir=base.local_dir,
                      expert_selection=selection)
    leg = train(tmp_path / "leg", contract, warm_start_dir=base.local_dir, expert_selection=selection)
    assert leg.metrics["power_ema_scored"] is False and "power_ema" not in _config(leg.local_dir)
    got, want = _bytes(leg.local_dir), _bytes(ref.local_dir)
    assert set(got) == set(want) | {EMA005, EMA010}
    assert {k: got[k] for k in want} == want                 # scored file bitwise raw
    assert checkpoint_hashes(leg.local_dir) == checkpoint_hashes(ref.local_dir)
    king = _weights(base.local_dir)
    cand = _weights(leg.local_dir)
    _assert_expert_rows(king, cand, selection, names)
    for name in (EMA005, EMA010):
        avg = _weights(leg.local_dir, name)
        _assert_expert_rows(king, avg, selection, names)
        assert any(not torch.equal(avg[k], cand[k]) for k in cand), name   # selected rows averaged
    for f in (STABLE_WEIGHTS_FILE, OPTIM_STATE_FILE, POWER_EMA_FILE, RAW):
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
    assert power_ema_settings() == ((0.05, 0.10), 1)        # computed on every leg, never scored


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
