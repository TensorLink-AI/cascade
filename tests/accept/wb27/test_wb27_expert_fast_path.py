"""2026-11-27 owner ("stop now, fix first"): lane B expert legs ran ~5x slower
per step than lane A on 8xH200 (0.34 vs ~1.9 steps/s). ``ExpertSlices`` swaps
each selected MoE layer for ``_SelectedPyramidFFN`` and the leg never bound
lane A's training kernels (``pyramid_fast.accelerate`` is lane A only), so
every MoE layer ran the reference ``MoEFFN.forward`` (a per-expert Python
loop with host syncs under skewed routing), in the no-grad pre-pass and the
real forward, and the conv embed stayed on the slow cuDNN kernel.

Here: ``pyramid_fast.accelerate_experts`` (bound by ``ExpertSlices``, so
every gang rank runs it) and the owner's lane B requirements, each proved
against the reference lane B path (``ExpertSlices(..., fast_kernels=False)``
or ``accelerate_experts`` patched out):

(a) only the selected leaves get gradients / updates; every other tensor is
    bit-unchanged; (b) routing (top-k, gates, loads, aux) is the reference's;
(c) forward / loss / gradients match within lane A's parity tolerances over
    uniform, skewed, collapsed routing, empty experts and a selected expert
    with no tokens; (d) a 2-rank gloo gang reduces only the leaves and equals
    the 1-rank and reference legs; ``sync_leaves`` runs under ``no_grad``;
(e) the leg's files and the validator's merge check / merged rebuild are
    unchanged in kind and rebuild identically; (f) a continuation resumes
    bit-identically; (g) lane A's kernels and step are untouched.

Marked ``torch``: CI only, never the 7 GB box.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)

REPO = Path(__file__).resolve().parents[3]
FAST = REPO / "cascade" / "trainer" / "pyramid_fast.py"
DP = REPO / "cascade" / "trainer" / "pyramid_dp.py"
# lane A's tolerances (tests/unit/test_pyramid_fast.py): fp64 kernel parity,
# fp32 step parity; and WB25's leg tolerance on the selected rows
KERNEL_ATOL, GRAD_ATOL = 1e-12, 1e-11
STEP_RTOL, STEP_ATOL = 1e-4, 1e-6
LEG_RTOL, LEG_ATOL = 1e-4, 1e-6
# fast vs reference LEG (3 fp32 steps, warmup lr): a sanity bound only. The
# exact parity proofs are the kernel (fp64, 1e-12) and one-step (fp32, lane A's
# 1e-4 / 1e-6) tests above. Over a leg, fp32 rounding of the im2col conv and
# the binned bmm (~1e-7 relative) passes through NorMuon, whose
# orthogonalisation and row normalisation are scale-invariant, and can flip a
# near-tied top-k pick in a later layer; with the 3-step warmup delta at
# ~1.1e-4, CI measured differences up to 5.7e-5 (2026-11-27 run 2). So no
# fraction-of-delta bound here.
REF_RTOL, REF_ATOL = 1e-3, 1e-4

# The lane A functions, as main @ 00f3139c (ast source segments, sha256).
LANE_A_FAST = {
    "accelerate": "296570e9c8519ca9f7074190371566f0f2cfbd018583d80bf2f114944f27b628",
    "causal_conv": "c4fa6481796fdae1af9f4b4eb3281a65326e2ad4c05643751b6b8a788cf5fe5f",
    "conv_embed_forward": "441ec84728a2433db666862497210bf2127f61eb4ef5f54fd3da3499eb7ba513",
    "binned_experts": "954d73e6eb2cbbfd8736b22f6b675eadfc96ed4e6a3a79d421e83d0f4cf6b572",
    "moe_forward": "4bf035cfbbd25e6681983e6613917347e3a8f83ef1e149a0afdb99fd20db764b",
    "finish_balance": "eff4ecc13d86bae8d68c1115e20a14f6039d9dfd030265b3accb046efd02cf3c",
    "finish_aux": "c62a4530b25048d52feb56ef761786d394008d42423f31d3611725e3fb451809",
}
LANE_A_DP = {
    "micro_batch_pass": "630b04017d15ea6e15e11cfcce17c469392e5f9eefadfe7073393f6013c80105",
    "broadcast_from_sources": "aced42debe7dbca85619dc526a02536809b606ed732877ad1dfede4ed229a2f4",
}


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _fast():
    """The lane B fast path (fails by assertion on a base without it)."""
    from cascade.trainer import pyramid_fast as pf

    for name in ("accelerate_experts", "selected_moe_forward", "selected_binned_experts"):
        assert hasattr(pf, name), f"pyramid_fast lacks the lane B fast path ({name})"
    return pf


# ── layer / model level ──────────────────────────────────────────────────────


def _cfg(**kw):
    from cascade.trainer.pyramid_model import CascadeMoEConfig

    base = dict(d_model=32, num_heads=4, head_dim=8, num_layers=3, dense_ffn=48, expert_hidden=4,
                shared_hidden=16, num_experts=32, experts_per_layer=(32,), moe_blocks=(1,),
                variate_blocks=(2,), context_length=128, top_k=4)
    return CascadeMoEConfig(**{**base, **kw})


SELECTION = {0: [0, 5, 17, 31]}
SCENARIOS = {
    "uniform": {},
    "skewed": {0: 3.0, 1: 1.5},
    "collapsed": {0: 50.0},
    "empty_experts": {e: -1e3 for e in range(16, 31)},     # half the experts idle
    "selected_gets_no_tokens": {31: -1e3, 0: 3.0},         # selected row 31 starves
}


def _pair(selection=SELECTION, bias=None, dtype=None):
    """(fast model, reference model, fast slices, reference slices) from one
    init: the same king, the selection's slices, the fast vs reference kernels."""
    import torch

    from cascade.trainer.expert_training import ExpertSlices
    from cascade.trainer.pyramid_model import CascadeMoEModel

    torch.manual_seed(0)
    king = CascadeMoEModel(_cfg())
    if dtype is not None:
        king = king.to(dtype)
    with torch.no_grad():
        for e, v in (bias or {}).items():
            king.moe_layers()[0].route_bias[e] = v
    fast, ref = copy.deepcopy(king).train(), copy.deepcopy(king).train()
    fs = ExpertSlices(fast, selection)
    rs = ExpertSlices(ref, selection, fast_kernels=False)
    return fast, ref, fs, rs


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_selected_layer_matches_the_reference_path(scenario):
    """(b) + (c) on one selected layer, fp64: the same top-k picks, gates,
    loads and aux loss (bit-equal), the output within lane A's kernel
    tolerance, and gradients only on the selected leaves (zero, not None,
    for a selected expert that got no tokens), within lane A's tolerance."""
    import torch

    pf = _fast()
    fast, ref, fs, rs = _pair(bias=SCENARIOS[scenario], dtype=torch.float64)
    lf, lr = fast.moe_layers()[0], ref.moe_layers()[0]
    assert lf.forward.__func__ is pf.selected_moe_forward
    assert "forward" not in lr.__dict__                   # reference kept its class forward
    torch.manual_seed(1)
    x0 = torch.randn(64, 32, dtype=torch.float64)
    g = torch.randn(64, 32, dtype=torch.float64)
    load = torch.randint(0, 50, (32,))
    outs = []
    for layer in (lf, lr):
        layer.balance_load_override = load.clone()        # the pre-pass's global count
        x = x0.clone().requires_grad_(True)
        y = layer(x, 8)
        ((y * g).sum() + layer.last_aux).backward()
        outs.append((x, y, layer))
    (xf, yf, lf), (xr, yr, lr) = outs
    assert torch.equal(lf.last_topk, lr.last_topk)
    assert torch.equal(lf.last_load, lr.last_load)
    assert torch.equal(lf.last_aux, lr.last_aux)
    torch.testing.assert_close(yf, yr, rtol=0, atol=KERNEL_ATOL)
    torch.testing.assert_close(xf.grad, xr.grad, rtol=0, atol=GRAD_ATOL)
    for (nf, a), (nr, b) in zip(fs.named_parameters(), rs.named_parameters(), strict=True):
        assert nf == nr
        assert a.grad is not None and b.grad is not None, nf
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=GRAD_ATOL)
    for model in (fast, ref):
        assert all(p.grad is None for p in model.parameters()), "a frozen tensor got a gradient"
    if scenario == "selected_gets_no_tokens":
        row = SELECTION[0].index(31)
        assert int(lf.last_load[31]) == 0
        for leaf in fs.parameters():
            assert torch.count_nonzero(leaf.grad[row]) == 0
    if scenario == "empty_experts":
        assert int((lf.last_load == 0).sum()) >= 15


def test_expert_model_forward_and_routing_match_the_reference():
    """(b) + (c) through the whole model, fp32 (conv embed, frozen and
    selected MoE layers): every layer's routing bit-equal, outputs within
    lane A's model-level tolerance; the
    conv embed and the frozen MoE layers run lane A's kernels; the leg keeps
    the counting pre-pass (``fast_training`` unset)."""
    import torch

    pf = _fast()
    from cascade.trainer.pyramid_model import DilatedConvEmbed

    cfg = dict(num_layers=4, moe_blocks=(1, 3), experts_per_layer=(32, 32))
    from cascade.trainer.expert_training import ExpertSlices
    from cascade.trainer.pyramid_model import CascadeMoEModel

    torch.manual_seed(0)
    king = CascadeMoEModel(_cfg(**cfg))
    fast, ref = copy.deepcopy(king).train(), copy.deepcopy(king).train()
    ExpertSlices(fast, {1: [2, 9]})                        # layer 0 frozen, layer 1 selected
    ExpertSlices(ref, {1: [2, 9]}, fast_kernels=False)
    assert getattr(fast, "fast_experts", False) is True
    assert not getattr(fast, "fast_training", False)
    convs = [m for m in fast.modules() if type(m) is DilatedConvEmbed]
    assert convs and all(m.forward.__func__ is pf.conv_embed_forward for m in convs)
    frozen, selected = fast.moe_layers()
    assert frozen.forward.__func__ is pf.moe_forward
    assert selected.forward.__func__ is pf.selected_moe_forward
    z = torch.randn(2, 3, 4, 32)
    yf, yr = fast(z, with_parameters=True), ref(z, with_parameters=True)
    for a, b in zip(fast.moe_layers(), ref.moe_layers(), strict=True):
        assert torch.equal(a.last_topk, b.last_topk) and torch.equal(a.last_load, b.last_load)
    torch.testing.assert_close(yf, yr, rtol=1e-5, atol=1e-5)    # lane A's model-level tolerance
    torch.testing.assert_close(fast.aux_loss, ref.aux_loss, rtol=1e-5, atol=1e-6)


def test_one_step_updates_only_the_selected_leaves_like_the_reference():
    """(a) one real step (micro_batch_pass with its pre-pass, combine, clip,
    optimizer, sync): losses, loads and gradients equal the reference within
    lane A's step tolerance; afterwards every tensor outside the selected
    rows is bit-identical to the king and the selected rows moved, and equal
    the reference step's within tolerance."""
    import torch

    _fast()
    from cascade.trainer.pyramid_dp import micro_batch_pass
    from cascade.trainer.pyramid_optimizer import build_optimizer
    from cascade.trainer.pyramid_trainer import combine_gradients
    from cascade.trainer.toto2_trainer import sample_cpm_masks

    fast, ref, fs, rs = _pair(bias=SCENARIOS["skewed"])
    king = {k: v.clone() for k, v in ref.state_dict().items()}
    rng = np.random.default_rng(0)
    values = rng.standard_normal((4, 3, 128)).cumsum(-1)
    masks = sample_cpm_masks(12, 4, c_max=16, p_max=0.4, rng=rng).reshape(4, 3, -1)
    splits = np.array_split(np.arange(4), 4)
    out = []
    for model, slices in ((fast, fs), (ref, rs)):
        params = list(slices.parameters())
        gen = torch.Generator().manual_seed(3)
        reference, lm, lj, load = micro_batch_pass(model, params, values, masks, None, splits,
                                                   set(range(4)), gen, "cpu", step=0, tokens=0,
                                                   gang=None)
        grads = [p.grad.clone() for p in params]
        combine_gradients(params, reference)
        torch.nn.utils.clip_grad_norm_(params, 7.0)
        opt = build_optimizer(slices, model.cfg.d_model)
        opt.step()
        slices.sync()
        out.append((reference, grads, lm, lj, [x.clone() for x in load], model.state_dict()))
    (r1, g1, m1, j1, l1, s1), (r2, g2, m2, j2, l2, s2) = out
    assert all(torch.equal(a, b) for a, b in zip(l1, l2, strict=True))
    assert m1 == pytest.approx(m2, rel=STEP_RTOL) and j1 == pytest.approx(j2, rel=STEP_RTOL)
    for a, b in zip(r1 + g1, r2 + g2, strict=True):
        torch.testing.assert_close(a, b, rtol=STEP_RTOL, atol=STEP_ATOL)
    rows = torch.tensor(SELECTION[0])
    owned = {n for n in fs.weights}
    moved = False
    for k, t in king.items():
        if k not in owned:
            assert torch.equal(s1[k], t), f"{k} changed outside the selection"
            continue
        outside = torch.ones(t.shape[0], dtype=torch.bool)
        outside[rows] = False
        assert torch.equal(s1[k][outside], t[outside]), f"{k}: an unselected expert row moved"
        torch.testing.assert_close(s1[k][rows], s2[k][rows], rtol=STEP_RTOL, atol=STEP_ATOL)
        moved |= not torch.equal(s1[k][rows], t[rows])
    assert moved, "no selected row trained"


def test_sync_leaves_broadcasts_under_no_grad():
    """``sync_leaves`` writes into leaves that require grad: grad mode must be
    off (``c10d::broadcast_`` has no autograd kernel: a warning and a hook
    per leaf per step otherwise), and the leaves still land in the rows."""
    import torch

    _fast()
    from cascade.trainer.pyramid_dp import sync_leaves

    _, _, fs, _ = _pair()
    seen = []

    def broadcast(t, src=0, async_op=False):
        seen.append(torch.is_grad_enabled())
        t.add_(1.0)                                         # stands in for the owner's bytes

    dist = types.SimpleNamespace(broadcast=broadcast, get_backend=lambda: "gloo")
    leaves = list(fs.parameters())
    want = [p.detach().clone() + 1.0 for p in leaves]
    with torch.enable_grad():
        sync_leaves(dist, fs, [0] * len(leaves), "cpu")
    assert seen and not any(seen), "sync_leaves broadcast with grad mode on"
    for leaf, w in zip(leaves, want, strict=True):
        assert torch.equal(leaf.detach(), w) and leaf.grad_fn is None
    for name, leaf in fs.named_parameters():
        assert torch.equal(fs.weights[name][fs.rows[name]], leaf.detach())


# ── legs through the production entry point ──────────────────────────────────


def _contract():
    from cascade.shared.config import parse_chain_config
    from scripts.pyramid_config import local_config

    c = parse_chain_config(local_config("2m")).training
    # micro_batches=4: the counting pre-pass runs and both ranks of a 2-rank gang own splits
    return replace(c, training_steps=3, batch_size=8, micro_batches=4, max_train_seconds=300)


def _data():
    rng = np.random.default_rng(84)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def _train(path, contract, **kw):
    import torch

    from cascade.trainer.toto2_trainer import Toto2Trainer

    torch.set_num_threads(1)
    return Toto2Trainer(device="cpu").train(iter(_data()), contract, training_seed=1,
                                            token_budget=contract.train_tokens, out_dir=path, **kw)


def _weights(d, name="weights.safetensors"):
    from safetensors.torch import load_file

    return load_file(str(Path(d) / name))


def _names(king_dir):
    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_model import moe_param_names

    return moe_param_names(load_king(king_dir))


def _selection(king_dir):
    """The three most-loaded experts per MoE layer plus its least-loaded one
    (a selected row that may get no tokens)."""
    import torch

    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_trainer import losses

    model = load_king(king_dir)
    model.train()
    losses(model, torch.as_tensor(_data()[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    out = {}
    for i, layer in enumerate(model.moe_layers()):
        load = layer.last_load
        out[i] = sorted({*load.topk(3).indices.tolist(), int(load.argmin())})
    return out


_LEGS: dict = {}


@pytest.fixture(scope="module")
def legdir(tmp_path_factory):
    return tmp_path_factory.mktemp("legs")


@pytest.fixture
def legs(legdir):
    """Shared legs, built on first use inside the test body (so on a base
    without the fast path the failure is the test's assertion, not a fixture
    error): ``legs(name)`` for king / fast / reference."""
    def get(name):
        if name in _LEGS:
            return _LEGS[name]
        contract = _contract()
        if name == "king":
            _LEGS[name] = _train(legdir / "king", contract).local_dir
        elif name == "selection":
            _LEGS[name] = _selection(get("king"))
        elif name == "fast":
            _LEGS[name] = _spied_leg(legdir / "fast", contract, warm_start_dir=get("king"),
                                     expert_selection=get("selection"))
        elif name == "reference":
            pf = _fast()
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(pf, "accelerate_experts", lambda model: model)
                _LEGS[name] = (_train(legdir / "reference", contract, warm_start_dir=get("king"),
                                      expert_selection=get("selection")), None)
        return _LEGS[name]
    return get


def _spied_leg(path, contract, **kw):
    """A leg with call counts on the lane B kernels (rank 0)."""
    pf = _fast()
    calls = {"selected": 0, "frozen": 0, "conv": 0}
    real = {"selected": pf.selected_moe_forward, "frozen": pf.moe_forward,
            "conv": pf.conv_embed_forward}

    def counting(key):
        def run(self, *a, **k):
            calls[key] += 1
            return real[key](self, *a, **k)
        return run

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pf, "selected_moe_forward", counting("selected"))
        mp.setattr(pf, "moe_forward", counting("frozen"))
        mp.setattr(pf, "conv_embed_forward", counting("conv"))
        leg = _train(path, contract, **kw)
    return leg, calls


def _assert_near_reference(fast, ref):
    """A fast leg's selected rows vs the reference leg's, within REF_* (see
    there: a sanity bound; the parity proofs are the kernel / step tests)."""
    import torch

    torch.testing.assert_close(fast, ref, rtol=REF_RTOL, atol=REF_ATOL)


def test_fast_expert_leg_trains_only_the_selected_rows(legs):
    """(a) end to end: the leg ran the fast kernels; every tensor outside the
    selected rows (router, routing bias, attention, embeddings, frozen
    experts) is bit-identical to the king; the selected rows moved and equal
    the reference leg's within REF_* (a sanity bound)."""
    import torch

    _fast()
    (leg, calls), (ref, _) = legs("fast"), legs("reference")
    king, selection = legs("king"), legs("selection")
    assert calls["selected"] > 0 and calls["conv"] > 0, calls
    ks, fs, rs = _weights(king), _weights(leg.local_dir), _weights(ref.local_dir)
    assert ks.keys() == fs.keys() == rs.keys()
    owned = {n: li for li, pair in enumerate(_names(king)) for n in pair}
    moved = False
    for k, t in ks.items():
        if k not in owned:
            assert torch.equal(fs[k], t), f"{k} changed outside the selection"
            continue
        rows = sorted(selection.get(owned[k], []))
        outside = torch.ones(t.shape[0], dtype=torch.bool)
        outside[rows] = False
        assert torch.equal(fs[k][outside], t[outside]), f"{k}: an unselected expert row moved"
        if rows:
            _assert_near_reference(fs[k][rows], rs[k][rows])
            moved |= not torch.equal(fs[k][rows], t[rows])
    assert moved, "no selected row trained"
    assert leg.metrics["steps"] == ref.metrics["steps"] == 3
    assert leg.metrics["final_loss"] == pytest.approx(ref.metrics["final_loss"], rel=1e-4)


def test_fast_leg_checkpoint_is_what_the_validator_merges(legs, tmp_path):
    """(e) the leg's files are the reference leg's (same names; config.json,
    model.py, forecast_wrapper.py byte-equal; every safetensors file with the
    same keys / shapes / dtypes, power EMAs included); the validator's merge
    check accepts it (strict), and its merged-init rebuild equals the
    trainer-side merge tensor for tensor; the leg is deterministic."""
    from safetensors.torch import save_file

    _fast()
    from cascade.trainer.expert_lane import apply_merges
    from cascade.validator.expert_lane import merge_check_dirs, merged_rebuild_check

    (leg, _), (ref, _) = legs("fast"), legs("reference")
    king, selection = legs("king"), legs("selection")
    fd, rd = Path(leg.local_dir), Path(ref.local_dir)
    files = sorted(p.name for p in fd.iterdir() if p.is_file())
    assert files == sorted(p.name for p in rd.iterdir() if p.is_file())
    for name in ("config.json", "model.py", "forecast_wrapper.py"):
        if (rd / name).is_file():
            assert (fd / name).read_bytes() == (rd / name).read_bytes(), name
    for name in files:
        if name.endswith(".safetensors"):
            a, b = _weights(fd, name), _weights(rd, name)
            assert {k: (tuple(v.shape), v.dtype) for k, v in a.items()} == \
                {k: (tuple(v.shape), v.dtype) for k, v in b.items()}, name
    assert merge_check_dirs(Path(king), fd, selection, strict=True) is None
    dirs = {"king": Path(king), "cand": fd}
    merges = [{"selection": {str(k): v for k, v in selection.items()},
               "rebased_pointer": "cand", "king_pointer": "king"}]
    want = apply_merges(_weights(king), merges, lambda p: _weights(dirs[p]), _names(king),
                        strict=True)
    target = tmp_path / "merged"
    target.mkdir()
    save_file({k: v.contiguous() for k, v in want.items()}, str(target / "weights.safetensors"))
    dirs["target"] = target
    assert merged_rebuild_check(lambda p: dirs[p], "king", merges, "target", strict=True) == \
        {"equal": True}
    again, _ = _spied_leg(tmp_path / "again", _contract(), warm_start_dir=king,
                          expert_selection=selection)
    for name in files:
        if name.endswith(".safetensors"):
            assert (Path(again.local_dir) / name).read_bytes() == (fd / name).read_bytes(), name


def test_fast_leg_resumes_from_a_continuation_bit_identically(legs, tmp_path):
    """(f) a fast leg killed after step 2 and resumed from its continuation
    writes the uninterrupted leg's bytes; a completed recovery adds no step."""
    _fast()
    from cascade.shared.manifest import contract_digest
    from cascade.trainer.continuation import Continuation, SnapshotStore

    (leg, _) = legs("fast")
    king, selection = legs("king"), legs("selection")
    contract = _contract()
    store = SnapshotStore(tmp_path / "snapshots", {"contract": contract_digest(contract), "seed": 1})

    class CrashAfterSecond(Continuation):
        def save(self, state):
            self.store.save(state["step"], state)
            if state["step"] == 2:
                raise RuntimeError("simulated pod loss")

    kw = {"warm_start_dir": king, "expert_selection": selection}
    with pytest.raises(RuntimeError, match="simulated pod loss"):
        _train(tmp_path / "failed", contract, continuation=CrashAfterSecond(store, interval=0), **kw)
    resumed, calls = _spied_leg(tmp_path / "resumed", contract,
                                continuation=Continuation(store, interval=0), **kw)
    assert calls["selected"] > 0
    assert resumed.metrics["steps"] == 3
    for p in sorted(Path(leg.local_dir).glob("*.safetensors")):
        assert (Path(resumed.local_dir) / p.name).read_bytes() == p.read_bytes(), p.name


def test_two_rank_fast_leg_reduces_only_leaves_and_matches(legs, tmp_path):
    """(d) a 2-rank gloo gang (the #960 CPU seam; every helper rank builds its
    ExpertSlices, so it runs the same kernels): every float all_reduce is a
    selected leaf's gradient or a loss pair, the frozen tensors stay
    bit-identical to the king, and the selected rows equal the 1-rank fast
    leg within WB25's leg tolerance and the reference leg within REF_*."""
    import torch
    import torch.distributed as dist

    _fast()
    from cascade.trainer import pyramid_dp

    (one, _), (ref, _) = legs("fast"), legs("reference")
    king, selection = legs("king"), legs("selection")
    names = _names(king)
    ks = _weights(king)
    leaf_shapes = {(len(selection[li]), *ks[n].shape[1:]) for li, pair in enumerate(names)
                   if selection.get(li) for n in pair}
    reduces, worlds = [], []
    real_reduce, real_gang = dist.all_reduce, pyramid_dp.Gang

    class SpyGang(real_gang):
        def __init__(self, world, *a, **k):
            worlds.append(int(world))
            super().__init__(world, *a, **k)

    def all_reduce(t, *a, **k):
        reduces.append((tuple(t.shape), t.dtype))
        return real_reduce(t, *a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pyramid_dp, "Gang", SpyGang)
        mp.setattr(pyramid_dp, "gang_size", lambda *a, **k: 2)
        mp.setattr(dist, "all_reduce", all_reduce)
        two, calls = _spied_leg(tmp_path / "two", _contract(), warm_start_dir=king,
                                expert_selection=selection)
    assert worlds == [2], worlds
    assert calls["selected"] > 0
    floats = {shape for shape, dtype in reduces if dtype.is_floating_point}
    assert floats - leaf_shapes <= {(2,), (1,)}, sorted(floats - leaf_shapes)
    a, o, r = _weights(two.local_dir), _weights(one.local_dir), _weights(ref.local_dir)
    owned = {n: li for li, pair in enumerate(names) for n in pair}
    for k, t in ks.items():
        if k not in owned:
            assert torch.equal(a[k], t), k
            continue
        rows = sorted(selection.get(owned[k], []))
        outside = torch.ones(t.shape[0], dtype=torch.bool)
        outside[rows] = False
        assert torch.equal(a[k][outside], t[outside]), k
        if rows:
            torch.testing.assert_close(a[k][rows], o[k][rows], rtol=LEG_RTOL, atol=LEG_ATOL)
            _assert_near_reference(a[k][rows], r[k][rows])


# ── lane A untouched ─────────────────────────────────────────────────────────


def _segments(path: Path) -> dict[str, str]:
    src = path.read_text(encoding="utf-8")
    return {n.name: hashlib.sha256(ast.get_source_segment(src, n).encode()).hexdigest()
            for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}


def test_lane_a_kernels_and_step_are_untouched(tmp_path):
    """(g) the lane A kernels and the step function are main's source; a lane
    A model binds exactly them (``fast_training``, no lane B kernel), and a
    lane A leg never calls a lane B kernel. (WB25's oracle tests prove the
    lane A legs byte for byte against main.)"""
    pf = _fast()
    from cascade.trainer.pyramid_model import CascadeMoEModel, DilatedConvEmbed, MoEFFN

    fast_src, dp_src = _segments(FAST), _segments(DP)
    assert {k: fast_src.get(k) for k in LANE_A_FAST} == LANE_A_FAST
    assert {k: dp_src.get(k) for k in LANE_A_DP} == LANE_A_DP
    model = pf.accelerate(CascadeMoEModel(_cfg()))
    assert model.fast_training is True and not hasattr(model, "fast_experts")
    for m in model.modules():
        if type(m) is MoEFFN:
            assert m.forward.__func__ is pf.moe_forward
        elif type(m) is DilatedConvEmbed:
            assert m.forward.__func__ is pf.conv_embed_forward
    calls = {"n": 0}
    real = pf.selected_binned_experts

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pf, "selected_binned_experts", spy)
        mp.setattr(pf, "accelerate_experts", lambda model: (_ for _ in ()).throw(
            AssertionError("lane A bound the lane B kernels")))
        leg = _train(tmp_path / "lane_a", _contract())
    assert calls["n"] == 0 and leg.metrics["steps"] == 3
