"""WB16 on real tiny pyramid expert legs (marked ``torch``: CI only, never
the 7 GB box). D40 (a): a lane B expert leg leaves every router weight and
routing bias byte-identical to the king, and the merge byte-check refuses a
candidate whose router moved. README.md has the contract.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from . import helpers as h

pytestmark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def contract():
    from cascade.shared.config import parse_chain_config
    from scripts.pyramid_config import local_config

    c = parse_chain_config(local_config("2m")).training
    return replace(c, training_steps=3, batch_size=8, max_train_seconds=90)


def _data(rng_seed):
    rng = np.random.default_rng(rng_seed)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def train(path, contract, *, seed=1, rng_seed=84, **kwargs):
    from cascade.trainer.toto2_trainer import Toto2Trainer

    return Toto2Trainer(device="cpu").train(iter(_data(rng_seed)), contract,
        training_seed=seed, token_budget=contract.train_tokens, out_dir=path, **kwargs)


def _selection(king_dir):
    """The four most-loaded experts per MoE layer (they receive tokens, so
    their rows really train)."""
    import torch

    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_trainer import losses

    model = load_king(king_dir)
    model.train()
    losses(model, torch.as_tensor(_data(84)[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    return {i: layer.last_load.topk(4).indices.tolist() for i, layer in enumerate(model.moe_layers())}


def _weights(d):
    from safetensors.torch import load_file

    return load_file(str(d / "weights.safetensors"))


def _names(king_dir):
    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_model import moe_param_names

    return moe_param_names(load_king(king_dir))


@pytest.fixture
def king(tmp_path, contract):
    return train(tmp_path / "king", contract).local_dir


def _leg(tmp_path, contract, king, name="leg", **kw):
    selection = _selection(king)
    leg = train(tmp_path / name, contract, warm_start_dir=king, expert_selection=selection, **kw)
    return leg, selection


def _assert_router_frozen(king_state, leg_state):
    import torch

    keys = h.router_keys(king_state)
    assert keys, "no router / route_bias keys found in the king"
    moved = [k for k in keys if not torch.equal(king_state[k], leg_state[k])]
    assert not moved, moved


def test_expert_slices_trains_no_router_parameter(king):
    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.expert_training import ExpertSlices

    model = load_king(king)
    slices = ExpertSlices(model, _selection(king))
    assert all(not p.requires_grad for p in model.parameters())
    names = [n for n, _ in slices.named_parameters()]
    assert names and all(n.endswith((".w_in", ".w_out")) for n in names), names
    routers = {id(layer.router.weight) for layer in model.moe_layers()}
    assert not routers & {id(p) for p in slices.parameters()}


def test_expert_leg_leaves_router_and_bias_byte_identical(tmp_path, contract, king):
    leg, _ = _leg(tmp_path, contract, king)
    _assert_router_frozen(_weights(king), _weights(leg.local_dir))


def test_router_slow_cannot_unfreeze_an_expert_leg(tmp_path, contract, king):
    """The DEC-CA-0050 slow-router flag scales lane A's router LR; on an
    expert leg the router stays frozen regardless."""
    slow = replace(contract, router_slow=True)
    leg = train(tmp_path / "slow", slow, warm_start_dir=king, expert_selection=_selection(king))
    _assert_router_frozen(_weights(king), _weights(leg.local_dir))


def test_only_the_selected_rows_move(tmp_path, contract, king):
    import torch

    leg, selection = _leg(tmp_path, contract, king)
    ks, ls = _weights(king), _weights(leg.local_dir)
    names = _names(king)
    owned = {n for pair in names for n in pair}
    for k in ks:
        if k not in owned:
            assert torch.equal(ks[k], ls[k]), k
    moved = False
    for li, pair in enumerate(names):
        rows = sorted(selection.get(li, []))
        for n in pair:
            outside = torch.ones(ks[n].shape[0], dtype=torch.bool)
            outside[rows] = False
            assert torch.equal(ks[n][outside], ls[n][outside]), n
            moved |= bool(rows) and not torch.equal(ks[n][rows], ls[n][rows])
    assert moved, "no selected expert row trained"


@pytest.mark.parametrize("module", ["cascade.trainer.moe_model", "cascade.trainer.pyramid_model"])
def test_merge_accepts_the_frozen_leg(tmp_path, contract, king, module):
    import importlib

    merge = importlib.import_module(module).merge_expert_slices
    leg, selection = _leg(tmp_path, contract, king)
    merge(_weights(king), _weights(leg.local_dir), selection, _names(king))


@pytest.mark.parametrize("module", ["cascade.trainer.moe_model", "cascade.trainer.pyramid_model"])
@pytest.mark.parametrize("which", ["router", "route_bias"])
def test_merge_byte_check_refuses_a_moved_router(tmp_path, contract, king, module, which):
    """Router byte-check: a candidate whose router weight or routing bias
    differs from the king's anywhere is refused (option (b) is not built)."""
    import importlib

    merge = importlib.import_module(module).merge_expert_slices
    leg, selection = _leg(tmp_path, contract, king)
    cand = _weights(leg.local_dir)
    key = next(k for k in h.router_keys(cand) if (".router." in k) == (which == "router"))
    t = cand[key].clone()
    t.view(-1)[0] += 1.0
    cand[key] = t
    with pytest.raises(ValueError, match="changed outside"):
        merge(_weights(king), cand, selection, _names(king))


def test_frozen_leg_is_deterministic(tmp_path, contract, king):
    one, _ = _leg(tmp_path, contract, king, name="one")
    two, _ = _leg(tmp_path, contract, king, name="two")
    a = (one.local_dir / "weights.safetensors").read_bytes()
    b = (two.local_dir / "weights.safetensors").read_bytes()
    assert a == b
