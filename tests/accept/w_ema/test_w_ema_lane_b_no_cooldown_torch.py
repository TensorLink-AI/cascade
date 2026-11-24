"""Plan rev6.20 on real tiny pyramid legs (marked ``torch``: CI only, never
the 7 GB box): a lane B expert leg under the ``warmup_constant_ema`` leg
contract trains with no cooldown, scores (weights.safetensors) its sigma_rel
0.10 power EMA with every frozen row at the king's bytes, and the rebased
checkpoint settlement builds (judged and merged) comes from those same
weights. Lane A is untouched (the W-EMA and WB25 byte tests). Contract:
README.md "rev6.20"."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from . import helpers as h

pytestmark = pytest.mark.torch

SCHEDULE = "warmup_constant_ema"


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


def _data(rng_seed=84):
    rng = np.random.default_rng(rng_seed)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def train(path, contract, **kwargs):
    from cascade.trainer.toto2_trainer import Toto2Trainer

    return Toto2Trainer(device="cpu").train(iter(_data()), contract, training_seed=1,
        token_budget=contract.train_tokens, out_dir=path, **kwargs)


def _selection(king_dir):
    import torch

    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_trainer import losses

    model = load_king(king_dir)
    model.train()
    losses(model, torch.as_tensor(_data()[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    return {i: layer.last_load.topk(4).indices.tolist() for i, layer in enumerate(model.moe_layers())}


@pytest.fixture
def king(tmp_path, contract):
    return train(tmp_path / "king", contract).local_dir


def _ema_leg(tmp_path, contract, king, monkeypatch):
    """The lane B EMA leg, with spies on both LR schedules."""
    from cascade.shared.config import lane_b
    from cascade.trainer import pyramid_trainer

    assert getattr(lane_b, "LANE_B_EMA_SCHEDULE", None) == SCHEDULE
    calls = {"wsd": 0, "lane_b": []}
    wsd, flat = pyramid_trainer.schedule_factor, pyramid_trainer.lane_b_schedule_factor

    def spy_wsd(step, total):
        calls["wsd"] += 1
        return wsd(step, total)

    def spy_flat(step, total):
        calls["lane_b"].append(flat(step, total))
        return calls["lane_b"][-1]

    monkeypatch.setattr(pyramid_trainer, "schedule_factor", spy_wsd)
    monkeypatch.setattr(pyramid_trainer, "lane_b_schedule_factor", spy_flat)
    selection = _selection(king)
    leg = train(tmp_path / "leg", replace(contract, lr_schedule=SCHEDULE),
                warm_start_dir=king, expert_selection=selection)
    return leg, selection, calls


def test_lane_b_ema_leg_has_no_cooldown(tmp_path, contract, king, monkeypatch):
    leg, _, calls = _ema_leg(tmp_path, contract, king, monkeypatch)
    assert calls["wsd"] == 0, "the lane B EMA leg must not use the WSD schedule"
    assert len(calls["lane_b"]) == 3 and calls["lane_b"][-1] == 1.0, calls
    assert leg.metrics["lr_schedule"] == SCHEDULE


def test_lane_b_ema_leg_scores_the_sigma_0_10_average(tmp_path, contract, king, monkeypatch):
    import torch
    from safetensors.torch import load_file

    from cascade.trainer.expert_lane import _moe_names
    from cascade.trainer.moe_model import merge_expert_slices

    leg, selection, _ = _ema_leg(tmp_path, contract, king, monkeypatch)
    files = set(h.file_bytes(leg.local_dir))
    assert {h.RAW, h.EMA005} <= files and h.EMA010 not in files, sorted(files)
    assert leg.metrics["power_ema_scored"] is True
    assert leg.metrics["power_ema_eval_sigma_rel"] == 0.10
    marker = h.config_of(leg.local_dir)[h.MARKER_KEY]
    assert marker == {**h.MARKER, "schedule": SCHEDULE}
    scored = load_file(str(leg.local_dir / "weights.safetensors"))
    raw = load_file(str(leg.local_dir / h.RAW))
    assert any(not torch.equal(scored[k], raw[k]) for k in raw), "scored file is the raw endpoint"
    # frozen rows at the king's bytes: the merge byte-check accepts the EMA
    kw = load_file(str(king / "weights.safetensors"))
    merge_expert_slices(kw, scored, selection, _moe_names(king))


def test_settlement_rebases_the_scored_ema_not_the_raw_endpoint(tmp_path, contract, king,
                                                                monkeypatch):
    """settle_leg (the judged + merged checkpoint) is built from the leg's
    scored weights.safetensors, and the trainer reads the leg as "ema010"."""
    import torch
    from safetensors.torch import load_file

    from cascade.trainer.expert_lane import _moe_names, settle_leg
    from cascade.trainer.moe_model import rebase_expert_delta

    leg, selection, _ = _ema_leg(tmp_path, contract, king, monkeypatch)
    from cascade.trainer.lane_b_driver import leg_scored_weights

    assert leg_scored_weights(leg.local_dir) == "ema010"
    rebased = settle_leg(king, king, leg.local_dir, selection, tmp_path / "rebased")
    got = load_file(str(rebased / "weights.safetensors"))
    kw = load_file(str(king / "weights.safetensors"))
    names = _moe_names(king)
    want = rebase_expert_delta(kw, kw, load_file(str(leg.local_dir / "weights.safetensors")),
                               selection, names)
    raw = rebase_expert_delta(kw, kw, load_file(str(leg.local_dir / h.RAW)), selection, names)
    assert all(torch.equal(got[k], want[k]) for k in want)
    assert any(not torch.equal(got[k], raw[k]) for k in raw)


def test_lane_a_leg_refuses_the_lane_b_schedule(tmp_path, contract):
    from cascade.trainer import pyramid_trainer

    assert hasattr(pyramid_trainer, "lane_b_ema_leg")
    with pytest.raises(ValueError, match="lane B only"):
        train(tmp_path / "a", replace(contract, lr_schedule=SCHEDULE))
