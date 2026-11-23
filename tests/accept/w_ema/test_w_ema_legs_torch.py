"""W-EMA on real tiny pyramid legs (marked ``torch``: CI only, never the
7 GB box). Each warm-start leg is compared, byte for byte, with the same leg
trained with the image's power EMA switched off (the module constants
monkeypatched to ``()`` / ``-1``, the "tests only" switch
``power_ema_settings`` documents): that leg's ``weights.safetensors`` IS the
raw cooled endpoint. Contract: README.md."""

from __future__ import annotations

import contextlib
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


@pytest.fixture
def plain(monkeypatch):
    """A context in which the image's power EMA is off (no average computed,
    the raw endpoint saved): the reference raw leg."""
    from cascade.trainer import pyramid_trainer

    @contextlib.contextmanager
    def ctx():
        with monkeypatch.context() as m:
            m.setattr(pyramid_trainer, "POWER_EMA_SIGMA_RELS", ())
            m.setattr(pyramid_trainer, "POWER_EMA_EVAL", -1)
            yield
    return ctx


def _data(rng_seed):
    rng = np.random.default_rng(rng_seed)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def train(path, contract, *, seed=1, rng_seed=84, **kwargs):
    from cascade.trainer.toto2_trainer import Toto2Trainer

    return Toto2Trainer(device="cpu").train(iter(_data(rng_seed)), contract,
        training_seed=seed, token_budget=contract.train_tokens, out_dir=path, **kwargs)


def _assert_raw_leg(leg, ref):
    """``leg`` (image EMA on) scores exactly what ``ref`` (EMA off) saved:
    every file ``ref`` has is byte-identical in ``leg`` (weights.safetensors,
    config.json without a marker, the raw lineage files); ``leg`` adds the
    two unscored averages and nothing else (no weights_raw: the raw endpoint
    is weights.safetensors)."""
    got, want = h.file_bytes(leg.local_dir), h.file_bytes(ref.local_dir)
    assert set(got) == set(want) | {h.EMA005, h.EMA010}, sorted(set(got) ^ set(want))
    assert got["weights.safetensors"] == want["weights.safetensors"]
    assert {k: got[k] for k in want} == want
    assert h.RAW not in got
    assert h.MARKER_KEY not in h.config_of(leg.local_dir)
    assert leg.metrics["power_ema_scored"] is False


def _assert_scored_hashes_equal(leg, ref):
    from cascade.validator.private_worker import checkpoint_hashes

    assert checkpoint_hashes(leg.local_dir) == checkpoint_hashes(ref.local_dir)


# ── lane A ──────────────────────────────────────────────────────────────────


def test_warm_start_king_leg_scores_the_raw_endpoint(tmp_path, plain, contract):
    init = train(tmp_path / "init", contract)
    with plain():
        ref = train(tmp_path / "ref", contract, warm_start_dir=init.local_dir)
    leg = train(tmp_path / "king", contract, warm_start_dir=init.local_dir)
    _assert_raw_leg(leg, ref)
    _assert_scored_hashes_equal(leg, ref)


def test_warm_start_challenger_leg_scores_the_raw_endpoint(tmp_path, plain, contract):
    init = train(tmp_path / "init", contract)
    with plain():
        ref = train(tmp_path / "ref", contract, warm_start_dir=init.local_dir,
                    seed=7, rng_seed=91)
    leg = train(tmp_path / "challenger", contract, warm_start_dir=init.local_dir,
                seed=7, rng_seed=91)
    _assert_raw_leg(leg, ref)


def test_warm_start_from_an_ema_marked_init_scores_the_raw_endpoint(tmp_path, plain, contract):
    """Even an init that itself carries today's marker (an EMA-scored king of
    an earlier image) yields a raw-scored leg."""
    init = train(tmp_path / "init", contract)
    h.mark_config(init.local_dir)
    with plain():
        ref = train(tmp_path / "ref", contract, warm_start_dir=init.local_dir)
    leg = train(tmp_path / "leg", contract, warm_start_dir=init.local_dir)
    _assert_raw_leg(leg, ref)


def test_warm_start_leg_writes_both_unscored_averages(tmp_path, contract):
    """Both EMA files are present on a warm-start leg, in the checkpoint's own
    format and keys, and they are real averages (not copies of the raw
    endpoint)."""
    import torch
    from safetensors.torch import load_file

    init = train(tmp_path / "init", contract)
    leg = train(tmp_path / "leg", contract, warm_start_dir=init.local_dir)
    present = set(h.file_bytes(leg.local_dir))
    assert {h.EMA005, h.EMA010} <= present, sorted(present)
    raw = load_file(str(leg.local_dir / "weights.safetensors"))
    for name in (h.EMA005, h.EMA010):
        avg = load_file(str(leg.local_dir / name))
        assert avg.keys() == raw.keys(), name
        assert all(avg[k].dtype == raw[k].dtype and avg[k].shape == raw[k].shape for k in raw), name
        assert any(not torch.equal(avg[k], raw[k]) for k in raw), f"{name} equals the raw endpoint"
    assert leg.metrics["power_ema_steps"] == 3


def test_random_init_leg_is_byte_identical_to_the_raw_trainer(tmp_path, plain, contract):
    """Today's random-init behaviour, which warm-start legs now copy."""
    with plain():
        ref = train(tmp_path / "ref", contract)
    leg = train(tmp_path / "leg", contract)
    _assert_raw_leg(leg, ref)
    _assert_scored_hashes_equal(leg, ref)


def test_raw_scored_leg_passes_the_checkpoint_guard(tmp_path, contract):
    from cascade.eval.checkpoint_guard import verify_checkpoint

    init = train(tmp_path / "init", contract)
    leg = train(tmp_path / "leg", contract, warm_start_dir=init.local_dir)
    verify_checkpoint(leg.local_dir, contract)


# ── lane B ──────────────────────────────────────────────────────────────────


def _selection(king_dir):
    import torch

    from cascade.trainer.expert_lane import load_king
    from cascade.trainer.pyramid_trainer import losses

    model = load_king(king_dir)
    model.train()
    losses(model, torch.as_tensor(_data(84)[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    return {i: layer.last_load.topk(4).indices.tolist() for i, layer in enumerate(model.moe_layers())}


def test_lane_b_leg_from_an_ema_marked_king_still_scores_raw(tmp_path, plain, contract):
    """§8 Tests: "a lane B leg from an EMA-marked king still scores raw"."""
    init = train(tmp_path / "init", contract)
    king = train(tmp_path / "king", contract, warm_start_dir=init.local_dir)
    if h.MARKER_KEY not in h.config_of(king.local_dir):
        h.mark_config(king.local_dir)          # an EMA-marked king of an earlier image
    assert h.config_of(king.local_dir)[h.MARKER_KEY] == h.MARKER
    selection = _selection(king.local_dir)
    with plain():
        ref = train(tmp_path / "ref", contract, warm_start_dir=king.local_dir,
                    expert_selection=selection)
    leg = train(tmp_path / "leg", contract, warm_start_dir=king.local_dir,
                expert_selection=selection)
    _assert_raw_leg(leg, ref)
    _assert_scored_hashes_equal(leg, ref)
