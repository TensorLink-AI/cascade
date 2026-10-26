"""CPU integration of dispatch, checkpoints, warm starts, expert merges and replay."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch
from safetensors.torch import load_file

from cascade.eval.checkpoint_guard import CheckpointTampered, verify_checkpoint
from cascade.shared.config import parse_chain_config
from cascade.shared.manifest import contract_digest, contract_payload
from cascade.trainer.continuation import Continuation, SnapshotStore
from cascade.trainer.pyramid_model import merge_expert_slices, moe_param_names
from cascade.trainer.pyramid_wrapper import Wrapper
from cascade.trainer.toto2_trainer import Toto2Trainer
from scripts.pyramid_config import dump_toml, local_config


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


def data():
    rng = np.random.default_rng(84)
    return [rng.normal(size=(8, 64)).cumsum(-1) for _ in range(3)]


def train(path, contract, **kwargs):
    return Toto2Trainer(device="cpu").train(iter(data()), contract,
        training_seed=1, token_budget=contract.train_tokens, out_dir=path, **kwargs)


def test_new_backend_dispatch_checkpoint_guard_and_repeatable_forecasts(tmp_path, contract):
    a = train(tmp_path / "a", contract)
    b = train(tmp_path / "b", contract)
    assert a.metrics["steps"] == 3
    assert verify_checkpoint(a.local_dir, contract).n_params == 17528416
    left, right = load_file(str(a.local_dir / 'weights.safetensors')), load_file(str(b.local_dir / 'weights.safetensors'))
    assert all(torch.equal(left[k], right[k]) for k in left)
    wrapper = Wrapper(a.local_dir)
    history = np.arange(4096.0)
    q = wrapper.forecast_quantiles_batch([history, history * 2], 64)
    assert q.shape == (2, 64, 9) and np.isfinite(q).all()
    assert (np.diff(q, axis=-1) > 0).all()
    paths = wrapper.forecast(history, 64, 4)
    assert paths.shape == (1, 4, 64) and np.isfinite(paths).all()
    assert np.array_equal(paths, wrapper.forecast_joint(history[None], 64, 4))
    assert np.array_equal(paths, wrapper.forecast(history, 64, 4))
    multivariate = wrapper.forecast_joint(np.stack([history, history * 2]), 64, 4)
    assert multivariate.shape == (2, 4, 64) and np.isfinite(multivariate).all()
    with pytest.raises(CheckpointTampered, match="contract pins cascade_moe"):
        verify_checkpoint(a.local_dir, replace(contract, base_arch="cascade_moe"))
    for file in ("weights_stable.safetensors", "optimizer.safetensors"):
        assert (a.local_dir / file).is_file()


def test_warm_start_and_subset_training_preserve_frozen_bytes(tmp_path, contract):
    from types import SimpleNamespace

    from cascade.trainer.expert_lane import load_king, probe_routing

    base = train(tmp_path / "base", contract)
    warm = train(tmp_path / "warm", contract, warm_start_dir=base.local_dir)
    verify_checkpoint(warm.local_dir, contract)
    model = load_king(base.local_dir)
    assert [m.w_in.shape[0] for m in model.moe_layers()] == [128, 256, 256, 512, 512, 512, 512]
    from cascade.trainer.pyramid_trainer import losses
    model.train()
    losses(model, torch.as_tensor(data()[0][None]), torch.zeros(1, 8, 2, dtype=torch.bool))
    selection = {i: layer.last_load.topk(4).indices.tolist() for i, layer in enumerate(model.moe_layers())}
    expert = train(tmp_path / "expert", contract, warm_start_dir=base.local_dir, expert_selection=selection)
    king = load_file(str(base.local_dir / "weights.safetensors"))
    candidate = load_file(str(expert.local_dir / "weights.safetensors"))
    merged = merge_expert_slices(king, candidate, selection, moe_param_names(model))
    assert all(torch.equal(candidate[k], merged[k]) for k in candidate)
    assert any(not torch.equal(king[k], candidate[k]) for k in king)
    probe = probe_routing(base.local_dir, [np.arange(128.0)], SimpleNamespace(affinity_windows=1))
    assert probe.affinity.shape == (7, 512)
    assert np.count_nonzero(probe.affinity[0, 128:]) == 0
    assert probe.topk[0].max() < 128


def test_restart_after_an_optimizer_boundary_reproduces_the_checkpoint(tmp_path, contract):
    expected = train(tmp_path / "expected", contract)
    store = SnapshotStore(tmp_path / "snapshots", {"contract": contract_digest(contract), "seed": 1})

    class CrashAfterSecond(Continuation):
        def save(self, state):
            self.store.save(state['step'], state)
            if state['step'] == 2:
                raise RuntimeError("simulated pod loss")

    with pytest.raises(RuntimeError, match="simulated pod loss"):
        train(tmp_path / "failed", contract, continuation=CrashAfterSecond(store, interval=0))
    result = train(tmp_path / "recovered", contract, continuation=Continuation(store, interval=0))
    left = load_file(str(expected.local_dir / "weights.safetensors"))
    right = load_file(str(result.local_dir / "weights.safetensors"))
    assert all(torch.equal(left[k], right[k]) for k in left)
    assert result.metrics["steps"] == 3
    # A completed recovery cannot add an extra update.
    again = train(tmp_path / "again", contract, continuation=Continuation(store, interval=0))
    final = load_file(str(again.local_dir / "weights.safetensors"))
    assert all(torch.equal(right[k], final[k]) for k in right)


def test_explicit_budget_and_complete_offline_configs_do_not_guess_throughput(cfg):
    import tomllib

    for size in ("2m", "7m", "28m", "120m"):
        raw = local_config(size)
        assert tomllib.loads(dump_toml(raw)) == raw
        c = parse_chain_config(raw).training
        assert c.ref_throughput_tokens_per_s == 0
        assert c.train_tokens == 57671680000
        shorter = c.for_hours(0.01)
        assert shorter.training_steps < c.training_steps
        assert shorter.train_tokens == shorter.training_steps * 512 * 4096
        assert not raw['v2'].get('training')
    old = cfg.training
    assert 'training_steps' not in contract_payload(old)
    assert contract_digest(old) == contract_digest(replace(old, training_steps=0))


def test_architecture_and_recipe_mismatches_refuse_before_training(tmp_path, contract):
    with pytest.raises(ValueError, match="d_ff"):
        train(tmp_path, replace(contract, d_ff=1))
    with pytest.raises(ValueError, match="copula_samples"):
        train(tmp_path, replace(contract, copula_samples=8))
    with pytest.raises(ValueError, match="selected experts require"):
        train(tmp_path, contract, expert_selection={0: [1]})


def test_shipped_pyramid_pins_match_the_implemented_source():
    import tomllib

    from cascade.shared.config.training import parse_training_contract
    from cascade.trainer.contract import compute_base_arch_digest
    from scripts.pyramid_config import ROOT

    for size in ('2m', '7m', '28m', '120m'):
        path = ROOT / f'deploy/presets/cascade-moe-v2-{size}.toml'
        c = parse_training_contract(tomllib.loads(path.read_text())['training'])
        assert c.base_arch_digest == compute_base_arch_digest(c)
    for name in ('chain.toml', 'chain.testnet.toml'):
        c = parse_chain_config(tomllib.loads((ROOT / name).read_text())).training.v2_contract
        assert c.base_arch_digest == compute_base_arch_digest(c)


@pytest.mark.parametrize('field,value', [('target_train_hours', 0), ('target_train_hours', float('nan')),
    ('batch_size', 0), ('micro_batches', 0), ('micro_batches', True), ('max_train_seconds', -1)])
def test_bad_update_budget_inputs_refuse_at_config_load(field, value):
    raw = local_config('2m')
    raw['training'][field] = value
    with pytest.raises(ValueError, match=field):
        parse_chain_config(raw)
