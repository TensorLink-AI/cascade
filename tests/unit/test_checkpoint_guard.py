"""cascade.eval.checkpoint_guard: a checkpoint is data, never code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cascade.eval.checkpoint_guard import (
    CheckpointTampered,
    expected_checkpoint_code,
    safetensors_header,
    verify_checkpoint,
    verify_checkpoint_code,
)

torch = pytest.importorskip("torch")


def _honest_checkpoint(tmp_path: Path, contract) -> Path:
    """Exactly what Toto2Trainer._save_checkpoint writes, from the repo's code."""
    from safetensors.torch import save_file

    from cascade.trainer.toto2_model import Toto2Config, Toto2Model

    d = tmp_path / "ckpt"
    d.mkdir()
    cfg = Toto2Config.from_contract(contract)
    model = Toto2Model(cfg)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(d / "weights.safetensors"))
    (d / "config.json").write_text(json.dumps({
        "arch": contract.arch_preset, "toto2": cfg.to_dict(),
        "quantile_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "input_transform": "arcsinh_causal"}))
    for name, data in expected_checkpoint_code().items():
        (d / name).write_bytes(data)
    return d


@pytest.fixture
def contract(cfg):
    return cfg.training.contract_for(cfg.training.arch_preset)


def test_honest_checkpoint_passes(tmp_path, contract):
    d = _honest_checkpoint(tmp_path, contract)
    report = verify_checkpoint(d, contract)
    assert report.n_params > 0 and report.n_tensors == len(safetensors_header(d / "weights.safetensors"))
    verify_checkpoint_code(d)               # code-only variant too


def test_modified_wrapper_is_refused(tmp_path, contract):
    d = _honest_checkpoint(tmp_path, contract)
    w = d / "forecast_wrapper.py"
    w.write_bytes(w.read_bytes() + b"\nimport os; os.system('id')\n")
    with pytest.raises(CheckpointTampered, match="forecast_wrapper.py differs"):
        verify_checkpoint(d, contract)


def test_modified_model_py_is_refused(tmp_path, contract):
    d = _honest_checkpoint(tmp_path, contract)
    m = d / "model.py"
    m.write_bytes(m.read_bytes().replace(b"class Toto2Config", b"class  Toto2Config", 1))
    with pytest.raises(CheckpointTampered, match="model.py differs"):
        verify_checkpoint_code(d)


def test_extra_python_is_refused(tmp_path, contract):
    d = _honest_checkpoint(tmp_path, contract)
    (d / "sitecustomize.py").write_text("print('hi')")
    with pytest.raises(CheckpointTampered, match="unexpected python"):
        verify_checkpoint_code(d)


def test_config_drift_is_refused(tmp_path, contract):
    d = _honest_checkpoint(tmp_path, contract)
    obj = json.loads((d / "config.json").read_text())
    obj["toto2"]["d_model"] = obj["toto2"]["d_model"] * 2
    (d / "config.json").write_text(json.dumps(obj))
    with pytest.raises(CheckpointTampered, match="config.json.*d_model"):
        verify_checkpoint(d, contract)


def test_weights_header_shape_games_are_refused(tmp_path, contract):
    from safetensors.torch import save_file

    from cascade.trainer.toto2_model import Toto2Config, Toto2Model

    d = _honest_checkpoint(tmp_path, contract)
    state = Toto2Model(Toto2Config.from_contract(contract)).state_dict()
    # an extra tensor
    bad = dict(state); bad["evil.extra"] = torch.zeros(4)
    save_file(bad, str(d / "weights.safetensors"))
    with pytest.raises(CheckpointTampered, match="extra="):
        verify_checkpoint(d, contract)
    # a reshaped tensor
    bad = dict(state); k = next(iter(bad)); bad[k] = torch.zeros(bad[k].shape + (1,))
    save_file(bad, str(d / "weights.safetensors"))
    with pytest.raises(CheckpointTampered, match="shape/dtype"):
        verify_checkpoint(d, contract)


def test_oversize_weights_are_refused_before_load(tmp_path, contract):
    d = _honest_checkpoint(tmp_path, contract)
    with pytest.raises(CheckpointTampered, match="budget"):
        verify_checkpoint(d, contract, max_bytes=1024)


def test_absurd_header_is_refused_without_allocating(tmp_path, contract):
    d = _honest_checkpoint(tmp_path, contract)
    # A header declaring a 1e12-element tensor with no data behind it.
    hdr = json.dumps({"w": {"dtype": "F32", "shape": [1_000_000, 1_000_000],
                            "data_offsets": [0, 8]}}).encode()
    (d / "weights.safetensors").write_bytes(len(hdr).to_bytes(8, "little") + hdr + b"\x00" * 8)
    with pytest.raises(CheckpointTampered, match="data_offsets"):
        verify_checkpoint(d, contract)


def test_evaluator_refuses_tampered_and_trust_flag_bypasses(tmp_path, contract):
    from cascade.validator.evaluator import EvaluatorError, load_forecaster

    d = _honest_checkpoint(tmp_path, contract)
    (d / "forecast_wrapper.py").write_text("class Wrapper:\n    def __init__(self, d, device='cpu'): pass\n")
    with pytest.raises(EvaluatorError, match="checkpoint_tampered"):
        load_forecaster(d, contract=contract)
    # Archived, operator-produced checkpoints can still be loaded explicitly.
    load_forecaster(d, trust_checkpoint_code=True)
