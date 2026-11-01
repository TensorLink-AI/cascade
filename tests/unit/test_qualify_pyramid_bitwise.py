"""The qualification harness's comparison is strict bitwise equality."""
from __future__ import annotations

import json

import torch
from safetensors.torch import save_file

from scripts.qualify_pyramid_bitwise import (
    compare_metrics,
    compare_run_dirs,
    compare_tensor_dicts,
    tensors_bitwise_equal,
)


def test_bitwise_equality_is_stricter_than_close():
    a = torch.tensor([1.0, 2.0, float("nan")])
    assert tensors_bitwise_equal(a, a.clone())                 # identical NaN bits match
    assert not tensors_bitwise_equal(a, a.double())             # dtype
    assert not tensors_bitwise_equal(torch.tensor([0.0]), torch.tensor([-0.0]))
    b = a.clone()
    b[0] = torch.nextafter(b[0], torch.tensor(2.0))
    assert not tensors_bitwise_equal(a, b)                      # one ulp
    assert tensors_bitwise_equal(torch.tensor(3.0), torch.tensor(3.0))   # 0-dim
    assert tensors_bitwise_equal(torch.tensor([True]), torch.tensor([True]))


def test_tensor_dict_reports_every_kind_of_mismatch():
    x = {"w": torch.ones(2, 3), "v": torch.zeros(4), "only_a": torch.ones(1),
         "s": torch.ones(2), "d": torch.ones(2)}
    y = {"w": torch.ones(2, 3), "v": torch.zeros(4) + 1e-30, "only_b": torch.ones(1),
         "s": torch.ones(3), "d": torch.ones(2).double()}
    got = {m.key: m.reason for m in compare_tensor_dicts(x, y)}
    assert set(got) == {"v", "only_a", "only_b", "s", "d"}
    assert "values differ" in got["v"] and "shape" in got["s"] and "dtype" in got["d"]
    assert compare_tensor_dicts(x, {k: t.clone() for k, t in x.items()}) == []


def test_metrics_compare_exact_floats_and_nested():
    a = {"final_loss": 0.1 + 0.2, "steps": 3, "channel_telemetry": {"p": [1.0, 2.0]}}
    assert compare_metrics(a, json.loads(json.dumps(a))) == []
    b = dict(a, final_loss=0.3)
    assert [m.key for m in compare_metrics(a, b)] == ["metrics.final_loss"]
    assert [m.key for m in compare_metrics(a, {"steps": 3, "final_loss": a["final_loss"]})] \
        == ["metrics.channel_telemetry"]


def _run(path, w, loss, logs):
    (path / "ckpt").mkdir(parents=True)
    save_file({"a": w}, str(path / "ckpt" / "weights.safetensors"))
    save_file({"m": w * 2}, str(path / "ckpt" / "optimizer.safetensors"))
    (path / "result.json").write_text(json.dumps({"metrics": {"final_loss": loss, "steps": 3},
                                                  "step_logs": logs, "contract_digest": "x"}))


def test_run_dirs(tmp_path):
    w = torch.randn(5)
    _run(tmp_path / "a", w, 1.5, [{"step": 1, "loss": 2.0}])
    _run(tmp_path / "b", w.clone(), 1.5, [{"step": 1, "loss": 2.0}])
    _run(tmp_path / "c", w.clone(), 1.5, [])
    w2 = w.clone()
    w2[0] += 1
    _run(tmp_path / "d", w2, 1.5, [{"step": 1, "loss": 2.0}])
    assert compare_run_dirs(tmp_path / "a", tmp_path / "b") == []
    assert [m.key for m in compare_run_dirs(tmp_path / "a", tmp_path / "c")] == ["step_logs"]
    assert compare_run_dirs(tmp_path / "a", tmp_path / "c", step_logs=False) == []
    keys = [m.key for m in compare_run_dirs(tmp_path / "a", tmp_path / "d")]
    assert keys == ["weights.safetensors:a", "optimizer.safetensors:m"]
