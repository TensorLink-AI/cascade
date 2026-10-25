"""Native FEV data loading and real MoE forecasting on a tiny local fixture."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from cascade_benchmark.direct_cli import canonical, verify_fev_manifest
from cascade_benchmark.parallel import run_sweep
from datasets import Dataset


def test_native_fev_pinned_data_and_real_forecasts(tmp_path, monkeypatch):
    import datasets

    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf-cache"))
    from cascade.trainer.moe_model import QUANTILE_LEVELS, CascadeMoEConfig, CascadeMoEModel
    from cascade.trainer.moe_trainer import _save_moe_checkpoint

    cfg = CascadeMoEConfig(
        d_model=16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        patch_size=2,
        context_length=8,
        horizon=2,
        max_patches=16,
        variate_blocks=(),
        moe_blocks=(1,),
        dense_ffn=24,
        num_experts=4,
        top_k=2,
        expert_hidden=4,
        shared_hidden=8,
        conv_channels=4,
        conv_kernel=3,
        conv_dilations=(1,),
        cpm_c_max=1,
    )
    model = CascadeMoEModel(cfg)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    _save_moe_checkpoint(
        checkpoint,
        model.state_dict(),
        cfg,
        QUANTILE_LEVELS,
        SimpleNamespace(arch_preset="direct-fixture", input_transform="arcsinh_causal"),
    )
    data = tmp_path / "fixture.parquet"
    Dataset.from_dict(
        {
            "id": ["a", "b"],
            "timestamp": [pd.date_range("2020-01-01", periods=32).tolist()] * 2,
            "target": [np.sin(np.arange(32) / 3).tolist(), np.cos(np.arange(32) / 3).tolist()],
        }
    ).to_parquet(data)
    task = tmp_path / "tasks.yaml"
    task.write_text(
        yaml.safe_dump(
            {
                "tasks": [
                    {"dataset_path": str(data), "task_name": "fixture/h2", "horizon": 2,
                     "seasonality": 1, "num_windows": 1},
                    {"dataset_path": str(data), "task_name": "fixture/h3", "horizon": 3,
                     "seasonality": 1, "num_windows": 1},
                ]
            }
        )
    )
    manifest = dict(
        tasks=dict(path=str(task), sha256=hashlib.sha256(task.read_bytes()).hexdigest()),
        datasets=[dict(path=str(data), sha256=hashlib.sha256(data.read_bytes()).hexdigest())],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    pinned_identity = verify_fev_manifest(path)
    normalized_tasks = yaml.safe_load(task.read_text())
    for row in normalized_tasks["tasks"]:
        row["dataset_path"] = {"dataset_sha256": manifest["datasets"][0]["sha256"]}
    expected = {"tasks": normalized_tasks, "datasets": [manifest["datasets"][0]["sha256"]]}
    assert pinned_identity == hashlib.sha256(canonical(expected).encode()).hexdigest()
    # Moving identical pinned bytes to another validator's mount preserves
    # both the data revision and task identities. Changing a horizon does not.
    moved = tmp_path / "other-mount"
    moved.mkdir()
    moved_data = moved / "data.parquet"
    moved_data.write_bytes(data.read_bytes())
    moved_tasks = moved / "tasks.yaml"
    moved_tasks.write_text(task.read_text().replace(str(data), str(moved_data)))
    moved_manifest = moved / "manifest.json"

    def pin_moved():
        moved_manifest.write_text(json.dumps({
            "tasks": {"path": str(moved_tasks), "sha256": hashlib.sha256(moved_tasks.read_bytes()).hexdigest()},
            "datasets": [{"path": str(moved_data), "sha256": manifest["datasets"][0]["sha256"]}]}))
        return verify_fev_manifest(moved_manifest)

    assert pin_moved() == pinned_identity
    changed = yaml.safe_load(moved_tasks.read_text())
    changed["tasks"][1]["horizon"] = 4
    moved_tasks.write_text(yaml.safe_dump(changed))
    assert pin_moved() != pinned_identity
    verify_fev_manifest(path)
    result = run_sweep(["fev"], [str(checkpoint)], num_samples=8, batch_size=1,
                       max_series=None, device="cpu", workers=1)[0][0]
    assert result.status == "ok" and not result.skipped, result.detail
    assert result.n_expected == len(result.rows) == 2
    assert np.isfinite(result.rows[0]["MASE"]) and np.isfinite(result.rows[0]["CRPS"])
    parallel = run_sweep(["fev"], [str(checkpoint)], num_samples=8, batch_size=1,
                         max_series=None, device="cpu", workers=2)[0][0]
    assert parallel == result
    data.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="dataset hash"):
        verify_fev_manifest(path)
