"""Real two-process Gloo training and crash/resume, using the actual MoE model."""

import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load

from cascade.direct.storage import FileObjects, publish
from cascade.shared.direct_moe import BatchSchedule
from cascade.trainer import moe_model


def training_spec(tmp_path):
    objects = FileObjects(tmp_path / "objects")
    cfg = moe_model.CascadeMoEConfig(
        d_model=16,
        num_layers=2,
        num_heads=2,
        head_dim=8,
        patch_size=2,
        context_length=8,
        horizon=2,
        max_patches=8,
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
    with torch.device("meta"):
        counts = moe_model.param_counts(moe_model.CascadeMoEModel(cfg))
    arr = np.random.default_rng(4).normal(size=(32, 2, 8)).astype(np.float32)
    stream = io.BytesIO()
    np.save(stream, arr, allow_pickle=False)
    spec = dict(
        job="trained-model",
        fence=1,
        terms_digest="fixed-admitted-contract",
        world_size=2,
        device="cpu",
        precision="float32",
        objective="causal_pinball_v1",
        cpu_threads=1,
        collective_timeout=10,
        seed=77,
        exposure=0,
        budget=135,
        micro_batch=1,
        checkpoint_steps=1,
        grad_clip=1.0,
        storage={"kind": "file", "root": str(objects.root)},
        architecture={
            "config": cfg.to_dict(),
            "expected_active": counts["active"],
            "expected_total": counts["total"],
            "source_sha256": hashlib.sha256(Path(moe_model.__file__).read_bytes()).hexdigest(),
        },
        schedule=asdict(
            BatchSchedule(
                context=8,
                channels=2,
                thresholds=(64, 128),
                batches=(4, 8, 8),
                horizon=1000,
                warmup=16,
            )
        ),
        data={"kind": "array", "object": publish(objects, stream.getvalue()), "epochs": 1},
        optimizer={
            "optimizer": "normuon_adamw",
            "base_lr": 0.001,
            "weight_decay": 0.0,
            "umup_base_d_model": 16,
        },
        output=str(tmp_path / "run"),
    )
    from cascade.direct.fingerprint import implementation_digest

    spec["implementation"] = implementation_digest()
    return spec, objects


def launch(spec, tmp_path, *extra):
    path = tmp_path / "spec.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec))
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={spec['world_size']}",
            "-m",
            "cascade.direct.torch_worker",
            "--spec",
            str(path),
            *extra,
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=90,
    )


def latest(spec):
    files = sorted(Path(spec["output"]).glob("checkpoint-*.json"))
    return json.loads(files[-1].read_text())["manifest"]


def test_real_distributed_rank_failure_exact_resume(tmp_path):
    spec, objects = training_spec(tmp_path)
    full = {**spec, "output": str(tmp_path / "full")}
    result = launch(full, tmp_path / "full-launch")
    assert result.returncode == 0, result.stdout
    expected = latest(full)
    interrupted = {**spec, "output": str(tmp_path / "interrupted")}
    result = launch(
        interrupted, tmp_path / "interrupted-launch", "--fail-rank", "1", "--fail-step", "1"
    )
    assert result.returncode != 0
    partial = latest(interrupted)
    assert partial["consumed"] == 64
    resumed = {**spec, "fence": 2, "resume": partial, "output": str(tmp_path / "resumed")}
    result = launch(resumed, tmp_path / "resumed-launch")
    assert result.returncode == 0, result.stdout
    actual = latest(resumed)
    assert actual["consumed"] == expected["consumed"] == 135
    for rank in ("0", "1"):
        for component in ("weights", "optimizer", "router"):
            a = load(objects.get(actual["ranks"][rank][component]["uri"]))
            b = load(objects.get(expected["ranks"][rank][component]["uri"]))
            assert a.keys() == b.keys()
            assert all(torch.equal(a[k], b[k]) for k in a), (rank, component)
    assert actual["ranks"]["0"]["weights"] == actual["ranks"]["1"]["weights"]
    assert actual["ranks"]["0"]["optimizer"] == actual["ranks"]["1"]["optimizer"]


@pytest.mark.parametrize("precision", ["float32", "bfloat16"])
def test_existing_single_cuda_device_real_training(tmp_path, precision):
    if not torch.cuda.is_available():
        pytest.skip("existing CUDA hardware unavailable; not a GPU validation pass")
    spec, objects = training_spec(tmp_path)
    spec.update(world_size=1, device="cuda", precision=precision)
    result = launch(spec, tmp_path / "gpu-launch")
    assert result.returncode == 0, result.stdout
    manifest = latest(spec)
    assert manifest["consumed"] == 135
    assert set(manifest["ranks"]) == {"0"}
    weights = load(objects.get(manifest["ranks"]["0"]["weights"]["uri"]))
    assert all(torch.isfinite(value).all() for value in weights.values())

    interrupted = {**spec, "output": str(tmp_path / "gpu-interrupted")}
    result = launch(interrupted, tmp_path / "gpu-fault", "--fail-rank", "0", "--fail-step", "1")
    assert result.returncode != 0
    partial = latest(interrupted)
    assert partial["consumed"] == 64
    resumed = {**spec, "fence": 2, "resume": partial, "output": str(tmp_path / "gpu-resumed")}
    result = launch(resumed, tmp_path / "gpu-resume")
    assert result.returncode == 0, result.stdout
    actual = latest(resumed)
    for component in ("weights", "optimizer", "router"):
        expected = load(objects.get(manifest["ranks"]["0"][component]["uri"]))
        restored = load(objects.get(actual["ranks"]["0"][component]["uri"]))
        assert expected.keys() == restored.keys()
        assert all(torch.equal(expected[k], restored[k]) for k in expected), component


def test_real_specialist_training_freezes_and_merges_continuation(tmp_path):
    from cascade.direct.experts import merge_checkpoint
    from cascade.trainer.direct_model import build_direct_model
    from cascade.trainer.moe_model import moe_param_names

    spec, objects = training_spec(tmp_path)
    result = launch(spec, tmp_path / "base-launch")
    assert result.returncode == 0, result.stdout
    base = latest(spec)
    expert = {
        **spec,
        "job": "specialist",
        "initial": base["ranks"]["0"],
        "selection": {"0": [0]},
        "exposure": 135,
        "output": str(tmp_path / "expert"),
    }
    result = launch(expert, tmp_path / "expert-launch")
    assert result.returncode == 0, result.stdout
    candidate = latest(expert)
    model = build_direct_model(spec["architecture"], device="meta")
    names = moe_param_names(model)
    original = load(objects.get(base["ranks"]["0"]["weights"]["uri"]))
    trained = load(objects.get(candidate["ranks"]["0"]["weights"]["uri"]))
    changed = False
    for name in original:
        if name in names[0]:
            assert torch.equal(original[name][1:], trained[name][1:])
            changed |= not torch.equal(original[name][0], trained[name][0])
        else:
            assert torch.equal(original[name], trained[name]), name
    assert changed
    combined = merge_checkpoint(
        objects, base, [(candidate, {0: [0]})], names, optimizer_policy="reset_edited_rows"
    )
    actual = load(objects.get(combined["ranks"]["0"]["weights"]["uri"]))
    assert all(torch.equal(actual[k], trained[k]) for k in actual)
    for rank in combined["ranks"].values():
        state = load(objects.get(rank["optimizer"]["uri"]))
        for name in names[0]:
            assert torch.all(state[f"muon.{name}.momentum"][0] == 0)
            assert torch.all(state[f"muon.{name}.row_v"][0] == 0)


@pytest.mark.skipif(
    not os.environ.get("CASCADE_SANDBOX_TEST_IMAGE"), reason="strict generator image not configured"
)
def test_real_distributed_training_from_archived_sandbox_generator(tmp_path):
    import zipfile

    from cascade.direct.data import tree_digest

    spec, objects = training_spec(tmp_path)
    root = tmp_path / "generator"
    root.mkdir()
    (root / "config.json").write_text("{}")
    (root / "requirements.txt").write_text("")
    (root / "generator.py").write_text(
        "import numpy as np\nfrom cascade.interface import DataGenerator\n"
        "class Generator(DataGenerator):\n"
        "    def __init__(self, config_dir, *, seed):\n        self.rng = np.random.default_rng(seed)\n"
        '    @property\n    def name(self): return "direct-fixture"\n'
        "    def generate(self, n_series):\n"
        "        for _ in range(n_series):\n            yield self.rng.normal(size=(2,8)).astype(np.float32)\n"
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        for path in sorted(root.iterdir()):
            z.writestr(path.name, path.read_bytes())
    spec["data"] = dict(
        kind="sandbox",
        archive=publish(objects, archive.getvalue()),
        source_sha256=tree_digest(root),
        generator_config=dict(
            corpus_n_series=32,
            min_length=8,
            max_length=8,
            max_total_points=512,
            max_generate_seconds=30,
            max_memory_mb=4096,
            max_channels=2,
            sandbox_strict=True,
            sandbox_mode="container",
            sandbox_image=os.environ["CASCADE_SANDBOX_TEST_IMAGE"],
        ),
        shard_examples=32,
        seed=77,
        blocked_imports=[],
    )
    result = launch(spec, tmp_path / "sandbox-launch")
    assert result.returncode == 0, result.stdout
    cp = latest(spec)
    assert cp["consumed"] == 135 and set(cp["ranks"]) == {"0", "1"}
    assert cp["ranks"]["0"]["weights"] == cp["ranks"]["1"]["weights"]
