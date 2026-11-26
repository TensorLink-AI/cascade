"""W16 CI tiny run (marked ``torch``: CI only, never the 7 GB box).

A real CPU fine-tune of the smallest pyramid size (``cascade-moe-v2-2m``)
from a random-init base, a few steps on a tiny public-data JSONL. The output
must be a lane D entry the W4 ingest accepts against the contract, trained
(weights moved), deterministic for a fixed seed on CPU, and the stdout
summary must name the entry's manifest digest (what W10 submits)."""

from __future__ import annotations

import json
import math

import pytest

from .helpers import (
    ALLOWED,
    last_json_line,
    pyramid_contract,
    read_header,
    refusal_codes,
    run_script,
    snapshot,
    tiny_base,
    write_data,
)

pytestmark = pytest.mark.torch   # needs torch; CI installs it (uv sync --all-extras)

CPU_ENV = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
RUN_ARGS = ("--steps", "3", "--lr", "0.01", "--seed", "4", "--batch-size", "2",
            "--device", "cpu", "--context-length", "256")


@pytest.fixture
def two_threads():
    state = {}
    yield state
    if "torch" in state:
        state["torch"].set_num_threads(state["threads"])


def _limit(state):
    import torch

    state.setdefault("threads", torch.get_num_threads())
    state["torch"] = torch
    torch.set_num_threads(2)


def _series():
    return [[math.sin(i / (5.0 + k)) * (1 + k) + 0.01 * i for i in range(1024)]
            for k in range(6)]


def _run(tmp_path, base, data, out, *extra):
    return run_script(tmp_path, "--base", str(base), "--data", str(data), "--out", str(out),
                      *RUN_ARGS, *extra, block_torch=False, extra_env=CPU_ENV, timeout=900)


def _bf16_of(base_dir):
    """The base's tensors as BF16 bytes (what an untrained output would hold)."""
    import torch
    from safetensors.torch import load_file

    out = {}
    for name, t in load_file(str(base_dir / "weights.safetensors")).items():
        out[name] = (t.to(torch.bfloat16) if t.is_floating_point() else t).contiguous()
    return out


def _ingest(tmp_path, out_dir, contract, tag):
    from cascade.eval.lane_d_ingest import ingest_entry
    from cascade.lane_d.intake import upload_checkpoint
    from cascade.lane_d.storage import FilesystemStorage

    store = FilesystemStorage(tmp_path / f"vault_{tag}")
    digest = upload_checkpoint(store, f"w16-{tag}", out_dir)
    got = ingest_entry(store, {"hotkey": f"w16-{tag}", "digest": digest},
                       tmp_path / f"ingested_{tag}", contract=contract)
    return digest, got


def test_cpu_tiny_run_from_king_a_f32_base(tmp_path, two_threads):
    import cascade.miner.lane_d_finetune  # noqa: F401  (the W16 module exists)
    # isort: split
    import torch
    from safetensors.torch import load_file

    from cascade.lane_d.intake import checkpoint_manifest, manifest_digest

    _limit(two_threads)
    contract = pyramid_contract()
    base = tiny_base(tmp_path / "base", contract, seed=1, bf16=False)
    base_before = snapshot(base)
    data = write_data(tmp_path / "data.jsonl", _series())
    out = tmp_path / "out"

    proc = _run(tmp_path, base, data, out)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert refusal_codes(proc.stderr) == []

    # exactly the four lane D files, nothing else (no logs, no temp files)
    assert sorted(p.name for p in out.iterdir()) == list(ALLOWED)
    assert all(p.is_file() for p in out.iterdir())
    # the base is read-only input
    assert snapshot(base) == base_before
    # code and config carried over from the base (= the repo's pyramid code)
    for name in ("model.py", "forecast_wrapper.py"):
        assert (out / name).read_bytes() == (base / name).read_bytes(), name
    assert json.loads((out / "config.json").read_text()) == json.loads(
        (base / "config.json").read_text())
    # BF16 for every floating tensor, same names and shapes as the base
    hb, ho = read_header(base / "weights.safetensors"), read_header(out / "weights.safetensors")
    assert set(ho) == set(hb)
    for name, meta in ho.items():
        assert meta["shape"] == hb[name]["shape"], name
        want = "BF16" if hb[name]["dtype"] in ("F32", "BF16") else hb[name]["dtype"]
        assert meta["dtype"] == want, name
    # trained: at least one tensor moved past BF16 rounding of the base
    trained = load_file(str(out / "weights.safetensors"))
    start = _bf16_of(base)
    assert any(not torch.equal(trained[k], start[k]) for k in start)
    assert all(torch.isfinite(t.float()).all() for t in trained.values() if t.is_floating_point())

    # the W4 lane D ingest accepts it against the contract
    digest, ingested = _ingest(tmp_path, out, contract, "f32")
    assert snapshot(ingested) == snapshot(out)

    # the summary line names the entry's manifest digest
    summary = last_json_line(proc.stdout)
    assert summary["out"] == str(out.resolve())
    assert summary["steps"] == 3
    assert isinstance(summary["final_loss"], float) and math.isfinite(summary["final_loss"])
    assert summary["digest"] == manifest_digest(checkpoint_manifest(out)) == digest

    # deterministic on CPU for a fixed seed
    out2 = tmp_path / "out2"
    proc2 = _run(tmp_path, base, data, out2)
    assert proc2.returncode == 0, proc2.stderr[-4000:]
    assert snapshot(out2) == snapshot(out)
    assert last_json_line(proc2.stdout)["digest"] == summary["digest"]


def test_cpu_tiny_run_from_bf16_lane_d_base(tmp_path, two_threads):
    """A lane D king or house (BF16) is a valid start point too."""
    import cascade.miner.lane_d_finetune  # noqa: F401  (the W16 module exists)

    _limit(two_threads)
    contract = pyramid_contract()
    base = tiny_base(tmp_path / "base", contract, seed=2, bf16=True)
    data = write_data(tmp_path / "data.jsonl", _series())
    out = tmp_path / "out"
    proc = _run(tmp_path, base, data, out)
    assert proc.returncode == 0, proc.stderr[-4000:]
    assert sorted(p.name for p in out.iterdir()) == list(ALLOWED)
    assert (out / "weights.safetensors").read_bytes() != (base / "weights.safetensors").read_bytes()
    _, ingested = _ingest(tmp_path, out, contract, "bf16")
    assert snapshot(ingested) == snapshot(out)


def test_dry_run_with_torch_installed_does_not_train(tmp_path, two_threads):
    import cascade.miner.lane_d_finetune  # noqa: F401  (the W16 module exists)

    _limit(two_threads)
    contract = pyramid_contract()
    base = tiny_base(tmp_path / "base", contract, seed=3, bf16=False)
    data = write_data(tmp_path / "data.jsonl", _series())
    out = tmp_path / "out"
    proc = _run(tmp_path, base, data, out, "--dry-run")
    assert proc.returncode == 0, proc.stderr[-4000:]
    plan = last_json_line(proc.stdout)
    assert plan["n_series"] == 6 and plan["out_dtype"] == "BF16"
    assert plan["n_tensors"] == len(read_header(base / "weights.safetensors"))
    assert not out.exists()
