"""Operator-owned GPU worker: private histories in, forecast samples out.

Targets, pool metadata, wallets and consensus decisions stay on the validator.
NPZ uses numeric arrays only, never pickle. This is an explicit trusted-host
transport, separate from public benchmarking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

SCHEMA = "cascade-private-eval/1"
CHECKPOINT_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_hashes(directory: Path) -> dict:
    return {name: file_hash(directory / name) for name in CHECKPOINT_FILES}


def write_request(path, checkpoint, windows, *, contract, num_samples):
    from ..eval.torch_runtime import capture_runtime

    metadata = {"schema": SCHEMA, "contract": asdict(contract),
                "runtime": capture_runtime(),
                "num_samples": int(num_samples),
                "horizons": [w.horizon for w in windows],
                "checkpoint": checkpoint_hashes(Path(checkpoint))}
    arrays = {f"h_{i}": np.ascontiguousarray(w.history, dtype=np.float64)
              for i, w in enumerate(windows)}
    arrays["metadata"] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
    np.savez(path, **arrays)
    return file_hash(Path(path))


def run_request(checkpoint, request, output, *, device="cuda", workers=1, qualify=False):
    import torch

    from ..eval.torch_runtime import apply_runtime
    from ..shared.config.training import SizeSpec, TrainingContractConfig
    from .evaluator import forecast_checkpoint_windows

    # The opt-in worker owns its process; avoid nested BLAS/torch pools.
    torch.set_num_threads(1)
    request_hash = file_hash(Path(request))
    with np.load(request, allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes())
        if metadata.get("schema") != SCHEMA:
            raise ValueError("incompatible private evaluation request")
        horizons = metadata["horizons"]
        if set(data.files) != {"metadata"} | {f"h_{i}" for i in range(len(horizons))}:
            raise ValueError("private evaluation input coverage mismatch")
        histories = [data[f"h_{i}"] for i in range(len(horizons))]
    if checkpoint_hashes(Path(checkpoint)) != metadata["checkpoint"]:
        raise ValueError("private evaluation checkpoint hash mismatch")
    apply_runtime(metadata["runtime"])
    fields = dict(metadata["contract"])
    fields["extra_sizes"] = tuple(SizeSpec(**s) for s in fields.get("extra_sizes", []))
    contract = TrainingContractConfig(**fields)
    baseline = None
    baseline_seconds = None
    if qualify:
        started = time.perf_counter()
        baseline = forecast_checkpoint_windows(
            checkpoint, histories, horizons, num_samples=int(metadata["num_samples"]),
            device=device, contract=contract, workers=1)
        baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    samples = forecast_checkpoint_windows(
        checkpoint, histories, horizons, num_samples=int(metadata["num_samples"]),
        device=device, contract=contract, workers=workers)
    seconds = time.perf_counter() - started
    if baseline is not None and (len(samples) != len(baseline) or any(
            not np.array_equal(a, b) for a, b in zip(samples, baseline, strict=True))):
        raise ValueError("private GPU qualification failed: forecasts differ from sequential")
    arrays = {f"s_{i}": s for i, s in enumerate(samples)}
    arrays["request_sha256"] = np.frombuffer(request_hash.encode(), dtype=np.uint8)
    np.savez(output, **arrays)
    return {"schema": SCHEMA, "request_sha256": request_hash, "windows": len(histories),
            "workers": workers, "seconds": seconds, "sequential_seconds": baseline_seconds,
            "qualified": bool(qualify), "device": device,
            "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
            "devices": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if str(device).startswith("cuda") else ["cpu"])}


def read_forecasts(path, request_hash, windows, num_samples):
    """Check exact coverage, request binding, shape and dtype before scoring."""
    import zipfile

    max_bytes = sum(w.n_channels * num_samples * w.horizon * 8 for w in windows)
    with zipfile.ZipFile(path) as archive:
        if sum(f.file_size for f in archive.infolist()) > max_bytes + (len(windows) + 1) * 4096:
            raise ValueError("private evaluation result exceeds its shape budget")
    with np.load(path, allow_pickle=False) as data:
        expected = {"request_sha256"} | {f"s_{i}" for i in range(len(windows))}
        if len(data.files) != len(expected) or set(data.files) != expected:
            raise ValueError("private evaluation output coverage mismatch")
        if data["request_sha256"].tobytes().decode() != request_hash:
            raise ValueError("private evaluation response belongs to another request")
        samples = []
        for i, window in enumerate(windows):
            arr = data[f"s_{i}"]
            if arr.dtype != np.float64 or arr.shape != (window.n_channels, num_samples, window.horizon):
                raise ValueError("private evaluation forecast shape/dtype mismatch")
            if not np.isfinite(arr).all():
                raise ValueError("private evaluation forecast is non-finite")
            samples.append(arr)
    return samples


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("request")
    p.add_argument("output")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--qualify", action="store_true",
                   help="also run sequentially; fail unless every forecast is exactly equal")
    args = p.parse_args(argv)
    report = run_request(args.checkpoint, args.request, args.output, device=args.device,
                         workers=args.workers, qualify=args.qualify)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
