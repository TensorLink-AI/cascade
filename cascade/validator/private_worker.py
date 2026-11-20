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
# DEC-CA-0067 ``quantile_head_after``: a request for the checkpoint's QUANTILES
# (``(C, H, num_q)`` per window at ``metadata["quantile_levels"]``) instead of
# samples. A distinct schema string, so a worker from before the knob refuses
# the request outright ("incompatible") instead of answering with samples the
# validator would then reject — either way the round holds as infrastructure.
SCHEMA_QUANTILES = "cascade-private-eval/2-quantiles"
# DEC-CA-0068: a quantile request decoded in fixed batches
# (``metadata["quantile_batch"]``, :func:`cascade.validator.evaluator.quantile_chunks`).
# A distinct schema, so a worker from before it REFUSES instead of answering at
# batch 1 (whose numerics differ) — the round holds as infrastructure.
SCHEMA_QUANTILES_BATCHED = "cascade-private-eval/3-quantiles-batched"
CHECKPOINT_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")


def _per_window_depth(num_samples, quantile_levels) -> int:
    """Rows per (channel, horizon step) in a reply: the quantile count on the
    quantile-head path, else the sample count."""
    return len(quantile_levels) if quantile_levels is not None else int(num_samples)


def _expected_shape(window, num_samples, quantile_levels) -> tuple[int, int, int]:
    if quantile_levels is not None:
        return (window.n_channels, window.horizon, len(quantile_levels))
    return (window.n_channels, int(num_samples), window.horizon)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_hashes(directory: Path) -> dict:
    return {name: file_hash(directory / name) for name in CHECKPOINT_FILES}


def write_request(path, checkpoint, windows, *, contract, num_samples, quantile_levels=None,
                  hashes=None, quantile_batch=1):
    """``quantile_levels`` set ⇒ a quantile-head request (:data:`SCHEMA_QUANTILES`);
    None ⇒ the sample request, byte-identical to before the knob. ``hashes``:
    the caller's :func:`checkpoint_hashes` of ``checkpoint`` (saves a re-hash)."""
    from ..eval.torch_runtime import capture_runtime

    metadata = {"schema": SCHEMA, "contract": asdict(contract),
                "runtime": capture_runtime(),
                "num_samples": int(num_samples),
                "horizons": [w.horizon for w in windows],
                "checkpoint": (dict(hashes) if hashes is not None
                                else checkpoint_hashes(Path(checkpoint)))}
    if quantile_levels is not None:
        metadata["schema"] = SCHEMA_QUANTILES
        metadata["quantile_levels"] = [float(q) for q in quantile_levels]
        if int(quantile_batch) > 1:
            metadata["schema"] = SCHEMA_QUANTILES_BATCHED
            metadata["quantile_batch"] = int(quantile_batch)
    elif int(quantile_batch) > 1:
        raise ValueError("a batched decode is a quantile-head request only")
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
        if metadata.get("schema") not in (SCHEMA, SCHEMA_QUANTILES, SCHEMA_QUANTILES_BATCHED):
            raise ValueError("incompatible private evaluation request")
        quantile_levels = None
        quantile_batch = 1
        if metadata["schema"] in (SCHEMA_QUANTILES, SCHEMA_QUANTILES_BATCHED):
            quantile_levels = tuple(float(q) for q in metadata["quantile_levels"])
            if not quantile_levels:
                raise ValueError("private evaluation quantile request names no levels")
        if metadata["schema"] == SCHEMA_QUANTILES_BATCHED:
            quantile_batch = metadata.get("quantile_batch")
            if isinstance(quantile_batch, bool) or not isinstance(quantile_batch, int) \
                    or quantile_batch < 2:
                raise ValueError("private evaluation batched request names no batch")
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
            device=device, contract=contract, workers=1, quantile_levels=quantile_levels,
            quantile_batch=quantile_batch)
        baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    samples = forecast_checkpoint_windows(
        checkpoint, histories, horizons, num_samples=int(metadata["num_samples"]),
        device=device, contract=contract, workers=workers, quantile_levels=quantile_levels,
        quantile_batch=quantile_batch)
    seconds = time.perf_counter() - started
    if baseline is not None and (len(samples) != len(baseline) or any(
            not np.array_equal(a, b) for a, b in zip(samples, baseline, strict=True))):
        raise ValueError("private GPU qualification failed: forecasts differ from sequential")
    arrays = {f"s_{i}": s for i, s in enumerate(samples)}
    arrays["request_sha256"] = np.frombuffer(request_hash.encode(), dtype=np.uint8)
    np.savez(output, **arrays)
    return {"schema": metadata["schema"], "request_sha256": request_hash, "windows": len(histories),
            "workers": workers, "quantile_batch": quantile_batch, "seconds": seconds, "sequential_seconds": baseline_seconds,
            "qualified": bool(qualify), "device": device,
            "torch": torch.__version__, "cuda": torch.version.cuda, "numpy": np.__version__,
            "devices": ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if str(device).startswith("cuda") else ["cpu"])}


def check_forecasts(path, request_hash, windows, num_samples, quantile_levels=None) -> None:
    """Check exact coverage, request binding, shape, dtype and finiteness of
    EVERY window before any score is computed — one window in memory at a
    time (the validator box holds 7 GB; a full duel's samples are ~GBs).
    ``quantile_levels`` set ⇒ the reply holds ``(C, H, num_q)`` quantiles."""
    import zipfile

    depth = _per_window_depth(num_samples, quantile_levels)
    max_bytes = sum(w.n_channels * depth * w.horizon * 8 for w in windows)
    with zipfile.ZipFile(path) as archive:
        if sum(f.file_size for f in archive.infolist()) > max_bytes + (len(windows) + 1) * 4096:
            raise ValueError("private evaluation result exceeds its shape budget")
    with np.load(path, allow_pickle=False) as data:
        expected = {"request_sha256"} | {f"s_{i}" for i in range(len(windows))}
        if len(data.files) != len(expected) or set(data.files) != expected:
            raise ValueError("private evaluation output coverage mismatch")
        if data["request_sha256"].tobytes().decode() != request_hash:
            raise ValueError("private evaluation response belongs to another request")
        for i, window in enumerate(windows):
            arr = data[f"s_{i}"]
            _check_window(arr, window, num_samples, quantile_levels)
            del arr


def _check_window(arr, window, num_samples, quantile_levels=None) -> None:
    if arr.dtype != np.float64 or arr.shape != _expected_shape(window, num_samples, quantile_levels):
        raise ValueError("private evaluation forecast shape/dtype mismatch")
    if not np.isfinite(arr).all():
        raise ValueError("private evaluation forecast is non-finite")


def iter_forecasts(path, windows, num_samples, quantile_levels=None):
    """Yield each window's samples (or quantiles) in window order, read lazily
    from the NPZ (call :func:`check_forecasts` first). Same arrays, same order
    as :func:`read_forecasts` — only the peak memory differs."""
    with np.load(path, allow_pickle=False) as data:
        for i, window in enumerate(windows):
            arr = data[f"s_{i}"]
            _check_window(arr, window, num_samples, quantile_levels)
            yield arr


def read_forecasts(path, request_hash, windows, num_samples, quantile_levels=None):
    """Check exact coverage, request binding, shape and dtype before scoring."""
    check_forecasts(path, request_hash, windows, num_samples, quantile_levels)
    return list(iter_forecasts(path, windows, num_samples, quantile_levels))


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
