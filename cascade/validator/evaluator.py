"""Load a trained checkpoint and score it on the shared eval windows.

This is the validator's torch boundary. The trained checkpoints are produced by
the *owner's* trainer (not miners), so their format is trusted and fixed: the
checkpoint directory exposes ``forecast_wrapper.py`` with a ``Wrapper`` class
that loads the model and implements ``forecast(history, horizon, num_samples)``
returning sample arrays. The evaluator adapts that to the numpy
:data:`cascade.eval.scoring.ForecastFn` and runs the pure scoring math.

Because both the king's and the challenger's checkpoints are evaluated on the
*same* :class:`EvalWindow` list with the *same* ``num_samples``, the resulting
:class:`WindowScore` lists are paired and ready for
:func:`cascade.eval.koth.evaluate_round`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from ..eval.scoring import (
    JointForecastFn,
    WindowScore,
    adapt_per_channel,
    score_joint_forecaster_on_windows,
)
from ..eval.window import EvalWindow


class EvaluatorError(RuntimeError):
    """Loading or running a trained checkpoint failed."""


def load_forecaster(
    checkpoint_dir: Path | str, *, device: str = "cpu", contract=None,
    trust_checkpoint_code: bool = False,
) -> JointForecastFn:
    """Import ``forecast_wrapper.Wrapper`` from a trained checkpoint and return
    a numpy JOINT forecaster ``f(history_2d, horizon, num_samples) -> (C, m, H)``.

    Wrappers exposing ``forecast_joint`` are called once per window with all
    channels; archived 1-D wrappers (``forecast`` only) are lifted through the
    permanent per-channel adapter — numerically identical at ``C = 1``.

    The checkpoint is DATA, not code: under miner-funded compute it comes off
    a pod the miner controls, so before anything is imported the guard
    (:mod:`cascade.eval.checkpoint_guard`) requires the shipped
    ``forecast_wrapper.py`` / ``model.py`` to be byte-identical to this
    release's copies, refuses any other ``.py``, and — when ``contract`` is
    given — pins ``config.json`` and the weights header to the contract's
    model before a tensor is allocated. ``trust_checkpoint_code=True`` skips
    the guard and is for ARCHIVED, operator-produced checkpoints only (audit
    of pre-guard rounds whose wrapper predates this release).
    """
    d = Path(checkpoint_dir)
    if not trust_checkpoint_code:
        from ..eval.checkpoint_guard import CheckpointTampered, verify_checkpoint

        try:
            verify_checkpoint(d, contract)
        except CheckpointTampered as e:
            raise EvaluatorError(f"checkpoint_tampered: {e}") from e
    wrapper_py = d / "forecast_wrapper.py"
    if not wrapper_py.is_file():
        raise EvaluatorError(f"missing forecast_wrapper.py in {d}")

    spec = importlib.util.spec_from_file_location("cascade_trained_wrapper", wrapper_py)
    if spec is None or spec.loader is None:
        raise EvaluatorError("wrapper_spec_failed")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cascade_trained_wrapper"] = module
    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        raise EvaluatorError(f"wrapper_import_failed: {type(e).__name__}: {e}") from e

    Wrapper = getattr(module, "Wrapper", None)
    if Wrapper is None:
        raise EvaluatorError("wrapper_class_missing (expected `Wrapper`)")
    try:
        wrapper = Wrapper(str(d), device=device)
    except Exception as e:  # noqa: BLE001
        raise EvaluatorError(f"wrapper_construct_failed: {type(e).__name__}: {e}") from e

    def forecast_fn(history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        out = wrapper.forecast(history, horizon, num_samples)
        arr = np.asarray(out, dtype=np.float64)
        if arr.shape != (1, num_samples, horizon):
            raise EvaluatorError(
                f"wrapper.forecast returned {arr.shape}; expected (1, {num_samples}, {horizon})"
            )
        return arr

    # Joint (C, L) contract (DEC-CA-0026): a wrapper that exposes
    # ``forecast_joint(history_2d, horizon, num_samples) -> (C, ns, H)`` is
    # called once per window with all channels; every archived 1-D wrapper is
    # lifted through the per-channel adapter instead — kept forever, so old
    # checkpoints stay scoreable without resubmission. At C = 1 (every window
    # today) the two paths are numerically identical.
    joint_raw = getattr(wrapper, "forecast_joint", None)
    if joint_raw is None:
        return adapt_per_channel(forecast_fn)

    def joint_fn(history: np.ndarray, horizon: int, num_samples: int) -> np.ndarray:
        arr = np.asarray(joint_raw(history, horizon, num_samples), dtype=np.float64)
        expect = (int(np.atleast_2d(history).shape[0]), num_samples, horizon)
        if arr.shape != expect:
            raise EvaluatorError(
                f"wrapper.forecast_joint returned {arr.shape}; expected {expect}"
            )
        return arr

    return joint_fn


def evaluate_checkpoint(
    checkpoint_dir: Path | str,
    windows: list[EvalWindow],
    *,
    num_samples: int,
    device: str = "cpu",
    contract=None,
    trust_checkpoint_code: bool = False,
    workers: int = 1,
) -> list[WindowScore]:
    """Load the checkpoint and score it on ``windows``. Convenience wrapper over
    :func:`load_forecaster` + :func:`score_joint_forecaster_on_windows`.
    ``contract`` (the entry's TrainingContractConfig) arms the full ingest
    guard — every live scorer should pass it."""
    if workers > 1:
        scores = forecast_checkpoint_windows(
            checkpoint_dir, [w.history for w in windows], [w.horizon for w in windows],
            num_samples=num_samples, device=device, contract=contract, workers=workers,
            trust_checkpoint_code=trust_checkpoint_code,
            _transform=lambda i, sample: score_joint_forecaster_on_windows(
                lambda *_: sample, [windows[i]], num_samples))
        return [score for chunk in scores for score in chunk]
    forecast_fn = load_forecaster(checkpoint_dir, device=device, contract=contract,
                                  trust_checkpoint_code=trust_checkpoint_code)
    return score_joint_forecaster_on_windows(forecast_fn, windows, num_samples)


def forecast_checkpoint_windows(checkpoint_dir, histories, horizons, *, num_samples: int,
                                device: str = "cpu", contract=None, workers: int = 1,
                                trust_checkpoint_code: bool = False,
                                _transform=None) -> list:
    """Parallel independent windows without changing a model's batch or RNG.

    Each worker owns a wrapper; no mutable model cache is shared between
    threads. Imports/guard checks are serialized before worker threads start.
    Explicit concurrency is opt-in and must be qualified on the chosen GPU.
    """
    import queue
    from concurrent.futures import ThreadPoolExecutor

    if workers < 1 or len(histories) != len(horizons) or num_samples < 1:
        raise ValueError("invalid private evaluation work shape or worker count")
    if not histories:
        return []
    count = min(workers, len(histories))
    devices = [device]
    if device == "cuda":
        import torch

        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or [device]
    forecasters = [load_forecaster(
        checkpoint_dir, device=devices[i % len(devices)], contract=contract,
        trust_checkpoint_code=trust_checkpoint_code) for i in range(count)]
    pending = queue.Queue()
    for i in range(len(histories)):
        pending.put(i)
    results = [None] * len(histories)

    def run(forecaster):
        while True:
            try:
                i = pending.get_nowait()
            except queue.Empty:
                return
            sample = forecaster(histories[i], int(horizons[i]), num_samples)
            results[i] = _transform(i, sample) if _transform is not None else sample

    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="private-eval") as pool:
        # Consume every future: a failed window cannot silently disappear.
        for future in [pool.submit(run, f) for f in forecasters]:
            future.result()
    return results
