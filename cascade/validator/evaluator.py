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

from ..eval.crps import DEFAULT_QUANTILE_LEVELS
from ..eval.scoring import (
    JointForecastFn,
    JointQuantileFn,
    WindowScore,
    adapt_per_channel,
    score_joint_forecaster_on_windows,
    score_joint_quantile_forecaster_on_windows,
)
from ..eval.window import EvalWindow


class EvaluatorError(RuntimeError):
    """Loading or running a trained checkpoint failed."""


def _import_wrapper(checkpoint_dir: Path | str, *, device: str, contract,
                    trust_checkpoint_code: bool):
    """Guard (unless trusted), import and construct a checkpoint's ``Wrapper``.
    Shared by the sample and the quantile-head loaders so both run the exact
    same ingest guard."""
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
        return Wrapper(str(d), device=device)
    except Exception as e:  # noqa: BLE001
        raise EvaluatorError(f"wrapper_construct_failed: {type(e).__name__}: {e}") from e


def quantile_indices(levels, wanted, *, tol: float = 1e-9) -> list[int] | None:
    """Positions of every ``wanted`` level inside a wrapper's ``levels``
    (first match within ``tol``), or ``None`` when one is missing — the
    quantile head is scored only on levels it emits; nothing is interpolated."""
    if levels is None:
        return None
    have = [float(v) for v in levels]
    out: list[int] = []
    for q in wanted:
        hits = [i for i, v in enumerate(have) if abs(v - float(q)) <= tol]
        if not hits:
            return None
        out.append(hits[0])
    return out


def load_quantile_forecaster(
    checkpoint_dir: Path | str, *, device: str = "cpu", contract=None,
    trust_checkpoint_code: bool = False,
    quantile_levels=DEFAULT_QUANTILE_LEVELS,
) -> JointQuantileFn:
    """Import a checkpoint's ``Wrapper`` and return a numpy JOINT QUANTILE
    forecaster ``f(history_2d, horizon) -> (C, H, num_q)`` at ``quantile_levels``
    (DEC-CA-0067 ``quantile_head_after``): the wrapper's
    ``forecast_quantiles_batch(histories, horizon) -> (B, H, num_q_head)`` is
    called once per window with the channels as the batch — on a univariate
    window (every private window today) the same forward the sample path
    runs; a multichannel window is scored channel by channel, the benchmark
    path's convention, where the sample path's ``forecast_joint`` conditioned
    across channels — and the scorer's grid is picked out of the head's
    ``quantile_levels`` by exact
    match (nothing interpolated). Runs the same ingest guard as
    :func:`load_forecaster`. A wrapper without a quantile head, or whose head
    lacks a grid level, raises :class:`EvaluatorError` (``quantile_head_
    unavailable``) — deterministic for the checkpoint's bytes."""
    wrapper = _import_wrapper(checkpoint_dir, device=device, contract=contract,
                              trust_checkpoint_code=trust_checkpoint_code)
    batch_fn = getattr(wrapper, "forecast_quantiles_batch", None)
    idx = quantile_indices(getattr(wrapper, "quantile_levels", None), quantile_levels)
    if batch_fn is None or idx is None:
        raise EvaluatorError(
            "quantile_head_unavailable: the checkpoint's wrapper exposes no "
            "forecast_quantiles_batch covering the scoring grid "
            f"{tuple(float(q) for q in quantile_levels)} (head levels: "
            f"{getattr(wrapper, 'quantile_levels', None)!r})")
    picks = np.asarray(idx, dtype=np.int64)
    n_q = len(picks)

    def quantile_fn(history: np.ndarray, horizon: int) -> np.ndarray:
        hist = np.atleast_2d(np.asarray(history, dtype=np.float64))
        try:
            arr = np.asarray(batch_fn(list(hist), int(horizon)), dtype=np.float64)
        except Exception as e:  # noqa: BLE001 — the wrapper's own refusal (e.g. a
            # non-finite head) is deterministic for these bytes: an entry fault,
            # never an endless retry. (An out-of-memory text stays transient:
            # EvaluationMixin._entry_fault_reason reads it.)
            raise EvaluatorError(
                f"quantile_head_forward_failed: {type(e).__name__}: {e}") from e
        expect = (int(hist.shape[0]), int(horizon))
        if arr.ndim != 3 or arr.shape[:2] != expect:
            raise EvaluatorError(
                f"wrapper.forecast_quantiles_batch returned {arr.shape}; expected "
                f"({expect[0]}, {expect[1]}, num_q)")
        return arr[:, :, picks].reshape(expect[0], expect[1], n_q)

    return quantile_fn


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
    wrapper = _import_wrapper(checkpoint_dir, device=device, contract=contract,
                              trust_checkpoint_code=trust_checkpoint_code)

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
    quantile_head: bool = False,
) -> list[WindowScore]:
    """Load the checkpoint and score it on ``windows``. Convenience wrapper over
    :func:`load_forecaster` + :func:`score_joint_forecaster_on_windows`.
    ``contract`` (the entry's TrainingContractConfig) arms the full ingest
    guard — every live scorer should pass it.

    ``quantile_head=True`` (DEC-CA-0067 ``quantile_head_after``, a v2
    settlement from the switch era on) scores the checkpoint's quantile head
    directly — :func:`load_quantile_forecaster` +
    :func:`score_joint_quantile_forecaster_on_windows`; ``num_samples`` is then
    unused (no Monte-Carlo step). False is byte-identical to before the knob."""
    if quantile_head:
        levels = DEFAULT_QUANTILE_LEVELS
        if workers > 1:
            scores = forecast_checkpoint_windows(
                checkpoint_dir, [w.history for w in windows], [w.horizon for w in windows],
                num_samples=num_samples, device=device, contract=contract, workers=workers,
                trust_checkpoint_code=trust_checkpoint_code, quantile_levels=levels,
                _transform=lambda i, q: score_joint_quantile_forecaster_on_windows(
                    lambda *_: q, [windows[i]], levels))
            return [score for chunk in scores for score in chunk]
        quantile_fn = load_quantile_forecaster(
            checkpoint_dir, device=device, contract=contract,
            trust_checkpoint_code=trust_checkpoint_code, quantile_levels=levels)
        return score_joint_quantile_forecaster_on_windows(quantile_fn, windows, levels)
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
                                _transform=None, quantile_levels=None) -> list:
    """Parallel independent windows without changing a model's batch or RNG.

    Each worker owns a wrapper; no mutable model cache is shared between
    threads. Imports/guard checks are serialized before worker threads start.
    Explicit concurrency is opt-in and must be qualified on the chosen GPU.

    ``quantile_levels`` set ⇒ the quantile-head path: each result is the
    window's ``(C, H, num_q)`` quantiles at those levels
    (:func:`load_quantile_forecaster`); ``num_samples`` is then only validated.
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
    if quantile_levels is not None:
        forecasters = [load_quantile_forecaster(
            checkpoint_dir, device=devices[i % len(devices)], contract=contract,
            trust_checkpoint_code=trust_checkpoint_code, quantile_levels=quantile_levels)
            for i in range(count)]
    else:
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
            if quantile_levels is not None:
                sample = forecaster(histories[i], int(horizons[i]))
            else:
                sample = forecaster(histories[i], int(horizons[i]), num_samples)
            results[i] = _transform(i, sample) if _transform is not None else sample

    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="private-eval") as pool:
        # Consume every future: a failed window cannot silently disappear.
        for future in [pool.submit(run, f) for f in forecasters]:
            future.result()
    return results
