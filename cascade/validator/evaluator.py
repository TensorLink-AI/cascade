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

from ..eval.scoring import ForecastFn, WindowScore, score_forecaster_on_windows
from ..eval.window import EvalWindow


class EvaluatorError(RuntimeError):
    """Loading or running a trained checkpoint failed."""


def load_forecaster(checkpoint_dir: Path | str, *, device: str = "cpu") -> ForecastFn:
    """Import ``forecast_wrapper.Wrapper`` from a trained checkpoint and return
    a numpy forecaster ``f(history_1d, horizon, num_samples) -> (1, m, H)``.

    The wrapper is owner-produced and trusted; no static guard or sandbox is
    applied here (unlike the miner-controlled generators on the trainer side).
    """
    d = Path(checkpoint_dir)
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

    return forecast_fn


def evaluate_checkpoint(
    checkpoint_dir: Path | str,
    windows: list[EvalWindow],
    *,
    num_samples: int,
    device: str = "cpu",
) -> list[WindowScore]:
    """Load the checkpoint and score it on ``windows``. Convenience wrapper over
    :func:`load_forecaster` + :func:`score_forecaster_on_windows`."""
    forecast_fn = load_forecaster(checkpoint_dir, device=device)
    return score_forecaster_on_windows(forecast_fn, windows, num_samples)
