"""Adapt a trained cascade checkpoint to the gluonts ``Predictor`` interface.

Every cascade checkpoint ships a ``forecast_wrapper.py`` exposing::

    Wrapper(checkpoint_dir, device).forecast(history_1d, horizon, num_samples)
        -> np.ndarray of shape (1, num_samples, horizon)

That is the *same* trusted inference path the validator scores on
(``cascade.validator.evaluator``), so wrapping it here — rather than depending
on a model-specific gluonts class — keeps the benchmark numbers consistent with
the in-protocol scores and makes the sidecar work for any backbone cascade
trains in the future.

gluonts drives evaluation by calling ``Predictor.predict(dataset)`` and reading
``SampleForecast`` objects back. We translate each gluonts entry's ``target``
(the context) into the wrapper's 1-D history, forecast ``prediction_length``
steps, and emit a ``SampleForecast``. Multivariate entries are forecast one
variate at a time and stacked — mirroring cascade's per-channel scoring.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from gluonts.dataset import Dataset
from gluonts.model.forecast import SampleForecast
from gluonts.model.predictor import Predictor


def _load_wrapper(checkpoint_dir: Path, device: str):
    """Import ``forecast_wrapper.Wrapper`` from the checkpoint and instantiate it.

    Mirrors ``cascade.validator.evaluator.load_forecaster`` — the wrapper is
    owner-produced and trusted, so no sandboxing is applied.
    """
    wrapper_py = checkpoint_dir / "forecast_wrapper.py"
    if not wrapper_py.is_file():
        raise FileNotFoundError(f"missing forecast_wrapper.py in {checkpoint_dir}")
    spec = importlib.util.spec_from_file_location("cascade_bench_wrapper", wrapper_py)
    if spec is None or spec.loader is None:
        raise ImportError("could not load forecast_wrapper spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cascade_bench_wrapper"] = module
    spec.loader.exec_module(module)
    Wrapper = getattr(module, "Wrapper", None)
    if Wrapper is None:
        raise AttributeError("forecast_wrapper.py defines no `Wrapper` class")
    return Wrapper(str(checkpoint_dir), device=device)


class CheckpointPredictor(Predictor):
    """gluonts predictor backed by a cascade checkpoint's ``forecast`` wrapper."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        prediction_length: int,
        *,
        num_samples: int = 100,
        device: str = "cpu",
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.num_samples = int(num_samples)
        self.device = device
        self._wrapper = _load_wrapper(Path(checkpoint_dir), device)

    def _forecast_1d(self, history: np.ndarray, horizon: int) -> np.ndarray:
        """Return samples of shape ``(num_samples, horizon)`` for one series."""
        out = np.asarray(
            self._wrapper.forecast(history, horizon, self.num_samples), dtype=np.float64
        )
        # The wrapper contract is (1, num_samples, horizon); drop the batch axis.
        if out.shape != (1, self.num_samples, horizon):
            raise ValueError(
                f"wrapper.forecast returned {out.shape}; "
                f"expected (1, {self.num_samples}, {horizon})"
            )
        return out[0]

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[SampleForecast]:
        from gluonts.dataset.field_names import FieldName

        for entry in dataset:
            target = np.asarray(entry[FieldName.TARGET], dtype=np.float64)
            start = entry[FieldName.START]
            item_id = entry.get(FieldName.ITEM_ID)
            h = self.prediction_length

            if target.ndim == 1:
                samples = self._forecast_1d(target, h)  # (num_samples, h)
            else:
                # Multivariate: (num_variates, time). Forecast each variate
                # independently and stack to gluonts' (num_samples, h, dim).
                per_variate = [self._forecast_1d(target[v], h) for v in range(target.shape[0])]
                samples = np.stack(per_variate, axis=-1)  # (num_samples, h, dim)

            yield SampleForecast(samples=samples, start_date=start + len(target), item_id=item_id)
