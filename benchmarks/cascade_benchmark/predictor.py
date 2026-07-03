"""Adapt a trained cascade checkpoint to the gluonts ``Predictor`` interface.

Every cascade checkpoint ships a ``forecast_wrapper.py``. CPM-era checkpoints
expose the quantile head directly::

    Wrapper(checkpoint_dir, device).forecast_quantiles_batch(histories, horizon)
        -> np.ndarray of shape (B, horizon, num_q)   # levels in .quantile_levels

which is what the benchmark metrics actually consume — MASE/MAE read the 0.5
quantile and CRPS (``mean_weighted_sum_quantile_loss``) reads the full 9-level
grid — so we emit gluonts ``QuantileForecast`` objects straight from the head:
no Monte-Carlo sampling, and the forward passes are batched across series
(``batch_size``), which is what makes full GIFT-Eval/BOOM sweeps tractable.

Older checkpoints only expose the validator contract::

    Wrapper(checkpoint_dir, device).forecast(history_1d, horizon, num_samples)
        -> np.ndarray of shape (1, num_samples, horizon)

for those we fall back to the original per-series ``SampleForecast`` path.

Multivariate entries are forecast one variate at a time and stacked — mirroring
cascade's per-channel scoring (both gluonts suites univariate-ize upstream, so
this branch is rarely hit).
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from gluonts.dataset import Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import Forecast, QuantileForecast, SampleForecast
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
    """gluonts predictor backed by a cascade checkpoint's forecast wrapper."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        prediction_length: int,
        *,
        num_samples: int = 100,
        device: str = "cpu",
        batch_size: int = 64,
    ) -> None:
        super().__init__(prediction_length=prediction_length)
        self.num_samples = int(num_samples)
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self._wrapper = _load_wrapper(Path(checkpoint_dir), device)
        levels = getattr(self._wrapper, "quantile_levels", None)
        self._use_quantiles = (
            hasattr(self._wrapper, "forecast_quantiles_batch") and levels is not None
        )
        if self._use_quantiles:
            self._forecast_keys = [f"{float(v):g}" for v in levels]

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        if self._use_quantiles:
            yield from self._predict_quantiles(dataset)
        else:
            yield from self._predict_samples(dataset)

    # ── quantile-head path (CPM checkpoints): batched, no sampling ────────────

    def _predict_quantiles(self, dataset: Dataset) -> Iterator[QuantileForecast]:
        pending: list[tuple] = []  # (start, item_id, ndim, n_time, [histories])
        n_series = 0
        for entry in dataset:
            target = np.asarray(entry[FieldName.TARGET], dtype=np.float64)
            histories = [target] if target.ndim == 1 else list(target)
            pending.append(
                (entry[FieldName.START], entry.get(FieldName.ITEM_ID),
                 target.ndim, target.shape[-1], histories)
            )
            n_series += len(histories)
            if n_series >= self.batch_size:
                yield from self._flush_quantiles(pending)
                pending, n_series = [], 0
        if pending:
            yield from self._flush_quantiles(pending)

    def _flush_quantiles(self, pending: list[tuple]) -> Iterator[QuantileForecast]:
        histories = [h for *_, hs in pending for h in hs]
        h = self.prediction_length
        q = np.asarray(
            self._wrapper.forecast_quantiles_batch(histories, h), dtype=np.float64
        )
        expected = (len(histories), h, len(self._forecast_keys))
        if q.shape != expected:
            raise ValueError(
                f"wrapper.forecast_quantiles_batch returned {q.shape}; expected {expected}"
            )
        i = 0
        for start, item_id, ndim, n_time, hs in pending:
            rows = q[i : i + len(hs)]
            i += len(hs)
            if ndim == 1:
                arrays = rows[0].T  # (num_q, h)
            else:
                # gluonts multivariate layout: (num_q, h, num_variates)
                arrays = np.stack([r.T for r in rows], axis=-1)
            yield QuantileForecast(
                forecast_arrays=arrays,
                forecast_keys=self._forecast_keys,
                start_date=start + n_time,
                item_id=item_id,
            )

    # ── legacy sample path (pre-CPM checkpoints) ──────────────────────────────

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

    def _predict_samples(self, dataset: Dataset) -> Iterator[SampleForecast]:
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

            yield SampleForecast(
                samples=samples, start_date=start + target.shape[-1], item_id=item_id
            )
