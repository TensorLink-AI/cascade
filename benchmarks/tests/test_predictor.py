"""CheckpointPredictor — quantile-head path (batched, QuantileForecast) and the
legacy sample-path fallback for pre-CPM checkpoints.

Stub wrappers are written into a tmp checkpoint dir so no torch model is
needed; what's under test is the batching, ordering, and forecast assembly.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.model.forecast import QuantileForecast, SampleForecast

from cascade_benchmark.predictor import CheckpointPredictor

QUANTILE_STUB = textwrap.dedent(
    """
    import numpy as np

    CALLS = []  # batch sizes seen, for the test to assert batching happened

    class Wrapper:
        quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        def __init__(self, checkpoint_dir, device="cpu"):
            pass

        def forecast_quantiles_batch(self, histories, horizon):
            CALLS.append(len(histories))
            # quantile q at every step = last history value + level, so the
            # test can tie each forecast back to its input series
            out = np.stack([
                np.tile(h[-1] + np.asarray(self.quantile_levels), (horizon, 1))
                for h in histories
            ])
            return out  # (B, horizon, 9)
    """
)

SAMPLE_STUB = textwrap.dedent(
    """
    import numpy as np

    class Wrapper:
        def __init__(self, checkpoint_dir, device="cpu"):
            pass

        def forecast(self, history, horizon, num_samples):
            h = np.asarray(history, dtype=np.float64).reshape(-1)
            return np.full((1, num_samples, horizon), h[-1])
    """
)


def _ckpt(tmp_path: Path, stub: str) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "forecast_wrapper.py").write_text(stub, encoding="utf-8")
    return d


def _entry(values, item_id: str):
    return {
        "target": np.asarray(values, dtype=np.float64),
        "start": pd.Period("2020-01-01", freq="D"),
        "item_id": item_id,
    }


def test_quantile_wrapper_yields_batched_quantile_forecasts(tmp_path: Path):
    ckpt = _ckpt(tmp_path, QUANTILE_STUB)
    predictor = CheckpointPredictor(ckpt, prediction_length=5, batch_size=3)
    dataset = [_entry(np.arange(10) + 100 * i, f"s{i}") for i in range(7)]

    forecasts = list(predictor.predict(dataset))
    assert len(forecasts) == 7
    assert all(isinstance(f, QuantileForecast) for f in forecasts)
    for i, f in enumerate(forecasts):
        assert f.item_id == f"s{i}"  # order preserved across batches
        last = dataset[i]["target"][-1]
        assert np.allclose(f.quantile("0.5"), last + 0.5)
        assert np.allclose(f.quantile("0.9"), last + 0.9)
        assert f.start_date == dataset[i]["start"] + 10

    calls = sys.modules["cascade_bench_wrapper"].CALLS
    assert calls == [3, 3, 1]  # 7 series flushed in batches of 3


def test_quantile_wrapper_multivariate_stacks_variates(tmp_path: Path):
    ckpt = _ckpt(tmp_path, QUANTILE_STUB)
    predictor = CheckpointPredictor(ckpt, prediction_length=4, batch_size=8)
    target = np.stack([np.arange(12.0), np.arange(12.0) + 50])
    (f,) = predictor.predict([{"target": target, "start": pd.Period("2020-01-01", freq="D")}])
    assert isinstance(f, QuantileForecast)
    assert f.forecast_array.shape == (9, 4, 2)
    assert f.start_date == pd.Period("2020-01-01", freq="D") + 12  # time, not variates


MISMATCHED_GRID_STUB = textwrap.dedent(
    """
    import numpy as np

    class Wrapper:
        quantile_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # not the metric grid

        def __init__(self, checkpoint_dir, device="cpu"):
            pass

        def forecast_quantiles_batch(self, histories, horizon):
            raise AssertionError("quantile path must not be used on a mismatched grid")

        def forecast(self, history, horizon, num_samples):
            h = np.asarray(history, dtype=np.float64).reshape(-1)
            return np.full((1, num_samples, horizon), h[-1])
    """
)


def test_mismatched_quantile_grid_falls_back_to_samples(tmp_path: Path):
    """A wrapper whose quantile grid differs from the one the metrics request
    must take the sample path: QuantileForecast would silently interpolate or
    tail-extrapolate the missing levels, making numbers non-comparable with
    same-grid checkpoints, while samples can serve any level correctly."""
    ckpt = _ckpt(tmp_path, MISMATCHED_GRID_STUB)
    predictor = CheckpointPredictor(ckpt, prediction_length=4, num_samples=7)
    (f,) = predictor.predict([_entry(np.arange(10), "a")])
    assert isinstance(f, SampleForecast)
    assert f.samples.shape == (7, 4)


def test_legacy_sample_wrapper_falls_back_to_sample_forecasts(tmp_path: Path):
    ckpt = _ckpt(tmp_path, SAMPLE_STUB)
    predictor = CheckpointPredictor(ckpt, prediction_length=5, num_samples=11)
    dataset = [_entry(np.arange(10), "a"), _entry(np.arange(20) + 7, "b")]

    forecasts = list(predictor.predict(dataset))
    assert len(forecasts) == 2
    assert all(isinstance(f, SampleForecast) for f in forecasts)
    assert forecasts[0].samples.shape == (11, 5)
    assert np.allclose(forecasts[1].samples, 26.0)
    assert forecasts[1].start_date == dataset[1]["start"] + 20
