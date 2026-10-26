"""A 21-level head must serve all four suites on their unchanged metric grids."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from cascade_benchmark.quantiles import forecast_on_grid, quantile_indices
from cascade_benchmark.suites.fev_bench import score_task
from cascade_benchmark.suites.time_bench import _instance_quantiles, _score_prepared

METRIC = tuple(i / 10 for i in range(1, 10))
LEVELS = (.01, *(i / 20 for i in range(1, 20)), .99)


class RichHead:
    quantile_levels = LEVELS

    def forecast_quantiles_batch(self, histories, horizon):
        return np.stack([np.tile(np.asarray(self.quantile_levels) + h[-1], (horizon, 1))
                         for h in histories])

    def forecast(self, *args):
        raise AssertionError('a complete Q21 head must never fall back to samples')


def test_selects_exact_levels_and_refuses_missing_ambiguous_or_invalid_grids():
    indices = quantile_indices(LEVELS, METRIC)
    np.testing.assert_array_equal(np.asarray(LEVELS)[indices], METRIC)
    assert quantile_indices((.1, .5, .9), METRIC) is None
    assert quantile_indices((.1, .1, .5, .9), METRIC) is None
    assert quantile_indices((.1, float('nan'), .9), METRIC) is None
    assert quantile_indices((0, .1, .9, 1), METRIC) is None
    assert quantile_indices(None, METRIC) is None
    assert quantile_indices(np.asarray(LEVELS, dtype=np.float32), METRIC) is not None


def test_q21_selects_columns_instead_of_reshaping_or_interpolating():
    q = forecast_on_grid(RichHead(), [np.arange(12.0)], 4, METRIC)
    assert q.shape == (1, 4, 9)
    np.testing.assert_array_equal(q[0, 0], 11 + np.asarray(METRIC))
    with pytest.raises(ValueError, match='every required'):
        forecast_on_grid(RichHead(), [np.arange(12.0)], 4, (.025, .5, .975))


def test_gift_and_boom_predictor_preserves_the_nine_level_metric_grid(tmp_path):
    from cascade_benchmark.predictor import CheckpointPredictor
    from gluonts.model.forecast import QuantileForecast
    from test_predictor import QUANTILE_STUB

    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    text = QUANTILE_STUB.replace(
        'quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]',
        f'quantile_levels = {list(LEVELS)!r}')
    (checkpoint / 'forecast_wrapper.py').write_text(text)
    predictor = CheckpointPredictor(str(checkpoint), prediction_length=4)
    outputs = list(predictor.predict([{'target': np.arange(12.0), 'start': pd.Period('2020-01', freq='M')}]))
    assert len(outputs) == 1 and isinstance(outputs[0], QuantileForecast)
    assert len(outputs[0].forecast_keys) == 9
    for level in METRIC:
        np.testing.assert_allclose(outputs[0].quantile(level), 11 + level)


def test_time_single_and_batched_paths_select_the_same_metric_levels(tmp_path, monkeypatch):
    from cascade_benchmark.suites import time_bench

    context = np.stack([np.arange(12.0), np.arange(12.0) + 20])
    direct = _instance_quantiles(RichHead(), context, 4, 8)
    assert direct.shape == (9, 2, 4)
    np.testing.assert_array_equal(direct[:, 0, 0], 11 + np.asarray(METRIC))
    seen = []

    def metrics(dataset, forecasts, *args):
        seen.append(forecasts.copy())
        return {'MASE': 1., 'CRPS': 1.}

    monkeypatch.setattr(time_bench, '_metrics_from_quantiles', metrics)
    prep = {'name': 'fixture', 'term': 'short', 'dataset': object(), 'pred_len': 4,
            'season': 1, 'eval_inputs': [{'target': context}], 'info': {}}
    _score_prepared(RichHead(), prep, tmp_path, 8, normalize=False, batch_size=1)
    assert seen[0].shape == (1, 9, 2, 4)
    np.testing.assert_array_equal(seen[0][0], direct)


def test_fev_scores_q21_on_the_same_grid_as_q9():
    from test_fev_bench import _Oracle, _Task, _Window

    class RichOracle(_Oracle):
        quantile_levels = LEVELS

        def forecast_quantiles_batch(self, histories, horizon):
            q9 = super().forecast_quantiles_batch(histories, horizon)
            return np.repeat(q9[..., :1], 21, axis=-1)

    task = _Task([_Window([np.arange(40.)], [np.arange(40., 44.)])])
    assert score_task(task, RichOracle()) == score_task(task, _Oracle())
