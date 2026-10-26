"""The core CI verifies the isolated sidecar's Q21 -> benchmark-grid boundary."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

from cascade.trainer.pyramid_model import QUANTILE_LEVELS

MODULE = Path(__file__).resolve().parents[2] / 'benchmarks/cascade_benchmark/quantiles.py'
SPEC = importlib.util.spec_from_file_location('benchmark_quantile_selection', MODULE)
selection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selection)


def test_public_metric_columns_are_exact_deciles_from_the_21_level_head():
    target = tuple(i / 10 for i in range(1, 10))
    indices = selection.quantile_indices(QUANTILE_LEVELS, target)
    np.testing.assert_array_equal(indices, np.arange(2, 20, 2))

    class Wrapper:
        quantile_levels = QUANTILE_LEVELS

        def forecast_quantiles_batch(self, histories, horizon):
            return np.broadcast_to(np.arange(21), (len(histories), horizon, 21))

    result = selection.forecast_on_grid(Wrapper(), [np.zeros(64), np.ones(64)], 8, target)
    assert result.shape == (2, 8, 9)
    np.testing.assert_array_equal(result[0, 0], np.arange(2, 20, 2))


def test_incomplete_grid_and_wrong_head_width_cannot_silently_pass():
    class Wrapper:
        quantile_levels = QUANTILE_LEVELS

        def forecast_quantiles_batch(self, histories, horizon):
            return np.zeros((len(histories), horizon, 9))

    with pytest.raises(ValueError, match='expected'):
        selection.forecast_on_grid(Wrapper(), [np.zeros(64)], 8, (.1, .5, .9))
    with pytest.raises(ValueError, match='every required'):
        selection.forecast_on_grid(Wrapper(), [np.zeros(64)], 8, (.025, .5, .975))
