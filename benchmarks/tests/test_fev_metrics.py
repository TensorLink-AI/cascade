"""Invalid forecasts cannot be hidden by masking unusable benchmark tasks."""

import numpy as np
import pytest
from cascade_benchmark.fev_metrics import TaskAccumulator, normalised_geomean


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_prediction_on_observed_target_is_a_checkpoint_fault(bad):
    q = np.ones((4, 9))
    q[1, 0] = bad
    with pytest.raises(ValueError, match="non-finite"):
        TaskAccumulator().add(np.arange(20.), np.arange(4.), q, 1)


def test_missing_target_can_mask_its_forecast():
    q = np.ones((4, 9))
    q[1] = np.nan
    acc = TaskAccumulator()
    acc.add(np.arange(20.), np.array([0., np.nan, 2., 3.]), q, 1)
    assert all(np.isfinite(x) for x in acc.result())


def test_one_quantile_cannot_broadcast_over_the_nine_level_grid():
    with pytest.raises(ValueError, match="shape"):
        TaskAccumulator().add(np.arange(20.), np.arange(4.), np.ones((4, 1)), 1)


def test_misaligned_task_lists_cannot_silently_truncate():
    with pytest.raises(ValueError):
        normalised_geomean([1., 100.], [1.])
