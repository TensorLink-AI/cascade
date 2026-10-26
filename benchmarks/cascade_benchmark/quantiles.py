"""Select an exact benchmark grid from a richer head; never interpolate levels."""
from __future__ import annotations

import numpy as np


def quantile_indices(source, target):
    if source is None:
        return None
    try:
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if (source.ndim != 1 or target.ndim != 1 or not source.size or not target.size
            or not np.isfinite(source).all() or not np.isfinite(target).all()
            or np.any(source <= 0) or np.any(source >= 1) or np.any(np.diff(source) <= 0)):
        return None
    matches = np.isclose(target[:, None], source[None, :], rtol=0, atol=1e-6)
    if not np.all(matches.sum(1) == 1):
        return None
    return matches.argmax(1)


def forecast_on_grid(wrapper, histories, horizon, target):
    source = getattr(wrapper, "quantile_levels", None)
    indices = quantile_indices(source, target)
    if indices is None:
        raise ValueError("wrapper does not contain every required benchmark quantile")
    out = np.asarray(wrapper.forecast_quantiles_batch(histories, horizon), dtype=np.float64)
    expected = (len(histories), horizon, len(source))
    if out.shape != expected:
        raise ValueError(f"wrapper.forecast_quantiles_batch returned {out.shape}; expected {expected}")
    return out[..., indices]
