"""Frequency → seasonal period mapping, ported from gluonts.

Mirrors ``gluonts.time_feature.seasonality.get_seasonality`` so MASE
seasonality can be derived from an ``EvalWindow.metadata["freq"]`` pandas-style
frequency string (``"H"``, ``"D"``, ``"W"``, ...). Re-implemented rather than
imported to keep gluonts an optional dependency. Values match gluonts 0.15.x.
"""

from __future__ import annotations

_DEFAULT_SEASONALITIES: dict[str, int] = {
    "S": 3600,
    "min": 60,
    "T": 60,
    "H": 24,
    "D": 1,
    "W": 1,
    "M": 12,
    "ME": 12,
    "MS": 12,
    "Q": 4,
    "QE": 4,
    "QS": 4,
    "A": 1,
    "Y": 1,
    "YE": 1,
}


def _normalise(freq: str) -> str:
    """Strip a leading integer multiplier and trailing offset suffixes."""
    f = freq.strip()
    i = 0
    while i < len(f) and (f[i].isdigit() or f[i] in "-+"):
        i += 1
    base = f[i:] or f
    base = base.split("-", 1)[0]
    return base


def get_seasonality(freq: str, default: int = 1) -> int:
    """Return the canonical seasonal period for a pandas-style freq string.

    ``default`` (1) degenerates MASE to the random-walk-scaled MAE — the same
    convention gluonts uses for unknown frequencies.
    """
    base = _normalise(freq)
    return _DEFAULT_SEASONALITIES.get(base, default)
