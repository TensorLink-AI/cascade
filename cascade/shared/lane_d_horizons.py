"""Lane D per-window horizon bands (GIFT-Eval terms; owner 2026-11-27).

Every lane D window is scored at its own horizon ``h``, drawn per window
from the GIFT-Eval-style bands of its series' frequency:

* **Base horizon** per frequency, GIFT-Eval's own tables
  (``SalesforceAIResearch/gift-eval`` ``src/gift_eval/data.py`` at
  ``1b520f911c24e3f45d990a6dc966fadab2ba475b``): ``PRED_LENGTH_MAP`` =
  ``{"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}``; GIFT-Eval's
  quarterly and yearly series are M4's, so ``Q`` / ``A`` take
  ``M4_PRED_LENGTH_MAP``'s ``8`` / ``6``.
* **Terms**: ``Term.multiplier`` (same file): ``short = 1``, ``medium = 10``,
  ``long = 15``; ``Dataset.prediction_length = multiplier * base``.
* **Patch alignment**: ``h = ceil(multiplier * base / 32) * 32`` (at least one
  32-point patch), capped at ``max_horizon`` (2048). Every band horizon is a
  whole number of model patches, so the masked span is never padded.

The frequency is normalised the way GIFT-Eval does it
(``norm_freq_str(to_offset(freq).name)``: the multiple and the anchor are
dropped, ``5min`` -> ``T``, ``W-SUN`` -> ``W``), with one addition so a
pool written in seconds keeps its real cadence: a tick frequency (seconds,
minutes, hours) is first re-expressed in its largest exact unit
(``21600S`` = ``6H`` -> ``H``; ``150S`` stays ``S``; ``240S`` = ``4min`` ->
``T``). Sub-second frequencies have no band.

Light imports only (stdlib): the validator, the pool builder and the audit
all import this module.
"""

from __future__ import annotations

import math
import re

# GIFT-Eval PRED_LENGTH_MAP (+ M4_PRED_LENGTH_MAP for Q / A), by base unit.
GIFT_BASE_HORIZONS: dict[str, int] = {"S": 60, "T": 48, "H": 48, "D": 30, "W": 8,
                                      "M": 12, "Q": 8, "A": 6}
# GIFT-Eval Term.multiplier.
GIFT_TERM_MULTIPLIERS: dict[str, int] = {"short": 1, "medium": 10, "long": 15}
BANDS: tuple[str, ...] = ("short", "medium", "long")
PATCH_POINTS = 32
MAX_HORIZON_CAP = 2048
GIFT_SOURCE = ("SalesforceAIResearch/gift-eval src/gift_eval/data.py "
               "@1b520f911c24e3f45d990a6dc966fadab2ba475b")

# The equal-weight cells of the combined statistic: (frequency class, band).
CELL_CLASS = {"S": "subdaily", "T": "subdaily", "H": "subdaily", "D": "daily",
              "W": "weekly", "M": "monthly", "Q": "quarterly", "A": "yearly"}

_UNITS = {
    "S": "S", "s": "S", "sec": "S", "Sec": "S",
    "T": "T", "min": "T", "Min": "T",
    "H": "H", "h": "H",
    "D": "D", "d": "D", "B": "D", "C": "D",
    "W": "W", "w": "W",
    "M": "M", "ME": "M", "MS": "M", "BM": "M", "BME": "M", "BMS": "M", "SM": "M", "SMS": "M",
    "Q": "Q", "QE": "Q", "QS": "Q", "BQ": "Q", "BQE": "Q", "BQS": "Q",
    "A": "A", "Y": "A", "YE": "A", "YS": "A", "AS": "A", "BA": "A", "BY": "A", "BYE": "A",
    "BYS": "A", "BAS": "A",
}
_TICK_SECONDS = {"S": 1, "T": 60, "H": 3600}
_FREQ_RE = re.compile(r"^\s*(\d*)\s*([A-Za-z]+)(?:-[A-Za-z]+)?\s*$")


def gift_base_freq(freq) -> str | None:
    """GIFT-Eval's base unit (``S T H D W M Q A``) of a pandas-style
    frequency string, or ``None`` (unknown or sub-second)."""
    m = _FREQ_RE.match(str(freq or ""))
    if not m:
        return None
    mult = int(m.group(1)) if m.group(1) else 1
    unit = _UNITS.get(m.group(2))
    if unit is None or mult < 1:
        return None
    if unit in _TICK_SECONDS:
        secs = mult * _TICK_SECONDS[unit]
        unit = "H" if secs % 3600 == 0 else "T" if secs % 60 == 0 else "S"
    return unit


def cell_class(freq) -> str:
    """The frequency class of a window's equal-weight cell."""
    return CELL_CLASS.get(gift_base_freq(freq) or "", "other")


def band_horizon(base: str, band: str, *, max_horizon: int = MAX_HORIZON_CAP,
                 patch: int = PATCH_POINTS) -> int:
    """``min(max_horizon, max(patch, ceil(multiplier * base_h / patch) * patch))``."""
    raw = GIFT_TERM_MULTIPLIERS[band] * GIFT_BASE_HORIZONS[base]
    return int(min(int(max_horizon), max(int(patch), math.ceil(raw / patch) * int(patch))))


def bands_for(freq, bands, *, max_horizon: int = MAX_HORIZON_CAP) -> dict[str, int]:
    """``{band: h}`` of ``freq`` over the enabled ``bands`` (in :data:`BANDS`
    order); ``{}`` for a frequency without a base horizon."""
    base = gift_base_freq(freq)
    if base is None:
        return {}
    return {b: band_horizon(base, b, max_horizon=max_horizon) for b in BANDS if b in set(bands)}


def band_table(bands, *, max_horizon: int = MAX_HORIZON_CAP) -> dict[str, dict[str, int]]:
    """``{base unit: {band: h}}``: the signed report's record of the rule."""
    return {u: {b: band_horizon(u, b, max_horizon=max_horizon) for b in BANDS if b in set(bands)}
            for u in GIFT_BASE_HORIZONS}


def band_slots(ts, freq, bands, *, slots: int, max_horizon: int = MAX_HORIZON_CAP
               ) -> list[tuple[str, int, int]]:
    """The forward candidate windows of one series' stored target region
    (``ts``, its data times, oldest first): per band whose ``h`` fits, up to
    ``slots`` non-overlapping targets tiled back from the newest point,
    ``(band, h, start index)``. A window's history is everything before its
    start (embargo-held points included: only the target must be unseen)."""
    n = len(ts)
    out = []
    for band, h in bands_for(freq, bands, max_horizon=max_horizon).items():
        for j in range(int(slots)):
            start = n - (j + 1) * h
            if start < 0:
                break
            out.append((band, h, start))
    return out


def cell_key(half: str, freq, band: str) -> str:
    """``"<half>|<frequency class>|<band>"`` (the verdict ``cells`` keys)."""
    return f"{half}|{cell_class(freq)}|{band}"


def horizons_record(cfg) -> dict:
    """The signed report's optional ``horizon_bands`` block (an extra the
    pre-band checker never reads): the rule every window's ``h`` follows."""
    bands = tuple(cfg.horizon_bands)
    return {"schema": "lane-d-horizons/1", "source": GIFT_SOURCE,
            "bands": [b for b in BANDS if b in set(bands)],
            "multipliers": {b: GIFT_TERM_MULTIPLIERS[b] for b in BANDS if b in set(bands)},
            "base_horizons": dict(GIFT_BASE_HORIZONS), "patch": PATCH_POINTS,
            "max_horizon": int(cfg.max_horizon), "band_slots": int(cfg.band_slots),
            "min_cell_windows": int(cfg.min_cell_windows),
            "table": band_table(bands, max_horizon=int(cfg.max_horizon))}


def bands_on(cfg) -> bool:
    """Per-window horizons are on (``[lane_d] horizon_bands`` non-empty);
    off is the fixed ``[lane_d] horizon`` rule, byte for byte."""
    return bool(tuple(getattr(cfg, "horizon_bands", ()) or ()))


__all__ = ["BANDS", "CELL_CLASS", "GIFT_BASE_HORIZONS", "GIFT_SOURCE", "GIFT_TERM_MULTIPLIERS",
           "MAX_HORIZON_CAP", "PATCH_POINTS", "band_horizon", "band_slots", "band_table",
           "bands_for", "bands_on", "cell_class", "cell_key", "gift_base_freq",
           "horizons_record"]
