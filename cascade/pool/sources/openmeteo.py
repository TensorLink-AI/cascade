"""Open-Meteo weather source — keyless, free, hourly, global.

Uses the historical archive API (https://open-meteo.com/en/docs/historical-weather-api),
which needs no API key. Each (location, variable) pair becomes one hourly series
with strong daily + yearly seasonality — the backbone domain of the pool, long
enough to fill the full ``context_length`` at hourly frequency.

Gaming resistance comes from harvesting *recent* windows and re-building
periodically: tomorrow's weather is not knowable at generator-submission time,
so the rotated pool cannot be memorised or distribution-matched in detail.

Locations default to a deterministic **global grid** (:func:`global_grid`) so the
pool scales to thousands of series without a hand-maintained list — one API call
per grid point returns all :data:`VARIABLES`. The default grid (~252 points) ×
12 variables is ~3000 raw series, comfortably above ``[eval] n_windows`` after
validation drops. Tune the grid steps (denser = more series) or roll ``as_of``.
A curated, named :data:`CITIES` list is also provided for callers that prefer it.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..source import FetchJson, HarvestContext, HarvestedSeries, daterange

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Continuous variables defined over both land and ocean (ERA5 reanalysis), so a
# global grid yields usable series everywhere. Sparse/zero-heavy fields
# (precipitation) and land-only fields (soil_*) are omitted — they would be
# dropped as degenerate / mostly-missing by the builder anyway.
VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "surface_pressure",
    "pressure_msl",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_gusts_10m",
)

# (label, latitude, longitude) — a curated, named seed across climates/continents.
CITIES: tuple[tuple[str, float, float], ...] = (
    ("tokyo", 35.69, 139.69), ("delhi", 28.61, 77.21), ("shanghai", 31.23, 121.47),
    ("sao_paulo", -23.55, -46.63), ("mexico_city", 19.43, -99.13), ("cairo", 30.04, 31.24),
    ("mumbai", 19.08, 72.88), ("beijing", 39.90, 116.41), ("dhaka", 23.81, 90.41),
    ("osaka", 34.69, 135.50), ("new_york", 40.71, -74.01), ("karachi", 24.86, 67.01),
    ("buenos_aires", -34.60, -58.38), ("istanbul", 41.01, 28.98), ("kolkata", 22.57, 88.36),
    ("lagos", 6.52, 3.38), ("manila", 14.60, 120.98), ("rio_de_janeiro", -22.91, -43.17),
    ("london", 51.51, -0.13), ("paris", 48.85, 2.35), ("moscow", 55.76, 37.62),
    ("jakarta", -6.21, 106.85), ("seoul", 37.57, 126.98), ("lima", -12.05, -77.04),
    ("bangkok", 13.76, 100.50), ("nairobi", -1.29, 36.82), ("johannesburg", -26.20, 28.05),
    ("toronto", 43.65, -79.38), ("sydney", -33.87, 151.21), ("madrid", 40.42, -3.70),
    ("chicago", 41.88, -87.63), ("los_angeles", 34.05, -118.24), ("tehran", 35.69, 51.39),
    ("bogota", 4.71, -74.07), ("santiago", -33.45, -70.67), ("reykjavik", 64.15, -21.94),
    ("singapore", 1.35, 103.82), ("anchorage", 61.22, -149.90), ("perth", -31.95, 115.86),
    ("vancouver", 49.28, -123.12),
)


def global_grid(lat_step: int = 12, lon_step: int = 17) -> tuple[tuple[str, float, float], ...]:
    """A deterministic lat/lon grid (60S–72N, global longitude).

    ~252 points at the defaults; smaller steps give a denser grid (more series).
    Ocean points are fine — atmospheric reanalysis is defined there too."""
    pts: list[tuple[str, float, float]] = []
    for lat in range(-60, 73, lat_step):
        for lon in range(-175, 176, lon_step):
            pts.append((f"grid_{lat}_{lon}", float(lat), float(lon)))
    return tuple(pts)


# Default to the global grid so a no-arg build clears n_windows = 2000.
LOCATIONS: tuple[tuple[str, float, float], ...] = global_grid()


class OpenMeteoSource:
    name = "openmeteo"

    def __init__(
        self,
        locations: tuple[tuple[str, float, float], ...] = LOCATIONS,
        variables: tuple[str, ...] = VARIABLES,
    ) -> None:
        self.locations = locations
        self.variables = variables

    def harvest(self, fetch: FetchJson, ctx: HarvestContext) -> Iterable[HarvestedSeries]:
        # Archive lags ~5 days; bias the window back so the request resolves.
        start, end = daterange(ctx.as_of, max(ctx.span_days, ctx.context_length // 24 + 14))
        emitted = 0
        for label, lat, lon in self.locations:
            if emitted >= ctx.max_series:
                return
            params = {
                "latitude": lat,
                "longitude": lon,
                "start_date": start,
                "end_date": end,
                "hourly": ",".join(self.variables),
                "timezone": "UTC",
            }
            data = fetch(ARCHIVE_URL, params)
            hourly = (data or {}).get("hourly") or {}
            times = hourly.get("time") or []
            if not times:
                continue
            for var in self.variables:
                if emitted >= ctx.max_series:
                    return
                raw = hourly.get(var)
                if not raw:
                    continue
                values = np.array([np.nan if v is None else v for v in raw], dtype=np.float64)
                yield HarvestedSeries(
                    series_id=f"openmeteo__{label}__{var}",
                    values=values,
                    freq="H",
                    domain="weather",
                    seasonal_period=24,
                    attrs={"lat": lat, "lon": lon, "variable": var},
                )
                emitted += 1
