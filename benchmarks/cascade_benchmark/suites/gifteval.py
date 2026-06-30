"""GIFT-Eval runner — Salesforce's general TS benchmark (97 configs).

gift-eval exposes no importable dataset list — its reference runner
(``notebooks/naive.ipynb``) hardcodes two space-separated strings and derives
terms from them. We embed those verbatim and replicate the term logic, the
per-config scoring (``_common.score_dataset``), and the official aggregation:
the Seasonal-Naive-normalized shifted geometric mean
(``cascade_benchmark.aggregate``), normalized against the vendored official
Seasonal-Naive results — so the headline numbers match the leaderboard.

The baseline-key construction (``name/freq/term``) reproduces ``naive.ipynb``
exactly (pretty-name remap + frequency from dataset_properties) — verified to
match all 97 official baseline keys.

Set ``GIFT_EVAL`` to the downloaded benchmark data.
``CASCADE_BENCH_GIFTEVAL_DATASETS`` restricts the config list.
"""

from __future__ import annotations

import os
import traceback

from ..aggregate import official_aggregate
from ..resources import load_json
from ..results import SuiteResult
from ._common import build_dataset, score_dataset

# Verbatim from gift-eval notebooks/naive.ipynb @ 1527c415 (the full lists; the
# notebook ships them commented out for a fast 2-dataset demo).
SHORT_DATASETS = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing kdd_cup_2018_with_missing/H "
    "kdd_cup_2018_with_missing/D car_parts_with_missing restaurant "
    "hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H "
    "LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W "
    "jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T "
    "bitbrains_rnd/H bizitobs_application bizitobs_service "
    "bizitobs_l2c/5T bizitobs_l2c/H"
)
MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T "
    "M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application "
    "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
)
# naive.ipynb pretty_names map (dataset dir name → baseline-key name).
_PRETTY = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def _baseline_key(ds_name: str, term: str, properties: dict) -> str:
    """Reproduce naive.ipynb's ``ds_config`` = ``{ds_key}/{ds_freq}/{term}``."""
    if "/" in ds_name:
        key, freq = ds_name.split("/")
        key = _PRETTY.get(key.lower(), key.lower())
    else:
        key = _PRETTY.get(ds_name.lower(), ds_name.lower())
        freq = properties[key]["frequency"]
    return f"{key}/{freq}/{term}"


def _name_term_pairs(max_tasks: int | None):
    override = os.environ.get("CASCADE_BENCH_GIFTEVAL_DATASETS", "").strip()
    names = override.replace(",", " ").split() if override else SHORT_DATASETS.split()
    med_long = set(MED_LONG_DATASETS.split())
    n = 0
    for ds_name in names:
        for term in ("short", "medium", "long"):
            if term in ("medium", "long") and ds_name not in med_long:
                continue
            yield ds_name, term
            n += 1
            if max_tasks and n >= max_tasks:
                return


def run(
    checkpoint_dir: str,
    *,
    num_samples: int = 100,
    max_series: int | None = None,
    device: str = "cpu",
) -> SuiteResult:
    if not (os.environ.get("GIFT_EVAL") or os.environ.get("CASCADE_BENCH_GIFTEVAL_DATASETS")):
        return SuiteResult(
            suite="gift-eval",
            status="skipped",
            detail="GIFT_EVAL not set; point it at the downloaded gift-eval benchmark data.",
        )
    try:
        properties = load_json("gifteval_dataset_properties.json")
        baseline = load_json("gifteval_seasonal_naive.json")
        rows = []
        for ds_name, term in _name_term_pairs(max_series):
            ds = build_dataset(ds_name, term)
            if ds is None:
                continue
            try:
                m = score_dataset(ds, checkpoint_dir, num_samples=num_samples, device=device)
            except Exception:  # noqa: BLE001 — one dataset must not abort the sweep
                continue
            rows.append({"full": _baseline_key(ds_name, term, properties), **m})

        if not rows:
            return SuiteResult(suite="gift-eval", status="error", detail="no datasets scored")
        agg = official_aggregate(rows, baseline)
        metrics = {k: agg[k] for k in ("crps", "mase", "crps_zero", "mae_zero") if k in agg}
        return SuiteResult(suite="gift-eval", status="ok", metrics=metrics, n_series=agg["n_scored"])
    except ImportError as e:
        return SuiteResult(suite="gift-eval", status="skipped", detail=f"gift-eval not importable: {e}")
    except Exception as e:  # noqa: BLE001
        return SuiteResult(
            suite="gift-eval",
            status="error",
            detail=f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}",
        )
