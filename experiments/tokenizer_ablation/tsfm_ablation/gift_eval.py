# ============================================================================
# GIFT-EVAL  ·  dev-subset harness (official evaluate_model path, CRPS ==
# leaderboard mean_weighted_sum_quantile_loss). DEV_SETS is the FIXED design
# subset — do NOT grow it while iterating on tokenizers, or you Goodhart the
# choice; the remaining ~85 tasks stay held out for the promoted winner.
# Unresolvable names are skipped with a warning (GIFT names occasionally
# shift between releases), so the harness degrades gracefully.
# ============================================================================
"""GIFT-Eval dev-subset harness + a cheap pre-flight wiring test.

Imports gluonts / gift_eval lazily inside the functions so the rest of the
package stays importable without the (numpy<2-pinned) eval stack installed.
"""
from __future__ import annotations

import os

import numpy as np

from .infer import batched_quantiles

QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# (name, term) — stratified across frequency; the 10T/15T/H entries form the
# "long-seasonality" slice where the iso-token pyramid (T2) should show its
# advantage if the long-history hypothesis is real. This is the FIXED design
# subset for iterating on tokenizers — do NOT grow it, or you Goodhart the
# choice; the full 97-config leaderboard (GIFT_ALL) stays held out for the
# promoted winner.
DEV_SETS = [
    ("m4_weekly", "short"), ("m4_daily", "short"), ("m4_hourly", "short"),
    ("m4_monthly", "short"),
    ("electricity/15T", "short"), ("electricity/H", "short"),
    ("solar/10T", "short"), ("solar/H", "short"),
    ("ett1/15T", "short"), ("ett2/H", "short"),
    ("jena_weather/10T", "short"),
    ("us_births/D", "short"), ("covid_deaths", "short"), ("hospital", "short"),
]
LONG_SEASONALITY_FREQS = {"10T", "15T", "H", "h", "10min", "15min"}

# The full GIFT-Eval enumeration (97 configs), verbatim from gift-eval's
# reference runner (notebooks/naive.ipynb): SHORT terms for every dataset, plus
# medium + long terms for the subset with long-enough series. Used by the
# `gift_all_specs()` full-leaderboard eval; the horizon-past-tail configs are
# handled by the block rollout in infer.batched_quantiles.
_SHORT_DATASETS = (
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
_MED_LONG_DATASETS = (
    "electricity/15T electricity/H solar/10T solar/H "
    "kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T "
    "M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H "
    "bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application "
    "bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
)


def gift_all_specs():
    """The full 97-config GIFT-Eval task list as ``(name, term)`` pairs."""
    med_long = set(_MED_LONG_DATASETS.split())
    specs = []
    for name in _SHORT_DATASETS.split():
        for term in ("short", "medium", "long"):
            if term in ("medium", "long") and name not in med_long:
                continue
            specs.append((name, term))
    return specs


def _download_dev_sets(gift_eval_dir, dev_sets=DEV_SETS):
    from huggingface_hub import snapshot_download
    pats = sorted({n.split("/")[0] for n, _ in dev_sets})
    snapshot_download("Salesforce/GiftEval", repo_type="dataset",
                      local_dir=gift_eval_dir, token=os.environ.get("HF_TOKEN") or None,
                      allow_patterns=[f"{p}/*" for p in pats] + [f"{p}*" for p in pats])
    os.environ["GIFT_EVAL"] = gift_eval_dir


class TSFMPredictor:
    """gluonts predictor around MiniTSFM2: forecasts each series' quantiles via
    the shared ``infer.batched_quantiles`` core (ragged-aware, block-rollout for
    horizons past the trained tail)."""

    def __init__(self, model, H, batch_size=256):
        self.model, self.H, self.bs = model, H, batch_size
        self.prediction_length = H

    def predict(self, dataset, **kwargs):
        from gluonts.model.forecast import QuantileForecast
        entries = list(dataset)
        targets = [np.asarray(e["target"], np.float32).reshape(-1) for e in entries]
        q = batched_quantiles(self.model, targets, self.H, self.bs)   # [N, H, Q]
        for j, e in enumerate(entries):
            yield QuantileForecast(
                forecast_arrays=q[j].T, forecast_keys=[str(p) for p in QL],
                start_date=e["start"] + len(np.atleast_1d(e["target"])),
                item_id=e.get("item_id"))


def eval_gift_dev(model, gift_eval_dir, dev_sets=DEV_SETS, tag=""):
    """Returns {dataset: {CRPS, MASE, freq}} + geometric-mean CRPS + slices."""
    from gift_eval.data import Dataset as GEDataset
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality
    _download_dev_sets(gift_eval_dir, dev_sets)
    model.eval()
    rows = {}
    for name, term in dev_sets:
        try:
            to_uni = GEDataset(name=name, term=term).target_dim != 1
            ds = GEDataset(name=name, term=term, to_univariate=to_uni)
            season = get_seasonality(ds.freq)
            res = evaluate_model(
                TSFMPredictor(model, ds.prediction_length), test_data=ds.test_data,
                metrics=[MASE(), MeanWeightedSumQuantileLoss(quantile_levels=QL)],
                batch_size=512, axis=None, mask_invalid_label=True,
                allow_nan_forecast=False, seasonality=season)
            rows[name] = {"CRPS": float(res["mean_weighted_sum_quantile_loss"].iloc[0]),
                          "MASE": float(res["MASE[0.5]"].iloc[0]), "freq": ds.freq}
            print(f"  {tag} {name:22s} CRPS={rows[name]['CRPS']:.4f} MASE={rows[name]['MASE']:.3f}")
        except Exception as ex:
            print(f"  {tag} {name:22s} SKIPPED ({type(ex).__name__}: {ex})")
    crps = [r["CRPS"] for r in rows.values() if np.isfinite(r["CRPS"]) and r["CRPS"] > 0]
    out = {"per_dataset": rows,
           "gm_crps": float(np.exp(np.mean(np.log(crps)))) if crps else float("nan")}
    ls = [r["CRPS"] for r in rows.values()
          if r["freq"] in LONG_SEASONALITY_FREQS and np.isfinite(r["CRPS"]) and r["CRPS"] > 0]
    ss = [r["CRPS"] for r in rows.values()
          if r["freq"] not in LONG_SEASONALITY_FREQS and np.isfinite(r["CRPS"]) and r["CRPS"] > 0]
    out["gm_crps_long_season"] = float(np.exp(np.mean(np.log(ls)))) if ls else float("nan")
    out["gm_crps_other"] = float(np.exp(np.mean(np.log(ss)))) if ss else float("nan")
    print(f"{tag} GM-CRPS={out['gm_crps']:.4f}  long-season={out['gm_crps_long_season']:.4f} "
          f" other={out['gm_crps_other']:.4f}  ({len(rows)}/{len(dev_sets)} datasets)")
    return out


def eval_gift_full(model, gift_eval_dir, tag=""):
    """Full 97-config GIFT-Eval leaderboard sweep. Same scoring as the dev
    subset, over every task; downloads the full benchmark on first run."""
    return eval_gift_dev(model, gift_eval_dir, dev_sets=gift_all_specs(), tag=tag)


def quick_wiring_test(gift_eval_dir, name="m4_weekly", term="short"):
    """Seasonal-naive on one dataset — surfaces env/token problems in seconds,
    before an A100-day. Returns True on PASS. No model involved."""
    from gift_eval.data import Dataset as GEDataset
    from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
    from gluonts.model import evaluate_model
    from gluonts.model.forecast import QuantileForecast
    from gluonts.model.predictor import RepresentablePredictor
    from gluonts.time_feature import get_seasonality
    from huggingface_hub import snapshot_download
    from scipy.stats import norm

    assert np.__version__.startswith("1."), "numpy is 2.x — the gift-eval env needs numpy<2."

    if not os.path.exists(os.path.join(gift_eval_dir, name)):
        print(f"[quick] downloading only {name} (first run)...")
        snapshot_download("Salesforce/GiftEval", repo_type="dataset",
                          local_dir=gift_eval_dir, token=os.environ.get("HF_TOKEN") or None,
                          allow_patterns=[f"{name}/*", f"{name}*"])
    os.environ["GIFT_EVAL"] = gift_eval_dir

    class _SNaive(RepresentablePredictor):
        def __init__(self, H, m):
            super().__init__(prediction_length=H); self.m = max(1, m)

        def predict(self, dataset, **kwargs):
            H, m = self.prediction_length, self.m
            for e in dataset:
                y = np.asarray(e["target"], np.float32)
                base = (np.tile(y[-m:], int(np.ceil(H / m)))[:H] if len(y) >= m
                        else np.full(H, y[-1] if len(y) else 0.0))
                r = float(np.std(np.diff(y))) if len(y) > 1 else 1.0
                q = np.stack([base + norm.ppf(p) * r for p in QL], axis=0)
                yield QuantileForecast(
                    forecast_arrays=q, forecast_keys=[str(p) for p in QL],
                    start_date=e["start"] + len(e["target"]), item_id=e.get("item_id"))

    to_uni = GEDataset(name=name, term=term).target_dim != 1
    ds = GEDataset(name=name, term=term, to_univariate=to_uni)
    season = get_seasonality(ds.freq)
    res = evaluate_model(
        _SNaive(ds.prediction_length, season), test_data=ds.test_data,
        metrics=[MASE(), MeanWeightedSumQuantileLoss(quantile_levels=QL)],
        batch_size=512, axis=None, mask_invalid_label=True,
        allow_nan_forecast=False, seasonality=season)
    print(f"PASS  {name}/{term}  freq={ds.freq}  H={ds.prediction_length}  "
          f"MASE={float(res['MASE[0.5]'].iloc[0]):.3f}  "
          f"CRPS={float(res['mean_weighted_sum_quantile_loss'].iloc[0]):.3f}")
    return True
