#!/usr/bin/env python
"""Refresh the GIFT-Eval leaderboard reference the stakeholder scoreboard reads.

The scoreboard's hero ("how close are we to beating the best") sets the king's
GIFT-Eval score against a few headline models from the OFFICIAL GIFT-Eval
leaderboard (Hugging Face space ``Salesforce/GIFT-Eval``). No number on the page
is typed in: this script downloads every model's per-configuration results
(``results/<model>/all_results.csv``) at a pinned space revision and writes a
small, versioned reference file with two readings per model:

* ``leaderboard`` — the space's OWN overall numbers, recomputed exactly as its
  ``src/utils.get_grouped_dfs`` does: every model's 97 per-config metrics,
  NaNs filled with the column mean, divided by Seasonal-Naive's, then a plain
  geometric mean (MASE[0.5], CRPS = mean weighted sum quantile loss) and the
  mean per-config rank (``MASE_Rank`` / ``CRPS_Rank``, ``rank(method="first")``
  over the whole field). These are what a visitor sees on the leaderboard.
* ``cascade`` — the same per-config rows re-aggregated by OUR benchmark
  sidecar's ``official_aggregate`` (Seasonal-Naive-normalised shifted
  geometric mean, zero-MASE configs pooled separately, against the vendored
  baseline). This is the reading the king's signed ``gifteval_mase`` /
  ``gifteval_crps`` are computed with, so THIS pair is the apples-to-apples
  comparison the page draws. The two readings differ slightly by
  construction (shifted gmean over the non-zero pool vs a plain gmean over all
  configs); the page labels which one it shows.

Usage::

    python scripts/refresh_leaderboard_reference.py                 # write the repo file
    python scripts/refresh_leaderboard_reference.py --publish       # + upload public-read
    python scripts/refresh_leaderboard_reference.py --models TimesFM-3,Chronos-2

The output (``cascade/website/data/gift-eval-reference.json``) is checked in,
so every refresh is a reviewable diff; ``--publish`` (or
``scripts/publish_website.py``) uploads it to
``benchmarks/leaderboard-reference.json`` in the manifest bucket, where
``stakeholders.html`` reads it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SPACE = "Salesforce/GIFT-Eval"
SPACE_URL = f"https://huggingface.co/spaces/{SPACE}"
API = f"https://huggingface.co/api/spaces/{SPACE}"
OUT = REPO / "cascade" / "website" / "data" / "gift-eval-reference.json"
PUBLISHED_KEY = "benchmarks/leaderboard-reference.json"
SCHEMA = 1

# The headline set (owner 2026-10-10: "key models, e.g. TimesFM 3"): the newest
# TimesFM plus the most-recognised foundation-model families on the board.
DEFAULT_MODELS = ("Timestral-209M", "TimesFM-3", "Chronos-2", "TiRex", "Moirai2", "Toto-2.0-2.5B")

MASE = "eval_metrics/MASE[0.5]"
MAE = "eval_metrics/MAE[0.5]"
CRPS = "eval_metrics/mean_weighted_sum_quantile_loss"
METRICS = ("eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]", MAE, MASE,
           "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]", "eval_metrics/MSIS",
           "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]", CRPS)
# the space's own dataset-name corrections (src/utils.standardize_df)
CORRECTIONS = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
               "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}
_FREQ = {"T": "Minutely", "H": "Hourly", "D": "Daily", "W": "Weekly", "M": "Monthly",
         "Q": "Quarterly", "Y": "Yearly", "A": "Yearly", "S": "Secondly"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cascade-leaderboard-refresh"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 — fixed https host
        return r.read()


def space_revision() -> str:
    return str(json.loads(_get(API))["sha"])


def model_dirs(rev: str) -> list[str]:
    tree = json.loads(_get(f"{API}/tree/{rev}/results"))
    return sorted(e["path"].split("/", 1)[1] for e in tree if e.get("type") == "directory")


def _raw(rev: str, path: str) -> bytes:
    quoted = "/".join(urllib.parse.quote(p) for p in path.split("/"))
    return _get(f"{SPACE_URL}/resolve/{rev}/{quoted}")


def load_results(rev: str, models: list[str]) -> list[dict]:
    rows: list[dict] = []
    for m in models:
        try:
            text = _raw(rev, f"results/{m}/all_results.csv").decode("utf-8")
        except Exception as e:  # noqa: BLE001 — a dir without results is not a model
            print(f"  skip {m}: {e}", file=sys.stderr)
            continue
        rows.extend(csv.DictReader(io.StringIO(text)))
    return rows


def _unify_freq(f: str) -> str:
    f = "".join(c for c in f if not c.isdigit()).split("-")[0]
    return _FREQ.get(f, f)


def _float(v) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return math.nan
    return x


def leaderboard_overall(rows: list[dict]) -> dict[str, dict]:
    """The space's overall table (``get_grouped_dfs()['overall']``)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["model", "dataset"]).reset_index(drop=True)
    df[["dataset", "frequency", "term_length"]] = df["dataset"].str.split("/", expand=True)
    df["frequency"] = df["frequency"].map(_unify_freq)
    for m in METRICS:
        df[m] = df[m].map(_float).astype(float)
        df[m] = df[m].fillna(df[m].mean())
    df["dataset"] = df["dataset"].replace(CORRECTIONS)
    for (ds, term, freq), idx in df.groupby(["dataset", "term_length", "frequency"]).groups.items():
        sn = df.loc[idx][df.loc[idx]["model"] == "Seasonal_Naive"]
        if sn.empty:
            raise RuntimeError(f"no Seasonal_Naive row for {ds}/{freq}/{term}")
        for m in METRICS:
            df.loc[idx, m] = df.loc[idx, m] / sn[m].values[0]
    keys = ["dataset", "term_length", "frequency"]
    df["rank"] = df.groupby(keys)[CRPS].rank(method="first", ascending=True)
    df["Rank_MASE"] = df.groupby(keys)[MASE].rank(method="first", ascending=True)
    out: dict[str, dict] = {}
    for model, g in df.groupby("model"):
        out[str(model)] = {
            "mase": float(math.exp(g[MASE].map(math.log).mean())),
            "crps": float(math.exp(g[CRPS].map(math.log).mean())),
            "mase_rank": float(g["Rank_MASE"].mean()),
            "crps_rank": float(g["rank"].mean()),
            "n_configs": int(len(g)),
        }
    return out


def model_config(rev: str, model: str) -> dict:
    try:
        return json.loads(_raw(rev, f"results/{model}/config.json"))
    except Exception:  # noqa: BLE001 — config is descriptive only
        return {}


def cascade_aggregate(rows: list[dict], model: str) -> dict:
    """``model``'s rows through OUR sidecar's leaderboard aggregation."""
    sys.path.insert(0, str(REPO / "benchmarks"))
    from cascade_benchmark.aggregate import official_aggregate
    from cascade_benchmark.resources import load_json

    baseline = load_json("gifteval_seasonal_naive.json")
    mine = []
    for r in rows:
        if r["model"] != model:
            continue
        name, freq, term = r["dataset"].split("/")
        name = CORRECTIONS.get(name.lower(), name.lower())
        mine.append({"full": f"{name}/{freq}/{term}", "MASE": _float(r[MASE]),
                     "MAE": _float(r[MAE]), "CRPS": _float(r[CRPS])})
    agg = official_aggregate(mine, baseline)
    return {"mase": agg.get("mase"), "crps": agg.get("crps"),
            "n_scored": agg.get("n_scored"), "n_nonzero": agg.get("n_nonzero")}


def _field(rows: list[dict], models: list[str]) -> dict:
    aggs = [cascade_aggregate(rows, m) for m in models]
    return {k: sorted(round(float(a[k]), 6) for a in aggs
                      if a.get(k) is not None and math.isfinite(a[k])) for k in ("mase", "crps")}


def build(models: list[str]) -> dict:
    rev = space_revision()
    names = model_dirs(rev)
    print(f"{SPACE}@{rev}: {len(names)} model dirs", file=sys.stderr)
    rows = load_results(rev, names)
    board = leaderboard_overall(rows)
    configs = {m: model_config(rev, m) for m in board}
    # the leaderboard's default view: test-leak "No" entries first; positions
    # here count only those (agents/ensembles that saw test data are not a bar)
    clean = [m for m in board if configs[m].get("testdata_leakage") == "No"]
    ranked = sorted(clean, key=lambda m: board[m]["mase_rank"])
    entries = []
    for m in models:
        if m not in board:
            raise SystemExit(f"{m!r} is not on the leaderboard at {rev}")
        cfg = configs[m]
        entries.append({
            "model": m,
            "org": cfg.get("org", ""),
            "model_type": cfg.get("model_type", ""),
            "model_link": cfg.get("model_link", ""),
            "testdata_leakage": cfg.get("testdata_leakage", ""),
            "leaderboard": {**board[m],
                            "position": ranked.index(m) + 1 if m in ranked else None},
            "cascade": cascade_aggregate(rows, m),
        })
    return {
        "schema": SCHEMA,
        "benchmark": "GIFT-Eval",
        "source": SPACE_URL,
        "space_revision": rev,
        "retrieved_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "n_models": len(board),
        "n_no_leak": len(clean),
        # every no-leak entry's overall geomeans (cascade aggregation == the
        # space's on GIFT-Eval: no zero-MASE configs), sorted — the page places
        # the king in this field by MASE / CRPS geomean
        "field": _field(rows, clean),
        "readings": {
            "leaderboard": "the space's overall table: gmean over all configs of Seasonal-Naive-"
                           "normalised MASE[0.5] / CRPS, mean per-config rank (position = by "
                           "MASE_Rank among test-leak 'No' entries, the space's default sort)",
            "cascade": "the same per-config rows through cascade's benchmark sidecar "
                       "(official_aggregate: shifted gmean, zero-MASE configs pooled apart) — "
                       "the aggregation behind a checkpoint's signed gifteval_mase / gifteval_crps",
        },
        "models": entries,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="Comma-separated leaderboard model dirs (default: %(default)s).")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--publish", action="store_true",
                    help=f"Also upload public-read to {PUBLISHED_KEY} in the manifest bucket.")
    ap.add_argument("--chain-toml", type=Path, default=None)
    args = ap.parse_args()

    doc = build([m.strip() for m in args.models.split(",") if m.strip()])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for e in doc["models"]:
        lb, ca = e["leaderboard"], e["cascade"]
        print(f"{e['model']:>16}  #{lb['position']:<3} leaderboard MASE {lb['mase']:.4f} "
              f"CRPS {lb['crps']:.4f} · cascade MASE {ca['mase']:.4f} CRPS {ca['crps']:.4f}")
    print(f"wrote {args.out}")
    if args.publish:
        publish(doc, args.chain_toml)


def publish(doc: dict, chain_toml: Path | None = None) -> str:
    from cascade.shared.config import load_chain_config
    from cascade.shared.hippius import S3Config, S3Store, publish_public_json

    cfg = load_chain_config(chain_toml)
    store = S3Store(S3Config.from_storage(cfg.storage, bucket=cfg.storage.manifest_bucket))
    key = publish_public_json(store, PUBLISHED_KEY, doc)
    print(f"published → {cfg.storage.manifest_bucket}/{key}")
    return key


if __name__ == "__main__":
    main()
