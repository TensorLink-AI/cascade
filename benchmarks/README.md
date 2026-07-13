# cascade-benchmark (sidecar)

Scores a trained cascade checkpoint on the public time-series benchmarks
**GIFT-Eval**, **BOOM**, and **TIME**, and writes the numbers as JSON. These are
**log-only** — they never feed miner scores, weights, or KOTH state. The
validator runs this out-of-process and just logs whatever JSON comes back.

## Why a separate project / environment

`gift-eval` (the harness behind both GIFT-Eval and BOOM) hard-pins
`numpy~=1.26`, `scipy~=1.11`, `datasets~=2.17`, `gluonts~=0.15`, `pandas==2.0.0`
and needs Python 3.11. Those caps **cannot coexist** in the cascade env with
`torch>=2.2` / `transformers>=4.40` / `bittensor` / the `hippius` extra — every
resolver upgrade re-breaks the install. So this lives in its **own locked
environment** and is invoked as a subprocess. The cascade core stays
`numpy`/`scipy`-only.

The boundary is dead simple: **a checkpoint dir in, a results JSON out.** Every
cascade checkpoint already ships `forecast_wrapper.py` (the same trusted
inference path the validator scores on), which we wrap as a gluonts predictor —
so this is model-agnostic and consistent with in-protocol scores.

CPM-era checkpoints expose their quantile head
(`forecast_quantiles_batch(histories, horizon)`), so the scorer emits gluonts
`QuantileForecast`s straight from one batched forward pass per `--batch-size`
series — no Monte-Carlo sampling. Older checkpoints that only implement the
sampling contract still work via the original per-series `SampleForecast` path
(slow; expect the historical hours-long runtimes for those).

## Docker (recommended): env + frozen data as one artifact

`benchmarks/Dockerfile` bakes the locked env **and** the three datasets at the
revisions pinned in `cascade_benchmark/datasets.py`, so any host with docker
(and optionally a GPU) scores checkpoints with zero setup and comparable
numbers:

```bash
# build (from benchmarks/); the data layer only rebuilds when revisions bump
docker build -t cascade-bench:<tag> .

# score a checkpoint (data is baked at /data; env vars pre-wired)
docker run --rm --gpus all \
    -v /path/to/checkpoint:/ckpt:ro -v /path/to/out:/out \
    cascade-bench:<tag> /ckpt /out/report.json --device cuda --batch-size 512
```

Every report records `data_revisions`, so historical numbers stay traceable to
the exact data they were computed against.

## Setup (uv)

```bash
# from the repo root — resolves in total isolation from the main env
uv sync --project benchmarks
```

## Run

```bash
uv run --project benchmarks cascade-benchmark \
    /path/to/checkpoint_dir /path/to/out.json \
    --suites gift-eval,boom,time --num-samples 100
```

Fast smoke run on a subset (avoids downloading/scoring the full benchmarks):

```bash
CASCADE_BENCH_GIFTEVAL_DATASETS="electricity/short" \
uv run --project benchmarks cascade-benchmark CKPT out.json --suites gift-eval --max-series 50
```

### Getting the data (auto-download)

The three benchmarks live in HuggingFace dataset repos (`Salesforce/GiftEval`,
`Datadog/BOOM`, `Real-TSF/TIME`). Fetch them all — or run the scorer with
`--data-dir` and let it download what's missing:

```bash
# download the full datasets once (set HF_TOKEN if a repo is gated)
uv run --project benchmarks cascade-benchmark-download --data-dir ./bench_data

# ...or fetch-and-score in one shot (each suite reads <data-dir>/<suite>)
uv run --project benchmarks cascade-benchmark CKPT out.json \
    --suites gift-eval,boom,time --data-dir ./bench_data --download
```

`--data-dir` (without `--download`) just wires `GIFT_EVAL` / `BOOM` /
`CASCADE_BENCH_TIME_DATASET` to `<data-dir>/<suite>` for data already present.
Downloads are resumable. BOOM is ~350M obs / large — expect a big pull for a full
run. The manual env vars below still work if you manage the data yourself.

### Datasets / env vars

- `GIFT_EVAL` — **required for GIFT-Eval.** Path to the downloaded gift-eval
  benchmark data (gift-eval's own env var). `CASCADE_BENCH_GIFTEVAL_DATASETS`
  (comma-separated `name` or `name/freq`) restricts the config list.
- `BOOM` / `CASCADE_BENCH_BOOM_PATH` — **required for BOOM.** Path to the
  downloaded [`Datadog/BOOM`](https://huggingface.co/datasets/Datadog/BOOM) data.
  `CASCADE_BENCH_BOOM_DATASETS` restricts the config list;
  `CASCADE_BENCH_BOOM_PROPERTIES` overrides the vendored manifest.
- `CASCADE_BENCH_TIME_DATASET` (or `TIME_DATASET`) — **required to enable TIME.**
  Path to the [`Real-TSF/TIME`](https://huggingface.co/datasets/Real-TSF/TIME)
  data. `CASCADE_BENCH_TIME_DATASETS` optionally restricts the `name/freq`
  configs (default: all of TIME's bundled config). Without it the `time` suite
  reports `skipped`.

## Output shape

```json
{
  "checkpoint": "/path/to/ckpt",
  "suites": [
    {"suite": "gift-eval", "status": "ok", "metrics": {"crps": 0.42, "mase": 0.81}, "n_series": 97, "detail": ""},
    {"suite": "boom",      "status": "ok", "metrics": {"crps": 0.55, "mase": 0.93}, "n_series": 32, "detail": ""},
    {"suite": "time",      "status": "skipped", "metrics": {}, "n_series": 0, "detail": "TIME loader not configured..."}
  ]
}
```

`status` is `ok` | `skipped` | `error` per suite — one broken suite never aborts
the others, and a skipped/errored suite is logged as such rather than emitting a
fabricated number.

## How each suite plugs in

- **GIFT-Eval** — gluonts-interface. The benchmark's dataset list is *not* an
  importable constant; gift-eval's reference runner (`notebooks/naive.ipynb`)
  hardcodes two strings, so we embed those verbatim (`suites/gifteval.py`,
  `SHORT_DATASETS` / `MED_LONG_DATASETS`) and replicate its term logic, scoring
  via the same `evaluate_model` call (`suites/_common.py`).
- **BOOM** — also gluonts/gift-eval, but its 2,807-config manifest (with each
  config's fixed term) is *not* shipped in gift-eval. We vendor DataDog's
  `boom_properties.json` (`data/`, Apache-2.0) and iterate it, one `Dataset` per
  config with its designated term.
- **TIME** — *not* gluonts; mirrors TIME's own `experiments/chronos2.py`: build
  `timebench.evaluation.data.Dataset`, feed quantile arrays (the wrapper's
  quantile head on TIME's 9-level grid; legacy checkpoints fall back to sample
  paths reduced to that grid) to `save_window_predictions`, and read the resulting
  `metrics.npz` — TIME's own metric code, so the per-task numbers match the
  [leaderboard](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard). Unlike
  GIFT-Eval/BOOM, no Seasonal-Naive baseline is *vendored* for TIME, so we score a
  point Seasonal-Naive forecast through the **same** saver+metric path per task and
  normalize the model metric by it — the ratio→shifted-geomean below — so TIME's
  headline `crps`/`mase` are on the same footing as the other two suites (this is
  also TIME's own leaderboard aggregation). The baseline is checkpoint-independent,
  so it is **cached** (`cache.py`, `~/.cache/cascade_benchmark/time_snaive` or
  `CASCADE_BENCH_TIME_BASELINE_CACHE`; `CASCADE_BENCH_NO_CACHE=1` disables) and
  computed at most once per task — the model forward is the only per-round cost.
  `CASCADE_BENCH_TIME_RAW=1` forces the legacy raw arithmetic mean (not
  baseline-normalized, so **not** comparable).

Both `GIFT_EVAL` and `BOOM` env vars must point at the respective downloaded
benchmark data (gift-eval layout); each suite `skip`s cleanly when unset.

### Aggregation (the headline number)

All three suites are **not** a plain mean of per-dataset metrics. The headline
is the official one (`aggregate.py`, a faithful port of DataDog's
`boom/utils/leaderboard.py`, the same methodology GIFT-Eval and TIME use):

> shifted geometric mean, across datasets, of each metric **normalized by the
> Seasonal-Naive baseline** — with the zero-inflated split (datasets where the
> baseline MASE is 0 use MAE instead) and BOOM's `LOW_VARIANCE_DATASETS`
> exclusion.

We normalize against the **vendored official Seasonal-Naive results** (`data/`,
from the upstream repos), keyed `name/freq/term`. GIFT-Eval keys are constructed
to match all 97 official baseline keys exactly; BOOM is driven directly off the
baseline keys. So `metrics.crps` / `metrics.mase` are leaderboard-comparable
(values near 1.0 ≈ Seasonal-Naive; lower is better). `crps_zero` / `mae_zero`
report the zero-inflated pool.

## Status

Data loading, per-config metrics, and aggregation are all written against — and
verified from — upstream source at the pinned commits (gift-eval `naive.ipynb` +
`data.py`; DataDog `boom/utils/leaderboard.py` + vendored Seasonal-Naive
results; TIME `chronos2.py` + `saver.py`). The aggregation math and GIFT-Eval/
BOOM key-alignment are unit-tested (`tests/test_aggregate.py`; 97/97 GIFT-Eval
keys match the baseline, model==baseline normalizes to 1.0).

What is **not** yet exercised: the actual model inference path
(`CheckpointPredictor` → gluonts `evaluate_model`), because that needs the
installed env + GB-scale data. **Smoke-test with `--max-series 1`** after
`uv sync --project benchmarks`. Refresh the embedded GIFT-Eval list / vendored
baselines if you bump the pinned commits.
