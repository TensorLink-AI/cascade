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

### Datasets / env vars

- `CASCADE_BENCH_GIFTEVAL_DATASETS` — comma-separated `name` or `name/term` to
  override the full GIFT-Eval config list.
- `BOOM` / `CASCADE_BENCH_BOOM_PATH` — path/HF repo for the `Datadog/BOOM`
  dataset; `CASCADE_BENCH_BOOM_DATASETS` to override its config list.
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

- **GIFT-Eval / BOOM** — gluonts-interface, scored via gift-eval's `Dataset` +
  `gluonts.model.evaluate_model` (shared loop in `suites/_common.py`). Both wrap
  the checkpoint as a gluonts predictor (`predictor.py`).
- **TIME** — *not* gluonts; mirrors TIME's own `experiments/chronos2.py`: build
  `timebench.evaluation.data.Dataset`, feed quantile arrays (sample paths from
  the wrapper reduced to TIME's 9-level grid) to `save_window_predictions`, and
  read the resulting `metrics.npz` — TIME's own metric code, so numbers match the
  [leaderboard](https://huggingface.co/spaces/Real-TSF/TIME-leaderboard).

## Status / TODO

**Smoke-test each suite once the env is built** (`--max-series 1`) — these
runners are written against the upstream APIs but have not been executed end to
end here. Likely adjustment points:

- GIFT-Eval / BOOM: the dataset-list symbols (`ALL_DATASETS`, `BOOM_DATASETS`)
  and metric column names in `_common.py` may differ in the pinned gift-eval
  commit — pass explicit `CASCADE_BENCH_*_DATASETS` to sidestep enumeration.
- TIME: confirm the `metrics.npz` keys land under expected metric names (the
  runner averages *every* array it finds, so it's robust to naming) and that
  `Dataset(...).test_data.input` matches the installed `timebench` version.
