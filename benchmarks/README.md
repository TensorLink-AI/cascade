# cascade-benchmark (sidecar)

Scores a trained cascade checkpoint on the public time-series benchmarks
**GIFT-Eval**, **BOOM**, **TIME** and **FEV** (the default `--suites`),
and writes the numbers as JSON. TIME and FEV need their installation extras;
FEV also requires a verified local task/data manifest. The private eval pool
still decides every duel; the public numbers are telemetry, the input to
warm-start promotion and, in v2, a gate that can only block a private-pool win
(DEC-CA-0054 / DEC-CA-0061, `cascade/eval/bench_gate.py`):

**For a protocol-required four-suite run, install both `time` and `fev` extras
and stage all four suites.** An optional dependency at installation time
can still be required by the selected competition rules.

- **Trainer-source policy.** Each finished leg's checkpoint is benched at leg completion (on
  the payer's pod for a funded challenger; the top entries are re-benched on the
  era king's operator pod, and only operator numbers are signed). The numbers
  (per suite × CRPS/MASE) ship in the trainer-signed
  `benchmarks/round-<id>.json` report that promotion selection reads.
- **Main validator (DEC-CA-0063).** From `[v2] bench_source_from_block` with
  `bench_source = "validator"` (the mainnet `chain.toml` value; testnet keeps
  `"trainer"`), the trainer skips lane-A sweeps. The main validator runs this
  sidecar on its own GPU, only for a challenger that won the pool (plus the
  king), within `[v2] bench_sweep_hours` / `bench_budget_hours`, with
  `[eval] bench_workers` / `bench_paired` passed through as `--workers` /
  `--extra-checkpoint`. It signs `bench_v/round-<id>.json`; other validators
  read that report and do not sweep.
- **Validator.** The optional GIFT-Eval no-regression gate
  (`[scoring] gift_gate_mode`) runs a pinned GIFT-Eval subset through this
  sidecar, locally or on an `--eval-hosts` pod.

## Why a separate project / environment

The benchmark harness has its own pinned scientific and dataset dependencies
in `benchmarks/pyproject.toml` and `benchmarks/uv.lock`. It runs in an isolated
Python 3.11 environment and is invoked as a subprocess, so changes to miner,
chain or training dependencies do not silently change benchmark libraries.
The core installation does not need the benchmark harness.

The interface is **a checkpoint directory in, a results JSON out.** Every
cascade checkpoint already ships `forecast_wrapper.py` (the same trusted
inference path the validator scores on), which we wrap as a gluonts predictor —
so this is model-agnostic and consistent with in-protocol scores.

CPM-era checkpoints expose their quantile head
(`forecast_quantiles_batch(histories, horizon)`), so the scorer emits gluonts
`QuantileForecast`s straight from one batched forward pass per `--batch-size`
series — no Monte-Carlo sampling. Cascade MoE Base's 21-level head includes
all nine levels requested by GIFT-Eval, BOOM, TIME and FEV. The adapters select
those columns directly, retaining each suite's scoring grid and aggregation;
missing levels are never interpolated into a pass. Older checkpoints that only implement the
sampling contract still work via the original per-series `SampleForecast` path
(slow; expect the historical hours-long runtimes for those).

## Docker (recommended): env + frozen data as one artifact

Use uv >=0.13.4 and a patched OS CA store. The benchmark lock includes
certifi >=2026.10.19. Run `benchmarks/.venv/bin/python scripts/check_tls_trust.py`
from the repository root after installing locally. See the
[TLS runbook](../docs/OPERATOR.md#tls-trust-and-the-october-2026-emudhra-incident)
for image rebuilds and rollout.

`benchmarks/Dockerfile` bakes the locked env **and** the three datasets at the
revisions pinned in `cascade_benchmark/datasets.py`, so any host with docker
(and optionally a GPU) scores checkpoints with zero setup and comparable
numbers:

```bash
# build from the repository root; the data layer only rebuilds when revisions bump
docker build -f benchmarks/Dockerfile -t cascade-bench:<tag> .

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
uv sync --project benchmarks --frozen --extra time --extra fev
```

## Run

```bash
uv run --frozen --extra time --project benchmarks cascade-benchmark \
    /path/to/checkpoint_dir /path/to/out.json \
    --suites gift-eval,boom,time --num-samples 100
```

For the current four-suite rule:

```bash
uv run --project benchmarks --frozen --extra time --extra fev cascade-benchmark \
    /path/to/checkpoint_dir /path/to/out.json --data-dir ./bench_data \
    --suites gift-eval,boom,time,fev --device cuda
```

Stage the [required datasets](#getting-the-data-auto-download) first. FEV uses
the pinned local data described below. Inspect each suite's status, finite metrics,
per-task rows and revisions; a written JSON file alone is not a passing sweep.

### Concurrent sweep (`--workers`) and paired sweeps (`--extra-checkpoint`)

```bash
uv run --frozen --extra time --extra fev --project benchmarks cascade-benchmark CKPT out.json --device cuda \
    --batch-size 512 --data-dir ./bench_data --workers auto \
    [--extra-checkpoint KING_CKPT king.json] [--timings-json timings.json]
```

Every suite is split into work units: one GIFT-Eval config, BOOM dataset,
TIME task or FEV task (`suites/_common.py`: `plan` / `run_unit` /
`assemble`). `--workers N` runs them over N **spawned** worker processes,
each loading every checkpoint once (`predictor.load_wrapper_cached`) and
pulling one unit at a time. Rows, aggregates and skip lists are assembled in
the fixed unit order. Qualification requires report JSON **byte-identical** to
`--workers 1` on the chosen hardware. Both modes use one CPU thread per worker:
same fp32 forward, same per-unit batches at `--batch-size`, same quantile
grid. A failing unit is recorded exactly as before. A worker that dies is
replaced and its unit re-run once; a unit that kills two workers ends the
run (exit 3), as it would have crashed the sequential sweep.

`--workers auto` sizes to the box (`cascade_benchmark/sizing.py`): min(by GPU
memory, effective CPU cores − 1, host RAM, `--max-workers` [16], units).
Affinity and cgroup quotas/limits are respected. By GPU memory means, per
visible GPU, ⌊(free − 2 GiB) ÷ per-worker⌋. Per-worker VRAM is measured: a
probe process loads the checkpoint(s), forwards one warm-up batch and reads
`torch.cuda.max_memory_reserved`, then the tool adds ×1.25 and 0.5 GiB of
CUDA context. `--worker-vram-gb` overrides the measurement. A CPU device
uses the same CPU/RAM limits. The default RAM allowance is 2 GiB for data plus
twice the checkpoint weight bytes per worker; `--worker-ram-gb` overrides it.
This is an estimate, so measure dataset peaks on the actual VM.
Workers are spread round-robin over the GPUs, one GPU each
(`CUDA_VISIBLE_DEVICES`).

`--extra-checkpoint DIR OUT` (repeatable) sweeps further checkpoints in the
SAME pass: each unit's dataset is built once and forecast by every
checkpoint, and each report is byte-identical to sweeping that checkpoint
alone. `--timings-json` writes per-suite wall times and the worker plan, and
never goes into the report. `scripts/bench_timing.py` (repo root) times
`--workers 1` against other settings and checks all four suites, their data pins,
coverage and identical reports. Missing/failed tasks cannot pass qualification.

Requirement: the wrapper must be stateless across calls, which cascade's
wrappers are: the quantile head is deterministic, and the sample path seeds
each window from its history. A wrapper that kept RNG state would give
schedule-dependent numbers with more than one worker.

No dataset cache is added. GIFT-Eval/BOOM data is already memory-mapped HF
arrow on local disk. A pickled cache of the built, univariate-ized entries
would add a second serialisation path (dtype, `pandas.Period` starts) that
could change numbers. A paired sweep already builds each dataset once per
pass. TIME's checkpoint-independent Seasonal-Naive metrics stay cached
(`cache.py`), scoped to the data revision, task settings and scorer. Unknown or
partial data is never persistently cached.

### Pinning FEV data

Stage the release's chosen FEV tasks and datasets under the same absolute
paths on every eval worker. Keep horizons, windows, seasonality and task names
when replacing Hub paths with local Parquet paths; set `dataset_config: null`.
No implicit Hub downloads are allowed in a normal FEV sweep.

```bash
uv run --frozen --extra fev --project benchmarks cascade-benchmark-pin-fev \
    /data/bench/fev/tasks.yaml /data/bench/fev/manifest.json
```

`--data-dir /data/bench` discovers that manifest automatically. Alternatively,
use `--fev-manifest PATH` or `CASCADE_BENCH_FEV_MANIFEST`. The runner verifies
task/data hashes before spawning workers and records the manifest identity in
`data_revisions.fev`. Changed tasks during execution abort the suite.
`--allow-unpinned-fev` is only for archived/diagnostic runs; its revision is
`unknown`, and it cannot pass qualification. The direct-runtime wrapper uses
the same verifier.

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
uv run --frozen --extra time --project benchmarks cascade-benchmark CKPT out.json \
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
- `CASCADE_BENCH_FEV_MANIFEST` — verified local FEV task/data manifest.
  `--data-dir` also discovers `<data-dir>/fev/manifest.json`. Normal runs use
  this manifest to set `CASCADE_BENCH_FEV_TASKS`; direct task-list/URL overrides
  are for explicitly unpinned diagnostic runs only.

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

A zero process exit means the report was written, not that every required
suite passed. Check suite status, finite metrics, required rows and data/task
revisions. Compare checkpoints only with compatible metric definitions,
aggregation, coverage and revisions; a truncated sweep is not a full-battery result.

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
  computed at most once per task — the model forward is the only per-checkpoint cost.
  `CASCADE_BENCH_TIME_RAW=1` forces the legacy raw arithmetic mean (not
  baseline-normalized, so **not** comparable).

Both `GIFT_EVAL` and `BOOM` env vars must point at the respective downloaded
benchmark data (gift-eval layout); each suite `skip`s cleanly when unset.

**FEV** uses its task and rolling-window API, with the scoring implementation in
`fev_metrics.py`. It records univariate task CRPS/MASE and an in-suite
Seasonal-Naive baseline; multivariate-target tasks are explicitly skipped.
Its headline uses a baseline-normalized geometric mean, while signed per-task
rows retain the inputs needed by the public guards. It is separate from the
three-suite shifted-geomean description below.
Reports explicitly label this `cascade-fev/1` adaptation. It is **not the full
official FEV leaderboard protocol**: covariates are unused and multivariate
targets remain outside its declared scope. Changing that scoring contract is
a coordinated rule change, not a performance optimization.

### Aggregation (the headline number)

GIFT-Eval, BOOM and TIME do **not** use a plain mean of per-dataset metrics. Their headline
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

## Verification

Data loading, per-config metrics, and aggregation are written against — and
verified from — upstream source at the pinned commits (gift-eval `naive.ipynb` +
`data.py`; DataDog `boom/utils/leaderboard.py` + vendored Seasonal-Naive
results; TIME `chronos2.py` + `saver.py`). The aggregation math and GIFT-Eval/
BOOM key-alignment are unit-tested (`tests/test_aggregate.py`; 97/97 GIFT-Eval
keys match the baseline, model==baseline normalizes to 1.0). The model
inference path needs the installed env and GB-scale data, so it is not in CI:
**smoke-test with `--max-series 1`** after `uv sync --project benchmarks`.
Refresh the embedded GIFT-Eval list / vendored baselines if you bump the pinned
commits.
