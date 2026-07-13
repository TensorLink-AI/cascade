# cascade base generator (genesis "base" king)

A self-contained cascade data generator that adapts a curated subset of
[TempoPFN](https://github.com/automl/TempoPFN)'s procedural synthetic time-series
priors into a single deterministic `Generator(DataGenerator)`. Intended as the
launch/genesis king for the cascade subnet.

## What's inside

| file | purpose |
|------|---------|
| `generator.py` | `class Generator(DataGenerator)` — the entrypoint |
| `config.json` | length band, per-family mixing weights, sanitisation knobs |
| `requirements.txt` | hash-locked, allowlisted deps (`numpy, scipy, pandas, torch, scikit-learn, gpytorch, networkx`) |
| `tempo_gen/` | vendored, import-rewritten TempoPFN subset (Apache-2.0) |
| `NOTICE`, `LICENSE` | TempoPFN attribution + Apache-2.0 text |
| `tests/` | contract tests + a diversity sanity check |

## Generator families

Ten families are mixed by weight: **ForecastPFN, SineWave, SawTooth, Step,
Anomaly, Spikes, OrnsteinUhlenbeck, GP-prior, KernelSynth, CauKer**. Each is
drawn at `generate_length` (4096) and deterministically random-cropped into the
`[min_length, max_length]` band for length diversity.

The GP/kernel family — **GP-prior** (gpytorch), **KernelSynth** (scikit-learn)
and **CauKer** (networkx + scikit-learn) — was added in v2: those deps are now
on cascade's allowlist (`chain.toml [dependencies]`). The TempoPFN ablation
shows this family carries a large share of the downstream signal. CauKer is
multivariate (an SCM DAG of GP-prior nodes); each channel is flattened into its
own univariate series so the emitted corpus stays 1-D. Its upstream GPU
(`cupy`) draw was replaced with NumPy's seeded `multivariate_normal` to keep the
generate path CPU-only and reproducible.

The pyo-backed **audio** generators remain **excluded**: pyo runs a real-time
audio server and seeds via `hash()`, both of which break the cross-process
determinism contract below (and pyo is not on the allowlist).

## Determinism

The corpus is a pure function of `(seed, n_series)`. We seed NumPy / torch /
Python `random` from `seed`, run torch on CPU with deterministic algorithms,
derive every per-family and per-series sub-seed from the master seed, and use a
separate seeded RNG for cropping. The upstream `hash()`-based per-generator seed
offset — salted per process by `PYTHONHASHSEED` — was replaced with a stable
`zlib.crc32` in `tempo_gen/.../abstract_classes.py`, so two audit runs in
*separate processes* produce byte-identical corpora.

## Verify / test

```bash
# from the cascade repo root, with deps installed and cascade importable
cascade verify ./base_generator                       # full trainer-side checks
PYTHONPATH=$PWD pytest base_generator/tests -q           # contract tests
PYTHONPATH=$PWD python base_generator/tests/diversity_check.py
```

## Config knobs (`config.json`)

- `min_length` / `max_length` — per-series length band (defaults 64 / 4096, matching `chain.toml` — and `[training] context_length`, so series can fill the model's context as Toto 2.0's do).
- `generate_length` — length generators are drawn at before cropping (>= `max_length`).
- `batch_size` — internal generation batch size (memory/throughput knob).
- `weights` — per-family mixing weights (normalised; families with weight 0 or absent are skipped).
- `standardize` — z-normalise each series (default `false`; off preserves scale diversity).
- `clip_sigma` — if > 0, clip each series to `mean ± clip_sigma·std` (default `0`, disabled).
- `max_abs_value` — hard magnitude clip to keep output trainer-safe finite.
