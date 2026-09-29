# cascade-fullctx-spectral-v12

A differentiated NumPy/SciPy full-context generator derived from the public
`custom-fullctx-v4` design.

## Design

- Emits only 4096-point series, matching the 128-patch training/eval geometry
  and minimizing the one non-target context patch paid per series.
- Samples trend as total end-to-end excursion, so trend strength is independent
  of series length.
- Mixes fourteen families: trend/seasonality, structural regimes,
  multiplicative series, AR/integrated/nonlinear dynamics, smooth spectral GP,
  power-law long memory, physical sensors, seasonal counts, intermittent
  demand, and outliers.
- Replaces the predecessor's slow 48-pass random-Fourier GP with one batched
  inverse FFT and adds persistent/anti-persistent spectral paths.
- Adds a small regime-switching mean-reverting family with bounded clustered
  volatility, heavy-tailed innovations, transient shocks, and seasonal means.
- Executes the OU recurrence with SciPy's compiled linear filter instead of a
  4095-step Python loop.
- Executes AR(1) and AR(2) recurrences with compiled SciPy filters; local
  microbenchmarks were 4–12× faster than scanning time in Python.
- Adds batched cadence-seasonal Poisson and gamma-mixed Poisson counts with
  signed trends and decaying bursts, preserving positive integer structure.
- Adds weekday/weekend interactions to a minority of count rows and piecewise
  spectral slopes to long-memory rows, covering calendar effects and
  scale-dependent roughness without another FFT.
- Gives 40% of trend-seasonal rows a low-innovation mode, teaching sharp,
  stable periodic reconstruction while retaining noisy seasonal coverage.
- Anchors most mixture mass on the five-seed-tested full-context core
  (trend/seasonal, regimes, multiplicative, AR2, and integrated paths), while
  retaining each newer prior at a conservative share.
- Uses published TempoPFN ablations to strengthen OU/SDE-like dynamics,
  spectral/long-memory paths, and transient events without letting one family
  dominate the corpus.
- Adds slowly modulated amplitude and phase to a minority of seasonal
  components; stationary cycles remain the majority. Multiplicative paths use
  the same full cadence bank instead of a four-period subset.
- Extends structural/event coverage with piecewise-affine regime trends,
  decaying shock recovery, event plateaus, and genuine held-constant runs.
- Applies low-rate reversal, censoring, quantization, and sample-and-hold
  artifacts to bridge clean priors to real measurement pipelines.
- Extends seasonality through 365/672/730-step cycles and adds a small generic
  physical-sensor family (smooth, bounded, pressure-like, and skewed-positive)
  without adopting the competitor's private-pool-shaped weather weighting.
- Generates lazy 2048-row random-family chunks, keeping every stream prefix
  mixed while amortizing Python dispatch. Local profiling found this about 6%
  faster than 1024 rows; 4096 rows regressed slightly.
- Evaluates optional seasonal components only for active rows while preserving
  the fixed RNG draw sequence, and caches the fixed cadence sine/cosine basis,
  reducing trigonometric work without narrowing the prior.
- Draws jump, shock, and heavy-tail values only where those sparse branches are
  active instead of allocating dense arrays whose values are mostly discarded.

On this VPS, an 8192-series benchmark improved from a v11 pre-optimization
median of 9.40M points/s to 11.38M points/s after v12 prefetching.
The generator is now well above the mainnet contract's 3.7M reference
throughput in isolation; end-to-end token completion also includes model
training and stream handoff.

An end-to-end isolation run found that synchronous generation left training
blocked on data for 21.9% of its wall (`2.13M` point-passes/s). A deterministic
one-chunk producer thread now overlaps NumPy/SciPy generation with GPU work,
cutting data wait to 3.9% and raising training throughput to `2.43M`
point-passes/s (+14.4%) on the A100. The same short contract budget then
completed without a deadline hit. Cached rows reached `2.74M`, confirming the
remaining gap to the live L40S reference is mostly model/device throughput.

A controlled 120-second parameter screen then compared the baseline mixture
with seasonal-, spectral-, and dynamics-heavy variants under the same model,
pool, budget, and seeds. Dynamics-heavy won all three validation seeds, reducing
mean local synthetic-pool geomean from `0.19097` to `0.18431` (3.5%; lower is
better). The applied weights increase AR(2), integrated, threshold-AR, chaotic,
regime-shift, and OU coverage while reducing stationary seasonal, spectral, and
sparse/count families. This remains a directional local result, not a live
validator verdict.

## Local training result

The v10 corpus was trained under the mainnet `chain.toml` contract on an A100
for the full 3-hour wall. It scored `0.13679` on the 64-window local synthetic
smoke pool (lower is better), improving from `0.15429` at the 30-minute heat
budget, while reaching 55% of the token budget. The optimized dynamics-heavy
v11 heat reached 59% (`3.90B / 6.66B`) and scored `0.15424`. The v12 prefetch
isolation test then cut data wait from 21.9% to 3.9% and raised end-to-end
throughput from `2.13M` to `2.43M` point-passes/s. These scores are directional
and are not live-validator verdicts; the A100 remains below the contract's
L40S-calibrated `3.7M` reference.

## Validate

```bash
python -m cascade.miner.cli verify ./cascade-v2 --chain-toml chain.toml
```

Contract validity and CPU throughput do not establish forecasting quality.
Run a production-faithful GPU A/B score against the current king before
deploying this candidate.
