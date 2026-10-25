---
id: DEC-CA-0064
type: decision
title: "MarketCast: a lane-C fine-tune eval on a pinned Timestral-28M base, judged paired vs the untouched base on forward-only market windows, one lane-C slot per suite (mkt:returns, mkt:vol, …); built (pool, trainer, validator, audit, intake), shipped off at from_block = 0"
status: proposed
date: 2026-10-19
tags: [consensus, scoring, lanes, fine-tuning, real-data, markets, v2]
revisit_when: "the owner arms it (types an era-boundary from_block after answering the Open list: licence, FX source for the other 25 crosses, measured validator-GPU cost, a testnet cycle); or the forward-only snapshots cannot supply min_windows over min_clusters assets within verdict_wait_blocks for a suite (lengthen the wait, publish snapshots more often, or merge suites); or fine-tunes win mkt:returns with mis-calibrated tails (tighten coverage_tolerance / tail_tolerance); or near_dup_tol flags independent fine-tunes as duplicates (measure the honest spread first); or the timestral library moves (the training path uses two private methods at the pinned commit)"
relations: {depends_on: [DEC-CA-0056, DEC-CA-0062, DEC-CA-0063], relates_to: [DEC-CA-0028, DEC-CA-0020, DEC-CA-0026, DEC-CA-0057, DEC-CA-0058]}
---

## Decision

MarketCast is a **fine-tune eval**, not a public benchmark. Miners fine-tune
a PINNED third-party base, `mistralai/Timestral-28M` (HF), for one MarketCast
suite. Each fine-tune is judged **paired against the untouched pinned base**
on that suite's forward-only market windows. A pass needs `gain LCB ≥
gain_margin` (0.01, never 0). Each suite is its own lane-C target slot,
`mkt:<suite>`, with lane C's payout schedule and TTL (DEC-CA-0056). The
emission comes from lane C's share, split equally across lane C's targets.

**Purpose.** Forecast the *underlying quantities* that the market subnets'
signals are built from: return distributions, realized vol, ranges, funding,
liquidity and dTAO alpha. We do not reproduce their exact signals (paths,
embeddings, portfolios, trades). A base that forecasts these well is useful
to miners on every one of those subnets.

**Shipped state: built, OFF.** `[marketcast]` parses onto
`ChainConfig.marketcast` (`cascade/shared/config/marketcast.py`) with
`from_block = 0` in both tomls. The pool, trainer, validator, audit, intake
and miner-CLI halves exist and are tested. Until the owner types an
era-boundary `from_block`:

- no `mkt:` target can be configured, funded or committed;
- verdicts, emissions, receipts, manifests and audit output are
  byte-identical to before (tested).

Arming is load-checked; see §Arming.

### Why a sibling `[marketcast]` table, not `[lane_c.marketcast]`

The DEC-CA-0057 overlay allow-list admits every key of `[lane_c]`
(`ALLOWED["lane_c"] = None`). A nested table there could be changed by an
overlay. MarketCast pins a new base architecture, its checkpoint digest and a
market-data contract, and all of those are release-only. So it gets its own
table. Updated 2026-10-21 (DEC-CA-0057): the listed numeric verdict, payout
and liveness controls are overlay-settable (`ALLOWED["marketcast"]`);
the base pins, activation, universe, data sources and window construction
remain release-only. This also leaves `LaneCConfig` (consensus terms the
audit replays) byte-identical. The two tables share only the target grammar:
`parse_target(spec, mkt_suites=…)` accepts `mkt:<suite>` only for the suites
that `MarketCastConfig.target_suites` admits. That set is empty while
MarketCast is inert. With no `mkt_suites`, the grammar and every error message
are exactly as before.

## Source research (subnet repos, read 2026-10-19)

| Subnet | Repo @ commit | What miners submit / how they are scored | Data |
|---|---|---|---|
| Synth SN50 | `synthdataco/synth-subnet` @ `d2daf20` (2026-10-14) | 1000 simulated price paths (24h @ 5m, 1h @ 1m); CRPS on bp price changes over several intervals; the 1h track adds realized-vol CRPS (std of 1m bp returns per block) | Binance / Hyperliquid 1m |
| Mantis SN123 | `Barbariandev/MANTIS` (main, 2026-10-18) | embeddings for binary / hit-first ±σ / σ-bucket / breakout / cross-sectional rank / funding-change rank / FLOW challenges; paid by marginal contribution in meta-models | owner R2 prices + funding (~1m) |
| Investing SN88 | `mobiusfund/investing` @ `4859a586` (2026-09-24) | portfolio weights (dTAO alpha long-only; ~565 US stocks/ETFs); return/maxDD × kelly-odds | stocks 30m OHLCV; alpha hourly pool state |
| Vanta SN8 | `taoshidev/vanta-network` @ `4b630b1` (2026-10-15) | trading signals on crypto perps, FX, commodities, equities; HWM-gated realized PnL, drawdown eliminations; costs = funding, σ-slippage, spread, FX rollover | Polygon, Tiingo, Hyperliquid |

Suite × subnet (● core driver, ○ secondary):

| Suite | Synth | Mantis | Investing | Vanta |
|---|---|---|---|---|
| `mkt:returns` | ● | ● | ○ | ● |
| `mkt:vol` | ● | ● | ○ | ○ |
| `mkt:range` | ○ | ● | ○ | ● |
| `mkt:funding` | | ● | | ● |
| `mkt:liquidity` | | ○ | ● | ● |
| `mkt:alpha` | | | ● | |

Vanta's costs also depend on funding, liquidity, range and sessions
(rollover / open / close), which is why those suites and the session
covariates exist.

## Suites and metrics (`cascade/eval/marketcast.py`, pure numpy)

| Suite | Series (built from 1m OHLCV + funding) | Horizons (placeholder) | Verdict loss | Also reported |
|---|---|---|---|---|
| `mkt:returns` | log return over `h` bars, bp (`log_returns_bp`) | 1m 5m 1h 4h 1d (5d dropped, §Review fixes) | CRPS in bp | coverage GATE at 50/80/98 %; rank IC / AUC (diagnostic only) |
| `mkt:vol` | realized vol per block: std(ddof=1) of 1m bp log returns, blocks with < 2 obs NaN (`realized_vol_bp`) | 5m 15m 1h 1d | CRPS on log RV | QLIKE on the median |
| `mkt:range` | ln(max H / min L), max up / down excursion vs the block open (`range_excursions`) | 1h 4h 1d | CRPS on log range | |
| `mkt:funding` | funding-rate change, bp (`funding_change_bp`); basis from the Hyperliquid premium (`basis_bp`); no historical OI source | 8h | CRPS | MASE of the median |
| `mkt:liquidity` | ln Σ volume per block (`log_volume`); spread (`corwin_schultz_spread_bp`) and ADV (`adv`) as diagnostics | 1h 1d | CRPS | |
| `mkt:alpha` | log return of the dTAO alpha price, bp (`alpha_price` = tao_in / alpha_in; depth `log_depth`) | 1h 1d | CRPS | |

- **CRPS** = `2 · mean_k pinball_k` over the base's knots. This is the
  repo's GIFT-Eval convention, left unnormalised, so returns in bp give CRPS
  in bp. Knot weights are uniform. The repo has no quantile-input CRPS
  (`crps.mwsql_components` takes samples), so the three-line pinball is
  local.
- **Coverage gate** (`coverage_gate`): the central interval at level `c` runs
  from the `(1−c)/2` quantile to the `(1+c)/2` quantile, interpolated from the
  knots. The gate FAILS if empirical coverage at any level is below
  `nominal − coverage_tolerance`. Both sides are reported, but over-coverage
  does not fail the gate (CRPS already penalises intervals that are too
  wide). The gate is AND-ed into the verdict on `mkt:returns` only
  (`COVERAGE_GATE_SUITES`).
- **Path quantities are their own series.** Timestral outputs MARGINAL
  quantiles, and horizons past 1024 steps autoregress on the median. So
  barrier, range and per-block vol quantities are never derived from return
  marginals.
- **Knot mapping** (`map_quantiles`): crossing knots are sorted first (the
  monotone rearrangement), then interpolated linearly in level space. Levels
  outside the source range clamp to the end knot. The mapping is exact on
  shared levels: Timestral's 21 knots map exactly onto the repo's 9-level
  grid, and mapping to a superset and back is the identity.

### Paired verdict (`judge_suite`)

- The per-window losses of the fine-tune and the base are taken on the SAME
  eligible windows.
- The statistic is `rel = 1 − geomean(ft)/geomean(base)`, with its LCB_α from
  the repo's `expert_lane.paired_geomean_lcb`.
- (Superseded by §Review fixes II: clusters are asset class × time block,
  horizons and classes weighted equally.) With clusters (the asset), each cluster is first collapsed to its
  per-model geomean loss. The bootstrap then resamples whole assets.
- A suite passes only if all of these hold:
  - `LCB ≥ gain_margin`;
  - the point gain is > 0;
  - there are ≥ `min_windows` windows over ≥ `min_clusters` assets;
  - for `mkt:returns`, the fine-tune's coverage gate passes.
- An untouched base never passes: identical losses give LCB = 0, which is
  below the margin.

### Forward-only windows (`window_eligible`)

A window is eligible for a submission only if its forecast start is strictly
after the submission's block time, and it can be scored only once its end
has been observed. Market data is public, so this rule is what makes leakage
impossible. Retired windows are revealed later for audit (the DEC-CA-0052
`reveal_after` pattern). Public bench guards are OFF for `mkt:` targets:
`parse_lane_c` refuses a `guard_public` entry on one.

### Covariates

Covariates are inputs, never targets: cross-asset prices, funding / OI,
volume, lagged indicators, session / calendar flags (FX 17:00 ET rollover,
equity open / close, weekends) and scheduled macro events as known-future
inputs. Timestral takes past and known-future covariates.

## Base: Timestral-28M (pinned 2026-10-19)

Pins, read from the Hub's public API:

- `base_ref = "mistralai/Timestral-28M@180698bbf5a50dda3516aa6419798437fb03a430"`,
  the `sha` of `https://huggingface.co/api/models/mistralai/Timestral-28M`
  (lastModified 2026-10-15).
- `base_sha256 = "dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d"`,
  the `model.safetensors` LFS oid from
  `https://huggingface.co/api/models/mistralai/Timestral-28M/tree/<sha>`
  (111,296,464 bytes, ≈ 28M float32).

`config.json` facts at that commit:

- patch 32;
- **`d_model` 384** (the README says "embedding dimension 256");
- 6 heads, MLP 1024;
- `block_schedule` of 12 layers (9 `time` + 3 `time_variate`, so
  `TIMESTRAL_BLOCKS = 12` is confirmed);
- `block_size` 32 patches (1024 steps per decode pass);
- `context_length` 32768;
- 21 knots 0.01 … 0.99;
- `covariate_enabled`, `use_asinh`.

The modelling code is the separate **`timestral` library**, read as text
only:

- source: `github.com/mistralai/timestral` @
  `a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff`, version 1.0.0;
- licence: Apache-2.0 for the code;
- it is NOT on PyPI. The PyPI JSON for `timestral` returns Not Found.

It is an OPTIONAL extra (`.[marketcast]`, a git-commit direct reference;
`uv.lock` not regenerated). The architecture differs from Cascade-MoE:
`cascade/trainer/moe_model.py` is untouched and Timestral never inits a king.

**Wrapper: real** (`cascade/eval/timestral_adapter.py`).

- `TimestralQuantileForecaster` implements the `QuantileForecaster`
  Protocol over the library's `TimestralForecaster.forecast`:
  - context `(B, T)` with NaN = missing;
  - `past_covariates (B, C, T)` and `known_covariates (B, C, T + H)`;
  - the library decodes horizons past 1024 steps block by block, feeding
    the median back;
  - the library's `symmetric_averaging` and `clip_nonnegative` defaults stay
    on, identically for the base and every fine-tune.

  The library is lazy-imported and takes a device argument.
- `load_pinned_base` resolves the commit and checks the sha256 BEFORE load.
- Fine-tune hook:
  - `select_trainable` covers `full`, `last_n_blocks`, `frozen_embeddings`
    and `lora`. **The library has no LoRA**, so `LoRALinear` wraps every
    `nn.Linear` of the transformer layers in our code. It starts exactly at
    the base (B = 0), and `merge_lora` folds it back so the export has the
    base's exact key set.
  - `forward_block` is one differentiable decode block. It uses the
    library's PRIVATE `_as_context` / `_assemble_rows` plus the public
    `model.predict_block`, because the public `forecast` runs under
    `torch.inference_mode`. This is why the pin is a commit.
  - `export_checkpoint` writes the base's `config.json`, a float32
    `model.safetensors` and `marketcast_meta.json`.
- Pure harvest checks: `safetensors_header`, `same_structure`,
  `safetensors_all_finite`.

CI has no `timestral` install. Tests run on stubs and a Timestral-shaped toy
model, and the one real-library test skips.

**Licence (owner / legal open item).** The weights are "Modified MIT". Clause
2 bars use by a company over USD 20M monthly revenue, and **it extends to
derivatives**: every fine-tune is one.

## Miner submission (owner 2026-10-19)

A submission has two parts:

- **(a) A declarative recipe**, `marketcast.json`, validated by
  `cascade/shared/marketcast_recipe.py:parse_marketcast_recipe`. Unknown keys
  are refused at every level and every bound comes from `[marketcast]`. The
  existing lane-C `recipe.json` keys are unchanged. The recipe covers:
  - `target`;
  - `data`: assets per class ⊆ config, class weights, frequencies, and a
    `start` / `end` date range whose end falls at least `embargo_days`
    before the submission;
  - `context_length` ≤ `max_context_length` (32768);
  - `horizon_mix` over the suite's horizons;
  - `covariates` from an allow-list;
  - `augmentations` from an allow-list, each with `p` and numeric params;
  - `objective`: per-suite and per-knot loss weights, plus an optional
    coverage penalty;
  - `optimizer`: lr ≤ `max_lr`, schedule, warmup ≤ 0.2, steps, batch,
    weight decay, clip, EMA 0 or [0.9, 1);
  - `trainable` scope: `full`, `last_n_blocks` (1–12),
    `frozen_embeddings`, or `lora` (rank ≤ `max_lora_rank`);
  - `synthetic_fraction`.
- **(b) An OPTIONAL generator tree.** It is fully MINER-DESIGNED code, using
  the same sandboxed `generate()` contract as the lane-A / lane-C trees (not
  a preset or a parameterised allow-list). Its synthetic series are mixed
  with the real data at `synthetic_fraction` ≤ `max_synthetic_fraction`
  (0.5). A recipe with `synthetic_fraction > 0` and no generator tree is
  refused.

MarketCast needs that generator to emit **target + covariate channels**.
DEC-CA-0020 `accepted_fields` and the DEC-CA-0026 roles are currently gated
to `values`. This DEC does not change the lane-A `accepted_fields`; a
MarketCast-only field set is an open item.

## Scoring on the main validator (DEC-CA-0063 pattern)

The main validator (`[netconfig] owner_hotkey`) runs the base and the
fine-tunes on its GPU over the suite's `pool_mkt` windows and publishes
signed per-window rows at `marketcast/round-<id>.json`
(`cascade/shared/marketcast_report.py`), with the forecasts themselves as
float32 quantile sidecars. Satellites and other validators replay the
verdict from those rows (DEC-CA-0058); they never run the model. The base's
quantiles are cached per window. Rows are applied only when a later
trainer-signed lane-C manifest records the report (§Review fixes); a
missing report never holds a settlement.

## Data sources (owner 2026-10-19)

Hyperliquid is PRIMARY wherever it lists the asset. The listed universe was
read from `POST https://api.hyperliquid.xyz/info` with `{"type":"meta"}`,
`{"type":"perpDexs"}` and `{"type":"meta","dex":"xyz"}`; the `[marketcast]
assets` lists are taken from it:

- crypto: 28 main-dex perps;
- fx: `EUR`, `JPY`, `GBP`. These are ALL the FX Hyperliquid lists (`xyz:`,
  vs USD). The other 25 of the 28 majors and crosses have no source;
- metals: `GOLD`, `SILVER`, `PLATINUM`, `PALLADIUM`;
- commodities: `CL`, `BRENTOIL`, `NATGAS`, `COPPER`;
- equity_etf: `SP500`, `XYZ100`, `JP225`, `KR200`, sector / country ETFs and
  seven mega-caps;
- alpha: eight netuids.

Feeds:

- Hyperliquid 1m `candleSnapshot` + hourly `fundingHistory`. Funding
  `premium` is the basis series.
- Binance spot 1m klines are the crypto cross-check, reported and never
  gating.
- dTAO alpha is read from the subnet pool state via the pinned SDK read path
  `ChainClient.subnet_economics` uses.

Hyperliquid `xyz` prices are **perp prices**: they carry a basis to the cash
market and drift off-hours. That is acceptable for a forecasting eval.

Cash-equity vendors (Polygon / Tiingo / Databento) are out of the plan. For
the remaining FX crosses the owner's candidates are Pyth Hermes (per-minute
polling by the main validator, redistribution terms unverified) with
Chainlink on-chain FX as a cross-check. They are recognised source ids with
no adapter, and a class that names one cannot be armed.

## Config (`[marketcast]`)

| Key | Shipped | Notes |
|---|---|---|
| `from_block` | `0` | an era boundary arms it (§Arming) |
| `base_ref` / `base_sha256` | the pins above | `<repo>@<40-hex>`; branches refused |
| `base_arch` | `timestral` | the only allowed value |
| `suites` | `[returns, vol]` | ⊆ the six |
| `assets` / `data_sources` | §Data sources | `"<primary>[+<cross-check>]"`; `""` refuses arming the class |
| `frequencies` / `horizons` | per suite | a horizon is forecast at the coarsest frequency dividing it (`window_geometry`) |
| `coverage_levels` / `coverage_tolerance` | `[0.5, 0.8, 0.98]` / `0.05` | gate on `mkt:returns` |
| `alpha` / `bootstrap_B` / `gain_margin` | `0.05` / `10000` / `0.01` | margin never 0 |
| `min_windows` / `min_clusters` | `200` / `8` | clusters = asset class × 6 h block (§Review fixes II) |
| `pool_prefix` | `pool_mkt` | |
| `eval_context_length` | `4096` | the context the base and the fine-tune both see |
| `windows_per_cell` | `48` | per (asset, horizon), spread evenly over the eligible span (§Review fixes II) |
| `start_margin_seconds` | `3600` | windows start ≥ 1 h after the train block |
| `verdict_wait_blocks` | `43200` | an inconclusive entry waits up to 6 days |
| `snapshot_days` | `21` | realized bars per eval snapshot |
| `coarse_history_days` | `180` | hourly bank for ≥ 1h series (§Review fixes; ≥ `snapshot_days`) |
| `report_timeout_settlements` | `8` | the trainer records a missing report `"absent"` after this (≥ 1) |
| `max_infra_deferrals` | `8` | infra deferrals before an uncharged drop (≥ 0) |
| `covariates` | `cross_asset, funding, volume, calendar, session` | `open_interest` / `macro_events` dropped (no historical OI from Hyperliquid, no pinned event calendar) |
| `augmentations` | `scale, jitter, crop, time_warp, mixup` | pure seeded transforms (`marketcast_leg.augment`) |
| recipe bounds | `max_lr 1e-3`, `max_steps 1e5`, `max_batch 512`, `max_lora_rank 64`, `embargo_days 30`, `max_synthetic_fraction 0.5` | |

New series builders (`cascade/eval/marketcast.py`):

- `funding_bp` and `basis_bp` (Hyperliquid `premium`);
- `corwin_schultz_spread_bp` (the high/low spread estimator, pairs inside a
  block);
- `dollar_volume` and `adv`;
- `alpha_price` (`tao_in / alpha_in`) and `log_depth`;
- `suite_series` (one value per frequency block, per suite), `cut_windows`
  (forward-only, origins on the absolute UTC grid so window ids are stable
  across snapshots), `build_covariates` (US Eastern session flags with the
  US DST rule, no tz database).

`mkt:liquidity` keeps log volume as its verdict target; spread and ADV are
builders and diagnostics. `mkt:alpha` forecasts log returns of the alpha
price.

## Arming

The table-local checks (`parse_marketcast`):

- `base_ref` is pinned to a commit and `base_sha256` is set;
- every class an enabled suite reads that lists assets has a non-empty data
  source with a built adapter.

The cross-table checks (`check_marketcast_arming`, in the loader):

- the block is an era boundary on the grid in force;
- it is ≥ `[v2] from_block` and ≥ `[lane_c] from_block`;
- `[v2] bench_source = "validator"` is in force at or before it (main-validator
  GPU scoring);
- `[netconfig] owner_hotkey` is set.

## Verdict lifecycle (as built)

Lane C's pending → confirm → slot machinery, with three MarketCast rules:

1. **Base never moves.** The pinned base is not the king, so no
   `max_base_age_eras` expiry and no `base_moved` / `stale` on a king
   change. `stale` means the entry's `marketcast` record names another base.
2. **Fresh confirm windows.** A confirm is judged on windows starting after
   the judge-phase snapshot's end.
3. **Waiting.** An entry inconclusive for lack of windows or clusters WAITS:
   it is re-deferred (keeping its origin round), or a confirm stays
   `pending`, while the settlement is within `verdict_wait_blocks` of its
   origin. After that it is dropped.

Receipt rows of `mkt:` targets carry an `mkt` record:

- the report key and sha256, and the snapshot sha256;
- `after`, `origin_block`, `within_wait`, `inconclusive`;
- `fresh_after`, and the coverage results.

Rows of other targets are unchanged.

## Open (owner to answer)

Answered on 2026-10-19:

- data sources (§Data sources);
- asset lists (Hyperliquid-listed);
- launch suites (returns + vol);
- timestral pin (above);
- verdict delay (`verdict_wait_blocks`, 6 days, which covers the 5d horizon
  plus snapshot lag).

Still open:

1. **Timestral licence:** the derivative revenue restriction (legal).
2. **FX beyond EUR / JPY / GBP:** Pyth Hermes (terms) and Chainlink
   cross-check. Each needs an adapter.
3. **Measured validator-GPU eval cost:** base + k fine-tunes ×
   `windows_per_cell` × assets × horizons per settlement. Measure it on
   testnet before arming.
4. **Coverage gate tolerance** (0.05), and whether the 98 % level gates or
   is only reported.
5. **Generator output fields for covariates:** synthetic samples carry NaN
   covariates today. A MarketCast field set (target + covariate channels)
   beside the gated lane-A `accepted_fields` is not built.
6. **"Bring your own checkpoint" intake:** not built.
7. **"cpm":** meaning unclear; ask.
8. **Unverified research items:**
   - Synth's 1h realized-vol term on mainnet;
   - Mantis FLOW's weight;
   - Vanta received funding;
   - the SDK 11.3.0 `get_dynamic_info` reserve keys `tao_in` / `alpha_in`
     (a missing key reads NaN).
9. **Per-asset vs time-block clusters** for the bootstrap.
10. **Hyperliquid history depth:** measured 2026-10-19 (§Review fixes): the
    most recent ~5000 candles per interval only. 1m ≈ 3.6 days, so a cold
    21-day minute grid is mostly missing and an embargoed (≥ 30 days old)
    training range has no minute bars at all on Hyperliquid. Open: carry
    minute bars forward across snapshots, and a minute-bar archive for
    training.

## What is built (code map)

- **Config:** `cascade/shared/config/marketcast.py` (values, sources,
  `arming_problems`, `check_marketcast_arming`, `active`), the loader, both
  tomls.
- **Pure scoring:** `cascade/eval/marketcast.py` (series builders, geometry,
  windows, covariates, coverage tallies, `judge_suite`).
- **Wrapper:** `cascade/eval/timestral_adapter.py`.
- **Feeds:** `cascade/pool/marketcast_feeds.py` — Hyperliquid / Binance
  request builders and parsers, injectable network, pagination, grid,
  cross-check, the alpha recorder.
- **Snapshot:** `cascade/pool/marketcast.py`
  - content-addressed (per-file sha256 + body digest) and signed by the main
    validator;
  - window draw `snapshot_windows`;
  - covariates `window_covariates`;
  - the bucket source `MktSnapshotSource`.
- **Pool CLI:** `cascade-pool publish --lane mkt`, `reveal --lane mkt`,
  `mkt-record-alpha`.
- **Trainer:**
  - `cascade/trainer/marketcast_leg.py`: sampler, augmentations, loss,
    loop, `run_mkt_leg`;
  - `cascade/trainer/marketcast_runner.py`: recipe fetch, dispatch on the
    miner's key, harvest check;
  - worker `--role mkt_finetune`;
  - `remote.worker_argv` / `receipt_role` / `LocalTrainReceipt.marketcast`;
  - the harvest branch in `loop._harvest_funded_checkpoint`;
  - `lane_c_driver` (dispatch only while active, no expiry / `base_moved`
    for mkt legs, the `marketcast` manifest record).
- **Manifest:** `LaneCEntry.marketcast` (dropped from the signed body when
  empty), and `parse_finetune_commit(..., mkt_suites=)`.
- **Validator:**
  - `cascade/validator/marketcast.py`: pre-checks, rows, verdict, the
    main / other report path;
  - `cascade/validator/lane_c.py`: routing, the wait rules in
    `resolve_outcomes` / `_pending_of`, `mkt` receipt rows;
  - `PendingEntry.mkt`;
  - `cascade/shared/marketcast_report.py`: signed rows,
    `verdict_from_rows`.
- **Audit:**
  - `check_marketcast` replays the signed rows, and joins `run_tier0` only
    when active;
  - `check_lane_c` uses the `[marketcast]` terms for `mkt:` rows;
  - `cascade-audit` fetches the reports.
- **Intake / miner CLI:**
  - `FundingIntake(mkt_suites=)` (`marketcast_inactive` while off) and
    `ChainRevealResolver.mkt_suites`;
  - `cascade lane-c check|commit|fund` accept `mkt:` targets +
    `marketcast.json` when active and refuse them while off.
- **W&B:** `lane:C/mkt`.

## Review fixes 2026-10-19

A review found five blockers; all are fixed in code, tested, and still
shipped OFF (`from_block = 0` in both tomls).

### B1 — liveness: MarketCast never holds a settlement

*Before:* a missing or unreadable main-validator report raised
`LaneCTransient`, which held the WHOLE settlement (lanes A, B and C) with
no bound; a deterministic error (`window_covariates` on an unknown
covariate name from the checkpoint's `marketcast_meta.json`) was a
permanent hold.

*Now* (`cascade/validator/marketcast.py`, `cascade/validator/lane_c.py`):

- MarketCast has its own pipeline. Each settlement REGISTERS its
  MarketCast entries plus every MarketCast entry left pending into one
  MarketCast round (`LaneCState.mkt_backlog`). Lanes A / B and
  non-MarketCast lane C proceed regardless.
- **The arbiter is the trainer-signed lane-C manifest**, not a clock (no
  chain head, no wall clock, so two validators can never disagree): a later
  manifest records `mkt_reports = {round_id: sha256 | "absent"}` —
  the sha256 of the exact report text the trainer read, or `"absent"` once
  `report_timeout_settlements` settlements passed without one. The field is
  dropped from the signed body when empty, so manifests without MarketCast
  are byte-identical. While MarketCast is active the trainer publishes a
  lane-C manifest every settlement it records something, even with no leg.
- Validators apply the backlog strictly in round order, each round exactly
  as recorded. A recorded sha256 ⇒ those bytes (a READ failure there holds,
  like any bound read: the bytes provably exist); bytes that do not hash to
  it, or an unparseable / unsigned report ⇒ every candidate an `infra`
  deferral (B2). `"absent"` ⇒ the same. A round no manifest ever records
  expires to absent after 2 × `report_timeout_settlements` MarketCast
  settlements — a count of folded settlements, deterministic.
- Deterministic per-entry errors are the miner's `fault`, never a hold:
  `marketcast_meta.json` is validated at harvest
  (`marketcast_runner.verify_harvested_mkt_checkpoint` →
  `marketcast_recipe.meta_problem`: required fields, the pinned base, an
  enabled target, covariate names unique and ⊆ the allow-list ∩ the
  builders, ≤ 64 KiB) and again on the validator inside the fault path
  (`compute_entry_rows`).
- The verdict wait is measured on the ROUND's block, so a late report never
  eats it. A win's `win_block` is the settlement that applied it.

Residual: the commitment-history read MarketCast registration needs is the
same chain read non-MarketCast lane C needs; its failure holds as before.

### B2 — a main-validator compute failure never costs the miner

Each entry's rows are retried `MKT_COMPUTE_ATTEMPTS` (2) times within the
round. Still failing (OOM, a library exception, the base not loading), the
signed report marks the entry `infra_deferred` (not judged, not dropped).
A CUDA OOM (`RuntimeError`) is no longer a fine-tune fault; only a
`ValueError` from the fine-tune on valid input is. Validators turn it into
an `infra` judgement: it waits (a `deferred` / `pending` carry) without
consuming `verdict_wait_blocks` (the origin shifts by the time between the
deferral and the next attempt), up to `max_infra_deferrals` (8). Past the
cap it drops with error class `operator` — the lane-C queue's existing
requeue-eligible, never-burned class (`funding/queue.py`
`OPERATOR_FAULT_CLASS`); the trainer requeues it uncharged when it reads a
report naming the drop. Each lane-C entry is judged independently
(`_mkt_apply_item` judges candidate by candidate), so one entry's infra
deferral never changes another's verdict. Residual: a drop whose last
attempts were all `"absent"` rounds is visible only in the receipt; the
trainer cannot requeue it (it never saw a report).

### B3 — verdicts checkable without trusting the main validator

- Every eligible window is a row (non-finite losses are `null`), and each
  entry binds a content-addressed sidecar of the base's and the fine-tune's
  per-window quantiles as float32 (`mkt_quantiles_key`, sha256 in the
  signed report). Losses are computed FROM those float32 values by one
  shared rule (`marketcast_report.score_windows`), so the audit recomputes
  every loss on CPU exactly.
- `check_marketcast` re-cuts the window set from the published snapshot and
  requires equality (no omission, no cherry-pick), recomputes every loss and
  coverage tally, and re-derives the verdict.
- `after` is now `train_time + start_margin_seconds`, with `train_time` the
  unix time the TRAINER dispatched the leg, stamped in the trainer-signed
  `LaneCEntry.marketcast` record beside `train_block` (separation of
  duties: the main validator's snapshot anchor no longer moves it; the audit
  notes an anchor more than an hour off).
- `cascade-audit marketcast <id> [--mkt-snapshots DIR] [--reproduce]`
  (`cascade/audit/marketcast_reproduce.py`): re-fetches a deterministic
  sample of the snapshot's bars from Hyperliquid's public `candleSnapshot`
  (closes within 1 bp) and, with torch + the `timestral` extra, re-forecasts
  a sample of windows from the public checkpoints (`|Δ| ≤ 1e-3 + 2e-3·|q|`
  per knot: float32 rounding, cross-GPU noise, one-window NaN padding).
  Reports; never changes consensus.
- **Decision for the owner to confirm — public fine-tune checkpoints.**
  Checkpoint visibility is not decided per checkpoint in code: harvested
  checkpoints (MarketCast's included, `trainer/loop.py`
  `upload_harvested_checkpoint`) go to the trainer's Hub namespace, which
  is public by design (`shared/hippius.py`, anonymous pulls; the HF
  fallback mirror creates public repos). DEC-CA-0036's "only thrones
  publish" governs generator CODE (`champions/`), not checkpoints. So
  judged MarketCast fine-tunes are public-read from harvest on — stronger
  than "public once the verdict is final", and what `--reproduce` needs. A
  public fine-tune cannot be re-submitted as weights (miners submit
  recipes) and a byte-identical re-train loses to the earlier commit (B5).
  Not verified against the live registry's project visibility.

### B4 — 1d horizons were unscorable; 5d dropped

`cut_windows` needs 32 finite context steps; a 21-day minute snapshot holds
21 daily bars, so every 1d window was unscorable. Measured 2026-10-19
(read-only `info` calls, BTC and `xyz:GOLD`): Hyperliquid's
`candleSnapshot` serves only the most recent ~5000 candles of an interval —
1m ≈ 3.6 days, 5m ≈ 17 days, 15m ≈ 52 days, 1h ≈ 208 days, 4h ≈ 833 days,
1d from listing (an `xyz` asset only since its listing).

- **Construction:** the snapshot carries a coarse HOURLY bank of
  `coarse_history_days` (180 < 208, fetched as `1h` candles + hourly
  funding) and every series forecast at ≥ 1h is cut from it
  (`coarse_frequency`), context and target alike. Realized vol needs ≥ 2
  returns inside a block, so it reads the bank only at ≥ 1d: daily RV is the
  std of the day's 24 hourly bp returns (one consistent construction across
  context and target; sub-daily vol stays on minute bars). The bank is
  absent from the signed body when not written (old digests unchanged).
- **Load check** (`scorability_problems`, run by `parse_marketcast`): every
  configured (suite, horizon) needs history ≥ 32 + steps at its frequency
  and a forward wait (`verdict_wait_blocks` × 12 s − `start_margin_seconds`)
  holding two aligned windows (≥ 3 × the horizon, since origins sit on the
  absolute UTC grid).
- **5d dropped** from the default `mkt:returns` horizons: a 6-day wait
  holds at most one aligned 5-day window. Raising the wait to ≥ 15 days for
  it would delay every verdict of the suite by more than two weeks, and one
  window per asset per wait is weak evidence for a per-asset cluster
  bootstrap. The shipped tomls pass the check.
- Not fixed (open item 10): a cold snapshot fills only the newest ~3.6 days
  of its 21-day minute grid, and an embargoed training range has no minute
  bars on Hyperliquid (the training snapshot now also fetches the hourly
  bank).

### B5 — slots against the pinned base; exact-duplicate dedup

- An `mkt:` judgement, pending entry and slot win carry `base_pointer =
  "<base_ref>|<base_sha256>"` (`MarketCastConfig.base_pointer`), so
  `win_beats` always falls through to the LCB comparison: a king change
  never lets a later (even copied) win take the slot. MarketCast judging no
  longer needs a king (the `no_king` early return only concerns
  non-MarketCast lane C). Legs still never expire on a king change.
- Exact identity only (DEC-CA-0008 spirit): a fine-tune whose weights'
  sha256 (`marketcast.checkpoint_sha256`) equals an entry committed EARLIER
  (any target; seniority = the commitment block as of `train_block`, then
  `train_block`, then hotkey — never a UID) is a `duplicate`. The registry
  (`LaneCState.mkt_seen`) keeps the earliest commit; an earlier-committed
  original arriving later evicts a later copy from its slot (an eviction
  row) and its carries lose at their next stage. `check_lane_c` accepts the
  `duplicate` and `infra` stages on `mkt:` rows only.

### New config keys

`coarse_history_days = 180`, `report_timeout_settlements = 8`,
`max_infra_deferrals = 8` (both tomls). New signed field:
`LaneCManifest.mkt_reports`; new record field `marketcast.train_time`;
report version 2 (`status`, `infra_reason`, `quantiles_key`,
`quantiles_sha256`, `null` losses).


## Review fixes II 2026-10-19

Second batch; still shipped OFF.

### M10 — a verdict sample spanning the period, clustered by regime

- `windows_per_cell` windows per (asset, horizon) are now spread EVENLY over
  the eligible span (`spread_select`, deterministic; ids stay grid-stable),
  not the most recent ones.
- Bootstrap clusters are **asset class × `cluster_block_hours` (6 h) time
  block** (`MktWindowRow.time_cluster`). Windows of one class in one block
  share a regime and are never independent evidence; per-asset clusters
  treated 28 correlated crypto perps as 28 independent units. Class × block
  (not block alone) keeps the cross-class information a single shock does
  not erase.
- The gain statistic weighs every HORIZON equally and, inside a horizon,
  every asset CLASS equally (`clustered_gain`, `window_weights`): `1 −
  exp(mean_h Σw·ln(ft/base) / Σw)`, same bootstrap bag for all horizons,
  percentile LCB.
- A verdict needs ≥ `min_time_blocks` (8) distinct blocks — two days of
  evidence; fewer is inconclusive (`time_blocks`) and waits.
- Defaults re-derived (the old 64 / 100 / 5 had no basis): over the 6-day
  wait there are 24 six-hour blocks; two windows per (asset, horizon, block)
  give `windows_per_cell = 48` (more windows inside one block add GPU cost,
  not evidence); `min_windows = 200`, `min_clusters = 8`. The load check
  refuses a wait shorter than `min_time_blocks × cluster_block_hours`.

### M11 — non-finite forecasts are never masked

A non-finite fine-tune quantile on any eligible window makes the entry a
deterministic `fault` (`compute_entry_rows`; the audit FAILs a report row
with a non-finite fine-tune loss). A non-finite BASE forecast leaves the
row's base loss `null`: the window is excluded for everyone and the receipt
row records `mkt.base_excluded`.

### M12 — embargo and forward-only enforced by the trainer

- At dispatch the trainer re-validates `marketcast.json` with
  `submission_time` = the commitment's time on the dispatch clock
  (`train_time − (started_block − commit_block) × 12 s`): `data.end` later
  than `embargo_days` before the commit is the miner's fault before
  anything is rented.
- The pod receives `train_time` in its recipe payload, re-checks the
  embargo against it and clips every fetched bar to before it
  (`build_training_snapshot(not_after=)`).
- **Residual risk:** the payer pod runs on the miner's OWN provider key, so
  the miner can exec into it and feed it anything. The harvest cannot
  cheaply detect post-`train_time` data use (the weights carry no data
  provenance). What bounds it: windows start after `train_time +
  start_margin_seconds`, so leaked post-dispatch data can only cover the
  first windows of a 6-day wait, and the result must still beat the base
  on windows that close long after the pod is gone. Not detected; accepted.

### M13 — trivial and illiquid wins

- (a) Windows of a price suite (`returns`, `alpha`) whose target's mean
  |value| is below `min_target_abs_bp` (0.1 bp) are not eligible — a stale /
  no-trade window's near-zero loss would blow up the ratio (the DEC-CA-0009
  lesson). `funding` changes are legitimately hundredths of a bp, so only an
  exactly flat one is out.
- (b) Liquidity floor (`liquidity_filter`): over the 24 h before a window,
  ≥ `min_bar_coverage` (0.8) live bars and ≥ `min_dollar_volume_24h`
  ($1M; 0 = off; not measured per asset — owner-tunable) traded.
- (c) Equal weight per asset class inside a horizon (M10).
- The bar remains **"beats the zero-shot base"** (the base gets no
  covariates). A SECOND, reported, non-gating baseline is computed when the
  fine-tune takes covariates: the base given the SAME covariates (Timestral
  accepts past / known covariates), per-window `bc` loss in the rows and
  `mkt.gain_vs_base_cov` in the receipt. The audit does not recompute it
  (its quantiles are not in the sidecar).

### M14 — coverage gate per horizon, two-sided

Each horizon must pass on its own (`coverage_by_horizon`); a gated level
fails below `nominal − coverage_tolerance` (0.05) AND above `nominal +
coverage_tolerance_over` (0.10: intervals that wide are useless
downstream). `coverage_gate_levels = [0.5, 0.8]`; 98 % stays report-only —
with a few hundred windows a horizon its expected misses are single digits,
too noisy to gate.

### M15 — a forward bar recorder

Hyperliquid serves only ~5000 candles per interval (1m ≈ 3.6 days), less
than the 6-day verdict wait and far less than the 21-day minute grid. The
main validator now RECORDS closed 1m bars + funding forward:
`cascade-pool mkt-record --store DIR` (`cascade/pool/marketcast_store.py`),
cron'd every few minutes, resumable (each pass re-fetches the last 48 h and
merges), gap-tolerant (an unrecorded minute stays NaN, never
interpolated), closed bars only. `cascade-pool publish --lane mkt` builds the
minute bank from it (`--bar-store DIR`) and refuses without one
(`--allow-cold` for tests). `snapshot_days` now means recorded history. The
config cannot see a recorder, so arming cannot check it: running it from
before the arming block (≥ `snapshot_days` ahead) is an operator step
(docs/OPERATOR.md). Recipe training data is unaffected by the recorder (the
store is the main validator's): an embargoed (≥ 30 days old) range has no
minute bars on Hyperliquid, so sub-hourly suites train only on what the
generator supplies, ≥ 1h ones on the hourly bank.

### New keys (both tomls)

`cluster_block_hours = 6`, `min_time_blocks = 8`, `min_target_abs_bp = 0.1`,
`min_bar_coverage = 0.8`, `min_dollar_volume_24h = 1000000.0`,
`coverage_tolerance_over = 0.10`, `coverage_gate_levels = [0.5, 0.8]`;
changed defaults `windows_per_cell = 48`, `min_windows = 200`,
`min_clusters = 8`. Lane-C side (DEC-CA-0056 §Review fixes):
`LaneCEntry.tree_digest`, `LaneCState.seen`, the `spent` driver state.

## Note 2026-10-20 — own full-pay tiers (owner: gain-tiered full-pay period)

An `mkt:<suite>` win pays through a lane-C slot (DEC-CA-0056), whose full-pay
period is now gain-tiered (DEC-CA-0056 "Gain-tiered full-pay period
2026-10-20"). MarketCast gets its OWN table, `[marketcast] full_pay_tiers`
(default `[[0.05, 10], [0.02, 5], [0.01, 2]]`, uncalibrated placeholders),
not lane C's. Both LCBs are relative loss reductions at a 0.01 margin, but
they are different statistics: lane C's is the cluster-bootstrapped
`sqrt(WQL · MASE)` gain of a fine-tune vs the KING on pool windows (sources as
clusters); MarketCast's is `clustered_gain` — a horizon-equal weighted
geomean gain on ONE suite's loss (CRPS-bp on returns, QLIKE-style vol, log
range, …) vs the pinned Timestral base, clustered by asset class × time
block. The same threshold is not the same evidence (a 2 % CRPS gain on
near-efficient returns is not a 2 % gain on a domain fine-tune), so one
calibration cannot serve both. Consequences: the key is independently
overlay-settable (DEC-CA-0057 flexibility update, 2026-10-21), its
`full_eras` must be `<= [lane_c]
win_ttl_eras` (load-checked: the slot TTL is lane C's), and the decay is lane
C's. The tier is computed at judge time (`judge_marketcast_entry` →
`LaneCJudgement.full_pay`) from the confirmed entry's gain LCB, recorded on
the `TargetWin`, and audited like any lane-C win (`check_lane_c`,
`_lane_c_paid_wins`).

## Red-team fixes 2026-10-20

A red team simulated the verdict against an ORACLE base (the true predictive
distribution: lognormal per-asset scale, lognormal per-(class, 6 h block)
regime, Student-t(3) returns, 48 windows per asset and horizon, the default
five classes and horizons). Every attack below WON or came within the
coverage gate of winning; after the fixes every one LOSES on the point gain
itself, and an honest better forecaster still wins. The sims are ported to
`tests/unit/test_marketcast_redteam.py` (24 windows per asset/horizon, B =
400). Numbers are point gain / gain LCB (margin 0.01), seeds 0–1.

| attack (vs an oracle base) | before | after |
|---|---|---|
| all knots ×0.9 / ×0.85 / ×0.8 | +2.3 / +3.4 / +4.4 % (coverage gate only) | −0.2 / −0.5 / −0.9 % |
| inner knots ×0.5, 10/25/75/90 % knots exact | **PASS**, LCB 0.0195 | −0.8 %, LCB −0.009 |
| tails → 10/90 % knots + inner ×0.5 | **PASS**, LCB 0.031 (pooled CRPS −2.9 %) | −3.0 %, LCB −0.032 |
| thin classes ×0.7 only | **PASS**, LCB 0.031 | −1.2 %, gate fails |
| thin ×0.7 + rest tails→10/90 + inner ×0.5 | **PASS**, LCB 0.047 | −2.4 % |
| mkt:vol log-RV ×0.8 / ×0.5 | +4.5 / +7.9 % (no gate) | −1.2 / −9.0 %, gate fails |
| lattice, mass moved off zero | +18.7 % (target filter) | −35 % |
| HONEST: base 30 % too wide, candidate = oracle | PASS | PASS, +1.2 / +1.4 %, LCB 0.0105 / 0.0121 |
| HONEST vol: base 40 % too wide | PASS | PASS, +3.3 %, LCB 0.030 |

### F1 — the log of a single window's CRPS is not proper

The old statistic averaged `ln(ft CRPS / base CRPS)` per window. A proper
score's MEAN is minimised by the true belief; its per-window LOG is not — a
too-narrow forecast lowers the typical per-window log loss while raising the
mean. Now (`cascade.eval.marketcast.clustered_gain_detail`) the windows'
losses are SUMMED first per (asset class × horizon) over the class's time
blocks, only that ratio of MEAN losses is logged, and the gain is the
geomean over classes (equal inside a horizon) and horizons (equal) — the
DEC-CA-0009 shape. The bootstrap still resamples whole asset class × 6 h
block clusters, the same bag for every horizon, each class ratio recomputed
from the bag's sums. Each ratio is clipped to `[1/cell_ratio_cap,
cell_ratio_cap]` (4.0), so one class's near-zero loss cannot carry a verdict.

Deviation from the red-team's suggestion (ratio per class × block × horizon
cell): measured, per-cell ratios over ~3 windows (fx at a sparse horizon)
keep a small-sample Jensen bias that still favours sharpening (×0.85:
+0.4 % point gain at 24 windows per asset/horizon; the honest gain read
+2.9 % against a pooled +1.4 %). Pooling the ratio over the class's blocks
gives −0.3 % for ×0.85 at 8 windows per asset/horizon and tracks the pooled
CRPS gain (honest +1.2 % vs +1.3 %). Cost: inside a class, a high-vol block
weighs more than a quiet one (CRPS keeps its bp units), as in pooled CRPS.

### F1b — tails and vol are gated

`COVERAGE_GATE_SUITES = ("returns", "vol")` (a constant, release-only). The
fine-tune's rows carry `tail` tallies per `tail_levels` level (default
`[0.05, 0.95]`): the finite steps below `q_0.05` / above `q_0.95`. The tail
hit rate must lie within `tail_tolerance` (0.025) of 5 % per horizon AND
pooled over horizons (equal horizon weight). The pooled row is the one with
power where a horizon is thin (1d: ~5 windows per asset in a 6-day wait); the
per-horizon rows catch a model that collapses only its 1-minute tails.
Collapsed 1/5/95/99 % knots (central gate exact) now fail at every horizon.

### F2 — calibration weighted like the gain

Rates are no longer tallied over raw windows (28 crypto perps outvoting 3 FX
pairs). Per horizon, each class's rate is its summed tallies over its summed
steps, and the horizon's rate is the MEAN over classes — the gain's
weighting. Gating per (class, horizon) instead was rejected: an fx cell at a
single-step horizon has 144 steps (3 assets × 48), so the 50 % rate's
standard error is 0.042 and a 0.05 tolerance would false-fail a calibrated
model about 24 % of the time per cell, near-certainly across 25 cells × 2
levels. Every tolerance is `max(tolerance, calibration_z × se)` with `se =
sqrt(Σ_c p(1−p)/n_c)/C` at the expected rate (`calibration_z = 3.0`), so a
thin horizon widens its own band instead of false-failing an oracle (the
oracle passes every row at seeds 0–2 in the tests). Rows record the `slack`
used. Thin-class ×0.7 now fails `1m@0.5` (class-weighted 0.42 vs pooled
0.47).

### F3 — the target never filters a window

`min_target_abs_bp` is REMOVED (the parser refuses it). It dropped windows
whose REALISED mean |y| was below 0.1 bp, so the test set was conditioned on
the outcome: a forecaster that knows the filter moves mass off zero (lattice
sim: +18.7 %). Eligibility now reads only what is known at a window's start:
the liquidity floor (`liquidity_filter`) and, for `returns` / `alpha`, a
stale CONTEXT (the last `MIN_CONTEXT` finite context values all equal:
`cut_windows(stale_context=True)`). A zero-scale CELL (class × block ×
horizon whose base mean loss is ≤ `zero_scale_frac` (1e-3) × the median cell
of its horizon) is masked from both sums. That rule reads only the base's
loss, never the candidate's forecast. The count is recorded on the receipt
row (`mkt.masked_cells`, first keys in `mkt.masked`) and the audit
re-derives it.

### F4 — behavioural near duplicates

Exact-sha dedup missed a copied public winner with one weight nudged. The
main validator now measures every computed entry against its SENIOR
references: the round's other computed entries on the same target with an
earlier `(commit block, train block, hotkey)` (never a UID, never the same
hotkey), plus the target's slot holder. It forecasts each reference on the
candidate's own windows, cached per (pointer, snapshot, window). The
distance is `mean |q_cand − q_ref| / w` over every step × knot, with `w` the
base's 5–95 % width at that step. The base is fixed, so the scale is the
same for every candidate. The closest reference's record goes into the
signed entry as `near_dup` = {hotkey, pointer, commit_block, mean_rel,
max_rel, n_windows, quantiles_key, quantiles_sha256}, and its forecasts are
published as a second sidecar `marketcast/round-<id>/r-<tag>.npy` =
`encode_quantiles(base, reference)`. Every validator applies `near_dup_tol`
(0.02): `mean_rel` below it gives stage `duplicate` (`judge_marketcast_entry`).
The audit recomputes the distance from the two sidecars, checks the base
planes agree, and checks that the stage follows the tolerance. Measured in
the validator test harness (base 3× too wide, so its width inflates the
denominator): a copy with every output nudged by one float32 ULP (2⁻²³)
measures mean 1.0e-8 / max 4.7e-8; a model scaled ×1.6 measures 0.053 (not
a duplicate); one scaled ×1.2 measures 0.018, so it IS flagged. At equal
base and candidate widths, 0.02 is roughly a 9 % rescale of the quantiles.
Uncalibrated: measure the spread between honest independent fine-tunes
before arming.

Not built: the weight-space cosine of `(w_cand − w_base)` vs
`(w_ref − w_base)`. A function-preserving permutation of hidden units (MLP
neurons, attention heads) sends it to ~0 while the forecasts stay
bit-identical, so it is weaker than the behavioural test. It would also need
both checkpoints in the audit. Residual: an adversary can perturb a copy
until it measures ≥ 2 % of the base width on average. That is a measurably
different model, which must still clear the LCB against the base and beat
the slot holder.

### F5 — the leg runs for `[lane_c] leg_hours`

`train_mkt_leg` handed the funded path the throne contract (5 h
`max_train_seconds`). It now uses `mkt_leg_contract` =
`for_hours([lane_c] leg_hours)` with the `[round]` guard knobs and the
stream-CPU corpus, so the rent, the corpus sizing and the harvest guard
agree with the worker's own `leg_hours` wall.

### F6 — sybils neither inflate k nor buy extra looks

The family is counted by OWNER. The trainer stamps the dispatching
commitment's coldkey into the signed `marketcast` record (`coldkey`), and
falls back to the hotkey when the reveal history has no coldkey (deregistered
— the metagraph is the only coldkey source, so it is stamped at dispatch,
deterministic for validators). Each judge-phase entry is judged at `α / (C ×
m)`: `C` = distinct owners on the target in the round, `m` = this owner's
entries (`mkt_family_sizes`). This is a weighted Bonferroni: the levels sum
to `α`, so one coldkey's sybils share one owner's `α / C`. The receipt row
records `mkt.owner`, and the audit re-derives `k`. The trainer also caps
open MarketCast legs per coldkey (`max_open_legs_per_coldkey = 1`, trainer
policy). A sybil with many coldkeys is not stopped by either rule.

### F7 — deterministic forward failures are faults

A `ValueError` from the fine-tune stays a fault. Now any other exception
raised INSIDE the fine-tune's forward that is not infra-shaped (`infra_like`:
MemoryError / OSError / connection / timeout, or a message naming CUDA,
cuDNN, cuBLAS, NCCL, out-of-memory, device-side, illegal memory, the
driver) becomes `MktForwardError`. When it is raised identically on every
`MKT_COMPUTE_ATTEMPTS` attempt, the entry is a `fault` ("forward failed
identically"). Anything else, including base-side failures, stays
`infra_deferred`. This is deliberately conservative: an intermittent model
bug is still infra.

### New / changed keys (both tomls)

Added `zero_scale_frac = 0.001`, `cell_ratio_cap = 4.0`, `tail_levels =
[0.05, 0.95]`, `tail_tolerance = 0.025`, `calibration_z = 3.0`,
`near_dup_tol = 0.02`, `max_open_legs_per_coldkey = 1`. Removed
`min_target_abs_bp`. The report is now version 3, adding `tail_levels`,
per-window `tail` and per-entry `near_dup`. Nothing is deployed, so there is
no compatibility shim.

## Note 2026-10-20 — MarketCast is lane C's only competition (owner)

DEC-CA-0056 "MarketCast-only lane C 2026-10-20": lane C now admits only
competitions with a forward-only eval, and MarketCast is the only one
(`COMPETITION_KINDS["mkt"]`, `forward_only=True`). The generic
`domain:` / `freq:` / `horizon:` targets and the `pool_c` pool are gone, so
the "with no `mkt_suites` the grammar is exactly the pre-MarketCast one"
rule no longer applies: a target with no admitted suites is refused with
"MarketCast admits no targets here". The shipped `[lane_c] targets` are
`["mkt:returns", "mkt:vol"]`, matching `[marketcast] suites`. The loader
admits them against the configured `suites`, so they load while
`from_block = 0` and count no active slot (lane C's share burns). Intake,
commitments and recipes still use `target_suites`, which is empty while
MarketCast is off. Nothing else here changes: the pinned base, the
`pool_mkt` snapshot, the verdict and the full-pay tiers.

## Note 2026-10-20 — MarketCast as a competition entry (concurrent competitions)

Lane C may now run several competitions at once (DEC-CA-0056 §Concurrent
competitions). MarketCast is the `mkt` entry: `[lane_c.competitions.mkt]`
sets its `weight` (shipped 1.0), optional `until_block` and wind-down rule.
Its start stays `[marketcast] from_block` (the kind registers
`from_block_table = "marketcast"`; `from_block` in its competition table is
refused), so this DEC's arming checks are unchanged. After an `until_block`
the validator's MarketCast pipeline judges nothing: new entries, carried
pending entries and backlog candidates get `ended` rows and the backlog
empties; held `mkt:` wins wind down per the competition's rule.
