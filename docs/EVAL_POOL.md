# Building the held-out eval pool

Validators score the king's and challenger's trained models on a **private,
rotating pool of real-world series** (`[eval] window_pool`, a Hippius Hub
`repo@digest`). This doc covers the **producer** side — turning real data into
that pool — with `cascade.pool` and the `cascade-pool` CLI. The consumer side
(fetch ref → slice into windows) lives in `cascade.validator.pool` / `.windows`.

## Why this design is hard to game

The eval is only contamination-resistant if a miner's generator cannot
**distribution-match** the eval set and the trained model cannot **memorise** it.
The pool leans on three levers:

1. **Privacy** — owner-controlled, never published as a named public benchmark.
2. **Freshness** — sources harvest *recent* data up to an `as_of` cutoff; you
   re-build periodically so each pinned pool rotates in time. Data that didn't
   exist at submission can't be matched or memorised. (The validator also rotates
   the *slice* per round via the block hash.)
3. **Breadth** — multiple real domains at sub-daily frequency, so the only way to
   score well is to forecast generally, not to fit one distribution.

Do **not** point the pool at a fixed public benchmark (GIFT-Eval, Monash, …):
those are the easiest thing for a generator to overfit, and they overlap what
time-series foundation models pretrain on.

## Two ways to ship the pool

1. **Daily publish to a bucket (recommended).** The owner orchestrator runs
   `cascade-pool publish` on a cron; validators pull the current snapshot from
   `[storage] pool_bucket` with **no `chain.toml` edit**. This is how the pool
   *rotates in time* — see "Daily rotation & consensus" below.
2. **Static ref.** `cascade-pool build --upload` pins one snapshot's Hub
   `repo@digest` in `[eval] window_pool`. Simple, but refreshing the data means
   editing `chain.toml` + redeploying. Use it for a fixed pool or local testing.

If `[storage] pool_bucket` is set, the validator uses the bucket; otherwise it
falls back to the static ref.

## Quick start

```bash
# Offline smoke test (no network): synthetic series through the full build path.
cascade-pool build --out ./pool --sources synthetic --overwrite

# One-off static pool: build + pin a Hub ref.
cascade-pool build --out ./pool --upload --hub-repo cascade/eval-pool
# → prints  window_pool = "cascade/eval-pool@sha256:…"  ← paste into [eval] in chain.toml

# Daily publish: build + push a snapshot to the pool bucket (no chain.toml edit).
cascade-pool publish --effective-block auto

cascade-pool sources   # list registered sources
```

Window geometry (`context_length` / `horizon`) defaults to `[eval]` in
`chain.toml`, so the pool matches what the validator expects.

## Daily rotation & consensus

`cascade-pool publish` builds a fresh pool, packs it to a deterministic tar,
uploads it to the pool bucket, and appends it to `pool/index.json` stamped with
an **`effective_block`** — the epoch-boundary block from which the snapshot is
active. Each validator, for a round, computes that round's epoch-boundary block
(`created_block` floored to the epoch grid, from the shared manifest) and selects
the snapshot with the greatest `effective_block ≤ that block` — the **same**
deterministic choice on every validator, so two validators that polled at
different times around the daily rollover still score the *identical* pool for a
given round (no latest-wins divergence). Integrity is the tar's sha256, verified
on fetch.

> **Why the block, not the round id.** A round id is the epoch-boundary block
> *hash* folded to a 64-bit seed (`ChainClient.block_seed`) — deliberately
> unpredictable, hence non-monotonic. Ordering snapshots by "greatest
> `effective_round ≤ round_id`" over random seeds is meaningless. The epoch
> block *number* is monotonic and every validator derives the same value from
> the manifest's `created_block`, so it is the correct consensus key. (Pool
> index schema v2 made this switch; a v1 index keyed by `effective_round` still
> parses — a redeploy republishes under `effective_block`.)

**Invariant the publisher must hold:** a new snapshot's `effective_block` is in
a *future* epoch (later than the current round). `--effective-block auto` enforces
this by reading the manifest `latest.json` `created_block`, flooring it to the
epoch grid, and adding `--round-buffer` epochs (default 1). Never publish a
snapshot that becomes active for an already-scored round, or validators would
disagree. (`--effective-round` remains as a deprecated alias; its value is now a
block.)

Example daily cron on the orchestrator:

```bash
# 03:00 UTC daily — fresh windows, active from the next round onward.
0 3 * * *  cascade-pool publish --as-of "$(date -u +\%F)" --effective-block auto
```

Validators pick up new snapshots automatically (they re-read the index each
round and fetch a snapshot once, cached by digest). No restart, no `chain.toml`
change.

### Backend: Hippius S3 or Cloudflare R2

The publisher and validators talk to one S3-compatible bucket. Defaults use the
Hippius S3 endpoint + `HIPPIUS_S3_*` credentials. To use R2 instead, set in
`chain.toml`:

```toml
[storage]
pool_bucket      = "cascade-eval-pool"
pool_s3_endpoint = "https://<account>.r2.cloudflarestorage.com"
pool_s3_region   = "auto"
```

and provide `POOL_S3_ACCESS_KEY` / `POOL_S3_SECRET_KEY` (an R2 token). When the
`POOL_S3_*` env is unset, the pool store falls back to the `HIPPIUS_S3_*`
credentials, so a Hippius-only operator needs nothing extra.

## Sources (shipped)

| name            | domain          | freq       | seasonality | notes |
|-----------------|-----------------|------------|-------------|-------|
| `openmeteo`     | weather         | H          | 24 (daily)  | keyless archive API; global grid (~252 pts × 12 vars), fills full context |
| `wikimedia`     | web_traffic     | D          | 7 (weekly)  | keyless REST API; ~85 articles, shorter-context breadth |
| `tsbench_forge` | 7 GIFT domains  | 30S–D      | per cadence | reads a synced [tsbench-forge](https://github.com/tensorlink-dev/TSBench-Forge) scraper mirror (~90 feeds, 36 DGP classes); needs the `pool-forge` extra |
| `synthetic`     | synthetic       | H          | 24          | offline/testing only — **not** for a real pool |

### tsbench-forge (bucket relay)

The broadest source is the tsbench-forge live catalog: ~90 verified,
daily-or-faster public feeds across the 7 GIFT-Eval domains, curated with a
contamination denylist and novelty vetting. The coupling is the scraper's
on-disk contract (`sources.yaml` + `data/<source_id>/<YYYY-MM-DD>.parquet`) —
cascade never imports forge code. The deployment is a two-bucket relay:

```
forge scrape host                owner orchestrator                    validators
scraper cron ─► raw-data bucket ─► cascade-pool publish ─► pool bucket ─► fetch by
                (parquet+catalog)  (sync, build, validate)  (tar+index)   effective_block
```

1. The forge cron ends with `scripts/publish_data_bucket.sh` (in the forge
   repo): an `aws s3 sync` of `data/` + `sources.yaml` to a **private** bucket.
2. The publish cron here syncs that bucket down and points the source at the
   mirror before building:

```bash
aws s3 sync "s3://$TSFORGE_BUCKET" ./tsforge --exact-timestamps
TSFORGE_DIR=./tsforge cascade-pool publish --sources tsbench_forge --effective-block auto
```

Install the producer extra first: `pip install "cascade[pool-forge]"` (pyarrow
+ pyyaml + pandas; validators never need it — they consume the built `.npy`
pool). Everything downstream — deterministic tar, `effective_block` consensus,
validator fetch — is unchanged.

Properties worth knowing:

* **Freshness cutoff** — only snapshots dated `<= as_of` are read and rows
  stamped after `as_of` are dropped, so a rebuild against the same mirror is
  reproducible (the dated bucket pins the audit inputs).
* **Staleness guard** — a feed whose newest snapshot is older than
  `max_stale_days` (default 4) before `as_of` is skipped; if *every* feed is
  stale the build fails loudly instead of republishing old data. With the crons
  decoupled, a dead scraper must not silently erode the freshness lever.
* **Cluster labels** — every series lands in `metadata.json` with
  `source = <catalog id>`. The validator's KOTH bootstrap resamples these
  *clusters* rather than windows (windows from one feed are correlated), and
  `[scoring] min_clusters` sets a breadth floor on the verdict. Raise it to
  ~30 once this source feeds the pool.
* Weekly-and-slower catalog entries, `disabled` feeds, and categorical feeds
  are skipped; panel-expanded feeds are capped at 200 series per source.

The defaults produce **~3000 raw series** (Open-Meteo's global grid dominates),
which clears `[eval] n_windows = 2000` with margin after validation drops — and
leaves the pool larger than `n_windows`, so each round draws a *different* 2000
slice (intra-day rotation), on top of the daily snapshot rotation.

**Scale the pool by data, not code**: make the Open-Meteo grid denser
(`global_grid(lat_step, lon_step)`), extend `VARIABLES` / `ARTICLES`, or add a
source. One Open-Meteo call per grid point returns all variables, so a denser
grid is the cheapest lever. The CLI warns if the pool falls below
`[scoring] min_windows`. (At ~252 grid points that's ~252 API calls + ~85 for
Wikimedia per daily build — well within the keyless free tiers.)

## Adding a source

Implement the `DataSource` protocol (`cascade/pool/source.py`) — a `name` and
`harvest(fetch, ctx) -> Iterable[HarvestedSeries]` — and register it in
`cascade/pool/sources/__init__.py`. Do **all** network I/O through the injected
`fetch` callable so the source is unit-testable against canned JSON (see
`tests/unit/test_pool_sources.py`). Yield raw series with a pandas-style `freq`;
the builder handles cleaning, gap-fill, length normalisation, degeneracy/dup
filtering, and seasonality.

Good keyless candidates to add: ENTSO-E / EIA grid load (hourly), USGS water
services (hydrology, hourly), air-quality (OpenAQ). Crypto OHLCV is huge and
future-unknowable but near-random-walk, so use it as a minor domain only.

## What the builder guarantees

- **On-disk format** matches `cascade.validator.pool` exactly: one
  `<series_id>.npy` per series (float32; loader upcasts), `metadata.json` keyed by
  `series_id` → `{freq, seasonal_period, domain}`, plus a `provenance.json` the
  loader ignores.
- **Determinism**: same harvested inputs ⇒ byte-identical directory ⇒ the same
  content pushed to the same Hub repo yields the same OCI `repo@digest`. Re-build
  to audit.
- **Cleaning**: gaps interpolated (series dropped above `--max-missing-frac`),
  truncated to the freshest `context_length + horizon` points, short/constant/
  multi-channel series dropped, exact duplicates de-duplicated.
