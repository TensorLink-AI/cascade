# Building the held-out eval pool

Validators score the king's and challenger's trained models on a **private,
rotating pool of real-world series** (`[eval] window_pool`, a Hippius registry
CID). This doc covers the **producer** side — turning real data into that pool —
with `metronome.pool` and the `metronome-pool` CLI. The consumer side (fetch CID
→ slice into windows) lives in `metronome.validator.pool` / `.windows`.

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
   `metronome-pool publish` on a cron; validators pull the current snapshot from
   `[storage] pool_bucket` with **no `chain.toml` edit**. This is how the pool
   *rotates in time* — see "Daily rotation & consensus" below.
2. **Static CID.** `metronome-pool build --upload` pins one snapshot's CID in
   `[eval] window_pool`. Simple, but refreshing the data means editing
   `chain.toml` + redeploying. Use it for a fixed pool or local testing.

If `[storage] pool_bucket` is set, the validator uses the bucket; otherwise it
falls back to the static CID.

## Quick start

```bash
# Offline smoke test (no network): synthetic series through the full build path.
metronome-pool build --out ./pool --sources synthetic --overwrite

# One-off static pool: build + pin a CID.
metronome-pool build --out ./pool --upload
# → prints  window_pool = "bafy…"   ← paste into [eval] in chain.toml

# Daily publish: build + push a snapshot to the pool bucket (no chain.toml edit).
metronome-pool publish --effective-round auto

metronome-pool sources   # list registered sources
```

Window geometry (`context_length` / `horizon`) defaults to `[eval]` in
`chain.toml`, so the pool matches what the validator expects.

## Daily rotation & consensus

`metronome-pool publish` builds a fresh pool, packs it to a deterministic tar,
uploads it to the pool bucket, and appends it to `pool/index.json` stamped with
an **`effective_round`**. Each validator, for a round at `round_id`, selects the
snapshot with the greatest `effective_round ≤ round_id` — the **same**
deterministic choice on every validator, so two validators that polled at
different times around the daily rollover still score the *identical* pool for a
given round (no latest-wins divergence). Integrity is the tar's sha256, verified
on fetch.

**Invariant the publisher must hold:** a new snapshot's `effective_round` is in
the *future* (greater than the current round). `--effective-round auto` enforces
this by reading the manifest `latest.json` round_id and adding `--round-buffer`
(default 1). Never publish a snapshot that becomes active for an already-scored
round, or validators would disagree.

Example daily cron on the orchestrator:

```bash
# 03:00 UTC daily — fresh windows, active from the next round onward.
0 3 * * *  metronome-pool publish --as-of "$(date -u +\%F)" --effective-round auto
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
pool_bucket      = "metronome-eval-pool"
pool_s3_endpoint = "https://<account>.r2.cloudflarestorage.com"
pool_s3_region   = "auto"
```

and provide `POOL_S3_ACCESS_KEY` / `POOL_S3_SECRET_KEY` (an R2 token). When the
`POOL_S3_*` env is unset, the pool store falls back to the `HIPPIUS_S3_*`
credentials, so a Hippius-only operator needs nothing extra.

## Sources (shipped)

| name        | domain      | freq | seasonality | notes |
|-------------|-------------|------|-------------|-------|
| `openmeteo` | weather     | H    | 24 (daily)  | keyless archive API; backbone, fills full context |
| `wikimedia` | web_traffic | D    | 7 (weekly)  | keyless REST API; shorter-context breadth |
| `synthetic` | synthetic   | H    | 24          | offline/testing only — **not** for a real pool |

The shipped location/article lists are a starting seed. **Scale the pool by
data, not code**: extend `LOCATIONS` / `VARIABLES` / `ARTICLES`, raise
`--max-series-*`, or add a source. Aim comfortably above `[scoring] min_windows`
(and ideally `[eval] n_windows`) — the CLI warns if the pool is too small.

## Adding a source

Implement the `DataSource` protocol (`metronome/pool/source.py`) — a `name` and
`harvest(fetch, ctx) -> Iterable[HarvestedSeries]` — and register it in
`metronome/pool/sources/__init__.py`. Do **all** network I/O through the injected
`fetch` callable so the source is unit-testable against canned JSON (see
`tests/unit/test_pool_sources.py`). Yield raw series with a pandas-style `freq`;
the builder handles cleaning, gap-fill, length normalisation, degeneracy/dup
filtering, and seasonality.

Good keyless candidates to add: ENTSO-E / EIA grid load (hourly), USGS water
services (hydrology, hourly), air-quality (OpenAQ). Crypto OHLCV is huge and
future-unknowable but near-random-walk, so use it as a minor domain only.

## What the builder guarantees

- **On-disk format** matches `metronome.validator.pool` exactly: one
  `<series_id>.npy` per series (float32; loader upcasts), `metadata.json` keyed by
  `series_id` → `{freq, seasonal_period, domain}`, plus a `provenance.json` the
  loader ignores.
- **Determinism**: same harvested inputs ⇒ byte-identical directory ⇒ identical
  Hippius CID (the registry packs a sorted, zeroed-metadata tar). Re-build to
  audit.
- **Cleaning**: gaps interpolated (series dropped above `--max-missing-frac`),
  truncated to the freshest `context_length + horizon` points, short/constant/
  multi-channel series dropped, exact duplicates de-duplicated.
