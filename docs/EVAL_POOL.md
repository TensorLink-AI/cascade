# Building the held-out eval pool

Validators score the king's and challengers' trained models on a **private,
rotating pool of real-world series** (`[eval] window_pool`, a Hippius Hub
`repo@digest`). This doc covers the **producer** side — turning real data into
that pool — with `cascade.pool` and the `cascade-pool` CLI. This is an operator
guide, not a source of training data for miners. The consumer side
(fetch ref → slice into windows) lives in `cascade.validator.pool` / `.windows`.

**Workflow:** [build locally](#quick-start) → [publish a future-effective snapshot](#daily-rotation--consensus)
→ [check source coverage](#sources-shipped). MarketCast uses its own
[forward-observation snapshots](#the-marketcast-pool---lane-mkt-dec-ca-0064).

## Why this design is hard to game

The private test should be difficult to reconstruct or memorize before a
submission. Three choices reduce that risk:

1. **Privacy** — active windows stay private; retired windows can be published
   later. A public benchmark is not a replacement for this pool.
2. **Freshness** — sources harvest *recent* data up to an `as_of` cutoff; you
   rebuild periodically so snapshots change over time. A/B snapshots may
   include observations that existed before submission; they do not have
   MarketCast's strict future-window guarantee. The validator also changes
   the selected slice each settlement using the block hash.
3. **Breadth** — multiple real domains reduce the value of fitting one narrow
   distribution. These controls reduce contamination risk; they do not prove
   that no miner could reconstruct a source.

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
active. Each validator, for a settlement, computes its epoch-boundary block
(`created_block` floored to the epoch grid, from the shared manifest) and selects
the snapshot with the greatest `effective_block ≤ that block` — the **same**
deterministic choice on every validator, so two validators that polled at
different times around the daily rollover still score the *identical* pool for a
given settlement. Integrity is the tar's sha256, verified on fetch.

The sha256 lives in `pool/index.json`, which is unsigned — so on its own it
protects against corruption, not against a hostile holder of the bucket's
write credentials. The trainer therefore **pins** the settlement's snapshot into
the signed manifest (`eval_pool_key` + `eval_pool_sha256`, resolved by the
same selection rule from the trainer's pool source): each validator's own
snapshot selection must match the pinned pair or the settlement is rejected
(`pool_pin_mismatch`), so pool integrity descends from the trainer signature
validators already trust, not from storage ACLs. A manifest without a pin
falls back to trusting the index. Never publish a snapshot whose
`effective_block` is at or before a pending settlement's boundary
(`--effective-block auto` guarantees this): it would flip validators'
selection away from the pinned snapshot.

> **Why the block, not the round id.** A round id is the epoch-boundary block
> *hash* folded to a 64-bit seed (`ChainClient.block_seed`) — deliberately
> unpredictable, hence non-monotonic. Ordering snapshots by "greatest
> `effective_round ≤ round_id`" over random seeds is meaningless. The epoch
> block *number* is monotonic and every validator derives the same value from
> the manifest's `created_block`, so it is the correct consensus key. (A v1
> index keyed by `effective_round` still parses.)

**Invariant the publisher must hold:** a new snapshot's `effective_block` is in
a *future* epoch (later than the latest settlement). `--effective-block auto`
enforces this by reading the manifest `latest.json` `created_block`, flooring it
to the epoch grid, and adding `--round-buffer` epochs (default 1). Never publish
a snapshot that becomes active for an already-scored settlement, or validators
would disagree. (`--effective-round` is a deprecated alias taking a block.)

Example daily cron on the orchestrator:

```bash
# 03:00 UTC daily — fresh windows, active from the next settlement onward.
0 3 * * *  cascade-pool publish --as-of "$(date -u +\%F)" --effective-block auto
```

Validators pick up new snapshots automatically (they re-read the index each
settlement and fetch a snapshot once, cached by digest). No restart, no `chain.toml`
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

1. The forge repo's scheduled scrape workflow (`.github/workflows/scrape-data.yml`)
   ends with `src/sources/sync_storage.py`: a boto3 mirror of `data/` +
   `sources.yaml` to the **private** Hippius bucket `tsbench-forge-sources`
   (endpoint `https://s3.hippius.com`, credentials `HIPPIUS_S3_ACCESS_KEY` /
   `HIPPIUS_S3_SECRET_KEY`). That bucket is the one canonical raw-data relay;
   the forge repo's `scripts/publish_data_bucket.sh` is only for self-managed
   alternatives and must then target the same bucket the step below syncs.
2. The publish cron here syncs that bucket down and points the source at the
   mirror before building:

```bash
: "${TSFORGE_BUCKET:=tsbench-forge-sources}" "${TSFORGE_S3_ENDPOINT:=https://s3.hippius.com}"
AWS_ACCESS_KEY_ID=$HIPPIUS_S3_ACCESS_KEY AWS_SECRET_ACCESS_KEY=$HIPPIUS_S3_SECRET_KEY \
  aws s3 sync "s3://$TSFORGE_BUCKET" ./tsforge --endpoint-url "$TSFORGE_S3_ENDPOINT" --exact-timestamps
TSFORGE_DIR=./tsforge cascade-pool publish --sources tsbench_forge --effective-block auto
```

Only the owner orchestrator holds forge-bucket credentials. Validators never
touch this bucket (or need the forge repo at all) — their sole data dependency
is the built pool snapshot published downstream.

Install the producer extra first: `pip install "cascade[pool-forge]"` (pyarrow
+ pyyaml + pandas; validators never need it — they consume the built `.npy`
pool). Everything downstream — deterministic tar, `effective_block` consensus,
validator fetch — is unchanged.

#### Multivariate coupling — the `mv_channels` contract (DEC-CA-0041)

A source may tag genuinely coupled channels for multivariate scoring via
`mv_channels` in its `sources.yaml` entry. This is a **cross-repo interface**,
versioned here (v1):

> **`mv_channels`** — an ordered list (≤ 8) of value-column names within a
> single `(source, panel_row)`, all on that row-group's shared timestamp index.
> The group packs to one `(C, L)` eval window and counts as **one series**
> against every cascade cap: the 200-series panel-expansion cap, the
> `[eval] n_windows` draw, and — since the KOTH bootstrap clusters by `source`
> — it adds no cluster beyond its source. Cross-predictiveness is verified by
> forge's audit: a group is tagged only if a sibling channel's *lags* improve a
> held-out forecast of a channel beyond its own lags (contemporaneous
> correlation alone does not admit).

**Packing.** `cascade-pool publish` packs `mv_channels` groups into `(C, L)`
windows iff the snapshot's `effective_block` is at or past `[scoring]
mv_score_from_block` (`--no-mv-pack` forces off, `--mv-pack` forces on), so the
pool and the multivariate scoring rule switch on the same boundary by
construction. `--max-channels` is raised to the `mv_channels` cap (8)
automatically when packing.

**Two consumers, deliberately different accounting.** Forge's own benchmark
expands a coupled group into C univariate challenges (C slots, group-level
bootstrap); cascade packs it to one window (1 slot, source-level clusters). The
same tag serves both — do not "fix" one to match the other.

Because a coupled group is free against every cascade cap (1 series, 1 forecast
call, 0 extra clusters), tagging costs the pool nothing and there is no
size-vs-evidence tension; the cross-predictiveness bar is the only gatekeeper.
Panel width buys no evidence — a 200-station network is one cluster — so harvest
breadth is counted in **distinct coupled sources** (≥ ~12 / ~38 % of the pool;
headroom toward `min_clusters ≈ 30`), never in channels or series.

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
which clears `[eval] n_windows` with margin after validation drops and leaves
the pool larger than any one draw, so each settlement scores a *different*
slice — a jittered domain mix of `[eval] mix_target_windows` windows
(DEC-CA-0019, DEC-CA-0032) — on top of the daily snapshot rotation.

**Scale the pool by data, not code**: make the Open-Meteo grid denser
(`global_grid(lat_step, lon_step)`), extend `VARIABLES` / `ARTICLES`, or add a
source. One Open-Meteo call per grid point returns all variables, so a denser
grid is the cheapest lever. The CLI warns if the pool falls below
`[scoring] min_windows`. (At ~252 grid points that's ~252 API calls + ~85 for
Wikimedia per daily build — well within the keyless free tiers.)

### v2 primary pool (DEC-CA-0050)

tsbench-forge is already the only source the production publish cron
(`.github/workflows/publish-pool.yml`) builds from, so making it the v2
primary pool is an operator step plus two build-time checks — no validator
change. Lane B judges each leg per domain, so v2 adds:

* `--require-domain` — drop series whose catalog entry carries no domain
  (the forge source labels those `unknown`); counted as `missing_domain` in
  provenance drops. Every window a lane-B verdict reads then has a real label.
* `--min-series-per-domain N` — the DEC-CA-0032 Part A capacity floor: the
  build REFUSES to write a snapshot in which any labeled domain holds fewer
  than `N` series (one window candidate each). v2 uses 300. A refused build
  publishes nothing, so the previous snapshot stays current — fix the
  catalog (add feeds to the thin domain), never the floor, per DEC-CA-0032.
* `--require-source` — the CLI switch for the existing `require_source`
  builder rule (every series carries its cluster key); turn it on with the
  other two.

Operator steps:

1. Add the three flags to the `cascade-pool publish` call in
   `publish-pool.yml`:

   ```bash
   TSFORGE_DIR=./tsforge cascade-pool publish \
     --chain-toml "$TOML" --sources tsbench_forge \
     --max-panel-series-per-feed 5 --max-series-per-domain-freq 300 \
     --max-series-per-domain 600 --selection balanced \
     --require-domain --require-source --min-series-per-domain 300 \
     --effective-block "$EFFECTIVE_BLOCK"
   ```

2. Dry-run first with `cascade-pool build --out /tmp/v2pool ...` (same flags)
   against the current mirror and read the `by domain:` line — any domain under
   300 must be fixed in the forge catalog before the cron flips.
3. Raise `[scoring] min_clusters` toward ~30 (see the cluster-labels note
   above) in the same coordinated release that arms lane B.

Signed window scores now carry each window's `domain`
(`WindowScoreRecord.domain`, drop-when-default), so per-domain statistics and
lane-B verdicts can be replayed from a receipt alone
(`cascade.eval.expert_lane.judge_expert_leg` over the paired signed scores).

## The MarketCast pool (`--lane mkt`, DEC-CA-0064)

Lane C has no pool of its own: the DEC-CA-0056 public-feed `pool_c` (and
`cascade-pool publish --lane c`) is retired, because its windows could be
scraped and memorised. Lane C's only competition, MarketCast, reads a
separate signed snapshot of realized market bars under `[marketcast]
pool_prefix` (`pool_mkt`), built by the main validator (`[netconfig]
owner_hotkey`):

```bash
*/1 * * * *  cascade-pool mkt-record-alpha --out /srv/cascade/alpha.jsonl   # one dTAO sample per subnet
*/5 * * * *  cascade-pool mkt-record --store /srv/cascade/bars              # closed Hyperliquid 1m bars + funding
30 */6 * * * cascade-pool publish --lane mkt --alpha-rows /srv/cascade/alpha.jsonl \
                 --bar-store /srv/cascade/bars --wallet-name owner --wallet-hotkey main
50 5 * * *   cascade-pool reveal  --lane mkt --hf-repo <org>/cascade-eval-pool
```

- `mkt-record` must run forward from at least `[marketcast] snapshot_days`
  (21) before arming: Hyperliquid serves only ~3.6 days of 1m candles, so
  `publish --lane mkt` refuses to run without `--bar-store` (`--allow-cold`
  builds from the API alone, for testing only). Each pass re-fetches the last
  `--lookback-hours` (at most ~83 h, the API's 1m depth), so a missed pass
  inside that window is healed.
- `publish --lane mkt` signs the snapshot with the given hotkey
  (`--wallet-*`) and refuses a signer other than `[netconfig] owner_hotkey`
  (validators would refuse it); `--effective-block` works as for lane A, and
  `--anchor-block` (default: the chain head) and `--snapshot-days` override
  what it reads from the chain and `[marketcast]`.

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
  `series_id` → `{freq, seasonal_period, domain}` with `source` and `dgp_class`
  when supplied, plus a `provenance.json` the
  loader ignores.
- **Determinism**: same harvested inputs ⇒ byte-identical directory ⇒ the same
  content pushed to the same Hub repo yields the same OCI `repo@digest`. Re-build
  to audit.
- **Cleaning**: gaps interpolated per channel (series dropped above
  `--max-missing-frac`), tails truncated under the configured keep length,
  short/constant/invalid series dropped, and exact duplicates removed.
  Multichannel series are allowed up to the effective `max_channels` limit;
  a coupled group is not rejected merely for having several channels.
