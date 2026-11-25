# W6b acceptance tests: `pool_d` history to 16384 + 64, timestamps, `frontier`, `feeds`

Plan rows: §3.0 **W6b** (rev3 extended history **to 16384 + 64** and
per-series (per-channel) `end_ts` / `first_target_ts`; rev6.1 `frontier` for
**every** series and a `feeds` entry for **every** feed, a `pool_d` missing
either is **invalid**; rev6.10 the effective per-feed `embargo_hours`; rev6.12
no sub-hourly zero; "`pool` bytes unchanged"), §1.7 "`pool_d` (D12)" and
"Metadata". Depends on W6a (selection), W6fs (first-seen) and W2e. Runs in
`builder`; Accept is `scripts/laned/accept W6b --evidence evidence/w6b.json`.

These tests are the spec. The builder may touch only `PATHS`:
`cascade/pool/pool_d_build.py` (new). On main every test fails at its first
import of `cascade.pool.pool_d_build`: each test imports it **before**
`cascade.pool.pool_d_select` (W6a) and `cascade.pool.first_seen` (W6fs).
Fixtures are a synthetic forge mirror (`sources.yaml` + dated parquet, the
`TsbenchForgeSource` contract), recorded by the W6fs recorder, selected by
the W6a join; no bucket, no network.

**W2e note.** `[lane_d] embargo_hours` (W2e) is not on main yet. The builder
reads it through the existing `[lane_d]` loader result with the plan default:
`lane_d_embargo_hours(lane_d_cfg)` is `lane_d_cfg.embargo_hours` when the
parsed config carries it, else **48.0** (rev6.10 (a), decided). When W2e
lands, the same function reads the parsed key; no test here changes.

`test_evidence_*` **skip** when `$LANED_EVIDENCE` is unset. L4 needs
`accept W6b --evidence evidence/w6b.json` green on main, with the file
written on the `pool_d` builder by the CLI below.

## Contract (`cascade/pool/pool_d_build.py`)

| Name | Contract |
|---|---|
| `POOL_D_SCHEMA` / `BUILDER_VERSION` / `HISTORY_LENGTH` / `DEFAULT_EMBARGO_HOURS` | `"pool_d/1"` / `"cascade-pool-d/1"` / `16384` / `48.0` |
| `PoolDBuildConfig` | frozen; `history_length=16384`, `horizon=64`, `min_context=256`, `max_missing_frac=0.2`, `min_std=1e-8`, `embargo_hours=48.0`; `PoolDBuildConfig.from_lane_d(lane_d_cfg)` takes `horizon` and `embargo_hours` from the parsed `[lane_d]` table |
| `lane_d_embargo_hours(lane_d_cfg)` | see the W2e note; refuses (`ValueError`) a negative or non-finite value |
| `effective_embargo_hours(entry, *, global_hours, publish_ahead)` | `0.0` when `publish_ahead`; else `min(entry["embargo_hours"], global_hours)` when the catalog entry carries one, else `global_hours`; `None` (fail closed) for a negative, non-finite, bool or non-numeric catalog value |
| `build_pool_d(*, forge_dir, selection, first_seen_dir, as_of_ts, out_dir, cfg=None, prior_dirs=(), overwrite=False)` | writes one `pool_d/1` snapshot; returns `PoolDSummary` (`snapshot`, `sha256`, `n_series`, `n_feeds`, `refused` = `{pool_d_series_id: reason}`, `out_dir`); raises `ValueError` if the result fails `validate_pool_d` |
| `validate_pool_d(pool_dir)` | list of problem strings (empty = valid) |
| `snapshot_sha256(pool_dir)` | equals `cascade.validator.lane_d.load_pool(pool_dir, h)["sha256"]` |
| `main(argv)` | the builder CLI (below) |

### What a snapshot holds

The directory has exactly `metadata.json`, `hist.npz` and `tgt.npz`, written
deterministically (same inputs ⇒ same bytes, whatever the wall clock: no zip
timestamps, sorted keys). `hist.npz` / `tgt.npz` hold one float32 array per
`series_id`: the history (up to `history_length` points, gaps interpolated)
and the target (the last `horizon` points, never interpolated).

`metadata.json`:

- top level: `schema`, `snapshot` (`f"pool_d-{as_of_ts}"`), `as_of_ts`,
  `horizon`, `history_length`, `builder_version`, `catalog_sha256` (sha256 of
  the `sources.yaml` bytes read), `embargo_hours_global`, `selection_sha256`
  (W6a), `first_seen_roots` (sorted, the committed W6fs roots of every point
  used), `feeds`, `series`, `refused` (sorted `[pool_d_series_id, reason]`);
- per series (one per selected channel): `series_id` (the W6a
  `pool_d_series_id`), `pool_series_id`, `source` (the lane A cluster key),
  `feed`, `panel`, `channel`, `column`, `freq`, `seasonal_period`, `domain`,
  `history_len`, `target_ts` and `first_seen_ts` (`horizon` ints each: data
  time and W6fs first-seen per target point), `first_target_ts`
  (= `target_ts[0]`), `end_ts` (= `target_ts[-1]`), `frontier` (a list,
  possibly empty: `[as_of_ts, end_ts]` for each earlier snapshot in
  `prior_dirs` holding the series, oldest first);
- per feed of any series: `feeds[feed] = {"mirror_lag_hours" (W6fs, number or
  null), "max_lead_hours", "publish_ahead", "forecast_product" (catalog bool,
  default false), "embargo_hours" (effective, above)}`.

Only points the W6fs index knows at `as_of_ts` are used (committed periods
ending by `as_of_ts`), so every `first_seen_ts` is `< as_of_ts`.

Refused selections (typed, in `refused`): `rollup_not_supported` (a `.rollup_*`
series), `bad_embargo` (the feed's catalog `embargo_hours` fails closed: the
feed is left out of `feeds` and its series out of the snapshot, so the
snapshot stays valid), `target_gap` (a non-finite value in the target),
`not_in_forge`, `too_short`, `too_much_missing`, `degenerate`.

Event / irregular feeds (#841, testnet 2026-11-24): data timestamps are
compared in epoch seconds, so sub-second events can repeat a timestamp. A
series whose repeated timestamps all carry the same value (and first-seen
time) is de-duplicated deterministically (first row of each run kept); any
disagreement refuses that series as `duplicate_ts_conflict`. A series cut
that would still break a per-series invariant `validate_pool_d` checks is
refused per series (`target_ts_not_increasing`, `first_seen_not_before_as_of`)
and the rest of the snapshot is built; it never voids the whole snapshot.
A feed without repeats is cut byte-identically to before.

`validate_pool_d` reports (at least): a series with no `frontier` (absent or
`null`), a series whose feed has no `feeds` entry, a feed entry missing
`embargo_hours` / `forecast_product` / `mirror_lag_hours`, an `embargo_hours`
above `embargo_hours_global` or non-zero on a `publish_ahead` feed, a
`target_ts` / `first_seen_ts` not of length `horizon`, an `end_ts` /
`first_target_ts` inconsistent with `target_ts`, a `first_seen_ts` not below
`as_of_ts`, and a missing history or target array.

### CLI (the owner's builder run)

```
python -P -m cascade.pool.pool_d_build --forge-dir D --first-seen-dir L \
    --lane-a-pool P --out OUT --as-of-ts T [--prior-dir PRIOR ...] \
    [--max-panel-series-per-feed 5] [--mv-pack] [--chain-toml chain.toml] \
    [--evidence-out evidence/w6b.json]
```

`--lane-a-pool` is an unpacked lane A snapshot (its `metadata.json`,
`*.npy` and `provenance.json`). The CLI rebuilds lane A from the same forge
mirror with that snapshot's recorded `PoolBuildConfig` (and `as_of`), using
W6a's `collect_with_selection`, and refuses (exit 3, message contains
`lane_a_replay_mismatch`) unless the rebuilt `metadata.json` and every
`*.npy` are byte-identical to `--lane-a-pool`. Only then is the W6a selection
used to build `pool_d`.

`--evidence-out` writes `{"schema": "laned-evidence/w6b/1", "wp": "W6b",
"source": "builder", "generated_by": "cascade.pool.pool_d_build", "host",
"generated_ts", "snapshot", "as_of_ts", "sha256", "n_series", "n_feeds",
"history_length", "max_history_len", "problems": [], "lane_a_unchanged":
true, "lane_a_pool_sha256", "catalog_sha256", "refused": {reason: count},
"digest"}`; `lane_a_pool_sha256` = sha256 of the sorted lines
`f"{name} {sha256}"` over `metadata.json` and every `*.npy`; `digest` = sha256
of the canonical JSON (sorted keys, `(",", ":")` separators) of the document
without `digest`.

### Unreadable partitions (2026-11-25, `test_w6b_quarantine.py`)

A partition the W6fs quarantine names at `as_of_ts` is never read, even once
it is readable again; any other partition that fails to read is skipped and
logged to stderr. The metadata gains `quarantine` (the W6fs
`quarantine_report(until_ts=as_of_ts)`: `kind`, `feed`, `partition`,
`partition_sha256`, `size`, `quarantined_ts`, plus `reason` or
`n_blocked_points`) and `unreadable_partitions` (`[[feed, partition,
reason], ...]`, sorted) ONLY when non-empty, so a build without them is
byte-identical to before; the evidence document carries the same two keys
under the same rule. More unreadable partitions than the W6fs
`unreadable_limit` over the files read (`PoolDBuildConfig.max_unreadable_files`
/ `max_unreadable_frac`, CLI `--max-unreadable-*`) aborts with
`too_many_unreadable`. Validators ignore both keys (`load_pool` reads named
fields only).

## What is pinned

| Row text | Test |
|---|---|
| history to 16384 + 64 | `test_history_extended_to_16384_plus_64`, `test_short_series_keeps_what_it_has` |
| per-series (per-channel) `end_ts` / `first_target_ts` | `test_target_timestamps_and_first_seen` |
| first-seen never later than the snapshot | `test_points_after_as_of_are_not_used` |
| `frontier` for every series, possibly empty | `test_frontier_every_series_and_prior_snapshots` |
| `feeds` entry for every feed (`mirror_lag_hours`, `forecast_product`) | `test_feeds_entry_for_every_feed` |
| invalid if any `frontier` or `feeds` entry missing | `test_validate_refuses_missing_frontier_and_feed` |
| effective `embargo_hours` (catalog, clamped, global, 0 publish-ahead) | `test_effective_embargo_hours_rule`, `test_feed_embargo_hours_pinned` |
| negative / non-numeric catalog value fails closed | `test_bad_catalog_embargo_fails_closed` |
| rev6.12: no sub-hourly zero | `test_no_sub_hourly_zero` |
| W2e through the existing loader | `test_global_embargo_from_lane_d_loader` |
| `pool` bytes unchanged | `test_lane_a_pool_bytes_unchanged` |
| deterministic, content-addressed, re-derivable | `test_build_is_deterministic`, `test_loads_with_lane_d_loader` |
| W6fs roots + verify | `test_first_seen_roots_and_verify` |
| typed refusals | `test_rollup_and_target_gap_refused` |
| event feeds: per-series refusal, dedupe (#841) | `test_w6b_event_feeds.py` |
| builder CLI + lane A replay guard | `test_cli_builds_with_lane_a_replay`, `test_cli_refuses_lane_a_mismatch` |
| evidence from the real run | `test_evidence_*` |
