# W6fs acceptance tests: first-seen recording (launch-blocking)

Plan rows: §3.0 **W6fs** (rev3 first-seen recording; rev5 B12 attestation;
rev6.10 release-time record; rev6.11 (d) "the W6fs first-seen Merkle root is
committed **per 3-hourly period**, not per day, so a snapshot never uses
records whose root isn't committed yet"). Runs in `builder`; Accept is
`scripts/laned/accept W6fs --evidence evidence/w6fs.json`.

These tests are the spec. The builder may touch only `PATHS`:
`cascade/pool/first_seen.py` (new). On main every test fails at its first
import of `cascade.pool.first_seen`. Fixtures are a synthetic forge mirror
(`sources.yaml` + `data/<feed>/<YYYY-MM-DD>.parquet` with `timestamp`,
`_panel_*` and value columns, the `TsbenchForgeSource` contract) written to
`tmp_path`; no bucket, no network.

`test_evidence_*` **skip** when `$LANED_EVIDENCE` is unset, so the module can
merge and `accept W6fs` reach L3 before the owner's builder run. L4 needs
`accept W6fs --evidence evidence/w6fs.json` green on main, with the file
written on the `pool_d` builder by `python -P -m cascade.pool.first_seen
evidence` (below).

## Contract (`cascade/pool/first_seen.py`)

Constants: `PERIOD_SECONDS = 10800` (3 h, rev6.11 (d)), `LOG_SCHEMA =
"pool-d-first-seen/1"`, `EVIDENCE_SCHEMA = "laned-evidence/w6fs/1"`.

| Name | Contract |
|---|---|
| `period_start(ts)` | `ts - ts % PERIOD_SECONDS` |
| `canonical_json(obj)` | `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()` |
| `panel_json(panel)` | `canonical_json(dict(panel)).decode()`; unpaneled = `"{}"` |
| `merkle_root(leaves)` | hex root over byte leaves: leaf hash `sha256(b"\x00" + leaf)`, node `sha256(b"\x01" + left + right)`, an odd last node is promoted unchanged; `ValueError` on no leaves |
| `release_ts(first_seen_ts, embargo_hours)` | `int(first_seen_ts) + round(embargo_hours * 3600)` (rev6.10: release is derived, never stored) |
| `FirstSeenError` | `ValueError` subclass with a string `.code` |
| `FirstSeenLog(log_dir)` | the append-only ingest log: `records.jsonl` + `roots.jsonl` in `log_dir` (by default `<forge mirror>/first_seen/`, which `TsbenchForgeSource` never reads) |
| `.record(forge_dir, *, now_ts)` | scans every `data/<feed>/*.parquet`; a partition whose (feed, name, sha256) is already logged is skipped unread; otherwise appends ONE record per partition holding only the point keys `(panel_json, data_ts)` never seen before for that feed. Returns `{"records": n, "points": n}`. Raises `FirstSeenError(code="period_committed")` when `period_start(now_ts)` already has a committed root |
| `.commit(*, now_ts)` | commits a root for every period with records that has ENDED (`period_start + PERIOD_SECONDS <= now_ts`) and has none yet; appends `{"period_start", "root", "n_records", "committed_ts": now_ts}` to `roots.jsonl`; returns the new entries |
| `.records()` / `.roots()` | parsed log lines / `{period_start: root entry}` |
| `.index(*, until_ts)` | a `FirstSeenIndex` over the records of the periods whose root was committed with `committed_ts <= until_ts` (a snapshot at `until_ts` can only use roots committed by then, and an old snapshot re-derived later sees exactly what it saw); recomputes every committed root from the log and raises `FirstSeenError(code="root_mismatch")` on any difference |
| `FirstSeenIndex.first_seen(feed, panel, data_ts)` | first `seen_ts` of the point, or `None` |
| `FirstSeenIndex.root_for(feed, panel, data_ts)` | the committed root covering that record, or `None` |
| `FirstSeenIndex.feed_lag()` | `{feed: {"mirror_lag_hours", "max_lead_hours", "publish_ahead"}}` |
| `verify_snapshot(meta, log_dir)` | list of problem strings for a `pool_d/1` metadata dict (empty = ok) |
| `main(argv)` | CLI `record`, `commit`, `evidence` (exit 0 on success) |

A record line is `{"schema": LOG_SCHEMA, "feed", "partition" (file name),
"partition_sha256", "seen_ts", "period_start", "baseline", "points":
[[panel_json, data_ts], ...]}` with `points` sorted; `baseline` is true for
every record of the first `record()` call that logs a feed (its points predate the recorder, so
they say nothing about mirror lag). The leaves of a period's root are
`canonical_json(record)` of its records in log order.

Lag, per feed, in hours: `mirror_lag_hours` = median of `first_seen −
data_ts` over non-baseline points with `data_ts <= first_seen` (`None` when
there are none); `max_lead_hours` = max of `data_ts − first_seen` over every
point (`None` when the feed has no points); `publish_ahead` =
`max_lead_hours > 0`. W6b uses `publish_ahead` to force `e = 0`.

`verify_snapshot` reads `meta["as_of_ts"]`, `meta["first_seen_roots"]` and per
series `feed`, `panel`, `target_ts`, `first_seen_ts`. A problem is reported
when a target point is unknown to the log, its `first_seen_ts` differs from
the log, its root is not committed with `committed_ts <= as_of_ts`, its root is
not listed in `first_seen_roots`, or the log no longer matches its committed
roots (a rewritten record).

## Evidence (`evidence/w6fs.json`)

`python -P -m cascade.pool.first_seen evidence --forge-dir D --log-dir L
--out evidence/w6fs.json` on the builder writes `{"schema": EVIDENCE_SCHEMA,
"wp": "W6fs", "source": "builder", "generated_by":
"cascade.pool.first_seen evidence", "host", "generated_ts", "n_records",
"n_points", "roots": [{"period_start", "root", "n_records",
"committed_ts"}], "feeds": {feed: lag record}, "digest"}`, `digest` =
sha256 of `canonical_json` of the document without `digest`.

## What is pinned

| Row text | Test |
|---|---|
| append-only ingest log, first time our mirror saw a point | `test_record_first_seen_per_point`, `test_log_is_append_only` |
| rewrites and backfills keep the first time | `test_rewrite_keeps_first_time`, `test_backfill_new_points_get_their_own_time` |
| first-seen never later than any snapshot containing the point | `test_index_until_never_exposes_later_records`, `test_index_excludes_uncommitted_period` |
| lag computed per feed | `test_mirror_lag_per_feed`, `test_publish_ahead_feed_measured` |
| per-3-hourly-period Merkle root committed before use | `test_commit_roots_per_3h_period`, `test_commit_skips_open_period`, `test_record_refuses_committed_period` |
| snapshot postdating its committed root refused | `test_verify_snapshot_refuses_uncommitted_and_late_roots`, `test_index_refuses_rewritten_log` |
| release recomputable from the signed record and pinned `e` alone | `test_release_recomputable_from_record_and_e` |
| `pool` bytes and lane A windows byte-identical with and without the recorder | `test_pool_bytes_identical_with_and_without_recorder` |
| evidence from the real run | `test_evidence_*` |
