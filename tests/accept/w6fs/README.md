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

### Unreadable partitions: the quarantine (2026-11-25)

Two forge objects with no parquet footer aborted every `record`. Now
(`test_w6fs_quarantine.py`):

* `record` skips a partition that fails to read, logs it to stderr
  (`QUARANTINE ...`) and appends to `quarantine.jsonl` (append-only, next to
  `records.jsonl`; restored/pushed with the rest of the log by R13
  `state-pull`/`state-push`) `{"schema": QUARANTINE_SCHEMA, "kind":
  "unreadable", "feed", "partition", "partition_sha256", "size", "reason",
  "quarantined_ts"}`. The same bytes are never re-read or re-logged.
* A quarantined partition NAME stays quarantined forever. When a later version
  of it reads, its points not already known are appended as `{"kind":
  "blocked", ..., "points"}` -- never to `records.jsonl` -- and are never
  recorded as first seen afterwards from any partition (a repaired object's
  old points can never count as fresh). Points recorded from other partitions
  before the quarantine keep their genuine first-seen time.
* `.index(until_ts)` also drops any record of a quarantined name with
  `seen_ts >= quarantined_ts` (written by a quarantine-unaware release).
* More than `unreadable_limit(n_partitions, max_files=50, max_frac=0.001)` =
  `min(max_files, max(1, floor(max_frac * n)))` unreadable partitions in one run
  (new ones plus quarantined bytes still present) raises
  `FirstSeenError("too_many_unreadable")` before anything is written (CLI
  `--max-unreadable-files` / `--max-unreadable-frac`, exit 2).
* `record` returns `{"records", "points"}` plus `quarantined` / `blocked` /
  `unreadable` counts only when non-zero.

### Incremental scans and the `ended` root rule (2026-11-28)

Every pool_d build re-hashed all ~630k forge partitions and re-parsed the
whole 20 GB `records.jsonl` (record, commit, the build's index), and cut the
snapshot from the root committed one build earlier (`test_w6fs_incremental.py`):

* `FirstSeenLog(log_dir, *, cache_dir=None, full=False,
  full_verify_seconds=FULL_VERIFY_SECONDS, clock=time.time)`. With a
  `cache_dir` (the CLI's default is `default_cache_dir(log_dir)` =
  `<log_dir>.cache`; `--no-cache` turns it off, `--full` ignores it for one
  run) it keeps `partitions.json` (`(size, mtime_ns, inode) -> sha256` per
  partition: only new or changed partitions are hashed) and
  `records.meta.json` (per log line: span, feed, partition, sha256, `seen_ts`,
  `period_start`, point count, data-time range, Merkle leaf; plus a
  fingerprint of the bytes covered): only lines appended since are parsed,
  and `record` loads a feed's known points only from the records whose
  data-time range meets the new partition's. Every result (records, roots,
  quarantine, index, lag) equals the uncached run. A cache that no longer
  fingerprints its log prefix is dropped (full scan); one older than
  `full_verify_seconds` (24 h) is replaced by a full scan that warns
  `DIVERGED` if it differed. The cache is never one of the pushed log files.
* `.index(*, until_ts, roots="committed")` / `.feed_index(...)` (plan rev6.26, owner-approved 2026-11-28): `roots` is
  `"committed"` (the rule above) or `"ended"`: every committed period that
  ENDED by `until_ts` (`period_start + PERIOD_SECONDS <= until_ts`, so all its
  `seen_ts < until_ts`), whenever its root was committed — the pool_d
  builder's own `commit`, run just after the boundary, makes the previous
  build's records usable in this build's snapshot. `"ended"` raises
  `FirstSeenError("uncommitted_period")` when an ended period has records but
  no root, so the usable set is final when the snapshot is cut.
* `.record` raises `FirstSeenError("period_frozen")` when `period_start(now_ts)`
  ended by the newest `committed_ts` (a backdated `now_ts`): no record can
  later enter a period an `ended` snapshot relied on, so a re-derivation from
  the log sees the same points.
* `verify_snapshot(meta, log_dir, *, roots="committed", cache_dir=None)`.

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
| incremental scans equal the full recompute; only new input read; stale cache caught | `test_cached_runs_equal_the_full_recompute`, `test_only_new_or_changed_partitions_are_hashed`, `test_only_appended_log_lines_are_parsed`, `test_rewritten_log_is_caught_through_the_cache`, `test_periodic_full_verify_rebuilds_a_diverged_cache` |
| `ended` root rule: same-build root, final, frozen periods | `test_ended_rule_uses_the_root_this_build_committed`, `test_ended_rule_refuses_an_uncommitted_ended_period`, `test_record_refuses_a_period_frozen_by_a_commit` |
| evidence from the real run | `test_evidence_*` |
