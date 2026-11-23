# W6d acceptance tests: `pool_d` pinning, digests, retention, 3-hourly cadence

Plan rows: §3.0 **W6d** ("Pinning on Hippius; digests in reports;
retention. **(rev6.11)** Cadence: one `pool_d` snapshot every 3 h …, each
build finishes before a settlement boundary (rec); retention keeps every
snapshot an open entry's eligible or frozen set references; the build wall
must stay < 3 h (alert otherwise). **(rev6.10)** The pinned snapshot (sha256
in every report) is the only source of the per-feed `embargo_hours` a verdict
used; the report also carries the sha256 of the catalog `sources.yaml` the
builder read, for provenance only"), rev6.11 (d), §1.7 "Build host" and
"Reveal" (no reveal obligation). Depends on W6c and W3a (the upload rule:
single PUTs are only safe up to 512 MiB, so large tars go multipart in parts
≤ 512 MiB). Runs in `builder`; Accept is `scripts/laned/accept W6d --evidence
evidence/w6d.json`.

These tests are the spec. The builder may touch only `PATHS`:
`cascade/pool/pool_d_publish.py` (new). On main every test fails at its first
import of `cascade.pool.pool_d_publish`. Storage is the module's own
filesystem store (`FsPoolStore`), wrapped by a recorder in the tests; the
Hippius adapter is only constructed, never called (no network).

**Digests in reports.** The lane D report already carries each judged
snapshot's sha256 (`pool.snapshots`, W7a). That sha256 covers
`metadata.json`, which carries the effective per-feed `embargo_hours` (W6b)
and the catalog `catalog_sha256`. So the pinned digest is what binds both;
W6d adds them to each pin and manifest and never re-reads the catalog.

`test_evidence_*` **skip** when `$LANED_EVIDENCE` is unset. L4 needs
`accept W6d --evidence evidence/w6d.json` green on main, written by the CLI's
`evidence` command after a live run (below).

## Contract (`cascade/pool/pool_d_publish.py`)

Constants: `CADENCE_SECONDS = 10800`, `MAX_BUILD_WALL_SECONDS = 10800`,
`PINS_SCHEMA` (equals `cascade.validator.lane_d_inputs.POOL_PINS_SCHEMA`),
`MANIFEST_SCHEMA = "lane-d-pool-manifest/1"`, `INDEX_SCHEMA =
"lane-d-pool-index/1"`, `EVIDENCE_SCHEMA = "laned-evidence/w6d/1"`,
`MULTIPART_CHUNK_BYTES` (≤ 512 MiB, W3a), `TERMINAL_STATUSES = ("crowned",
"lost", "closed")`.

| Name | Contract |
|---|---|
| `PublishError` | `ValueError` subclass with a string `.code` |
| `due_as_of(now_ts)` | the `as_of_ts` of the build due at `now_ts`: `now_ts` floored to `CADENCE_SECONDS` |
| `check_build_wall(*, as_of_ts, started_ts, finished_ts)` | `{"ok", "wall_seconds", "limit_seconds", "deadline_ts"}`; `deadline_ts = as_of_ts + CADENCE_SECONDS` (the next boundary); `ok` iff `wall_seconds < MAX_BUILD_WALL_SECONDS` and `finished_ts < deadline_ts`; when not ok it logs a warning on logger `cascade.pool` containing `pool_d_build_wall_exceeded` |
| `FsPoolStore(root)` | filesystem store: `put_file(key, path)`, `get_file(key, path)`, `put_bytes(key, data)`, `get_bytes(key)` (`cascade.shared.hippius.ObjectNotFound` when absent) |
| `HippiusPoolStore` / `open_pool_store(chain_toml)` | the same interface over Hippius S3, built from a chain.toml path: bucket `[lane_d_ops] pool_d_bucket` (parsed by `parse_lane_d_ops`; `.prefix` is `pool_d_prefix`), endpoint from `[storage]`, credentials from `HIPPIUS_S3_*` env only; lazy (no request until used); files above one part go multipart in `MULTIPART_CHUNK_BYTES` parts |
| `publish_snapshot(pool_dir, store, *, prefix, build_wall_seconds=None)` | splits the snapshot (W6c), uploads `<prefix>/<snapshot>/hist.tar` and `tgt.tar`, writes `<prefix>/<snapshot>/manifest.json` and adds it to `<prefix>/index.json` (once); returns the manifest. Re-publishing the same bytes is a no-op; a different snapshot under an existing name raises `PublishError(code="snapshot_conflict")` |
| `sync_pins(store, *, prefix, local_root, pins_path, from_block)` | box side: for every indexed snapshot not yet pinned, downloads both tars, checks `hist_sha256` / `tgt_sha256`, joins them (W6c) under `local_root/<snapshot>/`, checks the snapshot sha256, and appends a pin; returns the new pins. A digest failure raises `PublishError(code="digest_mismatch")` and pins nothing for that snapshot |
| `fetch_hist(store, *, prefix, snapshot, out_dir)` | eval-host side: downloads only the manifest and `hist.tar`, checks `hist_sha256`, returns the tar path (it never reads `tgt.tar`: invariant 8) |
| `retention_plan(pins_path, state, *, embargo_hours=48.0)` | `{"keep": [...], "expire": [...]}` (snapshot names, pin order) |
| `apply_retention(pins_path, local_root, plan)` | rewrites the pins file without the expired pins and removes their directories (only inside `local_root`); returns the removed snapshot names |
| `main(argv)` | CLI `publish`, `sync`, `retention`, `evidence` |

A **manifest** is `{"schema": MANIFEST_SCHEMA, "snapshot", "as_of_ts",
"sha256", "hist_sha256", "tgt_sha256", "hist_key", "tgt_key",
"catalog_sha256", "first_seen_roots", "n_series", "build_wall_seconds"}`;
the **index** is `{"schema": INDEX_SCHEMA, "snapshots": [{"snapshot",
"as_of_ts", "manifest_key"}]}` sorted by `as_of_ts`. A **pin** is the
`PinFilePool` record `{"dir" (relative to the pins file), "sha256",
"from_block"}` plus `snapshot`, `as_of_ts`, `hist_sha256`, `tgt_sha256`,
`catalog_sha256`; the pins file `{"schema": PINS_SCHEMA, "pins": [...]}`
lists them oldest first and is rewritten atomically.

**Retention.** An entry is *open* unless its `status` is in
`TERMINAL_STATUSES`; the open entries are `state["entries"].values()` plus
`state["house_entry"]` when present. A pinned snapshot is kept when it is the
newest pin, or its `as_of_ts > min(open reveal_ts) − embargo_hours × 3600`
(every snapshot an open entry's eligible union can still draw from), or it
holds a window id listed in an open entry's `frozen_set["window_ids"]`.
An open entry without a `reveal_ts` keeps every snapshot (fail safe).

### CLI (the owner's live run)

```
python -P -m cascade.pool.pool_d_publish publish --pool-d DIR --chain-toml chain.toml \
    [--prefix P] [--started-ts S --finished-ts F]
python -P -m cascade.pool.pool_d_publish sync --chain-toml chain.toml --pins PINS.json \
    --local-root DIR --from-block N
python -P -m cascade.pool.pool_d_publish retention --pins PINS.json --local-root DIR \
    --state lane_d_state.json [--apply]
python -P -m cascade.pool.pool_d_publish evidence --chain-toml chain.toml --out evidence/w6d.json
```

`--fs-store DIR` replaces `--chain-toml` with a filesystem store (testnet /
local runs). `evidence` re-reads the index, re-downloads every listed
snapshot's tars, re-checks every digest, and writes `{"schema":
EVIDENCE_SCHEMA, "wp": "W6d", "source": "builder", "generated_by":
"cascade.pool.pool_d_publish evidence", "host", "generated_ts", "bucket",
"prefix", "snapshots": [{"snapshot", "as_of_ts", "sha256", "hist_sha256",
"tgt_sha256", "build_wall_seconds", "verified": true}], "cadence_ok",
"walls_ok", "digest"}`. `cadence_ok` holds when at least two snapshots are
listed and consecutive `as_of_ts` differ by exactly `CADENCE_SECONDS`.
`walls_ok` holds when every recorded `build_wall_seconds` is below the
limit. `digest` is the sha256 of the canonical JSON of the document without
`digest`.

## What is pinned

| Row text | Test |
|---|---|
| one snapshot every 3 h | `test_due_as_of_cadence` |
| build wall < 3 h, alert otherwise; each build finishes before the boundary | `test_check_build_wall` |
| pinning on Hippius (digests) | `test_publish_writes_manifest_tars_index`, `test_publish_idempotent_and_conflict` |
| pins readable by the validator (`PinFilePool`) | `test_sync_pins_feeds_pin_file_pool`, `test_sync_refuses_tampered_tar` |
| targets stay on the box | `test_fetch_hist_never_reads_targets` |
| pinned snapshot is the only source of `embargo_hours`; catalog sha for provenance | `test_digest_binds_embargo_and_catalog` |
| retention keeps every referenced snapshot | `test_retention_plan_rules`, `test_apply_retention` |
| Hippius adapter, multipart ≤ 512 MiB, env-only credentials | `test_hippius_store_lazy_and_scoped` |
| CLI | `test_cli_publish_sync_retention_evidence` |
| evidence from the real run | `test_evidence_*` |
