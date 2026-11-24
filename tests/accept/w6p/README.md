# W6p acceptance tests: public-source index, mirror and allowed list (rev6.18 A1, A5, A6; F3)

W6p is new in plan rev6.18 (F3 row): the index builder and mirror script,
read-only on Hugging Face until the owner runs them with the bucket keys.
Accept: `scripts/laned/accept W6p` (CI). Fixtures: `fixtures.py` (tiny
synthetic GiftEvalPretrain-style arrow and RMISC-style parquet mirrors,
written by pyarrow; nothing is downloaded). Nothing here imports torch.

## Module contract: `cascade/eval/lane_d_public_index.py`

- `PUBLIC_SOURCES`: `gift_eval_pretrain` = `Salesforce/GiftEvalPretrain` @
  `6830b624…` (one window per series) and `rmisc` = `nju-zhangsq/RMISC` @
  `d0cdd935…` (several per series), each ONE 40-hex revision.
- `family_of`: `cmip6_*` / `era5_*` / `largest_*` (any case, `_` or `-`)
  merge into one family; any other folder is its own.
- The index: one row per series `{ordinal, series_id, dataset, family, freq,
  length, n_channels, file, rows}` (`rows` = `[start, stop)` row ranges in
  `file`; RMISC series = runs of equal `_original_filename`; `n_channels`
  counts the value columns holding a value, the `meta.json` timestamp and
  `_original_filename` excluded). `index.jsonl.gz` = canonical JSON lines
  (sorted keys, compact) in `(dataset, file, first row)` order;
  `index_sha256` = sha256 of the uncompressed bytes. `manifest.json`:
  `schema`, `source`, `repo`, `revision`, `n_series`, `index_sha256`, the file
  list (`path`, `size`, `sha256`), `files_sha256` = sha256 of its canonical
  JSON, `datasets`. `iter_index` / `load_manifest` refuse any mismatch.
- `read_series(data_root, row, source=)`: the `(C, L)` float64 series (NaN
  for nulls), reading one arrow row or only the parquet row groups that
  cover its ranges.
- The allowed list `data/lane_d/rmisc_allowed.toml` (`parse_allowed_list`):
  `schema = "lane-d-rmisc-allowed/1"`, the pinned RMISC `repo` and `revision`,
  `status` `pending` (no folder) or `final` (≥ 1), `generated_by`, and
  `[[folder]]` tables with `name`, `reason`, `licence`, `domain`, `freq`,
  `data_check = "pass"`, names unique and sorted, never KnownOpen / KnownOil.
  The repo copy is `pending` (no folder) until the owner's finalize run.

## Owner tools

`scripts/laned_public_mirror.py`:

- `size --source S` prints `SIZE <source> <repo>@<rev> datasets=… files=…
  bytes=… (… GiB)` from the HF tree metadata (RMISC: the final allowed
  folders only; a pending list is refused).
- `mirror` checks free disk ≥ size × 1.15 before downloading, downloads the
  pinned revision, checks every file size, builds the index and uploads data
  + index to Hippius and R2 under `lane_d_public/<source>/<revision>/`. Keys
  come from the environment by name only (`HIPPIUS_S3_*`, `BACKUP_S3_*`).
- `index` builds the index from an existing local copy.

`scripts/laned_rmisc_allowlist.py`: `fetch-listings` (HF metadata + RMISC
`meta.json` only) → `candidates` (offline: name match against GIFT-Eval
test, GiftEvalPretrain, fev_datasets, chronos_datasets, TIME @ 83e3d0b, BOOM
@ 69325b5; KnownOpen/KnownOil; non-forecasting; licences forbidding
redistribution) → `datacompare` (OWNER on a pod: winnowed K-gram hashes of
quantised values, every exclusion source's data) → `finalize` → `validate`.
Output formats are in the script's docstring.

| Item | Test |
|---|---|
| one pinned revision per source | `test_each_public_source_is_pinned_at_one_revision` |
| year-split families | `test_year_split_folders_merge_into_one_family` |
| index rows (GIFT, RMISC) | `test_gift_index_has_one_row_per_series`, `test_rmisc_index_groups_rows_by_original_filename` |
| index / file-list hashes | `test_index_hashes_are_canonical_and_reproducible`, `test_a_tampered_index_or_file_list_is_refused` |
| reader reads only the series | `test_read_series_returns_the_stored_series`, `test_read_series_reads_only_the_row_groups_of_that_series` |
| allowed-list format | `test_the_repo_allowed_list_is_a_valid_pending`, `test_the_allowed_list_format_round_trips_and_refuses_defects` |
| size check, disk check, keys by env name, index command | `test_size_check_prints_the_bytes_from_the_metadata`, `test_rmisc_size_and_mirror_need_a_final_allowed_list`, `test_mirror_refuses_to_start_without_disk_headroom`, `test_mirror_reads_store_keys_from_the_environment_by_name`, `test_index_command_builds_the_index_from_a_local_copy` |
| A5 candidates, data compare, finalize | `test_candidates_apply_the_a5_rules`, `test_candidates_refuse_unpinned_listings`, `test_datacompare_compares_data_not_names`, `test_finalize_writes_a_valid_final_list` |

## Hippius mirror (owner 2026-11-24, issue #894)

The canonical mirror is an object set on Hippius S3 (R2 the backup copy),
not a local copy on every eval host (`cascade/eval/lane_d_public_mirror.py`,
builder `cascade/eval/lane_d_public_build.py`, config
`deploy/laned/public_mirror.toml`). **The index format and bytes do not
change**: the streaming build writes the very `index.jsonl.gz` /
`manifest.json` that `build_index` writes over a full local mirror, so
`index_sha256` / `files_sha256` (signed report, draw, audit) are untouched.

- Objects under `<prefix>/<source>/<revision>/`: `shards/<sha256>.parquet`
  (content-addressed, ~256 MB, never spanning folders; columns `series_id`,
  `n_channels`, `n_points`, `values` list<float32> or — when the float32
  round trip is not exact — `values64` list<float64>; zstd, byte-stream
  split; row groups of whole series closed at 1 MiB raw) and
  `index/<index_sha256>/{index.jsonl.gz, manifest.json, shards.json,
  shardmap.jsonl.gz}`.
- `shards.json` (`lane-d-public-shards/1`) binds the index's source, repo,
  revision, `index_sha256`, `files_sha256`, `n_series`, the
  `shardmap_sha256` and each shard's key, size, sha256 and tail (footer)
  range + sha256. `shardmap.jsonl.gz`: one canonical line per index row in
  ordinal order: `ordinal, series_id, shard, row_group, row, offset, length,
  sha256` (the row group's byte range) and `n_channels, n_points,
  values_sha256` (the hash of `read_series` over the HF files).
- Eval host (`PublicMirror.reader(draw)`): fetches the index set once
  (verified against the draw's `index_sha256` / `files_sha256`), then ONLY
  the drawn series' row groups (+ each touched shard's tail) by ranged GET:
  the `BlobCache` (the Lium volume; sha256-addressed, byte-budget LRU, every
  read re-hashed) first, then the stores in order (Hippius, R2). A bad hash
  is refused and the next store tried; a decoded series must match its
  `values_sha256`, its `series_id` and the index length.
- Builder: per folder download → W6p index + file hashes → shards → upload
  → journal; resumable by shard, idempotent (`scripts/laned_public_mirror.py
  build|estimate|fetch-index`).

| Item | Test |
|---|---|
| index identical to a full-mirror build | `test_the_built_index_is_the_w6p_index_of_a_full_mirror` |
| content-addressed objects, index set uploaded | `test_objects_are_content_addressed_and_the_index_set_uploaded` |
| same canonical series | `test_every_series_decodes_to_read_series_exactly`, `test_nan_and_multichannel_series_round_trip` |
| ranged fetch + verify, R2 fallback | `test_ranged_fetch_verifies_and_falls_back_to_r2` |
| tampered shard / index set refused | `test_a_tampered_shard_is_refused_and_the_backup_used`, `test_a_tampered_index_set_is_refused`, `test_build_refuses_a_file_that_is_not_the_listing` |
| cache: warm reads, re-verify, refill, LRU budget | `test_the_cache_serves_repeat_reads_and_reverifies_every_read`, `test_cache_eviction_keeps_the_byte_budget_lru` |
| draw → fetch touches only the drawn series | `test_draw_then_fetch_touches_only_the_drawn_series` |
| resumable, idempotent build | `test_the_build_resumes_by_shard_and_is_idempotent` |
| box index fetch, estimate, config | `test_fetch_index_gives_the_box_a_verified_index`, `test_estimate_extrapolates_the_measured_ratio`, `test_mirror_config_names_stores_and_the_volume_cache` |
