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
