# W6a acceptance tests: `pool_d` selection joins `pool`'s selected series

Plan rows: §3.0 **W6a** ("`pool_d` selection = `pool`'s selected series ids
(joined, not re-selected)", the rev5 join rule), §1.7 "`pool_d` (D12)"
("the same selected series ids as `pool`, joined (not re-selected)"), and
"**Lane A must not change**" (`pool` untouched: same builder inputs, same
bytes). Runs in `ci + builder`; Accept is `scripts/laned/accept W6a`.

These tests are the spec. The builder may touch only `PATHS`:
`cascade/pool/pool_d_select.py` (new). `cascade/pool/builder.py` and every
other lane A file stay byte-identical; lane A's own unit tests are unchanged.

On main every test fails at its first import of `cascade.pool.pool_d_select`
(a module in PATHS). The tests use synthetic in-memory sources and a fixture
forge mirror (`sources.yaml` + `data/<feed>/<YYYY-MM-DD>.parquet`, the
real `TsbenchForgeSource` contract), never a live bucket.

## Why a join, and on which key

`builder.collect_records` assigns collision suffixes `-2`, `-3`, … in sort
order (`builder.py:361-370`, `:471-480`), so lane A's emitted `series_id`
moves whenever another colliding series appears or vanishes; `content_hash`
is the sha256 of the truncated values (`:295`), so it changes whenever the
tail moves. `pool_d` therefore joins on

```
key = (source, pre-suffix source series key, channel)
    = (str(hs.source), builder._sanitize_id(hs.series_id), channel)
```

built in the **same builder run** from the same harvested records. The
emitted id and the content hash are used only *inside one run* to find which
harvested item produced each lane A record; they are never recorded as the
join key and never compared across snapshots.

## Contract (`cascade/pool/pool_d_select.py`)

| Name | Contract |
|---|---|
| `REFUSAL_REASONS` | tuple containing `"key_collision"`, `"no_source"`, `"ambiguous_record"`, `"id_collision"` |
| `join_key(hs, channel=0)` | `(str(hs.source), builder._sanitize_id(hs.series_id), int(channel))` |
| `pool_d_series_id(key)` | `f"{base}~{builder._sanitize_id(source)}.c{channel}"`: stable across snapshots, unique per key, contains no `:` or `/` |
| `select_pool_d(harvested, records, cfg)` | `PoolDSelection` for lane A's `records` (the output of `collect_records(…, cfg, …)` over exactly `harvested`, in harvest order) |
| `collect_with_selection(sources, ctx, cfg, fetch)` | `(records, drops, selection)`: harvests **each source once**, runs the unchanged `builder.collect_records` over the replayed series, then `select_pool_d` |
| `build_pool_with_selection(sources, out_dir, ctx, cfg, *, fetch=None, overwrite=False)` | `(BuildSummary, PoolDSelection)`; the written pool is **byte-identical** to `builder.build_pool(sources, out_dir, ctx, cfg, …)` (domain floor check included) |

`PoolDSelection` (frozen):

- `series`: tuple of `SelectedSeries`, one per (lane A record, channel),
  sorted by `pool_d_series_id`;
- `refused`: tuple of `(pool_series_id, reason)` pairs, sorted; every lane A
  record is either selected (all its channels) or refused, never both;
- `join_keys()`: frozenset of the selected join keys;
- `to_json()`: `{"schema": "pool-d-selection/1", "series": [...], "refused":
  [...]}` (plain JSON types, deterministic);
- `sha256()`: sha256 of `json.dumps(to_json(), sort_keys=True,
  separators=(",", ":"))`.

`SelectedSeries` (frozen) fields: `key`, `pool_series_id` (lane A's emitted
id in this run), `pool_d_series_id`, `feed` (`hs.source`), `cluster`
(lane A's `metadata["source"]`, the KOTH cluster key), `channel`,
`n_channels`, `freq`, `domain`, `attrs` (a dict copy of `hs.attrs`).

Refusals (each also logged as a warning on logger `cascade.pool` whose
message contains `pool_d_join_refused` and the lane A id):

- `key_collision`: two harvested series of **one source** share
  `(source, base)`; both are refused from `pool_d`; lane A is unaffected;
- `no_source`: the harvested series has no `source`;
- `ambiguous_record`: the lane A record's values match harvested series with
  different keys (lane A deduplicated them), so the key is not determined;
- `id_collision`: two keys map to one `pool_d_series_id`.

## What is pinned

| Row text | Test |
|---|---|
| `pool` bytes unchanged | `test_pool_bytes_identical_to_build_pool` (both selection modes), `test_forge_mirror_pool_bytes_identical` |
| joined, not re-selected | `test_selection_is_lane_a_selected_records`, `test_lane_a_dropped_series_never_selected` |
| same builder run, same records | `test_each_source_harvested_once` |
| join key (source, pre-suffix key, channel), never emitted id | `test_join_key_survives_suffix_reassignment` |
| never content hash | `test_join_key_survives_tail_move` |
| key collision inside one source refuses (logged) | `test_key_collision_in_one_source_refused_and_logged` |
| per channel | `test_multichannel_record_one_key_per_channel` |
| fail closed | `test_no_source_refused`, `test_ambiguous_record_refused` |
| deterministic / content-addressed | `test_selection_deterministic_and_hashable` |
| forge mirror shape | `test_forge_mirror_keys_are_catalog_feed_and_panel` |
