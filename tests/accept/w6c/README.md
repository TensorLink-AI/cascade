# W6c acceptance tests: `pool_d_hist` / `pool_d_tgt` split

Plan rows: §3.0 **W6c** (`pool_d_hist` / `pool_d_tgt` split; depends on
W6b), §1.7 "**Artifacts:** two tars from one build. `pool_d_hist`: histories
only, **no targets**. It goes to the eval host … `pool_d_tgt`: targets plus
timestamps, small. It stays on the box. This keeps the existing privacy
rule: targets and scoring stay local (`eval_offload.py:7-10`)", and §3.2
invariant 8 "**Targets never leave the box.** (W6c)". Runs in `ci + builder`;
Accept is `scripts/laned/accept W6c`.

These tests are the spec. The builder may touch only `PATHS`:
`cascade/pool/pool_d_split.py` (new). On main every test fails at its first
import of `cascade.pool.pool_d_split`. The fixture `pool_d/1` snapshot is
hand-written in the exact shape the W6b builder writes and passes W6b's
`validate_pool_d`.

Not in scope: the proposed per-window MASE scale in `pool_d_tgt` (plan
"Proposed design changes" item 8, an owner decision, not applied).

## Contract (`cascade/pool/pool_d_split.py`)

Constants: `HIST_META_SCHEMA = "pool_d_hist/1"`, `HIST_MEMBERS =
("hist.npz", "hist_metadata.json")`, `TGT_MEMBERS = ("metadata.json",
"tgt.npz")`, `HIST_SERIES_FIELDS = ("first_target_ts", "freq",
"history_len", "seasonal_period", "series_id")`.

| Name | Contract |
|---|---|
| `SplitError` | `ValueError` subclass with a string `.code` |
| `hist_metadata(meta, snapshot_sha256)` | `{"schema": HIST_META_SCHEMA, "snapshot", "snapshot_sha256", "as_of_ts", "horizon", "history_length", "series": [...]}` with each series reduced to exactly `HIST_SERIES_FIELDS` (no target data, no target or first-seen timestamps beyond the window start) |
| `split_pool_d(pool_dir, out_dir)` | validates the snapshot with W6b's `validate_pool_d` (refuses an invalid one with `ValueError`), then writes `<snapshot>.hist.tar` (exactly `HIST_MEMBERS`; `hist.npz` byte-identical to the snapshot's) and `<snapshot>.tgt.tar` (exactly `TGT_MEMBERS`, both byte-identical); returns `SplitResult(snapshot, snapshot_sha256, hist_tar, tgt_tar, hist_sha256, tgt_sha256)` |
| `join_split(hist_tar, tgt_tar, out_dir)` | box side: extracts `hist.npz` from the hist tar and `metadata.json` + `tgt.npz` from the tgt tar into `out_dir` and returns it; the result loads with `cascade.validator.lane_d.load_pool` and hashes to `snapshot_sha256` |
| `read_hist(hist_tar)` | eval-host side: `(hist_metadata, {series_id: float array})` from the hist tar alone |
| `main(argv)` | CLI `split --pool-d DIR --out DIR` and `join --hist TAR --tgt TAR --out DIR` (exit 0 on success, non-zero on a `SplitError` / `ValueError`) |

Tars are uncompressed, deterministic and safe: members sorted, regular files
only, `mtime = 0`, `uid = gid = 0`, empty `uname` / `gname`, mode `0o644`;
the same snapshot gives byte-identical tars whatever the wall clock or the
source files' mtimes.

`join_split` / `read_hist` refuse with `SplitError` codes:

- `bad_members`: the member set is not exactly `HIST_MEMBERS` / `TGT_MEMBERS`
  (e.g. the tars swapped);
- `unsafe_member`: a member that is not a regular file, or whose name is not
  a bare file name (`..`, `/`, a link);
- `snapshot_mismatch`: the two tars belong to different snapshots;
- `digest_mismatch`: the joined files do not hash to the hist metadata's
  `snapshot_sha256`.

## What is pinned

| Row / invariant | Test |
|---|---|
| two tars from one build | `test_split_writes_two_tars_with_exact_members` |
| `pool_d_hist`: histories only, **no targets** (inv. 8) | `test_hist_tar_holds_no_target_data`, `test_hist_metadata_fields_exact` |
| `pool_d_tgt`: targets plus timestamps, small | `test_tgt_tar_holds_targets_and_timestamps` |
| box re-joins to the pinned snapshot | `test_join_restores_loadable_snapshot` |
| eval host reads histories only | `test_read_hist_from_hist_tar_alone` |
| deterministic, content-addressed | `test_split_is_deterministic` |
| fail closed | `test_join_refusals`, `test_split_refuses_invalid_pool_d` |
| CLI | `test_cli_split_and_join` |
