# W1 acceptance tests: the lane D measurement

W1 (plan §3.0 W1 row, S1 B2) measures what lane D's forward-only judging
can count on: forward-eligible windows and latency per frequency, the
long-history inventory per context variant, and `pool_d` size. The
implementation is `scripts/laned_w1_measure.py`, its unit tests, and the
read-only workflow `.github/workflows/laned-w1-measure.yml` that runs it in
CI with the forge mirror synced down (see `PATHS`).

These tests pin the **forge-mirror mode** (`--forge-dir`, label
`measured from forge mirror`). It sits next to the existing
`snapshot-diff bound` (unchanged).

| Item | Test |
|---|---|
| the unit suite `tests/unit/test_laned_w1_measure.py` passes | `test_unit_suite_passes` |
| `w1.json` schema: `labels`, `forge_mirror.per_freq` for every bucket (10s … slower), the four clocks T+6/9/12/24 h, `long_history.rungs` for 4096 … 131072 with `need_points = L + horizon`; the defaults (embargo 48 h, min fresh share 0.5, max windows 5000, forward margin 0, eval delay 6 h); the gate reads `LaneDConfig().min_long_series` and 20 sources | `test_forge_report_schema_and_defaults` |
| per-cadence sources/series, the reader's skips (slower than daily, disabled), first-seen-proxy latency from the partition object time, long-history counts | `test_mirror_inventory_latency_and_long_history` |
| rev6.14: `b = T + forward_margin` (default 0, so fresh windows are counted from T); embargo targets from `b − 48 h`; judging starts at `T + eval_delay` (a clock before it is "not run"); a 6 h margin counts fewer fresh windows | `test_fresh_counted_from_commit_time_and_eval_delay` |
| rev6.12 (b) set rule on every row: fresh first, embargo ≤ `F·(1−s)/s`, at most 5000 windows, embargo can't start a set, `met` ⇒ floors and share | `test_set_rule_holds_on_every_row` |
| read-only (the mirror's files are untouched) and deterministic (two runs give the same block) | `test_read_only_and_deterministic` |
| the workflow gives the forge sync `id: forge_sync`, runs the measurement with `--forge-dir … --embargo-hours 48` only when `steps.forge_sync.outcome == 'success'`, installs the locked `pool-forge` extra, and syncs bucket → runner only | `test_workflow_measures_forge_mirror_after_sync` |
| with `LANED_EVIDENCE` (`accept W1 --evidence evidence/w1.json`): the evidence file has the same schema and a non-empty sweep | `test_evidence_file` |

The fixture mirror is built by `helpers.build_mirror`:
- a 1-min feed with one partition per day, written at 23:30;
- a 2-panel hourly feed with 5000 points per panel;
- a weekly feed;
- a disabled feed.

No torch and no bucket access. Needs the `pool-forge` extra (pandas, pyarrow, pyyaml).
