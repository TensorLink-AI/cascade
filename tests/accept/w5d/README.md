# W5d acceptance tests: no-pad scorer, context variants, T7 evidence

W5d (plan rev6 §3.0 W5d row, §1.1, §1.6, §4 T7) deepens W1.5's lane D scorer
`cascade/eval/lane_d_scorer.py` and adds the T7 measurement helpers
`cascade/eval/lane_d_t7.py`. These tests are the spec. The builder may touch
only the paths in `PATHS`, and must keep `tests/accept/w1_5` green (its
`forecast_quantiles(dir, histories, context=4096, horizon=64)` call stays
valid).

Accept: `scripts/laned/accept W5d` (CI, tiny model) and, for T7,
`scripts/laned/accept W5d --evidence evidence/t7.json` (pod, real size).

## What is pinned

| Plan item | Test |
|---|---|
| quantile head at exactly the 9 `SCORING_LEVELS`, picked by exact match from the head's Q21 | `test_scoring_levels_are_lane_a_quantile_head_grid`, `test_level_indices_pick_exactly_the_nine_levels_from_the_head`, `test_score_window_refuses_a_grid_other_than_nine_levels` |
| per-window CRPS (and MASE) computed the same way as lane A under `[v2] quantile_head_after` | `test_score_window_crps_and_mase_equal_lane_a_per_window` (numpy: `score_joint_quantile_forecaster_on_windows` + `wql_per_window`) |
| (rev5) the house scored by the lane D scorer at 4096 equals its lane A per-window CRPS on the same windows | `test_house_at_4096_equals_its_lane_a_per_window_crps` (torch) |
| each `[lane_d] contexts` value is a declared scorer context; others are refused | `test_scorer_contexts_cover_every_lane_d_context`, `test_each_lane_d_context_is_a_declared_scorer_context`, `test_undeclared_context_is_refused`, `test_forecast_refuses_undeclared_context_before_loading` |
| horizons ≤ 512 (16 CPM patches × 32) | `test_forecast_refuses_horizon_over_512_before_loading` |
| no padding: model input length = available history (largest patch multiple), capped at context | `test_batch_plan_*`, `test_no_pad_model_input_is_real_history_capped_at_context` (torch, a forward pre-hook records the patch tensor), `test_no_pad_truncates_to_the_largest_patch_multiple` (torch) |
| T7 CI half: a 4k checkpoint and its declared-128k copy score identically on ≤ 4096-history windows | `test_t7_4k_checkpoint_and_declared_128k_copy_identical_on_short_windows` (torch; also 8k and 16k) |
| T7 pod half: wall time + memory per variant, armed only if it fits | `test_evidence_*` (read `$LANED_EVIDENCE`), `test_builder_*`, `test_verifier_*`, `test_measure_variant_and_short_window_check_on_a_tiny_model` (torch) |
| (RV3) a variant row is only certified on inputs that exercise it: `input_len == variant`, every history ≥ variant, memory = the variant's own CUDA peak | `test_builder_row_records_the_input_that_ran`, `test_builder_row_refuses_*`, `test_builder_pod_evidence_refuses_process_peak_rss`, `test_builder_refuses_a_hand_edited_row`, `test_measure_variant_refuses_short_history_before_loading`, `test_verifier_rejects_*_that_did_not_exercise_its_variant`, the tiny-model test (torch; a forward pre-hook checks the model input really is 8192 points) |

## Module contract: `cascade/eval/lane_d_scorer.py`

Everything W1.5 pins stays (`SCORING_LEVELS`, `MEDIAN_INDEX`,
`SCORING_METHOD == "quantile_head"`, `level_indices`, `usable_length`,
`score_window`, `forecast_quantiles`). W5d adds:

- `SCORER_CONTEXTS`: ascending tuple of ints, a superset of
  `cascade.shared.config.lane_d.LANE_D_CONTEXTS`; each a multiple of 32 in
  `[4096, 131072]`.
- `MAX_HORIZON == 512`.
- `declared_context(context, contexts) -> int`: returns `int(context)` when
  it is in `contexts` **and** in `SCORER_CONTEXTS`; otherwise `ValueError`.
- `batch_plan(history_lengths, *, context, patch_size) -> dict[int, list[int]]`:
  the no-pad plan. Key = model input length `(min(n, context) // patch) * patch`,
  value = ascending window indices with that length; keys ascending. A
  history shorter than one patch is a `ValueError`. Nothing is ever padded up.
- `forecast_quantiles(checkpoint_dir, histories, *, context, horizon, contexts=None)`:
  - `contexts` (keyword-only, default `None`) is the armed `[lane_d] contexts`;
    with it the context must pass `declared_context(context, contexts)`;
    without it the context must still be in `SCORER_CONTEXTS`.
  - `horizon` must be in `[1, MAX_HORIZON]`.
  - Both checks raise `ValueError` **before** anything is loaded or
    `torch` is imported (the tests make `import torch` fail and pass a
    checkpoint dir that does not exist).
  - The model input of each window is its **last** `batch_plan` length
    points, batched by `batch_plan` (one forward per length), plus
    `ceil(h/32)` masked horizon patches. No edge pad. So with history 330 the
    forecast equals that of its last 320 points.
  - Returns `(n, horizon, 9)` finite float64 at `SCORING_LEVELS`.

Lane A equality (torch): the house (F32, median bias +3) on 12 windows with
history ≥ 4096, forecast at context 4096 and scored with `score_window`,
equals `evaluate_checkpoint(..., quantile_head=True)` → `stack_components` →
`wql_per_window` per window to `rel=1e-5` (MASE too). Different batching
(lane A: one window per forward; lane D: by length) is why it is not exact.

## Arming (rev6.6, owner 2026-11-23): `test_w5d_arming.py`

Plan rev6.6 (§1.1 context bullet, §1.7 readiness, T7 row). `[lane_d] contexts`
lists every variant from launch and is the set a miner may **declare**. A
variant is **armed** at a settlement only when both gates pass. The armed set is
computed here, in the numpy-free half of `lane_d_scorer.py`, so the box-side
settlement (W7a) can call it. It is never read from `cfg.contexts`.

- `GATE_MIN_SOURCES == 20` and `READINESS_SCHEMA == "laned_readiness/1"`.
- `armed_record(contexts, *, readiness, t7, min_long_series) -> dict`.
  `readiness` and `t7` are the pinned documents' **bytes**, or `None` when
  absent. The record is
  `{"armed_contexts": [...], "evidence": {"min_long_series", "min_sources",
  "readiness_sha256", "t7_sha256", "variants": [{"context", "series",
  "sources", "t7_fits", "armed"}, ...]}}`.
  - `variants` has one entry per value of `contexts`, ascending.
    `armed_contexts` holds the variants whose `armed` is true, ascending.
  - `*_sha256` is the sha256 hex of the bytes given, or `null` for an absent
    document.
  - Gate 1 (readiness, W1/R16 per-rung counts) is
    `{"schema": "laned_readiness/1", "rungs": [{"context", "series", "sources"}]}`.
    It passes iff `series >= min_long_series` and `sources >= 20`. A variant
    with no rung counts as `series = sources = 0`. A wrong schema, a missing
    `rungs`, a non-int, bool or negative count, a rung context outside
    `SCORER_CONTEXTS` or a duplicate rung is a `ValueError`.
  - Gate 2 (T7) reads the `laned_t7/1` evidence. It passes for a variant iff
    the document's `source` is `"pod"` and the variant's row has
    `peak_mem_source == "cuda_max_allocated"` and fits its budget
    (`wall_s <= budget.wall_s` and `peak_mem_gb <= budget.mem_gb`,
    **recomputed**). A row whose stated `fits` disagrees with its numbers is a
    `ValueError`, because a hand edit is refused and never trusted. Without a
    row the gate fails. Unparseable bytes or a schema other than `laned_t7/1`
    is a `ValueError`.
  - `contexts` must be a non-empty subset of `SCORER_CONTEXTS`, else
    `ValueError`. The record is deterministic: the same bytes give the same
    record, whatever the order of `contexts`.
- `effective_context(declared, armed_contexts) -> int | None` returns
  `declared` when it is armed. Otherwise it returns the highest armed variant
  **not above** `declared`. ~~When every armed variant is above `declared`,
  the lowest armed one.~~ **(plan rev6.15, owner 2026-11-23: "Highest ≤
  declared, else skip")** When every armed variant is above `declared` it
  returns **`None`**: the entry is not scored that settlement (the settlement
  skips it `context_below_armed`, W7a). A model is never run past the context
  its miner declared. In the normal case the gates are monotone (a series
  usable at 64k is usable at 32k), so 4096 is armed whenever anything is and
  `None` is rare. The scored context
  switches at the settlement where the declared variant arms. An
  empty armed set, or a declared value that is not a scorer context (or is a
  bool or str), is a `ValueError`.
- `declared_context` and `forecast_quantiles(..., contexts=)` are unchanged.
  The caller passes the context to forecast at (the effective one) and the
  set it may run at (the armed set).

On main these tests fail on the `_s()` assertion (the names are absent).

## Module contract: `cascade/eval/lane_d_t7.py`

The pod run calls these and writes `evidence/t7.json` with `json.dump`.

- `SCHEMA == "laned_t7/1"`, `LAUNCH_VARIANTS == (4096, 8192, 16384)`,
  `S8_HOST == "valeval-laned"`.
- `settlement_budget(mem_gb, *, settlement_blocks=900, block_seconds=12.0,
  fraction=0.5) -> {"wall_s": blocks*seconds*fraction, "mem_gb": mem_gb}`
  (§1.6: ≤ 50% of a settlement; mem = the host GPU's memory).
- `ShortHistoryError`: a `ValueError` subclass. Raised whenever a variant
  would be measured on a window with fewer than `variant` points of history
  (RV3: the no-pad scorer would silently run it at `min(n, context)`).
- `variant_row(*, host, gpu, variant, input_len, history_min, history_max,
  wall_s, peak_mem_gb, peak_mem_source, budget, n_models, n_windows) -> dict`
  with exactly those keys plus `fits = wall_s <= budget["wall_s"] and
  peak_mem_gb <= budget["mem_gb"]`. `ValueError` when `variant` is not in
  `LANE_D_CONTEXTS`, `n_models < 10` (8 entries + KING_D + house),
  `n_windows < 1`, a time/memory is not positive and finite, a host/gpu is
  empty, or the host rule breaks: variants ≤ 16384 on the launch eval host
  (never `valeval-laned`), variants > 16384 on `valeval-laned`. RV3 adds:
  - `input_len` is the model input length in points that actually ran (the
    single `batch_plan` key); `ValueError` unless `input_len == variant`.
  - `history_min` / `history_max` are the shortest and longest window
    history; `ShortHistoryError` when `history_min < variant`, `ValueError`
    when `history_max < history_min`. Since every window reaches the
    variant, `n_windows` is the count of windows run at the variant.
  - `peak_mem_source` is `"cuda_max_allocated"` (this variant's own device
    peak, after `reset_peak_memory_stats`) or `"process_peak_rss"` (CPU
    `ru_maxrss`, the process-lifetime peak every later row inherits);
    anything else is a `ValueError`.
- `build_evidence(rows, *, short_window, tool, torch_version, source) -> dict`
  with keys `schema, wp ("W5d"), source, torch, tool, measured_at (ISO 8601
  UTC, now), rows (sorted by variant), short_window, armed, passed`.
  - It re-checks every row it is handed (a hand-edited dict too):
    `input_len != variant` is a `ValueError`, `history_min < variant` a
    `ShortHistoryError`.
  - With `source == "pod"`, a row whose `peak_mem_source` is not
    `"cuda_max_allocated"` is a `ValueError`: process RSS is never a budget
    input. Other sources (CI) may carry `"process_peak_rss"` rows; the
    verifier rejects such evidence.
  - `short_window` gets `identical = (max_abs_diff == 0.0)`, recomputed.
  - `armed` = ascending variants whose row fits.
  - `passed` = identical **and** every launch variant measured **and** 4096 armed.
- `measure_variant(checkpoint_dir, histories, *, context, horizon, host, gpu,
  n_models, budget, device) -> row`: times the lane D scorer on `histories`
  at `context` and projects the settlement wall time for `n_models`.
  - Any history shorter than `context` points is a `ShortHistoryError`
    **before** `torch` is imported or the checkpoint is read (never a silent
    truncation).
  - The row's `input_len` is the model input length that ran (`== context`),
    `history_min`/`history_max` the window history range.
  - `peak_mem_gb` is `torch.cuda.max_memory_allocated` after a reset
    (`peak_mem_source "cuda_max_allocated"`) on CUDA; on CPU it is labelled
    `"process_peak_rss"`.
- `short_window_check(checkpoint_dir, histories, *, horizon) ->
  {"base_context": 4096, "declared_context": 131072, "n_windows",
  "max_abs_diff", "identical"}`: the checkpoint at 4096 vs a byte copy at
  131072. A history over 4096 points is a `ValueError` before loading.

## T7 evidence (`laned_t7/1`) and its verifier

`helpers.evidence_problems` is the verifier (independent of `lane_d_t7`). Per
row: `host, gpu, variant, input_len, history_min, history_max, wall_s,
peak_mem_gb, peak_mem_source, budget {wall_s ≤ 5400, mem_gb}, fits, n_models,
n_windows`. Per row (RV3) it requires `input_len` an int equal to `variant`,
`history_min` an int ≥ `variant`, `history_max` an int ≥ `history_min`, and
`peak_mem_source == "cuda_max_allocated"` (pod evidence; CPU
`process_peak_rss` is refused). It requires `source == "pod"` (the builder's CI
runs use `"ci"`), `torch == "2.4.1+cu124"`, a clean `tool.git_sha` commit
that has `cascade/eval/lane_d_t7.py`, unique variants covering the launch set,
the host rule, and recomputes `fits`, `armed`, `short_window.identical` and
`passed`. The gate test also wants `passed: true`. The `"wp"`/`"passed"`
keys also satisfy `accept`'s generic evidence step.

## Failure on main and the `torch` marker

- On main the W5d tests fail on `cannot import name` (`SCORER_CONTEXTS`,
  `declared_context`, `batch_plan` from `cascade.eval.lane_d_scorer`), on
  `No module named 'cascade.eval.lane_d_t7'`, or on an assertion (the
  `contexts` keyword / `MAX_HORIZON` checks run before any call).
- The W1.5-level grid and lane A numpy-equality tests pass on main.
- RV3 correction (tests-correction mode, W5d built on main): every test that
  builds rows with `lane_d_t7` asserts the RV3 surface first (`_rv3`:
  `variant_row` has `input_len, history_min, history_max, peak_mem_source`
  and `ShortHistoryError` exists), so on main it fails on that
  `AssertionError`, never a `TypeError`. The verifier-only tests on the
  hand-written `_fixture()` evidence pass on main.
- `test_evidence_*` skip while `$LANED_EVIDENCE` is unset.
- Tests marked `torch` need torch (CI: `uv sync --all-extras`). The
  production box runs only `-m 'not torch'`. `torch` is not registered in
  pyproject, so pytest warns about an unknown mark.

## Multichannel windows (rev6.18 F2): `test_w5d_multivariate.py`

Plan rev6.18 A4 (DEC-CA-0041 applied to lane D). A lane D window is either
univariate (1-D history and target, as before) or multichannel: a `(C, L)`
history with a `(C, horizon)` target, C >= 2, forecast jointly by one model
call. A `(1, L)` series is the univariate window.

- `cascade/eval/lane_d_channels.py` (numpy + hashlib) picks the channels a
  multichannel window is scored on: `empty_channels(history)` (channels with
  no finite history value) drop first; `pick_channels(block_hash, series_id,
  candidates, max_channels)` keeps all when they fit, else the
  `max_channels` with the smallest `sha256(seed + ":" + index)`, `seed =
  sha256(block_hash + series_id + "lane_d_channels")`, ascending (`()`
  without a block hash). `max_channels` is `[generator] max_channels`
  (32), read from the config. `window_pick`, `channel_plan` (first version of
  an id in pin order decides), `apply_pick` (one channel left gives 1-D
  arrays; `(1, L)` is flattened; a univariate window is returned unchanged),
  `channel_record` (`series_id, n_channels_raw, empty_channels, n_channels,
  channels, channels_sha256`).
- `score_window(q, target, history, sp)` with `q` of shape `(C, horizon,
  9)`: each channel scored as a univariate window, `crps` averaged
  (arithmetic) over the channels with `sum|y_c| > 0` (zero-target channels
  masked from the quantile half only), `mase` over every channel;
  `ValueError` when no channel has scale or a shape disagrees. A 2-D `q`
  runs the pre-F2 path unchanged.
- `forecast_quantiles` takes `(C, L)` histories and returns one row per
  channel, window-major: `(sum forecast_rows(histories), horizon, 9)`.
  Windows batch by (channel count, input length); univariate rows of a mixed
  request are bit-identical to a univariate-only request (torch test).
- Transport: `encode_request` keeps a `(C, L)` history 2-D (univariate
  stays flat); `forecast_entry` wants `(sum rows, horizon, 9)`;
  `SSHEvalHost._cut` cuts every channel along time.

On main these tests fail on `No module named 'cascade.eval.lane_d_channels'`
or on an assertion (scorer and transport refusals are re-raised as
`AssertionError`).
