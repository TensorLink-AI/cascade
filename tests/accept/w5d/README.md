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

## Module contract: `cascade/eval/lane_d_t7.py`

The pod run calls these and writes `evidence/t7.json` with `json.dump`.

- `SCHEMA == "laned_t7/1"`, `LAUNCH_VARIANTS == (4096, 8192, 16384)`,
  `S8_HOST == "valeval-laned"`.
- `settlement_budget(mem_gb, *, settlement_blocks=900, block_seconds=12.0,
  fraction=0.5) -> {"wall_s": blocks*seconds*fraction, "mem_gb": mem_gb}`
  (§1.6: ≤ 50% of a settlement; mem = the host GPU's memory).
- `variant_row(*, host, gpu, variant, wall_s, peak_mem_gb, budget, n_models,
  n_windows) -> dict` with exactly those keys plus `fits = wall_s <=
  budget["wall_s"] and peak_mem_gb <= budget["mem_gb"]`. `ValueError` when
  `variant` is not in `LANE_D_CONTEXTS`, `n_models < 10` (8 entries + KING_D
  + house), `n_windows < 1`, a time/memory is not positive and finite, a
  host/gpu is empty, or the host rule breaks: variants ≤ 16384 on the launch
  eval host (never `valeval-laned`), variants > 16384 on `valeval-laned`.
- `build_evidence(rows, *, short_window, tool, torch_version, source) -> dict`
  with keys `schema, wp ("W5d"), source, torch, tool, measured_at (ISO 8601
  UTC, now), rows (sorted by variant), short_window, armed, passed`.
  - `short_window` gets `identical = (max_abs_diff == 0.0)`, recomputed.
  - `armed` = ascending variants whose row fits.
  - `passed` = identical **and** every launch variant measured **and** 4096 armed.
- `measure_variant(checkpoint_dir, histories, *, context, horizon, host, gpu,
  n_models, budget, device) -> row`: times the lane D scorer on `histories`
  at `context` and projects the settlement wall time for `n_models`;
  `peak_mem_gb` is the peak device memory (on CPU, the process peak RSS).
- `short_window_check(checkpoint_dir, histories, *, horizon) ->
  {"base_context": 4096, "declared_context": 131072, "n_windows",
  "max_abs_diff", "identical"}`: the checkpoint at 4096 vs a byte copy at
  131072. A history over 4096 points is a `ValueError` before loading.

## T7 evidence (`laned_t7/1`) and its verifier

`helpers.evidence_problems` is the verifier (independent of `lane_d_t7`). Per
row: `host, gpu, variant, wall_s, peak_mem_gb, budget {wall_s ≤ 5400, mem_gb},
fits, n_models, n_windows`. It requires `source == "pod"` (the builder's CI
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
- `test_evidence_*` skip while `$LANED_EVIDENCE` is unset.
- Tests marked `torch` need torch (CI: `uv sync --all-extras`). The
  production box runs only `-m 'not torch'`. `torch` is not registered in
  pyproject, so pytest warns about an unknown mark.
