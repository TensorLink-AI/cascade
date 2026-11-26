# W0b acceptance tests: context equivalence

W0b (plan rev6 §3.0 W0b row, §1.1, B7) pins two facts about the lane D
scorer `cascade/eval/lane_d_scorer.py` (W5d) that W5d's own tests do not:

1. **Quantile-output equality with the pinned wrapper.** On windows with
   ≥ 4096 history, `forecast_quantiles(..., context=4096)` equals the pinned
   `pyramid_wrapper` quantile head read through lane A's scoring path under
   `[v2] quantile_head_after` (`cascade.validator.evaluator.load_quantile_forecaster`:
   the checkpoint's `forecast_wrapper.py`, `forecast_quantiles_batch`, the 9
   grid levels picked by exact match). Quantiles are compared directly, not
   CRPS/MASE (W5d's `test_house_at_4096_equals_its_lane_a_per_window_crps`
   compares scores) and not samples.
2. **Strict loading at every context variant.** A 4096 checkpoint loads
   `strict=True` into the model the scorer builds at every
   `SCORER_CONTEXTS` value.

## What is pinned

| Item | Test |
|---|---|
| the variant list W0b sweeps is `lane_d_scorer.SCORER_CONTEXTS`; `[lane_d] contexts` (chain.toml) and `LANE_D_CONTEXTS` are subsets of it | `test_w0b_iterates_exactly_the_scorer_contexts`, `test_lane_d_contexts_are_a_subset_of_the_scorer_contexts` |
| the scorer reads the 9 `SCORING_LEVELS` out of the head's Q21 | `test_scorer_reads_the_nine_level_grid_out_of_the_head` |
| the fixture checkpoints are the pinned format (config 4096, byte-identical `model.py` / `forecast_wrapper.py`) | `test_checkpoint_is_the_pinned_4096_format` (torch) |
| lane D at 4096 vs the wrapper, one window per forward (lane A's `quantile_fn`): max abs diff ≤ `ATOL` | `test_scorer_at_4096_equals_pyramid_wrapper_quantile_head_per_window` (torch, F32 and BF16) |
| the same vs the wrapper's batched forward (`quantile_fn.batch`) | `test_scorer_at_4096_equals_pyramid_wrapper_quantile_head_batched` (torch, F32 and BF16) |
| control: the wrapper's edge pad on a 320-point window differs from the no-pad scorer by > 100 × `ATOL` (the tolerance can catch a real divergence) | `test_tolerance_discriminates_the_wrapper_edge_pad` (torch) |
| at every `SCORER_CONTEXTS` value the scorer calls `load_state_dict(strict=True)` once, with no missing / unexpected keys, on a model built at that context whose config differs from `config.json` only in `context_length`; afterwards the model's state dict has exactly the file's keys and every tensor equals the file's (floating upcast to fp32) | `test_4096_checkpoint_loads_strict_at_every_context_variant` (torch, F32 and BF16) |
| negatives: a copy with an extra key fails with `Unexpected key`, a copy with a dropped key with `Missing key`, at 4096 and 131072 | `test_checkpoint_with_an_extra_key_fails_strict`, `test_checkpoint_with_a_missing_key_fails_strict` (torch) |

The strict-load sweep observes the scorer's own load through a spy on
`CascadeMoEModel.load_state_dict` that calls the original unchanged (the
component is never replaced).

## Tolerance

`ATOL = 1e-4` (data units). Both sides run the same fp32 forward with the
same weights on the same last 4096 points (BF16 is upcast to fp32 on both:
the scorer before `load_state_dict`, the wrapper inside it). The fixture
series sit near 20, where one fp32 ulp is about 2e-6, so 1e-4 is about 50
ulp. It allows only reduction-order differences between batch shapes (lane
A decodes one window per forward, the scorer one forward per input length).
The edge-pad control shows a real divergence is orders of magnitude larger.

## Fixtures

Tiny random-init 2m pyramid checkpoints (`cascade-moe-v2-2m` via
`scripts/pyramid_config.py`) written by `pyramid_trainer.save_checkpoint`:
one F32 with a +3 median bias (a lane A king), one BF16 (a lane D entry).
Twelve windows of 4096 to 6000 points, including non-patch-multiple lengths.

## Verification WP and the `torch` marker

W0b is a verification WP. W5d's scorer on main already builds the model from
`config.json` with only `context_length` replaced, loads `strict=True` after
an fp32 upcast, and runs the wrapper's arithmetic without the pad, so these
tests are expected to pass on main. `PATHS` names the module that would
change if one failed. Tests marked `torch` need torch (CI:
`uv sync --all-extras`); the production box runs only `-m 'not torch'`.
