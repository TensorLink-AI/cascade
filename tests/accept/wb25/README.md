# WB25 acceptance tests: the contract they pin

WB25 (plan §3.0, lane B track, rev6.8, owner 2026-11-23: "Can make lane b
multi gpu") trains **lane B expert legs on every GPU of the rented layout**.
It runs in `ci + pod`. The builder may touch only the paths in `PATHS`:

- `cascade/trainer/pyramid_trainer.py` (`train_pyramid`)
- `cascade/trainer/pyramid_dp.py` (`Gang`, `_follower`, the reducers)

Not in PATHS (the tests rely on them unchanged): `expert_training.py`
(`ExpertSlices` is reused as is), `expert_lane.py` (selection, merge),
`pyramid_optimizer.py` (`assign_muon_owners` is reused), `pyramid_model.py`.
No dependency, `[training]`, `base_arch_digest` or contract-digest change.

**The gap on main (cbe4c79).** `train_pyramid` sets
`world = gang_size(trainer.device) if slices is None else 1`, so an
expert-slice leg trains on GPU0 even on an 8× pod (a WB19 run on 8×H200:
0.055 steps/s, the same as an RTX 4090). `_follower` builds lane A helpers
only ("the gang only trains lane A (no slices)").

## How the tests run

- `test_wb25_structure.py`: AST and text only. No torch, so it runs on the
  box and in CI.
- `test_wb25_gang_torch.py`: marked `torch`, **CI only** (CPU, no GPU).
  Tiny `2m` pyramid legs on CPU: `training_steps=3`, `batch_size=8`,
  **`micro_batches=4`** (so both ranks own micro-batches), one CPU thread
  per leg. The king is a 1-rank random-init lane A leg. The selection is the
  four most-loaded experts per MoE layer (as in WB16).

### The CPU seam (what the builder must provide)

- **World size.** A test forces a 2-rank gang by patching
  `cascade.trainer.pyramid_dp.gang_size` to return 2. `train_pyramid` must,
  as on main, look `gang_size` up on `pyramid_dp` when the leg starts
  (a function-local `from .pyramid_dp import ...`), and call it on
  `trainer.device` for every leg, lane A or lane B.
- **Backend follows the device.** On a CUDA trainer device the Gang is
  today's (NCCL with `NCCL_PIN`, rank r on `cuda:r`). On a CPU device every
  rank runs on CPU with the **gloo** backend. That is the only way CI (CPU
  only) can run a real multi-rank leg. Lane A's CUDA path is unchanged.
- **Spies on rank 0.** `pyramid_dp.Gang` is wrapped in a subclass that
  records `world`, so the Gang must be built as `pyramid_dp.Gang(world, ...)`
  looked up at leg time. `torch.distributed.all_reduce` is wrapped to record
  `(shape, dtype, numel)`. `pyramid_optimizer.assign_muon_owners` is wrapped
  to record `(param shapes, world)`. Helper ranks are spawned processes and
  are not recorded; their collectives mirror rank 0's.
- A 2-rank leg that raises (main's CUDA-only Gang on a CPU runner) is
  reported as an `AssertionError` naming the exception. A leg that does not
  build a 2-rank Gang fails `Gang worlds == [2]`. The shared 2-rank leg is
  built inside each test body (not in a fixture), so on main these are test
  failures, not fixture errors.

## Structure (`test_wb25_structure.py`)

| test | pins |
|---|---|
| `test_slice_legs_take_the_gang_size` | every assignment to `world` in `train_pyramid` is a call `gang_size(trainer.device ...)`: no conditional expression, and no `slices` / `expert_selection` in it |
| `test_no_world_one_override_anywhere_in_train_pyramid` | the text `if slices is None else 1` is gone |
| `test_gang_receives_the_selection` | every `Gang(...)` call in `train_pyramid` passes `slices` or `expert_selection` (positional or keyword, possibly inside an expression) |
| `test_helper_ranks_can_build_expert_slices` | `pyramid_dp` imports `ExpertSlices` from `.expert_training` and constructs it |
| `test_helper_ranks_accelerate_only_lane_a` | every `accelerate(...)` call in `pyramid_dp` sits under an `if` or a conditional expression. Rank 0 accelerates only lane A, so a helper rank of an expert leg must not either (a mismatch would put the ranks on different `micro_batch_pass` branches) |
| `test_lane_a_rank0_still_accelerates_only_without_slices` | guard: `train_pyramid`'s `accelerate` stays under `if slices is None:` |
| `test_lane_a_keeps_the_gang_path` | guard: `train_pyramid` still imports `Gang`, `gang_size`, `micro_batch_pass`, `owned_splits` from `.pyramid_dp` and calls `micro_batch_pass` and `shard_optimizer` |
| `test_gang_size_still_reads_cascade_train_gpus` | guard: `GANG_ENV = "CASCADE_TRAIN_GPUS"` and `gang_size` caps by `device_count` |
| `test_nccl_pin_unchanged` | guard: `NCCL_PIN` is exactly today's |
| `test_torch_pin_unchanged` | guard: `torch==2.4.1` in `pyproject.toml` and `2.4.1` / `2.4.1+cu124` in `uv.lock` |
| `test_main_snapshot_is_main` | the oracle files' sha256 (below) |
| `test_plan_row` | the plan's WB25 row: 6 cells, `rev6.8`, `gpu_n`, `CASCADE_TRAIN_GPUS`, Depends `WB16, WB19`, `ci + pod`, Accept `scripts/laned/accept WB25` |

## Lane B on the gang (`test_wb25_gang_torch.py`)

| test | pins |
|---|---|
| `test_slice_leg_trains_on_the_gang` | the 2-rank expert leg builds `Gang(2, ...)`, keeps `expert_training_backend == "compact_rows"`, records **`metrics["gpu_n"] == 2`**, and has the 1-rank leg's step count and `expert_trainable_param_count` |
| `test_one_rank_slice_leg_records_gpu_n_1` | a 1-rank expert leg records `gpu_n == 1` |
| `test_gang_all_reduces_only_selected_rows` | with `L` = the numel of the compact selected-row leaves and `S` steps, the float elements all-reduced on rank 0 over the leg are `≥ S·L` and `≤ S·(2L + 2)` (the marginal and copula gradients of the leaves, plus the two losses; integer loads and flags are not counted). No float all-reduce has the shape of a frozen tensor (more than 2 elements) that is not a leaf shape. Bucketing the leaves is allowed |
| `test_optimizer_sharded_over_selected_rows` | rank 0 calls `assign_muon_owners` with world 2 over exactly the leaves (shape multiset) |
| `test_router_and_bias_byte_identical_after_two_rank_leg` | every `.router.` weight and `route_bias` equals the king's (WB16, D40 a) |
| `test_two_rank_leg_matches_one_rank_leg` | the same file set as the 1-rank leg. Selected rows: `assert_close(rtol=1e-4, atol=1e-6)` with the 1-rank leg. Every other tensor and every unselected row: `torch.equal` to the 1-rank leg and to the king. At least one selected row moved. `final_loss` within `rel=1e-5` |
| `test_merge_accepts_a_multi_gpu_leg[moe_model\|pyramid_model]` | `merge_expert_slices(king, leg, selection, names)` accepts the 2-rank leg |
| `test_two_rank_leg_is_deterministic` | a second 2-rank leg gives identical file bytes and identical non-timing metrics |
| `test_gang_size_honours_cascade_train_gpus` | guard: with 8 visible CUDA devices `gang_size("cuda")` is 8, `CASCADE_TRAIN_GPUS=3` lowers it to 3, and `gang_size("cpu")` is 1 |

## Lane A unchanged (byte for byte)

The oracle is main's two PATHS files at `cbe4c79`, kept byte for byte in
`main_snapshot/` (`pyramid_trainer.py.txt` sha256 `4521dda1…dad79`,
`pyramid_dp.py.txt` sha256 `6842716e…afe8b`). `helpers.load_main_oracle`
loads them as `cascade.trainer._wb25_main_pyramid_trainer` and
`cascade.trainer._wb25_main_pyramid_dp`, wired to each other. Every other
import is the live module, and each oracle module's `__file__` is its live
sibling path, because `save_checkpoint` copies `pyramid_model.py` and
`pyramid_wrapper.py` from next to it. Both sides are called as
`train_pyramid(Toto2Trainer("cpu"), stream, contract, training_seed=1, ...)`
with the arguments `Toto2Trainer.train` passes.

| test | pins |
|---|---|
| `test_lane_a_random_init_leg_is_main_byte_for_byte` | every output file has the same bytes, and the non-timing metrics are equal (same keys, same values) |
| `test_lane_a_warm_start_leg_is_main_byte_for_byte` | the same, warm-started from the king |
| `test_one_rank_slice_leg_is_main_byte_for_byte` | a 1-GPU lane B leg is main's leg byte for byte. Only the new `gpu_n` metric is extra |
| `test_lane_a_metrics_gain_no_gpu_n` | lane A metrics (they feed receipts) gain no key |
| `test_lane_a_two_rank_leg_keeps_the_gang_path` | lane A on the 2-rank CPU gang equals the 1-rank leg within the same tolerance on every tensor. It all-reduces more float elements than the model has (the full-model path), and records no `gpu_n` |

The timing metrics left out of every comparison are `train_seconds`,
`throughput_tokens_per_s`, `data_wait_s`, `deadline_hit` and
`corpus_exhausted`.

## Out of scope

- The pod run (`ci + pod`): an 8× pod expert leg that uses every GPU. It is
  WB19's tool with a GPU count above 1 (optional). It is not pinned here.
- WB22 carries the recorded `gpu_n` into its audit re-derivation. That is
  WB22's PATHS, not WB25's.

## Failure on main (expected, for the right reason: assertions only)

The structure tests fail on the `world = ... if slices is None else 1`
override, on the selection never reaching `Gang(...)`, on `pyramid_dp`
never building `ExpertSlices`, and on the unconditional helper `accelerate`.
`test_plan_row` fails until plan rev6.8 merges. Each 2-rank expert test
fails with `Gang worlds: []` (the leg ran on one rank). The lane A 2-rank
test fails because main's Gang is CUDA-only (reported as an assertion).
`test_one_rank_slice_leg_records_gpu_n_1` fails because there is no
`gpu_n`. The guards and the byte-for-byte lane A and 1-rank lane B tests
pass on main.
