# WB23 acceptance tests: miner-chosen lane B leg length (D42, D42b; TB25–TB27)

Plan sources: the WB23 row (§3.0 lane B track), §12b.6 LB6 D42 item 2 and
item 5 (consensus), D42b = (a) **real hours on the chosen GPU** (owner
2026-11-21), T-rows TB25, TB26, TB27 (§4). Depends on WB10 (the keys
`max_leg_hours`, `max_leg_seconds`, `guard_factor`) and WB22 (lane B's GPU
list, the WB19 step-rate gate, `LaneBGpus.rate`).

The builder may touch only the paths in `PATHS`. Lane A stays byte-identical.
There is no dependency, `[training]` or contract-digest change:
`[training] max_train_seconds` is never edited. The lane B leg contract is
derived with the lane B cap. A leg funded **without** `train_hours` is
today's leg, byte for byte: same steps (`for_hours([lane_b] leg_hours)`),
same queue entry, same expert-manifest bytes. Nothing here needs torch or a GPU.

## `cascade/shared/lane_b_leg.py` (stdlib only, never imports torch)

| name | contract |
|---|---|
| `TRAIN_HOURS_HEADER` | `"X-Lane-B-Train-Hours"` |
| `HOURS_STEP`, `MIN_TRAIN_HOURS` | `0.5`, `0.5` |
| `BAD_TRAIN_HOURS` | `"bad_train_hours"` (typed refusal code) |
| `NEEDS_GPU` | `"train_hours_needs_gpu"`: D42b(a) hours are on the chosen GPU, so a `train_hours` fund must choose a measured type |
| `WITH_FALLBACK` | `"train_hours_with_fallback"`: a fallback type would train the fixed steps at another rate, so `train_hours` with a GPU fallback is refused |
| `UNAVAILABLE` | `"train_hours_unavailable"`: an intake with no lane B GPU list wired (dev/unit) |
| `parse_train_hours(raw, *, max_leg_hours) -> float \| None` | `raw` stripped; `""`/None gives None (no header: today's leg). A finite decimal that is a whole multiple of 0.5 inside `[0.5, max_leg_hours]` is returned as a float. Anything else (`"0"`, `"0.25"`, `"0.75"`, `"6.5"` at a 6.0 cap, `"nan"`, `"inf"`, `"-1"`, `"abc"`, `"1e400"`) raises `LegLengthError` with `.code == BAD_TRAIN_HOURS` |
| `LegLengthError(ValueError)` | carries `.code` and the message |
| `train_hours_tag(raw) -> str` | `"hours=<raw stripped>"`, or `""` when absent |
| `reference_steps(contract, train_hours) -> int` | `max(1, round(contract.training_steps * train_hours / contract.target_train_hours))` (the steps a lane A leg of those hours runs: `for_hours`'s formula) |
| `leg_steps(train_hours, steps_per_s, *, cap) -> int` | `min(floor(train_hours * 3600 * steps_per_s), cap)`; `ValueError` when that is < 1 |
| `leg_guard_seconds(train_hours, *, guard_factor, overhead_seconds, max_leg_seconds) -> int` | `min(ceil(train_hours * 3600 * guard_factor + overhead_seconds), max_leg_seconds)` |
| `LegLength` | frozen dataclass `(train_hours: float, steps: int, guard_seconds: int)` with `as_dict() -> {"train_hours", "steps", "guard_seconds"}` |
| `admit_leg_length(lb_cfg, rate, train_hours, *, contract) -> LegLength` | `rate` is a WB19 `StepRate`. Steps from `leg_steps(..., rate.steps_per_s, cap=reference_steps(contract, h))`. Guard from `lb_cfg.guard_factor`, `rate.overhead_seconds`, `lb_cfg.max_leg_seconds` |
| `leg_contract(base, length) -> contract` | `dataclasses.replace(base, training_steps=steps, target_train_hours=train_hours, max_train_seconds=guard_seconds, extra_sizes=())`. The guard may exceed `base.max_train_seconds` (the lane B cap replaces it). `base` is never mutated |
| `check_leg_record(*, train_hours, steps, guard_seconds, lb_cfg, contract) -> str \| None` | consensus check of a recorded leg. All three 0 gives None (a leg without `train_hours`: today's). Otherwise `"train_hours_out_of_range"` (not a 0.5 multiple in `[0.5, max_leg_hours]`), `"steps_out_of_range"` (not `1 ≤ steps ≤ reference_steps`), `"guard_out_of_range"` (not `min(ceil(h*3600*guard_factor), max_leg_seconds) ≤ guard ≤ max_leg_seconds`), checked in that order; else None. It needs no step-rate table (satellites never have one) |

## Intake (TB25)

`POST /v1/lane-b/fund` with `X-Lane-B-Train-Hours`:

- the header is parsed **before** anything is queued or vaulted. Refusals
  are 400 with the codes above. Nothing is queued, no key is vaulted, so
  nothing is charged. The checks run in this order: `UNAVAILABLE` (no
  `lane_b_gpus`), `BAD_TRAIN_HOURS`, `NEEDS_GPU` (no `X-Gpu-Type`),
  `WITH_FALLBACK` (`X-Gpu-Allow-Fallback` true). The WB22 GPU checks
  (`unknown_gpu`, `gpu_rate_unmeasured`) keep their place before these;
- the signed ref is `lane_b_signed_ref(targets, ref, declared_rows,
  train_hours=raw)`. With a header, `"|" + train_hours_tag(raw)` is appended
  after everything else (after the WB14 `rows=` tag). Without one, the field
  is today's, byte for byte;
- admitted: `LegLength` from `admit_leg_length(lane_b_cfg, rate, h,
  contract=<contract at [v2] from_block>)`. The queue entry carries
  `train_hours`, `leg_steps`, `leg_guard_seconds`. The 202 reply adds
  `train_hours`, `steps`, `guard_seconds` (and WB22's `gpu`, `steps_per_s`);
- `FundingIntake(..., lane_b_landing=None)`: an optional
  `callable(guard_seconds) -> dict | None`. When it is wired and returns a
  dict, the reply carries it as `landing` (the expected landing boundary:
  `target_boundary`, `era_index`, `cross_era`). An exception in it never fails
  the fund (no `landing` key).

## Queue

- `FundedEntry.train_hours: float = 0.0`, `leg_steps: int = 0`,
  `leg_guard_seconds: int = 0`. A queue file written before WB23 loads with
  zeros.
- `LaneBQueue.add_lane_b(..., leg_length=None)` stores them. `public_view()`
  rows show them when > 0.

## Trainer (TB25, TB26)

- `_mark_funded`'s funding record adds `train_hours`, `leg_steps`,
  `leg_guard_seconds` when the entry has them (> 0). Without them the record
  is today's.
- `_dispatch_funded`: a funding record with `train_hours > 0` trains
  `leg_contract(base, LegLength(...))`, with `base = throne_contracts_at(era
  start)[0]`. Without it the contract is today's `for_hours([lane_b]
  leg_hours, ...)`.
- `leg_length_admission(round_cfg, *, block_now, now, guard_seconds,
  margin_seconds, current_era)` (module function in `lane_b_driver.py`) is
  `rolling.admit(..., wall_seconds=guard_seconds, ...)`. A funded entry whose
  admission is `cross_era` is not started this era. It stays queued and
  uncharged, and status shows `deferred: leg_length` with the
  `target_boundary` (TB26).
- `status()["open"]` rows carry `train_hours`, `steps`, `guard_seconds` (0
  when not set) next to `gpu`.
- `settle` writes the funding record's leg length into
  `ExpertEntry(train_hours=, leg_steps=, leg_guard_seconds=)`.

## Signed leg record (`ExpertEntry`) and the validator (TB27)

- `ExpertEntry.train_hours: float = 0.0`, `leg_steps: int = 0`,
  `leg_guard_seconds: int = 0`. Each is dropped from `canonical_body()` at 0,
  so an entry without them keeps main's bytes. They survive `to_json` /
  `load_expert_manifest`. A negative value, a bool or a non-finite
  `train_hours` raises `ValueError`.
- `judge_entry(..., contract=None)` and `judge_manifest(..., contract=)`:
  when `contract` is given and `check_leg_record` returns a reason, the entry
  is refused with `stage == "leg_length"` and that reason, **before** any
  checkpoint fetch or evaluation. Every node computes the same verdict from
  the signed bytes, so satellite replay agrees. An entry without a leg length
  is judged exactly as today.
- `cascade/validator/lane_b.py` passes the round's training contract
  (`contract=`) into `judge_manifest`.

## Builder-chosen rules (flagged for owner review)

The plan doesn't pin these. The tests fix them:

1. A `train_hours` fund must choose a measured GPU and must not allow a
   fallback (D42b(a): hours on the chosen GPU).
2. The consensus steps cap is `reference_steps` (a lane B leg never trains
   more steps than a lane A leg of the same hours). Admission clamps to it.
3. The guard floor the validator accepts is `min(ceil(h*3600*guard_factor),
   max_leg_seconds)`. The measured overhead is trainer-side, so only the
   ceiling `max_leg_seconds` and that floor are consensus.

## Failure on main (expected, right reasons only)

`cascade.shared.lane_b_leg` doesn't exist (`ModuleNotFoundError` for a PATHS
module). The keyword, field, status and source checks fail with
`AssertionError`. A few guards pass on main: today's leg without
`train_hours`, and lane A unchanged.

Run locally (tests only, no torch):

```
nice -n 19 env PYTHONPATH=<wt> python -P -m pytest --rootdir <wt> \
  -p no:cacheprovider -m 'not torch' <wt>/tests/accept/wb23
```
