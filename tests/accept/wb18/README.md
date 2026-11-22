# WB18 acceptance tests: sentinel interaction (D35 / LB5; TB19, TB20, TS4)

WB18 is in the plan §3.0 lane B track. Its rules come from §12b.5 LB5, D35
(decided: "A sentinel suite is not a lane B target or guard that era") and
the T-rows TB19, TB20 and TS4. The sentinel is the §16 draw built by W-S1
(`cascade.shared.sentinel.sentinel_for_era` over the era-start block hash).
These tests are the spec. The builder may touch only the paths in `PATHS`.

On main, every failure is an `AssertionError`:

- with `[lane_b] suite_targets = true`, the slot list has no `suite:*`
  targets, so a suite-target entry can't be committed or funded;
- `RunnerLaneBOps` has no `block_hash`.

`test_plan_rows_pinned`, `test_shipped_bound_is_four`,
`test_inv1_targets_unchanged_when_unarmed` and
`test_inv1_sentinel_unchanged_when_unarmed` pass on main.

## What runs for real, and what is stubbed

This is the same harness as WB15.

- **Real:** funding goes through `FundingIntake.lane_b_fund` (with
  `X-Lane-B-Targets: suite:<s>`) into the `cascade.funding.queue.LaneBQueue`
  file. The commitment is built by `format_expert_targets`. Planning is
  `LaneBDriver.tick`. Status is `LaneBDriver.status()` and
  `publish_lane_b_status` (`status/lane_b.json`). The config comes from
  `parse_lane_b`. The draw is W-S1's `sentinel_for_era`, and the era grid
  comes from a real `RoundConfig` (1000 blocks per era, one settlement).
- **Stubbed (external or GPU only):**
  - the driver's ops: commitments, pod dispatch, and **`block_hash(block)`**,
    the chain read. The fixture's hashes are chosen so that W-S1 draws the
    wanted suite for each era.
  - the king forward, through a torch-free `RoutingProbe`.
  - the reveal oracle and the signature check.
  - an in-memory object store.
- **No torch:** an autouse guard makes any torch import raise.
- **The driver's `cfg`:** `lane_b`, `v2` (`sentinel_from_block`,
  `sentinel_suites`, ...) and `round`. The sentinel arms from era 1, and era 0
  is the gate-opening era.

## The contract

1. **Suite targets are slots (DEC-CA-0071 §1).** With `suite_targets = true`,
   `LaneBConfig.targets` keeps its domains and horizons and gains
   `suite:gift`, `suite:boom`, `suite:time` and `suite:fev`. Each one commits
   as `s-<suite>` (`target_code`) and `parse_expert_targets` round-trips it.
   At the default (`false`), the slot list is unchanged.
2. **The era's sentinel.** The driver reads it only while
   `sentinel_armed(cfg, era_start)`, as `sentinel_for_era(ops.block_hash(era_start),
   cfg.v2.sentinel_suites)`. It reads only the era-start block. Unarmed, it
   never calls `block_hash`.
3. **Deferral (TB19, TS4).** A suite-target entry whose suite is the era's
   sentinel doesn't start a leg. `status()["deferred"]` and the published
   `status/lane_b.json` carry a row with `hotkey`, `entry_ref` (the funded
   entry) and `deferred: "sentinel_target"`.
4. **No charge.** The funded entry stays `queued` with
   `attempts == requeues == 0` and `last_error_class == ""`. It never goes in
   flight, it stays pending in the driver queue, and it writes no `outcomes`
   row. This holds across repeated ticks in the same era.
5. **Other targets are unaffected.** A suite target that isn't the sentinel,
   and any domain target, start normally in the same tick.
6. **Release on rotation.** When the next era draws another suite, the entry
   starts on its own target (`sub.declared == ("suite:<s>",)`) and its funded
   entry goes `in_flight`. An entry on the new sentinel is deferred instead.
7. **A deferred claim holds no rows.** An overlapping declared-rows entry on
   another target starts with its own rows. Once the deferral lifts, WB15's
   `rows_held` rule applies to the claim.
8. **The bound (TB20).** An entry is deferred in at most
   `sentinel_max_defer_eras` distinct eras, and several ticks in one era count
   once. In the next era it starts on its target even if that target is still
   the sentinel. The start is logged on `cascade.trainer.lane_b` at INFO or
   higher, with the hotkey's first 8 characters and the word
   `sentinel_max_defer_eras`. The tests check bounds 1 and 3. The shipped
   default is 4.
9. **The count is durable.** A trainer restart (a new `LaneBDriver` on the
   same state file) between deferred eras doesn't reset the bound.
10. **Production wiring.** `RunnerLaneBOps.block_hash(block)` is the chain
    client's `block_hash(block)`.
11. **Determinism.** The same plan comes out under permuted funding order and
    chain order, and under `PYTHONHASHSEED` 0 / 1 / 4242. The hash-seed check
    runs `python -P -m tests.accept.wb18.helpers` in fresh interpreters.

## Invariants (1, paired per invariant 10)

| test | pins |
|---|---|
| `test_inv1_targets_unchanged_when_unarmed` / `..._armed_nonempty_lane_d_effect` | slot list unchanged at `suite_targets = false` / gains `suite:*` when armed |
| `test_inv1_sentinel_unchanged_when_unarmed` / `..._armed_nonempty_lane_d_effect` | all at default: domain plan as today, no hash read, nothing deferred / armed: the sentinel suite target is deferred and the domain entry is not |
| `test_inv1_suite_target_unchanged_when_unarmed` / `..._armed_nonempty_lane_d_effect` | suite targets on, sentinel off: the entry starts, no hash read, `deferred == []` / sentinel on: deferred, uncharged |

## Not pinned

These are left to the builder:

- extra fields on a `sentinel_target` row, such as the suite, the eras
  deferred or the expected release;
- the exact wording of the TB20 log line, beyond the hotkey and the key name;
- where the era-deferral count lives in the driver state;
- the lane B guard exclusion for the sentinel, which belongs to W-S2;
- validator-side judging and pay of suite targets.

## Files

| file | pins |
|---|---|
| `PATHS` | the modules WB18 may change |
| `helpers.py` | the scenario (real intake, queue, driver, W-S1 draw), the stub probe and ops, the torch guard, the hash-seed entry point |
| `test_wb18_sentinel_deferral.py` | the plan text (WB18 / TB19 / TB20 / TS4 / D35 / LB5 verbatim), contract items 1–11, the invariant pairs |
