# W-S2 acceptance tests: lane B guard/target exclusion for the sentinel (D33 rule 2, D35; TS4)

W-S2 is in the plan §3.0 table (rev4: now in §3):

> | W-S2 | **(rev4: now in §3)** lane B guard/target exclusion for the sentinel | — | W-S1 | ci | `scripts/laned/accept W-S2` (TS4) |

Its rule comes from §16 rule 2 ("**Lane B:** it's neither a guard nor a
target that era (D35)"), rule 3 ("**The other suites remain full regression
guards** in every lane"), D35 (decided: "A sentinel suite is not a lane B
target or guard that era") and TS4 ("Lane B entry targeting the sentinel |
Deferred (D35)"). The sentinel is W-S1's draw
(`cascade.shared.sentinel.sentinel_for_era` over the era-start block hash).
These tests are the spec. The builder may touch only the paths in `PATHS`.

**The target half is already built.** WB18 (on main) defers a suite-target
entry whose suite is the era's sentinel (`deferred: sentinel_target`). Its
tests pass on main and are not duplicated. Here
`test_ts4_sentinel_target_still_deferred` only checks that it still holds.
**The guard half is not built.** On main, every lane B public guard is
computed for all four suites in every era, so W-S2 isn't satisfied by WB18.

On main, every failure is an `AssertionError`:

- `judge_entry` / `judge_manifest` and `judge_acceptance` have no `sentinel`
  keyword;
- the validator's lane B judge still blocks a sentinel-only regression;
- the trainer's `settle()` holds when the king lacks a required sentinel
  suite, and calls `ops.judge` without the sentinel;
- `RunnerLaneBOps.judge` has no `sentinel` parameter.

These tests pass on main: `test_plan_rows_pinned`,
`test_ts4_sentinel_target_still_deferred` and the four
`test_inv1_*_unchanged_when_unarmed` tests.

## What runs for real, and what is stubbed

- **Real:**
  - the consensus judge, `cascade.validator.expert_lane.judge_manifest` →
    `judge_entry`, over a real `ExpertManifest` / `ExpertEntry` and real
    `WindowScore` lists. Its public guards come from the real
    `pair_public_rows` and are judged by the real `judge_expert_targets`;
  - `cascade.validator.merge_acceptance.judge_acceptance`;
  - `LaneBMixin._lane_b_judge_experts` on a runner that holds W-S1's
    recorded draw (`_settlement_sentinel`, read through the real
    `ValidatorRunner._sentinel_for`);
  - the trainer's `LaneBDriver.tick` and `settle`, on WB18's scenario (real
    intake, funding queue and driver). The config comes from `parse_lane_b`,
    and the era grid is a real `RoundConfig` (1000 blocks per era, one
    settlement).
- **Stubbed (GPU boundary or external only):**
  - `safetensors.torch.load_file`, `moe_model.merge_expert_slices` and
    `expert_lane._moe_names`, which are the torch tensor I/O and the
    expert-row byte check in `judge_entry`, plus the trainer's `settle_leg`
    (the torch rebase);
  - the window scores (the eval forward), checkpoint storage, the chain's
    block hashes, and the trainer's `ops.judge` / `bench_rows` / `upload` /
    `publish_expert_manifest`;
  - on the validator runner: `_lane_b_read`, `_verdict_windows`,
    `_evaluate`, `_fetch_checkpoint_dir` and `_training_for`.
- **No torch:** a meta-path guard makes any torch import raise.

The fixture leg targets `energy` with a 0.85× loss on it and leaves `web`
unchanged. The candidate's public item rows (40 items per suite) are 0.99× the
king's on every suite except the sentinel (`time`), where they are 1.30×.

## The contract

1. **Keyword.** `judge_entry(..., sentinel=None)` (or `judge_manifest`,
   which forwards keywords to it) and `judge_acceptance(..., sentinel=None)`
   take the era's sentinel suite name. `None` is the unarmed call.
2. **Not a guard.** With `sentinel=s`, the `s_crps` / `s_mase` public guards
   aren't built. They're absent from `public_lcbs` and from `unguarded`. A
   regression on `s` alone doesn't block, and the leg merges.
3. **The other suites stay full guards (§16 rule 3).** A regression on any
   other suite still blocks (`regression on <suite>_...`). Their LCBs, the
   domain guard LCBs and the target LCB are byte-identical to the unarmed
   judgement of the same rows. Pool-domain guards are unaffected.
4. **Not required.** `require_public` minus `s` is what must pair. Missing
   sentinel rows don't refuse the entry. A missing non-sentinel required
   suite still gives `public guard missing`.
5. **Rotation restores it.** With another suite as the sentinel, `s` is a
   guard again and blocks, and the new sentinel is excluded.
6. **Acceptance.** The same holds for the warm-start acceptance's public
   guards (`judge_acceptance`): a sentinel-only regression is accepted, a
   missing required sentinel suite doesn't refuse, and another suite's
   regression still rejects.
7. **The validator wiring.** `_lane_b_judge_experts` for a settlement
   judges with that settlement's recorded W-S1 draw
   (`_sentinel_for(round_id)["suite"]`). Its verdict row carries no
   `<s>_*` public LCB. With no record (unarmed), the verdict is today's.
8. **The trainer wiring.**
   - `settle()` reads the sentinel of the era the settlement belongs to
     (`settlement_era_start`: block 1000 for the boundary at 2000), through
     `ops.block_hash` and W-S1's draw. This happens **whenever the sentinel is
     armed**, with or without `[lane_b] suite_targets`. It is the same draw
     the validators record for that settlement.
   - It self-judges with `ops.judge(manifest, round_id, block, sentinel=s)`.
   - The king's `require_public` check doesn't require `s`. A missing
     non-sentinel suite still holds the settlement with `OSError`.
   - `RunnerLaneBOps.judge` has a `sentinel` parameter, defaulting to `None`,
     which it forwards to `judge_manifest`.
9. **Determinism.** The sentinel follows the block hash. An era-1 hash that
   draws `boom` gives `sentinel="boom"`.
10. **TS4 still holds.** A suite-target entry on the sentinel is deferred
    (WB18).

## Invariants (1, paired per invariant 10)

| test | pins |
|---|---|
| `test_inv1_judge_unchanged_when_unarmed` / `..._armed_nonempty_lane_d_effect` | no sentinel: all four suites guard and a missing required suite refuses / `sentinel="time"`: the sentinel-only regression merges |
| `test_inv1_acceptance_unchanged_when_unarmed` / `..._armed_nonempty_lane_d_effect` | the same for `judge_acceptance` |
| `test_inv1_validator_unchanged_when_unarmed` / `..._armed_nonempty_lane_d_effect` | no recorded draw: today's verdict / recorded `time`: excluded |
| `test_inv1_trainer_unchanged_when_unarmed` / `..._armed_nonempty_lane_d_effect` | `sentinel_from_block = 0`: no hash read, `ops.judge(manifest, round_id, block)` exactly as today, and a missing required suite still holds / armed: it settles and passes the sentinel |

## Not pinned

These are left to the builder:

- how the validator gets the sentinel for older settlements in the lane B
  catch-up walk (`_lane_b_walk_back`). Re-deriving it from the block hash or
  replaying the recorded receipt value (§16 rule 1) are both fine;
- how the trainer behaves when the settlement era's hash is unreadable;
- whether the published expert manifest's `public_rows` still carry the
  sentinel suite's rows. They're signed evidence, and judging ignores them;
- sentinel numbers on the status boards (W-S3);
- the DEC-CA-0071 amendment text (WB7 / Appendix C).

## Files

| file | pins |
|---|---|
| `PATHS` | the modules W-S2 may change |
| `helpers.py` | the scenario: real judge, acceptance, validator mixin and driver; the GPU-boundary stubs; the torch guard |
| `test_w_s2_guard_exclusion.py` | the plan text (W-S2 / TS4 / D35 / §16 verbatim), contract items 1–10, the invariant pairs |

## rev6.20 (owner 2026-11-24): stage `scored_weights`

`judge_entry` refuses an entry whose `scored_weights` is not `"ema010"`
(every lane B leg scores its power EMA), before any fetch.
`test_w_s2_scored_weights_stage.py`.
