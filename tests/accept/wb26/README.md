# WB26 acceptance tests: guards 4 and 5 on lane B suite slots

WB26 is in the plan §3 lane B table (rev6.19, orchestrator-assigned
2026-11-24, narrowed onto rev6.22). It builds ON rev6.22's
`judge_expert_slots` (#854). That function tests every leg on every slot
with CRPS or MASE, in two max-T families: declared at `gain_margin`,
undeclared at `undeclared_gain_margin`. WB26 leaves the bars and families
unchanged and adds DEC-CA-0071 §3's two suite guards. These tests are the
spec; the builder touches only the paths in `PATHS`.

## The rule

- **Guard 4:** a suite slot that clears is claimable only if the
  whole-pool paired point gain is `≥ 0`. This is the contamination tripwire.
- **Guard 5, the anchor guard:** a cleared suite slot is also claimable only
  if every guard of the leg holds against the anchor king, at
  `tau_anchor_domain` / `tau_anchor_public` (default 0.0). The leg's guards
  are every domain (`_rest` pooled), every horizon and every public metric.
  They are re-run by the same guard engine with the anchor as the baseline.
- **Missing anchor:** no anchor, unpaired anchor windows, or a guarded public
  metric the anchor lacks is `anchor_missing`. The suite slots are refused
  (fail-closed).
- **Computed once, lazily:** both guards run once per leg, and only when a
  suite slot would be claimed. Until then the anchor is never scored. Domain
  and horizon slots are never affected.
- **Anchor:** the era's promotion member from the signed training manifest:
  `warm_start_base` when `warm_start_merges` is set, else `warm_start_ckpt`
  (`lane_b_anchor_pointer`).
- **Report and audit:** a claimed or refused cleared suite slot records
  `suite_guards` {`pool_gain`, `anchor_pointer`, `anchor_guard_lcbs`,
  `anchor_public_lcbs`}. `check_lane_b` re-derives it and checks it is the
  same across the leg.
- **Trainer:** every settling leg carries `public_rows["anchor"]`, benched
  once per anchor. The self-judge runs against the same anchor. The rolling
  trainer passes the anchor while `suite_targets` is on.
- **No activation block** (owner 2026-11-24). `suite_targets` arming is
  unchanged: it still needs the contamination hash set.

## Files

- `test_wb26_suite_guards.py`: the real `judge_expert_slots`.
- `test_wb26_validator.py`: the real `judge_manifest`, `judgements_record`
  and `LaneBMixin._lane_b_judge_experts`, with W-S2's GPU-boundary stand-in
  (no torch).
- `test_wb26_audit.py`: the real audit row check on honest rows from the
  validator and on tampered copies of them.
- `test_wb26_config.py`: the real loader and overlay path (WB10's helpers).
- `test_wb26_trainer.py`: WB18's real intake, funding queue and driver.

## On main

Every test fails with an `AssertionError`. The causes are:

- the `anchor` / `anchor_pointer` keywords are missing;
- the keys are not parsed;
- no `suite_guards` is recorded;
- the audit accepts a tampered row.

The one exception is an `ImportError` for a WB26 name in a PATHS module
(`lane_b_anchor_pointer`). No PATHS module is imported at module level.
