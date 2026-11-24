# WB26 acceptance tests: lane B suite-target judging

WB26 is in the plan §3 lane B table (rev6.19, orchestrator-assigned
2026-11-24). It builds the judging side of DEC-CA-0071 suite targets with
the owner's 2026-11-24 rule (Amendment rev6.19, plan rev6.19 (G)). These
tests are the spec; the builder touches only the paths in `PATHS`.

## The rule

- **Gain:** a suite target `suite:<s>` wins when CRPS OR MASE clears
  `LCB_{α/(2k)} ≥ gain_margin_public` (0.01) on the whole suite's paired
  items with a positive point gain. Bonferroni α/2 per metric, on top of
  the leg's existing α/k.
- **The other metric:** if only one metric clears, the other must hold
  `LCB_α ≥ −tau_public`.
- **Floor:** fewer than `min_public_items` items ⇒ `inconclusive: items`.
- **Full pay:** `target_lcb`, the `full_pay` tier input, is the SMALLER
  cleared LCB (orchestrator 2026-11-24).
- **Guards:** every existing guard, minus the target suite's own two keys.
  Plus guard 4 (whole-pool point gain ≥ 0) and the anchor guard (every
  domain, horizon and public guard re-run against the anchor king at
  `tau_anchor_domain` / `tau_anchor_public`, default 0.0). Both run once per
  leg. A missing anchor is `anchor_missing` (fail-closed).
- **Anchor:** the era's promotion member from the signed training manifest:
  `warm_start_base` when `warm_start_merges` is set, else `warm_start_ckpt`.
  Its public rows are `ExpertEntry.public_rows["anchor"]`.
- **Report:** `suite_gain` per suite row; `suite_guards` per leg.
  `check_lane_b` re-derives both.

## Files

- `test_wb26_suite_gain.py`: the real `judge_expert_targets`.
- `test_wb26_validator.py`: the real `judge_manifest`, `judgements_record`
  and `LaneBMixin._lane_b_judge_experts`, with W-S2's GPU-boundary stand-in
  (no torch).
- `test_wb26_audit.py`: the real audit row check on honest rows from the
  validator and on tampered copies of them.
- `test_wb26_config.py`: the real loader and overlay path (WB10's helpers).
- `test_wb26_trainer.py`: the trainer side (below).

## On main

Every test fails with an `AssertionError` (the `anchor` / `anchor_pointer`
keywords are missing, the keys are not parsed, a suite target is
`inconclusive: windows`, or the audit accepts a tampered row). The one
exception is an `ImportError` for a WB26 name in a PATHS module
(`lane_b_anchor_pointer`). No PATHS module is imported at module level.

## Trainer side (in scope, orchestrator 2026-11-24)

`test_wb26_trainer.py` runs WB18's real intake, funding queue and driver. A
settling suite-target leg carries the era anchor's bench rows
(`public_rows["anchor"]`). They are benched once per anchor, and the
self-judge is called with the same `anchor_pointer`. The rolling trainer
derives the anchor with the validator's `lane_b_anchor_pointer` while
`suite_targets` is on. A domain leg never benches the anchor.

`suite_targets` arming is unchanged: it still needs the contamination hash
set.
