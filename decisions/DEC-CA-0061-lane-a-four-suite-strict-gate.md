---
id: DEC-CA-0061
type: decision
title: "Lane A public gate: four suites (GIFT-Eval, BOOM, TIME, FEV); rule v2 (2026-10-20) avg_noninf_winrate — per task, better on average, no suite significantly worse, win rate >= 0.5; the private pool still decides by geomean LCB"
status: proposed
date: 2026-10-19
tags: [consensus, scoring, benchmarks, v2]
revisit_when: "the seed-noise calibration (scripts/bench_seed_noise.py) shows same-recipe seed pairs exceeding 1 + bench_gate_tolerance on a suite's ratio LCB, or failing the 0.5 win rate, more than rarely; or genuine pool winners are blocked mostly by one condition; or FEV bench time pushes the bench report past the settlement it gates"
relations: {depends_on: [DEC-CA-0054], supersedes_behavior_of: [DEC-CA-0054], relates_to: [DEC-CA-0055, DEC-CA-0063]}
---

## Rule v2 2026-10-20 (owner)

Owner decision (2026-10-20): the public benches must show "improvement on
average and no statistically significant regression". This rule REPLACES
the strict rule below IN PLACE: same `RULE_V2` slot, same `[v2]
bench_gate_v2_from_block` key, new name `"avg_noninf_winrate"`. No
settlement was ever judged under the strict rule. Mainnet v2 is not live
(`from_block = 0`), and nothing on this branch ran on testnet (owner,
2026-10-20), so no replay has to be preserved. Testnet now also types
`bench_gate_v2_from_block = "v2"`. It stays inert there while testnet's
`[scoring]` / `[v2] bench_gate_mode` are `"off"`.

**Why the strict rule was dropped.** After ~50B-token legs from the same
init, challenger and king differ on the public suites by about training-seed
noise. Eight strict point comparisons then fail a real private-pool winner
on one hair-worse number almost every time. Bench forecasts of a 9-quantile
checkpoint are deterministic (`benchmarks/cascade_benchmark/predictor.py`),
so this noise is training noise across tasks, not sampling noise.

**The rule** (`cascade/eval/bench_gate.py`, `_evaluate_v2`). It can only
BLOCK a private-pool win.

- **Unit = one benchmark task.** A task is one signed item row: the
  sidecar's `full` key, i.e. a GIFT-Eval config, a BOOM dataset, a TIME
  task or an FEV task. Rows are already per task, so no further
  aggregation is needed. Series inside a task are never resampled on their
  own. Only tasks that both reports scored, with finite CRPS AND MASE > 0
  on both sides, count (`pair_public_rows`). Zero or non-finite
  denominators are masked, as DEC-CA-0009 masks them. Each suite records
  its `masked` count. A suite with fewer than `[scoring]
  bench_gate_min_items` (10) usable tasks is uncomputable, and an
  uncomputable gate holds the round under `enforce`.
- Per task: `r_crps = chal/king`, `r_mase = chal/king`, and the task score
  `s = sqrt(r_crps · r_mase)`. Per suite: `R_s` is the geomean of every
  `r_crps` and `r_mase`, which equals the geomean of the task scores.
1. **Average.** `geomean_s(R_s) < 1`, as a point estimate (the private pool
   already supplies significance). Each SUITE gets equal weight: the
   geomean is taken within each suite, then across the four. Otherwise
   GIFT-Eval's ~97 configs would swamp BOOM/TIME/FEV. The owner's intent is
   "on average across the benchmarks", not "across GIFT configs".
2. **No significant regression.** Per suite, a paired cluster bootstrap with
   tasks as clusters on `R_s` runs at `alpha / 4`. It uses the existing
   `paired_geomean_lcb` over the per-task geomean(CRPS, MASE) and the
   existing seed `{seed}|bench|{suite}`, so validators and the audit replay
   bit-identically. The one-sided lower bound of the ratio
   (`ratio_lcb = 1 − ucb`) must be `<= 1 + [scoring] bench_gate_tolerance`.
3. **Win rate.** Per suite, the challenger wins a task when `s < 1`, and an
   exact tie counts half. The suite-equal mean of the four shares must be
   `>= [scoring] bench_gate_min_win_rate`. This uses the same weighting as
   (1).

FEV stays required: missing FEV rows make the gate uncomputable. The
data-revision check, the shared-item join and the DEC-CA-0063
`checkpoint_fault` / infra / `budget_deferred` outcomes are unchanged. The
suite-level BenchScores are no longer read by the rule (they are still
logged).

**Keys / defaults** (`[scoring]`, CONSENSUS, overlay-able like the rest of
`[scoring]`, dropped from receipt params at their defaults):
`bench_gate_tolerance = 0.01` (in [0, 0.5)) and `bench_gate_min_win_rate =
0.5` (in [0, 1]).

**Receipts.** `verdict.bench_gate_detail` records `rule`, `computed`,
`passed`, `reason`, `tolerance`, `min_win_rate`, the three numbers
`avg_ratio` / `worst_lcb` (the max suite `ratio_lcb`) / `win_rate`, and per
suite `n`, `masked`, `ratio`, `ratio_lcb` and `win_rate`. The audit
(`check_bench_gate`) re-derives the three numbers from the recorded
per-suite ones (`detail_verdict`), and then replays the whole detail from
the signed rows.

**King check.** Under this rule the king check's bench half is the gate's
rule (1)+(2)+(3), with the KING as challenger and the INIT as reference, on
both entries' signed task rows (`king_check_gate`, seed
`{base_seed}|king-check`, thresholds from `cfg.koth_params(block)`). Missing
rows or too few tasks defer the check with the reason `bench rule
avg_noninf_winrate uncomputable: …`. A failure is recorded with the reason
`bench rule avg_noninf_winrate: <gate reason>`. The record's `bench.gate`
holds the gate detail. To make the rows available, the trainer now
publishes the init entry's rows (`rows_key`) in its bench report. The
validator bench source already published them. The audit replays the half
from those rows, and WARNs (`king-check-bench`) without them.

**Owner item before arming (calibration).** Run `scripts/bench_seed_noise.py`
on at least 3 seed replicas of the v2 recipe, trained from the same init.
It prints (a) how often seed noise alone flips an old strict comparison,
(b) the new rule's per-condition failure rate (the average condition fails
~half the ordered pairs by construction; regression and win rate are the
conditions to calibrate), and (c) the per-suite ratio-LCB distribution,
with the smallest tolerance no seed pair exceeds. Set
`bench_gate_tolerance` above that number, with margin. The 0.01 default is
unvalidated until this has been run.

The sections below describe the 2026-10-19 strict rule. They are kept as
history: wherever they conflict with this section, this section governs.

## Decision (2026-10-19, superseded in place by rule v2 above)

Owner (2026-10-19): the main (private) eval decides by the geomean LCB, as
today. On the external benchmarks a challenger must be better on everything.

1. **Private pool unchanged.** Paired geomean LCB ≥ the dethrone margin.
2. **Four suites.** GIFT-Eval, BOOM, TIME and FEV (DEC-CA-0054 had three).
   The trainer's signed bench report carries FEV item rows for lane A too.
3. **Strictly better, no statistics.** For each suite, the challenger's
   suite-level CRPS < the king's and its MASE < the king's: eight point
   comparisons, all required. Replaces DEC-CA-0054's six-number geomean.
4. **Not significantly worse.** For each suite, the paired bootstrap of
   per-item geomean(CRPS, MASE) at `alpha / 4` must not show the challenger
   worse (DEC-CA-0054 rule 1, now over four suites).
5. **Block only.** The gate can only block a private-pool win; it rides
   `[v2] bench_gate_mode = "enforce"`.

Lane B keeps statistical non-inferiority guards (raw "equal or better" on
~14 numbers would fail a neutral expert almost always).

## Supersession

This node supersedes DEC-CA-0054's RULE (three suites, six-number geomean)
for every settlement whose era starts at or after `[v2]
bench_gate_v2_from_block`: on mainnet that is `"v2"`, so from the v2 go-live.
It is a partial supersession (`supersedes_behavior_of`), not a full one:
DEC-CA-0054 still decides every earlier settlement and every testnet
settlement (testnet does not type the key), so DEC-CA-0054 keeps its status
and only carries a note.

## King check (resolved 2026-10-19)

The DEC-CA-0055 king check's bench half uses this rule's suite set whenever
it is in force for the settlement: the king's suite means must be strictly
below the init's on all EIGHT numbers (FEV joins), and missing FEV means
defer the check. The record's `bench.rule` says `"dec-ca-0061"` (dropped under
DEC-CA-0054, so those records keep their bytes) and the audit replays the
recorded rule against `bench_gate_rule_at`. The king check takes only the
strict half; its pool half stays the increment LCB of DEC-CA-0055 (the
per-item non-inferiority bootstrap is not added to it).

## Shipped

Built 2026-10-19 on `v2-moe-expert-comp`; inert until a `[v2]
bench_gate_v2_from_block` resolves to a block.

- **Versioning.** `[v2] bench_gate_v2_from_block` (CONSENSUS). An integer
  `N > 0` applies the rule to every settlement whose era starts at or after
  `N` (the same era-start arithmetic as `v2_settlement_active`, so the
  settlement AT `N` keeps the old rule). `0` (the default) means the
  DEC-CA-0054 rule everywhere. The string `"v2"` resolves at load to `[v2]
  from_block`. Mainnet ships `"v2"`, so the rule goes live with v2 and
  stays inert while `from_block = 0`. An already-judged settlement can only
  be re-judged if the owner types a past block, which is the same rule as
  `[v2] from_block`. Testnet does not type the key: it keeps DEC-CA-0054.
  A `"v2"` value resolves whenever the config is parsed — and a DEC-CA-0057
  overlay re-parses the merged raw table (`build_config` →
  `parse_chain_config`, `cascade/shared/netconfig.py`;
  `parse_bench_gate_v2_from_block`, `cascade/shared/config/v2.py`), so an
  overlay that moves `[v2] from_block` DOES move this rule with it (the same
  as `bench_source_from_block`). `bench_gate_v2_from_block` itself is not on
  the overlay allow list.
- **Gate.** `evaluate_bench_gate(..., rule=)`. Under "dec-ca-0054" the code
  path is unchanged. Under "dec-ca-0061" it uses four suites, `alpha / 4`
  and eight strict comparisons (a tie blocks). It runs only on a private-pool
  win, behind `bench_gate_mode_at` as before.
- **Like with like (2026-10-19 review).** The strict comparisons read the
  suite-level numbers only when both reports scored the SAME item set for
  that suite (nothing dropped from the inner join of their rows). When the
  sets differ — a cached king swept on another day, a partial suite — the two
  suite-level numbers aggregate different items, so both sides are instead
  the geometric mean of the per-item raw CRPS / MASE over the shared items,
  and the suite's detail records `basis = "shared_items"`. Validator reports
  (DEC-CA-0063) also carry each entry's pinned data revisions: two entries
  swept on different revisions make the gate uncomputable (the GIFT gate's
  rule), and the main validator re-sweeps a cached king whose revision
  differs from the challenger's.
- **Missing FEV** mirrors a missing suite under DEC-CA-0054. On the
  validator, missing FEV rows or FEV suite numbers make the gate
  uncomputable, so under `enforce` the round is inconclusive and the king
  holds. On the trainer, an operator bench in a DEC-CA-0061 era without FEV
  numbers counts as no bench (`extract_bench_scores(require_fev=True)`).
  The leg is then held and the king re-benched exactly as for a missing
  GIFT-Eval/BOOM/TIME. At the era's last settlement it settles anyway and is
  uncomputable. Payer-pod sweeps never need FEV: they only feed the
  forged-sweep check, and the operator re-bench publishes.
- **Trainer sweep.** `bench_suites_for` adds `fev` to `--bench-suites` when
  the rule is configured at some block. Testnet sweeps are unchanged. FEV is
  never a data-guard or sideload marker: fev pulls its tasks and datasets from
  the HF hub at sweep time. Every `uv run` then passes `--extra fev`.
- **Receipts.** `KothParams.bench_gate_rule` records the effective rule. It
  is dropped from params at "dec-ca-0054", so DEC-CA-0054 receipts keep
  their bytes. `VerdictRecord.bench_gate_detail` holds the rule, per suite
  the four numbers, `crps_better` / `mase_better`, `n` and `ucb`, and the
  reason. It is `None` and dropped under DEC-CA-0054. `load_receipt` now
  also parses `bench_gate_passed`, which it used to drop.
- **Audit.** The `bench-gate` Tier-0 check reads the trainer-signed report
  and its sha256-pinned rows, then replays both rules. The recorded rule
  must equal the one in force at the block. The outcome must match, and
  under DEC-CA-0061 so must every recorded comparison. Without the report
  the check WARNs, after testing that the recorded detail implies the
  recorded outcome.
- **Not changed.** Lane B is untouched. (The king check's bench half now
  follows this rule when it is in force — see "King check" above. Under a
  trainer bench source the init bench must then carry FEV means, or the
  check defers; whether it does was not verified. On mainnet the source is
  the main validator, which sweeps FEV whenever this rule is in force.)

### FEV cost (code facts, not measured)

- No FEV sweep timing exists in this repo, and the full FEV suite has not
  been qualified by a complete production-data sweep. Current setup is in
  `docs/OPERATOR.md`, "Before go-live / before a testnet cycle".
- The suite scores the univariate tasks of the upstream fev-bench list
  (multivariate-target tasks are skipped). It runs every rolling window per
  task and makes one more forecast pass, the in-suite Seasonal-Naive
  baseline (`benchmarks/cascade_benchmark/suites/fev_bench.py`).
- Its datasets (`autogluon/fev_datasets`) download from HF on each pod's
  first sweep. The 4.4 GB sideload does not cover them.
- Every operator bench in a DEC-CA-0061 era adds one FEV pass: the king,
  each operator-lane leg and each payer-verify re-bench. These run inside
  the existing bench window (`[eval] bench_hold_max_hours` when
  `cascade_bench_max_series = 0`, else 1800 s).
- Measure one full FEV sweep on the king pod SKU before go-live. If it
  pushes the verified bench past the settlement it gates, verification
  capacity has to grow (DEC-CA-0054 "Not done").

## Interfaces

| Surface | Name |
|---|---|
| Config | `[v2] bench_gate_v2_from_block` (int ≥ 0 or `"v2"`), `V2Config.bench_gate_v2_from_block`, `parse_bench_gate_v2_from_block` (`cascade/shared/config/v2.py`) |
| Rule resolution | `bench_gate_rule_at(cfg, boundary)`, `bench_gate_rule_for_era`, `bench_gate_v2_configured` (`cascade/shared/era.py`) |
| Gate | `evaluate_bench_gate(..., rule=, tolerance=, min_win_rate=)`, `RULE_V1 = "dec-ca-0054"`, `RULE_V2 = "avg_noninf_winrate"` (rule v2), `SUITES_V2`, `BenchGateResult.rule / suites / avg_ratio / worst_lcb / win_rate / detail()`, `combine_suites`, `detail_verdict`, `strict_eight` (report-only) (`cascade/eval/bench_gate.py`) |
| Thresholds | `[scoring] bench_gate_tolerance` (0.01), `bench_gate_min_win_rate` (0.5); `KothParams.bench_gate_tolerance / bench_gate_min_win_rate` |
| King check | `king_check_gate`, `KC_GATE_UNCOMPUTABLE` (`cascade/validator/king_check.py`); audit `_kc_gate_replay` |
| Calibration | `scripts/bench_seed_noise.py` (report-only) |
| Verdict | `KothParams.bench_gate_rule`, `RoundResult.bench_gate_detail`, `apply_bench_gate` (`cascade/eval/koth.py`); `EvaluationMixin._run_bench_gate(..., rule=)` |
| Receipt | `verdict.params.bench_gate_rule` (drop-when-default), `verdict.bench_gate_detail` (drop-when-None) |
| Bench report | `BenchScores.fev_crps / fev_mase` (optional, dropped when `None`: `bench_scores_body`); item rows `fev` (`item_rows_from_report`, unchanged) |
| Trainer | `extract_bench_scores(report, require_fev=)`, `bench_suites_for`, `_bench_fev_required(era)`, `_remote_bench_scores(..., require_fev=)` |
| Audit | `check_bench_gate(receipt, cfg, bench)`, `fetch_bench_inputs(cfg, receipt, store)` |
