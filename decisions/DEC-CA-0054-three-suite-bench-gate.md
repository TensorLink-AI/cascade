---
id: DEC-CA-0054
type: decision
title: "A dethrone must also not regress on public benchmarks: no suite of GIFT-Eval / BOOM / TIME significantly worse on a paired test, and the six-number geomean strictly better — read from trainer-signed per-item rows"
status: proposed
date: 2026-10-18
tags: [scoring, consensus, benchmarks, validation]
revisit_when: "shadow telemetry shows the gate blocking private-pool wins that later prove real (too strict), or never blocking anything (redundant); or a payer-pod sweep is found forged after verification (move every gated bench onto operator pods); or BOOM/TIME item counts drop below bench_gate_min_items; or promotion (DEC-CA-0044) adopts the same gate"
relations: {depends_on: [DEC-CA-0013, DEC-CA-0036, DEC-CA-0043, DEC-CA-0045], relates_to: [DEC-CA-0012, DEC-CA-0038, DEC-CA-0044]}
note: "DEC-CA-0061 supersedes this node's rule for settlements whose era starts at or after [v2] bench_gate_v2_from_block (mainnet and testnet: the v2 go-live, \"v2\"); this rule still decides every earlier settlement. DEC-CA-0061 rule v2 (2026-10-20) is avg_noninf_winrate"
---

## Decision

Owner (2026-10-18): a challenger "must beat on eval pool (ts benchforge) but
also not regress"; "none can be significantly worse on a paired test for the
evals and mean must be higher (no threshold) new > old, same rules as current
cascade for main eval"; benched "on the miner-funded pod they trained on".

On a private-pool win, a third AND-ed gate (after the margin and the
existing GIFT gate) — `[scoring] bench_gate_mode`, `cascade.eval.bench_gate`:

1. **No suite significantly worse.** For each of GIFT-Eval, BOOM and TIME,
   items shared by king and challenger are scored `geomean(CRPS, MASE)` and
   paired-bootstrapped exactly like the lane-B public guards
   (`expert_lane.paired_geomean_lcb`, seed = the round's block hash). A suite
   fails when the upper bound of the relative improvement at `alpha/3` is
   below 0. Bonferroni over three suites keeps the false-block rate at
   `alpha` for equal checkpoints (max-T, DEC-CA-0038, needs shared resamples
   and the suites share no items).
2. **The mean is better.** The six-number Cascade score of the challenger is
   strictly below the king's. No tolerance.

The gate can only block; it never grants a dethrone. Missing rows ⇒
uncomputable ⇒ inconclusive under `enforce` (king holds), exactly like the
GIFT gate.

**Activation rides the v2 go-live** (owner: everything "needs to be live when
the new incentive goes live"): `[v2] bench_gate_mode = "enforce"` applies to
every v2 settlement (`cascade.shared.era.bench_gate_mode_at`); before it,
`[scoring] bench_gate_mode` (off) governs. No separate readiness vote — the
same typed block as DEC-CA-0050.

## Where the numbers come from

- Each leg is benched at completion on the payer's pod (DEC-CA-0043), then
  re-benched on the era king's operator pod; only operator numbers publish
  (a payer number better than ours by > `funded_bench_verify_tolerance` is a
  forged sweep — dropped). The operator sweep's per-item rows are published
  at `benchmarks/rows/<sha256(pointer)>.json`, and the signed bench report
  entry pins them (`rows_key`, `rows_sha256` — omitted when empty, so every
  archived report signature still verifies).
- While the gate is on, the king's entry rides every settlement's bench
  report, and a leg settles only once its verified bench has landed — except
  at the era's last settlement, where it settles anyway (then uncomputable).
- Validators read identical signed numbers; no validator re-runs a sweep.
- Consensus hygiene (review, 2026-10-19):
  - With the gate on, the bench report publishes BEFORE the manifest. A
    validator that sees a settlement always sees its report, and a failed
    report publish withholds the settlement.
  - A republish (a bench that landed after its settlement) never adds gate
    rows. Only the first version's pointers keep `rows_key`, so validators
    that judged either version agree.
  - A store READ failure on the report or the rows raises, and the
    settlement retries. Only absence or a sha256 mismatch is uncomputable.
  - Legs also wait for the era king's bench. A king entry without one is
    re-benched every 10 min.
  - Receipts stamp the settlement's effective `bench_gate_mode` (v2 can
    switch it on).

## Not done

- Promotion (DEC-CA-0044: a candidate enters by beating the worst live member
  on the six-number score) does not yet apply the paired suite test.
- `cascade-audit` does not replay the bench gate: it applies the recorded
  `bench_gate_passed` outcome (as for the gift gate) under the receipt's
  stamped mode.
- Verification throughput: the operator re-benches one leg at a time on the
  king pod. Under enforce, a leg still unverified at the era's last
  settlement is uncomputable, so it can never be crowned, and it is still
  burned. Size verification capacity to intake, or requeue such legs
  unburned, before go-live.
- Typing `[v2] from_block` turns on row stashing (`rows_key` in reports)
  before the go-live block. Every validator must run the release first; this
  is the existing DEC-CA-0050 rule.
