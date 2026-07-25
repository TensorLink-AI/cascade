---
id: DEC-CA-0006
type: decision
title: "Heat ranks on the point estimate; the bootstrap LCB is a tracked diagnostic, never a gate"
status: active
date: 2026-07-25
tags: [heat, scoring, koth, statistics, screening]
revisit_when: "the tracked cut LCB comes in below the KOTH win margin on a sustained majority of contested rounds — that is the trigger to raise [round] finalists / heat_n_windows / heat_num_samples, not to change the ranking statistic; or finalists rises above 1, which changes what a marginal cut actually costs"
relations: {}
---
The heat screen ranks challengers on the OBSERVED `global_geomean`
(`sqrt(MWSQL × mean MASE)`, lower better) and advances the top
`[round] finalists`. It does NOT rank on the paired-bootstrap LCB that gates
dethroning. The LCB is computed once per round at the finalist boundary and
recorded as a diagnostic (`HeatResult.cut`).

Why not gate the heat on the LCB, given KOTH does:

- **Different shape of problem.** The LCB exists in KOTH because dethroning has
  an asymmetric default — weak evidence means the king holds
  (`inconclusive`, streak untouched). The heat has no such default: it must
  advance `finalists` entrants whatever the evidence says. Swapping the point
  estimate for a lower bound adds no "decline to decide" branch, it only
  reorders the field.
- **No reference model at heat budget.** The KOTH LCB's power is *pairing*
  against the king on identical windows. `plan_round` separates the king out and
  `_run_heat` trains challengers only, so there is no king heat checkpoint to
  pair against. The alternatives are a tournament against the running best
  (O(N) bootstraps, order-dependent) or an unpaired CI per entrant, which is
  dominated by window-difficulty variance common to every entrant and therefore
  near-constant across the field.
- **It would reward the wrong property.** Where LCB-ranking differs from
  point-ranking it favours the lower-variance entrant at equal expected score.
  Selecting generators on the tightness of their per-window distribution is a
  gradient toward blandly-average forecasts, not better ones.
- **Sampling noise is not the dominant term.** Under [[DEC-CA-0001]] heats
  deliberately time-truncate (`heat_guard_factor = 1.0`), so a heat score partly
  measures pipeline throughput. A bootstrap over eval windows says nothing about
  that.

What IS worth knowing is whether the cut is real, because `finalists = 1` plus
`one_submission_per_hotkey = true` means a noise-ordered boundary permanently
burns a miner's single lifetime submission and the runner-up never reaches the
final for the real statistic to settle it. So `cascade.eval.heat_cut` runs the
same paired cluster bootstrap on the last entrant that advanced vs the first
that did not — properly paired, since every entrant screens on the same
`windows_for_round(base_seed, n, block)` slice — and records
`lcb / p50 / p95 / observed` against the KOTH `win_margin_start` as the reference
bar. Logged at WARNING when `lcb < margin`, carried on the unsigned `heat` block
of the manifest and summarised on the round receipt.

The response to a chronically unseparated cut is MORE EVIDENCE OR MORE SLOTS
(`heat_n_windows` 256, `heat_num_samples` 20, `finalists` 1) — not a different
ranking statistic. Consensus-safe by construction: the heat is trainer-internal,
the cut never feeds `challenger_wins_round` or weights, and it rides outside
`canonical_body` so it cannot move a signature. See [[DEC-CA-0001]].
