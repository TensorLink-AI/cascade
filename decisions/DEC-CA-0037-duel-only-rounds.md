---
id: DEC-CA-0037
type: decision
title: "Duel-only rounds on the operator's final fleet: no heat, reveal-order seating under a per-round cap, one block-gated flip with the scored horizon ladder"
status: accepted
date: 2026-09-03
tags: [round, heat, duel, provisioner, eval, cadence]
revisit_when: "the waiting queue grows for several consecutive rounds (raise duel_field_cap, add a final pod, or shorten the grid); or a round's manifest lands inside the last 2h of its epoch twice (the cap is too high for the fleet's lanes); or the 3h grid arms (cap must then equal lanes − 1 so a round is exactly one wave); or a seated cohort's α/k bar is shown to block a genuine improvement the single-finalist rule would have crowned"
relations: {builds_on: [DEC-CA-0012, DEC-CA-0008, DEC-CA-0003], retires_operationally: [DEC-CA-0006, DEC-CA-0011], related: [DEC-CA-0036]}
---

From `[round] duel_from_block` (8992800 ≈ Fri 2026-09-04 08:30 UTC, an epoch
boundary) rounds run without a heat: the screened field seats straight into
the duel, earliest reveal block first, up to `duel_field_cap` (7)
challengers; entrants beyond the cap wait for a later round with their
submission intact. The king and every seated challenger train the full
`[training]` budget on the operator's L40S final fleet; legs beyond the
fleet's lane count queue on the same pods (2 pods × 2 lanes: 8 legs = 2
waves ≈ 6h inside the 12h epoch). The validator judges the whole seated
cohort under the existing α/k rule with k = the seated count, derived from
the signed manifest as before — no validator change, no contract change
(`expected_gpu` stays pinned to L40S). The provisioner rents no heat fleet
for these rounds and rents the final at the margin, sized 1 + seats.

The same block arms the scored horizon ladder (`[eval] scored_horizons =
[64, 256, 720]`, `scored_from_block`): the verdict draws one even-by-domain
rung per horizon over the round's snapshot and pools the rows through the
unchanged paired bootstrap. That half is consensus (release-then-activate,
every validator before the block); the duel-only half is trainer policy.

## Why

- One full-budget duel per entrant replaces a 1h screen plus a duel for
  three: the artifact judged is the artifact trained, and the screen's
  cross-SKU fleet (with its own provisioning failure class) goes away.
- Seating by reveal order is the seniority rule the heat's tie-break and the
  dedup screen already use; a UID recycles, a reveal block does not.
- The cap bounds cost and wall clock per round explicitly instead of through
  fleet sizing; the queue absorbs field size without changing k per round
  beyond the cap.

## Consequences owned

- The per-challenger dethrone bar tightens with the seated count (α/k):
  seven seats judge each at α/7. Deliberate — the cap is the lever.
- Standings carry no ranking on duel-only rounds; the public standings doc
  publishes the seated count, the cap, and the waiting entrants (`waiting`)
  so a queued miner can read that its submission is intact.
- The post-publish bench sweep grows with the cohort (each pod benches the
  checkpoints it trained, sequentially per pod); the provisioner's bench hold
  bounds it.
- Rollback of the duel-only half is `duel_from_block = 0` + a trainer
  restart; the ladder half cannot be rolled back unilaterally once past its
  block.
