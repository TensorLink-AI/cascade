---
id: DEC-CA-0037
type: decision
title: "Duel-only rounds on the operator's final fleet: no heat, the whole field seats in reveal order, the fleet sizes to the epoch, one block-gated flip with the scored horizon ladder"
status: accepted
date: 2026-09-03
tags: [round, heat, duel, provisioner, eval, cadence]
revisit_when: "the waiting queue grows for several consecutive rounds (raise the final pod ceiling or shorten the grid); or a round's manifest lands inside the last 2h of its epoch twice (the overhead constant or the waves-per-lane rule is wrong); or the 3h grid arms (one wave per round — the pod ceiling becomes the seat count); or a seated cohort's α/k bar is shown to block a genuine improvement the single-finalist rule would have crowned"
relations: {builds_on: [DEC-CA-0012, DEC-CA-0008, DEC-CA-0003], retires_operationally: [DEC-CA-0006, DEC-CA-0011], related: [DEC-CA-0036]}
---

From `[round] duel_from_block` (8992800 ≈ Fri 2026-09-04 08:30 UTC, an epoch
boundary) rounds run without a heat: the whole screened field seats straight
into the duel, earliest reveal block first, and the king plus every seated
challenger train the full `[training]` budget on the operator's L40S final
fleet. The provisioner rents no heat fleet and sizes the final at the margin
to fit `1 + field` legs inside the epoch — each lane runs
`floor((epoch_h − 1.5h) / 3h)` = 3 legs back to back, because pods carry a
one-epoch TTL — up to its pod ceiling (`[provisioner.final] max_pods`). The
trainer waits for that fleet, seats what its lanes can finish before the
boundary (2 pods × 2 lanes = 11 seats; +6 per extra pod), and only that
physical overflow waits for the next round with its submission intact.
`duel_field_cap` (0 = none) is an explicit per-round cap on top. The
validator judges the whole seated cohort under the existing α/k rule with
k = the seated count, derived from the signed manifest as before — no
validator change, no contract change (`expected_gpu` stays pinned to L40S).

The same block arms the scored horizon ladder (`[eval] scored_horizons =
[64, 256, 720]`, `scored_from_block`): the verdict draws one even-by-domain
rung per horizon over the round's snapshot and pools the rows through the
unchanged paired bootstrap. That half is consensus (release-then-activate,
every validator before the block); the duel-only half is trainer policy.
The same block also moves the fresh-king margin from 2% to 1%
(`[scoring] win_margin_start = 0.01`, `win_margin_start_prev = 0.02`,
`margin_activation_block = 8992800`): validators resolve the value from the
round's epoch boundary and the audit replays each receipt under its own
value, so the change is block-exact rather than restart-timed. And the
config_only dedup tier starts DROPPING at the same block
(`dedup_config_only_enforce = true`, `dedup_config_only_from_block =
8992800`; shadow-logged before): identical code with a different config is
the seat-spray pattern (19 of r58's 24 entrants came from 5 code groups; the
gate would have kept 10), and with no heat the seats are what spam takes.

## Why

- One full-budget duel per entrant replaces a 1h screen plus a duel for
  three: the artifact judged is the artifact trained, and the screen's
  cross-SKU fleet (with its own provisioning failure class) goes away.
- Seating by reveal order is the seniority rule the heat's tie-break and the
  dedup screen already use; a UID recycles, a reveal block does not.
- Cost scales with the field (the owner's pod ceiling is the brake); the
  wall clock is bounded by the epoch through the waves-per-lane rule, so a
  large field carries over instead of overrunning into the next boundary
  (where the provisioner's next rental and the pod TTL would collide with
  it).

## Consequences owned

- The per-challenger dethrone bar tightens with the seated count (α/k): an
  11-seat round judges each challenger at α/11. Deliberate; the explicit
  cap exists if that bar proves too stiff.
- Standings carry no ranking on duel-only rounds; the public standings doc
  publishes the seated count, the cap, and the waiting entrants (`waiting`)
  so a queued miner can read that its submission is intact.
- The post-publish bench sweep grows with the cohort (each pod benches the
  checkpoints it trained, sequentially per pod); the provisioner's bench hold
  bounds it.
- Rollback of the duel-only half is `duel_from_block = 0` + a trainer
  restart; the ladder half cannot be rolled back unilaterally once past its
  block.
