---
id: DEC-CA-0012
type: decision
title: "Duel margin decays on a stale throne (3 full-margin rounds, then halve toward the floor)"
status: active
date: 2026-08-05
tags: [koth, scoring, margin, lcb, consensus]
revisit_when: "kings churn every handful of rounds (the decay is doing the dethroning, not model quality — raise margin_decay_after_rounds or the floor), or a decayed-margin dethrone is followed by an immediate re-dethrone ping-pong (the floor is admitting statistical ties in practice — raise margin_floor), or cascade_enabled arms (a reign now triggers warm-start promotion, so 'stale throne' takes on a second meaning the decay schedule should be checked against)"
relations: {refines: DEC-CA-0004, depends_on: DEC-CA-0009}
---
The duel's win margin is no longer constant over a reign: once the king
survives `margin_decay_after_rounds` (3) rounds undethroned, the margin's
excess above `margin_floor` (0.0) is multiplied by `margin_decay_rate` (0.5)
each further round — 0.02 → 0.01 → 0.005 → … A dethrone resets tenure and
with it the schedule for the new king. `margin_decay_after_rounds = 0`
disables (the old flat-margin behaviour, and the default for older configs).

**Why.** With the throne persistent (DEC-CA-0004) and the margin flat at
0.02, a king that is merely *not 2% worse* than every challenger holds
indefinitely; incumbency compounds because nothing about the bar ever moves.
The decay converts "nobody has cleared the bar lately" into a progressively
lower bar: a challenger that is genuinely better — just not 2% better —
takes the throne within a few rounds instead of never.

**Why the floor stays at 0.** At full decay a challenger still needs
`LCB ≥ 0`: a statistically significant improvement at the bootstrap's alpha,
with the king holding ties. The asymmetric null ("king holds unless beaten")
is the duel's anti-churn property and is preserved; only the *extra* margin
decays. A negative floor would let statistically tied challengers through —
load-time validation rejects `margin_floor < 0` (and a non-decaying
`margin_decay_rate` ∉ (0,1), and a floor above the margin schedule).

**Boundary.** Rounds judged at tenure 0..2 carry the full margin; the round
judged at tenure 3 is the first decayed one (the king already survived 3
rounds to reach it).

**Replayability.** The decayed margin is a pure function of
`king_tenure_rounds`, which `VerdictRecord` already records alongside the
full `KothParams` — so `cascade-audit` replays decayed verdicts with zero
receipt-format change (the signed canonical body gains only the three new
keys inside the existing `params` dict; the golden fixture was regenerated,
archived receipts still verify against what they signed).

**Consensus.** The margin decides rounds, so all validators must run the
same `margin_decay_*` values — same deploy-together discipline as any
`[scoring]` change (cf. DEC-CA-0009). `check_koth_params` flags receipts
recorded under other values, as it does for any scoring drift.
