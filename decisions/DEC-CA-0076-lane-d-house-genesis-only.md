---
id: DEC-CA-0076
type: decision
title: "Lane D: the house is the genesis bar only; no house retake, no second bar, no burn while no miner king exists"
status: accepted
date: 2026-11-24
tags: [lane-d, validator, audit, satellite, consensus]
revisit_when: "the first miner crown on mainnet: check that the main validator, every satellite and cascade-audit agree that lane D stops paying lane A and starts paying the court at that settlement"
relations: {amends: [DEC-CA-0058], relates_to: [DEC-CA-0057]}
---

## Problem

Owner 2026-11-24: "I just want [the house] to be for genesis of the comp
until a king is crowned … remove the house-holds-king mechanism after that."

Before this decision, the house (our lane A king) was always entered in lane
D. A house KING_D's slot burned (D21, `house_king`). The house was also a
second bar for every miner challenger (W5c change 3), and a refreshed house
could challenge and retake KING_D from a miner (W5f).

## Decision

1. **Genesis.** Genesis means no miner has ever held KING_D. During genesis
   the house holds KING_D, and the first crown must beat it (the lane A
   king), as before.
   - Lane D's share is NOT burned. It stays with its `takes_from` lane: the
     vector is exactly the lane-D-off vector.
   - Genesis is read from the signed lane D report: `king_after.house` is
     true. The house can never hold KING_D again after a miner crown (rule
     2), so a house KING_D means genesis.
2. **After the first miner crown,** the house is gone from lane D:
   - it is no second bar;
   - a new house pointer opens no challenger (`refresh_house`);
   - it never retakes KING_D.
   A report whose KING_D is the house after a miner crown (prior kings
   present, or a miner king before it) is refused (`ld_crown`, burn class:
   lane D's share burns, lanes A/B/C unchanged). Challengers face only KING_D
   under the crown-bar rule, which decays from 3% to 1%.
3. **Court.** The court is KING_D plus up to two prior miner kings at
   `court_weights` 0.80 / 0.15 / 0.05. With fewer prior kings than slots,
   the weights renormalise over the kings that exist. A deregistered king's
   slot keeps today's rule: its uid does not resolve, so the slot burns to
   `burn_uid` (`burn_reason = "deregistered"`).
4. **Replay.** The main validator, the satellites (DEC-CA-0058) and
   `cascade-audit` derive the same vector from the same signed record:
   - the main validator through `LaneDStep.genesis_for`, which the weight fold
     reads;
   - satellites and the audit through `lane_d_pay(...).genesis` in
     `derive_weights`.

Lanes B, C and D are not rolled out, so there is no activation block. Lane A,
and lanes B and C with lane D off, are byte-identical.

## Not changed (open)

The W5e bench gate still sweeps the house as a reference: CRPS and MASE must
stay within `tau_public` of both KING_D and the house. That is a
public-benchmark floor, not the crown duel. Removing it changes the gate
record and the bench sweep, which needs a separate owner call.
