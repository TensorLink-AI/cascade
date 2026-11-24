---
id: DEC-CA-0073
type: decision
title: "MarketCast (lane C) pays in full for a gain-scaled duration (5 to 20 days), then the slot ends"
status: accepted
date: 2026-11-24
tags: [lane-c, marketcast, payout, consensus, audit]
revisit_when: "the first confirmed lane-C win — check its recorded full_eras against the formula and that its slot burns at that age"
relations: {amends: [DEC-CA-0056, DEC-CA-0064], depends_on: [DEC-CA-0057], relates_to: [DEC-CA-0050]}
---

## Problem

A lane-C (MarketCast) win paid 100 % of its slot for a gain-tiered period
(`[marketcast] full_pay_tiers = [[0.05, 10], [0.02, 5], [0.01, 2]]` eras),
then halved every `half_life_rounds` (1) era to a 10 % `floor`, and stopped at
`win_ttl_eras` (28). At the 1 % margin that is two eras of full pay, about one
day on the 12 h mainnet eras, and the decay tail keeps paying a stale holder.

## Decision (owner 2026-11-24)

A winning target slot pays FULL for a duration that scales with the
crowning gain LCB: **5 days minimum (at the 1 % gain margin) up to 20 days at
a 5 % or higher LCB gain**. Then the slot ENDS (it burns, `remainder =
"burn"`) unless a newer win replaces it. There is no decay tail and no TTL.

How the code reads it:

```
frac      = clamp((gain_lcb - gain_margin) / (full_pay_max_gain - gain_margin), 0, 1)
days      = full_pay_min_days + (full_pay_max_days - full_pay_min_days) * frac
full_eras = ceil(days * 86400 / (BLOCK_SECONDS * era_blocks))
```

* `gain_margin` is the judging competition's margin: `[marketcast]
  gain_margin` (0.01) for an `mkt:` target, `[lane_c] gain_margin` otherwise.
* `era_blocks` is the era length in force at the confirming settlement
  (`[round] epoch_blocks` x `era_settlements`; mainnet 900 x 4 = 3600 blocks
  = 12 h). Mainnet therefore pays 10 to 40 eras. Rounding is UP, so a win is
  never paid for less than the owner's duration.
* The arithmetic is exact (`fractions.Fraction`), so every validator gets the
  same `full_eras`.
* The duration is fixed at the win and recorded on it (`tier` = the clamped
  gain the duration came from, `full_eras`, `below_floor` = an LCB under the
  margin). It is never re-evaluated.
* The slot rule (`win_beats`) is unchanged. A holder whose duration has run
  out is evicted before the next settlement's offers, through the existing
  expiry path. Any confirmed win can then take the empty slot.

Keys in `[lane_c]` (CONSENSUS, overlay-settable like the rest of
`[lane_c]`): `full_pay_min_days = 5`, `full_pay_max_days = 20`,
`full_pay_max_gain = 0.05`. Loader checks: finite values, `0 <
full_pay_min_days <= full_pay_max_days`, `0 < full_pay_max_gain < 1`,
`[lane_c] gain_margin < full_pay_max_gain` and `[marketcast] gain_margin <
full_pay_max_gain`.

Retired (`RETIRED_KEYS`: a config that still has them loads and they are
ignored; an overlay that names them is refused): `[lane_c] full_pay_tiers`,
`half_life_rounds`, `floor`, `win_ttl_eras` and `[marketcast] full_pay_tiers`.

## No activation block

Lane C / MarketCast has not rolled out, so no lane-C win has ever been
recorded or paid under the old rule (owner 2026-11-24). The new rule is
simply the rule; there is no `pay_v2_from_block`. `[lane_c] from_block` is
unchanged. Every validator must still run this release before the first
lane-C confirmation, because netuid 91 has external validators and the
payout is CONSENSUS.

## Audit

`cascade-audit` re-derives every confirmed row's `full_pay` and every paid
lane-C win from its gain LCB on the era length at that settlement. It FAILs
a forged duration, a multiplier other than 1.0, and a win paid at or past its
own `full_eras`.

## Not changed

Lane A and lane B. Lane B keeps its own tiers, decay and TTL, and moves to
indefinite pay separately (plan rev6.22).
