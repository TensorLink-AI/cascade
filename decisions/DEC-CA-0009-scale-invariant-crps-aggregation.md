---
id: DEC-CA-0009
type: decision
title: "CRPS half of the round metric is a per-window geomean, not a pooled MWSQL"
status: active
date: 2026-07-28
tags: [koth, scoring, eval, bootstrap, heat, audit]
revisit_when: "a single cluster's share of the WQL half exceeds ~20% of the windows it is defined on (breadth collapse, the failure this replaced one form of); or the pool becomes count-imbalanced enough that the largest feed exceeds ~15% of the log-sum, at which point the two-layer GIFT-Eval form (equal weight per feed) stops being verdict-equivalent and should be re-measured; or the zero-|y| mask exceeds ~10% of windows, since two-layer discards none"
relations: {supersedes_behavior_of: DEC-CA-0006}
---
The CRPS-family half of `geomean(CRPS, MASE)` — the quantity the duel's LCB
bounds and the heat ranks on — is the **geometric mean of per-window WQL**, each
window normalised by its own `sum|y|`. It was a **pooled MWSQL**: numerators and
denominators summed across every window, divided once.

**Why the pooled form had to go.** Pooling a ratio implicitly weights every
window by its own magnitude. cascade's eval pool spans ~15 orders of magnitude
in `sum|y|`, so that is not a mild bias. Measured on round
10447302782510174565 (2000 windows):

    blockchain_btc_difficulty     62.7% of the denominator
    treasury_debt_to_penny_daily  18.6%
    treasury_daily_debt_to_penny  18.6%
    -> top 3 windows = 100.0%; the other 1997 contributed ~0%.

Half the decision statistic had an effective sample size of 3, and the single
cluster the challenger was worst on (treasury, 2.5x worse) was the most
over-represented cluster in the worst 5% of bootstrap bags. That is what
produced 8 consecutive king defences against challengers the shadow diagnostics
scored as better every round (win_rate 0.57-0.79, wilcoxon_p to 3e-44): the
bootstrap distribution was not wide, it was skewed by three windows. Point
estimates on that round favoured the challenger on *both* halves (geomean MASE
0.4400 vs 0.5338; MWSQL 0.0182 vs 0.0205; bootstrap p50 +0.1275, 76.7% of bags
favouring the challenger) while the 5th percentile sat at -0.4243.

**Why the geometric mean.** It mirrors the MASE half and for the same stated
reason: per-window ratios are heavy-tailed and an arithmetic mean lets one
window dominate. It is also scale-invariant, which is the property a
cross-domain pool actually needs.

**What the pooled form was right about.** It is immune to a zero denominator,
and the original docstring cited exactly that ("dividing per window blows up on
near-zero-mean windows"). This is real: ~3% of a live pool has `sum|y| == 0`
(idle bike-share docks, intermittent counts). Those windows are **masked out of
the WQL half**, not floored — flooring at eps injects values of order 1e11 into
a log-space mean and swamps it. They still count for MASE. So the pathology the
pooled form was defending against is handled by exclusion, not by reweighting
the entire pool toward its largest series.

**Blast radius, and what was deliberately NOT done.**
- Applies to the duel (validator) *and* the heat screen (trainer), which ranks
  on the same `global_geomean`. Both must be deployed together or the heat
  selects finalists on a different metric than the duel judges — the invariant
  DEC-CA-0006 depends on.
- `win_margin` and `bootstrap_alpha` were NOT touched. Lowering the margin
  0.02 -> 0 flips zero of the five archived duels (every LCB was negative) and
  raising alpha 0.05 -> 0.25 flips one; both paper over a broken statistic by
  lowering the evidence bar for everyone. The margin was never the constraint.
- No new `[scoring]` config key: the aggregation is the rule, not a tunable.
  `KothParams` is unchanged, so `cascade-audit`'s `check_koth_params` is
  unaffected.
- No new `VerdictRecord` field, and `RECEIPT_VERSION` was NOT bumped.
  `asdict(verdict)` goes verbatim into the signed `canonical_body`, so any new
  field re-serialises every archived receipt with bytes that were never signed
  and breaks the public audit trail; and `load_receipt` rejects any version but
  the current one, so a bump would stop archived receipts loading at all.
  Instead `check_verdict` replays under each known aggregation and reports which
  reproduces the recorded LCB — self-verifying, since reproducing an LCB to 1e-9
  under the wrong rule is not something a tampered receipt does by accident.
  Verified: all five archived duels still reproduce bit-for-bit under `pooled`.

**Effect on the archived duels** — all seven scored receipts in the bucket,
same seeds, alpha, margin, clusters. Note this is indicative, NOT a
counterfactual history: challengers optimised against the live rule, so it does
not establish that the throne was wrongly held.

    round                  pooled (live)   ->  per-window geomean   two-layer
    14567400215825813659      +0.1085 WIN  ->  +0.2058 WIN          +0.2311 WIN
    16369783168410731489      +0.0568 WIN  ->  +0.1492 WIN          +0.1432 WIN
    10447302782510174565      -0.4243 loss ->  +0.0659 WIN          +0.0929 WIN
    14597972378946389402      -0.1930 loss ->  +0.0812 WIN          +0.1003 WIN
    7738837034499501847       -0.0315 loss ->  +0.1545 WIN          +0.1539 WIN
    15787128089753493320      -1.8367 loss ->  -0.0822 loss         -0.1645 loss
    9797713724704223682       -0.1254 loss ->  +0.0168 loss         -0.0042 loss
                              2 WIN            5 WIN                5 WIN

It discriminates rather than flipping everything — the genuinely worse
challenger (15787…, a 2.6x blowout) still loses, and 9797… stays a near miss,
which is the behaviour a corrected statistic should show.

**Relation to GIFT-Eval, and why "two-layer" is a documented alternative, not
the choice.** GIFT-Eval computes CRPS in two layers: the pooled
`sum QL_q / sum |y|` ratio *within* one dataset/freq/term config, then a
**geometric mean** across its 97 configs (of Seasonal-Naive-normalised values).
The pooled form this decision removes was GIFT-Eval's *inner* layer applied to
the whole pool — one layer where GIFT-Eval runs two, with the cross-scale
aggregation missing. The per-window geomean is the analogue of the *outer*
layer, which is the one that does the cross-scale work.

A faithful two-layer rule (pool within feed, geomean across feeds; our cluster
= upstream feed is the config analogue) was measured and is the third column
above. It **agrees with the per-window geomean on all seven verdicts**, runs
slightly tighter, and discards nothing (0 feeds have `sum|y| == 0`, versus 60
of 2000 windows masked). It was NOT adopted, because:

- Its leaf is still a pooled ratio, and the concentration is still there at
  feed scale: in `coinbase_spot_15m` a single window is **96.2%** of that
  feed's `sum|y|` (median feed: the top window is ~25%). Two-layer bounds the
  damage to 1/185 of the log-sum rather than removing it.
- Identical verdicts on every archived round means switching buys no measured
  behaviour, at the cost of a second re-verification cycle and a third
  aggregation rule for `check_verdict` to replay.

The residual divergence from GIFT-Eval is **weighting**: our flat per-window
geomean weights a feed by its window *count* (the five largest feeds are ~7%
each of the log-sum), where GIFT-Eval Layer 2 gives every config an equal
1/97. Two-layer would fix that. See `revisit_when`.

**Seasonal-Naive normalisation is a no-op here and was not added.** GIFT-Eval
divides each config by Seasonal Naive's value for that config. Under a
geometric mean, dividing every window by a fixed positive constant is a
translation in log space, so it cancels exactly in the paired ratio
`(G_king - G_chal)/G_king` — verified numerically to 1e-12 by injecting an
arbitrary per-window normaliser (it cancels only because the duel is paired and
king/challenger share an identical validity mask; both hold). It does *not*
cancel under the pooled rule, where the same normaliser moved the ratio from
1.1243 to 1.1855. That normaliser exists to put a *leaderboard* on an absolute
scale; a king-of-the-hill duel is already relative. Note the MASE half is
seasonal-naive-scaled per observation by construction, so the two halves are
not symmetric in this respect and never were.

**Absolute geomeans are not comparable across this change.** `king_geomean` /
`chal_geomean` on receipts before and after are different quantities; only the
relative improvement is.
