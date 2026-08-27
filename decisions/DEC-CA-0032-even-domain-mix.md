---
id: DEC-CA-0032
type: decision
title: "Even-by-domain eval draws: pool capacity floors per domain + tightened mix jitter (alpha 4 -> 28), coordinated block-gated activation"
status: accepted
date: 2026-08-25
tags: [eval, pool, sampler, mix, consensus, goodhart]
revisit_when: "pool snapshots carry dgp_class/cadence labels broadly (arm the granularity axis as a second balanced tier); or realized effective_domains p5 drops below 6.5 after both parts land (caps binding again); or a domain's source dries up below its floor (rebalance the floor, not the sampler)"
---

# Even-by-domain eval draws

## Problem (owner report 2026-08-25, verified)

The DEC-CA-0019 jittered mix is uniform only IN EXPECTATION. Realized draws
are heavily uneven — r40 drew nature 30.5% / transport 26.7% / web 4.0% /
econ_fin 5.5% (effective_domains 5.49 of 7). Two compounding causes,
separated by simulation against the deployed `_capped_jittered_split` with
r40-era capacities (2,000 trials):

1. **Jitter width**: `mix_jitter_alpha = 4.0` lets an uncapped domain
   realize anywhere in 2%-35% of the round (web_cloudops hit 4% in r40 with
   ~300 windows of spare capacity).
2. **Pool capacity**: the pool itself runs ~600 transport / ~490 nature
   windows against ~93 econ_fin / ~116 sales. After the 0.7 series bag,
   econ_fin's cap is ~65 of a 171-window even share — **even with jitter
   disabled entirely, econ_fin=5.4%, sales=6.7%, healthcare=8.8%.** Evenness
   is unreachable by any sampler change alone.

Related pressure (docs/notes/2026-08-25-generator-lineage-r35-r40.md): miners
statistically characterize pool marginals ("stable across all 8 snapshots on
disk") and tune corpora toward them. A capacity-starved, stable pool
composition is both an uneven eval AND an easy Goodhart target.

## Decision (two parts, matching the two causes)

**Part A — pool capacity floors (owner-side, next snapshot builds).** The
pool builder targets >= 300 windows per domain per snapshot (an even 1200/7
share needs ~171; /0.7 bag ≈ 245; 300 gives headroom). Overweight domains
shrink toward parity as sources allow. Snapshot composition additionally
DRIFTS deliberately between builds (source rotation) so pool marginals stop
being a stable tuning target. Non-consensus: validators score whatever
snapshot is published; no coordination needed beyond normal rotation.

**Part B — tightened jitter (consensus constant, coordinated).**
`mix_jitter_alpha` 4.0 -> 28.0: per-domain realized share tightens to roughly
+/-2-3pp around even (simulated: effective_domains p5 rises 5.60 -> 6.12 on
TODAY'S skewed caps; on Part-A-rebalanced caps the draw becomes even +/- a
couple of points, which is the owner's spec). Still jittered per round —
composition stays unpredictable at submission time. Activation is
block-gated exactly like `mix_from_block` (all 6 external validators deploy,
flip at a pinned epoch boundary, audit replays each round's own alpha);
testnet cycle first. NOT a local config flip — a mid-window mismatch forks
verdicts.

## Interim mechanism (owner-accepted 2026-08-25): two-tier split

Until Part A's floors land, `_tiered_split` (this PR) replaces the flat
Dirichlet when `mix_tier_from_block` activates: domains whose capacity cannot
fill an even share of the shrinking remainder draw AT CAPACITY every round
(the flat jitter could starve them below even their caps — r40 gave
web_cloudops 48 of ~300 available); the surviving domains split the rest
under `mix_tier_jitter_alpha = 75` (a 4-group tier needs ~75 for the +/-2-3pp
band; 7 groups need ~28 — alpha is group-count-dependent). Simulated on r40
capacities: scarce three pinned at 5.5/6.8/9.1%, big four ~19.7% each,
realized band 16-25% vs the flat split's 2-35%. The scarce classification is
deterministic and consumes no randomness; when every domain clears an even
share the scarce set is empty and the split degenerates to a tight even
jitter over all domains — the design converges to uniform +/-2-3pp as Part
A's capacity lands, with no second migration.

## Explicitly deferred

Granularity/cadence as a balanced axis. The realized draw already publishes
`composition.cadences`; balancing it needs per-series cadence class labels in
snapshots first (same precondition as dgp_class rotation). Take up when the
pool carries labels (see revisit_when).

## Evidence

Simulation (deployed `_capped_jittered_split`, r40 caps [65, 290, 106, 345,
81, 420, 303], n=1200, block=8, 2000 trials):

| | eff_domains mean | p5 | uncapped-domain range |
|---|---|---|---|
| alpha 4 (live) | 5.99 | 5.60 | 2.0-35.0% |
| alpha 28 | 6.21 | 6.12 | ~10-32% (tails), +/-3pp typical |
| no jitter | 6.27 | 6.27 | fixed, but small domains stuck at 5.4/6.7/8.8% |

r40 realized (manifest composition block): nature 366, transport 320, energy
216, healthcare 109, sales 75, econ_fin 66, web 48 of 1200.
