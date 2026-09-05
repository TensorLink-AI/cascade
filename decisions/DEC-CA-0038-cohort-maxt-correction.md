---
id: DEC-CA-0038
type: decision
title: "Cohort duel family-wise correction: a shared-resample studentised max-T, not Bonferroni α/k — exact under the challengers' real correlation, block-gated"
status: accepted
date: 2026-09-05
tags: [scoring, koth, cohort, duel, statistics, consensus]
revisit_when: "a cohort round's dethrone is disputed and the max-T bound is shown to mis-cover under the observed correlation (re-examine the studentisation / centring); or cohort sizes routinely exceed ~15 (re-check the bootstrap tail has enough bags at the joint α); or the eval draw stops being shared across the cohort (the shared-resample premise breaks and the correction must be re-derived)"
relations: {revises: DEC-CA-0012, related: [DEC-CA-0006, DEC-CA-0009, DEC-CA-0016]}
---

DEC-CA-0012 corrects the k-challenger cohort duel with Bonferroni: each
challenger's LCB is read at `bootstrap_alpha / k`. That is the WRONG
correction for this structure. All k challengers are scored against the SAME
king on ONE shared window draw, so their relative-improvement tests are
strongly POSITIVELY correlated — and the Bonferroni union bound is loose
exactly there. At k = 11, `α/k = 0.0045` reads the 0.45th percentile of a
10k-bag bootstrap (~45 bags decide the bound), over-protecting the king — the
opposite of what DEC-CA-0016 intended when it lowered the fresh-king margin to
make him beatable.

DECISION: judge the cohort under a **shared-resample max-T** (Westfall–Young
step-down simultaneous band) instead. One shared cluster resample scores the
king and every challenger jointly (`cascade.eval.bootstrap.joint_bag_geomeans`,
already the heat's machinery), and the critical value is read off the ACTUAL
joint spread:

    rel[j,b] = (king_geo[b] - chal_geo[j,b]) / king_geo[b]
    D[b]     = max_j ( t[j] - rel[j,b] ) / se[j]        # studentised downward error
    c        = quantile(D, 1 - α)                       # one simultaneous critical value
    L[j]     = t[j] - c · se[j]                         # family-wise LCB per challenger

Exact under the real correlation: at ρ→1 `D` collapses to a single
challenger's deviation (no correction); at ρ→0 it widens to the true max of k
— never the fixed `α/k` tail. It reduces BIT-IDENTICALLY to the deployed
percentile LCB at k = 1 (the SE cancels), so single-challenger rounds — every
round before cohorts — never change, and it always sits BETWEEN Bonferroni
(lowest bound) and no correction (highest).

Two refinements the naive form gets wrong, both pinned by tests
(`tests/unit/test_cohort_maxt.py`): (1) it is CENTRED (basic bootstrap), not a
raw percentile of the per-bag max — `quantile(max_j rel[j,b], α)` is only
calibrated at ρ = 1 and grossly anti-conservative otherwise (at independence,
k = 11, it collapses to the ~76th percentile of a single test); (2) it is
STUDENTISED — a single additive critical value is hijacked by the
widest-spread challenger and can come out MORE conservative than Bonferroni;
dividing each deviation by its own bootstrap SE restores the per-challenger
scale-adaptivity the percentile bound has for free.

CONSENSUS, block-gated via `[scoring] cohort_maxt_from_block`
(release-then-activate, the `margin_activation_block` / `mix_from_block`
shape): every validator resolves the rule from the round's epoch block, so
restart timing never forks a verdict, and `cascade-audit` replays each round
under its own block's rule (`check_verdict` / `check_duel_cohort` take the
config and recompute the crowned/​published LCBs under the resolved rule —
`cascade.eval.koth.cohort_maxt_lcb_map` is the single implementation both the
validator and the audit call). The receipt records no new field: the
correction lives in the block gate, so `VerdictRecord.params` stays the
unmodified `[scoring]` set (`check_koth_params` still matches chain.toml) and
archived-receipt bytes are untouched. `0` = Bonferroni forever.

ARMED: mainnet `cohort_maxt_from_block = 9043200` (≈ Fri 2026-09-11 08:30 UTC,
an epoch boundary — one coordinated validator-upgrade window; external
validators diverge from that block until upgraded, exactly like the ladder /
margin flips). Testnet armed at 1 for the one-cycle multi-cohort validation.
