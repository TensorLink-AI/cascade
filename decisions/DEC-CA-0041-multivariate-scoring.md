---
id: DEC-CA-0041
type: decision
title: "Multivariate scoring for the private-pool duel is GIFT-Eval-weighted (per-variate metric, a window counts ONCE with its channels averaged in) and lands eval-first: the round point statistic is made to agree with the source-cluster bootstrap, block-gated and bit-identical on the univariate pool"
status: proposed
date: 2026-09-08
tags: [scoring, koth, eval, multivariate, gift, consensus, pool]
revisit_when: "before arming mv_score_from_block: the coupled-source count must be measured against the LCB-width requirement below (a realistic single-digit MV gain from ~3 coupled sources does NOT clear a 0.5% margin; ~12+ sources / ~38% pool share does), the settling ablation (DEC-CA-0026) must show MV training now HELPS the MV-aware eval, and WindowScoreRecord.channel serialization must be confirmed for audit replay; revisit the arithmetic-vs-geometric within-window combiner if the owner prefers geometric variate averaging; revisit if the eval pool stops being source-tagged (the cluster key, hence the N_eff argument, breaks)"
relations: {revises: DEC-CA-0026, depends_on: [DEC-CA-0009, DEC-CA-0012], relates_to: [DEC-CA-0038, DEC-CA-0040]}
---

Cascade already runs GIFT-Eval as the public no-regression gate (`gift_gate.py`,
Seasonal-Naive-normalised shifted geomean, per-config, paired bootstrap). This
node makes the PRIVATE pool score multivariate the SAME way, and fixes the one
place the machinery was not yet GIFT-faithful.

**FINDING (verified in code, not the DEC-CA-0026 text — which was 3 weeks
stale).** The eval path already ingests `(C, L)` windows end to end: the trainer
consumes every channel with correct token billing (the old channel-drop trap is
fixed, `toto2_trainer.py`), the validator scores native joint `(C, L)` forecasts
(`evaluator.load_forecaster`; archived 1-D wrappers lift through
`adapt_per_channel`), and the paired bootstrap clusters by `source`/`series_id`
so a C-channel window is ONE resampling unit (`koth._window_clusters`). There is
NO eval-side gate on `n_channels`; `max_channels` is generator-side only. So MV
windows in the pool are scored today with **zero switches flipped** — but the
round POINT statistic (`global_geomean`/`global_components`) geomeans over
`WindowScore` ROWS, one per `(window, channel)`, so a C-channel window votes
**C×**. The variance side is GIFT-faithful; the point side is not.

**DECISION: average a window's channels into ONE per-window contribution** — the
GIFT convention (each config/window counts once, variates averaged within),
arithmetic mean over variates for both halves (MASE; and per-window WQL over the
channels where `|y| > 0`, masking zero-target channels from the WQL half only,
exactly as `wql_per_window` does). NOT `sum(qloss)/sum(abs_target)` pooling — a
measured 0.28× scale-domination by the largest-|y| channel (DEC-CA-0009's lesson
across channels). `cascade.eval.scoring.collapse_channels_by_window`, applied in
`evaluate_round` when armed, feeds BOTH the point estimate and the bootstrap, so
they are computed on the same weighting.

**BIT-IDENTICAL on the univariate pool.** A single-channel window passes through
the collapse UNTOUCHED (no re-encode round-trip), so every univariate window —
i.e. the entire pool until coupled data lands — is a literal no-op. Pinned by
`tests/unit/test_mv_scoring.py::test_univariate_pool_is_bit_identical_mv_on_vs_off`
(exact `==` on lcb and both geomeans).

CONSENSUS, block-gated `[scoring] mv_score_from_block` (the `cohort_maxt_from_block`
/ `increment_from_block` shape): `koth_params(block)` resolves `KothParams.mv_score`
via `mv_score_active`, every validator resolves the rule from the round's epoch
block, and audit replays each round under its own block's rule. No receipt-format
change — the correction lives in the gate; per-channel `WindowScore` rows are still
what is recorded. `0` = per-channel forever. **UNARMED** (`mv_score_from_block = 0`).

**Rewarded the GIFT way, and the N_eff constraint that follows (measured on the
real cluster bootstrap; magnitudes illustrative).** MV is rewarded only because a
joint model that conditions across channels lowers each variate's per-window
error — there is no MV bonus. Two consequences the numbers make hard:
- **Channel count buys NO statistical evidence.** Under the source-cluster
  bootstrap the effective sample size is the distinct-coupled-**source** count.
  C=4 vs C=12 on the same 3 sources → identical point and LCB (+0.0371 / +0.0191).
- **The reward is bounded by coupled-source count AND pool share.** A realistic
  single-digit MV gain from ~3 coupled sources does NOT clear a 0.5% margin
  (LCB ~+0.001–0.005); ~12+ coupled sources (~38% pool share) is where a 3–5%
  gain clears. Rule of thumb: `gain × MV-share ≳ 1%`. So the beatability lever is
  distinct coupled SOURCES, harvested broadly — never channels stacked on a few.

**Channels are free; the coupled-source count is the currency.** A coupled group
packs to one `(C, L)` window (the `mv_channels` contract, `docs/EVAL_POOL.md`):
one series against the 200/source cap, one joint forecast call, and — the
bootstrap clustering by `source` — zero extra clusters. So tagging costs the pool
nothing and there is no size-vs-evidence tension; the cross-predictiveness audit
(lagged sibling info, not `|corr|`) is the only gatekeeper. NOTE the deliberate
forge/cascade accounting divergence: forge's OWN benchmark expands a group into C
univariate challenges (C slots, group-level bootstrap); cascade packs to one
window (1 slot, source-level clusters). Same tag, two consumers, NOT reconciled —
neither should be "fixed" to match the other.

**Rejected alternatives:** per-row weighting (over-weights a C-channel window C×
and makes it a high-leverage bootstrap unit — the current, non-GIFT default);
`sum/sum` WQL pooling (scale-dominated, above); scoring MV as C independent
univariate challenges (the tsbenchforge eval model — never lets the joint
forecaster condition, so MV is not rewarded); raising `bootstrap_alpha` to make
MV easier (buys beatability by loosening the noise gate — the wrong axis, cf.
DEC-CA-0040).

**Eval-first ordering (the hard sequence).** This aggregation change must be live
on every validator BEFORE any MV window enters the pool (a pool snapshot alone
would otherwise silently activate the per-row C× weighting). And the whole eval
must reward MV before the generator surface moves (DEC-CA-0026: training C>1 while
the pool is all-univariate is self-sabotage under the variate-layer regime
mismatch). Rollout:
1. **[THIS NODE — Phase 0, landed]** `collapse_channels_by_window` +
   `mv_score_from_block`, inert at C=1, tests, gate at 0. Deploy fleet-wide.
2. Confirm `WindowScoreRecord.channel` serialization for MV audit replay.
3. Pool builder: stop dropping multichannel at harvest; pack `mv_channels`-tagged
   columns (forge `sources.yaml`, an inert data-structuring change on the forge
   side) into one `(C, L)` `EvalWindow`; a cross-**predictiveness** screen (not
   `|corr|`); many distinct coupled sources per the N_eff numbers; raise
   `min_clusters`.
3.5 Shadow MV windows at a small share; measure LCB width vs coupled-source count.
4. Settling ablation (DEC-CA-0026): paired univariate vs coupled-C=4 corpora on
   the MV-aware eval — arm nothing until B ≥ A.
5. Raise `[generator] max_channels` (4–8, matched to coupled-group supply and the
   O(C²) variate-attention cost; not 10) and arm `mv_score_from_block` at the
   same coordinated block the first MV pool snapshot goes live.

## Phase 3 ablation results (2026-09-08, measured)

Run on an RTX 6000 Ada, Toto2 backbone, L=4096/P=128, equal 240s wall per arm,
3 seeds per config, synthetic coupled corpora from
`cascade/interface/mv_reference_generator.py`. Harnesses were scratch; the
numbers below are the durable part.

**1. Wide training is cheaper, not costlier.** Against a *token-matched*
univariate baseline (C=1 at batch 16·C, so both push identical sequence counts),
packing sequences on the variate axis is ~20% faster per step and ~40% lighter
in peak memory, consistently at every width to C=32:

    seqs/step   MV ms/step   UV ms/step   MV peak GB   UV peak GB
       64          11.1        13.2          0.71        1.12
      128          20.7        25.8          1.35        2.18
      256          42.5        55.0          2.66        4.31
      512          89.9       118.0          5.26        9.04

This corrects the O(C²) worry that motivated capping `max_channels` low. C=32 is
8× slower per step than C=1/batch-16, but that comparison is against an idle GPU
(0.32 GB peak); against equal work the variate axis wins.

**2. Wide training also improves univariate skill.** Held-out univariate pinball
loss vs the token-matched control, 3 seeds, sd ≈ 0.001:

    C=4  vs C=1/BS=64    -0.02331  (21 sd)
    C=8  vs C=1/BS=128   -0.01527  (13 sd)

So raising `max_channels` does not depend on the multivariate thesis at all — it
is free on throughput and positive on univariate quality. Note the univariate
control curve is non-monotonic in batch size at fixed LR 3e-4 (0.0803 → 0.0888 →
0.0759 → 0.0551 → 0.0519 for BS 16→512), so LR is not neutral across that range;
the MV-vs-control comparisons above are matched pairs and unaffected.

**3. Variate attention extracts real cross-channel information, and it must be
LEARNED.** Channel-shuffle test: score one model on C=8 windows built normally
vs windows whose channel k is drawn from a *different* series (same shape, same
per-channel marginals, co-membership destroyed). All arms at C=8, so no regime
mismatch. `eval_coup=0.0` is a null — true and shuffled are then the same
distribution, so the gap is required to be zero.

    train_coup  eval_coup   true     shuffled    gap        rel%
       1.2        0.0     0.05293   0.05303   -0.00009     0.17   <- null clean
       1.2        0.3     0.05951   0.06034   -0.00083     1.37
       1.2        0.6     0.05759   0.05964   -0.00205     3.43
       1.2        1.2     0.04929   0.05270   -0.00341     6.47
       1.2        2.4     0.04753   0.05136   -0.00383     7.46
       0.0        0.0     0.04431   0.04435   -0.00004     0.08   <- null clean
       0.0        0.3     0.06192   0.06190   +0.00002    -0.02
       0.0        0.6     0.06910   0.06901   +0.00009    -0.13
       0.0        1.2     0.06571   0.06562   +0.00009    -0.13
       0.0        2.4     0.06466   0.06450   +0.00016    -0.25

Null clean in both arms; monotonic dose-response in the coupled-trained arm; and
the model trained on *independent* channels gains exactly nothing at any
coupling strength. The capability is learned from coupled training data, not a
generic property of the architecture.

**Consequence for the A1 admission threshold.** With the DEC-CA-0040 rule of
thumb (`gain × MV-share ≳ 1%` to clear a 0.5% margin):

    coupling    joint-vs-marginal lift    MV pool share needed
    0.3 (weak)         1.4%                   ~73%  -- infeasible
    0.6 (moderate)     3.4%                   ~29%
    1.2 (strong)       6.5%                   ~15%

A1 should admit column groups whose measured **joint-vs-marginal lift is ≳3%**
and target ≳30% pool share. `|corr|` is not the bar and neither is "coupled in
principle" — weakly-coupled groups would need three-quarters of the pool to move
a duel and are not worth harvesting.

**Consequence for step 5.** Arming MV scoring rewards miners whose *generators
emit causally coupled channels*, since the skill cannot be picked up from
independent-channel corpora. That is an incentive gradient on generator design,
not just model architecture, and it is reachable — `mv_reference_generator`
is the worked example.

**Caveats.** Synthetic corpora throughout, generously coupled (dense random DAG,
lags ≤24); real harvested panel columns will sit lower on the dose-response
curve, so 3.4% at coupling 0.6 is the honest planning anchor and the 1.2/2.4
rows are ceilings, not forecasts. 240s per arm is short — these are directional.
The eval-first ordering above is unchanged: none of this licenses arming the
gate before MV windows are actually in the pool.

**Methodology note.** Four earlier designs returned nulls, each for a different
reason: a univariate eval set (variate attention inert at eval, null by
construction), a joint-vs-marginal comparison (C=8-trained model run at C=1 is a
regime mismatch, not an information test), and two runs invalidated by the
generator's `coupling_strength` config key not matching its `_coupling`
attribute — a silent no-op that made every "coupling" arm identical (fixed in
`041bb3e`). The harness now asserts that each coupling level produces a distinct
corpus. Worth repeating for any future ablation here: a knob that does not
change the data will otherwise return a confident, meaningless null.
