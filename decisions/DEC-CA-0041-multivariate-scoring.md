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
