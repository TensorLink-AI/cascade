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

## Memory footprint at production shape (2026-09-08, measured)

`batch_size=64`, `context_length=4096`, `patch_size=32` (P=128, the widest
bucket), float32 (the trainer uses no autocast), AdamW + EMA state included.
Measured on a **physical RTX 3090** (23.6 GB usable) and, identically, on an
RTX 6000 Ada with the allocator capped:

    C    peak alloc GB   reserved GB   fits 24 GB card?
    1        1.05           1.26            yes
    4        2.51           3.24            yes
    8        4.92           5.79            yes
   16        9.77          11.01            yes
   32       19.47          21.41            yes  (~2.2 GB headroom)

Scaling is **linear in C**, not quadratic: the O(C²) variate-attention matrix is
(B·P, C, C) ≈ 33 MB at C=32, while the dominant term is time attention at
(B·C, heads, P, P), linear in C. The backbone is 2.7M params, so fixed cost is
~0.05 GB and essentially all of this is activations. Mixing widths does not
help — buckets are keyed `(P, C)` and each fills to `batch_size`, so peak is set
by the widest bucket present.

**Consequence: memory is not an argument for or against any cap up to 32.** Even
C=32 fits the smallest plausible validator card. A miner who overreaches is
self-policing anyway — OOM is a challenger fault and never re-queues
(`loop.py`), so they lose their own round rather than the fleet's.

The cap recommendation of **8** therefore rests on exactly two things: (a) caps
are additive to raise and breaking to lower, so ship the smaller one and widen
when coupled-group supply justifies it; (b) regime match with the `mv_channels
≤ 8` eval contract — a model trained wider than the eval windows pays a
mismatch penalty (the C=8-trained-model-run-at-C=1 case measured ~5% relative).

Caveats: 5-step probe, so no allocator fragmentation accumulates over a real
round; headless datacenter cards (a desktop 3090 driving a display gives up
another ~0.5-1 GB); nothing else resident on the GPU.

## Coupled-group supply already in the pool (2026-09-09, measured)

Measured on the REAL mainnet pool at block 8989200 — the byte-identical reveal
from `Tensor-Link/cascade-eval-pool` (POOL_SHA256 verified against the published
marker), i.e. exactly what validators scored that day.

    total series: 2801    distinct sources: 1238
    sources with >=2 series: 118  covering 1681 series (60.0% of pool)
    sources with >=4 series: 107  covering 1654 series (59.1%)
    sources with >=8 series:  79  covering 1505 series (53.7%)

The arming targets were >=12 coupled sources and >=30% pool share. The pool
already carries **79 sources with >=8 series each, covering 53.7%** — past both
thresholds structurally, before any new harvesting. So A3's "breadth harvest" is
largely already done: what remains is A1 (does a candidate group actually carry
cross-predictive information) and A2 (tag it). That is a tagging problem, not a
harvesting problem, and much cheaper than the plan assumed.

**Pivot direction is the live design question.** `cdc_nssp_ed_{ari,covid19,
influenza,rsv}_daily_by_state` carries 25 series each = one per (condition,
state). Two groupings are possible and they are not equally good:

* **by state, across conditions** — C=4 (ARI/COVID/flu/RSV in one state),
  sharing epidemic and reporting dynamics; plausibly cross-predictive and fits
  `mv_channels <= 8` with no subsetting.
* **by condition, across states** — C=25, over the cap, coupled only through
  weak national co-movement.

The first is the NSSP CASE-pivot and is the one worth testing first. The same
question applies to `citibike_station_status` / `cabi_station_status` (25
stations each; spatially coupled, so nearby-station subsets of <=8 are the
natural group).

None of this changes the eval-first ordering: a group is admitted on MEASURED
joint-vs-marginal lift (>=3%, per the ablation above), never on being
structurally groupable.

## The cost of flipping early, measured on the real king (2026-09-09)

Measured end-to-end on production artifacts: the **generation-7 mainnet king**
(`ckpt-r6270219435605126478-challenger-toto2-4m-u51`, fetched anonymously), its
**own generator** (`tonybilling/gen-64a0412cd332`, chronoforge v216 — the corpus
the lineage actually learned from), and the **real scored pool** at block
8989200 (byte-identical HF reveal, POOL_SHA256 verified) over the full
`[64, 256, 720]` ladder at `num_samples=100`. Warm start from
`weights_stable.safetensors` (the lineage branch) and scored on each arm's own
EMA (`ema_decay=0.999`), matching what the duel scores.

Arms equalised on BOTH axes — same series consumed per step (128) and same
240s wall — so neither steps nor data can flatter either side:

    king EMA = 0.321283
            arm  n   bs  ema geomean       sd    vs king   steps
             c1  3  128     0.391922  0.000324  +0.070638    8288
             c8  3   16     0.404719  0.004291  +0.083436   10069
     c8_coupled  3   16     0.463933  0.000810  +0.142650    9987

      c8 vs c1          +0.012798  (+3.27%)  SEPARATED (4.2 pooled sd)
      c8_coupled vs c8  +0.059214 (+14.63%)  SEPARATED

**Flipping the generator to C=8 while the pool is univariate costs 3.27% of the
round statistic** — and conservatively so, since `c8` fitted MORE steps into the
same wall (variate packing is cheaper per step) and still lost. If the miner's
generator also emits coupled channels, the total cost is ~18%: the coupled arm's
extra 14.6% is corpus shift (a channel's marginal becomes base + lagged
siblings, away from the king's prior) with no univariate eval surface to earn it
back.

**This is the eval-first rule with a price on it.** The dethrone margin is
0.5-1%; the self-inflicted penalty is 3-6x that. Raising `[generator]
max_channels` before MV windows are in the pool is not merely premature, it
actively punishes anyone who uses the headroom. Keep `max_channels = 1` and
`mv_score_from_block = 0` until coupled windows are live, then raise both at one
coordinated block.

**Also settled, on the same real weights:**

* **Do not roll the lineage back.** Random init at C=8 scored 0.894915 vs the
  warm-started flip's 0.460017 at matched steps — **94.5% worse** (and 29.4%
  worse in an earlier run on a different corpus). Seven generations are worth
  roughly double; no confound touches this comparison.
* **Generation count is irrelevant to MV readiness.** In the real gen-7 weights
  the variate layer's Q/K match a fresh init's std and stable rank to ~0.1%
  (0.06245 vs 0.06237; srank 184.33 vs 183.76), because attention over a single
  key has softmax == 1 and passes them zero gradient. Gen 1 and gen 7 are
  equally unprepared, so "roll back for a cleaner MV start" has no mechanism.
  Its V, by contrast, trained hard (stable rank 184 -> 54).
* **A diagonal-dominant (`W_k := W_q`) variate init does not help** (+0.8%, not
  separated). No mitigation to build.

**Superseded:** an earlier synthetic ablation reported wide training IMPROVING
univariate skill by 0.015-0.023 at token-matched budgets. That measured
next-patch pinball loss on a toy prior (level + one sinusoid + AR(1)). On the
duel metric with the king's real corpus the sign reverses. Prefer these numbers.

**Caveats:** 240s arms (a real round trains far longer), one pool snapshot, and
the coupled arm's coupling is a linear lagged DAG, so the 14.6% is specific to
that mechanism. All arms still sit above the king (0.392 vs 0.321) after 240s of
fine-tuning, so these are short-run deltas between matched arms, not forecasts
of a settled round.
