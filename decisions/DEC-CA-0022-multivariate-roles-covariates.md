---
id: DEC-CA-0022
type: decision
title: "Multivariate arms eval-first: per-series C with bucketed batching, full-freight channel pricing, roles as a record field — and the channel-drop trap is fixed before any cap raise"
status: proposed
date: 2026-08-13
tags: [interface, generator, trainer, eval, scoring, koth, pool, dedup]
revisit_when: "the MV-under-univariate-eval ablation (below) shows multivariate training HELPS the current univariate eval — then the strict eval-first sequencing can relax to eval-parallel; or honest generators trip the shadow-logged channel-correlation diagnostic at material rates — then the proposed 0.999 bar is wrong and the gate design reopens; or future-known covariates are to be armed before the pool has a written exogeneity curation rule — which this node forbids"
relations: {depends_on: DEC-CA-0016, subsumes: DEC-CA-0020, constrained_by: DEC-CA-0012}
---

Gap 6 is where the dormant capacity is: the variate-attention layers exist,
initialise, and train (`toto2_model.py:111`, `_Block axis="variate"`; with
`num_layers = 4` and `layer_group_size = 4`, layer 3 is variate) but have
only ever executed at `C = 1`. It is also where the eval-precedence claim of
Part C **binds hardest** — confirmed below with a sharper mechanism than
budget economics — and where verification found a correctness trap that must
be fixed before `max_channels` moves at all.

## The trap first: today, C > 1 is billed C× and trained 1×

`iter_training_batches` reduces every `(C, L)` series to channel 0 — `s =
s[0]` (`toto2_trainer.py:122-124`) — while the stream bills all channels
against the token budget (`total += arr.size`, `stream.py:186`) and the
trainer counts only the surviving channel's tokens toward the training
budget. If `max_channels` were raised with no other change, a multivariate
miner would pay `C×` stream budget for `1×` training signal, the stream
would exhaust before the token budget, and the run would silently
under-train and flag `deadline_hit`. The eval side has the same shape:
the checkpoint wrapper's `_prep` accepts 1-D histories only. **No
`max_channels` raise, ever, until the trainer consumes every channel.**
This fix is trainer code — digest-blind — so it ships under the lockstep
deploy discipline (DEC-CA-0009 precedent) and is inert at `C = 1`.

## (a) C is per-series; batches bucket by (P, C)

`Toto2Model.forward` takes `(B, C, P, ps)` — uniform C per batch
(`toto2_model.py:352`). Rather than variate-axis padding plus an attention
mask (new masking semantics inside a pinned arch — rejected), extend the
existing length-bucketing (`iter_training_batches` already buckets by P) to
key on `(P, C)`. Identical machinery, no padding, no mask; the cost is more
partial-bucket flushes at stream end, which the current code already
tolerates. A generator may freely mix C = 1 and C = 4 series in one corpus.

## (b) Channel economics: full freight, no discount — and the dominance
problem is the eval's, not the budget's

A C-channel series prices at `C × L` points (bytes under G3). A discount is
rejected for the stated reason — it mints cheap points and invites channel
spam — but the sharper finding is that budget pricing is not what makes
multivariate dominated today. Once the trainer consumes all channels, `C×`
cost buys `C×` trained tokens: actuarially fair. What makes a rational miner
ignore the axis is the eval:

**Training at C > 1 and evaluating at C = 1 is a regime mismatch inside the
variate layers.** At eval the wrapper feeds single-channel histories, so
variate attention runs over a singleton set — a learned per-token transform
that saw a different input distribution than it trained under. Multivariate
training moves the variate layers' weights *away* from the C = 1 regime the
eval exercises; the null hypothesis is that it *hurts* the score being
paid. So under the current scorer, multivariate data is worse than ignored —
it is plausibly self-sabotage at full price. This is Part C's claim with a
mechanism attached: **the eval change precedes the surface change, hard.**

The settling experiment (testnet, one round's budget): train paired 4M runs
under one `RoundSeeds` — corpus A univariate, corpus B the same generator
emitting coupled C = 4 groups — and score both on the *current* univariate
eval. If B ≥ A, the sequencing can relax; if B < A (expected), the eval
lands first and the result is the number quoted to miners for why.

## The eval work that gates arming (with DEC-CA-0016 G6 already in place)

1. **Joint forecaster.** `ForecastFn` `(C, L) → (C, ns, H)` and
   `score_forecaster_on_windows` calling it once per window — shipped
   trivially at Stage 0 (carrier node). The wrapper for *new* checkpoints
   accepts `(C, L)` and decodes channels jointly through the variate axis;
   archived 1-D wrappers keep a per-channel adapter forever.
2. **Multivariate windows in the pool.** The loader and `EvalWindow` are
   MV-ready (`pool.py:14`, `window.py:22`), but the **builder drops
   multi-channel series at harvest** (`docs/EVAL_POOL.md`, "Cleaning") — so
   today the capability would be measured on zero windows. The builder
   keeps genuinely-coupled real groups (Open-Meteo's 12 variables per grid
   point are natural C = 12 candidates; forge feeds with sibling metrics).
3. **`source` on every multivariate window, and the cluster fallback fixed.**
   `_window_clusters` (`koth.py:159`) makes each sourceless *row* a
   singleton cluster — C perfectly-correlated channels of one window would
   bootstrap as C independent observations, inflating effective sample size
   exactly when MV windows enter. Fallback key becomes `series_id`
   (Stage 0, behaviour-identical at C = 1); `source` becomes guaranteed at
   pool build; `min_clusters` rises per the existing plan
   (`chain.toml:322`).
4. **Aggregation with MV windows in the minority.** Per-channel `WindowScore`
   rows geomean into the round statistic exactly as univariate rows do
   (DEC-CA-0009's per-window normalisation is per-row and scale-free), and
   clustering by series/source keeps a 12-channel window from voting 12
   times. No new aggregation rule is needed — the cluster key is the whole
   defence, which is why item 3 is non-negotiable.

## (c) Rank-collapse: a gate is proposed, but honestly it is data-quality,
not incentive defence

The feared exploit — C near-duplicate channels with 1e-9 jitter filling the
budget with no information, invisible to byte-dedup (`_series_key` is
per-series; channels within a series are never compared) — is real as
*waste* but self-financed: the miner pays full freight for redundant tokens
that train little, and eval windows are **real data the miner does not
shape**, so duplicated training channels buy no eval-side row inflation.
The failure mode is a miner harming their own model at their own expense,
plus corpus-quality erosion.

Proposal, in the DEC-CA-0010 shape (measure first):

- **Shadow**: log per-series max off-diagonal |Pearson| (on standardized
  channels) and effective rank (participation ratio of the channel-
  covariance spectrum) into the training log. Cost O(C²L) per series,
  negligible at small C.
- **Gate, when armed**: `max_channel_corr = 0.999` as a `check_series`
  argument beside `reject_constant` — a bar for *near-identity*, not
  similarity (DEC-CA-0008's lesson: ratio bars at meaningful thresholds are
  spacing-gameable and false-positive-prone; 0.999 targets only the jitter
  exploit, which has no reason to live below it).
- **Honest false positives at any tighter bar**: co-located sensor pairs,
  a metric and its EMA/cumulative transform, hard-coupled physical
  simulation outputs (voltage/current under near-constant load), saturated
  or clipped channels agreeing at the rail. At 0.999 these mostly survive;
  the shadow logs decide whether even 0.999 draws honest blood before it
  enforces.

## Variate roles and future-known covariates

**Roles ship as a record field, not positional convention + config.** The
positional proposal (`n_targets`/`n_past_cov`/`n_future_cov`, channels
ordered) was motivated by needing no schema change — but DEC-CA-0016 removes
that motivation, and positional+config is strictly worse on its own terms:
it splits one fact across two files, forces every series in a corpus to
share one role split (no mixed corpora), and creates an ordering invariant
that survives only by convention. `roles: (C,) uint8` (0 target, 1 past-cov,
2 future-known) is per-series, self-describing, additive, and absent means
all-targets — every deployed generator valid untouched.

**The CPM mask carries visibility with no new tensor — confirmed.**
`Toto2Model.forward` accepts patch-level `(B, C, P)` masks and expands them
per-entry (`toto2_model.py:368-370`); the trainer's mask construction
(`toto2_trainer.py:317-324`) is already per-row. Role-aware masking is
purely a construction rule: future-known channels keep their horizon region
unmasked (values visible), targets get it masked. Observability stays the
trainer's decision, as specified — the generator declares eligibility via
`roles`, the CPM mask decides visibility.

**The covariate incentive failure, worked through.** A miner emits a
future-known channel that is a near-deterministic function of the target's
future; the model learns to copy; it is useless on real data. Is it
self-correcting under our eval?

- **Under the current eval: yes, trivially** — no eval window carries
  covariates, so a copy-trained model just wasted budget and skewed its
  variate layers. Self-harm.
- **Under a covariate-aware eval: conditionally.** Self-correction holds iff
  eval windows' future-known channels are (i) genuinely ex-ante — values
  knowable before the target's realisation (calendars, schedules, published
  forecasts, posted prices) — and (ii) no more informative about the target
  future than real-world exogenous covariates are. If the pool ever includes
  a "covariate" that is a smoothed or lagged transform of the target itself,
  copy-training *pays* and the incentive inverts.
- **`docs/EVAL_POOL.md` guarantees none of this today** — it has no
  covariate concept. So the guarantee must be *written before the role
  arms*: a covariate-curation rule (exogeneity by provenance, not by
  statistics alone) plus a build-time informativeness screen — reject
  candidate future-known channels whose future explains the target future
  above a bar (R² threshold to be set from the measured distribution of
  honest exogenous covariates; that measurement is an open question below).
  Until that section exists, `roles` value 2 is refused even after values
  0/1 are accepted.

## Change surface, digest, migration

- `chain.toml`: `max_channels` raise (`[generator]`, **not** digest-bound —
  verified against `TrainingContractConfig`); `max_channel_corr` +
  shadow flags; at roles arming, the accepted-field fold into `[training]`
  (one digest bump, shared with DEC-CA-0019's if co-scheduled).
- Interface: record fields `roles` (this node) — `values` at `(C, L)` needs
  nothing new; validation of roles vocabulary and shape.
- Trainer: channel-drop fix + `(P, C)` bucketing (inert at C = 1, lockstep);
  role-aware CPM construction (at arming).
- Eval: items 1–4 above; wrapper joint decode for new checkpoints.
- Migration: univariate generators untouched at every stage; 1-D yields stay
  the common case indefinitely.
- Gaming: channel spam (self-financed, shadow-then-gate above); endogenous
  covariates (curation rule above); role-flipping a target to covariate to
  dodge loss on hard channels — priced correctly for free, since covariate
  channels carry no loss signal, the miner is just buying conditioning
  context at full freight.

## Open questions

1. MV-vs-univariate ablation under the current eval (sequencing hinge —
   experiment specified above).
2. Cap value at arming: C = 4 vs 8 — decided by measured variate-attention
   cost at the bucketed batch shapes and by what real coupled groups the
   pool can supply (Open-Meteo C = 12 argues for ≥ 8; compute argues small).
3. The exogeneity R² bar for eval covariates — measure the informativeness
   distribution of true ex-ante covariates (calendar dummies, published
   day-ahead forecasts) against pool targets before setting it.
4. Whether `max_channels` should gain a digest-visible `[training]` mirror
   at arming so a validator can detect a trainer running a different cap —
   today nothing would (the gate checks `[training]` only).

## Rank

#1 of the six by value-per-effort once the carrier exists: the capacity is
already trained-but-dormant in every checkpoint, the data shape is already
canonical, and the whole cost is concentrated in eval work that is needed
anyway (cluster hygiene, pool `source` guarantees) — against the prompt's
ordering, this displaces gaps 1–2 from the top.
