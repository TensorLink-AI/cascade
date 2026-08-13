---
id: DEC-CA-0021
type: decision
title: "Multivariate: bucket batches by (C, P) and never pad the variate axis; roles are a per-series KEY not a positional convention; no budget discount, ever, on a per-channel scorer"
status: proposed
date: 2026-08-13
tags: [interface, generator, model-arch, multivariate, covariates, eval, budget, incentives]
revisit_when: "the eval scores cross-variate skill on a pool slice large enough to move the round metric (≈10% of windows at a plausible within-slice effect) — before that, every part of this decision's payload is a capability miners are rationally required to ignore"
relations: {depends_on: [DEC-CA-0015, DEC-CA-0019], related: DEC-CA-0009}
---
Raising `max_channels` is one config line. Everything behind it is not.

## The variate path is unreachable, not dormant

`iter_training_batches` does `if s.ndim == 2: s = s[0]` (`toto2_trainer.py:120`)
— channels ≥ 1 are **dropped before training**. `causal_standardize` takes
`(B, L)`. The model is called with a 3-D tensor and takes its
`squeeze_variates` path. So raising `max_channels` today would silently discard
every extra channel *while the stream budget charged for it*. The variate layers
exist and are trainable, but nothing upstream can reach them.

## (a) `C` fixed or per-series: bucket by `(C, P)`, do not pad

`iter_training_batches` already buckets by patch count `P` so rows stack without
padding; the key becomes the pair `(C, P)`. Generators emit tens of thousands of
series, so bucket occupancy is not a problem.

Padding the variate axis is worse than it looks. `_Block` with `axis="variate"`
calls `F.scaled_dot_product_attention(q, k, v, is_causal=False,
scale=1/head_dim)` with **no `attn_mask` argument** (`toto2_model.py:303-305`).
Padded variates would attend into every real variate and corrupt them. Masking
them means editing `toto2_model.py`, which bumps `base_arch_digest` and
`contract_digest` and mismatches every archived checkpoint — paying the most
expensive digest on the roadmap to avoid a dictionary key.

Also required, both trainer-side and neither an arch change:
`causal_standardize` becomes per-(row, channel), and `sample_cpm_masks` returns
`(B*C, P)`.

## (b) Who pays for the channel axis: the miner, at full rate

The budget counts `C * L`. Today that understates the penalty: under
`corpus_mode = "stream_cpu"` the stream's counter is `arr.size`
(`stream.py:186-189`) while the trainer's token counter only counts channel-0
timesteps — so a `C = 4` corpus exhausts the stream at roughly 25% of
`train_tokens` and the run ends **under compute**, loudly (`tokens_frac`,
`deadline_hit`) and loses the round. Multivariate is not merely dominated today;
it is a self-inflicted 4× compute cut.

After the variate axis lands, multivariate stays dominated as long as the scorer
runs per channel with a univariate `forecast_fn` — a coupled `C = 4` group costs
four univariate series and earns four independent univariate scores. That is
exactly why Part C binds hardest here.

**A budget discount is refused permanently.** It would subsidise a capability
the eval cannot see, and it is payable by channel spam. The honest way to make
multivariate worth its cost is an eval that rewards cross-variate skill; then it
pays for itself at full price.

## (c) Rank collapse: do not build the gate — refuse the discount instead

The proposed defence (a `max_channel_corr` / `min_effective_rank` argument to
`check_series` beside `reject_constant`) is aimed at a miner filling the budget
with `C` near-duplicate channels at 1e-9 jitter, which byte-dedup misses.
Correct that byte-dedup misses it. Wrong that a threshold is the answer.

Under a `C * L` budget with **no discount** and a small `max_channels`, channel
spam is self-penalising: the miner pays C× for ~1 channel of information and
trains a worse model. The attack only becomes profitable if a discount is
granted — so refusing the discount *is* the mitigation, and it is exact rather
than approximate.

A hard bar meanwhile has real false positives on precisely the priors worth
having:

- **Genuinely coupled physics.** Open-Meteo's own variable list ships
  `temperature_2m`, `dew_point_2m` and `apparent_temperature` — near-deterministic
  functions of each other. A prior that models them is honest and would trip a
  correlation bar.
- **Common-factor panels.** 500 hosts of one metric during one incident are
  high-correlation *by construction* — the Chronos-2 target shape (DEC-CA-0019).
- **Short series.** At `min_length = 64` the correlation estimate is noisy
  enough that the bar's false-positive rate is a function of length, not of
  intent — the same "gameable by construction" defect that got the 0.99
  similarity threshold removed in DEC-CA-0008.

**DECISION: ship effective rank as a shadow diagnostic** — per-series
`exp(H(σ/Σσ))` over the standardized `(C, L)` matrix, logged with the corpus
summary, gating nothing. The DEC-CA-0006 / DEC-CA-0010 idiom: compute it, record
it, do not obey it. Revisit only if a discount is ever granted.

## Roles: a per-series key, not a positional convention

The case for positional (`n_targets` / `n_past_cov` / `n_future_cov` in config,
channels ordered targets → past-cov → future-known) rests on "it needs no schema
change and keeps deployed generators valid". **DEC-CA-0015 is already making the
schema change**, and after it a `roles` key costs nothing extra while positional
is strictly worse:

- it is a **global** split — one role layout for every series in a corpus, no
  per-series variation, which forecloses a prior that mixes coupled groups with
  bare univariate series;
- it lives in config, so changing it **silently reinterprets** an existing
  corpus (the same channels mean something different); a per-series key cannot;
- deployed generators stay valid either way, because they yield bare arrays.

**DECISION: `roles` is a per-series record key** under DEC-CA-0015, accepted in
the same release as `max_channels > 1`.

**Observability is not the generator's job — confirmed, and the mask carries
it.** `Toto2Model.forward` accepts `mask.dim() == 3` i.e. `(B, C, P)` and
expands it to per-entry itself (`toto2_model.py:368-370`); the trainer already
builds `step_mask` by exactly that expansion (`toto2_trainer.py:315-318`). A
role-conditioned mask — horizon region masked for targets and past-covariates,
unmasked for channels the generator declared future-known — therefore needs **no
new tensor and no arch change**. What it needs is a role-aware
`sample_cpm_masks` over `(B*C, P)`. Confirmed as hypothesised.

## The copy-the-covariate incentive failure

A miner emits a future-known channel that is a near-deterministic function of
the target's future; the model learns to copy the covariate instead of
forecasting, and is useless where scheduled-event covariates are weakly
informative.

**Self-correcting today, trivially and destructively.** `EvalWindow` carries
history and target only — the eval supplies no covariates at all. A copy circuit
has nothing to read at inference, so the miner has burned model capacity *and* a
C× budget multiple. It loses.

**It stops being self-correcting the moment the eval supplies future-known
covariates,** and then only under three conditions: the eval's covariate
informativeness must be (i) drawn from real data, (ii) not observable to miners,
and (iii) carried by enough of the pool to move the aggregate.

**`docs/EVAL_POOL.md` does not currently guarantee them.** It guarantees privacy,
freshness and breadth for *values* — (i) and (ii) by construction, once
covariates exist and are harvested the same way. It says nothing about
covariates, because there are none. (iii) it cannot guarantee at all, and (iii)
is quantitative: under DEC-CA-0009 the round metric is a per-window geomean, so a
fraction `f` of covariate-bearing windows with a within-slice gain `g` moves the
aggregate by roughly `f × g`. Against the flat 0.02 margin, a 20% within-slice
effect needs **`f ≥ 10%` of the pool** to be decidable — and the LCB has to clear
the margin, not the point estimate, so 10% is a floor. The missing guarantee is
a pool-composition rule, not a code change, and it is the same arithmetic that
governs whether *any* minority capability on this roadmap is worth competing for.

## Open questions

1. **Bucket occupancy under `(C, P)`.** Replay an archived corpus stream with a
   synthetic channel-count distribution and measure how many partial buckets
   flush at stream end. If the tail is material, the batcher needs a flush
   policy, not more channels.
2. **Does the variate layer learn anything at small `C`?** One layer in four is
   variate (`num_layers = 4` ⇒ exactly layer 3). Train the king's generator at
   `C = 1` and at `C = 4` with genuinely coupled variates under one `RoundSeeds`
   and compare per-channel scores. If coupling buys nothing at 4M, the whole
   multivariate workstream is premature at this model size and should wait for a
   larger fixed model — which would have to be costed explicitly.
