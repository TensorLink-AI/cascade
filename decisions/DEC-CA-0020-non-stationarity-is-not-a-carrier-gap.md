---
id: DEC-CA-0020
type: decision
title: "Non-stationarity is not a submission-surface gap — drift is already expressible; the open work is entirely pool-side"
status: proposed
date: 2026-08-13
tags: [generator, drift, eval-pool, diagnostics, non-stationarity]
revisit_when: "a drift bucket exists in pool metadata and the per-domain diagnostics show challengers that win overall losing on it — that is a real separable signal and would justify asking whether the corpus needs a way to declare drift rather than merely exhibit it"
relations: {depends_on: DEC-CA-0015, related: DEC-CA-0016}
---
The claim under review: with no time anchor there is no way to express concept
drift, and this largely falls out of gap 1 (DEC-CA-0016). **It does not fall out
of gap 1, and it needs nothing from the carrier at all.**

**Within-series drift is expressible today.** A generator changes its generating
parameters partway through a series — trend break, variance regime, seasonality
amplitude, changed AR coefficients — with a bare `np.ndarray` and no anchor.
Nothing in `check_series` or `drain_generator` objects. The model has no
absolute-time input to anchor to in any case (DEC-CA-0016), so a `start` would
be inert here even if it were carried.

**Corpus-level drift has no consumer.** The corpus is an unordered list; the
trainer buckets it by patch count and (post-DEC-CA-0021) by channel count, and
draws batches in stream order. "The corpus drifts over its length" is a property
nothing reads.

**The one version that needs anything is synchronised cross-series drift** —
"these 500 series all shift at the same instant" — which needs `group_id` +
`start`, and is unusable without the variate axis. But once the members sit
inside one variate axis it is again just a `(K, L)` series with a shared change
point, which is expressible today. So even the hard version reduces to
DEC-CA-0021's work plus nothing.

**DECISION: close gap 5 at the carrier and payload level.** No reserved name
beyond what DEC-CA-0016 already reserves, no trainer change, no contract churn.

**Where the actual work is: the eval.** The Impermanent premise and the Datadog
bottleneck are both statements about *held-out data containing drift*, not about
generators being able to declare it. The private pool's freshness lever
(`as_of`, daily rotation) gives it recent data, which is not the same thing as
drifting data — and the builder's cleaning path (interpolation, tail truncation
to `context_length + horizon`, degeneracy filters) has never been examined for
what it does to change points.

**Open question — the cheap instrument.** Compute a change-point statistic at
pool-build time (a mean/variance shift score over the kept window is enough) and
stamp a coarse `drift` bucket into `metadata.json` alongside `domain`. It then
rides the diagnostic that already exists: `RoundResult.per_domain_win_rate`
(`eval/koth.py:194-199`) is keyed on a metadata bucket and is already logged as
the "stop aggregating" tripwire. Read it on the drift bucket for a few rounds.

Two outcomes, both useful. If challengers that win overall systematically lose
on the drift bucket, there is a real separable capability the field is not
competing on, and *that* is the evidence for doing something structural. If the
drift bucket is thin or indistinguishable, the honest conclusion is that the
private pool does not contain the phenomenon the benchmark literature is
measuring — which is a statement about pool composition, and the fix is a
source, not an interface.
