---
id: DEC-CA-0018
type: decision
title: "The length cap does not foreclose long-range priors — long CONTEXT is deferred to the 22M size seam and costed there"
status: proposed
date: 2026-08-13
tags: [generator, training, eval, config, provision]
revisit_when: "the 22M size activates (bundle context growth with the recalibration that seam already forces); or measured L40S throughput at context 8192 contradicts the ~10% per-token estimate below by enough to change the trade; or the eval-pool sources can demonstrably supply 8192+horizon fresh points across enough domains that the pool-composition shift stops being a cost"
relations: {depends_on: DEC-CA-0016, constrained_by: DEC-CA-0002}
---

Gap 2 as framed: `max_length = 4096` forecloses competition on long context,
where the field's headroom sits (Timer-S1 11.5K, Chronos-2 8192, TiRex-2
streaming). **Half right.** The cap forecloses long-*context* competition —
which is a `[training]`/arch change, not a submission-surface change — and it
does NOT foreclose long-*range* priors, which are miner-expressible today.

## Two corrections from the code

1. **`max_length` is not digest-bound.** It lives in `[generator]`
   (`chain.toml:17`), which is not part of `TrainingContractConfig`
   (`shared/config.py:163`) and so not in `contract_digest`. The lockstep
   with `context_length` is a convention, not a coupling the digest enforces.
   The expensive, digest-bound half of this gap is `context_length` alone
   (`[training]` + `[eval]`, plus `base_arch_digest` via `max_patches`).

2. **The trainer already crops.** `iter_training_batches` keeps the *last*
   `context_length` points of any longer series
   (`trainer/toto2_trainer.py:127`, `s[-p * patch_size:]`). Raising
   `max_length` alone would work mechanically today — and buy almost
   nothing, because everything before the final window is generated, billed
   against the stream budget, and discarded.

## Why long-range structure is not actually blocked

A model with a 4096-step context cannot condition on structure longer than
4096 steps, period — no submission-surface change alters that. What a long
series would let a miner express is a *long consistent trajectory sampled
into training windows*: regime evolution, slow trends, low-frequency cycles,
all internally coherent. But that is computable inside the generator today:
simulate the 100K-step trajectory internally, yield its 4096-step crops.
Determinism holds (same seed, same crops), the budget prices only the emitted
points, and no interface changes. The "long series permitted but sampled into
shorter training windows" variant the question asks about therefore already
exists — with the sampling done by the party that owns the trajectory, which
is also the right place for it (the trainer sampling crops would be a
digest-blind semantics change to what a yielded series *means*, for zero
capability the miner doesn't already have). The residual value of raising
`max_length` alone is ergonomics, and it is declined to keep "one yield = one
training series" exact.

## Costing the real change: `context_length` 4096 → 8192

- **Per-token compute, estimated ~+10%.** At `d_model 256`, `d_ff 688`,
  patches P = 128: the sequence-dependent attention term (`2·P·inner`) is
  ~65K of ~680K mults per position (~10%); at P = 256 it doubles while
  everything else holds, so total per-token cost rises roughly 10%. On the
  fixed ~40B point-pass budget that is ~10% more wall-clock — inside the 3h
  wall only if `ref_throughput` is recalibrated, which is a
  `contract_digest` bump by itself. **Estimate, not measurement**; the
  settling experiment is one instrumented run on the reference L40S at both
  contexts.
- **Diversity halves at fixed budget.** The budget is point-passes; series
  twice as long means half as many distinct series seen per run. Long
  context is paid for in breadth — a real training-mix cost the throughput
  number doesn't show.
- **Eval doubles per window.** `[eval] context_length` must move in lockstep
  (`chain.toml:285`), so every heat and duel forward pass is ~2× — the 2000
  window heat screen (~34s/ckpt today) and the ~106s duel eval both roughly
  double. Tolerable, but it lands on the CPU path DEC-CA-0012 budgeted.
- **The pool must supply 8192 + 64 fresh points per window, and mostly
  cannot.** The builder truncates to the freshest `context+horizon`
  (`docs/EVAL_POOL.md`); at hourly that is ~344 days (Open-Meteo: fine), at
  daily it is ~22 years of fresh data (Wikimedia and most tsbench-forge
  daily feeds: impossible). Long-context eval therefore *shifts pool
  composition toward sub-daily domains* — a scoring-mix change that would
  need its own DEC, or short-history windows left-padded (the wrapper
  already pads, `_prep` in `toto2_trainer.py`), which evaluates long context
  without exercising it.
- **Digest/arch:** `context_length` is in `contract_digest`; `max_patches`
  derives from it (`toto2_model.py:103`) so `base_arch_digest` recomputes;
  trainer + validators restart in lockstep — the routine but heavyweight
  re-pin protocol.

## Decision

Defer the `context_length` raise to the **22M size activation** — that seam
already forces a throughput re-measurement, an arch-digest recompute, and a
coordinated restart (`chain.toml:124` block comment), so context growth rides
a migration that is happening anyway, on a model with enough capacity to
plausibly use it. `max_length` stays in lockstep at 4096 until then: raising
it alone buys ergonomics for real cost in "what does a yield mean"
ambiguity. No carrier work is needed for this gap at all — length is already
a per-series property of `values`.

Migration story when it lands: deployed generators (all `<= 4096`) remain
valid — the band widens, never narrows. Gaming surface: none new; a longer
series is just more billed points.

## Rank

Against the proposed ordering (#2 of six): **demoted to last**. The
submission surface is not the binding constraint; the model contract is, and
that contract's next scheduled move (22M) is the natural carrier for it.
