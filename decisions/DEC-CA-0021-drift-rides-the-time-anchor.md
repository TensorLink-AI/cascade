---
id: DEC-CA-0021
type: decision
title: "Corpus-level non-stationarity needs nothing beyond the time anchor — and the model cannot perceive it anyway; yield order is the one drift lever miners already hold"
status: proposed
date: 2026-08-13
tags: [interface, generator, trainer]
revisit_when: "an arch generation consumes absolute time (then cross-series drift becomes perceivable and start's payload decision reopens); or the training recipe adds any corpus-order-dependent element beyond SGD ordering (curriculum, annealed mixing), which would turn the yield-order lever below into a scored surface needing its own rules"
relations: {depends_on: DEC-CA-0017}
---

Gap 5: with no time anchor there is no way to express concept drift — the
Impermanent-benchmark premise. The question: does this need anything beyond
gap 1?

**Confirmed: nothing further — and less than that.** Three findings:

1. **Within-series drift is expressible today.** Regime shifts, trend
   breaks, and variance changes inside a 4096-step window are just shapes in
   `values`; competitive generators already emit them. The eval rewards them
   to exactly the degree real held-out windows contain them.

2. **Cross-series drift is imperceptible to the fixed model, anchor or not.**
   Batch elements are independent; positions are relative (xPos); there is
   no absolute-time input (DEC-CA-0017). A corpus whose series are stamped
   `start` values spanning a decade of simulated drift trains identically to
   the same series shuffled — the model has no mechanism that could tell.
   So gap 5's payload is inert for the same structural reason as gap 1's,
   and `start` stays reserved (one field, one reservation, shared).

3. **The one real drift lever already exists and is undocumented: yield
   order.** In the live `stream_cpu` mode the trainer consumes series in
   yield order with no shuffle beyond length-bucketing
   (`_FreshSeriesStream.series()` → `iter_training_batches`,
   `toto2_trainer.py:107`), under a cosine LR schedule. A miner can already
   order easy→hard, stationary→shifted, or anneal the mixture over the
   stream — a curriculum, expressed with zero interface change. This is a
   live, miner-controlled training-dynamics surface today. It is judged
   acceptable (it is part of "the data", the thing being competed; both
   duel sides hold the same lever), but it should be *named* in
   `docs/INTERFACE.md` rather than left as folklore, because it is the
   honest answer to "how do I express non-stationarity to this trainer" —
   order, not timestamps.

## Decision

No changes of its own: no `chain.toml`, no interface beyond DEC-CA-0017's
reservation, no trainer, no eval, no digest movement, no migration, no new
gaming surface (yield-order control is pre-existing, symmetric, and already
priced by the wall — a pathological order costs the miner's own throughput).
Document the yield-order lever. Close the gap into DEC-CA-0017.
