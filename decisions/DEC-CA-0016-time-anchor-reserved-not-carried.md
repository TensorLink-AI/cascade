---
id: DEC-CA-0016
type: decision
title: "Time anchor and frequency: reserve the names, refuse the payload — the fixed model has nowhere to put a calendar and Toto2 is timestamp-free by design"
status: proposed
date: 2026-08-13
tags: [interface, generator, model-arch, seasonality, eval, gift-eval]
revisit_when: "a shadow experiment (below) shows a calendar-feature input measurably improves the private-pool geomean on Toto2-4M under one round's seeds, OR the fixed model is replaced by one that already takes frequency/timestamp inputs — either makes the payload consumable and the arch-digest cost worth paying"
relations: {depends_on: DEC-CA-0015}
---
A generator cannot declare "hourly, starting Monday, with weekday seasonality",
while `eval/scoring.py::_resolve_seasonal_period` reads `seasonal_period`/`freq`
straight out of window metadata for MASE. The asymmetry is real. The conclusion
drawn from it — that this is the largest hole and close to free — is not.

**There is nowhere to put it.** `Toto2Model.patch_embed` is
`Linear(patch_size * 2, d_model)`: values ‖ mask channel, nothing else
(`toto2_model.py:327,371`). There is no covariate embedding, no frequency
embedding, no calendar path anywhere in the module. Adding one changes
`toto2_model.py`'s bytes, and `compute_base_arch_digest` hashes those bytes
alongside `_ARCH_FIELDS` (`trainer/contract.py:52-71`) — so it bumps
`base_arch_digest`, which is a `[training]` field, which bumps
`contract_digest`, which means a lockstep trainer+validator restart and an arch
mismatch against every archived checkpoint. That is the most expensive class of
change this repo has, and it is the *cheapest* way to make `(start, freq)` do
anything at all.

**Toto 2.0 is deliberately timestamp-free.** So is Chronos-2 for the target
series. Frequency embeddings are a Moirai/TimesFM design choice. `chain.toml`
pins `base_arch_digest` against the released `Datadog/Toto-2.0-4m`
`config.json`; adding calendar features is a decision to *diverge from the
pinned reference model*, which is a model-design question, not a
submission-surface one, and it should be argued as such.

**The eval already knows the frequency, and the model does not need to.** MASE's
seasonal denominator is a property of the *eval* series, computed from pool
metadata the builder stamps (`pool/builder.py:184-188`). A miner is not blocked
from competing on frequency-dependent behaviour: a generator expresses "hourly
with weekday seasonality" today by emitting period-24 and period-168 structure,
which is precisely what a frequency-agnostic model learns from. What the miner
cannot do is *label* it — and no consumer of that label exists.

**DECISION.** `start` and `freq` are reserved names in the DEC-CA-0015
namespace and are **not accepted**. Carrying them now would violate this
roadmap's own rule against unconsumed payload: untested, budget-costing, and a
free side channel.

**The one thing worth building meanwhile, at zero arch cost:** treat a *claimed*
frequency mix as **corpus-composition telemetry** — publish, per generator per
round, the frequency/seasonality profile its corpus actually exhibits (estimated
from the emitted series, not declared), next to the pool's profile. That is the
DEC-CA-0006 / DEC-CA-0010 idiom: compute the diagnostic, record it, do not obey
it. It answers "does the field's corpus cover the frequencies the pool scores?"
without a contract change, and it is what would make the declared-`freq` case
arguable later on evidence.

**Open question — the experiment that settles the payload.** Under one round's
`RoundSeeds`, train the reigning king's generator twice: once under the pinned
arch, once under an arch whose patch embedding also takes sin/cos calendar
features derived from a declared `(start, freq)`. Score both on the same private
slice. If the gap sits inside round-to-round noise, gap 1's payload is closed
permanently and the reservation can be retired. This is a shadow run
(DEC-CA-0014 Stage-1 shape), not a consensus change; it costs one lane.

**Not settled here:** whether a diverged arch could still be called
"Toto2-4M from scratch" for the purposes of the subnet's public claim. That is
an owner/positioning call, and it should be made before any calendar work, not
after.
