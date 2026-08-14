---
id: DEC-CA-0016
type: decision
title: "Warm-start recipe: WSD learning-rate schedule (warmup once per generation, flat across rounds) + optimizer-state continuity in the checkpoint"
status: active
date: 2026-08-14
tags: [cascade, warm-start, training-recipe, lr-schedule, optimizer, contract-digest]
revisit_when: "A release-checkpoint mechanism exists (then build the deferred decay branch: short cosine/linear anneal off the flat lineage when promotion cuts a generation, published beside — never instead of — the undecayed line); or shadow-scratch telemetry (DEC-CA-0014 Stage 1) shows flat-LR lineages plateauing where a decayed comparison keeps improving; or checkpoint upload size becomes a real cost (then gate the optimizer sidecar to final runs only)"
relations: {depends_on: DEC-CA-0013, informs: DEC-CA-0014}
---
Warm-start ([[DEC-CA-0005]]/[[DEC-CA-0013]]) turned rounds into continued
pretraining of a compounding lineage, but the recipe was still the
from-scratch one, per round: `warmup_cosine` re-warmed and re-decayed the LR
to zero EVERY round, and `warm_start_dir` loaded weights only, so Muon
momentum and the NorMuon row-EMA reset at each round boundary. Repeated
cosine restarts are the known-wrong shape for continued pretraining — each
round spends its start re-warming and its end at ~zero LR, and the restart
transient distorts what the lineage compounds.

DECISION — `lr_schedule = "wsd"` (warmup-stable-decay), flipped in both
`chain.toml` and `chain.testnet.toml` on 2026-08-14:

* **Warmup once per generation.** The from-scratch (generation-start) run
  does the linear warmup over `warmup_tokens`; every warm-started round is a
  continuation and re-enters FLAT at `base_lr` — no re-warmup, no restart.
  "Is this run warm-started" is the signal that keys it, which is shared
  king/challenger state (both roles get the identical init, DEC-CA-0004), so
  the controlled-experiment terms stay byte-identical.
* **No in-round decay.** Decay belongs to cutting a *release* checkpoint,
  which cascade does not have yet — the round loop must not fake one. Every
  in-lineage checkpoint (dueled, promoted, warm-started from) is an undecayed
  flat-LR checkpoint; comparisons stay paired, so the missing anneal biases
  no verdict.
* **Optimizer state rides the checkpoint.** wsd rounds write
  `optimizer.safetensors` beside the weights (Muon momentum + row-EMA + AdamW
  moments *and step count*, name-keyed, snapshot copies — never aliases of
  live buffers), and a warm-started run re-attaches it. ~3× checkpoint size,
  accepted. A member without the file starts fresh state loudly-logged
  (momentum rebuilds in ~hundreds of steps against 2k–150k steps/round — the
  sanctioned crossing for members promoted before this shipped); a PRESENT
  file with mismatched shapes aborts the run, like the strict weights load.
  Saving is gated on `lr_schedule = "wsd"` so the 3× cost and the digest flip
  travel as one deploy event; an unknown `lr_schedule` string aborts instead
  of silently falling back to cosine.

Consequences held by the design: the checkpoint stays deterministic (a
resumed run re-derives byte-identically, `optimizer.safetensors` included, so
`cascade-audit` Tier-2's per-file digest comparison still passes); the
validator's wrapper ignores the extra file; no wire/receipt change. The flip
IS a `[training]` edit: `contract_digest` changed, trainer and validator
restart together (operational invariant), and auditing a pre-flip round
requires the pre-flip config — inherent to every `[training]` edit.
