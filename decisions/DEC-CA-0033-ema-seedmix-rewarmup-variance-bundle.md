---
id: DEC-CA-0033
type: decision
title: "The measured variance bundle: EMA finished-form checkpoints, N-seed generation mix, warm-started re-warmup — shipped inert behind drop-when-default"
status: proposed
date: 2026-08-26
tags: [cascade, training-recipe, seed-variance, ema, lr-schedule, wsd, contract-digest, fairness]
revisit_when: "A testnet cascade runs a full round cycle with ema_decay (and optionally gen_seed_mix / rewarmup_fraction) armed — the gate for the mainnet cut; or DEC-CA-0029's fork-anneal is chosen for the finished-form role instead (then ema_decay stays 0 forever — never arm both); or the 22M size seam arms (re-measure decay=0.999 at size — it was calibrated on ~48k-step heat runs at 4M)"
relations: {depends_on: DEC-CA-0018, informs: DEC-CA-0017, alternative_to: DEC-CA-0029}
---
Measured 2026-08-26 on a rented 4×4090 pod: 3 r41 finalist generators × 4
seed combos × heat-length runs, all scored on the round's exact heat slice
(bit-deterministic eval; full grid + method in
docs/notes/2026-08-26-seed-variance-ema.md). Findings that force the design:

- **Generation seed is the heat's dominant noise term**: a single corpus
  redraw moves an entrant 0.0046–0.0111 geomean — 5–20× the real skill gaps
  between finalists (0.0002–0.0009) and up to ~1.3× the dethrone margin.
  Entrant-specific, i.e. the unfair term. Training seed (0.0011–0.0061) is
  common-mode within a round; the fixed eval slice contributes exactly 0.
- **EMA-0.999 of the weights fixes most of it**: shrinks the gen-seed term
  4–11× (to ~the real-gap scale) AND scores ~7% better absolutely, in 12 of
  12 runs — the endpoint is a noisy snapshot; the average sits in the basin.
- **The first optimizer step after a warm start is destructive**: fresh
  optimizer state (Adam/Muon step-1 ≈ full LR per coordinate) + wsd's
  skipped warmup costs +0.11 geomean at base_lr (the u158 no-op probe,
  reproduced in-run); at ¼ LR the kick is +0.045. No run at any LR
  recovered to the untouched init's level within a heat.

DECISION — three [training] knobs, one bundle, all inert at their defaults
via the drop-when-default convention (deployed digests untouched until an
operator arms them; arming is the deliberate contract cut —
release-then-activate, trainer + all validators together, testnet first):

1. `ema_decay` (0.0 = off; measured value 0.999) — the trainer maintains a
   per-step EMA of the weights and ships it as `weights.safetensors` (the
   artifact every scoring layer loads); the raw endpoint rides beside it as
   `weights_stable.safetensors`, reusing DEC-CA-0029's lineage-branch file
   convention so the warm-start loader needs no new case. EMA is the CHEAP
   alternative to fork-anneal for finished form (zero extra tokens, zero
   wall stretch) — never arm both; the trainer refuses the combination.
2. `gen_seed_mix` (1 = off; measured design 3) — the generator is invoked N
   times with N derived seeds (`_mix(generation_seed, "seed-mix", i)`),
   each ~1/N of the token budget, series interleaved round-robin; the
   corpus digest becomes the rolling digest of the interleaved sequence and
   the audit re-derivation replays the same rule. Residual seed noise drops
   ~√N. Miner interface unchanged (generators are already required to be
   pure functions of the passed seed). Mostly redundant on top of EMA —
   adoption order is EMA first, mix optional second.
3. `rewarmup_fraction` (0.0 = off) — warm-started wsd runs get a short
   linear LR ramp over this fraction of train_tokens instead of full
   base_lr against fresh optimizer state on step 1. From-scratch runs keep
   warmup-once semantics untouched. MEASURED 2026-08-26 (paired A/B,
   4×L40S): re-warmup does NOT improve end-of-run quality — slightly worse
   at full LR, a wash at ¼-LR; the first-step kick is a transient that
   constant-LR training re-perturbs past anyway. The knob ships inert and
   should STAY unarmed on current evidence; its only value is protecting
   very-early-truncated runs. The measured recipe that matters is
   base_lr→0.001 + ema_decay=0.999 (EMA-999 0.2114–0.2119 vs current
   0.2188, init 0.2086, r41 slice).

Deployment coupling: `tie_runoff_windows` stays 0 regardless — DEC-CA-0019's
jittered EVAL draw already forces it (an eval-window property; the corpus
seed mix itself does not touch the run-off's prefix assumption). EMA also
excludes `[telemetry] bench_anneal_fraction` (DEC-CA-0030) — the worker's
`anneal_recipe` refuses an EMA-armed contract; one finished-form mechanism
at a time, everywhere. EMA changes which artifact external
validators score, so score reuse across the activation boundary is invalid
for one round. The `base_lr` level itself (¼-LR dominated at every measured
checkpoint) is deliberately NOT part of this node — an LR retune under µP
transfer is its own contract decision.
