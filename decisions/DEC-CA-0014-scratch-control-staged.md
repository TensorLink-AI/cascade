---
id: DEC-CA-0014
type: decision
title: "From-scratch signal survives the warm-start era: shadow control first, reseed valve second, a second throne only on divergent-winners evidence"
status: active
date: 2026-08-10
tags: [cascade, warm-start, koth, control, incentives, staging]
revisit_when: "Stage-1 shadow data exists for ≥2 promotion generations: if the scratch-vs-lineage gap is closing (or closed), arm the Stage-2 reseed valve and re-examine the lineage; if scratch rounds and warm rounds would have crowned DIFFERENT generators repeatedly, open the Stage-3 two-throne design; if the gap compounds steadily and winners coincide, stay at Stage 1 and re-check a generation later"
relations: {depends_on: DEC-CA-0013}
---
Warm-start ([[DEC-CA-0013]]) deliberately weakens the contract's founding
invariant — "from scratch, the corpus is the only source of learned signal" —
because a fine-tune of a mature init is less sensitive to data quality than a
pretrain from nothing. Once armed, the subnet loses its only baseline for "how
much is the lineage buying?" and its only escape from a benchmark-overfit
basin. The full remedy discussed (interleave random-init rounds and warm-start
rounds as TWO SEPARATE THRONES with split emission) is a large consensus
change bought on a hypothesis. DECISION: stage it, measurement first
(the [[DEC-CA-0006]]/[[DEC-CA-0010]] pattern — measure, don't mechanize).

**Stage 1 — shadow scratch control (BUILT 2026-08-15; no consensus change).**
Every M rounds (start M=4) the trainer additionally trains a FROM-SCRATCH
model on the reigning king's generator under the identical contract, benches
it on the three suites, and publishes the numbers as telemetry beside the
bench report. No throne, no weights, no envelope change; cost ~1/M of one
lane. Output is the two curves every later stage depends on:
lineage-vs-scratch gap, and its trend (compounding / flat / closing).

Stage-1 implementation (branch `claude/margin-decay-dec-ca-0014-qorsop`):
`[telemetry] scratch_shadow_every_rounds` (M; 0 = off — deliberately not
`[training]`, so arming never touches `contract_digest`); the leg runs
strictly post-publish on the king's final pod inside the existing bench hold
(~5h at the mainnet budget, inside the 6h `bench_hold_max_hours`), from
random init at the byte-identical token budget, and publishes a signed,
in-band-labeled `benchmarks/scratch/round-<id>.json` + trend index
(`cascade.shared.scratch_report` — a separate artifact, NOT a `BenchReport`
entry, because validators parse that schema on the promotion-provenance
path). Structurally unable to feed scoring/selection/promotion: never a
manifest entry, never handed to `promotion.record_bench`. The remote leg
drives the PINNED worker image's existing CLI, so it ships trainer-side
unilaterally (no image re-pin). STATUS: testnet armed at M=2
(`chain.testnet.toml`) for the one-cycle validation in
`docs/SCRATCH_SHADOW.md`; mainnet stays 0 until that passes and the owner
arms M=4.

**Stage 2 — cross-band reseed valve (arm when the gap closes).** Scratch
checkpoints enter the promotion candidate pool through the EXISTING quality
floor: normally a 150k-step scratch model benches far below an N-generation
lineage and `cascade_quality_epsilon` excludes it automatically — but the
moment it clears the floor, the lineage has stopped compounding, and
promotion reseeds from scratch exactly when reseeding is warranted. No
"sick lineage" heuristic; the benches define it. This requires a deliberate,
documented exception to DEC-CA-0013's generation band (select within the
highest generation, EXCEPT a scratch candidate that clears the mature
floor) — the band exists to keep immature checkpoints out, and the valve is
the one crossing that means something.

**Stage 3 — second throne (only on divergent winners).** Alternate/asymmetric
rounds judged as a separate random-init KOTH with its own emission share is
justified ONLY if Stage-1 shadows show the two regimes reward different
generators — that is the sole outcome where a second reward stream pays for
its machinery (duplicate ChampionState/margins/court, throne-tagged receipts,
split weight vectors, per-throne round counting for every rounds-denominated
knob). Same generator would serve both thrones (one commit; a round's throne
derived from the epoch grid), preserving dedup/burn rules unchanged.

Why not jump straight to two thrones: emission splits weaken each
competition's security margin; per-throne cadence dilution silently retunes
every live-calibrated rounds-denominated parameter; and if data quality
correlates across regimes the same miner wins both — paying the full
machinery cost for the same payout distribution. The shadow buys the
distinguishing evidence at ~1/M of one lane.
