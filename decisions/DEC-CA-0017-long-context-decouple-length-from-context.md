---
id: DEC-CA-0017
type: decision
title: "Long context splits in two: decouple max_length from context_length and SAMPLE windows (cheap, do it); raising context_length is gated on eval CPU and pool supply (defer)"
status: proposed
date: 2026-08-13
tags: [generator, training, eval, heat, budget, long-context, pool]
revisit_when: "the pool build reports that ≥50% of windows carry unpadded real context at the target length AND the heat screen has moved off the orchestrator CPU (or heat_n_windows has been cut back) — the two constraints that make 2b measurable and affordable; the FLOP cost alone was never the blocker"
relations: {depends_on: DEC-CA-0015, refines: DEC-CA-0001}
---
`[generator] max_length = 4096` carries the comment "keep in lockstep with
`[training] context_length`". The lockstep is real but the reason is not
architectural — it is that a longer series is currently **charged for and thrown
away**. Two separable changes hide behind one number.

## 2a — decouple the cap and sample windows. Do this.

`iter_training_batches` keeps only the last `P = min(L // patch_size,
max_ctx_patches)` patches of every series (`toto2_trainer.py:107-133`,
`s[-p * patch_size:]`). The prefix is discarded. Meanwhile the live
`stream_cpu` budget counter charges **every emitted value** (`arr.size`,
`stream.py:186-189`). So a generator emitting 3× `context_length` today pays 3×
the budget and trains on the last third; the stream exhausts before the trainer
reaches `train_tokens`, and the run finishes under compute — visible as
`tokens_frac` / `deadline_hit`, and a lost round.

Fix: raise `[generator] max_length` freely and make the batcher **sample** a
context window from a long series (seeded from `training_seed`, so byte-exact
re-derivation is preserved) instead of taking the tail. A long-range prior then
expresses structure the model can be trained on at slices, while attention is
still only ever paid over `context_length`. No arch change, no eval change, no
`base_arch_digest` change, no miner migration.

**One trap.** `base_arch_digest` hashes `toto2_model.py` only; the batcher lives
in `toto2_trainer.py`, which nothing hashes. Changing the sampling rule silently
would alter the training recipe with **no gate on it at all** — the exact
failure mode `docs/VALIDATOR.md` warns about for scoring-rule changes ("nothing
detects it for you"). So the rule must be named in `[training]` (e.g.
`window_sampling = "tail" | "random"`), which does bump `contract_digest` and
does force the lockstep restart. Pay the digest deliberately rather than change
the recipe invisibly.

## 2b — raising `context_length`. Defer; it is gated, and not by FLOPs.

**Training cost is cheap.** At `d_model = 256`, `patch_size = 32`, the 4M
backbone is ~3.25M params ⇒ ~19.5 MFLOP per patch-token of training, against
`3 × 3072 P²` FLOPs of time-attention per row (3 of 4 layers are time-axis;
layer 3 is variate and runs at `C = 1`). Attention is therefore ≈ `4.7e-4 × P`
of the dense cost: **6% at P = 128 (ctx 4096), 12% at 8192, 24% at 16384**. At a
fixed token budget that is **+5.7% wall-clock at 8192 and +17% at 16384**.
Nothing in the 3h budget breaks.

**Eval cost is the binding constraint.** Per-window CPU forward scales with
`P + 4.7e-4 P²` — ≈5× per window at 4× context. `chain.toml` records the heat
screen at ~34 s/ckpt over 2000 windows, ~17 min/wave, and explicitly relies on
that fitting "inside the 1h wave-training shadow". At 16384 that becomes
~170 s/ckpt and ~85 min/wave — past the shadow, and the heat becomes the round's
critical path. DEC-CA-0012's 106 s paired final eval (the tie-break re-score
that makes the cohort duel affordable) becomes ~9 min, which changes that
decision's arithmetic.

**Pool supply is the second constraint.** `PoolBuildConfig` keeps
`context_length + horizon` with `span_days = 210` (≈5040 hourly points — enough
for 4096, not for 16384, which needs ≈700 days and ~3.3× the Open-Meteo
payload). Daily sources (Wikimedia's ~85 articles; most tsbench-forge feeds)
can never fill even 4096. `Wrapper._prep` left-pads short histories with the
first value, so on those windows the added context is a **constant pad** — a
capability miners would be asked to compete on and the eval could not see.

**Digest cost.** `context_length` is in `_ARCH_FIELDS`, so raising it bumps
`base_arch_digest` *and* `contract_digest`, mismatches every archived
checkpoint's arch, and (via `Toto2Config.max_patches`) changes the decode
window. This is the most expensive change on the roadmap.

**DECISION.** Ship 2a. Defer 2b behind two measurements, not behind an opinion.

**Open questions — the measurements that settle 2b.**
1. What fraction of the current pool's windows carry ≥4096 *unpadded* real
   context, and what would that be at 8192? Instrument `pool/builder.py` to emit
   the unpadded-context histogram into `provenance.json` on the next few builds.
   Below ~50% at the target length, 2b is unmeasurable and must not ship.
2. What does the heat wave actually cost at 8192 on the orchestrator's CPU?
   Time one archived checkpoint over 2000 windows at both geometries. If the
   wave exceeds the training shadow, 2b needs the heat moved to GPU
   (`--eval-hosts` already exists validator-side) *before* it needs a context
   raise.
