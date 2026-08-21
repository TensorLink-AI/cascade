---
id: DEC-CA-0029
type: decision
title: "GP-prior generation economics: the materialised corpus budget is denominated in POINTS (count free), the batch drain budget rises 1800 → 7200 CPU-s with the streaming stall window split off and pinned, and the relative-jitter Cholesky is documented miner-side"
status: accepted
date: 2026-08-21
tags: [generator, budget, gp-priors, sandbox, corpus, economics, docs]
revisit_when: "a GP/kernel-prior submission actually enters a round (validate the budgets against a live entry, not our benchmarks); or corpus_mode flips off stream_cpu (cache_reuse would make corpus_target_points the LIVE denomination and stream_gpu would moot it — both are digest-bound flips with their own coordination, see below); or interface_version 2 work starts (the calendar-field rejection is the remaining floor under TempoPFN-class generators regardless of any time budget)"
relations: {refines: DEC-CA-0001, enables: [], depends_on: []}
---

`chain.toml [dependencies]` has shipped `gpytorch`/`scikit-learn`/`networkx`
("GP/kernel/graph primitives for GP-prior generators") since launch, and torch
is annotated as a compute library for exactly this class of prior. But the
generation budgets priced them out: 1800 CPU-seconds over a fixed
16384-series drain is 109.9 ms/series, against measured costs at L=4096 of
12.76 s/series (KernelSynth, 116× over), 5.47 s (CauKer-univariate, 50×),
1.42 s (plain GP, 13×) — the exact priors our E9 ablation showed carry the
entire −11.77%. The priors were *permitted* but not *expressible*.

## The decision (three changes, one node)

1. **Points, not series** — `[generator] corpus_target_points` (armed at
   67,108,864 = 16384 × 4096, today's corpus size). When set, the
   materialised drain (`cache_reuse`, `cascade verify`) stops at the first
   series that reaches the target points; the series count is free inside
   the existing `[min_length, max_length]` band, and the drain requests the
   same prefix upper bound the stream uses (`target // min_length + 2`).
   GP draw cost is ~cubic in length, so a fixed count pinned those priors
   into their worst corner: at L=1024 a KernelSynth series costs ~64× less,
   and the same corpus-in-points lands near the time budget instead of two
   orders of magnitude over. 0 = the legacy exactly-`corpus_n_series` drain,
   byte-identical. The dedup probe explicitly zeroes the target (it compares
   small fixed-count draws). The streaming feeds were ALREADY
   points-denominated (they stop at the training token budget — roadmap
   Part D), so this aligns the materialised path with the live one.

2. **Batch drain budget 1800 → 7200 CPU-s, with the stall window split
   off.** `max_generate_seconds` had two jobs: the batch sandbox's
   RLIMIT_CPU + wall budget AND the streaming per-frame stall window (how
   long a silent generator holds a lane before being killed — the reason
   testnet had already cut it to 600). Raising one must not widen the other,
   so the stall window is now its own knob, `[generator]
   stream_stall_seconds` (0 = fall back to `max_generate_seconds`, the
   pre-split behaviour), pinned at the old values: mainnet 1800, testnet
   600. All streaming consumers read it through
   `GeneratorConfig.effective_stall_seconds`. At 7200 the plain-GP prior
   fits the full drain at L=4096 once change 3 lands, and the heavier priors
   fit at reduced length under change 1. RLIMIT_CPU still sums across
   threads and `multiprocessing` stays blocked — multi-core BLAS burns the
   budget faster, which is now documented instead of discovered.

3. **The factorisation is documented, not discovered** (docs/MINER.md §1a):
   relative-jitter Cholesky — diagonal nudge scaled to the kernel's mean
   diagonal, escalating ×10 per retry — replaces the Cholesky→SVD fallback
   pattern. Measured: ~5× per GP draw, SVD fallbacks 47 → 0 per corpus.
   Pure CPU-side algorithm choice, deterministic, no contract implication.

## Why this is operational, not consensus

No `[generator]` key is in `contract_digest` (verified: the digest is
computed over `TrainingContractConfig` only — `trainer/main.py`,
`shared/manifest.py`; the roadmap's Part D states the same corrected
premise). No coordinated trainer+validator restart, no manifest
invalidation, no receipt-format change. Audit re-derivation reads the same
`chain.toml [generator]` the trainer does, and at the shipped defaults every
historical replay is byte-identical (target=0 paths untouched; the armed
target only governs future materialised draws, and the live `stream_cpu`
digests never went through the materialised drain).

## What was considered and NOT done

- **`corpus_mode = "stream_gpu"`** dissolves the CPU economics outright
  (measured 451× on KernelSynth, 84× on CauKer) and the machinery exists —
  but `corpus_mode` IS in `contract_digest`, so the flip needs a coordinated
  trainer+validator restart at an epoch boundary and drops the audit from
  byte-exact to tolerance/same-hardware. That is its own decision with its
  own rollout; nothing here forecloses it.
- **Loosening the training wall** for slow generators — forbidden by
  DEC-CA-0001 ("wall is the law"): in the live stream feed, generator
  throughput remains a compute multiplier and mass `deadline_hit`s remain
  intentional. This node cheapens the *per-point cost* of GP priors so they
  can compete under that law; it does not bend the law.
- **`interface_version` 2 (calendar fields)** — `start`/`freq` stay
  hard-rejected (DEC-CA-0021: no consumer can exist under the calendar-free
  arch pin). Noted honestly: this keeps TempoPFN-class calendar augmentation
  inexpressible regardless of any time budget, which is why the −11.77% is a
  floor; parked with its own revisit condition, not smuggled in here.
