---
id: DEC-CA-0010
type: decision
title: "Host variance is MEASURED, not normalized — host telemetry is diagnostic only"
status: active
date: 2026-07-30
tags: [telemetry, heat, trainer, provisioning, observability]
revisit_when: "a round's roll-up shows host_bench spread above ~1.3x across the heat field, or a regression of throughput_tokens_per_s on host_bench + host_gen_cpu_slice attributes a material share of heat throughput dispersion to the pod — at which point per-host normalization of ref_throughput (or SKU-homogeneous heat fleets) becomes a real choice instead of an unmeasured one"
relations: {refines: DEC-CA-0001}
---
Each training run now stamps a `host` record into its public training log:
lane geometry, CPU/GPU capability, opaque pod + physical-machine ids, and a
**fixed calibration bench** (`host_bench_tokens_per_s`) timed before the run's
corpus stream opens. Nothing consumes it. The wall stays exactly as
DEC-CA-0001 set it, `ref_throughput` is not normalized per host, and no score,
weight, rank, or `p_best` reads a host field.

**Why this was needed.** A miner (uid 93/111 entrant, round 16968040428939357615)
reported that realized heat compute split into two clusters on nominally
identical 4090s — 24 runs at ~3.87M tok/s finishing 100% of budget, 12 at ~2.59M
finishing 13–96% — and made the argument that matters: `corr(throughput,
data_wait_frac) = +0.542`, i.e. the SLOW runs waited *less* for data. Generator
starvation gives the opposite sign, so starvation cannot be the explanation for
that cluster. They also had a same-operator pair where the *cheaper* generator
(0.416 vs 0.471 ms CPU/series) got *less* compute.

That argument was correct and unfalsifiable from public data at the same time:
`data_wait_frac` can rule the corpus OUT, but nothing in the record could rule
the pod IN. Host id, core count, and co-tenancy were simply absent. The gap was
ours, and a screen that prices compute owes entrants the ability to check which
compute they were priced on.

**Why measure rather than correct.** The tempting fix — divide realized
throughput by a per-host factor before the wall applies — is a consensus-relevant
change to the thing DEC-CA-0001 deliberately fixed, argued from an *inferred*
effect size. We do not know the magnitude yet. Two failure modes bracket it: if
the fleet turns out uniform, normalization adds a gameable denominator for
nothing; if it turns out badly non-uniform, the right fix is more likely
procurement (SKU-homogeneous heat fleets, which DEC-CA-0003's ladder already has
the machinery for) than arithmetic. Both need the number first. So this ships the
instrument and defers the policy, which is the same shape as DEC-CA-0006:
compute the diagnostic, record it, do not obey it.

**Design constraints held.**

- **No wire-format change.** Host facts ride channel 9 (the training-log JSONL +
  the wandb mirror), never the manifest or the receipt. `contract_digest` is
  untouched — the config lives in a new `[telemetry]` section, not `[training]`,
  so no validator restart and no signature break (cf. the operational invariant
  on `[training]` edits).
- **The bench is fixed, not configurable.** `[telemetry] host_bench` turns it
  off; it cannot resize it. A bench an operator can tune produces numbers that
  cannot be pooled across the fleet, which defeats the purpose. Any change to the
  workload must bump `HOST_BENCH_SPEC` so two versions' numbers are never
  silently compared.
- **Cannot touch the compute it measures.** The probe runs after the generator is
  fetched but before the corpus stream (hence the sandbox child) exists, so it
  measures the pod rather than the pod-plus-submission; `max_train_seconds`
  anchors at the first training batch and the budget is a token count, so the
  bench cannot eat either. It uses its own RNG, so the contract's pinned init and
  data order are unchanged.
- **Never fatal.** Every probe degrades to missing keys. A pod with no
  `nvidia-smi`, no `/proc`, or no CUDA still trains.

**Two scopes of host id, deliberately.** `host_id` (from `/etc/machine-id`)
groups the lanes of one pod; `host_boot_id` (from `/proc/sys/kernel/random/boot_id`,
which is not namespaced) groups pods on one physical machine. Runs sharing a
`host_boot_id` but not a `host_id` are co-tenants — the case a per-pod id can
never show, and one of the three things the miner named as missing.

**What answers the question.** Regress `throughput_tokens_per_s` on
`host_bench_tokens_per_s` and `host_gen_cpu_slice` across a round's heat field.
The per-round roll-up line carries the headline: `host_bench min/p50/max`,
`spread` (max/min — the bound on what the pods alone could explain), and distinct
pod / machine / SKU counts. At `spread=1.00x` the fleet is uniform and the
dispersion is the generators, which is DEC-CA-0001 working as intended.
