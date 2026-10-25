---
id: DEC-CA-0053
type: decision
title: "Generation moves off the GPU pod: an operator-run CPU tier on Targon and Shadeform materialises each leg's corpus into a tagged Hippius dataset, metered in CPU budget instead of GPU feed rate; series lengths up to ~32k behind a scheduled context"
status: proposed
date: 2026-10-18
tags: [generator, training, compute, provision, storage, consensus]
revisit_when: "Targon CPU XLarge capacity stays too thin to serve a settlement's legs (2 listed on 2026-10-18); or Shadeform's listing proves GPU-only with no vCPU-rich cheap shape; or measured generation on the pinned CPU type differs byte-wise from a second host of the same type (stream_cpu reproducibility broken: pin harder or audit by tolerance); or corpus storage cost exceeds the intake fee it is funded from; or king corpora outgrow publishing in full (switch to recipe + sample shards); or the first context step (8192) is scheduled"
relations: {depends_on: [DEC-CA-0031, DEC-CA-0036, DEC-CA-0042, DEC-CA-0045], relates_to: [DEC-CA-0001, DEC-CA-0010, DEC-CA-0022, DEC-CA-0050]}
---

## Decision

Owner (2026-10-18): generators must not be throttled by the GPU pod's CPU;
series "start 4096 but be possible to make from 8k to 32k"; each leg's data
stored "to a hippius dataset that's tagged by round, miner uid/hotkey, score";
50B time-series points per leg (pre-patch tokenisation); only kings' data
public; Lium has no CPU servers and two miner keys is clunky — "queue up
Shadeform and Targon as CPU generator providers".

1. **A CPU generation tier, run by the operator.** Before a leg trains, its
   generator runs on an operator-rented CPU VM (container sandbox,
   `sandbox_strict`, digest-pinned image, never the orchestrator). The tier
   fans the generator out over N sandbox children with derived seeds
   (the `gen_seed_mix` rule generalised; miner code still may not import
   multiprocessing) and writes digest-pinned shards to Hippius. The GPU pod
   reads shards and never runs miner code. Miners keep ONE key (the GPU
   payer key); the CPU cost is the operator's, recovered by an intake fee
   or the owner share (open, below).
2. **Providers, preference-ordered:**
   - **Targon CPU VMs** (live price list, 2026-10-18, owner-supplied):

     | Type | vCPUs | RAM | Price | Available |
     |---|---|---|---|---|
     | CPU Small (AMD) | 4 | 52 GiB | $0.15/h | 22 |
     | CPU Medium (AMD) | 8 | 104 GiB | $0.30/h | 11 |
     | CPU Large (AMD) | 16 | 209 GiB | $0.60/h | 5 |
     | CPU XLarge (AMD) | 32 | 419 GiB | $1.00/h | 2 |

     XLarge is the pinned shape. The initial proposal used a 5 h window
     ($5 per leg at the listed rate). Updated 2026-10-21: the configured
     generation cap is now 90 minutes on 32 vCPUs (48 vCPU-hours), stopping
     earlier at 50B points. At the listed rate that is $1.50 for the
     generation window alone; setup, finalization and upload are additional.
     Capacity is the risk (2 XLarge listed).
   - **Shadeform**: the adapter already exists (`provision/core.py`
     `ShadeformProvider`, operator GPU rentals). Its instance-types API is
     documented as GPU types, but every type reports `vcpus` and
     `memory_in_gb`, so the CPU path is a cheap vCPU-rich shape. To verify
     before arming: whether CPU-only shapes are listed at all.
   - Lium is not a CPU provider (no CPU-only executors; GPU executors list
     8–13 cores per consumer GPU, 16–32 per GPU on 8-GPU boxes, snapshot
     2026-10-18).
3. **A uniform CPU budget replaces the GPU feed rate as the generator's
   compute limit.** Every leg gets the same pinned shape (32 vCPU, ~2 GiB
   per sandbox child) for the same generation window. A generator still
   spends its own compute — a slow prior yields fewer points — but on a
   fixed, identical machine, not on whatever host the GPU leg landed on
   (DEC-CA-0010's host variance leaves the scored path). This narrows
   DEC-CA-0001: the wall still bounds training, it no longer prices data.
4. **50B points per leg, unique, pre-patch.** The GPU trains on the
   materialised corpus; the budget denomination (DEC-CA-0042/0047) still
   decides where the stream stops. At ~2.8M points/s over 5 h, 32 vCPU
   needs ~87k points/s per vCPU: the current king measured ~3M points/s on
   one warm Ryzen 8845HS core (~1–1.5M expected on a server vCPU), plain GP
   at L=4096 ~14k, CauKer ~750, KernelSynth ~320 (DEC-CA-0031). A king-speed
   mix leaves room for ~15% of points from plain GP at 4096; long expensive
   priors need spectral/sparse approximations — the intended axis.
5. **Length: context scheduled, series up to ~32k.** `[training]
   context_length` stays 4096 and becomes a scheduled contract value
   (`context_length_after(_block)`, the `budget_denomination_after` pattern),
   stepping to 8192 / 16384 / 32768 only by a DEC-CA-0045 on-chain switch.
   `[generator] max_length` rises to 32768. Until context grows, a series
   longer than the context is cut into seeded random context-length windows
   by the trainer, so a long trajectory is not wasted (reverses DEC-CA-0022's
   "trainer keeps the last window" only for stored corpora). The v2 lineage
   start (`[v2] from_block`, mainnet TBD) is the natural first step to 8192.
   Each step needs an eval pool that supplies context + horizon fresh points.
6. **The corpus dataset on Hippius S3** (dual-written to R2):

   ```
   corpora/round-<id>/<hotkey>/shard-<n>.parquet   series (float32 + per-series C, L)
   corpora/round-<id>/<hotkey>/meta.json           round, era, uid, hotkey, generator repo@digest,
                                                   seed, corpus_digest, n_series, points,
                                                   length histogram, channels, CPU-seconds,
                                                   CPU shape and model, provider
   corpora/index.parquet                           one row per leg; score columns filled after
                                                   settlement (CRPS/MASE geomean, verdict,
                                                   crowned, bench numbers)
   ```

   Keyed by hotkey; UID is a tag (UIDs recycle). Shards are immutable;
   scores land only in the index. The corpus digest is re-derived over the
   stored representation so an auditor hashes exactly what trained.
7. **Visibility.** Only kings' corpora are public, published when crowned
   (matching DEC-CA-0036: only kings publish). Every other leg's corpus is
   operator-private and deleted after the audit window. A 50B-point corpus
   is ~200 GB float32; at ~2 kings a day full publication is ~400 GB/day,
   so a king may instead publish recipe (generator ref + seed + digest) plus
   sample shards — byte-exact `stream_cpu` regenerates the rest.

## Consensus surface

`corpus_mode`, `long_series` and `context_length` are in `contract_digest`.
**They ride the v2 go-live** (owner, 2026-10-18: everything "needs to be live
when the new incentive goes live"): mainnet `[v2.training]` carries
`corpus_mode = "shards"` and `long_series = "crop"`, so they switch at
`[v2] from_block` with the rest of DEC-CA-0050 — no separate readiness vote.
Context growth past 4096 is a later typed step (`context_length_after` /
`_after_block` in `[v2.training]`). The tier runs for exactly the legs whose
effective contract is `shards`; the loader refuses a `shards` contract with
`[generation] enabled = false`. Testnet's v2 is already live (8190000), so its
`[v2.training]` is unchanged — testing the bundle there needs a fresh
`from_block` on a later era boundary.

## Settled since

- The subnet owner pays for CPU (owner, 2026-10-18); the budget unit is the
  vCPU, which is what Targon sells.

## Shipped (inert)

- Config: `[generation]` (`enabled = false`), `[training] long_series`,
  `context_length_after(_block)`, corpus mode `shards`.
- `cascade/provision/cpu.py`, `targon.py` (inventory API verified live;
  org-scoped VM calls written against the docs, never exercised),
  `ShadeformCpuProvider`.
- `cascade/generation/`: shard format, the fan-out tier (`cascade-generate`),
  the tagged dataset (upload, scores, king publication, retention).
- Trainer: `shards` stream, long-series crop, `--corpus-dir`, no generator
  fetch on a shards leg; audit re-digests shards.

## Transfer (owner, 2026-10-19: pre-signed links)

Neither machine holds a storage credential. The orchestrator signs
per-object, time-limited links: upload-only for every shard name the tier can
write plus `corpus.json` (12 h), download-only for one leg's shards (24 h).
The VM uploads shards, then the index last; the GPU pod gets `corpus.json` +
`urls.json` (written over ssh stdin, never a command line) and streams each
shard when its child reaches it, checks its sha256, deletes it once read
(~2 × 64 MB shards per child on disk). Shards are 16.8M points.

## The stage (built, inert until v2)

`cascade/generation/stage.py` + `trainer/runner_generation.py`: for a
`shards` leg, BEFORE the GPU rent — rent the CPU VM (Targon, then
Shadeform), bootstrap it with `deploy/pod_bootstrap.sh`, copy the generator,
run `cascade-generate --upload-urls` (strict sandbox), tag the corpus,
terminate the VM whatever happened. Generator fault ⇒ the miner's (rc 3);
everything else requeues. One corpus per leg is cached, so a rent that falls
back to an operator lane does not regenerate. Funded, operator-rented, king
and lane-B legs all use it. Admission budgets the tier's lead
(`window_seconds` + 2 h) on top of the GPU wall once a `shards` contract can
apply.

Review fixes (2026-10-19): the generator tree, dataset prefix and VM name
carry a ref digest (`round_id = <base_seed>-<sha256(ref)[:12]>`), so a
hotkey's second ref in one era never reuses the first ref's fetch or
overwrites its shards; one build per `(hotkey, seed, ref)` under a lock; the
fetched tree is removed after the stage; `corpora/index.json` writes hold a
process-wide lock; a VM that cannot isolate (`sandbox_isolation_unavailable`)
is infra, not the miner's; Targon terminate treats only 404/410 as gone and
raises otherwise; shard names in the uploaded index must be the planned ones;
lane B refuses its local fallback under `shards`.

## Before typing mainnet `[v2] from_block`

- Verify Hippius S3 honours SigV4 pre-signed PUT/GET (untested here).
- Set `[generation] vm_image` (Targon needs a VM image) and run one leg end
  to end on testnet (fresh testnet `from_block`); the Targon org-scoped VM
  calls have never been exercised.
- Fund the Targon account (owner pays) and check XLarge capacity (2 listed
  on 2026-10-18) against the expected legs per era.
- Consider a shorter `window_seconds`: lead (window + 2 h) plus the 5 h GPU
  wall is ~12 h, so legs start in the previous era's pre-train window.
- **Blocker — isolate miner code from the VM's filesystem.** `cascade-generate`
  forces `sandbox_strict` (no network), but under `sandbox_mode = "subprocess"`
  a child runs as the login uid and can rewrite the venv or the source tree.
  The parent then runs after the children with the upload links, so it could
  run the miner's code and upload an arbitrary corpus. The parent now
  pre-imports what it runs afterwards; that is a mitigation, not the fix. Before
  go-live: run children under `sandbox_mode = "container"` (digest-pinned
  image) or a separate uid with the tree and other children's output read-only.
- **Blocker — size the post-window budget from a measured run.** The stage
  allows `window_seconds + 1 h` for the children to wind down (a child can
  overrun by `stream_stall_seconds`), the single-threaded digest pass over
  ~200 GB, and the upload of ~3000 shards; the whole corpus also sits on the
  VM disk until upload. Measure one full leg; stream uploads as shards close
  and check VM disk at launch if it does not fit.

## Open

- Retention window for non-king corpora (default 14 days).
- Whether king corpora publish in full or as recipe + samples (default
  recipe).
- A context step past 4096 on the toto2 lineage changes `max_patches`, so it
  needs a new `base_arch_digest` (DEC-CA-0022); the MoE lineage derives it.
