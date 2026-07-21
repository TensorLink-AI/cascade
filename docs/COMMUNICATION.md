# Validator ↔ Trainer Communication Map

Every channel between the subnet's roles, in one place: who writes, who reads,
what carries integrity, and what happens when several validators run at once.
`docs/ARCHITECTURE.md` explains *why* the round works this way; this page is
the wire-level *how*.

Terminology worth pinning down first: the **trainer** publishes the round's
**manifest** (signed training receipts); each **validator** publishes a round
**receipt** (signed scoring/weights record). Validators never post manifests —
they gate, score, set weights, and publish receipts.

## The picture

```
                       Bittensor chain
        commitments ▲            │ commitments,      ▲ weights
        (gen refs)  │            │ incentive, blocks │ (per validator,
                    │            ▼                   │  independent)
   ┌────────┐   ┌─────────────────────┐        ┌──────────────┐
   │ miners │   │       trainer       │        │  validators  │  × N
   └────────┘   │ (owner GPU boundary)│        └──────────────┘
        │       └─────────────────────┘          ▲  ▲  ▲    │
        │ generator     │ checkpoints │ manifest │  │  │    │ receipts (signed,
        │ repos         ▼             ▼          │  │  │    ▼ per-validator prefix)
        ▼        ┌───────────┐  ┌─────────────────────────────────┐
   ┌───────────┐ │ Hippius   │  │ manifest bucket (Hippius S3)    │
   │ Hippius   │ │ Hub (OCI) │  │  manifests/…  logs/…            │
   │ Hub (OCI) │ └───────────┘  │  receipts/<hotkey>/…  index.html│
   └───────────┘                └─────────────────────────────────┘
                                                 ▲
                    ┌───────────┐   pool bucket  │ snapshots (tar+sha256,
   TSBench-Forge ─► │ owner pool│ ───────────────┘  selected by
   (private raw     │ builder   │                   effective_block)
    data bucket)    └───────────┘
```

## Channels

| # | Channel | Writer → Reader | Medium / key | Integrity | Multi-validator behaviour |
|---|---------|-----------------|--------------|-----------|---------------------------|
| 1 | Generator commitments | miner → trainer, validators | chain reveal commitments, `metro-v1:gen:hippius:<repo>@<digest>` | on-chain, digest-pinned | read-only; identical view per block |
| 2 | Round seed | chain → everyone | epoch-start block hash → `seed_from_block_hash` | consensus | deterministic, shared |
| 3 | Training manifest | trainer → validators | `manifests/round-<id>.json` + `manifests/latest.json` (manifest bucket) | signed by `[manifest] trainer_hotkey`; gated by `check_manifest` | read-only; single writer (the trainer) |
| 4 | Checkpoints / generators | trainer, miners → validators | Hippius Hub OCI, `repo@sha256:…` | content-addressed digest self-verifies | read-only |
| 5 | Eval pool | owner pool builder → validators | `pool/snapshots/block-<N>.tar` + `pool/index.json` (pool bucket) | sha256 pinned in the **signed manifest** (`eval_pool_key`/`eval_pool_sha256`, gated by `check_pool_pin`); index sha256 as legacy fallback for unpinned manifests | read-only; all validators resolve the same snapshot for a round and verify it against the pin |
| 6 | Weights | each validator → chain | `set_weights` | on-chain, per-hotkey | independent by construction — no shared state |
| 7 | Round receipts | each validator → auditors, dashboard | `receipts/<hotkey>/round-<id>.json` + `receipts/<hotkey>/latest.json`, shared `receipts/latest.json` pointer | signed per validator hotkey; public-read | **single-writer prefixes** — validators cannot clobber each other; only the convenience pointer is last-writer-wins |
| 8 | Receipts index | each validator → dashboard | `receipts/index.json`, entries keyed `(round_id, validator_hotkey)` | unsigned, presentational only | merge-keyed read-modify-write; a simultaneous write can drop one entry until that validator's next round — never audit- or weight-bearing |
| 8b | Live chain status | validator (or `scripts/publish_chain_status.py`) → dashboard | `status/chain.json`: block anchor, epoch grid, stage windows, revealed submissions (see `cascade.shared.chain_status`) | unsigned, presentational only | last-writer-wins whole-object overwrite — honest publishers write near-identical chain state; never audit- or weight-bearing |
| 9 | Training logs | trainer → observers | `logs/round-<id>/<role>.jsonl` (+ optional wandb) | none (observability) | read-only |
| 10 | Trainer ↔ GPU pods | trainer orchestrator ↔ rented pods | SSH, receipt-sentinel stdout | wallet never leaves the orchestrator | n/a |
| 11 | Validator ↔ eval pod | validator ↔ its own GPU pod | SSH+scp (`--eval-hosts`) | only public checkpoint + report cross | per-validator, private |

The channels a verdict depends on (1–6) are all either on-chain, signed, or
content-addressed *and* deterministic per round — N validators reach the same
verdict independently, which is why weights need no coordination. Channels
7–9 are evidence and observability, never inputs to consensus.

Because storage keys aren't prefix-scoped, anyone holding a bucket's write
credentials can overwrite *any* object in it. The design assumes that and
anchors integrity above the storage layer: manifests fail the trainer-hotkey
signature gate if touched, checkpoints are digest-pinned, and the eval pool is
pinned inside the signed manifest. What write access *does* buy an attacker is
vandalism — deleted receipts, a defaced dashboard, stalled rounds — which is
detectable (signatures, the R2 mirror) but not preventable; the mitigation is
key hygiene: one pair per bucket, pool keys separate from manifest keys, forge
keys never shared beyond the owner.

## What a validator needs access to — and only this

| Resource | Access | Credential |
|----------|--------|------------|
| Bittensor chain | read + `set_weights` | hotkey wallet |
| Manifest bucket | read `manifests/…`, write `receipts/<own hotkey>/…` + index | `HIPPIUS_S3_*` |
| Pool bucket | read-only snapshots | `POOL_S3_*` — issue a pair separate from the manifest-bucket keys (S3 keys aren't prefix-scoped; a shared pair could write the pool) |
| Hippius Hub | read (pull checkpoints/generators) | `HIPPIUS_HUB_TOKEN` |
| HF benchmark datasets + bench sidecar (opt-in only, see below) | public read | `HF_TOKEN` if gated |

The benchmark sidecar (`benchmarks/`, or an SSH eval pod via `--eval-hosts`) is
**not** part of a stock validator. All three of its call sites default off:
`[eval] run_benchmarks` (log-only numbers on a dethrone), `[scoring]
gift_gate_mode` (the gift-eval gate — the only mode in which the sidecar is
load-bearing, since `enforce` turns a sidecar failure into an inconclusive
round), and `[scoring] cascade_enabled` (warm-start promotion, which prefers
the trainer-signed `bench_scores` on the manifest and touches the sidecar only
as a non-consensus-safe fallback for manifests that carry none).

Deliberately **not** on the list: the TSBench-Forge raw-data bucket. Validators
consume forge data only through the built pool snapshots (channel 5). The raw
catalog + parquet relay (`tsbench-forge-sources` on Hippius — see
`docs/EVAL_POOL.md`) is an owner-orchestrator-side secret; keeping it private
is the anti-gaming lever that stops miners from fitting generators to the
held-out pool.

## TSBench-Forge → validator data path

```
forge repo cron (scrape-data.yml)
  └─ sync_storage.py ─► s3://tsbench-forge-sources        [private, owner-only]
owner orchestrator
  └─ aws s3 sync … ./tsforge
     cascade-pool publish --sources tsbench_forge ─► pool bucket snapshots
validators
  └─ select snapshot by effective_block, verify sha256, rotate windows by round seed
```

One bucket contract end to end: `sources.yaml` + `data/<source_id>/<date>.parquet`
at the relay, deterministic tar + `pool/index.json` at the pool. The forge cron
and the pool-publish cron are decoupled; the staleness guard
(`max_stale_days`, fail-loud when every feed is stale) lives in the pool
builder so a dead scraper can't silently erode freshness.
