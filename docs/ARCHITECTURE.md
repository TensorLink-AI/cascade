# Cascade architecture

This document describes the implemented rolling A/B/C system. It is an
implementation map, not live configuration. Discover deployment state through
`cascade status --json`; use signed contracts, manifests and receipts for
eligibility and outcomes. [Decisions](../decisions/) amend historical rules;
[operator](OPERATOR.md), [miner](MINER.md) and [validator](VALIDATOR.md) guides
cover role-specific procedures.

**Read by task:** [roles and flow](#roles-and-data-flow) · [scoring](#scoring)
· [status](#discovery-and-miner-status) · [communication map](#communication-map)
· [configuration](#configuration-and-compatibility) · [trust](#trust-model).

In this guide, a **leg** is one training run; an **era** fixes its shared
starting checkpoint and rules; a **settlement** judges compatible completed
work. The **king** is A's current winner. The **init** or **warm start** is the
shared starting checkpoint, which can differ from the king's trained weights.
The trainer signs **manifests**; validators sign **receipts** recording decisions.

## The thesis

Miners compete on synthetic training data under a fixed comparison contract.
A/B contributions can accumulate in a shared forecasting model lineage. C is a
separate application competition, currently MarketCast on a pinned Timestral
base. Its checkpoints do not merge into A/B.

Pre-v2 eras use the Toto2 lineage; the v2 candidate is Cascade MoE Base - 1.23B
(`cascade-moe-v2-120m`) under `[v2.training]`. It starts a fresh lineage at an era boundary. Historical
`moe_model.py` and `toto2_model.py` bytes remain unchanged for replay. The separate
direct runtime retains its own architecture contract.

The new backend is `cascade_pyramid`: `trainer/pyramid_model.py` supplies the
self-contained model, `pyramid_trainer.py` the objective and training loop,
`pyramid_optimizer.py` the width-corrected optimizer, and `pyramid_wrapper.py`
the standalone checkpoint loader. `trainer/contract.py` binds the model source
into its architecture digest; `eval/checkpoint_guard.py` verifies both shipped
Python files, configuration and tensor geometry before execution.
`Toto2Trainer` dispatches the effective architecture for workers, audits and
miner experiments. New update budgets are digest-bound; their default zero
leaves historical contract digests unchanged. Optional generator `groups`
are carried through schema-2 shards, equal-width batches, masked variate
attention and per-group copula sampling (DEC-CA-0065). Values-only paths retain
their numerical and corpus bytes; new pyramid source/contract pins still
require a matching release and checkpoint lineage.

The [model specification](MODEL.md) describes Cascade MoE Base variants at 17.58M,
69.02M, 273.84M and 1.23B total parameters. [Preset instructions](../deploy/presets/README.md) cover complete
local configs. A 56M identity stays blocked. Lane B resolves trusted checkpoint
code before loading, counts each layer's actual experts, and uses compact
trainable rows without changing the frozen weights. Routing matrices pad shorter
layers to 512 columns; zero-affinity padding cannot enter a target selection.

## Roles and data flow

### 1. Miner — submits a generator

A/B miners submit deterministic, code-only Python generators. B also declares
eligible targets; the trainer selects routed expert rows. C submits an eligible
MarketCast recipe, optionally including a generator. Inputs are immutable
refs bound to on-chain commitments and signed intake requests. A hotkey holds
one commitment, so replacing it can supersede another lane's queued entry.

| Lane | Entry path | Scientific state after training |
|---|---|---|
| A | `miner/cli.py` → `funding/intake.py` → `FundedQueue` | Settlement, private-pool comparison and applicable public benchmark gate |
| B | `miner/lane_b.py` → lane-B intake/queue → `trainer/lane_b_driver.py` | Target qualification, then warm-start merge acceptance when enabled |
| C | `miner/lane_c.py` → lane-C intake/queue → `trainer/lane_c_driver.py` | Forward judging, deferral or pending confirmation, then confirmed slot decision |

Paths in this map are relative to `cascade/`. Publication and funding are
explicit actions. The lane-specific CLI workflow and output contract are in
[AGENT_MINING.md](AGENT_MINING.md); source/data limits are in
[INTERFACE.md](INTERFACE.md).

`funding/providers.py` signs an ordered Lium/Shadeform/Vast/Runpod credential list; the
local vault retains immutable rental credentials under `leases/`. The public
ledger holds only provider and opaque credential references. `provision/payer_pool.py`
selects capacity before create, adopts existing pods after restart, and routes
cleanup to the original account. An uncertain create never falls through to a
second account. The separate direct runtime keeps its own provider rules.

Non-secret `create-intents/` records survive restarts and credential expiry.
An empty provider listing does not resolve them. A confirmed existing allocation
can be adopted, or a positively observed allocation can be terminated; only
then can recovery clear its intent. Ledger corruption fails closed, with atomic
writes and a backup; heartbeat refreshes cannot undo a credential rotation.

`miner/funding.py` shares signing and the validation/consent/wallet order across
A/B/C. Lane modules retain target checks, endpoint selection and result fields;
the public CLI helper functions remain compatibility entry points.

### 2. Trainer — owner-operated, the GPU boundary

`trainer/rolling.py` owns era context, A admission, completion and settlement.
`TrainerRunner` in `trainer/loop.py` supplies infrastructure through the
`runner_*` mixins. B/C drivers run alongside it, using shared admission and
GPU capacity, while keeping their own scientific state.

Paid A entry opening is resolved separately from training activation, using
finalized chain timestamps. The v2 intake rejects funding before the opening;
queued challengers report `waiting_launch` and the trainer starts no CPU/GPU
work until v2. Old reveals are refused without burning an attempt. Legacy
receipt replay retains its earlier floor. C drawn from A may launch before B;
the validator and audit reserve only the pools that have launched.

A funded entry becomes eligible after reveal and lane checks. It can wait for
CPU admission, corpus preparation, a fitting GPU, shared operator capacity,
worker startup or settlement. Intake acceptance does not reserve a GPU. Reveal
order is seniority subject to capacity, choice/caps and lane rules, not an ETA.

#### One training leg

1. Pin the leg's contract, seeds, base/init and immutable entry identity.
2. Prepare its corpus. V2 A uses `generation/stage.py` and `generation/tier.py`
   to create bounded sandboxed CPU shards before GPU rental. The generation
   window is separate from the GPU training wall. The worker reads the pinned
   shards; miner code does not need to execute on the GPU pod for that path.
3. Dispatch the training worker under the pinned image/code fingerprint. A
   trains the full model; B uses the selected-expert backend with other weights
   frozen; C follows its competition's pinned-base recipe and data rules.
4. Harvest, integrity-check and upload the checkpoint. Retained local harvest
   evidence allows an interrupted upload to retry without repeating training.
5. Persist completion before independent benchmark work. Compatible completed
   entries can settle; old-era A work cannot silently become a new-era result.

Core seams are `runner_funded_legs.py`, `runner_funded_pods.py`,
`runner_operator_pods.py`, `runner_generation.py`, `worker.py`, `contract.py`
and the model-specific trainers. Worker receipts and checkpoint measurements
bind artifacts to the effective contract. Tensor masks/compact rows and model
state names are scientific compatibility surfaces, not generic cleanup targets.

#### Capacity, queues and recovery

`trainer/admission.py`, `lane_pool.py` and `runner_capacity.py` coordinate shared
work. CPU generation has its own parallelism/capacity. King work has provisioning
priority; B and C retain their lane limits and compete for remaining resources.

`gpu_health.py` persists GPU-type cooldowns and `operator_health.py` runs an
independent watchdog with a durable alert outbox. `provision/operator_pool.py`
reserves capped operator spending before a create and binds cleanup to the
original account and physical ID. `king_hedge.py` persists a delayed replica's
first valid completion; it does not select on evaluation scores. These paid
recovery features require explicit limits and ship disabled.

`continuation.py` and `continuation_transport.py` implement the separately
activated `cascade-continuation/1` protocol for MoE A/B and MarketCast C.
Optimizer-boundary snapshots are pulled off-pod and checked against the job,
input replay and qualified runtime before reuse. The original MarketCast
training data travels with its continuation. The default empty protocol keeps
historical training digests unchanged. See the
[recovery controls and qualification procedure](OPERATOR.md#qualified-training-continuation)
for activation requirements and integration limits.

Queues are fresh file-backed instances with atomic writes and locking. A
running attempt has a unique identity; stale callbacks cannot relabel a retry
or consume a parked new commitment. Retry/backoff, deadlines and fault classes
are explicit. Operator-caused losses preserve the submission. With configured
operator lanes or capped operator rentals the retry runs on them; otherwise it
rents a new pod on the payer's key, and after `OPERATOR_FAULT_RETRIES` (3) such
losses the entry pauses for the operator.
No spare capacity means waiting/requeue, not a manufactured successful result.
See [the backlog runbook](OPERATOR.md#backlogs-and-recovery-across-lanes).

`runner_common.funded_queue_of` resolves each lane's queue from one path/TTL
rule, while runner-local specialist/king overrides remain in `TrainerRunner`.
`trainer/lane_status.py` shares best-effort publication but keeps independent
B/C rate limits. `shared/concurrency.call_with_deadline` bounds the caller's
wait, not the underlying operation: a timeout is not evidence that a remote
launch failed or was cancelled.

#### Settlements

Epoch boundaries define settlement opportunities; eras pin training context.
Mainnet's effective rolling grid is derived through activation, not by changing
a timing constant in isolation. A boundary belongs to the era containing the
preceding block. Manifests are hash-chained by `prev_round_id` and name the
training contract, era/init, entries and associated specialist/report digests.

Durable write-ahead state preserves publication bytes and recovery ordering.
The validator journals state transitions so partial writes, restarts and replay
cannot silently produce a second merge or reward. Missing evidence can hold or
defer work; it is not automatically a miner loss. The
[operator runbook](OPERATOR.md#lineage-measurements-and-state-upgrades)
describes recovery and state upgrades. Real provider behaviour and throughput
require integration testing on the selected deployment.

### 3. Validator — reads the manifest, decides the throne

Full evaluators check signatures, chain context, commitment/base identity,
contracts, manifests and report bindings; score the required private windows;
apply scientific rules; persist resulting state; and publish signed receipts
and chain weights. They do not train challengers. When validator-sourced benches
are active, the designated main validator also needs benchmark data and GPU
capacity. Other full evaluators replay those signed public rows under their
bindings while still scoring the private pool themselves.

A satellite verifies the designated validator's signed receipt trail and derives
weights without loading the training stack. Its role differs from an independent
GPU evaluator. See [VALIDATOR.md](VALIDATOR.md) for setup and recovery.

#### Lane B: qualification and merge acceptance

`trainer/lane_b_driver.py` plans expert rows and holds trained legs. Settlement
rebases a leg against the compatible current king, checks routing/base staleness
and byte-level row constraints, and publishes an expert manifest. Target gains
and non-inferiority guards determine qualification and slot selection.

When acceptance is enabled, a qualification neither starts payment nor replaces
the existing holder. At a scheduled warm start, `trainer/merge_acceptance.py`
tests compatible deltas incrementally against the checkpoint accepted so far,
with regression checks against the original init. `validator/merge_acceptance.py`
verifies the bound intermediate weights and evidence. Only committed accepted
contributions enter the init and begin rewards; rejection leaves them out.
Overlapping rows or already-taken target slots can defer a proposal to another
init. Existing pre-acceptance rules remain available only where configured.

Acceptance staging has a trainer time budget (`round.ops_acceptance_timeout_seconds`,
1,200 seconds by default). An overrun pins that era to the unmerged init; pending
qualifications remain unpaid and unconsumed. A late background result cannot
replace that pinned init. At most one acceptance worker runs at a time.

#### Lane C: forward judging and confirmation

`trainer/lane_c_driver.py`, `marketcast_runner.py` and `marketcast_leg.py` use
the configured Timestral base and recipe. The signed lane-C manifest binds the
trained artifact, recipe, training block and MarketCast evidence. The main
validator evaluates forward-only windows and publishes report evidence whose
exact digest is bound by a later manifest. Missing/infra evidence defers work
under the configured bounds; scientific rejection is a different outcome.

`validator/marketcast.py` and `validator/lane_c.py` maintain forward judging,
confirmation and target holders. An initial pass is pending; a later
confirmation must also satisfy the slot rule. A passing candidate below an
existing holder's bound does not take the slot. No C fine-tune merges into A/B.

Expired or deregistered holders are removed before confirmation. Only the final
holder is labelled `confirmed` when several backlog rounds resolve together;
judge-phase runners-up are `not_selected`. Losing every replica of a report
already recorded by digest holds the whole settlement. Recovery requires the
exact report bytes; a local timeout cannot safely substitute a different result.

### Cascade — king-reign promotion

The reigning king and an era's shared init are distinct. V2 discards the old
Toto2 warm-start members and observes its configured random-init window.
From the v2 king-ladder policy, only a king backed by a passing validator king
check can promote. `trainer/runner_promotion.py`, `trainer/promotion.py` and
`validator/cascade.py` bind the decision and install it at an effective era.
A merely announced future promotion is not today's init.

A promotion must still have the full required notice when it actually fires.
Its canonical digest is announced in a trainer-signed settlement before the
effective era, and later settlements bind that notice through predecessor
hashes. Catch-up validators require this chain evidence too. Each v2 trained
entry records its starting checkpoint and contract; restart recovery refuses an
incompatible entry. Resolving the calendar launch discards future eras prepared
under the legacy contract and requeues their work without burning an attempt.
The trainer also requires a published notice before adding a promotion to its
era ledger. An unannounced record leaves the previous init in use; unreadable
evidence holds scheduling for recovery. A record whose earliest chain-bound
notice is late is never effective for any node: validators record the
generation as skipped (the init stays the live generation's) and their
catch-up walk continues at the next record; the trainer's ledger keeps the
previous init and the next passing king check fires the following generation
under the same rule. A record that cannot be published at its fire boundary is
retracted and the check re-read at the next boundary rather than announced
late. Recovered promotions invalidate any prepared era whose generation/init
differs, including after a restart.

B acceptance can add permitted expert deltas to a scheduled init. Its base and
intermediate evidence must match that exact adoption; background acceptance
staging cannot silently install results for another era/base.

The earlier rolling-top-k algorithm, including losing-challenger candidates,
is retained for its historical block/config rules. It must not be described as
the v2 king-only rule. Promotion does not itself crown a different Lane A king.

## The controlled-experiment invariant

Within an A comparison, competitors use the same effective contract, training
context and shared init; their generator/corpus is the competitive variable.
The signed manifest records that context and validators gate the terms they
consume. B additionally binds its selected rows and base; C binds a separate
competition base/recipe and forward data. Those comparisons are not interchangeable.

Model-source bytes participate in architecture digests. State layouts, immutable
refs and signed serialization also carry compatibility. Reusing a helper must
not rewrite those bytes or change an archived receipt's canonical form.

## Scoring

A uses paired private-window comparisons of quantile-loss/CRPS-related and MASE
metrics under the effective horizon, domain and bootstrap rules. The applicable
margin can depend on tenure and shared-init improvement; cohort correction
handles multiple challengers. Rules are block-gated to keep historical replay.

A challenger that passes the private test must also pass any active public
benchmark gate. The v2
mainnet configuration selects the four-suite GIFT-Eval/BOOM/TIME/FEV rule through
its activation settings. With validator-sourced benches, the main validator
publishes signed rows and subsequent manifest bindings determine which bytes
other validators use. Missing reports and budget deferrals are not invented
scores or automatic scientific losses.

B uses target gains and regression guards, followed by merge acceptance when
active. C uses its competition's forward-only tests and confirmation. The
planned sequence A 100%, A/C 75/25, A/B/C 50/25/25 describes mainnet lane
pools; actual rewards also depend on A's ladder and B/C slot/decay/expiry rules.
Unused shares burn under the mainnet policy; allocating a lane pool does not
pay its entire amount to its current winners. The
[reward schedule](MINER.md#rewards-allocation-burn-and-progression) gives the
launch caps, ladder and slot clocks. `shared/emission.py`, `shared/lane_b.py`,
`shared/lane_c.py` and validator folds define the actual weight accounting.

## Discovery and miner status

`status/v2.json` publishes effective policy and the current init; chain and
lane feeds publish timestamps. `miner/live_status.py` and `submission_status.py`
separate queue execution from scientific progress, matching hotkey/ref/targets
and the exact funded attempt. They distinguish missing, stale and empty feeds.

B exposes qualification and acceptance observations. C retains bounded
per-entry receipt observations across later omitted receipts and restarts, with
the original receipt boundary. Historical confirmation is not evidence of
current payment. Pending reveals remain redacted; stage changes are not worker
heartbeats. Public telemetry is unsigned; signed reports remain authoritative.

## Communication map

The main channels between the subnet's roles: who writes, who reads, what carries
integrity, and what happens when several validators run at once. Setup steps
for these credentials are in [VALIDATOR.md](VALIDATOR.md) and
[OPERATOR.md](OPERATOR.md).

Terminology worth pinning down first: the **trainer** publishes one
**manifest** per settlement (signed training receipts); each **validator**
publishes a **receipt** per settlement (signed scoring/weights record). Storage
keys and the CLI call a settlement a "round" (`round-<id>.json`). Validators
never post manifests — they gate, score, set weights, and publish receipts.
Miners fund legs (and optionally submit code privately) over the trainer's
HTTPS intake (channel 1b) — the only non-chain, non-storage channel.

### The picture

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

### Channels

| # | Channel | Writer → Reader | Medium / key | Integrity | Multi-validator behaviour |
|---|---------|-----------------|--------------|-----------|---------------------------|
| 1 | Generator commitments | miner → trainer, validators | chain reveal commitments, `metro-v1:gen:hippius:<repo>@<digest>` | on-chain, digest-pinned | read-only; identical view per block |
| 1b | Funding / private submission | miner → trainer | HTTPS to `cascade-intake`: `POST /v1/fund` (`X-Lium-Api-Key` or signed `X-Payer-Credentials`), `POST /v1/submit` (private generator ZIP → `vault/direct@sha256:…` ref), public `GET /v1/queue` | hotkey-signed request; the key lands in the sealed vault and nowhere else | n/a — validators never see it |
| 2 | Seeds | chain → everyone | training seeds from the era/seed-epoch block hash under the effective rule; the settlement boundary's block hash keys the round id and eval draw | consensus | deterministic, shared; v2 king-ladder eras can reuse a seed epoch |
| 3 | Training manifest | trainer → validators | `manifests/round-<id>.json` + `manifests/latest.json` (manifest bucket), one per settlement, hash-chained by `prev_round_id` | signed by `[manifest] trainer_hotkey`; gated by `check_manifest` + the era envelope | read-only; single writer; validators walk the chain forward, `latest.json` is only the head hint |
| 3b | Bench reports | trainer → validators, dashboards | `benchmarks/round-<id>.json` (GIFT-Eval / BOOM / TIME, + FEV under DEC-CA-0061, per settlement checkpoint) | signed by the trainer hotkey | read-only; feeds promotion verification and, when the lane-A bench gate is on (DEC-CA-0054 / 0061, every v2 settlement), the gate's per-item rows |
| 3d | Validator bench reports | main validator → validators, auditors | `bench_v/round-<id>.json` + rows under `bench_v/rows/<id>/` (DEC-CA-0063; pool clearers + king, `infra_failed` / `budget_deferred` / `checkpoint_fault` records) | signed by `[netconfig] owner_hotkey`; its sha256 bound by a later trainer-signed manifest and stamped `bench_report` on receipts | write-once; other validators hold until the binding lands, then replay exactly those bytes |
| 3c | Promotion records | trainer → validators | `promotions/gen-<n>.json` | signed by the trainer hotkey; verified against the envelope | read-only; installed at the record's `effective_era` |
| 3e | Lane-B expert manifest | trainer → validators | `lane_b/round-<id>.json`, digest bound by the A manifest | trainer signature, immutable checkpoint refs and row/base checks | validators reproduce qualification and accepted merge evidence |
| 3f | Lane-C manifest and MarketCast report | trainer/main validator → validators | `lane_c/round-<id>.json`, `marketcast/round-<id>.json` and associated arrays | signed evidence; exact report digest bound by a later manifest | validators replay the bound forward/confirmation evidence |
| 4 | Checkpoints / generators | trainer, miners → validators | Hippius Hub OCI, `repo@sha256:…` | content-addressed digest self-verifies | read-only |
| 5 | Eval pool | owner pool builder → validators | `pool/snapshots/block-<N>.tar` + `pool/index.json` (pool bucket) | sha256 pinned in the **signed manifest** (`eval_pool_key`/`eval_pool_sha256`, gated by `check_pool_pin`) | read-only; all validators resolve the same snapshot for a settlement and verify it against the pin |
| 5b | Activation notes | each validator → everyone | plain chain commitment `cascade-ready:1:<feature>:<lock>:<block>` (DEC-CA-0045) | on-chain, per hotkey; tallied by permit-holding stake | each node tallies the same notes as of the boundary block |
| 5c | Network configuration | owner → trainer and validators | `config/network/seq-<n>.json` + `config/network/latest.json` | owner signature, supported keys, sequence and effective-block/stake adoption | durable config history keeps catch-up and replay on the applicable settings |
| 6 | Weights | each validator → chain | `set_weights` | on-chain, per-hotkey | independent by construction — no shared state |
| 7 | Round receipts | each validator → auditors, dashboard, satellites | `receipts/<hotkey>/round-<id>.json` + `receipts/<hotkey>/latest.json`, shared `receipts/latest.json` pointer | signed per validator hotkey; public-read | **single-writer prefixes** — validators cannot clobber each other; only the convenience pointer is last-writer-wins |
| 8 | Receipts index | each validator → dashboard | `receipts/index.json`, entries keyed `(round_id, validator_hotkey)` | unsigned, presentational only | merge-keyed read-modify-write; a simultaneous write can drop one entry until that validator's next settlement — never audit- or weight-bearing |
| 8b | Live chain status | validator (or `scripts/publish_chain_status.py`) → dashboard | `status/chain.json`: block anchor, epoch grid, activation, revealed submissions (`cascade.shared.chain_status`) | unsigned, presentational only | last-writer-wins whole-object overwrite; never audit- or weight-bearing |
| 8c | Funded roster | trainer → miners, dashboards, audit | `funded/round-<id>.json` + `funded/latest.json`: the settlement's seated, in-flight, waiting and terminal entries plus per-rent seniority evidence (`rents`) | unsigned; the audit's `funded-roster` check WARNs on a pass-over, never FAILs | single writer (the trainer) |
| 8d | Training summaries | trainer → dashboards | `training/round-<id>.json`: per-leg steps, tokens, width | unsigned, presentational only | single writer (the trainer) |
| 8e | Current policy and lane progress | trainer/intake → miners and dashboard | `status/v2.json`, `status/lane_b.json`, `status/lane_c.json`, intake queue routes | unsigned timestamps and attempt-bound observations | discovery only; stale/missing is unknown, not permission to train or proof of payment |
| 9 | Training logs | trainer → observers | `logs/round-<id>/<role>.jsonl` (+ optional wandb) — per-step metrics, the closing `summary`, and one `host` record per run (`[telemetry]`, `cascade.trainer.host_probe`) | none (observability) | read-only |
| 10 | Trainer ↔ GPU pods | trainer orchestrator ↔ rented pods | SSH, receipt-sentinel stdout; checkpoint harvested by the orchestrator | wallet and storage credentials never leave the orchestrator | n/a |
| 11 | Validator ↔ eval pod | validator ↔ its own GPU pod | SSH+scp (`--eval-hosts`; separate `--private-eval-hosts` opt-in) | public benchmarks transfer checkpoint/report; private offload transfers hash-bound checkpoint + selected histories and returns samples, with targets/scoring local | per-validator, operator-owned host |

The channels a verdict depends on (1–6) are all either on-chain, signed, or
content-addressed *and* deterministic per settlement — N validators reach the
same verdict independently, which is why weights need no coordination.
Signed receipts in channel 7 are also the authoritative input for satellite
replay. Unsigned indices, queue views and training logs in channels 8–9 are
observability and must not independently determine weights.

Because storage keys aren't prefix-scoped, anyone holding a bucket's write
credentials can overwrite *any* object in it. The design assumes that and
anchors integrity above the storage layer: manifests fail the trainer-hotkey
signature gate if touched, checkpoints are digest-pinned, and the eval pool is
pinned inside the signed manifest. What write access *does* buy an attacker is
vandalism — deleted receipts, a defaced dashboard, stalled rounds — which is
detectable (signatures, the R2 mirror) but not preventable; the mitigation is
key hygiene: one pair per bucket, pool keys separate from manifest keys, forge
keys never shared beyond the owner.

### What a validator needs access to — and only this

| Resource | Access | Credential |
|----------|--------|------------|
| Bittensor chain | read + `set_weights` | hotkey wallet |
| Manifest bucket | read `manifests/…`, write `receipts/<own hotkey>/…` + index | `HIPPIUS_S3_*` |
| Pool bucket | read-only snapshots | `POOL_S3_*` — issue a pair separate from the manifest-bucket keys (S3 keys aren't prefix-scoped; a shared pair could write the pool) |
| Hippius Hub | read (pull checkpoints/generators) | `HIPPIUS_HUB_TOKEN` (or username + password) |
| HF benchmark datasets + bench sidecar (required on the designated main validator when active) | public read | `HF_TOKEN` if gated |

The designated main validator needs the sidecar, staged data and GPU capacity
once validator-sourced lane-A benches or MarketCast scoring are active. Other
validators replay the signed, manifest-bound evidence for those paths.
Historical optional `[eval] run_benchmarks` and `[scoring] gift_gate_mode` are
separate call sites; they do not describe the v2 main validator's responsibilities.
Satellites verify/replay signed receipts and do not need the training stack.

Deliberately **not** on the list: the TSBench-Forge raw-data bucket. Validators
consume forge data only through the built pool snapshots (channel 5). The raw
catalog + parquet relay (`tsbench-forge-sources` on Hippius — see
[EVAL_POOL.md](EVAL_POOL.md)) is an owner-orchestrator-side secret; keeping it private
is the anti-gaming lever that stops miners from fitting generators to the
held-out pool.

### TSBench-Forge → validator data path

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

## Configuration and compatibility

Owner-signed network overlays can change only the keys supported by the installed
release. Signature, sequence, compatibility, stake lock-in, effective era and
durable local history govern adoption. Manifests and receipts bind the applied
configuration identity; historical replay uses the config in force at its block.
An overlay cannot install code, change architecture bytes or silently replace
release-only base/data pins. See [network configuration](VALIDATOR.md#owner-signed-network-config-dec-ca-0057).

## Trust model

Training is centralised in the owner's trainer and trust in `[manifest]
trainer_hotkey`; that is what makes the controlled experiment enforceable.
The seed derivations, corpus digests and signed contract make every run
reproducible, which is the hook for decentralising training later (validators
or a trainer quorum re-deriving and challenging a manifest).
