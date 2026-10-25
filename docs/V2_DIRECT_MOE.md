# V2 direct MoE competition

This experimental protocol is designed to train one ~220M-active MoE lineage
on miner-funded GPU groups and pay qualifying contributions in emissions.
Rules and rationale: DEC-CA-0050 (lanes) and
DEC-CA-0052 (infrastructure, validation, trust model, owner config, rounds).
Operator runbook: [V2_DIRECT_MOE_OPERATIONS.md](V2_DIRECT_MOE_OPERATIONS.md).

This is a separate experimental protocol. `DirectConfig.validate()` refuses
every nonzero direct activation, even with complete policy settings;
worker/provider qualification and a compatible activation release are required.
The commands below describe isolated direct-runtime operation, not rolling A/B/C
provider selection. See the [compute key guide](MINER.md#compute-provider-keys).

For current rolling mining instructions, use [MINER.md](MINER.md). The
specification below includes configurable requirements and unresolved policy
choices; it is not the reward schedule for rolling lanes A/B/C.

Contents: [how it runs](#how-it-runs) · [mining it](#mining-the-direct-lane) ·
[specification](#specification) · [files](#files-and-migration)

## How it runs

- **Compute.** One pod per gang on the marketplace the payer chose: Lium
  (`provision/core.py`), Vast.ai (`provision/vast.py`) or RunPod
  (`provision/runpod.py`), billed to the payer's own key
  (`cascade/direct/compute.py`). The pod name is the reservation identity, a
  launch is submitted at most once per key, and pod id and SSH host key are
  pinned on first sight.
- **Keys.** `POST /v2/direct/credentials` (hotkey-signed, binds provider and
  key) files the key in that marketplace's vault. Admission needs a key on file
  for every provider the signed terms name. Keys never enter a signed envelope
  or the journal.
- **Transport and storage.** `cascade/direct/pods.py` stages inputs by digest
  over SSH, launches the supervisor detached and harvests checkpoints, re-hashing
  every object before it reaches Hippius. Pods hold no credentials.
- **Rewards.** Emissions only: a payout boundary settles entitlements in the
  journal (`settle_boundary`) and validators turn it into weights
  (`direct/weights.py`).
- **Validation.** Validators fetch the trainer's signed journal from Hippius,
  verify it and replay every decision from the trainer's per-case scores. They
  never load a checkpoint or run a model.
- **Owner network config.** The owner's main validator signs a hash-chained
  config (trainer, journal, burn uid, all-burn switch, ledger and reward
  policies, allowed guards, private policies and panel schedule) and commits its
  digest on chain. Validators pin only the owner hotkey and the config bucket,
  so policy changes supported by the installed protocol can use a new signed
  config. New algorithms and incompatible schema changes still need a release.
- **Auditability.** Every decision carries its scoring binding and chain block,
  uses the panel scheduled for that block, binds publicly fetchable
  checkpoints, and its private panel must be revealed within `reveal_after` or
  the boundary burns.
- **Truth by reproduction.** `cascade-audit direct-moe --reproduce` re-scores
  every decision from the public checkpoints and revealed panels and FAILs any
  divergent score or verdict.
- **Rounds.** The reference schedule uses 6 eras per round (about 3 days on
  the rolling mainnet grid). One cohort
  per round; the round's winner advances the checkpoint only after the round
  closes; no winner means no progression; one payout per era.
- **Stale-king burn.** A king without a promotion of its own in the last full
  round loses `stale_burn` (90 %) of its throne share until its next promotion.

## Mining the direct lane

Everything goes through the `cascade` CLI (`cascade direct --help`). Every
command takes `--json`; see [AGENTS.md](../AGENTS.md) for the output contract.
Point the CLI at the owner config once:

```bash
export CASCADE_DIRECT_OWNER_HOTKEY=<owner ss58> CASCADE_DIRECT_CONFIG_BUCKET=<bucket>
```

1. **Read the round.** `cascade direct status --hotkey <you>` verifies the
   config and journal exactly as a validator does, then shows the round and
   when it closes, the current checkpoint, the king and whether it is stale,
   the open cohort, and your jobs and payouts.
2. **Commit your generator** as for the generator lane (`cascade submit
   --no-fund` or `cascade deploy`); admission needs a revealed commitment.
3. **File a compute key** for the marketplace you want to pay on. The key is
   read from the environment only (`LIUM_API_KEY`, `VAST_API_KEY` or
   `RUNPOD_API_KEY` by default):

   ```bash
   cascade direct deposit-key https://<intake> --provider vast --ref <repo@digest> \
       --wallet-name W --wallet-hotkey H
   ```

   Prefer a scoped key: Vast `instance_read/instance_write/misc`, a RunPod
   Restricted key with pod read/write. Lium keys are account-wide.
4. **Fund a job.** The terms (cohort, anchor, budget, guard, reference) come
   from the published journal and owner config; you choose the gang and the
   spend cap. Inspect first, then send:

   ```bash
   cascade direct fund https://<intake> --ref <repo@digest> --provider vast \
       --profile <profile> --world-size 8 --spend-cap <units> --dry-run ...
   cascade direct fund ... --max-spend <units> --yes
   ```

   It refuses without `--max-spend` and `--yes` (exit 5) and sends nothing.
   The job id is derived from payer, generator and cohort, so a retry is
   idempotent. Only the king may fund the incumbent job (`--role incumbent`).
5. **Follow it** with `cascade direct status --hotkey <you>`. An infrastructure
   failure keeps your admitted rights; a winning checkpoint advances when the
   round closes.

Kings: your generator must keep producing promotions. A full round without one
burns 90 % of your throne share until your next promotion.

## Specification

The owner's requirements for the lane (2026-10). Values called configurable
here are protocol settings that must be supplied before activation, not
defaults to infer.

### Scope and model

Compete directly on approximately 220M active MoE parameters: one accumulating
lineage, broad-generator competition (lane A) and targeted expert contributions
(lane B). The owner reports about 1.5 hours on eight H200 GPUs; verify budget,
configuration and measured times before using that for capacity planning. Do not
infer architecture dimensions from the active parameter count. Historical
backends and receipt replay keep working; the new preset and protocol ship
through explicit versioned contracts and activation gates.

### Training budgets and the lifetime batch ramp

One token is one scalar time-series value, counting all channels. Start with
25B values per broad candidate; 50B and 100B are configurable alternatives for
later calibration, never mixed within a cohort. Reference geometry: context 4096,
C=8. Missing/padded-value accounting and last-batch handling are pinned; an
overshot or incomplete run is never called exactly 25B.

For a declared 1T-value lineage, global batch is 64 below 150B exposure, 128 at
150B–400B and 256 from 400B: with 25B blocks, blocks 1–6, 7–16 and 17 onward
(about 11,921 / 5,960 / 2,980 updates per 25B). The schedule runs over the whole
lineage, not per round. Warmup (if 1 % of 1T) spans the first 10B values; base LR
stays fixed across batch transitions; cooldown is a separate evaluation policy.

Every candidate inherits the pinned initial checkpoint's exposure counter. Only
accepted sequential training advances it; competing branches, retries,
evaluations and cache reuse do not. Thresholds crossed inside a job are handled
explicitly. Routing assignments are counted across all devices and accumulation
passes before balancing. Continuation optimizer/RNG/data state is kept separate
from any evaluation-only cooled artifact.

### Founding, caching and king obligations

The model starts from random weights. Several founding comparisons share the
same random initial state before the first warm-start promotion; a
completed-candidate count and a deadline define the founding window.

Settlement (a possible dethronement) is separate from advancing the training
anchor. A cohort pins initial weights and state, generator versions, budget,
optimizer/LR/batch schedule, randomness and evaluation rules. Challengers fund
their own runs. The incumbent trains once per cohort and its checkpoint is reused
against later challengers; a winning challenger becomes the cached king with its
completed checkpoint. The anchor does not advance on every win.

When the anchor advances, the reigning king funds the next matched incumbent
block: an endpoint trained on the old anchor cannot defend against candidates
that got another full block. Specialist merges that change the anchor also
invalidate the old baseline. Changing evaluation windows means rescoring, not
retraining.

Admission pins payer, generator, anchor, contract, value budget and conditional
reward terms atomically. Dethronement never transfers or cancels an admitted job
or its completion entitlement: if A funds the next block and B dethrones A, A
completes that obligation and B takes the next uncommitted one. A new king takes
current-throne rewards at the payout boundary; rewards for funding accepted
continuation training are a separate portion. No success means no success
reward; infrastructure retries are not scientific failures.

### Broad and expert phases

The two lanes alternate mutations of the shared model: broad training and
decision → bounded specialist window → evaluation of the combined specialist
result → next broad anchor. Specialists open only after the 200B accepted-lineage
floor and routing stability, with separate exposure accounting; an empty window
expires instead of stalling broad training.

Specialist jobs select experts on the current model from an affinity slice
separate from scoring data, freeze the router, balancing bias and all unselected
parameters, and are judged on domain improvement and other-domain retention.
Individual changes and the final combined model are both evaluated before
acceptance; disjoint tensor slices do not imply independent forecast effects.

### Public benchmark guard

Private evaluation decides the duel; GIFT-Eval, FEV, TIME and BOOM are mandatory
public acceptance guards for broad candidates, king-funded continuations,
individual specialists and the final combined specialist result. A funded
continuation is not successful merely because it ran.

Metrics are lower-is-better. For benchmark b with pinned aggregates K_b
(reference) and C_b (candidate) over paired cases, r_b = 1 − C_b/K_b. The overall
estimate is R = Σ w_b r_b with fixed nonnegative weights summing to one (default
proposal: equal quarters). Acceptance needs ALL of:

1. R > 0 (observed improvement, not necessarily significant);
2. LCB_α(R) ≥ −τ_overall;
3. LCB_α(r_b) ≥ −τ_b for every required benchmark (and any mandated metric
   guards), so a material regression cannot hide behind gains elsewhere;
4. the private duel/domain gate and all retention requirements.

Bootstrap is a paired cluster bootstrap: the same cases for reference and
candidate, related series/horizons/windows grouped, normalization and the
combined statistic recomputed inside every draw, shared sources across suites
coupled, repeated forecast seeds treated as repeated measurements. α = 0.05
one-sided and 10,000 draws are exposed defaults, not calibrated tolerances;
τ values must be configured from measurement. Requiring LCB ≥ 0 would demand
superiority, which is not the approved rule. Missing suites, insufficient
coverage, NaN, mismatched cases or failed evaluations are non-accepting, never
dropped and renormalized into a pass.

### Rewards and royalties

Lane A versus lane B allocation is separate from allocation within lane A.
Proposed lane-A split: 50 % throne/duel, 50 % verified accepted king-funded
continuation. Prior-king royalties remain an open choice (50 % or 25 % of the
duel pool). Entitlements are finite and recorded with their admission terms;
dethronement does not erase earned entitlements; no indefinite jackpot.
Capacity outages are never treated as miner default.

### Capacity shortages and recovery

Failure recovery is protocol-visible state: admitted, waiting_capacity,
reserved, launching, running, checkpointing, retry_wait, evaluating,
inconclusive, accepted/rejected and attributable funding/contract failure, each
persisted with idempotency keys before external effects.

- **Gang allocation** of the complete multi-GPU group before launch; partial
  reservations expire. A different SKU or world size only if the contract
  permits it and admission checks pass; global batch, budget and data order stay
  invariant; never silently shrink context, channels, model, precision or steps.
- **No suitable GPUs:** bounded exponential backoff with jitter,
  `capacity_wait` exposed separately, no repeated charges for failed
  reservations, and scarcity pauses attributable deadlines under a bounded
  policy. Invalid keys and exhausted balances are distinguished from scarcity.
- **Backpressure:** bounded queued jobs, active obligations and reserved spend;
  running allocations are never evicted for a new king.
- **Interruption:** resume the same job from its latest verified complete
  checkpoint (all ranks' weights, optimizer, scheduler, router/bias, RNG,
  sampler position, consumed count); partial checkpoints are never resumable.
- **Durability:** immutable checksummed objects, verified manifest, then an
  atomic pointer advance; recovery uses verified remote storage only.
- **Distributed faults:** whole-group restart at a verified boundary; leases
  with fencing so an old worker and its replacement cannot both publish.
- **Evaluation outages:** evaluation queues independently, valid per-case
  outputs are cached by fingerprint, and a missing suite blocks acceptance
  without counting as a performance failure.
- **Economic attribution:** provider failure or capacity wait preserves the
  job's rights; invalid funding or revoked keys follow separate cure deadlines;
  spending above the admitted cap needs renewed authorization; repeated
  provider failure trips a visible circuit breaker.

## Files and migration

Runtime: `cascade/direct/` (`__main__`, `compute`, `data`, `evaluation`,
`evaluator`, `experts`, `fingerprint`, `initialize`, `intake`, `journal`,
`locked_launcher`, `network`, `pods`, `runtime`, `signing`, `storage`,
`supervisor`, `torch_worker`, `weights`). Protocol: `funding/direct_ledger.py`,
`eval/direct_guard.py`, `shared/direct_rewards.py`, `trainer/direct_moe.py`.
Validator: `validator/direct_moe.py`; audit: `audit/direct_reproduce.py`; miner
CLI: `miner/direct.py`; providers: `provision/{vast,runpod}.py`. Support:
`benchmarks/cascade_benchmark/direct_cli.py`, `deploy/direct-v2/`,
`tests/fixtures/direct_ledger_v1.json`.

Use a separate direct ledger and state directory; never rewrite historical
funding stores. v1 journals replay unchanged; new guards use version 2 while v1
receipts keep the old resampling. Back up the SQLite databases with their WALs
from a consistent snapshot. A worker-source change needs a new scientific
contract; never bypass a continuation fingerprint mismatch.
