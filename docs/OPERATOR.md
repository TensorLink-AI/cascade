# Operator runbook

For the subnet operator: the person running the trainer, the funding intake,
the provisioner and the public pages. Miners read [MINER.md](MINER.md);
validator operators read [VALIDATOR.md](VALIDATOR.md); the system design is
[ARCHITECTURE.md](ARCHITECTURE.md).

The trusted **orchestrator** holds the wallet and private test pool, runs the
intake, and signs/publishes training records. **Worker pods** run training and
leave checkpoints for the orchestrator to collect. With v2 CPU shards, separate
CPU workers run the generator before GPU rental; older feeds run it on the pod.
Challenger legs bill the submitting miner's selected provider account. Under v2 king-ladder
rules the king normally funds its own leg too; failed-key recovery, configured
make-good capacity and validator evaluation use operator resources. The scheduler is
rolling intake + era king (DEC-CA-0043, `cascade/trainer/rolling.py`).

The [implemented ladder](../deploy/presets/README.md) supplies Cascade MoE Base variants at 17.53M,
68.92M, 273.64M and 1.23B total parameters. `[v2.training]` selects 1.23B on mainnet and 17.53M
for short testnet runs. The older flat-expert candidate remains available for replay.
The new source uses an explicit update budget; it does not assume a measured
GPU rate. Qualify the worker image, CPU generation image, baseline generator,
GPU memory/throughput and full benchmark pipeline before activation. 56M
remains unspecified and blocked.

## Contents

1. [Before activation](#before-go-live--before-a-testnet-cycle): pins, calibration and launch order
2. [Pods and the worker image](#pods-and-the-worker-image): worker image,
   GPU type, rented pods, operator lanes, sandbox, `cascade-provisioner`
3. [Funded intake](#funded-intake): bring-up, fronting the intake, payer-pod
   credentials, bench, direct submissions, cost model
4. [Backlogs and recovery](#backlogs-and-recovery-across-lanes), including shared limits and operator-billed retries
5. [Security and credentials](#security-and-credentials)
6. [Dashboards and status](#stakeholder-scoreboard): publishing, benchmark sources and missing-data checks
7. [C competition changes](#lane-c-competitions-add-re-weight-end-dec-ca-0056)
8. [Network config overlays](#network-config-overlays-dec-ca-0057): change
   verdict / payout constants without a fleet release

## Before go-live / before a testnet cycle

Mainnet Lane A targets **30 October 2026, 12:00 UTC**. With `from_block = 0`,
`[v2] launch_at_utc` resolves the first era boundary at or after the target
from finalized chain timestamps. A typed block is an explicit override. A
missing timestamp read leaves launch unresolved; it never estimates a block.
Nodes observing the crossing use recent timestamps; a much later fresh node
may need an archive-capable RPC to derive that same historical boundary. A
pruned node that cannot read it fails with an actionable error; point it at an
archive endpoint or type the resolved `[v2] from_block`.

The v2 paid-entry opening is separately pinned as `[v2] commit_open_at_utc =
"2026-10-30T09:00:00Z"`. Its first finalized chain block becomes
`commit_from_block`; an explicit block pin is also supported. A configured pin
does not open intake early: the finalized head must reach it first. Late nodes must
be able to resolve both timestamps, or use both published block pins. This is
also required after a restart. A running intake retains each resolved pin
between requests so it does not repeatedly need state the node has pruned.
The submission rule is an opening floor, not a closing deadline. The updated intake refuses paid A
entries before it opens and refuses older reveals afterward. The updated
trainer holds queued A challengers until v2 is active, without CPU/GPU rental
or consuming attempts. Installing this release therefore pauses new challenger
dispatch under v1; historical replay and king work remain available.

The planned order is A → C → B. C with `takes_from = "lane_a"` can launch with
B disabled or scheduled later. It reserves 25% while A keeps 75%; launching B
later yields 50%/25%/25%. Legacy C taken from B still requires B to be live.
Upgrade every validator and satellite to this release before arming C before B.
While `launch_at_utc` is set and `from_block = 0`, the trainer and validator
services refuse to start unless the v2 configuration the date would activate
loads (for example `[netconfig] owner_hotkey` for `bench_source = "validator"`).
Complete that configuration or clear the date before deploying a release.
`cascade-audit` and satellites resolve the same launch from the chain
(`receipt_calendar_config`): without a chain connection or a typed
`[v2] from_block` the audit reports a `calendar-launch` warning and a satellite
holds, rather than trusting a receipt's own stamp.
The 95% pre-launch burn is independent: `prelaunch-95-r2` must reach the
configured 51% of eligible permit stake, then starts at the next settlement.
Do not type a burn block merely to bypass the readiness vote.

In order. Each item names the code or load check it comes from; anything the
loader checks fails `cascade-trainer --offline` / `cascade-validator
--offline` (both only load the config and print), so run those against the
edited `chain.toml` before shipping it.

1. **Lock file.** `uv.lock` includes the git-pinned `marketcast` extra.
   Use the committed lock with the extras required by the selected role;
   validate lock consistency before building a release. Pods need the training
   dependencies (`train`) and MarketCast workers also need `marketcast`.
2. **Worker image.** `deploy/Dockerfile` installs
   `.[train,hippius,wandb,marketcast]`. Rebuild from the intended release
   and qualify the image, including `--router-slow` and `mkt_finetune`.
   Then run `scripts/repin_worker_image.sh <published-tag> chain.toml`.
   It verifies the published package matches this release and atomically
   updates both training digests, `[round] funded_pod_image` and
   `[round] worker_code_fingerprint`. Set `CASCADE_PYTHON=.venv/bin/python`
   if the project dependencies are installed in that environment. A locally
   built image is not deployable until the registry can serve its pinned digest.
3. **Benchmark sidecar env** on whichever box sweeps lane A (the main
   validator under DEC-CA-0063, else the trainer's pods): the DEC-CA-0061
   rule needs FEV, which is the optional `fev` extra
   (`uv sync --project benchmarks --extra fev --extra time`;
   `benchmarks/Dockerfile` includes both extras), plus data staged with
   `cascade-benchmark-download` and `[eval] gift_gate_data_dir`.
4. **Hardware qualification.** The pyramid contract uses explicit
   `training_steps`; `ref_throughput_tokens_per_s = 0` means unmeasured, not
   zero work. Measure the 1.23B port on the intended GPU, including accumulation,
   full-batch gradients, checkpoint I/O and the four public suites. The research
   run's four-GPU expert parallelism is not a feature of this single-device port.
   Keep the measured configuration and its image pinned before deployment.
5. **CPU generation tier** (DEC-CA-0053, rides `[v2.training] corpus_mode =
   "shards"`): set `[generation] vm_image` (empty in both TOMLs), verify
   Hippius pre-signed URLs, run one testnet leg end to end. Each leg gets
   32 vCPUs for up to 90 minutes (`window_seconds = 5400`), stopping earlier
   at 50B points. Provisioning, digest finalization and upload add time
   outside that generation window. This is operator-side configuration;
   changing the window does not require other validators to update.
   Bootstrap imports every supported generator library and prints its version
   using `scripts/check_generation_runtime.py`; retain that manifest when
   qualifying the production VM image. A dependency import failure must stop
   provisioning before any miner generator runs.
6. **Main validator identity.** Mainnet pins `[netconfig] owner_hotkey` to
   its configured main validator. Testnet keeps a separate unset identity;
   set it when enabling validator-sourced benches there. The
   loader refuses `[v2] bench_source = "validator"` with a resolved block >
   0 unless `owner_hotkey` is set, `[v2] king_ladder = true`, the block is an
   era boundary at/after `[v2] from_block`, and `owner_hotkey` equals
   `[manifest] validator_hotkey` when both are set
   (`cascade/shared/config/v2.py` `check_bench_source`). Mainnet ships
   `bench_source_from_block = "v2"`, so typing `[v2] from_block` without
   `owner_hotkey` fails to load.
7. **Testnet only, optional.** Testnet ships `king_ladder = false` and no
   `bench_source` (trainer benches). To exercise DEC-CA-0055 / DEC-CA-0063
   there, set `king_ladder = true`, `bench_source = "validator"`,
   `bench_source_from_block = "v2"` (or an era boundary after it) and the
   testnet `owner_hotkey` (= testnet `[manifest] validator_hotkey`).
8. **Calibrate the placeholders** (all CONSENSUS; each is overlay-settable
   except where noted):
   - `[scoring] bench_gate_tolerance` (0.01) / `bench_gate_min_win_rate`
     (0.5): `scripts/bench_seed_noise.py` on ≥ 3 seed replicas of the v2
     recipe (DEC-CA-0061).
   - `[v2] bench_sweep_hours` (2.0) / `bench_budget_hours` (6.0)
     (overlay-settable): `scripts/bench_timing.py` on the sweep box, with the
     `[eval] bench_workers` / `bench_paired` you will run (see
     [Sizing the sweep](#sizing-the-sweep-concurrent-workers-paired-passes)).
   - `[v2] ladder_start_tiers`: seed replicas on the private pool (the 0.50
     tier must be rare under noise); keep the lowest threshold at `[v2]
     win_margin_end`.
   - `[lane_b]` / `[lane_c] full_pay_tiers` and `[marketcast]
     full_pay_tiers` (overlay-settable): from real win LCBs (testnet receipts'
     `target_lcb` / `gain_lcb`); lowest threshold at `gain_margin`.
   - `[marketcast] near_dup_tol` (0.02, overlay-settable) and the liquidity
     floor `min_dollar_volume_24h` (release-only): once the recorder shows
     real volumes.
9. **`[v2] from_block`.** Type an era boundary in the future (the loader
   refuses a non-boundary; a passed block judges the eras in between under
   the wrong contract). Trainer and every validator run the v2 release
   BEFORE it. Testnet is armed at 8190000: confirm it is still ahead of the
   head before deploying. Leave B and C `from_block = 0` for the A-only
   launch: lane A gets the full allocation, with its existing ladder and
   burn rules. Announce B separately at an era start ≥ `[v2] from_block`;
   B reserves 25% from its first settlement; A keeps 75% if C is off and
   50% if C is already live. Empty B slots burn.
   `[lane_b] acceptance_from_block = "v2"` makes individual wins unpaid
   qualifications: only compatible deltas accepted at a warm start start
   earning. Testnet leaves it at 0 to preserve settled history. To upgrade
   a live lane, drain legacy pending merges, announce a future era boundary,
   and upgrade trainer and validators before that block. The loader validates
   alignment. Do not disable acceptance while qualified merges are pending;
   the trainer refuses to merge them through the legacy path.
   Preserve the lane-B state file: after acceptance activates, a missing
   state is rebuilt from the full settlement chain back to lane launch (or
   the saved marker), because accepted-delta identities outlive payout TTLs.
   Historical rounds fold oldest first; a TTL-sized replay cannot safely
   reconstruct which expert deltas have already entered the model.
   Acceptance benchmarks run in a background job. The new init waits while
   intake and status ticks continue; no new-init leg starts early. Staged
   results consume pending deltas only when the scheduler adopts that exact
   era/base, so obsolete jobs and process restarts leave the queue intact.
   `[lane_c] from_block`, if typed, must be an era boundary with B live by
   then; mainnet C takes its share from A (`takes_from = "lane_a"`), giving
   A/B/C = 50/25/25. Testnet keeps its existing B-funded C split. The settlement
   AT a lane's launch boundary still belongs to the preceding era.
10. **MarketCast** (only when arming `[marketcast] from_block`): start
    `cascade-pool mkt-record` (and `mkt-record-alpha` if `alpha` assets are
    listed) at least `snapshot_days` (21) before the block — the loader
    cannot see the recorder; install the `marketcast` extra on the main
    validator; the loader then requires an era boundary ≥ `[v2]` and
    `[lane_c] from_block`, the validator bench source in force by then,
    `owner_hotkey`, a commit-pinned `base_ref` + `base_sha256`, and a built
    `[marketcast.data_sources]` entry for every class that lists assets
    (`cascade/shared/config/marketcast.py` `arming_problems`,
    `check_marketcast_arming`). See [MarketCast](#marketcast-dec-ca-0064-snapshot-feeds-validator-gpu).

## Pods and the worker image

The transport is plain SSH (`cascade/trainer/remote.py`), so any
SSH-reachable GPU host looks the same to the trainer.

### Build and pin the worker image

```bash
docker build -f deploy/Dockerfile -t <registry>/cascade-worker:<tag> .
docker push <registry>/cascade-worker:<tag>
docker inspect --format='{{index .RepoDigests 0}}' <registry>/cascade-worker:<tag>
```

Pin the digest in `chain.toml`, never a mutable tag:

```toml
[training]
train_image_digest = "<registry>/cascade-worker@sha256:<digest>"
```

Every pod receives the same digest at launch as `CASCADE_TRAIN_IMAGE_DIGEST`
(a container cannot introspect its own OCI digest). With the pin set,
`cascade-train-worker` refuses a run whose runtime doesn't report it, the
trainer probes each pod's worker runtime before dispatch, and `cascade-audit`
Tier 2 uses the match to decide when a byte-exact checkpoint comparison
applies. Rented pods boot `[round] funded_pod_image`; pin it as described in
[Credentials on payer pods](#credentials-on-payer-pods).

### GPU type

The GPU type is chosen **per leg**: the trainer rents the cheapest fitting
executor across `[round] funded_pod_skus` under the price caps
(`funded_sku_per_leg`), with `[training] expected_gpu = ""`. Equal compute
comes from the fixed series-point budget, not from matching silicon.

Pinning one SKU (`expected_gpu = "<exact torch.cuda.get_device_name string>"`)
is a LOCKED contract term — a coordinated validator upgrade — and buys
byte-exact re-derivation of checkpoints on that SKU + image.

### Miner provider lists and recovery

Rolling submit/fund supports an ordered Lium/Shadeform/Vast.ai/Runpod key list. Upgrade intake
and trainer together, then confirm `status/v2.json` advertises
`compute.rolling_providers`. Miners use `--provider-key-env PROVIDER=ENV_NAME`;
legacy `LIUM_API_KEY` requests remain valid. `cascade credentials` updates the
list without queuing work. The signed bundle binds provider names, order and
key digests; keys never enter public queues or rental ledgers.

Read-only capacity selection preserves GPU/hourly/estimated-leg caps. Once a
create is attempted, no second provider is tried on an uncertain result.
Existing allocations are discovered first, including via retained credentials
from an earlier list. Each rental writes a provider and `payer_credential_id`
into its ledger before use. Credential snapshots live under the payer vault's
`leases/` directory, with the same sealing key, permissions and TTL. Preserve
that directory during backups/restarts until all rentals are confirmed gone.
Opaque Shadeform UUIDs are resolved alongside instance names during orphan
cleanup. An inaccessible account blocks new spending; surface it for repair.
A successful authentication check does not establish available credit.

Vast.ai follows the official `vast-cli` request shapes: offers by `gpu_name`
(spaces, e.g. `RTX 4090`), on-demand, verified, priced by `dph_total` under the
hourly and per-leg caps; creates with `client_id = "me"`; the paged
`/api/v1/instances/` list. An explicit refused create is capacity; a lost,
malformed or contradictory response, timeout or 5xx keeps the create unresolved.
Cleanup requires a successful listing with valid rows, counts and complete
pagination; an API error or partial listing cannot establish that a pod is gone.
Qualify it on a real account before relying on it: GPU
name strings for A6000/H100, listing lag after create, and that deletion stops
billing. Runpod is accepted by rolling intake and cleanup; its on-demand
adapter refuses creates with an hourly or estimated-leg cap. Provider selection
skips that capped account and may choose another listed provider before any
create. With no compatible account, status reports `price_cap_unavailable`.
Only an explicitly chosen GPU with `--max-price 0` uses Runpod's uncapped path.
Do not interpret this as a hard account or total-dollar budget.

Lium documents key budgets configured by the human. Shadeform has a native
per-instance auto-delete spend threshold: applicable `funded_max_leg_cost_usd`
is sent to the create API, rounded down to cents. It covers that instance only.
Vast and Runpod hard per-key budgets remain unconfirmed; permissions, alerts
and balances are not substitutes. Discovery publishes these distinctions in
`compute.spending_controls`, without claiming to verify any key's allowance.

### Pods the trainer rents itself

The trainer rents miner-funded pods on Lium, Shadeform, Vast.ai or Runpod
(`cascade/provision/funded.py`, `cascade/trainer/loop.py`):

* **Payer pods**: one per funded challenger leg, billed to the miner's own
  provider key from the sealed vault. Torn down after harvest and any required
  trainer-sourced benchmarks. Validator-sourced A benchmarks run separately.
* **King training**: under v2 `king_ladder`, normally billed to the king's
  own key; earlier rules and failed-key/operator recovery can use the operator
  path (`[round] funded_king_rent`). Seed epochs can reuse a trained king
  across eras. Trainer-source benchmark verification can retain an operator
  king pod; do not assume a new operator-funded king rental every era.
* **Operator-billed legs**: only with `[round] funded_operator_fallback =
  true` (shipped `false`). Then a leg still waiting for marketplace capacity
  takes a free operator lane from `hosts.toml`, and a leg whose payer key has
  expired runs on an operator lane or, with none on file, on a pod the
  trainer rents on the operator's key. With the flag off, payer legs never
  run on operator hardware, and a leg whose key has expired is released as
  `auth`, unburned.

Every one of these pods is **isolated and credential-free** (details:
[Credentials on payer pods](#credentials-on-payer-pods)). Public Hub
generators and warm-start inits pull anonymously; private (vault)
submissions are staged onto the pod as a ZIP. Pod names are
deployment-scoped (`cascade-n<netuid>-<round>-funded-<slug>`) so co-hosted
deployments on a shared payer account never reap each other, and the trainer
reconciles and reaps them through its write-ahead ledger
(`<work_root>/funded_pods.json`, swept at trainer startup; pods of legs still
in flight and the current eras' king pods are kept and re-attached). The
provisioner's orphan reaper never touches them.

The trainer needs `LIUM_API_KEY` (operator account) and a `hosts.toml`
whose first `final`/`any` entry is the pod profile every rented pod mirrors
(image paths, the orchestrator SSH keypair `<key_path>` + `<key_path>.pub`).
A profile-only entry (`profile_only = true`, or a loopback address) is
enough; it is never a dispatch target.

### Operator lanes (`hosts.toml`)

Hand-rented GPU hosts serve the operator-billed legs above. Point the
trainer at a hosts file; it is re-read on every use, so a lane added later is
picked up without a restart:

```bash
cascade-trainer --remote-hosts hosts.toml   # + wallet/chain flags
```

```toml
[[host]]
name          = "a6000-oslo-g0"
host          = "203.0.113.10"
user          = "root"
key_path      = "~/.ssh/trainer_orchestrator"   # the PRIVATE key
remote_python = "/root/cascade/.venv/bin/python"
workdir       = "/root/cascade"                  # matches the image WORKDIR
cuda_device   = "0"
isolated      = true
```

Two GPUs on one box → one entry per card (`cuda_device = "0"` / `"1"`); see
`scripts/remote_hosts.example.toml`. A lane joins the pool only after a gate:
SSH answers, the pod runs this release's worker image, and the host clears the
per-SKU bench floor. A lane whose SSH is refused is quarantined.

Keep `isolated = true` on every lane: miner code runs there. `forward_env`
(forwarding `HIPPIUS_*` credentials so the pod pushes its own checkpoint)
remains only as a deliberate exception, and the trainer logs a WARNING at
startup for every non-isolated host that forwards secrets.

Launching a lane: run the image with SSH exposed and your orchestrator public
key injected, e.g. on a base GPU pod with `docker` + `nvidia-container-toolkit`:

```bash
docker run -d --gpus all -p 22:22 \
  -e SSH_PUBKEY="ssh-ed25519 AAAA... trainer-orchestrator" \
  -e CASCADE_TRAIN_IMAGE_DIGEST="<registry>/cascade-worker@sha256:<digest>" \
  <registry>/cascade-worker@sha256:<digest>
```

### Lock down generator execution

Pods run **miner-controlled generator code**. Configure the sandbox in
`chain.toml [generator]`:

```toml
[generator]
# EITHER: subprocess sandbox that REFUSES to run when the host cannot provide
# a network namespace. Needs unprivileged user namespaces on the pod
# (`unshare --user --map-root-user --net true` must succeed).
sandbox_strict = true

# OR: kernel-enforced container sandbox — docker/podman with --network=none,
# --cap-drop=ALL, no-new-privileges, read-only rootfs, tmpfs workdir and
# memory/pids/cpu limits. Needs a container runtime and docker.sock access.
sandbox_mode   = "container"
sandbox_image  = "<registry>/cascade-worker@sha256:<digest>"
sandbox_python = "/root/cascade/.venv/bin/python"
```

Never disable the checked-in strict sandbox in production (`sandbox_mode =
"subprocess"`, `sandbox_strict = false`): on a host without unprivileged user
namespaces it silently leaves only the in-process socket guard between miner
code and the network. The same rule holds on the orchestrator, which holds
the private eval pool and the trainer wallet: generator code never runs there
without `sandbox_mode = "container"` or `sandbox_strict = true`.

Two layers enforce this (`cascade/shared/sandbox_policy.py`). A **load
check** refuses any config knob that would run generator code on a weak
sandbox (today only the retired `[round] dedup_probe_series > 0`, which
nothing reads). A **runtime guard**: `cascade-trainer` and
`cascade-validator` mark themselves at start as holding secrets, and every
generator entry point (`trainer/sandbox.py`, `trainer/corpus.py`) then
refuses in-process generator execution always, and a sandbox child unless
the sandbox is hardened (`SandboxPolicyError`, an operator fault, never booked
against the miner). Payer / worker pods, the CPU generation tier and miner
tools never mark themselves, so nothing changes there.

**No-egress lanes.** A credential-free worker needs outbound access only to
the generator / init sources it pulls from (the Hippius registry and S3) plus
inbound SSH from the orchestrator. Deny everything else at the pod boundary:

```bash
for host in registry.hippius.com s3.hippius.com; do
  for ip in $(getent ahostsv4 "$host" | awk '{print $1}' | sort -u); do
    iptables -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT
  done
done
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT  # SSH replies
iptables -P OUTPUT DROP
```

Providers with security groups: express the same policy there — 22/tcp
inbound from the orchestrator only, outbound only to the storage endpoints.

### `cascade-provisioner`

`cascade-provisioner` (`cascade/provision/`, config in
`scripts/provision.example.toml` / `deploy/provision.*.toml`, unit
`deploy/cascade-provisioner.service`) runs beside the trainer. It never calls
the trainer; the two meet only through files and stores. It holds the
marketplace API keys, read credentials for the manifest store and the
orchestrator SSH keypair — never the wallet. Each cycle it:

* **Eval stage.** When a new manifest appears, rents one GPU pod for the
  validator's heavy evals and publishes it to a separate `eval_hosts_path`
  file the validator re-reads per eval (never the trainer's hosts file). It
  tears the pod down when the settlement's receipt publishes, a newer
  manifest supersedes it, or the TTL fires. No capacity or a failed health
  gate means the validator evaluates locally — degraded, never lost.
* **Hosts file.** Publishes the trainer's hosts file atomically: the
  `static_hosts` fragment (re-read every publish; a broken edit keeps the last
  good copy) plus any `[[host]]` entries it does not own, preserved verbatim.
  Ownership is by address. Do not name a hand-rented pod in the provisioner's
  own shapes (`cascade-<provider>-<8 hex>-g<n>` lanes, or
  `cascade-<digits>-(heat|final|eval)…` on the provider), or it is treated as
  the provisioner's and dropped or reaped.
* **Orphan reconcile.** Terminates live provider pods tagged with its own
  name shape that its ledger does not own, so a crash between launch and
  ledger-save never leaks a billing box.
* **TTL backstop.** Every pod it rents dies one epoch after rent no matter
  what; `max_spend_per_round` caps the worst case before anything is rented.
* **Adopted-pod re-gate.** After a restart, pods adopted from the ledger are
  re-checked by the health gate before reuse. The gate's `fresh_boot` check
  also fails a recycled container (one that already pins
  `CASCADE_TRAIN_IMAGE_DIGEST` or has a non-empty `_train_work`).
* **Grid activation.** Reads the DEC-CA-0045 on-chain activation so its epoch
  grid switches at the same block as the trainer and validators.

Its config accepts and ignores the tables of the retired per-round fleet
(`[provisioner.heat]`, `[provisioner.final]`); the trainer rents the king and
every leg itself. On funded pods the provisioner must never: rent them on
the operator key (`provision/funded.py` has no such path — keep it that way);
forward shared credentials (`HIPPIUS_*`, `HF_TOKEN`, `WANDB_*`); auto-retry a
rent in place (every attempt spends the miner's budget — retry cadence lives
on queue requeues, `should_recover` + attempt caps); or touch any
`cascade-n<netuid>-…` pod.

## Funded intake

Design rationale: DEC-CA-0036 (funded legs, intake, vault, direct
submissions) and DEC-CA-0043 (rolling intake). The miner side is documented
once, in [MINER.md §5](MINER.md#5-submit) (submitting and funding) and
[§8](MINER.md#8-failure-classes) (what re-queues, what spends the
submission); keep that table in sync with `classify_funded_worker_failure`
and the queue's terminal states when either changes.

| piece | where |
|---|---|
| Payer-key vault (TTL, sealed at rest, hydrate-for-teardown) | `cascade/funding/vault.py` |
| Rent-fault taxonomy (auth / rate / capacity / infra, no-burn rules) | `cascade/funding/faults.py` |
| Funded queue (reveal-block seniority, one live entry per hotkey, `in_flight` state) | `cascade/funding/queue.py` |
| Intake service (`cascade-intake`, signed `X-Lium-Api-Key` POST, direct submissions) | `cascade/funding/intake.py`, `main.py`, `store.py` |
| Rolling scheduler (admission, per-leg GPU choice, settlements, era king) | `cascade/trainer/rolling.py` |
| Per-payer rentals, verified teardown, write-ahead ledger `funded_pods.json`, payer-key reap | `cascade/provision/funded.py`, `cascade/trainer/loop.py` |
| Duplicate registry (earliest commit keeps; king + published champions included) | `cascade/trainer/dedup_registry.py` |
| Checkpoint guard (a checkpoint is data, never code) | `cascade/eval/checkpoint_guard.py` |
| Transparency: roster `funded/round-<id>.json` + `funded/latest.json`, `cascade queue`, tier-0 `funded-roster` audit check | `trainer/rolling.py`, `miner/cli.py`, `audit/checks.py` |
| Miner commands (`cascade submit`, `cascade fund` / `--withdraw`, `cascade queue`, `cascade iterate`, `cascade direct`; `--json` contract) | `cascade/miner/{cli,iterate,direct,output}.py` |

### Backlogs and recovery across lanes

Validator settlement recovery uses `<state_db>.settlement.json` as an undo
journal across champion, Cascade, lane-B and lane-C files. Keep these files
together on durable local storage. An interrupted fold is restored before
replay; a committed fold retains its pending receipt/weight follow-up. Disk
write failures hold progress. Do not remove a prepared journal to bypass an
error. The validator entrypoint holds `<state_db>.lock` so two processes cannot
write the same bundle. A fresh bundle still needs the historical evidence
needed to reconstruct consensus state; it is not a substitute for backups.

After a restart, persisted open B/C legs count as operator-side losses. A
verified pending upload can still be reused; a required retrain follows the
`operator_billed` rules below. An expired lane-B leg keeps its hotkey busy
until its worker returns; scientific expiry does not terminate a GPU process.

Required lane-B public suites must have `min_public_items` usable pairs for
both CRPS and MASE. Different known data revisions, one missing revision pin,
missing suites or insufficient required evidence hold trainer settlement or
merge preparation. Historical evidence with neither revision pin remains
readable; it does not prove the data versions were identical.

Once lane B, lane C or validator-sourced lane-A benchmarks are active,
every boundary with a valid era king can publish a king-only settlement.
Lane A needs no new challenger for B/C results, benchmark acknowledgements,
confirmations and reward clocks to advance. The existing signed manifest
format and validator checks apply; an unavailable king still holds publication.

`[round] funded_field_cap` now also caps shared admission of miner-funded
A/B/C jobs preparing data, waiting for capacity or training. Waiting jobs
enter FIFO at this shared gate; per-lane seniority and target caps still
apply. The king has reserved admission and gets first GPU acquisition.
`funded_field_cap` is overlay-settable. CPU provisioning has a separate
`[generation] max_parallel_jobs` cap (2 in both TOMLs), so a large GPU queue
cannot launch unbounded generation VMs. Admission waits do not rent a GPU.
These are per-trainer-process controls; run one authoritative scheduler.

GPU selection and launch stay serialized until the executor is claimed.
Once its identity is known, boot, SSH and runtime checks release that
selection lock, allowing other jobs to acquire different GPUs. Unknown
executor identities retain the lock conservatively. All three lanes use
their own queue and actual leg budget throughout rent and harvest; lane B
uses its `leg_hours` in the orchestrator as well as the worker.

An operator/provider failure after allocation, or a completed leg lost to
an era/routing change on our side, marks that entry `operator_billed`.
The flag persists across requeues and restarts and is visible in queue
JSON, with `operator_faults` counting each loss. Without operator capacity,
every allocation that billed the miner and delivered nothing is a loss of its
own — a lemon that never became ready, a pod released for a stale image or a
slow bench — and a replacement is rented within the same attempt only while
the losses stay under the cap, so a systemic fault bills the miner at most
`OPERATOR_FAULT_RETRIES` allocations in total before the pause (not three
pods per attempt). A retry first reuses any
verified pending upload. When an operator lane or the capped operator rental
policy below is configured, the retrain runs there (B/C keep their role, base
and recipe/selection). Otherwise the flag is cleared and the leg rents a new pod
on the payer's key; a pod whose release is unconfirmed still blocks that rent.
After `OPERATOR_FAULT_RETRIES` (3) losses the entry pauses with an `operator`
failure instead of billing the miner again; fix the cause, then (with the
trainer stopped) remove `operator_faults` from that queue entry to resume it.
The flag is a recovery billing restriction,
not a refund of already consumed GPU time. No-capacity failures before an
allocation remain ordinary retries without an attempt burn.

### Outage protection, alerts and capped recovery

The checked-in mainnet and testnet presets allow one active miner job per
chain-resolved coldkey across A/B/C. The shared gate covers generation,
capacity waits, training and fallback; king work has reserved admission.
Unknown or stale ownership waits for fresh chain metadata. An already verified
upload can finish without acquiring another compute slot.

Both presets also avoid a GPU type for three hours after two distinct pods of
that type fail within one hour. `gpu_health.json` preserves this across restarts.
Account, rate-limit and sold-out responses do not count as hardware failures.
A first transient host fault gets a 30-minute exclusion; repeated host faults
get three hours, while a rejected stale runtime can warrant 24 hours.

The independent operator watchdog records low or unknown account balance,
oldest queue waits by lane, held king settlements, repeated king failures,
GPU cooldowns and chain timestamp drift. A missing Shadeform balance reader is
reported as unknown. `operator_health.json` stores active/resolved events and
an outbox; unsuccessful delivery retries the same event ID. Set
`ops_alert_webhook_env` to the name of an environment variable containing the
HTTPS receiver URL. Without it, events remain available locally and in logs.
Receivers should deduplicate IDs. Public `operations` telemetry includes its
sample time and stale flag, never account keys or exact balances. A hung sampler
does not spawn additional samplers. An advancing but stale RPC head is
reconnected and refused until its timestamp is fresh; block heights still
determine eras and settlements.

These `[round]` controls require the matching installed release:

| Controls | Purpose |
|---|---|
| `ops_max_active_per_coldkey` | Shared owner limit; 1 in the presets. |
| `ops_gpu_failure_threshold`, `ops_gpu_failure_window_seconds`, `ops_gpu_cooldown_seconds` | Persistent GPU-type outage detection; 2 / 3600 / 10800 in the presets. |
| `ops_alert_interval_seconds`, `ops_alert_webhook_env` | Watchdog cadence and optional delivery; interval 0 disables it when the operator pool is off. |
| `ops_low_balance_usd`, `ops_oldest_wait_seconds`, `ops_settlement_hold_seconds`, `ops_clock_drift_seconds` | Alert thresholds; timestamp freshness also gates the trainer's chain head. |
| `ops_operator_providers` | Ordered operator accounts, using `LIUM_API_KEY` / `SHADEFORM_API_KEY` in the trainer environment. |
| `ops_operator_leg_cap_usd`, `ops_operator_daily_cap_usd` | Required positive limits before enabling the operator pool. |
| `ops_paid_fallback_after_seconds` | After a known payer-capacity shortage and this much total queue age, permit capped operator rental. Zero disables it. An explicit miner GPU choice is preserved. |
| `ops_king_backup_after_seconds`, `ops_king_backup_cap_usd` | Delayed operator-paid king replica and its additional cap; zero delay disables it. |

Provider fallback and king replicas ship disabled. The operator must supply
per-leg and daily limits before enabling them. The spending journal reserves a
conservative amount at the hourly price ceiling for training, startup, harvest
and any declared hold. Closed attempts still consume their reservation; retries
cannot reset the daily or leg cap. These are allocation limits, not a guarantee
against a cloud provider continuing to bill after an unsuccessful deletion.
The watchdog reaps expired allocations while the trainer is running; independent
provider-side billing alerts remain necessary for a complete control-plane outage.
The capped operator pool requires that watchdog to remain enabled.

`operator_spending.json` binds each create intent to its original account and
physical instance ID. Operator credential snapshots stay under
`operator_credentials/`, using the configured vault sealing key and local
permissions. Preserve both directories with the rental ledger. An uncertain
create retains its reservation and blocks a second create for that job, even
when a subsequent listing is empty. Resolve that outcome with the provider;
do not delete journals to reset spending. An unused failing account can be
skipped only when no unresolved or legacy allocation needs inspection there.
Shadeform deletion is checked against physical IDs, not instance-name prefixes.

Miner-funded rentals keep the same fail-closed rule in the payer vault's
`create-intents/<lease>.json` (non-secret: scope, provider, state). Such an
intent holds only that round's leg; it never expires and no command clears it.
After confirming with the provider, on the miner's account, that the create
did not allocate (or after terminating what it did allocate), delete that one
file. Never delete it while the provider might still list the allocation.

The king replica uses the same pinned generator, seeds, contract and init on a
separate allocation. Where the primary's allocation is known, the backup must
use another provider or GPU type. The first valid training completion wins;
evaluation scores do not choose the replica. The choice is durable, and late
callbacks only clean up their own allocation. Operator backup work does not
establish a king-key lapse; a proven primary-key failure retains the existing
funding/slash classification. King-only settlements still require a valid king
result; redundancy reduces that dependency's failure rate.

### Qualified training continuation

Continuation ships disabled and is a release change. Declare
`continuation_protocol = "cascade-continuation/1"` in the applicable
`[v2.training]` or `[marketcast]` contract, qualify the worker image, then set
`[round] ops_continuation_image_digest` to that image's immutable digest and
`ops_continuation_interval_seconds` to a positive cadence. Detached dispatch is
required. Size `ops_continuation_max_bytes` for the full model, optimizer,
averages and any pinned MarketCast data; the default bound is 16 GiB. The
empty protocol preserves earlier training digests. An overlay does not install
this code or activate a release-only contract field on an older validator.

A snapshot follows a complete optimizer step and contains weights, optimizer
state, random generators, routing/averaging state, consumed data position and
remaining active training budget. It uses JSON and safetensors, with atomic
generation pointers and digest checks. The orchestrator periodically pulls the
committed generation to `_continuations/` and restores its last verified copy
on a replacement pod. Failed transfers leave the previous copy intact.
Credentials never go to the worker. Copying a warm-start weights file alone
does not resume a leg.

The protocol also records a yielded shard series before returning it to the
trainer. Earlier shard streams account for it only on the next iterator read;
that historical behaviour remains available for replay. The declared protocol
makes the final consumed series part of the new receipt digest without reading
an extra series. Trainer, local scoring and audit replay use the same rule.

Recovery replays completed input batches without optimizer updates and checks
their digest before restoring the saved random state. MarketCast also restores
the original training snapshot instead of downloading revised bars. A different
base, recipe, selection, seed, contract, dtype or qualified runtime is refused.
Exact continuation requires the same GPU model; moving to an unqualified type
must not silently reuse optimizer state. Retried work retains the operator-billed
restriction. Before the first durable off-pod copy, a lost pod still requires a
full restart; later losses can repeat work since the last copied boundary.

This protocol counts active training wall across attempts. Snapshot I/O,
reconstruction and replacement startup are recovery overhead; they do not add
optimizer steps or token budget. Qualify the cadence, transfer size, available
local disk and provider hold before activation. Completed-job mirrors need
operator retention management; the per-snapshot bound is not a total disk quota.

Run this offline smoke check in an empty temporary directory:

```bash
python -m cascade.trainer.qualify_continuation --tiny --device cpu --work-dir /tmp/cascade-continuation-check
```

It compares fresh-process uninterrupted training with a worker that exits
immediately after a durable snapshot and is resumed in another process. It
checks cold/warm A, selected-expert B and annealing. The JSON report explicitly
labels the tiny CPU scope. For qualification on an already available GPU, omit
`--tiny`, select `--device cuda` and the deployed `--dtype`, and use a new work
directory. This command neither rents nor accesses storage services. It does
not qualify MarketCast's real Timestral base, cross-provider SSH lifecycle,
long-budget throughput, concurrent rentals, or recovery after a complete
orchestrator outage. Exercise those on testnet before enabling the policy.

### Lineage measurements and state upgrades


CPU recovery resolves Targon launch intents using the documented,
paginated exact-name VM listing. Empty/error/malformed listings retain the
intent and prevent a second launch. Targon does not sweep unrelated workload
names. See the [provider API](https://docs.targon.com/api/workloads).

Funded B and MarketCast C recovery can use configured operator GPU lanes
through the same admission gate and lane pool as A. The fallback retains B's
selection/base or C's recipe/base and harvests through the existing dispatcher.
It never rents an additional operator pod. Enable the existing
`[round] funded_operator_fallback` and provision sufficient operator final lanes
before enabling recovery. With no lanes, recovery stays pending; a full pool waits
at most 30 minutes before requeuing. This is make-good compute, not a monetary
refund or a guarantee of recovery throughput.

Worker checkpoints include `training_measurement.json`, bound by the
immutable checkpoint pointer. It binds generator, corpus, effective contract,
warm start and measured training-budget points. The lineage counter excludes
the fork-anneal tail and caps batch overshoot at the contracted budget. It uses
the contract's denomination (`series_points` and scalar channel points are
different). Each completed seed epoch is durably registered before the rolling
scheduler retires it; counter and epoch receipt update together. Replay after
a scheduler crash cannot count the epoch twice. Missing/incompatible evidence
stays pending and is retried; unsigned dashboard telemetry cannot open the gate.
These are trainer-produced measurements, not independent hardware attestation.

Back up the **complete** trainer and validator state bundle before upgrading.
On first load, the old budget-proxy count moves to `legacy_budget_tokens` and
does not satisfy a closed gate's measured threshold. A gate already latched
open is not automatically closed. Old checkpoint measurements are not invented
from their budgets: reconcile historical epochs from retained evidence or wait
for newly measured epochs. Review this explicitly if upgrading a live network.
Discovery exposes the meter type, legacy estimate and unmeasured epoch count.
Build/pin a matching worker image and code fingerprint before deploying; old
workers produce no new measurement and will leave epochs unmeasured.


### Bringing it up

Submission tracking requires the matching CLI, intake and trainer release.
Run `cascade discovery-check --json`; `submission_tracking` must pass. The
trainer publishes a fresh `status/v2.json` heartbeat at its existing interval
even when the block/policy body is unchanged. Queue responses carry `as_of` and
use `Cache-Control: no-store`; C's trainer status now includes `funded_queue`
alongside the existing count summary. Agents can use fresh public queues or
request intake queues directly with `cascade status --hotkey ... --lane ...
--ref ... --intake ... --json`.

Each admitted attempt has a unique `attempt_id` and bounded `phase_history`
(16 events), `phase_at`, requeue count and retry timestamp. Updates require the
same hotkey/ref, admission block, funding time and attempt ID, so late callbacks
cannot relabel a new attempt. Stages describe preparation, compute waits,
allocation/startup, training dispatch, harvest and settlement wait. They contain
fixed reason codes; provider exception text and credentials are not published.
Telemetry write failures cannot change the training outcome. Phase changes are
not worker heartbeats or token progress; a healthy intake does not prove a worker
is alive. Old `in_flight` entries remain explicitly unclassified until new work
reports a stage. Timelocked pending reveals remain count-only.

B and C also publish bounded `submission_progress` rows keyed to the funded
attempt. B persists qualification observations separately from fault outcomes;
only committed acceptance is reported as accepted. C retains exact trained
entry receipt observations, including their original boundary, across omitted
later receipts and restarts; older receipts cannot roll observations back.
Neither projection sets weights or changes qualification/merge/slot rules.
The CLI requires a fresh scientific feed independently of the intake, and
reports `reward_status=not_verified`. These are unsigned observations, not a
replacement for signed receipts or proof of on-chain payment. Old entries
without an attempt binding remain unknown rather than inheriting a result.

Publish the updated `llms.txt` and miner guides with the release. This metadata
does not change emission, admission or scoring rules and cannot resolve an
actual provider capacity shortage. Qualify the recorded stages, automatic retry
and stale-feed behavior through an isolated lifecycle before publishing live.

1. **Intake.** `cascade-intake --queue-path <work_root>/funded_queue.json
   --vault-dir <work_root>/payer_vault --submission-dir <work_root>/submission_vault
   --chain-toml chain.toml` behind a TLS terminator (see
   [Fronting the intake](#fronting-the-intake)). `--queue-path` is REQUIRED
   and must be the exact path the trainer resolves for `[round]
   funded_queue_path` (work_root-relative) — there is deliberately no
   default, because a CWD-relative fallback split-brains the queue (funds
   return 202 but never enter). Likewise `--vault-dir` = `[round]
   payer_vault_dir`, `--submission-dir` = `[round] submission_vault_dir`, and
   `--ttl-hours` = `[round] funded_entry_ttl_hours` (36 h). Queue writes are
   flock-serialised via a sibling `funded_queue.json.lock` — keep both on one
   local filesystem (flock over NFS is not a lock). Loopback bind is the
   default; the key header must never cross the wire in clear. The vault and
   submission dirs stay off git, off rsync'd pod trees, and off backups that
   leave the box. NO `--trust-refs` and NO `--no-require-signature` on
   mainnet, ever.
2. **Vault seal key.** Set `CASCADE_VAULT_KEY_FILE` (32 raw bytes or 64 hex,
   e.g. `head -c 32 /dev/urandom > /root/.cascade/vault.key; chmod 600`) to
   the SAME file for the intake and the trainer.
3. **Trainer config** (`[round]`, trainer restart; no consensus impact):
   `funded_pods = "rent"` (every challenger leg on its payer's own key),
   `funded_king_rent = true` (the trainer rents, ledgers and sweeps the
   operator king pod itself; requires `[training] expected_gpu = ""`),
   `funded_sku_per_leg = true` with `funded_pod_skus` (the allowed GPU types;
   each leg rents the cheapest fitting executor), the price guards
   `funded_max_leg_cost_usd` / `funded_max_price_per_hour` with
   `funded_sku_wall_seconds` (measured wall per SKU; drives the per-leg cost
   cap and the latest safe start — re-measure when the budget changes),
   `funded_cpu_blocklist`, `funded_host_bench_floor`, `funded_field_cap`
   (legs in flight), `funded_pod_image` + `worker_code_fingerprint` (re-derive
   on every image re-pin), and `funded_pod_checkpoint = "harvest"`.
4. **Publish the intake URL** to miners and watch: `cascade queue --intake
   <url>`, `funded/latest.json`, `cascade-audit latest` (funded-roster check)
   and the trainer log's admission / SKU lines.

### Fronting the intake

The intake runs on the orchestrator — the box holding the trainer wallet and
the private eval pool — so exposing it raw is not an option. The division of
labour:

**The front proxy does volumetric defence.** Terminate TLS in nginx/caddy (or
put the DNS behind an edge like Cloudflare) and configure, at minimum:

- per-IP connection and request-rate limits (nginx: `limit_conn` /
  `limit_req`; the fund/withdraw endpoints are a few requests per miner per
  day — single-digit req/s per IP is generous),
- header/body read timeouts (`client_header_timeout`, `client_body_timeout`),
- a body size limit matching the intake's ZIP cap (`client_max_body_size` =
  `--max-zip-mb`, or ~1 MB if the intake is funding-only),
- proxy buffering ON, so a slow client dribbles at the proxy, not at a
  handler thread.

**The intake keeps in-app backstops** so a missing or misconfigured proxy
degrades gracefully instead of taking the orchestrator down:

- `--request-timeout` (default 30 s): per-socket read/write timeout — a
  slowloris client gets its connection closed, not a parked thread (the
  stdlib serves with NO timeout by default).
- `--max-connections` (default 64): concurrent-handler cap; over it, new
  connections get an immediate `503` + `Retry-After` and are closed without
  spawning a thread, so a connection flood costs one small write each.
- `--chain-timeout` (default 30 s): deadline on the reveal poll — a hung
  substrate websocket serves the stale reveal table instead of pinning every
  handler slot behind one dead chain connection.
- `--request-deadline` (default 120 s): whole-connection wall clock enforced
  by a reaper — a drip-feed client sending one byte per interval defeats a
  per-op timeout but not this.
- `--max-uploads` (default 4): concurrent submit-body buffers (each is up to
  the ZIP cap in RAM).
- `--per-ip-rpm` (default 60) / `--per-ip-connections` (default 8): a
  sliding-window request cap and a concurrent-connection cap per client IP
  (429 + Retry-After; `/health` exempt). Behind the proxy every connection
  arrives from the proxy's IP, so pass `--trusted-proxy <proxy-ip>` and the
  client is read from `X-Forwarded-For`'s last hop — never honoured from any
  other peer.
- Identity floors: submits (and funds) require a hotkey REGISTERED on the
  subnet (metagraph oracle, fail-closed 503 when the chain is unreachable);
  each hotkey gets `--max-hotkey-mb` (default 256 MiB) of stored
  submissions; ZIPs are capped by bytes, decompressed bytes, AND member
  count; signed timestamps are NaN-proof, strictly increasing per (action,
  hotkey), and the v2 signature binds the sha256 of the key header, so a
  captured request cannot be replayed with a swapped key.
- Structural caps already in the request path: declared-length `413` before
  any body byte is read, the signature gate before body buffering, and the
  streaming decompression cap in the store.

Per-identity limits (registration + quotas) and per-IP backstops together
mean a single attacker needs many registered hotkeys AND many IPs to hurt
the intake; the front proxy still carries the volumetric load. A minimal
nginx front:

```nginx
limit_req_zone  $binary_remote_addr zone=intake_req:10m rate=2r/s;
limit_conn_zone $binary_remote_addr zone=intake_conn:10m;
server {
    listen 443 ssl;  server_name intake.example;
    # ssl_certificate / ssl_certificate_key …
    client_max_body_size 128m;          # = cascade-intake --max-zip-mb
    client_header_timeout 15s;  client_body_timeout 60s;
    proxy_request_buffering on;         # slow bodies dribble at nginx, not us
    location / {
        limit_req  zone=intake_req burst=10 nodelay;
        limit_conn intake_conn 8;
        proxy_pass http://127.0.0.1:8790;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 180s;
    }
}
```

Then run `cascade-intake … --trusted-proxy 127.0.0.1`.

### Credentials on payer pods

A funded pod runs on the MINER's provider account: the payer has console access,
so everything in the worker's environment is theirs to read. Rules, enforced
in code:

- The pod is **isolated**: `RemoteHost.isolated` drops every `forward_env`
  and the dispatcher's global extras (`WANDB_API_KEY`); nothing from the
  orchestrator's environment travels. Pulls are anonymous (the repos are
  public by design).
- **Credential-free by default** (`[round] funded_pod_checkpoint =
  "harvest"`): the worker runs `--local-only` — it trains and leaves the
  checkpoint on disk, uploading nothing, and the pod's environment holds NO
  credential of any kind. The orchestrator then pulls the checkpoint over the
  same pinned ssh (`harvest_remote_dir`, tar streamed through Python's
  `tarfile` `filter="data"` so a hostile archive with `..`/absolute/symlink
  members fails the harvest rather than writing outside it), runs the ingest
  guard on the local copy (deviation ⇒ `tamper`), and uploads it under the
  OPERATOR's identity — the manifest pointer is a checkpoint whose bytes the
  orchestrator inspected before anything else could fetch them. A harvest
  transport failure settles as infra (requeue), never tamper. Requires a
  `funded_pod_image` built from a `--local-only`-aware release.
  - **Pin `funded_pod_image` as `repo:tag@sha256:…`, not digest-only.** Lium's
    launch API rejects the canonical digest ref and the client degrades a
    digest-only pin to the repo's floating tag (`:latest`), so a host with an
    OLD image cached under that tag boots stale code while the env digest gate
    still reads "pinned" (it attests the requested ref, not the runtime). Give
    the pin an immutable version tag so the degraded form stays on-release, and
    the trainer functionally probes each rented pod's `worker --help` for the
    dispatch flags (and hashes its files against `worker_code_fingerprint`)
    before any bytes move — a stale pod is requeued as infra, never burned.
- `"robot"` mode (fallback for a pre-harvest image only): the pod pushes with a
  **Harbor robot** scoped to the checkpoint project
  with `repository:push` only — no delete, no other project, no S3, no HF:
  1. **Per-pod robot (preferred):** minted at rent, revoked at teardown and by
     every sweep, Harbor expiry `[round] funded_robot_duration_days` as the
     backstop. Harbor forbids robots managing robots, and the operator's
     everyday Hub identity IS a project robot (`robot$cascade+cascade-bot`),
     so minting needs a project-admin USER login on the orchestrator:
     `CASCADE_HUB_ADMIN_USERNAME` / `CASCADE_HUB_ADMIN_PASSWORD` (used only
     by the minter; never forwarded anywhere).
  2. **Static funded robot (fallback):** a push-only robot you create in the
     Hippius UI, handed to the trainer as `CASCADE_FUNDED_HUB_USERNAME` /
     `CASCADE_FUNDED_HUB_PASSWORD`; rotate it by hand. A user login in that
     slot is refused.
  3. **Neither → the leg fails CLOSED** (skipped, unburned). The operator's
     own Hub login is never an option.
- **Pod identity is pinned for the leg.** Names are owner-chosen and
  reusable, so the trainer records the platform's pod id at rent and the
  container's SSH host key (generated at first boot), dispatches with
  `StrictHostKeyChecking=yes` against that key only, and re-checks the
  platform id before dispatch and when the leg returns. A pod relaunched
  under the same name, or a different container answering at its address,
  settles the entry as **`tamper`** — miner fault, terminal, hotkey spent —
  never as infra. These checks bind the observed pod identity; they do not
  attest that a miner-controlled host executed training honestly. Keep the
  harvest/checkpoint checks and credential isolation even with pinned SSH.
- **A checkpoint is data, never code.** A miner's pod produces the
  checkpoint, so no scorer may execute anything shipped inside it.
  `cascade/eval/checkpoint_guard.py` requires the checkpoint's
  `forecast_wrapper.py` / `model.py` to be byte-identical to the release's
  own copies (any other `.py` is refused), `config.json` to equal the
  contract's model config, and the safetensors HEADER (names/dtypes/shapes)
  plus file size to match the pinned model — before a tensor is allocated.
  The trainer runs the guard on every funded leg's checkpoint before it can
  enter a manifest (a deviation settles as `tamper`); validators, audit and
  bench run it again on their side.
- **The payer vault is sealed at rest** when `CASCADE_VAULT_KEY_FILE` is
  set: AES-256-GCM per entry with the hotkey as associated data, so a copied
  vault directory is useless without the key. Unset = plaintext 0600 with a
  startup warning — never run mainnet that way. Plaintext entries still
  hydrate and re-seal on their next insert.

### Bench at leg completion (miner-billed)

This is the trainer-sourced benchmark path. Validator-sourced v2 A benchmarks
skip payer sweeps and their verification; see the next section.

With `[telemetry] funded_bench = true`, a payer pod benches its own
checkpoint the moment the leg finishes — harvest → ingest-verify → bench →
tear down — and is never held to a settlement boundary. Expect ~1 h of extra
pod time per submission; `[round] funded_bench_margin_seconds` (default 0)
reserves time for it before a leg's target boundary. The bench data is
sideloaded from the trainer box and **no HF token** goes to an isolated host
(a failed sideload degrades to an anonymous download).

**Payer numbers are a filter, never a published fact.** A pod the miner
controls can write any `benchmark_report.json` it likes, and the signed
bench report feeds the DEC-CA-0013 promotion pool (the shared init for
every future era). So the top `funded_bench_verify_top` (default 1)
payer-reported challengers are **re-benched on the era king's pod** from the
orchestrator's guard-verified copy of the checkpoint. Within
`funded_bench_verify_tolerance` (2 % relative, one-sided) the operator's
numbers replace the payer's; a payer number better than the re-bench by more
than that is a forged sweep — the entry is dropped from the report (its
submission is spent anyway; the promotion seat is what the drop denies) and
logged as TAMPER. Unverified payer numbers (below the top-N) are logged
only. Everything the trainer signs was produced on operator hardware; the
payer bench decides only *which* challenger is worth the king pod's time.
`BenchReport` carries no provenance field (validators sign-check the
canonical body; adding one would fork them).

### Lane-A benches on the main validator's GPU (DEC-CA-0063)

From `[v2] bench_source_from_block` (mainnet: `"v2"`, i.e. the v2 go-live)
with `[v2] bench_source = "validator"`, lane A is no longer benched at leg
completion. The trainer runs no challenger or king sweep, no payer-pod
verification and no bench hold for those eras. Only the once-per-seed-epoch
init bench stays: it publishes the init pointer the king check needs.

The **main validator** (the wallet whose hotkey is `[netconfig]
owner_hotkey`) does the sweeps instead, and only when they are needed:

* the challenger that cleared the private pool, plus the era king (its rows
  are cached per pointer, so it is swept once per king checkpoint);
* the king and the current init when the DEC-CA-0055 king check is due.

It signs `bench_v/round-<id>.json` (rows under `bench_v/rows/<id>/`) with its
own hotkey and publishes it before the settlement is marked handled, even
with no swept entries. When a clearer or due king check needs this evidence,
every evaluator, including the main validator, holds until a later
trainer-signed manifest acknowledges the exact report digest (or records
`"absent"` after the configured timeout). A quiet settlement that needs no
report can proceed without one. Satellites replay the bound evidence in the
main validator's receipt.

What the main validator needs:

* A GPU: `--device cuda`, or an offload pod via `--eval-hosts` (the same
  path the gift gate uses, `run_bench_via_host`).
* The benchmark sidecar (`[eval] benchmark_project_dir`) and the benchmark
  data staged locally: `uv run --project benchmarks cascade-benchmark-download
  --data-dir <dir>`, then `[eval] gift_gate_data_dir = "<dir>"`. Coverage and
  the time limit follow `[eval] cascade_bench_max_series` (0 = full battery)
  and `bench_hold_max_hours`. FEV is added when the DEC-CA-0061 rule is in
  force; it pulls its own data from the HF hub. That rule ("avg_noninf_winrate",
  rule v2 2026-10-20) reads the per-task rows only: average ratio < 1, no
  suite's ratio LCB above 1 + `[scoring] bench_gate_tolerance` (0.01), win
  rate >= `[scoring] bench_gate_min_win_rate` (0.5). Before arming it, run
  `scripts/bench_seed_noise.py` on seed replicas of the recipe to calibrate
  the tolerance (DEC-CA-0061 "Rule v2").
* Write access to the manifest bucket. It already has this, because it
  publishes receipts there.

Budget: historical trainer timings do not qualify the current four-suite run
on Targon; measure the actual checkpoint, GPU and data first. The ceiling is
`bench_hold_max_hours`, which is 9 h on mainnet. The sweep runs inside the settlement, so
a winning settlement takes one sweep (king cached) or two (new king pointer)
longer to judge, and the next settlement waits behind it. Use
`cascade_bench_max_series` or a faster offload GPU if that comes close to the
3 h cadence. Each checkpoint's sweep has its own allowance, `[v2]
bench_sweep_hours` (mainnet 2.0, its retry included); the king's sweep never
spends a clearer's. `[v2] bench_budget_hours` (mainnet 6.0) caps the
settlement: a clearer starts only while the cap still holds its whole
allowance (plus the king's when the king is not cached), otherwise it is
published `budget_deferred` — skipped on every validator (never a loss) and
carried by the trainer into the settlement that records the report (at most
2 carries, inside the era), where it is judged again.

A failed sweep is classified. When both attempts show checkpoint evidence
(non-finite forecasts, or every config failing at scoring) and nothing
environmental, the sweep is published `checkpoint_fault`, and a challenger
listed there fails the gate on every validator. Everything else is
published `infra_failed`: OOM / CUDA, timeouts, store / network, dataset
fetch, a sidecar crash, or an allowance spent before the retry. An
`infra_failed` clearer is skipped and never takes a loss. With nobody left
under `enforce`, the round is inconclusive and the king holds. Size
`bench_sweep_hours` from a measured sweep: an allowance that is too short
turns every sweep into an infra timeout. If the publish itself fails, the
settlement is rolled back and held until the next poll.

#### Sizing the sweep (concurrent workers, paired passes)

The sidecar runs each suite as independent work units (one GIFT-Eval config,
BOOM dataset, TIME task or FEV task) over N worker processes, each loading
the checkpoint(s) once. Several workers can share a GPU while others prepare
data or compute metrics. Each retains every paired model; larger checkpoints
therefore fit fewer workers. Qualify identical reports on the actual hardware
before choosing concurrency. Only the
main validator sweeps; every other validator replays its signed rows.

* `[eval] bench_workers = "auto"` (or an int; `1` = sequential). With `auto`,
  a short probe process loads the checkpoint(s), runs one warm-up batch at
  the sweep's batch size and reads the allocator peak. Per-worker VRAM is
  that peak × 1.25 + 0.5 GiB of CUDA context. The worker count is
  min(Σ per GPU ⌊(free − 2 GiB) ÷ per-worker⌋, effective CPU cores − 1,
  available host RAM / worker allowance, 16, units),
  spread round-robin over the visible GPUs with one GPU per worker
  (`CUDA_VISIBLE_DEVICES`). `--device cuda:K` pins the sweep to GPU K. A
  failed probe falls back to one worker. The chosen plan is logged as a
  `[sweep] plan:` line on the sidecar's stderr. On an offload pod
  (`--eval-hosts`), the sizing runs on the pod. CPU affinity and container
  quotas apply; BLAS and torch use one CPU thread per worker. RAM sizing is
  conservative but estimated; the sidecar's `--worker-ram-gb` can override it.
* `[eval] bench_paired = true` sweeps challenger + an unswept king (and
  king + init at the king check) in ONE pass, so each dataset is built once.
  The pass may run for both members' `bench_sweep_hours`. Its wall time counts
  once toward `bench_budget_hours` and half toward each member's allowance,
  and it is each member's first attempt: a member whose report failed retries
  alone with what is left. A pass starts only while the cap holds both
  allowances.
* Measure before setting the DEC-CA-0063 budgets. On the sweep box (or pod):

  ```bash
  python scripts/bench_timing.py <checkpoint> --data-dir <dir> --device cuda \
      --suites gift-eval,boom,time,fev --workers auto,4 \
      [--extra-checkpoint <king checkpoint>] --keep /tmp/cascade-bench-qualification
  ```

  It runs the sidecar with `--workers 1` and then with each setting you
  list, prints the wall time per suite and the speedup, and exits 1 unless
  every report is byte-identical to the sequential one. It also rejects
  unknown/partial data pins, missing suites and failed tasks. FEV's explicitly
  reported multivariate scope exclusions are allowed; they are not full FEV
  leaderboard coverage. Stage its [local manifest](../benchmarks/README.md#pinning-fev-data)
  before this run. All four suites are selected by default. Then set `[v2]
  bench_sweep_hours` to cover the measured (paired) sweep plus its retry,
  and `bench_budget_hours` to cover the clearers a settlement should judge.
  The current 2 h / 6 h values were set before any timing existed.

#### Targon GPU evaluation

A prepared Targon GPU VM works through the existing SSH interface. This path
does not automatically rent Targon GPUs; the automatic Targon adapter remains
for CPU generation. Use the VM's assigned SSH address, port and login user,
install this release and both benchmark extras with the checked-in lock, and
stage all four suites on local disk. Do not forward wallets or provider keys.

Point `cascade-validator --eval-hosts` at its `hosts.toml`. Leave `cuda_device`
unset to expose all visible GPUs, or list one entry per GPU on that same VM;
the resolver combines those masks. It selects one physical machine, not a
distributed fleet. `--device cuda` runs on a GPU VM hosting the validator itself.
MarketCast currently uses the validator's own `--device`.

Private-pool offload is separately opt-in through `--private-eval-hosts`;
see [validator compute](VALIDATOR.md#compute-and-offloading). Only histories
cross that boundary; targets and scoring stay local. Parallel private workers
use independent model copies and preserve the original per-window batch/RNG.
They default to one worker and require hardware qualification before enabling.
The worker's `--qualify` mode compares every sample with a sequential run:

```bash
python -m cascade.validator.private_worker CHECKPOINT REQUEST.npz OUTPUT.npz \
    --device cuda --workers 2 --qualify
```

`REQUEST.npz` is made locally with `cascade.validator.private_worker.write_request`
using the effective training contract and selected windows. It contains no
targets or pool metadata. Keep this private fixture and checkpoint local to the
operator-owned VM. Save the returned timing/runtime JSON with the public
qualification results. A failed comparison must block enabling that mode.

### MarketCast (DEC-CA-0064): snapshot, feeds, validator GPU

**Shipped OFF** (`[marketcast] from_block = 0`). To arm it, type an era
boundary that is:

- ≥ `[v2] from_block` and ≥ `[lane_c] from_block`;
- at or after `[v2] bench_source = "validator"` takes effect;
- set while `[netconfig] owner_hotkey` names the main validator.

The loader refuses anything else. Lane C is MarketCast-only (DEC-CA-0056,
owner 2026-10-20): the shipped `[lane_c] targets` are already the
`mkt:<suite>` targets of `[marketcast] suites` (`mkt:returns`, `mkt:vol`),
and the loader refuses any non-`mkt:` target ("lane-C competitions need a
forward-only eval — only MarketCast (mkt:) has one") and `[lane_c]
pool_prefix` (retired; there is no `pool_c` and no `cascade-pool publish
--lane c`). With lane C armed and MarketCast off, no target is active, so
lane C's share burns. Add a suite to `[lane_c] targets` only when it is in
`[marketcast] suites`. The loader also refuses arming while a class that lists
assets has an empty or unbuilt `[marketcast.data_sources]` entry. Empty the
class, or pick a built source.

**Feeds** (public, no key; used by the snapshot/training pipeline and optional
audit reproduction, not by every validator to derive the recorded verdict):

- Hyperliquid `POST https://api.hyperliquid.xyz/info` is primary wherever it
  lists the asset:
  - `candleSnapshot` 1m and `fundingHistory`;
  - the main dex for crypto;
  - the `xyz` dex (`xyz:<SYMBOL>`) for FX, metals, commodities, indices,
    ETFs and mega-caps.

  These are perp prices: they carry a basis and drift off-hours, which is
  acceptable for a forecasting eval.
- Binance spot 1m klines are the crypto cross-check. Its result is reported
  in `snapshot.json` `cross_check` and never gates.
- dTAO alpha comes from `cascade-pool mkt-record-alpha --out <file>`, cron'd
  every minute on a box with chain access. It reads the pinned SDK path
  (`prices.alpha_price` + `SubnetInfoRuntimeApi.get_dynamic_info`).
  `publish --lane mkt --alpha-rows <file>` folds the samples in. The
  reserve keys `tao_in` / `alpha_in` are not verified against SDK 11.3.0;
  a missing key reads as missing.
- Open owner choices for FX, the other 25 crosses:
  - Pyth Hermes polling by the main validator (redistribution terms
    unverified);
  - Chainlink on-chain FX as a cross-check.

  Neither has an adapter yet.

**Snapshot** (signed by the main validator's hotkey; validators refuse any
other signer):

```bash
*/1 * * * *  cascade-pool mkt-record-alpha --out /srv/cascade/alpha.jsonl
*/5 * * * *  cascade-pool mkt-record --store /srv/cascade/bars
30 */6 * * * cascade-pool publish --lane mkt --alpha-rows /srv/cascade/alpha.jsonl \
                 --bar-store /srv/cascade/bars --wallet-name owner --wallet-hotkey main
50 5 * * *   cascade-pool reveal --lane mkt --hf-repo <org>/cascade-eval-pool
```

**The forward bar recorder is required** (review M15). `mkt-record`
appends every CLOSED Hyperliquid 1m bar + funding sample to the bar store as
it closes; each pass re-fetches the last 48 h and merges, so an outage
shorter than ~3 days loses nothing, and an unrecorded minute stays missing
(never interpolated). `publish --lane mkt` builds the snapshot's minute bank
from the store and refuses to run without `--bar-store` (`--allow-cold`
only for tests). Start the recorder at least `snapshot_days` (21) before
arming: the config cannot see it, so arming cannot check it.

It carries the last `snapshot_days` (21) of bars on one UTC-minute grid,
plus a block → time anchor, plus a coarse HOURLY bank of
`coarse_history_days` (180) that every series forecast at ≥ 1h is cut from
(realized vol at ≥ 1d only: the std of the day's 24 hourly returns). It goes
to `pool_mkt/snapshots/block-<N>.tar` plus `pool_mkt/index.json` in the pool
bucket. Publish often: an entry's windows exist only once a snapshot
covering them is active. The verdict waits up to `verdict_wait_blocks` for
them.

Hyperliquid history depth (measured 2026-10-19 with read-only `info`
calls, BTC and `xyz:GOLD`): `candleSnapshot` serves only the most recent
~5000 candles of an interval — 1m ≈ 3.6 days, 5m ≈ 17 days, 15m ≈ 52 days,
1h ≈ 208 days, 4h ≈ 833 days (an `xyz` asset only since its listing), 1d
from listing. So:

- the hourly bank fits (180 < 208 days), and is what makes the 1d horizons
  scorable (21 daily bars could never hold the 32-step minimum context);
- the API alone fills only the newest ~3.6 days of the 21-day minute grid
  (less than the 6-day verdict wait) — hence the recorder above;
- a recipe's embargoed training range (≥ 30 days old) has no minute bars on
  Hyperliquid; the training snapshot also fetches the hourly bank, so ≥ 1h
  suites train on real data and sub-hourly ones only on the generator's.

Window eligibility and the verdict (review II, 2026-10-19): per (asset,
horizon) `windows_per_cell` (48) windows spread evenly over the eligible
span; a window needs ≥ `min_bar_coverage` (0.8) live bars and ≥
`min_dollar_volume_24h` ($1M; 0 = off) traded in the 24 h before it, and a
price window (`returns`, `alpha`) a context that is not flat. Nothing reads
the realised target: `min_target_abs_bp` was removed (red-team F3,
2026-10-20) and the loader refuses it. The bootstrap clusters are asset
class × `cluster_block_hours` (6 h). The gain is the geomean over (class ×
horizon) of the ratio of MEAN CRPS (red-team F1); horizons and asset classes
weigh equally; a verdict needs ≥ `min_time_blocks` (8) blocks. A class × block
× horizon cell whose base mean loss is ≤ `zero_scale_frac` × its horizon's
median is masked (counted on the receipt row as `mkt.masked_cells`), and each
ratio is clipped to `cell_ratio_cap`. The calibration gate runs on
`mkt:returns` AND `mkt:vol`. Per horizon it is two-sided on
`coverage_gate_levels` (`coverage_tolerance` below, `coverage_tolerance_over`
above). The tail hit rates at `tail_levels` must be within `tail_tolerance`
of 5 %, per horizon and pooled. Every rate is class-weighted like the gain,
and every band is widened to `calibration_z` standard errors. Tune the
liquidity floor once the recorder shows real volumes: it was not measured
per asset.

The loader refuses a configuration with an unscorable (suite, horizon): its
frequency's history (`coarse_history_days` for a ≥ 1h series, else
`snapshot_days`) must hold the 32-step context + the horizon, and the
forward wait (`verdict_wait_blocks` × 12 s − `start_margin_seconds`) must
hold two aligned windows (≥ 3 × the horizon). `5d` was dropped from
`mkt:returns` for that reason.

**Main validator**:

- install the release's locked dependencies:
  `uv sync --frozen --extra train --extra hippius --extra chain --extra marketcast`.
  The Timestral library is a git-pinned dependency in the committed lock.
  Do not regenerate that lock as a routine deployment step.
- use a GPU (`--device cuda`).

Every settlement registers its MarketCast entries (and every entry left
pending) as ONE MarketCast round. For each round without a published report
the main validator:

1. loads the base at its pinned commit, sha256-checked before load;
2. fetches each fine-tune, which must match the base's architecture, be
   finite and carry a valid `marketcast_meta.json` (covariates in the
   allow-list), else the entry gets a deterministic `fault`;
3. forecasts the base and the fine-tunes on EVERY eligible window of the
   snapshot active at the round's block (base quantiles cached per window);
   a compute failure (OOM, a library exception, the base not loading) is
   retried twice, then the entry is marked `infra_deferred` — never judged,
   never a loss;
4. measures each computed fine-tune against its SENIOR references (red-team
   F4): earlier-committed entries of the same suite in the round and the
   suite's slot holder, forecast on the candidate's own windows. It records
   the closest reference as `near_dup`, with that reference's forecasts as a
   second sidecar (`marketcast/round-<id>/r-*.npy`). This is one extra pass
   per reference not already forecast;
5. publishes the float32 quantile sidecars and then
   `marketcast/round-<id>.json`, signed, public-read. It never overwrites a
   report; a missing snapshot or a failed publish is retried at the next
   settlement.

A non-infra exception raised inside a fine-tune's forward identically on
both attempts is that entry's `fault` (red-team F7). The infra shapes stay
`infra_deferred`: OOM, OSError, and messages naming CUDA / cuDNN / NCCL /
the driver.

Missing reports that have not yet been recorded can defer MarketCast work.
A round is applied
only when a later TRAINER-signed lane-C manifest records its report in
`mkt_reports` (the sha256 of the text the trainer read, or `"absent"` after
`report_timeout_settlements` (8) settlements); validators apply rounds
strictly in order and use exactly the recorded bytes. A recorded report
whose bytes cannot be read **holds the whole settlement, including A/B**, like
any bound read. Restore the exact recorded report and its sidecars from a
verified replica; never replace its digest, overwrite a report or make a local
decision to treat it as absent. Permanently losing all copies has no automatic
consensus-safe recovery in this release. Qualify report replication and restart
recovery before enabling C; C remains disabled in the checked-in mainnet config.
`"absent"` defers every
entry as `infra`; an entry deferred more than `max_infra_deferrals` (8)
times drops with the `operator` error class and the trainer requeues it
uncharged. Infra time never consumes an entry's `verdict_wait_blocks`. A
round no manifest ever records expires to absent after 2 ×
`report_timeout_settlements` MarketCast settlements.

Cost per round: one base pass over the windows, plus one pass per
fine-tune. Each pass covers `windows_per_cell` (48) × assets × horizons
windows. The cost on the validator GPU has not been measured (owner open
item).

**Trainer:** a MarketCast leg rides the lane-C queue and payer-pod path. The
pod runs `cascade-train-worker --role mkt_finetune --local-only` and needs
the `marketcast` extra in the worker image. The leg rents and harvests under
the throne contract reduced to `[lane_c] leg_hours` (`mkt_leg_contract`,
red-team F5), not the 5 h throne wall. The trainer stamps the leg's dispatch
time (`train_time`, unix) into the signed `marketcast` record — the
forward-only clock. It also stamps the dispatching commitment's `coldkey`
(red-team F6: validators count the Bonferroni family by owner), and keeps at
most `max_open_legs_per_coldkey` (1) MarketCast legs open per coldkey. The harvest checks the weights' sha256, the base's
architecture, the base's `config.json`, finite weights and
`marketcast_meta.json` (required fields, the pinned base, covariates in the
allow-list, ≤ 64 KiB). While MarketCast is active the trainer reads the main
validator's reports from the manifest bucket at every settlement and records
them in the lane-C manifest (`mkt_reports`), publishing a lane-C manifest
even when no leg finished.

Fine-tune checkpoints are uploaded under the trainer's Hub namespace like
every other checkpoint, which is public by design (`cascade/shared/
hippius.py` notes the namespace serves anonymous pulls), so a judged
MarketCast fine-tune is public-read from its harvest on — what
`cascade-audit marketcast --reproduce` needs.

### Direct submissions (private code, champion-only publication)

With `--submission-dir` on the intake and `[round] submission_vault_dir` +
`champion_publish` on the trainer, code never touches a miner-hosted repo:

- `POST /v1/submit` (body = the ZIP, signature over its sha256, optionally
  `X-Lium-Api-Key` in the SAME request) stores it in the operator-private
  store and returns a `vault/direct@sha256:<hex>` ref + the exact chain
  payload — a byte-ordinary hippius commit, so no validator changes.
  A submit-with-key parks `pending_reveal` and auto-queues when the chain
  reveal resolves (the intake sweeps on every request).
- Ownership: earliest upload owns a digest; another hotkey committing your
  digest is dropped at field entry; byte-copies still die at dedup, and so
  does an entry that carries most of your unpublished module inside another
  tree (`private_copy` — identifier renames do not help; only published
  code is anyone else's to build on).
- Publication: ONLY thrones publish (`champions/<digest>.zip` + index,
  public-read on the manifest bucket) per `champion_publish`: `crown`
  (published when crowned; the shipped value) / `delay` (after
  `champion_publish_delay_rounds`) / `dethrone`. Losers never publish.
- Pods: a dispatch stages exactly ONE entry's ZIP
  (`SubmissionStore.stage_for_dispatch` → the pod's `$CASCADE_VAULT_DIR`);
  the king's published code also resolves via `$CASCADE_CHAMPION_BASE`
  (`{s3_endpoint}/{manifest_bucket}`). Never rsync the whole store anywhere.
- Key TTL: a submit-with-key entry parks `pending_reveal` and its key lives
  in the vault under the same TTL, so a reveal timed beyond the TTL fails the
  entry (the key it needs is gone). Keep `[round] funded_entry_ttl_hours` =
  `cascade-intake --ttl-hours` (36 h).

### Cost model

Costs follow the active contract and benchmark source:

| Work | Normal payer |
|---|---|
| A/B/C challenger training | Miner, through the selected rolling provider key |
| V2 A CPU shard preparation | Operator; the generation window is separate from GPU time |
| V2 king-ladder defense training | King, when a new leg is needed; a cached king can cover several eras |
| Required operator-caused retraining and failed-key fallback | Operator resources, with the applicable recovery/key-lapse rules |
| Validator-sourced A benchmarks and MarketCast judging | Main validator/operator |
| Earlier trainer-source payer benchmarks | Miner pod, with extra rental time; operator verification uses operator hardware |

Budget for the actual GPU wall, startup, harvest and applicable benchmark work;
the configured reference-compute hours are not a marketplace invoice or ETA.
Compatible completed work still needs signed publication and judging. Late,
stale or deferred work follows the documented recovery rules.

## Security and credentials

- **The wallet never leaves the orchestrator.** Pods cannot sign; a bad pod
  can only return a checkpoint the validator's gates reject.
- **Pods carry no credentials.** Isolated, `--local-only`, checkpoint
  harvested by the orchestrator. No secrets in the image; `SSH_PUBKEY` at
  launch, key-only SSH, no baked host keys.
- **Miner code is caged.** Container sandbox or strict-netns subprocess, a
  no-egress firewall behind it.
- **The runtime is pinned.** `train_image_digest` +
  `CASCADE_TRAIN_IMAGE_DIGEST`, checked before dispatch.

Storage-layer integrity (who can write what, and why a leaked bucket key
buys vandalism but not a forged verdict) is in
[ARCHITECTURE.md](ARCHITECTURE.md#channels).

### TLS trust and the October 2026 eMudhra incident

Use **uv 0.13.4 or newer**, **certifi 2026.10.19 or newer**, and a patched OS
CA store on the orchestrator, validators and every CPU/GPU worker. If using
pip, use **pip 26.2.2 or newer** as well: it carries its own CA bundle.
Upgrade through a trusted, patched installer; do not use an old pip or curl
trust store to bootstrap its own replacement. Do not disable TLS verification.
See the [uv advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62)
and [certifi release](https://github.com/certifi/python-certifi/releases/tag/2026.10.19).

After refreshing OS CA packages, check each application's interpreter:

```bash
.venv/bin/python scripts/check_tls_trust.py
benchmarks/.venv/bin/python scripts/check_tls_trust.py
```

The checker exits nonzero for affected trust anchors, old/missing certifi,
unreadable CA overrides or an uninspectable default store. It checks default
OpenSSL files **and directories**, certifi, Requests, botocore, pip's vendored
bundle when installed, and CA environment overrides. It makes no network
requests. Before installing dependencies, `--system-only` checks OS/custom
trust without requiring certifi. A clean result covers these stores in that
interpreter; it does not establish whether earlier traffic was intercepted.

Hosted CI uses `scripts/prepare_ci_tls.py` before dependency downloads. It
copies the runner's CA bundle into a job-local store, removes only the four
known affected root fingerprints, and points Python, Git and HTTP clients at
that store. The unchanged trust checker must then pass. This does not modify
the runner's OS store or replace patching production hosts.

The pod bootstrap transfers the orchestrator's patched `uv` over SSH and
checks its SHA-256 before use. The local binary and worker must use compatible
CPU architectures; `CASCADE_UV_BIN` can select the vetted binary. It checks
system trust before fetching Git dependencies, then checks each installed
environment. Eval dependencies install even without a local data cache;
failed installation or unsafe trust stops provisioning. Bootstrap refuses
uv system/custom/insecure trust overrides; ordinary application CA overrides
must pass the checker.

For a small benchmark-image trust check, build the `tls-trust` target before
fetching the scientific stack or datasets:

```bash
docker build --pull --no-cache --target tls-trust -f benchmarks/Dockerfile -t cascade-tls-check .
```

Rebuild worker and benchmark images with `--pull --no-cache`; build both from
the repository root (`-f deploy/Dockerfile` or `-f benchmarks/Dockerfile`).
Qualify the resulting images on a disposable worker, publish immutable image
digests, and rotate the fleet using the normal contract/era rollout. Existing
image digests and running workers do not change when this source is updated.
Preserve queues and settlement state during replacement. Check the generation
CPU image too; an installed Python package update alone cannot patch OS trust.

### Rotating the storage credentials

Rotate whenever a pod ever held them. Trainer (`.env`, restart after): the
Hippius S3 pair (`HIPPIUS_S3_ACCESS_KEY` / `HIPPIUS_S3_SECRET_KEY` — read/write
on the manifest and logs buckets, read on the eval-pool bucket), the Hub login
(`HIPPIUS_HUB_USERNAME` / `HIPPIUS_HUB_PASSWORD` or `HIPPIUS_HUB_TOKEN` — push
and pull on the checkpoint repos), `HF_TOKEN` (fallback dataset write) and
`WANDB_API_KEY`. Validators need only READ access on the manifest bucket
(`manifests/`, `benchmarks/`, `promotions/`) plus write on their own receipt
prefix: issue scoped, expiring sub-tokens per validator and revoke any shared
pair. Each validator operator swaps the environment value and restarts.

## Stakeholder scoreboard

The stakeholder scoreboard (`cascade/website/stakeholders.html`, served at
`/scoreboard` and `/stakeholders`) is deliberately short (owner 2026-10-10:
key models, tokens, competitions; little text). Three blocks:

1. **Hero — GIFT-Eval gap to the best.** The king's latest GIFT-Eval MASE
   (toggle: CRPS) against a few headline models from the official
   leaderboard: the gap to Timestral-209M as one big number, a ranked bar list,
   and where the king's
   geomean would place among the leaderboard's no-leak entries.
2. **Five tiles.** Lineage tokens (king lineage; the trainer's counter from
   v2, else one contracted leg per promotion generation plus the king's
   own), competitions (settlements in `receipts/index.json`, with eras and
   judged challenger legs), miners (distinct hotkeys judged; eligible in the
   latest settlement), domain experts merged (v2 lane B), and v2 status
   (live / go-live block + countdown / not announced).
3. **One trend.** The king's GIFT-Eval gap to Timestral-209M, settlement by
   settlement.

Every figure comes from a public object in the manifest bucket; a missing
object renders a pending state, never a guessed number.

| Object | Producer | Feeds |
|---|---|---|
| `receipts/index.json` | validator | competitions, miners, lineage legs |
| `benchmarks/round-<id>.json` and `bench_v/round-<id>.json` | trainer and designated main validator (signed reports) | king scores and trend; the page prefers validator entries for matching checkpoint/role |
| `benchmarks/leaderboard-reference.json` | operator (`scripts/refresh_leaderboard_reference.py`) | the reference models |
| `status/v2.json` | trainer (`cascade.shared.v2_status`) | v2 status, lineage tokens (v2), experts merged |
| `promotions/index.json` | trainer | lineage depth before v2 |
| `status/chain.json` | validator | era length |

### Publish the page

```bash
python scripts/publish_website.py --only stakeholders
```

This also uploads the versioned GIFT-Eval reference
(`cascade/website/data/gift-eval-reference.json` →
`benchmarks/leaderboard-reference.json`). Re-run after any page change.

### Refresh the GIFT-Eval reference

```bash
PYTHONPATH=$PWD python scripts/refresh_leaderboard_reference.py            # rewrite the repo file
PYTHONPATH=$PWD python scripts/refresh_leaderboard_reference.py --publish  # + upload it
```

The script downloads every model's per-config results from the Hugging Face
space `Salesforce/GIFT-Eval` at a pinned revision (≈3 min), recomputes the
space's overall table (gmean of Seasonal-Naive-normalised MASE / CRPS, mean
per-config rank) and re-aggregates the same rows through our benchmark
sidecar's `official_aggregate` — the aggregation behind every checkpoint's
signed `gifteval_mase` / `gifteval_crps`. GIFT-Eval has no zero-MASE config,
so the two readings agree (a unit test pins this): the hero compares like
with like. Commit the rewritten file so each refresh is a reviewable diff.
`--models` changes the headline set; `REF_HEADLINE` in the page names the
model the gap is measured to.

Caveats the page states: testnet benches a truncated battery
(`[eval] cascade_bench_max_series`), so its gap is labelled not comparable;
the leaderboard sorts by mean per-config rank, which needs per-config rows
our signed report does not carry, so the king is placed by geomean.

### The v2 status feed (`status/v2.json`)

The trainer also publishes `warm_start` directly from the current era's
`warm_start_ckpt`, before the first settlement. It includes the era start,
generation, member index and size, and identifies the actual merged init
rather than the trained king. The mining page links it at the top and hides
an init from a different era until fresh status arrives.

`policy` exposes selected values from the running trainer's resolved config:
training budgets, CPU generation limits, expert-selection/acceptance limits
and the pinned MarketCast base. These fields are unsigned discovery data,
not a replacement for the enforced contract or signed receipts.

`cascade status --json` re-fetches the chain, v2 and all lane queue feeds,
reports timestamps/availability, and exits 3 if required live feeds are stale
or missing. Unknown policy fields are retained for agents. The new fields
need the updated trainer; they are not inferred from a miner's local config.

`scripts/publish_website.py --dry-run` includes `llms.txt`; publishing the
site also uploads it as plain text. Both pages advertise the guide. Static
Vercel deployments serve the packaged file with a short cache lifetime.
The guide describes read-only discovery and requires human authorization
for submissions, reveals and paid actions; publishing a guide does not
change those CLI safeguards.

Unsigned and presentational, published by the trainer from
`RollingScheduler.tick` (rate-limited to one upload per 2 min, with a fresh
timestamp even for unchanged state; also before v2 is live, when it carries `golive.state =
"unannounced"` for the `[v2] from_block = 0` placeholder or `"scheduled"`).
Schema 1:

- `golive {from_block, state, blocks_to_go}`; `block`, `as_of`, `netuid`;
- `model` — the MoE preset's card (name, total / active params, experts,
  top-k, MoE blocks);
- `king {era, pointer, routing_health {experts, bins, layers[{layer, tokens,
  max_over_mean, cv, dead, entropy, heat[32]}]}, health_source}` — the
  current king's per-expert load summed over the domains' routing probes;
- `lane_b {active, from_block, domains, targets, horizon_targets, max_targets_per_leg,
  payout {unit, full_pay_tiers, half_life, floor, ttl, share, slots,
  remainder},
  verdict {...}, gate {open, opened_block, reason, lineage_tokens,
  open_tokens, routing_overlap, routing_jaccard, latest, history[{era, block,
  lineage_tokens, overlap[], jaccard[]}]}, legs {...}, per_domain {target:
  {pending, open, finished, pending_merge, selection, dropped}}, merges[],
  merged_total}` — `targets` is every DEC-CA-0059 emission slot (the domains,
  then `horizon:<h>` per `[eval] scored_horizons` rung: 10 slots on mainnet);
  `payout.slots` counts targets; `from_block` is B's independent era-start
  launch (0 = off), not the v2 lane-A launch. `acceptance_from_block` is
  the separate acceptance-rule activation; pending merges are qualifications,
  and `outcomes` in the trainer's lane-B status distinguishes `accepted`
  from `acceptance_rejected` (an accepted row is not a dropped leg);
- `lane_a {round_id, epoch_start_block, ladder {king_share, king_share_bp,
  prev_bp, event, crown_lcb, crown_units, crown_tier, crown_start_bp, court_hotkey, court_until_block, slashed, slash_reason,
  slash_until, key_lapses, funded_by, former_share, king_slash, lane_a_cap,
  prelaunch_burn}, king_check {result, round_id, block, lcb, margin,
  king_pointer, king_hotkey, reasons}}` — the DEC-CA-0055 records of the
  latest receipt (present once a receipt carries them). A crowning's start
  share is margin-tiered (owner 2026-10-20): `[v2] ladder_start_tiers`
  (`[[0.50, 8000], [0.25, 6500], [0.10, 5000], [0.01, 2500]]`, placeholders)
  on the crowned challenger's verdict LCB when it is in increment units, else
  the fixed `[v2] ladder_start_level_bp` (5000); the `crown_*` fields are
  null on every other settlement. The audit's weights check re-derives the
  units, the tier and the start from the verdict (a forged `crown_lcb`, tier
  or start share fails it). Calibrate the tiers on seed replicas before
  arming (the 0.50 tier must be rare under noise); keep the lowest threshold
  at `[v2] win_margin_end`. A "crowning LCB below the lowest
  ladder_start_tiers threshold" warning in the validator log means the two
  disagree. The key slash
  escalates per reign (owner 2026-10-20): the first operator-paid lapse
  holds 1 % for at least `[v2] key_slash_min_eras` eras (6), then until
  recovery is recorded under a working rotated key; a seed-epoch carry can
  record recovery without retraining. A second lapse holds it until the reign ends;
  `operator_fault` never counts. Validators derive it from the manifests'
  `funded_by` (`ChampionState.king_key_lapses / king_lapse_open /
  king_slash_until_block`); the audit replays it from the ladder record's
  `prev_*` fields, satellites bind those to the previous `state_post`. The
  trainer keeps the failed-key digests in `era_state.json`
  (`king_failed_keys`): deleting that file (or the key) lets the same failed
  key lift the slash — don't. The "key fixed" probe is auth-only (no Lium
  balance read exists);
- `lane_c {active, from_block, state, share, takes_from, targets, payout,
  verdict, legs, per_target {t: {holder, queued, open, finished, leg_slots,
  pending[]}}, holders_source, trainer}` — only once `[lane_c] from_block`
  is typed (absent ⇒ the pages read "not live"). A holder carries its win's
  `tier` / `full_eras` / `below_floor`.
- **Gain-tiered full-pay period (owner 2026-10-20).** `[lane_b]` / `[lane_c]
  full_pay_tiers` (overlay-settable) and `[marketcast] full_pay_tiers`
  (overlay-settable) replace `full_rounds` (a config still typing it is refused
  at load): `[[lcb_threshold, full_eras], ...]`, default `[[0.05, 10], [0.02,
  5], [0.01, 2]]`, thresholds strictly descending in (0, 1), `full_eras`
  non-increasing and `<= win_ttl_eras` (lane C's TTL for `[marketcast]`). A
  win pays in full for the `full_eras` of the highest threshold its gain LCB
  meets, then the unchanged decay and TTL; the tier is fixed at the win and
  stamped on every `lane_b_pay` / `lane_c_pay` win (`gain_lcb`, `tier`,
  `full_eras`, `below_floor`) and on the `state_post` slot. The defaults are
  UNCALIBRATED placeholders — lane-B domain LCBs are noisy (few windows and
  sources per domain): before arming, set the thresholds from real win LCBs
  (e.g. the testnet receipts' passing `target_lcb` / confirmed `gain_lcb`) and
  keep the lowest threshold at `gain_margin` (a validator log line "BELOW the
  lowest tier" means a win cleared `gain_margin` under the lowest threshold).
  A tier change applies to wins made after it only.

The trainer holds no validator slot state, so the lane-C **holders** and the
lane-A ladder are read off the latest receipt (the anchor's
`receipts/<validator_hotkey>/latest.json`, else the shared pointer):
`lane_c_pay.wins` are the paid live slots, enriched with the holder's
`confirmed` row; rows with outcome `pending` / `deferred` are the pending
confirmations (`holders_source = "receipt"`). An expired or deregistered
holder's slot burns and shows no holder. The read is unverified: the doc is
presentational, the pages read the signed receipt themselves.

The mining dashboard's lane tabs and v2 strip read it with the signed
receipts (`lane_b`, `lane_b_pay`, `lane_c_pay`, `emission`, `king_check`,
`weights`) and, per settlement, the receipt index row's `lane_b_verdicts`,
`lane_b_pay`, `lane_c_verdicts`, `lane_c_pay`, `emission_split` (lane C
included, never counted as burn), `ladder` and `king_check`; the lane-C page
also reads the trainer's `status/lane_c.json` (queue, open legs, outcomes).
`tests/unit/test_website_lanes.py` checks that the fields the page reads are
fields these writers emit. Verify:
`curl <endpoint>/<bucket>/status/v2.json | jq '.golive, .lane_a, .lane_c.per_target'`.

`cascade round` (the miner CLI, `cascade/miner/dashboard.py`) prints the
same lane rows from the receipt index: the lane-A ladder share / court / key
slash / king check, lane-B and lane-C verdict counts and paid slots, and the
emission split.

### Live training on W&B (lanes, tags, events)

Observability only (`[wandb]`; numbers never feed scoring or weights). One run
per leg, tagged so a public project filters cleanly:

- every run: `round:<id>`, `hotkey:<hk>`, `uid:<n>`, `size:<preset>`,
  `role:<log role>` and **`lane:A|B|C`**; lane-B / lane-C legs add
  `target:<t>` (a DEC-CA-0059 multi-target leg's targets comma-joined, e.g.
  `target:energy,horizon:64`). The run config carries `lane` and `target`;
  a generic lane-C `finetune` run would also carry the recipe
  (`recipe_lr_scale`, `recipe_steps_fraction`, `recipe_warmup_fraction`,
  `recipe_schedule`), but lane C is MarketCast-only today and the
  `mkt_finetune` worker path (`cascade/trainer/worker.py`,
  `marketcast_leg.py`) opens no W&B run, so lane-C legs log nothing.
  Expert and fine-tune legs keep the `challenger-<size>` role (the receipt
  role, `remote.receipt_role`, is unchanged) — filter on `lane:`.
- run ids are deterministic in `(round, role, hotkey)` plus, off lane A,
  `(lane, target)` (`wandb_sink.wandb_run_id`): a retried leg resumes its run,
  and the same hotkey's lane-A, lane-B and lane-C legs in one round never
  share one.
- trainer events (short runs, `job_type` = the event, tag `event:<kind>`):
  `king_check` — each DEC-CA-0055 king check the trainer reads off the
  receipts (result, LCB, margin, king pointer, ladder share, reasons; once
  per check); `promotion` — each published promotion record (generation,
  members, effective era).
- validator summary (`[wandb] validator_summary = true`, **off by
  default**, independent of `enabled`): one `settlement` event per published
  receipt — lane-A verdict, lane-B / lane-C verdict counts and paid slots,
  emission split, ladder and king check. Logged after the weights and the
  receipt are out; no key, no package or a W&B outage is a silent no-op.
  Validators never need W&B.

**Funded legs** run on the payer's isolated
pod, which receives no orchestrator environment: W&B reaches it only through
`[wandb] funded_key_env`, the NAME of an orchestrator env var holding a
SEPARATE key scoped to the public project, forwarded to the pod as
`WANDB_API_KEY` (the operator's own `WANDB_API_KEY` is refused). Mainnet
names `WANDB_API_KEY_FUNDED`: export it on the orchestrator (an unset
variable is skipped silently). The testnet template sets no
`funded_key_env`, so testnet funded legs log nothing. The payer can read the
key from their pod.

### Benchmark reports (GIFT-Eval / BOOM / TIME)

The hero and trend read both `benchmarks/round-<id>.json` and
`bench_v/round-<id>.json`, preferring the validator entry for the same
checkpoint/role. Under validator-source A rules, the trainer report can contain
only the init while the main validator report supplies king/challenger scores.
If figures stay missing while settlements land:

1. `curl -sI <endpoint>/<bucket>/benchmarks/round-<id>.json` — a **403** means
   the doc is there but private (backfill the canned ACL with
   `scripts/make_bench_public.sh`), a **404** means the bench never ran or
   never published — check the responsible trainer/main-validator log. Check
   both report prefixes; a missing trainer king row is expected under validator source.
2. On the Vercel origin, `curl -sI https://<domain>/benchmarks/round-<id>.json`
   — a 404 there with a 200 at the bucket means the corresponding rewrite
   (`/benchmarks/:path*` or `/bench_v/:path*`) is missing from
   `cascade/website/vercel.json`.
3. Only a report carrying a `role = "king"` entry with all six numbers moves
   the hero.

To backfill a report by hand: `scripts/publish_round_bench.py` (see its
`--help`; reports are immutable once published).

The dashboard's **Benchmarks** tab shows GIFT-Eval, BOOM, TIME and FEV
(CRPS and MASE). Older reports without FEV show `--`; FEV does not change
the existing six-score promotion geomean.

Its fixed comparison target is **Timestral-209M**, loaded from
`benchmarks/reference-timestral-209m.json`. Score the official pinned
checkpoint with the same sidecar, data revisions and full battery as the
king (`--suites gift-eval,boom,time,fev`), then prepare the document with:

```bash
PYTHONPATH=$PWD python scripts/publish_reference_bench.py /out/report.json \
  --preset timestral-209m \
  --source 'mistralai/Timestral-209M@<40-character-revision>' --dry-run
```

Remove `--dry-run` to publish after reviewing the reference. This preset
requires all eight measured losses and a pinned model revision. Until the
document exists, the page shows the reference as pending. Do not substitute
the model card's FEV skill scores for the sidecar's losses. The stakeholder
scoreboard separately compares GIFT-Eval against the refreshed Timestral-209M
leaderboard entry using the sidecar's aggregation.

The Benchmarks chart compares models on a Pareto frontier. Select a benchmark
and Size × MASE, Size × CRPS or MASE × CRPS. Size is total checkpoint
parameters; active MoE parameters appear in point details. Coral arrows follow
the last scored checkpoint of each king's reign in settlement order. Missing
scores/sizes and known data-revision mismatches with the current king are
omitted rather than estimated. Records without data revisions are historical
comparisons, not verified same-data runs.

`data/benchmark-models.json` supplies the packaged model catalog; the page
prefers `benchmarks/model-frontier.json`, uploaded by `publish_website.py`.
The initial catalog has published GIFT-Eval losses using Cascade's existing
97-configuration aggregation. Those external models were not rerun by Cascade.
BOOM/TIME/FEV references require measured sidecar results. To merge one:

```bash
PYTHONPATH=$PWD python scripts/publish_reference_bench.py /out/report.json \
  --preset timestral-209m \
  --source 'mistralai/Timestral-209M@<40-character-revision>' \
  --dry-run --out /tmp/timestral-reference.json
PYTHONPATH=$PWD python scripts/refresh_pareto_reference.py \
  --reference-doc /tmp/timestral-reference.json
```

The refresh fetches public leaderboard/checkpoint metadata and counts Cascade
preset parameters on a meta device; it runs no model inference and publishes
nothing. Review the generated catalog before the normal website publication.

### Verify the page

1. Top bar shows LIVE; the hero names Timestral-209M with a gap and a ranked bar
   list including "Cascade king".
2. Five tiles: lineage tokens, competitions, miners, experts merged, v2.
   Before v2 the last two read "v2 domain experts" / "go-live not announced"
   (or the go-live block with a countdown).
3. The trend has one point per benched settlement.
4. Dark mode and phone width: tiles fall to 3 then 2 columns; the hero
   stacks.

### Design constraints (do not regress)

- Three blocks, five tiles, one chart; labels and tooltips, no prose.
- No invented numbers: reference figures come only from the refresh script,
  king figures only from signed bench reports; a missing source pends.
- The king's number and the reference must share an aggregation; if that
  ever stops holding, label it on the page.
- Detail (verdicts, cohort, lane B, MoE health, training curves) belongs on
  the mining dashboard (`index.html`), not here.

## Lane-C competitions: add, re-weight, end (DEC-CA-0056)

Lane C can run several fine-tune competitions at once. Each one is a
`[lane_c.competitions.<kind>]` table in `chain.toml` (`weight`,
`from_block`, `until_block`, `wind_down`, `wind_down_max_eras`). The shipped
config has one: `mkt` (MarketCast), weight 1, no end.

**Add a competition (a release).** A competition kind is code: its eval must
be forward-only (it scores only windows that close after the entry is
trained).

1. Register the kind with its eval in `cascade/shared/config/lane_c.py`
   (`register_competition_kind(kind, eval=..., forward_only=True,
   from_block_table=..., parse_value=...)`). Set `from_block_table` when the
   eval has its own go-live table (MarketCast: `"marketcast"`); otherwise
   the start is the competition table's own `from_block`.
2. Ship its judging pipeline (validator), its trainer leg and its audit
   replay with the same release.
3. In `chain.toml` add its targets to `[lane_c] targets` and add
   `[lane_c.competitions.<kind>]` with `weight` and an ERA-BOUNDARY
   `from_block` at or after `[lane_c] from_block`. The loader refuses a
   non-boundary block, an unknown or non-forward-only kind, a target whose
   kind has no table, and a negative weight.
4. Every validator and the trainer must run the release before
   activation. Weights are normalized over every configured competition,
   including pending competitions, so the new share is reserved once that
   configuration applies. Until the competition starts, its part burns. Tell miners the new per-target
   share before you ship.

**Re-weight or end a competition (an overlay or a release).** `weight` and
`until_block` of a configured competition are overlay-settable
(DEC-CA-0057). An overlay replaces the whole `[lane_c] competitions` table,
so restate every competition with all its keys:

```toml
[lane_c.competitions.mkt]
weight             = 1.0
until_block        = <era boundary>
wind_down          = "finish_full_pay"
wind_down_max_eras = 6
```

`until_block` must be an era boundary after the start, and the overlay's
apply block must be before it (an end is never moved to or from a block
already passed). The settlement AT `until_block` is the last one judged.
From that block the intake answers `409 competition_ended`; the trainer
fails a queued entry whose leg (`[lane_c] leg_hours`) cannot finish before
it, as `competition_ended` (uncharged). Winners wind down per `wind_down`.
An overlay cannot add or remove a competition, change `from_block` or the
wind-down rule; those need a release. The receipt's `lane_c_pay` records
each competition's share and state, and the audit replays it.

## Network config overlays (DEC-CA-0057)

The owner changes `[scoring]`, `[eval]`, the `[round]` schedule keys
(`epoch_blocks`, `epoch_blocks_prev`, `epoch_activation_block`,
`rolling_from_block`, `era_settlements`, `round_hours`, `blocked_hotkeys`,
`blocked_from_block`, `commit_floor_block`, `funded_field_cap`), `[v2]` runtime policy
(never `[v2.training]`), `[lane_b]`, `[lane_c]` and the listed numeric
`[marketcast]` controls by publishing a signed overlay. Validators adopt
it by stake. Everything outside the allow-list is still a release.

Install the release supporting these controls on the fleet once. After
that, supported tuning needs no validator restart or code update. Inspect
the installed release's exact policy locally (no wallet or network):

```bash
cascade-validator --netconfig-policy
```

| Control | Update path |
|---|---|
| Existing scoring rules' thresholds, eval sample counts, lane shares and reward tiers | Signed overlay |
| Existing lane-A benchmark source, rule activation, report timeout and sweep budgets | Signed overlay; source uses the already pinned trainer/main-validator identities |
| MarketCast gain/confidence/calibration thresholds, minimum evidence, reward tiers and report/infra limits | Signed overlay; exact keys in `--netconfig-policy` |
| CPU generation time, resources and corpus target | Operator config; no other validator update |
| Trainer-declared optimizer/recipe/budget terms | Trainer config and signed manifests; locked terms remain release-only |
| New model/scoring code, locked training terms, trusted keys, storage locations, MarketCast base/universe/data contract | Release |

Every overlay still passes the normal loader and cross-field checks, is
owner-signed, requires the configured permit-stake lock-in (51% in the
shipped config), and takes effect at an era boundary. Historical
settlements keep their original configuration. Changing a parameter does
not implement an algorithm a validator's code does not already support.

**One-time setup (a release):** set `[netconfig] owner_hotkey` in
`chain.toml` to the owner's MAIN validator hotkey and ship it to every
validator and the trainer. `[activation] feature` must stay set: adoption
rides the readiness note. Validators need read access on
`config/network/` in the manifest bucket. Publishing needs write access on
that prefix (the trainer's Hippius S3 pair works).

**Each change:**

1. Write the FULL set of overrides as a TOML file. Overlays are snapshots,
   not deltas: seq N+1 replaces seq N, so restate every key you want to keep.
   For example:

   ```toml
   [scoring]
   min_windows = 250
   [lane_b]
   share = 0.45
   ```

2. Pick `effective_block`: an era start comfortably ahead of the head. If
   the 51 % lock-in lands after it, the overlay applies at the first era
   start after the lock-in, and every changed `*_from_block` in it must then
   still be at or after that block, or the fleet refuses it.
3. Dry run (the default; signs and writes nothing):

   ```bash
   cascade-validator --netconfig-overlay overlay.toml \
       --netconfig-effective-block <era start>
   ```

   It prints the next `seq` / `prev`, the digest and its `cfg-<8 hex>` note
   segment, the merged diff against `chain.toml`, and every check a
   validator runs (allow-list, merged load checks, era start,
   retroactivity), then `VALID` or `INVALID`.
4. Publish with the owner wallet:

   ```bash
   cascade-validator --netconfig-overlay overlay.toml \
       --netconfig-effective-block <era start> --netconfig-publish \
       --wallet-name <owner> --wallet-hotkey <main validator hotkey>
   ```

   It refuses a wallet whose hotkey is not `[netconfig] owner_hotkey`. It
   writes `config/network/seq-<n>.json`, then `config/network/latest.json`.
   Nothing is written on chain: your validator signals the `cfg-` segment in
   its own note on its next poll.
5. Watch the logs for `netconfig: overlay seq N … verified`, then
   `ADOPTED — in force from block B` (or `REFUSED`, with the reason; fix it
   and publish the next seq). Receipts from block B on carry
   `netconfig_seq` / `netconfig_digest`.

The trainer runs the same resolver read-only (`netconfig_state.json` in its
work root). It never signals, and it switches its rolling scheduler and
promotion engine to the overlay at the same block.

### V2 generator channel policy

The release presets require **8–32 channels in every synthetic yield**, including
optional synthetic generators supplied with MarketCast recipes. The private eval
pool keeps its separate channel rules. Verify the source with the new CLI before
uploading; `--skip-runtime` does not check output width. `cascade init` creates an
8-channel starter. Stream parents, CPU corpus generation and GPU shard readers
check widths; incompatible cached corpora are regenerated before GPU rental.

Before activating v2, verify the **chosen initial king/baseline generator** too.
An inherited v1 source producing 1–7 channels does not satisfy the v2 policy;
prepare a compliant baseline source/ref before launch. Archived source is kept
unchanged for replay, and historical pre-v2 training resolves the original floor
of one channel. Do not pad or duplicate archived data silently to pass the gate.

Run the launch qualification against the immutable source that the initial
king will actually use, on a host with the production generator sandbox:

```bash
cascade-trainer --chain-toml chain.toml --work-root ./_train_work --preflight-v2-baseline NAMESPACE/GENERATOR@sha256:DIGEST
```

This fetches the pinned tree, runs full bounded verification, and records a
local proof tied to its ref, contract, generator policy and verifier source.
It rents nothing. The first v2 king leg refuses rental without a matching
proof and a pinned validator receipt anchor. A launch root survives a later
v2 dethrone; changing the verifier or policy requires qualification again.
The trainer and validator may share the configured signer; receipts then share
that trust anchor rather than providing an independent attestation. The benchmark
owner is pinned to the main validator in the mainnet template. Do not assume
installation has qualified the baseline or completed release configuration.
`status/v2.json` publishes `launch_preflight=blocked` with its reason, and
the CLI does not report training readiness while that blocker is present.

A fired promotion is not sufficient to change the training init: it must have
a timely published settlement notice. A record that cannot be published at
its fire boundary is retracted (`generation N RETRACTED`) and the passing
check re-read at the next boundary. If a notice still missed its window, the
generation is dead for every node: the trainer keeps the prior verified
generation and logs `promotion N not installed: … announced too late`, every
validator skips that generation (`never effective … skipped`), and the next
passing king check fires the following generation. A `promotion record publish
REFUSED` (another body already published for that generation) needs operator
reconciliation. If the bound evidence is unreadable, scheduling holds until
it can be read. Do not backdate a record or edit the era ledger to force it
through. Verify the notice/report replica and restart paths during the
integration qualification.
