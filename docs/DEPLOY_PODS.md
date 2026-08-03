# Deploying trainer-worker pods (Shadeform / Targon / Lium)

The trainer splits into a **control plane** (the orchestrator — holds the wallet,
signs + publishes the manifest, runs on a trusted CPU box) and a **data plane**
(GPU pods that only fetch a generator, train one checkpoint, push it back, and
print a receipt). This doc covers standing up the data-plane pods from one
portable image, on any SSH-reachable GPU marketplace.

The transport is plain SSH, so the provider is interchangeable — Shadeform,
Targon, Lium, or bare metal all look identical to `cascade/trainer/remote.py`.
Mix providers in one `hosts.toml` if you like.

## 0. Prerequisites (once)

- A container registry the pods can pull from (GHCR, Docker Hub, ECR…).
- An SSH keypair for the orchestrator. The **public** key goes on every pod
  (`SSH_PUBKEY`); the private key stays on the orchestrator (`hosts.toml`
  `key_path`).
- Hippius registry + S3 credentials (read the generator, write the checkpoint).

## 1. Build & push the image

```bash
docker build -f deploy/Dockerfile -t <registry>/cascade-worker:<tag> .
docker push <registry>/cascade-worker:<tag>
# Record the pushed digest — pin pods to the DIGEST, not a mutable tag:
docker inspect --format='{{index .RepoDigests 0}}' <registry>/cascade-worker:<tag>
```

Pinning by digest (`...@sha256:...`) makes the numeric stack identical on every
pod and every audit re-run. Treat the digest as part of the reproducibility
contract, alongside `[training] expected_gpu` — and pin it **on-chain-visibly**
in `chain.toml`:

```toml
[training]
train_image_digest = "<registry>/cascade-worker@sha256:<digest>"
```

Then inject the same digest into every pod at launch as
`CASCADE_TRAIN_IMAGE_DIGEST` (a container cannot introspect its own OCI
digest). With the pin set, `cascade-train-worker` **refuses a final run** whose
runtime doesn't report the pinned digest, and `cascade-audit` Tier 2 uses the
match to decide when a byte-exact checkpoint comparison applies.

## 2. Pick ONE GPU SKU and stick to it

Every pod, on every provider, must be the **same** GPU SKU — otherwise the
`expected_gpu` pin fails and numerics drift past tolerance. Filter each
marketplace to a single SKU (e.g. always `NVIDIA A10`). Confirm on a pod with:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Then set that exact string in `chain.toml`:

```toml
[training]
expected_gpu = "NVIDIA A10"
```

## 3. Launch pods (per provider)

Each provider does the same three things: run the image, expose SSH (port 22)
with your `SSH_PUBKEY`, and pass the Hippius creds (or forward them per-dispatch,
see step 4). Filter to your chosen SKU.

- **Shadeform** — launch via the REST API with a container config: image =
  your digest, env = `SSH_PUBKEY` (+ optionally the `HIPPIUS_*` creds), port 22
  exposed. Filter instances by GPU type.
- **Targon** — same pattern: launch the image, inject `SSH_PUBKEY`, expose SSH.
- **Lium** — either launch your image directly, or SSH into a base GPU pod and
  `docker run` it (needs `docker` + `nvidia-container-toolkit` on the base):

  ```bash
  docker run -d --gpus all -p 22:22 \
    -e SSH_PUBKEY="ssh-ed25519 AAAA... trainer-orchestrator" \
    <registry>/cascade-worker@sha256:<digest>
  ```

Two GPUs on one box → run the container once and pin each card with a separate
`hosts.toml` entry (`cuda_device = "0"` / `"1"`); the entrypoint's sshd serves
both. See `scripts/remote_hosts.example.toml`.

## 4. Wire the orchestrator (`hosts.toml`)

Collect each pod's public IP and add an entry. Forwarding the Hippius creds here
(rather than baking them at launch) keeps them off the pod's disk:

```toml
[[host]]
name          = "a10-shadeform"
host          = "203.0.113.10"
user          = "root"
key_path      = "~/.ssh/trainer_orchestrator"   # the PRIVATE key
remote_python = "/root/cascade/.venv/bin/python"
workdir       = "/root/cascade"                  # matches the image WORKDIR
cuda_device   = "0"
forward_env   = ["HIPPIUS_HUB_TOKEN", "HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY"]

[[host]]
name          = "a10-lium"
host          = "198.51.100.20"
user          = "root"
key_path      = "~/.ssh/trainer_orchestrator"
remote_python = "/root/cascade/.venv/bin/python"
workdir       = "/root/cascade"
cuda_device   = "0"
forward_env   = ["HIPPIUS_HUB_TOKEN", "HIPPIUS_S3_ACCESS_KEY", "HIPPIUS_S3_SECRET_KEY"]
```

The first host trains the king, the second the challenger; more hosts form a
round-robin pool for the heat and multi-finalist finals.

## 5. Run the round

Point the orchestrator at the host file (the wallet + `chain.toml` live here, not
on the pods):

```bash
cascade-trainer --remote-hosts hosts.toml   # + your usual wallet/chain flags
```

The orchestrator SSHes into each pod, runs `cascade.trainer.worker`, fetches the
checkpoints back, screens/assembles locally, and signs + publishes the manifest.

## 6. Spin down

The trainer reads a **static** `hosts.toml` — it does not provision or destroy
pods. For elastic spin-up/down, wrap steps 3–5 in a provisioning script:

```
launch pods (provider API)  →  poll SSH-ready, collect IPs
  →  template hosts.toml     →  cascade-trainer --remote-hosts
  →  destroy pods (provider API)
```

Only these GPU-hours are the variable cost; the orchestrator stays up cheaply on
CPU between rounds.

## 7. Lock down generator execution (no-egress + sandbox)

Pods run **miner-controlled generator code** when they build the corpus. Two
layers of the sandbox are configured in `chain.toml [generator]`; harden the
deployment around them.

**Pick a sandbox mode.** On any production pod, one of:

```toml
[generator]
# EITHER: subprocess sandbox that REFUSES to run when the host cannot provide
# a network namespace (instead of silently degrading to the Python-level
# socket guard alone). Needs unprivileged user namespaces on the pod
# (`unshare --user --map-root-user --net true` must succeed).
sandbox_strict = true

# OR: kernel-enforced container sandbox — docker/podman with --network=none,
# --cap-drop=ALL, no-new-privileges, read-only rootfs, tmpfs workdir, and
# memory/pids/cpu limits, with the rlimited subprocess kept inside as defense
# in depth. Needs a container runtime on the pod and docker.sock access.
sandbox_mode   = "container"
sandbox_image  = "<registry>/cascade-worker@sha256:<digest>"   # digest-pinned
sandbox_python = "/root/cascade/.venv/bin/python"
```

Don't ship the permissive default (`sandbox_mode = "subprocess"`,
`sandbox_strict = false`) to mainnet: on a hardened host without unprivileged
userns it silently leaves only the in-process socket guard between miner code
and your network.

**No-egress pods.** The worker only ever needs to reach the Hippius registry
and S3 endpoints (plus the orchestrator's inbound SSH). Deny everything else at
the pod boundary so even a full sandbox escape has nowhere to call home:

```bash
# On the pod (or bake into the provider's firewall / security-group config):
# resolve the storage endpoints once, then default-drop outbound.
for host in registry.hippius.com s3.hippius.com; do
  for ip in $(getent ahostsv4 "$host" | awk '{print $1}' | sort -u); do
    iptables -A OUTPUT -d "$ip" -p tcp --dport 443 -j ACCEPT
  done
done
iptables -A OUTPUT -o lo -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT  # SSH replies
iptables -P OUTPUT DROP
```

Providers with security groups (Shadeform et al.): express the same policy
there instead — allow 22/tcp inbound from the orchestrator's IP, outbound only
to the storage endpoints.

**Write-only-prefix S3 credentials.** The `HIPPIUS_S3_*` pair a pod receives
via `forward_env` should not be the owner's root credentials. Issue each pod a
scoped key that can only write where a worker legitimately writes — per-round
logs — and read nothing it doesn't need:

```json
{
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:PutObject"],
     "Resource": "arn:aws:s3:::cascade-logs/logs/*"},
    {"Effect": "Deny", "Action": ["s3:GetObject", "s3:DeleteObject", "s3:PutObject"],
     "Resource": "arn:aws:s3:::cascade-manifests/*"}
  ]
}
```

The manifest and receipt buckets stay writable **only** by the orchestrator's
credentials (the wallet box): a compromised pod then cannot overwrite
`manifests/latest.json` or `receipts/latest.json`, only append noise to its own
log prefix. Checkpoint uploads go through the Hub token — scope it to the
`ckpt-*` repos if the registry supports it, and rotate both after any incident.

## Security recap

- **Wallet never leaves the orchestrator.** Pods can't sign; a bad pod can only
  return a checkpoint the validator's contract/eval gate rejects.
- **No secrets in the image.** `SSH_PUBKEY` at launch; Hippius creds via
  `forward_env` (preferred) or launch env.
- **Key-only SSH.** The image disables password auth and bakes no host keys.
- **Miner code is caged.** Container sandbox or strict-netns subprocess (step
  7); no-egress firewall behind it; write-only-prefix S3 creds behind that.
- **The runtime is pinned.** `train_image_digest` + `CASCADE_TRAIN_IMAGE_DIGEST`
  make a final run refuse an off-contract stack (step 1).

## The trainer ↔ provisioner contract (`cascade-provisioner`)

Everything above provisions pods **by hand**. `cascade-provisioner` (see
`cascade/provision/`, config in `scripts/provision.example.toml`, unit in
`deploy/cascade-provisioner.service`) automates it **per round**: rent shortly
before the epoch boundary, health-gate, publish, tear down per stage. The two
services never call each other — they meet only through files and stores, and
each side degrades safely without the other:

- **`hosts.toml`** (the trainer's `--remote-hosts`). The provisioner writes it
  atomically with heat- and final-tagged `[[host]]` entries — one entry **per
  GPU** (a 2×L40S final pod becomes two entries with `cuda_device = "0"/"1"`,
  so king and finalist train on the same physical box and trivially satisfy
  the validator's `expected_gpu` pairing). The trainer re-reads the file at
  every round start (`--hosts-wait-seconds` covers pod boot time after the
  boundary).
- **Empty `hosts.toml` = local fallback.** When no provider has capacity, the
  fleet fails its health gate, or the budget breaker refuses the round, the
  provisioner publishes an *empty* file. `load_hosts` raises on zero entries
  and the trainer trains that round locally — degraded, never lost.
- **`cascade-trainer --plan-only`** is the sizing input: the provisioner runs
  it inside the trigger margin (timed reveals have landed, so the eligible
  field is countable) and sizes the heat fleet off `eligible_challengers`,
  slot-based for multi-GPU pods.
- **The shared work-root** carries the mid-round teardown signal: the trainer
  drops `work_root/<round_id>/heat_complete.json` when the heat settles
  (field screened, hotkeys burned, finalists chosen — no heat dispatch can
  occur afterwards). The provisioner then kills the heat fleet **while the
  final still runs** and re-renders `hosts.toml` final-only. The provisioner
  keys rounds by boundary block and can't know `round_id` (= base_seed, the
  boundary block's *hash*) in advance, so any marker newer than its rent time
  is accepted — only one round runs at a time.
- **The round manifest** (`manifests/round-<id>.json` / a `latest.json`
  round-id change) ends the round: final pods die when it publishes.
- **TTL backstop.** Every pod dies one epoch after rent no matter what — a
  crashed trainer, an unreachable store, or a lost ledger can cost at most
  the round's worst-case projection, which `max_spend_per_round` caps before
  anything is rented. Orphan reconcile (live `cascade-*`-tagged pods not in
  the provisioner's ledger) runs every cycle.

**Security split.** The wallet never leaves the trainer/orchestrator box, and
the provisioner never needs it: it holds only the marketplace API keys
(`SHADEFORM_API_KEY`, `LIUM_API_KEY`), read credentials for the manifest store
(reachability probe + round-end watch), and the orchestrator SSH keypair. A
compromised provisioner can spend your provider balance (bounded by the budget
breaker) but cannot sign a manifest, and the systemd unit deliberately has no
dependency on any trainer unit — either service restarts freely without the
other.

## Marketplace adapters

Four are registered (`cascade.provision.core._PROVIDER_FACTORIES`); a stage's
`providers = [...]` names them in priority order and the ladder walks
`(candidate × provider)` until one has capacity for the **whole** fleet.

| adapter | API key | boot | pod shape | notes |
|---|---|---|---|---|
| `lium` | `LIUM_API_KEY` | CLI, docker template | multi-GPU executors | Bittensor-native; mostly lists `L40`, which **fails** an `L40S` pin |
| `shadeform` | `SHADEFORM_API_KEY` | docker or bare VM | `configuration.num_gpus` | a **broker** — it resells other clouds, so the offer you get is whichever backend was cheapest |
| `runpod` | `RUNPOD_API_KEY` | docker (REST v1) | `gpuCount` | defaults to **Secure Cloud** (RunPod's own/partner DCs) |
| `vast` | `VAST_API_KEY` | docker (REST) | `num_gpus` | cheapest consumer silicon; a marketplace of individual hosts |

**Pick per stage, not globally.** The `final` stage rents the pinned SKU for the
duel and a zombie rental costs the round, so it wants a real operator:
`runpod` (Secure Cloud) or `shadeform`. The `heat` stage screens the field on
disposable silicon and absorbs a dud through the replacement path, so it can
chase price.

**RunPod.** GPU-type ids *are* the nvidia-smi device strings (`NVIDIA L40S`,
`NVIDIA GeForce RTX 4090`), so a stage needs no `market_sku`. Prices are quoted
**per GPU-hour** and the adapter multiplies by the pod shape before handing the
number to the budget breaker, which bills per pod-hour. Availability is a
"sold on this tier" probe — RunPod publishes no per-shape stock — so the dud
counter and replacement path, not the probe, are what bound a bad round. There
is no `machine_of`: RunPod picks the machine, so a lemon cannot be excluded
from a replacement the way it can on lium and vast.

**Vast.** The most heterogeneous supply in the ladder, which is a real hazard
for the **heat**: the screen ranks runs across pods, so machine-to-machine
spread becomes rank spread (`decisions/DEC-CA-0010`). The adapter therefore
carries quality floors as part of its contract, not as tuning:

```toml
[provisioner.provider_options.vast]
verified_only         = true   # datacenter-verified hosts only
min_reliability       = 0.98   # vast's own reliability2 score
min_cpu_cores_per_gpu = 4.0    # the corpus streams from a CPU generator
disk_gb               = 60
```

`min_cpu_cores_per_gpu` is the one that bites: `corpus_mode = "stream_cpu"`
means each lane's generator is a CPU process whose threads are capped to its
slice of the box (`trainer.sandbox._lane_cpu_slice`), so a CPU-thin machine
starves training and reads as a slow *generator* in the heat. Loosen these and
the heat inherits the variance.

**Naming.** `providers` is a *stage*-level list but each marketplace spells the
same silicon differently — shadeform `RTX4090`, vast `RTX 4090`, runpod
`NVIDIA GeForce RTX 4090`. Matching folds whitespace and case, so those three
collide correctly, but it will never merge genuinely different devices (`L40`
vs `L40S`). Where a name differs by more than spacing, give the rung its own
`[[provisioner.<stage>.candidate]]` with that marketplace's `market_sku` —
`SkuCandidate` carries one per rung. The `sku` field stays the exact
nvidia-smi string in every case; that is what the health gate asserts on the
pod that actually booted.

Vast rents at most one pod per `machine_id` — two lanes on one physical box are
co-tenants, and a "2-pod" fleet that lands twice on the same machine bills
double while sharing a memory bus. It also supports `machine_of`, so the
replacement path can exclude a machine that just failed its boot gate.

```toml
[provisioner.provider_options.runpod]
cloud_type = "SECURE"   # or "COMMUNITY" for the cheaper host marketplace
disk_gb    = 60
```

Unknown keys in these tables are rejected at config load against the adapter's
own field list — a silently-ignored `min_reliabilty` would rent the heat off
unvetted machines while the config claims otherwise.

### What every adapter owes the loop

Beyond the five protocol verbs, the loop duck-types four optional methods, and
two of them are load-bearing for money:

- **`list_tagged(prefix) -> [(name, handle)]`.** Both halves, always. The
  orphan reaper decides *what* is ours by matching the pod NAME against
  `cascade-<round>-<stage>`, then terminates by the provider's HANDLE — an
  opaque id on every marketplace except lium. Returning bare handles makes
  `is_provisioner_pod_name` see a uuid, judge it "not ours", and skip it, so
  orphans bill until someone notices by hand. Use `filter_tagged_pods`.
- **`offer_price(sku, *, gpus) -> USD per POD-hour.`** Not per GPU-hour. The
  round breaker bills pods, so an adapter quoting per-GPU must multiply and one
  listing per-instance must scan the right shape. Getting it wrong
  under-projects `max_spend_per_round` silently, in the permissive direction.
- `machine_of(handle)` enables lemon exclusion on the replacement path; omit it
  where the marketplace picks the machine (runpod).
- `launched_image_digest(handle)` is the provider-side attestation the health
  gate falls back to when the pod cannot report its own launch env.

Batch launches must be **atomic**: `_rent_stage` records the ledger only after
`launch` returns, so an adapter that raises mid-batch strands whatever it
already rented. Both REST adapters unwind on failure rather than leaving it to
the reaper.

Transport retries are bounded and apply to `GET`/`DELETE` only. A create that
appears to fail may still have been accepted upstream, so retrying `POST`/`PUT`
risks double-renting a fleet the ledger never learns about.

### Preflight before arming a rung

```bash
cascade-provisioner --config provision.toml --chain-toml chain.toml --check-providers
```

Walks every configured `(stage × rung × provider)` against the live APIs and
**rents nothing**. It exists because an adapter's real risk is not its control
flow — that is unit-tested — but its assumptions about somebody else's JSON:
endpoint paths, auth header, response field names, whether a price is dollars
or cents and per-GPU or per-pod, and whether the pod listing carries the name
the orphan reaper matches on. Every one of those fails looking like "no
capacity" rather than "misconfigured", which is exactly the kind of fault that
hides for weeks.

```
ok  heat[0] 4×NVIDIA GeForce RTX 4090 (RTX4090) on vast: capacity, $1.10/pod-hr (cap $2.60)
--  heat[1] 4×NVIDIA RTX A6000 (A6000) on vast: no capacity, price unknown (rung would bill at its cap)
ok  final[0] 2×NVIDIA L40S (NVIDIA L40S) on runpod: capacity, $2.40/pod-hr (cap $2.60)
ok   runpod: list_tagged → 0 tagged pod(s)
```

It runs before the chain client and manifest store are opened, so it works on
a box with marketplace credentials and nothing else — including before the
service has ever been armed. Exit status is non-zero only for a hard fault
(auth failure, a raising adapter, a listing in the wrong shape); no-capacity
and over-cap are reported, not failed.

**Both new adapters still need a live run before they carry a `final`.** Their
response parsing is unit-tested against recorded shapes and preflight checks
the rest against real credentials, but no cascade round has rented through
them. Preflight, then a single heat rung on testnet, then the duel.
