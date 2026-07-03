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
