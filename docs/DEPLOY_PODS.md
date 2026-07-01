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
contract, alongside `[training] expected_gpu`.

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

## Security recap

- **Wallet never leaves the orchestrator.** Pods can't sign; a bad pod can only
  return a checkpoint the validator's contract/eval gate rejects.
- **No secrets in the image.** `SSH_PUBKEY` at launch; Hippius creds via
  `forward_env` (preferred) or launch env.
- **Key-only SSH.** The image disables password auth and bakes no host keys.
