# Validator guide — score rounds and set weights

A validator reads the owner-trainer's signed manifest each round, pulls the
king's and challenger's trained checkpoints, scores them on the private rotating
eval pool, runs the king-of-the-hill verdict, and sets weights on chain. It also
publishes a signed public **receipt** for every round so anyone can verify your
work. You never train — that's the owner's trainer; your job is inference-only
scoring, which is why the hardware bar is modest (see below).

At a glance:

```
pick a shape (GPU box / CPU box + eval pod) → install → make a wallet
   → register + stake → configure chain.toml + creds → cascade-validator
   → confirm weights set + receipts published → audit as health check
```

## Hardware

Two supported shapes:

| shape | what runs where |
|---|---|
| **GPU box** (recommended) | everything — wallet, private-pool duel, and benchmark evals — on one CUDA machine; run with `--device cuda` |
| **CPU orchestrator + eval pod** | wallet, private-pool duel, and **all consensus decisions** stay on a cheap CPU box; the GPU-heavy public benchmarks are offloaded over SSH with `--eval-hosts` (only a public checkpoint and a report cross to the pod — never keys) |

Minimums (the GPU row applies to whichever box runs benchmarks — the GPU box
itself, or the eval pod in the split shape):

| resource | minimum | notes |
|---|---|---|
| GPU | 1× CUDA GPU, **24 GB VRAM** (e.g. RTX 3090/4090, A5000, L4, L40S) | the Toto2 checkpoints being scored are tiny (4M / 22M params) — 24 GB is headroom for the benchmark sweeps (batched forecasts, `batch_size` up to 512), not the model |
| CPU / RAM | 8 cores / 32 GB | the private-pool duel and the paired bootstrap are CPU-tuned |
| disk | 100 GB free | eval-pool snapshots, fetched checkpoints, benchmark data caches |
| network | stable outbound HTTPS + SSH | Hippius S3/Hub and the chain endpoint; SSH to the pod in the split shape |

What actually exercises the GPU depends on config:

* **Pure KOTH (the shipped defaults** — `gift_gate_mode = "off"`,
  `cascade_enabled = false`**)**: only the private-pool duel runs. It is
  CPU-tuned, so a CPU-only box *works* — but run it on the GPU anyway
  (`--device cuda`) when you have one: rounds finish faster and you're already
  covered the moment the owner turns the gates below on. Treat CPU-only as the
  floor for the split shape, not the default. Without any GPU (local or pod),
  enabling those gates makes your rounds inconclusive/degraded.
* **GIFT-Eval gate on** (`gift_gate_mode = "shadow"` / `"enforce"`): every
  round scores *both* king and challenger on gift-eval — GPU, every round.
* **Cascade on** (`cascade_enabled = true`): normally free for you (the trainer
  ships the numbers in the signed manifest), but the fallback runs the full
  three-suite battery on your GPU — see
  [Bench scores](#bench-scores-trainer-stamped-validator-fallback).

## 0. Install

The evaluator needs torch; add the Hippius (fetch checkpoints/pool) and chain
(metagraph + weights) extras:

```bash
git clone https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[train,hippius,chain]'
# GPU-box shape only — on a CPU orchestrator this prints False and that's fine:
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

## 1. Make a wallet, register, and stake

```bash
btcli wallet new-coldkey --wallet-name my-validator
btcli wallet new-hotkey  --wallet-name my-validator --wallet-hotkey default
btcli subnets register --netuid 259 --network test \
  --wallet-name my-validator --wallet-hotkey default        # mainnet: --netuid 91 --network finney
```

Setting weights requires a **validator permit**, which requires stake above the
subnet threshold. Add stake to your hotkey:

```bash
btcli stake add --netuid 259 --network test \
  --wallet-name my-validator --wallet-hotkey default --amount <TAO>
btcli subnet list --network test        # check the permit / stake threshold
```

## 2. Configure `chain.toml`

Point the validator at the subnet and the trust anchors. Use `chain.testnet.toml`
for testnet; the keys that matter to a validator:

```toml
[subnet]
netuid = 259                       # 91 on mainnet

[manifest]
trainer_hotkey   = "5Cyver…"       # the ONLY trainer whose manifest you trust
validator_hotkey = "5F1Vm…"        # your hotkey — the receipts you publish are signed with it

[storage]
manifest_bucket = "cascade-testnet-manifests"   # where manifests + receipts live
pool_bucket     = "cascade-testnet-eval-pool"    # daily eval-pool snapshots (recommended)
# …or, instead of pool_bucket, pin a static pool:
# [eval] window_pool = "namespace/eval-pool@sha256:…"

[scoring]
# min_windows / min_clusters gate whether a round is conclusive; leave the
# shipped values unless the owner tells you otherwise.
```

`base_arch_digest`, the contract, and the eval geometry are shipped in the file
and must match the trainer's — don't change them, or your digest gate rejects
every (valid) manifest.

## 3. Set credentials

Storage credentials come from the environment, never `chain.toml`:

```bash
export HIPPIUS_S3_ACCESS_KEY=...    # read manifests, write your receipts
export HIPPIUS_S3_SECRET_KEY=...
export POOL_S3_ACCESS_KEY=...       # read eval-pool snapshots (falls back to
export POOL_S3_SECRET_KEY=...       #  HIPPIUS_S3_* when unset — see note below)
export HIPPIUS_HUB_USERNAME=...     # (or HIPPIUS_HUB_TOKEN) to pull checkpoints from the registry
export HIPPIUS_HUB_PASSWORD=...
```

The Hub credential can be **your own** Hippius account — pulls are digest-pinned
and namespace-independent, so the owner doesn't need to share theirs. The S3
pairs come from the owner. Owners: hand out a *separate* key pair for the pool
bucket (`POOL_S3_*`) rather than reusing the manifest-bucket pair — S3 keys
aren't prefix-scoped, so any key that can write a bucket can overwrite
everything in it, and the eval pool is the one store where an overwrite could
touch scoring (the manifest's signed pool pin catches it, but least privilege
beats detection). The same reasoning says never reuse the TSBench-Forge relay
keys here, even though they read the same `HIPPIUS_S3_*` env names in that repo.

If `[storage] backup_s3_endpoint` is set (a Cloudflare R2 backup of the
manifest/receipt bucket — every object is dual-written there, and reads fall back
to it when Hippius S3 is down), also export the R2 token:

```bash
export BACKUP_S3_ACCESS_KEY=...     # R2 backup of manifests/receipts
export BACKUP_S3_SECRET_KEY=...
```

Smoke-test the config with no chain/GPU I/O first:

```bash
cascade-validator --offline --chain-toml chain.testnet.toml
# prints netuid, king, dethrone_cp, manifest bucket, eval-pool source
```

## 4. Run

**GPU box** (recommended) — everything local, duel included, on the GPU:

```bash
cascade-validator --chain-toml chain.testnet.toml --network test \
  --wallet-name my-validator --wallet-hotkey default --device cuda
# mainnet: --network finney
```

**CPU orchestrator + eval pod** — `--device` stays at its `cpu` default and the
benchmark evals go to the first `final`/`any`-stage host in a `hosts.toml`
(same format as the trainer's `scripts/remote_hosts.example.toml`):

```bash
cascade-validator --chain-toml chain.testnet.toml --network test \
  --wallet-name my-validator --wallet-hotkey default \
  --eval-hosts hosts.toml
# log line to expect:
#   GIFT-Eval gate offloaded to <name> (<host>); wallet + consensus stay local
```

On startup it loads the eval pool (`loaded eval pool snapshot@block-… series=…`)
and polls the manifest bucket. Each new round you'll see it gate, score, decide,
and set weights:

```
new manifest round=… entries=2 (king:uid3,challenger:uid2); gating + scoring …
round=… lcb=0.0000 margin=0.0200 win=False loss king=… tenure=…
round=… weights set: reward_uids=[3] (n_uids=9, burn_uid=0)
published scored receipt round=… signed=True → s3://…/receipts/<hotkey>/round-….json
```

Run it under a process manager (systemd, tmux, supervisor) so it survives
restarts; it resumes cleanly from its persisted champion state
(`[validator] state_db_path`).

## 5. Confirm it's working

Three signals:

1. **Weights on chain** — `round=… weights set …` in the log, and
   `btcli wallet overview` / the metagraph shows your hotkey emitting weight.
2. **Receipts published** — a signed `receipts/<your-hotkey>/round-<id>.json` per
   round in the manifest bucket (the dashboard reads these via the shared index).
3. **Audit as health check** — verify your own latest round end to end:

   ```bash
   cascade-audit latest --config chain.testnet.toml --network test
   # all-PASS = signatures, seeds, digests, verdict, and weights all reproduce
   ```

   A FAIL here means your validator and the audit disagree — investigate before
   trusting the round. See [`AUDIT.md`](AUDIT.md).

## Bench scores: trainer-stamped, validator fallback

When Cascade (`[scoring] cascade_enabled`) is on, each reign's checkpoints are
ranked on six public-benchmark numbers — GIFT-Eval, BOOM, and TIME, CRPS + MASE
each. Those numbers reach your validator one of two ways:

1. **Trainer-stamped (authoritative).** The owner-trainer runs the benchmark
   sidecar once on the king's checkpoint and stamps the six numbers onto that
   entry in the **signed** manifest (`bench_scores`). Every validator reads the
   identical values, so Cascade's warm-start selection is deterministic across
   validators — and it costs you zero GPU time. This is the normal path.
2. **Validator sidecar fallback (degraded).** Only when a manifest entry
   carries **no** `bench_scores` (e.g. a trainer predating the Cascade hook)
   does your validator score the checkpoint itself: the full
   GIFT-Eval + BOOM + TIME battery via the out-of-process `benchmarks/` sidecar,
   on your local `--device` or offloaded to your `--eval-hosts` pod. Two
   caveats: it's expensive (full BOOM alone ≈ 26 min on an RTX 5090 — this is
   the load the 24 GB VRAM minimum is sized for), and independently-run GPU
   sweeps are **not bit-reproducible**, so validators on the fallback can log
   slightly different numbers. Prefer the trainer-stamped path; treat sustained
   fallback as something to raise with the owner.

Either way, these public-benchmark numbers drive **only** Cascade's warm-start
promotion. The dethrone verdict itself stays entirely on the private eval pool.

## What can go wrong

| symptom | cause |
|---|---|
| `rejecting manifest … contract_digest_mismatch` | your `chain.toml` contract differs from the trainer's — sync the file (this is the digest gate doing its job) |
| `rejecting manifest … signature_invalid` | wrong `[manifest] trainer_hotkey`, or the trainer published unsigned |
| `no eval-pool snapshot published` | `pool_bucket` set but the owner hasn't published a snapshot, or wrong bucket/creds |
| weights never set | no validator permit (insufficient stake), or the weight extrinsic is failing — check `btcli` and the log's `weight set failed` line |
| `gift-eval sidecar unavailable/errored` | the gift gate needs a GPU: pass `--device cuda` on a GPU box, or point `--eval-hosts` at a pod (see [Hardware](#hardware)); under `enforce` this makes rounds inconclusive |
| CUDA errors with `--device cuda` | torch can't see the GPU — reinstall with `pip install -e '.[train]'` on a CUDA box and re-check `torch.cuda.is_available()` |
| `--eval-hosts …: no 'final'/'any'-stage host found` | every host in the file is tagged `stage = "heat"` — the validator only offloads to `final`/`any`-stage hosts |
| audit WARNs on `block-hash-onchain` / `commit-cutoff` | you're on a lite node without the historical block/commitment; point `--network` at an archive node for zero WARNs |

## Rewards

Weight is split equally across the current king plus up to
`[scoring] reward_prior_kings` recent distinct kings still registered (burning to
`burn_uid` if none are). You don't tune this — it's consensus config in
`chain.toml`; every honest validator computes the identical weight vector, which
is exactly what `cascade-audit`'s `weights` check reproduces.
