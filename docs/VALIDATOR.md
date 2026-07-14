# Validator guide — score rounds and set weights

A cascade validator reads the owner-trainer's signed manifest each round, pulls
the two trained checkpoints, scores them on the rotating eval pool, runs the
king-of-the-hill verdict, sets weights, and publishes a signed public receipt.
You never train. CPU is enough (a GPU is optional, see below).

## Everything you configure — the complete list

**1. `chain.toml` — you change NOTHING.** Pick the file for your network and
pass it on the command line:

| network | file | netuid |
|---|---|---|
| mainnet (`finney`) | `chain.toml` | 91 |
| testnet (`test`) | `chain.testnet.toml` | 259 |

Every value in it ships correct and consensus-critical: `[training]` folds into
the contract digest (edit it and you reject every valid manifest), `[scoring]`
and `[round]` define the verdict every honest validator must compute
identically, `[manifest] trainer_hotkey` is the only trainer to trust, and the
buckets are where the subnet's data actually lives. Your identity needs no toml
entry — receipts are signed with your **wallet hotkey** automatically.

*One optional exception*: `[manifest] validator_hotkey` is read only by
`cascade-audit` — set it to **your own** hotkey and the audit pins receipts to
exactly that signer; left empty, the audit still verifies the signature but
WARNs that the signer is unpinned. The running validator never reads it.
(Auditing someone else? `cascade-audit … --validator <their-hotkey>` does the
same without editing the file.)

**2. Environment variables — exactly these** (env only, never in the toml):

```bash
# From the owner (read manifests/logs, publish your receipts):
export HIPPIUS_S3_ACCESS_KEY=...
export HIPPIUS_S3_SECRET_KEY=...

# Your OWN Hippius account (https://hippius.com) — pulls checkpoints from the
# registry; digest-pinned, so it needn't be the owner's account:
export HIPPIUS_HUB_USERNAME=...
export HIPPIUS_HUB_PASSWORD=...        # or HIPPIUS_HUB_TOKEN=...

# Only if the owner hands them out (least-privilege pool + R2 backup keys):
export POOL_S3_ACCESS_KEY=...          # eval-pool snapshots
export POOL_S3_SECRET_KEY=...
export BACKUP_S3_ACCESS_KEY=...        # R2 fallback when Hippius S3 is down
export BACKUP_S3_SECRET_KEY=...
```

> ⚠️ Do NOT set `POOL_S3_*` unless the owner also told you to point
> `[storage] pool_s3_endpoint` elsewhere — with the default Hippius endpoint
> they sign requests with the wrong keys and every pool read fails
> `SignatureDoesNotMatch`.

**3. Command-line flags** — wallet and network, nothing else:

```bash
cascade-validator --chain-toml chain.toml --network finney \
  --wallet-name my-validator --wallet-hotkey default
# testnet: --chain-toml chain.testnet.toml --network test
```

That is the entire configuration surface.

## Setup

```bash
git clone https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[train,hippius,chain]'
```

The install pins `bittensor==10.5.0` — **don't upgrade it**; other SDK lines
write on-chain commitments in an encoding this subnet's decoder rejects.

Register and stake as you would on any subnet (netuid above; weight-setting
needs a validator permit, i.e. stake above the subnet threshold).

Smoke-test the config with zero chain/GPU I/O:

```bash
cascade-validator --offline --chain-toml chain.toml
# prints netuid, king, dethrone_cp, manifest bucket, eval-pool source
```

Then run it under a process manager (systemd/tmux); it resumes cleanly from
its persisted state. A healthy round looks like:

```
new manifest round=… entries=2 (king:uid…,challenger:uid…); gating + scoring …
round=… lcb=… margin=0.0200 win=… king=… tenure=…
round=… weights set: reward_uids=[…] (n_uids=…, burn_uid=0)
published scored receipt round=… signed=True → s3://…/receipts/<your-hotkey>/round-….json
```

## Cascade-specific rules (the things other subnets don't have)

- **Restart after any announced `[training]` change.** The contract digest is
  computed at startup; a validator running across the owner's `[training]` edit
  rejects every round (`contract_digest_mismatch`) until restarted. Owner
  announcements of a re-pin = pull + restart.
- **CPU is enough.** A duel scores in well under a minute on CPU. The GPU-heavy
  parts (GIFT-Eval gate, cascade bench) can be offloaded with
  `--eval-hosts eval_hosts.toml` — one `[[host]]` entry, re-read at every eval,
  so an elastic per-round pod works; a missing/empty file just means local
  evals. Your wallet and all consensus decisions never leave your box.
- **Gift gate**: with `gift_gate_mode = "shadow"` (launch default) you'll see
  one `gift-gate round=… passed=…` log line per duel, log-only. When the owner
  flips `"enforce"`, failing the gate vetoes takeovers — enforce mode needs the
  pinned bench data (`[eval] gift_gate_data_dir`) or an eval pod.
- **Verify yourself**: `cascade-audit latest --config chain.toml --network
  finney` re-derives your latest round end-to-end (signatures, seeds, digests,
  verdict, weights). All-PASS or investigate. See [`AUDIT.md`](AUDIT.md).

## What can go wrong

| symptom | cause |
|---|---|
| `rejecting manifest … contract_digest_mismatch` | your file's `[training]` differs from the trainer's — pull the current file AND restart |
| `rejecting manifest … signature_invalid` | wrong `[manifest] trainer_hotkey` (don't edit it), or the trainer published unsigned |
| `no eval-pool snapshot published` | owner hasn't published a snapshot, or wrong pool creds |
| weights never set | no validator permit (insufficient stake), or the extrinsic fails — check `btcli` and `weight set failed` log lines |
| every pool read `SignatureDoesNotMatch` | you set `POOL_S3_*` with the default endpoint — unset them |
| gift gate / bench slow | those want a GPU — `--eval-hosts`, or pin `gift_gate_data_dir` on a CUDA box |
| audit WARNs on `block-hash-onchain` | lite node without history — use an archive node for zero WARNs |

## Rewards

Weight splits across the current king plus recent distinct kings (geometric
decay, `[scoring] king_decay`), burning to `burn_uid` when none are registered.
You don't tune any of it — every honest validator computes the identical
vector, which is exactly what `cascade-audit`'s `weights` check reproduces.
