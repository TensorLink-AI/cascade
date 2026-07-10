# Miner guide — submit a data generator

You compete by submitting a **data generator**: purely-algorithmic code that
produces synthetic time series. The owner's trainer trains a fixed Toto2-4M
forecaster from scratch on your data; you win rounds when your data trains a
better forecaster than the reigning king's, scored on a private, rotating
held-out set you never see. No GPU, no shipped weights — you compete the
*prior*. The submission contract (what the code must be) is
[`INTERFACE.md`](INTERFACE.md); this is the end-to-end operator walkthrough.

At a glance:

```
fork a generator → cascade verify → make a wallet → register on the subnet
   → set Hippius creds → cascade deploy → confirm it competes in a round
```

## 0. Install

Miners need no GPU. Install the core package with the Hippius (registry push)
and chain (on-chain commit) extras:

```bash
git clone https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[hippius,chain]'      # numpy/scipy + hippius-hub + boto3 + bittensor
```

## 1. Write (or fork) your generator

Start from a reference and edit — the shipped examples are real, deployable
generators:

```bash
cp -r scripts/example_generator my-generator      # minimal trend+seasonal+AR(1)
# or one of the richer priors: gen_changepoint, gen_chaotic, gen_garch, base_generator
```

Your repo directory must contain:

```
generator.py        # exposes `class Generator(DataGenerator)`
config.json         # any JSON your generator reads (band lengths, weights, …)
requirements.txt    # hash-locked deps from the allowlist (numpy, scipy, torch, …)
```

The one hard rule that trips people up: **determinism**. `generate()` must be a
pure function of the `seed` passed to `__init__` — two runs at the same seed
produce byte-identical corpora. Seed every RNG (numpy, torch, `random`) from it,
avoid `hash()`/wall-clock/network. See `INTERFACE.md` for the full contract and
the dependency allowlist (`chain.toml [dependencies]`).

## 2. Verify locally

`cascade verify` runs **every check the trainer runs** — layout, the static
import guard, hash-locked deps, and the determinism check (it builds your corpus
twice and compares digests). Fix anything it flags *before* you spend a
registration:

```bash
cascade verify ./my-generator --chain-toml chain.testnet.toml
# → OK: generator would be accepted by the trainer.
#   corpus_digest (seed=0): 3ff20660d2fd1c55…  [deterministic]
```

A green `[deterministic]` line means the trainer will accept it.

## 3. Make a wallet and register

You need a bittensor wallet (a coldkey + a hotkey) and a UID on the subnet.

```bash
# create keys (skip if you already have a wallet)
btcli wallet new-coldkey --wallet-name my-miner
btcli wallet new-hotkey  --wallet-name my-miner --wallet-hotkey gen1

# register on the subnet (burns a small amount of test/real TAO for the slot)
btcli subnets register --netuid 259 --network test \
  --wallet-name my-miner --wallet-hotkey gen1
# mainnet: --netuid 91 --network finney
```

`btcli subnet list --network test` shows the current registration cost. One
hotkey = one UID = one competing generator; register more hotkeys to run several
priors in parallel.

## 4. Set your Hippius credentials

`cascade deploy` pushes your generator to the Hippius Hub registry, which needs
registry auth in the environment (never in `chain.toml`):

```bash
export HIPPIUS_HUB_USERNAME=...     # or a token: HIPPIUS_HUB_TOKEN=...
export HIPPIUS_HUB_PASSWORD=...
```

You do **not** need S3 credentials — those are the trainer/validator's.

Optionally, set a HuggingFace token if you want the outage fallback in
[§5a](#5a-if-the-hippius-hub-is-down) — it is **only** used when the Hub is down:

```bash
export HF_TOKEN=hf_...               # only needed for `--hf-repo`
```

## 5. Deploy

`cascade deploy` re-verifies locally, pushes the generator to the registry
(content-addressed by `repo@digest`), and commits the on-chain pointer
`metro-v1:gen:hippius:<repo>@<digest>`:

```bash
cascade deploy ./my-generator \
  --chain-toml chain.testnet.toml --network test \
  --wallet-name my-miner --wallet-hotkey gen1 \
  --hub-repo my-namespace/my-generator
# → pushed to Hippius Hub: my-namespace/my-generator@sha256:…
#   committed: metro-v1:gen:hippius:my-namespace/my-generator@sha256:…
```

Re-deploy any time to submit a new version — the latest pre-cutoff commit per
hotkey is the one that competes.

### 5a. If the Hippius Hub is down

Miner submission uploads to the Hippius **Hub** (the OCI registry) — a different
service from Hippius **S3** (which only the trainer/validator use). If the Hub is
having an outage, the upload fails with `registry upload failed: …` (exit 4). Pass
`--hf-repo` to mirror your generator to HuggingFace instead so you can still
submit:

```bash
cascade deploy ./my-generator \
  --chain-toml chain.testnet.toml --network test \
  --wallet-name my-miner --wallet-hotkey gen1 \
  --hub-repo my-namespace/my-generator \
  --hf-repo  my-hf-namespace/my-generator      # fallback, needs HF_TOKEN
# → Hippius Hub upload failed (…); falling back to HuggingFace mirror …
#   mirrored to HuggingFace: my-hf-namespace/my-generator@hf:<sha>
#   committed: metro-v1:gen:hippius:my-hf-namespace/my-generator@hf:<sha>
```

How it works, and what to know:

- **Hippius is priority one.** The Hub is *always* tried first; HF engages **only**
  if that push fails. `--hub-repo` is required — you cannot submit straight to HF
  while the Hub is healthy. (`--hf-repo` alone is refused.)
- **It's a real submission.** The chain commit records `repo@hf:<sha>`, and the
  trainer/validators/auditors fetch, train, and score it exactly like a Hub one —
  the `hf:` ref just tells them to fetch from HuggingFace.
- **Keep the HF repo public and don't delete it** while that commit is your active
  submission — the trainer fetches it anonymously, so a private/deleted repo means
  it can't be evaluated. (A newly-created repo is public by default.)
- **The commit stays on HF until you replace it.** When the Hub recovers it does
  *not* auto-migrate — re-deploy with just `--hub-repo` to move your submission back
  onto the content-addressed Hub (the preferred, audit-anchored form).

## 6. Confirm it's competing

Your commit is now on chain. Verify it two ways:

```bash
# it shows in the revealed commitments for the netuid …
# (the trainer reads these before each epoch boundary)
python - <<'PY'
from cascade.shared.chain import ChainClient
from cascade.shared.config import load_chain_config
cfg = load_chain_config('chain.testnet.toml')
c = ChainClient(netuid=cfg.netuid, network="test")
for cm in c.poll_commitments():
    print(cm.uid, cm.hotkey[:10], cm.payload[:60])
PY
```

Then watch the **public round receipts** (or the dashboard): committed *before*
the epoch boundary, your generator enters the next round's **heat**, gets
trained and scored, and appears in that round's receipt participant set with
your `gen_ref`. If it wins the heat it advances to the full final against the
king. Verify any round independently with `cascade-audit latest` (see
[`AUDIT.md`](AUDIT.md)).

## Study the competition

Every committed generator is content-addressed and **public** — that's what
makes the eval re-derivable, and it makes the current best openly studyable.
Pull the reigning king (or any competitor) and read its code:

```bash
cascade fetch king --network test --chain-toml chain.testnet.toml
# → fetched king-uid3: cascade/testnet-smoothgp@sha256:…
#   inspect it, or fork + improve it:  cascade verify ./fetched-king-uid3

cascade fetch 13 --out ./chal13      # a specific UID
cascade fetch 5Haf…                  # a specific hotkey (ss58)
cascade fetch namespace/repo@sha256:…  --verify   # a raw ref; --verify runs the checks
```

This is the game: the best generator is visible, and you win by **improving**
on it, not hiding — a byte-identical copy of the king is dropped before it
trains (it can only tie), so you have to genuinely beat it. Read-only; no wallet
needed, just Hub read credentials.

## Common failures

| symptom | cause |
|---|---|
| `cascade verify` fails determinism | an unseeded RNG, `hash()`, wall-clock, or set iteration order — make `generate()` pure in `seed` |
| `blocked_import` | a banned import (`socket`, `subprocess`, `pickle`, …); see `chain.toml [static_guard]` |
| `requirement_not_hash_locked` | every `requirements.txt` line needs `--hash=sha256:…`; only allowlisted packages |
| deploy: Hub auth error | `HIPPIUS_HUB_USERNAME`/`PASSWORD` (or `HIPPIUS_HUB_TOKEN`) not exported |
| `registry upload failed` (Hub outage) | the Hippius Hub is down — retry, or add `--hf-repo` + `HF_TOKEN` to submit via the HuggingFace fallback ([§5a](#5a-if-the-hippius-hub-is-down)) |
| committed but never in a receipt | committed *at/after* the epoch boundary → it competes next round; or it failed to train (heat drops it) |
| loses every heat | expected while you iterate — the pool is broad real-world data; widen your prior (mix families) rather than fitting one shape |
