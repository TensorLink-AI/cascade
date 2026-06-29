# Testnet deployment — two VMs

A minimal metronome testnet runs on **two machines**, mapped to the two GPU roles
in `docs/ARCHITECTURE.md`:

| VM | role | package | why | sizing |
|----|------|---------|-----|--------|
| **A — orchestrator** | trainer (owner) | `metronome.trainer` | trains two Toto2-4M models **from scratch** per round (45 min each) and signs the manifest with the owner wallet | **expensive** GPU box |
| **B — validator** | validator | `metronome.validator` | only **scores** the two finished checkpoints on the held-out windows (forward passes, no training) and sets weights | **cheap** box (small GPU or CPU) |

Miners need **neither GPU nor a VM here** — they run `metronome deploy` from
anywhere to upload a generator and commit an on-chain pointer (`docs/INTERFACE.md`).

The asymmetry is the point: training from random init is the cost, scoring is not.
The orchestrator carries the GPU bill; the validator is cheap and can be replicated
for consensus. (If you want the orchestrator itself to stay cheap, it can dispatch
the two trainings to rented GPU pods over SSH instead of training locally — see
[Variant: remote GPU pods](#variant-remote-gpu-pods).)

---

## 0. Prerequisites (both VMs)

* A **Bittensor testnet** wallet. Create a coldkey + hotkey and register the
  relevant hotkey on the test netuid:
  ```bash
  btcli wallet new_coldkey --wallet.name metronome
  btcli wallet new_hotkey  --wallet.name metronome --wallet.hotkey trainer   # on VM A
  btcli wallet new_hotkey  --wallet.name metronome --wallet.hotkey validator # on VM B
  btcli subnet register --netuid <TESTNET_NETUID> --subtensor.network test \
      --wallet.name metronome --wallet.hotkey <trainer|validator>
  ```
* **Hippius storage** reachable from both VMs:
  * an IPFS node (the registry backend) — set `IPFS_NODE_URL`;
  * Hippius **S3** credentials for manifests + logs — `HIPPIUS_S3_ACCESS_KEY` /
    `HIPPIUS_S3_SECRET_KEY` (optional `HIPPIUS_ENCRYPTION_KEY`).
* Python ≥ 3.11.

Credentials are **never** committed — they come from the environment (see
`chain.toml [storage]`).

---

## 1. Shared `chain.toml`

All three roles read the **same** `chain.toml`. Set these before launch (the rest
of the file ships with sensible defaults):

```toml
[subnet]
netuid = <TESTNET_NETUID>          # the test netuid you registered on

[training]
target_train_hours = 0.75          # 45-min screening round (testnet default)
expected_gpu       = ""            # pin to torch.cuda.get_device_name(0) for byte-exact audit

[eval]
window_pool = "<HELD_OUT_POOL_CID>"  # upload your private pool, pin its registry CID

[manifest]
trainer_hotkey = "<VM_A_TRAINER_SS58>"  # the ONLY signer validators trust

[storage]
ipfs_api_url = "http://127.0.0.1:5001"  # or your node; overridden by IPFS_NODE_URL
```

The **45-minute** budget (`target_train_hours = 0.75`) is the short *screening*
round used to rank the field fast on a daily cadence. See
[`OPEN_QUESTIONS.md` #9](../OPEN_QUESTIONS.md) for the intended daily-seed / king
re-use cadence and the longer ~3h king-defence run that layers on top of it.

Upload the held-out eval pool once and pin its CID:
```python
from metronome.shared.hippius import RegistryConfig, upload_dir_to_registry
from metronome.shared.config import load_config
cfg = load_config()
print(upload_dir_to_registry("path/to/held_out_pool", RegistryConfig.from_storage(cfg.storage)).cid)
```

### Registration & queue sizing (testnet)

Testnet UID slots are limited (~64). How many miners can register and how fast is
a **chain hyperparameter on the netuid** (`max_allowed_uids`, registration
cost/adjustment, immunity) — *not* something this repo sets. Cap
`max_allowed_uids` so the registered field can't outrun what the trainer can
screen in a day. The trainer's submission backlog is otherwise unbounded; keep the
processable queue to **≲40** (calibrate to challengers screenable per seed-day).
Tracking item + the queue-cap knob to add: [`OPEN_QUESTIONS.md` #11](../OPEN_QUESTIONS.md).
Dereg/re-deploy of a *queued* miner is already handled — the backlog is pruned to
the live on-chain field each round ([`OPEN_QUESTIONS.md` #12](../OPEN_QUESTIONS.md)).

---

## 2. VM A — orchestrator / trainer (expensive GPU)

```bash
git clone <repo> metronome && cd metronome
python -m venv .venv && . .venv/bin/activate
pip install -e '.[train,hippius,chain]'

export IPFS_NODE_URL=...           # registry
export HIPPIUS_S3_ACCESS_KEY=...
export HIPPIUS_S3_SECRET_KEY=...
# export HIPPIUS_ENCRYPTION_KEY=...   # optional

# Smoke (no chain/GPU): prints the contract digest + derived seeds.
metronome-trainer --offline --trainer metronome.trainer.toto2_trainer:Toto2Trainer

# Live: poll the chain, train king + 1 challenger per round, sign & publish.
metronome-trainer \
  --trainer metronome.trainer.toto2_trainer:Toto2Trainer \
  --network test \
  --wallet-name metronome --wallet-hotkey trainer \
  --max-challengers 1
```

The trainer needs the **owner wallet** (it signs the manifest) — keep that wallet
only here. It reads the reigning king as the highest-incentive UID on the
metagraph (validators own the dethrone decision; the trainer just trains who they
crown — `OPEN_QUESTIONS.md` #3).

### Run it as a service

```ini
# /etc/systemd/system/metronome-trainer.service
[Unit]
Description=metronome trainer (orchestrator)
After=network-online.target
Wants=network-online.target

[Service]
WorkingDirectory=/root/metronome
EnvironmentFile=/root/metronome/.env          # IPFS_NODE_URL, HIPPIUS_S3_* (chmod 600)
ExecStart=/root/metronome/.venv/bin/metronome-trainer \
  --trainer metronome.trainer.toto2_trainer:Toto2Trainer \
  --network test --wallet-name metronome --wallet-hotkey trainer --max-challengers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl enable --now metronome-trainer
journalctl -u metronome-trainer -f
```

---

## 3. VM B — validator (cheap)

```bash
git clone <repo> metronome && cd metronome
python -m venv .venv && . .venv/bin/activate
pip install -e '.[train,hippius,chain]'      # 'train' brings torch for the evaluator

export IPFS_NODE_URL=...
export HIPPIUS_S3_ACCESS_KEY=...
export HIPPIUS_S3_SECRET_KEY=...

# Smoke (no chain/Hippius): prints champion state and exits.
metronome-validator --offline

# Live: poll manifests, score both checkpoints, run KOTH, set weights.
metronome-validator \
  --network test \
  --wallet-name metronome --wallet-hotkey validator \
  --device cpu                                  # or cuda:0 on a small GPU
```

The validator only does **forward passes** on two 4M checkpoints over the eval
windows, so `--device cpu` is viable on a cheap box; a small GPU just speeds it up.
Every validator independently scores the **full** eval set and reaches the same
verdict from the same seeded window slice — redundancy is the consensus mechanism
(`OPEN_QUESTIONS.md` #10). It gates every manifest on the trainer-hotkey signature,
so VM A's `trainer_hotkey` in `chain.toml` must match.

Service unit is the same shape as VM A's, with
`ExecStart=.../metronome-validator --network test --wallet-name metronome
--wallet-hotkey validator --device cpu`.

---

## 4. End-to-end check

1. **Both smokes pass** — `metronome-trainer --offline …` prints a contract digest;
   `metronome-validator --offline` prints state. Same `chain.toml` ⇒ the trainer's
   contract digest is what the validator will require.
2. **A miner deploys** — `metronome deploy ./generator --wallet-name … --wallet-hotkey …`
   uploads a generator and commits `metro-v1:gen:hippius:<cid>` (`docs/INTERFACE.md`).
3. **A round completes** — VM A logs `published manifest round=… entries=2`; the
   manifest lands in the S3 `manifest_bucket` (`round-<id>.json` + `latest.json`).
4. **The validator acts** — VM B logs `round=… lcb=… win=… king=…` and sets weights
   (`btcli wallet overview --subtensor.network test` shows the king's incentive).

---

## Variant: remote GPU pods

To keep the orchestrator cheap, train the king and challenger **in parallel on
rented GPU pods** (Lium/Targon) over SSH while VM A only orchestrates and signs:

```bash
metronome-trainer \
  --trainer metronome.trainer.toto2_trainer:Toto2Trainer \
  --network test --wallet-name metronome --wallet-hotkey trainer \
  --remote-hosts hosts.toml
```

`hosts.toml` lists the pods (see `scripts/remote_hosts.example.toml`). The pods run
`metronome-train-worker`, need torch + a GPU + registry/S3 access, and **never hold
the wallet** — signing/publishing stays on VM A. For byte-exact re-derivation, rent
the **same GPU SKU** on both pods and pin it in `chain.toml [training] expected_gpu`
(the validator then rejects any round not trained on that SKU). `hosts.toml` is
trainer-local — keep it out of git.

---

## Notes / open items

* **45-min screen vs. daily king run** — this default trains the king *every*
  round, which is wasteful when idle; the intended cadence is one seed-init per day
  (train king once/day, defend it against challengers) plus a longer ~3h run. Knobs
  to add: [`OPEN_QUESTIONS.md` #9](../OPEN_QUESTIONS.md).
* **Scaling the fixed model** (interleaving 4M/20M sizes, best-checkpoint
  warm-starts): [`OPEN_QUESTIONS.md` #13](../OPEN_QUESTIONS.md).
* **Eval battery as a binary gate** (BOOM / GIFT-Eval / TIME / Tempus-bench /
  DS-systems): [`OPEN_QUESTIONS.md` #15](../OPEN_QUESTIONS.md).
