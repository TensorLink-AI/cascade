# Validator guide — score settlements and set weights

Validators check published evidence and set the reward weights on chain.
Choose your role before installing: satellites replay signed results; full
evaluators also run private forecasting tests. Neither trains challengers.

Mainnet v2 private verdicts use `[v2] eval_n_windows = 4000`; the settlement
at activation still judges the preceding era under its earlier budget. Run
the same release/config as the other validators before that future era.
New pool snapshots record conservative source-family cluster keys in `source`;
correlated feeds resample together. Historical snapshots retain their recorded
keys. Counts of windows and independent clusters are different evidence;
see [pool rules](EVAL_POOL.md#v2-pool-size-channel-width-and-evidence).

## Choose your role

| Role | What you run | What you need | Setup |
|---|---|---|---|
| Satellite | Verify the main validator's signed receipts, replay decisions and derive weights | A CPU host, chain/wallet access and read access to published evidence | [Satellite setup](#satellite-validators-dec-ca-0058) |
| Full evaluator | Check trainer records, fetch checkpoints and score the private pool; replay the main validator's public benchmark and MarketCast evidence | Model dependencies, private-pool access and suitable inference compute | [Configure](#configure-your-validator), [install](#setup), [run](#run--this-is-the-validate-command) |
| Designated main validator | Full evaluation plus the public benchmark sweeps and MarketCast forecasts when enabled | GPU capacity, benchmark environment/data and report-write access; identity must match the configured owner | Full-evaluator setup plus the [operator runbook](OPERATOR.md#lane-a-benches-on-the-main-validators-gpu-dec-ca-0063) |

The sections through **Run** describe full evaluators. A satellite can start at
its linked setup. All roles need the correct network/release and durable state.

V2 Lane A targets **30 October 2026, 12:00 UTC**, at the first era boundary at
or after that time, derived from finalized chain timestamps. The separate 95%
pre-launch burn waits for 51% of eligible validator stake to signal this
release's `prelaunch-95-r2` feature; it starts at the next settlement boundary.
Installing the release does not activate B/C. Preserve activation and receipt
state when upgrading, and check fresh status before assuming a lane is open.

The paid A entry window opens **30 October 2026, 09:00 UTC**, three hours
before the noon launch target. Its reveal floor is resolved independently
from finalized timestamps (`[v2] commit_from_block`). Trainer dispatch waits
for actual v2 activation. Receipts keep the historical floor for v1 and the
v2 opening floor for v2. C can launch before B when it takes its share from A;
the A+C allocation is 75%/25%, becoming 50%/25%/25% when B launches.

The mainnet template pins `[netconfig] owner_hotkey` to the same public main
validator address as `[manifest] validator_hotkey`. It signs benchmark reports
and network policy. Satellites and full validators must use that release pin;
an empty owner makes the scheduled validator-sourced benchmark contract
unloadable and services refuse startup. The trainer may share that signer;
these receipts then share its trust anchor rather than being independent
attestations. A separate deployment must configure its own public identities.
Dated launch replay needs chain timestamps (archive access for late catch-up),
or an explicitly resolved `[v2] from_block`. Offline audits defer unresolved
calendar rules; satellites hold rather than applying pre-v2 rules by default.

**Reference:** [competition rules](#competition-rules) · [rewards](#rewards)
· [failure handling](#what-can-go-wrong) · [signed configuration](#owner-signed-network-config-dec-ca-0057).

**Terms:** a *manifest* is the trainer's signed training record; a *receipt*
is a validator's signed decision and weight record. A *settlement* (round)
judges completed work. An *era* fixes the training rules and starting model.
The *king* is A's current winner; the *warm start* is the shared checkpoint
from which a new training run starts. Those checkpoints can differ.

## How a settlement reaches you

Every epoch boundary (~3 h) the trainer publishes at most ONE manifest, `manifests/round-<id>.json`, carrying
the era king's entry plus every challenger finished since the previous
settlement. With no finished A challenger, king-only envelopes can still advance
B/C work, confirmations and validator-benchmark acknowledgements. A missing or
incompatible trained king prevents publication. Manifests are
hash-chained (`prev_round_id`): your validator walks them **forward from the
last settlement it handled**, never jumping to `manifests/latest.json`, so after
downtime it catches up through every settlement, dethrones included. A walk
that cannot reach your last handled settlement (a missing `round-<id>.json`)
handles nothing that poll and retries deeper.

An *era* is `[round] era_settlements` (4) consecutive settlements — ~12 h —
pinning a compatible contract and init. With v2 `king_ladder`, several eras
can share a seed epoch and its trained king; do not assume every era retrains
the king with new seeds. Each
manifest's `era` stamp is checked against the era derived from its boundary,
the init against the promotion ledger, and every entry's ref against the
hotkey's on-chain commitment as of the entry's `train_block`; any mismatch
rejects the manifest (`era_*` reasons below). Tenure and promotion ripeness
are counted in blocks (`[scoring] margin_warmup_blocks`,
`cascade_reign_blocks`).

The grid is 900 blocks. `chain.toml` ships `[round] epoch_blocks = 3600`;
your validator applies `[activation] epoch_blocks_after = 900` from the
rolling-intake rollover block decided on chain (DEC-CA-0045, below), along
with rolling intake, the era king and tenure in blocks. Do not edit
`epoch_blocks` yourself.

## Configure your validator

**1. Use the release configuration for your network.** Do not independently
tune scoring or reward constants. Select the matching file:

| network | file | netuid |
|---|---|---|
| mainnet (`finney`) | `chain.toml` | 91 |
| testnet (`test`) | `chain.testnet.toml` | 259 |

Use the operator's validated release configuration. Checked-in templates can
contain inactive gates and unqualified deployment values. `[training]` defines
the training contract manifests are gated on (edit a locked term and you reject
every valid manifest), `[scoring]`
and `[round]` define the verdict every honest validator must compute
identically, `[manifest] trainer_hotkey` is the only trainer to trust, and the
buckets are where the subnet's data actually lives. Your identity needs no toml
entry — receipts are signed with your **wallet hotkey** automatically.

`[manifest] validator_hotkey` also participates in main-validator benchmark
identity checks: when set alongside `[netconfig] owner_hotkey` for validator
benchmarks, they must agree. Do not replace it with each satellite's identity.
For auditing a particular validator, use `cascade-audit … --validator
<their-hotkey>` instead; an unpinned receipt signer is reported as WARN.

**2. Storage credentials** for the selected role (env only, never in the toml):

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

**3. Command-line flags** — wallet and network for the standard full-validator setup:

```bash
cascade-validator --chain-toml chain.toml --network finney \
  --wallet-name my-validator --wallet-hotkey default
# testnet: --chain-toml chain.testnet.toml --network test
```

The main evaluator also needs inference/benchmark configuration where enabled;
satellites have a separate small installation below. `cascade-validator --help`
lists local device, offload, cache and release-management options.

## Setup

Use a patched installer: uv >=0.13.4, or pip >=26.2.2 with a patched OS CA
store. This checkout requires certifi >=2026.10.19. After installing, run
`python scripts/check_tls_trust.py` with the same interpreter as the CLI.
See [TLS setup and worker rollout](OPERATOR.md#tls-trust-and-the-october-2026-emudhra-incident)
if the check fails; never bypass certificate verification.

```bash
git clone --branch cascade-v2 https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[train,hippius,chain]'
```

The install pins `bittensor==11.3.0` — **don't change it**: the port to the
11.x SDK (DEC-CA-0051) keeps the subnet's reveal encoding, the activation
tally's stake source and the u16 weight quantisation bit-identical to the
10.5 fleet, and other SDK lines differ on at least one of those. Upgrade
with `pip install -e '.[train,hippius,chain]'` (or `uv sync --all-extras`)
and restart; 10.5-era signatures, keyfiles and on-chain reveals all keep
working.

Register and stake as you would on any subnet (netuid above; weight-setting
needs a validator permit, i.e. stake above the subnet threshold).

Smoke-test the config with zero chain/GPU I/O:

```bash
cascade-validator --offline --chain-toml chain.toml
# prints netuid, king, dethrone_cp, manifest bucket, eval-pool source
```

## Run — this is the validate command

```bash
# mainnet (chain.toml and --network finney are the defaults, shown for clarity):
cascade-validator --chain-toml chain.toml --network finney \
  --wallet-name my-validator --wallet-hotkey default

# testnet:
cascade-validator --chain-toml chain.testnet.toml --network test \
  --wallet-name my-validator --wallet-hotkey default
```

Keep it under a process manager (systemd/tmux); it resumes cleanly from its
persisted state. A healthy settlement looks like:

```
new manifest round=… entries=3 (king:uid…,challenger:uid…,challenger:uid…); gating + scoring …
round=… k=2 decided=… lcb=… margin=… win=… king=… tenure=…
round=… weights set: reward_uids=[…] (n_uids=…, burn_uid=0)
published scored receipt round=… signed=True → s3://…/receipts/<your-hotkey>/round-….json
```

## Competition rules

Judge each settlement with the contract and signed configuration in force at
its boundary. Checked-in templates are not proof that a feature is live.

| Lane | Comparison | Required decision | Payment starts |
|---|---|---|---|
| A | Fully trained challenger versus the era king, with the pinned starting model and seeds | Private confidence-bound gain must clear the current dethrone margin; any active public benchmark gate must also pass | A v2 crowning sets the king's share; scheduled king checks later change it |
| B | The selected-expert update versus its compatible king/base | Each declared target must improve; other-domain, horizon and public benchmark regression guards must pass | With acceptance enabled, only committed acceptance at a scheduled warm start replaces paid holders |
| C | MarketCast fine-tune versus the untouched pinned Timestral base | Eligible future windows, gain/evidence thresholds, calibration and duplicate checks, then confirmation on fresh windows and the slot-holder comparison | A confirmed win that takes the target slot |

A's mainnet v2 public gate uses GIFT-Eval, BOOM, TIME and FEV. It requires a
suite-equal overall task ratio below 1, no suite's bootstrap ratio bound above
1 plus the configured tolerance (default 1%), and a suite-equal task win rate
of at least the configured minimum (default 50%, ties count half). Missing
required rows cannot establish a win. [Benchmark evidence](#cascade-specific-rules-the-things-other-subnets-dont-have)
below explains report binding, timeouts and fault classification.

B accepts up to three declared targets under the mainnet template. The trainer
selects their expert rows; the leg trains their union under one budget. The
measured 200B-lineage-point and routing-stability gate must open before B
starts. During merge acceptance, compare each update with the checkpoint
accepted so far, and test regression limits against the original init. Check
the permitted weight rows, original base, intermediate pointers and signed
evidence. Rejected updates neither merge nor earn. Deferred updates remain
unpaid, with existing eligible holders left in place.

C currently supports only the configured MarketCast targets, returns and
volatility. Its base is separate from the A/B lineage; its weights never merge
into A/B. The first pass creates a pending confirmation. Confirmation uses
windows starting after that first snapshot ended and must also take the slot;
a scientific pass below the holder's required bound is unpaid.

Infrastructure deferrals are not research losses. Store/read failures hold or
retry according to the bound evidence; malformed/non-finite model output can
be a checkpoint fault. Queue completion is not a verdict. Use the receipt's
stage, error class, reward state and exact report binding when explaining a
miner's result. The [miner guide](MINER.md#7-how-the-verdict-works) gives the
participant-facing rules and the [architecture](ARCHITECTURE.md) maps them to code.

## Cascade-specific rules (the things other subnets don't have)


### Training contract changes

Restart after an announced change to a locked `[training]` term. The
trainer publishes its full training contract inside each signed manifest,
and your validator gates only the terms it consumes — `arch_preset`,
`base_arch_digest`, `expected_gpu`, `train_seed_salt`, and each extra size's
preset / GPU pin. A mismatch there rejects every manifest
(`contract_locked_mismatch`) until you pull and restart. Every other
`[training]` change (worker-image re-pin, optimizer or schedule, budget)
needs nothing from you: your validator keeps scoring, and `cascade-audit`
reports the moved fields as a `contract-declaration` WARN.

### Scoring-rule changes

Adopt scoring-rule changes through the supported coordinated path.
New code and release-only fields require an upgrade before the announced
boundary. Supported numeric/rule settings can use
[owner-signed overlays](#owner-signed-network-config-dec-ca-0057) once the
installed release supports them; adoption checks and durable history govern
the effective block. Manually diverging `[scoring]`, `[eval]` or the cascade
envelope can still produce different verdicts and weights. For a release,
pull, `uv sync --frozen`, and restart before the announced activation block.
For a supported overlay, follow its adoption/effective-block evidence without
manually changing constants. Confirm the applied config identity and next
settlement's vector agree with the expected network state.

### Stake-based activation

Some changes are decided on chain by validator stake (DEC-CA-0045). At
startup your validator posts one plain commitment from your hotkey — the
readiness note, `cascade-ready:1:rolling-era-king:<lock>:<block>` plus one
`<feature>:<lock>:<block>` segment per pending on-chain decision (king
forfeiture `forfeit-<hash>`, dethrone bar v2 `margin-v2-<hash>`). You'll see
`activation: signalled …` in the log. Posting is a signed `set_commitment`
(the intent is logged before signing; a rejected extrinsic is retried next
poll). At every boundary every validator sums the stake of permit-holding
validators behind each segment **as of the boundary block**; the first
boundary at or over `[activation] threshold` (51%) locks that feature in,
one-way, and it applies from the next boundary (margin v2: the next era
start). A lock-in logs `activation: <feature> LOCKED IN at boundary B (…)
— rollover at block R`; for the rolling-intake feature the validator then
logs `activation: DEC-CA-0043 rollover ARMED at block R …` as it writes
the block into its live config; the new rules apply from block R.
Lock-ins are persisted beside your state (`activation_state.json`,
`activation_forfeit_state.json`, `activation_margin_v2_state.json`),
stamped on receipts (`activation_block`), and written back into your note so a validator that
restarts or joins late adopts them from the notes alone. Lock-in is
one-way: stake drifting back under the line changes nothing. The count is
taken as of the boundary block; if your endpoint cannot serve it (a pruned
node) the validator logs `activation: as-of read at boundary B failed …`
and retries THAT boundary — never a later view — while it is young. Once
the boundary is older than the endpoint's prune window (300 blocks) and no
validator's note names a lock-in there, it is skipped (`boundary B is
unreadable here … skipping it`) and the next boundary is counted as of its
own block; a lock-in named in the notes is always adopted. A validator on
a release without a feature cannot parse the extended note and counts you
as not signed — install announced releases promptly; a validator still on
the old release when a feature applies falls out of consensus until
upgraded. A block typed into `chain.toml` always wins (the owner override).
`status/chain.json` carries the live tally (`activation.tally`,
`activation.forfeit`, …) for the dashboards.

### Compute and offloading

Compute depends on the role. Satellites replay signed evidence on CPU.
Full validators run private-pool inference and need the model stack and
suitable compute. Public benchmark evaluation can
be offloaded with `--eval-hosts eval_hosts.toml`, re-read at every eval.
Entries for GPUs on the same VM are combined; other machines are not pooled.
A missing/empty file just
means local evals. Your wallet and all consensus decisions never leave your
box. Exception: the MAIN validator (`[netconfig] owner_hotkey`) needs a GPU
once `[v2] bench_source = "validator"` is in force (it runs the lane-A
benches, DEC-CA-0063) and for MarketCast (DEC-CA-0064). Other full validators
replay those benchmark/MarketCast rows but still score the private pool.

Prepared Targon GPU VMs use this SSH interface; use the assigned port/user and
stage GIFT-Eval, BOOM, TIME and the [pinned FEV manifest](../benchmarks/README.md#pinning-fev-data).
The automatic Targon provider currently provisions CPU generation VMs only.
Public benchmark invocations use the frozen lock and enable both needed extras.

Private evaluation can run on the validator's `--device cuda`, or separately
with `--private-eval-hosts trusted_eval_hosts.toml`. That explicit flag sends
selected histories to the operator-owned GPU VM; targets, metadata, wallets
and score calculation stay local. A missing host, timeout or invalid reply
holds the round instead of switching to local inference or penalizing a miner.
Transient directories are removed after success/failure; cleanup failures are
logged for the operator to remove on the worker.

`--private-eval-workers N` defaults to `1`. Parallel workers each load a model
and keep the original per-window forecast call and deterministic draws; local
parallel mode limits torch to one CPU thread per worker. Size this explicitly
for GPU/RAM capacity. Qualify exact sequential/parallel forecasts on the intended
GPU/runtime before enabling it; the [operator procedure](OPERATOR.md#targon-gpu-evaluation)
covers this check. `--private-eval-timeout` bounds the complete offload, including
transfers (default 3,600 seconds). Other full validators still need a compatible
private evaluation runtime; the main validator's public-report replay does not
replace their private inference.
Private requests bind torch/numpy versions and numerical backend switches;
a worker applies those switches and refuses incompatible versions. CPU and CUDA
use different sampling backends, so qualification must use the intended device
backend and full validators must coordinate that choice before a live switch.

### Optional GIFT-Eval gate

`gift_gate_mode = "off"` ships in `chain.toml`. Under
`"shadow"` you see one `gift-gate round=… passed=…` log line per duel,
log-only; under `"enforce"` failing the gate vetoes takeovers and needs the
pinned bench data (`[eval] gift_gate_data_dir`) or an eval pod.

### Lane A public benchmark gate

This check can block a challenger that passed the private test. It reads a
signed benchmark report and its sha256-pinned per-item rows. Replaying this
report needs no benchmark GPU; full evaluators still need private-pool
inference compute. The
mode is `[v2] bench_gate_mode = "enforce"` on every v2 settlement, and
`[scoring] bench_gate_mode` (off) before v2. The rule depends on the
settlement:

- **DEC-CA-0054** (default). The challenger may not be significantly worse
  on GIFT-Eval, BOOM or TIME (paired bootstrap of per-item geomean(CRPS,
  MASE) at alpha/3), and its six-number geomean must be strictly better.
- **DEC-CA-0061** (`bench_gate_rule = "avg_noninf_winrate"`), for eras
  starting at or after `[v2] bench_gate_v2_from_block` (mainnet: `"v2"`,
  the v2 block). FEV becomes a fourth suite and the unit is one task. The
  challenger passes when all three hold: the suite-equal average of its
  per-task CRPS/MASE ratios to the king is below 1; no suite's
  paired-bootstrap ratio bound at alpha/4 exceeds 1 + `[scoring]
  bench_gate_tolerance` (0.01); and its suite-equal task win rate (a tie
  counts half) is at least `[scoring] bench_gate_min_win_rate` (0.5).

Who runs the benchmarks: before `[v2] bench_source_from_block` (and on testnet, where
`bench_source` stays `"trainer"`) the rows come from the trainer-signed
`benchmarks/round-<id>.json`. From it on mainnet (`bench_source =
"validator"`, `bench_source_from_block = "v2"`, DEC-CA-0063) they come
from the MAIN validator (`[netconfig] owner_hotkey`), which sweeps on its
own GPU only a challenger that cleared the pool, plus the cached king (and
king + init when the king check is due), and signs
`bench_v/round-<id>.json`. Every other validator HOLDS that settlement
until a later trainer-signed manifest records the report's sha256 (or
`"absent"` after `[v2] bench_report_timeout_settlements`; its clearers are
then infra, never a loss), and judges from exactly those bytes. An
`infra_failed` or `budget_deferred` entry is never a loss; a
`checkpoint_fault` fails the gate. The main validator needs a GPU
(`--device cuda` or `--eval-hosts`) and the benchmark data staged
(`cascade-benchmark-download`); `[eval] bench_workers` /
`bench_paired` size its sweep and `[v2] bench_sweep_hours` /
`bench_budget_hours` bound it (per checkpoint / per settlement).

Under the v2 task gate, a challenger-only missing, duplicate or non-finite
required score fails the comparison. An unavailable suite/report remains an
infrastructure deferral; it cannot establish a win. Tasks unusable on the king
side can be masked only up to 10% of a suite. Capped or incomplete reports cannot
establish a pass. Under enforce, an uncomputable gate leaves the king in place.
A store read failure retries the settlement. Log line:
`bench-gate round=… mode=… rule=… passed=… reason=…`. Receipts stamp
`params.bench_gate_rule`, under DEC-CA-0061 also
`verdict.bench_gate_detail` (`avg_ratio`, `worst_lcb`, `win_rate` and the
per-suite numbers), and under DEC-CA-0063 `bench_report`. The audit
`bench-gate` / `bench-source` / `bench-coverage` checks replay them from
the signed rows.

### Check your receipts

`cascade-audit latest --config chain.toml --network
finney --validator YOUR_SS58` checks your latest signed receipt (signatures,
seeds, digests, era, verdict and weights). Without `--validator`, `latest`
can select another validator's receipt from the shared pointer. Investigate
FAILs and read WARNs for missing evidence. See
[`AUDIT.md`](AUDIT.md).

## What can go wrong

| symptom | cause |
|---|---|
| `rejecting manifest … contract_digest_mismatch` | a manifest without a published contract body whose digest differs from your `[training]` — pull the current file AND restart |
| `rejecting manifest … contract_locked_mismatch` | a LOCKED contract term (arch, size, GPU pin, seed salt) differs — pull + restart; unlike the declared terms these are never waved through |
| `rejecting manifest … contract_body_mismatch` | the trainer's published contract doesn't hash to the digest it declared — do NOT restart into this; report it, the two claims disagree |
| `rejecting manifest … signature_invalid` | wrong `[manifest] trainer_hotkey` (don't edit it), or the trainer published unsigned |
| `rejecting manifest … era_mismatch` / `era_init_mismatch` / `era_king_pointer_mismatch` | the manifest's era stamp, init or king pointer disagrees with what your state derives — check you run the current release; report it if you do |
| `rejecting manifest … era_ref_unverifiable` / `era_ref_mismatch` | an entry's ref could not be checked (or does not match) against the hotkey's commitment history at its `train_block` — usually a chain endpoint without history |
| `manifest chain: … does not reach this validator's last handled settlement` | a `round-<id>.json` is missing from the bucket; the validator retries deeper each poll — report it if it persists |
| `no eval-pool snapshot published` | owner hasn't published a snapshot, or wrong pool creds |
| weights never set | no validator permit (insufficient stake), or the extrinsic fails — check `btcli` and `weight set failed` log lines |
| every pool read `SignatureDoesNotMatch` | you set `POOL_S3_*` with the default endpoint — unset them |
| gift gate / bench slow | those want a GPU — `--eval-hosts`, or pin `gift_gate_data_dir` on a CUDA box |
| audit WARNs on `block-hash-onchain` | lite node without history — use an archive node for zero WARNs |
| `activation: as-of read at boundary B failed …` repeating | your endpoint pruned the boundary block; it retries while the boundary is under 300 blocks old, then skips it unless a note names a lock-in there — an archive endpoint avoids it |
| `lane_c_unverifiable: … registration of slot holder … as of block B unreadable` holding | MarketCast deregistration eviction reads the metagraph AS OF the settlement boundary (historical state): your endpoint pruned it — the settlement holds (retried every poll) until it reads; an archive endpoint is required while lane C has slot holders |
| `marketcast round=R: the chain client has no uid_for_hotkey_at_block hook` | this client build cannot read historical registrations: a deregistered holder keeps its slot on this validator (verdicts unchanged, expiry still evicts) — run the current release against an archive endpoint |

**Emergency: stop endorsing without going dark.** If something looks wrong —
a suspect manifest, a scoring result you don't trust, a possibly-compromised
key — don't stop the process (your last weights linger on chain and you go
Yuma-inactive). Instead restart with `--force-burn` (or set
`[validator] force_burn = true`): every weight-set burns to the subnet's
burn UID while your `last_update` stays fresh, so you remain an active
validator endorsing nobody. Your champion state keeps evolving normally on
disk; drop the flag and restart to resume voting it. Every push logs
`FORCE-BURN active` so it can't be forgotten.

## Rewards

Use the settlement's effective policy; validator operators do not set their
own payout constants. Historical settlements retain the older geometric court
rule. For mainnet v2:

<!-- cascade:rewards:summary:start -->
These are the checked-in mainnet v2 rules. Check live activation and signed policy before submitting.
Percentages are shares of the competition reward vector. A lane's pool is its budget; miners receive only the shares they earn.

| Launched lanes | A pool | B pool | C pool |
|---|---:|---:|---:|
| A | 100% | 0% | 0% |
| A + C | 75% | 0% | 25% |
| A + B | 75% | 25% | 0% |
| A + B + C | 50% | 25% | 25% |

- **At launch:** a separately activated pre-v2 policy burns 95%. Until the first v2 winner is crowned, the inherited king/court receives at most 5% of total emissions; the rest of A burns.
- **A winner:** a random-init or other level-unit win starts at 50% of A. Increment-unit wins use the duel's lower confidence bound (LCB): LCB ≥ 0.5 → 80%; LCB ≥ 0.25 → 65%; LCB ≥ 0.1 → 50%; LCB ≥ 0.01 → 25%. These are the duel's increment units, not percentages of absolute forecast-error reduction.
- **A progression:** each passed scheduled king check adds 10 percentage points of A, up to 95%. Failure sets 10%; a deferred check holds the share. Time alone does not increase it.
- **Former king:** only the immediate former king can receive 5% of A for 36,000 blocks, while eligible. A qualifying compute-key lapse reduces the current king's payout to 1% of A, without reducing separately earned B/C rewards.
- **B winner:** with merge acceptance enabled, qualification is unpaid. Acceptance into a scheduled shared starting checkpoint starts payment. The 10 targets each reserve 2.5% of total emissions at full pay.
- **C winner:** an initial test pass is unpaid. A later confirmation must win the target slot. C's pool is divided by competition weight, then target count. The current sole competition, MarketCast, has 2 targets: each full-pay slot is 12.5% of total emissions.
- **Full-pay duration:** B: LCB ≥ 0.05 → 10 eras, LCB ≥ 0.02 → 5 eras, LCB ≥ 0.01 → 2 eras; MarketCast C: LCB ≥ 0.05 → 10 eras, LCB ≥ 0.02 → 5 eras, LCB ≥ 0.01 → 2 eras. B uses accepted marginal gain; C uses confirmed gain. Each win keeps its recorded tier.
- **B/C progression:** after full pay, B's per-era slot multipliers are 50% → 25% → 12.5%, then a 10% floor. Payment ends at age 28 eras, including the full-pay period. C has its own policy (currently a 10% floor and age 28 expiry). Age starts at B acceptance or C confirmation. A new holder stops the old holder's pay.
- **Burn:** unpaid A, empty/decayed/expired slots in launched lanes and inactive competition allocations go to `burn_uid`; they are not shared among other miners. An unlaunched lane has no reserved pool.
- **Check payment:** read signed reward records and on-chain weights. Queue completion, qualification and pending confirmation are not paid wins. Public status reports `reward_status=not_verified` because it does not verify payment.
<!-- cascade:rewards:summary:end -->

Derive the vector from `emission`, accepted B holders and confirmed C holders;
verify expiry, registration and all remaining burn. Do not normalize rewards
among the non-burn recipients. `cascade-audit` replays this accounting.
The [complete schedule](MINER.md#rewards-allocation-burn-and-progression)
includes every clock, key-recovery condition and a worked allocation.

## King forfeiture (DEC-CA-0048)

`[scoring] forfeit_hotkeys` / `forfeit_from_block` / `forfeit_successor_hotkey`
are consensus constants, inert while the list is empty. From the block, every
validator strips the listed hotkeys from the throne and the payout court and
crowns the named successor (else the most recent eligible former king, else a
vacant throne) — on the first poll that reaches the block, no manifest needed
(standing weights are re-pushed as soon as the chain's rate limit allows); a
settlement whose boundary reaches the block applies the same, idempotently.
With `forfeit_from_block = 0` and a non-empty list the block is decided on
chain through the readiness note's `forfeit-<hash>` segment (see above); a
typed block is the owner override, and then every validator must carry the
same value before that boundary or weights fork there.

## cascade v2 go-live (DEC-CA-0050)

V2 starts at the configured era boundary, `[v2] from_block`. There is no
separate v2 readiness vote: validators must already support its code and
contract. A release can set the block in `chain.toml`; an installed compatible
fleet can also adopt this supported field through an owner-signed overlay,
which has its own stake/adoption rules. Upgrade before activation or the
validator can judge the same settlement under different rules.

- `from_block = 0` (checked-in mainnet template): v2 is off; nothing below changes.
- From the first era starting at `from_block`:
  - Lane A is judged under the `[v2.training]` contract (Cascade MoE Base - 1.23B total parameters):
    the contract gate, `base_arch_digest`, the checkpoint guard and the
    eval all resolve the settlement's contract at its ERA START (pure block
    arithmetic). The settlement AT `from_block` still belongs to the last
    toto2 era.
  - The era envelope requires random init for the new lineage: a toto2
    generation never initialises a v2 era. The random-init window comes from
    `[v2] random_init_eras` (checked-in mainnet: 14), in addition to the
    protocol's minimum warm-start offset; do not assume the third era is warm.
  - Lane A keeps the full allocation until B or C launches independently
    (`from_block = 0` means off). C taken from A may launch first; A/C then
    split 75/25. B's launch must be an era start at or after v2;
    the settlement AT its launch still uses the preceding era's allocation.
    B reserves 25% when launched: A/B split 75/25 while C is off, and
    A/B/C split 50/25/25 while C is on. B's share has one slot per
    `[lane_b]` domain plus one per `[eval]
    scored_horizons` rung (mainnet 7 + 3 = 10; testnet 7), each win paid
    100 % for the `full_eras` of the highest `[lane_b] full_pay_tiers`
    threshold its gain LCB meets (fixed at the win), then halving per era to
    a 10 % floor, and nothing after 28 eras; the unpaid remainder burns.
    `replay_settlements` defaults to cover that window. With `[lane_c]`
    armed, mainnet C takes its share from A (A 50 / B 25 / C 25). The
    explicit `takes_from = "lane_b"` retains the existing testnet split.
    Lane-C recovery replays back to its saved boundary, or activation for
    missing/torn state. Its holders and pending report identities survive
    payout expiry, so `lane_c.replay_settlements` cannot truncate recovery.
  - Every receipt stamps `v2_block`; `cascade-audit` checks it (`v2`).
  - From `[lane_b] acceptance_from_block` (mainnet: `"v2"`; testnet: 0),
    individual passing legs are unpaid qualifications. A warm-start merge
    must carry signed intermediate checkpoint pointers and public benchmark
    rows. Validators rebuild each step, re-score its private windows, require
    marginal target gain against the accepted checkpoint so far, and check
    regression limits against the original init across every domain and
    horizon. Only accepted targets replace paid holders, with the clock at
    that warm start and tiers from the accepted marginal gain. Replays keep
    that clock; rejected combinations neither merge nor earn and cause no
    slash. Public rows retain the existing trainer-evidence trust model.
- Upgrade checklist: pull the v2 release, keep `chain.toml` in sync with the
  published one (the `[v2]`, `[v2.training]` and `[lane_b]` tables), restart
  before the block. Nothing else to configure.

## Owner-signed network config (DEC-CA-0057)

This release uses `cascade-netconfig/2`. Overlay bodies bind both the chain
deployment (`[netconfig] network_id`) and `netuid`; v1/unbound envelopes must
be replaced with a freshly reviewed, signed v2 envelope. Burn destinations
(`scoring.burn_uid`, `lane_b.remainder`) are release-only. Existing nodes
retain rollback protection in durable overlay history. Bootstrap a replacement
node from authenticated state; a rolled-back store cannot prove its own latest
version to a fresh node without a trusted external checkpoint.

Verdict and payout constants can change without a release: the owner signs
an overlay and the fleet adopts it by stake. You configure nothing beyond
keeping `chain.toml` current.

- `[netconfig] owner_hotkey = ""` (both checked-in templates): completely
  inert. Nothing is fetched or signalled and every config, note and receipt
  is byte-identical to a release without the feature.
- With the owner key set (the owner's main validator hotkey), every poll
  re-reads `config/network/latest.json` from the manifest bucket (every
  `poll_seconds`, default 300). Your validator needs READ access on
  `config/network/` beside `manifests/`. It verifies the owner's signature
  and the `seq` / `prev` hash chain, refuses a rollback or a fork, and
  re-parses `chain.toml` + the overlay through the normal loader (era
  boundaries, seams, lane checks). An overlay may only touch `[scoring]`,
  `[eval]`, the `[round]` schedule keys, `[v2]` runtime policy,
  `[lane_b]`, `[lane_c]` and the listed numeric `[marketcast]` controls;
  anything else is refused and logged. Run `cascade-validator
  --netconfig-policy` to inspect the keys supported by your installed
  release without contacting the chain or storage.
- A verified overlay is signalled as a `cfg-<8 hex>` segment of your
  readiness note (DEC-CA-0045). 51 % of permit stake locks it in. It applies
  at its `effective_block` (an era start) if the lock-in lands before it,
  otherwise at the first era start after the lock-in. An overlay whose
  changed `*_from_block` values would lie before that block, or that would
  move the grid before it, is refused and never applied (log line
  `netconfig: … REFUSED`).
- Adopted overlays are kept as history in `netconfig_state.json` beside
  your state file (atomic writes with `.bak`). Do not delete it: a node
  without it judges replays and catch-up under the wrong config. Every
  settlement is judged with the config in force at its block, and its
  receipt stamps `netconfig_seq` / `netconfig_digest`. `cascade-audit`
  fetches that overlay and re-derives under it (`netconfig` check).
- The note is 128 bytes. When it overflows, the `cfg-` segment is the last
  one dropped. An overlay that types a decided block (`forfeit_from_block`,
  `margin_v2_from_block`, …) removes that feature's segment.
- These still need a release: new code, locked `[training]` / `[v2.training]`
  terms, the trainer or owner hotkey, the `[storage]` keys you read, and the
  `[chain]` SDK pin. MarketCast's base pins, activation, suites, data
  sources and window construction remain release-only. Existing benchmark
  source selection, rule activation, timeouts and compute budgets, and the
  listed MarketCast numeric thresholds/pay tiers can use overlays. A node
  needs the release supporting these keys once; later supported changes
  need neither a restart nor an update. An overlay cannot install new code.

## Satellite validators (DEC-CA-0058)

A satellite does not evaluate anything. It replays the main validator's
signed receipts and sets the vector that replay derives. It needs no GPU, no
eval pool, no checkpoints and no torch. The main validator is the owner's
validator whose hotkey is `[netconfig] owner_hotkey` (the same key that signs
the DEC-CA-0057 overlays). Satellites are inert while that key is empty.

**Install (small CPU box, no `[train]` extra):**

```bash
git clone --branch cascade-v2 https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[satellite]'          # == .[hippius,chain]: boto3, hippius-hub, bittensor 11.3.0
export HIPPIUS_S3_ACCESS_KEY=...       # READ access on the manifest bucket
export HIPPIUS_S3_SECRET_KEY=...       # (manifests/, receipts/, benchmarks/, bench_v/, marketcast/, config/network/)
```

**Run:**

```bash
cascade-satellite --chain-toml chain.toml --network finney \
  --wallet-name my-validator --wallet-hotkey default
# same thing: cascade-validator --satellite … (its --cache-dir is the state dir)
# one pass, prints the status JSON: add --once
```

`chain.toml` must carry `[netconfig] owner_hotkey` and `[manifest]
trainer_hotkey`. The satellite refuses to start without them. Register and
stake as for a full validator. State lives in `--state-dir` (default: the
directory of `[validator] state_db_path`): `satellite_state.json`, the
activation records and `netconfig_state.json`. Do not delete them: the
anchor is what continuity is checked against.

**What it checks, per settlement, oldest first.** It walks the main
validator's `receipts/<main>/round-<id>.json` back from `latest.json` along
the trainer's signed `prev_round_id` links (the receipt index is the fallback)
to the last receipt it replayed. Then, for each receipt:

| check | what |
|---|---|
| `main-signature` | the receipt is signed by `[netconfig] owner_hotkey` |
| `manifest-signature` / `manifest-fetch` / `manifest-digest` | the trainer signed the manifest; `manifests/round-<id>.json` is published, its sha256 is the receipt's `manifest_sha256`, and it equals the embedded manifest |
| `continuity` | the trainer's hash link to the previous settlement, and the receipt's `state_prior_sha256` equals the previous receipt's `state_post` (king, tenure, court, ladder, lane-B/C slots and pending). Missing receipts in the chain also fail here. |
| `lane_c_chain` | lane-C pending entries carry over exactly (`check_lane_c_chain`) |
| every `cascade-audit` tier-0 row | verdict, cohort correction, KOTH params, bench gate (replayed from the signed bench report: the trainer's, or the main validator's `bench_v/` one under DEC-CA-0063), king check / ladder / emission, lane-B/C payouts, weights. These run under the receipt's DEC-CA-0045 blocks and its owner-signed overlay. |
| `weights-derive` | the satellite's own rebuild of the vector equals the recorded one |
| `reward-uid` (each push) | every paid UID is registered on the satellite's own metagraph to a hotkey the receipt pays |

**What it sets.** All checks pass ⇒ the vector it derived, through the normal
weight path (float32 → u16, `WEIGHT_RETRY_BLOCKS` retry). It re-asserts every
`weight_set_interval_blocks`. A receipt without a vector (a gate rejection)
keeps the standing vector. Any failed check ⇒ it burns to `burn_uid` and stays
active. The failing check is logged at every push until a later receipt
replays cleanly. A failed receipt still becomes the continuity anchor when its
main signature verifies, so recovery needs only one clean receipt. A check
that cannot run is retried, never treated as a verdict. Examples: storage
down, the bench report or overlay unreadable, no sr25519 verifier.

**Main validator silent.** It re-asserts the last verified vector until
`[satellite] stale_after_blocks` (default 3600 = one era) past that
settlement's boundary. After that it burns until a new receipt verifies.

**Status.** `<state-dir>/status/satellite-<hotkey>.json` and one log line per
poll give the mode (`following` / `holding` / `burning`), the last verified
round and the last failing check.

**Readiness notes.** A satellite posts its DEC-CA-0045 / DEC-CA-0057 note like
a full validator. It runs the same activation code with its own persisted
records, so its stake counts toward lock-ins. The overlay it replays under is
the one each receipt stamps (fetched by digest, signature checked against the
owner key).

**What it cannot detect.** Fabricated score rows. The satellite replays the
recorded per-window scores; it never re-scores a checkpoint. Reproduction
from the public checkpoints covers that (`cascade-audit round <id> --tier 2`,
experimental: GPU and the `[train]` extra), not satellites. A main validator
that never publishes
holds satellites on their last vector for `stale_after_blocks`, then burns
them.
