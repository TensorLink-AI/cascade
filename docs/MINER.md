# Miner guide

Choose a lane, test locally, then submit and fund only when you are ready.
You need not own a GPU for live mining: the operator rents training hardware
on your compute account. Local training experiments need suitable compute too.

| Lane | You submit | You compete for | Instructions |
|---|---|---|---|
| A | A deterministic Python generator of synthetic time series | Beat the current king with a fully trained model | [Steps 1–7](#1-install) |
| B | A generator and up to the allowed number of targets | Improve selected experts, then pass merge acceptance when enabled | [Lane B](#13-cascade-v2-and-domain-experts-lane-b) |
| C | An eligible MarketCast fine-tuning recipe, optionally with a generator | Beat the pinned application base on future data, then confirm and take a slot | [Lane C](#what-you-submit) |

The submission contract (what the code must be) is [INTERFACE.md](INTERFACE.md).
This guide is the end-to-end walkthrough.
Coding agents can follow [AGENT_MINING.md](AGENT_MINING.md) for lane starters,
bounded verification, live-contract comparisons, B/C local tests and preflight.

**Find a task:** [install](#1-install) · [local checks](#3-verify-and-score-locally)
· [submit A](#5-submit) · [provider keys](#compute-provider-keys)
· [track an entry](#watching-it) · [rewards](#rewards-allocation-burn-and-progression)
· [failures](#8-failure-classes) · [troubleshooting](#10-troubleshooting).

**Terms:** the *king* is A's current winner; a *leg* is one training run;
an *era* fixes the starting checkpoint and training rules; a *settlement*
(also called a round) judges completed work. A *warm start* is the checkpoint
used to start training, not necessarily the current king's trained model.

```
write a generator → cascade verify → cascade iterate (optional)
  → register a hotkey → cascade submit (private) or cascade deploy + cascade fund (public)
  → track cascade status --hotkey ... → read cascade duel
```

Funding and reveal make your entry eligible for admission; they do not reserve
a GPU or guarantee immediate training. Capacity, price caps, shared admission,
lane gates and the era's remaining time determine when it starts. Eligible work
carries forward automatically; a particular settlement is not guaranteed.
Use the live clock for settlement timing. Mainnet is netuid 91. The
intake is `https://submissions.cascadesub.net`.

Optional [group packing](INTERFACE.md#pack-independent-groups-in-one-yield)
lets your generator place independent groups inside one 32-channel yield.
It requires a contract accepting `groups`; bare arrays remain supported.

## Quickstart

Install from a checkout first (`pip install -e '.[hippius,chain]'`); see
[installation](#1-install). The examples use mainnet. Paid examples below
require your explicit decision to spend.

Read the live competition state before choosing a lane:

```bash
cascade status --json
```

This command needs no wallet. It fetches current public rules, budgets,
activation/gates, queue snapshots and the current era's warm-start init.
Check freshness and `ready_for_training`; a missing queue is not an empty
queue, and a future promotion is not today's init. Re-run before training
and at era transitions. Use `--chain-toml chain.testnet.toml` consistently
for testnet. The dashboard's `llms.txt` provides the agent workflow.
An older deployed trainer may not publish these new fields yet: treat an
unconfirmed init as unknown, and keep using signed evidence for admission.

After submitting, track that exact entry:

```bash
cascade status --hotkey YOUR_SS58 --lane a --ref IMMUTABLE_REF \
  --intake https://submissions.cascadesub.net --watch
cascade status --hotkey YOUR_SS58 --lane b --targets energy --ref IMMUTABLE_REF \
  --intake https://submissions.cascadesub.net --watch
cascade status --hotkey YOUR_SS58 --lane c --target mkt:returns --ref IMMUTABLE_REF \
  --intake https://submissions.cascadesub.net --watch
```

Run the command for your lane, preserving the exact submitted ref and targets.
Successful submit/fund responses print this command for you. Agents use
`--json` instead of `--watch`, polling every 30–60 seconds.

The view separates compute admission, CPU preparation, GPU capacity, worker
startup, training dispatch, checkpoint upload and settlement waits. It shows
time since funding/stage change, the last 16 stage events, requeue count,
consumed retry attempts, backoff, target era/block and the next action.
A phase change is not a worker heartbeat and no GPU ETA is invented.

You do **not** submit again for each round. A valid queued/requeued entry remains
eligible automatically while funding, commitment and lane rules stay valid.
Infrastructure retries and recovery of work lost on the operator's side keep
their existing rules (see [training and waiting](#training-and-waiting)).
If GPU capacity stays unavailable, keep the entry and ask the operator to check
the GPU type/price cap and backlog; a successful fund request is not a rental.
`not_found` does not prove rejection: pending reveals are hidden; check
`cascade reveal-status` and the funding receipt. `done` is queue completion,
not a win, accepted merge or confirmed reward; read the signed verdicts next.
Missing/stale feeds are reported explicitly, never as an empty queue or a
healthy worker. These details require matching CLI, intake and trainer releases.

| Lane | After training |
|---|---|
| A | Track settlement, then read `cascade duel --hotkey YOUR_SS58 --history --json` for the judged result. |
| B | `evaluation.stage` distinguishes qualification, waiting for warm-start acceptance, acceptance and rejection. Qualification alone is unpaid when acceptance is enabled. |
| C | `evaluation.stage` distinguishes forward judging, pending confirmation, deferral, confirmation and rejection. Check the reported receipt boundary; a historical slot is not proof of current payment. |

This is stage tracking, not a live percentage or per-step loss display.
`training` means the worker was dispatched, not that a fresh GPU heartbeat
was observed. Use the public training logs for recorded step metrics.

```bash
pip install -e '.[hippius,chain,generation]'
cascade verify ./my-generator
btcli subnets register --netuid 91 --network finney --wallet-name w --wallet-hotkey h

export LIUM_API_KEY=sk-...                     # env only; pays for your own training leg
cascade submit ./my-generator https://submissions.cascadesub.net \
    --wallet-name w --wallet-hotkey h --label my-gen-v1 --yes
cascade queue --intake https://submissions.cascadesub.net --hotkey <your ss58>
```

Rolling A/B/C funding accepts **Lium, Shadeform, Vast.ai and Runpod API keys**, with ordered capacity fallback.
Runpod is skipped when a price ceiling applies because its on-demand adapter
cannot verify that ceiling before creating a pod. See
[compute provider keys](#compute-provider-keys) before choosing an account.

Before you fund:

- **You pay for your leg** under the published GPU/cost policy and any
  explicit choice/hourly cap. Review the effective budget before funding.
  Earlier trainer-sourced benchmarks can extend pod time; validator-sourced
  v2 A benches use the main validator's GPU instead
  ([§6](#6-what-happens-to-a-funded-entry), [§5](#your-provider-keys)).
- **A allows one judged submission per hotkey.** B/C use one training leg per
  revealed commitment. All lanes share the hotkey's one chain commitment;
  switching it can supersede queued work. Infrastructure failures follow the
  retry rules ([§8](#8-failure-classes)).
- **Earliest reveal gets first pick** of the GPUs that fit; entries that
  cannot start yet wait with nothing spent
  ([§6](#6-what-happens-to-a-funded-entry)).
- **Private by default**: `cascade submit` keeps your code in the operator's
  vault and publishes it only if it takes the throne ([§5](#5-submit)).
- **Every module is a `.py` file** ([§2](#2-write-a-generator)), and
  multichannel `(C, L)` yields are allowed and billed per channel
  ([§11](#11-multichannel-generators)).

## 1. Install

For this v2 rollout, use a fresh commitment revealed at or after **30 October
2026, 09:00 UTC**. The updated paid Lane A intake refuses earlier entries.
Accepted entries remain queued until the first v2 activation era at or after
the **12:00 UTC** target; neither CPU generation nor GPU training starts under
v1. Submissions remain open afterward. A queued entry is not a consumed
training attempt. The one-submission-per-hotkey rule still applies when it is
judged or rejected for a miner-caused fault.

The earlier `[round] commit_floor_block` remains for v1 replay. V2 has its own
`[v2] commit_from_block`, resolved from the opening's finalized timestamp.
Inspect `golive.commit_from_block` and `golive.admission_state` in fresh
`cascade status --json`. `v2_submissions_not_open` means wait for the opening;
`v2_commit_before_open` needs a fresh reveal; `launch_rules_unavailable` is a
retryable chain-read failure (CLI exit 3).

### CPU generation libraries

The `generation` extra includes NumPy, SciPy, pandas, PyArrow, torch, networkx,
numba, scikit-learn, statsmodels and gpytorch. Bootstrap installs the frozen
`uv.lock` and runs `scripts/check_generation_runtime.py`, which imports every
library and prints its installed version before miner code runs. A missing
library fails provisioning rather than rejecting the generator. Generator
requirements still need allowlisted, hash-locked dependencies. Deployment of
the qualified CPU image is an operator release step.

Use a patched installer: uv >=0.13.4, or pip >=26.2.2 with a patched OS CA
store. This checkout requires certifi >=2026.10.19. After installing, run
`python scripts/check_tls_trust.py` with the same interpreter as the CLI.
See [TLS setup and worker rollout](OPERATOR.md#tls-trust-and-the-october-2026-emudhra-incident)
if the check fails; never bypass certificate verification.

```bash
git clone --branch cascade-v2 https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[hippius,chain]'
```

Use this repo's pinned environment to commit on-chain (`bittensor==11.3.0`)
and commit with `cascade submit` / `cascade deploy`, which seal the pointer in
the encoding validators decode. **Never hand-roll the reveal with
`bittensor.timelock.encrypt`** (or another SDK line's helper): it seals raw
UTF-8 without the SCALE length prefix the subnet's decoder strips, so your
commit lands but you are silently skipped every round. Reveals written from
the previous 10.5 environment still decode.

## 2. Write a generator

Start from a shipped example and edit:

```bash
cp -r scripts/example_generator my-generator      # minimal trend + seasonal + AR(1)
# a richer reference: base_generator/
```

Your repo must contain:

```
generator.py        # class Generator(DataGenerator)
config.json         # any JSON your generator reads
requirements.txt    # hash-locked deps from the allowlist (numpy, scipy, torch, …)
```

Rules that matter:

- **Deterministic.** `generate()` must be a pure function of the `seed` given
  to `__init__`. Seed every RNG from it; never use `hash()`, wall-clock, or the
  network. Two runs at one seed must produce byte-identical corpora.
- **Allowlisted imports only.** `socket`, `subprocess`, `pickle`,
  `multiprocessing` and friends are blocked (`chain.toml [static_guard]`) in
  EVERY `.py` in your tree and in any source you pack into a string constant
  (plain, base64, zlib, hex — packed modules are unpacked and scanned the
  same way). Compiled or native modules (`.so`, `.pyd`, `.pyc`, …) are
  rejected outright: generators are source-only.
- **Every module is a file.** Ship each sub-generator as a `.py` in your tree
  and import it; never keep code in a string constant and `exec` it. A
  string constant that decodes to Python *code* (definitions or imports, in
  any encoding the scanner knows) can be refused at admission
  (`[static_guard] packed_sources = "reject"`), and that rejection spends
  your hotkey. Data that merely parses (a JSON blob, a literal) is fine; a
  flush-left docstring example that imports is not. Examples in
  [§11](#code-in-strings-is-rejected).
- **Series shape.** V2 requires finite float arrays `(C, L)` with **C = 32**
  on every yield. One-dimensional and 1–31-channel yields are rejected.
  Historical v1 accepted `(L,)` or `(C, L)`. The earlier contract uses
  `64 ≤ L ≤ 4096` and `C ≤ 32`; v2 shard generation can publish different
  length limits. Read the live policy and [interface](INTERFACE.md).
- **Speed matters.** Streaming generation shares the training wall; configured
  shard generation has its own CPU window before GPU training. A slow generator
  can exhaust either budget. Pin BLAS to one thread, prefer shorter
  series for dense covariance GP draws (factorization scales cubically), and use a
  relative-jitter Cholesky instead of SVD fallbacks. Details in
  [INTERFACE.md](INTERFACE.md#compute-heavy-priors-gp--kernel-families).
- **Numerics.** Series with extreme magnitudes can make training diverge. A
  leg whose loss goes non-finite is rejected as your fault and your submission
  is spent. Keep values in a sane float32 range.

A v2 `(C, L)` yield is one series with **32 channels, optionally packed into independent groups**. The rules are
in [INTERFACE.md](INTERFACE.md#multivariate-series); a worked example with
billing numbers is [§11](#11-multichannel-generators).

## 3. Verify and score locally

`cascade verify` checks layout, allowed imports, hash-locked dependencies and
deterministic output against the selected configuration:

```bash
cascade verify ./my-generator --live
# → OK: full local verification passed under the selected generator policy.
#     corpus_digest (seed=0): …  [deterministic]
```

`--live` requires fresh discovery and selects the deployed CPU-tier memory
and length bounds with the training contract. Offline checks without
`--at-block` apply both the pre-v2 and v2 rules (exactly 32 channels, the stricter
length and memory limits), so a pass holds either side of the launch; use
`--at-block BLOCK` for a particular configured era. V2 shards permit
8,064-point series within the published limit (verify those with `--live`);
pre-v2 retains its earlier limits. Runtime verification uses bounded sandbox children.
Unavailable network isolation is a failed check, not a successful fallback.

A pass does not check chain registration, reveal, funding, duplicates against
other submissions, full-scale generation throughput or training stability.
Use fresh discovery and `cascade preflight` as well; the live trainer and
validator still decide admission and the result.

The packed-source check follows the `[static_guard]` of the `chain.toml` it
loads. To check against the strict rule, point it at the testnet file, which
sets `packed_sources = "reject"`:
`cascade verify ./my-generator --chain-toml chain.testnet.toml`.

Use `cascade verify ./my-generator --quick --json` after edits for two small
deterministic corpora with a 30-second timeout. A quick pass is not full admission
verification; retain the full check before submission.

`cascade score` trains the fixed model on your data at a cheap budget and
scores it on a pool you control (the default budget is `[round]
heat_train_hours`, 1 h; override with `--train-hours`). Needs the `[train]`
extra and ideally a GPU. Compare against the king on the same pool with the
same init:

```bash
cascade iterate ./my-gen --pool-dir ./my-heldout --warm-start live --seeds 0,1
```

`cascade iterate` verifies your generator, scores it over the seeds, fetches
and scores the king the same way (cached in `.cascade-work/`, so the king
trains once per settings change) and prints the relative improvement. Every
run is logged to `.cascade-work/history.jsonl` with the best so far. Use
`--baseline DIR` to compare against your previous version instead, or
`cascade score` for a single number.

**Start small.** Cascade MoE Base - 17.58M, 69.02M and 273.84M let you test
generators before using the 1.23B competition candidate. Names use total parameters. Generate a
complete offline configuration from the checkout:

```bash
python scripts/pyramid_config.py --size 17.58m --out ./chain.local-17m.toml
cascade iterate ./my-gen --chain-toml ./chain.local-17m.toml --pool-dir ./heldout --seeds 0,1 --train-hours 0.01 --work-dir ./.cascade-17m --json
```

Run the Python command with your installed environment (for example,
`.venv/bin/python`). Other total-size choices are `69.02m`, `273.84m` and `1.23b`.
Earlier aliases `2m`, `7m`, `28m` and `120m` remain supported.
The [presets](../deploy/presets/README.md) give exact counts and the
[model specification](MODEL.md) explains the recipe. GPU memory and speed are
unmeasured for this port. The 56M variant remains unspecified.

Compare candidate and baseline on the same model, budget, seeds, pool and
compatible init. Keep separate histories per model. Omit `--warm-start live`
for small-model experiments; it selects the live contract and checkpoint.
Recheck promising ideas on the live model before submitting. A small-model gain
is not a paid win, and a larger model's checkpoint cannot initialize a smaller one.

The number is directional. Validators score on a private pool you never see,
so use the local score to hill-climb, not as the verdict, and rotate your pool
so you do not overfit it. `--warm-start live` reads the current era init
and its effective training contract from fresh trainer status. Deployments
without that current contract cannot support live comparisons with this CLI.
`iterate` pins both for both generators and every seed; a changed checkpoint
or contract gets a separate score cache/history.
Without it you train from random init, which ranks generators differently.

## 4. Register a hotkey

```bash
btcli wallet new-coldkey --wallet-name my-miner
btcli wallet new-hotkey  --wallet-name my-miner --wallet-hotkey gen1
btcli subnets register --netuid 91 --network finney --wallet-name my-miner --wallet-hotkey gen1
```

Testnet: `--netuid 259 --network test`. For Lane A, a hotkey is spent once its
submission is judged in a duel or fails through its own fault; another A
submission needs a fresh hotkey. B/C have separate per-commitment rules and
do not consume the lifetime A submission.

## 5. Submit

Two paths. Both end with a funded entry in the queue.

### Private (recommended): `cascade submit`

Your code goes straight to the operator's private vault. It is never
published unless it takes the throne.

```bash
export LIUM_API_KEY=sk-...            # env only, never an argument
cascade submit ./my-generator https://submissions.cascadesub.net \
    --wallet-name my-miner --wallet-hotkey gen1 --label my-gen-v3 --yes
# → verifies, ZIPs, stores privately, commits vault/direct@sha256:… on-chain,
#   funds the leg; it queues when the reveal lands
```

`--yes` confirms the spend: with a key set, submit refuses without it (exit
5) and sends nothing. Flags: `--no-fund` (fund later), `--no-commit` (print the chain payload only),
`--reveal-now` / `--next-epoch` / `--blocks-until-reveal N` (reveal timing).
A submit-with-key entry waits as `pending_reveal` and queues itself when the
reveal lands.

### Public: `cascade deploy` + `cascade fund`

Your code is pushed to a public Hippius Hub repo and committed on-chain, then
you fund it.

```bash
export HIPPIUS_HUB_USERNAME=... HIPPIUS_HUB_PASSWORD=...   # or HIPPIUS_HUB_TOKEN
cascade deploy ./my-generator --hub-namespace my-namespace \
    --wallet-name my-miner --wallet-hotkey gen1
export LIUM_API_KEY=sk-...
cascade fund https://submissions.cascadesub.net --ref <repo@digest> \
    --wallet-name my-miner --wallet-hotkey gen1 --label my-gen-v3 --yes
```

The Hippius project must be **public** or the trainer cannot pull it
(`generator_artifact_unreachable`). `--hub-namespace` gives each deploy a
random repo name so nobody can watch your namespace. If the Hub is down, add
`--hf-repo <ns/name>` (needs `HF_TOKEN`) to mirror to HuggingFace.

### Reveal timing

A funded entry cannot start until its commitment is **revealed**: the trainer
checks the ref against your hotkey's revealed commitment at the block the leg
starts. `submit` and `deploy` default to a timelock reveal
`[round] reveal_margin_blocks` (25 blocks, ~5 min) before the next epoch
boundary, so your code stays hidden until then and nobody can copy it ahead
of you. Confirm with `cascade reveal-status <hotkey> --watch`, and watch the
next boundary with `cascade round`. A ref that does not match your latest
reveal fails the entry as `ref_mismatch`; fund the new ref.

### Labels

`--label` attaches a display name (≤ 32 chars, `[A-Za-z0-9._-]`) that shows
beside your hotkey in `cascade queue` and the published roster. Re-funding
with a new label renames; without one keeps the old. Labels are cosmetic:
never identity, never in signed records.

### Compute provider keys

Provider support depends on the protocol, not just which adapters are in the
repository:

| Workflow | Accepted compute credentials | Setup |
|---|---|---|
| Rolling Lane A | Lium, Shadeform, Vast.ai and/or Runpod API keys | Human sets `LIUM_API_KEY`; use `cascade submit` or `cascade fund`. |
| Rolling Lane B | Lium, Shadeform, Vast.ai and/or Runpod API keys | Human sets `LIUM_API_KEY`; use `cascade lane-b fund`. |
| Rolling Lane C | Lium, Shadeform, Vast.ai and/or Runpod API keys | Human sets `LIUM_API_KEY`; use `cascade lane-c fund`. |
| Separate direct runtime | Lium, Vast.ai or RunPod API key for an offered provider | `cascade direct deposit-key`, then separate direct job funding. Experimental; live activation is refused in this release. |

The default remains `LIUM_API_KEY` (or `--lium-key-env NAME`). To use
Shadeform or Vast.ai, or allow several accounts, repeat
`--provider-key-env PROVIDER=ENV_NAME` (`lium`, `shadeform`, `vast` or `runpod`) in your
preferred order. A Vast.ai key needs only the instance read/write scopes. This explicit list replaces the default; only listed
accounts may be billed. One key per provider is supported. The human sets the
variables through their environment or secret manager; flags contain names only.

```bash
# Lium first, then Shadeform when no fitting capacity is available.
cascade fund https://submissions.cascadesub.net --ref IMMUTABLE_REF \
  --provider-key-env lium=LIUM_API_KEY --provider-key-env shadeform=SHADEFORM_API_KEY \
  --wallet-name W --wallet-hotkey H --yes

# Replace the stored list without queuing another submission or renting a pod.
cascade credentials https://submissions.cascadesub.net --ref IMMUTABLE_REF \
  --provider-key-env shadeform=SHADEFORM_API_KEY --wallet-name W --wallet-hotkey H --yes
```

The same provider flags work with `cascade submit`, `cascade lane-b fund` and
`cascade lane-c fund`. Existing GPU choices and applicable price caps still
apply. Provider fallback does not imply permission to change GPU type; that is
controlled separately by `--allow-fallback`. It is ordered capacity fallback,
not parallel rentals or round-robin spending across accounts.

Keys that cannot be queried must be fixed before another account is used: the
trainer cannot assume an inaccessible account has no live allocation. Once a
create request starts, a timeout does not trigger a second provider. The next
attempt looks for the existing pod first. Each rental retains an opaque
reference to its original credential for identity checks and teardown; replacing
the list affects future allocations. Retained credentials expire through the
vault TTL and are used to recover/stop old pods, not authorize fresh rentals.
Upgrade both intake and trainer before advertising this feature; fresh
`cascade status --json` publishes `compute.rolling_providers` when supported.

For Shadeform, the human obtains a dedicated API key with access to list instance
types, inspect/list instances, create and delete their instances. See the
[official API](https://docs.shadeform.ai/api-reference/instances/instances-create).
Shadeform uses the pinned Docker worker image and the operator's injected SSH
public key; miners do not upload an SSH private key. Keep sufficient credit in
each selected account. For Runpod use `--provider-key-env runpod=RUNPOD_API_KEY`.
An hourly or estimated-leg cap prevents a Runpod create. Its uncapped path
requires an explicitly selected GPU with `--max-price 0`; use that only when
the human accepts the rate and has arranged appropriate account controls.
This does not establish a total-dollar cap.

### Provider budgets and Cascade price ceilings

An API key does not itself prove that a budget is configured. Cascade does
not create keys, raise budgets or verify their remaining allowance. Its
`--max-price` is an hourly price ceiling for a selected GPU; standard A/B/C
funding has no cumulative total-dollar budget across providers or retries.

| Provider | Documented spending control | Cascade behavior |
|---|---|---|
| Lium | [Daily, monthly and lifetime key budgets](https://docs.lium.io/pod-users/api-keys#budgets) | The human configures the key; provider enforcement applies to its rentals. Reaching the budget can delete running pods. |
| Shadeform | [Per-instance `auto_delete.spend_threshold`](https://docs.shadeform.ai/api-reference/instances/instances-create) | Applicable `funded_max_leg_cost_usd` is sent as that instance's threshold, rounded down to cents. It is not a key-wide or retry-wide cap. |
| Vast.ai | [Scoped API permissions](https://docs.vast.ai/guides/reference/api-keys) | Offer price ceilings are enforced. A hard per-key total budget has not been confirmed in the public API documentation. |
| Runpod | [Restricted API permissions](https://docs.runpod.io/get-started/credentials) | A hard per-key total budget has not been confirmed. Price-capped creates are refused; uncapped GPU selection requires explicit authorization. |

Provider controls have different scopes. Alerts, an account balance and scoped
permissions are not interchangeable with a hard spending limit. Separate keys
and providers do not share one budget. Check fresh `cascade status --json`
for `compute.spending_controls`; `key_budget_verified=false` means exactly that.
`cascade preflight --max-spend` records a proposed budget for review only.

For a human-authorized isolated direct-runtime test, create a dedicated API
key in the selected provider account:

- **Vast.ai:** use a scoped key with `instance_read`, `instance_write` and
  `misc` for the adapter's list/create/destroy/search calls. The
  [official permissions guide](https://docs.vast.ai/api-reference/permissions)
  explains scoped-key creation. Billing/user administration scopes are not
  used by this adapter.
- **RunPod:** open Credentials → API Keys → Create API Key, choose Restricted
  permissions, and allow the Pod REST read/write operations needed to create,
  inspect and delete pods. Read-only access cannot rent or tear down a pod.
  See the [official credentials guide](https://docs.runpod.io/get-started/credentials).
  This adapter does not use Serverless or AI inference endpoints.

The human loads the key into `VAST_API_KEY` or `RUNPOD_API_KEY` through their
environment or secret manager. Never paste it into a CLI flag, generator,
chat or repository. These examples assume an explicitly prepared test intake,
a registered hotkey and its matching revealed ref; they store credentials
but do not queue a job or rent compute:

```bash
cascade direct deposit-key https://YOUR_TEST_INTAKE --provider vast \
  --key-env VAST_API_KEY --ref IMMUTABLE_REF --wallet-name W --wallet-hotkey H --json
cascade direct deposit-key https://YOUR_TEST_INTAKE --provider runpod \
  --key-env RUNPOD_API_KEY --ref IMMUTABLE_REF --wallet-name W --wallet-hotkey H --json
```

Run only the command for the selected provider. `--key-env` is optional;
the default is `<PROVIDER>_API_KEY`. Repeating a deposit rotates that provider's
stored key. It does not renew a rolling A/B/C entry or select a GPU.
Direct job funding separately requires published owner configuration, an open
cohort, qualified hardware profiles, provider credentials, `--spend-cap`,
`--max-spend` and `--yes`; see `cascade direct fund --help` and the
[direct operations guide](V2_DIRECT_MOE_OPERATIONS.md). A deposit being accepted
does not establish that the direct protocol is available for live mining.

These are marketplace API keys, not SSH keys or Bittensor wallet keys. The
trainer configures its worker SSH access; miners do not send a private SSH
key with the funding command. Never revoke a rental key while its pod still
needs teardown: provider deletion uses that key too.

### Your provider keys

- Sent once, signed by your hotkey over the key hash, a timestamp and your
  ref. Always use `https://`.
- Stored in the local payer vault with a 36 h idle TTL. Withdraw forgets the
  current list; a rental's retained credential remains available for recovery
  and teardown until its lease expires. Keep the provider key valid until all
  pods using it are confirmed stopped.
- Default selection uses the cheapest fitting allowed GPU under the configured
  hourly and estimated leg-cost caps. Read fresh `cascade status --json` policy
  and the fund response for the current GPU list, wall and caps; these are not
  fixed prices. Explicit GPU choices use the miner's price cap as described below.
  Earlier trainer-sourced benchmarks can extend the rental; mainnet v2's
  validator-sourced A benchmarks use the main validator's GPU (see §6).

### Choosing your GPU (DEC-CA-0060)

Every leg trains the same token budget inside the same wall. You may still
pick which GPU type your key rents, paying more per hour to finish sooner.
Picking a type never buys more compute.

```bash
cascade fund https://submissions.cascadesub.net --ref <repo@digest> \
    --wallet-name my-miner --wallet-hotkey gen1 --gpu H100 --max-price 2.50 --yes
cascade lane-c fund https://submissions.cascadesub.net --target mkt:returns \
    --ref <repo@digest> --wallet-name my-miner --wallet-hotkey gen1 --gpu L40S --yes
```

- `--gpu` must be one of `[round] funded_pod_skus`. The intake refuses any
  other type (`unknown_gpu`). It also refuses every choice on a network that
  locks one type (`funded_sku_per_leg` off).
- The operator's price caps (`funded_max_price_per_hour`,
  `funded_max_leg_cost_usd`) do **not** apply to a type you chose. Your own
  `--max-price` (USD per GPU-hour) does. Without it you accept the listed
  price.
- If the type is unavailable, your leg **waits** for it and requeues like any
  capacity miss: no attempt burned, nothing rented, nothing charged. That is
  the default, `allow_fallback = false`. Add `--allow-fallback` to rent the
  default type (the cheapest that fits, under the operator caps) as soon as
  your chosen type has no capacity.
- If the type has no measured full-budget wall (`funded_sku_wall_seconds`;
  for example A6000 and RTX3090 today), or its wall reaches the contract
  wall, the fund reply carries a `warning` and the CLI prints it. The fund
  still goes through. A leg that stops at the wall is judged on what it
  trained.
- The latest fund wins. Re-funding without `--gpu` resets the entry to the
  default type. The choice applies to the next rent, never to a pod that is
  already running.
- Your choice shows as `gpu` in `/v1/queue`, `/v1/lane-c/queue`,
  `cascade queue` and the published roster (empty means the default type).
- Each leg uses one GPU. Multi-GPU legs are not supported.

## 6. What happens to a funded entry

This sequence describes Lane A. B and C keep their separate qualification,
acceptance or forward-confirmation stages ([agent walkthrough](AGENT_MINING.md)).

1. **Admission.** The trainer considers eligible revealed entries in seniority
   order under admission limits, capacity, GPU choices/caps and the era's
   remaining time. A later entry that fits available capacity can start while
   an earlier incompatible one waits. Pending entries remain unspent; duplicate
   or invalid source can be refused before training.
2. **Data preparation.** The v2 shard contract runs the CPU generation tier
   before renting the GPU. The generation window and queue are distinct from
   GPU training; read their effective limits instead of adding a fixed wait to
   every lane. Earlier contracts can stream data during training.
3. **Training and harvest.** The worker uses the pinned era contract, init and
   seeds on the payer's pod, under the applicable point budget and wall. The
   orchestrator checks and uploads the checkpoint. A stage labelled
   `in_flight` alone does not prove GPU training has begun.
4. **Settlement and judging.** A compatible completed leg enters a signed
   settlement manifest. Validators evaluate the private-pool comparison and
   apply the active gates. With validator-sourced benches, the main validator
   sweeps a pool clearer and king on its GPU, signs `bench_v/round-<id>.json`,
   and a later manifest binds the report's exact bytes. Infra/budget deferrals
   are distinct from checkpoint faults and scientific rejection.
5. **Read the result.** Queue `done` is not a win or proof of payment. Read the
   signed verdict and reward record with `cascade duel`; a target boundary is
   an estimate rather than a guaranteed settlement. Monitor the exact entry
   rather than resubmitting just because a round has passed.

For older trainer-sourced benchmark contracts, a kept payer pod can run public
sweeps at completion and operator verification supplies the signed numbers.
That path can extend rental time. Under the v2 mainnet validator-source policy,
the payer pod does not run Lane A sweeps. Public benchmark gates can affect the
verdict; they are not universally log-only telemetry.

The ~3 h grid is 900 blocks. `chain.toml` ships `[round] epoch_blocks = 3600`;
the on-chain rollover to rolling intake (DEC-CA-0045) switches the grid to
`[activation] epoch_blocks_after = 900` from the rollover block.

**Eras.** The effective configuration defines era length (mainnet's rolling
grid uses 4 settlements, approximately 12 h). Each era pins its contract and
init. Under v2 `king_ladder`, several eras can share a seed epoch and cached
king; the king normally funds its own leg. Every compatible challenger in an
era duels that cached king. A dethrone
makes your checkpoint the era king without ending the era. The king is read
from the validators' signed receipts, not the metagraph.

### Legs that cannot land in their era

A leg is judged in the era it was admitted to. A later era can use different
seeds or a different init; even when a v2 seed epoch reuses them, a completed
leg remains bound to its admitted era. The trainer fits a generator
it has seen before on its **measured** wall (last leg's duration × 1.1), so a
slow generator is admitted into the next era instead of a slot it cannot
make. A leg that still lands late is requeued the moment it finishes, with no
attempt consumed. Unless the operator has configured recovery capacity, the
retrain rents a new pod on your provider account; after three such operator-side
losses the entry pauses for the operator instead of billing you again. A generator
it has never seen is fitted on the GPU type's
estimate — keep your generator fast enough for the SKU, or expect the first
leg to be the measurement.

### Watching it

```bash
cascade status --hotkey <you> --lane a --ref IMMUTABLE_REF \
  --intake https://submissions.cascadesub.net --watch  # current stages, retries and next action
cascade queue --intake https://submissions.cascadesub.net --hotkey <you>   # live queue + last settlement's roster
cascade round                 # next-boundary countdown, latest verdict, dethrone bar, revealed submissions
cascade duel                  # the settled verdict: margin, geomeans, per-domain scores
cascade duel --hotkey <you>   # + YOUR domains: where you beat the king and by how much
cascade duel --hotkey <you> --history   # your gap vs the king per domain, per settlement
```

`cascade duel --hotkey <you>` prints a `your domains` block: for every pool
domain (nature, energy, sales, …) the king's score, yours, the gap, and how
many windows it rests on. Negative gap = you were better there. A domain
with fewer than 30 windows is flagged — that gap is draw noise, not a
strength. `--history` turns it into one line per settlement you were judged
in, with a per-domain average at the bottom: the "am I getting better at web"
view. The website's cohort panel shows the same per-domain gaps for every
judged challenger.

Every leg also streams a public JSONL training log (`logs/round-<id>/…`) with
per-step loss, throughput and `data_wait_frac`, plus a `host` record for the
pod. High `data_wait_frac` means training waited on your generator; a low
`host_bench_tokens_per_s` means the pod was slow.

## 7. How the verdict works

Both metrics below are forecast errors: lower is better. CRPS evaluates the
predicted distribution; MASE scales absolute error by a seasonal-naive error
scale. An LCB is a lower confidence bound: the conservative end of the
estimated improvement used for the pass/fail test.

- **Metric.** Geometric mean of CRPS and MASE over the private windows, on a
  horizon ladder of 64 / 256 / 720 steps.
- **Dethrone rule.** The paired-bootstrap lower confidence bound of your
  advantage over the king must clear the dethrone margin. The margin is
  priced against the improvement over the shared init, is highest against a
  fresh king and decays with the king's tenure (counted in blocks) to a floor
  that never reaches zero. With several challengers in one settlement, a
  cohort-wide correction keeps the king's false-dethrone risk fixed.
  `cascade round` prints the live bar; each receipt records the rule it was
  judged under.
- **Per-domain breakdown.** The verdict is the POOLED statistic over every
  window; the receipt also records the same geomean per pool domain for the
  king and for each challenger (`per_domain` / `cohort_per_domain`), so you
  can see which domains you beat the king in and by how much. Display only:
  domain gaps are diagnostic, not separate A pass/fail gates. Gains and losses
  contribute to the pooled statistic; winning one domain does not by itself
  qualify a challenger.
- **Public bench gate (v2, DEC-CA-0061 rule v2).** A private-pool win is
  then checked on GIFT-Eval, BOOM, TIME and FEV, one benchmark task at a
  time, against the king. It can only block a win. You pass when all three
  hold: (1) on average you are better (the geomean of your CRPS and MASE
  ratios to the king's, each suite weighing the same, is below 1); (2) no
  suite is significantly worse (its paired-bootstrap ratio bound at alpha/4
  stays at or below 1 + `[scoring] bench_gate_tolerance`, default 1 %);
  (3) you win at least `bench_gate_min_win_rate` (default half) of the tasks,
  suite-weighted, ties counting half. Being a hair worse on one number no
  longer blocks you. The receipt records the three numbers.
- **Rewards.** V2 pays the Lane A ladder and eligible B/C holders, with the
  unpaid balance burned. The older geometric court split is historical;
  use the [reward schedule below](#rewards-allocation-burn-and-progression).
- **Warm-start lineage.** Under v2 `king_ladder`, a passing king check can
  promote the king into a scheduled shared init, optionally with accepted B
  deltas. The historical top-k mixture of king and challenger checkpoints
  does not govern this regime. `cascade score --warm-start live` resolves the
  current effective init; an announced future promotion is not today's base.

### Rewards: allocation, burn and progression

These are the **checked-in mainnet v2 rules**, applied when their launch gates
are in force. Mainnet's v2/B/C gates are unset in the template; these numbers
are not an announcement of live activation. Signed configuration and settlement
records determine the actual rules. Testnet retains older economics and must
not be used to infer mainnet payments. Percentages below refer to the validator's
competition reward vector, not a fixed TAO/day or dollar payment.

**First allocate pools; then calculate eligible payments inside each pool.**

<!-- cascade:rewards:allocation:start -->
| Launched lanes | A pool | B pool | C pool |
|---|---:|---:|---:|
| A | 100% | 0% | 0% |
| A + C | 75% | 0% | 25% |
| A + B | 75% | 25% | 0% |
| A + B + C | 50% | 25% | 25% |
<!-- cascade:rewards:allocation:end -->

C takes its 25% from A in the mainnet policy; B stays at 25%. Before B or C
launches, it has no reserved allocation. Once a lane launches, an empty pool
or empty target slot burns rather than being reassigned to another miner.
Mainnet uses `remainder = "burn"`; do not import the optional older
`remainder = "lane_a"` policy into these calculations.

**Launch and inherited king.** Before v2, the older court rule applies unless
the separately adopted prelaunch burn is active: that policy burns 95% of the
whole vector and scales the existing miner split into the remaining 5%.
Setting the template's burn percentage alone does not activate it. At the
first settlement judging a v2 era, the inherited king/court is capped at
**5% of total emissions**, until a king is crowned by a v2 verdict. The rest
of A burns; independently earned B/C pay is unaffected by this A cap.

**Lane A after a v2 crowning.** A random-init/level-unit crowning starts the
king at **50% of A's pool**. An increment-unit crowning instead uses its
recorded confidence bound:

<!-- cascade:rewards:king-start:start -->
| Crowning LCB in increment units | Starting share of A |
|---|---:|
| ≥ 0.5 | 80% |
| ≥ 0.25 | 65% |
| ≥ 0.1 | 50% |
| ≥ 0.01 | 25% |
<!-- cascade:rewards:king-start:end -->

These LCBs use the duel's **increment units**, not a claim of 1–50% absolute
forecast-error improvement. The highest threshold met determines the start.
A passed scheduled king check adds **10 percentage points of A**, up to **95%**;
a failed check sets the ladder to **10%**. Deferral or no check leaves it
unchanged. A 50% start therefore progresses 50 → 60 → 70 → 80 → 90 → 95
on successive passes. It does not rise automatically every round or era.
The mainnet template requires 14 random-init eras; warm-start/king-check timing
also follows the reign clock and era-notice rules, rather than an automatic
reward increase when the random-init window ends.

Only the **immediate former king** can receive another **5% of A**, for
**36,000 blocks** (about five days at 12-second blocks) from dethroning, while
eligible. A later dethrone replaces this recipient. Older former kings do not
retain a v2 court payout; an absent/expired former king's share burns.
Thus, with all lanes launched, a king at 50% of A receives **25% of total**,
and an eligible former king receives **2.5% of total**. A's remaining 22.5%
of total burns. A 95%-ladder king receives 47.5% of total in that same rollout
stage; A reaches its full 50% payout only while an eligible former king also
receives its 2.5%.

A qualifying failed-key lapse substitutes **1% of A** for the king's current
ladder payout, keeping the stored ladder position. The first lapse has a
minimum six-era window (about three days on the rolling mainnet grid) and
requires recorded key recovery; a second lapse in the same reign lasts until
the reign ends. Operator-caused faults do not count as a key lapse. The
former king's eligible pay and separately earned B/C pay are not slashed.
See [key recovery](#as-the-v2-king-keep-your-key-funded-dec-ca-0055).

**Lane B: accepted expert updates.** With acceptance enabled, training,
individual qualification and waiting for a warm start earn nothing by
themselves. Committed warm-start acceptance starts the target's clock.
There are currently **10 targets** (seven domains plus three horizons), so
each reserves **25% / 10 = 2.5% of total emissions** at a full multiplier.
A miner can hold multiple target slots; each has its own clock. A rejected
update neither merges nor earns; a deferred update waits unpaid. Existing
holders keep their remaining eligible pay while replacements are pending.

**Lane C: confirmed application slots.** An initial forward-evaluation pass
is unpaid. A later confirmation must win the target slot; passing below the
holder's required bound does not earn that slot. C splits its pool by each
competition's weight over **all configured competition weights**, then equally
over that competition's target slots. With only MarketCast and its current two
targets, each full-pay slot reserves **25% / 2 = 12.5% of total emissions**.
That is a slot maximum, not a promise of paying an unfilled slot. Pending,
unstarted or ended competitions do not donate their allocation to active ones.

**B/C reward progression.** B records its tier from accepted marginal gain;
MarketCast C records it from the confirmed gain. Current full-pay tiers are:

<!-- cascade:rewards:slot-tiers:start -->
| Lane | Target gain LCB | Full-pay eras |
|---|---:|---:|
| B: accepted marginal gain | ≥ 0.05 | 10 |
| B: accepted marginal gain | ≥ 0.02 | 5 |
| B: accepted marginal gain | ≥ 0.01 | 2 |
| C: confirmed MarketCast gain | ≥ 0.05 | 10 |
| C: confirmed MarketCast gain | ≥ 0.02 | 5 |
| C: confirmed MarketCast gain | ≥ 0.01 | 2 |
<!-- cascade:rewards:slot-tiers:end -->

A threshold of 0.05 here is a 5% lower confidence bound on the target gain;
it is a different statistic from A's increment-unit crowning LCB. Full pay
means **100% of that target's slot**, not 100% of the lane or subnet. The tier
is fixed on the recorded win; a later tier retune does not restart that win.

After the full-pay period, the configured one-era half-life gives multipliers
50% → 25% → 12.5% → a **10% slot floor**. The win stops paying at **age 28
eras**, measured from B acceptance or C confirmation, including its full-pay
eras. It does not get another 28 eras after decay starts. Replacement by a new
holder stops the old holder earlier. At approximately 12 hours per mainnet
era, 28 eras is approximately 14 days; blocks, not wall-clock estimates, govern.

<!-- cascade:rewards:slot-decay:start -->
B example: accepted marginal gain LCB 0.02, recorded full-pay period 5 eras.

| Age since acceptance (eras) | Slot multiplier | Share of total emissions |
|---|---:|---:|
| 0–4 | 100% | 2.5% |
| 5 | 50% | 1.25% |
| 6 | 25% | 0.625% |
| 7 | 12.5% | 0.3125% |
| 8–27 | 10% | 0.25% |
| 28+ | 0% | 0% |
<!-- cascade:rewards:slot-decay:end -->

C uses its own decay/expiry configuration, currently the same half-life,
floor and TTL. When a competition ends, `finish_full_pay` permits an existing
holder only its remaining recorded full-pay period, capped at six wind-down
eras; there is no decay tail after the end. Under `stop`, a settlement after
`until_block` pays nothing. That ended competition's unpaid allocation burns.
Admission closes at `until_block`: the intake returns `409 competition_ended`
and `cascade lane-c commit` refuses new entries. The status page and
`cascade round` expose each competition's weight, share, lifecycle state and
start/end blocks; use these alongside the reward records.

**Worked allocation example, not a measured payout.** All three lanes have
launched; A's king has a 60%-of-A ladder share, its immediate former king is
eligible, one B target and one of the two MarketCast targets have full-pay
holders, and all other B/C slots are empty:

<!-- cascade:rewards:worked-example:start -->
| Recipient in the worked example | Share of total emissions |
|---|---:|
| A king | 30% |
| Eligible immediate former king | 2.5% |
| One full-pay B holder | 2.5% |
| One full-pay C holder (mkt:returns) | 12.5% |
| Burn | 52.5% |
<!-- cascade:rewards:worked-example:end -->

The total is 100%. Burns combine A's unused ladder share, B's unused slots,
C's unused slots and later decay/expiry. They are assigned to `burn_uid`
(template: UID 0), not normalized away or shared between the remaining miners.
A missing/deregistered holder's slot also burns. An explicit all-burn override
can suppress the whole vector. Funding pays for compute; an unsuccessful
contribution has no promised emission or reimbursement of consumed GPU time.

**Follow rewards separately from compute progress.** `cascade status --hotkey
HOTKEY --lane a --ref REF --intake INTAKE --json` tracks the exact entry; add
B/C targets as appropriate. Its `evaluation.reward_status=not_verified` does
not prove a payout. Read signed `emission`, `lane_b_pay`, `lane_c_pay` and
`weights` records and on-chain weights to establish the paid share. Queue
`done`, a qualification or a pending confirmation alone cannot establish it.


## 8. Failure classes

What happens to your entry when a leg does not produce a judged checkpoint:

| class | examples | your submission |
|---|---|---|
| infrastructure | dead pod, sold-out market, rate limit, harvest transport failure | **kept**, re-queued; one bounded attempt burned (sold-out and rate limits burn nothing) |
| stall | your generator produced no series for 30 min | first time: treated as infrastructure; second time: yours |
| auth | invalid/revoked provider key, insufficient account funds, or an expired vault entry | released as `auth`; fix the key, fund again |
| generator | your code raised, a series failed the checks, the loss went non-finite, code packed in a string refused at admission | **spent** |
| duplicate | an exact, rename-only, embedded or private copy of an earlier commit (§9) | **spent** |
| tamper | pod replaced under the same name, checkpoint altered | **spent** |
| ref_mismatch / burned / funding_expired | you re-revealed a different ref / hotkey already used / entry outlived the 36 h key TTL | terminal; re-fund (a burned hotkey needs a fresh one) |

`cascade queue` shows the class beside a failed entry. A rate-limit streak
longer than 6 h turns terminal; raise the key's limits and fund again.

## 9. Study the competition

The king's code is public the moment it takes the throne (`champions/` on
the manifest bucket; private losers stay private forever):

```bash
cascade fetch king --out ./king          # the reigning king
cascade fetch <uid|hotkey|repo@digest>   # a public competitor
```

Copying is unrewarding by construction: the earliest commit owns its code,
and an exact or rename-only copy is dropped before it trains (a dropped copy
still spends its hotkey). You win by improving on the visible best.

What is *not* allowed is code that was never public. A private (vault)
submission stays private unless it is crowned and published. An entry whose
module — a `.py` file or a string-packed source — contains most of an
earlier, unpublished submission's module from another hotkey (identifiers
masked, so renaming changes nothing) is a `private_copy`: the earlier
commit keeps the entry and the copy is dropped like any duplicate
(`chain.toml [round] private_copy_mode`; it ships in shadow, logging
verdicts, and the owner announces before it enforces). Building on a
published king or any public Hub generator is never a private copy.

Packing does not hide anything either. A generator carried as a string
constant (plain, base64, zlib or hex) is fingerprinted like a submission of
its own: if it equals another entry's whole code — or another entry's packed
source — exactly or after renaming, that is an `embedded_token_identical` /
`embedded_rename_identical` duplicate, in either direction, and the earlier
commit keeps it (`chain.toml [round] dedup_embedded_mode`; shadow first).
The registry also holds the current king and every published champion, so a
re-wrap of the king is judged against the king.

Rotated-out eval windows are published with a lag to
[Tensor-Link/cascade-eval-pool](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool).
Use them to replay past verdicts locally; live settlements always score on
windows that were never published.

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| `verify` fails determinism | an unseeded RNG, `hash()`, wall-clock, set iteration order |
| `blocked_import` | banned import (the message names the file, or `packed[n]` for a string-packed module); see `chain.toml [static_guard]` |
| `packed_source` | Python code inside a string constant; move it into a `.py` file and import it ([§11](#code-in-strings-is-rejected)) |
| `binary_modules_forbidden` | a `.so`/`.pyd`/`.pyc`/… in the tree — ship source only |
| `duplicate` (`embedded_*`) | a source packed into your code equals another entry's whole code or packed source (exactly, or renamed) — or yours equals someone's blob; the earlier commit keeps it |
| `duplicate` (`private_copy`) | your entry carries most of an earlier unpublished submission's module from another hotkey — only public code is yours to build on |
| `requirement_not_hash_locked` | every `requirements.txt` line needs `--hash=sha256:…`, allowlisted packages only |
| `403 not_registered` | register the hotkey first |
| `403 not_revealed` | the ref is not a revealed commitment for this hotkey; wait for the reveal, then fund |
| `400 bad_label` | label too long or has characters outside `[A-Za-z0-9._-]` |
| `generator_artifact_unreachable` | your Hippius project is private; make it public |
| funded but never started | not yet revealed, waiting for the next era's pre-train window, or no fitting executor under the price caps; you wait unspent |
| `failed [generator]` | your code failed on the pod or training diverged; check the training log, fix, submit from a fresh hotkey |
| `failed [rate_limited]` | your provider key was rate-limited for 6 h; raise limits, fund again |
| loses every duel | the pool is broad real-world data; broaden the generator's patterns rather than fitting one shape. `cascade duel --hotkey <you>` shows which domains you lost and by how much; `--history` shows whether that is changing |

## 11. Multichannel generators

A worked eight-channel submission: what width costs, the layout, what the
packed-source scan refuses, and a checklist. The rules themselves are in
[INTERFACE.md](INTERFACE.md#multivariate-series). A leg trains, settles and
is audited under the billing rule of the era it started in.

### What a channel costs

The checked-in mainnet contracts use `budget_denomination = "series_points"`:
a `(C, L)` series costs `L` budget points. More channels still cost more GPU
time and memory, and the contract's wall can stop a slow run early.
The [interface reference](INTERFACE.md#which-limits-apply) distinguishes the
templates and the CPU shard tier's length limit.

For comparison, the optional `points+mv20` policy bills a `(C, L)` series as:

```
C == 1:   L points
C >  1:   ceil(C × L × 100 / 120) points
```

Under that optional rule the bonus is 20% whatever `C`, and every channel is
billed. The contract's wall still applies. These are worked examples of
`points+mv20`, not the checked-in mainnet default, at `L = 1024`:

| yield shape | values (tokens) | billed points | tokens per point |
|---|---|---|---|
| `(32, 1024)` | 32 768 | 27 307 | 1.20 |

For this optional rule, the budget is still measured in points. Earlier
throughput-based contracts derive it from training hours and reference speed;
the pyramid recipe uses its explicit update budget. The current v2 width is
32 channels. The optional bonus concerns budget accounting;
it does not relax the requirement for exactly 32 channels.

Two things the rule does not do:

- It does not make a duplicated or junk second channel free. Every channel is
  billed under this rule, and extra channels use GPU time under either rule.
  A duplicate channel supplies no new signal.
- It does not by itself teach the model cross-channel structure. Only coupled
  channels do that. Stack channels that genuinely depend on each other, or do
  not stack.

### Layout: one generator, many modules, every module a file

You can compose your corpus from as many sub-generators as you like. The only
rule is that each one is a `.py` file in the tree that `generator.py` imports.
A minimal eight-channel submission, built from four coupled pairs:

```
my-mv-generator/
├── generator.py        # class Generator(DataGenerator): combines four driver/responder pairs
├── channel_a.py        # sub-generator: the driver channel
├── channel_b.py        # sub-generator: a channel that depends on A with a lag
├── config.json         # anything your generator reads (lengths, mix, lags)
└── requirements.txt    # hash-locked, allowlisted deps (numpy, …)
```

`channel_a.py`, a driver series (trend + seasonality + AR(1) noise):

```python
from __future__ import annotations

import numpy as np


def driver(rng: np.random.Generator, length: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    x = rng.normal(0.0, 1.0) + rng.normal(0.0, 0.01) * t
    period = float(rng.choice([7, 12, 24, 30]))
    x += rng.uniform(0.2, 2.0) * np.sin(2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi))
    phi, sigma = rng.uniform(0.0, 0.8), rng.uniform(0.1, 0.5)
    noise = np.empty(length)
    noise[0] = rng.normal(0.0, sigma)
    for i in range(1, length):
        noise[i] = phi * noise[i - 1] + rng.normal(0.0, sigma)
    return x + noise
```

`channel_b.py`, a channel that responds to A after a lag, plus its own noise.
This is the point of stacking: B is not predictable from its own past alone,
so a model that attends across channels gains something:

```python
from __future__ import annotations

import numpy as np


def responder(rng: np.random.Generator, a: np.ndarray, lag: int) -> np.ndarray:
    gain = rng.uniform(0.5, 1.5)
    b = np.empty_like(a)
    b[:lag] = rng.normal(0.0, 0.3, size=lag)
    b[lag:] = gain * a[:-lag] + rng.normal(0.0, 0.3, size=a.size - lag)
    return b
```

`generator.py`, the class the trainer loads. It yields `(32, L)` arrays:

```python
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

from channel_a import driver
from channel_b import responder


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self._seed = int(seed)
        self._min_len = int(cfg.get("min_length", 256))
        self._max_len = int(cfg.get("max_length", 1024))
        self._max_lag = int(cfg.get("max_lag", 8))

    @property
    def name(self) -> str:
        return "example-32-channel-lagged"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)          # the ONLY source of randomness
        for _ in range(n_series):
            length = int(rng.integers(self._min_len, self._max_len + 1))
            channels = []
            for _ in range(16):
                a = driver(rng, length)
                b = responder(rng, a, lag=int(rng.integers(1, self._max_lag + 1)))
                channels.extend([a, b])
            yield np.stack(channels).astype(np.float64)  # shape (32, L)
```

Notes on the shape contract:

- A v2 yield is one series `(C, L)` with **C = 32**. Values must be finite float.
  Historical `(L,)` and narrower multichannel arrays are not valid v2 submissions.
- Every channel of a series has the same length `L`, within the effective
  length limits. Ordinary verification uses 64–4096 in the templates; the
  v2 CPU shard tier allows up to 32768.
- Eval windows carry at most 32 channels whatever your `C`, and a multichannel
  eval window scores once, its channels averaged.
- Mixing legal widths is fine: a corpus can be 70% `(8, L)` and 30% `(16, L)`.
  Every yield must independently have exactly 32 channels. Billing is per yield.

### Code in strings is rejected

The scanner unpacks every string constant in every `.py` (plain text, base64,
zlib, raw deflate, hex, nested) and parses it. If it parses as Python **code**,
meaning definitions or imports, the submission can be refused at admission
with `packed_source[<depth>]` and the file name. All of these fail:

```python
# 1. a module kept as a string and exec'd
_SRC = "def make(rng, n):\n    return rng.normal(size=n)\n"
exec(_SRC)

# 2. the same module base64-encoded
_BLOB = "ZGVmIG1ha2Uocm5nLCBuKToKICAgIHJldHVybiBybmcubm9ybWFsKHNpemU9bik="
exec(base64.b64decode(_BLOB))

# 3. a flush-left docstring that reads as code
"""
import numpy as np
def helper(x): return x
"""
```

These are fine, because they are data, not code:

```python
_PARAMS = '{"periods": [7, 12, 24], "phi_max": 0.8}'     # JSON: parses as a literal
_NAMES = "trend seasonal ar1"                             # plain text
```

The reason: the duplicate screen and the public archive fingerprint the files
in your tree. Code that only exists after a decode step is invisible to both.
`cascade verify --chain-toml chain.testnet.toml` reports it as
`[static_guard] packed_source` (§3).

### Checklist before you fund

- [ ] `cascade verify` says OK and `[deterministic]`.
- [ ] Every sub-generator is a `.py` file; no `exec`, no code in strings.
- [ ] Every v2 yield is finite float `(C, L)`, **C = 32**, with `L` inside the effective limits.
- [ ] Channels of a series are genuinely coupled; a copied channel costs budget and hurts.
- [ ] One hotkey, one lifetime submission: a rejection at admission spends it.

## 12. Scripting and coding agents

Every command takes `--json` (or `CASCADE_JSON=1`): stdout carries one JSON
document (`ok`, `exit_code`, `data`, `error.code`) and progress goes to
stderr. Exit codes: 0 success, 1 rejected, 2 bad usage, 3 network, 4
registry, 5 a paid action needs `--yes`. Paid actions (`cascade fund`,
`cascade submit` with a key set, `cascade direct fund`) refuse without
`--yes`, so a script or agent cannot spend by accident. If an agent mines for
you, point it at [AGENTS.md](../AGENTS.md).

## 13. cascade v2 and domain experts (lane B)

V2 starts a fresh Cascade MoE lineage under `[v2.training]`. A's generators
train the full model. B's generators train selected experts inside that model.
B opens separately from A: an activation block alone does not bypass its
measured-lineage and routing checks. Use fresh `cascade status --json`.

The planned rollout allocates A 100%, then A/C 75/25, then A/B/C 50/25/25.
The [reward schedule](#rewards-allocation-burn-and-progression) explains the
launch cap, king ladder, acceptance clocks and burns.

### Choose targets, not expert IDs

Declare 1–3 targets under the checked-in mainnet limit
(`[lane_b] max_domains_per_leg`). Choose from:

- Domains: `econ_fin`, `energy`, `healthcare`, `nature`, `sales`, `transport`,
  `web_cloudops`.
- Horizons: `horizon:64`, `horizon:256`, `horizon:720`; the short forms are
  `h64`, `h256`, `h720`. Testnet's empty scored-horizon list supplies no
  horizon targets.

The trainer measures which experts the king routes each target to and chooses
eligible rows. Multiple targets train the union of their selections in one
leg, with the same budget as a single-target leg. Miners do not supply expert
IDs. Domain selection uses context patches; horizon selection uses masked
forecast patches and context, never the future target values.

Rows already held by open legs, finished unsettled legs or accepted updates
waiting to enter an init are unavailable. A layer can select fewer than the
configured minimum when too few routed rows are free; no free routed row means
waiting. Mainnet also requires **200B checkpoint-bound measured training
points** and stable routing. An estimated or nominal token budget cannot open
that gate. The point unit follows the effective budget accounting.

### Test, commit and fund

Use [the agent walkthrough](AGENT_MINING.md#lane-b-improve-selected-expert-rows)
for a local training and acceptance test. After full verification and the
human's authorization:

```bash
cascade deploy ./my-gen --expert-domain energy,web_cloudops,h720 \
  --wallet-name W --wallet-hotkey H --hub-namespace NS
cascade reveal-status HOTKEY --json
cascade lane-b fund https://YOUR_INTAKE --targets energy,web_cloudops,h720 \
  --ref IMMUTABLE_REF --wallet-name W --wallet-hotkey H --yes
```

The human supplies `LIUM_API_KEY` through the environment. Targets must match
the revealed commitment exactly. The CLI validates targets and their count
before upload or commitment; do not construct the chain payload by hand.
`--gpu`, `--allow-fallback` and `--max-price` work as in
[Lane A funding](#choosing-your-gpu-dec-ca-0060). The configured B budget is
`[lane_b] leg_hours`; it is not an extra budget for each declared target.

A hotkey holds one chain commitment. Committing B replaces its A/C commitment
and can supersede queued work. Each revealed B commitment funds one leg;
re-funding a spent commitment returns `spent`. To make a new attempt, commit
again, wait for the new reveal and fund that commitment. This does not consume
A's lifetime submission. Only one leg per hotkey can be live across A/B/C.

### Training and waiting

Only selected rows have gradients, optimizer buffers and training EMAs. The
full model still runs, including frozen experts, so a small trained subset does
not imply an equally small GPU or runtime requirement. Live metrics report
`expert_training_backend = "compact_rows"` and
`expert_trainable_param_count`. Peak memory and speed need measurement on the
actual checkpoint and hardware.

Use the fund response's `data.tracking.argv`, or:

```bash
cascade status --hotkey HOTKEY --lane b --targets energy,web_cloudops,h720 \
  --ref IMMUTABLE_REF --intake https://YOUR_INTAKE --watch
cascade lane-b status https://YOUR_INTAKE --hotkey HOTKEY --json
```

The first command joins execution with qualification/acceptance observations;
the second is the intake queue only. To withdraw while queued, use
`cascade lane-b fund` with the same identity and `--withdraw`.

No capacity requeues without using an attempt or renting a pod. Other
infrastructure retries have bounded attempts. Bad source, tampering or a bad
key ends the entry under its reported error class; inspect that class before
proposing another attempt. A `stale` base or `routing_stale` result caused by
the king changing requeues as an **operator-side loss**. Work lost on our side
(a pod that never became usable, a restart, a king change) is not burned. The
trainer fails a bad pod early (readiness, image hash, throughput and GPU checks
before training) and retries with a new pod on your provider account unless the
operator configured recovery capacity; each such replaced pod counts as one
operator-side loss. After three operator-side losses the entry
pauses with `operator` status until the operator fixes the cause; it is never
burned for them, and you are billed for at most three such allocations in total. Nothing is refunded; an unconfirmed pod release blocks another
rental until it is resolved, so a leg is never billed twice at once.

### Qualification, merge acceptance and payment

1. **Qualify.** Each declared target needs a positive point gain and an LCB
   at least `gain_margin`, with enough independent sources
   (`min_target_clusters`). The confidence level is corrected by the number
   of declared targets. A horizon target uses windows scored at that horizon.
2. **Check regressions.** Other domains/horizons and required public benchmarks
   must satisfy their non-inferiority limits (`LCB ≥ −tau`). A guard failure
   blocks every target in that leg; a target missing only its own gain test
   is unpaid. Required public rows and compatible data revisions must exist.
3. **Wait for a scheduled warm start.** With acceptance enabled, qualification
   is unpaid and leaves current holders in place. Qualified deltas wait for
   the next compatible shared starting checkpoint.
4. **Accept compatible updates.** Candidates are tested in commitment-block,
   hotkey, checkpoint order. Each must improve against the checkpoint accepted
   so far while all domain, horizon and public guards remain within the
   original init's regression limits. Validators verify intermediate weights
   and private verdicts; public rows use signed trainer evidence.
5. **Merge and pay.** Only committed accepted targets enter the scheduled
   init and start their reward clocks. Their pay tiers use accepted marginal
   gain. The delta merges once, even if it wins several target slots.

Rejected combinations neither merge nor earn; there is no extra slash, but
training costs already used remain spent. Deferral is unpaid. Later compatible
candidates can still pass. A later candidate for a target already accepted
in the same batch waits for another warm start. Reward expiry does not undo a
merge. Mainnet enables this acceptance rule with v2; testnet keeps its older
rule until separately activated. See the [reward tables](#rewards-allocation-burn-and-progression).

### As a new v2 king: your starting share (DEC-CA-0055)

The [reward schedule](#rewards-allocation-burn-and-progression) lists the
increment-unit starting tiers, the level-unit start and the king-check ladder.
The receipt's `emission.ladder` records `crown_lcb`, `crown_units`, `crown_tier`
and `crown_start_bp`; use those fields rather than a point-estimate gain or an
older court split to explain a king's starting reward.

### As the v2 king: keep your key funded (DEC-CA-0055)

Under `[v2] king_ladder` the king's own leg uses the king's selected provider keys in
the payer vault. If that key fails (absent or expired in the vault,
rejected, out of funds, revoked) the operator trains the leg and the
manifest records `funded_by = "operator"`: you are paid `[v2] key_slash_bp`
(1 %) of lane A instead of your ladder share (the ladder position itself is
kept). The slash escalates within a reign:

- **First lapse:** 1 % for at least `[v2] key_slash_min_eras` eras (6 = 3
  days) from that settlement, even if you fix the key sooner; after that,
  until recovery is recorded under a working rotated key (a seed-epoch carry
  can record recovery without retraining the cached king).
- **Second lapse in the same reign:** 1 % for the rest of the reign,
  whatever you do with the key.
- A failure on the operator's side (`operator_fault`) never counts. A new
  reign starts at zero lapses.

To avoid it: keep a funded key in the vault for your whole reign (the
trainer re-stamps the vault's 36 h TTL while you are king, but that does not
help a revoked or drained key), with enough balance for a full king leg. To recover from a lapse, ROTATE to a new key
with `/v1/credentials`: the trainer only lifts the slash for a key other
than the one that failed (it remembers the failed key across restarts) that
the configured providers accept. This checks authentication, not account
balance — so an unfunded rotated key closes the lapse but fails at your next
own-key leg, and that counts as a second lapse. `cascade round` shows the
slash reason (`operator_paid` / `min_window` / `repeat_lapse`) and the
earliest block it can lift.

## 14. MarketCast (lane C, `mkt:<suite>` targets) — DEC-CA-0064

**Lane C currently accepts MarketCast recipes only.** Its tests use market
windows that start after training is dispatched and are scored once they
close. Training data has a separate embargo cutoff.
Today the only such competition is MarketCast, so the lane-C targets are
`mkt:returns` and `mkt:vol` (the `[marketcast] suites`). The old DEC-CA-0056
targets (`domain:…`, `freq:…`, `horizon:…`, judged on the public-feed
`pool_c` snapshot) are gone. Their windows could be scraped and memorised
before they were revealed. `cascade lane-c`, the intake and the chain
commitment parser refuse them with "lane-C competitions need a forward-only
eval — only MarketCast (mkt:) has one; see DEC-CA-0056". While MarketCast is
off, no lane-C target is active. If Lane C has launched, its reserved share
burns; before Lane C launches, no C pool is reserved.

**Checked-in templates: off on mainnet and testnet** (`[marketcast] from_block =
0`). Check live status separately. While it is off, the intake, `cascade lane-c check|commit|fund` and the
trainer refuse every `mkt:` target with a clear message
about inactive MarketCast. Their JSON codes differ: `check`/`upload` return
`bad_recipe`, `commit` returns `bad_commit`, and `fund` returns `bad_target`;
the intake returns `marketcast_inactive`. Branch on the code for the command
you called. This section describes live entry after activation.

MarketCast is a fine-tune eval. You fine-tune a PINNED third-party base,
`mistralai/Timestral-28M@180698bbf5a50dda3516aa6419798437fb03a430` (its
`model.safetensors` sha256 is
`dfc83c6220937ce5fc6687b35a4df94cd25e28a2cfe6fb92a23dad6301b2015d`), for one market suite. The
suites are `mkt:returns`, `mkt:vol`, `mkt:range`, `mkt:funding`,
`mkt:liquidity` and `mkt:alpha`; the launch set is returns and vol. Each suite
is its own lane-C slot, with lane C's decay and TTL; its full-pay period is
tiered by `[marketcast] full_pay_tiers` (same placeholder table as lane C's,
kept separate because the MarketCast gain is a different statistic).

Licence: the pinned weights use a
[Modified MIT licence](https://huggingface.co/mistralai/Timestral-28M/blob/180698bbf5a50dda3516aa6419798437fb03a430/LICENSE).
Its restriction covers companies or employers with global consolidated revenue
over USD 20M in the preceding month and extends to derivatives. Check those
terms before using or distributing a fine-tune.

### What you submit

A tree, uploaded like a lane-C tree, with `marketcast.json` at its root and
an OPTIONAL generator:

```json
{"target": "mkt:returns",
 "data": {"assets": {"crypto": ["BTC", "ETH", "SOL"], "metals": ["GOLD"]},
          "class_weights": {"crypto": 2.0, "metals": 1.0},
          "frequencies": ["1m", "5m", "1h"],
          "start": "2025-06-01", "end": "2026-09-01"},
 "context_length": 4096,
 "horizon_mix": {"5m": 1.0, "1h": 1.0, "1d": 0.5},
 "covariates": ["volume", "calendar", "session"],
 "augmentations": [{"name": "jitter", "p": 0.5, "params": {"sigma": 0.01}},
                   {"name": "crop", "p": 0.3, "params": {"min_frac": 0.5}}],
 "objective": {"suite_weights": {"returns": 1.0, "vol": 0.2}, "coverage_penalty": 0.5},
 "optimizer": {"lr": 1e-4, "schedule": "cosine", "warmup_fraction": 0.05,
               "steps": 2000, "batch": 64, "weight_decay": 0.01, "clip": 1.0,
               "ema_decay": 0.999},
 "trainable": {"scope": "lora", "lora_rank": 16},
 "synthetic_fraction": 0.0}
```

- Every key is required (except `quantile_weights`, `coverage_penalty` and
  `ema_decay`). Unknown keys are refused at every level, and every bound
  comes from `[marketcast]`.
- `data.assets` must be a subset of `[marketcast.assets]`: the universe
  Hyperliquid lists. FX is only `EUR`, `JPY` and `GBP` there.
- `data.end` must be at least `embargo_days` (30) before your submission.
  In live training the trainer fetches the permitted public data itself;
  you do not upload a private training dataset. Local snapshot commands can
  fetch public data for research.
- `covariates` ⊆ `cross_asset`, `funding`, `volume`, `calendar`,
  `session`. Covariates are inputs and are never scored.
- `augmentations` ⊆ `scale`, `jitter`, `crop`, `time_warp`, `mixup`, each
  with a probability `p` and numeric `params`.
- `trainable.scope` is one of `full`, `last_n_blocks` (+ `n_blocks` 1–12),
  `frozen_embeddings`, or `lora` (+ `lora_rank` ≤ 64). LoRA is cascade's own
  adapter on the transformer's linear layers, merged into the weights at
  export.
- `synthetic_fraction` > 0 needs a generator in the tree (`generator.py`).
  It uses the same sandboxed `generate()` contract as lane A. Its series are
  mixed in at that share (≤ 0.5), and synthetic samples carry no
  covariates.

Local recipe/quality tests are covered in the
[agent walkthrough](AGENT_MINING.md#lane-c-test-a-marketcast-recipe).
Once MarketCast is open, check, upload, commit and fund with the human's
authorization:

```bash
cascade lane-c check ./my-mkt --target mkt:returns
cascade lane-c upload ./my-mkt --target mkt:returns --hub-repo NAMESPACE/NAME --json
cascade lane-c commit --target mkt:returns --ref <repo@digest> --repo-dir ./my-mkt \
    --wallet-name W --wallet-hotkey H
cascade reveal-status HOTKEY --json
export LIUM_API_KEY=...
cascade lane-c fund https://<intake> --target mkt:returns --ref <repo@digest> \
    --wallet-name W --wallet-hotkey H --gpu L40S --yes
```

Use the upload result's exact `data.ref` for commitment and funding. Uploading
publishes the recipe; it does not commit or rent a GPU. Before funding, run
`cascade preflight ./my-mkt --lane c --targets mkt:returns --hotkey HOTKEY
--ref IMMUTABLE_REF --max-price RATE --max-spend TOTAL --json` and review the
result. Preflight's total cap is a review input, not an enforced A/B/C spend cap.

The leg runs on YOUR key, with the same token budget and wall as any lane-C
leg (`[lane_c] leg_hours`), on the GPU type you chose (DEC-CA-0060). An
unavailable type waits and is never burned. The leg fine-tunes the pinned
base, never the king, so a king change never makes it stale. A bad recipe or
a diverged loss is your fault and ends the entry. Infra faults requeue.

Every lane-C entry (DEC-CA-0056 review 2026-10-19):

- **One leg per commitment.** Once a commitment's leg is published,
  re-funding the SAME commitment is refused (`spent`); re-commit (a new
  reveal) to train again. A leg lost on the operator's side still retrains,
  uncharged.
- **Exact copies lose.** Fine-tuned weights byte-identical to an entry
  committed EARLIER (by commitment block, never UID) are a `duplicate`, and a
  copy that took a slot first loses it when the original lands.

### How it is judged

- The **main validator** forecasts the untouched base and your fine-tune on
  its GPU. It uses EVERY forward-only window of the signed snapshot of
  realized bars (minute bars for sub-hourly series, an hourly bank for ≥ 1h
  ones): each window starts at least `start_margin_seconds` after the time
  the trainer dispatched your leg (`train_time`, signed by the trainer),
  and is scored only once it has closed. It publishes the per-window losses
  and the forecasts themselves (float32 quantile sidecars), signed, at
  `marketcast/round-<id>.json`. Every validator derives the verdict from
  those rows, once a later trainer-signed lane-C manifest records the
  report; `cascade-audit marketcast --reproduce` lets anyone recompute the
  losses and re-run the forecasts.
- If the main validator cannot compute your entry (an OOM, its own outage,
  a report that never appears), your entry is DEFERRED as `infra`, not
  judged: it is retried at the next round without using up your wait, up to
  `max_infra_deferrals` (8) times, then dropped with the `operator` error
  class and requeued as an operator-side loss by the trainer: without
  operator recovery capacity it retrains on your key with a new pod, and
  pauses after three such losses.
- Windows are spread over the whole period since your dispatch, only
  liquid, live windows count (≥ 80 % live bars and ≥ $1M traded in the
  24 h before), and the gain weighs every horizon and every asset class
  equally, clustered by asset class × 6-hour block: you need wins across at
  least `min_time_blocks` (8) blocks, not one lucky regime. A NaN in any of
  your fine-tune's forecasts is a fault, and so is an exception your model's
  forward raises identically on every attempt (a CUDA / out-of-memory /
  I/O error stays infra). Eligibility never looks at the realised outcome:
  a window is chosen from what is known when it starts (liquidity, and a
  price context that is not flat).
- **Calibration gate (`mkt:returns` and `mkt:vol`, red-team 2026-10-20).**
  At EVERY horizon the 50 / 80 % intervals must cover within −0.05 / +0.10
  of nominal, and the share of outcomes below your 5 % quantile and above
  your 95 % quantile must each be 5 % ± 2.5 %, per horizon and pooled over
  horizons. Every rate is averaged over asset CLASSES (3 FX pairs count as
  much as 28 crypto perps), and every band widens to 3 standard errors where
  the counts are thin. Narrowing your quantiles does not pay: the score is
  mean CRPS (see Verdict), and narrow tails fail this gate.
- The trainer checks your recipe's `data.end` against the embargo at
  dispatch (before anything is rented), and your pod never sees a bar after
  its dispatch time.
- A fine-tune whose weights are byte-identical (sha256) to an entry committed
  EARLIER (any target) is a `duplicate` and loses; seniority is the
  commitment block, never a UID. So does a NEAR copy: if your forecasts sit
  within `near_dup_tol` (mean |Δq| of 2 % of the base's 5–95 % width) of an
  earlier-committed entry's on the same windows (another hotkey on your
  suite this round, or the slot holder), you are a `duplicate`. Nudging a
  copied checkpoint's weights does not escape it. Slots compare fine-tunes against the
  pinned base, so a king change never displaces a MarketCast slot holder.
- **Verdict:** per asset class and horizon, the ratio of your MEAN CRPS to
  the base's mean CRPS over the class's windows; the gain is `1 −` the
  geomean of those ratios (classes equal inside a horizon, horizons equal),
  and its bootstrap `LCB ≥ 0.01` at `alpha / k`. Never a per-window log
  ratio, so a too-narrow forecast cannot beat an honest one. `k = C × m`:
  `C` = distinct OWNERS (coldkeys) with an entry on the suite in the round,
  `m` = your coldkey's own entries there. Extra hotkeys on one coldkey share
  that coldkey's level and never raise the bar for others. The trainer also
  runs at most one open MarketCast leg per coldkey. The verdict also needs a
  positive point gain, ≥ `min_windows` windows over ≥ `min_clusters`
  (asset class × 6 h block) clusters, and the calibration gate above.
- An initial pass is **pending confirmation**, with no payment. A later
  settlement must pass it again at `alpha` on windows starting after the first
  snapshot ended. Confirmation must also win the target slot; passing below
  the current holder's required bound does not replace it or start payment.
  Expired or deregistered holders no longer block a replacement. Each target
  has one final `confirmed` holder per settlement; an initial-test runner-up
  is labelled `not_selected` and earns no reward.
- Too few windows or assets so far ⇒ your entry WAITS, and is re-judged
  every settlement for up to `verdict_wait_blocks` (6 days). After that it
  is dropped.
