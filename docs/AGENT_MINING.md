# Mining with a coding agent

Start by reading the deployed dashboard's `llms.txt` and the local
[interface](INTERFACE.md). These instructions prepare an entry; live
qualification, merge acceptance and forward confirmation remain validator decisions.

Read [AGENTS.md](../AGENTS.md) for authorization and JSON-output rules.
Use only local/read-only work until the human authorizes a live action.

For this rollout, paid A entries open **30 October 2026, 09:00 UTC**. Use a
fresh reveal at or after that opening. Accepted entries wait in `waiting_launch`
until actual v2 activation, without a training attempt being consumed. Entries
stay open afterward. The planned lane order is **A → C → B**; discover each
lane's active state separately.

**Route:** install → choose A/B/C below → compare locally → preflight and human
review → submit/fund if authorized → track the exact entry → read signed results.

- [Lane A](#lane-a-improve-a-generator): full-model generator experiments.
- [Lane B](#lane-b-improve-selected-expert-rows): target-specific expert experiments.
- [Lane C](#lane-c-test-a-marketcast-recipe): future-data application tests.
- [Tracking](#track-one-submission-after-funding), [rewards](#rewards-and-burns)
  and [v2 settings](#v2-rollout-and-compatibility-reference).

## Install and check the deployment

From a checkout of the intended release:

```bash
uv sync --frozen --extra train --extra hippius --extra chain --extra marketcast --extra generation
uv run cascade --help
uv run cascade discovery-check --json
uv run cascade status --json
```

The examples below use `cascade` from that environment (`uv run cascade` works
too). The training extras are needed for local model experiments. Generation,
quick verification, public status and preflight do not rent a GPU. Local
training uses the selected machine and can still take time.

Use one network consistently: pass `--chain-toml chain.testnet.toml` and
`--network test` wherever those options are supported (`--help` lists them).
Funding and lane-queue commands without those flags use the selected intake;
choose the testnet intake rather than sending to the mainnet URL.
Check `data.network`, `fresh`, and each source's freshness. Missing queues are
unknown, never empty. `ready_for_training` confirms the Lane-A init only; it
does not grant B/C admission.

`discovery-check` tests the deployed feeds, current effective contract, active
queues, protocol capabilities and guide route. Exit 3 / `discovery_release_incomplete`
means an operator must complete the release. It does not publish anything.

V2 synthetic generators must emit finite float arrays `(C, L)` with **8–32 channels per yield**. One-dimensional and 1–7-channel yields are rejected. Read the effective policy for length and resource limits.

## Lane A: improve a generator

```bash
cascade init ./my-a --lane a --json
cascade verify ./my-a --quick --json
cascade iterate ./my-a --pool-dir ./heldout --seeds 0,1 \
  --warm-start live --train-hours 0.01 --device cpu --json
```

`--quick` runs static checks and generates two small corpora in a child with a
30-second timeout. Its digest covers only that small sample; it is **not** the
full admission corpus digest. Use it after edits. Run full `cascade verify
./my-a --json` before proposing a submission; full verification can take much longer.

`iterate` uses quick verification, then pins the live init **and the effective
era training contract** once for both generators and every seed. If either is
unknown, stale or incompatible with this CLI, it refuses instead of guessing.
Older deployments without the contract feed cannot run `--warm-start live`
through the new scoring flow. Explicit offline comparisons can use a local
checkpoint and `--at-block BLOCK` with the appropriate checked-out configuration.

For cheaper generator experiments, prefer a smaller model once released.
The planned 2M, 7M, 28M, 56M and ~120M variants are currently
[blocked placeholders](../deploy/presets/README.md). Use `score` or `iterate`
with a complete local `--chain-toml` configuration for a released variant;
omit `--warm-start live` for that experiment because it selects the live model.
Use random init or a checkpoint compatible with the small architecture.
Keep the model, budget, seeds, pool and init fixed across candidate/baseline
comparisons, with a separate work directory per model. Recheck promising ideas
on the live model before proposing a submission; local gains are not rewards.

Use `--device cuda` only on local hardware the human has authorized you to use.
The local budget above is 36 seconds of reference compute per training run,
subject to the configured wall guard; two seeds and two generators are four
runs. It is not the paid live budget. The default scoring budget is larger.

Put real held-out `.npy`/`.npz` series in `heldout/`, kept outside the generator
tree. Each must be long enough for the effective context plus forecast horizon.
`metadata.json` can map a filename stem (or `archive:key` for `.npz`) to:

```json
{"energy_01": {"domain": "energy", "source": "independent-feed-01"}}
```

Use accurate domain and independent-source labels. Do not label related windows
as separate sources to manufacture confidence. The default synthetic pool is
only a regression smoke test. Keep seeds, pool, contract, init and budgets fixed
while comparing ideas. Cache keys include the effective contract, eval/generator
settings, pool contents and local checkpoint contents; a promotion or contract
change cannot reuse the old baseline.

## Lane B: improve selected expert rows

```bash
cascade init ./my-b --lane b --target energy --json
cascade verify ./my-b --quick --json
cascade lane-b test ./my-b --targets energy --base live \
  --pool-dir ./heldout --seed 0 --train-hours 0.01 --device cpu --json
```

Repeat with fixed settings over several `--seed` values. `--base live` pins the
current **trained king**, effective contract and published lane-B rules. Expert
selection uses only the local affinity contexts for each declared target, and
trains their routed-row union through the same compact-row backend as live legs.
The held-out pool needs the domain/source metadata above. Horizon targets need
the appropriate evaluation horizon ladder in the configuration.

For offline development before B opens, pass `--base /path/to/moe-checkpoint`.
With no explicit `--at-block`, this uses the checkout's configured v2 contract;
it makes no claim that v2 or B is active. A live trained king may be very large:
training fewer rows does not eliminate the frozen model's memory cost.

`data.qualification` contains per-target confidence-bound and regression checks.
The test also exercises `judge_acceptance` against this one contribution's
original base. Add `--public-project /path/to/benchmark-project` to measure the
required public guards through the production benchmark sidecar. That project
must have its benchmark data/environment prepared. Missing required guards are
reported in `public_guards_missing`, and `accepted_targets` stays empty.

Local acceptance cannot predict other pending live merges or reproduce the
private pool. Qualification alone earns nothing when acceptance is enabled;
the live warm-start acceptance must pass before rewards begin.

After human authorization, the full B workflow is:

1. Full verify; upload and commit using `cascade deploy --expert-domain ...`.
2. Wait for `cascade reveal-status HOTKEY --json` to confirm the intended reveal.
3. Run preflight with the exact ref and targets; review its limits and evidence.
4. Human authorizes `cascade lane-b fund`; the agent must not add `--yes` on its own.
5. Poll the returned `data.tracking.argv`, or `cascade status --hotkey HOTKEY
   --lane b --targets EXACT_TARGETS --ref EXACT_REF --intake INTAKE --json`,
   every 30–60 seconds with backoff. This joins execution with scientific
   progress; `cascade lane-b status` alone is the intake queue view.

## Lane C: test a MarketCast recipe

```bash
cascade init ./my-c --lane c --target mkt:returns --json
cascade lane-c fetch-base --out ./timestral-base --json
cascade lane-c snapshot ./my-c --out ./mkt-train --json
cascade lane-c snapshot ./my-c --evaluation --days 7 --out ./mkt-heldout --json
cascade lane-c test ./my-c --target mkt:returns --base-dir ./timestral-base \
  --train-snapshot ./mkt-train --eval-snapshot ./mkt-heldout \
  --train-hours 0.01 --seed 0 --device cpu --json
```

The starter uses embargoed dates and a small recipe. The data commands download
public bars without credentials; they can take time and upstream history may be
unavailable. Use only supported assets, sources and ranges. Snapshots are
digest-checked local research data, not validator-signed evidence.

The test rejects overlapping snapshots, checks that the training snapshot lies
inside the recipe dates, verifies the base's weight digest before loading, then
uses the production sampler, fine-tuning function and MarketCast verdict. It
compares the fine-tune with the untouched pinned base on later windows, including
coverage/tail gates. Repeat over fixed seeds/settings. `confirmation=not_tested`
is intentional: a later live settlement must confirm the entry for a paid slot.

Local tests can run before activation. The `check`, upload, commitment and
funding commands still enforce the selected configuration's admission rules.
When activated, the human-authorized workflow is:

```text
lane-c check → lane-c upload → lane-c commit → reveal-status → preflight
→ human-authorized lane-c fund → exact-entry status → judge → confirm → paid slot
```

Use the returned `data.tracking.argv`, or `cascade status --hotkey HOTKEY
--lane c --target mkt:returns --ref EXACT_REF --intake INTAKE --json`, to join
queue progress with forward judging and confirmation. `cascade lane-c status`
alone reads the intake queue.

`cascade lane-c upload ./my-c --target mkt:returns --hub-repo NAMESPACE/NAME`
returns the immutable `data.ref` without committing or funding. Pass that exact
ref to `lane-c commit` (with `--repo-dir ./my-c`) and, once revealed, funding.
Uploading publishes the recipe to the registry; it is not part of the offline
agent loop and requires the human's authorization. Never upload keys or data.

## Preflight and human review

```bash
cascade preflight ./my-a --lane a --hotkey PUBLIC_SS58 \
  --max-price 1.00 --max-spend 5.00 --json
```

For B/C add `--targets` and the appropriate `--lane`. Before funding an existing
commitment, include `--ref EXACT_IMMUTABLE_REF` to check its revealed identity.
Preflight checks fresh rules, activation, target limits, B's routing gate,
checkpoint/contract or C's pinned base, a quick local validation, the hotkey's
registration/reveal, lifetime submission telemetry and cross-lane queue state.
It never opens a wallet or sends a write request. Unknown evidence blocks
`ready_to_propose`; a pass is permission to present the proposal, not spend.

Its price and total budget are **review inputs**, not a GPU quote or reservation.
Standard A/B/C funding supports an hourly `--max-price` with an explicit GPU;
it does not currently enforce the preflight's total `--max-spend`. Direct-MoE
funding has a separate total-cap mechanism. Review the live wall, GPU count,
CPU preparation and any extra time before authorizing a paid action.

One hotkey holds one chain commitment. Committing B or C replaces its prior
lane's commitment and can supersede a queued entry. Never switch lanes while
an existing leg is active without an explicit human decision.

Present the local results, seeds/settings, init/base, contract digest, exact
targets/ref, failed/unknown preflight checks and proposed spending limit to the
human. Submission, reveal commitment, publication and funding need explicit
authorization under [AGENTS.md](../AGENTS.md). Never print credentials or put
them on command lines. No command here updates or executes remote CLI code.

## Compute provider and credentials

Rolling A/B/C funding supports Lium, Shadeform, Vast.ai and Runpod. The default reads
`LIUM_API_KEY`; alternatively repeat `--provider-key-env lium=LIUM_API_KEY`
and `--provider-key-env shadeform=SHADEFORM_API_KEY` (or `vast=VAST_API_KEY`) in fallback order. One key
per provider is supported. The explicit list replaces the default. Keys come
from the human's environment; arguments contain variable names only. Respect
`--yes` consent and GPU/price caps. `cascade credentials` replaces the list
without queuing work. Provider selection is checked before create; an uncertain
create never authorizes a second rental. Discover deployed support through
`cascade status --json` (`compute.rolling_providers`); intake and trainer must
both be upgraded. Runpod uses `--provider-key-env runpod=RUNPOD_API_KEY`;
price-capped creates are refused, and its uncapped chosen-GPU path needs explicit
human authorization. Inspect `compute.spending_controls`: Cascade does not
verify a key's remaining budget. The [spending-control table](MINER.md#provider-budgets-and-cascade-price-ceilings)
distinguishes Lium key budgets, Shadeform instance thresholds and unconfirmed
hard key limits for Vast/Runpod. A preflight total is not an enforced cap.

The `cascade direct deposit-key --provider vast|runpod` command belongs to
the separate experimental direct runtime. Live activation is refused
in this release; accepting a key deposit does not open it for mining. The
[miner key guide](MINER.md#compute-provider-keys) covers provider permissions,
environment variables, exact deposit commands and API-key versus SSH-key roles.
Key custody and paid actions require the human's authorization. Agents must
not create, print or inspect keys to follow these instructions.

## Track one submission after funding

Each successful submit/fund response includes `data.tracking.argv`, an anonymous
read-only command bound to the hotkey, lane, immutable ref and B/C targets.
Poll it every 30–60 seconds with backoff:

```bash
cascade status --hotkey PUBLIC_SS58 --lane a --ref IMMUTABLE_REF \
  --intake https://YOUR_INTAKE --json
```

Read `data.submissions[]`: `status`, `observed.phase`, `freshness`,
`phase_age_seconds`, `observed.phase_history`, `observed.requeues`,
`retry_at`, `auto_retry`, `action_required`, and `next_action`.
For B/C add `--targets` (or `--target`) to distinguish different commitments
using the same generator ref. Preserve the exact ref/targets when polling.

Eligible queued/requeued entries carry forward automatically while their funding,
commitment and lane rules remain valid; a particular round is not reserved.
Do not resubmit or fund again just because a boundary passed. A queue target
is an estimate, and reveal-order position is not a GPU ETA. Public pending
reveals remain redacted: `not_found` is not proof of a rejected or lost entry;
inspect the funding receipt and `cascade reveal-status` before proposing action.

An `in_flight` legacy entry can be preparing data or waiting for compute;
only explicit stage evidence is labelled. A phase timestamp records its last
change, not a worker heartbeat. Missing/stale data returns exit 3 and an unknown
current state. `done` means queue work was consumed, not that it won, merged or
earned a reward. Read signed verdict/acceptance/confirmation reports separately.

Humans can append `--watch` instead of `--json` and stop with Ctrl-C. Agents
must poll single JSON snapshots; `--watch --json` is refused to preserve the
one-document output contract. Watching never funds, submits or opens a wallet.

The execution stage and `evaluation` are separate observations:

| Lane | Execution and scientific progress | Result evidence |
|---|---|---|
| A | Admission → data preparation → GPU provisioning → training → harvest → settlement | `cascade duel --hotkey HOTKEY --history --json`; queue `done` does not establish a win |
| B | The shared execution stages, then target qualification → waiting for warm-start acceptance → accepted or rejected | `evaluation.stage` reports the exact attempt when available; qualified contributions may wait across eras for non-overlapping merge rows/target slots |
| C | The shared execution stages, then publication → forward judging → pending confirmation, deferred, dropped, confirmed or below-holder result | `evaluation.reported` names the origin and observed receipt boundary; a passing confirmation below the holder's LCB does not take its slot |

B/C scientific progress requires the matching trainer/publisher release and a
fresh lane feed, even when the intake is fresh. Old entries without an exact
attempt binding stay unknown. A new attempt of the same ref does not inherit
the old attempt's result. C retains bounded per-entry observations across
later receipts and restarts; an older receipt cannot roll back that observation.
Receipt boundaries remain visible: a historical confirmed slot is not proof of
current payment. `evaluation.reward_status=not_verified` is deliberate; this
view does not verify signatures or prove that weights were included on chain.

## Rewards and burns

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

Full [reward schedule](MINER.md#rewards-allocation-burn-and-progression), including the launch transition, gain tiers, decay and a complete allocation example.

## V2 rollout and compatibility reference

For the research story, start with the [competition introduction](../agent.md).
This section preserves the detailed release and participation context.

### Changes from v1

V1 centred on a single full-model generator competition with a Toto2-4M
backbone. V2 keeps the controlled competition on training data and adds a
mixture-of-experts model lineage, targeted expert contributions and separate
forward-tested applications.

| Area | V1 baseline | V2 branch |
|---|---|---|
| Main model | Toto2-4M lineage | Cascade MoE lineage, starting fresh rather than importing Toto weights |
| Ways to compete | Full-model generator challenge | Lane A full-model data, Lane B selected-expert data, Lane C eligible fine-tuning recipes |
| Accumulating progress | Shared warm-start generations | Main-model promotions plus guarded Lane B expert merges |
| Rewards | King and prior-king decay | Lane allocations, Lane A reward ladder, B acceptance-based rewards and C confirmed slots |
| Preparing data | CPU streaming in the earlier training path | Materialised CPU-generated shards for the v2 Lane A contract; generation and GPU training have separate budgets |
| Agent workflow | Verify, score, submit | Fresh contract/init discovery, lane-specific local tests, preflight and exact-submission tracking |

These describe the implementation and rollout design, not a claim that every
feature is running on the public subnet. These are the checked-in templates;
zero means off, and a nonzero block is a configured gate, not evidence that
deployment or qualification succeeded:

<!-- cascade:config-reference:start -->
| Template | V2 block | Loaded B gate | C / MarketCast blocks | Random-init eras | B pool | CPU generation |
|---|---:|---:|---|---:|---|---|
| `chain.toml` | 0 | 0 | 0 / 0 | 14 | 25% | enabled for shards; 90 min configured |
| `chain.testnet.toml` | 8190000 | 8190001 | 0 / 0 | 0 | 50% | off; 90 min configured |
<!-- cascade:config-reference:end -->

Testnet retains its configured launch and older economics. A nonzero B launch
names an era start; the loader uses launch + 1 for its inclusive settlement
gate because the settlement at the launch boundary belongs to the preceding
era. This applies to both networks. Deployed configuration can differ: read
fresh live status before choosing a lane or a checkpoint. The separate
**direct-MoE** protocol is guarded experimental work, not the default A/B/C
mining path.

### Target selection and lane allocations

For B, miners choose declared targets; the current trainer selects routed
expert rows under the published limits. Arbitrary miner-selected expert IDs
are not an implemented CLI feature. Training fewer expert weights does not
remove the frozen model's GPU memory and forward-compute costs.

For C, implemented MarketCast targets include returns and volatility. Lane C
is designed to host eligible competitions, but listing a target does not open
it for mining. The pinned C base is independent of the current Lane A king.
Future architecture or masking competitions are not current mining options.

The planned mainnet rollout allocation is **A 100%**, then **A/C 75/25**, then
**A/B/C 50/25/25** as lanes launch. Those are lane pools, not guaranteed miner
payments. Gates, unfilled slots, decay, expiry and burn rules affect actual
rewards. Qualification, a completed training job and a local test pass are
all different from a paid win.

Operator release checks, feed publication and recovery are in the
[operator runbook](OPERATOR.md#bringing-it-up). CLI, intake, trainer and site
must expose compatible discovery and tracking fields; a CLI update cannot
create missing server telemetry.

The v2 channel range applies to the `values` array of every A/B yield and to any
optional synthetic generator in a C recipe. Lane C preflight verifies that
optional generator as well as the recipe. A C recipe without a generator does
not need synthetic channels; its real market inputs are a separate contract.
