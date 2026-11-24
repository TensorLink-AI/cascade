# Cascade: synthetic data, real forecasting tests

**Cascade's ambition is a model that sees the future.**

Contributors compete to write data generators: small deterministic programs
whose synthetic time series teach a shared model to forecast real data. Each
generator encodes a prior, a hypothesis about patterns a forecaster should learn,
such as changing electricity demand, shifting business seasonality or a sudden
regime change. Controlled training and held-out tests measure whether that prior
taught the model something useful.

**A small program can have a lasting effect.** Winning data can shape the model
lineage through checked promotions; accepted specialist updates can enter future
starting checkpoints. Miners compete individually, while later competitors can
build on earlier discoveries. That is the idea behind the name **Cascade**:
measured improvements become the starting point for further improvements.

The competition runs on Bittensor, where qualifying contributions can earn
emissions. The research question is whether independent experiments can produce
a general forecasting model that becomes more useful over time. The ambition
has to be earned in testing; it is not a claim that the model predicts events
with certainty or has already achieved superior performance.

**For a coding agent:** read [AGENTS.md](AGENTS.md) before acting, then choose
the [A/B/C walkthrough](docs/AGENT_MINING.md). Local experiments are the
default; publication, submission and spending require human authorization.

The v2 Lane A launch target is **30 October 2026, 12:00 UTC**, aligned to the
first era boundary at or after that time using finalized chain timestamps.
The **95% pre-launch burn** activates separately after **51% of eligible
validator stake** signals this release. Installation alone activates neither.
Read fresh live status before proposing mining work; B/C have separate gates.

Paid A entries open at **30 October 2026, 09:00 UTC**. Use a fresh revealed
commitment from that opening; accepted entries wait for the actual v2 era,
and submissions stay open after launch. Lanes B, C and D open together at
block 9586800 (about 2026-11-25 22:28 UTC).

## One contribution, from idea to the next experiment

*Illustrative example, not a reported experiment or measured result:*

A contributor writes a generator in which a business's weekly sales pattern
changes as its customer mix changes. Cascade trains the fixed model on that
generator's output, then compares its forecasts with the reigning model on
held-out real series. A useful-looking generator can still lose this test.

If the challenger wins and later meets the promotion checks, its trained
checkpoint can become the shared starting point for a later generation. New
experiments then build from that checkpoint. The contributor's original program
may be small, but the improvement it helped produce can outlive its own round.

## Two ways to improve the shared model, one separate application lane

| Lane | What a contributor does | Where a successful contribution goes |
|---|---|---|
| **A — full model** | Designs synthetic data that trains the full fixed model to challenge the reigning model | A winning checkpoint can become king and, after promotion checks, advance the shared starting model |
| **B — selected experts** | Designs targeted synthetic data to improve specialist subnetworks inside the model | A qualified update must pass merge acceptance when enabled before it enters a future starting checkpoint and starts rewards |
| **C — applications** | Provides an eligible fine-tuning recipe; currently MarketCast on a separately pinned Timestral base | A separate application checkpoint, tested against future observations and confirmed later; it does **not** merge into A/B |

A and B are paths into the shared model lineage. C asks whether an application
fine-tune helps on genuinely future observations, using its own base and reward
slots. Not every lane or target is necessarily open on a deployment.

## The four competitions at launch, and the offline loop for each

Mainnet launch (the signed overlay `deploy/overlays/launch-20261125.mainnet.toml`):
lanes B, C (MarketCast) and D open together at block **9586800** (about
Wed 2026-11-25 22:28 UTC). From that block the emission split is
**A 0 / B 35 / C 15 / D 50**; lanes C and D carve their shares from lane A.
Until a miner first wins the lane D crown, lane D's 50% stays with lane A
(owner decision DEC-CA-0076: the house, our lane A king, only seeds lane D).
Lane D launches **live-only** (`public_weight = 0.0`); the 50/50 live/public
rule is switched on later by overlay. Read `cascade status --json` (`lanes`)
for what is armed now.

| Lane | What to optimise | Offline loop (local, free, never spends) | Submit and fund (human authorisation) |
|---|---|---|---|
| **A** | Generator data that trains the fixed MoE to beat the king on private windows (geomean of CRPS and MASE) | `cascade score DIR --pool-dir P`; `cascade iterate DIR --pool-dir P --seeds 0,1` (add `--warm-start live` for the final check) | `cascade submit`, or `cascade deploy` plus `cascade fund`; your compute key |
| **B** | Generator data for 1–4 declared targets, training only selected expert rows: gain on a target, no regression elsewhere | `cascade lane-b test DIR --targets energy --base live --pool-dir P` (the validators' slot judge on every slot) | `cascade deploy DIR --expert-domain energy` plus `cascade lane-b fund`; your compute key |
| **C** (MarketCast) | A fine-tuning recipe on the pinned Timestral base that beats it on future market windows | `cascade lane-c fetch-base`, `cascade lane-c snapshot`, then `cascade lane-c test DIR --target mkt:market --base-dir B --train-snapshot S --eval-snapshot E` (the validators' MarketCast scorer and verdict) | `cascade lane-c upload`, `cascade lane-c commit`, `cascade lane-c fund`; your compute key |
| **D** | BF16 `cascade_pyramid` weights trained anywhere that beat the lane D king on forward-only live windows (before the first crown: the house, the lane A king) | `cascade lane-d test CKPT --king KING --context C --pool-dir P` (or `--series S.npz`): the validators' window draw, `score_window` and `judge_cohort`, against the crown bar (3% at a new king's tenure 0, falling to 1%; `--tenure`); `--public-weight 0.5` previews the later rule | `cascade lane-d export`, `verify`, `submit`, `ready`; free, no compute key |

Every offline result is directional: the live pools are private and newer
than anything a miner holds. The per-lane briefs are
[A](docs/MINER_LANE_A.md), [B](docs/MINER_LANE_B.md),
[C MarketCast](docs/MINER_LANE_C_MARKETCAST.md) and [D](docs/MINER_LANE_D.md).

## Lane D: open weights on the same architecture

Lane D asks how far the fixed cascade architecture can go when training is
unconstrained. A contributor trains a `cascade_pyramid` checkpoint anywhere,
on any data and with any recipe, and submits only the BF16 weights; the
operator only evaluates, on forward-only private windows whose targets reached
our data after both models were committed. Our own lane A king (the house)
is the bar for the first crown, so lane D pays only for weights better than
our pipeline. On mainnet the signed launch overlay arms it at block 9586800,
scored on live windows only at first. The guide is
[docs/LANE_D.md](docs/LANE_D.md); the brief is
[docs/MINER_LANE_D.md](docs/MINER_LANE_D.md).

An agent can preview (`cascade lane-d test`), export and verify an entry
locally. Uploading, attesting the
licence and committing on chain need the human's explicit authorisation, as
for every submission, publication or spend ([AGENTS.md](AGENTS.md)).

V2 synthetic generators must emit finite float arrays `(C, L)` with **exactly 32 channels per yield**. One-dimensional and 1–31-channel yields are rejected. Read the effective policy for length and resource limits.

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
- **B winner:** with merge acceptance enabled, qualification is unpaid. Acceptance into a scheduled shared starting checkpoint starts payment. The 10 slots each pay 2.5% of total emissions to their holder every settlement, with no decay or expiry, until a newer leg's accepted merge takes the slot. Every leg is tested on every slot: up to 4 declared targets clear at a 1% gain LCB, any other slot at 2%; one miner may hold every slot.
- **C winner:** an initial test pass is unpaid. A later confirmation must win the target slot. C's pool is divided by competition weight, then target count. The current sole competition, MarketCast, has 1 target: each full-pay slot is 25% of total emissions.
- **C full-pay duration:** MarketCast C: 5 days (10 eras) at LCB 0.01, rising linearly to 20 days (40 eras) at LCB ≥ 0.05. C uses confirmed gain; each win keeps its recorded duration. B has no tiers.
- **C progression:** C has no decay: its slot ends when the full-pay duration ends. Age starts at C confirmation. A new holder stops the old holder's pay.
- **Burn:** unpaid A, empty B slots, empty/ended C slots and inactive competition allocations go to `burn_uid`; they are not shared among other miners. An unlaunched lane has no reserved pool.
- **Check payment:** read signed reward records and on-chain weights. Queue completion, qualification and pending confirmation are not paid wins. Public status reports `reward_status=not_verified` because it does not verify payment.
<!-- cascade:rewards:summary:end -->

Full [reward schedule](docs/MINER.md#rewards-allocation-burn-and-progression), including the launch transition, gain tiers, decay and a complete allocation example.

## Follow the research without mining

Open the [public dashboard](https://s3.hippius.com/cascade-manifests/index.html).
The **king** is the reigning Lane A competitor; a **warm start** is the shared
checkpoint used to begin a later training experiment. They are different roles.

For a concrete way to inspect progress in the v2 dashboard, open **Benchmarks**,
choose **GIFT-Eval**, select **MASE × CRPS**, and leave **King trajectory** enabled.
Each axis measures forecast error, so lower is better. Follow the current and
prior king points, then open **Models and scores** to inspect the plotted values.
A move toward lower values on both axes is an improvement on that benchmark;
a trade-off between them needs closer inspection. Arrows show reign order,
not a guarantee that every new king improves every public benchmark.

Compare matching tasks and data revisions. One benchmark trajectory does not
establish broad forecasting progress. Missing reports provide no result to
compare. The deployed dashboard may lag this branch; if these controls are
absent, they describe the v2 view awaiting release, not an observed live result.

The [public receipt index](https://s3.hippius.com/cascade-manifests/receipts/index.json)
is another concrete artifact to inspect: choose a scored settlement and follow
its receipt key to the manifest, evaluation and decision record. The
[audit guide](docs/AUDIT.md) explains how to check the signed evidence. Public
charts are discovery views; signed verdicts determine competition outcomes.

For an actual historical record, open this
[scored receipt published on 9 September 2026](https://s3.hippius.com/cascade-manifests/receipts/5FcCsoB2R5M55RdrNrvPQv7xmLJRFzs7R7C73UUSxgUCyfsw/round-26346216572362586.json).
It contains the training manifest, evaluation scores and recorded verdict.
It is an artifact from the earlier competition, not the illustrative example
above or evidence of v2 performance.

## What changed from v1

V1 centred on one full-model generator competition with a Toto2-4M backbone.
V2 starts a fresh mixture-of-experts lineage, adds specialist contributions in
B and separately tested applications in C. It also adds guarded expert merges
and agent tools for discovering the current rules and testing ideas locally.
The [rollout reference](docs/AGENT_MINING.md#v2-rollout-and-compatibility-reference)
holds the migration table, activation details and lane allocation schedule.

## For an agent ready to experiment

A generator is an inspectable Python program: propose one idea, implement it,
verify determinism, train and compare on real held-out data. Keep the base,
settings and seeds fixed while comparing ideas. A local win is directional;
live evaluation can produce a different result.

The new [Cascade MoE Base - 1.23B](docs/MODEL.md) has 1.234B total parameters
and 112.07M active per token. Smaller variants, **Cascade MoE Base - 17.58M,
69.02M and 273.84M**, give miners a cheaper way to improve generators. Model
names use total parameters. Use `cascade score`
or `cascade iterate` with a [complete local config](deploy/presets/README.md),
then recheck promising ideas on the live model. Keep both sides' settings and
compatible starting checkpoint fixed. `--warm-start live` selects the actual
competition contract and init. Small-model gains do not establish paid wins.
GPU performance is unmeasured for this port; 56M remains unspecified.

Start with the [agent operating instructions](AGENTS.md), read the
[generator interface](docs/INTERFACE.md) before editing a generator, and use the
[lane-specific walkthrough](docs/AGENT_MINING.md) for installation, local tests,
preflight, costs and submission tracking. Source checks can run on a CPU;
meaningful model experiments need suitable compute, and live training uses
funded hardware. Paid actions and live submissions require human authorisation.

Discover what is actually active using `cascade status --json` and check the
release with `cascade discovery-check --json`. Neither static prose nor an old
checkpoint is a substitute for fresh rules. The [architecture](docs/ARCHITECTURE.md)
and [decisions](decisions/) explain the implementation and its amendments.

Optional generator `groups` labels pack independent channel groups into each
32-channel yield. Miners choose the packing and group-size mix; the
starter distribution is an example, not an admission rule. Use them only when the effective contract accepts
`groups`; omission or `None` keeps the values-only path. See the group-packing
section of [the interface](docs/INTERFACE.md#pack-independent-groups-in-one-yield).
