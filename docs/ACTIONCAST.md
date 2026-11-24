# ActionCast — design

**ActionCast** is a lane-C competition. Miners write *decision software*, a
**harness**, that turns our fine-tuned models' forecasts into actions. It
trades a risk mandate, dispatches a battery, orders stock, or covers any
vertical we fine-tune for. Harnesses are judged forward-only on real data,
in simulated episodes, against a reference harness on the same model.

- **Decision record:** [DEC-CA-0073](../decisions/DEC-CA-0073-actioncast.md)
  holds what is decided, the owner rulings and the open questions (`Q1`…`Q10`
  below refer to it).
- **Status: design, not built, not armed.** Every number is a proposed
  default.
- **Plugin interface:** [LANE_C_COMPETITIONS.md](LANE_C_COMPETITIONS.md).
  ActionCast is the `act` competition kind; targets are `act:<scenario>`.

**The Cast set.** Lane C competitions are named in one family. Each
vertical has a fine-tune (the model) and an ActionCast scenario (the
software that uses it):

| vertical | fine-tune | kind | ActionCast scenario |
|---|---|---|---|
| markets | **MarketCast** (live) | `mkt:` | `act:perp` |
| energy | **EnergyCast** (to build) | `nrg:` | `act:bess` |
| inventory | **InventoryCast** (to build) | `inv:` | `act:inventory` |
| next vertical | **<Vertical>Cast** | `<abc>:` | `act:<scenario>` |

The rule (owner): **models are `<Vertical>Cast`; software is always
ActionCast.** A new vertical never gets its own software competition. It
gets a `<Vertical>Cast` fine-tune and an `act:` scenario on the shared kit.

## Contents

1. [Why](#1-why)
2. [System at a glance](#2-system-at-a-glance)
3. [Lifecycle of an entry](#3-lifecycle-of-an-entry)
4. [Verdict and payout](#4-verdict-and-payout)
5. [Vertical pairing](#5-vertical-pairing)
6. [Harness stack: forecasts, Engy LLMs, decisionmodels.io](#6-harness-stack-forecasts-engy-llms-decisionmodelsio)
7. [Primitive kit](#7-primitive-kit)
8. [Scenarios](#8-scenarios)
9. [Rewarding creative use of forecasts](#9-rewarding-creative-use-of-forecasts)
10. [Explaining actions to the customer](#10-explaining-actions-to-the-customer)
11. [Incentives: king of the hill, privacy, duplicates](#11-incentives-king-of-the-hill-privacy-duplicates)
12. [Submission](#12-submission)
13. [Rules](#13-rules)
14. [Implementation map](#14-implementation-map)
15. [Config sketch](#15-config-sketch)
16. [Rollout](#16-rollout)

## 1. Why

Lane C pays for better **models**: MarketCast fine-tunes, judged on CRPS.
Nobody is paid to turn a forecast into a **decision**: how much to hold, when
to charge, how much to order. That step is where a forecast earns its value,
and most of it is not forecasting. It covers sizing on uncertainty, costs,
constraints, risk, and what to do when the forecast is wrong.

ActionCast pays miners for that software. It also gives each fine-tune a
decision-based test on top of CRPS. And it grows a shared, open kit of
building blocks that every new vertical reuses.

## 2. System at a glance

```
 MINER                          TRAINER (miner's key)            MAIN VALIDATOR                       EVERYONE
 ─────                          ─────────────────────            ──────────────                       ────────
 harness tree ──commit──▶ chain ──▶ CPU qualifier leg            forward recorders (prices, demand,
 (policy.py,  act:<scen>          · static guard + smoke run      text feeds) ─────────┐
  harness.json)                   · historical backtest vs ref                         ▼
                                  · freeze pointer ──manifest──▶  episode runner (sandbox, no network)
                                                                    │  miner harness
                                                                    │  reference harness
                                                                    │  miner harness on naive forecasts
                                                                    ▼
                                                                 model gateway ── pin · budget · cache · record
                                                                    ├─▶ vertical fine-tune (GPU pod)
                                                                    ├─▶ Engy LLMs (SN53, with proofs)
                                                                    └─▶ decisionmodels.io
                                                                    ▼
                                                                 score terms → paired bootstrap → verdict
                                                                    ▼
                                                                 report + transcripts ──────────────▶ satellites verify digests
                                                                                                     auditor replays on CPU
                                                                                                     winners published
```

## 3. Lifecycle of an entry

1. **Build.** The miner writes a harness on the kit (`cascade.harness`) and
   backtests it locally on recorded history (`cascade harness backtest`).
2. **Upload, commit and fund.** The miner uploads the harness to the
   private vault, commits
   `metro-v1:ft:act:<scenario>:hippius:vault/direct@sha256:<digest>` and
   funds a short CPU leg.
3. **Qualify.** The trainer runs the static guard and a smoke run. It then
   backtests the harness against the reference on public history. Failing
   entries stop here, before they cost validator GPU. The trainer freezes
   the pointer into the signed lane-C manifest.
4. **Wait for the future.** Episodes must start after the leg's
   `train_block` plus an embargo. The validator waits until enough forward
   data has been recorded (`verdict_wait_blocks`).
5. **Judge.** The main validator draws the round's episodes (variants and
   hard tiers from the round seed). It runs these passes on the same
   episodes: the miner, the reference, the holder and every published
   harness, and the miner on naive forecasts.
   Every model call goes through the gateway into a transcript.
6. **Verdict.** Score terms → paired cluster bootstrap → the gates in §4.
   Explanations are checked in the same pass (§10): grounding, truth,
   counterfactual replays and the simulatability reader.
   The best passer per target becomes pending.
7. **Confirm.** At the next settlement the pending entry is re-judged on
   fresh episodes, on the current fine-tune holder. Pass → it takes the main
   slot or the novel-use slot.
8. **Pay, then publish.** The slot pays on the lane-C schedule while the
   harness stays private. When it is dethroned or decays to the floor, its
   code, transcripts and explanations are published and join the reference
   set.
9. **Audit.** After publication, anyone replays the reign from the
   transcripts on CPU. They
   check Engy proofs, re-query a sample of decision-model calls, and
   re-derive every score and LCB.

## 4. Verdict and payout

A judge-phase entry passes when **all** of these hold, at `alpha / k`
(Bonferroni over the target's entries this settlement):

| gate | rule |
|---|---|
| gain | composite gain LCB over the reference ≥ `gain_margin` (0.01), with a strictly positive point gain |
| no term worse | every score term's LCB ≥ −`tau`: a gain on one term cannot buy a confident loss on another |
| model lift | the harness's score with the real fine-tune beats its own naive-forecast run (`LCB ≥ model_lift_margin`) |
| clusters | at least `min_target_clusters` distinct clusters (start days / regions / SKUs) |
| factual | false-claim and unsupported-sentence rates are under `max_false_rate` / `max_unsupported_rate` at their upper bounds (§10) |
| explanation | explanation score LCB over the reference's ≥ −`tau`, so explanations can't be traded away for decision gain (§10) |

Then the DEC-CA-0056 machinery applies: the best passer becomes pending
and is confirmed at uncorrected `alpha` at the next settlement. The slot
rule is stricter than lane C's (§11): to take an occupied slot, the
challenger must beat the holder **and every published harness**, paired on
the same episodes, by `increment_margin` at the LCB. Each `act:<scenario>` has two slots:

| slot | who | share |
|---|---|---|
| main | best confirmed qualifier | 1 − `novel_share` |
| novel-use | best confirmed qualifier whose per-episode gains correlate ≤ `novel_max_corr` with the main holder's, and whose forecast value ≥ the reference's | `novel_share` (e.g. 0.3) |

Payout is the competition's weight in `[lane_c.competitions.act]`, split
equally across its targets, then split per slot, on the gain-tiered lane-C
schedule with its TTL. Unpaid slots burn.

## 5. Vertical pairing

Each vertical has two lane-C competitions that feed each other:

| vertical | fine-tune target (the model) | harness target (the app) |
|---|---|---|
| markets / trading | **MarketCast**, `mkt:market` (live) | `act:perp` |
| energy | **EnergyCast**, an energy-price fine-tune, `nrg:<market>` (to build) | `act:bess` |
| inventory / demand | **InventoryCast**, a demand fine-tune, `inv:<suite>` (to build) | `act:inventory` |

Rules:

- **The forecaster is the fine-tune's slot holder.** At judging, the validator
  loads the trained pointer of the paired fine-tune target's current holder.
  Before that target has a holder, it uses the fine-tune competition's pinned
  base (Timestral-28M for MarketCast). This is the same model a winning
  fine-tune would ship.
- **Harness entries are judged on one model.** Each judge-phase row records
  the forecaster pointer. The miner's run, the reference run and the
  naive-model run in a verdict all use that pointer.
- **When the fine-tune holder changes.** The harness slot holder keeps its
  slot, but its next confirm runs on the new model. A pending entry whose
  confirm would run on a different model than its judge phase is dropped as
  `stale` and requeued uncharged, like `base_moved` in DEC-CA-0056. A harness
  that only worked with the old weights loses its slot at that confirm.
- **An app arms only after its vertical's fine-tune.** `act:<scenario>`
  needs its paired fine-tune target configured (load-checked). Running on the
  pinned base until the first holder is allowed.
- **Feedback to the fine-tune (later, not consensus at first).** Each verdict
  also reports the reference harness's score on the new holder against the
  old one: a decision-based check on whether a CRPS win helped in use. It is
  published as a diagnostic only. Making it a guard on fine-tune wins is a
  later decision.

## 6. Harness stack: forecasts, Engy LLMs, decisionmodels.io

A harness is more than a forecast plus an optimiser. It can reason over text
and context with an LLM, and ask a decision model discrete questions. Both
sit around the vertical's fine-tune, not in place of it.

| role | provider | what it gives the harness |
|---|---|---|
| forecasts | our lane-C fine-tune for the vertical (main-validator pod) | quantile forecasts |
| LLM | **Engy** (Bittensor SN53) | text and tool-use reasoning, with proof of which model ran |
| decision model | **decisionmodels.io** | probabilities over a typed set of allowed answers, given a state |

```
            ┌──────────── harness (miner code, sandboxed, no network) ─────────┐
 obs, spec, │  policy.py ── models.forecast | models.llm | models.decide        │ → action
 text feed  └──────────────────────────────┬────────────────────────────────────┘
                     model gateway (main validator): pin, budget, cache, record
               ┌──────────────┬────────────┴──────────────┐
         fine-tune pod     Engy (SN53)            decisionmodels.io
                  every request, reply (and Engy proof) goes in the transcript
```

> **Provider facts to confirm.** These are taken from public pages
> (github.com/hanlinai/engy, a third-party listing for decisionmodels.io),
> not from the providers. The earlier Ephemeris router experiment
> (`doc:exp-decision-model-router`) also could not reach decisionmodels.io
> from the sandbox. Before anything below is fixed, the provider facts must
> be checked against each provider's own API docs. That covers the API
> shapes, the model names, version pinning and prices.

### One model gateway

The sandbox still has no network. The harness calls `models.*`, and the
gateway on the main validator makes the outside call. The gateway:

- **Pins.** Each call names a registry entry. The gateway sends the pinned
  model id, version or hash and fixed decoding settings: temperature 0, or a
  seed from the round seed.
- **Budgets.** Tokens and decision-model calls are counted per episode, and
  cadence caps apply per scenario.
- **Caches.** Replies are stored by the hash of (pinned model, request). The
  miner run and the naive-model run share cached replies where their
  requests match. A judging restart never pays twice or gets a different
  answer.
- **Records.** Every request and reply goes in the transcript, with Engy's
  inference proof where there is one. The auditor replays the harness from
  the transcript on CPU, with no calls to any provider.
- **Fails safe.** A provider outage or timeout is a transient: the
  settlement holds and retries, as in the MarketCast B2 rule. It is never
  the miner's fault. Repeated outages past `report_timeout` make the round
  `absent`, not a loss.
- **Is main-validator only.** Satellites never call the providers. They
  verify the main validator's report digests and transcripts, the existing
  arbiter pattern.

### LLMs from Engy

- **Why Engy fits.** Engy's pitch is verifiable inference: a proof that the
  model requested is the model that ran. That solves the audit problem for
  LLMs. Each transcript entry stores the proof, and the auditor checks it
  without regenerating. Spot-check regeneration (the earlier plan) becomes a
  fallback for calls without a proof.
- **Registry.** `[actioncast.providers.engy]` lists the allowed models by
  Engy's model id and weights hash. Only models whose training cutoff is
  before the competition's `from_block` are allowed.
- **No tools beyond the gateway.** Requests go out with no provider-side
  tools (no web search, no browsing). An LLM can only see what the harness
  passes it. This matters more than with a self-hosted model: judging runs
  **after** the episode's data exists, so a model with web access could look
  the outcome up.
- **What an LLM can read.** The observation, plus forward-recorded text
  feeds: the event calendar with descriptions, the mandate or operating
  brief written as text, and later licensed headlines. All text is data: the
  simulator validates the action whatever the LLM said.
- **Tool use.** The kit's primitives are exposed as tool schemas
  (`cascade.harness.agent`). An Engy LLM can run an agent loop inside a
  step, for example forecast, check the tail, then size, and every tool call
  is a gateway call against the same budgets.
- **Cadence.** For example, one LLM step per hour for `act:perp`, per 2 h
  for `act:bess` and per day for `act:inventory`, plus a token budget per
  episode. The LLM plans, and fast numeric code acts every step.

### Decision models from decisionmodels.io

- **What they are for.** As listed, a call sends a *state* and *typed
  questions* and gets back a probability for each allowed answer. That maps
  onto the discrete choices inside a harness, for example:
  * which regime this is (calm / trending / stressed);
  * which decision kernel to use now;
  * which forecaster, horizon or context length to ask (the Ephemeris
    router experiment, task #30);
  * whether to hedge, rebalance or order (yes / no or a few buckets);
  * which tier of the mandate brief applies.

  The harness turns the probabilities into an action with kit primitives,
  for example a probability-weighted kernel blend.
- **Kit call.** `models.decide(name, state, questions)` returns
  `{question: {answer: p}}`. `state` is built with kit helpers from the
  observation and forecasts. Forecast quantiles are a natural input, which
  keeps the fine-tune in the loop.
- **Registry.** `[actioncast.providers.decisionmodels]` pins the model names
  and versions. The listing names `s1-fast`, `s1-pro`, `s1-vision` and
  `Quyet-1.0-Large`, said to be Apache-2.0: to confirm.
- **Determinism and audit.** Replies (probabilities) go in the transcript,
  and the harness replays from them. To trust the replies themselves, the
  auditor re-queries a seeded sample. Probabilities must match within a
  tolerance, so the provider must be deterministic for a pinned version
  (to confirm). If it is not, and if the models really are open weights,
  the fallback is to self-host the pinned weights on the main-validator pod
  behind the same gateway call.
- **Miner-shipped decision models.** Shipping weights in the tree (an RL
  policy, trees, a small net, run on CPU) is no longer the main path. Q10
  asks whether to keep it alongside decisionmodels.io. Either way, the
  model-lift gate, the forecast-value gate and an attribution row (the
  score with the fine-tune against forecasts zeroed to naive, other models
  held fixed) make sure the gain still depends on the fine-tune.

### Who pays

Forecasts stay on the main validator, as in MarketCast. Engy and
decisionmodels.io calls cost real money per token or per call. Proposal:
meter every external call per entry, and charge it to the miner's funded
key, capped by the episode budgets. An entry that makes no external calls
costs the same as before (Q10).

## 7. Primitive kit

We fine-tune a model per vertical, and every fine-tune should get an app
without new competition code. Under the surface, the three scenarios above
are the same few problems:

- the battery is a **store** (level, charge and discharge, capacity, losses);
- inventory is a **store** whose outflow is the forecast series (demand);
- the trading mandate is a **position** held against a risk budget.

So the kit is split by layer. Each layer is pure numpy / scipy,
deterministic and versioned (`api = 1`, additive-only; a breaking change
bumps `api`, and older harnesses keep running on a shim, like
`[generator] interface_version`). It ships as one package used by three
groups: miners write harnesses on it, the owner writes scenarios on it, and
the validator and auditor run both on it.

### Layer 1 — data (`cascade.harness.data`)

| primitive | what |
|---|---|
| `Series` | values, timestamps, frequency, mask, channels, metadata (`vertical`, `unit`, `source`) — the existing carrier fields |
| `Observation` | a past-only view of every series in the episode, plus state; cannot be indexed past "now" |
| `Spec` | the episode's constants and variant draws, delivered at `reset` |
| transforms | `returns`, `log`, `resample`, `deseasonalise`, `basket(weights)`, `spread(a, b)`, `aggregate(window)`, `align` |

### Layer 2 — forecasts (`cascade.harness.forecast`)

| primitive | what |
|---|---|
| `Forecaster` | the handle: `forecast(contexts, horizon, levels, covariates, model)`, `models`, `levels`, `max_horizon`, `budget_left()` |
| `QuantileForecast` | the reply: `quantile(p)`, `interval(a)`, `cdf(x)` / `ppf(p)` interpolated, `median`, `spread`, `pit(y)` for checking past forecasts |
| `Scenarios` | the harness builds sample paths from marginals with a copula it chooses (`independent`, `comonotone`, `gaussian(rho)`), labelled as its own construction, never as model paths |
| `RiskFromQuantiles` | VaR / ES, upside / downside, and probabilities of crossing a threshold, all read off the marginals for one horizon step |
| `Calibrator` | online recalibration of the model's quantiles from the harness's own past PIT values: a sanctioned, deterministic place for "the model is a bit wide here" |

### Layer 3 — decisions (`cascade.harness.decide`)

Vertical-agnostic decision kernels that miners compose, replace or beat:

| primitive | what |
|---|---|
| `newsvendor(q, under, over)` | the critical-fractile quantity |
| `base_stock` / `s_S` | inventory-style replenishment policies on lead-time demand |
| `store_dispatch(scenarios, store, prices)` | a stochastic or rolling LP for any store (battery, warehouse, reservoir, capacity pool, cash) |
| `size_to_risk(target, risk)` | scale a position to a vol, VaR or ES target |
| `hedge(book, instruments, risk)` | the minimum-variance or tail hedge ratio |
| `band(target, current, cost)` | a no-trade or no-order band, to limit churn |
| `mpc(model, horizon, solver)` | receding-horizon control around any of the above |
| `commit_then_adjust` | a two-stage decision (day-ahead commitment plus real-time adjustment) |

The reference harness for every scenario is just a composition of these
kernels. Miners see exactly what they have to beat.

### Layer 4 — environments (`cascade.harness.env`, owner-side)

| primitive | what |
|---|---|
| templates | `Store` (level, in / out controls, capacity, efficiency, leakage), `Position` (exposure, hedges, mandate), `Commit` (a two-stage commitment with imbalance settlement). A scenario is one template or a few composed |
| `ActionSpace` | box, integer, simplex; the simulator clips and logs every action |
| `CostModel` | linear, fixed, slippage from spread / volume, holding, shortage, degradation, imbalance |
| `Constraint` | limits, ramp rates, capacity shared across items, utilisation floors |
| `Oracle` | perfect-foresight LP / MIP for any `Store` or `Commit` scenario; none for `Position` |
| `ScoreTerm` | a loss, how it aggregates (sum ratio against the reference), its weight and its `tau` guard; the library has tracking, tail, FZ0, cost, regret ratio and capture ratio |
| `VariantSampler` | seeded ranges for every spec constant (the creativity variants) |
| `Tier` | an episode selector (events, stress, spikes, cold start) with a share |

### A new vertical = a data adapter + a scenario spec

```toml
# cascade/harness/scenarios/hydro.toml   (example: a hydro reservoir)
scenario  = "hydro"
model     = "inv:inflow"            # the paired fine-tune target
template  = "Store"
data      = "adapters.river_flow"   # forward recorder + adapter (code, reviewed)
step      = "1d"
episode   = "28d"

[store]       # with variant ranges
capacity   = [500, 2000]
efficiency = [0.85, 0.95]
inflow     = "series:inflow"        # the forecast series drives the store
price      = "series:power_price"

[score]
terms     = ["capture_ratio", "spill", "cost"]
reference = "mpc(store_dispatch, horizon=14)"

[[tier]]
name = "drought"
share = 0.2
select = "inflow_pct < 10"
```

- **What is code.** Each data adapter, plus any genuinely new template, cost
  or score term. These are reviewed and shipped in a release (consensus).
- **What is spec.** Everything else. Load-checked: the model target exists,
  every series is forward-recorded, the terms and reference resolve, and the
  variant ranges are bounded.
- **What it unlocks.** The same competition kind (`act`) serves every
  vertical. Arming a vertical's app is a release with its spec, then a
  `from_block`, as for any lane-C competition. Candidate next verticals on
  the existing templates:
  * cloud capacity (`Store` on `domain:web_cloudops`);
  * staffing (newsvendor `Position` on call or footfall counts);
  * fleet and bike rebalancing (a multi-site `Store`);
  * fuel and power hedging (`Position`);
  * day-ahead bidding (`Commit`).

### Harnesses that carry across verticals

Because harnesses see templates, not verticals, a harness written for
`Store` runs on any `Store` scenario. `harness.json` may declare
`"scenarios": ["bess", "inventory", "hydro"]` and enter each one. Each entry
is still judged and paid per scenario, by the rules above.

Optional (Q9): a **generalist slot** `act:store` (and `act:position`),
scored on the geometric mean of the gains across every armed scenario of
that template. It would include one scenario added after the harness was
committed, so the harness cannot have been tuned to it. This pays for
software that works on the next vertical we fine-tune, not only the ones
it was tuned on.

### Growing the kit

The kit starts as the owner's set. Ideas that keep winning (a calibrator, a
scenario builder, a dispatch heuristic) can be promoted into the kit in a
later release, crediting the commitment that introduced them. That makes it
the shared base for the next round of harnesses. Promotion is an owner call,
not a consensus rule. The same kit can also sit on top of our hosted
forecasting API, so apps that win here can be used outside the subnet.

## 8. Scenarios

Each scenario fixes the observation, the action space, the cost model, the
episode length, a reference harness and (where one exists) a
perfect-foresight oracle. Where there is an oracle, the score is
oracle-normalised and so scale-free across assets, markets and days.

### `act:bess` — battery arbitrage on real-time power prices (recommended first)

- **Data.** Public 5-min regional prices that need no key: AEMO NEM
  (5 regions) first. ERCOT / CAISO are candidates once their terms are
  checked. A forward recorder, like the MarketCast bar recorder (M15), keeps
  the series.
- **Asset.** A notional 1 MW / 2 MWh battery, round-trip efficiency 0.88,
  SoC limits 5–95 %, a degradation cost per MWh throughput, a price-floor
  and price-cap clip.
- **Step / action.** Every 5 min: a power setpoint in [−1, 1] MW, clipped to
  feasibility by the simulator.
- **Episode.** 24 h from a random interval start, one region.
- **Oracle.** The perfect-foresight LP (`scipy.optimize.linprog`, HiGHS) on the
  realised prices.
- **Score.** Capture ratio `profit / oracle_profit`. The gain is the paired
  difference against the reference harness.
- **Reference harness.** A rolling-horizon LP on the forecaster's median
  (P50) path, re-solved every hour.
- **Forecaster.** The energy-price fine-tune's slot holder (`nrg:<market>`).
  That competition is MarketCast-shaped: a pinned base, forward-only CRPS on
  the same AEMO prices. Both the recorder and the data adapter are shared
  between the two.

This is the cleanest scenario. It has a bounded oracle, so the score is
scale-free and the noise is low. Quantile forecasts help in a way you can
measure: reserving SoC for spikes depends on the upper tail.

### `act:inventory` — periodic-review replenishment

- **Data.** Real daily count series recorded forward. Candidates that are
  public and keyless: Wikipedia pageviews per article (as SKU demand, scaled
  to integer units), and city bike-share daily trips per station. Each SKU
  has a unit cost, a holding cost and a stockout penalty, drawn from a seed
  fixed at the round.
- **Step / action.** Daily: an order quantity per SKU (≥ 0, integer), with
  lead time L ∈ {1, 3, 7} days, a shared capacity cap, and MOQ / case packs
  in harder tiers.
- **Episode.** 28 days × 20 SKUs.
- **Oracle.** A perfect-foresight min-cost plan (an LP / MIP that is small at
  this size).
- **Score.** `1 − (cost − oracle_cost) / (ref_cost − oracle_cost)`, clipped,
  or the cost ratio. This is Q3.
- **Reference harness.** A base-stock policy at the critical fractile
  `p / (p + h)`, using the forecaster's quantile of lead-time demand.
- **Forecaster.** The demand fine-tune's slot holder (`inv:<suite>`), trained
  on the same forward count series.

### `act:perp` — risk-managed perp mandate (owner, 2026-11-24: not scored on trading success)

Profit is too noisy to judge, and most of it comes from the direction call,
which this app does not test. Instead, **the harness is forced to hold a
position and is scored only on how well it manages that position's risk**.
Every score term depends on volatility, tails and costs. Those are what the
MarketCast fine-tune forecasts (its `vol`, `range`, `high` / `low` suites),
and they are far less noisy than returns.

- **Data.** The MarketCast universe and recorder: Hyperliquid perps, 1h bars,
  with funding and spread.
- **The mandate (forced).** Each episode, the round seed draws a mandate: a
  basket of 2–8 assets, a fixed direction and weight for each (long or short),
  a target volatility σ* (10–60 % annualised), a drawdown limit, and a
  tail-loss budget. The harness cannot change the basket, the directions or
  the weights. A seeded schedule may also change the mandate mid-episode
  (a "client flow"), which forces a rebalance.
- **What the harness controls.** Each hour it sets:
  * **size:** a gross scale `s_t ∈ [s_floor, s_cap]` (for example
    [0.25, 2.0]) on the mandate book. It must stay invested: average
    utilisation below `min_util` (for example 0.6) fails the episode;
  * **hedge:** an optional position in listed hedge instruments (BTC, ETH
    perps), `|h| ≤ 0.5 ×` gross;
  * **execution:** how fast to move to the new size (a partial fill rate), to
    trade cost against tracking;
  * **risk declaration:** its 1-hour-ahead VaR and ES of the book at 97.5 %.
- **Costs.** A taker fee, spread-plus-impact slippage from the recorded
  `spread` / volume, and funding.
- **Episode.** 7 days of hourly steps, from a random start.
- **Score terms** (per episode, each a loss; lower is better):

  | term | loss | what it rewards |
  |---|---|---|
  | vol tracking | `(log(σ_realised / σ*))²` | sizing to the risk target, both over and under |
  | tail control | drawdown beyond the limit, plus loss beyond the tail budget | cutting size before stress, not after |
  | risk forecast | Fissler–Ziegel FZ0 loss of the declared (VaR, ES) on the realised hourly book return | honest, calibrated risk numbers (a proper score, so over-wide and over-tight both lose) |
  | cost | turnover, slippage and funding, in bps of average gross | not churning |

  Each term is summed over the verdict's episodes and divided by the
  reference harness's sum on the same episodes (paired, cluster bootstrap by
  start day). The composite gain is the weighted geometric mean of those
  ratios, as in MarketCast. A pass needs a composite `gain LCB ≥
  gain_margin` and **no term confidently worse** than the reference
  (`LCB ≥ −tau`). Without that guard, a harness could win on costs by never
  rebalancing.
- **Diagnostics only (not scored).** Book PnL, Sharpe and hit rate are
  published on each row. There is no reward for them, so there is no reason
  to chase direction.
- **Why it is less noisy.** Realised vol over 168 hourly returns has a
  relative standard error of about 5 %, while the mean return over a week is
  mostly noise. The FZ0 loss scores every hour, not only the breach hours.
  The Q5 noise study still sets the episode count.
- **Gaming checks.**
  * Going flat fails the utilisation floor.
  * Declaring a huge ES loses on FZ0.
  * A hedge that cancels the whole book is capped at 0.5 × gross, and
    under-shooting σ* costs on tracking.
  * Timing size to the mandate's direction can add PnL, but PnL is not
    scored.
- **Reference harness.** Size `s_t = σ* / σ̂_t`, with σ̂ taken from the
  forecaster's quantile spread for the book (P90 − P10 over horizon), clipped
  to bounds. No hedge. A fixed partial fill rate. VaR and ES are read off the
  same forecast quantiles.
- **Forecaster.** The `mkt:market` slot holder (the MarketCast fine-tune), or
  the pinned Timestral-28M base before the first holder.

It is a risk-management score, not a profit claim, and the docs must say so.

Later candidates: `act:hedge` (fuel / power hedging ratio), `act:capacity`
(cloud autoscaling on `domain:web_cloudops` series), `act:staffing`.

## 9. Rewarding creative use of forecasts

The aim is to find new ways to turn our models' forecasts into decisions, not
small tweaks of one winning harness. Nothing here judges creativity directly.
A human or LLM judge would be subjective, non-deterministic and gameable.
Instead, the rules pay for outcomes that only new uses of the forecasts can
reach, and give harnesses room to find them.

### 1. An open forecast API, with a budget instead of a fixed call pattern

What a harness may ask the model is wide open. The limit is a per-episode
budget of forecast series (for example 2,000 series × horizon steps), not a
fixed number of calls per step. Spending it well is part of the game.

- **Any context it builds.** The harness may forecast any series it derives
  from the past: returns, spreads, the mandate book's value, a synthetic
  basket, resampled bars, residuals from its own model, a detrended or
  deseasonalised series.
- **Any of the 21 quantile levels, any horizon** up to the model's maximum,
  and **covariates it constructs**: its own calendar features, or another
  series as a cross channel.
- **What-if queries.** A harness may edit a context to ask "what if?"
  (append a shock, roll a P50 or P90 forward as a scenario path, change a
  covariate). These are ordinary calls against the budget. The docs must say
  that marginal quantiles are not paths, and that rolled scenarios are the
  harness's own construction.
- **Event-driven spending.** A harness can save budget in quiet hours and
  spend it in stress hours, or re-forecast only when its own trigger fires.
- **Cross-vertical models.** `harness.json` may list other verticals'
  fine-tunes it also calls, for example `act:bess` asking `mkt:market` for
  gas or power-adjacent markets. The main model must still be used (the
  model-lift gate). Other models are extra inputs and count against the same
  budget.
- **Batching.** One call can forecast many contexts. The budget counts series,
  not calls, so batching is free.

Every request and reply still goes in the transcript, so audit replay is
unchanged.

### 2. A novel-use slot per scenario

Each `act:<scenario>` has two slots:

| slot | who takes it | share of the scenario's payout |
|---|---|---|
| main | best confirmed qualifier (the current rule) | 1 − `novel_share` |
| novel-use | best confirmed qualifier that wins *differently* from the main holder | `novel_share` (for example 0.3) |

**"Differently"** is measured, not judged. Take the per-episode gains (over
the reference) of the candidate and of the main holder, on the same
episodes. Their correlation must be at most `novel_max_corr` (for example
0.5). The candidate must also pass the full gate on its own: gain LCB,
model lift, and no term confidently worse.

This is the DEC-CA-0015 idea (pay for decorrelated errors), applied to
decisions. A harness that does what the leader does a bit better still
fights for the main slot. A harness that wins on different episodes (in
stress, around events, on other assets or regions) can earn without beating
the leader head to head. When the main slot changes hands, the novel-use
holder is re-checked against the new main holder at its next confirm. If it
is now too similar, it loses the novel slot but can contest the main one.

### 3. Forecast value as a scored term

The model-lift gate already makes a harness beat its own naive-forecast run.
On top of that, each verdict records the harness's **forecast value**: its
score with the real model minus its score with the naive forecaster. That
lift is compared with the reference harness's own lift.

- **Novel-use slot:** forecast value is a tie-break and a gate (it must be at
  least the reference's). This means "different" cannot come from ignoring
  the model.
- **Main slot:** forecast value is published but not scored at first. Q8
  covers adding it to the composite.

### 4. Variants that stop hard-coding

Each round's seed draws scenario variants that the harness only learns at
`reset(spec)`: cost levels, limits, battery size and efficiency, lead times,
asset baskets, σ* and drawdown limits, which regions or SKUs. A harness tuned
to one setting loses across the draw. One that reads the spec and the
forecasts carries over.

### 5. Hard tiers where simple use is not enough

Part of every verdict's episodes come from tiers where reading the median is
not enough:

- **perp:** scheduled events (FOMC, CPI), mandate changes mid-episode, and
  stress windows sampled more often;
- **bess:** price-spike days, and a day-ahead commitment plus real-time
  dispatch, which needs both short and long horizons;
- **inventory:** long lead times, shared capacity, and new SKUs with short
  history.

The reference harness is weak on these tiers on purpose, so that is where
the room for new ideas is.

### 6. Open source, without paying copies

Harnesses are published **after** their reign, not during it (§11). A
published harness then becomes part of the bar every new entry must beat,
so copying it earns nothing. Building on a published idea is fine, but
winning still needs a confident gain over the holder and over everything
published.

## 10. Explaining actions to the customer

**Owner (2026-11-24):** explaining decisions to the customer, factually, is
part of what the software does. It is not optional and it is not a separate
slot. Every ActionCast entry must explain its actions, and the explanations
are scored.

"Factual" has a precise meaning here: **every statement is tied to evidence
in the transcript and can be checked by machine.** An LLM may write the
words, but it never grades them. The harness is deterministic and can be
replayed, so an explanation can be tested against what the harness actually
does.

### What the harness returns

`step` returns an action and, for every **material action**, an
`Explanation`. A material action is an action that changes by more than the
scenario's threshold, a limit that binds, or a mandate change.
`episode_report` returns a customer summary at the end of each episode.

```python
return Decision(
    action=self.plan(q, obs.soc),
    explanation=Explanation(
        claims=[
            ex.forecast("c1", q, series="price.NSW1", level=0.9, horizon="16:00-18:00"),
            ex.observation("c2", obs, "soc", value=0.42),
            ex.constraint("c3", "soc_max", binding=False),
            ex.counterfactual("c4", input="forecast.price.NSW1.p90", change=-0.3,
                              expect="discharge_mw <= 0.2"),
        ],
        text="Holding charge for the evening peak: there is a 10% chance the price "
             "tops $300/MWh between 16:00 and 18:00 [c1], and the battery is at 42% [c2]. "
             "If that risk were 30% lower, it would sell now instead [c4].",
    ),
)
```

| claim type | what it states | how it is checked |
|---|---|---|
| `observation` | an input's value at the decision time | equals the observation |
| `forecast` | a model's quantile or interval, with model, level and horizon | equals the recorded reply for that call |
| `constraint` | a limit, and whether it binds | equals the simulator's state |
| `cost` | a fee, slippage or penalty incurred or avoided | equals the simulator's cost ledger |
| `counterfactual` | "if input X changed by d, the action would satisfy Y" | the validator re-runs the harness with X changed (below) |
| `driver` | "this action depends mostly on these inputs" | matches measured sensitivity (below) |
| `outcome` | what then happened (episode reports only) | equals the realised data |

### Checks

1. **Grounding (no LLM).** Every sentence of `text` must cite at least one
   claim, `[c1]`. Every number in a sentence must match a number in a cited
   claim, after unit and rounding rules. Uncited sentences and unmatched
   numbers count as **unsupported**.
2. **Truth (no LLM).** Every claim is checked against the transcript,
   simulator state or realised data, by the table above. A wrong claim is
   **false**.
3. **Counterfactual faithfulness (no LLM).** The validator replays the
   harness with the claimed change, and also with changes it samples
   itself, which the miner cannot choose. Claimed counterfactuals must hold.
   Sampled ones measure how sensitive the action really is to each input.
4. **Driver coverage (no LLM).** The listed drivers must account for at least
   `driver_coverage` (for example 70 %) of the measured sensitivity.
   "I used the forecast" earns nothing. The real reasons must be named.
5. **Simulatability (Engy LLM as reader, not judge).** A pinned Engy model
   reads the explanations only, never the code, and then predicts the
   harness's actions on held-out states from the same episodes. The score is
   prediction accuracy against the harness's real actions. Padding doesn't
   help. Text is length-capped and stripped of instructions before the
   reader sees it, and the reader only outputs a prediction.
6. **Plain language (diagnostic only).** A readability measure and an
   optional pinned-LLM pairwise preference against the reference
   explanation are published but not paid. A tone or style rule can be
   promoted to a gate later, once there is evidence it cannot be gamed.

**Forecasts must be stated as forecasts.** A sentence about the future must
cite a `forecast` claim, and its wording must carry the probability ("a 10%
chance…", "likely between…"). The grounding checker refuses future-tense
certainty ("the price will…") with no probability attached.

### Scoring

- **Factual gate.** False claims and unsupported sentences must be under
  `max_false_rate` (for example 0.5 % of claims) and `max_unsupported_rate`
  (for example 1 % of sentences), at their upper confidence bounds. Failing
  either fails the entry, whatever its decision gain.
- **Explanation score.** A composite of counterfactual accuracy, driver
  coverage and simulatability, compared with the reference harness's
  template explanations on the same episodes. It enters the verdict through
  the "no term worse" guard (§4): an entry cannot pass by explaining worse
  than the reference. Q11 asks whether it should also add to the composite
  gain.
- **Cost.** Explanations count against the Engy token budget when an LLM
  writes them. Template explanations, built from kit builders, cost nothing.

### What the customer gets

- A plain-language reason for every material action, each sentence
  footnoted to its evidence (the forecast, the reading, the limit).
- A "what would change my mind" line (the counterfactual) that has been
  tested against the software itself.
- An episode report: what was done, why, and how it turned out against the
  forecast, with the same footnotes.
- The evidence trail (transcript references), so an auditor or the
  customer's own team can check any statement.

### Kit support (`cascade.harness.explain`, layer 5)

- **Claim builders.** `ex.forecast`, `ex.observation`, `ex.constraint`,
  `ex.cost`, `ex.counterfactual`, `ex.driver`. Each one captures its
  evidence reference from the transcript automatically, so a harness can't
  cite a value it never saw.
- **Renderers.** Deterministic templates per claim type and locale (English
  in v1). An optional Engy-written `text` must still pass grounding.
- **The checker itself,** so miners run exactly the validator's checks
  locally (`cascade harness explain-check`).
- **Reference explanations.** The reference harness ships template
  explanations. They are the bar every entry must not fall below.

## 11. Incentives: king of the hill, privacy, duplicates

### The game

| | |
|---|---|
| **Players** | Miners (each can hold many hotkeys and coldkeys); the main validator (operator-run, sees everything); satellites and auditors; copycats watching what is published. |
| **What a miner can do** | Submit original work; resubmit near-copies to fish for noise; copy a published winner and nudge it; clone a winner's behaviour from its published actions or explanations; use many identities; hold a slot and stop improving. |
| **Payoff** | Emissions, by far. Status (leaderboard, being the published winner) is a real second. |
| **Costs** | Building the harness; the CPU qualifier leg; metered Engy and decisionmodels.io calls. |
| **Information** | Code is private to the miner and the operator until publication. Scores are public. |
| **Objective** | Pay for the best decision software on each vertical, keep it improving, and never pay for a copy. |

### Recommendation: king of the hill with a paired increment, plus a novel-use slot

**Yes, king of the hill, with four changes.**

1. **Winner-take-the-slot, not a split by rank.** A rank split pays copies
   and sybils: ten near-clones collect ten shares. With one slot holder per
   scenario, a copy of yourself earns nothing, and a copy of someone else
   must beat them. This is the existing lane-C slot model.
2. **A paired challenge against the holder (better than lane C's rule).**
   Lane C compares LCBs recorded at different settlements, because
   re-training a fine-tune is costly (DEC-CA-0056). A harness is cheap to
   re-run: it is CPU code on cached forecasts. So at each verdict the
   holder **and every published harness** run on the challenger's episodes,
   as extra references. To take the slot, a challenger must beat the
   *strongest* of these, paired, by `increment_margin` (for example 1 %) at
   the LCB. Noise alone does not flip a slot, and a nudged copy of the
   holder cannot clear 1 % on the same episodes.
3. **A decaying reign.** The gain-tiered full-pay period (lane C's
   `full_pay_tiers`) then halves down to a floor, with a 28-era TTL. A
   holder that stops improving earns less over time, and the slot opens up.
4. **A second slot for a different way to win** (the novel-use slot, §9):
   it rewards a different approach, so the field is not just tweaks of one
   leader. It is the main defence against "the leader is unbeatable, so
   nobody plays".

**Alternatives considered**

| Alternative | Why not |
|---|---|
| Pay top-k by rank | Pays copies and sybils; rewards being close, not better. |
| Shapley or ensemble-contribution payouts | Gameable: miners craft entries that look additive. Hard to audit. Expensive. |
| Bounties (fixed prize for beating a bar once) | Good for launch. Could seed a new scenario's first holder, then hand over to king of the hill. |

### Privacy: private until it earns, published when it stops

The DEC-CA-0036 vault pattern, applied to ActionCast:

- **Private intake.** The miner uploads the harness ZIP to the operator
  vault. On chain they commit only the digest:
  `metro-v1:ft:act:<scenario>:hippius:vault/direct@sha256:<hex>`. The
  earliest uploader owns a digest, and a copied digest committed by another
  hotkey is dropped.
- **What is public during evaluation:** the commitment, the verdict rows
  (scores, LCBs, gate outcomes), and the digests of the transcripts and the
  report. **Not public:** the code, the per-step actions or the explanation
  texts. Actions and explanations would let anyone train a clone by
  imitation.
- **When the harness is published** (`[actioncast] publish`, proposed
  `"dethrone"`): when the holder loses the slot, or when its reign decays to
  the floor, whichever comes first. Then the code, transcripts, actions and
  explanations all become public. Losers are never published.
- **What publication does:** the published harness joins the **reference
  set** (item 2 above), so every later entry must beat it by the increment
  margin. Copying it earns nothing.
- **Audit, stated plainly.** While a harness reigns, its verdicts are
  verifiable only against the main validator's signed reports and digests.
  Once it is published, anyone can replay the whole reign from the
  transcripts, and the digests prove nothing changed. Every paid reign is
  fully auditable, just later. Non-winners can only be audited by trusting
  the operator. This is the same posture as DEC-CA-0036, and the price of
  stopping copy-watching.
- **Customer explanations** are a product output. They go to the customer
  (Ephemeris or apps), not to the public, during the reign.

### Duplicates and copies

Each layer catches what the previous one misses:

| Layer | Catches | Rule |
|---|---|---|
| 1. Digest ownership | byte copies | the earliest uploader owns a digest |
| 2. Structural hash | renamed or reformatted copies | normalised AST hash (comments, whitespace and identifiers stripped); matching a senior entry or a published harness → duplicate |
| 3. Behavioural near-duplicate | refactored copies, wrappers | action correlation ≥ 0.98 (and explanation-claim overlap) with a senior entry, the holder or a published harness, on the same episodes |
| 4. Published-set reference | "copy the published winner plus a tweak" | must beat every published harness by `increment_margin`, paired (item 2) |
| 5. Fishing limits | many near-variants hoping noise passes one | owner-weighted Bonferroni (MarketCast F6: one coldkey's entries share one owner's α), one open entry per coldkey, a paid leg per entry, and confirmation on fresh episodes |

Seniority is always the commitment block, never the UID.

### How each type of player fares

- **Honest builder:** submits privately. If better by 1 % against the
  holder and the published set, it takes the slot, earns while private, and
  is published when dethroned. Best response: keep improving, because the
  reign decays.
- **Copycat:** can only copy published harnesses. Those are in the
  reference set, so a copy earns nothing; even a good tweak must clear the
  margin. It can't see the live holder's code, actions or explanations.
- **Noise fisher:** many coldkeys, small variants. Each entry pays a leg and
  must pass at α / k, then pass a fresh confirmation and the paired margin
  against the holder. The false-pass rate under the null is about α × α, so
  expected value is negative once the fee is above that rate times the
  slot's value. Q5's noise study sets the fee floor.
- **Lazy player:** ships the reference harness plus a tweak. The reference
  is the bar, so it earns nothing without a real 1 % gain.
- **Incumbent who coasts:** the reign decays and the slot opens. The incumbent
  can defend only by submitting a better version, which is fine.
- **Insider risk:** the operator sees all code. The mitigations are vault
  access logs, per-dispatch staging (only the entry being judged is on the
  pod), and full publication of every reign later. This is trust, stated as
  trust.

### Participation

Winner-take-all can deter entrants. The counterweights:
- **More slots:** a main and a novel-use slot per scenario, across many
  scenarios and verticals, and possibly generalist slots (Q9).
- **Cheap entry:** a CPU leg, with external calls metered by use.
- **Local checking:** miners can backtest and run the explanation checker
  locally, so they know their odds before paying.
- **Status:** a public leaderboard of verdict scores for every entry,
  showing no code.
- **Launch bounties:** a bounty can seed each new scenario's first holder.

### What to monitor

- **Slot concentration** by coldkey, and how long it takes to dethrone a
  holder.
- **Near-duplicate distances** per round. A spike just under 0.98 means
  tuned evasion; tighten the threshold.
- **Pass rate of the owner's null entries:** the reference plus
  noise-only changes. It must stay near zero.
- **Entries per owner, and new coldkeys per round.**
- **Clones soon after a publication:** gain against the published harness
  clustered just above the margin.
- **Holders whose explanation scores fall** while their decision gains rise
  (trading explanations away).

## 12. Submission

The miner uploads the tree to the private vault and commits
`metro-v1:ft:act:<scenario>:hippius:vault/direct@sha256:<digest>` (the
existing lane-C form, with the DEC-CA-0036 vault ref; §11). The tree:

```
harness.json      # {"scenario": "bess", "api": 1, "models": ["nrg:aemo", "mkt:market"], ...}
policy.py         # class Harness(HarnessBase): reset(spec) / step(obs, fc) -> Decision(action, explanation) / episode_report()
lib/…             # optional pure-Python helpers, size-capped
```

```python
class Harness(HarnessBase):
    def reset(self, spec):            # scenario constants, asset limits, seed
        ...
    def step(self, obs, fc):          # obs: history to now (past only)
        q = fc.forecast(obs.prices, horizon=288, levels=(0.1, 0.5, 0.9))
        if self.spike_risk(q):        # spend budget only when it matters
            shocked = self.what_if(obs.prices, shock=+3.0)
            q2 = fc.forecast([obs.prices, shocked], horizon=288, levels=fc.LEVELS)
        return self.plan(q, obs.soc)  # action, validated and clipped by the sim
```

`fc.forecast(contexts, horizon, levels, covariates=…, model=…)` is the only
way to reach a model. It returns quantiles from the vertical's fine-tuned
model, or from another model listed in `harness.json`. Calls are batched, and
they draw on a per-episode budget of forecast series × horizon steps, with a
context-length cap. `fc.budget_left()` reports what remains. Over budget
means the call is refused, not that the episode fails.

## 13. Rules

- **Sandbox.** The same static guard and network-isolated sandbox as
  generators (`cascade/interface/static_guard.py`): no network, no clock, no
  filesystem outside the tree, single-threaded BLAS, numpy / scipy only,
  and a per-step CPU-time limit (for example 200 ms; over the limit = a
  no-op action plus a penalty). Code is capped (for example 2 MB). The only
  way out of the sandbox is the model gateway (forecasts, Engy,
  decisionmodels.io).
- **Determinism.** Same observations and model replies (forecasts, Engy
  replies, decision-model probabilities) give the same actions and the same
  explanations, byte for byte. The validator records every forecast request and
  reply in a per-episode transcript (hashed into the receipt row). The
  auditor replays the harness on CPU from the transcript, without a GPU.
  This is the DEC-CA-0064 B3 pattern: you can check a verdict without
  trusting the main validator.
- **Forward-only.** Episodes start after the leg's `train_block`, plus an
  embargo. The harness can only see the past inside the episode.
- **Duplicates.** Five layers (§11): digest ownership, a structural (AST)
  hash, behavioural near-duplicates (action correlation ≥ 0.98, against
  senior entries, the holder and published harnesses), the published-set
  reference, and fishing limits. Seniority goes to the earlier commitment
  block.
- **Privacy.** Submissions go through the private vault. Code, actions and
  explanation texts stay private until the harness is dethroned or its reign
  decays to the floor (§11).
- **Model lift.** See §4.

## 14. Implementation map

| piece | where | what |
|---|---|---|
| kind | `cascade/shared/config/lane_c.py` | `register_competition_kind("act", eval="paired episode decision score", forward_only=True, parse_value=check_scenario)` |
| kit | `cascade/harness/` (new: `data`, `forecast`, `decide`, `env`) | the primitive set, with golden-vector tests per primitive; published as a standalone package for miners |
| scenarios | `cascade/harness/scenarios/<name>.toml` + `adapters/` | one spec per scenario, plus its data adapter and forward recorder |
| model gateway | `cascade/validator/harness_gateway.py` (new) | Engy and decisionmodels.io clients: registry pinning, budgets, request-hash cache, transcript (with Engy proofs), transient handling, per-entry metering |
| agent tools | `cascade/harness/agent.py` | kit primitives as tool schemas for LLM tool use |
| sandbox runner | `cascade/eval/harness/runner.py` | runs `Harness` in the sandbox, proxies `fc.forecast`, writes the transcript |
| trainer half | `cascade/trainer/lane_c_competitions.py` | `HarnessTrainer`: `read_recipe` checks the tree. `train_leg` is a short CPU leg (DEC-CA-0053 CPU tier, on the miner's key): static guard, a smoke run, and a **historical qualifier** (it must beat the reference on public backtest episodes, so junk is filtered before it costs validator GPU). It then freezes the pointer. `leg_hours` ≈ 0.25 |
| validator half | `cascade/validator/lane_c_competitions.py` | `HarnessValidator.judge`: on the main validator, for entries whose forward window has closed, it runs N episodes × {miner, reference, naive-model miner} through `MktSession.forecaster(...)` (reusing the pod via `MktLease(owner="act")`), scores them and emits `LaneCJudgement` rows. Uses `generic_fold = True` unless the forecaster-change rule needs state |
| reports | same | main-validator per-round report (`report_key`), as MarketCast does, so satellites need no GPU |
| explanations | `cascade/harness/explain.py` + `cascade/eval/harness/explain_check.py` | claim builders, renderers, grounding / truth / counterfactual / driver checks, the simulatability reader |
| audit | `cascade/audit/harness_reproduce.py` + a branch in `checks.py` | replays transcripts on CPU and re-derives scores and LCBs |
| docs | `docs/ACTIONCAST_MINER.md` | the miner guide, in the MARKETCAST.md shape |

## 15. Config sketch

```toml
[lane_c]
targets = ["mkt:market", "act:bess", "act:inventory"]

[lane_c.competitions.act]
weight      = 0.5
from_block  = 0
until_block = 0

[actioncast]                    # sibling table, like [marketcast] (overlay allow-list)
episodes_per_verdict = 400
verdict_wait_blocks  = 21600    # ~3 days of forward data
gain_margin          = 0.01
model_lift_margin    = 0.0
forecast_budget      = 2000      # series x horizon/100 per episode (unit to fix)
novel_share          = 0.3
novel_max_corr       = 0.5
hard_tier_fraction   = 0.3
increment_margin     = 0.01     # paired gain over holder and published set
publish              = "dethrone"  # dethrone | floor | crown | delay:<eras>
max_open_per_coldkey = 1
ast_dedup            = true
near_dup_action_corr = 0.98
step_cpu_ms          = 200
tree_max_bytes       = 2_000_000

[actioncast.explain]
max_false_rate       = 0.005    # claims, at the upper bound
max_unsupported_rate = 0.01     # sentences, at the upper bound
driver_coverage      = 0.7
counterfactual_samples = 4      # validator-sampled per material action
text_max_words       = 80
reader_model         = "<engy-model-id>@<weights-hash>"

[actioncast.providers.engy]         # LLMs; cutoff must precede from_block
models             = ["<engy-model-id>@<weights-hash>"]   # to choose (Q10)
decoding           = "greedy"
tools              = "gateway-only"   # no provider-side web or browse tools
tokens_per_episode = 20000
require_proof      = true

[actioncast.providers.decisionmodels]
models               = ["<model>@<version>"]   # e.g. s1-fast, to confirm
calls_per_episode    = 500
spot_check_fraction  = 0.02
prob_tolerance       = 1e-6

[actioncast.bess]
model      = "nrg:aemo"   # paired fine-tune target; its holder (or pinned base) is the forecaster
regions    = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
episode_h  = 24
```

## 16. Rollout

0. Build the kit first (layers 1–4, golden vectors), and write the three
   first scenarios as specs on it. If a scenario needs a one-off module, the
   kit is missing a primitive.
   Build the model gateway and the transcript format in the same step: the
   forecast path first, then the Engy and decisionmodels.io clients once
   their API facts are confirmed (Q10).
1. Build `act:perp` plumbing on `mkt:market` first: it is the only vertical
   whose fine-tune exists, so the forecaster pairing can be tested end to
   end. In parallel, build the `nrg:` fine-tune and `act:bess` together, since
   they share the AEMO recorder. Run the Q5 noise study offline for each
   scenario.
2. Testnet with the reference harness and two or three variants written by
   us (good, bad, model-blind), to check that the verdicts rank them
   correctly.
3. Ship the release, then arm each `act:` scenario at an era boundary, after
   its fine-tune target and its own noise study. With risk scoring, `act:perp`
   can arm first, because `mkt:market` already exists.

