---
id: DEC-CA-0073
type: decision
title: "Lane C harness competition (`app:<scenario>`): miners submit decision software that calls our fine-tuned model for the vertical; judged forward-only on simulated real-data episodes, paired against a reference harness on the same model"
status: draft
date: 2026-11-24
tags: [lane-c, competitions, harness, applications, v2]
revisit_when: "the owner settles the open questions below; or a scenario cannot supply enough forward episodes per settlement; or slots stay unpaid for weeks because decision-score noise swamps the margin"
relations: {depends_on: [DEC-CA-0056, DEC-CA-0064, DEC-CA-0063], relates_to: [DEC-CA-0053, DEC-CA-0069]}
---

> **Draft for owner review. Nothing here is built or armed.** It is a design
> that fits the lane-C plugin interface (`docs/LANE_C_COMPETITIONS.md`). Every
> number is a proposed default. The open questions are at the end.

## Problem

Lane C pays for a better **model** (MarketCast fine-tunes of Timestral-28M,
judged on CRPS). Nobody is paid to turn a forecast into a **decision**: how
much to trade, when to charge a battery, how much stock to order. That step
is where a forecast earns its value, and most of it is not forecasting. It
covers sizing on uncertainty, costs, constraints, risk and what to do when
the forecast is wrong.

A harness competition pays miners to write that software. It also tests the
fine-tuned models with a decision-based measure, not only CRPS.

## Decision (proposed)

1. **What miners submit.** A *harness*: deterministic Python policy code plus
   a `harness.json` manifest, for one declared scenario. The harness gets an
   observation and a handle to a validator-hosted forecaster, and returns an
   action at each decision step. It is not trained and needs no GPU of its
   own.
2. **The model is our fine-tune for that vertical (owner, 2026-11-24).** We
   fine-tune one model per vertical in lane C. Each harness scenario is paired
   with its vertical's fine-tune target and always calls that target's
   current slot holder (see **Vertical pairing**). Miners compete on decision
   logic, not on which forecaster they load. A harness may bring its own
   *decision* models and call pinned LLMs (Decision 10), but the
   vertical's fine-tune stays the forecaster it is judged on.
3. **Scenarios are targets.** A target is `app:<scenario>`. Each scenario is a
   simulator driven by real data that is recorded **after** the harness was
   committed (forward-only, like MarketCast). Proposed first scenarios:
   `app:perp` (risk-managed trading mandate), `app:bess` (battery energy arbitrage) and
   `app:inventory` (replenishment).
4. **Judged paired against a reference harness.** Every episode is run twice,
   with the miner's harness and with the owner-shipped reference harness, on
   the same data and the same forecaster. The score is a paired, cluster-
   bootstrapped gain LCB on a per-episode, scale-free decision score. A pass
   needs `gain LCB ≥ gain_margin`. Then the DEC-CA-0056 Bonferroni, confirm
   and slot rules apply unchanged.
5. **It must use the model.** The harness is also run with the forecaster
   replaced by a naive one (seasonal-naive / empirical quantiles). It passes
   only if its real-model score beats its naive-model score with
   `LCB ≥ model_lift_margin`. This keeps the competition about *using the
   forecaster*, not about pure rule-based strategies. See open question Q2.
6. **Payout.** Its own lane-C competition weight
   (`[lane_c.competitions.app] weight`), split equally across its targets,
   with the existing gain-tiered slot schedule and TTL.
7. **Separate product.** Winning harnesses are published (open source, with
   their recorded episodes) as reference apps for the models.
8. **Creative use of forecasts is paid for (owner, 2026-11-24).** The
   forecast API is open-ended, and each scenario has a second, *novel-use*
   slot for a qualifier that wins in a different way from the main holder.
   See **Rewarding creative use of forecasts**.
9. **One primitive set for every vertical (owner, 2026-11-24).** Harnesses
   and scenarios are built from a shared, versioned kit (`cascade.harness`).
   A vertical we fine-tune for gets an app by writing a data adapter and a
   scenario spec, not a new competition. See **Primitive set**.
10. **LLMs and decision models are part of the harness stack (owner,
   2026-11-24).** A harness may call pinned open-weight LLMs hosted next to
   the forecasters, and may ship its own small decision models. All of them
   go through one model gateway: typed, budgeted and recorded. See **Models
   in the harness stack**.

## Vertical pairing

Each vertical has two lane-C competitions that feed each other:

| vertical | fine-tune target (the model) | harness target (the app) |
|---|---|---|
| markets / trading | `mkt:market` (MarketCast, live) | `app:perp` |
| energy | an energy-price fine-tune, e.g. `nrg:<market>` (to build) | `app:bess` |
| inventory / demand | a demand fine-tune, e.g. `dmd:<suite>` (to build) | `app:inventory` |

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
- **An app arms only after its vertical's fine-tune.** `app:<scenario>`
  needs its paired fine-tune target configured (load-checked). Running on the
  pinned base until the first holder is allowed.
- **Feedback to the fine-tune (later, not consensus at first).** Each verdict
  also reports the reference harness's score on the new holder against the
  old one: a decision-based check on whether a CRPS win helped in use. It is
  published as a diagnostic only. Making it a guard on fine-tune wins is a
  later decision.

## Scenarios

Each scenario fixes the observation, the action space, the cost model, the
episode length, a reference harness and (where one exists) a
perfect-foresight oracle. Where there is an oracle, the score is
oracle-normalised and so scale-free across assets, markets and days.

### `app:bess` — battery arbitrage on real-time power prices (recommended first)

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

### `app:inventory` — periodic-review replenishment

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
- **Forecaster.** The demand fine-tune's slot holder (`dmd:<suite>`), trained
  on the same forward count series.

### `app:perp` — risk-managed perp mandate (owner, 2026-11-24: not scored on trading success)

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

Later candidates: `app:hedge` (fuel / power hedging ratio), `app:capacity`
(cloud autoscaling on `domain:web_cloudops` series), `app:staffing`.

## Primitive set: one kit for every vertical

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
model     = "dmd:inflow"            # the paired fine-tune target
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
- **What it unlocks.** The same competition kind (`app`) serves every
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

Optional (Q9): a **generalist slot** `app:store` (and `app:position`),
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

## Models in the harness stack: LLMs and decision models

A harness is more than a forecast plus an optimiser. It can also reason over
text and context with an LLM, and run its own learned policy. Both are
allowed. They sit around the vertical's fine-tune, not in place of it.

```
            ┌──────────── harness (miner code, sandboxed) ────────────┐
 obs, spec, │  policy.py  ── decision models (miner weights, CPU)      │ → action
 text feed  │      │                                                   │
            │      └── model gateway: forecast | llm | (later) others  │
            └────────────────────┬─────────────────────────────────────┘
                                 │ every request and reply goes in the transcript
               vertical fine-tune (GPU pod)    pinned LLMs (GPU pod)
```

### One model gateway

Every model call goes through `fc` (renamed `models` in the kit; `fc` stays
as an alias):

| call | what | budget |
|---|---|---|
| `models.forecast(...)` | the vertical's fine-tune, or another listed fine-tune | forecast series × horizon |
| `models.llm(name, messages, tools=…, max_tokens, schema=…)` | a pinned LLM from the registry | tokens per episode, plus a cadence cap |
| `models.decide(name, inputs)` | the harness's own decision model, run in-process on CPU | CPU time per step |

The gateway is the only door out of the sandbox. It types every request
and checks it against the budgets. It writes every request and reply to the
transcript.

### LLMs

- **Pinned, open-weight, validator-hosted.** A registry (`[harness.llm]`)
  names each LLM by repo and revision, as Timestral is pinned. The LLMs run on
  the main validator's pod (a vLLM-type server next to the `mkt_worker`
  forecasters, held with the same `MktLease`). There are no external LLM APIs
  in v1:
  * they change silently;
  * they cannot be replayed or audited;
  * hosted tools such as web search could leak the future into a
    forward-only test.

  Every pinned LLM's training cutoff must be before the competition's
  `from_block`. With no tools beyond the gateway, it cannot know the episode.
- **Deterministic decoding.** Greedy, or seeded sampling from the round seed;
  fixed batch-invariant kernels on a fixed pod type; a fixed
  `max_tokens`. Replies go in the transcript, so the auditor's harness replay
  needs no GPU. Trust in the replies themselves comes from a **spot-check**:
  the auditor (or any validator with a GPU) regenerates a random, seeded
  sample of LLM calls. Any mismatch beyond a token-level tolerance flags the
  round. This is the same idea as the MarketCast re-forecast tolerance.
- **What an LLM can read.** Anything the harness passes it from the
  observation, plus scenario text feeds recorded forward like any series:
  * the event calendar, with descriptions (FOMC, CPI, grid notices,
    holidays);
  * the mandate or operating brief written as text (the variants are
    rendered as prose as well as numbers);
  * later, licensed headline feeds.

  All text is data: the simulator validates the harness's action whatever
  the LLM said.
- **Tool use.** The kit exposes its primitives as tool schemas
  (`cascade.harness.agent`): forecast, quantile and risk read-outs, the
  decision kernels, and the calibrator. An LLM can then run an agent loop
  inside a step, for example "forecast this spread, check the tail, size
  the hedge", with every tool call a gateway call against the same budgets.
  This is the "software that uses the fine-tune to make calls" in its fullest
  form.
- **Cadence.** LLM calls are expensive next to forecasts. Each scenario sets a
  cadence: for example at most one LLM step per hour of episode time for
  `app:perp`, per 2 h for `app:bess` (5-min steps), and per day for
  `app:inventory`. There is also a token budget per episode. Typical use:
  the LLM plans or reclassifies the regime at that cadence, and fast
  numeric code acts every step.
- **Later: our own LLM fine-tunes.** If lane C later fine-tunes LLMs per
  vertical, they join the registry exactly as forecasters do. The harness
  calls the slot holder, under the same pairing and change rules.

### Decision models (miner-supplied)

- **Allowed.** Small learned models a harness ships in its tree: an RL
  policy, a regime classifier, gradient-boosted trees, a small net. The cap
  is for example 50 MB in total. Formats are `safetensors` or `.npz` run by
  kit code, or ONNX with `onnxruntime` pinned and single-threaded on CPU. No
  pickles.
- **Trained offline on history.** The kit's environments run in backtest mode
  on public recorded data (`cascade harness backtest`). Miners can train
  policies against the same simulator, costs and variants that judge them.
  The forward-only rule still holds, because judging is on episodes after
  commit.
- **Not a second forecaster.** A decision model could hide its own price
  forecaster. Three rules limit that:
  * the model-lift gate (it must beat its own naive-forecast run);
  * the novel-use forecast-value gate;
  * a published **attribution** row: the score with the vertical fine-tune
    against the score with forecasts zeroed to naive, with the decision
    model held fixed.

  A harness whose gain does not depend on the fine-tune does not pass.

### Who pays

Forecasts stay on the main validator, as in MarketCast. LLM tokens are
metered per entry. Q10 asks whether the miner pays for them: a metered
charge on the funded key, or an LLM pod rented on the miner's key for that
entry's eval. An entry with no LLM calls costs the same as before.

## Rewarding creative use of forecasts

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
  fine-tunes it also calls, for example `app:bess` asking `mkt:market` for
  gas or power-adjacent markets. The main model must still be used (the
  model-lift gate). Other models are extra inputs and count against the same
  budget.
- **Batching.** One call can forecast many contexts. The budget counts series,
  not calls, so batching is free.

Every request and reply still goes in the transcript, so audit replay is
unchanged.

### 2. A novel-use slot per scenario

Each `app:<scenario>` has two slots:

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

Winning harnesses are published with their transcripts, so the field learns
from each other. Copies are not paid: the exact-digest rule, the
behavioural near-duplicate rule (action correlation ≥ 0.98) and the novel-use
correlation rule all go to the earlier commitment. Building on a published
idea is fine. Winning still needs a confident gain over the current holder,
or a different way of winning.

## Submission

The miner commits `metro-v1:ft:app:<scenario>:hippius:<repo>@sha256:<digest>`
(the existing lane-C form, parsed generically). The tree:

```
harness.json      # {"scenario": "bess", "api": 1, "models": ["nrg:aemo", "mkt:market"], ...}
policy.py         # class Harness(HarnessBase): reset(spec) / step(obs, fc) -> action
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

## Rules

- **Sandbox.** The same static guard and network-isolated sandbox as
  generators (`cascade/interface/static_guard.py`): no network, no clock, no
  filesystem outside the tree, single-threaded BLAS, numpy / scipy (plus a
  pinned `onnxruntime` for decision models) only, and a per-step CPU-time
  limit (for example 200 ms; over the limit = a no-op action plus a
  penalty). Code is capped (for example 2 MB). Decision-model weights are
  capped (for example 50 MB, safetensors / npz / ONNX, no pickles). The
  only way out of the sandbox is the model gateway.
- **Determinism.** Same observations and model replies (forecasts and LLM
  outputs) give the same actions, byte for byte. The validator records every forecast request and
  reply in a per-episode transcript (hashed into the receipt row). The
  auditor replays the harness on CPU from the transcript, without a GPU.
  This is the DEC-CA-0064 B3 pattern: you can check a verdict without
  trusting the main validator.
- **Forward-only.** Episodes start after the leg's `train_block`, plus an
  embargo. The harness can only see the past inside the episode.
- **Duplicates.** An exact tree digest is a duplicate. So is a behavioural
  near-duplicate: action correlation ≥ 0.98 with an earlier entry on the
  same episodes (the MarketCast F4 approach). Seniority goes to the earlier
  commitment block.
- **Model lift.** See Decision 5.

## Plugin mapping (code to write)

| piece | where | what |
|---|---|---|
| kind | `cascade/shared/config/lane_c.py` | `register_competition_kind("app", eval="paired episode decision score", forward_only=True, parse_value=check_scenario)` |
| kit | `cascade/harness/` (new: `data`, `forecast`, `decide`, `env`) | the primitive set, with golden-vector tests per primitive; published as a standalone package for miners |
| scenarios | `cascade/harness/scenarios/<name>.toml` + `adapters/` | one spec per scenario, plus its data adapter and forward recorder |
| LLM serving | `cascade/validator/llm_worker.py` (new) | pinned LLM registry, deterministic decoding, token metering; on the `mkt` pod via `MktLease(owner="app")` |
| agent tools | `cascade/harness/agent.py` | kit primitives as tool schemas for LLM tool use |
| sandbox runner | `cascade/eval/harness/runner.py` | runs `Harness` in the sandbox, proxies `fc.forecast`, writes the transcript |
| trainer half | `cascade/trainer/lane_c_competitions.py` | `HarnessTrainer`: `read_recipe` checks the tree. `train_leg` is a short CPU leg (DEC-CA-0053 CPU tier, on the miner's key): static guard, a smoke run, and a **historical qualifier** (it must beat the reference on public backtest episodes, so junk is filtered before it costs validator GPU). It then freezes the pointer. `leg_hours` ≈ 0.25 |
| validator half | `cascade/validator/lane_c_competitions.py` | `HarnessValidator.judge`: on the main validator, for entries whose forward window has closed, it runs N episodes × {miner, reference, naive-model miner} through `MktSession.forecaster(...)` (reusing the pod via `MktLease(owner="app")`), scores them and emits `LaneCJudgement` rows. Uses `generic_fold = True` unless the forecaster-change rule needs state |
| reports | same | main-validator per-round report (`report_key`), as MarketCast does, so satellites need no GPU |
| audit | `cascade/audit/harness_reproduce.py` + a branch in `checks.py` | replays transcripts on CPU and re-derives scores and LCBs |
| docs | `docs/HARNESS.md` | the miner guide, in the MARKETCAST.md shape |

## Config sketch (inert)

```toml
[lane_c]
targets = ["mkt:market", "app:bess", "app:inventory"]

[lane_c.competitions.app]
weight      = 0.5
from_block  = 0
until_block = 0

[harness]                       # sibling table, like [marketcast] (overlay allow-list)
episodes_per_verdict = 400
verdict_wait_blocks  = 21600    # ~3 days of forward data
gain_margin          = 0.01
model_lift_margin    = 0.0
forecast_budget      = 2000      # series x horizon/100 per episode (unit to fix)
novel_share          = 0.3
novel_max_corr       = 0.5
hard_tier_fraction   = 0.3
step_cpu_ms          = 200
tree_max_bytes       = 2_000_000
weights_max_bytes    = 50_000_000

[harness.llm]                    # pinned registry; cutoff must precede from_block
models       = ["<open-weight-llm>@<rev>"]   # to choose (Q10)
decoding     = "greedy"
tokens_per_episode = 20000
spot_check_fraction = 0.02

[harness.bess]
model      = "nrg:aemo"   # paired fine-tune target; its holder (or pinned base) is the forecaster
regions    = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
episode_h  = 24
```

## Open (owner to answer)

- **Q1. Forecaster change.** Proposed in **Vertical pairing**: the holder
  keeps its slot, and its next confirm runs on the new model. Confirm this,
  or choose to re-score every harness holder at each fine-tune change.
- **Q2. Model-lift gate.** Should a harness be *required* to beat its own
  naive-forecast run? Requiring it keeps the competition about the model.
  Dropping it widens the field to any good policy.
- **Q3. Score per scenario.** Use the oracle-normalised capture or regret
  ratio, or the raw cost ratio? For `app:perp`: the term weights, σ* range,
  `min_util`, and whether execution / client-flow steps are in v1 or later.
- **Q4. The energy and demand fine-tunes.** Resolved in principle: each
  vertical uses its own fine-tune (owner, 2026-11-24). Still open: the pinned
  base and suites for `nrg:` and `dmd:` (Timestral-28M again?), and whether
  each app scenario arms with its fine-tune or one era after it.
- **Q5. Noise budget.** Before any `from_block`, run the reference harness
  against small perturbations of itself on recorded data. Measure how many
  episodes a 1 % gain needs to pass at α/k. This decides
  `episodes_per_verdict`. For `app:perp`, measure each score term's noise
  separately; a term too noisy to move the composite is dropped or
  down-weighted.
- **Q6. Who pays for eval.** Should the main validator pay for forward
  episodes, as MarketCast forecasting does, with the miner's CPU leg
  qualifier as the spam filter? Or should the miner also fund eval
  forecasts?
- **Q7. Weight.** Should the app competition take part of lane C's 25 %, and
  how much (0.5 in the sketch above, against MarketCast's 1.0)?

- **Q8. Creativity terms.** Agree the novel-use slot (`novel_share`,
  `novel_max_corr`). Should forecast value also count in the main composite,
  or stay a novel-use gate? And how large should the forecast budget be? Too
  small blocks new ideas; too large makes the main validator's GPU cost
  unpredictable.

- **Q9. Generalist slots.** Should there be `app:store` / `app:position`
  slots scored across every scenario of a template, including one added after
  commit? And what weight should they get against the per-scenario slots?

- **Q10. LLMs.** Which open-weight LLMs go in the registry, and at what
  size? Size sets the pod and the cost. Does the miner pay for LLM tokens
  (a metered charge, or an eval pod on their key)? Should external LLM APIs
  stay out for good, or come later as unscored and unaudited extras? Should
  miners be allowed to ship a LoRA adapter for a pinned LLM, under the
  weights cap?

## Rollout

0. Build the kit first (layers 1–4, golden vectors), and write the three
   first scenarios as specs on it. If a scenario needs a one-off module, the
   kit is missing a primitive.
   Build the model gateway and the transcript format in the same step: the
   forecast path first, the LLM path once Q10 is settled.
1. Build `app:perp` plumbing on `mkt:market` first: it is the only vertical
   whose fine-tune exists, so the forecaster pairing can be tested end to
   end. In parallel, build the `nrg:` fine-tune and `app:bess` together, since
   they share the AEMO recorder. Run the Q5 noise study offline for each
   scenario.
2. Testnet with the reference harness and two or three variants written by
   us (good, bad, model-blind), to check that the verdicts rank them
   correctly.
3. Ship the release, then arm each `app:` scenario at an era boundary, after
   its fine-tune target and its own noise study. With risk scoring, `app:perp`
   can arm first, because `mkt:market` already exists.
