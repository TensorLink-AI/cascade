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
   logic, not on which weights they load. A harness may not ship weights or a
   model of its own beyond a size cap (see Rules).
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

## Submission

The miner commits `metro-v1:ft:app:<scenario>:hippius:<repo>@sha256:<digest>`
(the existing lane-C form, parsed generically). The tree:

```
harness.json      # {"scenario": "bess", "api": 1, "calls_per_step": 4, ...}
policy.py         # class Harness(HarnessBase): reset(spec) / step(obs, fc) -> action
lib/…             # optional pure-Python helpers, size-capped
```

```python
class Harness(HarnessBase):
    def reset(self, spec):            # scenario constants, asset limits, seed
        ...
    def step(self, obs, fc):          # obs: history to now (past only)
        q = fc.forecast(obs.prices, horizon=288, levels=(0.1, 0.5, 0.9))
        return self.plan(q, obs.soc)  # action, validated and clipped by the sim
```

`fc.forecast(context, horizon, levels, covariates=…)` is the only way to reach
the model. It returns quantiles from the vertical's fine-tuned model. Each
step has a call budget (`calls_per_step`, cap 8) and a context-length cap.

## Rules

- **Sandbox.** The same static guard and network-isolated sandbox as
  generators (`cascade/interface/static_guard.py`): no network, no clock, no
  filesystem outside the tree, single-threaded BLAS, numpy / scipy only, and
  a per-step CPU-time limit (for example 200 ms; over the limit = a no-op
  action plus a penalty). The tree is capped (for example 2 MB, no weights
  files).
- **Determinism.** Same observations and forecast replies give the same
  actions, byte for byte. The validator records every forecast request and
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
| scenarios | `cascade/eval/harness/` (new) | one module per scenario: data adapter, simulator, oracle, reference harness, score. Pure numpy / scipy, deterministic |
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
calls_per_step_max   = 8
step_cpu_ms          = 200
tree_max_bytes       = 2_000_000

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

## Rollout

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
