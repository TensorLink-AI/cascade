# MarketCast miner guide

MarketCast is lane C's competition (DEC-CA-0064). You write a **fine-tune
recipe** for a pinned forecasting model, pay for one GPU leg that trains it,
and win the `mkt:market` slot if your fine-tune forecasts the market better
than the untouched model on market windows its training never saw.

This page is the step-by-step path. The full rules (recipe schema, every gate,
every number) are in [MINER.md §14](MINER.md#14-marketcast-lane-c-the-composite-mktmarket-target--dec-ca-0064);
when they disagree, §14 and the live `chain.toml` win.

> **Status.** The checked-in `chain.toml` arms MarketCast at block
> **9586800** (`[marketcast]` and `[lane_c] from_block`; about Wed 25 Nov 2026
> 22:28 UTC; owner 2026-11-24 moved it from 9579600). Once validators and the trainer run this release you can commit
> and fund `mkt:market`, but no leg rents (and nothing is charged) before the
> first lane-C era starts at that block; queued entries go first. On a
> release with `from_block = 0` every `mkt:` command is refused
> (`marketcast_inactive`). Check the live network before spending anything.

## What you are trying to do

Make a small, general forecaster **better at describing what real markets do
next** — as a full probability distribution, not one number. Your fine-tune of
Timestral-28M wins if, for the listed assets and every timeframe from 1-minute
to daily bars, its quantile forecasts of the coming path are:

- **More accurate as a distribution.** The score is CRPS, which rewards the
  whole spread of outcomes you predict. A confident wrong call costs more
  than an honest, wider one; an over-wide one costs too.
- **Calibrated.** Your 50 % and 80 % intervals contain the realised value
  that often, and your 5 % / 95 % tails are hit about 5 % of the time.
- **Better everywhere, worse nowhere.** The composite is a geometric mean
  over seven suites (location, volatility, range, highs, lows, liquidity,
  OHLCV) and every asset class. A gain in one place never buys a confident
  loss in another; the guards and the simple baselines enforce that.
- **Good on market it has not seen.** Your training data ends 30+ days
  before you commit, and you are scored on windows drawn from those
  embargoed days (plus the short paths' windows after your leg starts),
  chosen by a block hash that does not exist while you train. What wins is
  structure that holds across regimes, not a memorised stretch of history.

### Where the gains are

Directions, not a recipe — the field is open:

- **Market structure the base model lacks:** volatility clustering and its
  decay, intraday and weekly session patterns, fat tails, the link between
  volume and volatility, cross-asset moves (BTC/ETH, equities, gold, FX).
- **Scheduled events.** FOMC, CPI and payrolls are known in advance (the
  `calendar`, `session` and `events` covariates); markets behave differently
  around them.
- **The right covariates.** Past `volume`, `funding`, `open_interest`,
  `spread` and the `cross_*` series are available; use the ones that carry
  signal and leave out the ones that add noise.
- **Coverage the bars lack.** A generator (`synthetic_fraction` > 0) can
  teach regimes the recorded history is too short for, and with
  `synthetic_channels` its samples carry their own covariate rows (volume,
  funding, cross-asset, a clock for calendar / session / events) — but
  synthetic data that does not look like real markets hurts.
- **Small, careful updates.** The test is "confidently better, nowhere
  worse", so a conservative fine-tune (LoRA, a low learning rate, enough
  steps) that keeps what the base does well usually beats an aggressive
  full fine-tune.

### What this is not

- **Not a trading competition.** Nothing scores profit or direction alone;
  getting the width and shape of the distribution right is most of the score.
- **Not about the eval windows.** No leg reads the embargoed days, which
  ones are drawn is decided after your leg, and a generator that embeds
  embargoed prices is refused (`embargo_series_embedded`).
- **Not about copying.** A near-duplicate of an earlier entry is judged a
  duplicate; seniority is the commitment block.
- **Not about hardware.** The model is 28M parameters; a bigger GPU buys
  speed you rarely need.

## At a glance

| | |
|---|---|
| Target | `mkt:market` — one composite slot over 7 suites (`ohlcv`, `returns`, `vol`, `high`, `low`, `range`, `liquidity`) |
| Model you fine-tune | `mistralai/Timestral-28M@180698bbf5a50dda3516aa6419798437fb03a430` (pinned; never the lane-A king) |
| What you submit | A tree with `marketcast.json` (+ optional `generator.py`) |
| Who pays | You pay the training pod (your provider key). The main validator pays for forecasting and scoring |
| Leg length | Your `leg_hours` (≤ 12), default `[lane_c] leg_hours` = 1 h |
| Training data | Public market bars the trainer fetches itself, ending ≥ 30 days before you submit |
| Judged on | Windows drawn from the 28 days before your leg's dispatch (`embargo_eval_days`; inside the 30-day embargo no leg reads), 48 per asset, series and timeframe, ranked by the registering settlement's block hash; plus the closed `1mx60` / `5mx36` forward windows (DEC-CA-0074) |
| Time to a verdict | The first settlement after your leg finishes (an inconclusive entry waits ≤ 13 h, `verdict_wait_blocks` = 3900) |
| To win | Composite gain LCB ≥ 0.01 over the base, no suite / guard / baseline confidently worse, calibration gate, then a **confirmation** on fresh windows and a paired win over the current holder |
| Reward | Lane C's share (25 % of emissions, taken from lane A), paid to the slot holder by a gain-tiered schedule |
| Licence | Timestral's [Modified MIT licence](https://huggingface.co/mistralai/Timestral-28M/blob/180698bbf5a50dda3516aa6419798437fb03a430/LICENSE) restricts companies over USD 20M revenue and their derivatives — read it first |

## What you control and what you can access

### What you control

Everything below is yours to choose in `marketcast.json` or at funding.
Bounds are the shipped `[marketcast]` values; the live `chain.toml` wins.

| Lever | Your options |
|---|---|
| Assets | Any subset of the universe, with a weight per class (`data.assets`, `data.class_weights`). Scored classes: `crypto` (28 perps: BTC, ETH, SOL, XRP, HYPE, BNB, DOGE, ADA, AVAX, LINK, SUI, TRX, LTC, DOT, BCH, NEAR, APT, ARB, OP, TAO, ENA, AAVE, UNI, INJ, TIA, SEI, WIF, JUP), `equity_etf` (SP500, XYZ100, JP225, KR200, SMH, XLE, TLT, XBI, EWJ, EWY, EWZ, EWT, AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA), `metals` (GOLD, SILVER, PLATINUM, PALLADIUM), `commodities` (CL, BRENTOIL, NATGAS, COPPER), `fx` (EUR, JPY, GBP) |
| Training range | `data.start` … `data.end`, ending ≥ 30 days before you submit |
| Bar sizes | `data.frequencies` ⊆ `1m`, `5m`, `15m`, `1h`, `4h`, `1d` |
| Horizon mix | Weights over `1mx60`, `5mx36`, `15mx96`, `1hx48`, `4hx42`, `1dx7` (all six are judged whatever you weight) |
| Context | `context_length` 32 – 32768 steps (the eval feeds up to 32768: as much of that as the bar's recorded history holds, so longer contexts pay off more as the recording grows) |
| Inputs | `covariates` ⊆ `volume`, `funding`, `open_interest`, `spread`, `cross_asset`, `cross_eth`, `cross_equity`, `cross_gold`, `cross_fx`, `calendar`, `session`, `events` |
| Augmentations | Up to the allowed set — `scale`, `jitter`, `crop`, `time_warp`, `mixup` — each with a probability and numeric params |
| Loss | `objective.suite_weights`, optional `quantile_weights` (one per each of the 21 quantile knots, 1 %…99 %), `coverage_penalty` 0 – 10 |
| Optimiser | `lr` ≤ 1e-3; `cosine` / `linear` / `constant`; warmup ≤ 0.2; `steps` ≤ 100 000; `batch` ≤ 512; weight decay 0 – 1; clip ≤ 10; optional EMA |
| What trains | `full`, `last_n_blocks` (1 – 12 of Timestral's 12 blocks), `frozen_embeddings`, or `lora` (rank ≤ 64) |
| Synthetic data | Your own `generator.py`, mixed at `synthetic_fraction` 0 – 1 (1 = generator only); `synthetic_channels` gives its samples their own covariates and bar size |
| Warm start | `init`: an earlier MarketCast checkpoint the trainer produced — yours or the current holder's |
| Compute | `leg_hours` ≤ 12 (you pay for it); GPU type, max price and provider order at `fund` |

### What you can access

- **The model.** Timestral-28M at its pinned revision, weights public on
  Hugging Face; `cascade lane-c fetch-base` downloads and digest-checks it.
  12 transformer blocks, patch 32, context up to 32768, 21 quantile knots.
- **Public market data, locally.** `cascade lane-c snapshot ./my-mkt --out
  train-snap` fetches your recipe's training range from the public feeds;
  `--evaluation --days 7` fetches a recent held-out snapshot to test on. Use
  them with `cascade lane-c test`.
- **What your live leg trains on.** The trainer relays the operator's bar
  recording (Hyperliquid and Hyperliquid xyz, the classes' primary sources)
  for exactly your recipe's assets and range (`data.start` to `data.end` +
  1 day), at every bar size your recipe uses. The recording is young: sub-hour bars only exist from the day it
  started recording them, so an embargoed range may have no `1m` / `5m`
  bars yet — those horizons then train on your generator or not at all.
  `open_interest` and `spread` exist only from 2026-11-18 (NaN before).
- **Every verdict, in full.** After scoring, the signed report
  (`marketcast/round-<id>.json`) publishes every window's losses — yours,
  the base's and the baselines' — and the forecasts themselves as quantile
  sidecars. `cascade-audit marketcast --reproduce` recomputes them. Your
  receipt row breaks the gain down per suite, guard and baseline.
- **Other entries.** Uploaded recipe trees are public and commitments
  reveal on chain, so recipes can be read once revealed; trained checkpoints
  are published as trained pointers (usable as `init`). Copying does not
  pay: a byte-identical or near-identical fine-tune of an earlier commitment
  is a `duplicate`.
- **The queue.** `cascade lane-c status <intake>` shows the public lane-C
  queue; `cascade status --lane c` tracks your entry.

### What you cannot access

- **The bars you are scored on**, in training. Every judged window lies in
  your embargo (days no leg reads) or opens at least an hour after your leg
  is dispatched; which embargo windows are drawn is decided by a block hash
  after your leg. A generator may not embed embargoed prices: the tree is
  scanned before your leg rents (`embargo_series_embedded`).
- **Anything after your leg's dispatch time.** The pod never receives a bar
  at or after it, and never anything after `data.end` + 1 day.
- **Your own dataset.** There is no upload of private training data; a
  generator is the only way to add series.
- **The network, from your generator.** It runs sandboxed on the training
  pod: no network (network namespace + socket block), 4 GB memory, a
  128 MB code-only tree (no weights), exactly 32 rows per yield, series of
  64 – 4096 steps. It does not see the market data either.
- **The training pod's controls.** The pod is rented on your provider key
  and billed to you, but the trainer drives it, relays its inputs and
  verifies the checkpoint it copies back; a copy that keeps differing from
  the pod's own listing is treated as tampering.
- **The holder's verdict on other weeks.** To take the slot you must beat
  the holder on *your* windows; its old LCB does not count.

## What you need

- A registered hotkey on the subnet and this repo's pinned environment
  (`bittensor==11.3.0`; commit with `cascade`, not a hand-rolled extrinsic).
- A provider API key with a **budget set on the key itself**: Lium,
  Shadeform, Vast.ai or Runpod ([MINER.md — your provider keys](MINER.md#your-provider-keys)).
  The key goes to the operator's intake only when you fund; never paste it
  on a command line — pass the *name* of the env var.
- Optional, for local testing: a CUDA GPU (any 16 GB+ card fits the 28M model).

**One hotkey, one live competition.** A hotkey holds one chain commitment
shared by lanes A, B and C. Committing a MarketCast recipe replaces that
hotkey's A/B commitment and can supersede its queued work, and only one leg
per hotkey can be live across A/B/C. Use a separate hotkey per lane if you
want to compete in several at once. The trainer also runs at most one open
MarketCast leg per **coldkey** (`max_open_legs_per_coldkey` = 1).

## Step by step

### 1. Test locally (no chain, no rental)

```bash
cascade lane-c fetch-base --out ./timestral --json                 # pinned Timestral, digest-checked
cascade lane-c snapshot ./my-mkt --out train-snap                  # your recipe's training range
cascade lane-c snapshot ./my-mkt --out eval-snap --evaluation --days 7   # recent held-out data
cascade lane-c test ./my-mkt --target mkt:market --base-dir ./timestral \
    --train-snapshot train-snap --eval-snapshot eval-snap --device cuda --json
```

`lane-c test` trains your recipe and scores it against the base the way the
validator will, on your own snapshots. The
[agent walkthrough](AGENT_MINING.md#lane-c-test-a-marketcast-recipe) has the
snapshot flags and a worked example. A local win is necessary, not
sufficient: the live verdict uses windows nobody has seen yet.

### 2. Write `marketcast.json`

The full schema with every key is in
[§14 — What you submit](MINER.md#what-you-submit). The decisions that matter most:

- **`data.end`** must be ≥ 30 days before your submission (`embargo_days`).
  The days in between are the embargo: no leg reads them, and your eval
  windows are drawn from them.
- **`horizon_mix`** — horizons are `<bar>x<steps>`: `1mx60`, `5mx36`,
  `15mx96`, `1hx48`, `4hx42`, `1dx7`. Every suite is scored on every
  horizon, so a recipe that only trains short horizons still gets judged on
  the long ones.
- **`trainable.scope`** — `full`, `last_n_blocks`, `frozen_embeddings` or
  `lora` (rank ≤ 64). Lighter scopes regress less where you did not train.
- **`objective`** — `suite_weights` and `coverage_penalty` shape the loss.
  The verdict is a weighted geomean across suites and fails if *any* suite
  is confidently worse than the base, so do not trade one suite for another.
- **`leg_hours`** — the wall clock you pay for. Your forward windows start at
  dispatch, so a longer leg does not delay the verdict.
- **`init`** — optionally warm-start from an earlier MarketCast checkpoint
  the trainer produced (e.g. the current holder's). You are still judged
  against the pinned base, and against the holder to take the slot.
- **`synthetic_fraction`** — > 0 needs a `generator.py` (lane A's sandboxed
  `generate()` contract). A horizon with no real bars in your range trains on
  generator series alone, so a generator is how you train the short
  horizons while the minute recording is still younger than the embargo.
- **`synthetic_channels`** — name your generator's rows (`"target"` first,
  then past covariates and an optional `"time"` clock) so synthetic samples
  carry their own covariates instead of missing ones. The clock's step sets
  the bar size the array is used at, so 1 m and 1 h series can differ. Units and an example
  are in [§14](MINER.md#what-you-submit).

### 3. Check, upload, commit

```bash
cascade lane-c check  ./my-mkt --target mkt:market
cascade lane-c upload ./my-mkt --target mkt:market --hub-repo NAMESPACE/NAME --json
cascade lane-c commit --target mkt:market --ref <data.ref from upload> \
    --repo-dir ./my-mkt --wallet-name W --wallet-hotkey H
```

- Use the upload result's exact `data.ref` (`repo@sha256:…`) everywhere after.
- `commit` uses a **timed reveal**: it prints the block your payload
  decrypts at (just before the next settlement). You cannot fund until it
  has revealed — `cascade reveal-status HOTKEY --json` tells you.

### 4. Preflight, then fund

```bash
cascade preflight ./my-mkt --lane c --targets mkt:market --hotkey HOTKEY \
    --ref <ref> --max-price RATE --max-spend TOTAL --json
export LIUM_API_KEY=...            # or SHADEFORM / VAST / RUNPOD
cascade lane-c fund https://<intake> --target mkt:market --ref <ref> \
    --wallet-name W --wallet-hotkey H --gpu RTX3090 --max-price 0.5 --yes
```

- `--gpu` picks the GPU type; an unavailable type **waits** and is never
  burned. Add `--allow-fallback` to take other allowed types.
- `--max-price` caps the hourly rate. Preflight's `--max-spend` is a review
  number, not an enforced cap — the enforced cap is the budget on your key.
- `--provider-key-env PROVIDER=ENV_NAME` (repeatable) gives an ordered
  fallback across providers.
- Funding rents nothing by itself; the trainer rents your pod when your leg
  is admitted. `cascade lane-c fund --withdraw ...` pulls a still-queued entry.

### 5. Track it

```bash
cascade status --hotkey YOUR_SS58 --lane c --target mkt:market --ref <ref> \
    --intake https://<intake> --watch          # agents: --json, poll every 30–60 s
cascade lane-c status https://<intake> --hotkey YOUR_SS58 --json   # the public lane-C queue
```

## What happens after you fund

1. **Queued → admitted.** Earliest reveal gets first pick of the GPUs that
   fit. MarketCast legs dispatch on the era's lane-C settlement cadence.
2. **Your leg.** The trainer rents a pod on *your* key, relays the permitted
   bars to it, and fine-tunes the pinned base (or your `init`). The pod never
   sees a bar after its dispatch time. The trainer copies the checkpoint
   back, verifies it, publishes it, and releases the pod.
3. **Settlement.** The next boundary publishes a signed lane-C manifest
   naming your trained checkpoint.
4. **Scoring.** The main validator forecasts the base and your fine-tune on
   the windows drawn for your entry from a signed market snapshot and publishes
   the per-window losses and the forecasts at `marketcast/round-<id>.json`.
   Anyone can recompute them with `cascade-audit marketcast --reproduce`.
5. **Judgement.** Every validator derives the same verdict from those rows.
   Not enough windows yet (≥ 200 over ≥ 8 clusters and ≥ 8 six-hour blocks)
   ⇒ you **wait**, re-judged every settlement for up to 13 h. Normally the
   verdict comes at the first settlement after your leg.
6. **Pending.** An initial pass is `pending` — no payment yet.
7. **Confirmed.** A later settlement must pass you again on the other half of
   the embargo windows (never the ones the initial test scored, plus any
   forward windows after the first snapshot ended), and you must beat the current
   holder on *your* windows (paired `vs_holder` gain). Then you take the slot.
8. **Payout** (DEC-CA-0073). 100 % of the slot for `5 + 15 × clamp((LCB −
   0.01) / 0.04, 0, 1)` days (`[lane_c] full_pay_min_days` /
   `full_pay_max_days` / `full_pay_max_gain`, ramped from `[marketcast]
   gain_margin`), rounded up to whole eras: 10 to 40 twelve-hour eras. Then
   the slot ends and burns unless a newer win has replaced it. There is no
   decay tail and no TTL. The win records its `full_eras`; `tier` is the
   clamped gain the duration came from. Empty slots burn.

## Reading your verdict

`cascade status` explains the state in words. The verdict row behind it
carries:

| Field | Meaning |
|---|---|
| `passed`, `stage`, `reason` | The outcome and the first rule that decided it, e.g. `verdict` / `gain LCB -0.0049 < margin 0.01` |
| `gain`, `gain_lcb` | Composite gain over the base (point estimate and bootstrap lower bound). Positive = better than the base |
| `mkt.suites.<suite>` | Per-suite `gain`, `gain_lcb`, `gain_ucb`, `n` — find the suite that sank you |
| `mkt.vs_baseline` / `mkt.base_vs_baseline` | Your fine-tune / the base vs the simple baselines (random walk + EWMA/FHS, seasonal AR(1)) |
| `mkt.coverage`, `mkt.coverage_passed` | Empirical interval coverage and tail hit rates per horizon — the calibration gate |
| `mkt.vs_holder` | Paired gain against the slot holder (confirmation only) |
| `inconclusive` | Non-empty while you are waiting for windows |
| `full_pay` | The payout tier a confirmed win earned |

### Outcomes

| You see | Meaning | What to do |
|---|---|---|
| waiting / `inconclusive` | Too few windows | Nothing; re-judged every settlement for up to 13 h |
| `infra` | The main validator could not compute your entry | Nothing; retried without using your wait, up to 8 times, then requeued uncharged |
| `verdict`, not passed | Your fine-tune did not prove better, or was confidently worse somewhere | Read `mkt.suites`; re-commit a new recipe |
| `fault` | NaN forecasts, or your model's forward fails identically every time | Fix the recipe; this is your fault |
| `duplicate` | Byte-identical or near-identical (`near_dup_tol` 2 %) to an earlier-committed entry | Seniority is the commitment block; copying does not pay |
| `pending` | Initial pass | Wait for the confirmation settlement |
| `not_selected` | Passed, but another entry took the slot this settlement | No reward from this attempt |
| `confirmed` | You took the slot | Payment follows the schedule above |
| `spent` (on fund) | This commitment already trained | Re-commit (new reveal) to train again |

## Costs

You pay for the training pod only: its hourly rate × your leg (default 1 h)
plus boot and copy-back time. Lane-C legs are small — on the 2026-11-20
testnet rehearsal a 1500-step leg on a Lium RTX 3090 billed at $0.18/h and
its checkpoint was 0.11 GB. Prices move; your `--max-price` and your key's
budget are what bound you. A leg lost on the operator's side retrains
uncharged.

## Tips

- **Do not regress anywhere.** The composite fails on any suite, guard
  (`mase`, `qlike`, `tail`) or baseline that is *confidently* worse. On the
  testnet rehearsal a short full-fine-tune recipe improved `liquidity` and
  `range` but lost on `returns`, `vol` and `ohlcv`, and failed overall
  (gain −0.54 %). One data point, but the pattern is the trap.
- **Calibrate.** `returns` and `vol` must cover their 50 / 80 % intervals and
  5 / 95 % tails within tolerance at every horizon. Narrow quantiles fail.
- **Cover every horizon.** Long horizons (`4hx42`, `1dx7`) are judged too.
- **Test against the baselines locally.** The untouched base only ties the
  simple baselines on returns and loses on 1-day returns and highs; those are
  the cheapest places to gain, and the places a fine-tune most often breaks.
- **Spend on the recipe, not the GPU.** The model is 28M parameters; a bigger
  card mostly buys time you do not need.

## FAQ

**Can I enter lane A and MarketCast with one hotkey?** Yes, but not at the
same time: one commitment per hotkey, and committing MarketCast replaces the
A one. Use separate hotkeys to compete in both at once.

**Does my leg depend on the lane-A king?** It fine-tunes the pinned base,
never the king, and slots compare against the pinned base, so a king change
never displaces a MarketCast holder.

**Do I upload training data?** No. The trainer fetches the permitted public
bars itself. A generator (`synthetic_fraction` > 0) is the only way to add
your own series, with their own covariates if you set `synthetic_channels`.

**Can I see the windows I am scored on?** The embargoed bars are public
market data, but which windows are drawn is fixed by a block hash after your
leg, and no leg trains on them. After scoring, the signed report (with the
draw) and the forecasts are public.

**Where are the rules for X?** [MINER.md §14](MINER.md#14-marketcast-lane-c-the-composite-mktmarket-target--dec-ca-0064)
and the `[marketcast]` / `[lane_c]` tables of the live `chain.toml`.
