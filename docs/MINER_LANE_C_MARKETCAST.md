# Lane C miner guide: MarketCast

This is the competition brief for lane C, whose one competition today is
MarketCast. The full recipe reference is [MARKETCAST.md](MARKETCAST.md), and
the walkthrough is [MINER.md lane C](MINER.md#14-marketcast-lane-c-the-composite-mktmarket-target--dec-ca-0064).
The live numbers are in `cascade rules c`, read from the same `chain.toml`
the trainer and validators load.

```bash
cascade rules c            # headline lane C and MarketCast rules
cascade rules c --all      # every [lane_c] and [marketcast] key
cascade status --json      # whether MarketCast is admitting entries now
```

## Intent

MarketCast asks for the **best fine-tuning recipe for market forecasting**.
Every entry fine-tunes the same pinned base model, Timestral-28M (`base_ref`
in `cascade rules c`), on market bars. Only the recipe differs: which data,
which windows, which augmentations, and how much synthetic data. The winner
is the recipe whose model forecasts real markets best on data that arrives
after you submit.

## What you submit

- **A recipe tree** with `marketcast.json` at its root: data range, assets,
  timeframes, covariates, augmentations, learning rate, steps, batch, LoRA
  rank and leg hours. A sandboxed `generator.py` (the lane A interface) is
  optional and is mixed in at a `synthetic_fraction` from 0 to 1.
- **One target:** `mkt:market`, the composite of all scored suites.
- **Your own compute key.** The trainer fine-tunes the base on your provider
  key.

The universe has 66 symbols in six asset classes (crypto 28, equity/ETF 19,
metals 4, commodities 4, FX 3, Bittensor alpha 8) on six timeframes: 1m, 5m,
15m, 1h, 4h and 1d. The scored horizons are 1m×60, 5m×36, 15m×96, 1h×48,
4h×42 and 1d×7. The alpha class feeds only the `alpha` suite, which is off,
so 58 assets are scored.

```bash
cascade lane-c fetch-base --out ./timestral
cascade lane-c snapshot ./my-recipe --out ./train-snap
cascade lane-c snapshot ./my-recipe --out ./eval-snap --evaluation --days 7
cascade lane-c test ./my-recipe --target mkt:market --base-dir ./timestral \
    --train-snapshot ./train-snap --eval-snapshot ./eval-snap
cascade lane-c check ./my-recipe --target mkt:market
cascade lane-c upload ./my-recipe --target mkt:market --hub-repo my-ns/my-recipe
cascade lane-c commit --target mkt:market --ref <repo@digest> --repo-dir ./my-recipe \
    --wallet-name my-miner --wallet-hotkey mkt1
export LIUM_API_KEY=...   # env only, never an argument
cascade lane-c fund https://submissions.cascadesub.net --target mkt:market --ref <repo@digest> \
    --wallet-name my-miner --wallet-hotkey mkt1 --yes
cascade lane-c status https://submissions.cascadesub.net --hotkey <ss58>
```

## How it is scored

- **Seven suites, one composite.** `ohlcv`, `returns`, `vol`, `high`, `low`,
  `range` and `liquidity`, combined as a weighted geometric mean: returns
  0.25, ohlcv 0.20, vol 0.18, range 0.12, liquidity 0.10, high 0.075 and low
  0.075 (`composite_suite_weights`).
- **Per window,** CRPS is scaled by the seasonal-naive error. Windows are
  grouped into cells (asset class × horizon × 6 h block), each clipped to
  [1/4, 4] against the base.
- **The pass rule.** The composite gain's LCB must be ≥ `gain_margin` (1%)
  with a positive point gain, at alpha corrected for your coldkey's entries.
  You need at least `min_windows` (200) windows, `min_clusters` (8) clusters
  and 8 time blocks. No suite and no guard (MASE, QLIKE, tails) may be
  confidently worse. The model must also beat simple baselines (random walk
  with EWMA volatility, seasonal AR(1)), and its 50% / 80% intervals and
  5% / 95% tails must be calibrated.
- **Timing.** Your training data must end at least `embargo_days` (30) before
  you submit. The 4h×42 and 1d×7 horizons are judged on that embargo gap, so
  they need no wait. The other four horizons are forward-only: windows start
  at least 1 h after dispatch, and an entry waits up to
  `verdict_wait_blocks` (43,200 blocks, about 6 days).
- **Stages.** A first pass is `pending` and unpaid. A later settlement on
  fresh windows must confirm it, and you must beat the current slot holder
  with a strictly higher LCB to take the slot.

## How you are paid

Lane C splits its pool by competition weight and then over each
competition's targets. MarketCast has one target, so its holder can receive
the **whole lane C pool**. The confirmed gain's LCB sets the full-pay period
(`full_pay_tiers`):

| Confirmed gain LCB | Full-pay eras | About |
|---:|---:|---:|
| ≥ 0.05 | 10 | 5 days |
| ≥ 0.02 | 5 | 2.5 days |
| ≥ 0.01 | 2 | 1 day |

After that the slot halves every era (`half_life_rounds`) down to a 10%
floor, and stops at age `win_ttl_eras` (28 eras, about 14 days). A new
holder replaces you earlier.

**Worked example at the owner-proposed split A0 / B35 / C15 / D50.** This is
the owner's proposal (plan rev6.24, signed overlay), not a guaranteed live
value. Check `cascade rules c` for `share`. A holder confirmed at LCB 0.02
receives:

| Age since confirmation (eras) | Slot multiplier | Share of emissions |
|---|---:|---:|
| 0–4 | 100% | 15% |
| 5 | 50% | 7.5% |
| 6 | 25% | 3.75% |
| 7 | 12.5% | 1.875% |
| 8–27 | 10% | 1.5% |
| 28+ | 0% | 0% |

## Strategies worth trying

- **Regime-aware data.** Markets switch between calm, trending and crisis
  regimes. Weight or sample training ranges so each regime is represented,
  rather than fitting the most recent month.
- **Aim at the heavy suites.** Returns (0.25), OHLCV (0.20) and volatility
  (0.18) carry most of the composite. A recipe that improves volatility and
  range forecasts often helps several suites at once.
- **Liquidity and volatility suites reward covariates.** Volume, funding and
  cross-asset covariates (`covariates` in `cascade rules c`) carry
  information the price series alone does not.
- **Calibrate the tails.** The calibration and tail guards refuse an
  over-confident model even when its CRPS improves. Augmentations such as
  `scale`, `jitter` and `mixup` can widen intervals sensibly.
- **Use synthetic data sparingly and on purpose.** A generator can add rare
  regimes (crashes, gaps, illiquid hours) that are thin in real history.
- **Test on a recent held-out snapshot** (`cascade lane-c snapshot
  --evaluation`) before you pay. A gain on your own training range proves
  nothing.

## Costs

A MarketCast leg is short: Timestral-28M is small, the default leg is
`leg_hours` (1 h) and the recipe may ask for up to `max_leg_hours` (12 h).
You pay the provider's listed price for the GPU you rent. Pick it with
`--gpu` and cap it with `--max-price`. A failed or losing leg is not
refunded.

## What gets you refused

- **Recipe limits:** learning rate above `max_lr` (1e-3), more than
  `max_steps` (100,000) steps, a batch above `max_batch` (512), LoRA rank
  above `max_lora_rank` (64), context above `max_context_length` (32,768),
  leg hours above 12, or an unknown `marketcast.json` key.
- **Embargo:** training data ending less than 30 days before submission.
- **Targets:** anything but `mkt:market` (generic `domain:` / `freq:` /
  `horizon:` targets are refused).
- **Duplicates:** a recipe byte-identical or near-identical to an earlier
  one (`duplicate`).
- **Caps:** one commitment per hotkey (shared with lanes A and B), one live
  leg per hotkey across A/B/C, and `max_open_legs_per_coldkey` (1) per
  coldkey. Re-funding a spent commitment returns `spent`.
- **Lifecycle:** entries outside the competition window
  (`marketcast_inactive`, `competition_ended`).
- **Licence:** Timestral is released under a modified MIT licence with a
  revenue restriction that extends to derivatives. Check those terms before
  you reuse its outputs elsewhere. Its outputs may not be used as a lane D
  teacher.

## FAQ

**Is the base the lane A king?** No. Every recipe fine-tunes the pinned
Timestral base, never the lane A or lane D king.

**Can I start from my previous MarketCast checkpoint?** Only from one the
trainer produced (the recipe's `init`).

**Why is my entry still pending after a pass?** The first pass is unpaid.
It needs a confirmation on fresh windows and a higher LCB than the holder's.

**Do I need a GPU of my own?** Only for local tests. Live legs rent on your
provider key.

**Which assets count?** The 58 assets outside the alpha class. The alpha
suite is off.
