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
- **Embargo-data eval (no forward wait).** Your training data must end at
  least `embargo_days` (30) before you submit. Every suite and horizon is
  judged on windows drawn from the embargo: the `embargo_eval_days` (28) before
  your leg's dispatch, which your recipe could not train on. Up to
  `backtest_windows_per_cell` (48) windows are drawn per asset, series and
  horizon, ranked by a hash of the settlement's block hash, so nobody can pick
  them in advance. The short horizons (1m×60, 5m×36) also score forward
  windows that have already closed. No window is waited for.
- **Verdict at the first settlement after the leg.** An entry is judged at
  the first settlement after its leg finishes. An inconclusive entry may wait
  at most `verdict_wait_blocks` (3,900 blocks, about 13 h). That cap is fixed
  in code.
- **Stages.** The first pass (`judge`) and the confirmation (`confirm`) use
  disjoint halves of the drawn windows. A first pass is `pending` and unpaid
  until it confirms, and you must beat the current slot holder with a
  strictly higher LCB to take the slot.

## How you are paid

Lane C splits its pool by competition weight and then over each
competition's targets. MarketCast has one target, so its holder can receive
the **whole lane C pool**. The slot pays **100% for 5 to 20 days**, set by
the confirmed gain's LCB, and then it ends:

```
frac = clamp((LCB - gain_margin) / (full_pay_max_gain - gain_margin), 0, 1)
days = full_pay_min_days + (full_pay_max_days - full_pay_min_days) * frac
eras = ceil(days / era length)        # at least 1
```

With `gain_margin` 0.01, `full_pay_min_days` 5, `full_pay_max_days` 20 and
`full_pay_max_gain` 0.05, and 12 h eras:

| Confirmed gain LCB | Days at full pay | Eras |
|---:|---:|---:|
| 0.01 | 5 | 10 |
| 0.02 | 8.75 | 18 |
| 0.03 | 12.5 | 25 |
| 0.04 | 16.25 | 33 |
| ≥ 0.05 | 20 | 40 |

There is no decay tail and no floor: after the full-pay period the slot's
pay burns. A new holder that beats you replaces you earlier.

**Worked example at the owner-proposed split A0 / B35 / C15 / D50.** This is
the owner's proposal (plan rev6.24, set by the launch overlay), not a
guaranteed live value. Check `cascade rules c` for `share`. A holder
confirmed at LCB 0.03 receives **15% of emissions for 25 eras (12.5 days)**,
then nothing. At LCB 0.05 or more it is 15% for 40 eras (20 days).

## Historical data

The recorder's MarketCast bars are published once they are older than the
training embargo, so you can try recipes locally and compare them with
prior rounds:

- **HF dataset:** [tensorlink-dev/cascade-marketcast-bars](https://huggingface.co/datasets/tensorlink-dev/cascade-marketcast-bars)
- **Hippius mirror:** <https://s3.hippius.com/cascade-manifests/marketcast/public-bars/index.json> (anonymous HTTPS, same files)

Only days strictly before `today - embargo_days` (30) are public; the last
30 days stay private, because the eval draws its windows from the
`embargo_eval_days` (28) before each leg's dispatch and your recipe's
training data must end `embargo_days` before submission. The layout is
`data/<class>/<SYMBOL>/<tf>/<YYYY-MM>.parquet` with `tf` one of `1m`, `5m`,
`15m`, `1h` or `funding`: columns `timestamp` (UTC, bar open) and float64
`open`, `high`, `low`, `close`, `volume` (funding: `rate`, `premium`). A
bar nobody recorded is an absent row. Aggregate 1h bars for 4h and 1d.
`index.json` lists every file's sha256. The data is Hyperliquid perpetuals
(core dex for crypto, the `xyz` dex for equities, FX, metals and
commodities), redistributed under Hyperliquid's terms; the dataset card has
the details. It updates daily.

```bash
cascade lane-c data --symbols BTC,ETH,SP500 --timeframes 1m,1h,funding \
    --start 2026-10-01 --end 2026-10-31 --out ./bars
cascade lane-c data --symbols GOLD --out ./bars --source hippius
```

The download needs no keys, verifies every file's sha256 and resumes. The
default source is `hf`, falling back to the Hippius mirror. Validators score
on the private signed snapshot, never on this dataset.

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

The leg runs on one GPU, so you may rent any Lium GPU type. The recommended
types are in `[marketcast] funded_pod_skus` (shipped: `1xRTX4090`, `1xA100`,
`1xH100`, `8xH100`). Without `--gpu`, the leg rents the cheapest of these that
fits. You can choose any other Lium type (for example `1xRTX3090` or
`1xH200`) at your own risk. The fund reply then carries a warning. The leg is
still capped by `max_leg_hours` and your recipe's `leg_hours`. It rents on
your own Lium key: send a plain Lium key, because MarketCast never rents on
Shadeform. Nothing is refunded if the type is slow, cannot boot, runs out of
memory or crashes.

## What gets you refused

- **Recipe limits:** learning rate above `max_lr` (1e-3), more than
  `max_steps` (100,000) steps, a batch above `max_batch` (512), LoRA rank
  above `max_lora_rank` (64), context above `max_context_length` (32,768),
  leg hours above 12, or an unknown `marketcast.json` key.
- **Embargo:** training data ending less than 30 days before submission, or a
  generator that embeds series from the embargo window
  (`embargo_series_embedded`, checked by the trainer before the leg).
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
It needs a confirmation on the other half of the drawn windows and a higher
LCB than the holder's.

**How long until a verdict?** The first settlement after your leg finishes.
An inconclusive entry waits at most about 13 h (`verdict_wait_blocks`).

**When does MarketCast open?** At block 9,586,800 (about 2026-11-25 22:28
UTC) on mainnet, with lanes B and D. Read `from_block` in `cascade rules c`.

**Will there be other lane C competitions?** Possibly. New competitions are
defined by a spec in a signed overlay (DEC-CA-0075), so they can start at an
era boundary without a validator release. Each has its own targets, start,
optional end and weight in lane C's pool. `cascade rules c` and
`cascade status --json` list what is live. Your leg's embargo values are
recorded when it dispatches, so a later overlay change does not move them.

**Do I need a GPU of my own?** Only for local tests. Live legs rent on your
provider key.

**Which assets count?** The 58 assets outside the alpha class. The alpha
suite is off.
