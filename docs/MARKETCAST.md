# MarketCast miner guide

MarketCast is lane C's competition (DEC-CA-0064). You write a **fine-tune
recipe** for a pinned forecasting model, pay for one GPU leg that trains it,
and win the `mkt:market` slot if your fine-tune forecasts the market better
than the untouched model on windows that had not happened when it trained.

This page is the step-by-step path. The full rules (recipe schema, every gate,
every number) are in [MINER.md §14](MINER.md#14-marketcast-lane-c-the-composite-mktmarket-target--dec-ca-0064);
when they disagree, §14 and the live `chain.toml` win.

> **Status.** The checked-in `chain.toml` ships MarketCast **off**
> (`[marketcast] from_block = 0`). Until the owner arms it, every `mkt:`
> command is refused (`marketcast_inactive` at the intake). Check the live
> network before spending anything.

## At a glance

| | |
|---|---|
| Target | `mkt:market` — one composite slot over 7 suites (`ohlcv`, `returns`, `vol`, `high`, `low`, `range`, `liquidity`) |
| Model you fine-tune | `mistralai/Timestral-28M@180698bbf5a50dda3516aa6419798437fb03a430` (pinned; never the lane-A king) |
| What you submit | A tree with `marketcast.json` (+ optional `generator.py`) |
| Who pays | You pay the training pod (your provider key). The main validator pays for forecasting and scoring |
| Leg length | Your `leg_hours` (≤ 12), default `[lane_c] leg_hours` = 1 h |
| Training data | Public market bars the trainer fetches itself, ending ≥ 30 days before you submit |
| Judged on | Forward-only windows that open ≥ 1 h after your leg is dispatched (+ embargo-gap windows for `4hx42` / `1dx7`) |
| Time to a verdict | Up to 6 days (`verdict_wait_blocks` = 43200) for enough windows to close |
| To win | Composite gain LCB ≥ 0.01 over the base, no suite / guard / baseline confidently worse, calibration gate, then a **confirmation** on fresh windows and a paired win over the current holder |
| Reward | Lane C's share (25 % of emissions, taken from lane A), paid to the slot holder by a gain-tiered schedule |
| Licence | Timestral's [Modified MIT licence](https://huggingface.co/mistralai/Timestral-28M/blob/180698bbf5a50dda3516aa6419798437fb03a430/LICENSE) restricts companies over USD 20M revenue and their derivatives — read it first |

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
cascade lane-c fetch-base --json                       # pinned Timestral, digest-checked
cascade lane-c snapshot ...                            # public bars → local train / eval snapshots
cascade lane-c test ./my-mkt --target mkt:market \
    --train-snapshot TRAIN --eval-snapshot EVAL --device cuda --json
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
  The days in between are the embargo gap: no leg reads them, and the
  `4hx42` / `1dx7` windows are cut from them.
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
  `generate()` contract).

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
   every eligible forward window of a signed market snapshot and publishes
   the per-window losses and the forecasts at `marketcast/round-<id>.json`.
   Anyone can recompute them with `cascade-audit marketcast --reproduce`.
5. **Judgement.** Every validator derives the same verdict from those rows.
   Not enough windows yet (≥ 200 over ≥ 8 clusters and ≥ 8 six-hour blocks)
   ⇒ you **wait**, re-judged every settlement for up to 6 days.
6. **Pending.** An initial pass is `pending` — no payment yet.
7. **Confirmed.** A later settlement must pass you again on windows that
   started after the first snapshot ended, and you must beat the current
   holder on *your* windows (paired `vs_holder` gain). Then you take the slot.
8. **Payout.** 100 % of the slot for a gain-tiered period (`full_pay_tiers`:
   LCB ≥ 0.05 → 10 eras, ≥ 0.02 → 5, ≥ 0.01 → 2), then halving every era to
   a 10 % floor, stopping 28 eras (14 days) after the win. Unpaid decay and
   empty slots burn.

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
| waiting / `inconclusive` | Too few closed windows yet | Nothing; re-judged every settlement for up to 6 days |
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
your own series.

**Can I see the windows I am scored on?** Not before they happen — that is
the point. After scoring, the signed report and the forecasts are public.

**Where are the rules for X?** [MINER.md §14](MINER.md#14-marketcast-lane-c-the-composite-mktmarket-target--dec-ca-0064)
and the `[marketcast]` / `[lane_c]` tables of the live `chain.toml`.
