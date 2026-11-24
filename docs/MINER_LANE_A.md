# Lane A miner guide: synthetic-data generators

This is the competition brief for lane A. The full walkthrough (install,
verify, submit, provider keys, troubleshooting) is [MINER.md](MINER.md), and
the generator contract is [INTERFACE.md](INTERFACE.md). The live numbers are
in `cascade rules a`, read from the same `chain.toml` the trainer and
validators load.

```bash
cascade rules a            # headline lane A rules
cascade rules a --all      # every [v2] and [scoring] key
cascade round              # the live bar, countdown and king
```

## Intent

Lane A competes on **data alone**. The model, its training recipe and its
token budget are fixed. You write a Python generator of synthetic time
series. The trainer trains the shared model on your data, and that model is
judged on real series it has never seen. Better synthetic data means a better
forecaster. The winning generator becomes the king, and its model can become
the starting point for everyone's next era.

## What you submit

- **A deterministic, sandboxed Python generator** (no network, no weights),
  emitting finite float arrays of exactly 32 channels per yield. See
  [INTERFACE.md](INTERFACE.md) for shapes, dependencies and limits.
- **Your own compute key.** The trainer rents the leg on your provider key
  (Lium, Shadeform, Vast.ai or Runpod). Mainnet rents `8xH100` or `8xH200`
  pods. Picking a faster GPU type finishes sooner but never buys more
  training.

```bash
cascade verify ./my-generator
cascade iterate ./my-generator              # local score vs the king, never spends
export LIUM_API_KEY=...                      # env only, never an argument
cascade submit ./my-generator https://submissions.cascadesub.net \
    --wallet-name my-miner --wallet-hotkey gen1 --label my-gen --yes
cascade status --hotkey <ss58> --lane a --json
cascade duel
```

`cascade submit` keeps your code private unless it takes the throne. The
public path is `cascade deploy` plus `cascade fund`.

## How it is scored

- **Metric.** The geometric mean of CRPS and MASE over private windows, on
  horizons 64, 256 and 720.
- **The bar.** The paired-bootstrap LCB of your gain over the king, in
  increment units (gain over the shared init), must clear the dethrone
  margin. On v2 settlements it is `win_margin_start` (0.10) against a fresh
  king and decays to `win_margin_end` (0.01) over `margin_warmup_blocks`
  (18,000 blocks, about 2.5 days) of tenure. With several challengers, a
  cohort-wide max-t correction keeps the king's false-dethrone risk fixed.
- **The public bench gate** (`bench_gate_mode`) can only block a win. On
  GIFT-Eval, BOOM, TIME and FEV you must be better on average, no suite may
  be significantly worse (by more than `bench_gate_tolerance`, 1%), and you
  must win at least `bench_gate_min_win_rate` (half) of the tasks.

## How you are paid

**Under the owner-proposed split A0 / B35 / C15 / D50 (plan rev6.24, signed
overlay), lane A's pool is 0% once lane D is armed.** A lane A king then
earns no emissions from lane A. Its model still matters: the lane A king is
the **house** entry in lane D, and promoted kings become the shared init for
later eras. Until lane D arms, lane D's share stays with lane A.

Under today's checked-in template (all of A, B and C launched, A at 50%) the
v2 ladder applies:

| Event | Share of A's pool |
|---|---:|
| Crowned, LCB ≥ 0.5 / 0.25 / 0.1 / 0.01 (increment units) | 80% / 65% / 50% / 25% |
| Each passed king check | +10 points, up to 95% |
| A failed king check | set to 10% |
| Immediate former king, for `court_blocks` (36,000 blocks, about 5 days) | 5% |

**Worked example (template, A at 50%).** A king crowned at LCB 0.25 starts
at 65% of A, which is 32.5% of emissions. After two passed checks it is at
85%, or 42.5%. Its predecessor gets 2.5% for about 5 days. The rest of A
burns. See [the reward schedule](MINER.md#rewards-allocation-burn-and-progression)
for key lapses and the inherited-king cap.

## Strategies worth trying

- **Study what the king's data lacks.** Receipts carry a per-domain
  breakdown (`per_domain`). Domains where the king loses are where new data
  helps most.
- **Diversity over volume.** The token budget is fixed, so a generator that
  covers more realistic seasonalities, trends, regime changes and noise
  shapes usually beats one that repeats a single family.
- **Use the 32 channels.** Group packing lets one yield carry several
  independent or related series. Related channels teach cross-series
  structure (see [multichannel generators](MINER.md#11-multichannel-generators)).
- **Iterate cheaply.** Use `cascade score` / `cascade iterate` on the small
  Cascade MoE Base presets (17.58M, 69.02M) first, then recheck the best idea
  with `--warm-start live` before you pay for a live leg.
- **Mind the bench gate.** A pool win that is worse on a public suite is
  blocked. Keep enough general-purpose data in the mix.

## Costs

You pay the provider for one full leg per entry. A full 23,841-step leg of
the 1.23B model takes about 3.5 h on 8×H100 or 8×H200 (mainnet,
2026-11-01/02). Multiply by your provider's listed pod price. The operator's
price caps and your `--max-price` apply as described in
[MINER.md](MINER.md#choosing-your-gpu-dec-ca-0060). A losing leg is not
refunded.

## What gets you refused

- **Verification:** non-determinism, wrong shapes or channel counts,
  disallowed imports, code hidden in strings, or a tree over the size limit.
  Run `cascade verify` first.
- **Reveal:** a ref that does not match your latest revealed commitment
  (`ref_mismatch`), or a reveal before the opening.
- **Funding:** no `--yes` (exit 5), or a GPU type outside `[round]
  funded_pod_skus` (`unknown_gpu`).
- **Duplicates:** byte-identical or invalid source can be refused before
  training.

See [failure classes](MINER.md#8-failure-classes) for the full list.

## FAQ

**Is lane A still worth mining under A0?** It pays nothing directly under
that split. Its kings feed lane D (as the house) and the shared init. Mine it
for that, or move to lane B, C or D for direct pay.

**Can I change the model or recipe?** No. That is lane D.

**Does funding guarantee a settlement?** No. It makes the entry eligible.
Capacity and the era's remaining time decide when it trains.

**Where is the live bar?** `cascade round` prints it, and each receipt
records the rule it was judged under.
