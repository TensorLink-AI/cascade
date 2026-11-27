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

> **Lane A is closed to new challengers from block 9,586,800** (mainnet
> launch, about 2026-11-25 22:28 UTC). From that block the intake refuses a new
> lane A fund or funded submit (`409 lane_ended`, nothing charged), the trainer
> seats no new challenger, and king checks and promotions stop
> (`[v2] reign_warm_start_blocks = 0`). The lane A king (uid 164) is frozen as
> the reference model for lanes B and D. `cascade deploy` and `cascade submit`
> refuse a lane A entry past the cutoff before uploading or committing
> (`--allow-closed-lane-a` overrides; the entry still will not be seated).
> In-flight legs, the king leg and king-only settlements continue. Compete in
> [lane B](MINER_LANE_B.md), [C](MINER_LANE_C_MARKETCAST.md) or
> [D](MINER_LANE_D.md). The rest of this brief describes lane A as it ran.

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
public path is `cascade deploy` plus `cascade fund`. Both refuse a new lane A
entry from block 9,586,800 (see the notice above).

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

**From block 9,586,800 the step 1 launch overlay sets A 50 / B 35 / C 15,
and the frozen lane A king keeps its ladder share of A's 50% (no further
king checks change it); the rest of A burns.** A second overlay arms lane D
(at 9,586,800 if it locks in, otherwise at a later era start) and sets the
owner-proposed split A0 / B35 / C15 / D50. The lane A king is then the
**house** entry in lane D, and D's 50% stays with it until the first miner is
crowned lane D king. With promotions off, no king becomes a new shared init.

The v2 ladder below set the king's share while lane A ran. On the checked-in
template (all of A, B and C launched, A at 50%) it reads:

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

## Past evaluations

Every settlement's lane A result is public and anonymously readable (no
keys) in the `cascade-manifests` bucket at
`https://s3.hippius.com/cascade-manifests/<key>`. Settlements are about 3 h
apart. A round id is a seed, not a counter, so find rounds through the index
rather than computing them.

| What | Key | Notes |
|---|---|---|
| Round index | `receipts/index.json` | Rolling list of the last 400 settlements, oldest first. Each row has `round_id`, `published_at`, `status`, `reject_reason`, `receipt_key`, and the verdict summary. Mirrored at the HF dataset [`tensorlink-dev/cascade-backup`](https://huggingface.co/datasets/tensorlink-dev/cascade-backup/resolve/main/receipts/index.json) |
| Latest receipt | `receipts/latest.json`, or `receipts/<validator_hotkey>/latest.json` | The newest signed receipt |
| One round's receipt | `receipts/<validator_hotkey>/round-<round_id>.json` | Signed. Carries the full verdict, the per-window `entry_scores`, the round `manifest` inline, the `eval_context` (pool snapshot, window ids) and the weights |
| Public bench report | `bench_v/round-<round_id>.json` | Signed by the main validator. GIFT-Eval, BOOM, TIME and FEV CRPS/MASE per checkpoint (`role` = `king`, `challenger` or `init`). It is empty (`entries: []`) when nothing was benched that settlement. Per-task rows are at each entry's `rows_key` |

The trainer's own `manifests/round-<id>.json` and `benchmarks/round-<id>.json`
are not public. Each receipt embeds the round manifest it judged.

**Am I competitive?** In a `status: "scored"` row of `receipts/index.json` (or
the receipt's `verdict`), compare `chal_geomean` with `king_geomean` (lower is
better), and `lcb` with `margin`. You dethrone only when `lcb` clears
`margin` and `dethroned` is true. `cohort_lcbs` gives one LCB per challenger
when several were judged. `per_domain` and `per_horizon` (receipt `verdict`
fields `cohort_per_domain` / `cohort_per_horizon`) show where you won or lost.
Rows with `reject_reason: "no_king_challenger_pair"` were king-only
settlements: nothing was judged.

Examples (all verified readable on 2026-11-27):

- Last scored duel, 2026-11-25 07:41 UTC (king uid 164 held against uid 107,
  `lcb` −0.47 vs `margin` 0.01):
  <https://s3.hippius.com/cascade-manifests/receipts/5FcCsoB2R5M55RdrNrvPQv7xmLJRFzs7R7C73UUSxgUCyfsw/round-764824171903801846.json>
- A king-only settlement, 2026-11-27 07:34 UTC:
  <https://s3.hippius.com/cascade-manifests/receipts/5FcCsoB2R5M55RdrNrvPQv7xmLJRFzs7R7C73UUSxgUCyfsw/round-13004096062141715539.json>
- Last non-empty bench report (king vs init), 2026-11-24 13:43 UTC:
  <https://s3.hippius.com/cascade-manifests/bench_v/round-13058784948867351860.json>
- Index: <https://s3.hippius.com/cascade-manifests/receipts/index.json>

## FAQ

**Is lane A still open?** No. It takes no new challengers from block
9,586,800; the intake answers `409 lane_ended`. Move to lane B, C or D.

**Can I change the model or recipe?** No. That is lane D.

**Does funding guarantee a settlement?** No. It makes the entry eligible.
Capacity and the era's remaining time decide when it trains.

**Where is the live bar?** `cascade round` prints it, and each receipt
records the rule it was judged under.
