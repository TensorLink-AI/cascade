# Cascade v2: what changes for miners

v2 launches on mainnet (netuid 91) at **2026-10-30 12:00 UTC**; commits open
at 09:00 UTC. This page lists only what is different from today's Lane A.
Everything not listed works as before ([MINER.md](MINER.md)). Signed receipts
and `cascade status --json` are authoritative over this page.

## At a glance

| | Before v2 | v2 |
|---|---|---|
| Model | `toto2-4m` | `cascade-moe-v2-120m`, 1.23B-parameter MoE, trained from scratch for 7 days |
| Generator output | 1–32 channels per yield | **exactly 32 channels** per yield |
| Data feed | streamed while the GPU trains | generated first on CPU machines, then trained from stored shards |
| Batch | 64 series | **16 series × 32 channels** per step |
| Budget | per-series points | **≈ 50B points**: 23,841 steps × 16 × 32 × 4096 |
| Submissions | one per hotkey, for life | **one per hotkey per round** |
| Dethrone bar | 0.4 → 0.2 increment units | **1% → 0.5%** level (first 7 days), then **0.10 → 0.01** increment |
| Public benchmarks | GIFT-Eval, BOOM, TIME | adds **FEV**; run on the main validator's GPU |
| Rewards | court split | **king ladder**, see below |
| GPUs | one card per leg | **4×/8× multi-GPU pods** (see below) |

## 32-channel generators

Every yield must be a `(32, L)` array, length 64–4096. A generator that
emits fewer channels is refused before any compute is rented. You may pack
independent groups inside the 32 channels ([INTERFACE.md](INTERFACE.md)).
`cascade init` creates a 32-channel starter; check yours with
`cascade verify`.

The only exception is the carried-over king (below).

## Generation, then training

v2 splits a leg in two:

1. **Generation (CPU, operator-paid).** Your generator runs on rented CPU
   machines (32 vCPU) inside a 90-minute window and its output is stored as
   shards. You pay nothing for this stage.
2. **Training (GPU, your key).** Only when the corpus is ready is the GPU
   rented. Training reads the shards.

**Generation time is not part of the training wall.** The 18,000 s (5 h)
wall starts when GPU training starts. Generation does count when the
trainer decides whether your leg can still finish inside the current era:
it budgets the generation window plus the training wall. A slow generator
therefore costs you a later start, not training time.

## Batch and budget

A step trains 16 packed series of 32 channels at context 4096, about 2.1M
points. 23,841 steps make the ≈ 50B-point budget. Every leg in an era trains
the same model, recipe, init and seeds; your corpus is the only difference.

## One submission per hotkey per round

A round is one 900-block settlement (about 3 hours). Once your leg is
judged, or refused for your own fault, that hotkey cannot submit again in
the round the leg started in. From the next round it may commit and fund a
new generator. It must stay registered, and it can have only one live leg
at a time. Burns from before v2 are lifted. Operator-side losses never burn.

## Rounds and eras

- **Era:** 4 rounds (about 12 hours). An era fixes the contract, the
  starting checkpoint and the seeds.
- **Eras 1–14 (the first 7 days):** every leg, the king's included, trains
  from random init. Wins are measured in level units.
- **After that:** a king that passes its scheduled king check is promoted
  into the shared starting checkpoint, first about 21,600 blocks (≈ 3 days)
  into a reign. Wins are then measured in increment units.

## Scoring

The metric and test are unchanged: the geomean of CRPS and MASE on the
private windows at horizons 64/256/720, and a paired-bootstrap lower
confidence bound (LCB) on your advantage over the king.

- **Margin:** in increment units the bar is **0.10** against a fresh king,
  falling linearly to **0.01** over 18,000 blocks (≈ 2.5 days) of tenure.
  Random-init eras have no init to compare against, so they use the level
  bar: **1%** of the king's score, falling to **0.5%** over 28,800 blocks
  (≈ 4 days). See the [FAQ](V2_FAQ.md) for worked examples.
- **Public benchmark gate:** a private-pool win must also hold up on
  GIFT-Eval, BOOM, TIME and now **FEV**. You pass when you are better on
  average, no suite is significantly worse (within 1%), and you win at
  least half of the tasks. The main validator runs these on its own GPU,
  so your pod does not.

## Rewards

Lane A holds 100% of the vector until Lane B or C launches. The unpaid rest
burns.

| Crowning LCB (increment units) | King's starting share of A |
|---|---:|
| ≥ 0.50 | 80% |
| ≥ 0.25 | 65% |
| ≥ 0.10 | 50% |
| ≥ 0.01 | 25% |
| crowned in level units (first 7 days) | 50% |

- **King checks:** each pass adds 10 points, up to 95%. A failure drops
  the share to 10%.
- **Former king:** the king just dethroned keeps 5% of A for 36,000 blocks
  (≈ 5 days).
- **King's own key:** the king funds its own leg. If that key fails and the
  operator pays, the king gets 1% of A instead: at least 6 eras on a first
  lapse, the rest of the reign on a second.

## The carried-over king

Today's king, **uid 31** (`5Gsa19SF…`), is the first v2 king.

- **Channels:** its generator predates the 32-channel rule, so its current
  commitment alone may train at 1–32 channels. A re-commit, or any newly
  crowned king, must emit 32.
- **Pay cap:** it is capped at 5% of total emissions until a v2 verdict
  crowns a king.
- **Duels:** every challenger in an era duels its v2-trained leg.
  Dethroning it makes you era king at once.

The pin is uid 31's commitment as of block 9,339,038. If uid 31 is
dethroned or re-commits before launch, the operator updates it.

## GPUs

v2 legs train on multi-GPU pods. Each leg rents one pod on your key and
trains on all its GPUs: the 8 micro-batches of every step are split across
the GPUs and their gradients summed, so a 4× and an 8× leg train the same
model, step for step. More GPUs only finish sooner.

| Layout | v2 wall (est.) | $/h | Est. per leg | |
|---|---:|---:|---:|---|
| `8xH200` | 2.3 h | 18.00 | $42 | default |
| `8xH100` | 2.4 h | 24.00 | $59 | default |
| `4xH200` | 3.5 h | 18.00 | $62 | default |
| `4xH100` | 3.7 h | 24.00 | $88 | opt-in |

Walls are the owner's Lium timings for 27,500 updates of the same 512-row
step, scaled to v2's 23,841 steps; prices are advertised Lium rates per pod.

- **Default:** without `--gpu`, the leg rents the cheapest layout whose
  estimated leg costs **$75 or less**.
- **Opt-in:** name a dearer layout with `--gpu`; your own `--max-price`
  (USD per pod-hour) applies instead of the operator's caps:

  ```bash
  cascade fund https://submissions.cascadesub.net --ref IMMUTABLE_REF \
    --wallet-name W --wallet-hotkey H --gpu 4xH100 --max-price 30 --yes
  ```

- **Providers:** layouts rent on **Lium**. Other providers do not offer
  them yet.
- **Blackwell (B200, B300) is not available at launch.** The training image
  is built for CUDA 12.4, which cannot run on Blackwell. `4xB200`,
  `8xB200`, `4xB300` and `8xB300` join when the image moves to CUDA 12.8.
