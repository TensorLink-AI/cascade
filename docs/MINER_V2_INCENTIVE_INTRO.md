# Cascade v2: what changes for miners

v2 has been live on mainnet (netuid 91) since **2026-10-30**: commits opened
at 09:00 UTC and the first v2 era started at the first era boundary after
12:00 UTC (block 9,399,600). This page lists only what differs from the
pre-v2 Lane A.
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
| GPUs | one card per leg | **8-GPU pods: `8xH100` or `8xH200`** (see below) |

## 32-channel generators

Every yield must be a `(32, L)` array (deployed `[generator] min_channels =
max_channels = 32`). A generator that emits any other width is refused
before any compute is rented. The trainer never pads or merges channels, so
32 is forced on you: if your process has fewer related channels, pack
independent series into the 32 rows yourself. Optional `groups` labels mark
which rows belong together ([INTERFACE.md](INTERFACE.md#pack-independent-groups-in-one-yield)).
Length: 64–4096 in ordinary verification; the v2 CPU shard tier accepts
longer series (check `cascade verify --live`), which training crops into
random 4096-point windows (`[v2.training] long_series = "crop"`).
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

A step trains 16 series of 32 channels at context 4096, about 2.1M
points (`[v2.training] batch_size = 512` counted in `"sequences"`: 512 ÷ 32
channels = 16 series, split into 8 micro-batches). 23,841 steps make the ≈ 50B-point budget. Every leg in an era trains
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

The private pool on mainnet is built only from the
[TSBench-Forge](https://github.com/tensorlink-dev/TSBench-Forge) live
catalog, labelled with the 7 GIFT-Eval domains (`econ_fin`, `energy`,
`healthcare`, `nature`, `sales`, `transport`, `web_cloudops`). Retired
snapshots are revealed with a lag at
[Tensor-Link/cascade-eval-pool](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool);
use them as held-out data for `cascade score`/`iterate --pool-dir`, never as
training data ([MINER.md](MINER.md#held-out-data-the-public-eval-pool)).

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

The king at launch, **uid 31** (`5Gsa19SF…`), was carried over as the first v2 king.

- **Channels:** its generator predates the 32-channel rule, so its current
  commitment alone may train at 1–32 channels. A re-commit, or any newly
  crowned king, must emit 32.
- **Pay cap:** it is capped at 5% of total emissions until a v2 verdict
  crowns a king.
- **Duels:** every challenger in an era duels its v2-trained leg.
  Dethroning it makes you era king at once.

The pin is uid 31's commitment as of block 9,339,038
(`[generator] king_min_channels_refs`).

## GPUs

v2 legs train on multi-GPU pods. Each leg rents one pod on your key and
trains on all its GPUs: the 8 micro-batches of every step are split across
the GPUs and their gradients summed, so every layout trains the same model,
step for step. More GPUs only finish sooner.

**Rentable on mainnet today: `8xH100` and `8xH200` only** (deployed
`[round] funded_pod_skus`). A `--gpu` outside that list is refused at
funding (`unknown_gpu`); 4× layouts and Blackwell are not currently offered.

| Layout | Measured wall (mainnet) | Lium $/h actually charged | Typical leg | **Plan for** |
|---|---:|---:|---:|---:|
| `8xH100` | ≈ 3.5 h | $14.40–23.20 | ~$60–95 | **up to ~$105** |
| `8xH200` | ≈ 3.5 h | $24–46 (median ~$31) | ~$125 | **up to ~$210** |

*Measured 2026-11-01/02 on mainnet:* a full v2 leg is 23,841 steps, and
legs ran at about 112–115 steps per minute on **both** 8xH100 and 8xH200,
so ≈ 3.5 h of GPU training either way. Prices are what Lium charged for
8× pods on 2026-11-01/02, not advertised rates; they move by the hour. The
trainer's own planning walls (deployed `funded_sku_wall_seconds`, measured
2026-10-31 with a 5% margin) are 12,270 s (3.4 h) for 8xH200 and 13,600 s
(3.8 h) for 8xH100.

- **Typical leg** = price × (wall + ~30 min boot, data and upload).
- **Plan for** = the highest hourly rate you will accept × (wall + 1 h),
  i.e. × 4.5 h. Use that number for `cascade preflight --max-spend` and for
  your Lium key budget. Standard Lane A funding enforces only the hourly
  cap (`--max-price`, or the operator caps below); preflight's
  `--max-spend` is a review input, not an enforced total.
- A slow host can run to the 18,000 s (5 h) hard training wall, and a leg
  that fails on a bad host and reruns also bills the pod time it already
  used. Keep room for one rerun if your key's budget is tight.

- **Default:** without `--gpu`, the leg rents the cheapest layout under the
  operator's caps: currently **$40 per pod-hour** and an estimated
  **$110 per leg** (price × the planning wall above; read the live values
  with `cascade status --json`). At recent prices that is usually `8xH100`;
  an 8xH200 above ~$32/h fails the $110 leg cap.
- **Opt-in:** name a layout with `--gpu`; your own `--max-price` (USD per
  pod-hour, compared with the pod's listed hourly price) applies instead of
  the operator's caps:

  ```bash
  cascade fund https://submissions.cascadesub.net --ref IMMUTABLE_REF \
    --wallet-name W --wallet-hotkey H --gpu 8xH200 --max-price 35 --yes
  ```

- **Providers:** layouts rent on **Lium**. Other providers do not offer
  them yet.
- **Blackwell (B200, B300) is not available.** The training image is built
  for CUDA 12.4 (`torch==2.4.1+cu124`), which cannot run on Blackwell; a
  CUDA 12.8 image has not shipped. The `4xB200`/`8xB200`/`4xB300`/`8xB300`
  entries in the repository's wall table are not rentable until both the
  image and `funded_pod_skus` change.
