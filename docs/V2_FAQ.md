# Cascade v2 FAQ

Answers to miners' questions about v2 (live on mainnet, netuid 91, since
**2026-10-30**: commits opened 09:00 UTC, the first v2 era started at the
first era boundary after 12:00 UTC, block 9,399,600). For what changes overall,
see [MINER_V2_INCENTIVE_INTRO.md](MINER_V2_INCENTIVE_INTRO.md). Signed
receipts and `cascade status --json` are authoritative over this page.

## Competition rules

### Does Lane A keep the competition rules, such as the LCB margin?

Yes. A challenger still has to beat the king on the private pool with a
paired-bootstrap lower confidence bound (LCB, 95% one-sided) that clears a
margin, and then pass the public benchmark gate. What changes in v2 is the
margin schedule (next question) and that FEV joins the benchmark gate.

### What is the dethrone margin, and what is it relative to?

It depends on whether the era has a shared starting checkpoint (init).

**Random-init eras (the first 14 eras, about 7 days): level units.** There
is no shared init to measure gains against, so the statistic is the
relative improvement on the private geomean:

    stat = (king − challenger) / king

The bar is **1%** against a fresh king, falling linearly to **0.5%** over
28,800 blocks (about 4 days) of the king's tenure. The `[v2]
win_margin_start = 0.10` bar does **not** apply here; validators switch to
the level bar whenever a round has no init to score.

Example: the king's geomean is 0.800. A fresh king needs the LCB of
`(0.800 − challenger) / 0.800` to reach 0.01, so the challenger's geomean
must be about 0.792 or lower even at the conservative end of the bootstrap.

**Warm-start eras (after a king is promoted into the shared init):
increment units.** The shared init is scored as a third model, and the
gap is measured against how much the two models improved on it:

    Δking = init − king
    Δchal = init − challenger
    unit  = max((|Δking| + |Δchal|) / 2, 0.01 × init)
    stat  = (Δchal − Δking) / unit

So 0.10 means "10% of a typical improvement over the init", not 10% of the
absolute score. The bar is **0.10** against a fresh king, falling linearly
to **0.01** over 18,000 blocks (about 2.5 days) of tenure.

Example: the init scores 0.800 and the king 0.780 (Δking = 0.020). To clear
0.10, `(Δchal − 0.020) / ((Δchal + 0.020) / 2) ≥ 0.10`, so Δchal ≥ 0.0221:
the challenger needs about 0.778 or lower at the LCB. That is roughly 0.3%
better than the king in absolute terms.

`cascade round` prints the live bar and units, and every receipt records
the rule it was judged under.

## The king

### Which generator is the first v2 king?

The mainnet king at launch, **uid 31** (`5Gsa19SF…`), with this
commitment (deployed `[generator] king_min_channels_refs`):

    vault/direct@sha256:04e226724e3be088e01134a189598300b9c381779df59b159059c62f5afec73e

It carries into v2 as the first era king. Its v2 leg is trained on the v2
contract from random init, like every other leg. The current king can
change at any settlement; check it with `cascade round`.

### Is its code published, so I can test against it?

Yes. It is public now:

```bash
cascade fetch vault/direct@sha256:04e226724e3be088e01134a189598300b9c381779df59b159059c62f5afec73e
cascade iterate ./my-gen --baseline king
```

`--baseline king` resolves the current king (it never picks the burn uid).
You can also pass the ref above directly with `--baseline <ref>`.

### The king's generator yields (1, L) series, but v2 refuses anything that isn't 32 channels. How is the king evaluated?

The 32-channel rule is for training data, not for evaluation.

- **Training:** the king's current commitment is the one ref allowed to
  train at 1–32 channels (`[generator] king_min_channels_refs`). Every other
  generator, including that king's own re-commit and any newly crowned
  king, must emit exactly 32 channels.
- **Evaluation:** validators judge the trained checkpoint, never the
  generator's width. The king's model and every challenger's model are
  scored on the same private windows, with the same horizons and the same
  test.

### What is the king paid at launch?

The carried-over king is capped at 5% of total emissions until a v2 verdict
crowns a king. The rest of Lane A burns until then.

## Submitting

### Can a hotkey be reused? The code says `one_submission_per_hotkey = true`.

It can, once per round. The setting stays `true`, but since PR #382 it
means **one Lane A submission per hotkey per round**, not per lifetime. A
round is one 900-block settlement (about 3 hours). After your leg is judged,
or refused for your own fault, the hotkey may commit and fund again from
the next round. It must stay registered and can have only one live leg at a
time. Burns from before v2 are lifted. This has applied since launch.

### Can a commit made before launch be judged under v1 rules?

No (this mattered on launch day, 2026-10-30). `[round] commit_floor_block` is the v1 floor and is kept only so v1
history replays. v2 has its own floor, `[v2] commit_from_block`, resolved
from the 09:00 UTC opening (`commit_open_at_utc`).

- **Before the opening:** the intake answers `v2_submissions_not_open` and
  nothing is queued or rented.
- **Revealed before the opening:** refused with `v2_commit_before_open`.
  Make a fresh commitment; no attempt is used.
- **Between 09:00 and 12:00 UTC:** entries queue and wait. No CPU
  generation or GPU training runs under v1.

Check `golive.commit_from_block` and `golive.admission_state` in
`cascade status --json`.

## Generation and hardware

### Which packages are available when my generator runs?

The generation environment installs the `generation` extra from the frozen
`uv.lock`:

| Package | Version |
|---|---|
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| pandas | 3.0.3 |
| pyarrow | 25.0.0 |
| torch | 2.4.1 |
| networkx | 3.6.1 |
| numba | 0.66.0 |
| scikit-learn | 1.9.0 |
| statsmodels | 0.14.6 |
| gpytorch | 1.15.2 |

Before any miner code runs, each machine runs
`scripts/check_generation_runtime.py`, which imports every library and
prints its version. A missing library fails provisioning; it does not
reject your generator. Anything else must come from your
`requirements.txt` and be on the dependency allowlist, hash-locked.

### Does generation time count toward the 5-hour wall?

No. Generation runs first on CPU machines (32 vCPU, 90-minute window, paid
by the operator). The 18,000 s wall starts when GPU training starts. The
generation window does count when the trainer decides whether your leg can
still finish inside the current era, so a slow generator gets a later
start, not less training.

### Is the architecture final? What about loss masking, the output head and the smaller variants?

The launch architecture is the one pinned in `chain.toml` `[v2.training]`.
It is part of the contract digest, so it cannot change silently.

- **Model:** `cascade-moe-v2-120m`, 1.23B total parameters, d_model 1088,
  10 layers, patch size 32, context 4096.
- **Loss masking:** `contiguous_patch`.
- **Output head:** a 21-quantile head with a copula loss term.
- **Smaller variants:** Cascade MoE Base - 17.58M, 69.02M and 273.84M
  (earlier aliases `2m`, `7m`, `28m`) are in `deploy/presets/` for cheaper
  local iteration with `cascade score` and `cascade iterate`;
  `python scripts/pyramid_config.py --size 17.58m --out ./chain.local-17m.toml`
  writes a complete `--chain-toml`.

Any later architecture change is a new contract, announced in advance with
a new digest.

### What GPU memory and throughput have been measured?

Wall time for the 1.23B competition model is measured on mainnet; GPU
memory and the smaller variants' speed are not.

- **Measured 2026-11-01/02:** a full leg is 23,841 steps; legs ran at about
  112–115 steps per minute on **both** 8×H100 and 8×H200, ≈ 3.5 h of GPU
  training either way. Legs train data-parallel across all 8 GPUs.
- **Rentable layouts:** only `8xH100` and `8xH200` (deployed `[round]
  funded_pod_skus`). 4× layouts are not offered on mainnet, and Blackwell
  (B200, B300) needs a CUDA 12.8 image that has not shipped.
- **Default choice:** without `--gpu` a leg rents the cheapest layout under
  the operator's caps (currently $40 per pod-hour and an estimated $110 per
  leg; see `cascade status --json`). That is usually 8×H100.
- **Prices Lium actually charged, 2026-11-01/02:** 8×H100 $14.40–23.20/h;
  8×H200 $24–46/h (median ~$31). Budget the highest hourly rate you would
  accept × (wall + 1 h): about $105 for 8×H100 or $210 for 8×H200 at the
  top of those ranges.

The table is in [MINER_V2_INCENTIVE_INTRO.md](MINER_V2_INCENTIVE_INTRO.md#gpus).

### What exactly is the training geometry?

Every yield is `(32, L)`: deployed `[generator] min_channels =
max_channels = 32`. The trainer never pads or merges channels, so a process
with fewer channels must be packed into the 32 rows by your generator
(optional `groups` labels say which rows belong together; see
[INTERFACE.md](INTERFACE.md#pack-independent-groups-in-one-yield)). Each step
trains 16 series × 32 channels at context 4096 (`[v2.training] batch_size =
512` counted in sequences, 512 ÷ 32 = 16, in 8 micro-batches), for 23,841
steps (≈ 50B points). Series longer than 4096 are cropped into random
4096-point windows (`long_series = "crop"`).

## Local scoring and the eval pool

### Where do I get held-out data to score against?

Use the public eval pool:
[Tensor-Link/cascade-eval-pool](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool).
It is a lagged public reveal of **retired** held-out eval pool snapshots,
byte-identical to what validators scored. Use the
`snapshots/<as_of>-block-<N>/` folders; the legacy `snapshots/<YYYY-MM-DD>/`
folders (through 2026-08-03) are rebuilds, not byte mirrors. The latest
revealed snapshot is `snapshots/2026-10-31-block-9388800` (3,841 series,
sha256 `cb9b64d24f30fc41321058bb56583a60986a0a8e480bd5967962d21fe6896649`).
Download and score instructions are in
[MINER.md](MINER.md#3-verify-and-score-locally).

It is an **evaluation** set: use it as held-out data for scoring, never as
generator training data. The live private pool is newer and never revealed,
so local scores are directional.

### What does the private pool contain?

On mainnet the pool is built only from the
[TSBench-Forge](https://github.com/tensorlink-dev/TSBench-Forge) live catalog
(`tsbench_forge` source, via the bucket relay), labelled with the 7
GIFT-Eval domains: `econ_fin`, `energy`, `healthcare`, `nature`, `sales`,
`transport`, `web_cloudops`. Other sources the pool builder supports are not
used on mainnet ([EVAL_POOL.md](EVAL_POOL.md#sources-shipped)).

### What do `cascade score` and `cascade iterate` do, and not do?

- They really train: the same trainer as live legs (`Toto2Trainer`, which
  hands the v2 contract to the pyramid MoE trainer), from random init or
  `--warm-start live`.
- The budget is `--train-hours`, default `[round] heat_train_hours` (1 h):
  the step count is scaled from the full leg (23,841 steps over 3 h of
  reference compute), and the run also stops at that many hours of wall
  time (at least 900 s). `--train-hours 0.01` (79 steps) is a smoke test.
- Your generator runs locally **without the sandbox**, as your own trusted
  code.
- They score only on the pool you give them (`--pool-dir`, else `--pool`,
  else a small synthetic sample).
- They do **not** run the public GIFT-Eval/BOOM/TIME/FEV benchmarks. Those
  run in the separate `benchmarks/` sidecar ([benchmarks/README.md](../benchmarks/README.md)
  shows how to run it on a checkpoint). On mainnet the main validator runs
  them on its own GPU only for a private-pool winner, as a gate that can only
  block that win.

## Lanes

### In what order do the lanes launch?

**A, then C, then B.** Lane A launches with v2 and holds 100% of the vector.
Lane C (MarketCast) is next and takes its 25% from Lane A
(`[lane_c] takes_from = "lane_a"`), so it does not need Lane B. Lane B
comes after, giving A/B/C 50/25/25. Each lane is switched on separately.

The rule that Lane C needs Lane B live applies only to the legacy setting
where C takes its share from B, which mainnet does not use.
