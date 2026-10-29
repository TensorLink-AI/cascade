# Mining Cascade v2 (Lane A)

How v2 rounds run, how to submit and pay for your leg, which GPU and provider
to use, how a challenger beats the king, and what happens to today's king at
launch. Full reference: [MINER.md](MINER.md). Signed receipts and
`cascade status --json` are authoritative over this page.

| | |
|---|---|
| Commits open | 2026-10-30 09:00 UTC |
| v2 launch | 2026-10-30 12:00 UTC |
| Netuid | 91 |
| Intake | `https://submissions.cascadesub.net` |
| Model | `cascade-moe-v2-120m`, 1.23B-parameter MoE |

## Rounds and eras

- **Settlement:** every 900 blocks (~3 h). Validators judge every leg that
  has finished.
- **Era:** 4 settlements (~12 h). An era fixes the training contract, the
  starting checkpoint (init) and the seeds, so all legs in one era are
  comparable.
- **Random-init window:** eras 1–14 (the first 7 days). Every leg, the
  king's included, trains from random init. Challengers are judged in
  **level units**, because there is no shared init yet.
- **Warm starts:** after the window, a king that passes its scheduled king
  check is promoted into the shared init. The first promotion comes about
  21,600 blocks (~3 days) into a reign. Gains are then judged in
  **increment units**.

You do not resubmit each round. A funded, revealed entry is admitted when
capacity and the era's remaining time allow, and it carries forward on its
own. A leg is judged in the era it was admitted to. If it lands too late,
it is requeued without using an attempt.

## What a leg trains

Your generator only supplies data. Every leg in an era uses the same model,
recipe and budget.

| Setting | v2 contract |
|---|---|
| Context / horizon | 4096 / 64 |
| Batch | 16 packed 32-channel series per step (512 channel rows), 8 micro-batches |
| Steps | 23,841: 16 × 32 channels × 4096 points ≈ 2.1M points per step, ≈ 50B points in total |
| Wall | 18,000 s (5 h); a leg that hits it is judged on what it trained |
| Generator output | exactly **32 channels** per yield, length 64–4096 |

Data comes first. An operator-paid CPU tier (32 vCPU, 90-minute window)
runs your generator and stores the corpus as shards. Only then is your GPU
rented to train on them. Your provider key pays for the GPU leg only.

## Submitting

1. Write and check a 32-channel generator:
   ```bash
   pip install -e '.[hippius,chain,generation]'
   cascade init ./my-gen      # 32-channel starter
   cascade verify ./my-gen
   ```
2. Register a hotkey:
   ```bash
   btcli subnets register --netuid 91 --network finney --wallet-name W --wallet-hotkey H
   ```
3. Submit privately and fund the leg. The code goes to the operator's
   vault and is published only if it takes the throne. The key is read
   from the environment only:
   ```bash
   export LIUM_API_KEY=...
   cascade submit ./my-gen https://submissions.cascadesub.net \
     --wallet-name W --wallet-hotkey H --label my-gen-v1 --yes
   ```
4. Wait for the reveal. The commitment is timelocked to reveal 25 blocks
   (~5 min) before the next boundary, and the entry queues itself once the
   reveal lands:
   ```bash
   cascade reveal-status YOUR_SS58 --watch
   ```
5. Track the exact entry, then read the verdict:
   ```bash
   cascade status --hotkey YOUR_SS58 --lane a --ref IMMUTABLE_REF \
     --intake https://submissions.cascadesub.net --watch
   cascade duel --hotkey YOUR_SS58 --history
   ```

**One judged submission per hotkey per round in Lane A** (a round is one
900-block settlement, about 3 hours). A new commit replaces the queued one.
Once your leg is judged, or refused for your own fault, that hotkey cannot
submit again in the round your leg started in. From the next round it may
commit and fund a new generator; it must stay registered. A hotkey still
has at most one live leg at a time. Burns from before v2 do not carry over. Losses on
the operator's side never burn. Queue state `done` means training finished, not that you won.

## GPU and providers

Each leg rents **one GPU** on your own provider account. Every type trains
the same budget inside the same wall. A faster card finishes sooner; it
never buys more training.

| Type | Memory | For v2 |
|---|---:|---|
| L40S | 48 GB | preferred |
| L40 | 48 GB | preferred |
| A6000 | 48 GB | fine (slower) |
| RTX4090 | 24 GB | use with care |
| RTX3090 | 24 GB | use with care |

**Not yet measured on v2.** The per-GPU walls on file come from the previous,
smaller contract. The 1.23B model at batch 16 × 32 channels has not been measured on a
24 GB card, so the 48 GB types are the safer choice until GPU qualification
results are published.

The default is the cheapest allowed type under the operator's price caps.
To choose a type, fund (or re-fund) the revealed ref with `cascade fund`:

```bash
cascade fund https://submissions.cascadesub.net --ref IMMUTABLE_REF \
  --wallet-name W --wallet-hotkey H --gpu L40S --max-price 1.20 --yes   # your own USD/GPU-hour cap
cascade fund ... --gpu L40S --allow-fallback --yes   # default type if L40S is sold out
```

- **Price cap:** once you choose a type, the operator's caps no longer
  apply; your `--max-price` does. Without it, you accept the listed price.
- **Sold out:** without `--allow-fallback`, your leg waits for the type.
  Nothing is rented or charged and no attempt is used.
- **Changing your mind:** the latest fund request wins. Funding again
  without `--gpu` resets you to the default type.

**Providers.** Lane A accepts Lium, Shadeform, Vast.ai and Runpod keys,
tried in the order you list them. The next account is used only when the
earlier one has no fitting capacity.

```bash
cascade submit ... --provider-key-env lium=LIUM_API_KEY \
  --provider-key-env shadeform=SHADEFORM_API_KEY --yes
```

- **Lium:** the default. Set a daily or lifetime budget on the key.
- **Shadeform:** each instance gets a spend threshold from the operator's
  leg-cost cap.
- **Vast.ai:** needs a scoped key with instance read/write.
- **Runpod:** skipped whenever a price cap applies, because its API cannot
  check the price before creating a pod.

Keep the key valid and funded until your pod has been torn down, because
teardown uses the same key. A revoked, empty or expired key releases the
entry as `auth`; fix the key and fund again.

## Scoring and the margin

Validators score your checkpoint and the king's on the same private,
rotating windows.

- **Metric:** geometric mean of CRPS and MASE over the private windows, at
  horizons 64, 256 and 720. Lower is better.
- **Test:** a paired bootstrap gives a 95% one-sided lower confidence bound
  (LCB) on your advantage over the king. You win if the LCB clears the
  margin.
- **Several challengers:** a cohort-wide max-T correction keeps the king's
  false-dethrone risk fixed, however many challengers there are.

| When | Units | Bar |
|---|---|---|
| After the random-init window | increment (share of the typical gain over the shared init) | **0.10** against a fresh king, falling linearly to **0.01** over 18,000 blocks (~2.5 days) of tenure, then held |
| Random-init eras | level (share of the king's absolute score) | the level bar; `cascade round` prints it |

The margin never reaches zero, so a challenger that merely ties the king
cannot win on luck.

**Public benchmark gate.** A private-pool win is then checked against the
king on GIFT-Eval, BOOM, TIME and FEV. The gate can only block a win. You
pass when all three hold:

1. On average you are better: the suite-weighted geomean of your CRPS and
   MASE ratios to the king is below 1.
2. No suite is significantly worse: its ratio bound stays within 1 + 1%.
3. You win at least half of the tasks, suite-weighted, with ties counting
   half.

Per-domain gaps in `cascade duel --hotkey` are for diagnosis only. The
verdict uses the pooled statistic.

## Rewards

Lane A holds 100% of the competition vector until Lane B or C launches. The
king is paid a share of A on a ladder, and the unpaid rest is burned.

| Crowning LCB (increment units) | Starting share of A |
|---|---:|
| ≥ 0.50 | 80% |
| ≥ 0.25 | 65% |
| ≥ 0.10 | 50% |
| ≥ 0.01 | 25% |
| crowned in level units (random-init window) | 50% |

- **King checks:** each pass adds 10 points, up to 95%. A failed check
  drops the share to 10%.
- **Former king:** the king just dethroned gets 5% of A for 36,000 blocks
  (~5 days).
- **Own funding:** the king funds its own leg each era. If the king's key
  fails and the operator has to pay, the king is paid 1% of A instead. A
  first lapse lasts at least 6 eras; a second lapse lasts the rest of the
  reign. Operator-side faults never count.

## The current king

v2 does not start with an empty throne. Today's mainnet king, **uid 31**
(`5Gsa19SF…`), carries into v2 as the first era king.

- **Retrained on v2:** its generator was committed before the 32-channel
  rule, so it alone may train at 1–32 channels. The exception covers that
  commitment only. A re-commit, or any newly crowned king, must produce 32
  channels.
- **Capped pay:** it is capped at 5% of total emissions until a v2 verdict
  crowns a king. The rest of A burns in the meantime.
- **Every challenger duels its leg** for that era, trained on the same
  contract, init and seeds.
- **Dethroning it makes your checkpoint the era king at once**, without
  ending the era, and later challengers in that era duel you. The king is
  read from validators' signed receipts, never from the metagraph.

The carried-over king is pinned to uid 31's commitment as of block
9,339,038. If uid 31 is dethroned or re-commits before launch, the operator
updates that pin. Check the live king with `cascade round`.
