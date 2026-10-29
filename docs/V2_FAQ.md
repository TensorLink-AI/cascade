# Cascade v2 FAQ

Answers to miners' questions before the v2 launch (mainnet, netuid 91,
**2026-10-30 12:00 UTC**; commits open 09:00 UTC). For what changes overall,
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

Today's mainnet king, **uid 31** (`5Gsa19SF…`), with its current
commitment:

    vault/direct@sha256:04e226724e3be088e01134a189598300b9c381779df59b159059c62f5afec73e

It carries into v2 as the first era king. Its v2 leg is trained on the v2
contract from random init, like every other leg. If uid 31 is dethroned or
re-commits before launch, the operator updates this pin; check with
`cascade round`.

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
time. Burns from before v2 are lifted. This ships with the launch release.

### Can a commit made before launch be judged under v1 rules?

No. `[round] commit_floor_block` is the v1 floor and is kept only so v1
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
- **Smaller variants:** `cascade-moe-v2-2m`, `-7m` and `-28m` are released
  in `deploy/presets/` for cheaper local iteration with `cascade score` and
  `cascade iterate`.

Any later architecture change is a new contract, announced in advance with
a new digest.

### What GPU memory and throughput have been measured?

None yet for the 1.23B v2 model. Published per-GPU wall times come from the
previous, smaller contract. Multi-GPU layouts (4×/8× B200, B300, H200,
H100) are planned but not live at launch. The table and the ≤ $75 default
rule are in [MINER_V2_INCENTIVE_INTRO.md](MINER_V2_INCENTIVE_INTRO.md#gpus-planned).
Until they ship, a Lane A leg rents one GPU from `[round] funded_pod_skus`.
Prefer the 48 GB types (L40S, L40, A6000).

## Lanes

### In what order do the lanes launch?

**A, then C, then B.** Lane A launches with v2 and holds 100% of the vector.
Lane C (MarketCast) is next and takes its 25% from Lane A
(`[lane_c] takes_from = "lane_a"`), so it does not need Lane B. Lane B
comes after, giving A/B/C 50/25/25. Each lane is switched on separately.

The rule that Lane C needs Lane B live applies only to the legacy setting
where C takes its share from B, which mainnet does not use.
