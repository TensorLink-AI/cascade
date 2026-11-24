# Lane B miner guide: domain and suite experts

This is the competition brief for lane B. The step-by-step walkthrough and
every refusal code are in [MINER.md lane B](MINER.md#13-cascade-v2-and-domain-experts-lane-b).
The live numbers are in `cascade rules b`, which reads the same `chain.toml`
the trainer and validators load.

```bash
cascade rules b            # headline lane B rules
cascade rules b --all      # every [lane_b] key
cascade status --json      # whether lane B (and suite targets) are armed now
```

## Intent

The shared v2 model is a mixture of experts. Lane B asks you to **make a few
experts better at one job** (a domain such as energy, a horizon, or a public
benchmark suite) without making the model worse anywhere else. Your
generator's data trains only the expert rows you pick, with the router frozen.
If the gain is real, the update merges into the shared model and every later
miner builds on it.

## What you submit

- **A code-only generator** (the lane A interface, see
  [INTERFACE.md](INTERFACE.md)), committed with up to
  `max_domains_per_leg` (3) targets:
  - **domains:** econ_fin, energy, healthcare, nature, sales, transport,
    web_cloudops;
  - **horizons:** `horizon:64`, `horizon:256`, `horizon:720`;
  - **suites:** `suite:gift`, `suite:boom`, `suite:time`, `suite:fev`, when
    `suite_targets` is on.
- **Optionally, declared expert rows** (`--declared-rows FILE`): the experts
  you train, keyed by MoE layer ordinal. Per layer you may take between about
  3% and 15% of its experts (128 → 4..19, 256 → 8..38, 512 → 16..76). Rows
  are exclusive: the earliest reveal holds them, and a later leg on the same
  rows waits (`rows_held`).
- **Your leg levers** (plan rev6.21), all optional, signed with the fund
  request:
  - **learning-rate scale** 0.5–2.0 (default 1.0);
  - **curriculum** of 1–3 named stages whose fractions sum to 1;
  - **context length** 4096 (the only value accepted today; 8k–32k are
    refused `context_length_unsupported` while the pinned model's context is
    4096);
  - **EMA width** σ_rel from {0.03, 0.05, 0.10, 0.15, 0.20} (default 0.10).
    Legs train with a power EMA and no LR cooldown (plan rev6.20), and the
    EMA weights are what is scored;
  - **a token limit and/or an hour limit**, up to 12 h. The leg stops at
    whichever comes first.
- **Your own compute key.** The trainer rents the leg on your provider key.
  The lane B GPU list is `funded_pod_skus` (8×H200, 8×H100, 4×H200, 4×H100,
  8×RTX4090). With a token or hour limit set, any Lium GPU type may be chosen
  at your own risk (recorded `gpu_measured: false`; a failure there is
  `own_risk_gpu_failed`, with no requeue or refund).

```bash
cascade lane-b verify-repo ./my-generator           # the contamination scan, locally
cascade lane-b test ./my-generator --targets energy,h720 --base live --pool-dir ./pool
cascade deploy ./my-generator --hub-namespace my-ns --expert-domain energy,h720 \
    --wallet-name my-miner --wallet-hotkey expert1
export LIUM_API_KEY=...   # env only, never an argument
cascade lane-b fund https://submissions.cascadesub.net --targets energy,h720 \
    --ref <repo@digest> --declared-rows rows.json --gpu 8xH100 \
    --wallet-name my-miner --wallet-hotkey expert1 --yes
cascade lane-b status https://submissions.cascadesub.net --hotkey <ss58>
```

One commitment funds one leg. Re-funding a spent commitment returns `spent`.

## How it is scored

- **Domain and horizon targets.** Your merged model is compared with the
  current model on the private pool. Per target you need a positive point
  gain and an LCB ≥ `gain_margin` (1%) at `alpha` / k, where k is the number
  of targets you named. Each target needs `min_target_windows` (100) windows
  from `min_target_clusters` (5) clusters.
- **Suite targets (plan rev6.19).** A suite target passes when **CRPS or
  MASE** improves with LCB ≥ 1% (at alpha / 2k) and a positive point gain,
  and the other metric is non-inferior (LCB ≥ −1%). The full-pay tier uses
  the smaller cleared LCB.
- **Guards, for every target.** No other domain may regress by more than
  `tau_domain` (1%). No public suite in `require_public` (GIFT-Eval, BOOM,
  TIME, FEV) may regress by more than `tau_public` (1%). One failed guard
  blocks every target in the leg.
- **Acceptance.** A qualified leg earns nothing until its merge is accepted
  at a warm start. A suite target that is the era's sentinel waits
  (`sentinel_target`) for up to `sentinel_max_defer_eras` (4) eras.

## How you are paid

Each target is a **slot**. Lane B's pool is split equally over its slots:
10 today (7 domains + 3 horizons), 14 with suite targets on. A slot holder
is paid the **full slot with no decay and no time limit** until a newer leg
passes on top of it (plan rev6.22). A tie or an inconclusive challenger keeps
the holder. One hotkey may hold several slots.

**Worked example at the owner-proposed split A0 / B35 / C15 / D50.** This is
the owner's proposal (plan rev6.24, signed overlay), not a guaranteed live
value. Check `cascade rules b` for `share`.

| Lane B slots | One slot's share of emissions | You hold energy + h720 |
|---|---:|---:|
| 10 (suite targets off) | 35% / 10 = 3.5% | 7% |
| 14 (suite targets on) | 35% / 14 = 2.5% | 5% |

An empty slot burns. It is never shared out to other holders.

## Strategies worth trying

- **Target where the king is weakest.** The receipts' `per_domain` numbers
  and the bench preview show where the current model loses. A 1% gain is
  easier on a weak domain or suite than on a strong one.
- **Choose experts by routing affinity.** Declare the rows the target's data
  already routes to (the routing artefact in
  [AGENT_MINING.md](AGENT_MINING.md)). Training experts the router never
  sends that data to wastes the leg. Rows are first-come, so commit early on
  contested rows.
- **Short wide-EMA legs vs long narrow-EMA legs.** A short leg with a wide
  EMA (σ_rel 0.15–0.20) averages more of a brief run. A long leg with a
  narrow EMA (0.03–0.05) tracks late, better weights. Test both locally with
  `cascade lane-b test` before you pay.
- **Use the curriculum for focus.** For example, use broad data first and
  then target-heavy data in a final stage, so the experts specialise without
  forgetting.
- **Keep the guards in mind.** Many legs fail on a guard, not on the target.
  Mix in enough general data that other domains and the public suites hold.
- **Never embed benchmark series.** The contamination scan refuses them (see
  below). Suite targets reward generalisation to the suite, not copies of it.

## Costs

You pay the provider directly for the leg. These are measured lane B leg
rates (operator measurements, 2026-11-24; WB19 evidence for 8×RTX4090):

| GPU pod | Steps/s | Tokens/s | Approx. cost per billion tokens (spot) | 3 h leg | 12 h leg |
|---|---:|---:|---:|---:|---:|
| 8×H100 | 0.404 | ≈ 847k | ≈ $4.6 | ≈ 9.1B tokens, ≈ $42 | ≈ 36.6B tokens, ≈ $168 |
| 8×RTX4090 | 0.137 | ≈ 286k | ≈ $3.3 | ≈ 3.1B tokens, ≈ $10 | ≈ 12.4B tokens, ≈ $41 |
| 4×H100 | 0.212 | ≈ 444k | not measured | ≈ 4.8B tokens | ≈ 19.2B tokens |

Spot prices move. A token limit makes your cost predictable across GPU
types. A failed or rejected leg is not refunded.

## What gets you refused

- **Contamination:** a tree that embeds benchmark series
  (`benchmark_series_embedded`, `contamination_budget_exceeded`). Run
  `cascade lane-b verify-repo` first. Suite-target repos are capped at
  `suite_target_max_repo_mb` (8 MB).
- **Targets:** a target not in the lists above, more than
  `max_domains_per_leg`, or a suite target while `suite_targets` is off
  (`bad_targets`).
- **Declared rows:** out of the per-layer range, a bad layer, a bad count or
  malformed JSON (`declared_rows_out_of_range`, `declared_rows_bad_layer`,
  `declared_rows_count`, `declared_rows_malformed`).
- **Levers out of range:** `lr_scale_out_of_range`, `bad_curriculum`,
  `context_length_unsupported`, `ema_out_of_range`,
  `train_tokens_out_of_range`, `train_hours_out_of_range`.
- **GPU:** a type outside the list without a token or hour limit, or an
  unmeasured type (`gpu_rate_unmeasured`).
- **Funding:** no `--yes` (exit 5), a spent commitment (`spent`), or a key
  that expired before the leg rented (`funding_expired`).

## FAQ

**Do I pick expert IDs or targets?** Targets. Declared rows are optional and
only narrow which experts train.

**Does my leg get paid when it qualifies?** No. Pay starts when the merge is
accepted at a warm start.

**How long does a slot pay?** Until another leg passes on top of yours.
There is no decay and no time limit.

**Can I run a longer leg to get a bigger gain?** Yes, up to 12 h, at your
cost. Gains per token usually fall with length, so test locally first.

**Can I use a cheaper GPU?** Any Lium type, at your own risk, once you set a
token or hour limit. An unmeasured type that fails is not requeued.

**What does `cascade lane-b test` prove?** That your entry trains and how it
scores on a local pool. A local gain is not a live verdict.
