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
miner builds on it. You choose how the leg trains: learning rate, curriculum,
context, EMA width and how long.

## What you submit

- **A code-only generator** (the lane A interface, see
  [INTERFACE.md](INTERFACE.md)), committed with 1 to `max_domains_per_leg`
  (4) **declared targets**:
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
- **Your leg levers**, all optional, signed into the fund request
  (`X-Lane-B-Levers`) by `cascade lane-b fund`:

| Flag | What it sets | Allowed | Default |
|---|---|---|---|
| `--lr-scale X` | multiplies the leg's base learning rate | `lr_scale_min`..`lr_scale_max` = 0.5..2.0, at most 3 decimals | 1.0 |
| `--curriculum NAME:FRAC,...` | up to 3 ordered stages, e.g. `easy:0.3,hard:0.7`; each stage is its own corpus, generated with `curriculum_stage=<name>` | 1–3 stages, names `[a-z][a-z0-9_-]`, fractions sum to 1 (2 decimals) | one stage |
| `--context-length N` | the leg's training context | 4096, 8192, 16384 or 32768 (`max_context_length`); must fit the GPU's measured memory | 4096 |
| `--ema SIGMA` | the scored power-EMA width σ_rel | `sigma_rel_choices` = 0.03, 0.05, 0.10, 0.15, 0.20 | 0.10 |
| `--train-tokens N` | token limit, e.g. `20e9` | integer > 0 | none |
| `--train-hours H` | hour limit on your chosen GPU | 0.5 to `max_leg_hours` (12), 0.5 h steps | none |

  The leg stops at the token limit **or** the hour limit, whichever comes
  first. Tokens alone run up to the 12 h limit. Both limits need `--gpu`. A
  curriculum needs a generator whose `__init__` takes the `curriculum_stage`
  keyword: check it with `verify-repo --curriculum` before you fund.
- **How legs train.** Legs train with a power EMA and no LR cooldown: the LR
  warms up and then holds at peak to the last step. Only the EMA you chose is
  computed, and that EMA is what is scored (`ema010` at the default). No key
  turns this on or off.
- **Your own compute key.** The trainer rents the leg on your provider key.
  The measured lane B GPU list is `funded_pod_skus` (8×H200, 8×H100, 4×H200,
  4×H100, 8×RTX4090). **Any other Lium GPU type is allowed at your own risk**
  when you set a token or hour limit, use no fallback and pay with your Lium
  key. It is recorded `gpu_measured: false`, the memory-fit check is skipped,
  and there is no refund if the leg runs out of memory, cannot boot or
  crashes.

```bash
cascade lane-b verify-repo ./my-generator           # the contamination scan, locally
cascade lane-b test ./my-generator --targets energy,h720 --base live --pool-dir ./pool
cascade deploy ./my-generator --hub-namespace my-ns --expert-domain energy,h720 \
    --wallet-name my-miner --wallet-hotkey expert1
export LIUM_API_KEY=...   # env only, never an argument
cascade lane-b fund https://submissions.cascadesub.net --target energy --target h720 \
    --ref <repo@digest> --declared-rows rows.json --gpu 8xH100 \
    --wallet-name my-miner --wallet-hotkey expert1 --yes
cascade lane-b status https://submissions.cascadesub.net --hotkey <ss58>
```

Add any of the lever flags above to the `fund` line, for example
`--train-tokens 20e9 --ema 0.15 --lr-scale 1.5`. Repeat `--target` once per
declared target (or give them comma-separated with `--targets`). One
commitment funds one leg. Re-funding a spent commitment returns `spent`.

## How it is scored

Every judged leg is tested on **every** slot, not only the ones you declared.
The slots fall into two test families, each corrected on its own with a
Westfall–Young max-T bound at the full `alpha` (0.05), over CRPS and MASE per
slot:

| Family | Slots | Bar (LCB of the gain) |
|---|---|---|
| Declared | the 1–4 targets you committed | `gain_margin` = **1%** |
| Undeclared | every other slot (domains, horizons, suites when on) | `undeclared_gain_margin` = **2%** |

- **A slot clears** when CRPS **or** MASE has a family LCB at or above its
  bar with a positive point gain, and the other metric is non-inferior (its
  LCB ≥ −`tau_domain`, or −`tau_public` for a suite, both 1%). Every cleared
  slot is claimable, so one leg can win slots you never declared, at the
  higher bar.
- **Guards, for the whole leg.** No other domain may regress by more than
  `tau_domain` (1%). No public suite in `require_public` (GIFT-Eval, BOOM,
  TIME, FEV) may regress by more than `tau_public` (1%). One failed guard
  blocks every slot in the leg.
- **Acceptance.** A cleared leg earns nothing until its merge is accepted at
  a warm start. A suite target that is the era's sentinel waits
  (`sentinel_target`) for up to `sentinel_max_defer_eras` (4) eras.

## How you are paid

Each target is a **slot**. Lane B's pool is split equally over its slots:
10 today (7 domains + 3 horizons), 14 with suite targets on. A slot holder
is paid the **full slot every settlement, with no decay and no time limit,
until a newer leg is accepted on that slot**. The newer leg is judged on top
of a base that already includes the holder's update, so it has to beat the
holder, not the old model. A tie or an inconclusive challenger keeps the
holder. One hotkey may hold several slots. An empty or deregistered slot
burns. It is never shared out to other holders.

**Worked example at the owner-proposed split A0 / B35 / C15 / D50.** This is
the owner's proposal (plan rev6.24, set by the launch overlay), not a
guaranteed live value. Check `cascade rules b` for `share`.

| Lane B slots | One slot's share of emissions | You hold energy + h720 |
|---|---:|---:|
| 10 (suite targets off) | 35% / 10 = 3.5% | 7% |
| 14 (suite targets on) | 35% / 14 = 2.5% | 5% |

Lane B opens on mainnet at block 9,586,800 (about 2026-11-25 22:28 UTC),
with suite targets not armed at launch.

## Strategies worth trying

- **Target where the king is weakest.** The receipts' `per_domain` numbers
  and the bench preview show where the current model loses. A 1% gain is
  easier on a weak domain or suite than on a strong one.
- **Declare what you aim at; let the rest be a bonus.** Declared slots clear
  at 1% and undeclared ones at 2%. Declare the targets you train for (up to
  4). A broad improvement can still win undeclared slots.
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
- **Longer context for long-horizon targets.** `horizon:720` and slow
  seasonal domains may gain from 16k–32k context. It costs memory, so check
  that it fits your GPU.
- **Cap spend with tokens.** A token limit fixes the work done whatever the
  GPU. An hour limit fixes the wall. Set the one you care about.
- **Keep the guards in mind.** Many legs fail on a guard, not on the target.
  Mix in enough general data that other domains and the public suites hold.
- **Never embed benchmark series.** The contamination scan refuses them (see
  below). Suite targets reward generalisation to the suite, not copies of it.

## Costs

You pay the provider directly for the leg. These are measured lane B leg
rates (operator measurements, 2026-11-24: WB19/WB25 runs at context 4096,
batch 512, on the current worker image):

| GPU pod | Steps/s | Tokens/s | Approx. cost per billion tokens (spot) | 3 h leg | 12 h leg |
|---|---:|---:|---:|---:|---:|
| 8×H100 | 0.404 | ≈ 847k | ≈ $4.6 | ≈ 9.1B tokens, ≈ $42 | ≈ 36.6B tokens, ≈ $168 |
| 8×RTX4090 | 0.137 | ≈ 286k | ≈ $3.3 | ≈ 3.1B tokens, ≈ $10 | ≈ 12.4B tokens, ≈ $41 |
| 4×H100 | 0.212 | ≈ 444k | not measured | ≈ 4.8B tokens | ≈ 19.2B tokens |

One step is 2,097,152 tokens, so a token limit converts to steps exactly.
Spot prices move. When the worker image has no measured walls yet, every
leg's wall is capped at `unmeasured_image_wall_cap_seconds` (18,000 s,
5 h), so a 12 h request trains for at most 5 h until the new image is
measured. A failed or rejected leg is not refunded.

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
- **Levers out of range:** `bad_levers`, `lr_scale_out_of_range`,
  `bad_curriculum`, `context_length_out_of_range`, `context_length_unfit`
  (does not fit the GPU's memory), `ema_out_of_range`,
  `generator_curriculum_unsupported`.
- **Limits:** `bad_train_hours`, `bad_train_tokens`,
  `train_tokens_out_of_range`, a limit without `--gpu`
  (`train_hours_needs_gpu`) or with `--allow-fallback`
  (`train_hours_with_fallback`).
- **GPU:** an unmeasured type without a limit, or without your Lium key
  (`gpu_rate_unmeasured`).
- **Funding:** no `--yes` (exit 5), a spent commitment (`spent`), or a key
  that expired before the leg rented (`funding_expired`).

## FAQ

**Do I pick expert IDs or targets?** Targets. Declared rows are optional and
only narrow which experts train.

**Can I win a slot I did not declare?** Yes, if that slot clears the 2% bar
in its family and the guards hold.

**Does my leg get paid when it clears?** No. Pay starts when the merge is
accepted at a warm start.

**How long does a slot pay?** Until another leg is accepted on top of yours.
There is no decay and no time limit.

**Can I run a longer leg to get a bigger gain?** Yes, up to 12 h, at your
cost (5 h while the image is unmeasured). Gains per token usually fall with
length, so test locally first.

**Can I use a cheaper GPU?** Any Lium type, at your own risk, once you set a
token or hour limit. An unmeasured type that fails is not requeued or
refunded.

**What does `cascade lane-b test` prove?** That your entry trains and how it
scores on a local pool. A local gain is not a live verdict.
