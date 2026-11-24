# Lane D miner guide: open weights

This is the competition brief for lane D: what the subnet wants, what you
submit, how you are judged and paid, what to try, what it costs and what gets
you refused. The mechanics (file format, every refusal and skip code, the
intake protocol) are in [LANE_D.md](LANE_D.md). The live numbers are in
`cascade rules d`, which reads the same `chain.toml` the validators load.

```bash
cascade rules d            # headline lane D rules from chain.toml
cascade rules d --all      # every [lane_d] key
cascade status --json      # whether lane D is armed on this network right now
```

## Intent

Lane D asks for the **best forecasting model you can train, by any recipe
you like**. Lanes A–C keep the training recipe fixed and compete on data or a
narrow fine-tune. Lane D removes that limit. You train a `cascade_pyramid`
checkpoint anywhere, on your own hardware and data, with your own schedule,
distillation or context length, and submit only the weights. The subnet
publishes the winner under MIT so everyone can build on it.

Your model has to forecast data that did not exist when you submitted it.
Half the score comes from those fresh windows, so memorising public
benchmarks cannot win. A model that is better at real forecasting can.

## What you submit

- **A checkpoint directory with exactly four files:** `config.json`,
  `weights.safetensors`, `model.py` and `forecast_wrapper.py`. The two code
  files must be byte-identical to the repo's `cascade/trainer/pyramid_model.py`
  and `pyramid_wrapper.py`: you change weights, not code.
- **BF16 weights only.** An F32 file is refused (`dtype_mismatch`). At the
  120m pyramid size an entry is about 2.5 GB.
- **A declared context**, one of 4096, 8192, 16384, 32768, 65536 or 131072
  (`contexts` in `cascade rules d`).
- **An MIT licence attestation.** MIT is the only accepted licence (D13). You
  attest that the weights and the training data may be published under it and
  that you used no restricted-licence teacher outputs. For example,
  Timestral outputs fall under MarketCast's licence terms.
- **A dedicated hotkey.** A lane D commitment replaces that hotkey's lane A
  pointer. Do not reuse a hotkey that holds a lane A entry.

No compute key is needed: you train wherever you like, and judging runs on
the validators' eval host.

```bash
cascade lane-d export ./my-trained-ckpt --out ./entry
cascade lane-d verify ./entry
cascade lane-d submit https://submissions.cascadesub.net ./entry --context 16384 \
    --wallet-name my-miner --wallet-hotkey laned1
cascade lane-d ready https://submissions.cascadesub.net --digest <digest> --context 16384 \
    --licence MIT --attest --wallet-name my-miner --wallet-hotkey laned1
cascade lane-d status https://submissions.cascadesub.net --hotkey <ss58>
```

`submit` uploads straight to the private vault and commits nothing. `ready`
commits on chain with an immediate reveal. Commit within `ready_ttl_blocks`
(7200 blocks, about 24 h) of submit or the entry expires (`ready_expired`).

## How it is scored

- **Metric.** The geometric mean of the CRPS and MASE ratios against the
  reference, over windows with a 64-step horizon (`horizon`). A paired cluster
  bootstrap over sources gives a one-sided lower confidence bound (LCB) at
  `alpha` = 0.05.
- **The data: live half plus public half (plan rev6.18).**
  - **Live (50%)** is forward windows only: target data that arrived after
    your entry (plus the 48 h embargo top-up, `embargo_hours`), from the
    private live pool. Nobody, including the operator, has those targets
    when you submit.
  - **Public (50%)** comes from GiftEvalPretrain and RMISC, sampled from the
    whole dataset. Multivariate sets are scored as `(C, L)` panels (F2).
  - A challenger must improve on the public half as well as the live half.
- **Timing (plan rev6.14).** Judging starts no earlier than 6 h after your
  entry is ready (`eval_delay_hours` = 6). Settlements are about 3 h apart.
  Up to `max_windows_per_settlement` (5000) windows are drawn per settlement,
  and a verdict needs at least `min_windows` (1000) windows from
  `min_clusters` (20) sources, no source over `max_source_share` (25%).
- **Context.** You are scored at your declared context when that context is
  armed, else at the highest armed context below it. If none is armed below
  it, the entry is skipped `context_below_armed` and waits. You are never run
  above your declared context. A context arms once enough long series exist
  and the eval host passes its wall-time and memory check.
- **The bar.** You crown when the LCB of your gain clears the crown bar
  against **both** the lane D king (KING_D) **and the house** (the lane A
  king, always entered). In effect you must beat max(KING_D, house).
  - The bar decays with KING_D's tenure (plan rev6.23 part 2): **3%** against
    a fresh king, falling linearly to **1%** (`d_margin`) over 32 settled
    lane D settlements (about 4 days). House tenure counts too. A settlement
    with no lane D scoring does not age the king.
  - The crown is decided at the first verdict. There is no confirmation duel.
- **The gate.** Each settlement draws one public bench suite from GIFT-Eval,
  BOOM, TIME and FEV (minus the era's sentinel). Your CRPS and MASE must each
  stay within `tau_public` (1%) of both KING_D and the house, or the entry is
  `gate_regression`. A bench outage holds the verdict (`bench_hold`) for up to
  `max_bench_holds` (4) settlements.
- **The sentinel** is a held-out suite that is published and never gates.

## How you are paid

Lane D pays a **court**: the current king and up to two prior kings, by the
court weights **80% / 15% / 5%** (plan rev6.23). Prior kings are paid with
no time limit. A missing seat renormalises over the seats that are filled.
If the house holds the lane D crown, the king seat burns (`house_king`).
A deregistered holder's seat burns too.

**Worked example at the owner-proposed split A0 / B35 / C15 / D50.** This is
the owner's proposal (plan rev6.24, signed overlay), not a guaranteed live
value. Check `cascade rules d` and `cascade status --json`.

| Court | King | Prior king 1 | Prior king 2 |
|---|---:|---:|---:|
| Full court (80/15/5 of D's 50%) | 40% of emissions | 7.5% | 2.5% |
| King + one prior (renormalised over 0.95) | 42.1% | 7.9% | — |
| King only | 50% | — | — |
| House holds the crown | burns | 7.5% | 2.5% |

These are shares of the validators' weight vector, not a TAO or dollar
figure. Until lane D arms on a network, its share stays with the lane it is
carved from (`takes_from`).

## Strategies worth trying

- **Train for real forecasting, not for the public half.** The live half is
  data nobody has seen and the gate checks a public bench suite every
  settlement. A model fitted to GiftEvalPretrain or RMISC will look good
  locally and then lose the live half or trip the gate.
- **Long context is a real lever.** You may declare up to 131072. A model
  that uses 32k–128k of history well can beat a 4k king on series with long
  seasonality. Declare only what the model really supports: you are scored at
  the highest armed context at or below the one you declare, so an
  over-declared entry gains nothing and can wait `context_below_armed`.
- **Multivariate data.** Panels are scored as panels. Training on related
  channels (cross-series covariates, sensor groups, sales hierarchies) is
  where a model can beat a univariate king.
- **Fine-tune the king rather than starting from scratch.** Kings are public
  under MIT (Hub, R2, HF mirror `tensorlink-dev/cascade-lane-d-kings`). Use
  `python -P -m cascade.miner.lane_d_finetune` (`--base`, `--data`, `--out`,
  `--steps`, `--lr`, `--context-length`, `--dry-run`) as a starting point.
- **Distil only from teachers you may republish.** A restricted-licence
  teacher's outputs breach the MIT attestation and lead to a takedown.
- **Time your entry against a fresh king.** The bar is 3% against a new king
  and 1% after about 4 days of tenure. An entry ready while the king is old
  faces the lower bar.
- **Expect forward-only latency.** Daily and weekly series need days of new
  data before they fill windows. See LANE_D.md for expected latency by
  frequency.

## Costs

Training is yours, on any hardware. These measured rates are for the
same `cascade_pyramid` family on lane B legs (operator measurements,
2026-11-24; WB19 evidence for 8×RTX4090). Use them as a planning reference,
not a quote:

| GPU pod | Steps/s | Tokens/s | Approx. cost per billion tokens (spot) |
|---|---:|---:|---:|
| 8×H100 | 0.404 | ≈ 847k | ≈ $4.6 |
| 8×RTX4090 | 0.137 | ≈ 286k | ≈ $3.3 |
| 4×H100 | 0.212 | ≈ 444k | not measured |

Submitting and judging cost you nothing. There is no entry fee. A losing
entry earns nothing and is deleted after `loser_ttl_eras` (4).

## What gets you refused or skipped

- **Format:** any file beyond the four (`file_not_allowed`), non-BF16 weights
  (`dtype_mismatch`), changed code bytes (`code_mismatch`) or a changed
  config (`config_mismatch`, `config_key_not_allowed`).
- **Licence:** anything but `--licence MIT` (`licence_not_open`), or no
  `--attest` (`licence_attestation_missing`).
- **Context:** a value not in `contexts` (`bad_context` at the intake;
  `context_not_armed` locally).
- **Duplicates:** the same weights as another entry (`duplicate_weights`).
- **Coldkey caps:** `max_open_per_coldkey` (2) open entries
  (`coldkey_cap_open`), `max_judged_per_coldkey_per_settlement` (2) judged per
  settlement, and `max_entries_per_settlement` (8) overall. An entry pending
  longer than `max_reveal_delay_days` (14) closes.
- **Timing:** committing after `ready_ttl_blocks` (`ready_expired`).

The full refusal and skip lists are in [LANE_D.md](LANE_D.md#lane-d-faq).

## FAQ

**Do I need a GPU key?** No. You train on your own hardware and the
validators judge. Lane D never rents on your key.

**Can I submit a different architecture?** No. Lane D is the
`cascade_pyramid` code with your weights. Architecture changes are a
roadmap decision, not a submission.

**Why was my entry not judged this settlement?** Read the typed skip reason
in `cascade lane-d status`. Common ones: `pending_floor` (not enough fresh
windows yet), `context_below_armed`, `coldkey_cap_settlement`, `bench_hold`.

**Is the 3% bar live?** The decaying bar is plan rev6.23 part 2. Read
`d_margin` and the crown bar fields in `cascade rules d` and the signed
report for the value your entry was judged under.

**Does winning lane A help me in lane D?** The lane A king is the house in
lane D. It is a competitor you must beat, never a paid seat.

**Where are kings published?** Hippius Hub, the R2 public mirror and the HF
mirror, indexed in the signed `champions/lane_d/index.json`.

**What happens if I distilled from a restricted model?** The entry breaches
the attestation. The takedown path removes it and its pay. Use teachers whose
outputs you may republish under MIT.
