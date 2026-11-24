# Lane D miner guide: open weights

Lane D's competition brief. The mechanics (file format, every refusal and
skip code, the intake protocol) are in [LANE_D.md](LANE_D.md). Live values
come from `cascade rules d`, which reads the same `chain.toml` the
validators load.

```bash
cascade rules d            # headline lane D rules
cascade rules d --all      # every [lane_d] key
cascade status --json      # whether lane D is armed now, and the live overlay values
```

## Intent

Lane D wants **the best forecasting model you can train, by any recipe**.
Lanes A–C fix the recipe and compete on data or a narrow fine-tune. Lane D
does not. You train a `cascade_pyramid` checkpoint anywhere, on your own
hardware and data, and submit only the weights. The winner is published
under MIT so everyone can build on it.

Your model is judged on data that did not exist when you submitted it, so
memorising public benchmarks cannot win.

**Launch.** Lane D opens on mainnet at block 9586800 (about Wed 2026-11-25
22:28 UTC), set by the signed launch overlay. At launch it is scored on
**live windows only** (`public_weight = 0.0`). A 50/50 live/public score is
the template rule and switches on later by overlay, once the public mirror
exists.

## What you submit

- **A directory of exactly four files:** `config.json`, `weights.safetensors`,
  `model.py` and `forecast_wrapper.py`. The two code files must be
  byte-identical to the repo's `cascade/trainer/pyramid_model.py` and
  `pyramid_wrapper.py`. You change weights, not code.
- **BF16 weights only.** F32 is refused (`dtype_mismatch`). A 120m entry is
  about 2.5 GB.
- **A declared context:** 4096, 8192, 16384, 32768, 65536 or 131072
  (`contexts`).
- **An MIT licence attestation.** MIT is the only licence accepted. You
  attest that the weights and training data may be published under it and
  that you used no restricted-licence teacher outputs (Timestral outputs,
  for example, fall under MarketCast's licence).
- **A dedicated hotkey.** A lane D commitment replaces that hotkey's lane A
  pointer.

You need no compute key. You train where you like; the validators' eval host
judges.

Score your checkpoint offline first, with the validators' own lane D code
(the window draw, `score_window`, `judge_cohort` and the crown bar). Pass
the king's checkpoint as `--king`; before the first crown that is the house
(the lane A king). It prints per-window and aggregate CRPS/MASE, the LCB
against the bar and whether it would clear. It is live-only by default, as
at launch; `--public-weight 0.5 --public-pool-dir DIR` previews the later
rule, and `--tenure N` sets the bar for a king of that tenure.

```bash
cascade lane-d test ./my-trained-ckpt --king ./king --context 16384 \
    --pool-dir ./pool_d --device cuda          # a pool_d/1 snapshot
cascade lane-d test ./my-trained-ckpt --king ./king --context 4096 \
    --series ./my-series.npz --windows-per-series 20   # your own series
```

The result is directional. Your local windows are not the private live
pool, and the bench gate, caps and queue run only live.

```bash
cascade lane-d export ./my-trained-ckpt --out ./entry
cascade lane-d verify ./entry
cascade lane-d submit https://submissions.cascadesub.net ./entry --context 16384 \
    --wallet-name my-miner --wallet-hotkey laned1
cascade lane-d ready https://submissions.cascadesub.net --digest <digest> --context 16384 \
    --licence MIT --attest --wallet-name my-miner --wallet-hotkey laned1
cascade lane-d status https://submissions.cascadesub.net --hotkey <ss58>
```

`submit` uploads to the private vault and commits nothing. `ready` commits
on chain with an immediate reveal. Run it within `ready_ttl_blocks` (7200
blocks, about 24 h) of `submit`, or the entry expires (`ready_expired`).

## How it is scored

- **Metric.** The geometric mean of the CRPS and MASE ratios against the
  king, on 64-step forecasts (`horizon`). Each window's ratio is clamped to
  [0.5, 2] and log-averaged. A paired cluster bootstrap over sources gives a
  one-sided lower confidence bound (LCB) at `alpha` = 0.05.
- **Live windows.** Forward windows from the private live pool. A window is
  *fresh* when all its target data arrived after your entry, or *embargo*
  when it was held back up to 48 h (`embargo_hours`) and released after your
  entry. At least 20% of live windows are fresh (`min_fresh_window_share`).
  Nobody, the operator included, had those targets when you submitted.
- **Public windows (off at launch).** When an overlay raises
  `public_weight` above 0, each settlement also draws 5000 windows from each
  of `Salesforce/GiftEvalPretrain` and `nju-zhangsq/RMISC`. The score becomes
  `L = (1 − public_weight) · L_live + public_weight · L_public`, and you must
  gain on each half separately. The template value is 0.5.
- **Volume.** Up to `max_windows_per_settlement` (5000) windows per
  settlement. A verdict needs `min_windows` (1000) from `min_clusters` (20)
  sources, no source above `max_source_share` (25%).
- **Timing.** Judging starts at least 6 h after your entry is ready
  (`eval_delay_hours`). Settlements are about 3 h apart. Slow series (daily,
  weekly) take days to fill forward windows.
- **Context.** You are scored at your declared context if it is armed,
  otherwise at the highest armed context below it. If none is armed below
  it, the entry waits (`context_below_armed`). You are never run above your
  declaration. A context arms when enough long series exist and the eval host
  passes its time and memory check.
- **The bar.** Your gain's LCB must clear the crown bar against the lane D
  king (KING_D). The house (the current lane A king) exists only for lane
  D's genesis (owner 2026-11-24, DEC-CA-0076).
  - **Before the first miner crown,** the house holds KING_D, so the first
    crown must beat max(house, lane A king). The house is the lane A king,
    so this is the lane A king.
  - **After the first crown,** the house is gone. It is no second bar and it
    never retakes the crown, so you only need to beat KING_D.
  - The bar is **3%** (`d_margin_start`) against a fresh king and falls
    linearly to **1%** (`d_margin`) over 32 lane D settlements of the king's
    tenure (`d_margin_warmup_settlements`, about 4 days). The signed report
    records the bar used (`crown_bar`).
  - The crown is decided at the first verdict. There is no confirmation
    duel.
- **The gate.** Each settlement draws one public bench suite from GIFT-Eval,
  BOOM, TIME and FEV (minus the era's sentinel). Your CRPS and MASE must each
  stay within `tau_public` (1%) of both KING_D and the house, or the entry is
  `gate_regression`. A bench outage holds the verdict (`bench_hold`) for up to
  `max_bench_holds` (4) settlements.

## How you are paid

Lane D pays a **court**: the king and up to two prior kings, split
**80% / 15% / 5%** (`court_weights`). Prior kings are paid with no time
limit. Empty seats renormalise over the miner kings that exist. A
deregistered king's seat burns to `burn_uid` (unchanged): its uid no longer
resolves, so it is never paid to whoever holds that uid now.

**Genesis (DEC-CA-0076).** Until a miner is first crowned, lane D pays nobody
and burns nothing. Its share stays with lane A, its `takes_from` lane, so the
weights are exactly what they would be with lane D's share at 0.

**Worked example at the owner-proposed split A0 / B35 / C15 / D50.** This is
the owner's proposal (plan rev6.24), set by the mainnet launch overlay at
block 9586800. Check `cascade status --json` for the live value.

| Court | King | Prior king 1 | Prior king 2 |
|---|---:|---:|---:|
| Full court (80/15/5 of D's 50%) | 40% of emissions | 7.5% | 2.5% |
| King + one prior (renormalised over 0.95) | 42.1% | 7.9% | — |
| King only | 50% | — | — |
| Genesis (no miner crowned yet) | lane D's 50% stays with lane A | — | — |

These are shares of the validators' weight vector, not TAO or dollars.

## Strategies worth trying

- **Train for real forecasting.** Live windows are data nobody has seen, and
  the gate checks a public suite every settlement. A model fitted to public
  benchmarks looks good locally and then loses.
- **Use long context.** You may declare up to 131072. A model that uses
  32k–128k of history well can beat a 4k king on long-seasonality series.
  Declare only what the model supports: an over-declared entry gains nothing
  and may wait `context_below_armed`.
- **Train on multivariate data.** Panels are scored as panels. Related
  channels (covariates, sensor groups, sales hierarchies) are where a model
  can beat a univariate king.
- **Fine-tune the king.** Kings are public under MIT (Hub, R2, HF mirror
  `tensorlink-dev/cascade-lane-d-kings`). `python -P -m
  cascade.miner.lane_d_finetune` (`--base`, `--data`, `--out`, `--steps`,
  `--lr`, `--context-length`, `--dry-run`) is a starting point.
- **Distil only from teachers you may republish.** Restricted-licence
  outputs breach the MIT attestation and lead to a takedown.
- **Time your entry.** The bar is 3% against a new king and 1% after about
  4 days of tenure.

## Costs

Training is yours, on any hardware. These measured rates are for the same
`cascade_pyramid` family on lane B legs (operator measurements, 2026-11-24;
WB19 evidence for 8×RTX4090). Use them for planning, not as a quote:

| GPU pod | Steps/s | Tokens/s | Approx. cost per billion tokens (spot) |
|---|---:|---:|---:|
| 8×H100 | 0.404 | ≈ 847k | ≈ $4.6 |
| 8×RTX4090 | 0.137 | ≈ 286k | ≈ $3.3 |
| 4×H100 | 0.212 | ≈ 444k | not measured |

Submitting and judging are free. A losing entry earns nothing and is
deleted after `loser_ttl_eras` (4).

## What gets you refused or skipped

- **Format:** an extra file (`file_not_allowed`), non-BF16 weights
  (`dtype_mismatch`), changed code (`code_mismatch`) or a changed config
  (`config_mismatch`, `config_key_not_allowed`).
- **Licence:** anything but `--licence MIT` (`licence_not_open`), or no
  `--attest` (`licence_attestation_missing`).
- **Context:** a value not in `contexts` (`bad_context` at the intake,
  `context_not_armed` locally).
- **Duplicates:** the same weights as another entry (`duplicate_weights`).
- **Caps:** `max_open_per_coldkey` (2) open entries (`coldkey_cap_open`),
  `max_judged_per_coldkey_per_settlement` (2) judged per settlement, and
  `max_entries_per_settlement` (8) overall. An entry pending longer than
  `max_reveal_delay_days` (14) closes.
- **Timing:** committing after `ready_ttl_blocks` (`ready_expired`).

Full lists: [LANE_D.md](LANE_D.md#lane-d-faq).

## FAQ

**Do I need a GPU key?** No. Lane D never rents on your key.

**Can I submit a different architecture?** No. Lane D is the
`cascade_pyramid` code with your weights.

**Is the public half scored?** Not at launch. The verdict is the live half
alone until an overlay raises `public_weight`.

**Why was my entry not judged this settlement?** Read the skip reason in
`cascade lane-d status`. Common ones: `pending_floor` (not enough fresh
windows yet), `context_below_armed`, `coldkey_cap_settlement`, `bench_hold`.

**What bar will I face?** 3% against a fresh king, down to 1% after 32
settlements of its tenure. The signed report records the bar each verdict
used.

**Does winning lane A help in lane D?** The lane A king is the house. It is
the bar for the first crown only, and never a paid seat.

**Where are kings published?** Hippius Hub, the R2 public mirror and the HF
mirror, indexed in the signed `champions/lane_d/index.json`.
