# Plan: lane D "open weights" (anything goes on our architecture), with lane E and lane A flexibility — v3

Status: **PLAN v3, ~~rev6.12~~ ~~rev6.14~~ ~~rev6.15~~ ~~rev6.16~~ ~~rev6.17~~ ~~rev6.18~~ ~~rev6.20~~ rev6.23 (proposed 2026-11-22; ~~rev6.12~~ ~~rev6.14~~ ~~rev6.15~~ ~~rev6.16~~ ~~rev6.17~~ rev6.18 2026-11-23; rev6.20, rev6.23 2026-11-24)**: v2's design, made buildable by an
autonomous loop. A work package is done **only when its Accept command exits
0**; **(rev6)** its level is only what `scripts/laned/status` computes (§17),
never a claim. Owner decisions are marked **[DECIDE]**. Nothing is armed. **No calendar
dates**: stages are gated (§5).

## rev6.23 (owner 2026-11-24; main @ b65f25d): lane D court pays 80 / 15 / 5

One owner decision, for **lane D only** (owner 2026-11-24, DECIDED): lane
D's paid court shares become **explicit weights**: current KING_D
**80%**, prior king 1 **15%**, prior king 2 **5%** of lane D's share. This
replaces the geometric `king_decay = 0.5` (which gave 57 / 29 / 14). Plan
text only; the code lands as one feature PR (rev6.17 rule), tests and code
together. Changed text elsewhere is marked **(rev6.23)**; superseded text
is struck, not deleted.

**Config (W2 / W2e).**

- New key `[lane_d] court_weights = [0.80, 0.15, 0.05]`: a list of
  numbers, one per court slot (KING_D first, then prior miner kings by
  recency). The loader refuses it unless its length is
  `1 + reward_prior_kings`, every entry is a finite number `>= 0` (bools
  refused), and the entries sum to 1 within `1e-9`. Overlay-settable like
  every other lane D tuning key (`ALLOWED["lane_d"] = None`; the overlay
  merge is re-parsed through the same loader, so a bad overlay is refused).
- `reward_prior_kings` stays 2. Changing it alone (file or overlay) is a
  length mismatch and is refused; the two keys move together.
- `[lane_d] king_decay` is **retired**: it joins `RETIRED_KEYS["lane_d"]`,
  so older files that still set it load and the value is ignored. It is no
  longer a `LaneDConfig` field. **Lane A's `[scoring] king_decay` is a
  different key and is untouched.**

**Pay (unchanged otherwise).** Slot `i` of the paid court is worth
`court_weights[i]`, normalised over the slots the court has, times lane D's
share. Every other court rule is exactly today's:

- the court shifts on a new crown (the new KING_D in slot 0, the old one
  becomes prior king 1, prior king 2 drops off);
- the house is never paid: when the house holds KING_D its slot pays
  nobody and burns (`house_king`), exactly as today;
- a deregistered member's slot burns (`deregistered`);
- a court with fewer than `1 + reward_prior_kings` slots (no prior kings
  yet) renormalises over the slots it has, as today's geometric split
  did: KING_D alone takes the whole share; KING_D plus one prior king
  take `0.80 / 0.95` and `0.15 / 0.95`;
- a failed lane D check burns only lane D (rev6.18 F1/B1).

**Not an activation change.** Lane D is not armed on mainnet
(`from_block = 0`). This changes a default in the shipped files and the
parser; no activation block, no `cfg-` vote, no overlay is needed. The
change reaches a validator with the release; the arming overlay (D10/D45)
carries whatever `court_weights` the owner sets then.

**Report and audit (W7a, W8b, W9).** The signed `lane-d-report/1`
`params` echo carries `court_weights` (a JSON list, as parsed) in place of
`king_decay`; the audit's expected params match it. Every re-derivation of
the paid court uses the parsed `court_weights`, never the producer's
emission: `cascade.audit.lane_d_chain.court_shares` (W8b) and the audit's
lane D part in `cascade.audit.checks` (`lane_d_pay`, W9: the "weights match
the paid court" burn check). The validator's builder
(`cascade.shared.emission.lane_d_vector`) takes `court_weights` too.

**Text struck elsewhere:** §1.4 (court decay), the W2 key list, D7 (§6 and
§17.4), D25's 57 / 29 / 14.

## rev6.20b (owner 2026-11-24): no activation block for lane B EMA

Owner 2026-11-24: "lanes B, C and D are NOT rolled out yet, so **no
activation blocks**." This amends rev6.20 below. Plan text only.

- **EMA, no cooldown is simply lane B.** Every lane B leg contract carries
  `lr_schedule = "warmup_constant_ema"` (`lane_b_leg_contract`, the one
  helper the trainer and the worker share). Every lane B leg scores and
  merges its σ_rel 0.10 power EMA. No key, flag or block arms it.
- **~~`[lane_b] ema_no_cooldown_from_block`~~ is not built** (struck in
  rev6.20 below). It never reached main, so there is nothing to retire.
- **Kept:** the signed record and the judge check. Every expert manifest
  entry carries `scored_weights = "ema010"`, always in the signed body. A
  manifest without the field loads as `""`, and `judge_entry` refuses any
  entry whose value is not `"ema010"` at stage `scored_weights` (in the
  audit's `LANE_B_STAGES`).
- **Dropped:** the trainer's per-leg marker check (`scored_weights_mismatch`).
  The receipt check (`receipt_contract_ok`) already binds the pod to the
  EMA leg contract.
- **Unchanged:** lane A (W-EMA, cooldown, `contract_digest`, `[training]`,
  pinned deps). The on-pod part still ships in a shared-image repin, never
  mid seed epoch. Lane B goes live only with its own rollout
  (`[lane_b] from_block`).
- **WPs (#845):** W-EMA, WB10 (`lane_b_leg_contract`, no key), WB22, W-S2,
  W9, WB20 (MINER.md).

## rev6.20 (owner 2026-11-24; main @ 052eabe): lane B legs train with EMA, no cooldown

**(rev6.20b) The activation key below is superseded: no key; the rule is
lane B itself (rev6.20b above).**

One owner decision, for **lane B only** (owner 2026-11-24, DECIDED: "let's
just do EMA, no cooldown"). Plan text only; the code lands as one feature
PR (rev6.17 rule), tests and code together. Changed text elsewhere is
marked **(rev6.20)**; superseded text is struck, not deleted.

**The decision.** A lane B expert leg:

1. **trains with no LR cooldown.** The WSD decay tail (the last
   `round(steps / 11)` steps, `pyramid_optimizer.schedule_factor`) is
   removed for lane B legs: the LR warms up exactly as today (the same
   warmup steps) and then stays at its peak to the last step;
2. **is scored on its power-EMA average, not the raw endpoint.** The scored
   `weights.safetensors` of the leg is the image's `sigma_rel = 0.10` power
   EMA of the leg's own training (fresh at the base king's scored weights,
   per-leg scope). Every frozen row is restored to the king's exact bytes
   before export (`restore_frozen`), so the merge byte-check is unchanged.
   The raw endpoint rides along unscored as `weights_raw.safetensors` with
   `weights_ema005.safetensors` (never published, `UNPUBLISHED_PATTERNS`).

**Which EMA width is scored: `sigma_rel = 0.10`** (`POWER_EMA_SIGMA_RELS
[POWER_EMA_EVAL]`, `pyramid_trainer.py`). That is the owner's standing
power-EMA decision (OPSLOG 2026-11-01: "sigma_rels (0.05, 0.10), score the
0.10 average, per-leg scope"; `docs/MODEL.md` "Power EMA"; DEC-CA-0050
"Scored EMA per leg": "Lane-B legs score a power EMA of their own training
too … `power_ema_eval` of the contract"). §8's rev3 worry about σ 0.10
(the lineage-wide window making warm-start legs look worse) is a lane A
lineage effect and doesn't apply: a lane B leg's average starts fresh at
its base king and covers only its own training. No new width or image
constant is introduced.

**Lane A is untouched.** Lane A keeps the W-EMA rule (§8): every lane A leg
(random init, warm-start king or challenger, heat or full) scores the raw
cooled endpoint and keeps the WSD cooldown. `power_ema_active` stays `False`
for every input. Lane A training, the lane A `contract_digest`, every
`[training]` / `[v2.training]` key and every pinned dependency are
byte-identical; the existing byte-identity tests (WB25 `main_snapshot`
oracle, W-EMA EMA-off comparisons) prove it in CI.

**Consensus surface (validators judge lane B legs).** ~~One typed activation
key, the repo's `*_from_block` pattern:~~ **(rev6.20b) no key.**

- ~~**`[lane_b] ema_no_cooldown_from_block`** (integer ≥ 0; default **0 =
  never**, today's behaviour). When typed, an era boundary at or after
  `[v2] from_block` (loader-checked, like `acceptance_from_block`).
  Overlay-settable like every `[lane_b]` key except the release-only hash
  set. A lane B leg whose dispatch block (`--block`, recorded as the expert
  manifest entry's `train_block`) is ≥ the key trains under the new rule.~~
- **Lane B leg contract.** From that block the trainer and the worker derive
  the leg contract as today (`for_hours`) and then set `lr_schedule =
  "warmup_constant_ema"`, one shared helper on both sides. The worker's
  receipt carries that leg contract's digest and the trainer's existing
  receipt check (`receipt_contract_ok`) refuses a pod that trained anything
  else. `train_pyramid` refuses the schedule on a lane A leg (no expert
  selection) and on an image with no power EMA.
- **Signed record.** The leg checkpoint's `config.json` `power_ema` marker
  gains `"schedule": "warmup_constant_ema"`. The expert manifest entry gains
  **`scored_weights`** (`"ema010"` when armed; `""` = the raw cooled
  endpoint, dropped from the canonical body so an unarmed entry keeps its
  bytes). The trainer refuses to publish a leg whose checkpoint marker
  disagrees with the rule at its `train_block` (outcome
  `scored_weights_mismatch`).
- **Judges re-derive it.** `judge_entry` (validator, trainer self-judge and
  cascade-audit share it) re-derives the expected `scored_weights` from
  `train_block` and the key and refuses a mismatch at a new stage
  `scored_weights` (added to the audit's `LANE_B_STAGES`; such a row never
  passes).
- **Merge consistency.** The rebased checkpoint (`settle_leg`:
  `now + (leg − base)` on the leg's rows) is built from the leg's
  `weights.safetensors`, which is the scored EMA, so the judged checkpoint
  and the merged rows (`merge_expert_slices` byte-check,
  `rebase_expert_delta`, `apply_merges`) are the same weights. Nothing reads
  `weights_raw` / `weights_ema005` for a verdict or a merge.

**Activation (owner steps).** The on-pod part is a worker-image change
(shared image with lane A): image built from the feature PR's merge, 5-pin
repin + wall gate, never mid seed epoch (memory: "Image repin: 5 pins + wall
gate"), release-then-activate on netuid 91 (six external validators): ship
the validator release, wait for ≥ 2/3 observed stake, then type
~~`ema_no_cooldown_from_block` at a future era boundary. Testnet (netuid 259)
first. Until the key is typed nothing changes, on any image.~~ **(rev6.20b)**
no key to type: lane B gets the rule when lane B rolls out.

| Item | Change | Where |
|---|---|---|
| (a) | Lane B legs: no cooldown, scored = σ 0.10 power EMA, ~~behind `[lane_b] ema_no_cooldown_from_block`~~ **(rev6.20b) for every lane B leg** | §8 W-EMA (lane B struck), W-EMA row, §3.2 lane B bullet, §12b.0, D44 note |
| (b) | Feature PR WPs: W-EMA (`pyramid_trainer.py`, `worker.py` ride-along), WB10 (key), WB22 (`expert_manifest.py`, `lane_b_driver.py`), WB23 (`validator/expert_lane.py`), W9 (`LANE_B_STAGES`) | §3 |

## rev6.18 (owner 2026-11-23; main @ 78b1d45): eval v2 (live 50% + public 50%), multivariate, one checker, burn scoping, GPU environments

One owner brief (2026-11-23, "now also do this"), Parts A and B, and the
owner's answers to its four open items (B12, B9 owner keys, A3 clamp, A5
BOOM; (owner 2026-11-23, DECIDED: "they're all reasonable")). Plan text
only. Every changed item elsewhere is marked **(rev6.18)**; superseded text is
struck, not deleted. Under the rev6.17 rule this is the ONE plan PR for the
batch; the code lands as the feature PRs in **rev6.18 (F)** below, tests and
code together.

**Owner decisions (final, verbatim list; not reopened by this rev).**

- Miners may submit any context up to 128k.
- Entries are scored as soon as possible, first come first served.
- Live targets are the 48 h embargo data plus the data after submission.
- No confirmation duel. Crown at the first verdict.
- No false-win test. Do not add one or gate the release on one.
- No shadow modes. Everything goes live at one block.
- Lane D may be scored on any GPU, including RTX PRO 6000 Blackwell.
- A challenger must IMPROVE on public pretrain data as well as live data.
- The score is 50% live, 50% public.
- GiftEvalPretrain and RMISC count together as one public half.
- Public data is sampled from the whole dataset, not a fixed subset.
- RMISC is scored as multivariate.
- ~~The slow-cadence backtest windows stay, at 5%.~~ **Superseded (owner 2026-11-24, DECIDED): the slow-cadence backtest windows (W5i) are DROPPED; the public half replaces them** (rev6.18 (D24) below).
- The low-frequency check A10 is removed.
- The coldkey caps stay as built. No entry fee at launch.
- Disputes and wrong crowns: the owner decides; pay already sent stands.

### rev6.18 (D24) Backtest windows dropped (owner 2026-11-24, DECIDED)

The slow-cadence backtest windows (W5i) are dropped; the public half
(GiftEvalPretrain + RMISC) replaces them. This supersedes the brief's
"backtest stays at 5%" decision and its A7 keep line, rev6.9 (a) (the
backtest part of it) and rev6.15 (b) (backtest ON at launch). Concretely:

- **Live half = forward windows only** (fresh + 48 h embargo top-up, the W5j
  forward cells and the W2e embargo rule, unchanged). No backtest cell is
  built, drawn or scored; slow cadences enter the live half only as
  forward windows that pass W5a.
- **W5i is not built; it is retired** (row struck below).
- **`[lane_d] backtest_weight_cap` and `backtest_ratio_floor`** (W2e, on main,
  default 0.05 / 0.85) are **deleted in F4**: the loader drops their parse
  and the `1 − floor**cap < d_margin` inequality check, and lists both in
  `RETIRED_KEYS["lane_d"]` so a deployed file that still carries them
  loads; their `chain.toml` lines go. The audit's two-part
  `combined_log_ratio` (`cascade/audit/lane_d.py:841-850`) and the report's
  forward/backtest sub-score split go with them.
- **`public_weight` stays at 0.50** (overlay-settable, unchanged), so the
  score is 50% live (forward) + 50% public.

### rev6.18 (V) Verification against main @ 78b1d45

The brief's references were read at 1605ff7. Each was re-read on 78b1d45.
"STOP" means the item is not specified further here and needs the owner.

| Item | Reference | Matches main? |
|---|---|---|
| B1 | `cascade/audit/checks.py:2958-2961` (`_lane_d_court` returns `(None, "lane_d_report_unverified: …")` when `check_lane_d` is not PASS) | yes (same lines). The caller `derive_weights` (`checks.py:3784-3786`) returns `(None, why)` for the WHOLE vector |
| B1 | `cascade/validator/satellite.py` burns the whole vector on any failure | yes: `replay_receipt` → `_fail("weights-derive", …)` (`satellite.py:374-376`) and `desired()` returns `burn_vec` (`:629-647`), module docstring `:38-41` |
| B2 | `cascade/satellite/lane_d_replay.py` imported only by tests | yes, still: importers are `tests/accept/{w3b6b,w5e,w7d,w8b}/…` and the package docstring `cascade/satellite/__init__.py`. W3b6b **extended** it (now **1,885** lines, not 1,701) but added no production importer. The running satellite verifies only through `audit_receipt` → `check_lane_d` |
| B3 | refusal codes | `check_lane_d` (`cascade/audit/lane_d.py:1205-1340`) returns one name (`NAME`) with free-text detail: there are **no per-check codes** today, only `LANE_D_WEIGHT_ERRORS` (`checks.py:2874-2876`). The code table below is new |
| B4 | `cascade/shared/config/lane_d.py:59`, `:255`; `chain.toml` `[lane_d] forward_margin_hours = 6.0` (`:1653`); no `eval_delay_hours` in `cascade/` | yes, all. **In flight** (rev6.14 build, branch `laned/rev612-rev614`); not duplicated here |
| B5 | plan §1.6 `:2703`, D11 `:3448` and `:4565` | yes (same lines). Also `:3577-3581` (S8 host text), `:4193` (risk row), `:1292` (rev6.6 (a) open flag) |
| B7 | `cascade/eval/lane_d_handshake.py:32` | yes: `HANDSHAKE_FIELDS = ("git_head", "package_hash", "schema")`, code identity only |
| B8 | `ab0b412` | **the SHA is not on origin/main.** Its change landed squashed in `0c52d07a` (#471, with the weight keep-alive and bench evidence). The live validator runs `/root/cascade-v2` @ `5e3fade3` (#394), which has neither. `ab0b412` applies cleanly onto `5e3fade3` (merge-tree, no conflict). Deploy-gate needs: see B8 below |
| B9 | "52 unmarked keys" | **count does not reproduce.** A flattened-key TOML diff of `/root/cascade/chain.toml` vs `origin/main:chain.toml` gives 36 unmarked deployed-side keys + 12 main-only keys in shared tables = 48 (plus 98 keys of tables the deployed file predates). The two owner keys are present as described. Classification below; `cascade-audit decided` today: 2 TYPE findings, exit 1 |
| B11 | `lane_d.py:86` and `:100-115` | yes (same lines): `:86` `confirm_on_king_change`; `:100-115` the DEC-CA-0062 keys `targets … win_ttl_eras`. No lane D code reads any of them (every same-named read is on `[marketcast]`, `[lane_b]`, `[lane_c]` or the trainer config). `confirm_on_king_change` is still parsed and validated (`lane_d.py:264-266`) |
| B12 | readers of `init_baseline_geomean` / `init_floor_passed` | **STOP.** They are read: `cascade/shared/receipt.py:974-978` parses both back from archived receipts (the shadow gate has written them into signed verdict bodies since 2026-08-27, so receipt signature verification and audit replay depend on the parse); `receipt.py:498-501` (drop-when-None body); `cascade/validator/loop.py:583-592` (log); `cascade/eval/koth.py:233,454,594-618` (producer). `receipt.py:266` also carries `init_gate_mode` in the default verdict params. Deleting the gate can't delete these fields. ~~Owner call: keep the receipt fields parse-only (read, never written) and delete only the producer + key, or keep the gate~~ **Resolved (owner 2026-11-23, DECIDED: "they're all reasonable"): parse-only receipt fields, producer + key deleted (B12 below)** |
| B13 | `dedup_probe_*` ignored by the loader | yes: `[round]` `dedup_probe_mode`, `dedup_probe_series`, `dedup_probe_budget_seconds` are in `RETIRED_KEYS["round"]` (`cascade/shared/config/loader.py:86-97`, tested in `tests/unit/test_config.py`). One reader remains: `cascade/shared/sandbox_policy.py:93` refuses `dedup_probe_series > 0` on a weak sandbox; with the line deleted it reads the default 0, so behaviour is unchanged. Safe to delete (`chain.toml:498-500`) |
| B14 | plan `:3622`, `:3680` (30 h) | yes. No code implements the alarm yet (no `pool_d_stale` in `cascade/`, `deploy/`, `scripts/`) |
| A4 | no channel handling in `cascade/eval/lane_d_scorer.py`, `cascade/validator/lane_d.py` | yes: lane D is univariate. `_forward` takes `(B, L)` and lifts it to one variate (`lane_d_scorer.py:301-324`, `[:, None]`); `forecast_quantiles` returns `(n, horizon, 9)`. So F2 (C, L) is the first eval feature |
| A4 | `decisions/DEC-CA-0041-multivariate-scoring.md` | yes (rule as quoted) |
| A4 | `chain.toml max_channels = 32` | yes, `[generator] max_channels = 32` (`chain.toml:52`) |
| A5 | TIME dataset list in `cascade/eval/benchmarks.py` | **moved**: `benchmarks.py` only shells out to the sidecar. The TIME source is `Real-TSF/TIME` @ `83e3d0b3be28d11c7182bffcc1892d19b36c4da1` (`benchmarks/cascade_benchmark/datasets.py:56-59`); the task list is timebench's bundled config (`benchmarks/cascade_benchmark/suites/time_bench.py:93-103`). Note: `Datadog/BOOM` (`datasets.py:52-55`) is also a gate suite and is not in the brief's exclusion list; **added to it (owner 2026-11-23, DECIDED: "they're all reasonable")** |

### rev6.18 Part A: eval v2 (decided spec)

**A1 Sources.** Three sources per settlement, all drawn from the settlement
block hash:

| Source | Windows | Half |
|---|---|---|
| live (the existing W5j per-bin draw; forward windows only — ~~+ W5i backtest cells~~, D24) | ≤ 5000 (`max_windows_per_settlement`) | live |
| `Salesforce/GiftEvalPretrain` (Apache-2.0; 71 univariate + 17 multivariate datasets) | 5000 | public |
| `nju-zhangsq/RMISC` (MIT card; allowed folders only, A5) | 5000 | public |

GiftEvalPretrain + RMISC are one **public half** of 10000 windows.

**A2 Public draw.** Pure function of (block hash, index hash, config):

- one window per series, cut point uniform inside the series' valid range
  (history ≥ the scored context's minimum, target = `horizon` points);
- cells = (dataset family, frequency); equal share per cell; a scarce cell is
  taken whole and its surplus re-split over the rest (the W5j rule, same code
  path);
- families merge year-split folders: `cmip6_*`, `era5_*`, `largest_*` are
  one family each (63 of ~150 GiftEvalPretrain folders are climate);
- a small RMISC folder with few long series gets several windows per series
  at distinct cut points (non-overlapping targets), until its cell share is
  met or the series run out;
- the same window list for KING_D, the house and every challenger.

**A3 Winning rule.**

- **(D24)** `L_live` = the mean per-window log-ratio over the forward
  windows (fresh + embargo). ~~`L_live` = the W5i combined log-ratio over forward + backtest windows
  (`backtest_weight_cap = 0.05`, `backtest_ratio_floor`, unchanged; loader rule
  `1 − floor**cap < d_margin` unchanged).~~
- `L_pub` = mean per-window log-ratio over the public windows.
- **Per-window clamp:** every per-window ratio (live forward, ~~backtest,~~ public)
  is clamped to `[window_ratio_floor, 1 / window_ratio_floor]` before it
  enters a half. New key `[lane_d] window_ratio_floor`, overlay-settable,
  default **0.5**: each window's ratio is clamped to **[0.5, 2]** (one window
  can at most halve or double its contribution) (owner 2026-11-23, DECIDED: "they're all reasonable").
  Loader: `0 < window_ratio_floor < 1`.
- `L = (1 − w) · L_live + w · L_pub`, `w = [lane_d] public_weight`.
- **`public_weight`**: new `[lane_d]` key, **overlay-settable, default
  0.50**, range `[0, 1)`. It is **not** subject to the `backtest_weight_cap`
  loader rule ~~(that rule stays as is for the backtest cap)~~ **(D24: that rule and key are deleted in F4)**. `w = 0` turns the
  public half off (not drawn; its point-gain condition not applied).
- Crown iff **all** of: (1) the combined LCB on `L` (one max-T paired cluster
  bootstrap over the cohort, same `alpha`, `bootstrap_B`) ≥ `d_margin`;
  (2) point gain > 0 on the live half; (3) point gain > 0 on the public half
  (when `w > 0`); then (4) the existing §1.3b benchmark gate, unchanged.
- **Bootstrap clusters:** live = source (unchanged); public = dataset family
  (A2). Each bootstrap replicate resamples clusters within each half and
  recombines with `w`, so both halves enter one max-T LCB ~~(the W5i
  machinery, extended from two parts to three)~~ **(D24: a two-part
  live/public combination built in F4; W5i is retired, so there is no
  backtest part)**.
- No confirmation duel, no false-win test, no shadow stage: the rule is live
  at the lane D activation block.

**A4 Multivariate** (DEC-CA-0041, applied to lane D for RMISC and the 17
multivariate GiftEvalPretrain datasets):

- a window is the whole `(C, L)` series, forecast jointly by one model call;
- per-channel CRPS/MASE, channels averaged (arithmetic) into ONE per-window
  score; zero-target channels masked from the quantile (CRPS/WQL) half only;
- a window is one resampling unit; evidence counts clusters (sources /
  families), never channels;
- `C ≤ 32` (`[generator] max_channels`); a wider series keeps 32 channels
  picked by `sha256(block_hash ‖ series_id ‖ "lane_d_channels")`, the same
  for every model;
- drop all-empty columns left by RMISC's file merge (before the 32 cap);
- skip RMISC `KnownOpen` / `KnownOil` duplicate folders;
- univariate windows (`C = 1`) score bit-identically to today.

**A5 RMISC allowed list.** Start from the RMISC folders whose names are not in
`Salesforce/GiftEval`, `Salesforce/GiftEvalPretrain`,
`autogluon/fev_datasets`, `autogluon/chronos_datasets`, the TIME task list
(`Real-TSF/TIME` @ `83e3d0b`; see (V)), or **`Datadog/BOOM`** (@ `69325b5`, a
gate suite; (owner 2026-11-23, DECIDED: "they're all reasonable")) — about 71 of 197 by name. Then per
folder: (1) confirm by **data** comparison (series hashes on sampled
windows), not names; (2) drop non-forecasting sets (EEG/EMG, activity
recognition, classification, anomaly); (3) read `meta.json`, drop any licence
that forbids redistribution. Commit the list as
`data/lane_d/rmisc_allowed.toml` (folder, reason kept, licence, revision).
Publish the filtered copy + list as the official training source (19 RMISC
folders are GIFT-Eval test data). The data comparison in step (1) covers
BOOM as well as the other four sources and TIME.

**A6 Storage.** Mirror full GiftEvalPretrain and the allowed RMISC folders,
each at one pinned HF revision, to Hippius and R2 (owner provides bucket and
keys; size on disk measured from HF metadata before the copy and reported).
Per dataset an **index**: one row per series (family, frequency, length,
channel count, file, row/offset). Each settlement draws from the index and
reads only the drawn series. The signed report carries `public.index_sha256`,
`public.files_sha256` (the file list) and `public.revisions`.

**A7 Removals and keeps.** ~~W5i backtest windows stay as built.~~ **(D24) W5i
backtest windows are dropped (not built); `backtest_weight_cap` /
`backtest_ratio_floor` are deleted in F4.** **A10 / W3b5
is removed** from code (`cascade/validator/lane_d_lowfreq.py`, its call in
`validator/lane_d.py`), the report schema (`lowfreq` verdict block,
`lowfreq_regression` reason; a new schema minor, see F4), the audit
(`audit/lane_d.py` `_lowfreq_problem`), config (`[lane_d] lowfreq_freqs`,
`tau_public` stays only if the §1.3b gate uses it — it does) and the plan
(rows struck below).

**A8 Reporting.** Every entry's report row carries: combined point and LCB;
per half (live, public) point gain; per source (live, GiftEvalPretrain,
RMISC) point gain; per domain (family) point gain and window count.

**A9 Timing (before launch).** 15000 windows × 10 models on the scoring GPU
must fit in ≤ ½ settlement (≤ 1.5 h at 900 blocks). Owner approves the pod,
cost cap **$40**. Evidence file `evidence/lane_d_a9_timing.json`.

### rev6.18 Part B (design text)

**B1 Burn scoping.** A failed lane D check burns **only lane D's share**.
`derive_weights` builds lanes A, B, C as today and, when `_lane_d_court`
returns a BURN code, substitutes a lane-D-burn part (lane D's share →
`burn_uid`) instead of returning `None`. The satellite then replays a vector
whose A/B/C parts are unchanged. Done: a deliberately broken lane D report
on a satellite leaves lane A, B and C weights unchanged (test).

**B2 One checker.** The lane D checker for outside validators is
**`cascade.audit.lane_d.check_lane_d`**, reached by the satellite through
`audit_receipt` → `_lane_d_court`. `cascade/satellite/lane_d_replay.py`
(1,885 lines) and its tests are deleted (`tests/accept/w7d/*`, the
`lane_d_replay` helpers in `tests/accept/{w3b6b,w5e,w8b}`; the W3b6b
re-derivation of bench rows moves into `check_lane_d` first if not already
there — F1 checks). W7d's row becomes "satellite = audit checker".
Invariant 12 stays (the audit never imports producer code).

**B3 Refusal codes.** Only four checks stop pay (BURN, lane D share only):
report signature; config echo = `config_at(boundary)`; scores reproduce the
named winner (`rows_sha256` and bootstrap); weights match the paid court.
Everything else logs, alerts (`cascade-ops`), and keeps paying the signed
court. `check_lane_d` returns a code per check (`code: detail`), and the
audit row keeps the code.

| Code | Check | Class |
|---|---|---|
| `lane_d_record_missing` | lane D pays but no record | burn |
| `lane_d_record_malformed` | record not `{report, receipt}` | burn |
| `lane_d_boundary_mismatch` | report/record boundary ≠ settlement | burn (signature binding) |
| `lane_d_report_mismatch` | receipt binds another report body | burn (signature binding) |
| `ld_schema` | unknown report schema | burn |
| `ld_signature` | report signature | **burn** |
| `ld_not_armed` / `lane_d_not_armed` | config says lane D does not judge this boundary | burn (config echo) |
| `ld_params` | stated params ≠ `config_at(boundary)` | **burn** |
| `ld_pay_share` | `pay.share` ≠ config | burn (config echo) |
| `ld_seed` | bootstrap seed ≠ `lane-d/1:{boundary}:{pool}` | burn (reproduce) |
| `ld_rows_sha256` | rows ≠ `rows_sha256` | **burn** |
| `ld_rows_count` / `ld_rows_duplicate` | row set ≠ cohorts | burn (reproduce) |
| `ld_cohort` | verdict / LCB / sub-scores don't re-derive from rows | burn (reproduce) |
| `ld_crown` | crowned ≠ re-derived winner (incl. house retake) | **burn** |
| `ld_court` | court ≠ KING_D first / too many slots / house as prior | **burn** |
| `lane_d_carve_unsupported` | lane D share can't be carved | burn (weights) |
| `ld_malformed` | report unreadable (`KeyError`/`TypeError`/`ValueError`) | burn (cannot reproduce) |
| `ld_scoring_method` | not `quantile_head` | alert |
| `ld_sentinel` | era sentinel pin | alert |
| `ld_row_order` | row order | alert |
| `ld_clock` | both-clock eligibility | alert |
| `ld_embargo` | embargo / draw rules | alert |
| `ld_draw` | window draw re-derivation (live and public) | alert |
| `ld_judged_set` | coldkey caps, queue, registration | alert |
| `ld_chain` | `held_from` provenance in prior reports | alert |
| `ld_snapshot` | snapshot / pool files | alert |
| `ld_public_index` | public index / file list / revision hash (A6) | alert |
| `ld_bench` | §1.3b gate rows (W3b6b) | alert |
| (WARN) verifier unavailable | sr25519 verifier missing | transient (retry; neither) |

Tests: one burn case (`ld_rows_sha256`: lane D burns, A/B/C unchanged) and
one alert case (`ld_clock`: alert row, pay unchanged).

**B4 Timing keys.** In flight (rev6.14 build, `laned/rev612-rev614`):
`eval_delay_hours = 6`, `forward_margin_hours` default 0, `chain.toml` line
deleted, `boundary_ts` from chain `Timestamp`, fail closed. Not repeated.

**B5 Qualified scoring environments.** §1.6 / D11's "pinned torch 2.4.1 only,
not Blackwell" is replaced by a list `deploy/laned/scoring_envs.toml`: each
entry = `name`, lock file + sha256 (a separate scoring venv lock; the lane A
`uv.lock` / `torch==2.4.1+cu124` contract is untouched), torch build, CUDA,
allowed GPU models, driver floor. Launch list: `cu124-torch241` (the release
venv) and one sm_120 environment for RTX PRO 6000 Blackwell (the bench
sidecar's torch line). **Qualifying a new environment:** (1) PR adding the
entry; (2) on that GPU run the W0b context-equivalence test and a golden set
(500 live + 500 public windows, 3 models) against `cu124-torch241`: per-window
score relative difference ≤ 1e-3 and identical crown decisions on a replayed
cohort; (3) A9 timing passes; (4) evidence `evidence/lane_d_env_<name>.json`
committed; owner merges. Lane D scores are only compared within one cohort
on one machine (B6), so cross-environment bit-identity is not required;
satellites re-derive from signed rows, not by re-forecasting.

**B6 Same machine, same run.** Every model in a cohort (entries, KING_D,
house) is forecast on one host in one run. The score-cache key becomes
`(digest, context, window_id, host_identity)`, `host_identity =
sha256(env name ‖ GPU model ‖ driver ‖ torch build)`; a host change forces a
re-score of the cohort.

**B7 Report host fields.** The signed report carries `host.{gpu_model,
driver, torch_build, env}`; the handshake (`lane_d_handshake.py`) adds
`env` + `torch_build` to `HANDSHAKE_FIELDS` and refuses an environment not in
B5's list.

**B8 Ship the rate-limit emission fix alone.** Content = `ab0b412` (one hunk
in `cascade/validator/weights.py` + one test). Receipt-only: the vector,
ladder state and satellite derivation are unchanged; not a consensus change.
`scripts/deploy_gate.py check` as written refuses it twice: (1)
`provenance` — `ab0b412` is not an ancestor of origin/main (its squash is
`0c52d07a`); (2) `consensus` — `cascade/validator/weights.py` matches
`CONSENSUS` and the diff has no `*_from_block` key; the consensus rule can't
be acked. To accept it the gate needs one of: (a) a hotfix provenance rule —
a `release/*` revision whose every commit is `git patch-id`-equal to a
commit on origin/main (here `ab0b412` ≙ the weights hunk of `0c52d07a`); and
(b) a `--receipt-only` class for consensus-path diffs that is refused unless
the CI receipt-replay job is green on that revision, recorded in the gate
output. Alternatively the owner deploys main (≥ `0c52d07a`), which brings
507 commits and is not "on its own". Owner deploys; gate change is optional
work (not in this batch).

**B9 Config clean-up (S0a).** Classes for the 48 measured keys (V):

| Class | Keys | Action |
|---|---|---|
| operator-only (box money/ops; never read by other validators) | `eval.calib_num_samples`, `eval.calib_windows`, `generation.exclude_offers`, `round.funded_host_blocklist`, `round.operator_balance_reserve_usd`, `round.operator_daily_spend_cap_usd`, `round.operator_inventory_host_files`, `round.operator_lanes_operator_billed_only`, `round.operator_max_price_per_hour`, `round.operator_standing_pods_usd_per_hour.{valeval-blackwell2,valpriv-4090}`, `round.ops_continuation_{image_digest,interval_seconds,max_bytes}`, `round.payer_projected_rate_per_hour`, `round.progress_first_step_seconds`, `round.receipt_watch_validator_hotkey`, `round.funded_max_leg_cost_usd`, `round.funded_ready_timeout_seconds` (19) | mark in the deployed file (dedrift `operator` marker); main unchanged |
| main ← deployed (measured live values) | `round.funded_sku_wall_seconds.{8xH100,8xH200}`, `round.funded_sku_wall_measured_image` (3) | type into main with their evidence (#532 pattern) |
| release lag (main ahead; deployed takes them at its next release) | `lane_c.from_block`, `lane_c.targets`, `marketcast.{from_block,suites,coarse_history_days,eval_context_length,max_synthetic_fraction,snapshot_days}`, `v2.commit_from_block`, `scoring.crowning_anchor_from_block`, `v2.{funded_pod_image_after,funded_sku_wall_measured_image_after,image_after_unqualified_override,prior_train_image_digests,sentinel_from_block,sentinel_ratchet,sentinel_suites,sentinel_tau,train_image_digest_after,worker_code_fingerprint_after}` (20) | none in main. Note: main arms `[marketcast]`/`[lane_c] from_block = 9579600`; the deployed file has 0 |
| decided votes to type (`cascade-audit decided` findings) | `v2.prelaunch_burn_from_block = 9486900`, `v2.image_decoupled_from_block = 9493200` (2; deployed 0) | owner types both into the deployed file at its next quiet restart ⇒ `decided` exits 0 |
| owner keys, **decided: add to main at live values** (owner 2026-11-23, DECIDED: "they're all reasonable") | `training.budget_denomination_after(_block)`, `static_guard.packed_sources(_from_block)` (2 + 2 gates) | F6 types `[training] budget_denomination_after = "points+mv20"`, `budget_denomination_after_block = 9165600` and `[static_guard] packed_sources = "reject"`, `packed_sources_from_block = 9169200` into main's `chain.toml` |

The 10 already-marked drifted keys stay with #532 (R6x). Done when
`cascade-audit decided --config <deployed>` exits 0.

*Owner keys (live values). **Decided (owner 2026-11-23, DECIDED: "they're all reasonable"): both are added to main at their live values (the "Add to main" branch of each bullet below).***

- `[training] budget_denomination_after = "points+mv20"`, `_block = 9165600`.
  The DEC-CA-0047 scheduled budget switch of the **v1** contract
  (`TrainingContractConfig.at_block`). v2 is live from 9399600 and
  `at_block` returns the v2 contract there (the loader refuses a schedule
  inside `[v2.training]`, `loader.py:324`), so the key only shapes eras
  [9165600, 9399600). **Add to main:** audit/replay of those v1 rounds
  computes the contract they trained under. **Leave out of main (delete from
  deployed):** a validator on main re-derives those rounds' contract as
  `series_points`, so their contract digests won't match the archived
  receipts; live rounds are unaffected.
- `[static_guard] packed_sources = "reject"`, `_from_block = 9169200`.
  Live: miner source packed in string constants is refused at admission
  (`trainer/rolling.py:2739`). Main's default is `"scan"` (unpack and
  inspect). **Add to main:** every validator/trainer on main keeps
  rejecting; the release matches live. **Leave out of main:** the next
  deploy relaxes the static guard to `scan` (CLAUDE.md treats relaxing
  `[static_guard]` as Blocked-tier), and admission outcomes for packed
  submissions change.

**B10 Overlay proof (S0c).** One overlay that restates current values,
testnet then mainnet. Owner publishes. No code.

**B11 Legacy lane D keys.** Delete `confirm_on_king_change`
(`lane_d.py:86`, parse `:264-266`, `:336`) and the DEC-CA-0062 block
(`lane_d.py:100-115`) with their `chain.toml` `[lane_d]` lines
(`confirm_on_king_change` and `targets … gain_margin`). The loader then
**ignores** them (listed in `RETIRED_KEYS["lane_d"]` so a deployed file still
loads).

**B12 Init gate.** ~~**STOPPED** (see V): the receipt fields are read. Not
specified further until the owner picks.~~ **Decided (owner 2026-11-23, DECIDED: "they're all reasonable").** Delete the
init gate: `[scoring] init_gate_mode` / `init_gate_tolerance` (config,
loader parse, `chain_config.py:209-223` pass-through, `chain.toml:910-911`),
the producer in `cascade/eval/koth.py` (`KothParams.init_gate_mode`, the floor
computation and both `enforce` branches, `:233`, `:454`, `:594-618`) and the
log in `cascade/validator/loop.py:583-592`. **Keep** `cascade/shared/receipt.py`
parsing `init_baseline_geomean` / `init_floor_passed` (`:350-351`, `:453-455`,
`:498-501`, `:974-978`): read and re-serialised drop-when-None, never
written again, so every archived receipt still verifies. The `[round]
init_gate_mode` retired key (`loader.py:91`) is unchanged. This is lane A
consensus config: it ships in a **release with an activation block**
(DEC-CA-0045 typed `*_from_block`; before the block the gate runs as today in
shadow, from it the producer stops), never as a hand edit. Test: an archived
shadow-era receipt with both fields round-trips byte-identically and its
signature verifies.

**B13 Dedup probe lines.** Delete `chain.toml:498-500`
(`dedup_probe_mode`, `dedup_probe_series`, `dedup_probe_budget_seconds`).
Loader behaviour unchanged (V). The `RETIRED_KEYS` entries stay.

**B14 Stale pool alarm.** ~~30 h~~ **6 h** (`pool_d` builds every 3 h, so
6 h = two missed builds). Lines struck below.

**B15 Miner status context.** `status/lane_d.json` (W17) and `cascade
lane-d status` (W10) show, per entry, `declared_context` and
`scored_context` (from the signed report's per-entry effective context;
`null` + `context_below_armed` when skipped, rev6.15).

**B16 Disputes and wrong crowns.** The owner decides every dispute and every
claimed wrong crown; there is no automatic reversal and no appeal process.
Pay already sent stands: nothing is clawed back and no past weight vector is
re-set. A reversal is prospective only:

- the owner publishes a signed overlay (or a release note for a code fault)
  naming the reversed crown and the boundary it takes effect from;
- the next lane D report states `reversal: {reversed_digest, from_boundary,
  reason, overlay_sha256}`; its `king_before` is the reversed king and its
  `king_after` is the king the owner restores (the previous KING_D or the
  house);
- `lane_d_state.json` gets the restored king and a reversal entry; earlier
  posterior digests are untouched, so the state chain and every archived
  report still verify;
- the reversed digest drops out of the court from that boundary; no other
  entry's status is changed retroactively.

**B17 (deferred).** After Part A and B1–B7 land: a short current-rules spec
and the rev history moved to a changelog. Separate PR.

### rev6.18 (F) Feature build list

One PR per feature, tests and code together (rev6.17 rule). WP labels name
the rows each PR touches.

| # | Feature | Items | WPs touched | Depends |
|---|---|---|---|---|
| F1 | Burn scoping + one checker + refusal codes | B1, B2, B3 | W9 (`audit/lane_d.py` codes, `checks.py` `_lane_d_court`/`derive_weights`), W7d (deleted; satellite = audit), W8b (satellite path), W11 (alert patterns) | — (first: it removes the second checker before F2–F5 would change two) |
| F2 | (C, L) multivariate lane D scoring | A4 | W5d (scorer `(B, C, L)`, channel pick, empty-column drop), W5e (per-window collapse), W7a (per-window `n_channels`, `channels_sha256`), W9 | F1 |
| F3 | Public sources: allowed list, index, draw, mirror tooling | A1, A2, A5, A6 (code); data ops = owner | new W5k (public draw), new W6p (index builder + mirror script, read-only on HF until owner keys), W7a (`public.*` hashes), W9 (`ld_draw`, `ld_public_index`) | F2 |
| F4 | Combined winning rule (live forward + public) + per-half reporting + A10 removal + backtest removal (D24) | A3, A7, A8, D24 | W2e (add `public_weight`, `window_ratio_floor`; **delete** `backtest_weight_cap`, `backtest_ratio_floor` and the loader inequality, both keys to `RETIRED_KEYS["lane_d"]`, `chain.toml` lines out), W5i (retired, not built), W5j (forward cells only), W5e (two-part live/public max-T LCB), W3b5 (removed), W7a (schema minor: per-half/source/domain fields; `lowfreq` and the forward/backtest sub-score split dropped), W9 (`combined_log_ratio` removed; re-derives the new rule), W12a, W17 | F3, F1 |
| F5 | GPU environments + handshake + cache key | B5, B6, B7 | W4 (handshake fields), W15 (cache key), W7a (`host.*`), W6-S8 / T7 (env list), D11 | — (parallel with F2) |
| F6 | Config cleanup + init-gate removal | B9 (main-side typing incl. both owner keys), B11, B12, B13 | W2, `chain.toml`, R6x / #532, `koth.py`, `loop.py`, `scoring.py`/`loader.py` (B12 is lane A consensus: release + activation block) | — |
| F7 | Miner status context | B15 | W17, W10, W12b | rev6.15 W7a/W8c landed |
| F8 | A9 timing harness + evidence | A9 | T7, W1 (`scripts/laned_a9_time.py`) | F4, F5; owner pod |

Not in the list: B4 (in flight), B8 (owner deploy; gate change optional),
B10 (owner overlay), B14 and B16 (this plan text), B17 (deferred).

**Owner steps left:** type the two decided votes into the deployed file;
buckets + keys for A6 and the mirror run (data ops); approve the A9 pod ($40
cap); deploy B8; publish the B10 overlay (testnet, then mainnet); pick the B12
activation block when the F6 release is cut. (Resolved (owner 2026-11-23, DECIDED: "they're all reasonable"): B12
option, both B9 owner keys, A3 clamp 0.5, BOOM in the A5 exclusion set.)
## rev6.17 (owner 2026-11-23; main @ 78b1d45): one PR per feature; tests+code together; PATHS ride along; one plan PR per batch of decisions

One owner decision about the build loop's PR rules ("stop the PR
bureaucracy"). The design doesn't change. Changed text is marked
**(rev6.17)**, and superseded text is struck through, not deleted. The
tooling is W0c (`scripts/laned/accept` FEATURE rule and LABELLED L2 rule,
`accept --help`), with its acceptance tests in `tests/accept/w0c/`.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Tests + code together (owner 2026-11-23, DECIDED).** A PR may change a WP's `tests/accept/<wp>/` and its code in one diff. CI still proves the contract: the WP's NEW tests, taken from the PR head, run against the base's code and must fail there for a right reason (a new test that already passes on the base proves nothing and is refused, unless the WP is a declared verification WP); then the full Accept must pass on the head. Tests-only and tests-correction PRs stay available but are no longer required. Deleting or weakening an existing assertion (or dropping a `PATHS` entry) needs the `laned:tests-correction` label. The stub scan, invariant 10 and the protected files (`docs/plans/`, `scripts/laned/` outside W0c, status files) stay | §3.1 rules 1, 6; §3 roles; W0c (`accept --combined`) |
| (b) | **PATHS ride along (owner 2026-11-23, DECIDED).** A PR may add entries to `tests/accept/<wp>/PATHS` and change those files in the same PR: scope is the base's PATHS plus the added entries; a removed entry still counts from the base | §3.1 rule 1; W0c |
| (c) | **One PR per feature (owner 2026-11-23, DECIDED).** A PR whose labels are all `laned:<WP>` may carry every WP a feature touches: no 3-WP cap, no disjoint-PATHS or no-dependency refusal, no `coupled_batches` declaration. Every WP still runs its own Accept (or combined run) and CROSS, with one ledger row each. A tests-only batch has no cap and may mix built and unbuilt WPs | §3.1 rules 6, 7; W0c |
| (d) | **L2 needs the label (orchestrator 2026-11-23, follows (c)).** A squash merge `(#N)` is a WP's L2 commit only if PR N carried `laned:<WP>` and changed at least one non-test path in its PATHS (#817 made W5i read L2 through broad PATHS). A merge that changes the WP's tests is also its L1 when its combined run is green | §17.1 L1, L2; W0c (status) |
| (e) | **One plan PR per batch of decisions (owner 2026-11-23, DECIDED).** Plan text stays out of WP PRs (`docs/plans/` is protected); the decisions of one owner session land in one plan PR | §17.6 |

## rev6.16 (owner 2026-11-23; main @ 65bcb4b): partial WB19 evidence

One owner decision. Plan text only. The changed row is marked **(rev6.16)**.

**The problem.** WB19's `combine` fails its `coverage` check unless every
`[lane_b] funded_pod_skus` type and every OA8 type has an expert run. The
step-rate reader refuses any evidence whose `passed` is not true. Only 8×H200
(before WB25, single-GPU path) and 8×RTX4090 (with WB25) are measured, so
every lane B type was refused `gpu_rate_unmeasured`, including the two that
were measured.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Allow partial WB19 evidence (owner 2026-11-23, DECIDED, verbatim choice: "Allow partial").** A `laned_wb19/1` file whose **only** failing check is `coverage` still loads, and its measured rows are usable. Every type with no row (from the lane B list or from OA8) stays `gpu_rate_unmeasured`. Any other failing check (`memory`, `full_leg`), malformed checks, a missing or mismatched body hash refuses the whole file. `passed` stays the honest conjunction of the checks. `combine --note SKU=TEXT` marks a row measured on a superseded path. The 8×H200 row is marked pre-WB25 single-GPU: it understates a multi-GPU leg. Rates are never invented for unmeasured types | WB19 row; WB22 row; WB19 / WB22 tests-correction and WB19 code PR (`cascade/funding/step_rates.py`, `scripts/laned_wb19_measure.py`) |

## rev6.15 (owner 2026-11-23; main @ 9888c1a)

Two owner decisions (a, b) and their follow-ups (c). Plan text only. Every
changed item is marked **(rev6.15)**; superseded text is struck, not
deleted.

**The problem (a).** Since rev6.6 the plan says an entry that declared an
unarmed context variant "is scored at the highest armed variant" (§1.1
context bullet, rev6.6 (a), D2, T7, §11.2 `docs/LANE_D.md` row). Read
literally, that can run a model **above** the context its miner declared.
The W5d code on main (`cascade/eval/lane_d_scorer.py`, `effective_context`)
takes the highest armed variant ≤ declared, but when every armed variant is
above the declared one it falls back to the **lowest armed** variant, which
is still above the declaration. Its tests pin that fallback
(`tests/accept/w5d/test_w5d_arming.py::test_never_scored_above_the_declared_context`,
`ec(4096, (8192, 16384)) == 8192`; `tests/accept/w8c/test_w8c_armed.py`
case `(4096, (8192, 16384), 8192, 8192)`).

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Highest armed ≤ declared, else skip (owner 2026-11-23, DECIDED, verbatim choice: "Highest ≤ declared, else skip").** An entry is scored at its declared context when that variant is armed, else at the **highest armed variant below its declared context**. If no armed variant is ≤ the declared context, the entry is **not scored this settlement**: it stays `pending` with the new typed skip reason **`context_below_armed`**, and it is never run above its declared context. Spec below | §1.1 context bullet; rev6.6 (a); W5d, W7a, W7d, W8c, W9 rows; T7; D2 (§6); §11.2 `docs/LANE_D.md` row; §17.4 D2 |
| (b) | ~~**Backtest windows ON at launch (owner 2026-11-23, DECIDED).**~~ **Superseded (owner 2026-11-24, DECIDED) by rev6.18 D24 (backtest dropped).** `[lane_d] backtest_weight_cap` default **0.05** (was 0, "off by default", rev6.9). `backtest_ratio_floor` stays 0.85; the loader inequality `1 − 0.85^0.05 ≈ 0.0081 < d_margin` (0.01) holds at the defaults, so the default config loads. With the default, the slow-cadence backtest cells (W5i) are built and scored from launch. Setting the cap to 0 by overlay still turns them off (the rev6.9 "at 0" behaviour is kept as the off switch). The W2e default and its tests are changed by a separate builder; this rev changes plan text only | rev6.9 "Off by default" bullet; rev6.10 "What gets drawn" bullet; §1.7 horizon paragraph; W2 row; W5i row |
| (c) | **Follow-ups** (WP list below) | this section |

**(a) Spec: the scored context.**

- **`effective_context(declared, armed_contexts) -> int | None`** (W5d,
  `cascade/eval/lane_d_scorer.py`): `declared` when it is armed; else
  `max(c for c in armed if c < declared)`; else **`None`** (no armed variant
  is ≤ `declared`). `ValueError` stays for an empty armed set and for a
  `declared` outside the scorer contexts. Pure and deterministic.
- **Which forecasts W8c asks for** (`cascade/validator/lane_d_inputs.py`,
  `_needed`): a digest whose effective context is `None` is **not
  forecast** (no eval-host call). That covers an entry, an open record, and a
  miner KING_D alike. No new `LaneDInputsError` code: a skipped digest is a
  settlement outcome, not a missing input. **The house is exempt.** It
  declares no context; it is ours, trained at 4096. It keeps the rev6.6 W8c
  rule: 4096 when armed, else the highest armed variant below it, else the
  lowest armed one. So the house bar is always forecast, and one unarmed
  4096 variant doesn't stall the whole lane.
- **What the settlement does** (W7a, `cascade/validator/lane_d.py`
  `settle_lane_d`, only when the `armed` record is given): after the W5h
  queue selection, each chosen member whose declared `context` has no armed
  variant ≤ it is removed from judging. It gets reason `context_below_armed`,
  keeps status `pending` (no frozen-set advance, no verdict), and is listed
  in `skipped`. The house challenger is exempt (no declared context, see
  above). The test is `min(armed_contexts) >
  context`, computed from the signed `armed_contexts`, so audit and
  satellites re-derive it. It does not import the scorer. A cohort whose
  miner KING_D has no forecast for this reason already skips its members
  `king_unscored` (existing rule, unchanged). A held gate candidate
  is re-gated, not re-duelled, and is unaffected. The legacy call
  (`armed=None`) is unchanged.
- **Queue slot.** The skipped member keeps the slot it was selected into for
  this settlement (no re-selection inside a settlement, so the result stays
  one deterministic pass). This only happens when the 4096 variant is
  unarmed while a higher one is armed. The gates make that unlikely: readiness
  counts can only fall as context grows, and T7 is cheapest at 4096.
- **Report** (W7a): `context_below_armed` joins `SKIP_REASONS` (and so
  `ENTRY_REASONS`) in `cascade/validator/lane_d_report.py` and both enums of
  `cascade/validator/schemas/lane_d_report_1.json`. It stays on
  `lane-d-report/1`: the enum grows, and no existing report changes.
  Cross-check: a `context_below_armed` skip needs the report's
  `armed_contexts` present with `min(armed_contexts) > entries[].context`.
- **Name.** `context_below_armed` sits next to the existing
  `context_not_armed`, which means a context outside `[lane_d] contexts`
  (closed at registration). The new name says what happened: the declared
  context is below every armed variant. The owner's example name
  `context_not_armed_le_declared` was a placeholder ("pick a name consistent
  with existing skip reasons").

**(c) Follow-ups.** One tests-correction batch first, then one
implementation PR per WP (W5d and W8c can't share a batch: W8c depends on
W5d):

1. **Tests-correction W5d + W8c + W7a** (one batch, ≤ 3 built WPs): W5d's
   fallback test becomes `None`. W8c's `(4096, (8192, 16384))` case becomes
   "entry not forecast, the house at the lowest armed", plus an
   open-record case and a determinism case. W7a adds the enum value to its pins and a settle test
   (`context_below_armed`, status `pending`, listed in `skipped`, report
   valid).
2. **Tests-correction W5g**: `context_below_armed` gets a reason case (or an
   `OWNED_ELSEWHERE` entry naming W7a), so W5g's "every enum reason is cased"
   check stays green after the W7a implementation.
3. **W12a**: `docs/LANE_D.md` FAQ names `context_below_armed`; its "lowest
   armed" sentence (`:73`) is struck. This must land before the W7a
   implementation PR, because W12a's FAQ test names every skip reason.
4. **Implementation W5d** (`effective_context` → `None`).
5. **Implementation W8c** (`_needed` skips `None`).
6. **Implementation W7a** (settle skip, enum, schema, cross-check).
7. **W9 and W7d** (later, their own tests first): audit `check_lane_d` and
   satellite replay re-derive `context_below_armed` from the signed
   `armed_contexts` and each entry's declared context. They refuse a report
   that skips an entry that had an armed variant ≤ its context, or that
   judges one that had none.
8. **Text-only stragglers**: `tests/accept/w3/README.md:151`,
   `tests/accept/w5d/README.md:99-104` and `tests/accept/w8c/README.md:214-216`
   still say "highest armed variant" or "lowest armed". Each is fixed in its
   WP's next tests PR (W5d and W8c in item 1).
9. **(b)**: the W2e default and its tests are another builder's PR. W5i's
   tests must not assume the cap defaults to 0; check them when W5i is
   built.

## rev6.14 (owner 2026-11-23; main @ f3e4f6c)

Corrects one conflation (a) and one formula (b), and lists the follow-ups
(c). Plan text only. Every changed item is marked **(rev6.14)**; superseded
text is struck, not deleted. Found by the W1 measurement agent.

**The problem.** Since rev6.10 the plan has used one value,
`b = wall(max(T_E, T_K)) + forward_margin_hours` (6 h), for two jobs:

1. the **judging start**: no set frozen or judged before `b`;
2. the **fresh / eligibility boundary**: fresh ⇔ `data_ts > b` and
   `first_seen > b` (rev6.10 (a), the §1.3 rule block, rev6.12).

So at S1, the first settlement ≥ `b`, the fresh target interval is
`(T+6 h, a]`, only 0–3 h long. Hardly any fresh window exists there. With
`min_fresh_window_share` = 0.5, nothing can be judged until fast feeds have
built up 64 points after T+6 h. That contradicts the owner's decision,
verbatim (2026-11-23): "use 6 hours after submission so get data impossible
to be trained but supplement the hour + freq with embargo data" and "I want
to delay eval by 6 hours and use series after this time plus embargo data".
Data that arrives **after submission T** is the fresh (impossible to train
on) target. The 6 h is the **wait before judging**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Split the two roles (owner 2026-11-23, DECIDED: the owner's stated rule).** New key **`[lane_d] eval_delay_hours` = 6** is the judging-start rule: no set is frozen or judged for an entry at a settlement before `j = wall(max(T_E, T_K)) + eval_delay_hours`. The eligibility / fresh boundary stays `b = wall(max(T_E, T_K)) + forward_margin_hours`, with the default of **`forward_margin_hours` changed 6 → 0** (key kept). Fresh and embargo formulas unchanged, with the new `b`. Per-cadence latency redone (table below) | rev6.12 (a) rule bullet and latency caption / table note / per-cadence note; rev6.10 (a) `forward_margin_hours` bullet and latency caption; §1.3 rev6.10 rule block (and its stale rev6.12 share-cap leftover), "Superseded by the two-clock rule" bullet, "Evidence accumulates" freeze bullet; §1.7 floor maths + 5-min latency bullet; W1; W2e; W5a; W5e; W7a; W7d; W9; T8e; v3 self-check T8e row; §6 D5; §14 rev6.10 row; §17.4 D5 |
| (b) | **ERRATUM: the embargo bound.** rev6.12 (b) step 3's `F × s / (1 − s)` is wrong except at s = 0.5. The bound that keeps the fresh share F / (F + E) ≥ s is **`E ≤ F × (1 − s) / s`**. The W5j acceptance tests (`tests/accept/w5j/README.md` step 7, `F ≥ s · (F + K)`) already use it; `cascade/validator/lane_d_draw.py` isn't on main yet and must follow the tests. Both values are F at s = 0.5, so the live default is unaffected | rev6.12 (b) step 3; W5j |
| (c) | **Follow-ups** (WP list below) | this section |

**(a) Spec: judging start vs eligibility boundary.**

- **Two times per entry.** With T = `wall(max(T_E, T_K))`:
  - `b = T + forward_margin_hours` (default **0**, so `b = T`): the
    eligibility / fresh boundary;
  - `j = T + eval_delay_hours` (default **6**): the judging start. The
    entry is first drawn, frozen or judged at the first settlement whose
    boundary wall time is ≥ `j`.
- **Rules (formulas unchanged, new `b`).** For every target point p:
  - **fresh** ⇔ `data_ts(p) > b` **and** `first_seen(p) > b` (the e = 0
    rule);
  - **embargo** ⇔ `release(p) = first_seen(p) + e > b` **and**
    `data_ts(p) > b − e`, with `e` as in rev6.12 (a). A window that isn't
    fresh at every point (incl. mixed) is an embargo window.
  - S0 / backfill use `b − e` (unchanged).
  - Mirror-lag, forecast-product and missing-clock refusals are unchanged.
- **Invariant:** `forward_margin_hours ≤ eval_delay_hours`, so `b ≤ j`.
  Otherwise judging could start before any fresh point can exist. W2e
  refuses a config that breaks it.
- **Why `forward_margin_hours` = 0.** The first-seen clock (W6fs) proves a
  point didn't exist at commit, not a time margin:
  - A point with `data_ts > T` from a feed that doesn't publish ahead
    didn't exist anywhere at T, so no checkpoint fixed at T can have
    trained on it.
  - A feed that does publish ahead is caught by the first-seen clock
    whenever our mirror saw the value by T (`publish_ahead`). W6b /
    W6fs mark it per feed. A fixed margin of a few hours wouldn't stop
    day-ahead publication anyway.
  - So the six hours after submission are usable target, as the owner
    ruled.
- **Residual clock-skew argument (stated; no margin recommended).**
  - T comes from the chain block timestamp. Block time is ~12 s, and the
    timestamp can drift a few seconds from true time. `first_seen` comes
    from our NTP-synced box clock. `data_ts` is the source's own stamp.
  - The combined skew is seconds, below one step of the fastest (10 s)
    cadence, and far below what a miner needs to train on a point and
    commit.
  - Interval-labelled bars (a 5-min bar stamped at its start) still pass
    `data_ts > T` only if the bar opened after T, so that's safe.
  - **Rec:** keep 0. Only if W6fs measures box-vs-chain skew above 60 s,
    set `forward_margin_hours` to the measured skew rounded up to whole
    minutes, capped at 0.1 h (6 min). It's a float key in hours
    (`cascade/shared/config/lane_d.py:59`), so minutes fit.
  - The owner's rule, that the 6 h after submission is usable target,
    stands either way.
- **What changes at the boundary.** With `b = T`:
  - points with data time in `(T, T+6 h]` become fresh;
  - embargoed points are those with data time in `(T − 48 h, T]` and
    `release > T`;
  - points released in `(T, T+6 h]` are now admitted. Those weren't
    released by us before commit either, so this isn't a new exposure.

**Per-cadence latency (rev6.14; h = 64; e = 48 for all; 3-hourly aligned
pool_d builds; 3 h settlements).**

- `b = T`, `j = T + 6 h`. S1 is the first settlement ≥ `j`, in
  [T+6 h, T+9 h). The snapshot is at `a ≈ S1`.
- Fresh target interval `(T, a]`: 6–9 h. Embargo-eligible interval
  `(T − 48 h, T]` (48 h, the same length as rev6.12's whole interval),
  plus mixed windows that straddle T.
- Disjoint windows per series are tiled from `a` back. Fresh =
  ⌊fresh points / 64⌋. Embargo = all windows in `(T − 48 h, a]` minus
  fresh, so mixed windows count as embargo.
- This is arithmetic, not measurement; W1 and B2 replace it.

| Cadence | Fresh points per series at S1 = T+6 h (T+9 h) | Fresh windows per series at S1 (T+6 h / T+9 h) | Embargo windows per series at S1 (incl. mixed; ≈ rev6.12) | First settlement with a fresh window | First settlement with any eligible window |
|---|---|---|---|---|---|
| 10 s | 2160 (3240) | **33** / 50 | 270 | S1 | S1 |
| 1-min | 360 (540) | **5** / 8 | 45 | S1 | S1 |
| 5-min | 72 (108) | **1** / 1 | 9 | S1 | S1 |
| 10-min | 36 (54) | 0 / 0 | 5 (4 + 1 mixed) | ≈ T+10.7–13.7 h | S1 |
| 15-min | 24 (36) | 0 / 0 (mixed windows count as embargo) | 3 | ≈ T+16–19 h | S1 |
| 30-min | 12 (18) | 0 / 0 | 1 | ≈ T+32–35 h | S1 |
| Hourly | 6 (9) | 0 / 0 | 0 | ≈ T+64–67 h | ≈ T+16–19 h (was T+22–25 h) |
| Daily short h 7–30 (rev6.9, if armed) | — | 0 | 0 | — | T+5 d … T+28 d, then the next settlement (was T+5.25 d … T+28.25 d) |

- Without aligned builds, take the 3 h-older snapshot: subtract up to
  3 h of fresh points (10 s: 1080 → 16 windows at T+6 h).
- **Fresh share vs floors (pending W1 numbers).**
  - At `min_fresh_window_share` s = 0.5, kept embargo windows are
    ≤ F × (1 − s) / s = F. So the 1000-window floor needs **F ≥ 500**
    fresh windows in the cohort's draw, plus ≥ 20 sources and
    `max_source_share` 0.25.
  - At S1 (T+6 h), 500 fresh windows is ≈ 16 10 s series (33 each), or
    ≈ 100 1-min series (5 each), or ≈ 500 5-min series (1 each), or a mix.
    15-min to hourly supply only the embargo half at S1.
  - Under rev6.12 the same S1 had a 0–3 h fresh interval (10 s ≤ 16,
    1-min ≤ 2, 5-min 0 per series). rev6.14 gives about 2–3× the fresh
    windows at the first verdict and lets 1-min and 5-min feeds count.
  - **Whether the floors are met at S1 is plausible for a cohort with
    tens of 10 s or ~100 1-min series, but unmeasured.** W1 must count
    10 s / 1-min / 5-min sources at the rev6.14 boundary. If a cohort falls
    short, it stays `pending` (`fresh_share_short`) and is re-tried each
    settlement; the share is never relaxed (rev6.12 (b)).

**(b) Erratum (detail).** For F fresh and E embargo windows, F / (F + E) ≥ s
⇔ E ≤ F (1 − s) / s. rev6.12 wrote `F × s / (1 − s)`, which is that bound
inverted; the two agree only at s = 0.5. Example: at s = 0.75, the correct
bound is E ≤ F/3, and the rev6.12 text would have allowed E ≤ 3F (fresh
share 25%). Corrected in rev6.12 (b) step 3 and the W5j row. The W5j
acceptance tests already use `F × (1 − s) / s` (their README notes the
plan's wording matches only at 0.5), so no test follow-up; the W5j
implementation (`lane_d_draw.py`, not on main yet) follows the tests.

**(c) Follow-ups (WPs).**

- **W2 / W2e (config).** Add `eval_delay_hours` (default 6, float, ≥ 0)
  with loader parse, round-trip and mutation test (a dataclass field
  without a loader parse is a silent no-op). Change the
  `forward_margin_hours` default 6 → 0 (dataclass `lane_d.py:59` and loader
  `:244`). Refuse `forward_margin_hours > eval_delay_hours`. Both are
  overlay-settable lane D tuning (§14.2).
  - **Consensus note:** a config default is consensus-relevant, because
    every validator must derive the same `b` and `j`. Lane D is unarmed
    (`active()` hardcoded False, `from_block > 0` refused), so the change
    has **no live effect**. It ships in a release before arming. Before
    arming, diff the deployed chain.toml for an explicit
    `forward_margin_hours = 6` override, which would silently keep the old
    boundary.
- **W5a (b).** `forward_bound()` (`cascade/validator/lane_d_forward.py:81`)
  keeps `max(T) + forward_margin`; with the default 0 it returns T. Fresh /
  embargo classing is unchanged against the new `b`. The "no freeze before
  `b`" text becomes "no freeze before `j`" (W5e owns `j`).
- **W5j.** Strike `F × s / (1 − s)`, use `F × (1 − s) / s` (text only; the
  tests already match, and the implementation follows them).
- **W5e / W7a (judging start).** W5e: an entry is not drawn, frozen or
  judged at a settlement whose boundary wall time is < `j`, with skip
  reason `before_eval_delay`, distinct from `fresh_share_short` and the
  floors. W7a: config echo `eval_delay_hours` and `forward_margin_hours`;
  per entry `b` and `j`; cross-check freeze settlement ≥ `j` and `b ≤ j`.
- **W7d.** The satellite re-derives `b` and `j` from the signed T values
  and the config echo. It refuses a set frozen before `j`
  (`frozen_before_eval_delay`) and re-classes fresh / embargo at the new
  `b`.
- **W9.** The audit independently re-derives both values (`b` for
  eligibility and fresh / embargo class, `j` for the freeze settlement),
  with the same refusals.
- **T-rows.**
  - **T8e** adds: a point with data time and first-seen in `(T, T+6 h]` is
    fresh; a settlement in `[b, j)` freezes nothing; the
    `forward_margin_hours > eval_delay_hours` config is refused; the share
    bound at s = 0.75 (E ≤ F/3).
  - **T8**'s both-clocks cases run at margin 0.
- **W1 (measurement).** `scripts/laned_w1_measure.py` has one
  `--margin-hours` (default 6) used as both the cutoff and the count time.
  It needs **`--fresh-from` semantics matching (a)**: fresh / eligibility
  counted from `T + forward_margin_hours` (default 0), and counts taken at
  the first snapshot / settlement ≥ `T + eval_delay_hours` (default 6),
  as two flags, reporting fresh and embargo windows per frequency
  separately.

**Changed locations (rev6.14).**

- Status line.
- This section.
- rev6.12:
  - (a) spec, eligibility-rule bullet (`b` … (6 h));
  - (b) spec, step 3 (erratum);
  - latency caption (`b = T + 6 h`), a new (rev6.14) note bullet, and the
    per-cadence note (struck as superseded).
- rev6.10 (a): the `forward_margin_hours` = 6 bullet and the latency
  caption.
- §1.3:
  - rev6.10 rule block: `forward_margin_hours` = 6, "before `b`", and the
    stale `embargo_window_share_cap` leftover rev6.12 missed;
  - the "Superseded by the two-clock rule" bullet (recommended 6 h);
  - "Evidence accumulates" freeze bullet.
- §1.7: floor maths; forward latency (5-min bullet's "6 h margin").
- §3.0 rows: W1, W2e, W5a, W5e, W5j, W7a, W7d, W9.
- §4 row: T8e. v3 self-check "T-rows → WPs": T8e row.
- §6 D5, §14.2 rev6.10 row, §17.4 D5.

## rev6.13 (owner 2026-11-23, "build this"; main @ 00223fd)

Adds one transport WP, split in two (a), and records the W8c armed-context
follow-up (b). Plan text only. Every changed item is marked **(rev6.13)**.
**Ordering note:** rev6.12 (#752) merged first (c80e3a4). This section edits none of its
rows; only the top insertion point is shared, and the status line is left
alone. Whichever merges second keeps both sections, newest on top.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **New WP W3b6, lane D bench transport (owner 2026-11-23: "build this").** Gap: an armed lane D can't crown, on testnet or mainnet, because no production code moves lane D candidates' public-benchmark rows to the validator. W8b's step passes `bench=` (W8c's `source.bench`), but `load_inputs_services` returns `bench_host=None` ("no lane D bench host transport exists yet"). Every crowning candidate is therefore `bench_hold`, and after `max_bench_holds` it re-queues. The rows the §1.3b gate decides on also never leave the main validator: the report records the four LCBs only, so a satellite can check the outcome rule but not the LCBs. W3b6 reuses lane A's signed bench path: the same sidecar (`cascade/eval/benchmarks.py`, `item_rows_from_report`), the same pod-side content-addressed checkpoint layout (`eval_offload.BENCH_UPLOAD_FILES`), the same `bench_v/` prefix and the same store and signer as the lane D report. It adds no chain write and no bucket. It is split into **W3b6a** (the sweep transport) and **W3b6b** (publish + bind + satellite re-derive). The two have disjoint PATHS and no dependency on each other, so they can batch | §3.0 W3b rows (after W3b5); this section |
| (b) | **W8c computes the armed context set (rev6.6 follow-up).** W8c's source reads the pinned `laned_readiness/1` and `laned_t7/1` documents named in the `--lane-d-inputs` TOML (`readiness`, `t7`, both required, re-read every settlement). It computes the W5d `armed_record`, forecasts each digest at `effective_context`, and hands the record to `settle_lane_d(armed=)` through `LaneDInputs.armed`. A missing or invalid document, or an empty armed set, fails closed (`readiness_unavailable`, `t7_unavailable`, `armed_invalid`, `no_armed_context`). Tests-correction #753 adds `lane_d_wiring.py` to W8c's PATHS | W8c row |

### W3b6 spec (rev6.13)

**What exists (main @ 00223fd).**

- **Lane A** (DEC-CA-0063, `cascade/validator/validator_bench.py`): the main validator sweeps a clearing challenger and the king on the bench / eval-offload pod (`eval_offload.run_bench_via_host_many`). It ships the checkpoint into the pod's content-addressed cache `_eval_offload/cache/<sha256-…>` (files `BENCH_UPLOAD_FILES`, an upload manifest), runs the sidecar there and parses rows with `benchmarks.item_rows_from_report`. It then publishes the rows content-addressed (`bench_report.validator_bench_rows_key`, `bench_v/rows/<round>/<sha(pointer)>/<rows_sha256>.json`) and a signed `bench_v/round-<id>.json` that pins them by `rows_sha256`. Every validator reads those signed rows and runs the gate itself.
- **Lane D, gate side** (W3b1–W3b5, merged): `lane_d_bench.SweepSession(session_id, sweep)` calls `sweep(digests, suites) -> {digest: {suite: {"status", "rows", …}}}`, which is `item_rows_from_report`'s shape. One session per boundary (`lane-d-bench/<boundary>`, W8c). `paired_gate` decides on the three records. A sweep that raises is `bench_hold` (`lane_d._run_gate`). The report's `gate.results[]` carries `{digest, outcome, lcb, blocked_by, session}`. W7d re-derives the outcome from those LCBs.
- **Lane D, missing:** a production `sweep` (W8c ships `bench_host=None`). A lane D checkpoint is `config.json` + `model.safetensors` (BF16 allowed, W4). It has no `forecast_wrapper.py` / `model.py`, so the sidecar refuses it (`_invoke_sidecar_many` needs the pinned wrapper and `verify_checkpoint_code`). §10.2 forbids weights on the box, so lane A's box→pod `scp` can't carry it. The rows are not published.

**W3b6a — the sweep transport.**

- **Pod side** (new `cascade/eval/lane_d_bench_sweep.py`, runs on the bench host): `python -m cascade.eval.lane_d_bench_sweep hello|sweep`. The request is canonical JSON on stdin: digests, suites, each digest's effective context (rev6.6) and, per digest, the W8c presigned GET grant or `null`. The house and a house KING_D are expected in the cache, as on the eval host.
  - Each digest goes through the W15 `PrefetchCache` on the pod, with the eval host's own rule (`lane_d_forecasts._checkpoint`): a verified hit is used as is, and a miss with a grant runs W4 `prefetch_grant`, so the ingest guards run before admission.
  - **Infrastructure fails the whole sweep** (non-zero exit, the box raises, `bench_hold`): a miss without a grant, a vault or fetch failure, or `over_budget`.
  - **The miner's bytes:** a W4 refusal gets that digest's `{"status": "error", "rows": []}` on every suite, which is the gate's `checkpoint_fault` on a candidate and a hold on a reference.
  - The cached entry already has lane A's sidecar layout (W4 admits exactly `config.json`, `forecast_wrapper.py`, `model.py`, `weights.safetensors`, with the code byte-identical to the release's).
  - It is materialised into a per-(digest, context) sidecar dir: the three other files are copied as is, and `config.json` gets `context_length` = the effective context. The dir must pass `verify_checkpoint_code`. BF16 weights load into the pinned wrapper's F32 parameters (`load_state_dict` copies), so no conversion is needed.
  - Then one paired sidecar pass over all digests (`benchmarks._invoke_sidecar_many`, `--extra-checkpoint`), with `item_rows_from_report` per digest.
  - The reply is `{"schema": "lane-d-bench-reply/1", "rows": {digest: {suite: entry}}, "fingerprint": {...}}`. The fingerprint is the GPU model, driver and sidecar torch, as `bench_lease` tags lane A rows. A digest the sidecar returns no report for gets `{"status": "error", "rows": []}` per suite. A whole-pass failure (no report at all) exits non-zero.
  - Torch-free at import.
- **Box side** (new `cascade/validator/lane_d_bench_host.py`): `SSHBenchHost(hosts_path, *, grants, contexts, timeout_s, run=None)` is the `sweep` callable.
  - It resolves the first `final`/`any` `[[host]]` of `hosts_path` at every call, as `eval_offload.make_eval_host_fn` does, so a `bench_lease`-published file works unchanged. A missing or empty file raises, which gives `bench_hold`.
  - It runs the pod command over SSH (`trainer.remote.build_ssh_argv`) and validates the reply shape. Anything malformed raises (`bench_hold`, never a guessed row).
  - It sends no weights: grants only (D12, §10.2).
- **Wiring** (`cascade/validator/lane_d_inputs.py` `load_inputs_services`): an optional `bench_hosts` key in the `--lane-d-inputs` TOML. Absent, `bench_host = None` (today's behaviour: every crowning candidate holds). Present, it is `SSHBenchHost(bench_hosts, grants=<the W8c VaultGetGrants>, contexts=…)`. The source sets the effective contexts per settlement, as `set_armed` does for the eval host.
- **PATHS:** `cascade/eval/lane_d_bench_sweep.py` (new), `cascade/validator/lane_d_bench_host.py` (new), `cascade/validator/lane_d_inputs.py`.
- **Depends:** W3b2, W8c, W15, W4. **Runs in:** ci (fake host / stubbed sidecar) + pod (one live paired sweep; owner step).
- **Accept:** `scripts/laned/accept W3b6a`. The tests cover:
  - the request/reply round trip through a fake `run`, and a hostile reply (wrong schema, extra digest, non-mapping) refused;
  - a missing hosts file raising, so settle gives `bench_hold`;
  - the pod side's materialised dir passing `verify_checkpoint_code`, with `context_length` set and the other three files unchanged;
  - a miss without a grant failing the sweep, and a W4 refusal being `error` for that digest only;
  - torch-free import;
  - `load_inputs_services` without `bench_hosts` returning `bench_host=None`.

**W3b6b — publish, bind, re-derive.**

- **Bind** (`cascade/validator/lane_d_bench.py`, `cascade/validator/lane_d.py`): `sweep_record_text(rec) -> str` is the canonical JSON of one sweep record (sorted keys, no whitespace, finite floats only). Every gate result with swept records gains `rows: {"candidate", "king_d", "house"}`, the sha256 of each record's text, inside the signed body. A `bench_hold` from a raising sweep has no `rows`. Schema `lane_d_report_1.json`: an optional `rows` on `gate_result` (additive; the report stays `/1`).
- **Publish** (`cascade/validator/lane_d_wiring.py`): before the report, the step PUTs each bound record's text at `bench_v/lane_d/<boundary>/<sha256>.json` through the step's existing store (the lane D report's bucket and ACL, R2 dual-write unchanged). The key is content-addressed, so a re-sweep after a restart lands at a new key and never overwrites bytes a published report pins (lane A's rule, `validator_bench_rows_key`). A failed PUT is `publish_failed`: the court doesn't move and the tick retries.
- **Re-derive** (`cascade/satellite/lane_d_replay.py`, W7d's independent replay): for every gate result with `rows`, the satellite fetches the three records by key, checks each against its sha256 (`gate_rows_mismatch`), and recomputes the four LCBs with `cascade.eval.expert_lane.paired_geomean_lcb`. That is lane A's eval math, not producer verdict code, so invariant 12 holds. It uses the report's seed rule and compares within 1e-12 (`gate_lcb_mismatch`). A crowned digest whose pass result has no `rows` is refused (`gate_rows_missing`).
- **PATHS:** `cascade/validator/lane_d_bench.py`, `cascade/validator/lane_d.py`, `cascade/validator/schemas/lane_d_report_1.json`, `cascade/validator/lane_d_wiring.py`, `cascade/satellite/lane_d_replay.py`.
- **Depends:** W3b2, W7a, W7d, W8b. **Runs in:** ci.
- **Accept:** `scripts/laned/accept W3b6b`. The tests cover:
  - the bound sha256 equals the published bytes';
  - a report without a gate is byte-identical to today's;
  - a re-sweep landing at a new key;
  - a tampered record, a tampered LCB and a missing `rows` on a crowned pass, each refused by the satellite;
  - unarmed lane D publishing nothing;
  - lane A's `bench_v/round-*` keys untouched.

**Consensus (both halves).**

- **Deterministic:** the same records give the same text, sha256 and LCBs on every machine. The paired-gate seed is already a pure function of the boundary and its block hash.
- **Signed:** the record digests sit inside the lane D report body that the main validator signs.
- **Content-addressed:** each record lives at its own sha256.
- **Satellites re-derive:** they recompute the LCBs from published bytes instead of trusting the four numbers.
- **GPU sweeps aren't bit-reproducible:** satellites never re-sweep. A satellite verifies the published rows, as lane A validators verify `bench_v` rows.
- **Lane A is byte-identical:** no lane A module changes. `bench_v/lane_d/` is a new key space under an existing prefix.
- **Unarmed is byte-identical:** no source, so nothing is swept or published (invariant 1).
- **No new chain reads or writes, no new bucket.**
- **Pod and bench runs stay owner steps:** rent, seed the house into the pod cache, and run the live paired sweep for W3b6a's pod evidence.

**Readings chosen (flagged).**

- **The sidecar runs the release's pinned wrapper at `context_length` = the effective context.** The wrapper edge-pads short histories, unlike the no-pad W5d scorer. That is the same for the candidate, KING_D and the house, so the paired ratio stays fair. A no-pad bench wrapper would be a sidecar change, which is out of scope.
- **BF16 weights are loaded as is.** The pinned wrapper's strict `load_state_dict` copies them into its F32 parameters, as the W5d scorer upcasts at load.
## rev6.12 (owner 2026-11-23; main @ 315d240)

Records two owner decisions, (a) and (b), on top of rev6.11 (#751, merged
as 315d240). Plan text only. Every changed item is marked **(rev6.12)**,
and the reversed rev6.11 text is struck, not deleted.

**Terms used below.** A **fresh** window is an eligible forward window whose
every target point is post-submission, i.e. it passes the e = 0 rule (data
time and first-seen `> b`). An **embargo window** (rev6.11's term; "embargo-only"
in the owner's words) is an eligible forward window that fails the e = 0
rule, so a mixed window (target straddling `b`) counts as an embargo window.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **The 48 h embargo top-up applies to every cadence (owner 2026-11-23, DECIDED: "that's why we use the embargo data, so embargo + data after 6 hours", in reply to the 5-min / 15-min latency question).** rev6.11 (e)'s "sub-hourly e = 0" is **reversed**. For every series, of every `freq`, `e = min(catalog(feed), [lane_d] embargo_hours)`, with `embargo_hours` = 48. The per-feed catalog override can still only shorten it (e.g. 0 for feeds known to be public in real time). A window's target may combine embargoed points (`release(p) = first_seen(p) + e > b`, data time in `(b − e, b]`) with post-submission points (data time `> b`). The separate forward vs embargo sub-score reporting is **kept**. ~~`[lane_d] embargo_window_share_cap` (0.25) … **kept**~~ (the share rule is replaced by (b)) | rev6.11: (e) item row, (b) "Embargo mitigation" bullet, Commit-to-crown bullet, (d) "First verdict" bullet, (e) spec (heading, formula, W5a bullet, arithmetic, table, cadence bullets); rev6.10 (a): effective-embargo bullet, accepted-risk bullet, latency caption, Commit to crown; §1.3 rev6.10 rule block; §1.7 floor maths + forward latency bullets; W5a; W6b; W7a; W7d; W9; T8e; §6 D5; §17.4 D5 |
| (b) | **Fresh data first, with a minimum fresh share (owner 2026-11-23, DECIDED: "we obviously prefer fresh data, it should always be in there, but need to have a trade off").** (1) **Fresh-first draw:** the W5j per-bin draw takes fresh windows first, then fills with embargo windows. (2) **Minimum fresh share:** `[lane_d] embargo_window_share_cap` (0.25) is **replaced** by **`[lane_d] min_fresh_window_share` = 0.5**: at least 50% of an entry's frozen set must be fresh, so embargo windows are at most 50%. (3) **Trade-off rule:** if fresh windows can't make up ≥ 50% of a set that meets the floors (`min_windows`, `min_clusters`), the entry is **not frozen**. It stays `pending` until a later settlement has enough fresh windows. The share is never relaxed to meet the floors. (4) **Reporting:** forward/fresh and embargo sub-scores per entry; the audit (W9) and replay (W7d) re-derive the share | rev6.11: (e) item row, (b) "Embargo mitigation" bullet, (e) "Embargo windows and the share cap" bullet; rev6.10 (a) effective-embargo + accepted-risk bullets; §1.3 rev6.10 rule block; W2e (key rename); W5a; W5j; W7a; W7d; W9; T8e; §14.2 rev6.10 row; §6 D5; §17.4 D5 |

**(a) Spec: embargo for every cadence.**

- **Effective embargo per series** (replaces the rev6.11 (e) formula):

  ```
  e(series) = min(catalog(feed) or [lane_d] embargo_hours,
                  [lane_d] embargo_hours)                 for every freq
  ```

  - There's no `freq` term. W5a, W9 and W7d read `e` only from the
    snapshot's pinned `feeds[feed].embargo_hours` (W6b).
  - The eligibility rule is the rev6.10 rule, unchanged: for **every**
    target point p, `first_seen(p) + e > b` and `data_ts(p) > b − e`, with
    `b = wall(max(T_E, T_K)) + forward_margin` ~~(6 h)~~ **(rev6.14)**
    (default 0, so `b = T`; the 6 h is the separate judging start
    `j = T + eval_delay_hours`, rev6.14 (a)). S0 and backfill use
    `b − e`.
  - **Mixed targets are allowed.** One window's 64 target points may straddle
    `b`: the points with data time in `(b − e, b]` pass by the embargo, the
    points after `b` pass as post-submission data. Both halves are checked
    by the same rule.
  - A mixed window is an embargo window (terms above), not a fresh one.
- **Kept from rev6.11.** The report keeps the **forward/fresh** and
  **embargo** sub-scores separate (the `embargo_window` flag per window,
  W7a), next to rev6.9's forward / backtest split. The rev6.11 share cap is
  replaced by (b)'s minimum fresh share.
- **Accepted risk (owner).** A feed that's public at the source in real
  time can be scraped by a miner, so its embargoed targets may have been
  seen before the commit. rev6.12 widens this exposure from hourly-and-slower
  series to **every** cadence. Mitigations:
  - the per-feed catalog override (`e = 0` for known real-time-public
    feeds; W6b forces `e = 0` for publish-ahead feeds, rec);
  - the minimum fresh share ((b): embargo windows ≤ 50% of a frozen set);
  - the forward/fresh vs embargo sub-scores in the report;
  - B1 calibration: the null FPR is measured with embargo windows at the
    maximum share (50%), and B1 reports the gain an adversary gets from
    perfect scores on the embargo share.
  - **Rec (not applied):** if B1 shows that gain can exceed `d_margin`,
    clamp the embargo-only windows' contribution the way rev6.9 clamps
    backtest windows. At a 50% embargo share this matters more than at
    rev6.11's 25%.

**(b) Spec: fresh-first draw and the minimum fresh share.**

- **Key.** `[lane_d] min_fresh_window_share`, default **0.5** (owner), in
  [0, 1]; 1 = fresh windows only (the rev3 rule). It replaces
  `embargo_window_share_cap`, which no code parses yet (no shipped key, so
  no legacy parse is needed). Loader parse, round-trip and mutation test
  (W2e).
- **Draw (W5j), per cohort, per settlement:**
  1. Per (timeframe, bin) cell, the seeded draw takes **fresh** candidates
     first, then fills the cell's quota from **embargo** candidates, each
     group in the seeded order. Scarce cells are taken whole and surplus is
     re-split as before.
  2. `max_source_share` applies after, no refill (unchanged).
  3. Let F and E be the fresh and embargo windows left. Embargo windows
     above ~~`F × s / (1 − s)`~~ **(rev6.14 erratum)** `F × (1 − s) / s`
     (s = `min_fresh_window_share`; = F at 0.5)
     are dropped deterministically, last-drawn first, no refill. The total
     stays ≤ `max_windows_per_settlement`.
  4. Floors (`min_windows`, `min_clusters`) are checked on the result.
- **Trade-off rule.** If the step-3 set doesn't meet the floors, the entry
  is **not frozen** at this settlement and stays `pending`
  (`fresh_share_short`, a new skip reason next to the floor reasons). It's
  re-tried at each later settlement as fresh windows accrue. **The share is
  never relaxed to meet the floors**, and no embargo window is kept above
  step 3's bound.
- **Reporting and audit.**
  - Per entry: the forward/fresh sub-score and the embargo sub-score, the
    fresh count F, the embargo count E, and the share F / (F + E).
  - Cross-check (W7a): F / (F + E) ≥ `min_fresh_window_share` on every
    frozen set; the config echo carries the key.
  - W9 and W7d re-derive each window's fresh / embargo class from the signed
    clocks and `b`, re-derive the step-3 trim and the share, and refuse a
    frozen set below the share (`fresh_share_short`).

**Per-cadence latency (rev6.12; h = 64; e = 48 for all; 3-hourly aligned
pool_d builds; 3 h settlements).** **(rev6.14) Superseded by the rev6.14
table** (`b = T`, judging from `j = T + 6 h`). ~~`b = T + 6 h`. S1 is the
first settlement ≥ `b`,~~ in [T+6 h, T+9 h). An aligned build lands just before each
settlement (rev6.11 (d)), so S1's snapshot is at `a ≈ S1`. The eligible
target interval is `(b − 48 h, a] = (T − 42 h, a]`, ≥ 48 h long at S1
(≥ 45 h if S1 only sees the previous build). A window is eligible once
that interval holds its whole target span. This is target-span arithmetic,
not measurement; W1 and B2 replace it.

| Cadence | Target span | Disjoint windows per series at S1 (aligned / 3 h-old build) | First settlement with an eligible window | First verdict, subject to floors and the (b) fresh share | Post-submission (e = 0) windows first appear (rev6.11 (e) table) |
|---|---|---|---|---|---|
| 10 s | 0.18 h | 270 / 253 | S1 | **≈ T+6–9 h** | T+6.2 h |
| 1-min | 1.07 h | 45 / 42 | S1 | **≈ T+6–9 h** | T+7.1 h |
| 5-min | 5.33 h | 9 / 8 | S1 | **≈ T+6–9 h** (was ≈ T+11–14 h) | T+11.3 h |
| 10-min | 10.7 h | 4 / 4 | S1 | **≈ T+6–9 h** | T+16.7 h |
| 15-min | 16 h | 3 / 2 | S1 | **≈ T+6–9 h** (was ≈ T+22–25 h) | T+22 h |
| 30-min | 32 h | 1 / 1 | S1 | **≈ T+6–9 h** (was ≈ T+38–41 h) | T+38 h |
| Hourly (as rev6.11) | 64 h | 0 / 0 | snapshot ≥ T+22 h | ≈ T+22–25 h | T+70 h |
| Daily short h 7–30 (rev6.9, as rev6.11) | 7–30 d | 0 | T+5.25 d … T+28.25 d | ≈ +3 h after that | — |

- Without aligned builds, add up to 3 h.
- **"Subject to floors"** means the cohort's draw at S1 holds ≥ 1000
  windows from ≥ 20 sources after `max_source_share` **and the minimum
  fresh share** ((b) step 3). Otherwise the entry stays pending until later
  settlements add fresh windows ((b) trade-off rule).
- **(b) The fresh share sets the pace.** Before `b + span`, every eligible
  window of a cadence is an embargo window (fully embargoed or mixed). At
  `min_fresh_window_share` = 0.5 the kept embargo windows are ≤ F, so the
  1000-window floor needs **F ≥ 500 fresh windows** (and the clusters
  floor). Embargo windows can at most double a set; they can't start one.
- **(rev6.14)** The per-cadence note below (and the table's right-hand
  column) counts fresh windows only after `T+6 h`, the conflated boundary.
  With `b = T` the fresh counts at S1 are rev6.14's (10 s 33, 1-min 5,
  5-min 1 per series at T+6 h). The bullet above (F ≥ 500, with `b + span`
  read at the new `b`) still holds.
- ~~**Per-cadence note (owner).**~~ **(rev6.14: superseded, see the rev6.14 table)** 10 s and 1-min feeds supply fresh windows by
  about T+6–9 h (right-hand column: ≤ 16 and ≤ 2 per series in the T+6…9 h
  snapshot; e.g. 500 fresh windows ≈ 32 10 s series at 16 each). **So in
  practice the first verdict stays at ≈ T+6–9 h when those feeds meet the
  floors**, with 5-min to 30-min (and hourly from T+22 h) supplying the
  embargo half. A cohort without enough 10 s / 1-min fresh windows waits on
  the fresh schedule: 5-min fresh from ≈ T+11–14 h, 15-min from
  ≈ T+22–25 h, hourly from ≈ T+70–73 h. **W1 must count 10 s and 1-min
  sources; none is measured yet.**

**Changed locations (rev6.12).**

- Status line.
- This section.
- rev6.11 section:
  - the (e) item row (sub-hourly e = 0 sentence and the share cap struck);
  - the (b) "Embargo mitigation: resolved by (e)" bullet (two sub-bullets
    and the 0.25 cap struck);
  - the Commit-to-crown bullet;
  - the (d) "First verdict" bullet;
  - the (e) spec: heading, formula, the W5a bullet, "Embargo windows and
    the share cap", per-cadence arithmetic, the table's `e` column, and the
    cadence bullets.
- rev6.10 (a):
  - the effective-embargo bullet (e for all; cap → fresh share);
  - the accepted-risk bullet;
  - the latency caption;
  - Commit to crown.
- §1.3 rev6.10 rule block.
- §1.7: floor maths; forward latency (5-min and hourly bullets).
- §3.0 rows: W2e (key rename), W5a, W5j, W6b, W7a, W7d, W9.
- §4 row: T8e.
- §6 D5, §14.2 rev6.10 row, §17.4 D5.

## rev6.11 (owner 2026-11-23; main @ 25d2b1d)

Records six owner decisions, (a)–(f), on top of rev6.10 (#749, merged as
df7a2fc with the 2000 cap and the confirmation duel still in). Plan text
only. Every changed item is marked **(rev6.11)**.

**Ordering note:** rev6.9 (#745) is still open. rev6.11 again edits none of
its lines (the W2 row, the §14 lane D tuning row, §1.1 Horizon, §1.3 Window
identity, the §1.6 Box row, and the W5g / W6a rows around its W5i insertion).
For that reason **the W5g follow-up is written into the W5e row and this
section, not the W5g row.** Only the status line and the top insertion point
are shared. Whichever merges second keeps both sections, newest on top.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **`[lane_d] max_windows_per_settlement` = 5000 (owner 2026-11-23: "we don't [want] eval to be 2000 series, use 5000 series"), replacing rev6.10's rec 2000.** The loader check is unchanged: a value below `min_windows` (1000) is refused. The per-bin quota (rev6.10 (b)) splits up to 5000 windows. **Flag:** the brief said this matches "lane A's 5000-series eval (DEC-CA-0068 rising 4000 → 5000)". The committed and the deployed `[eval] shared_ladder_windows` are both **4000** (`chain.toml:603`; the live box's chain.toml carries the same; DEC-CA-0068 title "the SAME 4000 series"). No 5000 is in the repo yet, so this plan doesn't claim the two match. If lane A moves to 5000, the two will agree then. T7 must show that 5000 × (entries + KING_D + house) forward passes fit ≤ 50% of a settlement (§1.6) | rev6.10 (b) compute bound; W2e; §14.2 rev6.10 row; §6 D5; §17.4 D5 |
| (b) | **The confirmation duel is DROPPED (owner 2026-11-23, DECIDED).** A crown is decided at the **first verdict**: the paired cluster bootstrap LCB against KING_D with max-t over the cohort (`cohort_maxt_lcbs`, `bootstrap.py:393-418`), on the seeded per-bin draw of up to 5000 windows (W5j), plus the §1.3b bench regression gate (W3b1/W3b2, in W5e) and the A10 low-frequency non-inferiority check (W3b5). Owner rationale: the LCB covers sampling luck, the random draw over many windows covers window-set variance, and the regression gate stops bad kings. See the spec below | §1.3 Verdict (confirmation-duel bullet, D20 bullet, repeated attempts, `d_margin` calibration), Cohorts; §1.5 report and state bullets, deregistration; §1.7 floor maths + latency; rev6.10 (a) item cell, accepted-risk mitigation, latency caption + commit-to-crown, (b) quota + freeze step 6 + compute bound; §1.3 rev6.10 rule block + safety bullet; W5a; W5j; W5e; W3b5; W7a; W7c; W7d; W9; W12a; W17; T8e; T9b; T9d; T14; §6 D20, D46; §11.2 rows; §17.4 D20; Appendix B1 row |
| (c) | **W6rv is NOT adopted: "not needed (owner 2026-11-23)" ("don't need for lane A, it's only for lane D").** Lane A's `publish-pool.yml` stays daily and its reveal logic (`select_reveals`, `keep_newest = 2`) is unchanged. **Guard note (kept):** lane A's publish cadence must not increase unless W6rv ships first. A faster lane A cadence would shorten the ~48 h reveal delay below `embargo_hours`, because `keep_newest = 2` × cadence is the delay | rev6.10 (a) "The value, 48 h" bullet; §1.3 rev6.10 safety bullet; W6rv row; §14.2 rev6.10 row |
| (d) | **`pool_d` snapshots every 3 h (owner 2026-11-23: "update every 3 hours … so we can get 6 hours of unseen data much faster").** It's built on lane D's own CPU builder (§1.7), independent of lane A's daily `pool`. **Rec:** each build finishes before a settlement boundary, so a settlement always sees a snapshot ≤ 3 h old | W6d; W6fs (3-hourly Merkle roots); §1.3 "Evidence accumulates"; §1.7 Build host + latency; rev6.10 (a) latency caption |
| (e) | ~~**The embargo applies only to hourly-and-slower feeds**~~ (owner 2026-11-23: "use 6 hours after submission so data is impossible to be trained, but supplement the hour+ freq with embargo data"). ~~A **sub-hourly** series (pool `freq` finer than 1 h) uses **e = 0**: post-submission data only, the rev3 rule at `b`. An **hourly-or-slower** series uses~~ **(rev6.12)** Reversed for sub-hourly: **every** series uses `e = min(catalog, [lane_d] embargo_hours)` = 48 by default (rev6.12 (a)). The per-feed catalog override is kept, and it can only shorten. ~~New key **`[lane_d] embargo_window_share_cap`** (rec **0.25**): at most this share of a frozen set's windows may be *embargo windows* (ones that pass only because e > 0).~~ **(rev6.12)** Replaced by `[lane_d] min_fresh_window_share` = 0.5 (rev6.12 (b)). This resolves the rev6.11 (b) embargo flag | rev6.10 (a) effective-embargo + accepted-risk bullets; §1.3 rev6.10 rule block; W2e; W5a; W5j; W6b; W7a; W7d; W9; T8e; §14.2 rev6.10 row; §6 D5; §17.4 D5 |
| (f) | **WB4 (lane B) keeps its own confirmation duel, unchanged.** That's a separate lane B decision, not a [DECIDE]. Lane B code doesn't use lane D's confirm stage today (no lane B module references it). So W5e's removal stays lane D only, and WB4 builds its duel in lane B code | WB4 row; this section |

**(b) Spec: crown at the first verdict.**

- **The crown path.** For a cohort frozen at a settlement:
  1. the challenger clears iff `LCB_α ≥ d_margin` (max-t over the cohort)
     and the point gain is strictly positive;
  2. a cleared entry goes straight to ~~A10 (W3b5) and~~ the §1.3b gate **(rev6.18: A10 removed)**. If it
     passes both, it's crowned at that settlement or the next one the gate
     can decide (`bench_hold`, W5g).

  There's no `confirm_pending` state, no second set, and no D20 king-change
  rule. A loser stays never re-judged (§1.3).
- **W5e: remove the stage (chosen), not a `confirmation_duel = false`
  key.** Why this is the cleaner option, from the code on main:
  - the stage is wired through `cascade/validator/lane_d.py` (≈ 40 refs:
    `OPEN_STATUSES`, the `confirm` record, `_confirm_king_ts`, the
    king-change sweep, stage `confirm` in judging), `cascade/audit/lane_d.py`
    (≈ 58), `cascade/satellite/lane_d_replay.py` (≈ 30),
    `lane_d_report.py` (`ENTRY_STATUSES`, `CONFIRMATION_STAGES`,
    `VERDICT_OUTCOMES`) and the schema enums;
  - a key would keep both paths in consensus code, and in **three
    independent implementations** (producer, W9, W7d), each branching on it.
    That doubles the test matrix for a feature the owner dropped;
  - lane D has never been armed, so no mainnet state or report holds a
    confirm stage.

  **Kept for compatibility:** `[lane_d] confirm_on_king_change` stays a
  **legacy key, parsed and ignored**. The shipped `chain.toml:1661` carries
  it, and the plan's rule is that placeholder keys keep parsing (W2). It's
  dropped from the report's config echo. (The §14.2 tuning row isn't edited,
  because #745 touches it. The key just becomes a no-op there.)
- **Follow-up code and tests (per WP).**
  - **W5e:** remove the `confirm` stage.
    - A first clear becomes a crown candidate.
    - Delete `confirm_pending` from `OPEN_STATUSES` and the king-change
      sweep.
    - Drop `lost_confirmation`; a loss is `lost_duel`.
    - Verdict outcomes become `lost` | `crowned`, plus the existing
      `bench_hold` hold.
    - `confirm_on_king_change` becomes parse-only.
  - **W7a:** schema `lane-d-report/1` stages.
    - Remove `confirm_pending`, `confirmed`, `lost_confirmation`, stage
      `confirm` and the `confirm_rule` def from the enums.
    - The cohort `stage` field is fixed to `first`, or dropped.
    - The `confirmation` block is removed.
    - **Rec:** stay on `/1`, because lane D was never armed on mainnet. If
      any testnet report signed with stage `confirm` must stay verifiable,
      bump to `/2` instead, and the satellite refuses `/1` reports carrying
      it.
  - **W7c:** recovery of a `confirm_pending` state in `lane_d_state.json`
    (testnet only). On load, an entry in `confirm_pending` becomes a **crown
    candidate on its first-clear set** if its `confirm.king_digest` still
    equals KING_D. Otherwise it's `pending` / `pending_floor` (first clear
    discarded, the old D20 (a) outcome). Either way it's logged with a typed
    tag (`confirm_state_migrated`), and the prior/posterior digest binding
    covers the migration. **Rec:** the migration runs once, at the first
    settlement after the release.
  - **W7d and W9:** replay and audit expect **no** confirmation stage.
    - A report carrying one is refused (`confirmation_stage_present`).
    - The confirmation rule check comes out of W9's re-derived set.
    - Both re-derive crowning as first clear ∧ ~~A10 ∧~~ gate **(rev6.18)**.
  - **W5g (not edited here; see the ordering note):** the hold rules that
    reference the confirm stage change. A `bench_hold` now covers "a
    first-clear candidate the gate could not decide", not "a cleared
    confirmation". The sweep that lifts holds at stage `confirm`
    (`lane_d.py` docstring `:42-56`) keys on the first clear. The W5g
    reason-enum assertions drop `confirm_pending`.
  - Also:
    - **W3b5:** "confirm candidate" becomes "crown candidate (a verdict that
      cleared at the first stage)".
    - **W17 / §11.2:** the status board drops `confirm_pending` and the
      confirm stage.
    - **W12a:** the LANE_D.md FAQ loses "confirmation latency".
    - **T9b:** superseded.
    - **T14 / B1:** drop the "confirmation-duel FPR".
- **B1 calibration task (needed before S4).** Without the duel, the per-era
  false-crown rate is `1 − (1 − p)^k`, with `p` the **single-test** null FPR
  at (`alpha`, `d_margin`) on a W5j draw, not the confirmed `p_conf ≈ p²`.
  - At k = 32 and the 5% per-era target (OA3), `p ≤ 1 − 0.95^(1/32) ≈ 0.0016`.
  - B1 (`scripts/null_fpr.py`) must measure the single-test FPR, on
    5000-window per-bin draws, and report the `alpha` / `d_margin` pair that
    meets 1%, 5% and 10% per era.
  - Until it runs, **the `d_margin` 0.01 default is unverified for the
    no-duel path.** The rev3 text already noted that p ≈ 1% gives ≈ 27% per
    era at k = 32.
  - The S4 gate (T14) is unchanged in form, but it's measured on the
    single-test FPR.
- **Embargo mitigation: resolved by (e).** With the duel gone, the
  embargo-0 confirmation set no longer guarded against crowns resting on
  scrapable embargoed points. (e) restores that guard:
  - ~~sub-hourly windows are post-submission only (e = 0);~~
  - ~~embargo windows can only come from hourly-and-slower series, and they~~
    ~~make up at most `embargo_window_share_cap` (0.25) of a frozen set.~~
    **(rev6.12)** Embargo windows now come from every cadence (rev6.12 (a))
    and make up at most 50% of a frozen set: fresh-first draw,
    `min_fresh_window_share` = 0.5, else not frozen (rev6.12 (b)). B1
    measures at that 50% share.

  Other mitigations kept: the per-feed override, the forward/backtest
  sub-scores (rev6.9), the §1.3b gate ~~and A10~~ **(rev6.18)**. **B1 also checks the cap.**
  The null FPR is measured with embargo windows at the cap. The share cap
  bounds the windows, not the log ratio: memorised embargo windows push CRPS
  toward the floor, which is the same issue rev6.9 meets for backtest
  windows. So B1 reports the gain an adversary gets from perfect scores on
  the embargo share. **Rec:** if that gain exceeds `d_margin`, clamp the
  embargo sub-score the way rev6.9 clamps backtest windows. Not applied
  here.
- **Commit-to-crown (crown at the first verdict; h = 64; 3-hourly pool_d,
  (d); ~~sub-hourly e = 0, hourly-and-slower e = 48, (e)~~ **(rev6.12)**
  e = 48 for every cadence).** The arithmetic is ~~below~~ **(rev6.12)** in
  the rev6.12 table. The §1.3b gate can add a settlement or more (`bench_hold`, D6).
- **Lane B (WB4) keeps its own confirmation duel** (f). W5e's removal
  doesn't touch it.

**(c) Spec: W6rv not needed.** Lane D's only release paths are lane A's
`pool` reveal (unchanged, ~48 h: `keep_newest = 2`,
`cascade/pool/reveal.py:66-79`, × the daily cron, `publish-pool.yml:45`) and
`pool_d_tgt` after the dispute window. `embargo_hours` = 48 stays valid
while lane A's cadence stays daily.

- **Guard note:** lane A's `publish-pool.yml` cadence must not increase
  unless W6rv (or an equivalent age guard) ships first.
- Two real snapshots published within 48 h (a manual non-sentinel dispatch)
  also shorten the delay. Ops avoid that while lane D is armed. **Rec:** a
  W11 cascade-ops check that alerts when consecutive lane A pool snapshots
  are < 24 h apart.

**(d) Spec: 3-hourly `pool_d`.**

- **Builder and storage.**
  - The builder runs every 3 h on its own host (§1.7 Build host).
  - Each snapshot is pinned with its sha256 (W6d).
  - Retention keeps every snapshot any open entry's eligible set or frozen
    set references, then expires the rest.
  - The W6fs first-seen Merkle root is committed **per 3-hourly period**, not
    per day, so a snapshot never uses records whose root isn't committed yet.
- **First verdict.** It comes at the first settlement ≥ `b` whose latest
  snapshot holds a draw meeting the floors. ~~For sub-hourly feeds at e = 0,
  that's the first settlement after the first 3-hourly snapshot taken after
  T + 6 h that holds a full h = 64 target span. For 10 s data it's ≈ T+6–9 h
  (below).~~ **(rev6.12)** With e = 48 for every cadence, 10 s to 30-min
  data has eligible windows at S1 (≈ T+6–9 h), subject to the floors and the
  share cap (rev6.12 table).

**(e) Spec: embargo ~~only for hourly-and-slower~~ (rev6.12: for every cadence).**

- **Effective embargo per series:**

  ```
  e(series) = 0                                                  if freq < 1 h   [struck, rev6.12]
            = min(catalog(feed) or [lane_d] embargo_hours,
                  [lane_d] embargo_hours)                        otherwise
  ```

  **(rev6.12)** The `freq < 1 h ⇒ 0` branch above is struck: the second
  line applies to every `freq` (rev6.12 (a) formula).

  - ~~`freq` is the series' pool `freq`, the same field rev6.9's bins read.~~
  - The feed's value stays pinned in `feeds[feed].embargo_hours` (W6b).
  - ~~W5a applies the sub-hourly zero from the series' `freq`.~~
  - Audit and replay re-derive ~~both~~ **(rev6.12)** it from the pinned value.
- **Embargo windows and the share cap.**
  - An *embargo window* is an eligible forward window that fails the e = 0
    rule.
  - ~~After the W5j draw and `max_source_share`, embargo windows above
    `embargo_window_share_cap` × the set size are dropped deterministically,
    with no refill (as the source cap).~~ **(rev6.12)** Replaced by the
    fresh-first draw and `min_fresh_window_share` = 0.5 (rev6.12 (b)):
    embargo windows ≤ fresh windows, else trimmed; below the floors at that
    share the entry isn't frozen.
  - The floors are checked after this drop.
- **Per-cadence arithmetic.**
  - Settlements are every 3 h. The 3-hourly snapshot lands ≤ 3 h before
    each settlement (rec (d)).
  - `b = T + 6 h`.
  - A window is eligible once a snapshot holds its full h = 64 target span:
    - e = 0: from `b`, so the snapshot must be at `a ≥ T + 6 h + span`;
    - e = 48: from `b − 48 h = T − 42 h`, so `a ≥ T − 42 h + span`.
  - The verdict comes at the first settlement ≥ max(`b`, a). This is
    target-span arithmetic, not measurement; W1 and B2 replace it.

  **(rev6.12)** Superseded for 10 s to 30-min: those series now use e = 48
  (the rev6.12 table). Their `e` cells are struck to 48, but the other
  cells of the sub-hourly rows stay as the e = 0 timing,
  i.e. when each cadence's **post-submission** windows first appear, which
  still matters for the share cap (rev6.12 flag).

  | Cadence | e | Target span | First eligible snapshot ≥ | First verdict (aligned builds) | Windows per series in the T+6…9 h snapshot |
  |---|---|---|---|---|---|
  | 10 s | ~~0~~ 48 | 0.18 h | T+6.2 h | **≈ T+6–9 h** (S1 or S2) | 16 |
  | 1-min | ~~0~~ 48 | 1.07 h | T+7.1 h | ≈ T+7–10 h | 2 |
  | 5-min | ~~0~~ 48 | 5.33 h | T+11.3 h | ≈ T+11–14 h | 0 (1 by T+12 h) |
  | 15-min | ~~0~~ 48 | 16 h | T+22 h | ≈ T+22–25 h | 0 |
  | 30-min | ~~0~~ 48 | 32 h | T+38 h | ≈ T+38–41 h | 0 |
  | Hourly | 48 | 64 h | T+22 h | ≈ T+22–25 h (vs ≈ T+70–73 h at e = 0) | 0 |
  | Daily short h 7–30 (rev6.9) | 48 | 7–30 d | T+5.25 d … T+28.25 d | ≈ +3 h after that | 0 |

  - Without aligned builds, add up to 3 h.
  - ~~**"About T+6–9 h" holds for 10 s data, and nearly for 1-min.** 5-min
    and 15-min need their own target span after T+6 h, as the e = 0 rule
    requires.~~ **(rev6.12)** With e = 48 for all, every cadence up to
    30-min has eligible windows at S1 (rev6.12 table).
  - The first verdict at ≈ T+9 h needs the 10 s and 1-min cells to hold
    ≥ 1000 windows from ≥ 20 sources. Otherwise the entry stays pending
    until slower cells add windows. **W1 must count 10 s and 1-min sources;
    none is measured yet.**
  - ~~Hourly is the only h = 64 cadence the embargo speeds up, by about two
    days.~~ **(rev6.12)** The embargo now speeds up every cadence; hourly's
    gain (about two days) is unchanged.

**Changed locations (rev6.11).**

- Status line.
- This section.
- rev6.10 section:
  - (a) item cell;
  - the accepted-risk bullet;
  - the latency caption;
  - "Commit to crown";
  - the (b) Quota bullet;
  - freeze step 6;
  - the compute bound (5000).
- §1.3:
  - the rev6.10 rule block;
  - the rev6.10 safety bullet;
  - Cohorts step 2;
  - the Verdict bullets: confirmation duel, D20, repeated attempts and
    `d_margin` calibration.
- §1.5:
  - the report bullet (`confirmation` block);
  - the state bullets (`confirm`, `confirm_pending`);
  - deregistration.
- §1.7:
  - floor maths;
  - the latency lead-in;
  - the "With the mandatory confirmation duel" bullet.
- §3.0 rows:
  - W2e (5000);
  - W5a;
  - W5j;
  - W5e;
  - W3b5;
  - W7a;
  - W7c;
  - W7d;
  - W9;
  - W12a;
  - W17.
- §4 rows: T8e, T9b, T9d, T14.
- §6 D5, D20 and D46.
- §11.2 rows: `docs/LANE_D.md`, `docs/VALIDATOR.md` and status / website.
- §14.2 rev6.10 row (5000).
- §17.4 D5 and D20.
- Appendix B1 measurement row.
- **(c)–(f):**
  - rev6.10 (a): the "The value, 48 h" bullet, effective embargo, accepted
    risk, the latency caption and assumptions;
  - §1.3: the rev6.10 rule block, the rev6.10 safety bullet and "Evidence
    accumulates";
  - §1.7: floor maths, Build host, the latency bullets;
  - W2e, W5a, W5j, W6fs, W6b, W6d, W6rv, W7a, W7d, W9;
  - T8e;
  - WB4;
  - D46;
  - §14.2 rev6.10 row;
  - §6 and §17.4 D5.

## rev6.10 (owner 2026-11-23; main @ d3e3001)

Records two owner decisions, (a) ("I want to delay eval by 6 hours and use
series after this time plus embargo data"; context: "eval we release is days
old and consider this embargo"; value: "Eval is released 48 [h] later in lane
A, so maybe the same") and (b) ("Will also use randomly sampled eval from each
bin"). Plan text only. Every changed item is marked **(rev6.10)**.
**Ordering note:** rev6.8 (#741) is merged and is the newest rev on main, so
rev6.10 sits directly above it. rev6.9 (#745, slow-cadence bins) is still
open. It touches the W2 row, the §14 lane D tuning row, §1.1 Horizon, §1.3
Window identity and §1.6 Box row, and rev6.10 edits **none of those lines**:
its config and overlay additions are new rows (W2e; a new §14 row). The only
shared lines are this status line and the top-of-file insertion point.
Whichever merges second keeps both sections, newest on top, with the status
line naming the higher rev. Where (b) refers to "the rev6.9 bins", it means
#745's table. If #745 merges first, the rev6.9 per-window seed
`sha256(block_hash ‖ series_id ‖ "lane_d_backtest")` in its spec and W5i row
is superseded by (b)'s draw seed; if it merges second, it must strike that
text on its rebase.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Embargo-aware forward eligibility with a 6 h eval delay (owner 2026-11-23, DECIDED).** Each point gets a release time to miners, `release(p) = first_seen(p) + embargo_hours(feed)`. `[lane_d] embargo_hours` = **48** (decided: it matches lane A's ~48 h eval release delay, see the spec). A per-feed override `embargo_hours` in the TSBench-Forge catalog (`sources.yaml`) can only shorten it, e.g. 0 for feeds known to be public in real time. `forward_margin_hours` stays **6** (the owner's "delay eval by 6 hours"). ~~The confirmation duel is **unchanged**: its second set satisfies the embargo-0 rule~~ **(rev6.11)** The confirmation duel is dropped (rev6.11 (b)) | §1.3 rule (new block under the rev3 quote), "Why it's built this way" (new bullet), "Evidence accumulates" bullets, confirmation-duel bullet; §1.5 report bullet; §1.7 floor maths + forward latency; W2e (new row after W2b); W5a; W6fs; W6b; W6d; W6rv (new row after W6d); W7a; W7d; W9; T8; T8e (new); v3 self-check "T-rows → WPs" table (T8e, T9d rows); §6 D5; §14 (new row after the sentinel row); §17.4 D5 |
| (b) | **Seeded random draw from every horizon bin (owner 2026-11-23, DECIDED).** Each settlement's lane D windows are a seeded draw, equal per (timeframe, bin) cell, from the boundary block hash via lane A's blake2b seed map. King and cohort get identical draws. The frozen set is the draw at the freeze settlement. Windows per settlement are capped by `[lane_d] max_windows_per_settlement` | §1.3 "Evidence accumulates" + Cohorts; §1.5 report bullet; W2e; W5b; **W5j** (new row after W5b); W7a; W7d; W9; T9d (new); v3 self-check "T-rows → WPs" table; §14 new row |

**(a) Spec: embargo-aware forward eligibility.**

- **The rule.** Let `b = wall(max(T_E, T_K)) + forward_margin` and, per
  feed, `e = embargo_hours(feed)` (effective value below). A forward window
  is eligible only if, for **every** target point p:
  1. **first-seen clock:** `release(p) = first_seen(p) + e > b`, and
  2. **data clock:** `data_ts(p) > b − e`.

  Everything else in §1.3 stays as it is. The feed's mirror lag must be
  ≤ `max_mirror_lag_hours`. A forecast-product feed is never eligible. A
  series without both timestamps, `frontier` or a `feeds` entry is never
  eligible. S0 and the backfill rule (`lane_d_forward.py` rules 5 and 6) use
  the **feed bound** `b − e` in place of `b`.
- **How it generalises the rev3 both-clocks rule.** Both conditions are the
  rev3 rule with the bound moved from `b` to `b − e(feed)`: first-seen
  `> b − e` and data time `> b − e`. So **e = 0 ⇒ identical to today's rule**:
  same bound, same refusal reasons, same precedence. `forward_bound()` in
  `cascade/validator/lane_d_forward.py` gains a per-feed subtraction and
  nothing else. The refusal labels keep their meaning against the feed bound:
  `publish_ahead` = the first-seen clock fails (`release(p) ≤ b`),
  `late_mirror` = the data clock fails, `before_margin` = both fail.
- **Why it's safe at e > 0.** The embargo asserts that we don't release a
  point to miners before `first_seen + e`. A point with `release(p) > b`
  wasn't released by us before the entry's (or king's) commit plus the
  margin, even though its data time is up to `e` hours before the commit.
  The data clock `> b − e` keeps backfilled old values out: a value whose
  data time is older than the embargo window is refused even if our mirror
  saw it late.
- **Effective embargo per feed.** `e(feed) = min(catalog, [lane_d]
  embargo_hours)`, where `catalog` is the feed's `embargo_hours` in
  `sources.yaml`, or `[lane_d] embargo_hours` when the feed has no such
  field. The catalog can only **shorten** the embargo. It lives outside
  consensus (the forge repo), so it must never widen what a validator
  admits. The pool_d builder writes the effective `e` into the snapshot's
  `feeds[feed].embargo_hours` (W6b), so the value a verdict used is pinned
  with the snapshot sha256 (W6d) and read back by audit and replay, never
  re-read from the catalog. **(rev6.11)** ~~**Sub-hourly series use e = 0**
  whatever the feed value. The embargo only supplements hourly-and-slower
  series, and~~ **(rev6.12)** Every cadence uses this `e` (rev6.12 (a)), and
  embargo windows are capped at ~~`embargo_window_share_cap`
  (0.25) of a frozen set (rev6.11 (e))~~ **(rev6.12)** 50% of a frozen set:
  at least `min_fresh_window_share` (0.5) must be fresh (rev6.12 (b)).
- **`max_mirror_lag_hours` is kept** (6, A1), unchanged. **The publish-ahead
  protection is kept:** the first-seen clock still refuses a future-stamped
  value that our mirror saw before `b − e` (`publish_ahead`), and
  forecast-product feeds stay refused outright (`forecast_product`) at any
  `e`. **Rec:** W6b forces `e = 0` for any feed whose W6fs record shows data
  timestamps ahead of first-seen (`max(data_ts − first_seen) > 0` on any
  point). That's a publish-ahead feed whose values are public at the source
  before their data time, so an embargo on our release can't protect them.
- **The value, 48 h, and where lane A's 48 h comes from.** Lane A's delay is
  **not a config value.** It comes out of two things:
  - the reveal guard withholds the newest `keep_newest = 2` real snapshots
    (`cascade/pool/reveal.py:17-24`, `select_reveals` at `:66-79`, CLI
    default `cascade/pool/cli.py:384-388`; the workflow passes no override);
  - the daily publish cron is `0 3 * * *` (`.github/workflows/publish-pool.yml:45`;
    the comments at `:28` and `:251` say "~48h lag").

  So a point first seen just before a 03:00 build is revealed about 48 h
  later, and one first seen just after it about 72 h later. That's a
  steady-state figure. The guard reveals later than necessary under
  publisher gaps, never earlier, **but** two real (non-sentinel) snapshots
  published within 48 h, e.g. a manual `workflow_dispatch`, shorten it.
  There is no single value lane D could read. ~~**Rec (W6rv, needs an owner
  yes, because it changes the lane A reveal job's timing; pool bytes and
  lane A windows don't change):** make the delay explicit and enforced on the
  reveal side. `select_reveals` also withholds every snapshot built less
  than `[lane_d] embargo_hours` ago, and so does every reveal prefix that
  can carry forge points (`--lane`, e.g. pool_mkt). Both sides then read one
  value from chain.toml, and `embargo_hours ≤ actual reveal delay` holds by
  construction.~~ **(rev6.11)** W6rv **not needed** (owner 2026-11-23).
  Lane A's reveal is unchanged. Guard note: lane A's cadence must not
  increase without W6rv (rev6.11 (c)). Lane D's own `pool_d_tgt` is published only after the
  dispute window (§1.7 Reveal), well after 48 h.
- ~~**`forward_margin_hours` = 6** (the owner's "delay eval by 6 hours"). It's
  already the default (`cascade/shared/config/lane_d.py:59`, parsed at
  `:244`; D5), and no lane D text gives another value, so no margin text
  changes.~~ **(rev6.14)** The owner's "delay eval by 6 hours" is the new
  key `eval_delay_hours` = 6; `forward_margin_hours` defaults to **0**
  (rev6.14 (a)). rev6.10 adds the **judging-start rule**: no window set is frozen or
  judged for an entry at a settlement before ~~`b`~~ **(rev6.14)**
  `j = wall(max(T_E, T_K)) + eval_delay_hours`. Its first judgement uses
  the draw (b) over windows that satisfy the rule at the first settlement
  ≥ ~~`b`~~ **(rev6.14)** `j`.
- **Floors are unchanged:** `min_windows` 1000, `min_clusters` 20,
  `max_source_share` 0.25.
- **Accepted risk (owner).** **(rev6.11)** ~~Narrowed by rev6.11 (e) to
  hourly-and-slower series,~~ **(rev6.12)** Applies to every cadence
  (rev6.12 (a)), with embargo windows ≤ ~~25%~~ **(rev6.12)** 50% of a
  frozen set (rev6.12 (b)). A
  feed that is public at the source in real time
  can be scraped by a miner in real time. Its embargoed target points (data
  time in `(b − e, b]`) may then have been seen before the commit. Kept
  mitigations:
  - the per-feed override (`e = 0` for known real-time-public feeds);
  - forward and backtest sub-scores (rev6.9, #745) stay separate;
  - ~~the **confirmation duel is unchanged and forward-only.** Its second,
    disjoint set must satisfy the **embargo-0 rule** (`e = 0` for every feed,
    i.e. the rev3 rule at bound `b`), so no crown rests on embargoed points
    alone.~~ **(rev6.11)** Dropped with the duel. This mitigation no
    longer exists (see the rev6.11 (b) flag).

  This stays as decided unless the owner decides otherwise.
- **Backtest windows (rev6.9) are not affected.** They're knowingly
  contaminable hold-outs, so they carry no forward eligibility rule. A
  daily-short forward window (rev6.9) obeys this rule.
- **Dependencies.**
  - W6fs (first-seen recorder) is **required** to compute `release(p)`.
  - The audit (W9) and satellite replay (W7d) re-derive eligibility from the
    signed first-seen data (Merkle-rooted, B12) plus the pinned `e` values.
  - W5a changes (the rule).
  - The config keys land in **W2e** (a new row, so W2's row text stays
    untouched while #745 edits it).
  - The pool_d builder reads the catalog field. W6a selects series and
    doesn't need it; its row isn't edited, because #745 inserts W5i directly
    above it. The read lands in W6b, which already writes `feeds`, and W6d
    pins it.
- **Catalog contract for TSBench-Forge (`sources.yaml`).** Optional per-feed
  key `embargo_hours`: a non-negative number, or absent. Absent means
  `[lane_d] embargo_hours`. The builder clamps it to the global value and
  refuses a negative or non-numeric value (the feed is then unlisted in
  `feeds`, so `unlisted_feed`, fail closed). The forge repo documents it next
  to `frequency` / `domain`. A change takes effect at the next pool_d build
  and never retroactively: frozen sets keep the `e` pinned in their
  snapshot.

**Per-cadence latency (h = 64; e = 48, the decided value, and e = 0, an
overridden feed ~~or the confirmation duel's set~~).** Assumptions: T is the
commit wall time, ~~`b = T + 6 h`~~ **(rev6.14: `b = T`, `j = T + 6 h`;
current table in rev6.14 (a))**, settlements every 3 h (S1 is the first
settlement ≥ ~~`b`~~ **(rev6.14)** `j`, in [T+6 h, T+9 h)), ~~daily pool_d snapshots (a settlement
sees one 0–24 h old)~~ **(rev6.11)** superseded: pool_d is now 3-hourly and
~~sub-hourly series use e = 0. The current table is in rev6.11 (e).~~
**(rev6.12)** e = 48 for every cadence; the current table is in rev6.12 (a). A window becomes eligible once a snapshot holds its
whole target span. The eligible target interval at a snapshot built at time
`a` is `(b − e, a]`. These numbers are target-span arithmetic only, not
measurements; W1 and B2 replace them.

| Cadence | Target span | e = 48: disjoint windows per series at S1 (snapshot 0 h / 24 h old) | e = 48: first settlement with an eligible window | e = 0: first settlement with an eligible window |
|---|---|---|---|---|
| 1-min | 1.07 h | 45 / 22 | S1 | T+9 h … T+34 h |
| 5-min | 5.33 h | 9 / 4 | S1 | T+12 h … T+38 h |
| 10-min | 10.7 h | 4 / 2 | S1 | T+18 h … T+44 h |
| 15-min | 16 h | 3 / 1 | S1 | T+24 h … T+49 h |
| 30-min | 32 h | 1 / 0 | S1 … T+17 h | T+39 h … T+65 h |
| Hourly | 64 h | 0 / 0 | T+24 h … T+49 h (≈ 1–2 d) | T+72 h … T+97 h (≈ 3–4 d) |
| Daily short, h 7–30 (rev6.9, if armed) | 7–30 d | 0 | ≈ T+5.3 d (h 7) … T+29.4 d (h 30) | ≈ T+7.3 d (h 7) … T+31.4 d (h 30) |

- **Floors at e = 48.** Fast cadences (1–15 min) have eligible windows at
  S1. 30-min has them at S1 only with a fresh snapshot. So the entry
  **meets the floors at S1 if** those cells hold ≥ 1000 windows from ≥ 20
  sources after the (b) draw and `max_source_share`. For example, 1000
  windows is about 112 5-min series at 9 per series, or 250 at 4 per series.
  That's a conditional, not a measured claim: W1's per-frequency counts
  decide it. Hourly first contributes at ≈ 1–2 days.
- **Floors at e = 0** (today's rule): no cadence has a window at S1. The
  numbers match §1.7's earlier ≈ 1–1.5 days (5-min) and ≈ 4 days (hourly).
- **Commit to crown.** ~~The first clear can now come at S1 (e = 48), and the
  confirmation duel's set follows the e = 0 column. So commit-to-crown is
  about one embargo-0 freeze period instead of two: ≈ 1.5 days for 5-min-heavy
  sets and ≈ 4 days for hourly-heavy sets, down from §1.7's 2–3 and ≈ 8 days.~~
  **(rev6.11)** The crown comes at the first verdict. See the ~~rev6.11 (e)
  table (3-hourly pool_d, sub-hourly e = 0)~~ **(rev6.12)** rev6.12 table
  (3-hourly pool_d, e = 48 for every cadence).

**(b) Spec: seeded random draw from every horizon bin.**

- **What gets drawn.** Each settlement's lane D windows are a seeded random
  draw from every horizon bin. A **cell** is one (timeframe, bin) pair:
  - every sub-daily pool `freq` is a timeframe with one forward bin at
    **h = 64**;
  - the rev6.9 slow cadences (daily, weekly, monthly, quarterly, yearly)
    have the short / medium / long bins of #745's table. Their backtest cells
    exist only while `backtest_weight_cap > 0` (~~rev6.9: off by default;~~
    **(rev6.15)** on by default, cap 0.05; at 0
    daily stays one forward bin at h = 64).

  Within a cell, **both the window origin (target start) and the exact h**
  are drawn: h uniform within the bin (h = 64 for fast cells), and the
  origin uniform among the cell's candidates whose h-point target satisfies
  the window's rule (the (a) rule for forward windows, the rev6.9 hold-out
  for backtest ones).
- **Rec: no overlap.** Two windows of one series in one draw never overlap
  in target points.
- **Seed.** `_seed_to_int` (`cascade/eval/bootstrap.py:29`, the blake2b seed
  map that lane A's windows import at `cascade/validator/windows.py:38-40`)
  over the boundary block hash. **Rec:** `_seed_to_int(f"{boundary_block_hash}:lane_d_windows")`,
  one `np.random.default_rng` stream consumed in a fixed order: cells sorted
  by (timeframe, bin), candidates sorted by (series_id, origin). That's the
  lane A pattern at `windows.py:405`. This supersedes rev6.9's per-window
  sha256 seed (ordering note).
- **Quota.** Let N = `max_windows_per_settlement` ~~minus the windows already
  drawn this settlement for confirmation duels~~ **(rev6.11)** (5000; there are
  no confirmation draws). N is split **equally per
  cell, subject to availability.** A cell whose capacity is below its share
  is taken whole, and the surplus is re-split equally over the larger cells,
  smallest first. That's the rule of lane A's even-by-domain pick
  (`_pick_series`, `windows.py:461-493`; DEC-CA-0032; DEC-CA-0068), without
  the jittered split. **Rec:** a cell's capacity is the number of
  non-overlapping eligible windows at the bin's lower h. `max_source_share`
  then applies to the drawn set (W5b, T18e): the excess is dropped
  deterministically with no refill (rec), so the draw stays auditable in one
  pass.
- **Fairness.** The king and every challenger in a cohort get the identical
  draw: it's made once over the cohort's intersection eligible set (§1.3
  Cohorts), never per entry. Every drawn window is recorded in the signed
  report: window id, `timeframe`, `bin`, `h`, origin, `kind`. Audit (W9) and
  satellite replay (W7d) re-derive the draw from the block hash and the
  pinned candidate set.
- **Freeze and floors (explicit).**
  1. At each settlement ≥ `b`, the draw is made over each pending entry's
     (its cohort's) eligible set, i.e. the union over pool_d snapshots
     (§1.3).
  2. The floors (`min_windows`, `min_clusters`, forward windows only) are
     checked **on the draw**, after `max_source_share`.
  3. If the draw meets them, **the frozen set is the draw at that freeze
     settlement**. The entry is judged once on it and it's never re-drawn.
  4. If it doesn't, the entry stays `pending`. Nothing is forecast for it
     (counting needs no forecasts), and the next settlement draws afresh
     with that boundary's seed over the grown eligible set.
  5. Latest-T deferral (§1.3 Cohorts step 3) re-draws over the reduced
     cohort's intersection with the same seed.
  6. ~~A confirmation duel's set is the draw at its own freeze settlement,
     over the **embargo-0** eligible set, excluding every window whose target
     points overlap the first set (disjoint by target points, not only by
     id).~~ **(rev6.11)** There is no confirmation set: the crown is decided
     on the frozen draw (rev6.11 (b)).
- **Compute bound.** New key `[lane_d] max_windows_per_settlement`
  (~~**rec 2000**~~ **(rev6.11)** **5000**, owner 2026-11-23): the total windows forecast per settlement across the
  judging cohort ~~and the confirmation duels, with confirmations drawn first~~.
  The loader refuses a value below `min_windows`. A cohort whose remaining
  budget can't meet the floors is deferred with the typed reason
  `window_budget`. T7 checks that the cap × (entries + KING_D + house)
  forward passes fit ≤ 50% of a settlement on the eval host (§1.6).

**Changed locations (rev6.10).**

- Status line.
- This section.
- §1.3:
  - a new **(rev6.10)** rule block under the rev3 quote;
  - a new "Why it's built this way" bullet;
  - "Evidence accumulates" bullets 1 and 2 (struck and replaced);
  - Cohorts step 2;
  - the confirmation-duel bullet (unchanged-rule note).
- §1.5: the report bullet "per window, both clocks…".
- §1.7: "Floor maths at h = 64" (struck and replaced) and the "Forward
  latency" lead-in.
- §3.0 rows:
  - **W2e** (new, after W2b);
  - W5a;
  - W5b;
  - **W5j** (new, after W5b);
  - W6fs;
  - W6b;
  - W6d;
  - **W6rv** (new, after W6d; rec, needs an owner yes);
  - W7a;
  - W7d;
  - W9.
- v3 self-check "T-rows → WPs" table: rows T8e, T9d.
- §4: the T8 row; **T8e** and **T9d** (new, after T9b).
- §6 D5.
- §14.2 table: a new row after the cross-lane sentinel row.
- §17.4 D5.

## rev6.9 (owner 2026-11-23; main @ 60abbef)

Records one owner decision, (a) ("We accept its gameable for longer series
but we know there is not many in eval" + "Do ur recs there"). Plan text only.
Every changed item is marked **(rev6.9)**. **Ordering note:** ~~rev6.8 (#741,
WB25 multi-GPU lane B expert legs) was still open when this was written, so
rev6.9 sits above rev6.7, the newest rev on main. The two touch no common
table row; whichever merges second keeps both sections, rev6.9 on top.~~
**(on merge with main @ df7a2fc)** rev6.8 (#741) and rev6.10 merged first,
so rev6.9 sits between them in rev order. Per rev6.10's ordering note,
rev6.9's per-window sha256 draw seed is struck in favour of rev6.10 (b) / W5j
(the Draws bullet and the W5i row), and W5i now also depends on W5j.

| Item | Decision | Landed in |
|---|---|---|
| (a) | ~~**Slow-cadence backtest windows, owner horizon bins, 5% weight cap (owner 2026-11-23, DECIDED).**~~ **Superseded (owner 2026-11-24, DECIDED) by rev6.18 D24: backtest windows dropped, W5i retired; the struck spec below is kept for history.** See the spec below | §1.1 Horizon; §1.3 Window identity; §1.6 Box row; W2 row (keys); **W5i** row (new, after W5g); §14 lane D tuning row |

**(a) Spec.** **(rev6.18 D24, owner 2026-11-24): the backtest parts of this
spec ("Everything else is a backtest window", the weight cap, the
combination rule and its loader check, the backtest draw and sub-scores) are
superseded: no backtest window is built or scored, and
`backtest_weight_cap` / `backtest_ratio_floor` are deleted (F4).**

- **Horizon bins (points).** The forward pass appends `ceil(h/32)` masked
  patches and scores only the first h points (`lane_d_scorer.py:142, 156`).
  Every bin is ≤ `MAX_HORIZON` = 512 (`lane_d_scorer.py:52`), so the §1.1
  hard limit holds.

  | Cadence (pool `freq`) | Short | Medium | Long |
  |---|---|---|---|
  | Daily (`D`) | 7–30 | 31–90 | 91–180 |
  | Weekly (`W`) | 4–13 | 14–26 | 27–52 |
  | Monthly (`M`, `ME`, `MS`) | 3–12 | 13–24 | 25–36 |
  | Quarterly (`Q`, `QE`, `QS`) | 2–4 | 5–8 | 9–12 |
  | Yearly (`A`, `Y`, `YE`) | 1–2 | 3–4 | 5–6 |

  The freq strings are today's `LANE_D_LOWFREQ_FREQS`
  (`cascade/shared/config/lane_d.py:38`), exact match. Faster cadences keep
  forward-only h = `[lane_d] horizon` (64) unchanged.
- **Daily short stays forward-only.** The §1.3 both-clocks rule is
  unchanged. A daily window whose draw lands in the short bin is a
  **forward** window with that drawn h ∈ [7, 30]. It must pass W5a on all h
  target points, and it counts toward the floors like any forward window.
- ~~**Everything else is a backtest window.**~~ **(rev6.18 D24) superseded.** That's daily medium and long,
  plus every weekly, monthly, quarterly and yearly bin. A backtest window is
  a hold-out of the last h points of the series in the boundary's pinned
  `pool_d` snapshot, with the history before it as context (truncated to the
  declared context, no padding, §1.1). It's on already-published history,
  so it's **knowingly contaminable**: weights are open and miners can train
  on these targets. The owner accepts this because few such series are in
  eval. **Rec:** a series with fewer than 32 points (one patch) of history
  before the hold-out is skipped with the typed reason
  `backtest_short_history`.
- ~~**Weight cap.**~~ **(rev6.18 D24) superseded.** Together, backtest windows carry at most
  `[lane_d] backtest_weight_cap` = 0.05 of the lane D score; forward-only
  windows carry ≥ 0.95. **Rec:** the effective weight is
  `c = min(backtest_weight_cap, n_bt / (n_bt + n_fwd))`, so a handful of
  backtest windows is never up-weighted past its natural share.
- **The memorisation bound needs a formula (flagged).** The verdict
  statistic is a geomean ratio (`_geo` at `cascade/validator/lane_d.py:321`;
  gain = 1 − geo_challenger / geo_bar; CRPS floored at 1e-12). Perfect
  memorisation drives backtest CRPS to the floor. The log ratio is then
  unbounded, so a weight cap alone does **not** bound the gain. For example,
  at a typical CRPS of 0.1 a 5% weight still moves the gain by about 0.47.
  **Rec, the combination rule:** score the forward and backtest sub-scores
  separately (each a geomean over its own windows, as today). Then combine
  them log-linearly with the backtest ratio clamped:
  `log R = (1 − c)·log R_fwd + c·log clamp(R_bt, r_floor, 1/r_floor)`,
  gain = 1 − R. `r_floor` is the new key `[lane_d] backtest_ratio_floor`
  (**rec 0.85**). Perfect memorisation (with forward equal) gains at most
  `1 − r_floor^c`.
  - ~~**Loader check (W2/W5i):**~~ **(rev6.18 D24) deleted in F4.** Whenever `backtest_weight_cap > 0`, refuse
    the config (and any overlay) unless
    **`1 − backtest_ratio_floor ** backtest_weight_cap < d_margin`**,
    equivalently `backtest_weight_cap · ln(1 / backtest_ratio_floor) <
    −ln(1 − d_margin)`.
  - At the defaults, cap 0.05 and floor 0.85: 1 − 0.85^0.05 ≈ 0.0081 <
    `d_margin` 0.01. The floor bound is `r_floor > (1 − d_margin)^(1/c)` ≈
    0.818.
  - If B1 lowers `d_margin`, the floor must rise with it. The check runs at
    every overlay adoption, so an overlay that breaks it is refused.
  - The clamp is applied **inside every bootstrap replicate**, so the bound
    holds for the LCB as well as the point gain.
  - (Under a plain linear blend, gain_bt ≤ 1 would need `cap < d_margin`,
    which fails at 0.05 vs 0.01. That's why this is a formula and not a
    bare cap.)
- **Draws.** Per window, the bin and the exact h are drawn deterministically
  from the boundary block hash. ~~The seed is
  `sha256(block_hash ‖ series_id ‖ "lane_d_backtest")`; bin uniform over the
  cadence's three bins, h uniform within the bin.~~ **(rev6.9, on merge with
  rev6.10)** Superseded by rev6.10 (b) / W5j: one
  `_seed_to_int(f"{boundary_block_hash}:lane_d_windows")` stream, an equal
  quota per (timeframe, bin) cell, and the origin and h drawn within the
  cell. W5i supplies the backtest cells and the hold-out rule. King and every challenger
  in a cohort get the same draws. The draws are recorded per window in the
  signed report (`kind` forward|backtest, `freq_class`, `bin`, `h`), and
  audit (W9) and satellite replay (W7d) re-derive them. **Rec:** the
  confirmation duel re-draws at its own boundary. Disjointness (§1.3)
  applies to forward windows only, because the backtest series set is small
  and repeats.
- **Floors and freeze.** `min_windows`, `min_clusters` and
  `max_source_share` count **forward windows only** (rec). Backtest windows
  never freeze, never union across snapshots, and never defer a cohort.
- **Sub-scores.** Each entry and each bar gets a separately reported
  `forward` and `backtest` sub-score (crps, mase, geomean, n), plus the
  combined score, so memorisation is visible. **W11 follow-up:** a
  cascade-ops alert fires when an entry's backtest relative gain exceeds
  its forward relative gain by more than an ops-side threshold (non-consensus;
  rec 0.15).
- ~~**Off by default.** `backtest_weight_cap = 0` until the owner arms it by
  overlay/config.~~ ~~**(rev6.15, owner 2026-11-23) On at launch:**~~ **(rev6.18 D24, owner 2026-11-24) superseded: dropped, key deleted.**
  `backtest_weight_cap` defaults to **0.05**. Setting it to 0 by overlay or
  config turns it off: at 0 no backtest window is built or scored, the
  per-window daily h draw is not applied (daily stays h = 64 forward), and
  testnet isn't blocked.
- **A10 is unchanged.** The low-frequency non-inferiority check
  (`lane_d_lowfreq.py`, W3b5) stays a guard at `[lane_d] horizon` and never
  grants gain. Moving it onto the bins is a later call, not decided here.

## rev6.8 (owner 2026-11-23; main @ cbe4c79)

Records one owner request, (a) ("Can make lane b multi gpu"). Plan text
only; nothing else in the design changes. The changed item is marked
**(rev6.8)**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **New WP WB25, lane B expert legs on every GPU of the rented layout (owner 2026-11-23).** Gap (verified on main): `train_pyramid` (`cascade/trainer/pyramid_trainer.py`) sets `world = gang_size(trainer.device) if slices is None else 1`, so an expert-slice leg trains on one GPU even on an 8× pod. A WB19 run on 8×H200 used GPU0 only (0.055 steps/s, the same as an RTX 4090). Lane A legs already use the data-parallel `Gang` of `cascade/trainer/pyramid_dp.py`: micro-batches dealt round-robin, sums all-reduced, rank 0 owns the corpus and the optimizer, "the only difference from one GPU is the order of floating-point sums". WB25 runs lane B legs on that same Gang, extended for `ExpertSlices`. The optimizer sharding and the gradient all-reduce cover only the trainable (selected) expert rows. The router and the routing bias stay frozen (WB16, D40 option a). The leg equals the 1-GPU leg up to floating-point summation order. The leg record keeps the GPU count (`gpu_n`), and WB22's audit re-derivation is pinned to the recorded `gpu_n`. `CASCADE_TRAIN_GPUS` may lower it. Lane A legs stay byte-for-byte unchanged. No dependency, `[training]` or contract-digest change | Lane B table, WB25 row (after WB24) |

## rev6.7 (owner 2026-11-23; main @ d5878dd)

Records one owner decision, (a) ("Yes w8c"). Plan text only; nothing else in
the design changes. The changed item is marked **(rev6.7)**. It follows
rev6.6 (#728, merged first); the two touch no common table row.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **New WP W8c, the production `LaneDInputs` source (owner 2026-11-23, DECIDED: "Yes w8c").** Gap: W8b (`cascade/validator/lane_d_wiring.py`, `main.py`) calls `settle_lane_d` with the `LaneDInputs` its `inputs` source returns, and nothing in production supplies one: an armed lane D refuses `inputs_unavailable` at every settlement and its share burns. W8c builds the inputs per armed boundary from signed or pinned sources only (intake entries, eval-host forecasts by the W5d scorer through the W15 cache and W4 ingest, the pinned `pool_d` snapshot, the W5f house, the bench session); deterministic, fail closed with a typed code per missing input, off the poll thread, and wired through `build_lane_d_step`'s `inputs` / `bench` parameters | Lane D core table, W8c row (after W8b) |

## rev6.6 (owner 2026-11-23; main @ 9917c83)

Records one owner decision, (a) ("can we allow up to 128k for lane D … so
we just [use] as much as the model they submit allows"; "for lane D they can
train for however long they up 128k context"). Plan text only. Every changed
item is marked **(rev6.6)**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Lane D contexts up to 128k from launch, each armed when its gates pass (owner 2026-11-23, DECIDED).** `[lane_d] contexts` lists **every variant from launch: {4096, 8192, 16384, 32768, 65536, 131072}**. A miner declares any of them at `ready`; a value not in `contexts` is still refused (§1.1: "the miner declares a context from `[lane_d] contexts`"). The scorer keeps today's rule: it truncates each window to the history it really has, ≤ the declared context, no padding. So a model uses as much context as it supports and the data allows. **Arming is automatic per variant, not a separate S8 overlay.** A variant is armed once **both** gates pass: (1) ≥ `min_long_series` (300) series from ≥ 20 sources can use it (W1/R16 readiness counts, measured after the forge gap fix); (2) T7's wall time and memory for that variant pass on the eval host that scores lane D (≤ 50% of a settlement, §1.6). **Until a variant is armed, an entry that declared it is scored at ~~the highest armed variant~~ (rev6.15) the highest armed variant below its declared context, and is skipped `context_below_armed` (stays pending) when none is below it** (the declaration is valid because the value is in `contexts`; it isn't refused, and it is scored at its own context from the settlement its variant arms). The main validator computes the armed set per settlement and records it, with the evidence it used, in the signed report, so satellites verify it instead of re-deriving it (W7a field, follow-up). Quality beyond 16k stays unproven (32× the trained length, §1.1); T7 measures it. The comparison stays fair because king and challenger are scored on identical targets. Training length and recipe are the miner's (lane D is open weights). **Open, flagged (not decided here):** the brief named the launch eval host as "a 96 GB GPU (the bench pod valeval-blackwell4 is an RTX PRO 6000 96 GB)". D11 still says the launch eval host is a pinned-torch 2.4.1+cu124 pod and **not** Blackwell sm_120, because the pinned torch can't run there (§1.6, §10.2, D11). This rev keeps D11. Gate (2) is measured on whichever pinned-torch host scores lane D. On the ≥ 24 GB launch host, 32k–128k will likely fail gate (2) until the 96 GB pinned-torch valeval-laned is provisioned (W6-S8). Moving the launch eval host to a 96 GB pinned-torch pod is a separate owner call: **[DECIDE D11]** | §1.1 context bullet; §1.7 "Moved to stage S8" + harvest bullets; W2 row (`contexts` default); W6-S8 row; T7 row; §5 S8 row; D2, D10 (§6); §11.2 `docs/LANE_D.md` row; §17 S8 row; §17.4 D2 |

## rev6.5 (owner 2026-11-23; main @ e77ee6e)

Records one owner decision, (a). Plan text only; nothing else in the design
changes. Every changed item is marked **(rev6.5)**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **WB19 skips the full lane A 8×H100 leg (owner 2026-11-23, DECIDED: "Yes skip it").** The measured `[round] funded_sku_wall_seconds` in the deployed chain.toml stands for the full-leg wall. WB19 measures only the per-GPU expert legs (cost cap $20) | WB19 row |

## rev6.4 (owner 2026-11-23; main @ 54e7f23)

Records two owner decisions, (a) and (b), and one owner finding, (c). Plan
text only; nothing else in the design changes. Every changed item is marked
**(rev6.4)**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **W8b accepted (owner 2026-11-23, DECIDED: "i accept w8b").** The rev6.2 (g) proposed WP W8b (validator production wiring of `settle_lane_d`) is accepted; its row is otherwise unchanged | Lane D core table, W8b row marker |
| (b) | **CI-first testing for loop agents (owner 2026-11-23, DECIDED).** An agent runs only its own WP's `tests/accept/<wp>/` dir locally; every other suite (unit shards, receipt-replay, CROSS) runs in CI, and all are required green before merge. Why: the production box (4 cores / 7 GB) also scores the validator, so local full suites stall scoring | §3.1 rule 10 |
| (c) | **Hippius has no Object Lock (owner finding 2026-11-23).** `GetObjectLockConfiguration` / `GetBucketVersioning` return 403 for a full-access bucket-scoped sub-token and for the main key; the console offers no lock permission. W3a re-runs its probe with `--no-object-lock` (lock probes recorded as `skipped`). Consequence: lane D kings are **not** locked by a COMPLIANCE copy at crowning; their immutability rests on the content-addressed path, the ingest re-hash on every read, the signed report's digest, and the R2/HF mirrors | §1.2 Storage (Object Lock bullet, "lock only kings" bullet); W3a row; rev5 B9 row |

## rev6.3 (owner 2026-11-22/23; main @ a244271)

Adds one proposed WP, (a), and one owner decision, (b). Plan text only;
nothing else in the design changes. Every changed item is marked **(rev6.3)**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Proposed (owner 2026-11-22 approved adding it):** WB24 lane B miner CLI: `cascade lane-b fund` gains `--declared-rows <file>` (signed WB14 header + `rows=` tag, local refusal with `DECLARED_ROWS_CODES`), `--target suite:<s>` (WB18) and `--train-hours` (WB23, once built); `lane-b status` shows the deferral/refusal reasons; new `lane-b verify-repo <dir>` runs the WB13 scan locally (no upload, never executes the repo) | Lane B table, WB24 row |
| (b) | **Batch implementation PRs (owner 2026-11-23, DECIDED).** An implementation PR may batch **up to 3 WPs** when (1) their PATHS sets are disjoint, (2) none depends on another in the batch, (3) each WP's tests are already on main. Each WP in the batch is still checked on its own: scope (its PATHS only, files attributed per WP), its own accept, and CROSS; the ledger records one row per WP. A red WP is split out of the batch by the builder (the others proceed); WPs whose PATHS overlap stay serial. Tests-corrections found in one pass may likewise be batched in one PR (one dir per WP). Labels: one `laned:<WP>` per WP in the batch | §3.1 rule 7 (former rules 7–8 renumbered 8–9); §5 S2 Build row |

## rev6.2 (owner decisions 2026-11-22; main @ 6108ebd)

Records five decisions the owner approved on 2026-11-22, (c)–(f) ("do all ur
recommendations") and (h), plus one proposed WP, (g). Plan text only; nothing else in the design changes. Every
changed item is marked **(rev6.2)**. They are build-level rules inside
existing WPs, not new §6 D-numbers, so the §17.4 table is unchanged; the
dated entries are listed under it (§17.4, "Owner decisions 2026-11-22").

| Item | Decision | Landed in |
|---|---|---|
| (c) | LB1 normalised hashes use the WB11 **schema 2** z-grid rule (`cascade_contamination_hashset/2`, encoding `cascade-zgrid/2`), replacing "rounded to 4 significant figures"; low-information windows are excluded at build and ignored at lookup (RV6). PRs #617/#621, #632 | §12b.1; WB11 row |
| (d) | While a lane B funded entry is deferred (`sentinel_target` or `rows_held`), its payer key in `PayerKeyVault` gets its TTL refreshed on each deferred tick, the same rule as the entry (WB18 RV6, #629/#633). No WP owns the vault, so it's a follow-up note on WB18 | WB18 row |
| (e) | The W18 vault ledger `laned_vault_ledger/1` (current era, entries with statuses / `closed_era`, open disputes) is produced by W17 | W17 row |
| (f) | A W-S2 warm-start acceptance made at an era-init block is governed by the sentinel of the era that init block opens; a follow-up for the acceptance callers (trainer `accept_candidates`, validator `verify_acceptance`) | §16 Work and tests; W-S2 row |
| (g) | **Proposed (owner to accept):** new WP W8b wires `settle_lane_d` into the production validator loop (boundary hash, sentinel, bench `SweepSession`, `expect_prior_sha256`, lowfreq, house/king state; then sentinel sweep/board, signed report + receipt, pay into W8 emission only via the report); fail-closed, restart-safe, off or bounded against the weight loop | Lane D core table, W8b row |
| (h) | **Batch tests-only PRs (owner 2026-11-22).** Implementation, tests-correction, evidence and review PRs stay one WP per PR; a tests-only PR may carry several WPs' `tests/accept/<wp>/` directories, one `laned:<WP>` label per WP, each WP's expect-fail (or verification) accept run executed and ledger-recorded separately, the whole PR failing if any WP's check fails; dependents still only after the dependency's code is on main | §3.1 rule 6 (+ restatements: rev6 change 16, §3.0 Roles, §17 S2 row, risk table "PR hygiene") |

## rev6.1 (RV1 rubric fixes; main @ bdeed4d, 2026-11-21; RV3 fixes, main @ 45f8b61, 2026-11-22)

Plan text only, **no design change** (proposals 23–26 are not applied).
Every changed item is marked **(rev6.1)**. Sources: RV1 (`evidence/rv1.md`
on `laned/rv1`), non-blocking notes 1, 2, 10, 12 and the lane A consequence
of R3's lint (items 1–3); RV3 (`evidence/rv3.md` on main), BLOCKING 1 and
notes 2, 5, 10, 11, 21 (items 4–10).

| Item | Change | Landed in |
|---|---|---|
| 1 | W0 `Runs in` `ci + owner merge` → `ci`. The owner merge is the PR merge every WP has (§3.1 rule 6); as written it made W0 "external", so it needed an `evidence/w0.json` that nothing produces, blocking W0 at L4/L5 and so S2 (RV1 note 1) | §3.0 W0 |
| 2 | "Build WP" defined for §17.2: the §3.0 WPs whose `Runs in` is ci-only, or ci plus pod/testnet for their CI-checkable parts; owner execution WPs (R2x, R6x, R7t, R7m, R9, R13, R14, R15, R16, X*) are excluded and gate their own stages. Consistent with the §3.0 Review WPs count (RV1 note 2) | §17.2 S2 row |
| 3 | Proposed design changes 23 (code-enforced stake gate for `[lane_d] from_block`), 24 (`[lane_c] remainder` restatement refused) and 25 (lane A tuning after lane D activation). Not applied: owner decision (RV1 notes 10, 12) | Proposed design changes 23–25 |
| 4 | W6b must write `frontier` for every series (one record per earlier snapshot holding it) and a `feeds` entry (`mirror_lag_hours`, `forecast_product`) for every feed in `pool_d`; a `pool_d` missing either is invalid (RV3 BLOCKING 1) | §3.0 W6b, §1.3 (W6 metadata bullet), §1.7 Metadata |
| 5 | W9 refuses any snapshot/series lacking `frontier` or with an unlisted feed, and re-derives S0 / backfill / lag eligibility without importing `cascade/validator/lane_d_forward` (invariant 12) (RV3 BLOCKING 1) | §3.0 W9 |
| 6 | T8 clock labels corrected to §1.3 and `lane_d_forward.py:103-110`: data time passes, first-seen fails ⇒ `publish_ahead`; first-seen passes, data time fails ⇒ `late_mirror` (data old, mirror saw it late); missing `frontier` / unlisted feed added as refused cases (RV3 note 5) | §4 T8 |
| 7 | Pre-S5-restart snapshot extended to the trainer state under `/root/cascade/_train_work/` (`era_state.json`, `trainer_submissions.json`, `funded_queue.json`, `trainer_promotion.json`, `champion_publisher.json`, `lane_b_driver.json`, `coldkey_identities.json`, `published_settlements.json`, its `activation_*` / `netconfig` files), `provisioner_state.json`, `pod_monitor_state.json` and the deployed `chain.toml` (live overrides) with `hosts.toml` / `provision.toml` (RV3 note 21) | §10.8, §3.0 R5, §10.5 S5 |
| 8 | Wiring trap: `validator/weights.py` and the audit pass lane D's element to `emission_vector` only when a B or C pay record exists, so lane D's share would be lost; the WP that wires lane D pay must add the lane D record to that condition, and W9 decides invariant 12 for the shared lane D weight math (RV3 notes 10, 11) | §3.0 W7c, W9 |
| 9 | W3b5 defines the low-frequency result for a confirm candidate whose frozen set holds low-frequency windows: `pass`/`fail` with the true `n_windows` (replacing W7a's interim `not_candidate`); enum already holds both, extended only for a new value | §3.0 W3b5 |
| 10 | Proposed design change 26: arming the §16 sentinel (W-S1) gated on R4's observed `cfg-` stake ≥ 2/3 **and** all satellites upgraded. Not applied: owner decision (RV3 note 2) | Proposed design change 26, §16 (pointer) |

## rev6 (completion rubric, loop-readiness fixes, stale items; main @ 07117e1, 2026-11-21)

Plan text only. No code, chain.toml or prod-box changes, and **no design
change**. Every changed item is marked **(rev6)**. Anything else I'd change is
under "Proposed design changes" (change 22), not applied.

**Base.** The branch `claude/sleepy-cray-7uutm1` is on the remote and already
contains origin/main `07117e1` (main hasn't moved since rev5), so no rebase was
needed and no force-push was done. The repo facts below were re-checked on
`07117e1`.

**Changelog (brief item → where it landed):**

| Item | Change | Landed in |
|---|---|---|
| 1 | §17 "Completion rubric and status board": `scripts/laned/status` → `docs/plans/STATUS.md` + `status/laned_build.json` on `laned/build-state` (never merged); 17.1 levels L0–L7 + side states; 17.2 stage completion; 17.3 counters; 17.4 decision register (status, option, class S/O, who/when) + owner inputs OA1–OA8; 17.5 milestones; 17.6 agent rules | §17 (new); status line; §3 conventions |
| 2a | §3.1 rule 6: a tests-only PR then ONE implementation PR; consistent with rule 1. Rule 7 and the Review WPs paragraph count WPs at L3 | §3.1, §3.0 Review WPs |
| 2b | W0c adds (i) `scripts/laned/status` + `accept --gate`; (ii) `laned-accept.yml` (PRs labelled `laned:<WP>`, expect-fail mode for tests-only PRs, main after merge); (iii) the rule-1 CI check (and the `PATHS` check); (iv) self-tests (fixture repo ⇒ exact levels; planted rule-1 violation fails) | §3.0 W0c |
| 2c | Roles named (test author, builder, preparer, reviewer); a builder never edits `tests/accept/`, the plan, `scripts/laned/` or the status files | §3 conventions, §17.6 |
| 3a | S6 is one stage (D45): X6b removed; X6a Accept = observed `cfg-` stake ≥ 2/3 before `effective_block`, webhook test alert, first paid settlement's vectors match on ≥ 2 validators and satellite replay matches; no "≥ 2 eras of reports" | §3.0 X6a/X6b, Self-check stage map |
| 3b | S2b: "diffs only W2–W12" → "diffs only the lane D and lane B build WPs (§5 S2)" | §5 row 7, §14.5 |
| 3c | Appendix A → a pointer to §17 and STATUS.md; OA1–OA8 moved to §17.4 with status, what each unblocks and its evidence file; `LANE_D_PROGRESS.md` retired (pointer) | Appendix A, §17.4, Appendix B, `LANE_D_PROGRESS.md`, `decisions/DEC-CA-0070-…` |
| 3d | Leftover shadow / separate pay stage: §1.4 D21 table ("share-0 shadow (S6a)"), §1.4 "S6b signed overlay", §5 row 12b (struck), §13 row, §15 "before S6b", v3 changelog items 4, 6, 14, Proposed change 5; `LANE_D_S1_RESULTS.md` ("S6a shadow") is retired to a pointer (change 21) | those places |
| extra | W14 was still "blocked: needs owner (Proposed change 4)", but change 4 is accepted: W14 unblocked; T27 and the §5 kill-switch text now say "until W14 lands" | W14, T27, §5 kill switch, rev5 A2 rows |
| extra | §5 S0 gate reads the §17.4 register | §5 row 1 |
| extra | Proposed change 22: five §15 rows (D25–D27, D29, D31) have no recommended option; recorded as Option A, owner to confirm | Proposed design changes, §17.4 |

**Propagation (item × place):**

| Item | § | WP | T-row | Stage map | Docs |
|---|---|---|---|---|---|
| 1 §17 | done (§17, status line) | done (W0c builds the script) | n/a: no T-row tests build tooling | n/a: §17.2 is the stage rule itself | done (STATUS.md is generated, not committed) |
| 2a rule 6 | done (§3.1 rules 6, 7; Review WPs) | n/a: applies to all WPs | n/a | n/a | n/a |
| 2b W0c | done (§17.1 L1 expect-fail) | done (W0c) | n/a: W0c's own self-tests | n/a | n/a |
| 2c roles | done (§3 conventions, §17.6) | n/a | n/a | n/a | n/a |
| 3a S6 | done (§17.2 S6) | done (X6a, X6b removed) | n/a: T27 and T29 unaffected | done (S6 row) | n/a |
| 3b S2b | done (§5 row 7, §14.5) | n/a: R9's Accept unchanged | n/a | n/a | n/a |
| 3c Appendix A | done (Appendix A, §17.4, Appendix B) | n/a | n/a | n/a | done (`LANE_D_PROGRESS.md`, `LANE_D_S1_RESULTS.md`, DEC-CA-0070 pointer) |
| 3d shadow leftovers | done (§1.4, §5 row 12b, §13, §15, v3 changelog, change 5) | n/a | n/a | done (S6) | done (`LANE_D_S1_RESULTS.md`) |
| W14 unblocked | done (§5 kill switch) | done (W14) | done (T27) | n/a | n/a |

**Repo facts that turned out wrong or missing:**

- **W14 "blocked: needs owner (Proposed change 4)"** was stale: changes 1–17
  were accepted in rev5.
- **"The lane D release diffs only W2–W12"** (S2b) and Appendix A's "W2–W12 are
  not built": the build set is W1.5–W19, W-S1–W-S3, WB10–WB23 and W-EMA (§5 S2).
- **`LANE_D_S1_RESULTS.md`** still said the forward re-check "happens at the S6a
  shadow"; Appendix B had the rev5 fix.
- **Nothing runs `tests/accept/` in CI today.** `.github/workflows/ci.yml` runs
  `pytest tests/unit` only. L1 and L3 need W0c's workflow (ii).
- **There's no Makefile on main**, so W1.5 creates `make laned-e2e` from
  nothing (a new top-level `Makefile`).
- **No `laned:*` GitHub labels exist.** Creating them is an owner step before
  the first tests-only PR (W0c row).
- **`tests/accept/`, `scripts/laned/`, `evidence/` and branch
  `laned/build-state` don't exist yet**, as expected before W0c.


## rev5 (code alignment and review gaps; main @ 07117e1, 2026-11-21)

Plan text and decision drafts only. No code, chain.toml or prod-box changes.
Every changed item is marked **(rev5)**.

**Base.** The branch `claude/sleepy-cray-7uutm1` is rebased onto main `07117e1`
(2026-11-21) and pushed. Every fact below was re-checked on `07117e1`. The
MarketCast PRs #475 and #478 were merged into stacked base branches, but their
content is on main through #453's squash: the trees are identical. `3c67b64`,
cited in earlier revisions, was an older commit of this branch.

**Changelog (item → where it landed):**

| Item | Change | Landed in |
|---|---|---|
| A1 | Training rows 128 patches; 8k/16k/32k/128k = 2×/4×/8×/32×; CPM span ≤ 16 patches ⇒ horizons ≤ 512 points | §1.1 (architecture facts, 128k bullet, Horizon "longer rungs") |
| A2 | Missing lane D report = check can't run: old vector re-asserted for `stale_after_blocks` (3600), then a whole-vector burn; W14 blocked on Proposed change 4 (**(rev6)** change 4 accepted; W14 unblocked) | §5 kill switch, §10.8, W14, T27, Proposed change 4 |
| A3 | `[generation] king_publish` (`chain.toml:1555` on `07117e1`) | §15 D26 |
| A4 | King check = `king_check_gate` over four suites; W-S1 names it | §16, W-S1 |
| A5 | `netconfig.py:210`; `lane_d.py:78-79` | W2 |
| A6 | TIME → the era's sentinel; `unselected_suite` dropped (duplicates `[v2] sentinel_suites`) | §1.3b, §1.6, W2, W3b3, T7b, §10.4, §11.1, §11.2, §14.2 |
| A7 | Lane B record renumbered **DEC-CA-0071** (main has 0069 MarketCast hooks; 0070 = lane D); file renamed | everywhere (20 refs), Appendix C, `decisions/DEC-CA-0071-…`, DEC-CA-0070 numbering note |
| A8 | S0 and W0 cover D1–D43, D45–D48, D42b; D25–D32 rows in §6 | §5 row 1, W0, §6 |
| A9 | T8 two-clock rule; launch eval host (not valeval-laned) at S4; S2 = all build WPs; self-check T14 → B1 + S4 gate; §12 and §5 row 12b; §2 items 7/8; OA order | T8, §1.2, §1.5, §1.7, §5 S2/S4, X4, §10.2, §10.4, §10.5 S4, T7, §11.2, self-check, §12, §2, App. A |
| B1 | R8: AREA for `cascade/lane_d/`, every-path-classifies test, on-pod trainer modules in `WORKER_IMAGE`; consensus and worker-image paths listed | R8, §3 Services touched |
| B2 | `declared_rows` keys = MoE ordinals 0–6 | LB2, WB14, TB13, Appendix C |
| B3 | Clamp minimum only on layers with ≥ 1 row | LB2, WB14, TB15, TB15b, Appendix C |
| B4 | Shared image; dispatch `_dispatch_funded` → `_run_funded_leg`; declared rows need no image change (confirmed: `--selection-json`); D40(b) a shared repin; D40(b) and every repin ≥ rc16 depend on W-EMA | §3, D40, WB16, SB track |
| B5 | Release-only: `(lane_b, contamination_hashset)`, `(lane_c, remainder)`, `(lane_d, remainder)` | §14.2, WB10, Appendix C, §14.2 lane D row |
| B6 | W6a join on (source, pre-suffix source key, channel), same build | W6a, Proposed change 7 |
| B7 | Quantile head, 9-level grid, no copula sampling; W0b compares quantile-head outputs | §1.5, W0b, W5d, W7a |
| B8 | Deregistration: entry voided `deregistered`; a court slot burns | §1.5 state, W5h, W8, T18c |
| B9 | ~~Lock only kings (copy at crowning);~~ **(rev6.4)** no lock: Hippius has no Object Lock, so kings rest on content addressing + re-hash on every read + signed report digest + R2/HF mirrors; pending = content addressing + digest re-check; W3a probes `UploadPartCopy` | §1.2 storage, W3a |
| B10 | `[lane_d] ready_ttl_blocks` ⇒ `ready_expired` | §1.5 state, W2, W3, W5h, T18d, §14.2 |
| B11 | **W-EMA**: score the raw cooled endpoint on every leg; EMA files unscored; telemetry only | §8, W-EMA row (§3), §3 Services touched, D40, WB16 |
| B12 | Daily Merkle root of the first-seen log committed before use | W6fs, W7a, §9 conflict of interest |
| B13 | Licence attestation at `ready`; Hub/HF takedown path | W3, W13, §11.2 LANE_D.md |
| B14 | `[lane_d] max_source_share` | W5b, T18e, D29 (§6), W2, §14.2 |
| B15 | Rented eval host sees loser weights and `pool_d_hist` | §9 |
| C1 | **D46** house T (lineage commit block recommended) | §6; Proposed change 18 withdrawn (wrong premise) |
| C2 | **D47** house precision under BF16 | §6 |
| C3 | **D48** where D's share comes from inside A; `lane_pool_shares` signature | §6, W8 |
| D | Deployment facts (below) | this changelog |
| D42 | Owner: lane B GPU list + miner `train_hours` 0.5–6 h; D42b | §12b.0, §12b.6, D42, D42b, WB19, WB22, WB23, TB23–TB27, OA8, App. C |
| D43 | Owner: lane D reveals immediately | §1.2 step 4, W10, D43 |
| D44 | Drafted (EMA instead of cooldown), **withdrawn by the owner** | §6 (withdrawn row) |
| D45 | Owner: no shadow modes; one activation block | §5, D10, D19, D37, X6a, §14.5, §16, W-S1, SB5 |
| Harvest | Owner: long-history harvest to 128k now, scored from S8 | §1.7, R16, §5 S1 |

**Propagation table** ("n/a" says why):

| Item | § | WP | T-row | Key list | §14.2 | Report schema | State | Docs |
|---|---|---|---|---|---|---|---|---|
| A1 | done | n/a: fact only | n/a: T7 measures contexts | n/a | n/a | n/a | n/a | n/a: LANE_D.md cites §1.1 |
| A2 | done | done (W14) | done (T27) | n/a | n/a | n/a: change 4 undecided (**(rev6)** accepted) | n/a | n/a: OPERATOR row points at §5 |
| A3 | done | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| A4 | done | done (W-S1) | n/a: TS1–TS3 unchanged | n/a | n/a | n/a | n/a | n/a |
| A5 | done | done (W2) | n/a | n/a | n/a | n/a | n/a | n/a |
| A6 | done | done (W2, W3b3) | done (T7b) | done (dropped) | done | done (index row) | n/a | done (LANE_D.md) |
| A7 | done | done (WB21) | n/a | n/a | n/a | n/a | n/a | done (DEC files) |
| A8 | done | done (W0) | n/a | n/a | n/a | n/a | n/a | n/a |
| A9 | done | done (X4, S2 list) | done (T7, T8) | n/a | n/a | n/a | n/a | done (OPERATOR row) |
| B1 | done | done (R8) | n/a: R8 accept runs the classify test | n/a | n/a | n/a | n/a | n/a |
| B2 | done | done (WB14) | done (TB13) | n/a | n/a | n/a | n/a | done (App. C) |
| B3 | done | done (WB14) | done (TB15, TB15b) | n/a | n/a | n/a | n/a | done (App. C) |
| B4 | done | done (WB16, W-EMA) | n/a: (b) undecided | n/a | n/a | n/a | n/a | n/a |
| B5 | done | done (WB10, W2) | done (WB10/W2 accept: overlay refused) | done | done | n/a | n/a | done (App. C) |
| B6 | done | done (W6a) | n/a: W6a accept | n/a | n/a | n/a | n/a | n/a |
| B7 | done | done (W0b, W5d) | done (W5d CI test) | n/a | n/a | done (W7a `scoring_method`) | n/a | n/a |
| B8 | done | done (W5h, W8) | done (T18c) | n/a | n/a | done (typed reason) | done (`deregistered`) | n/a: LANE_D.md FAQ via WB20/W-docs |
| B9 | done | done (W3a) | n/a: W3a probe evidence | n/a | n/a | n/a | n/a | n/a |
| B10 | done | done (W2, W3, W5h) | done (T18d) | done | done | done (typed reason) | done (`ready_expired`) | n/a: LANE_D.md covers the ready flow |
| B11 | done (§8) | done (W-EMA) | done (W-EMA tests) | n/a: image constant, not a key | n/a | n/a: telemetry, not in the receipt | n/a | n/a |
| B12 | done (§9) | done (W6fs, W7a) | n/a: W6fs accept | n/a | n/a | done (W7a roots) | n/a | n/a |
| B13 | done | done (W3, W13) | n/a: W3/W13 accept | n/a | n/a | n/a | done (attestation at ready, W3) | done (LANE_D.md) |
| B14 | done | done (W5b, W2) | done (T18e) | done | done | n/a: windows already listed | n/a | n/a |
| B15 | done (§9) | n/a: accepted risk | n/a | n/a | n/a | n/a | n/a | n/a |
| C1–C3 | done (§6) | done (W8 for C3) | n/a: until decided | n/a | n/a | n/a | n/a | n/a |
| D42 | done | done | done (TB23–TB27) | done | done | done (signed leg record) | n/a | done (WB20) |
| D43 | done | done (W10) | done (W10 test) | n/a | n/a | n/a | n/a | n/a |
| D45 | done | done (X6a, W-S1) | n/a: removes shadow rows | n/a | n/a | n/a | n/a | n/a |
| Harvest | done | done (R16) | n/a: R16 accept | n/a | n/a | n/a | n/a | n/a |

**New keys** (each: loader parse, ALLOWED or release-only, round-trip and
mutation tests):

| Key | Parse + tests in | Overlay? | Tests named |
|---|---|---|---|
| `[lane_d] ready_ttl_blocks` | W2 | `ALLOWED["lane_d"]` | W2 accept: parse, round-trip, mutation, `unread_keys.py` |
| `[lane_d] max_source_share` | W2 | `ALLOWED["lane_d"]` | W2 accept: same |
| `[lane_d] remainder` (existing placeholder) | W2 | **release-only** (`netconfig.py:210` tuple) | W2 accept: an overlay setting it is refused |
| `[lane_c] remainder` (existing) | W2 (shared release) | **release-only** (newly) | W2 accept: overlay refused; unarmed digests byte-identical |
| `[lane_b] contamination_hashset` | WB10 | **release-only** | WB10 accept: overlay refused |
| `[lane_b] funded_pod_skus`, `max_leg_hours`, `max_leg_seconds`, `guard_factor` | WB10 | `ALLOWED["lane_b"] = None` | WB10 accept: parse, digest drop-when-default, round-trip, mutation |

**Repo facts in the brief that turned out wrong or imprecise:**

- **A1:** `max_patches` isn't "read by nothing". Lane B's horizon-affinity
  decode reads it (`expert_lane.py:365`); training doesn't.
- **A3:** on `07117e1`, `king_publish` is at `chain.toml:1555` (it was `:1554`
  on `5ce1f88`).
- **B6:** "source id + content hash" can't be the join key. `content_hash` is
  the sha256 of the values (`builder.py:295`), so it changes as tails move,
  and `pool` and `pool_d` histories hash differently. The plan uses (source,
  pre-suffix key, channel) within one build.
- **C3:** the king's ladder share runs from **10%** (after a fail) to 95%, not
  25–95% (`chain.toml:1223-1226`).
- **A7:** PR #479 also claimed DEC-CA-0069; **(rev6)** renumbered DEC-CA-0072 after merge.
- **B11:** rc15n (#477's minimal image, built on rc15 `5e3fade` + the pin)
  doesn't contain power EMA. W-EMA blocks #477's repin only if that image is
  built from main.
- **Earlier session errors corrected:**
  - "Stop publishing ⇒ immediate whole burn" was wrong; it's a one-era
    re-assert first (A2).
  - "The king retrains every era" was wrong; it trains once per seed epoch
    (Proposed change 18 withdrawn).
  - "`ALLOWED["lane_b"] = None` blocks overlays" was wrong; it allows every
    key.

**D: deployment-only facts (read-only checks on the box):**

| Fact | Result |
|---|---|
| rc37 validator / rc41 trainer | **verified:** service PYTHONPATHs are `/root/cascade-val-rc37` and `/root/cascade-rc41` |
| 21 LIVE OVERRIDEs | **verified:** 21 in the deployed chain.toml |
| keep-alive (#471) absent from rc37 | **verified:** `_weights_keepalive_tick` not in rc37 |
| valpriv-4090 spec (24 GB) | host listed in `valpriv-eval.toml`; **VRAM unverified (box)** |
| valeval-blackwell4 spec | **unverified (box):** not in the deployed config (only blackwell2's standing cost) |
| `[lane_d] takes_from = "lane_b"` | **verified** on the deployed file and on main |
| Lane B image = lane A image | **verified:** both pins are rc15 `5d41d94b…`; no image key under `[lane_b]` |
| OPSLOG line refs | **drifted** (newest-first log). `:2329` is replaced by its entry heading; `:164` is unverified |
| Any image ≥ rc16 in production? | **No evidence.** rc16 was built and qualified on 2026-11-01, but the switch was held as consensus-unsafe and the trainer stayed on rc15. rc15n (#477) was tried on 2026-11-20 and reverted. The deployed pin is rc15. Experiment pods aren't covered by this check |

## rev4 (lane B pass, 2026-11-21)

Lane B stays trainer-run, with miner-chosen expert rows (DEC-CA-0071 Open
option (b)) and one-suite targets ("improve one, regress none"); see §12b.
The miner-weights variant (§12) is parked. D32 is resolved, and D36–D42 are
new. Every changed item is marked **(rev4)**.

**rev4 mappings:**

| Item | Maps to |
|---|---|
| TB10–TB12 | WB11, WB13 |
| TB13, TB15 | WB14 |
| TB14 | WB16 |
| TB16, TB21 | WB15 |
| TB17, TB18 | WB17 |
| TB19, TB20 | WB18 |
| TB22 | WB10 |
| TS1–TS3, TS5 | W-S1 |
| TS4 | W-S2, WB18 |
| Sentinel board | W-S3 |
| Every new `[lane_b]` key | WB10 (plus the owning WP named in the §3 table) |
| LB6 measurement | WB19 |
| Docs | WB20 |
| DEC-CA-0071 amendment | WB21 |
| Lane B track | SB0–SB5 (§5) |

## rev3 (design pass, 2026-11-21)

This pass changes the design: items A1–A11 of the rev3 brief, plus the
incentive review in §15. Every changed item is marked **(rev3)**. New
decisions are D20–D24 and D25–D32 (§15). The S1 measurements are in
`LANE_D_S1_RESULTS.md` and the run status in `LANE_D_PROGRESS.md` **(rev6: both retired; status is
§17, results are Appendix B)**. The
v3 loop-readiness material below predates rev3. A new loop-readiness pass
runs only after the owner settles rev3's [DECIDE]s.

**rev3 additions to the v3 mappings:**

| Item | Maps to |
|---|---|
| T4b | W3b5 |
| T6b | W5h, W3, W9 |
| T9b | W5e |
| T18b | W5h |
| T14 (redefined) | B1 `scripts/null_fpr.py`, W5e, S4 gate |
| W6fs | S1 / S2 (launch-blocking) |
| W6-S8 and R15 | S8 and S3 |
| New stage S8 | W6-S8 |

## v3 changelog

The design is unchanged: lane D rules, judging, storage, pay, rollout order
and decisions D1–D18 stay as they were in v2. Changes:

**Contradictions fixed**

1. **§5 S2 gate.** Receipt replay now reads: unarmed ⇒ byte-identical; armed
   ⇒ lane A verdicts, court and lane A weight entries identical, and lane D
   fields present and schema-valid. It matches invariant 2.
2. **§10.5 S5** aligned with §5 S5. The order is eval host → validator →
   lane D intake → trainer, and all 8 snapshot files from §10.8 are listed.
3. **§14.4 / §14.5:** "S1b" renamed to S0a–S0c (§5).
4. **§13** "live overrides" row: "typed-block activation" → "signed-overlay
   activation (D10; S0c, S6a)" **(rev6: S6a is now S6, D45)**.
5. **§1.2:** sub tokens were said to suit the eval pod, which contradicted
   "presigned GETs only, no credentials on pods". It now says builder only;
   the pod gets presigned GETs.
6. **§1.4:** the `takes_from` change goes by the S6b signed overlay, not a
   chain.toml edit (consistent with §5 S6b and §14.2). **(rev6)** S6b is
   merged into S6: it's the one S6 activation overlay (D45).
7. **§1.3:** "judged once on that frozen set" → "judged once on its cohort's
   intersection set, ⊆ its frozen set" (consistent with §1.3 Cohorts).
8. **§1.3b:** "rows come from cached sweeps" → "re-swept in the same host
   session" (consistent with the paired-on-one-host rule).
9. **§1.2:** the cross-reference "§2.2" (which doesn't exist) → "§2 item 1".

**Citations re-verified against the repo**

10. `store.py:337-344` → `store.py:61, 340-367` (`DigestOwned` is defined at
    `:61` and raised at `:348`, `:365`). §1.2.
11. `builder.py:262-277` → `builder.py:279-290` (the metadata dict). §1.3.
12. `netconfig.py:495-505` → `netconfig.py:346-354` (`build_config` merges
    onto `effective_raw(base)`). §5 S0a.

    Checked and still correct: `checkpoint_guard.py:139-171, 151-153,
    175-196, 290-302`; `pyramid_model.py:89-93, 210-211, 259`;
    `pyramid_wrapper.py:35-38, 51`; `lane_c_manifest.py:199-203`;
    `chain.py:495-535`; `store.py:258-275`; `scripts/deploy_gate.py:55`;
    `builder.py:264-271`; `tsbench_forge.py:149, 204, 380-399, 446-460`;
    `windows.py:433-449, 556-611`; `bootstrap.py:225, 393-418`;
    `emission.py:330-420`; `eval_offload.py:7-10`; `validator/pool.py:71-77`;
    `publish-pool.yml:72-79` (timeout 180 at `:79`), `121-131` (10 GB at
    `:129`); `activation.py:1334-1357`; `lane_d.py:65-68`;
    `netconfig.py:83-118, 195-217`; `loop.py:1030, 1078, 1410`;
    `lane_b_driver.py:777`.

**Additions**

13. **§3.2 invariants 9–12:**
    - 9: `share = 0` gives an identical vector plus a valid report.
    - 10: paired unarmed and armed tests.
    - 11: typed reason enums.
    - 12: audit and replay code is independent of producer code.

    The invariants also now name their owning WPs.
14. **§5 S6a and §6: D19**, an owner-run shadow test entry on a dedicated
    hotkey (recommended). **(rev6) Superseded** by D45 and D19 (rev5): the test
    entries run on testnet in S3; there's no mainnet shadow and no S6a.
15. **§3 restructured for the loop:**
    - Accept and Runs-in columns.
    - The `scripts/laned/accept` convention.
    - External WPs are blocked, never mocked; owner steps are prepared but
      not run.
    - New WPs:
      - W0c loop tooling (stub scan, unread keys, follower patterns);
      - **W1.5 walking skeleton** (`make laned-e2e`);
      - W2b `[lane_d_ops]`;
      - W5 split into W5a–W5g; W6 into W6a–W6d plus W6s-*n* per source;
        W3b into W3b1–W3b4; W7 into W7a–W7d;
      - W3 and W10 are gated on W3a's storage decision record and build only
        the chosen path.
    - New owned work:
      - W13 king publication; W14 fast kill; W15 prefetch LRU; W16 reference
        fine-tune; W17 `status/lane_d.json`; W18 TTL sweeper;
      - W19 off-poll-thread + keep-alive;
      - W12 split into W12a docs and W12b status/discovery.
    - Rollout tooling R1–R14:
      - block arithmetic, the note-budget decoder plus its live run, overlay
        lint, the stake tracker;
      - previous-release worktrees and restart scripts;
      - the de-drift classifier plus its execution, overlay bring-up
        artefacts plus testnet and mainnet runs;
      - `deploy_gate` for the new services, the catch-up release;
      - the state-copy upgrade harness, the shipped-config startup-gate
        test;
      - the eval-host provisioning kit, builder provisioning, testnet stack
        provisioning.
    - Review WPs RVn and execution WPs X3–X7.
16. **§3.1 build rules:**
    - tests-first in read-only `tests/accept/`;
    - mock only external services;
    - no placeholders (stub scan in every Accept);
    - per-key parse, ALLOWED entry, round-trip and mutation tests, with an
      unread-key script;
    - follower patterns checked by a script;
    - one PR per WP, dependents only after the dependency is on main
      **(rev6.2:** implementation, tests-correction, evidence and review PRs
      stay one WP per PR; a tests-only PR may batch several WPs' tests, one
      `laned:<WP>` label and one separately recorded accept run per WP;
      dependents only after the dependency's code is on main**)**;
    - a fresh-context review after every 5 build WPs;
    - blocked beats faked.
17. **§3 "Services touched"** now lists `deploy_gate` and the trainer
    (capability release only).

### Self-check

**Every WP has an Accept and a Runs-in value: yes.** W1.5's Accept is
`make laned-e2e`, run in ci.

**T-rows → WPs** (all also executed in X3):

| Row | WPs |
|---|---|
| T1 | W5f, W8 |
| T2 | W3 |
| T3 | W16, W5e, W3b2, W13 |
| T4 | W5e, W3b2 |
| T5 | W4 |
| T6 | W3, W4 |
| T7 | W5d, W0b |
| T7b | W3b1–W3b4 |
| T7c | W3, W10, W15 |
| T8 | W5a, W6b |
| T9 | W5c |
| T10 | W5f |
| T11 | W7c |
| T12 | W7d, W9 |
| T13 | W1.5, W8 |
| T14 | **(rev5)** B1 (`scripts/null_fpr.py`, S1) + the **S4 gate**; W5e |
| T15 | W19 |
| T16 | W3, W5f |
| T17 | W3b4, W5g |
| T18 | W2, W5b, W14 |
| T19 | W4, R12 |
| T20 | W11 |
| T21 | R3, R7, W2 |
| T22 | R2 |
| T23 | W7d |
| T24 | R3, W2 |
| T25 | R10 |
| T26 | R11 |
| T27 | W14, R7 |
| T28 | W4, R8 |
| T29 | W12a (announcement), R14 |
| T8e | **(rev6.10)** W5a, W6b, W6fs, W2e, W9, W7d, **(rev6.14)** W5e, W5j |
| T9d | **(rev6.10)** W5j, W5b, W9, W7d |

**Invariants → WPs:**

| # | WPs |
|---|---|
| 1 | W1.5, W2, W8 |
| 2 | W1.5, W8, W9 |
| 3 | W4 |
| 4 | W5a, W9 |
| 5 | W7c |
| 6 | W19 |
| 7 | W7d |
| 8 | W6c |
| 9 | W1.5, W8 |
| 10 | W0c |
| 11 | W5g, W7a |
| 12 | W7d, W9 |

**Stage gates → WPs:**

| Stage | WPs |
|---|---|
| S0 | W0 |
| S0a | R6, R6x |
| S0b | R2, R2x |
| S0c | R7, R7t, R7m |
| S1 | W1, W0b, W3a |
| S2 | all W, plus RVn |
| S2b | R9 |
| S3 | R14, X3 |
| S4 | R12, X4 |
| S5 | R5, R8, X5 |
| S6 **(rev6)**: one stage (D45); S6a/S6b and X6b removed | R4, X6a |
| S7 | W11, X7 |

**§11.1 → WPs:**

| Item | WPs |
|---|---|
| vault | W3a, W3 |
| retention / archive | W18 |
| kings, index, cards, one-digest check | W13 |
| `pool_d` pins | W6d |
| eval cache | W15 |

**(rev4)** Lane B is no longer unchanged: its WPs are WB10–WB21 (§3) and its T-rows TB10–TB22 (§4). Lane A and the generator rows are unchanged, so no WP is needed for
them.

**§11.2 → WPs:**

| Item | WPs |
|---|---|
| LANE_D.md, INTERFACE, MINER, AGENT_MINING, AGENTS/agent.md, llms.txt (both), VALIDATOR, OPERATOR, ARCHITECTURE, website, README/CHANGES | W12a |
| `status/lane_d.json` | W17 |
| `cascade status` / `discovery-check` | W12b |
| Decisions (DEC-CA-0070, DEC-CA-0062 superseded) | W0 |

**Unmapped after the additions in item 15: none.** Before them, six items
had no WP: T15 / invariant 6, T25, T26, S4 eval-host provisioning, §10.2
builder provisioning, and §10.3 testnet provisioning. They're now owned by
W19, R10, R11, R12, R13 and R14.

**Nothing in §1–§2 design changed in v3.**

### Proposed design changes (NOT applied: owner decision)

**(rev5) Owner 2026-11-21: changes 1–17, 20 and 21 are ACCEPTED.** 18 is
withdrawn and 19 is adopted as D43. Each accepted change is built in the WPs
its "Affects" line names. The § text it affects is rewritten when that WP
lands, so until then the change text below is authoritative over the older
§ wording it contradicts.

These come from review round 2 (fresh context, read-only, checked against
the code). Each is verified where marked.

1. **Forward rule admits pre-T data.**
   - "Absent from the first snapshot after max(T)" is also true for values
     our mirror got late, and for series not selected in that snapshot
     (every point "absent").
   - Proposed rule: eligible iff (a) the series is present in S0 with a
     recorded `end_ts`, (b) `target_start_ts > max(T_E,T_K) + margin` in data
     time, and (c) every target ts > `end_ts(S0)`. Series absent from S0
     start their clock at first appearance.
   - T8's "backfills never eligible" needs this. Affects §1.3, W5a.
2. **Publish-ahead feeds aren't excluded automatically.** The forge reader
   drops rows with ts ≥ as_of + 1 day (`tsbench_forge.py:393-396`). Proposed:
   eligibility metadata from the mirror's landing time, or a per-source
   `publishes_ahead` deny-list. Affects §1.3, W6b.
3. **Copy of a fresh house can win.** `cohort_maxt_lcbs` compares challengers
   only against the king (`bootstrap.py:393-418`). Proposed:
   - a miner challenger must also clear `d_margin` against the current house
     on the same set;
   - the house ranks first on ties.

   Affects §1.3, T16.
4. **"Stop publishing ⇒ only lane D burns" is false.** **(rev5)** A missing
   report is a check that can't run: the satellite re-asserts the last
   verified vector for `stale_after_blocks` (3600), then burns the **whole**
   vector (`cascade/validator/satellite.py:30-45`). Proposed: a structured `lane_d: {status: "absent"}` receipt
   state with a lane-D-only burn, reusing MarketCast's `REPORT_ABSENT` /
   `INFRA_DEFERRED` (verified: `marketcast_report.py:32-34`). Affects §5 kill
   switch, §10.8, W14, T27.
5. **Refused overlays stay refused after upgrade.** The digest persists in
   `state.refused` (verified: `netconfig.py:822-826`). Proposed:
   - publish a restating seq N+1 after the stake gate if any node is known to
     have refused seq N;
   - or a code change that clears refusals on release upgrade.

   Affects §5 S6 **(rev6)**, §14.1, T23.
6. **Reuse MarketCast** (DEC-CA-0064: forward windows, absolute-grid window
   ids, signed per-window report with quantile sidecars, absent/deferred
   states) instead of reinventing W5/W7. Affects W5a–W7d.
7. **The window origin grid is undefined.** Proposed:
   - non-overlapping origins on the absolute grid `(ts/freq) % 64 == 0`;
   - every eligible origin included;
   - a stable source-based `series_id`. **(rev5)** Today's ids aren't
     stable: collision suffixes follow sort order (`builder.py:361-370,
     471-480`), and `content_hash` follows the values (`:295`). Proposed
     `series_id = <source>:<source-native key>[:<channel>]`, with no suffixes
     and in-source collisions refused. W6a uses that join rule from the
     start.

   Affects §1.3, W6a.
8. **MASE scale.** The box has only targets; per-model contexts would also
   change the scale. Proposed: a per-window MASE scale in `pool_d_tgt` from a
   fixed history slice. Affects §1.7, W6c.
9. **Target revisions and interpolation.** Proposed:
   - targets come from the first complete snapshot;
   - interpolated or missing target points are masked;
   - per-channel `end_ts`.

   Affects §1.3, W6b.
10. **Gate context.** The runners use the pinned 4096 wrapper. Proposed:
    either "gate always at 4096", or runners honour the declared context.
    Affects §1.3b, W3b2.
11. **Transfer host** for publishing kings, the vault mirror and the HF
    mirror (credentials can't be on the box or pods). Proposed: the builder
    VM with scoped write tokens. Affects §10.2, §10.6, W13.
12. **Loser audit trail.** Proposed:
    - per-model quantile sidecars in the private bucket;
    - a signed per-snapshot metadata digest published at build time.

    Affects §1.5, W7a.
13. **Testnet overlay is inert:** `owner_hotkey = ""`
    (`chain.testnet.toml:903`, verified). It's handled as a setup task in
    R14, with no design change, but noted here.
14. **(rev5) Moot under D45 (no shadow).** ~~Shadow semantics.~~ Proposed: S6a crowns are provisional and
    unpublished, and `lane_d_state` resets at the S6b `effective_block` (or
    say explicitly that they carry). Affects §5 S6a.
15. **Invariant 8 may not hold over time.** Later `pool_d_hist` snapshots
    contain earlier windows' targets as history. Under forward-only rules
    target secrecy after T protects nothing. Proposed: restate D18's
    rationale, or cut `pool_d_hist` so it never covers an open frozen window.
    Affects D18, §1.7.
16. **BF16 needs a lane-D-only guard path.** Today's guard requires the
    pinned F32 spec (verified: `checkpoint_guard.py:323-328`). This is
    implementation, recorded in W4; no design change.
17. **Minor design gaps:**
    - the house's declared context (4096);
    - what replaces "gate vs house" when the house is the challenger;
    - whether a house refresh replaces a pending house entry;
    - T for a payload revealed twice (first reveal, recorded in state);
    - `pool_d` snapshot retention;
    - bench-pod disk budget for 3 pinned checkpoints.
18. **(rev5) Withdrawn: wrong premise.** It claimed the lane A king's leg is
    retrained every era. It isn't: the king leg trains **once per seed epoch**
    (from one warm start to the next) and is carried across that epoch's eras
    (`trainer/rolling.py:125-128`, `shared/era.py:440-457`). The real issue,
    the house's T resetting on every KING_A change, is now owner decision
    **D46** (§6).
19. **(rev5) ADOPTED by the owner as D43 (§1.2 step 4, W10).** *Lane D
    reveal timing.* The standard timelock helper lands a
    reveal at `boundary − reveal_margin_blocks` (25) (`shared/chain.py:89-100`;
    copy-contest protection for lane A pointers). For lane D the weights stay
    private in the vault and exact-digest dedup applies, so revealing at once
    (`min_blocks`) moves T, and the forward clock, earlier by up to one
    settlement (3 h). Alternatively, take T as the commit-inclusion block,
    which binds the digest at commit, if poll_commitments can read it.
    Affects §1.2 step 4, §1.3.
20. **(rev5) Lanes A and B aren't forward-only.** Their pool snapshots "may
    include observations that existed before submission"
    (`docs/EVAL_POOL.md:22-24`). A generator can embed recent public series
    from the same feeds. Proposed: a **monitoring metric** (never gating, and not a rollout mode, so D45 isn't affected), once W6 timestamps
    exist, comparing each new lane A king's paired LCB on windows whose
    targets precede the generator's reveal against windows after it. A large
    gap is a memorisation flag, logged and never gating.
21. **(rev5) Retire `LANE_D_PROGRESS.md` and `LANE_D_S1_RESULTS.md`.**
    Appendices A and B duplicate them, and the progress file still has the
    old OA order and OA8 text. Keep one copy (the plan).
    **(rev6) Applied:** `LANE_D_PROGRESS.md` and Appendix A are retired in
    favour of the §17 status board; the owner inputs moved to §17.4;
    `LANE_D_S1_RESULTS.md` is now a pointer to Appendix B (its copy still
    said the forward re-check "happens at the S6a shadow").
22. **(rev6) §15 rows with no recommendation.** D25, D26, D27, D29 and D31
    give Option A (keep v2) and Option B, with no recommended option, so "all
    other recs" doesn't strictly settle them. §17.4 records them as **decided:
    Option A (keep v2, the design the WPs build)**, so they don't block W8 or
    W5b. Proposed: the owner confirms Option A, or names the Option B rows to
    adopt; any Option B is a structural change to W8 (court pay) and goes in
    as a new WP row.
23. **(rev6.1) Code-enforced stake gate for activation.** Today only the
    runbook enforces "observed `cfg-` stake ≥ 2/3 before `effective_block`"
    (§5 row 11, X6a, D10). `ALLOWED["lane_d"] = None` lets an overlay
    adopted at the 51% permit-stake lock-in set `[lane_d] from_block`, and
    old-release nodes refuse that overlay whole (§14.1), so arming early
    splits the fleet (RV1 note 10). Proposed: R3 (lint) or R7 (publish
    wrapper) refuses to publish an overlay that sets `[lane_d] from_block`
    unless R4's tracker output shows ≥ 2/3 observed `cfg-` stake.
    Affects R3, R4, R7, §5 S6.
24. **(rev6.1) `[lane_c] remainder` restatement.** It is now release-only
    (W2, `netconfig.py:213-215`). An overlay that merely restates
    `[lane_c] remainder = "burn"` would be refused whole by new nodes and
    accepted by old ones in a mixed fleet (RV1 note 12). R3 already refuses
    it; proposed: say so in §14.2 and in the operator runbook, so no
    overlay body ever carries it. Affects §14.2, R3, `docs/OPERATOR.md`.
25. **(rev6.1) Lane A tuning after lane D activation.** R3's lint, as its
    row is written, applies to any overlay containing a `[lane_d]` table,
    and overlays are full snapshots. After S6 every overlay carries
    `[lane_d]`, so any lane A key change is refused: lane A can no longer be
    tuned by overlay. Options: (a) accept it (lane A changes go through a
    release); (b) allow lane A keys to change in a seq that leaves every
    `[lane_d]` key byte-identical to seq N-1. Affects R3, §5, §14.2.
26. **(rev6.1) Stake-and-satellite gate for arming the §16 sentinel
    (W-S1).** `ALLOWED["v2"]` now admits the four `sentinel_*` keys
    (`netconfig.py:111-112`). A pre-W-S1 node refuses an overlay that
    carries them, and refuses it whole, so arming by overlay before the
    fleet upgrades splits the fleet. A pre-W-S1 satellite still verifies an
    armed receipt's signature (it uses the received body, `receipt.py:1220`),
    but its replay re-runs the gate on all four suites, the detail
    mismatches, and every armed settlement with a gate or king check fails
    replay there (RV3 note 2). Proposed: the R7 publish wrapper refuses an
    overlay that sets `[v2] sentinel_from_block` (non-zero) unless R4's
    tracker output shows observed `cfg-` stake ≥ 2/3 **and** every known
    satellite reports the W-S1 release; §5 states the same gate on the
    sentinel arming step. Affects R7 (publish wrapper), §5, §16.

## 0. Summary

- **Goal.** Find out how far our fixed architecture can go when training is
  unconstrained.
  - Miners train **anywhere, on anything, with any recipe or compute**: from
    scratch, from a public cascade checkpoint, by distillation.
  - They submit **only weights**. We **only evaluate**.
- **Winning.** A challenger beats the current lane D king on **forward-only
  private windows**: forecast targets that first appeared in our data after
  both models were committed. Nothing either side trained on can contain them.
- **The house.** Our own lane A king is always entered (the "house"). Lane D
  pays only for weights better than our pipeline produces.
- **Direction check.** A rotating public-benchmark gate runs, plus one
  never-gated benchmark, so we can see whether we're heading the right way.
- **Context.** Miners may run their weights at up to **128k context**, a
  scorer-side setting (§1.1).
- **Later phases.** Lane E (test-time optimisation) and the lane A recipe menu
  are phases 2 and 3 (§7, §8). **(rev4)** Lane B ships as part of this plan **trainer-run with miner-chosen experts and suite targets** (§12b: WB10–WB21, TB10–TB22, track SB0–SB5). The miner-weights variant (§12) is parked.

| Lane | Miner submits | Who trains | Judged on | Rollout |
|---|---|---|---|---|
| A (live) | data generator | us (fixed recipe) | private pool + bench gate | live |
| B (built, off) | **(rev4)** code-only generator + optional declared expert rows + optional one public-suite target | us (trainer-run) | per-target gain + guards + anchor + §16 sentinel; contamination-gated | **(rev4)** lane B track SB0–SB5 (§5), parallel to lane D S2–S3 |
| C (built, off) | fine-tune recipe | us | per-target eval | off |
| **D (this plan)** | **weights, our architecture** | **the miner, anywhere** | **forward-only private windows vs KING_D** | **gated stages S0–S7 (§5)** |
| E (phase 2) | test-time procedure | none | private pool vs KING_A plain | after D |
| A menu (phase 3) | generator + declared recipe | us | lane A duel | after E |

This supersedes DEC-CA-0062's recipe design, in which we trained a declared
recipe with an Ephemeris teacher. A lane D miner who wants distillation calls
Ephemeris, or anything else, on their own account. Recorded as **DEC-CA-0070**.

---

## 1. Lane D design

### 1.1 What an entry is, and how context works

**Checkpoint = a standard cascade_pyramid checkpoint, unchanged format.**

- `model.py` and `forecast_wrapper.py` must be byte-identical to the repo's
  `pyramid_model.py` and `pyramid_wrapper.py`. This is checked by
  `verify_checkpoint_code`, `cascade/eval/checkpoint_guard.py:139-171`.
- `config.json` must equal the contract's `from_contract().to_dict()`
  (`:290-302`). Its `context_length` is **4096**, because
  `CascadeMoEConfig.from_contract` requires it (`pyramid_model.py:89-93`).
- `weights.safetensors` must be **BF16 only** (D14, rev3): about 2.5 GB.
  - It's scored in lane A's eval precision: upcast to fp32 at load, the
    same for every entry.
  - F32's 4.9 GB sits close to the 5 GiB presigned single-PUT limit, which
    is W3a's fallback upload path.
  - Today's guard requires the pinned F32 spec
    (`checkpoint_guard.py:323-328`), so W4 adds a lane-D-only dtype path and
    leaves lane A's guard unchanged.
- **No other file is allowed.** Today's guard only refuses extra `.py` files
  (`checkpoint_guard.py:151-153`), so W4 adds an exact-filename allowlist that
  also refuses `.pt`, `.bin` and pickles.
- NaN and Inf weights are refused by `verify_weights_finite` (`:175-196`).
- Nothing here changes `pyramid_model.py`, `pyramid_wrapper.py`,
  `base_arch_digest` or any lane A contract term.

**Context is a scorer-side setting, not a checkpoint property.**

- At `ready` the miner declares a context from `[lane_d] contexts`.
  ~~**Launch arms `{4096, 8192, 16384}` (rev3).** 32768, 65536 and 131072 are
  armed later by overlay at stage S8 (§5), through the same `contexts` key
  (D10).~~ **(rev6.6, owner 2026-11-23)** `contexts` lists every variant
  from launch: `{4096, 8192, 16384, 32768, 65536, 131072}`. A miner may
  declare any of them; a value not in `contexts` is refused. Each variant is
  **armed automatically** once both gates pass: (1) `min_long_series` (300
  series from ≥ 20 sources) can use it (W1/R16, measured after the forge gap
  fix), and (2) T7's wall time and memory for it pass on the eval host that
  scores lane D. No overlay is needed. **An entry that declared an unarmed
  variant is scored at ~~the highest armed variant~~** **(rev6.15) the
  highest armed variant below its declared context** until its own arms;
  **(rev6.15)** if no armed variant is below it, it is not scored that
  settlement (typed skip `context_below_armed`, it stays `pending`), and it
  is never run above its declared context. The
  main validator records the armed set and its evidence in the signed report.
  Contexts are never a typed block.
- **Architecture facts (rev3, verified in `pyramid_model.py`):**
  - `patch_size = 32` (`:32`). **(rev5) Training rows are 128 patches**
    (4096 points): `pyramid_trainer.py:360-362` passes `max_ctx_patches=128`
    to the batch source. `max_patches = 136` (`pyramid_model.py:36`) is not
    the training length. Training never reads it; it's read only by lane B's
    horizon-affinity decode (`expert_lane.py:365`). So **8k = 2×, 16k = 4×,
    32k = 8×, 128k = 32× the trained length**.
  - **(rev5) Masked span limit.** CPM masks contiguous spans of ≤ 16 patches
    (`cpm_c_max = 16`; `pyramid_trainer.py:402`, the 120m preset and
    chain.toml). **Horizons > 512 points (16 × 32) are outside anything the
    model was trained to forecast.** That bounds every future lane D horizon
    (below).
  - `_xpos` uses `scale_base = 512.0` (`:154`) and a fixed, non-persistent
    `zeta = (idx + 0.4) / 1.4` buffer (`:211`). The q·k scale between
    patches d apart is `zeta^(d/512)`.
  - For the lowest-zeta dimensions (zeta ≈ 0.286) that is about **4.4e-5
    at d = 4096 patches**.
  - 128k context is 4096 patches, **(rev5) 32× the trained 128-patch rows**. The
    long-range signal in those dimensions is almost entirely suppressed, so
    quality beyond 16k is unproven and is left to ~~S8's~~ T7 measurements
    **(rev6.6:** T7 gates each variant's arming. The comparison stays fair
    because king and challenger are scored on identical targets; training
    length and recipe are the miner's**)**.
- A new lane D scorer module (`cascade/eval/lane_d_scorer.py`, outside the
  pinned wrapper) builds `CascadeMoEConfig` directly with that context and
  loads the same `state_dict`.
- **Why the weights fit any context:** the positional buffers (`inv_freq`,
  `zeta`) are non-persistent and sized by `head_dim`, and the route bias is
  sized by E (`pyramid_model.py:210-211, 259`). W0b pins this with a CI test:
  a 4096 checkpoint loads `strict=True` into every context variant.
- **No padding.** The pinned wrapper edge-pads short histories to the full
  context with `row[0]` (`pyramid_wrapper.py:35-38`). The lane D scorer
  instead **truncates each window's history to the largest patch multiple it
  really has** (≤ the declared context) and batches windows of equal length.
  - A long-context model gains only where long history exists.
  - Cost grows with real history.
  - A 4k model and a declared-128k copy of it score identically on windows
    with ≤ 4096 history (T7).
- **Precision:** scored in fp32, as the existing wrapper does
  (`pyramid_wrapper.py:51`). The context risk T7 measures is **extrapolation
  quality and memory**, not numeric overflow.
- **Decode cost (rev3, verified).**
  - The pyramid wrapper appends `ceil(h/32)` masked patches and runs **one
    forward pass** (`pyramid_wrapper.py:41-58`).
  - No KV cache exists anywhere in `cascade/`.
  - `STABLE_DECODE_STEPS = 768` is the **toto2** wrapper's constant
    (`toto2_trainer.py:1367`), not the pyramid's.
  - At h = 64 a window is therefore one pass over `context/32 + 2` patches.
    Cost is **roughly quadratic in context** for that single pass (time
    attention), and there's no per-horizon-step multiplier.
- **The comparison:** both king and challenger are scored by the lane D
  scorer, each at its declared context, on identical forecast targets. There
  is **one** lane D board across all contexts.

**Horizon:** lane D scores ~~**h = 64 only** at launch~~ **(rev6.9)** forward-only windows at **h = 64** for fast cadences (unchanged unless changed later); rev6.9 adds the owner's slow-cadence bins (daily short forward-only at h 7–30; daily medium/long and weekly-to-yearly as capped backtest windows, W5i, ~~off until `backtest_weight_cap` is armed~~ **(rev6.15)** on at launch, `backtest_weight_cap` 0.05; 0 turns them off) (A3: already in v2; rev3 makes §1.7 and the floors consistent with it). Lane A's 64/256/720
ladder would push forward-only latency on hourly data to about 30 days at 720
(§1.3). Longer rungs are a later typed change. **(rev5) Hard limit on that
change:** any lane D horizon must be ≤ 512 points (CPM `cpm_c_max = 16`
patches × 32). Longer horizons are outside training, so 720 is refused
unless the training contract changes first.

### 1.2 Identity, submission, storage

**Hotkey and commitment**

- A hotkey holds one commitment at a time. A lane D commit would replace that
  hotkey's lane A pointer, the same rule as lanes B and C
  (`lane_c_manifest.py:199-203`). **Lane D uses a dedicated hotkey.** It never
  consumes a lane A lifetime submission.
- Lane D reads reveals with `poll_commitments(include_history=True)`
  (`chain.py:495-535`).
- The commit grammar is `metro-v1:weights:c<context>:hippius:<vault-ref>`.

**Ownership and dedup.** All of these run before any evaluation spend.

- **Uploader must be the committer** (the `DigestOwned` rule,
  `cascade/funding/store.py:61, 340-367`). Another hotkey committing your digest
  is dropped.
- **Exact-digest dedup.** Refuse any entry whose weights sha256 equals:
  - a published checkpoint (every lane A leg or king on the registry, every
    lane D king),
  - the current house,
  - any open entry.

  This closes the front-run in which a freshly crowned, published lane A king
  is resubmitted into lane D (§2 item 1).
- **Near-copies** are handled by the margin. A near-copy of a king scores
  about the king, LCB ≈ 0, below `d_margin`.

**Rate limits (sybil and griefing)**

**(rev3) Final rule set, counted per lane: lane D has its own keys.**
Precedents are `[marketcast] max_open_legs_per_coldkey`
(`cascade/shared/config/marketcast.py:315, 671`) and
`[round] ops_max_active_per_coldkey` (`cascade/shared/config/rounds.py:455`).

"Open" means ready and not yet judged; an open entry can span eras. "Per
settlement" is a separate, judging-time rule.

| Rule | Key | Default | Status |
|---|---|---|---|
| One open entry per hotkey | (structural) | 1 | kept from v2 |
| Open entries per coldkey | `max_open_per_coldkey` | 2 | kept; lane D's own key |
| Judged per coldkey per settlement | `max_judged_per_coldkey_per_settlement` | 2 | **new (rev3)** |
| Judged per settlement (all) | `max_entries_per_settlement` | 8 | kept |
| Per-era count | — | — | none existed in v2; none added (the per-settlement key replaces the idea) |

- A `pending` entry holds its open slot.
- Exact-dup refusal is free (no eval).
- k for §1.3 = 32 per era.
- **Coldkeys are free (§9):** these caps limit casual flooding, not a
  determined Sybil. Registration cost and an eval fee (D4) are the remaining
  levers.

**(rev3) Enforcement**

- **The intake** `/v1/lane-d/ready` refuses over-cap requests with a typed
  code (`coldkey_cap_open`). This is **advisory**.
- **The validator's queue selection is the authority.**
  - It walks ready entries FIFO by T.
  - An entry over `max_judged_per_coldkey_per_settlement` or the global cap
    is **skipped with `coldkey_cap_settlement`**, carried to the next
    settlement, and not charged.
  - The judged set and every skip reason are structured report fields.
    `check_lane_d` re-derives the judged set.
- **Visibility:** per-entry status in `status/lane_d.json`, and a LANE_D.md
  FAQ entry. Test row T6b.

**(rev3) Coldkey resolution: [DECIDE D22]**

- **What satellites can query (verified).** `chain.py` can read the
  metagraph at a past block (`_metagraph(block=…)` `:361-371`,
  `uid_for_hotkey_at_block` `:454`). But past state needs an archive node,
  and the activation design deliberately needs none
  (`cascade/shared/activation.py:27, 344`). Public endpoints prune: a
  past-block read was "unreadable at entry (pruned; skipped per #304)" about
  2 h after its boundary (**(rev5)** OPSLOG entry "2026-09-23 11:54–12:10 UTC — ROUND 7384490546013170946 SCORED"; the old line ref `:2329` drifted to `:2338`, because the log is newest-first). So satellites **cannot** be assumed
  to read the metagraph at T.
- **Option a:** resolve the coldkey at T via archive state. Every validator
  then needs archive access.
- **Option b, recommended:** record the coldkey early and carry it in the
  signed report.
  - The intake records it at `/v1/lane-d/ready` time. That's earlier than T,
    and always inside the unpruned window.
  - The validator re-checks it at reveal, still inside the window. A
    mismatch is a typed refusal, `coldkey_changed`.
  - Satellites and the audit replay the signed value, at the same trust
    level as the rows. An archive check is optional for the audit.
  - Reasons: it works on pruned nodes, satellites need no new capability,
    and it follows §14.3 (satellites verify and do arithmetic only).
- **An unresolvable coldkey is never a silent pending.**
  - The entry gets the typed reason `coldkey_unresolved` in state and
    report, plus a cascade-ops alert (`lane_d_coldkey_unresolved`).
  - After `max_coldkey_unresolved_settlements` (recommended 4) it closes
    unjudged (`closed_unjudged: coldkey_unresolved`), and that's logged.

**Storage: the vault for lane D is new.**

- Today's vault is a **local 0700 directory on the box**
  (`store.py:258-275`), not a bucket, and can't hold 2.5 GB entries (BF16, rev3). Lane D
  adds a **dedicated private bucket** `cascade-lane-d-vault`. It is never the
  shared manifest bucket: private code once leaked through a shared-bucket
  archive tar (§13).
- **W3a prerequisite: what Hippius S3 offers** (docs checked 2026-11-21:
  docs.hippius.com/storage/s3/compatibility and /use/console/s3; live probe
  still needed):
  - **No STS and no IAM.** Short-lived per-entry credentials don't exist.
  - **Sub tokens** are bucket-scoped (not prefix-scoped), read-only or
    read-write, with custom lifespan and immediate revocation. They're created
    in the console, so they aren't per-entry. They suit the
    `pool_d` builder. The eval pod gets **presigned GETs only**, never a
    credential.
  - **Presigned GET and PUT are supported** (max 7 days). Presigned
    **UploadPart is not documented**. Multipart itself is supported (≤ 512 MiB
    per part, ≤ 10,000 parts).
  - ~~**Object Lock COMPLIANCE is supported.** Lock headers must go on
    CreateMultipartUpload; they're not supported on Complete.~~
    **(rev6.4, owner finding 2026-11-23) Hippius has no Object Lock in
    practice:** `GetObjectLockConfiguration` and `GetBucketVersioning` return
    403 for a full-access bucket-scoped sub-token and for the main key, and
    the console offers no lock permission. W3a re-runs its probe with
    `--no-object-lock` (lock probes recorded as `skipped`).
  - **Lifecycle rules are accepted but not applied.** TTL deletion must be our
    own job (D16).
  - **Bucket policies** use AWS policy language, but with no IAM it's unclear
    whether a policy can scope by prefix. The probe tests it.
  - **Consequence.** The miner path is **presigned PUT per part** if the probe
    shows presigned UploadPart works. Otherwise the fallbacks are a single
    presigned PUT (BF16 weights at ~2.5 GB, if the single-PUT size limit
    allows) or a small upload relay off the box.
  - **(rev5) Immutability: lock only kings.** COMPLIANCE retention is fixed
    when the object is created (the lock headers go on
    CreateMultipartUpload) and **can't be shortened**, so locking every entry
    would block loser deletion (`loser_ttl_eras`, D16). **Pending entries**
    rely on content addressing (the path includes the manifest digest) plus
    the ingest digest re-check, so an overwrite fails `digest_mismatch`. At
    **crowning**, the king is server-side copied into a locked prefix. W3a
    also probes `UploadPartCopy` for ~2.5 GB objects, because a single
    CopyObject may cap below that.
    **(rev6.4) Superseded for kings: there is no Object Lock**, so lane D
    kings are **not** locked by a COMPLIANCE copy at crowning. A king's
    immutability rests on the content-addressed path (manifest digest in the
    object path), the ingest re-hash on every read (an overwrite fails
    `digest_mismatch`), the signed report's digest, and the R2/HF mirrors
    (§11.1). Pending entries are unchanged (content addressing + digest
    re-check).
  - If prefix-scoped, short-lived credentials exist, use them.
  - Otherwise use **presigned multipart PUT URLs** minted per part with short
    expiry. Presigned URLs can't be revoked before expiry, so integrity comes
    from content addressing instead: the object path includes the manifest
    digest, ingest re-hashes, and any later overwrite fails `digest_mismatch`
    (the miner's fault).
- **Credentials and URLs are never logged, printed or put on a command line,**
  and never placed on pods. Pods get presigned GETs only.

**Flow** (a new **`cascade/lane_d/` service**, `cascade-laned-intake`):

- It reuses the `/v1/submit` identity, signature, registration and quota
  *rules* by import of the shared signing helpers.
- It does **not** live under `cascade/funding/`. `deploy_gate.py:55` maps
  `cascade/funding/` to the **trainer** service, so touching it would force a
  trainer restart.

1. **`POST /v1/lane-d/submit`**
   - Inputs: signed headers, `X-Content-Digest` (manifest sha256) and
     `X-Context-Length`.
   - Returns upload grants for `lane-d/<hotkey>/<digest>/`.
2. The miner's CLI does a **direct multipart upload**. Nothing transits the
   box (130 MB Caddy cap, 7 GB RAM).
3. **`POST /v1/lane-d/ready`**, signed.
   - Checks the object list and sizes against the manifest. It never
     downloads the weights.
   - Records `ready_block` and returns `commit_payload`.
4. **The miner commits** through a timelocked reveal. The **reveal block is
   T**, the entry's commit time. Weights are immutable from here.
   - **(rev5, owner-adopted, D43) Immediate reveal.** The lane D CLI sets
     `blocks_until_reveal = min_blocks` (reveal now). It does **not** use the
     lane A timing, which lands the reveal at `next boundary −
     reveal_margin_blocks` (25) (`shared/chain.py:89-100`) to stop copiers of
     public generator pointers.
   - **Why that's safe for lane D:** the commitment reveals only the vault
     ref and digest; the weights stay private in the vault; exact-digest dedup
     refuses resubmissions; and `DigestOwned` drops a digest committed by a
     hotkey other than its uploader.
   - **Effect:** T, and with it the forward-data clock (§1.3), starts up to
     one settlement (≈ 3 h, 900 blocks) earlier.
   - The validator doesn't depend on the timing: it takes T from the observed
     reveal block either way. A miner who uses a delayed reveal only delays
     their own clock.
5. **Prefetch.** On reveal, **(rev5)** the launch eval host (D11: valpriv-4090-class pinned-torch pod) prefetches the entry into a
   sha-keyed, resumable cache, using presigned GETs. Hippius outages and slow
   transfers are the history (§13).

**Visibility and retention**

- **Losers** stay private:
  - removed from the vault `loser_ttl_eras` (recommended 4) after judgement;
  - then held in an **owner-only archive** for the dispute window
    (`retention_days`, 14);
  - then deleted.

  Automated deletion is an Approve-tier destructive operation, so it needs
  the owner's **standing approval (D16)**.
- **Kings** are published (§11.1).

### 1.3 Judging: forward-only windows

**The rule.** For an entry E with reveal time T_E, against king K with
commit time T_K:

> **(rev3)** A window is eligible only if, for **every** target point,
> **both** its **data timestamp** **and** its **first-seen time** (when our
> mirror first ingested it) are after **wall(max(T_E, T_K)) +
> `forward_margin`**. Its feed's measured mirror lag must also be within
> `[lane_d] max_mirror_lag_hours`; a feed above the limit is **never
> eligible**. A series without both timestamps is never eligible.

> **(rev6.10, owner 2026-11-23) Embargo-aware generalisation.** With
> `b = wall(max(T_E, T_K)) + forward_margin` and `e = embargo_hours(feed)`
> (effective: `min(catalog, [lane_d] embargo_hours)`, pinned in the
> snapshot's `feeds` entry; **(rev6.11)** ~~**0 for a sub-hourly series**,~~
> **(rev6.12)** the same rule for every cadence, and
> embargo windows ≤ ~~`embargo_window_share_cap`~~ **(rev6.14, rev6.12 leftover)**
> `1 − min_fresh_window_share` of a frozen set), every target point p needs
> **`release(p) = first_seen(p) + e > b`** and **`data_ts(p) > b − e`**.
> That's the rule above with the bound moved per feed from `b` to `b − e`.
> S0 and the backfill rule use `b − e` too. **e = 0 ⇒ identical to the rev3
> rule.** `[lane_d] embargo_hours` = 48 (lane A's ~48 h reveal delay), and
> `forward_margin_hours` = ~~6~~ **(rev6.14)** 0, so `b = T`: data after
> submission is the fresh target. No set is frozen or judged before ~~`b`~~
> **(rev6.14)** `j = wall(max(T_E, T_K)) + eval_delay_hours` (6 h, the
> owner's eval delay; `b ≤ j` enforced; rev6.14 (a)). ~~The
> **confirmation duel's second set uses the embargo-0 rule** (unchanged).~~
> **(rev6.11)** No confirmation duel: the crown is decided at the first verdict.
> Mirror-lag, forecast-product and missing-clock refusals are unchanged.
> Full spec and per-cadence latency: rev6.10 (a).

- **(rev3) First-seen doesn't exist today.** Verified: `grep -rnE
  "first_seen|ingested_at|ingest_time|landed_at|LastModified" cascade/pool/`
  finds nothing. Recording first-seen is the **launch-blocking WP W6fs**
  (§3).
- **(rev3) Both clocks, because each one alone admits pre-T data:**
  - data time alone admits values that were public before T but carry a
    future timestamp (publish-ahead feeds, day-ahead prices, schedules,
    forecast products such as `openmeteo`);
  - first-seen alone admits values that were public before T but reached our
    mirror late (outages, backfills, series absent from a snapshot).
- **(rev3) The v2 rule "absent from the first snapshot after max(T) + margin"
  is UNSAFE as a substitute.** It's a first-seen proxy only. It counts a
  series missing from that snapshot as "absent", so pre-T points pass. It
  must not be used for verdicts. It survives only as the **upper bound** B2
  reports ("snapshot-diff bound").

Why it's built this way:

- **Pool records carry no timestamps today.** Pool `metadata.json` holds
  `{freq, seasonal_period, domain, source, …}` (`builder.py:279-290`). The
  forge source parses timestamps and then discards them
  (`sources/tsbench_forge.py:380-399`). Some feeds have none.
- So **W6 stamps per-series `end_ts` into `pool_d` metadata**, and a series
  without timestamps is never eligible. **(rev6.1)** Likewise a series
  without `frontier`, or a feed without a `feeds` entry, is never eligible:
  W6b must write both, and W9 refuses a snapshot that lacks them. There is no build-time fallback: a
  series' tail can be up to `max_stale_days = 4` old, and
  `_freshest_segment` can fall back to an old segment (`:149, :204,
  :446-460`).
- **(rev3) Superseded by the two-clock rule above.** The v2 claim that
  publish-ahead feeds are "excluded automatically" was false: the forge
  reader drops rows with `ts ≥ as_of + 1 day` (`tsbench_forge.py:393-396`),
  so those values never appeared in the snapshot. `forward_margin`
  (~~recommended 6 h~~ **(rev6.14)** default 0; the 6 h wait is
  `eval_delay_hours`) is now applied to **both** clocks.
  `max_mirror_lag_hours` (recommended 6) bounds how far our mirror may trail
  the public source per feed. It's measured by W6fs from the source's
  publish stamp, where the feed provides one, against the first-seen time.
- **Both T values count.** max(T_E, T_K) matters: a king committed after a
  pending challenger could have trained on that challenger's targets.
- **The cost is about one extra day of latency.** W1 measures it.
- **(rev6.10) Why the embargo is safe to count.** We release a point to
  miners no earlier than `first_seen + embargo_hours` (lane A reveals
  retired pool snapshots about 48 h later, `cascade/pool/reveal.py:17-24`;
  ~~W6rv makes that a guarantee~~ **(rev6.11)** W6rv isn't needed; lane A's
  cadence must not increase without it). So a point released after `b` was unseen at
  commit through us, even when its data time is up to `e` before it. The
  residual risk is a feed that's public at the source in real time. That's
  owner-accepted, and mitigated by the per-feed override (`e = 0`) ~~and the
  embargo-0 confirmation duel~~ **(rev6.11)** (the duel is dropped; see the
  rev6.11 (b) flag).

**Window identity.** Each window has a stable id
~~`<series_id>:<target_start_ts>:h64`~~ **(rev6.9)**
`<series_id>:<target_start_ts>:h<h>` (h = 64 for fast cadences; the drawn h
for a daily-short forward window), and `<series_id>:<target_start_ts>:h<h>:bt`
for a W5i backtest window. Lane A's `h<h>-s<pool_index>` ids aren't
stable across snapshots (`windows.py:433-449`).

**Evidence accumulates, then freezes.**

- An entry's eligible set is the union over every ~~daily~~ **(rev6.11)** 3-hourly `pool_d` snapshot
  ~~since T~~ **(rev6.10)** that can hold an eligible window (built after
  `b − max e`, so embargoed snapshots from before T count).
- It **freezes** at the first settlement where it meets the floors:
  `min_windows` (1000) and `min_clusters` (20 sources). **(rev6.10)** The
  floors are checked on that settlement's **seeded per-bin draw** over the
  eligible set (rev6.10 (b), W5j), never before ~~`b`~~ **(rev6.14)** `j`
  (`T + eval_delay_hours`). The frozen set **is**
  the draw at the freeze settlement. A pending entry is re-drawn afresh each
  settlement and isn't forecast until it freezes.
- It is **judged once**, on its cohort's intersection set (§1.3 Cohorts), which is ⊆ its frozen set. A loser is never re-judged on new
  windows: rotation alone swung one pair's LCB by 0.62 (§13).
- Until then the entry is `pending`.

**Cohorts (several entries in one settlement).** `cohort_maxt_lcbs` assumes
every challenger shares one window draw and the king's scores
(`bootstrap.py:393-418`). So:

1. The cohort is every entry that freezes this settlement.
2. The window set is the intersection eligible for max(T over cohort ∪ king).
   **(rev6.10)** It's the one seeded per-bin draw (W5j) over that
   intersection, identical for the king and every cohort member, capped by
   `max_windows_per_settlement`. A deferral (step 3) re-draws over the
   reduced cohort with the same seed.
3. If the intersection misses the floors, the latest-T entries are deferred to
   the next settlement until it meets them.
4. The cohort is then judged with max-T on that shared set.

**Verdict.**

- The statistic is a paired cluster bootstrap of the CRPS/MASE geomean ratio
  (`paired_bootstrap_lcb_aggregated`, `bootstrap.py:225`), clusters = source.
- The challenger wins if `LCB_α ≥ d_margin` and the point gain is strictly
  positive.
- `d_margin` is **lane D's own level-unit bar** (recommended 0.01). It is not
  lane A's v2 increment bar, which runs 0.10 → 0.01 with tenure decay.
- ~~**(rev3) Mandatory confirmation duel.** A candidate that clears must
  **clear again on the next disjoint eligible window set** (the next freeze
  of new forward windows, sharing no window with the first) before it's
  crowned. **(rev6.10) Unchanged by the embargo:** the second set must
  satisfy the **embargo-0 rule** (`e = 0` for every feed, bound `b`), and it
  shares no target point with the first set. It's the W5j draw at its own
  freeze settlement. This holds unless the owner decides otherwise.~~
  - ~~It costs one more freeze period of latency: **about 1–4 days depending
    on data frequency** (5-min-heavy sets at the low end, hourly at the high
    end; B2 refines it).~~
  - ~~While confirming, the entry's state is `confirm_pending`.~~
- **(rev6.11)** **The confirmation duel is DROPPED (owner 2026-11-23).** A
  crown is decided at the **first verdict**: the max-t LCB against KING_D
  over the cohort, on the W5j per-bin draw of up to 5000 windows, plus A10
  (W3b5) and the §1.3b gate. No `confirm_pending` state, no second set. See
  rev6.11 (b).
- ~~**[DECIDE D20] (rev3): a king change between first clear and
  confirmation** (house refresh or KING_D change):~~
  - ~~(a) **restart the duel against the new king** (first clear discarded;
    the entry re-freezes against the new king; **recommended**);~~
  - ~~(b) confirm against the king that held at the first clear.~~

  ~~The chosen rule is the `[lane_d] confirm_on_king_change` key (`"restart"`
  default | `"original"`). It's written into `lane_d_state`
  (`confirm.king_digest`, `confirm.rule`), the report schema (`confirmation`
  block) and tests (T9b).~~ **(rev6.11)** D20 is moot: there's no
  confirmation. `confirm_on_king_change` stays a legacy key, parsed and
  ignored (`chain.toml:1661` carries it).
- **(rev3) Repeated attempts.**
  - Cohort max-T corrects only **within one settlement**. Near-copies on
    separate hotkeys get independent window sets, so an adversary gets **k
    independent attempts per era**.
  - The per-era false-crown probability ≈ **1 − (1 − ~~p_conf~~ p)^k**, where
    ~~p_conf is the confirmation-duel null FPR~~ **(rev6.11)** p is the
    **single-test** null FPR (no duel). This assumes independent
    attempts; it's an approximation, conservative when attempts share
    windows.
  - **k at the A8 caps:** coldkeys are free, so the binding cap is global:
    `max_entries_per_settlement` (8) × `era_settlements` (4) = **k = 32
    judged attempts per era**. A single coldkey gets at most
    `max_judged_per_coldkey_per_settlement` (2) × 4 = 8 per era, and fewer in
    practice: `max_open_per_coldkey` and the freeze latency bound it.
- **(rev3) `d_margin` is calibrated to the per-era target, not a per-entry
  quantile.** Choose the margin so that 1 − (1 − ~~p_conf~~ p)^k ≤ the owner's
  per-era target (OA3; recommended 5%). **(rev6.11)** Without the duel, that
  needs p ≤ ≈ 0.0016 at k = 32 and 5%. B1 must re-check `alpha` and
  `d_margin` on the single test (rev6.11 (b)).
  - A per-entry null p99 alone (p ≈ 1%) would give about 27% false crowns
    per era at k = 32, which is too loose.
  - T14 is redefined (§4): the null FPR per entry for each null-pair type,
    plus the per-era FPR at the caps. It's measured at S1 (B1,
    `scripts/null_fpr.py`) and is an **S4 gate** against the owner's per-era
    target.
  - B1 reports the margin needed for per-era targets of 1%, 5% and 10%. The
    `d_margin` default stays 0.01 until B1 runs.

### 1.3b Rotating external-benchmark gate (direction check)

- **Gate suites:** **(rev5)** the four suites (GIFT-Eval, BOOM, TIME, FEV)
  **minus this era's §16 sentinel**, run with the existing runners
  (`cascade/eval/benchmarks.py`, `public_guards.py`). TIME is no longer a
  fixed held-out suite.
- **The draw:** one suite per crowning, drawn from the **settlement boundary's
  block hash**. Every validator agrees on it, and no miner can know it at
  commit. Weights start equal.
- **Rule on the drawn suite.** It runs paired on one host in one sweep, never
  against stored absolute numbers, because driver and SKU shift absolute
  scores. CRPS and MASE must each satisfy `LCB_α ≥ −tau_public` (0.01) against
  both:
  1. the current KING_D;
  2. **the current house**: our own, uncontaminated pipeline.

  There's no "best-ever miner king" ratchet. A contaminated king would set an
  inflated best-ever and block honest successors. The house is the anchor.
- **(rev3) Held-out suite = the cross-lane rotating sentinel (§16).** The
  v2 fixed "TIME unselected" is replaced.
  - Each era one suite is the held-out **sentinel** for all lanes (A, B and
    D).
  - Lane D's drawn gate suite comes from the remaining suites.
  - The sentinel is swept for each new KING_D and published only.
  - If the sentinel falls for 3 kings in a row, that's an alert and an owner
    review.
- **Faults:** a candidate whose sweep errors, returns partial, NaN or a
  missing metric is a **`checkpoint_fault`, the miner's fault** (T17).
  - Suite identity is verified per drawn suite. Scope-skips are not
    "partial" (§13).
  - The house and KING_D are re-swept in the same host session as the candidate (cached rows are re-used only within that session).
- **(rev6.18) REMOVED (owner 2026-11-23: "The low-frequency check A10 is removed"); the text below is kept struck.** ~~**(rev3) Always-run low-frequency check (A10).**~~ Alongside the drawn
  suite, every crowning candidate also runs a **paired non-inferiority
  check on a low-frequency slice of the private pool**.
  - The slice: `[lane_d] lowfreq_freqs`, recommended daily and slower.
    Forward-only rarely reaches these series (§1.7).
  - The test: CRPS and MASE `LCB_α ≥ −tau_public` against both KING_D and
    the house, on the same window set, scored by the lane D scorer.
  - On failure: `lowfreq_regression` (typed), and the candidate isn't
    crowned.
  - **Limit (§9):** the private pool isn't forward-only, so a contaminated
    model passes this check more easily. It **catches honest regressions
    only**.
- **Pay:** public suites never pay and are never targets.
- **Progress board:** every king's suite numbers (gate and **(rev5)** the era's sentinel, suite named) go into the
  signed lane D index (§11.1) and `status/lane_d.json`.

### 1.4 Kings, the house, pay

**KING_D**

- There's one KING_D at a time.
- The court is KING_D plus `reward_prior_kings` (2) prior **miner** kings,
  ~~decayed by `king_decay` (0.5) inside lane D's share~~ **(rev6.23)** paid
  by the explicit `court_weights` `[0.80, 0.15, 0.05]` inside lane D's share
  (owner 2026-11-24).
- The house is never in the court.

**The house (our lane A king)**

- **Identity:** the house is the checkpoint digest lane A currently treats as
  king. Under v2 that is the epoch's king leg (DEC-CA-0055).
- **Refresh:** it's refreshed whenever that digest changes (a crowning, a
  seed-epoch rollover, a promotion). It enters at that settlement, and its
  T is the block at which that digest was published.
- **Forfeit or king-check fail:** the house is whatever lane A treats as king.
  If lane A has none, the previous house stays.
- **Pay:** if the house holds KING_D, the lane D king slot goes unpaid.
  Prior miner kings keep their decaying tails, because those were real
  crowns. Where the unpaid slot goes is **[DECIDE D21] (rev3)**, a
  tokenomics call, not adopted here:

  | | Option 1: **burn** (v2 default) | Option 2: **return to lane A** while the house reigns |
  |---|---|---|
  | For | Matches §14.2, where `remainder` is a release-only burn destination. Lane A's share is fixed, so invariant 2 and satellite weight derivation are unchanged (**(rev6)** the share-0 shadow no longer exists, D45) | The house is lane A miners' work, so lane A's contributors are paid for it |
  | Against | Emission withheld whenever our pipeline is the best | §14.2 must change (a new release-only `house_remainder` destination). Lane A's share varies per settlement, which touches invariant 2 and satellite derivation (W7d), and adds test rows (**(rev6)** no shadow stage, D45) |
  | Dethroned house | Never enters the court (the court is miner kings only). No tail | Same: no tail. The slot moves to the new miner king at once |

  Under **both** options the house is never a prior king, and prior miner
  kings' tails continue unchanged.

**Share [DECIDE D1]**

- **(rev5, owner)** `[lane_d] share = 0.25`, `takes_from = "lane_a"`, `remainder = "burn"`.
- The deployed file has `takes_from = "lane_b"`. It's changed **by the S6
  activation overlay** **(rev6;** S6b merged into S6, D45**)** (§5, §14.2), not by
  a chain.toml edit.
- When B and C also arm, the split is **(rev5, owner) A 25 / B 25 / C 25 / D 25**.
- Live emission is `lane_parts` + `lane_pool_shares` + `emission_vector`
  under the DEC-CA-0055 ladder (`cascade/shared/emission.py:330-420`). The
  legacy `split_emissions` helper is not used. W8 extends `emission_vector`
  with lane D.

### 1.5 Consensus, trust and state

**Who scores**

- **The main validator (ours)** orchestrates lane D:
  - **(rev5)** the launch eval host (D11: valpriv-4090-class pinned-torch pod) computes forecasts (valeval-laned only from S8);
  - **(rev5) Scoring method = lane A's quantile head.** The lane D scorer
    reads the model's quantile head directly at the 9-level scoring grid, with
    **no copula sampling and no Monte-Carlo noise**, as
    lane A does under `[v2] quantile_head_after = true` (DEC-CA-0067,
    `chain.toml:1247`, `ChainConfig.quantile_head_at`). It never uses the
    sample path. So lane D CRPS is computed the same way as the house's lane A
    CRPS, and a house-vs-entry comparison has no method gap;
  - **targets and scoring stay on the box** (§10.2);
  - the main validator signs the **lane D report**: frozen window ids, per-window
    CRPS/MASE rows for every scored model, the bootstrap seed, the cohort,
    the drawn suite, verdicts and `pool_d` sha256.
  - **(rev3) The report also carries:**
    - `rows_sha256` (A9);
    - per window, both clocks and the feed's mirror lag (A1);
      **(rev6.10)** plus the effective `embargo_hours`, `min(release)`, the
      rule it passed (`embargo` | `embargo0`), and its draw record
      (`timeframe`, `bin`, `h`, origin), with the block hash the draw
      used;
    - the judged set and per-entry skip reasons, with `coldkey` as recorded
      at `ready` (A8);
    - ~~the `confirmation` block: stage, first-clear set id, king at the first
      clear, rule (A2, D20);~~ **(rev6.11)** removed (no confirmation duel);
    - ~~the low-frequency check result (A10).~~ **(rev6.18)** removed.
- **Satellites** hold no pool (DEC-CA-0058:17-33), and `pool_d` is private.
  They check the signature, the structural rules (declared timestamps
  against declared T values, floors, cohort construction), and they re-run
  the bootstrap from the signed rows, then set the same weights. They
  **cannot** re-derive targets.
- **The trust level is lower than lane A's bench numbers.** Lane A rows are
  reproducible from public checkpoints (`cascade-audit bench --reproduce`). A
  lane D loser's rows are not, because its weights stay private. Crowned
  kings are public, so their rows become re-scorable once a `pool_d` snapshot
  is revealed. **We accept this explicitly (D17).**
- **(rev3) Rows digest.** Every verdict publishes `rows_sha256`, the sha256
  of the exact scored per-window rows (canonical serialisation, both models,
  frozen set order). Anyone holding the rows can prove they're the ones the
  verdict used.
- **(rev3) Maximum reveal delay.** `[lane_d] max_reveal_delay_days`
  (recommended 14). An entry still pending at the limit closes unjudged
  (`closed_unjudged: reveal_delay_exceeded`), logged, and frees its slots.
- **(rev3) Loser re-scoring: [DECIDE D23].**
  - (i) Publish `pool_d_hist` after the dispute window, so losers can fully
    re-score their own weights against the published targets.
  - (ii) Soften the claim to "losers can verify targets, window ids,
    `rows_sha256` and the bootstrap; histories stay private". v2 keeps long
    history private on purpose (§1.7, lane A mimicry).

  The default is **(ii)**, matching v2. Choosing (i) also removes the
  private-history rationale in §1.7.
- **No free text in replay comparisons.** A transient fetch error is never
  read as "absent" (§13).

**State**

- A new `lane_d_state.json` holds KING_D, the court, the house digest and T,
  pending entries with their eligible-window progress, frozen sets, and a
  score cache keyed by (model digest, context, window id).
- **(rev3) It also holds, per entry:**
  - `coldkey_at_ready` and its re-check result (A8);
  - `ready_block`, `reveal_block` (T) and `reveal_deadline` (A9);
  - ~~`confirm` (first-clear set id, `king_digest`, `rule` per D20) (A2);~~ **(rev6.11)** removed; W7c migrates a testnet `confirm_pending`;
  - `reason`, a typed enum: `pending_floor`, ~~`confirm_pending`,~~
    `coldkey_cap_settlement`, `coldkey_unresolved`, `coldkey_changed`,
    `reveal_delay_exceeded`, `mirror_lag_exceeded`, `lowfreq_regression`,
    `checkpoint_fault`, `bench_hold`, **(rev5)** `deregistered` (B8), `ready_expired` (B10);
  - per feed, `measured_mirror_lag_hours` (A1).
- **(rev5) Deregistration (B8).** A pending, frozen ~~or `confirm_pending`~~ entry
  whose hotkey loses its UID is **voided**: typed `deregistered`, slot freed,
  never judged. A **court** member (KING_D or a prior king) that deregisters
  has its slot **burn** (precedent: lane B drops deregistered winners so the
  unpaid slot burns, `shared/lane_b.py:250-262`). It's checked at every
  settlement against the settlement's metagraph and recorded in the report.
- **(rev5) Ready → commit deadline (B10).** `[lane_d] ready_ttl_blocks`: an
  entry readied but not revealed within that many blocks of `ready_block`
  expires (typed `ready_expired`), its open slot is freed and its vault
  object becomes a loser for retention.
- Its digest is in the receipt's prior and posterior digests (DEC-CA-0058),
  and the report digest is bound into the receipt as the bench source is.
- `checks.derive_weights` and `check_lane_d` re-derive the lane D weights
  from the receipt chain.
- It's protected like `cascade_state.json`: snapshot before every restart,
  never hand-edited.

### 1.6 Capacity

| Resource | Load | Plan |
|---|---|---|
| **(rev6.18)** Lane D scoring host: any GPU running a qualified scoring environment (rev6.18 B5; RTX PRO 6000 Blackwell allowed). ~~**(rev3) Launch eval host:** valpriv-4090 or another pod running the pinned torch 2.4.1+cu124 venv~~ | prefetch on reveal; forecasts for up to 8 entries + KING_D + house per settlement on frozen sets, at ≤ 16k context; one forward pass per window, cost roughly quadratic in context (§1.1) | ~~**Not valeval-blackwell4**: sm_120 can't run the pinned torch (only the sidecar venv), so numerics would differ from testnet and the worker handshake fails.~~ **(rev6.18)** Superseded by B5/B6: one cohort is scored on one host, and the handshake checks the qualified environment. **S1 check (B2, OA4):** does a 16k-context batch fit in 24 GB on valpriv-4090? If not, name the smallest pinned-torch pod that fits. valeval-laned (96 GB) is rented **only at S8**. Measure in T7; arm only variants that fit ≤ 50% of a settlement |
| Box | target slicing, scoring and bootstrap for returned forecasts; report signing | Targets only: small, ~~h=64~~ **(rev6.9)** h = 64, or ≤ 180 points for W5i slow-cadence bins. **No histories, weights or torch on the box.** Bound the report size |
| valeval-blackwell4 (43 GB, ~19 GB cache) | per crowning: **2.5 GB (BF16, rev3)** upload + 1 drawn suite; per new king: **(rev5)** the era's sentinel suite | Byte-based LRU (#467); lane A first; second bench pod if crownings slip > 1 settlement (D6) |
| Weight loop | must never wait on lane D | Lane D runs off the poll thread; keep-alive (#471) live before S6 (T15). **(rev3) Status:** #471 is merged on main (0c52d07; `_weights_keepalive_tick` at `cascade/validator/weights.py:317`, wired into `validator_bench.py:901`) but **not deployed**: the live rc37 validator has no `_weights_keepalive_tick`. It ships at the latest with S2b |

### 1.7 Increasing the context of the cascade eval pool

**How it works today:**

- `publish-pool.yml` (daily 03:00 UTC) truncates each series to its freshest
  `context_length + keep_horizon` points: 4096 + 720, from
  `[eval] context_length` and the longest scored horizon.
- The validator cuts **one window per series per horizon rung** (the last h
  steps as target, up to `[eval] context_length` before it as history;
  `windows.py:556-611`).
- The pool snapshot has about 4.6k series (snapshot 9509400: 4,633).

**Lane A must not change.**

- **`pool` is untouched.** Same builder inputs, same bytes, same receipts,
  same reveal.
- A longer keep can't simply be applied to `pool`. The builder truncates
  **before** its degenerate-series check and dedup (`builder.py:264-271`), so
  a longer keep changes which series pass, and therefore lane A's windows.

**`pool_d` (D12)**

- **(rev3) Launch `pool_d` is still a separate tar.** The main pool's keep
  length is **never** raised: `builder.py:264-271` truncates to
  `keep_length` before the `too_short` and degenerate checks (and dedup runs
  later still), so a longer keep would change which series pass and
  therefore lane A's windows. The lane A byte-identity test stays exactly as
  strict.
- **(rev3) Launch construction:** the **same selected series ids as `pool`**,
  joined (not re-selected), with history extended to **16384 + 64**. It
  carries the A1 timestamps (data time and first-seen per target point) and
  the hist/tgt split (D18).
- **(rev3) Moved to stage S8** (~~armed by overlay through `contexts`, D10~~
  **(rev6.6:** no overlay; each variant arms automatically when its two gates
  pass, §1.1**)**): history to 131072 + 64, deep backfill, the new long sources (option 3
  below), and the 96 GB eval host. **(rev5) ~~Only scoring and arming wait
  for S8.~~ (rev6.6) Scoring at 32k–128k waits only for each variant's gates; in practice the history and the 96 GB host come from S8. Harvesting long history starts now** (below, WP R16).

**(rev5) Harvest now, score later (up to 128k).**

- **Why now.** Sub-hourly history that isn't recorded is gone. The forge
  mirror prunes sub-hourly partitions (Getting the history, item 2), and many
  source APIs serve only recent data. 128k points at 5 min is about 1.2
  years. If recording starts at S8, the 32k–128k variants can't arm for
  months after S8, because `min_long_series` (300 from ≥ 20 sources) can't
  be met.
- **What starts now (before lane D launches, from stage S1/S2):**
  1. stop pruning sub-hourly forge-mirror partitions for every source that
     feeds `pool` (an owner/builder retention change, not a code path
     validators run);
  2. start recording the deep-history live sources (item 3 below);
  3. the `pool_d` builder (R13) appends each selected series' new points to a
     **long-history archive**, keeping up to **131072 + 64** points per series
     and channel. It uses the W6a join key, carries the per-point first-seen
     record (W6fs) once that exists, and lives in a private bucket,
     sha-pinned per day.
- **What does not change:** `pool` and lane A (byte-identity test as
  strict); nothing is scored, sent to an eval host or revealed from the
  archive before its context is armed ~~at S8~~ **(rev6.6:** by its gates, §1.1**)**; nothing runs on the box.
- **Size bound (arithmetic, not measured):** 4,633 series × 131,136 points ×
  4 B (float32) is about **2.4 GB** for a full copy. Stored append-only, the
  daily delta is small. Most series never reach 128k (see the table below),
  so the real size is below this.
- **Readiness signal:** R16 reports, per context rung (32k/64k/128k), how many
  series and sources have reached it. ~~S8 arms a rung only when
  `min_long_series` is met.~~ **(rev6.6)** A rung arms only when
  `min_long_series` is met **and** T7 passes for it (§1.1); no S8 overlay.
- **Metadata:** per-series (per-channel) `end_ts`, `first_target_ts` and the
  per-point first-seen record (W6fs). **(rev6.1)** Plus `frontier` for every
  series and a `feeds` entry (`mirror_lag_hours`, `forecast_product`) for
  every feed; a `pool_d` missing either is invalid (W6b).
- **Artifacts:** two tars from one build.
  - `pool_d_hist`: histories only, **no targets**. It goes to **(rev5)** the eval host (the launch eval host; valeval-laned from S8).
  - `pool_d_tgt`: targets plus timestamps, small. It stays on the box.

  This keeps the existing privacy rule: targets and scoring stay local
  (`eval_offload.py:7-10`).
- **"One longer pool" is ruled out on RAM.** The box validator loads every
  series as float64 (`validator/pool.py:71-77`). 1,000 series × 131k × 8 B is
  about **1 GB**, on a 7 GB box.

**Getting the history** (cheapest first):

1. **Extend what the forge mirror already has** for the selected series.
2. **Stop pruning sub-hourly mirror partitions,** and backfill full history
   from source APIs where allowed.
3. **Add deep-history live sources:** grid load and generation at 5–15 min,
   weather stations hourly, traffic at 5 min, market bars at 1–5 min. Each
   carries a domain and a source key.

**What 128k means in wall-clock history:**

| Frequency | 128k context | Usable? |
|---|---|---|
| 1 min | ~91 days | yes |
| 5 min | ~1.2 years | yes |
| 15 min | ~3.7 years | yes, with backfill |
| hourly | ~15 years | rare archives only |
| daily or slower | centuries | no: those series keep what they have |

A context variant is armed only if at least `min_long_series` (300, from
≥ 20 sources) can use it (W1).

**(rev3) Floor maths at h = 64.** These are consistent with A3. A window's
target is ~~the last 64 steps, so for an entry committed at T the earliest
eligible window has its first target point after T + `forward_margin` by
both clocks~~ **(rev6.10)** 64 steps at a drawn origin (W5j). For an entry
committed at T, with `b = T + forward_margin`, the earliest eligible window
has its first target point after `b − embargo_hours(feed)` by both clocks,
and is first judged at the first settlement ≥ ~~`b`~~ **(rev6.14)** `j =
T + eval_delay_hours`, with `b = T` by default (rev6.14 (a)). ~~The confirmation duel
(§1.3) needs a second, disjoint set, so expect **two freeze periods** before
crowning. **(rev6.10)** The first is now at or near S1 for fast cadences at
e = 48, and the second uses the embargo-0 rule.~~ **(rev6.11)** One freeze
period: the crown is decided at the first verdict. ~~With 3-hourly pool_d
and sub-hourly e = 0, that's ≈ T+6–9 h for 10 s data and ≈ T+22–25 h for
15-min and hourly (rev6.11 (e) table).~~ **(rev6.12)** With 3-hourly
pool_d and e = 48 for every cadence, that's ≈ T+6–9 h for 10 s to 30-min
data and ≈ ~~T+22–25 h~~ **(rev6.14)** T+16–19 h for hourly, subject to the floors and the ~~share cap~~
**(rev6.14)** minimum fresh share (~~rev6.12 table~~ **(rev6.14)** rev6.14 table; F ≥ 500 fresh windows at S1 is pending W1). B2 measures the counts.

**Forward latency at h = 64** ~~:~~ **(rev6.10)**, at embargo 0 (~~the
confirmation set, or~~ a feed overridden to 0). For e = 48 (the decided
default) see the per-cadence table in rev6.10 (a):

- 5-min data: about 5.3 h of target span, plus up to a day for the next
  snapshot, plus 6 h margin ⇒ **about 1–1.5 days** **(rev6.14:** the 6 h
  is now the judging delay, not a margin; at e = 0 with `b = T`, 5-min has
  ≈ 1 fresh window per series at S1, ≈ T+6–9 h, rev6.14 table**)**. **(rev6.11)** With
  3-hourly snapshots: ≈ T+11–14 h. **(rev6.12)** That's the e = 0 timing;
  at e = 48 (now every cadence) 5-min has eligible windows at S1,
  ≈ T+6–9 h (rev6.12 table).
- Hourly: about 2.7 days of span, plus a day ⇒ **about 4 days**. **(rev6.11)**
  With 3-hourly snapshots: ≈ T+70–73 h at e = 0, or ≈ T+22–25 h with the
  48 h embargo ~~(hourly is embargo-eligible)~~ **(rev6.12)** (every cadence
  is embargo-eligible).
- W1 measures the real distribution. More sub-hourly sources shorten it.
- ~~**(rev3) With the mandatory confirmation duel**, add one more freeze
  period before crowning. That's **about 2–3 days (5-min-heavy) to about 8
  days (hourly-heavy)** from commit to crown.~~ **(rev6.11)** No
  confirmation duel: the crown comes at the first verdict (the rev6.11 (e)
  table). These latencies are the
  target-span arithmetic only, not measurements; B2 replaces them.
- **(rev3) The 6 h margin applies to both clocks (A1).** Daily-or-slower
  series almost never qualify, which is why A10's low-frequency check uses
  the private pool instead.

**Build host.**

- GitHub Actions `ubuntu-latest` has a 180-minute wall and a 10 GB cache
  quota (`publish-pool.yml:72-79, 121-131`). That's too small for multi-year
  backfill.
- **`pool_d` builds on a dedicated CPU builder**, never the box. It has its
  own disk, wall, retry and size budget, and the `TSFORGE_S3` read
  credentials live there only.
- The ~~daily~~ **(rev6.11)** 3-hourly (owner 2026-11-23) `pool_d_*` snapshots are pinned on Hippius with sha256 in every
  lane D report.

**Reveal.** `pool_d` adds **no reveal obligation**:

- weights are fixed at `ready`;
- the same series' recent tails are revealed in `pool` at about 48 h anyway;
- long history is withheld so lane A generators can't mimic it.

After the dispute window, `pool_d_tgt` and the window ids are published, so
anyone can check a crowned king's targets afterwards.

---

## 2. How checkpoints interact across lanes

```
     miner generators
            │
            ▼
   LANE A (we train) ──► KING_A (Hippius, public) ◄── LANE B merges (era boundary)
            │                     │
            │                     ├──► miners may start lane D runs from it
            │                     │
            │       house refresh whenever KING_A's digest changes
            ▼                     ▼
   LANE D (miners train anywhere): private vault ─► reveal T ─► prefetch ─►
        forward-only duel vs KING_D (house or miner) ─► gate ─► KING_D
            │
            └── no automatic flow back into lane A ("we just eval");
                an owner-gated promotion would be its own decision
```

1. **A → D:** the house enters whenever KING_A's digest changes (§1.4).
   Exact-digest dedup refuses resubmitted published checkpoints.
2. **The house can retake KING_D** when a better lane A king beats it on
   forward windows. The lane D king slot then burns.
3. **D → A: none.** KING_D never changes lane A's init, contract or receipts.
   Promoting a lane D king into lane A would be a separate decision, because
   lane D contamination would enter the lineage.
4. **B → D:** only via the house.
5. **C ↔ D:** none. **(rev3) Correction:** lane D does **not** cover lane C. Lane C is MarketCast on a pinned third-party Timestral-28M base (`[marketcast] base_ref`, DEC-CA-0064), not our architecture. The v2 text said lane C's fine-tune purpose is largely covered by lane D at
   zero training cost to us. Whether C stays on the roadmap is a later
   [DECIDE].
6. **E (phase 2)** targets KING_A. Whether it also applies to KING_D is a
   phase 2 decision.
7. **Contexts:** one board. Each model is scored at its declared context on
   the same targets. *(rev5: items 7 and 8 un-garbled.)*
8. **(rev4) Split of purpose.** **Lane B improves the lane A model on declared targets** (trainer-run, merged into KING_A, contamination-gated). **Lane D explores the architecture with unconstrained training** (miner-run, never merged, forward-only judged). That's why miner-trained weights are allowed in D and parked for B (§12).

---

## 3. Implementation (work packages)

The plan is built by an **autonomous loop**. A WP is done **only when its
Accept command exits 0**, never on a builder's word. Conventions:

- `scripts/laned/accept <WP> [--evidence <file>]` (built by W0c, then
  read-only to builders) runs, in order:
  1. `uv run --frozen pytest -q tests/accept/<wp>/`;
  2. the stub scan over the WP's declared paths (`tests/accept/<wp>/PATHS`);
  3. `make laned-e2e` (from W1.5 on);
  4. for external WPs, the evidence verifier.

  It exits non-zero on the first failure.
- **Runs in:**
  - `ci`: GitHub Actions; never the prod box.
  - `builder`: the `pool_d` builder VM.
  - `pod`: a rented GPU eval pod.
  - `testnet`: the netuid 259 stack (§10.3).
  - `owner`: the owner runs it.
- **External WPs** (live Hippius, a GPU, the builder or the chain) are
  **blocked** when the resource is unavailable. They are **never mocked**.
  Their Accept verifies evidence JSON written by the real run (endpoint,
  request ids, object digests, block numbers), and the evidence verifier
  cross-checks against the live resource where it can.
- **Owner steps** (remote writes, spend, merges, publishes, announcements):
  the loop prepares the script and artefacts and runs their dry-run checks;
  the owner executes. Accept checks the recorded result.
- **(rev6) Roles.** Each is a fresh-context agent; one agent never holds two
  roles on the same WP.
  - **Test author:** writes only `tests/accept/<wp>/` and its `PATHS`, from the
    WP row, its T-rows and the invariants it owns, copied from this plan. The
    tests must fail on main for the right reason (§17.1 L1). Opens a
    tests-only PR labelled `laned:<WP>` **(rev6.2:** or one tests-only PR
    batching several WPs, one `laned:<WP>` label per WP, per §3.1 rule 6**)**.
  - **Builder:** starts from origin/main after L1 and opens ONE implementation
    PR labelled `laned:<WP>`, staged with explicit paths. **A builder never
    edits ~~`tests/accept/`,~~ this plan, `scripts/laned/` or any status file
    (`status/laned_build.json`, `docs/plans/STATUS.md`).**
    **(rev6.17)** One agent may write the tests and the code of a feature in
    one PR (§3.1 rules 1 and 7), labelled with every WP it touches; CI proves
    the new tests fail on the base and pass on the head. It never deletes or
    weakens an existing assertion without the `laned:tests-correction` label.
  - **Preparer** (external and owner WPs): writes the scripts and runs their
    dry-runs only; the owner executes and the evidence verifier checks the
    result.
  - **Reviewer** (RVn): read-only on main; writes `evidence/rv<n>.md`.
  - Completion is never self-reported: only `scripts/laned/status` moves a
    level (§17.6).

### 3.0 Work packages

**Build-rule tooling and decisions (first)**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| W0c | Loop tooling: `scripts/laned/accept`; `stub_scan.py` (refuses `TODO`, `FIXME`, `pass`-only bodies, `NotImplementedError`, `...` bodies, hardcoded sample values and mock imports outside `tests/`); `unread_keys.py` (every `[lane_d]`/`[lane_d_ops]` key is read by a parser); `follower_patterns.py` (every lane D log call has a cascade-ops follower pattern). **(rev6) Also:** (i) `scripts/laned/status` per §17: computes every WP's level, the stage and milestone state, the decision register and the §17.3 counters from `git log origin/main`, GitHub Actions results on main and `evidence/`, and writes `docs/plans/STATUS.md` + `status/laned_build.json` to branch `laned/build-state` (never merged to main); `accept --gate <stage>` verifies a stage's exit-gate evidence (§17.2); (ii) a GitHub Actions workflow `.github/workflows/laned-accept.yml` that runs `scripts/laned/accept <WP>` on every PR labelled `laned:<WP>` and, on push to main, for every WP at ≥ L2. On a tests-only PR it runs in **expect-fail** mode: it passes only if the new tests are collected and fail for the right reason (an assertion failure, or a missing module/attribute named in the WP's `PATHS`), never on a syntax, fixture or import error elsewhere. Today `ci.yml` runs only `pytest tests/unit`, so nothing runs `tests/accept/` yet; (iii) the CI check that fails any PR that edits `tests/accept/` together with a path outside `tests/accept/` (rule 1), and any implementation PR touching a path not in its WP's `PATHS`; (iv) self-tests for (i)–(iii), listed in Accept. Creating the `laned:<WP>` label set is an owner step (`gh label create`); none exists today | — | ci | `scripts/laned/accept W0c` (self-tests: each script fails on a planted violation fixture and passes on a clean one; **(rev6)** `scripts/laned/status` on a fixture repo (fixture git history, CI results and evidence) yields the exact expected level for every fixture WP, including side states and a WP that regresses from L3 to L2; a planted rule-1 violation (tests + implementation in one PR) fails the workflow's check; an expect-fail run passes on correctly failing tests and fails on a syntax-error test file) |
| W0 | DEC-CA-0070 (supersedes DEC-CA-0062's recipe design) recording **(rev5)** D1–D43, D45–D48 and D42b (D44 withdrawn) (D25–D32 from §15, D33–D35 §16, D36–D42 §12b). **(rev6.1)** Owner-merged like every WP (§3.1 rule 6); that merge doesn't make W0 external | — | ci **(rev6.1)** | `scripts/laned/accept W0` (front matter valid; `supersedes: DEC-CA-0062`; **(rev5)** every decision D1–D43, D45–D48 and D42b answered or marked resolved) |
| W0b | Context-equivalence test: the lane D scorer at 4096 equals the pinned `pyramid_wrapper` output (within tolerance) on windows with ≥ 4096 history; a 4096 checkpoint loads `strict=True` at every context variant  **(rev5)** It compares **quantile-head** outputs at the 9-level grid (the lane A scoring path under `quantile_head_after`), not samples. | W5d | ci | `scripts/laned/accept W0b` |

**Rollout tooling (prerequisites for S0a–S0c, S4–S6)**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| R1 | Block-arithmetic tool `cascade-laned blocks` (era starts, activation and effective blocks from the live grid; cross-checked from two computations) | — | ci | `scripts/laned/accept R1` (golden vectors including past off-by-one cases) |
| R2 | Note-budget decoder (S0b): decode live readiness notes, simulate adding `cfg-`, report trimmed segments | — | ci (fixtures) | `scripts/laned/accept R2` (the 2026-11-13 crash-loop note fixture reproduces the refusal) |
| R2x | S0b live run: decode mainnet notes | R2 | owner (chain read) | `scripts/laned/accept R2x --evidence evidence/s0b_notes.json` (no pending segment trimmed) |
| R3 | Overlay lint: refuse any `[lane_d]` overlay that touches non-`lane_d` keys (except a byte-identical restatement of seq N-1) or `remainder` | W2 | ci | `scripts/laned/accept R3` |
| R4 | Stake-weighted upgrade tracker: per-hotkey `cfg-<hex>` and release from chain notes, stake-weighted %, names the laggards | R2 | ci | `scripts/laned/accept R4` (fixture metagraph + notes ⇒ exact %) |
| R5 | Previous-release worktrees + restart scripts for validator, intake, trainer and eval host (§10.8), with `--dry-run`. **(rev6.1)** The pre-restart snapshot covers the full §10.8 list: the validator files, **plus** the trainer state under `/root/cascade/_train_work/` (`era_state.json`, `trainer_submissions.json`, `funded_queue.json`, `trainer_promotion.json`, `champion_publisher.json`, `lane_b_driver.json`, `coldkey_identities.json`, `published_settlements.json`, its `activation_*` and `netconfig` files), `provisioner_state.json`, `pod_monitor_state.json`, and the deployed `chain.toml` (live overrides) with `hosts.toml` / `provision.toml` (RV3 note 21); a file absent on the box is reported as absent | — | ci (scripts) + owner (execute) | `scripts/laned/accept R5` (shellcheck; dry-run prints the exact plan; refuses during a scoring or settlement window fixture) |
| R6 | De-drift classifier (S0a): classify each `LIVE OVERRIDE` in a deployed chain.toml as main / overlay / operator-only; emit the main PR diff and the overlay body | — | ci | `scripts/laned/accept R6` (fixture with the 21 markers ⇒ deterministic classification) |
| R6x | S0a execution: main PR merged; deployed file reconciled | R6 | owner | `cascade-audit decided --config /root/cascade/chain.toml --state-dir /root/cascade` exits 0 **and** `scripts/laned/accept R6x --evidence evidence/s0a_diff.json` (consensus-key diff empty) |
| R7 | Overlay bring-up artefacts (S0c): benign seq-1 body generator, a refused-overlay body, runbook wrapper around `cascade-validator --netconfig-overlay … --netconfig-publish` (Approve-tier, dry-run shown to the owner) | R1 | ci | `scripts/laned/accept R7` (generated bodies pass `validate_overlay` on a fixture base; the refused body is refused whole) |
| R7t | S0c testnet drill (needs testnet `[netconfig] owner_hotkey`, today `""` at `chain.testnet.toml:903`) | R7 | testnet + owner | `scripts/laned/accept R7t --evidence evidence/s0c_testnet.json` (seq-1 adopted ≥ 51%; refused body applied nothing) |
| R7m | S0c mainnet no-op overlay | R7t, R6x, R2x | owner | `scripts/laned/accept R7m --evidence evidence/s0c_mainnet.json` (no 404 on S3 or HF; ≥ 51% `cfg-`; receipts stamp `netconfig_seq`; trainer apply block logged) |
| R8 | `deploy_gate` support for the new services (`--service intake-laned|eval-host`), since today only trainer/validator are accepted (`scripts/deploy_gate.py:290,298`). **(rev5)** Plus: (i) a new AREA `lane_d_intake` for `^cascade/lane_d/` (its own service). Today `cascade/lane_d/` matches **no** AREA (`deploy_gate.py:41-58`), so it passes unclassified; (ii) a CI test that **every path in the repo classifies** into at least one AREA, or is on an explicit docs/tests allow-list; (iii) `WORKER_IMAGE` gains the on-pod trainer modules (`cascade/trainer/expert_training.py` and what `worker.py --role expert` imports). The Dockerfile bakes the whole repo (`COPY . /root/cascade`, `deploy/Dockerfile:36`), but the AREA lists only `worker.py`, `interface/`, the Dockerfile and the lock files | — | ci | `scripts/laned/accept R8` |
| R9 | Catch-up release (S2b): main-as-of-now without lane D to validator, trainer and satellites | R6x | owner | `scripts/laned/accept R9 --evidence evidence/s2b.json` (2 settlements, identical vectors, no satellite burn) |
| R10 | Upgrade-from-mainnet-state harness (T25): restores a copy of `validator_state.db`, `activation_*`, `netconfig_state.json` and `cascade_state.json` into a testnet node and starts the release | — | ci (fixture state) + testnet | `scripts/laned/accept R10` and, for T25, `--evidence evidence/t25.json` |
| R11 | Shipped-config startup-gate test (T26): run the shipped `chain.toml` and `chain.testnet.toml` through every validator and trainer startup gate (activation note, loader, preflight) at a pinned live head | W2 | ci | `scripts/laned/accept R11` |
| R12 | Eval-host provisioning kit: on-rent check (`nvidia-smi -L`, VRAM, driver, disk), pinned known_hosts, venv sync, handshake, owner-paste setup and recovery scripts (§10.4) | W4 | ci (scripts) + pod + owner | `scripts/laned/accept R12` (shellcheck + dry-run) and `--evidence evidence/r12_pod.json` for a live run |
| R13 | `pool_d` builder VM provisioning: disk, wall, retry, credentials placement (no secrets printed), systemd timer | — | builder + owner | `scripts/laned/accept R13 --evidence evidence/r13.json` |
| R14 | Testnet stack provisioning (§10.3): testnet-ctl, testnet-eval, testnet `[netconfig] owner_hotkey` set (release-level edit), hourly testnet `pool_d`, pod TTLs and verified teardown | R13 | testnet + owner | `scripts/laned/accept R14 --evidence evidence/r14.json` |
| R16 | **(rev5)** Long-history harvest (§1.7 "Harvest now, score later"): sub-hourly forge-mirror retention on; deep-history sources recording; the builder's append-only per-series archive up to 131072 + 64 (W6a join key, private bucket, daily sha pin); per-rung readiness counts. Never scored before S8; never on the box | R13, W6a | builder + owner | `scripts/laned/accept R16 --evidence evidence/r16.json` (archive grows daily; `pool` bytes unchanged) |
| R15 | **(rev3)** Testnet pool_mkt cadence for lane D: a scheduled publish for the testnet lane D path that doesn't depend on `[marketcast] from_block` (today scheduled runs skip while it is 0, `publish-mkt-pool.yml:84-95`) | R14 | ci + testnet | `scripts/laned/accept R15 --evidence evidence/r15.json` (≥ 4 consecutive 6 h snapshots published on testnet) |

**Lane D core**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| W1 | **(rev3)** Measurement (= S1 B2, `scripts/laned_w1_measure.py`): first-seen existence check; snapshot-diff **upper bound** on forward-eligible counts at both clocks; pool_mkt at 6 h; (§1.3, §1.7): forward-eligible counts and latency per frequency; long-history inventory per context variant; `pool_d` size. **(rev6.14)** The script's single `--margin-hours` (default 6) is split to match rev6.14 (a): `--fresh-from` semantics (fresh / eligibility counted from `T + forward_margin_hours`, default 0) and counts taken at the first snapshot / settlement ≥ `T + eval_delay_hours` (default 6); fresh and embargo windows reported separately per frequency (10 s / 1-min / 5-min sources counted) | — | builder (needs `TSFORGE_S3`) | `scripts/laned/accept W1 --evidence evidence/w1.json` |
| W2 | `[lane_d]` config: the keys `from_block`, `until_block`, `share`, `takes_from`, `remainder`, `contexts`, `horizon`, `forward_margin_hours`, `min_windows`, `min_clusters`, `min_long_series`, `d_margin`, `alpha`, `bootstrap_B`, `gate_suites`, `gate_weights`, ~~`unselected_suite`~~ **(rev5) dropped:** the held-out suite is the §16 sentinel, drawn from `[v2] sentinel_suites` (W-S1); a lane D key would duplicate it, `tau_public`, `reward_prior_kings`, ~~`king_decay`~~ **(rev6.23)** `court_weights` (`[0.80, 0.15, 0.05]`; length `1 + reward_prior_kings`, each ≥ 0, sum 1 within 1e-9; `king_decay` → `RETIRED_KEYS["lane_d"]`), `max_entries_per_settlement`, `max_open_per_coldkey`, `max_bench_holds`, `loser_ttl_eras`, `dtypes`, **(rev3)** `max_mirror_lag_hours` (A1), `confirm_on_king_change` (D20), `max_judged_per_coldkey_per_settlement` (A8), `max_coldkey_unresolved_settlements` (A8), `max_reveal_delay_days` (A9), `lowfreq_freqs` (A10); **(rev5)** `ready_ttl_blocks` (B10), `max_source_share` (B14); **(rev6.9)** `backtest_weight_cap` (default ~~0~~ **(rev6.15)** 0.05) and `backtest_ratio_floor` (rec 0.85), parsed and range-checked here, with the inequality check in W5i; `contexts` default ~~`[4096, 8192, 16384]` (A5)~~ **(rev6.6)** `[4096, 8192, 16384, 32768, 65536, 131072]` (every variant declarable from launch; arming per variant is computed, not configured, §1.1); `dtypes` default `["BF16"]` (A6). **(rev3) Size M, not S:** `lane_d.py` hardcodes `active() = False` (`:65-68`), refuses `from_block > 0` (**(rev5)** `:78-79`) and carries DEC-CA-0062 placeholder keys that must keep parsing (`until_block` and `share` overlay-settable; activation by signed-overlay `from_block`, never a DEC-CA-0045 vote), plus legacy keys still parsed (the deployed file carries `targets`, `teacher_*`, `gain_margin`, …); `ALLOWED["lane_d"]` with `remainder` refused via the `(table, key)` special case (**(rev5)** `netconfig.py:210`); loader parse, round-trip and **mutation** test per key; the 131072 cap; `LaneDConfig.active()` real | W0c | ci | `scripts/laned/accept W2` (includes `unread_keys.py`) |
| W2b | `[lane_d_ops]` non-consensus table (vault bucket, `pool_d` location, eval host, queue policy), read only by the main validator and the intake; never in receipts | W2 | ci | `scripts/laned/accept W2b` (satellite code path imports nothing from it) |
| W2e | **(rev6.10, owner 2026-11-23)** `[lane_d]` keys added to W2's set (a separate row so W2's text stays untouched while #745 edits it): `embargo_hours` (default **48**, decided: lane A's ~48 h reveal delay; ≥ 0; 0 = the rev3 rule), `max_windows_per_settlement` (~~rec 2000~~ **(rev6.11)** **5000**, owner 2026-11-23; refused below `min_windows`); ~~**(rev6.11)** `embargo_window_share_cap` (rec **0.25**, in [0, 1]; 0 = no embargo windows)~~ **(rev6.12)** renamed and redefined: `min_fresh_window_share` (**0.5**, owner 2026-11-23, in [0, 1]; 1 = fresh windows only; the minimum fresh share of a frozen set, rev6.12 (b)). ~~Also confirms `forward_margin_hours` = **6** (already the default, `cascade/shared/config/lane_d.py:59`, `:244`).~~ **(rev6.14)** Adds `eval_delay_hours` (default **6**, float, ≥ 0; the judging start `j`) and changes the `forward_margin_hours` default **6 → 0** (`cascade/shared/config/lane_d.py:59`, loader `:244`); refuses `forward_margin_hours > eval_delay_hours`. A config default is consensus-relevant (every validator must derive the same `b`, `j`); lane D is unarmed, so no live effect; before arming, diff the deployed chain.toml for an explicit `forward_margin_hours = 6` override. Loader parse, round-trip and mutation test per key (a dataclass field without a loader parse is a silent no-op); both are overlay-settable lane D tuning (§14.2) | W2 | ci | `scripts/laned/accept W2e` (includes `unread_keys.py`) |
| W1.5 | **Walking skeleton** `make laned-e2e` (CI): fixture `pool_d` (with `end_ts`) → tiny checkpoint → intake (filesystem adapter behind the real storage interface) → ingest guards → scorer → signed report → `lane_d_state.json` → emission → `check_lane_d` | W2 | ci | `make laned-e2e` asserts: non-zero lane D weight for the challenger; audit passes; one tampered row fails; unarmed byte-identical; `share = 0` leaves weights unchanged and still yields a valid report; `LaneDConfig.active()` is not a constant. **Every later WP keeps it green and adds ≥ 1 assertion** |
| W3a | Storage probe scripts (presigned UploadPart, single-PUT size limit, Object Lock on multipart, prefix bucket policy, sub-token scope) + the **storage decision record** `decisions/STORAGE-LANE-D.md` naming the chosen upload path  **(rev5)** Also probes `UploadPartCopy` for ~2.5 GB objects (~~locking a king by copy at crowning~~) and ~~confirms COMPLIANCE retention can't be shortened~~. **(rev6.4)** Hippius has no Object Lock (403 on `GetObjectLockConfiguration` / `GetBucketVersioning` for sub-token and main key): the probe re-runs with `--no-object-lock` and the lock probes are recorded as `skipped`; kings are not locked by copy (§1.2). | — | owner (live Hippius) | `scripts/laned/accept W3a --evidence evidence/w3a_probe.json` (every probe has a recorded live result; the decision record names exactly one path) |
| W3 | `cascade/lane_d/` intake service: submit/ready for **only the W3a-chosen upload path**; ownership; **(rev3)** records `coldkey_at_ready` and refuses over-cap with `coldkey_cap_open` (advisory, A8); dedup; rate limits; queue; systemd unit with the code-path guard; Caddy route `/v1/lane-d/*`; RAM budget  **(rev5)** Enforces `ready_ttl_blocks` (B10); records the licence attestation at `ready` (B13). | W3a, W2b, R8 | ci (fs adapter) + testnet | `scripts/laned/accept W3` |
| W4 | Ingest guards: presigned fetch; sha256; `verify_checkpoint`; exact-filename allowlist; **lane-D-only dtype path for BF16** (today's guard requires the pinned F32 spec, `checkpoint_guard.py:323-328`, which lane A keeps unchanged); `verify_weights_finite`; worker handshake (git head + package hash) | W1.5 | ci + pod (handshake live) | `scripts/laned/accept W4` (the T5 malformed set is refused with exact codes) |
| W5a | Forward eligibility (§1.3): **(rev3) both clocks** (data time and first-seen > wall(max T) + margin) and `max_mirror_lag_hours`; the v2 snapshot-only rule is refused. **(rev6.10) Embargo-aware:** per feed `e = min(feeds[feed].embargo_hours, [lane_d] embargo_hours)` (a missing or non-numeric `embargo_hours` in a listed feed fails closed as `unlisted_feed`); ~~**(rev6.11)** `e = 0` for a series whose pool `freq` is finer than 1 h;~~ **(rev6.12)** the same `e` for every `freq` (no sub-hourly zero); a target may mix embargoed and post-submission points; each eligible window is classed **fresh** (passes at e = 0) or **embargo** (incl. mixed) for W5j's fresh-first draw and `min_fresh_window_share` (rev6.12 (b)); eligible iff `first_seen + e > b` and `data_ts > b − e` for every target point, `b = wall(max(T_E, T_K)) + forward_margin`; S0/backfill at `b − e`; e = 0 byte-identical to the rev3 verdicts (test diffs both); ~~an `embargo0` mode (all `e` = 0) for the confirmation set;~~ **(rev6.11)** no confirmation set, so `embargo0` is only the e = 0 per-feed case; no freeze before ~~`b`~~ **(rev6.14)** `j` (W5e owns the judging start). **(rev6.14)** `b = wall(max(T_E, T_K)) + forward_margin` with default margin 0, so `forward_bound()` returns max(T) by default; fresh ⇔ `data_ts > b` ∧ `first_seen > b` (points in `(T, T+6 h]` are fresh); formulas otherwise unchanged. Touched: `cascade/validator/lane_d_forward.py` (`forward_bound`, `window_reason`, `eligible_windows`) | W1.5, W2e, W6b | ci | `scripts/laned/accept W5a` |
| W5b | Union across snapshots + freeze at floors  **(rev5)** Cap any one source's share of a frozen set's windows at `[lane_d] max_source_share` (recommended 0.25), dropping the excess deterministically (T18e). **(rev6.10)** The union starts at snapshots built after `b − max e`; floors are checked on the W5j draw (after the source cap), and the frozen set is that draw | W5a | ci | `scripts/laned/accept W5b` |
| W5j | **(rev6.10, owner 2026-11-23)** Seeded per-bin window draw (rev6.10 (b)). Cells = (timeframe, bin): each sub-daily `freq` one forward bin at h = 64, plus the rev6.9 slow bins when #745's W5i is armed. Seed `_seed_to_int(f"{boundary_block_hash}:lane_d_windows")` (`cascade/eval/bootstrap.py:29`, as lane A, `windows.py:38-40, :405`), one stream in a fixed cell / candidate order; h uniform in the bin, origin uniform over eligible non-overlapping candidates; equal quota per cell, **(rev6.12)** fresh candidates first within each cell, then embargo candidates (fresh-first draw), scarce cells taken whole, surplus re-split (as `_pick_series`, `windows.py:461-493`); `max_source_share` after, no refill; ~~**(rev6.11)** then embargo windows (pass only at e > 0) capped at `embargo_window_share_cap` × set size, excess dropped deterministically, no refill, floors checked after;~~ **(rev6.12)** then embargo windows above ~~`F × s / (1 − s)`~~ **(rev6.14 erratum)** `F × (1 − s) / s` (s = `min_fresh_window_share` = 0.5, so ≤ F; the W5j tests already keep `F ≥ s · (F + K)`) dropped deterministically, last-drawn first, no refill; floors checked after; a set below the floors at that share is **not frozen** (entry stays `pending`, `fresh_share_short`), the share never relaxed; total ≤ `max_windows_per_settlement` (**(rev6.11)** 5000) ~~with confirmation draws first~~, `window_budget` deferral otherwise; one draw per cohort over its intersection set (identical for king and challengers); ~~the confirmation draw over the embargo-0 set, disjoint by target points from the first set.~~ **(rev6.11)** No confirmation draw. Report records each drawn window (`timeframe`, `bin`, `h`, origin, `kind`) and the block hash. Touched: `cascade/validator/lane_d.py` (`judge_cohort` and the windows build), `cascade/validator/lane_d_frozen.py`, `cascade/validator/lane_d_inputs.py` (W8c) | W5a, W5b, W2e | ci | `scripts/laned/accept W5j` |
| W5c | Cohort intersection + latest-T deferral | W5b | ci | `scripts/laned/accept W5c` |
| W5h | **(rev3)** Queue selection: FIFO by T; `max_judged_per_coldkey_per_settlement`, the global cap, coldkey re-check at reveal (`coldkey_changed`), `coldkey_unresolved` bound, `max_reveal_delay_days` close; typed skip reasons in the report (A8, A9)  **(rev5)** `deregistered` void (court slot burns, with W8) and `ready_expired` (`ready_ttl_blocks`) handling, both typed in state and report. | W5b, W3 | ci | `scripts/laned/accept W5h` (T6b, T18b) |
| W5d | No-pad scorer + context variants (`cascade/eval/lane_d_scorer.py`). **(rev5)** Quantile-head scoring at the 9-level grid, the same method as lane A under `[v2] quantile_head_after = true`; a CI test that the house scored by the lane D scorer at 4096 equals its lane A per-window CRPS on the same windows. **(rev6.15)** `effective_context(declared, armed)` returns `declared` when armed, else the highest armed variant below it, else **`None`** (~~else the lowest armed variant~~); it never returns a context above `declared` | W1.5 | ci (tiny model) + pod (real size, T7) | `scripts/laned/accept W5d` and, for T7, `--evidence evidence/t7.json` |
| W5e | Verdict (paired cluster bootstrap, `d_margin`, max-T; ~~**(rev3)** mandatory confirmation duel on the next disjoint set, D20 king-change rule,~~ `rows_sha256` (A2, A9)). **(rev6.11)** **Crown at the first verdict** (owner 2026-11-23): first clear (max-t LCB vs KING_D on the W5j draw) ∧ A10 (W3b5) ∧ §1.3b gate ⇒ crowned. **Remove the `confirm` stage** (chosen over a `confirmation_duel` key; rev6.11 (b)): drop `confirm_pending` from `OPEN_STATUSES` (`lane_d.py`, `lane_d_queue.py:42`, `lane_d_status.py:51`), the `confirm` record, `_confirm_king_ts`, the D20 king-change sweep and `lost_confirmation`; `confirm_on_king_change` parse-only (legacy). **Includes the W5g follow-up** (the W5g row isn't edited, so it doesn't conflict with #745): `bench_hold` = a first-clear candidate the gate couldn't decide; the hold sweep keys on the first clear; the reason-enum assertions drop `confirm_pending`. **(rev6.14) Judging start:** an entry is not drawn, frozen or judged at a settlement whose boundary wall time is < `j = wall(max(T_E, T_K)) + eval_delay_hours` (skip reason `before_eval_delay`, distinct from `fresh_share_short` and the floor reasons) | W5c, W5d | ci | `scripts/laned/accept W5e` |
| W5f | House entry and refresh | W5e | ci | `scripts/laned/accept W5f` |
| W5g | Holds, `checkpoint_fault`, typed reason enums | W5e | ci | `scripts/laned/accept W5g` (asserts the exact reason enum per case) |
| W5i | **RETIRED (owner 2026-11-24, DECIDED), rev6.18 D24: not built; the public half replaces it.** ~~**(rev6.9, owner 2026-11-23)** Slow-cadence backtest windows and their scoring weight (rev6.9 spec). (1) **Windows builder:** the per-cadence horizon bins; bin and h drawn per window from the boundary block hash ~~(`sha256(block_hash ‖ series_id ‖ "lane_d_backtest")`)~~ **(on merge with rev6.10)** by the W5j seeded per-cell draw (W5i supplies the slow-cadence cells and the hold-out rule), the same for king and every cohort member; daily short ⇒ a forward window at the drawn h through W5a's both-clocks rule; every other slow bin ⇒ a backtest hold-out (the last h points of the pinned `pool_d` snapshot's series, history before it, no padding); `backtest_short_history` skip; floors, freeze and `max_source_share` count forward windows only. (2) **Scorer/verdict:** forward and backtest sub-scores, combined as `log R = (1 − c)·log R_fwd + c·log clamp(R_bt, r_floor, 1/r_floor)` with `c = min(backtest_weight_cap, n_bt/(n_bt + n_fwd))`, clamped inside every bootstrap replicate of the max-T cohort LCB. (3) **Config:** `backtest_weight_cap`, `backtest_ratio_floor`; the loader refuses `backtest_weight_cap > 0` unless `1 − backtest_ratio_floor ** backtest_weight_cap < d_margin` (re-checked on overlay adoption); ~~default 0 = nothing built or scored~~ **(rev6.15)** default 0.05 (on at launch); 0 = nothing built or scored. (4) **Report:** per window `kind`, `freq_class`, `bin`, `h`; per entry and bar `forward` and `backtest` sub-scores plus the combined score. (5) **Audit + replay:** re-derive the draws, the clamp and the combined verdict independently. Touched modules: `cascade/validator/lane_d.py` (`load_pool`, `window_id`, `judge_cohort`), `cascade/validator/lane_d_forward.py` (per-window h for daily short), `cascade/validator/lane_d_frozen.py` (forward-only floors), `cascade/eval/lane_d_scorer.py` (h ≤ 180 per window; already capped at 512), `cascade/eval/bootstrap.py` (the combined statistic for `cohort_maxt_lcbs`), `cascade/shared/config/lane_d.py` (keys + inequality), `cascade/validator/schemas/lane_d_report_1.json` and `cascade/validator/lane_d_report.py` (fields + cross-field checks), `cascade/audit/lane_d.py` (W9 re-derivation), `cascade/satellite/lane_d_replay.py` (W7d, independent implementation), `cascade/validator/lane_d_inputs.py` (W8c supplies backtest histories and targets). **Follow-up (W11):** the cascade-ops alert on backtest gain ≫ forward gain~~ | W5a, W5b, W5d, W7a, W9, W7d, W5j | ci + testnet | `scripts/laned/accept W5i` |
| W6a | `pool_d` selection = `pool`'s selected series ids (joined, not re-selected). **(rev5) Join rule:** the emitted `series_id` isn't stable. Collision suffixes `-2`, `-3`… are assigned in sort order (`builder.py:361-370, 471-480`), and `content_hash` is the sha256 of the (truncated) values (`:295`), so it changes whenever the tail moves and differs between `pool` and `pool_d` histories. Join on **(source, the pre-suffix source series key, channel)**, both built in the **same builder run** from the same forge records, never by emitted id or content hash. A key collision inside one source refuses the series from `pool_d` (logged) | W1 | ci + builder | `scripts/laned/accept W6a` (`pool` bytes unchanged) |
| W6fs | **(rev3, launch-blocking)** First-seen recording: an append-only ingest log in the forge mirror path recording, per partition or point, the first time our mirror saw it, plus a per-feed mirror-lag measurement (source publish stamp where available). **Scope:** a recorder + reader used only by the `pool_d` builder. **Invariant:** `pool` bytes and lane A windows byte-identical (the test diffs a `pool` build with and without the recorder). **Tests:** first-seen never later than any snapshot containing the point; rewrites and backfills keep the *first* time; lag computed per feed. Not implemented in this PR  **(rev5) Attestation (B12):** a ~~daily~~ **(rev6.11)** per-3-hourly-period digest (Merkle root) of the ingest log is committed **before** any snapshot uses that ~~day's~~ period's records: in the signed receipt, or on chain via the main validator's commitment. A snapshot whose first-seen records postdate their committed root is refused. **(rev6.10) Release-time record:** W6fs is **required** for the embargo rule (`release(p) = first_seen(p) + e`). The recorder doesn't store `release` (it's derived, so the catalog can't rewrite history). It stores first-seen per point under the B12 Merkle root, plus per feed the measured `max(data_ts − first_seen)`, which W6b uses to force `e = 0` on publish-ahead feeds (rec). Test: release is recomputable from the signed first-seen record and the pinned `e` alone | W1 | builder | `scripts/laned/accept W6fs --evidence evidence/w6fs.json` |
| W6b | **(rev3)** Extended history **to 16384 + 64** + per-series (per-channel) `end_ts` / `first_target_ts`. **(rev6.1)** Also writes, for **every** series, `frontier` (a list, possibly empty: one record per earlier snapshot holding that series), and, for **every** feed in `pool_d`, a `feeds` entry (`mirror_lag_hours`, a number or `null`, and `forecast_product`). A `pool_d` in which any series lacks `frontier` (absent or `null`) or any feed lacks a `feeds` entry is **invalid**; the builder's tests assert that none does (RV3 BLOCKING 1: W5a's eligibility fails open on a missing key). **(rev6.10)** Each `feeds` entry also carries `embargo_hours`, the **effective** value: the TSBench-Forge catalog's per-feed `embargo_hours` (`sources.yaml`, read by the `pool_d` builder alongside the catalog fields `tsbench_forge.py:340` already reads), clamped to `[lane_d] embargo_hours`; the global value when the catalog has none; **0** for a feed W6fs shows as publish-ahead (rec). ~~**(rev6.11)** The sub-hourly zero is applied per series by W5a from `freq`, not written here.~~ **(rev6.12)** No sub-hourly zero: the pinned value applies to every series of the feed. A negative or non-numeric catalog value leaves the feed out of `feeds` (fail closed). `pool` bytes unchanged | W6a, W6fs, W2e | builder | `scripts/laned/accept W6b --evidence evidence/w6b.json` |
| W6c | `pool_d_hist` / `pool_d_tgt` split | W6b | ci + builder | `scripts/laned/accept W6c` |
| W6d | Pinning on Hippius; digests in reports; retention. **(rev6.11)** **Cadence: one `pool_d` snapshot every 3 h** (owner 2026-11-23), built on lane D's own CPU builder, independent of lane A's daily `pool`; each build finishes before a settlement boundary (rec); retention keeps every snapshot an open entry's eligible or frozen set references; the build wall must stay < 3 h (alert otherwise). **(rev6.10)** The pinned snapshot (sha256 in every report) is the only source of the per-feed `embargo_hours` a verdict used; the report also carries the sha256 of the catalog `sources.yaml` the builder read, for provenance only (never re-read by audit or replay) | W6c, W3a | builder | `scripts/laned/accept W6d --evidence evidence/w6d.json` |
| W6rv | **(rev6.11)** **Not needed (owner 2026-11-23: "don't need for lane A, it's only for lane D"); not built.** **Guard note:** lane A's `publish-pool.yml` cadence must not increase unless this WP ships first. ~~**(rev6.10, rec; needs owner yes: changes lane A reveal timing, not pool bytes)**~~ Enforce the embargo at the reveal side so lane A's ~48 h (today emergent: `keep_newest = 2`, `cascade/pool/reveal.py:66-79`, `cli.py:384-388`, × daily cron `publish-pool.yml:45`) can't drift below `[lane_d] embargo_hours`: `select_reveals` also withholds any snapshot built less than `embargo_hours` ago, for every reveal prefix (`--lane`) that can carry forge points; both read the one chain.toml value. **Tests:** three real snapshots published inside 48 h (keep_newest alone would reveal the oldest) ⇒ the oldest is withheld until 48 h after its own build, so every point is revealed ≥ `first_seen + embargo_hours`; revealed bytes unchanged; sentinel handling unchanged | W2e | ci | `scripts/laned/accept W6rv` |
| W6s-*n* | **(rev3: moved to S8)** One WP per deep-history source (option 3, §1.7), each passing cleaning, domain and source keys | W6b | builder | `scripts/laned/accept W6s-<name> --evidence evidence/w6s_<name>.json` |
| W6-S8 | **(rev3)** S8 long-context work: history to 131072 + 64, deep backfill, valeval-laned (96 GB pinned-torch) provisioning. **(rev6.6)** S8 is no longer the gate for the 32k–128k contexts (they are in `contexts` from launch and arm automatically on `min_long_series` + T7, §1.1). W6-S8 still owns the 131072 + 64 history/backfill work and valeval-laned provisioning, which in practice is what lets those variants pass their gates | W6d | builder + pod | `scripts/laned/accept W6-S8 --evidence evidence/w6s8.json` |
| W3b1 | Gate suite draw from the boundary block hash | W1.5 | ci | `scripts/laned/accept W3b1` |
| W3b2 | Paired sweep vs KING_D and house on one host | W3b1 | ci (recorded suite outputs) + pod | `scripts/laned/accept W3b2` |
| W3b3 | **(rev5)** Sentinel sweep + publish: the era's §16 sentinel suite (not fixed TIME), swept and published for every new king, never gating | W3b2 | ci | `scripts/laned/accept W3b3` |
| W3b4 | Suite identity + partial classification | W3b2 | ci | `scripts/laned/accept W3b4` |
| W3b5 | **(rev6.18) REMOVED (A7): code, schema block, audit rule and tests deleted in F4.** ~~**(rev3)** Always-run low-frequency private-pool non-inferiority check (`lowfreq_freqs`, `tau_public`; `lowfreq_regression`) (A10). **(rev6.1)** Result for a ~~confirm candidate (a verdict that cleared at stage `confirm`)~~ **(rev6.11)** crown candidate (a verdict that cleared at the first, only, stage) whose frozen set **does** hold low-frequency windows: `pass` or `fail` (`fail` = `lowfreq_regression`, not crowned), with the true `n_windows`; it is never `not_candidate`. W3b5 replaces W7a's interim path, which emits `not_candidate` with the true count (`validator/lane_d.py:371-387`). `no_windows` stays for a candidate with none; `not_candidate` (with `n_windows = 0`) stays for every non-candidate verdict. The W7a enum (`schemas/lane_d_report_1.json`, `LOWFREQ_RESULTS`) already holds `pass`/`fail`; W3b5 adds a value to it only if it introduces a new one | W5d | ci | `scripts/laned/accept W3b5` (T4b) |
| W3b6a | **(rev6.13)** Lane D bench sweep transport (rev6.13 W3b6 spec). Pod side: `cascade/eval/lane_d_bench_sweep.py` (`hello` / `sweep`): W15 cache + W4 `prefetch_grant` per digest (grants only, no weights through the box; infra faults fail the sweep, a W4 refusal is that digest's `error`), materialised per (digest, context) in lane A's sidecar layout (W4-verified release code, `config.json` `context_length` = the effective context, weights as ingested); one paired `_invoke_sidecar_many` pass; `item_rows_from_report` rows + hardware fingerprint. Box side: `cascade/validator/lane_d_bench_host.py` `SSHBenchHost`, the `SweepSession` sweep over SSH to the first `final`/`any` host of a re-read hosts file (a `bench_lease` file works); any defect raises ⇒ `bench_hold`. `cascade/validator/lane_d_inputs.py`: optional `bench_hosts` TOML key (absent ⇒ `bench_host = None`, today's behaviour). PATHS: those three files | W3b2, W8c, W15, W4 | ci + pod | `scripts/laned/accept W3b6a` |
| W3b6b | **(rev6.13)** Lane D bench rows published, bound and re-derived (rev6.13 W3b6 spec). `lane_d_bench.sweep_record_text` (canonical JSON); each swept gate result gains `rows: {candidate, king_d, house}` sha256 inside the signed body (schema: optional, additive, `/1` kept); W8b's step PUTs each record at `bench_v/lane_d/<boundary>/<sha256>.json` through its existing store before the report (`publish_failed` on error); W7d's replay fetches, checks the sha256 (`gate_rows_mismatch`), recomputes the four LCBs with `expert_lane.paired_geomean_lcb` (`gate_lcb_mismatch`) and refuses a crowned pass without `rows` (`gate_rows_missing`). Deterministic, signed, content-addressed; satellites re-derive, never re-sweep; lane A byte-identical. PATHS: `cascade/validator/lane_d_bench.py`, `cascade/validator/lane_d.py`, `cascade/validator/schemas/lane_d_report_1.json`, `cascade/validator/lane_d_wiring.py`, `cascade/satellite/lane_d_replay.py` | W3b2, W7a, W7d, W8b | ci | `scripts/laned/accept W3b6b` |
| W7a | Report schema `lane-d-report/1`, including §14.3 forward fields and **(rev3)** `rows_sha256`, both clocks per window, mirror lag, the judged set and skip reasons, `coldkey`, ~~the `confirmation` block~~ **(rev6.11)** (removed: no `confirmation` block; enums lose `confirm_pending`, `confirmed`, `lost_confirmation`, stage `confirm`, `confirm_rule`; `confirm_on_king_change` leaves the config echo; stay on `/1` unless a testnet `confirm` report must stay verifiable, then `/2`), the low-frequency result, **(rev5)** a `scoring_method: "quantile_head"` field (refused if absent or different)  **(rev5)** Plus the first-seen Merkle root(s) each frozen set relies on (B12). **(rev6.10)** Per window: effective `embargo_hours`, `min_release_ts`, `rule` (`embargo` \| `embargo0`), `timeframe`, `bin`, `h`, origin, `kind`; per draw: the boundary block hash, `max_windows_per_settlement`, the per-cell quota and capacity, and the per-cell candidate window ids the draw was made from (so W7d can re-draw; bounded by §1.6's report-size limit, else their sha256 with the list published after the dispute window); config echo `embargo_hours`, **(rev6.11)** ~~`embargo_window_share_cap`~~ **(rev6.12)** `min_fresh_window_share`, each window's series `freq` and an `embargo_window` flag, and the set's embargo-window count **(rev6.12)** plus fresh count and the per-entry fresh and embargo sub-scores (cross-check: ~~count ≤ cap × set size~~ **(rev6.12)** fresh / (fresh + embargo) ≥ `min_fresh_window_share`; ~~no `embargo_window` on a sub-hourly series~~ **(rev6.12)** any cadence may carry `embargo_window`); cross-field checks (`rule = embargo0` ⇒ every point passes at `e = 0`; ~~the confirmation set is `embargo0` and disjoint by target points;~~ a window's `embargo_hours` ≤ the config echo). **(rev6.14)** Config echo `eval_delay_hours` and `forward_margin_hours`; per entry `b` and `j`; cross-checks: freeze settlement's boundary wall time ≥ `j`, `b ≤ j`, and each window's fresh class at the echoed `b`. **(rev6.15)** Skip reason `context_below_armed` (enum in `lane_d_report.py` and the schema file; stays `/1`): with the `armed` record given, `settle_lane_d` skips each chosen member whose declared context is below every armed variant (`min(armed_contexts) > context`), status stays `pending`; the house challenger is exempt (it declares no context); cross-check: such a skip needs `armed_contexts` present and below-armed | W1.5 | ci | `scripts/laned/accept W7a` (JSON-schema; unknown version refused) |
| W7b | Report signing | W7a | ci | `scripts/laned/accept W7b` |
| W7c | `lane_d_state.json` + receipt prior/posterior digest binding. **(rev6.1)** Wiring trap (RV3 note 10): `validator/weights.py:517-524` passes the second element to `emission_vector` only when a lane B or C pay record exists (`pay is not None or pay_c is not None`); if lane D pays while B and C are off, the lane D share is dropped and `set_weights` normalisation hands it to lane A. The WP that wires lane D pay into weights (W7c, or the one that passes `lane_d_cfg` first) adds the lane D pay record to that condition, with a test for "lane D on, B and C off". **(rev6.11)** **Recovery of a `confirm_pending` state** (testnet only; never armed on mainnet): on load, an entry in `confirm_pending` whose `confirm.king_digest` equals KING_D becomes a crown candidate on its first-clear set; otherwise `pending` / `pending_floor` (first clear discarded). Logged `confirm_state_migrated`, once, at the first settlement after the release, covered by the prior/posterior digest binding. Test: both branches | W7b | ci | `scripts/laned/accept W7c` |
| W7d | **(rev6.18) B2: deleted — the satellite verifies through `audit_receipt` → `check_lane_d` (W9), the one lane D checker (F1).** ~~Satellite replay: an **independent** implementation,~~ no imports from producer eligibility, verdict or weight code (invariant 12). **(rev6.10)** Re-derives eligibility with the per-feed bound `b − e` from the signed per-window `min_target_ts`, `min_first_seen_ts` and `embargo_hours` (today it checks both against one bound, `cascade/satellite/lane_d_replay.py:891-895`); refuses a window `embargo_hours` above the signed config value; **(rev6.11)** ~~applies e = 0 to sub-hourly series (from the signed `freq`) and~~ **(rev6.12)** applies the signed `e` to every cadence and **(rev6.11)** checks the embargo-window share ≤ ~~`embargo_window_share_cap`~~ **(rev6.12)** 1 − `min_fresh_window_share` (re-derives each window's fresh / embargo class and the fresh-first trim; refuses a set below the share, `fresh_share_short`); ~~checks the confirmation set at `e = 0`;~~ **(rev6.11)** expects **no** confirmation stage (refuses a report with one, `confirmation_stage_present`) and re-derives crowning as first clear ∧ A10 ∧ gate; re-derives the W5j draw (seed, quotas, picks) from the block hash and the signed candidate list. **(rev6.14)** Re-derives `b` (forward_margin, default 0) and `j` (eval_delay_hours) from the signed T values and the config echo; re-classes fresh / embargo at `b`; refuses a set frozen before `j` (`frozen_before_eval_delay`). **(rev6.15)** Re-derives `context_below_armed` from the signed `armed_contexts` and each entry's declared context: refuses a report that skips an entry with an armed variant ≤ its context, or judges one with none | W7c | ci | `scripts/laned/accept W7d` (import-graph check included) |
| W8 | Emission: lane D in `lane_parts` / `lane_pool_shares` (`emission.py:330-383`, `lane_c.py`), court decay, house burn  **(rev5)** A deregistered court member's slot burns (B8). **C3 (D48)** decides where lane D's share comes from inside lane A; W8 also changes `lane_pool_shares` (`shared/lane_c.py:584`, returns a `(B, C)` tuple, callers `emission.py:361`, `lane_c.py:605`), a consensus signature at every caller. | W1.5 | ci | `scripts/laned/accept W8` |
| W8b | **(rev6.2) (rev6.4, accepted owner 2026-11-23)** Production wiring: today nothing outside tests calls `cascade.validator.lane_d.settle_lane_d`. The validator loop calls it at each lane D settlement boundary with: the boundary block hash (read-only chain read); the era's sentinel (W-S1 draw from the era-start hash); a bench `SweepSession` on the configured bench host (the W3b2/W5e gate inputs; an armed crown without them is a `bench_hold`); `expect_prior_sha256` from the last published receipt (W7c); the low-frequency check (W3b5); and the house/king state. After settlement: W3b3 `sweep_and_publish` for a new king; W-S3 `build_sentinel_board` with the era's kings list (from W7c state / W17); publish the signed report + receipt; lane D pay reaches weights (W8 emission) **only via the published report**. **Fail-closed:** any unreadable input ⇒ no settlement this tick (retry next tick), never a guessed input. **Restart-safe:** a re-call after restart is idempotent (W7c). **Weight loop:** lane D's bench work runs off the weight loop (W19) or is bounded so weight reasserts are never delayed past a stated bound (2026-11-22: weights went stale ≥ 2 h during a synchronous king-check sweep). **Unarmed:** byte-identical (invariant 1). Ships release-then-activate (external validators on netuid 91) | W5e, W3b5, W7c, W8, W3b3, W-S3, W17 | ci + testnet | `scripts/laned/accept W8b` |
| W8c | **(rev6.7, owner 2026-11-23)** Production `LaneDInputs` source: W8b's step calls `settle_lane_d` with a `LaneDInputs` from its `inputs` source, but nothing in production supplies one, so an armed lane D refuses `inputs_unavailable` at every settlement. W8c builds a `LaneDInputs` per armed boundary from signed or pinned sources only: (1) the queued entries from the W3 intake state and vault (`cascade/lane_d/`), revealed by the boundary; (2) each entry's, KING_D's and the house's forecasts on the boundary's frozen windows, computed on the lane D eval host by the W5d scorer (`cascade/eval/lane_d_scorer.py`) over checkpoints fetched through the W15 prefetch cache (`cascade/eval/lane_d_cache.py`) and W4 ingest, behind the W4 handshake; (3) the pinned `pool_d` snapshot(s) (W6d), each re-hashed against its pin, and their windows (W5a/W5b); (4) the house record from the W5f signed pointer, and the bench `SweepSession` (W3b*) for W8b's `bench`. **Deterministic:** the same boundary and sources give the same `LaneDInputs`. **Fail-closed:** each missing or unverifiable input is its own typed code (no inputs, no settlement this tick; W8b retries), never a guessed or default input. **Off the poll thread:** it runs inside the W19 worker's step; its chain reader is built on that thread. **No new chain reads beyond `ChainClient`.** **Unarmed:** no source is built and nothing is read (invariant 1); nothing reachable from `cascade.audit` imports it (invariant 12). Wired into `cascade/validator/main.py` through `build_lane_d_step`'s `inputs` and `bench` parameters. **(rev6.13, rev6.6 follow-up)** The source also computes the settlement's armed set: W5d `armed_record` over the pinned `laned_readiness/1` / `laned_t7/1` bytes named in the inputs TOML (`readiness`, `t7`, required, re-read per settlement), each digest forecast at `effective_context`, the record passed to `settle_lane_d(armed=)` via `LaneDInputs.armed` (W8b field, default `None`); missing/invalid documents or nothing armed fail closed (`readiness_unavailable`, `t7_unavailable`, `armed_invalid`, `no_armed_context`). **(rev6.15)** A digest whose `effective_context` is `None` (no armed variant ≤ its declared context) is not forecast: an entry or open record is then skipped by the settlement (`context_below_armed`, W7a); a miner KING_D without a forecast makes its cohort `king_unscored` (existing rule). The house declares no context and keeps its rule (4096 if armed, else the highest armed below, else the lowest armed), so it is always forecast. No new input error code | W8b, W5d, W15, W4, W3, W6d (`pool_d` is owner/forge data; tests use fixtures), W5f | ci + testnet | `scripts/laned/accept W8c` |
| W9 | Audit `check_lane_d` + `derive_weights` (**rev3:** re-derives the judged set, coldkey caps, both-clock eligibility, ~~the confirmation rule~~ and `rows_sha256`), independent of producer code. **(rev6.1)** Refuses any judged snapshot in which a series lacks `frontier` (absent or `null`) or a feed is not listed in `feeds` (W6b). It re-derives S0 presence (`absent_from_s0`), the backfill rule and mirror-lag eligibility (`over_lag`, `forecast_product`) from the snapshot itself, **without importing `cascade/validator/lane_d_forward`** (invariant 12; RV3 BLOCKING 1); its reason labels follow §1.3 (`publish_ahead` = first-seen fails, `late_mirror` = data time fails). **(rev6.10)** Re-derives the embargo-aware eligibility (`first_seen + e > b`, `data_ts > b − e`, S0/backfill at `b − e`) from the signed first-seen record (B12 roots) and the `e` pinned in the snapshot's `feeds` entries, never from the catalog; **(rev6.11)** ~~applies e = 0 to sub-hourly series and~~ **(rev6.12)** applies the pinned `e` to every cadence and **(rev6.11)** re-checks the embargo-window share ~~cap~~ **(rev6.12)** against `min_fresh_window_share` (re-derives fresh / embargo classes, the fresh-first trim and the share; refuses a frozen set below it, `fresh_share_short`); refuses an `e` above `[lane_d] embargo_hours` or a non-zero `e` on a feed W6fs marks publish-ahead; ~~checks the confirmation set under `e = 0`;~~ **(rev6.11)** expects no confirmation stage (refuses one) and re-derives crowning as first clear ∧ A10 ∧ gate; re-derives the W5j draw (seed, quotas, picks, source cap, `max_windows_per_settlement`) independently of `lane_d.py`. **(rev6.14)** Re-derives both times independently: `b = max(T) + forward_margin_hours` (default 0) for eligibility and the fresh / embargo class, and `j = max(T) + eval_delay_hours` for the freeze settlement; refuses a set frozen before `j` or a config with `b > j`. **(rev6.15)** Re-derives `context_below_armed` independently: an entry is skipped with it iff the report's signed `armed_contexts` has no variant ≤ its declared context; anything else is refused. **(rev6.1)** Wiring trap (RV3 note 10): the audit's `emission_vector` call (`audit/checks.py:2834-2841`, `:3660`) passes the second element only when a lane B or C pay record exists, so a lane D share with B and C off would be dropped and handed to lane A by normalisation; W9 must add the lane D pay record to that condition. **(rev6.1)** W9 decides invariant 12 for the shared lane D weight math (`lane_pool_shares` / `lane_parts` / `lane_d_vector`, RV3 note 11) before its tests are written: either re-derive lane D's carve and court vector without these functions, or record them in the plan as shared consensus code exempt from invariant 12 | W7c, W8 | ci | `scripts/laned/accept W9` |
| W10 | Miner CLI `cascade lane-d export \| verify \| submit \| ready \| status`, **W3a-chosen upload path only**. **(rev5, D43)** The commit reveals immediately (`blocks_until_reveal = min_blocks`), never at boundary − margin; a test asserts the lane D commit path never uses the lane A boundary timing | W3a, W3, W4 | ci + testnet | `scripts/laned/accept W10` |
| W11 | cascade-ops checks (§10.7) + follower patterns | W5g, W7c | ci (cascade-ops repo) | `scripts/laned/accept W11` (includes `follower_patterns.py`) |
| W12a | Docs and agent surfaces (§11.2 text surfaces, website; **rev3:** LANE_D.md FAQ on coldkey caps, ~~confirmation latency~~ **(rev6.11)** crown at the first verdict, both clocks, BF16; VALIDATOR.md on what satellites verify for the new report fields; OPERATOR.md on launch eval host, S8, first-seen recorder) | W10 | ci | `scripts/laned/accept W12a` (every §11.2 surface mentions lane D; links resolve; `llms.txt` lane list matches the status fixture) |
| W12b | `cascade status --json` and `discovery-check` lane D fields | W2, W17 | ci | `scripts/laned/accept W12b` |
| W13 | King publication (§11.1): Hippius Hub, R2, HF mirror, signed `champions/lane_d/index.json`, `laned-king-sync.yml`, cards, one-digest check  **(rev5)** A takedown path for the Hub and HF mirrors (licence or rights claim ⇒ unpublish the mirrors, keep the digest and verdict record). | W5e, W3a | ci + testnet | `scripts/laned/accept W13` |
| W14 | ~~**(rev5) blocked: needs owner (Proposed design change 4).**~~ **(rev6) Unblocked:** change 4 was accepted (owner 2026-11-21). Lane D-only fast kill: a `lane_d: {status: "absent"}` receipt state with a lane D-only burn. Today a missing lane D report makes satellites re-assert the old vector for one era (`stale_after_blocks = 3600`) and then burn the whole vector (`satellite.py:30-45`) | W7c | ci + testnet | `scripts/laned/accept W14` |
| W15 | Prefetch cache, LRU with KING_D and house pinned | W4 | ci + pod | `scripts/laned/accept W15` |
| W16 | Reference fine-tune script (miner kit) | W10 | ci (CPU tiny run) + pod | `scripts/laned/accept W16` |
| W17 | `status/lane_d.json` producer (queue, pending progress, king, board; **rev3:** per-entry reason enum including `coldkey_cap_settlement`, ~~`confirm_pending`~~ and reveal deadline). **(rev6.2, owner 2026-11-22)** Also produces the W18 vault ledger `laned_vault_ledger/1`: the current era, the vault entries with their statuses and `closed_era`, and the open disputes | W5b, W7c | ci | `scripts/laned/accept W17` |
| W18 | TTL sweeper: vault → owner-only archive → delete (live runs need D16) | W3a | ci (fs adapter) + owner | `scripts/laned/accept W18` |
| W19 | Lane D work runs **off the weight/poll thread**; the weight-loop keep-alive (#471) is in the release (invariant 6, T15) | W5e | ci + testnet | `scripts/laned/accept W19` (a fixture slow lane D step leaves weight-set timing unchanged) |

**(rev4) Lane B track (trainer-run, miner-chosen experts, suite targets; §12b)**

Every new `[lane_b]` key below ships with:

- a loader parse;
- the **drop-when-default** digest rule (unarmed digests don't move);
- `ALLOWED["lane_b"]` coverage, which is overlay-settable (`remainder` stays
  release-only);
- round-trip and mutation tests in the named WP.

| WP | What | Keys / fields | Depends | Runs in | Accept |
|---|---|---|---|---|---|
| WB10 | Lane B config keys (parse, digest drop-when-default, overlay, round-trip + mutation tests). **(rev5)** `contamination_hashset` is release-only (added to the `netconfig.py:210` tuple; test: an overlay setting it is refused); new D42 keys `funded_pod_skus`, `max_leg_hours`, `max_leg_seconds`, `guard_factor` | `suite_targets`, `contamination_hashset`, `contamination_min_hits`, `suite_target_max_repo_mb`, `min_experts_frac`, `max_experts_frac`, `router_rows`, `traffic_steal_bound`, `open_rule`, `sentinel_target_rule`, `sentinel_max_defer_eras` | W0c | ci | `scripts/laned/accept WB10` |
| WB11 | Contamination hash-set builder: exact + normalised sub-window hashes from the pinned suite data; versioned artefact + sha256. **(rev6.2)** Normalisation is schema 2 (`cascade_contamination_hashset/2`, `cascade-zgrid/2`) with RV6's low-information exclusion, per §12b.1 (#617/#621, #632) | `contamination_hashset` | WB10 | ci + owner (pin) | `scripts/laned/accept WB11` (hash set reproducible from the suite identities) |
| WB12 | Routing information producer (LB3): extend PR #481 telemetry to per-expert counts at phase level; per-suite context-only affinity on the bench host; digest-pinned `status/lane_b_routing-<king8>.json` | `status/lane_b.json` `routing_info` pointer + `stale_since_block` | PR #481 merged | ci + pod | `scripts/laned/accept WB12` |
| WB13 | Intake contamination check (LB1): AST literals, `config.json`, non-code bytes vs the hash set; typed `benchmark_series_embedded`; `suite_target_max_repo_mb` (D38) | receipt field `contamination: {hashset_version, result, suite, window_id}` | WB11 | ci | `scripts/laned/accept WB13` (TB10–TB12) |
| WB14 | Declared-rows parsing + per-layer clamps (LB2): **(rev5)** keys = MoE ordinals 0–6 (not block indices); clamp minimum only on layers with ≥ 1 row; validated vs `experts_per_layer`; codes `declared_rows_out_of_range`, `declared_rows_bad_layer`, `declared_rows_count` | manifest field `declared_rows` | WB10 | ci | `scripts/laned/accept WB14` (TB13, TB15, TB15b) |
| WB15 | Row claims + overlap deferral: earliest reveal; both modes coexist (declared first, then ESFT excluding held rows) | `status/lane_b.json` `deferred: rows_held` | WB14 | ci | `scripts/laned/accept WB15` (TB16) |
| WB16 | Router-row decision (D40): (a) the frozen path is today's backend, with a test that it stays frozen; if (b): unfreeze the declared router rows, router byte-check, `traffic_steal_bound` guard. **(rev5)** (b) is a worker-image change (shared with lane A): 5-pin repin + wall gate + DEC-CA-0067 qualification, after the scored EMA constant is fixed  **(rev5)** Depends on W-EMA. | receipt field `traffic_share_delta` (b only) | WB14 | ci + pod | `scripts/laned/accept WB16` (TB14) |
| WB17 | Opening rule (D41, LB4): `open_rule` implementation; driver state records the rule and the satisfying evidence | `lane_b_driver` state `gate_rule`, `gate_evidence` | WB10 | ci + testnet | `scripts/laned/accept WB17` (TB17–TB18) |
| WB18 | Sentinel interaction (D35/LB5): suite-target deferral, `sentinel_max_defer_eras`, no charge on deferral. **(rev6.2, owner 2026-11-22) Follow-up WP note:** while a funded entry is deferred (`sentinel_target` or `rows_held`), its payer key in `PayerKeyVault` (`cascade/funding/vault.py`) has its TTL refreshed on each deferred tick, the same rule as the entry (RV6, #629/#633), so a deferral never outlives the key the leg rents on. No WP owns the vault; the follow-up adds the vault hook and its own test (flagged, not pinned, in `tests/accept/wb18/README.md`) | `status/lane_b.json` `deferred: sentinel_target` | W-S1, WB10 | ci | `scripts/laned/accept WB18` (TB19–TB20, TS4) |
| WB19 | **(rev5)** Expert-leg measurement (LB6), **per candidate GPU type**: expert-only step rate, leg overhead split from training, peak memory at contract micro-batching, image qualification; ~~plus one full lane A leg on 8×H100 (is 13,600 s training-only?)~~ **(rev6.5) full lane A leg skipped, owner 2026-11-23**. **(rev6.16) Partial evidence allowed (owner 2026-11-23):** coverage may be the only failing check; measured rows are usable, unmeasured types stay `gpu_rate_unmeasured` (WB22) | `evidence/wb19.json` per-GPU table | — | pod (owner yes, OA8) | `scripts/laned/accept WB19 --evidence evidence/wb19.json` |
| WB22 | **(rev5)** Lane B GPU list (D42): `[lane_b] funded_pod_skus` separate from `[round]`; `GpuChoice` reads it for lane B legs only; trainer-side per-GPU step-rate table from WB19; audit re-derivation pinned to the recorded `gpu_name` | loader parse + round-trip + mutation test; lane A choice unchanged | WB10, WB19 | ci | `scripts/laned/accept WB22` (TB23, TB24) |
| WB23 | **(rev5)** Miner-chosen leg length (D42, D42b): `train_hours` on the funding request (0.5–`max_leg_hours`, 0.5 h steps); steps fixed at admission and signed in the leg record; guard = `train_hours × guard_factor` + overhead, capped at `[lane_b] max_leg_seconds` (lane B contract derivation; `[training] max_train_seconds` untouched); expected landing boundary returned to the miner; validator check of recorded steps and guard | keys `max_leg_hours`, `max_leg_seconds`, `guard_factor`; `status/lane_b.json` shows steps, GPU and landing boundary | WB10, WB22 | ci | `scripts/laned/accept WB23` (TB25–TB27) |
| WB24 | **(rev6.3, proposed)** Lane B miner CLI. Today `cascade lane-b fund` / `lane-b status` (`cascade/miner/lane_b.py`) do not send the declared-rows header, so the WB20 docs have miners hand-build that request. `cascade lane-b fund` gains `--declared-rows <file>` (builds and signs the WB14 `X-Lane-B-Declared-Rows` header and its signed `rows=` tag; refuses locally with the same `DECLARED_ROWS_CODES` the intake uses), `--target suite:<s>` (suite targets, WB18) and `--train-hours` (WB23, once built). `cascade lane-b status` shows the new deferral and refusal reasons (`rows_held`, `sentinel_target`, the `gate_closed_reason` values, the WB13 contamination codes incl. `contamination_budget_exceeded`) from `status/lane_b.json` and the intake replies. New `cascade lane-b verify-repo <dir>` runs the WB13 scan (`scan_repo`) locally against the pinned hash set (no upload; never executes the repo) so a miner can check before funding. Never prints keys; unarmed defaults unchanged | CLI flags `--declared-rows`, `--target suite:<s>`, `--train-hours`; subcommand `verify-repo` | WB13, WB14, WB15, WB17, WB18 (WB23 for `--train-hours`) | ci | `scripts/laned/accept WB24` |
| WB25 | **(rev6.8, owner 2026-11-23)** Multi-GPU lane B expert legs. Today `train_pyramid` (`cascade/trainer/pyramid_trainer.py`) forces `world = 1` for an expert-slice leg (`gang_size(trainer.device) if slices is None else 1`), so a lane B leg trains on GPU0 of an 8× pod (WB19 on 8×H200: 0.055 steps/s, the same as an RTX 4090). WB25 trains lane B expert legs on every GPU of the rented layout through the existing `pyramid_dp` `Gang` (micro-batches dealt round-robin, sums all-reduced, rank 0 owns the corpus and the optimizer), extended for `ExpertSlices`: the helper ranks hold the same compact selected-row leaves; the optimizer sharding and the gradient all-reduce cover only the trainable (selected) expert rows; the router and the routing bias stay frozen (WB16, D40 option a). The leg is identical to the 1-GPU leg up to floating-point summation order, and frozen tensors and unselected rows are exactly the king's. The leg record keeps the GPU count (`gpu_n`; WB22's audit re-derivation is pinned to the recorded `gpu_n`); `CASCADE_TRAIN_GPUS` may lower it. Lane A legs (no slices) are byte-for-byte unchanged and keep the same Gang path. No dependency, `[training]` or contract-digest change | leg metric `gpu_n` | WB16, WB19 | ci + pod | `scripts/laned/accept WB25` |
| W-EMA | **(rev5)** Score the raw cooled endpoint on every leg (lanes A and B **(rev6.20b: lane A only; every lane B leg trains with no cooldown and scores the σ 0.10 power EMA, recorded as the expert entry's `scored_weights`, no activation key)**): `power_ema_active` → raw for warm-start legs; keep writing `weights_ema005`/`weights_ema010` unscored; main-validator EMA telemetry off the critical path (§8). Blocks every repin ≥ rc16 and D40(b) | worker image (`pyramid_trainer.py`) | — | ci + testnet | `scripts/laned/accept W-EMA` |
| WB20 | Lane B docs (MINER, INTERFACE, AGENT_MINING, VALIDATOR) | — | WB13–WB18 | ci | `scripts/laned/accept WB20` |
| WB21 | DEC-CA-0071 amendment (this PR drafts it; owner accepts) | — | — | ci + owner | `scripts/laned/accept WB21` |
| W-S1 | **(rev4: now in §3)** §16 sentinel draw + lane A gate rule behind `sentinel_from_block` + receipt field + `[v2]` overlay keys (**(rev5)** no shadow logging, D45). **(rev5) Changed consensus code:** the bench gate (`cascade/eval/bench_gate.py`) **and `king_check_gate` (`cascade/validator/king_check.py:93`)**, both of which must drop the sentinel suite from the rule; block-gated by `sentinel_from_block` | `[v2] sentinel_from_block`, `sentinel_suites`, `sentinel_ratchet`, `sentinel_tau` | W0c | ci | `scripts/laned/accept W-S1` (TS1–TS3, TS5) |
| W-S2 | **(rev4: now in §3)** lane B guard/target exclusion for the sentinel. **(rev6.2, owner 2026-11-22) Follow-up:** a warm-start acceptance made at an era-init block is governed by the sentinel of the era that init block opens; applies to the acceptance callers (trainer `accept_candidates`, validator `verify_acceptance`) (§16) | — | W-S1 | ci | `scripts/laned/accept W-S2` (TS4) |
| W-S3 | **(rev4: now in §3)** sentinel numbers on the status boards | — | W-S1 | ci | `scripts/laned/accept W-S3` |


**Review WPs.** After every 5 build WPs reach **L3** (**(rev6)**, §17.1; counting W/R build WPs, not
owner execution WPs), a **fresh-context review WP** runs. RV1, RV2, … are
read-only reviewers against main with the plan as the spec. Accept is
`scripts/laned/accept RVn --evidence evidence/rv<n>.md`: no unresolved
BLOCKING item, and each finding is either fixed or filed as a WP.

**Execution WPs (stage gates, §5)**

| WP | Stage | Runs in | Accept |
|---|---|---|---|
| X3 | S3 testnet matrix T1–T29 | testnet | `scripts/laned/accept X3 --evidence evidence/s3_matrix.json` (each row's recorded pass, 6 clean eras, ≥ 1 crowning, ≥ 1 house retake) |
| X4 | S4 release + **(rev5)** launch eval host bring-up (D11; valeval-laned only at S8) + dry-run + announcement + staged revert overlay | owner | `scripts/laned/accept X4 --evidence evidence/s4.json` |
| X5 | S5 capability-release upgrade (eval host → validator → intake → trainer; satellites) | owner | `scripts/laned/accept X5 --evidence evidence/s5.json` (lane A unchanged ≥ 2 settlements; `--netconfig-policy` lists `lane_d`) |
| X6a | **(rev5)** S6 readiness probe and activation evidence (observed `cfg-` stake before the activation block); D19 test entries moved to testnet **(rev6)** One stage: S6 (D45). | owner | `scripts/laned/accept X6a --evidence evidence/s6a.json` (**(rev6)** observed `cfg-` stake ≥ 2/3 before `effective_block` (R4); webhook test alert delivered; from the block, the first paid settlement's vectors match on ≥ 2 validators and satellite replay matches. No "≥ 2 eras of reports" before activation) |
| X7 | S7 operate | owner | `scripts/laned/accept X7 --evidence evidence/s7.json` |

### 3.1 Build rules

1. **Tests first.** Acceptance tests are written first in `tests/accept/<wp>/`
   ~~and are **read-only to builders**. CI fails a PR that edits
   `tests/accept/` together with that WP's implementation.~~
   **(rev6.17, owner 2026-11-23)** and MAY land in the same PR as the code.
   "First" is proved by CI, not by PR order: the WP's new tests, taken from
   the PR head, must fail on the base's code for a right reason, then pass
   on the head with the full Accept (`accept <WP> --combined`). A new test
   that already passes on the base proves nothing and is refused, unless the
   WP is a declared verification WP. A PR may add `PATHS` entries and change
   those files in the same PR (scope = base PATHS + added entries; removals
   still count from the base). Deleting or weakening an existing assertion,
   or dropping a `PATHS` entry, needs the `laned:tests-correction` label.
2. **Mock only external services**, never the component under change.
3. **No placeholders.** No `TODO`, stubs, `pass`, `NotImplementedError`,
   hardcoded sample values or mocks outside `tests/`. The stub scan runs in
   every Accept.
4. **Every `[lane_d]` key** ships with a loader parse, an `ALLOWED` entry, a
   round-trip test and a mutation test. `unread_keys.py` fails on any key no
   parser reads.
5. **Every lane D log line** has a follower pattern, checked by
   `follower_patterns.py`.
6. ~~**(rev6) Each WP has a tests-only PR (`tests/accept/<wp>/`) followed by
   ONE implementation PR.**~~ **(rev6.17) One PR per feature** (rule 7);
   tests-only and tests-correction PRs remain available, not required.
   Explicit paths, no stacks, owner-merged;
   dependents start only after the dependency's code is **on main** (verify `git log
   origin/main`, never a MERGED badge) **(rev6.17:** or ride in the same
   feature PR**)**. ~~This matches rule 1: the tests land
   first and alone, and the implementation PR never touches them.~~
   **(rev6.2, owner 2026-11-22)** ~~Implementation, tests-correction, evidence
   and review PRs stay **one WP per PR**.~~ **(rev6.17)** Evidence and review
   PRs stay one WP per PR. A **tests-only PR** (new acceptance
   tests, no implementation yet) MAY carry several WPs' `tests/accept/<wp>/`
   directories, with one `laned:<WP>` label per WP; each WP's expect-fail (or
   verification) accept run is executed and ledger-recorded separately, and the
   whole PR fails if any WP's check fails. **(rev6.17)** No cap, and built
   and unbuilt WPs may mix (each runs its own mode).
7. ~~**(rev6.3, owner 2026-11-23) Implementation PRs MAY batch up to 3 WPs**
   when (1) their PATHS sets are disjoint, (2) none depends on another in the
   batch, and (3) each WP's tests are already on main. Each WP in the batch is
   still checked on its own: scope (its PATHS only, files attributed per WP),
   its own accept, and CROSS; the ledger records one row per WP. A red WP is
   split out of the batch by the builder (the others proceed); WPs whose PATHS
   overlap stay serial. Tests-corrections found in one pass may likewise be
   batched in one PR (one dir per WP). Labels: one `laned:<WP>` per WP in the
   batch. This narrows rule 6's "one WP per PR" for implementation and
   tests-correction PRs; evidence and review PRs stay one WP per PR.~~ (Former
   rules 7–8 are now 8–9.)
   **(rev6.17, owner 2026-11-23) One PR per feature.** A PR whose labels are
   all `laned:<WP>` may carry every WP a feature touches, with tests and code
   together: no WP cap, no disjoint-PATHS or no-dependency refusal, no
   `coupled_batches` declaration. Each WP is still checked on its own: its
   scope (base PATHS + added entries; a file in several WPs' PATHS counts for
   each), its own Accept or combined run, CROSS, and one ledger row. Every
   label must own a changed file or test. Every WP whose code the PR changes
   must carry its label: the squash merge is a WP's L2 commit only if the PR
   carried `laned:<WP>` and changed a non-test path in its PATHS (§17.1).
8. **A fresh-context review WP** after every 5 build WPs reach L3 **(rev6)**.
9. **Blocked is acceptable; faked completion is not.** If a WP can't be fully
   built, the builder stops and records the blocker in `evidence/blocked/<wp>.md`.
10. **(rev6.4, owner 2026-11-23) CI-first testing for loop agents.** An agent
    runs **only its own WP's accept dir** (`tests/accept/<wp>/`) locally.
    Every other suite (unit shards, receipt-replay, CROSS) runs in CI, and all
    of them are **required green before merge**. Why: the production box
    (4 cores / 7 GB) also scores the validator, so a local full suite stalls
    scoring.

### 3.2 Invariants

These are pinned by tests in the WPs named.

1. **Unarmed ⇒ byte-identical.** With `from_block = 0`, every receipt,
   manifest, verdict and weight vector matches today's. (W1.5, W2, W8)
2. **Armed ⇒ lane A verdicts and court unchanged.** Lane A's share-of-total
   changes by design, and the receipt gains lane D fields. (W1.5, W8, W9)
3. **No miner bytes execute.** Only repo code imports; the file allowlist is
   exact. (W4)
4. **Forward-only cannot be violated.** A scored window failing §1.3 for
   max(T_E, T_K) is an audit failure. (W5a, W9)
5. **Restarts are idempotent.** No re-score, no double crown. (W7c)
6. **Weights never wait for lane D** (off the poll thread; keep-alive). **(rev3)** #471 is merged (0c52d07) but not deployed (rc37 lacks it); it ships with S2b.
   (W19, X3 T15)
7. **Replay compares structured fields only.** (W7d)
8. **Targets never leave the box.** (W6c; see the proposed design changes in
   the changelog)
9. **Armed with `share = 0` ⇒ the weight vector is identical to unarmed, and
   a valid signed lane D report is still produced.** (W1.5, W8)
10. **Every "unchanged when unarmed" test has a paired "armed ⇒ non-empty
    lane D effect" test.** (every WP; enforced by `scripts/laned/accept`)
11. **`pending`, `hold` and `burn` always carry a typed reason enum,** and
    tests assert the exact reason. (W5g, W7a)
12. **Audit and satellite-replay code never import the producer's
    eligibility, verdict or weight functions.** (W7d, W9; import-graph check)

**Services touched:**

- the validator;
- the lane D intake service (new);
- the audit;
- the CLI;
- cascade-ops;
- the eval host;
- the `pool_d` builder;
- `deploy_gate`;
- **the trainer**, through the capability release only. The overlay resolver
  in `netconfig.py` gains `lane_d`. There's no trainer behaviour change, image
  or repin, and no `cascade/funding|trainer|…` path changes
  (`scripts/deploy_gate.py:55`). §14.5.

**(rev5) Blast radius by path** (`deploy_gate.py:41-58`):

- **CONSENSUS diffs** (every validator must compute them identically, so each
  is block-gated and ships in the capability release):
  - W2 (`cascade/shared/config/`);
  - W4 (`cascade/eval/checkpoint_guard.py`);
  - W5d (`cascade/eval/lane_d_scorer.py`);
  - W7–W9, including W8 (`cascade/shared/emission.py`) and the
    `cascade/validator/lane_d*.py` / `cascade/audit/` halves.
- **Worker-image changes:** any change to `uv.lock` or `pyproject.toml`
  (`WORKER_IMAGE`), and any on-pod trainer code (lane B D40(b), below). Each
  is a repin with 5 pins and the wall gate, never mid-seed-epoch.
- **Lane B (rev5, B4 and owner note).** Lane B shares lane A's worker image
  (`[round] funded_pod_image` and `[v2.training] train_image_digest`, both
  rc15 `5d41d94b…`; `[lane_b]` has no image key of its own).
  - Dispatch path: `lane_b_driver._dispatch_funded` (`:1297`) → `_run_funded_leg`
    (`:1329`), the lane A funded-leg path, so the same image pin.
  - **Declared rows need no image change.** The trainer passes the selection
    to the pod as data: `--selection-json` (`worker.py:119, 218-227`), keyed by
    MoE ordinal. The on-pod code is unchanged.
  - **D40(b) router unfreeze does need one.** It changes
    `expert_training.py`, which runs on the pod under `worker.py --role
    expert`. That's a **shared-image repin that also moves lane A**, with
    DEC-CA-0067 qualification.
  - Any image ≥ rc16 turns on power EMA for lane A at the next warm-start
    epoch (`POWER_EMA_SIGMA_RELS` is image-only). So **(rev5) D40(b) and every
    repin to an image ≥ rc16 depend on W-EMA** (§8): the next image scores the
    raw cooled endpoint on every leg.
  - **(rev6.20)** Lane B EMA / no cooldown is on-pod code too
    (`pyramid_trainer.py`, `worker.py`), so it ships in a shared-image repin;
    **(rev6.20b: no key; it applies to every lane B leg once lane B rolls
    out)** ~~it changes nothing until `[lane_b] ema_no_cooldown_from_block` is typed~~
    (release-then-activate, after the repin).

---

## 4. Testnet plan (netuid 259)

**Setup**

- The testnet stack (§10.3) is **fully separate from the prod box**.
- `[v2] from_block = 8190000`; `epoch_blocks = 150`; `era_settlements = 4`.
  Lane D arms at a 600-block boundary at least one era out.
- **(rev3) Floors on testnet.** A daily pool can't meet the mainnet floors
  inside a testnet day, so the testnet forward path uses **`pool_mkt`**
  (`.github/workflows/publish-mkt-pool.yml`, cron `30 */6 * * *` since #480;
  `cascade/eval/marketcast.py`) and/or **testnet-only floors**.
  - **(rev3, verified)** pool_mkt's *scheduled* runs **skip while
    `[marketcast] from_block = 0`** (`publish-mkt-pool.yml:84-95`). That's
    the current value in both `chain.toml` and `chain.testnet.toml`, so
    today pool_mkt is published only by **manual `workflow_dispatch`**.
  - Using it for lane D testnet needs either a dispatch per run (an owner
    step), or a lane-D-specific schedule. The latter is a new WP, `R15`, and
    must not arm MarketCast. **The
  mainnet floor values (1000 windows / 20 sources) are therefore untested on
  testnet.** B2 measures them on mainnet snapshots instead.
- **The testnet `pool_d` builds hourly**, not daily, so the forward path runs
  inside a testnet day. Margin is 30 min.
- **The testnet house is a deliberately weakened checkpoint** (e.g. an early
  lane A leg), so an honest fine-tune can beat it deterministically.
- "House changes" are simulated by swapping the testnet house pointer through
  the testnet overlay. No real 8×H100 testnet lane A dethrone is needed.
- Test hotkeys are owner-registered; **never** the mainnet wallet.

**Matrix**

| # | Case | Expected |
|---|---|---|
| T1 | House only, 2 eras | KING_D = house; slot burns; lane A verdicts and court unchanged |
| T2 | Exact copy of the house | Refused at dedup (no eval spend) |
| T3 | Honest fine-tune of the weakened house on public data | Wins forward duel and gate; crowned; published; lane D share paid |
| T4 | Domain-overfit model | Loses, or blocked by the drawn suite, with the reason logged |
| T5 | Malformed: extra `.py`, edited `model.py`, wrong `config.json`, `.pt`/`.bin`/pickle, wrong dtype or shape, NaN, oversize | Each refused at ingest with a specific code; nothing executed |
| T6 | Second entry for a hotkey; coldkey cap; upload after `ready` (overwrite) | Refused; refused; `digest_mismatch` at ingest |
| T6b | **(rev3)** Three entries from one coldkey ready in one settlement; one coldkey changed between ready and reveal; an unresolvable coldkey | 2 judged, 1 skipped `coldkey_cap_settlement` and carried; `coldkey_changed` refusal; `coldkey_unresolved` alert, then closed after the bound |
| T7 | Each context variant: wall time and memory on **(rev5)** the launch eval host for 4k/8k/16k (valeval-laned for 32k–128k ~~at S8~~ **(rev6.6:** once provisioned, W6-S8**)**); 4k model vs declared-128k copy on ≤ 4096-history windows. **(rev6.6)** T7 is gate (2) of each variant's automatic arming (§1.1): it runs per variant on the eval host that scores lane D, and its pass/fail is recorded as evidence the main validator cites in the signed report. Also: an entry declaring an unarmed variant is scored at ~~the highest armed variant~~ **(rev6.15)** the highest armed variant below its declared context (skipped `context_below_armed` when none is below), and the same entry scores at its own context once that variant arms | Identical scores on short windows; a variant is armed only if it fits the budget **(rev6.6:** ≤ 50% of a settlement, and `min_long_series` met**)**; unarmed-declared entry scored at ~~the highest armed variant~~ **(rev6.15)** the highest armed variant below its declared context, never above it |
| T7b | Gate: two validators derive the same suite from the block hash; regression vs KING_D or vs house blocks; **(rev5)** the era's sentinel suite swept and published only, never gating | As stated |
| T7c | 2.5 GB (BF16, rev3) direct upload (multipart or single presigned PUT per W3a); prefetch; box RAM and disk flat | As stated |
| T8 | **(rev3)** Forward rule, both clocks: **(rev6.1, labels corrected to §1.3 and `lane_d_forward.py:103-110`)** data-time-only pass but first-seen fail (publish-ahead, `publish_ahead`: a future-stamped value our mirror saw before T); first-seen-only pass but data-time fail (late mirror, `late_mirror`: data old, mirror saw it late); both fail (`before_margin`); a series lacking `frontier` or a feed not listed in `feeds` (refused); a feed over `max_mirror_lag_hours`; entry and king T values in both orders; a forecast-product feed; a backfilled partition | **(rev5)** Eligible only if **both** clocks (data time and first-seen) are after wall(max(T_E, T_K)) + `forward_margin` for every target point, and the feed's mirror lag ≤ `max_mirror_lag_hours`; the late-mirror, publish-ahead, over-lag, forecast-product and backfill cases are all ineligible; the snapshot-absence rule is never used for verdicts. **(rev6.10)** All of the above at `embargo_hours = 0` (verdicts byte-identical to the rev3 rule); embargo cases are T8e |
| T9 | Three entries freezing in one settlement with different T | Intersection set; latest-T deferred if below floors; max-T crowns the best |
| T9b | ~~**(rev3)** Clear, then a house refresh before confirmation~~ **(rev6.11)** Superseded (no confirmation duel); replaced by: a `confirm_pending` testnet state loaded after the release | ~~Per D20: restart vs new king (default) — first clear discarded, recorded in state and report~~ **(rev6.11)** W7c migration: same king ⇒ crown candidate on the first-clear set; changed king ⇒ `pending`; logged `confirm_state_migrated` |
| T8e | **(rev6.10)** Embargo-aware forward rule at `e = 48`: a point with data time in `(b − 48 h, b]` and first-seen in `(b − 48 h, b]`; data time ≤ `b − 48 h` (old backfill seen late); ~~**(rev6.11)** a 5-min series on a feed with catalog 48 (⇒ e = 0);~~ **(rev6.12)** a 5-min series on a feed with catalog 48 (⇒ e = 48); a 15-min window whose target straddles `b` (mixed); a set at S1 with no post-submission windows (F = 0); ~~embargo windows above `embargo_window_share_cap`;~~ **(rev6.12)** embargo windows above F (share 0.5); a set meeting the floors only with embargo windows > 50%; a fresh-first draw with surplus fresh and embargo candidates; a catalog `embargo_hours` of 0, of 100 (above the global), missing, negative; a publish-ahead feed with a non-zero catalog value; ~~the confirmation set;~~ a settlement before ~~`b`~~ **(rev6.14)** `j`; e = 0 vs the rev3 verdicts on a fixture pool; **(rev6.14)** a point with data time and first-seen in `(T, T+6 h]` at the default margin 0; a settlement in `[b, j)`; a config with `forward_margin_hours > eval_delay_hours`; the share bound at s = 0.75 | Eligible iff `first_seen + e > b` and `data_ts > b − e` for every point; the backfill case is `late_mirror`; ~~**(rev6.11)** sub-hourly ⇒ e = 0 whatever the catalog;~~ **(rev6.12)** every cadence ⇒ `e = min(catalog, 48)`; the mixed window is eligible and flagged `embargo_window`; F = 0 ⇒ no embargo windows kept ⇒ not frozen, `pending` (`fresh_share_short`); embargo windows above F trimmed last-drawn first; a floors-only-with-> 50%-embargo set ⇒ not frozen, share never relaxed; the fresh-first draw picks all fresh candidates before any embargo one in a cell; ~~embargo windows over the cap dropped deterministically;~~ catalog 0 ⇒ rev3 rule for that feed; 100 clamped to 48; missing ⇒ 48; negative ⇒ `unlisted_feed`; publish-ahead feed ⇒ `e = 0`; ~~the confirmation set uses `e = 0`;~~ nothing frozen before ~~`b`~~ **(rev6.14)** `j`; **(rev6.14)** the `(T, T+6 h]` point is fresh; the `[b, j)` settlement freezes nothing (`before_eval_delay`); `b > j` refused at load; s = 0.75 keeps E ≤ F/3; e = 0 verdicts byte-identical to rev3; W9 and W7d agree with the producer on every case |
| T9d | **(rev6.10)** Seeded per-bin draw: same boundary hash twice; king vs two challengers in one cohort; a cell below its share; one source above `max_source_share`; budget below `min_windows` ~~after confirmation draws~~; a pending entry across two settlements; ~~a confirmation draw overlapping the first set~~ | Identical draws for identical hashes and for every cohort member; the small cell taken whole and the surplus re-split; excess source windows dropped with no refill; `window_budget` deferral; a fresh draw per settlement until freeze and the frozen set equals the freeze-settlement draw; ~~confirmation windows share no target point with the first set;~~ W9 and W7d re-derive the same picks |
| T10 | House pointer swapped to a stronger checkpoint | House retakes; slot burns; prior miner tails keep paying |
| T11 | Validator restart mid-judgement | No re-score, no double crown |
| T12 | Satellite replay + `cascade-audit`; one tampered row | Identical weights; `check_lane_d` passes; tamper fails |
| T13 | Lane A verdicts and court before and after arming | Identical |
| T14 | **(rev3)** Null FPR per entry for each null-pair type (prior-king pairs, seed replicas, perturbed king), ~~confirmation-duel FPR~~ **(rev6.11)** single-test FPR (no duel), per-era FPR at k = 32 (S1: B1 / `scripts/null_fpr.py`) | **S4 gate:** per-era FPR ≤ the owner target (D24, OA3) at the chosen `d_margin` |
| T15 | 2 h lane D sweep across weight intervals | No weight gap > `weight_set_interval_blocks` |
| T16 | Front-run: published house resubmitted, plus a 1-ulp perturbation, in the crowning settlement | Exact refused; perturbation loses to the house |
| T17 | Checkpoint crafted to make a suite partial or NaN | `checkpoint_fault`; KING_D not held |
| T18 | Entry `ready` in an era's last settlement; `until_block` with entries pending (set by overlay) | Carries across eras; at `until_block` pending entries close unjudged and unpaid |
| T18b | **(rev3)** An entry pending at `max_reveal_delay_days` | `closed_unjudged: reveal_delay_exceeded`, logged, slots freed |
| T18c | **(rev5)** A pending entry's hotkey deregisters; a prior king deregisters | Entry voided `deregistered`, slot freed; the prior king's slot burns; satellite replay agrees |
| T18d | **(rev5)** Entry readied but not revealed within `ready_ttl_blocks` | `ready_expired`, open slot freed, nothing judged |
| T18e | **(rev5)** One source supplies more than `max_source_share` of a frozen set's windows | Excess windows from that source dropped deterministically (seeded by the settlement); the set re-checks its floors |
| T4b | **(rev3)** A model that regresses only on low-frequency series | Blocked by `lowfreq_regression` |
| TB10 | **(rev4)** Generator embedding an exact GIFT/BOOM/TIME/FEV test window as a literal | Refused at intake, `benchmark_series_embedded`, before any compute |
| TB11 | **(rev4)** Same window scaled ×3.7, offset +100 and rounded to 3 dp, in `config.json` | Refused (normalised match) |
| TB12 | **(rev4)** A clean generator; and a hash-set version change | Passes; the receipt records the hash-set version used |
| TB13 | **(rev4)** Declared rows out of range for a 128-expert layer (ordinal 0); **(rev5)** a key outside MoE ordinals 0–6 (e.g. 7, or block index 8) | `declared_rows_out_of_range`; `declared_rows_bad_layer` |
| TB14 | **(rev4, only if D40 (b))** Router-row delta pulling non-target routing toward its rows by > `traffic_steal_bound` | Refused, `traffic_stealing` |
| TB15 | **(rev4)** Declared counts at each layer size outside 4..19 / 8..38 / 16..76, **(rev5)** on layers with ≥ 1 declared row | `declared_rows_count` |
| TB15b | **(rev5)** Rows on 2 of 7 layers, each within its clamp, 0 rows on the other 5 | Accepted (ESFT-consistent) |
| TB16 | **(rev4)** Two entries on overlapping rows | The earliest reveal holds; the other shows `deferred: rows_held` with the holder and expected release |
| TB17 | **(rev4)** Opening rule (D41 default) on fixtures: tokens met + king-check pass | Gate opens; driver state records rule + evidence |
| TB18 | **(rev4)** Tokens met, king check failing (or tokens short) | Gate stays closed |
| TB19 | **(rev4)** Suite-target entry on the current sentinel suite | `deferred: sentinel_target`, not charged; judged in a later era |
| TB20 | **(rev4)** Same entry past `sentinel_max_defer_eras` | Judged on its target, logged |
| TB21 | **(rev4)** No declared rows | ESFT selection excluding held rows (both modes coexist) |
| TB22 | **(rev4)** Lane B unarmed with all new keys at default | Byte-identical digests, receipts and weights |
| TB23 | **(rev5)** Lane B leg on a cheap GPU in `[lane_b] funded_pod_skus` | Rents that type; `gpu_name` recorded; a lane A leg requesting the same type is still refused |
| TB24 | **(rev5)** GPU type in the lane B list with no measured step rate | Funding refused with a typed reason; no rental |
| TB25 | **(rev5)** `train_hours` = 6.0, and = 6.5 | 6.0 admitted with steps from the per-GPU rate and the guard ≤ `max_leg_seconds`; 6.5 refused at funding, nothing charged |
| TB26 | **(rev5)** 6 h leg funded late in an era | Held to the next era's pre-train window (cross-era admission); landing boundary shown to the miner |
| TB27 | **(rev5)** Leg record whose steps or guard exceed the consensus caps | Validator refuses the entry; satellite replay agrees |
| TS1 | **(rev4: now in §4)** Sentinel draw on two validators; recorded value replayed when the hash is unreadable | Identical |
| TS2 | **(rev4: now in §4)** Lane A with sentinel armed | Three-suite verdicts; sentinel-only regression blocked iff ratchet on |
| TS3 | **(rev4: now in §4)** Lane A sentinel unarmed | Byte-identical |
| TS4 | **(rev4: now in §4)** Lane B entry targeting the sentinel | Deferred (D35) |
| TS5 | **(rev4: now in §4)** Draw determinism and uniformity | As stated |
| T19 | Eval worker on stale code | Handshake mismatch ⇒ CRIT; nothing scored |
| T20 | Hippius primary write fails, R2 succeeds | Alert fires |
| T21 | Kill switch through the signed overlay | Share returns to lane A from the stated era; no release needed |
| T22 | Readiness note of the exact mainnet shape plus the `cfg-` segment | No pending segment trimmed |
| T23 | Old-code satellite with a new main validator | Matches until lane D arms; burns only after `from_block`; recovers after upgrade |
| T24 | Old-code trainer with a `[lane_d]` overlay published | Trainer refuses that overlay. Proves why the trainer is in S5, and that lane A stays safe while it waits |
| T25 | Upgrade from a **copy of mainnet state** (`validator_state.db`, `activation_*`, `netconfig_state`, `cascade_state.json`) | Clean start; identical next verdict |
| T26 | Shipped chain.toml through every validator and trainer startup gate at the live head | Starts (the #457/#460 class) |
| T27 | Revert overlay; stop-publishing path | **(rev5)** Overlay kill latency measured. Stop-publishing: the old vector (incl. the old lane D king) is re-asserted for `stale_after_blocks`, then the **whole** vector burns, logged (today's behaviour). **(rev6)** Change 4 is accepted: once W14 is on main, T27 adds the lane D-only burn row (stop-publishing then burns lane D only) |
| T28 | Eval host updated before and after the validator | Handshake fails closed when mismatched; works in the documented order |
| T29 | A docker-based satellite (code in `/app`) following the announcement verbatim | Upgrades and signals `cfg-` |

**Exit gate:** every row passes. **(rev3, A4) That includes T7, T7b, T7c and T14–T20 (already true in v2: "every row"), plus the new T4b, T6b, T9b and T18b.** At least one real crowning (T3) and one house
retake (T10) are observed end-to-end. No unexplained ERROR for 6 testnet eras.

---

## 5. Rollout stages (gated, no dates)

No calendar dates (owner 2026-11-21). Each stage starts only when the
previous gate is fully met; a failed gate goes back to the stage that owns
the fix.

Shaped by the rollout-mistake review (§13, §14). The worst past costs came
from four things:

- config drift between our box, main and the satellites;
- consensus changes landing before the fleet was ready;
- big-bang release bundles;
- announcements we couldn't verify.

So the order is: **de-drift → prove the overlay → catch-up release →
capability release → (rev5) one activation block, live everywhere.**

**(rev5) Go-live rule (owner 2026-11-21, D45): no shadow modes.**

- **Every feature goes live at one announced activation block, on every node,
  at once, at full effect.** That covers lane D, lane B, the sentinel, the D42
  leg options and D43. There are no `share = 0`
  eras, no log-only or "would-be verdict" periods, and no partial or
  per-validator rollout. Pay starts at the activation block.
- **What replaces shadow evidence:** CI, plus a **full testnet run
  (netuid 259)** that exercises real duels, real payouts and satellite replay
  (§4), plus the S1 measurements. Calibration that shadow eras used to provide
  (`d_margin`, gate tolerances, the sentinel) must come from testnet and S1
  **before** the block.
- **What stays (it isn't shadow):**
  - **Release-then-activate.** Six external validators run netuid 91. The
    activation overlay is published with an `effective_block` far enough
    ahead, and the **observed** `cfg-` stake must reach ≥ 2/3 (incl.
    5E2LP6) **before** that block. If it doesn't, publish seq N+1 with a later
    block. That's an upgrade check before the switch, not a trial after it.
    Without it the fleet forks.
  - **Activation at an era boundary.** Contract changes are never made
    mid-seed-epoch. Image repins follow DEC-CA-0067 (image-decoupled from
    `image_decoupled_from_block = 9493200`; legs are gated on the numerics
    digest), with 5 pins plus the wall gate.
  - **The kill switch** (revert overlay, §5).
- **Consequence:** a bug at activation pays out live until the revert overlay
  lands (up to about one era). Testnet exit gates (§4) are therefore stricter:
  every T-row must pass on testnet before the mainnet block is announced.

| # | Stage | Work | Exit gate (all must hold) |
|---|---|---|---|
| 1 | **S0 Decide** | **(rev5)** D1–D43, D45–D48 and D42b (incl. the §15 incentive [DECIDE]s D25–D32, the §16 sentinel D33–D35 and lane B D36–D42) (§6) in writing; W0 | All answered **(rev6)**: every D-number in the §17.4 register is decided, withdrawn or deferred-to-stage |
| 2 | **S0a De-drift** | Classify the 21 `LIVE OVERRIDE`s in the deployed chain.toml: main, overlay, or documented operator-only; type the decided blocks in main. Overlays merge onto **each node's own base** (`build_config(base, body)` merges onto `effective_raw(base)`, `netconfig.py:346-354`), so base drift survives any overlay | `cascade-audit decided --config /root/cascade/chain.toml` = 0 drift; the validator-read consensus-key diff between deployed and main is empty |
| 3 | **S0b Note budget** | Decode the live mainnet readiness notes and add the `cfg-` segment. It's trimmed **last** (`activation.py:1334-1357`), so older pending segments go first | No still-pending segment would be trimmed (type their decided blocks in main first if needed); T22 passes on testnet |
| 4 | **S0c Overlay bring-up** | Publish a **benign seq-1 overlay** (restates current values only) on testnet, then mainnet. Then a deliberately refused overlay on testnet | The 404-every-poll is gone. Fetched from S3 **and** the HF mirror by our validator, the trainer and satellites. `cfg-` lock ≥ 51% in `status/chain.json`. Receipts stamp `netconfig_seq`; the `cascade-audit` netconfig check passes; the trainer logs the same apply block. The refused overlay applies nothing |
| 5 | **S1 Measure** | W1, W0b, W3a (live Hippius probe); **(rev5)** start R16 long-history harvest (runs continuously from here, scored only from S8) | Forward floors met at the chosen margin (or re-decided); ≥ 1 context variant has `min_long_series`; storage mechanism chosen |
| 6 | **S2 Build** | **(rev5) All build WPs**, one PR each **(rev6.2: one implementation PR each; tests-only PRs may batch WPs per §3.1 rule 6; rev6.3: implementation PRs may batch up to 3 WPs per §3.1 rule 7)**: W1.5, W2 (with `ALLOWED["lane_d"]`) then W3–W19, W-S1–W-S3, the WB track (WB10–WB23) and W-EMA | CI green. Receipt replay on the last N mainnet receipts at the release head: **unarmed ⇒ byte-identical; armed ⇒ lane A verdicts, court and lane A weight entries identical, and lane D fields present and schema-valid** (W3b touches `bench_gate.py`/`public_guards.py`, which lane A replays). Startup-gate test of the shipped chain.toml at the live head for validator **and** trainer (T26). Docs gate |
| 7 | **S2b Catch-up release** | Ship main-as-of-now **without** lane D code to our validator, trainer and satellites, so the lane D release diffs only the lane D and lane B build WPs (§5 S2) **(rev6)** | 2 settlements with identical vectors; no satellite burn |
| 8 | **S3 Testnet** | §4 matrix including T22–T29 | §4 exit gate |
| 9 | **S4 Release** | Cut the **capability release** (lane D unarmed, `lane_d` allow-listed). **(rev5)** Bring up the launch eval host (D11; not valeval-laned, which is rented only at S8) (on-rent check, pinned known_hosts, handshake); dry-run the house. **Stage and review the revert overlay.** Send the announcement (§11.3) | Dry-run within the measured cross-host tolerance; announcement sent; revert overlay reviewed |
| 10 | **S5 Upgrade** | Snapshot all state (§10.8). Order, each through `deploy_gate check`/`verify` in quiet windows: **eval host → validator → lane D intake (systemd unit with the code-path guard) → trainer** (no legs in flight; memory-burst check; confirm preflight and worker-fingerprint lines unchanged). Satellites upgrade. Arm the broad watcher | Lane A unchanged on ≥ 2 settlements; `cascade-validator --netconfig-policy` lists `lane_d` on our nodes |
| 11 | **(rev5) S6 Activate (no shadow, D45)** | **D19 (rev5):** the owner-run test entries run on **testnet** (S3), not in a mainnet shadow. Overlay seq N, **one overlay**: `[lane_d] from_block` at a far era boundary **with `share = 0.25` (rev5, D1) and `takes_from = "lane_a"`**, all lane A keys **byte-identical** to seq N-1 (no lane A tuning in a lane D overlay, ever). ~~share = 0 shadow~~ | **Observed** `cfg-` stake ≥ 2/3 (incl. the largest holder, 5E2LP6 at about 49%) at a boundary before `effective_block`. That's the only stake-weighted proof of upgrade, because only nodes on the capability release can verify a `[lane_d]` overlay. Else publish seq N+1 with a later block. Webhook test alert delivered. Keep-alive live. **(rev5)** All §4 T-rows green on testnet before the block is announced. From the block: lane D live and paid everywhere; first paid settlement's vectors match on ≥ 2 validators, satellite replay matches, `check_lane_d` passes |
| 12 | ~~S6b Pay~~ **(rev5) merged into S6 (D45)** | — | — |
| 12b | **(rev5) NOT PLANNED.** ~~(rev3) S6c Lane B~~ (rev4: superseded by the SB track below; it was for the parked §12 variant) | ~~Overlay: `[lane_b] from_block` at an era boundary (shadow first: `[lane_b] share = 0` for ≥ 2 eras, then the D1 split)~~ **(rev6)** struck: no shadow (D45); lane B arms at SB5 | ~~TB1–TB8 passed in S3; ≥ 2 eras of valid lane B reports at share 0 with satellite replay matching; then first paid settlement's vectors match on ≥ 2 validators~~ |
| 12c | **(rev4) Lane B track SB0–SB5** (runs in parallel with lane D's S2–S3; depends on **S0a–S0c** de-drift, note budget and overlay bring-up, and on the shared S5 capability release) | **SB0** decide D35–D42, D42b · **SB1** build WB10–WB18, **(rev5)** WB22–WB23, W-S1–W-S2 · **SB2** WB19 measurement per GPU type (owner pod yes) · **SB3** testnet: TB10–TB27 (rev5), TS1–TS5, contamination check passing · **SB4** capability release (shared S5) · **SB5 (rev5, D45)** one overlay, once the D41 gate rule is satisfied and SB3 has passed (LB1): `[lane_b] from_block` with full share and `suite_targets` together, live everywhere at that block; no `share = 0` period. **(rev5)** If D40(b) is chosen: an image repin step (EMA constant fixed first; repin outside a seed epoch; moves lane A too) before SB3 | SB3 green including LB1's contamination check; **(rev5)** from the SB5 block, lane B live everywhere: first paid settlement's vectors match on ≥ 2 validators and satellite replay matches; gate-open evidence recorded |
| 13 | **S7 Operate** | §10.4 runbook; §10.7 checks; revert overlay stays staged | No pending past its latency expectation without progress; first crowning (reported immediately, with seated and judged counts) or a house hold observed end-to-end |
| 14 | **(rev3) S8 Long context** | W6-S8; rent valeval-laned (96 GB pinned torch) with owner yes; T7 at 32k/64k/128k; ~~then an overlay adds the passing variants to `contexts` (D10)~~ **(rev6.6:** no overlay: 32k–128k are in `contexts` from launch and each arms automatically when its gates pass, §1.1**)** | Each ~~added~~ **(rev6.6)** armed variant: T7 inside the settlement budget, W1 `min_long_series` met~~, overlay adopted ≥ 51%~~ |

**Block arithmetic.** Every activation or effective block is computed with
code (`is_era_start`, the activation helpers) from the live chain and
cross-checked from two sources. There are past off-by-ones (§13).

**Kill switch and latency.**

- An overlay can't roll back: a fix is **forward only**, as seq+1 (`until_block`
  or `share = 0`).
- That needs a fresh ≥ 51% lock-in and an era start, so **kill latency is up
  to about one era (≈ 12 h) or more**.
- A reviewed revert overlay is always staged. T27 measures the latency on
  testnet.
- **(rev5) There is no lane D-only fast kill today.** A missing lane D
  report is **not** a failure for a satellite. It's a check that can't run,
  so the receipt is retried and the satellite **keeps re-asserting the last
  verified vector**, still paying the old lane D king. That lasts for
  `[satellite] stale_after_blocks = 3600` (one era) past that settlement's
  boundary. After that the satellite burns the **whole** vector, every lane,
  until a new receipt verifies (`cascade/validator/satellite.py:30-45`;
  `shared/config/satellite.py:13`). So stopping lane D reports first freezes
  lane D's pay (no kill), then burns everything.
  "Stop publishing ⇒ only lane D burns" was wrong.
  - **(rev6)** Proposed design change 4 (a `lane_d: {status: "absent"}`
    receipt state with a lane D-only burn) is accepted and built by W14.
    Until W14 ships, **the only kill is the revert overlay above**.
  - T27 measures the overlay path, and tests stop-publishing as "old vector
    re-asserted for one era, then a whole-vector burn" until W14 lands **(rev6)**.

## 6. Owner decisions (recommended defaults)

**(rev5) Owner answers, 2026-11-21.** Every recommended default in this table
is **accepted** unless a row says otherwise. The explicit answers:

- **D28/D1/D48:** an equal split, **A 25 / B 25 / C 25 / D 25**.
- **D24:** 5% per-era false-crown target (OA3 answered).
- **D46:** (a), date the house by its lineage.
- **D42b:** (a), real hours on the chosen GPU.
- **The duel reference is confirmed:** an entry must beat the lane A king
  (the house) or, once the house has been beaten, the current lane D king.
  That's KING_D as designed (§1.4).
- **D21:** stays at its default (burn), under "all other recs".
- **D16:** the standing deletion approval is granted.
- **Proposed design changes 1–17, 20 and 21:** accepted. 18 is withdrawn and
  19 is adopted as D43.
- **Not covered by these answers (needs a separate yes with cost caps):** the
  pod rentals in OA4, OA8 and §12b.6, and the data-host choices in OA5 and
  OA6. Renting pods needs an explicit yes plus a cap.

| # | Decision | Recommended |
|---|---|---|
| D1 | Share and source | **(rev5) DECIDED (owner): 25% from the total; equal split A 25 / B 25 / C 25 / D 25 when all lanes are armed.** Mechanically `[lane_d] share = 0.25`, `takes_from = "lane_a"`: lane A gets "the rest" (`lane_pool_shares`, `shared/lane_c.py:584-605`), and the king ladder works inside lane A's 25%. While a lane is off, its share stays with lane A (today's code); while lane B is armed but gated, its share burns (D37). Remainder burns |
| D2 | Context variants | ~~**(rev3)** Launch arms {4k, 8k, 16k}. {32k, 64k, 128k} arm at **S8** by overlay (`contexts`), each only after T7 and W1 `min_long_series` (owner goal 128k kept)~~ **(rev6.6, owner 2026-11-23)** `contexts` = {4k, 8k, 16k, 32k, 64k, 128k} from launch; each variant arms automatically once W1 `min_long_series` and T7 pass; a declared unarmed variant is scored at ~~the highest armed variant~~ **(rev6.15, owner 2026-11-23)** the highest armed variant below it, else not scored that settlement (`context_below_armed`) |
| D2b | Gate | **(rev3)** One gate suite drawn per crowning from the suites **not** held out this era; the held-out suite is the §16 cross-lane sentinel (replaces fixed "TIME unselected"); `tau_public` 0.01; anchors = KING_D and house |
| D3 | Rate limits | 1 open per hotkey; 2 open per coldkey; 8 judged per settlement |
| D4 | Eval fee | None at launch |
| D5 | Margins and floors | `d_margin` 0.01 (level units, re-checked by T14); `forward_margin` ~~6 h~~ **(rev6.14)** 0 h (fresh = data after submission T) and `eval_delay_hours` 6 h (judging start; owner 2026-11-23); 1000 windows / 20 sources. **(rev6.10, owner 2026-11-23)** `embargo_hours` 48 h (lane A's ~48 h reveal delay; per-feed catalog override can only shorten); `max_windows_per_settlement` ~~rec 2000~~ **(rev6.11)** 5000 (owner 2026-11-23); **(rev6.11)** ~~embargo only for hourly-and-slower series,~~ **(rev6.12)** embargo for every cadence (owner 2026-11-23), ~~`embargo_window_share_cap` rec 0.25~~ **(rev6.12)** fresh-first draw, `min_fresh_window_share` 0.5 (owner 2026-11-23; below it at the floors ⇒ not frozen, pending); floors unchanged |
| D6 | Bench capacity | Queue behind lane A; second pod if crownings slip > 1 settlement |
| D7 | Court | 2 prior miner kings, ~~decay 0.5~~ **(rev6.23)** `court_weights` 80 / 15 / 5 (owner 2026-11-24); house never in the court |
| D8 | Losers | Private; vault TTL 4 eras; owner-only archive 14 days; then delete |
| D9 | House paid? | No: the slot burns; prior miner tails keep paying |
| D10 | Activation | **(rev5, D45)** One overlay: `from_block` with **`share = 0.25`** (D1) and `takes_from = "lane_a"` together (no `share = 0` shadow), published with an `effective_block` far enough ahead that observed `cfg-` stake reaches ≥ 2/3 before it; kill switch by a staged revert overlay (latency up to about one era); ~~**(rev3)** the S8 long-context arming is also an overlay (`contexts`), live at its block, no shadow~~ **(rev6.6)** long-context arming needs no overlay: every variant is in `contexts` from launch and arms automatically on its gates (§1.1, D2) |
| D11 | Eval host | **(rev6.18)** Any GPU with a qualified scoring environment (B5), incl. RTX PRO 6000 Blackwell (owner 2026-11-23). ~~**(rev3)** Launch: valpriv-4090 or another pinned-torch 2.4.1+cu124 pod (≥ 24 GB; the 16k fit is checked in B2). **Not** Blackwell sm_120.~~ valeval-laned (H100/A100 80 GB class, pinned torch) only at S8 |
| D12 | Pool packaging | `pool` untouched (keep length never raised); **(rev3)** launch `pool_d` = `pool`'s series ids joined, history to 16384 + 64, A1 timestamps, `pool_d_hist` (no targets) to the eval pod, `pool_d_tgt` on the box; 32k–128k history work at S8; dedicated builder |
| D13 | Weights license | MIT only (owner 2026-11-24; was Apache-2.0 / OpenMDW-1.1); kings are published |
| D14 | dtype | **(rev3)** BF16 only (~2.5 GB), scored in lane A's eval precision (fp32 upcast); F32 refused (4.9 GB is near the 5 GiB single-PUT limit) |
| D15 | Arch generations | Lane D generation = lane A generation; a change opens a new board |
| D16 | Standing approval | Automated vault TTL deletion and archive expiry (otherwise Approve-tier per deletion) |
| D17 | Trust | Accept that lane D loser rows are signed-but-not-reproducible (satellites verify structure and bootstrap; kings become re-scorable after publication) |
| D18 | Data path | **Targets stay on the box** (histories-only to the pod, D12), keeping the private-offload rule. Alternative: let the pod hold targets, which is a new rule; **(rev3)** launch `pool_d_hist` carries ≤ 16384 history per series |
| D19 | **(rev5)** Test entry | Owner-run test entries on dedicated **testnet** hotkeys during S3, so a real duel and payout are exercised before the mainnet block (no mainnet shadow, D45) |
| D20 | ~~**(rev3)** King change between first clear and confirmation~~ **(rev6.11)** Moot: confirmation duel dropped (owner 2026-11-23) | ~~(a) restart the duel against the new king (**recommended**); (b) confirm against the king at the first clear~~ `confirm_on_king_change` parsed and ignored |
| D21 | **(rev3)** Unpaid lane D king slot while the house reigns | Option 1 burn (v2 default, kept) vs option 2 return to lane A. Tokenomics call, not adopted (§1.4 table) **(rev5) DECIDED: burn** (the default, accepted). |
| D22 | **(rev3)** Coldkey resolution | (b) record at `/v1/lane-d/ready`, re-check at reveal, carry in the signed report (**recommended**); (a) archive-state resolution at T |
| D23 | **(rev3)** Loser re-scoring | (ii) targets, window ids, `rows_sha256` and bootstrap verifiable; histories private (**default, as v2**); (i) publish `pool_d_hist` after the dispute window |
| D24 | **(rev3)** Per-era false-crown target (sets `d_margin` via B1) | 5% recommended (OA3); `d_margin` stays 0.01 until B1 runs **(rev5) DECIDED: 5%** (owner; OA3 answered). |
| D25 | **(rev5 row; text in §15)** Court rent and salami-slicing | See §15 D25 |
| D26 | **(rev5 row; text in §15)** Lane A as a free input to D; D's share out of A | See §15 D26 |
| D27 | **(rev5 row; text in §15)** Withholding generators from A | See §15 D27 |
| D28 | **(rev5 row; text in §15)** Lane split (A/B/C/D) | See §15 D28 **(rev5) DECIDED: A 25 / B 25 / C 25 / D 25** (owner). |
| D29 | **(rev5 row; text in §15)** Frequency bias in frozen sets | See §15 D29  **(rev5)** Partly addressed by `max_source_share` (B14, W5b). |
| D30 | **(rev5 row; text in §15)** TIME isn't truly unselected (now handled by the §16 sentinel, D33) | See §15 D30 |
| D31 | **(rev5 row; text in §15)** Sybil queue griefing | See §15 D31 |
| D32 | **(rev5 row; text in §15)** Lane B vs D | Resolved in rev4 (§12b.7); see §15 D32 |
| D33 | **(rev3)** Cross-lane rotating sentinel (§16) | **Adopt**: one suite per era, drawn from the era-boundary block hash, held out of every gate and target in lanes A, B and D, swept and published for every new king. Alternative: no sentinel (v2: lane A gates all four; lane D holds TIME out) |
| D34 | **(rev3)** Sentinel ratchet vs the house | **Optional, recommended on**: every lane's candidates must not fall below the current lane A king on the sentinel by more than `sentinel_tau` (0.01), re-scored paired on one host. Off = the sentinel is publish-only, and a regression on it alone is never blocked |
| D35 | **(rev3)** Sentinel vs DEC-CA-0071 (pending: PR #482; rev5 renumber) lane B suite targets | While a suite is the sentinel it is **not** a lane B target or guard that era. Entries already declared on it are judged on the remaining suites, or deferred to the next era. Recommended: **defer** (the target stays honest) |
| D36 | **(rev4) resolved:** lane B arms **trainer-run** (generator + optional declared rows + optional suite target). The miner-weights variant (§12) is not planned |
| D37 | **(rev4)** Lane B share and what it does before the gate opens | Deployed `[lane_b] share = 0.25` (A/B 75/25 while C off; DEC-CA-0050 burns empty slots). **Assumption stated:** while lane B is armed but its opening gate is closed, every lane B slot is empty, so **its whole share burns**. Recommended: set `[lane_b] from_block` only once the gate rule (D41) is satisfied, or arm at `share = 0` until the gate opens (SB5). The split itself is D28. **(rev5, D45)** No `share = 0` arming: `[lane_b] from_block` is set only once the D41 gate rule is satisfied, and lane B goes live at full share at that block |
| D38 | **(rev4)** `max_repo_mb` for lane B suite-target entries (LB1) | (a) 8 MB via `[lane_b] suite_target_max_repo_mb` (**recommended**); (b) keep 128 MB (`chain.toml:47`) |
| D39 | **(rev4)** Per-layer expert clamp (LB2) | Fractions `min_experts_frac = 0.03125`, `max_experts_frac = 0.1484375` ⇒ 128: 4..19, 256: 8..38, 512: 16..76 (**recommended**); or fixed per-layer lists |
| D40 | **(rev4)** Router rows (LB2) | (a) frozen (**recommended for first arming**; no image change); (b) FlexOlmo unfreeze of the declared rows' router entries and bias, with the `traffic_steal_bound` guard. **(rev5)** (b) changes on-pod `expert_training.py` ⇒ a **shared worker-image repin that also moves lane A** + DEC-CA-0067 qualification, and since any image ≥ rc16 arms power EMA for lane A, (b) is **blocked on fixing the scored EMA constant first** (§3 Services touched)  **(rev5)** Depends on W-EMA (§8). |
| D41 | **(rev4)** Lane B opening rule (LB4) | (a) `open_tokens` + a passing lane A king check (**recommended**); (b) `open_tokens` alone; (c) co-gate with re-measured thresholds |
| D42 | **(rev5) DECIDED (owner 2026-11-21):** expert-leg cost (LB6) | **Cheaper GPUs allowed** via a lane B-only `[lane_b] funded_pod_skus` (each type image-qualified and measured first), and **miner-chosen leg length** `train_hours` 0.5–6.0 h (`[lane_b] max_leg_hours = 6.0`), guard capped by a new `[lane_b] max_leg_seconds`, never by editing `[training] max_train_seconds`. Operator funding not adopted. §12b.6 |
| D42b | **(rev5)** What `train_hours` means (§12b.6) | (a) wall hours on the chosen GPU, steps fixed at admission from the measured per-GPU rate (**recommended**); (b) reference hours, steps = 23,841 × h / 3 on any GPU **(rev5) DECIDED: (a) real hours on the chosen GPU** (owner). |
| D43 | **(rev5) DECIDED (owner 2026-11-21):** lane D reveal timing | **Immediate reveal** (`min_blocks`), not lane A's boundary − 25 blocks; T = the observed reveal block. Starts the forward clock up to ≈ 3 h earlier. §1.2 step 4, W10. (The commit-inclusion-block alternative is not adopted) |
| D44 | **(rev5) WITHDRAWN (owner 2026-11-21)** | Power EMA instead of cooldown was drafted and then dropped by the owner. ~~Lanes A and B keep today's cooldown.~~ Lane A keeps today's cooldown. Not planned **for lane A**. **(rev6.20, owner 2026-11-24)** Lane B only: EMA, no cooldown, decided separately (rev6.20 / rev6.20b: every lane B leg, no activation key); D44 itself stays withdrawn |
| D45 | **(rev5) DECIDED (owner 2026-11-21):** go-live rule | No shadow modes: every feature goes live at one activation block on every node at full effect. Testnet + S1 replace shadow evidence; release-then-activate (observed stake ≥ 2/3 **before** the block) and the kill switch stay. §5 |
| D46 | **(rev5, C1)** House T, which fixes starvation (§1.3, §1.4; interacts with D20) | Today the house's T resets on every KING_A change (a dethrone or a promotion; the king leg trains once per seed epoch, `shared/era.py:440-457`), so pending entries restart. If lane A changes king faster than lane D's commit-to-crown latency (~~about 2–8 days with confirmation~~ **(rev6.11)** ≈ T+6–9 h (10 s data) to ≈ 1 day (15-min, hourly) with 3-hourly pool_d, no confirmation duel), no challenger can crown. (a) **T_house = the latest commit (reveal) block of any generator in its lineage (recommended)**; (b) an entry keeps the T_K it froze against; (c) unchanged. B2 measures house refresh frequency **(rev5) DECIDED: (a)** (owner). |
| D47 | **(rev5, C2)** House precision under BF16-only | (a) **score the house as a BF16 cast of KING_A, and dedup entries against BF16 casts of every published checkpoint (recommended)**, so the house plays by the entry rules and a cast copy can't dodge dedup; (b) the house stays F32 |
| D48 | **(rev5, C3)** Source of lane D's share inside lane A (W8) | (a) before the ladder (cuts the lane A king); (b) from the ladder's burned remainder: free to lane A miners, but D's budget varies with the king's ladder share (10% after a fail, up to 95%, `chain.toml:1223-1226`) and can be ~5%; (c) from the total. W8 changes `lane_pool_shares` (`shared/lane_c.py:584`), a consensus signature at every caller **(rev5) DECIDED by the equal split:** fixed 25% shares of the total. Lane D takes from lane A, whose 25% is the pot the ladder works within, so lane D's budget never varies with the ladder. |

---

## 7. Phase 2: lane E (test-time optimisation), sketch

- **Entry:** a test-time procedure on KING_A with fixed weights (view
  ensembling, context adaptation, recalibration).
- **Judging:** on the private pool vs KING_A plain, with "improve one, regress
  none" guards (DEC-CA-0071 (pending: PR #482; rev5 renumber) pattern), at ≤ 4× base inference compute.
- **Blocker:** lane E runs miner code, so it needs a sandbox (no network,
  read-only weights, time and memory caps).
- **Phase 2a:** a fixed menu of procedures we implement, with miners choosing
  parameters only. No miner code; it proves the eval-cost model first.

## 8. Phase 3: lane A recipe menu, sketch

- **Declared options:** window and context sampling, domain mix weights,
  cooldown fraction plus an optional higher-LR continuation, and power-EMA
  averaging. **(rev3) Correction:** power EMA isn't a config knob.
  It's **image-only** (`POWER_EMA_SIGMA_RELS = (0.05, 0.10)`, hard-coded in
  `pyramid_trainer.py`, owner decision 2026-11-01). The contract's
  `power_ema_sigma_rels = []` only pins the contract value. The live rc15
  image has no EMA code, but **the first repin to rc16 or later turns EMA on
  automatically** at the next warm-start seed epoch, scoring the σ_rel 0.10
  average. That would make warm-start legs look worse. Fix the scored
  constant (0.05, raw, or cooldown-only averaging) **before** that repin.
- **(rev5) WP W-EMA (blocks every repin to an image ≥ rc16, including any
  image #477's NCCL pin ships in if it's built from main):**
  - The next image **scores the raw cooled endpoint on every leg**, in lanes
    A ~~and B~~. **(rev6.20/rev6.20b, owner 2026-11-24) Lane B is
    superseded:** every lane B leg trains with no cooldown and scores its
    σ_rel 0.10 power EMA (rev6.20 above), with no activation key. Lane A
    keeps this rule unconditionally. `power_ema_active` is changed so warm-start legs behave like
    random-init legs do today: `weights.safetensors` is the raw endpoint, and
    `weights_ema005` / `weights_ema010` are written alongside, **unscored**.
  - The main validator scores the EMA files **off the critical path** and
    publishes them as telemetry only. They never gate, never pay and never
    enter a receipt's verdict (so D45 isn't affected: it's telemetry, not a
    rollout mode).
  - **Never "best of both" per model.** The owner decides later whether to
    switch the scored file, as one activation for every leg (D45).
  - Tests: a warm-start leg's scored file is byte-identical to the raw
    endpoint; both EMA files are present; a lane B leg from an EMA-marked king
    still scores raw **(rev6.20b: only when trained directly on the `wsd`
    era contract; the dispatched lane B leg contract is the EMA schedule, so
    a dispatched lane B leg scores its own σ 0.10 EMA whatever its king)**.
- **Reproducibility:** options are folded into each leg's effective contract
  and replayed by the trainer, validators and audit.
- **Cost:** it's a lane A contract change (image, repin and seed-epoch rules;
  release-then-activate), so it has the highest blast radius of the three.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Too few forward-eligible windows, or long latency | W1 first; floors make it `pending`, never a guess; add sub-hourly sources |
| First-seen rule weaker than assumed (mirror lag > margin) | Margin ≥ the measured scrape lag; T8 forecast-feed and backfill cases |
| Main-validator centralisation of lane D scoring | Signed rows; satellite bootstrap replay; structural checks; kings re-scorable; D17 |
| **(rev3)** Coldkeys are free | The A8 caps limit casual flooding only, not a determined Sybil (one IP drove ~70 hotkeys across coldkeys; **(rev5) unverified:** the `OPSLOG:164` line ref has drifted and a grep for the fact didn't re-find it). Registration cost, the eval fee (D4), and the §15 pre-screen [DECIDE] are the levers. k = 32 per era is sized for this in §1.3 |
| **(rev3)** Owner conflict of interest | The owner runs the judging (main) validator **and** enters the house. Mitigations: the house is never paid (D9/D21); verdicts carry `rows_sha256` and signed rows; satellites re-run the bootstrap; `check_lane_d` re-derives the judged set and eligibility; kings are public and re-scorable; D23 (i) would let losers fully re-score; the builder's per-snapshot metadata digest is published (§15 / review round 2 item 12). **(rev5)** The first-seen log's daily Merkle root is committed before any snapshot uses it (W6fs), so we can't backdate first-seen to admit or exclude windows after seeing entries |
| **(rev5)** Rented eval host sees private data | Loser weights and `pool_d_hist` (histories, no targets) are visible to the rented (Lium) eval host. That's the same accepted risk as lane A's private offload: targets and scoring stay on the box, the host gets presigned GETs only, and losers are deleted on TTL |
| ~~**(rev3)** The low-frequency check is weak against contamination~~ **(rev6.18)** A10 removed | The private pool isn't forward-only, so a contaminated model passes A10's check more easily. It catches honest regressions only |
| **(rev4)** Procedurally recreated benchmark series in lane B | The static contamination check (LB1) can't see a series regenerated from fitted parameters. **Guard 4** (whole-pool point gain ≥ 0) and the **anchor guard** are the backstop: a gain that exists only on a public test set is refused before merge |
| Bench-pod contention and disk | Lane A first; byte LRU; second pod on demand |
| Eval GPU can't hold 64k/128k | Arm only measured variants; larger pinned-torch GPU later |
| Satellites lag the upgrade | Stake-weighted gate; later boundary |
| Scope pressure | No dates; arm a subset of contexts rather than drop a test row |

---

## 10. Operations: everything that has to run, end to end

### 10.1 What runs today (inventory 2026-11-21)

| Component | Where | Role |
|---|---|---|
| Trainer (rc41), validator (rc37), intake (`cascade.funding.main` :8790 behind Caddy, 130 MB cap), cascade-ops follower and health timer, three guards | prod box: 4 cores / 7 GB RAM / 154 GB disk (50%) | lane A, monitoring |
| valeval-blackwell4 | Lium RTX PRO 6000 96 GB, $1.48/h, driver 595; **benchmarks sidecar venv** (torch 2.12.1+cu130) | public bench sweeps |
| valpriv-4090 | Lium RTX 4090 24 GB, $0.60/h; pinned venv | lane A private eval offload |
| Lane A training pods | miner keys | lane A legs |
| Pool | GitHub Actions daily; Hippius pin; reveal at ~48 h | private windows |
| Storage | Hippius Hub/S3, R2, HF fallback; vault = local dir on the box | artifacts |
| Validators | ours (main, full) + satellites | weights |

Three `lium up` CLI processes on the box have been running for 3.6–8.3 days.
They look like hung rental calls. Check on-pod and confirm ownership before
anything else touches them.

### 10.2 What lane D adds

- **Training images:** none. The lane A image is unchanged, and lane D never
  triggers a repin.
- **Miner kit** (public): `cascade lane-d export | verify | submit | ready |
  status`; the latest public KING_A as a start point; a reference fine-tune
  script (runs on the miner's GPU).
- **(rev3) Launch eval host:** valpriv-4090 or another pinned torch 2.4.1+cu124 pod (≥ 24 GB; the 16k fit is checked in B2, OA4). **valeval-laned is rented only at S8:**
  - Hardware: ~~a **pinned-venv-capable GPU** (H100 or A100 80 GB; not
    Blackwell sm_120);~~ **(rev6.18)** any GPU with a qualified scoring environment (B5); ≥ 300 GB disk for the prefetch cache and
    `pool_d_hist`.
  - Software: the release's `uv sync --frozen` venv (torch 2.4.1+cu124),
    re-synced with every validator deploy, **with the worker handshake**.
  - Access: pinned known_hosts per pod, and an on-rent check (`nvidia-smi -L`,
    VRAM, driver, disk) before wiring.
  - Cost: about $1.3–2.5/h (Lium pricing varies; measured at rent).
- **Data paths** (nothing large through the box):
  1. Miner → vault bucket: direct multipart (§1.2).
  2. Vault → **(rev5)** the launch eval host (valeval-laned from S8): presigned GET prefetch at reveal; sha-keyed LRU
     with KING_D and the house pinned.
  3. Builder → Hippius: `pool_d_hist` (to the pod) and `pool_d_tgt` (to the
     box).
  4. Pod → box: forecasts only. The box scores against `pool_d_tgt`, then
     signs the report to S3 and R2.
- **Box load:** the lane D intake service (small), forecast scoring at h = 64
  (small), the bootstrap. **No weights, histories or torch.** `box_ram` is
  watched.
- **pool_d builder:** a dedicated small CPU VM with its own disk. It holds
  `TSFORGE_S3` and Hippius pin credentials only; never the prod box, never
  rented GPU pods.
- **Satellites:** no new hardware; the release before the activation block.

### 10.3 Testnet stack (separate from production)

- **testnet-ctl** (cheap CPU VM): testnet validator, lane D intake,
  cascade-ops; RC commit; testnet wallets only.
- **testnet-eval:** a **24 GB pinned-torch pod (rev3: launch contexts ≤ 16k)**, rented just-in-time for the T7
  rows and terminated after an on-pod check. A cheaper pinned-torch GPU
  handles the other rows.
- **Testnet `pool_d`:** hourly builds on the builder, testnet prefix.
- **Testnet bench:** valeval-blackwell4 **only outside lane A's king-check and
  dethrone windows** (check `v2_check_due`), or reduced-series suites on
  testnet-eval.
- **Test miners:** us. T3 needs one rented GPU for a few hours with **public
  data only**. T5 artifacts are made by script.
- Every testnet pod gets a TTL and verified teardown.

### 10.4 Per-settlement runbook

```
boundary B
 ├─ refresh house if KING_A digest changed (T = its publish block)
 ├─ newest pool_d_hist/_tgt pinned? (stale > ~~30 h~~ **(rev6.18) 6 h** ⇒ CRIT; entries stay pending)
 ├─ for each open entry: update eligible union (§1.3, max(T_E, T_K) + margin)
 │     → frozen this settlement? → cohort (intersection; defer latest-T if short)
 ├─ eval host (rev5: launch eval host; valeval-laned from S8): forecasts for cohort + KING_D + house on the frozen set
 │     (handshake first; cached rows re-used by (digest, context, window id))
 ├─ box: score vs pool_d_tgt → bootstrap → max-T → candidate or none
 ├─ candidate: drawn suite (block hash) on valeval-blackwell4, behind lane A,
 │     paired vs KING_D + house; sentinel after crowning (publish only, rev5)
 ├─ signed lane D report + lane_d_state.json → receipt digests → weights
 └─ crowned: publish king (async mirror; never gates the verdict); index; board
```

| Fault | Outcome |
|---|---|
| Eval pod down; vault or Hippius unreadable | Entries stay queued or pending, no cost; CRIT after 1 settlement |
| Bench pod down before the candidate's sweep | Crowning held; CRIT after 1; after `max_bench_holds` (4) the candidate re-queues unjudged. Deliberately not lane A's never-timeout hold |
| Candidate sweep error, partial, NaN, missing metric | `checkpoint_fault`: entry refused; evidence kept |
| Handshake mismatch (stale eval code) | Nothing scored; CRIT; stderr captured |
| `pool_d` stale or missing | Entries pending; CRIT |
| Validator restart | Idempotent from state and reports |
| Hippius primary failure with R2 success | Alert (fallback must not hide it) |

Pre-staged, reviewed **owner-paste scripts** cover **(rev5)** launch-eval-host (and, from S8, valeval-laned) setup and
recovery, because remote writes are often classifier-blocked.

### 10.5 Operations checklist by stage

| Stage | Action | Needs |
|---|---|---|
| S1 | W1 on GH Actions or the builder; W3a storage answer | — |
| S3 | Rent testnet-ctl, testnet-eval and the builder; deploy the RC; hourly testnet `pool_d`; arm | Owner yes on rentals |
| S3 | T3 fine-tune pod and T7 pod just-in-time; terminate after on-pod checks | Owner yes |
| S4 | **(rev5)** Bring up the launch eval host (D11; valeval-laned is rented only at S8) (on-rent check, known_hosts, handshake); dry-run; add `laned-eval.toml` to **`operator_inventory_host_files`** and **`operator_standing_pods_usd_per_hour`** (that list still names valeval-blackwell2; an Approve-tier chain.toml edit through the Edit tool, typed in main the same day, never left as a live-only override) | Owner yes |
| S4 | Cut the release; the owner sends the validator announcement (release, activation block, "no new hardware", how to verify) and the miner announcement | Owner |
| S5 | Snapshot **all 8 state files** first (§10.8: `validator_state.db`, `cascade_state.json`, `warm_start_init.json`, `lane_d_state.json`, `netconfig_state.json`, `activation_*_state.json`, the satellite state file, `decided_votes.toml`; **(rev6.1)** plus the trainer state under `/root/cascade/_train_work/` (`era_state.json`, `trainer_submissions.json`, `funded_queue.json` and the rest of the §10.8 list), `provisioner_state.json`, `pod_monitor_state.json` and the deployed `chain.toml` with its live overrides, `hosts.toml`, `provision.toml`). Then restart in order **eval host → validator → lane D intake → trainer** (no legs in flight), each with `python -P`, the code-path guard and `deploy_gate check`/`verify` (R8). **Outside settlement, harvest and scoring windows; never during a bench sweep; never Wed 22:00 – Thu 08:00 UTC.** Satellites upgrade per the announcement | Notify (plan and timing to the owner first) |
| S5 → S6 | Go/no-go on the stake gate; webhook test alert delivered | Owner |
| S6–S7 | First settlements watched end-to-end; first crowning reported immediately; daily load, cost and vault size | — |

### 10.6 Cost (operator-paid; training stays miner-paid)

| Item | Recurring | One-off |
|---|---|---|
| **(rev3)** Launch: valpriv-4090 (shared, outside lane A duel windows) **or** a dedicated 24 GB pinned-torch pod | ~$0 extra (shared) or ~$14–20/day (dedicated 4090-class, measured at rent) | — |
| **(rev3) S8 only:** valeval-laned (80–96 GB pinned-torch GPU) | ~$31–60/day (measured at rent) | — |
| pool_d builder (CPU VM + disk) | ~$2–5/day | — |
| Second bench pod (only if D6 triggers) | ~$36–46/day | — |
| Testnet ctl + eval + T3/T7 pods | — | per testnet day ~$10–30 plus a few hours of 80 GB GPU |
| Vault + `pool_d` storage | small | — |

### 10.7 Monitoring (cascade-ops, W11)

- `lane_d_report`: a signed report each settlement.
- `lane_d_pending`: a pending entry past its latency expectation with no
  progress.
- `lane_d_eval_pod`: reachable, handshake OK, disk ≥ cache budget, GPU
  visible.
- `lane_d_bench_hold`: a held crowning.
- `pod_disk` for valeval-blackwell4 under the added lane D load.
- `pool_d_stale`: the newest snapshot is older than about ~~30 h~~ **(rev6.18) 6 h** (two missed 3 h builds).
- `runway`: the balance covers ≥ 3 days of **all** standing pods.
- `primary_store_failed`.
- `vault_lane_d_bytes`.
- `box_ram`.
- **A follower pattern for every lane D log line.**

**Alert delivery.** `/etc/cascade-ops.env` is empty, so alerts reach only
`alerts.jsonl`. The owner fills a webhook (I never write that file), and a
delivered test alert is part of the S6 gate.

### 10.8 Rollback

- **Config:** forward-only overlays (§5 kill switch). The revert overlay is
  staged and reviewed before S4.
- **Code:** keep the previous release worktree and restart script for every
  service (validator, intake, trainer, eval host). A rollback goes through the
  same quiet-window rules and the same `deploy_gate` verify.
- **Snapshots before every S5 restart:** `validator_state.db`,
  `cascade_state.json`, `warm_start_init.json`, `lane_d_state.json`,
  `netconfig_state.json`, `activation_*_state.json`, the satellite state
  file, `decided_votes.toml`. **(rev6.1)** Plus (RV3 note 21) the trainer
  state under `/root/cascade/_train_work/`: `era_state.json`,
  `trainer_submissions.json`, `funded_queue.json`,
  `trainer_promotion.json`, `champion_publisher.json`,
  `lane_b_driver.json`, `coldkey_identities.json`,
  `published_settlements.json` and its own `activation_*` and `netconfig`
  files; `provisioner_state.json` and `pod_monitor_state.json`; and the
  deployed `chain.toml` (with its live overrides) together with
  `hosts.toml` / `provision.toml`. Copy them under `_state_backups/` with a
  timestamp.
- **Never restore validator state across a settlement.** A restore is only
  valid back to the last settlement it already covers. Otherwise forward-fix.
- **Lane D only:** revert overlay, forward-fix, republish. Lane A never
  depends on lane D state. **(rev5)** Stopping lane D reports doesn't stop
  lane D's pay. Satellites re-assert the last verified vector for
  `stale_after_blocks` (3600, one era), then burn the **whole** vector
  (`satellite.py:30-45`). It's not a lane D kill until Proposed design change
  4 ships.

---

## 11. Artifacts, publication and documentation

### 11.1 Where checkpoints live

| Artifact | Location |
|---|---|
| Lane A legs and kings | Hippius Hub `{ns}/ckpt-r{base_seed}-{role}-{size}{suffix}@sha256` (`loop.py:1030,1078,1410`); R2; HF fallback; non-kings expire after 14 days. Unchanged; the house reads KING_A by its existing ref |
| Lane B merges | `ckpt-r{rid}-expert-{label}` (`lane_b_driver.py:777`). Unchanged |
| Lane A king generator | `champions/<digest>.zip` (`champion_publish = "crown"`) and repo `champions/king/` via `king-sync.yml`. Unchanged |
| Lane D entries **(rev3: BF16, ~2.5 GB each)** | private bucket `cascade-lane-d-vault/lane-d/<hotkey>/<digest>/`; private R2 mirror; TTL then owner-only archive (§1.2) |
| Lane D kings | public Hippius Hub `{ns}/laned-king-c{context}-{hotkey8}@sha256` (the same digest as the vault), R2 public mirror, HF mirror `tensorlink-dev/cascade-lane-d-kings`; kept permanently |
| Lane D index | signed `champions/lane_d/index.json`: every king, context, digest, crown block, LCB, gate and **(rev5)** sentinel numbers (suite named), license |
| Repo cards | `champions/lane_d/<digest>/{PROVENANCE.json, MODEL_CARD.md}` via `laned-king-sync.yml`; no weights in git |
| `pool_d_hist` / `pool_d_tgt` | Hippius pins; sha256 in every lane D report; `pool_d_tgt` and window ids published after the dispute window |

One digest, one identity: vault, public repo, index and report must agree,
and the audit checks it.

### 11.2 Documentation and agent surfaces (ship with the release; S2 docs gate)

| Surface | Change |
|---|---|
| `docs/LANE_D.md` (new) | what lane D is; checkpoint format; context as a declared scorer setting; export, verify, submit, ready, commit; dedicated hotkey; **(rev5)** the licence attestation at `ready` (weights and training data licensed for Apache-2.0/OpenMDW publication; no restricted-licence teacher outputs, e.g. Timestral under MarketCast's `[marketcast]` licence terms) and the takedown path; forward-only rule and **expected latency by frequency**; house; gate and **(rev5)** the rotating sentinel; pay and court; license; FAQ; **(rev3)** coldkey caps and the typed skip reasons, ~~confirmation-duel latency~~ **(rev6.11)** crown at the first verdict, both clocks and mirror lag, BF16 only, ~~launch contexts 4k–16k (S8 later)~~ **(rev6.6)** contexts 4k–128k declarable from launch, each armed when its gates pass (unarmed ⇒ scored at ~~the highest armed~~ **(rev6.15)** the highest armed below the declared context, else skipped `context_below_armed`), max reveal delay |
| `docs/INTERFACE.md` | a lane D checkpoint interface section |
| `docs/MINER.md` | lane table, link, "no compute keys needed for lane D" |
| `docs/AGENT_MINING.md` | lane D agent workflow; **local scores on the revealed pool overstate forward gains** |
| `AGENTS.md`, `agent.md` | lane D role; human authorization still needed for spend and publish |
| `llms.txt` (repo + website) | lane D links; fix the stale "Planned order A → C → B" and activation text |
| `docs/VALIDATOR.md` | lane D replay: what satellites verify and what they can't; **(rev3)** the new report fields (`rows_sha256`, `coldkey`, judged set, ~~`confirmation`~~, low-frequency result, both clocks) **(rev6.11)** no confirmation stage |
| `docs/OPERATOR.md` | **(rev5)** the launch eval host (valeval-laned at S8), builder, vault, retention, runbook, kill switch; **(rev3)** launch eval host (pinned torch, not Blackwell), S8 long context, the first-seen recorder (W6fs), the coldkey alert |
| **(rev4) Lane B docs** | `docs/MINER.md` lane B: declaring rows, suite targets, the contamination refusal, deferrals. `docs/INTERFACE.md`: the ban on embedded benchmark series and the `declared_rows` format. `docs/AGENT_MINING.md`: using the routing artefact (LB3). `docs/VALIDATOR.md`: what validators check (hash-set version, declared rows, the opening rule) |
| `docs/ARCHITECTURE.md` | lane D flow |
| `status/lane_d.json`, website | queue, pending progress, king, board; **(rev3)** per-entry reason enum, ~~confirm stage~~, reveal deadline |
| `cascade status --json`, `discovery-check` | lane D activation, contexts, king, endpoints |
| `README.md`, `CHANGES.md` | summary, release notes |
| Decisions | DEC-CA-0070 new; DEC-CA-0062 superseded |

Docs gate: all of the above in the release PR set, links resolve, and
`llms.txt` matches `cascade status --json` on testnet.

### 11.3 Other items

- **Arch generations:** D15.
- **Announcements:** drafted at S4; the owner sends them. Validator
  announcement content:
  - the release commit and docker tag;
  - "pull main's chain.toml, no local edits, keep every state file";
  - expected confirmation lines: `cascade-validator --netconfig-policy` lists
    `lane_d`, `activation: signalled … cfg-<hex>`, `git log -1`, satellite
    status `mode` / `last_verified_round`;
  - the overlay seq and digest and the expected lock boundary;
  - what lagging costs (burn after `from_block`);
  - the deadline: one era before `effective_block`.

  We track a per-hotkey, stake-weighted checklist from the chain notes. A
  feature name is **never renamed after it's announced**.
- **Support:** `status/lane_d.json` shows each pending entry's eligible-window
  progress against the floors.
- **Leaks:** a validator operator leaking private weights gains nothing in
  lane D (copies can't beat the margin), but losers stay private and
  short-lived regardless.

---

## 12. Lane B variant: miners submit expert weights — **not planned (rev4: parked)**

**(rev4)** Parked by owner direction (2026-11-21): miner-trained rows can train directly on public test sets, which breaks suite targets and lets contamination into the lineage. Lane B stays trainer-run with miner-chosen experts (§12b). The text below is kept for the record; WB1–WB7 and TB1–TB8 in this section are **not planned**.


~~(rev3) In scope: built, tested and armed as part of this plan, armed at S6c.~~ **(rev5) Not planned: parked in rev4 (see the heading). The S6c stage is gone; lane B arms on the SB track (§5 row 12c).** Historical rev3 text follows. **(rev3)** Caveat: most low-frequency domains will rarely reach the forward-only floors, so forward-only judging of per-domain expert deltas works mainly for sub-hourly domains. It reuses lane D's vault, forward-only
judging and GPU ingest.

- **Entry:**
  - the base-king digest (≤ `max_base_age_eras` old);
  - the declared (layer, expert) rows;
  - optionally those rows' router and route-bias entries (FlexOlmo pattern;
    revives dead experts);
  - a safetensors delta of only those rows;
  - the declared target (DEC-CA-0059 / 0069).
- **Verification** uses existing code: `merge_expert_slices` (byte-check),
  `rebase_expert_delta` onto the current king, then re-run the verdict and
  guards. No training on our side.
- **Changes vs today's lane B:**
  - Rows are miner-claimed, held from `ready` to verdict or merge, earliest
    reveal wins.
  - Judging moves to forward-only windows (miners can train on anything).
  - No compute funding.
  - Router-row deltas need a deliberate traffic-stealing test case blocked by
    the guards.
- **Interaction:** a won delta merges into KING_A at the era boundary. It's
  the only path by which miner-trained weights reach lane A, bounded by the
  byte-check, the rebase and re-verification.
- **New design:** this is *not* DEC-CA-0071 (pending: PR #482; rev5 renumber)'s Open
  option (b). That option is miner-declared rows with trainer-run training.
  **(rev3) WB7 writes the DEC-CA-0071 amendment** that records this variant,
  the §16 sentinel rule (D35) and forward-only judging.

**(rev3) Execution in this plan**

- **Deployed state:** `[lane_b] from_block = 0` (off) and `leg_funding =
  "miner"`. The A/B split is 75/25 while C is off.
- **Arming:** `[lane_b]` is fully overlay-settable (`ALLOWED["lane_b"] =
  None`, `netconfig.py:83-87`; `remainder` stays release-only), so (rev5: historical, parked variant) S6c arms
  lane B **by signed overlay**, with no validator action. The capability
  release (S5) carries the lane B code.
- **Existing code reused:**
  - `merge_expert_slices` and `rebase_expert_delta` (`pyramid_model.py:706,
    738`) for the byte-check and rebase;
  - `check_lane_b` (`cascade/audit/checks.py:2887`) for the audit;
  - lane D's W3 intake, W3a vault, W4 ingest, W5a/W5b forward eligibility,
    W7 report pattern and W15 prefetch.

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| WB1 | Expert-delta submission format (base-king digest, declared (layer, expert) rows, optional router and route-bias rows, BF16 safetensors delta of only those tensors, declared target) + intake via the lane D service (`/v1/lane-b/submit`, `/ready`) | W3, W3a | ci + testnet | `scripts/laned/accept WB1` |
| WB2 | Verification: `merge_expert_slices` byte-check (only declared rows differ from the base king); `rebase_expert_delta` onto the current king; `max_base_age_eras`; re-run verdict and guards on the rebased model | WB1, W4 | ci + pod | `scripts/laned/accept WB2` |
| WB3 | Row claims: rows held from `ready` to verdict or merge, earliest reveal wins; **row-level dedup against published KING_D rows** (D32; cosine/L2 within ε ⇒ refuse) | WB1, W13 | ci | `scripts/laned/accept WB3` |
| WB4 | Forward-only per-target judging for lane B (lane D's W5a/W5b eligibility on the target's windows; floors; the confirmation duel per D20 **(rev6.11)** **kept unchanged**: a separate lane B decision (owner 2026-11-23); lane D's removal in W5e doesn't apply to lane B) + guards on every other domain and suite + §16 sentinel exclusion (W-S2) | WB2, W5a, W5b, W-S2 | ci | `scripts/laned/accept WB4` |
| WB5 | Lane B report fields (rows digest, claims, rebase base digest, merge identity), satellite replay and the `check_lane_b` extension | WB4, W7a, W9 | ci | `scripts/laned/accept WB5` |
| WB6 | Lane B docs: `docs/MINER.md` and `docs/AGENT_MINING.md` lane B sections for expert deltas, the LANE_D.md cross-reference, VALIDATOR.md/OPERATOR.md lane B notes | WB5 | ci | `scripts/laned/accept WB6` |
| WB7 | DEC-CA-0071 (pending: PR #482; rev5 renumber) amendment: expert-delta variant, forward-only judging, the D35 sentinel rule | — | ci + owner merge | `scripts/laned/accept WB7` |

| Test | Case | Expected |
|---|---|---|
| TB1 | Delta touching an undeclared row | Refused at the byte-check |
| TB2 | Delta trained on a base king 1 era old | Rebased onto the current king and re-verified; a base older than `max_base_age_eras` is refused |
| TB3 | Router-row delta that pulls other domains' traffic | Blocked by the other-domain guards |
| TB4 | Two entries claiming overlapping rows | The earliest reveal holds the claim; the other is refused with a typed reason |
| TB5 | Delta copied from a published KING_D's rows | Refused by the row-level dedup |
| TB6 | Lane B unarmed | Byte-identical receipts and weights (invariant 1) |
| TB7 | A lane B entry targeting the current sentinel suite | Deferred (D35) |
| TB8 | A won delta merges at the era boundary | KING_A updated; the house refreshes in lane D (§1.4) |

---

## 12b. Lane B (rev4): improve one, regress none, miner-chosen experts

**Direction (owner, 2026-11-21).** Lane B stays trainer-run:

- the miner submits a **code-only generator**, and our trainer trains the leg;
- the miner may also **(1) choose which expert rows are trained**
  (DEC-CA-0071 Open option (b));
- and **(2) target one public suite** (DEC-CA-0071 suite targets): improve it
  significantly while regressing nowhere — other suites, every private-pool
  domain, every horizon, the whole-pool point gain, and the fixed anchor king.

The rules below are recorded in the DEC-CA-0071 amendment
(`decisions/DEC-CA-0071-lane-b-public-suite-targets.md`, §"Amendment rev4").

### 12b.0 What carries over unchanged from DEC-CA-0071

- **Gain test:** CRPS `LCB_α ≥ gain_margin_public` (0.01) on the whole suite;
  MASE non-inferior; at least `min_public_items` items.
- **Guards:** every other suite, every pool domain, every horizon; the
  whole-pool point gain must be ≥ 0 (guard 4); the fixed anchor king (guard
  5).
- **Pay:** 7 domains + 3 horizons + 4 suites = 14 slots, weighted by
  `target_weights`, with decay, TTL, burn and merge-once.
- **Merge:** `merge_expert_slices` byte-check and `rebase_expert_delta`
  (`pyramid_model.py:706, 738`), re-verified on the current king.
- **Judging:** `judge_expert_targets` (`cascade/eval/expert_lane.py:249`).
- **Training backend:** the compact-rows backend. Every parameter is set
  `requires_grad_(False)` and only the selected rows get trainable leaves
  (`expert_training.py:3-6, 112-122`); the router and shared weights are
  reused as-is (`:49-51`).
- **(rev6.20, owner 2026-11-24) Schedule and scored weights.** ~~A lane B
  leg uses lane A's WSD schedule (cooldown over the last 1/11 of its steps)
  and scores the raw cooled endpoint (W-EMA).~~ **(rev6.20b: no key)** Every
  lane B leg has **no cooldown** (warmup, then constant LR to the end) and
  its scored and merged weights are its **σ_rel 0.10 power EMA**; every
  expert entry records `scored_weights = "ema010"` and judges refuse any
  other value. Lane A is unchanged (rev6.20).
- **Legs:** `[lane_b] leg_hours = 3.0`, `leg_funding = "miner"` (deployed).
  **(rev5)** Replaced by miner-chosen `train_hours` (0.5–6 h) on a lane B GPU
  list (D42, §12b.6); `leg_hours` stays the default when a miner sets none.

### 12b.1 LB1: contamination check — a precondition for arming

- **Arming rule.** `[lane_b] suite_targets` may arm **only after** the static
  contamination check ships **and passes on testnet** (stage SB3). It's a
  gate, not an open item.
- **What it checks.** Hashes of every GIFT-Eval, BOOM, TIME and FEV **test
  window**, in two forms:
  - **exact:** sha256 of the float64 values;
  - **normalised (rev6.2, owner 2026-11-22; WB11 schema 2,
    `cascade_contamination_hashset/2`, encoding `cascade-zgrid/2`; PRs
    #617/#621, #632):** sha256 over sliding sub-windows of 32 points, each
    window z-scored with its own mean and **population** std, every value
    binned as `floor(z / 0.5)`. At lookup, a value within 0.03 of a bin edge
    also probes the neighbouring bin (nearest edge first, at most 12 such
    values per window, at most 4096 probes per window); both the window and
    its negation are looked up. This is robust to scale, offset and
    rounding. It replaces "rounded to 4 significant figures", which TB11's
    evasions defeated at some scales.
  - **low-information windows (RV6):** a window with fewer than 4 distinct
    values, one value making up ≥ 3/4 of the window, std ≤ 2^-46 · |mean|,
    or a near-perfect linear ramp is excluded from both hash sets at build
    and ignored at lookup.
- **Matched against** everything in the submission:
  - numeric literals in `.py` files (AST walk, all number constants and list
    or array literals);
  - every value in `config.json`;
  - any non-code bytes (decoded as float32/float64/int arrays).
  - A match on ≥ `contamination_min_hits` sub-windows (recommended 1 exact or
    3 normalised) is a refusal.
- **When.** At intake, **before any compute**. The typed refusal is
  `benchmark_series_embedded`, with the suite and window id in the receipt
  field.
- **The hash set.**
  - Built from the pinned benchmark data the bench sidecar uses: the suite
    identities recorded on the bench pod (gift 30841734 / boom 69325b54 /
    time 83e3d0b3 / fev 6de245af, OPSLOG).
  - Published as a versioned, digest-pinned artefact
    (`benchmarks/contamination/hashset-v<n>.json.zst`, sha256 in
    `[lane_b] contamination_hashset`).
  - Refreshed **only** when a suite identity changes (a new benchmark
    release), as a new version under a new digest.
- **Backstop (stated in §9).** A miner can recreate a benchmark series
  procedurally (fit it, then generate it from parameters), and a static hash
  check can't see that. **Guard 4** (whole-pool point gain ≥ 0) and the
  **anchor guard** remain the backstop: a gain that exists only on a public
  test set is refused before merge.
- **[DECIDE D38] `max_repo_mb` for suite-target submissions.**
  - Today `[generator] max_repo_mb = 128` (`chain.toml:47`). DEC-CA-0028 notes
    `config.json` can carry bulk data and plans a reduction ("no longer any
    excuse for numbers in `config.json`", DEC-CA-0028:56).
  - (a) **Yes: 8 MB for lane B suite-target entries (recommended).** Less room
    to hide benchmark data, and a faster check.
  - (b) No: keep 128 MB for everyone.
  - Either way it's a new key, `[lane_b] suite_target_max_repo_mb`.

### 12b.2 LB2: rules for miner-chosen experts

- **Declaration.** At commit, `declared_rows = {layer: [expert indices]}`,
  validated against `experts_per_layer = (128, 256, 256, 512, 512, 512, 512)`
  (`pyramid_model.py:42`; counts are **per layer**).
  - **(rev5) Keys are MoE ordinals 0–6**, the existing selection format
    (`trainer/expert_lane.py:108-109`, one entry per `moe_layers()`; the
    worker parses `{int: [int]}`, `worker.py:226-227`). They are **not** block
    indices: the MoE blocks are `(1, 2, 3, 5, 6, 7, 8)`
    (`pyramid_model.py:39`). Ordinal i has `experts_per_layer[i]` experts.
  - `declared_rows_out_of_range`: an index ≥ that layer's expert count.
  - `declared_rows_bad_layer`: **(rev5)** a key outside 0–6.
  - `declared_rows_count`: a count outside the per-layer clamp.
  - **(rev5) The clamp minimum applies only to layers with ≥ 1 declared
    row.** Zero rows on a layer is allowed, matching ESFT, where a layer may
    take fewer than `min_n` rows, possibly none (`expert_lane.py:95-99`). At
    least one row in total is required (TB21 covers "no declared rows").
- **Per-layer clamp.** `min_experts = 8` / `max_experts = 38` were set for 256
  experts, i.e. 3.125% / 14.84% of a layer. Replaced by fractions:
  - `min_experts_frac = 0.03125` (8/256) and `max_experts_frac = 0.1484375`
    (38/256), the exact ratios of the current values;
  - per layer, `min = ceil(frac_min · E_l)`, `max = floor(frac_max · E_l)`;
  - ⇒ **128 → 4..19, 256 → 8..38 (unchanged), 512 → 16..76.**
  - This keeps ESFT's "5–15% of experts per layer" regime (DEC-CA-0050) at
    every layer size.
  - **[DECIDE D39]:** the fractions (recommended above), or fixed per-layer
    lists.
- **Router rows [DECIDE D40].**
  - (a) **Router and bias stay frozen (recommended for first arming).** This
    is today's backend (`expert_training.py:112-122`). Dead experts can't be
    revived, because a frozen router sends them no tokens.
  - (b) **FlexOlmo pattern:** only the declared rows' router entries and bias
    unfreeze, which can revive dead experts. If (b):
    - the merge byte-check also proves no other router row moved;
    - a **traffic-stealing guard** applies: on each **non-target** domain's
      windows, the routing share captured by the declared rows may rise by at
      most `traffic_steal_bound` (recommended 0.02 absolute per layer),
      measured on the rebased model against the current king, else refused
      with `traffic_stealing`;
    - T-row TB14.
- **Overlaps.** Claims on overlapping rows follow the existing merge-acceptance
  deferral: rows are exclusive against open legs, finished-but-unsettled legs
  and pending merges (DEC-CA-0059).
  - **Claim order:** earliest reveal.
  - A deferred claim shows `deferred: rows_held` in `status/lane_b.json`,
    with the holder's entry id and the expected release (the holder's
    verdict or merge era).
- **Trainer-picked rows (ESFT) remain the default** when a miner declares none.
  **Both modes coexist:** a declared entry claims its rows first, then
  ESFT-selected entries are planned excluding all held rows.

### 12b.3 LB3: routing information for miners

- **What's published, per king:**
  - per-layer **per-expert** load on the training stream;
  - dead and near-dead experts (load < 0.1 × mean);
  - per-suite routing affinity: routing counts over each public suite's
    **context inputs only**, with no targets and no training on them.
- **Repo fact (rev4).** PR #481's telemetry is **open, not merged, not
  deployed**. It records per-layer **summary** stats only (`eff`, `dead`,
  `max_mean` per 50-step window and per trunk/cooldown phase), **not**
  per-expert counts. The producer therefore extends it to emit the per-expert
  count vector at the phase level (WB12).
- **Where.** A digest-pinned artefact `status/lane_b_routing-<king8>.json`
  (sha256 recorded in `status/lane_b.json`), refreshed when the king
  changes. The producer is **WB12**; the suite affinity runs on the bench
  host's existing suite inputs.
- **It's informational:** never scoring, never consensus. Stale info
  (king changed, no new artefact yet) shows `stale_since_block`. Miners
  declare rows at their own risk, and a stale artefact never affects a
  verdict.

### 12b.4 LB4: replacing the opening gate

- **Today:** `open_tokens = 200B`, `routing_jaccard = 0.7`, `routing_overlap =
  0.8`, `rebase_jaccard = 0.5`, `rebase_overlap = 0.7` (deployed `[lane_b]`).
  The gate check is `gate_open(lineage_tokens, routing, lb)`
  (`lane_b_driver.py:503-510`).
- **Repo fact (rev4): the co-gate has never been measured on mainnet.**
  - The mainnet driver state (`/root/cascade/_train_work/lane_b_driver.json`,
    last written 11-15) has `gate_opened_block = 0`, `routing = {}`, an empty
    `routing_history` and `lineage_tokens = 0`.
  - Lane B is off (`[lane_b] from_block = 0`), so the tick that measures it
    never runs.
  - The only measurement is the 2026-11-20 manual probe: king vs init top-8
    overlap about 1.1/8, far below 0.8.
- **[DECIDE D41] the opening rule:**
  - (a) **`open_tokens` + a passing lane A king check (recommended).** It
    proves the lineage is maturing without needing routing to stop moving.
  - (b) `open_tokens` alone.
  - (c) keep the co-gate with re-measured thresholds.

  The new key is `[lane_b] open_rule` (`"tokens_king_check"` | `"tokens"` |
  `"cogate"`; default `"cogate"`, so unarmed digests don't move).
- **Consensus classification (§14.2).**
  - `open_rule` and the `[lane_b]` thresholds are overlay-settable
    (`ALLOWED["lane_b"] = None`).
  - The code for (a) is new, so it needs the **capability release**, then an
    overlay switches it.
  - `remainder` stays release-only.

### 12b.5 LB5: sentinel interaction (D35)

- With 4 suite slots and one suite held out each era (§16), **about a quarter
  of suite targets are deferred at any time.**
- **[DECIDE D35, rev4]:**
  - (a) **Accept and document (recommended).** A suite-target entry whose
    target is the current sentinel defers to the next era.
  - (b) Exempt lane B's own declared target from the sentinel, so it's held
    out of guards only.
- **What the miner sees.** The next era's sentinel **can't be known at
  commit**, by design (drawn from the era-boundary block hash).
  - `cascade status` shows the current era's sentinel.
  - A deferred entry shows `deferred: sentinel_target`.
  - **A deferral doesn't consume the miner's per-round submission:** the
    entry stays queued and uncharged until its target isn't the sentinel.
  - There's a bound, `sentinel_max_defer_eras` (recommended 4), after which
    it's judged on its target regardless, logged.

### 12b.6 LB6: miner cost of expert-only legs

**How a leg is budgeted today (verified).**

- `contract.for_hours(leg_hours)` (`cascade/shared/config/training.py:380-413`)
  sets `training_steps = 23,841 × leg_hours / target_train_hours`. Deployed
  `[v2.training] target_train_hours = 3.0` and `[lane_b] leg_hours = 3.0`, so
  a lane B leg runs **the same 23,841 steps as a full lane A leg**.
- The leg stops at the step count or at the wall guard,
  `max(heat_guard_factor × hours, heat_guard_floor_seconds)` = 1.0 × 3 h =
  10,800 s, **capped at the contract's `max_train_seconds` (18,000 s,
  `[v2.training]`)**. A time stop is flagged `deadline_hit`.
- **The budget is steps, not hours.** The miner pays until the leg ends, so a
  faster expert-only step does lower the bill. (An earlier session answer said
  "rent is SKU × hours"; that was wrong.)
- **Guard-vs-wall risk.** The measured full-leg wall on 8×H100 is 13,600 s
  (`[round] funded_sku_wall_seconds`), above the 10,800 s guard. Whether that
  figure is training-only or the whole leg (boot, corpus, harvest) is
  **unverified**. If it's training-only, an expert-only step must be ≥ about
  1.26× faster or every 8×H100 lane B leg truncates. LB6 splits the two.
- GPU choice today: lane B uses `GpuChoice` from the **shared** `[round]
  funded_pod_skus` (deployed `["8xH100", "8xH200"]`, `funded_sku_per_leg =
  true`, $20/h cap for no-choice legs). `lane_b_driver.py:1305`.

**D42 decided (owner, 2026-11-21): cheaper GPUs allowed, and the miner chooses
the leg length, up to 6 h.**

1. **Lane B's own GPU list.** New `[lane_b] funded_pod_skus`, separate from
   `[round]`, so adding a cheap GPU for lane B never opens it to lane A legs.
   - A GPU type enters the list only after (i) the pinned image (torch
     2.4.1+cu124) is qualified on it, (ii) its expert-only step rate and leg
     overhead are measured (LB6), and (iii) the leg fits memory **without
     changing micro-batching** (a different accumulation changes numerics).
   - The type is recorded on the leg (`gpu_name`, already passed to `done()`),
     and an audit re-derivation must use the same type, because a different GPU
     count changes reduction order.
2. **Miner-chosen leg length.**
   - The miner sets `train_hours` at funding: `0.5 ≤ train_hours ≤
     [lane_b] max_leg_hours` (**6.0**), in 0.5 h steps.
   - **[DECIDE D42b] What "hours" means:**
     - (a) **wall hours on the chosen GPU (recommended).** At admission,
       `steps = floor(train_hours × 3600 × measured_steps_per_s[gpu])`,
       fixed then and recorded in the signed leg record. Reproducible, and it
       matches what the miner pays for. A cheap GPU buys fewer steps per hour.
     - (b) **reference hours.** `steps = 23,841 × train_hours / 3.0` on any
       GPU. Equal training for an equal setting, but wall time on a slow GPU
       can be several times longer, which collides with the 5 h contract cap
       and the era timing below.
   - **Guard:** `train_hours × [lane_b] guard_factor` (recommended 1.25) plus the
     measured overhead, capped at **`[lane_b] max_leg_seconds`** (recommended
     6 h × 1.25 = 27,000 s). This is a new lane B cap. **`[training]
     max_train_seconds` is a contract-digest field and is never edited for
     this.** The lane B leg contract is derived with the lane B cap instead
     (a code change in `for_hours`'s caller, release-gated).
   - **Era timing.** The existing cross-era admission (`rolling.py:401-425`)
     already holds a leg that can't land inside the current era until the next
     era's pre-train window. A 6 h leg in a 12 h era therefore starts only in
     the first ~half of an era (or waits). The miner sees the expected landing
     boundary at funding.
3. **Pay and gains.** Longer legs are a miner trade of cost for gain; pay
   tiers, `gain_margin` and the regress-none guards are unchanged. More drift
   per merge is what the guards are for.
4. **Spam.** Cheaper legs mean more entries. The existing per-coldkey caps
   apply unchanged; revisit if the queue grows.
5. **Consensus.** `max_leg_hours`, `max_leg_seconds`, `guard_factor` and the
   `train_hours` range are consensus (validators check the leg's recorded steps
   and guard against them). The per-GPU step-rate table is **trainer-side
   operational config**: steps are fixed at admission and carried signed, so
   satellites never need the table. **(rev5 correction)** `ALLOWED["lane_b"] =
   None` means every `[lane_b]` key is overlay-settable (`netconfig.py:83-87`,
   `:213`; only `remainder` is release-only). The **new keys** still need the
   capability release, because nodes must parse and enforce them. After that,
   their values (`max_leg_hours`, `funded_pod_skus`, …) can change by signed
   overlay.
6. **Miner guide.** A per-GPU table of measured step rate, $/h cap and cost for
   1, 3 and 6 h, generated from the LB6 evidence (WB20).

**LB6 now measures, per candidate GPU type (one pinned-torch testnet pod per
type, OA8):**

- expert-only step rate and leg overhead (boot, corpus, harvest), measured
  separately;
- a full lane A leg on 8×H100 for comparison, which settles whether 13,600 s is
  training-only;
- peak memory at the contract micro-batching;
- image qualification (driver and pinned torch runs).

**Status: not run: blocked: needs owner (pod approval, OA8).**

### 12b.7 Split of purpose (rev4; also stated in §2)

- **Lane B improves the lane A model on declared targets:** trainer-run,
  merged into KING_A, contamination-gated.
- **Lane D explores the architecture with unconstrained training:**
  miner-run, never merged, forward-only judged.

---

## 13. Lessons from past incidents (OPSLOG review 2026-11-21)

| Past failure class | Example | Defence here |
|---|---|---|
| Bench hold that never ends; partial counted as infra | 11-15 / 11-16 "sidecar produced no report", `gift-eval:partial` | §1.3b `checkpoint_fault`; §10.4 hold cap; evidence kept; T17 |
| Bench pod disk full | 11-15 at 98%; leftovers deferred king checks | byte LRU; `pod_disk` under lane D load; no ad-hoc work on eval pods |
| Sweep blocks the weight loop | weight gaps during sweeps | invariant 6; keep-alive before S6; T15 |
| Eval-host code drift | valpriv `TypeError` for 8 h; bench image lacking `fev` | worker handshake; stderr capture; eval host synced at S5; T19 |
| Data time ≠ first-known time | partitions landing late, backfills | §1.3 first-seen rule; T8 |
| Front-running / copying | incentive lag after dethrones; scraping public repos | house refresh on every KING_A change; exact dedup; T16 |
| Private artifact leaks | private code via a shared-bucket tar; credentials in `ps` | own bucket; no credentials on command lines; presigned only |
| Hippius credential scope and rotation | rotated key lacking bucket access | W3a; rotation runbook; T20 |
| Hippius outages, slow transfers | 500s; upload timeouts; 1 MB/s scp | prefetch at reveal; resumable cache; async mirror |
| Box OOM / restart in scoring windows | 10-31 OOM killed all services | nothing big on the box; S5 rules; report size bound; `box_ram` |
| Live overrides vs repo; startup gates at externals | 21 unPR'd overrides; a 128-byte note crash-loop | keys typed in main the same day; CI startup gates; signed-overlay activation (D10; S0c, S6 **(rev6)**) |
| Stale service code | intake ran stale code for a day | `python -P` + code-path guard; deploy-gate verify |
| Settlement-timing traps | S+2 rule; legs after the last settlement | pending carries across eras; T18 |
| Satellite replay divergence | reason-string mismatch ⇒ burn | invariant 7; replay CI; T12 |
| Window-rotation variance | 0.62 LCB swing | judged once on a frozen set; T14 |
| Cross-host numerics | driver 595 vs 580 | paired-on-one-host gate; S4 tolerance |
| Lium lemons / wrong listings | "8×H100" booting as 4× PCIe | on-rent check; replacement runbook |
| Stale known_hosts | king leg hung on a host key | per-pod pinned known_hosts |
| Balance zero / orphan pods | Lium terminated all pods at zero | `runway`; testnet TTLs |
| Alerts not delivered | `bench_infra` into an empty webhook | delivered test alert in the S6 gate; follower patterns |
| Contract-digest drift | a runtime field blocked the king leg | lane D touches no contract term; a pending entry is pinned to its `ready` contract |
| PR hygiene | stacked merges; `git add -u` swept an override | one PR per WP (rev6.2: tests-only PRs may batch WPs, §3.1 rule 6); explicit paths; verify main |
| Identity / disputes | recycled uids | hotkey + digest identity; owner-only archive 14 days |
| Pool publish fragility | the job died for 6 days | dedicated builder; `pool_d_stale` |
| Suite identity / partial | FEV scope-skips counted as partial | W3b identity checks |
| Remote writes need the owner | classifier blocks | owner-paste scripts |
| ~~Blackwell vs pinned torch~~ | ~~sm_120 unsupported by torch 2.4.1~~ **(rev6.18)** handled by qualified scoring environments (B5) | D11 eval-host GPU class |
| Private offload rule | targets stay local | D18 / invariant 8 |

---

## 14. chain.toml, the signed overlay, and keeping validator updates rare

**Owner goal (2026-11-21):** codify as much as possible so that validators,
external satellites especially, do **not** have to keep updating. In the past
we drifted into relying on hand-edited files and ad-hoc satellite updates.

### 14.1 Where we are today (verified)

- **The signed overlay exists in code but has never run on mainnet.**
  - DEC-CA-0057's owner-signed network config (`config/network/seq-<n>.json`
    and `latest.json`, adopted at 51% stake, applied at an era start) is built
    (`cascade/shared/netconfig.py`).
  - It has **never been published**. The live validator still logs
    `S3 get failed for config/network/latest.json … HF fallback … 404` on
    every poll (`/root/cascade/validator.log`, 2026-11-21 11:27).
- **Config still changes by hand.**
  - The deployed `/root/cascade/chain.toml` carries **21 `LIVE OVERRIDE`
    markers** and differs from `origin/main` by about 119/116 lines.
  - Decided-vote blocks typed only into our deployed file caused the
    external-validator crash-loop and PR #457 (OPSLOG 2026-11-13).
  - The drift guard (`decided_votes.toml` + `cascade-audit decided`, PR #458)
    exists to catch this.
- **An old node can't partly accept an overlay.** An overlay with any table
  the node doesn't know is **refused whole**: `"[{table}] (not on the
  DEC-CA-0057 allow-list)"` (`netconfig.py:195-217`). `ALLOWED` has no
  `lane_d` today (`netconfig.py:83-118`).
  - An overlay that touches `[lane_d]` before a node runs the capability
    release would be refused **along with every other key in that overlay**.
  - The **trainer runs the same resolver** (DEC-CA-0057 §6), so it must take
    the capability release too.
- **The readiness note is full** (128 bytes). Overlay adoption uses the
  `cfg-` segment, which outranks older feature segments, so lane D must not
  add a note segment of its own.

### 14.2 The rule: one capability release, then overlays only

| Change type | How it ships | Validator action |
|---|---|---|
| New code (lane D scorer, report schema, replay rules, intake, audit) | **One capability release** | Update **once** |
| Lane D tuning: `from_block`, `until_block`, `share`, `takes_from`, `contexts`, `horizon`, `forward_margin_hours`, floors, `d_margin`, `alpha`, `bootstrap_B`, `gate_suites`, `gate_weights`, ~~`unselected_suite`~~ **(rev5) dropped:** the held-out suite is the §16 sentinel, drawn from `[v2] sentinel_suites` (W-S1); a lane D key would duplicate it, `tau_public`, court, rate limits, TTLs, `max_bench_holds`, **(rev3)** `max_mirror_lag_hours`, `confirm_on_king_change`, `max_judged_per_coldkey_per_settlement`, `max_coldkey_unresolved_settlements`, `max_reveal_delay_days`, `lowfreq_freqs`, **(rev5)** `ready_ttl_blocks` (B10), `max_source_share` (B14); **(rev6.9)** `backtest_weight_cap`, `backtest_ratio_floor` (W5i; the overlay is refused if it breaks the `d_margin` inequality); `remainder` is release-only (B5) | **Signed overlay** (add `ALLOWED["lane_d"] = None` minus `remainder`, which stays release-only as a burn destination, the same rule as `lane_b.remainder`) | **None**: nodes poll every 300 s, adopt by stake, apply at `effective_block` |
| **(rev3)** Cross-lane sentinel (§16): `[v2] sentinel_from_block`, `sentinel_suites`, `sentinel_ratchet`, `sentinel_tau` | **Signed overlay** once the capability release adds them to `ALLOWED["v2"]` (`netconfig.py:99-108`) | None after that release |
| **(rev6.10)** Lane D tuning, added: `embargo_hours` (48), `max_windows_per_settlement` (~~rec 2000~~ **(rev6.11)** 5000), **(rev6.11)** ~~`embargo_window_share_cap` (0.25)~~ **(rev6.12)** `min_fresh_window_share` (0.5), **(rev6.14)** `eval_delay_hours` (6; `forward_margin_hours` default now 0) (W2e); the per-feed `embargo_hours` lives in the forge catalog, is clamped to the global and pinned per snapshot (W6b), so it needs no overlay | **Signed overlay**, as the lane D tuning row above (`ALLOWED["lane_d"]`). A raise of `embargo_hours` above the enforced reveal delay is unsafe: ~~W6rv makes the reveal job read the same value~~ **(rev6.11)** W6rv isn't built, so `embargo_hours` must stay ≤ lane A's actual reveal delay (~48 h while lane A's cadence stays daily) | None |
| **(rev4)** Lane B: `suite_targets`, `contamination_hashset`, `contamination_min_hits`, `suite_target_max_repo_mb`, `min_experts_frac`, `max_experts_frac`, `router_rows`, `traffic_steal_bound`, `open_rule`, `sentinel_target_rule`, `sentinel_max_defer_eras` | **Signed overlay** (`ALLOWED["lane_b"] = None`, `netconfig.py:83-87`) once the capability release ships the code; `remainder` stays release-only, **(rev5) and so do `contamination_hashset`, `[lane_c] remainder` and `[lane_d] remainder`**: the release adds `("lane_b", "contamination_hashset")`, `("lane_c", "remainder")` and `("lane_d", "remainder")` to the release-only tuple (`netconfig.py:210`). Lane C's remainder is overlay-settable today (`ALLOWED["lane_c"] = None`, `:87`, not in the tuple), an existing gap closed in the same release, because overlays can't replace release-only data pins (`docs/ARCHITECTURE.md:471`). The other new keys (incl. rev5 `funded_pod_skus`, `max_leg_hours`, `max_leg_seconds`, `guard_factor`) stay overlay-settable. Each key drops from the digest at its default, so unarmed digests don't move | None after that release |
| Activation | Overlay types `[lane_d] from_block` (at an era boundary, as lanes B and C allow today) | None |
| Kill switch | Overlay sets `until_block` or `share = 0` | None |
| Main-validator-only operations (vault bucket, `pool_d` location, eval host, sub tokens, bench queue policy) | A **non-consensus** `[lane_d_ops]` table read only by our main validator and the lane D intake; satellites never read it | Our main validator only |
| New benchmark suite for the gate | Runner code is **main-validator-only** (satellites replay rows; suite names are data in the signed report) | Our main validator only |
| Locked training terms, security roots, validator-read storage keys, SDK pin | Release (unchanged DEC-CA-0057 rule) | Update |

**(rev3) If D21 option 2 is chosen** (return the unpaid king slot to lane
A), `[lane_d]` gains a **release-only** destination key, `house_remainder`,
refused by the overlay lint and `check_overrides` like `remainder`. Under
option 1 nothing changes here.

### 14.3 Design rules so satellites rarely need code

1. **Satellites only verify and do arithmetic.** Anything that needs private
   data, GPUs or benchmarks (forecasting, scoring, sweeps, eligibility against
   `pool_d`) runs **only on our main validator**. Satellites verify the
   signature, check that the parameters stated in the report equal
   `config_at(boundary)` from the adopted overlay, check structural rules,
   re-run the bootstrap from signed rows, and derive weights.
2. **The report carries its parameters.** Every value a satellite needs (cohort, frozen window
   ids, margins, alpha, B, seed, suite name, court, share) is **in the signed
   report** and cross-checked against the overlay. New knobs then need no
   satellite logic, as long as the arithmetic is generic.
3. **Anticipate later changes in schema v1** (`lane-d-report/1`): a horizon
   *list*, a suite *list*, per-entry context, room for multiple court
   members, an optional field map. Satellites refuse an unknown schema version
   cleanly, so a schema bump is a known, rare release, never a surprise fork.
4. **Structured comparisons only.** No free text in anything replayed (the
   reason-string burn incident).
5. **Code and config keys are added together.** Every new key ships with its
   loader parse, its `ALLOWED` entry and a round-trip test in the same PR
   (the config-knob memory rule).

### 14.4 Prerequisites before lane D (fix the overlay pipeline first)

These come before S2's lane D work, as stages **S0a (de-drift), S0b (note
budget) and S0c (overlay bring-up)** in §5:

1. **Reconcile config.**
   - Diff the deployed chain.toml against main and classify each of the 21
     overrides: belongs in main, belongs in an overlay, or is an operator-only
     knob.
   - Then PR main and run `cascade-audit decided` clean.
   - **No new hand edits on any validator** from here on.
2. **Testnet overlay drill.** Publish a **no-op overlay (seq 1)**, then a
   real one (one harmless allow-listed key). Show fetch, verify, `cfg-`
   signalling, the stake lock, application at `effective_block`, and
   `config_at` replay by the audit. Then show a deliberately refused overlay
   (an unknown table), refused cleanly with **no** partial application.
3. **Mainnet overlay drill.** A no-op seq-1 overlay on mainnet, to show
   satellites actually fetch and signal it, before any overlay carries a real
   change.
4. **Runbook.** Every future config change is an overlay plus a PR to main
   the same day, never a validator edit. Rollover-decided blocks are typed
   by overlay (DEC-CA-0057 §7). Run `cascade-audit decided` after every
   settlement.

### 14.5 Consequences for the rollout stages (§5)

All folded into §5:

- S0a de-drift, S0b note budget and S0c overlay bring-up before any lane D
  code ships.
- S2b catch-up release, so the lane D release diffs only the lane D and lane B
  build WPs (§5 S2) **(rev6)**.
- S5 capability release to every node **including the trainer** (shared
  overlay resolver).
- **(rev5)** S6: activation and payment are **one overlay** (D45, no
  `share = 0` shadow). The observed `cfg-` stake before its `effective_block`
  is the upgrade proof.
- Every `[lane_d]` overlay restates lane A keys byte-identically, because an
  old node refuses the whole overlay.

---

## 15. Incentive review (rev3): game theory across lanes

This comes from a read-only mechanism-design review against the deployed
economics (`/root/cascade/chain.toml`), the lane decisions and OPSLOG miner
behaviour.

**Everything here is a tokenomics or owner call: both sides are given and
nothing is adopted.** The judging core (forward-only windows, weights frozen
at `ready`) was found sound against target leakage. The weak part is pay.

| # | Issue | Option A (keep v2) | Option B (proposed) |
|---|---|---|---|
| **D25** | **Court pays rent indefinitely; salami-slicing.** The court is rank-decayed (~~57/29/14% of D's share at `king_decay` 0.5~~ **(rev6.23)** 80/15/5% of D's share by `court_weights`, 2 priors) with no TTL, and the bar is a fixed 0.01. A 3% gain released as three 1% crowns fills the whole court | Simple; matches lane A's court shape | Pay D wins on lanes B and C's `full_pay_tiers` + `half_life_rounds` + `win_ttl_eras` shape, keyed to the crowning LCB over max(KING_D, house). Or a tenure-decaying bar (e.g. 0.03 → 0.01). Or time-bound prior-king tails, like lane A's `court_bp` / `court_blocks` |
| **D26** | **Lane A is a free input to D, and D's 20% comes out of A.** **(rev6:** D's share is now 25% of the total, D1/D48**)** The cheapest winning D entry is "fine-tune the published house". Generators and recipe are published (**(rev5)** `[generation] king_publish = "recipe"`, `chain.toml:1555` on main `07117e1`; not `[v2]`) | Lane A is paid by its own ladder; D pays only for beating the house | A **lineage royalty**: a fixed ρ ≈ 15–20% of each D win's pay goes to the lane A king hotkey that was the house at the winner's T_E. Calibrate ρ in shadow with a weight-space nearest-ancestor measurement. Never self-declared lineage |
| **D27** | **Withholding from A.** A strong generator may earn more used privately in D than published in A | Accept; monitor how many D kings are far from every published A checkpoint | The D26 royalty, plus keep A's share ≥ D's |
| **D28** | **Split.** v2 has A 30 / B 25 / C 25 / D 20 when all are armed | v2 split | Reviewer: **A 40 / D 25 / B 20 / C 15**, with D ramping 15 → 25% after ≥ 3 miner crowns pass the gate with the unselected suite not falling. Now (B, C off): A80/D20 is acceptable. Interacts with **D21**: the reviewer prefers returning the unpaid house slot to lane A over burning it **(rev5) DECIDED (owner): A 25 / B 25 / C 25 / D 25** (§6 D1, D28). |
|  | *(rev4) Share assumption: B launches trainer-run, so its share pays generator-funded legs merged into KING_A. Until B's gate opens its share burns (D37). The A40/D25/B20/C15 proposal stands as an option; no default change.* | | |
| **D29** | **Frequency bias in frozen sets.** Sub-hourly data reaches the floors first, so miners specialise on sub-hourly | v2 freeze rule | Per-frequency-bucket and per-domain window quotas inside a frozen set, equal cluster weights. Freeze when every bucket meets its quota or a max wait passes |
| **D30** | **TIME isn't truly unselected.** It's in lane A's `require_public` gate and the king check, and DEC-CA-0071 (pending: PR #482; rev5 renumber) adds `suite:time` as a lane B target, so the house is optimised on it | Keep TIME as the unselected suite | Drop `s-time` from DEC-CA-0071, or choose a sentinel that no lane gates or pays on |
|  | *(rev3) D30 is addressed by D33: with a rotating sentinel no suite is permanently gated or targeted, so the held-out reading stays unbiased in expectation.* | | |
| **D31** | **Sybil queue griefing.** Cheap coldkeys can fill the 8 judged slots with junk | The A8 caps + registration cost | Round-robin by coldkey; priority by hotkey age or prior non-junk LCB; a cheap pre-screen (small window subset; drop LCB < −x before a full frozen set) |
| **D32** | **Lane B vs D.** Once D runs it dominates lane B as designed (one D king slot ~11–20% of emission vs ~1.8% per B slot) | Keep B as trainer-run legs | Re-scope B to the §12 expert-delta variant when D arms, with row-level dedup against published D kings' rows |
|  | *(rev4) **D32 resolved:** keep lane B trainer-run with miner-chosen experts (§12b). Lane B and D have distinct purposes (§2 item 8), so D's higher king pay doesn't make B redundant: B is the only lane whose wins merge into KING_A.* | | |

Synergies from the same review:

- **Keep:** the house as anchor and floor (A → D).
- **Reject for now:** D → A warm start, and D as a distillation teacher for A
  (contamination and centralisation risk).
- **Later:** lane E on any king, judged only against a plain king so E pay
  never stacks on D pay for the same gain.

The reviewer's view on the v3 "Proposed design changes": adopt 1–4, 7–10, 12
and 14 before S6 (**(rev6)** S6b is merged into S6, D45) (with 10 = gate at 4096 for everyone). Treat 5, 6, 11, 13,
15 and 16 as incentive-neutral ops. Those remain owner decisions and are
**not** applied here.

---

## 16. Cross-lane rotating held-out sentinel (rev3)

**Why.** Every public suite we gate or pay on gets optimised by miners and by
our own pipeline. Today:

- lane A gates on all four suites (`[v2] bench_gate_mode = "enforce"`, the
  `avg_noninf_winrate` rule; `require_public` for lane B);
- **(rev5)** the lane A v2 king check is the bench gate's rule over the same
  **four** suites (GIFT-Eval, BOOM, TIME, FEV; `BENCH_FIELDS_V2`,
  `validator/king_check.py:77-82`): `king_check_gate` (`:93`) calls
  `evaluate_bench_gate` with the king as challenger and the init as reference
  (average ratio, per-suite non-regression at `alpha / 4`, win rate). It does
  not "average six numbers";
- DEC-CA-0071 (pending: PR #482; rev5 renumber) would add suites as lane B targets.

So no suite stays a clean "are we really improving" reading. That's the D30
problem. A fixed held-out suite doesn't fix it either: a suite held out
forever drops out of regression coverage, and miners can still optimise for
it.

**Rule (D33).**

1. **The draw.** At each era boundary, one suite from `sentinel_suites`
   (default `["gift", "boom", "time", "fev"]`) is drawn from the
   **era-boundary block hash**, the same for every validator and
   unpredictable at commit time.
   - The main validator records the drawn suite in the settlement receipt
     and the lane D report as a structured field. Satellites verify it from
     the block hash; block hashes stay readable on pruned nodes, unlike
     past-block state.
   - A satellite that can't read the hash replays the recorded value.
2. **Held out that era, in every lane:**
   - **Lane A:** the gate and the king check use the remaining three suites.
     The sentinel is swept for each candidate that reaches the gate.
   - **Lane B:** it's neither a guard nor a target that era (D35).
   - **Lane D:** it isn't the drawn gate suite (§1.3b); it's swept for each
     new KING_D.
3. **The other suites remain full regression guards** in every lane. That's
   stronger than a rotating gate, which would check only one suite per
   crowning.
4. **Published every era:** each lane's new king's sentinel numbers, plus the
   house's, on a per-suite board in `status/lane_d.json` and the lane A
   status.
5. **Optional ratchet (D34).** A candidate in any lane must satisfy
   `LCB_α(sentinel) ≥ −sentinel_tau` against the **current lane A king**,
   re-scored paired on one host, never against stored numbers. With the
   ratchet on, a regression on the held-out suite is still blocked; it's
   just never used to *select* or *pay*.

**Cost.** Lane A's bench sweep stays at four suites: three gate and one
sentinel. With the ratchet off, the sentinel sweep can run after the verdict
and stays off the critical path. Lane D: one drawn suite plus the sentinel
per new king, roughly as v2.

**Consensus and rollout.**

- **Lane A:**
  - This is a change to lane A's gate rule, so it's a capability release,
    then a signed-overlay switch-on.
  - New `[v2]` keys: `sentinel_from_block` (0 = off), `sentinel_suites`,
    `sentinel_ratchet` (bool), `sentinel_tau`. Each needs a loader parse, an
    `ALLOWED["v2"]` entry (`netconfig.py:99-108` lists the `[v2]` allow-set
    explicitly) and round-trip and mutation tests.
  - It ships in the **S2b catch-up** or the S5 capability release.
  - **(rev5, D45)** It arms by overlay at an era boundary and affects the
    gate from that block, with no shadow eras. Its behaviour is proven on
    testnet first (TS1–TS5).
  - **(rev6.1) Mixed-fleet hazard (RV3 note 2; not applied, see Proposed
    design change 26).** A pre-W-S1 node refuses a whole overlay carrying
    `sentinel_*` keys, and a pre-W-S1 satellite fails replay on every armed
    receipt with a gate or king check. Change 26 proposes gating the arming
    overlay on R4's observed `cfg-` stake ≥ 2/3 **and** every known
    satellite on the W-S1 release.
- **Lane B:** it's off, so it adopts the sentinel from its first armed era.
  DEC-CA-0071 (pending: PR #482; rev5 renumber) needs a one-paragraph amendment for D35.
- **Lane D:** built in from launch (§1.3b).
- **Satellites:** they verify the recorded draw against the block hash and
  re-run the gate arithmetic from signed rows (§14.3). They need no bench
  hardware.

**Work and tests.**

- **WP W-S1:** the sentinel draw, the lane A gate rule change behind
  `sentinel_from_block`, the receipt field and the overlay keys (**(rev5)**
  no shadow logging, D45).
- **WP W-S2:** the lane B guard/target exclusion (with the DEC-CA-0071
  amendment).
  - **(rev6.2, owner 2026-11-22) Era-init acceptances.** A warm-start
    acceptance made at an era-init block is governed by the sentinel of the
    **era that init block opens**, not the era it closes. A follow-up for
    the acceptance callers: the trainer's `accept_candidates`
    (`cascade/trainer/merge_acceptance.py`) and the validator's
    `verify_acceptance` (`cascade/validator/merge_acceptance.py`) both pass
    that era's sentinel.
- **WP W-S3:** publication on the status boards.
- **Test rows:**
  - **TS1:** two validators derive the same sentinel from the block hash, and
    a satellite replays the recorded value when the hash is unreadable.
  - **TS2:** lane A with the sentinel armed. Verdicts use three suites, and
    a candidate regressing only on the sentinel is blocked iff the ratchet is
    on.
  - **TS3:** lane A **unarmed ⇒ byte-identical** receipts and verdicts
    (invariant 1 applies to lane A here).
  - **TS4:** a lane B entry targeting the sentinel suite is deferred (D35).
  - **TS5:** draws are independent per era (a repeat has probability
    1/4). The test checks the draw is deterministic from the hash and
    uniform over `sentinel_suites`.

---

## 17. Completion rubric and status board (rev6)

**(rev6)** This section is the one source of truth for completion.

- `scripts/laned/status` (built by W0c) computes every level below from git
  (`git log origin/main`), GitHub Actions results on main and the files under
  `evidence/`. It writes two outputs to branch **`laned/build-state`**, which
  is never merged to main:
  - `docs/plans/STATUS.md`, for people;
  - `status/laned_build.json`, for agents.
- **Agents never hand-edit status.** "Done" claimed in a PR, in chat or in any
  other file doesn't count. The rest of this plan's "a WP is done only when
  its Accept exits 0" is the L3 rule below.
- **Bootstrap.** Until W0c is at L3, the orchestrator maintains
  `status/laned_build.json` on `laned/build-state` by hand, using these levels
  and the same evidence. W0c is the first WP. From W0c at L3 on, status is
  only what the script computes.

### 17.1 WP levels

Each level needs all the levels below it.

| Level | Name | Condition |
|---|---|---|
| L0 | todo | Listed in §3 |
| L1 | specified | `tests/accept/<wp>/` and `PATHS` merged on main, and those tests fail on main before L2 (checked by the W0c workflow's expect-fail run). **(rev6.17)** Or merged with the code in one PR whose `combined` run proved the new tests fail on the base and pass on the head |
| L2 | built | Implementation PR merged on main (checked with `git log origin/main`), explicit paths only (every changed path is in the WP's `PATHS`). **(rev6.17)** The merged PR carried `laned:<WP>` and changed at least one non-test path in the WP's `PATHS`; a PR without the label never builds the WP, however broad its `PATHS` |
| L3 | accepted | `scripts/laned/accept <WP>` green in CI on main after the merge. A later red run on main drops the WP back to L2 |
| L4 | evidenced | External and owner WPs only (Runs in includes `builder`, `pod`, `testnet` or `owner`): evidence JSON present, and the verifier passes against the live resource. n/a for ci-only WPs, which pass L4 automatically |
| L5 | reviewed | Covered by an RVn whose `evidence/rv<n>.md` has no open BLOCKING item for this WP |
| L6 | released | Its commit is an ancestor of a tagged capability release |
| L7 | live | Its activation block has passed on mainnet and its §10.7 monitor is green for ≥ 1 era. **Owner-confirmed:** the owner merges `evidence/l7/<wp>.json` (activation block, monitor, era range) |

**Side states**, always with a reason:

- `blocked:<decision Dnn | external | owner | dependency>`;
- `parked` (e.g. the §12 variant).

### 17.2 Stage completion

A stage is complete when every WP it lists reaches its required level **and**
its §5 exit gate has an evidence file the verifier accepts. That's the stage's
execution-WP evidence where one exists (e.g. X3's `evidence/s3_matrix.json`),
otherwise `evidence/gates/<stage>.json`, checked by
`scripts/laned/accept --gate <stage>`.

| Stage | Required |
|---|---|
| S0 | Every §6 decision is decided, withdrawn or deferred-to-stage in the §17.4 register; W0 at L3 |
| S0a | R6x at L4 |
| S0b | R2x at L4 |
| S0c | R7t and R7m at L4 |
| S1 | W1, W0b and W3a at L4; R16 archive growing (R16 evidence) |
| S2 | Every build WP at L5; the S2 gate evidenced. **(rev6.1)** A **build WP** is a §3.0 WP whose `Runs in` is ci-only, or ci plus pod/testnet for its CI-checkable parts. Owner execution WPs (R2x, R6x, R7t, R7m, R9, R13, R14, R15, R16 and every X*) are **not** build WPs; they belong to their own stages (S0a–S0c, S1, S2b, …). This is the set the §3.0 Review WPs paragraph counts ("W/R build WPs, not owner execution WPs") and the set "all build WPs" means in S4 and S5 |
| S2b | R9 at L4 |
| S3 | X3 at L4, with every §4 T-row passed |
| S4 | X4 at L4; all build WPs at L6 |
| S5 | X5 at L4; all build WPs at L6 |
| S6 | X6a at L4; the lane D WPs (§3.0 "Lane D core") at L7 |
| SB0–SB5 | The same levels per §5 row 12c: SB0 as S0 for D35–D42 and D42b; SB1 as S2 for its WPs; SB2 WB19 at L4; SB3 as S3 for TB10–TB27 and TS1–TS5; SB4 = S5 (shared release); SB5 the lane B WPs at L7 |
| S7 | X7 at L4 |
| S8 | W6-S8 at L4; each ~~added~~ **(rev6.6)** armed context rung at L7 |

### 17.3 Coverage counters

Published on every run, in both outputs:

- **T-rows** passed on testnet / total (S3 needs 100%);
- **invariants 1–12**, each with ≥ 1 test at L3 (S2 needs 100%);
- **orphan check:** every WP row, T-row and decision maps to an owning WP;
- **blocked count** by reason, oldest first.

### 17.4 Decision register and owner inputs

Each D-number has a status (`open` | `decided` | `withdrawn` |
`deferred-to-stage`), the option chosen, the date and who decided. The status
script parses this table on main; it changes only by an owner-merged plan PR.

- An **open STRUCTURAL (S)** decision, one that changes code structure or a
  release-only rule (e.g. D21, D40, D46, D48), **blocks its WPs**.
- A decision that only sets an **overlay-tunable (O)** value, or an ops-only
  choice, **doesn't block**. The WP builds with the recommended default and
  lists it under "defaults assumed" in its PR.

"Default" below means the §6 recommended default, accepted by the owner's
2026-11-21 answer ("all other recs"; §6 header).

| D | Status | Option chosen | Class | Decided |
|---|---|---|---|---|
| D1 | decided | `[lane_d] share = 0.25` from the total, `takes_from = "lane_a"`; A 25 / B 25 / C 25 / D 25 | O | owner, 2026-11-21 |
| D2 | decided | ~~Launch {4k, 8k, 16k}; {32k, 64k, 128k} at S8 by overlay~~ **(rev6.6)** All six contexts declarable from launch; each auto-armed on `min_long_series` + T7; **(rev6.15)** an unarmed declaration is scored at the highest armed variant below it, else skipped | O | owner, 2026-11-23 (rev6.6; was owner default 2026-11-21) |
| D2b | decided | One gate suite per crowning from the suites not held out; the sentinel is held out | S | owner (default), 2026-11-21 |
| D3 | decided | 1 open per hotkey, 2 per coldkey, 8 judged per settlement | O | owner (default), 2026-11-21 |
| D4 | decided | No eval fee at launch | S | owner (default), 2026-11-21 |
| D5 | decided | `d_margin` 0.01 (re-checked by T14), `forward_margin` ~~6 h~~ **(rev6.14)** 0 h + `eval_delay_hours` 6 h (owner 2026-11-23), 1000 windows / 20 sources; **(rev6.10)** `embargo_hours` 48 h (owner 2026-11-23), `max_windows_per_settlement` ~~rec 2000 (open)~~ **(rev6.11)** 5000 (owner 2026-11-23); ~~embargo hourly-and-slower only~~ **(rev6.12)** embargo for every cadence, ~~`embargo_window_share_cap` rec 0.25~~ **(rev6.12)** fresh-first draw, `min_fresh_window_share` 0.5, not frozen below it (owner 2026-11-23) | O | owner (default), 2026-11-21; owner, 2026-11-23 |
| D6 | decided | Queue behind lane A; second pod if crownings slip > 1 settlement | O (ops) | owner (default), 2026-11-21 |
| D7 | decided | 2 prior miner kings, ~~decay 0.5~~ **(rev6.23)** `court_weights` 80 / 15 / 5; house never in the court | O | owner (default), 2026-11-21; owner, 2026-11-24 (rev6.23) |
| D8 | decided | Losers private; vault TTL 4 eras; archive 14 days; delete | O | owner (default), 2026-11-21 |
| D9 | decided | House unpaid: the slot burns | S | owner (default), 2026-11-21 |
| D10 | decided | One activation overlay (D45) | S | owner, 2026-11-21 |
| D11 | decided | ~~Launch eval host: valpriv-4090 or a pinned-torch pod (≥ 24 GB)~~ **(rev6.18)** any GPU in a qualified scoring environment (B5) | O (ops) | owner (default), 2026-11-21 |
| D12 | decided | `pool` untouched; `pool_d` joined to `pool`'s ids, history to 16384 + 64 | S | owner (default), 2026-11-21 |
| D13 | decided | Open licences only; kings published | S | owner (default), 2026-11-21 |
| D14 | decided | BF16 only | S | owner (default), 2026-11-21 |
| D15 | decided | Lane D generation = lane A generation | S | owner (default), 2026-11-21 |
| D16 | decided | Standing approval for automated TTL deletion granted | O (ops) | owner, 2026-11-21 |
| D17 | decided | Loser rows signed-but-not-reproducible accepted | S | owner (default), 2026-11-21 |
| D18 | decided | Targets stay on the box | S | owner (default), 2026-11-21 |
| D19 | decided | Owner-run test entries on testnet in S3 | O (ops) | owner, 2026-11-21 |
| D20 | decided | ~~(a) restart the duel against the new king~~ **(rev6.11)** moot: confirmation duel dropped (owner 2026-11-23) | S | owner (default), 2026-11-21 |
| D21 | decided | Option 1: burn | S | owner (default), 2026-11-21 |
| D22 | decided | (b) record at `ready`, re-check at reveal | S | owner (default), 2026-11-21 |
| D23 | decided | (ii) targets, ids, `rows_sha256`, bootstrap verifiable | S | owner (default), 2026-11-21 |
| D24 | decided | 5% per-era false-crown target | O | owner, 2026-11-21 |
| D25 | decided | Option A (keep v2) **(see Proposed change 22)** | S | owner ("all other recs"), 2026-11-21 |
| D26 | decided | Option A (keep v2); share source is D48 **(Proposed change 22)** | S | owner ("all other recs"), 2026-11-21 |
| D27 | decided | Option A (accept) **(Proposed change 22)** | S | owner ("all other recs"), 2026-11-21 |
| D28 | decided | A 25 / B 25 / C 25 / D 25 | O | owner, 2026-11-21 |
| D29 | decided | Option A (keep v2) plus `max_source_share` (W5b) **(Proposed change 22)** | O | owner ("all other recs"), 2026-11-21 |
| D30 | decided | Handled by the §16 sentinel (D33) | S | rev3 (D33) |
| D31 | decided | Option A (the A8 caps) **(Proposed change 22)** | O | owner ("all other recs"), 2026-11-21 |
| D32 | decided | Lane B trainer-run (§12b.7) | S | rev4 |
| D33 | decided | Adopt the cross-lane rotating sentinel | S | owner (default), 2026-11-21 |
| D34 | decided | Sentinel ratchet on | S | owner (default), 2026-11-21 |
| D35 | decided | A sentinel suite is not a lane B target or guard that era | S | owner (default), 2026-11-21 |
| D36 | decided | Lane B arms trainer-run | S | rev4 |
| D37 | decided | `[lane_b] from_block` only once D41's gate is met; full share at that block (D45) | O | owner (default), 2026-11-21 |
| D38 | decided | (a) 8 MB `suite_target_max_repo_mb` | O | owner (default), 2026-11-21 |
| D39 | decided | Fractions 0.03125 / 0.1484375 | O | owner (default), 2026-11-21 |
| D40 | decided | (a) router rows frozen | S | owner (default), 2026-11-21 |
| D41 | decided | (a) `open_tokens` + a passing lane A king check | S | owner (default), 2026-11-21 |
| D42 | decided | Cheaper GPUs via `[lane_b] funded_pod_skus`; miner-chosen leg ≤ 6 h | S | owner, 2026-11-21 |
| D42b | decided | (a) real hours on the chosen GPU | S | owner, 2026-11-21 |
| D43 | decided | Immediate reveal (`min_blocks`) | S | owner, 2026-11-21 |
| D44 | withdrawn | — (lane A keeps cooldown; **(rev6.20)** lane B: EMA, no cooldown, owner 2026-11-24) | — | owner, 2026-11-21 |
| D45 | decided | No shadow modes; one activation block | S | owner, 2026-11-21 |
| D46 | decided | (a) house dated by its lineage | S | owner, 2026-11-21 |
| D47 | decided | (a) score the house as a BF16 cast; dedup against BF16 casts | S | owner (default), 2026-11-21 |
| D48 | decided | Fixed 25% shares of the total; lane D takes from lane A's 25% | S | owner, 2026-11-21 |

**No decision is open, so no WP is blocked on a decision.** Proposed change 22
asks the owner to confirm the five §15 rows recorded from "all other recs".

**(rev6.2) Owner decisions 2026-11-22** ("do all ur recommendations"). Build
rules inside existing WPs, not §6 D-numbers, so they aren't table rows:

| Item | Decision | WP | Decided |
|---|---|---|---|
| (c) | LB1 normalisation = WB11 schema 2 z-grid with RV6 low-information exclusion (§12b.1; #617/#621, #632) | WB11 | owner, 2026-11-22 |
| (d) | A deferred lane B funded entry's `PayerKeyVault` key has its TTL refreshed on each deferred tick (#629/#633) | WB18 follow-up | owner, 2026-11-22 |
| (e) | `laned_vault_ledger/1` is produced by W17 | W17 | owner, 2026-11-22 |
| (f) | An era-init warm-start acceptance is governed by the sentinel of the era its init block opens | W-S2 follow-up | owner, 2026-11-22 |
| (h) | Tests-only PRs may batch several WPs (one `laned:<WP>` label and one separately recorded accept run per WP; whole PR fails if any WP fails); ~~implementation, tests-correction, evidence and review PRs stay one WP per PR (§3.1 rule 6)~~ **(rev6.17)** one PR per feature, tests + code together, any number of `laned:<WP>` labels; evidence and review PRs stay one WP per PR (§3.1 rules 1, 6, 7) | all build WPs | owner, 2026-11-22; rev6.17 owner, 2026-11-23 |

**Owner inputs** (moved from Appendix A, rev6). These are inputs to runs, not
design decisions. They're never guessed: the dependent WP stays
`blocked:owner` until the input's evidence file exists. "Here" in the list
below now means that evidence file.

1. **OA1, prior-king pairs (B1 null type 2).** List the checkpoint refs
   (`repo@sha256:…`) of ≥ 3 consecutive lane A v2 kings that are light
   fine-tunes of each other (same lineage, adjacent reigns). Write them here
   as `king_t, king_t+1` pairs.
2. **OA2, seed replicas (B1 null type 2b).** Do any same-corpus,
   different-seed replicas exist (DEC-CA-0061, docs/OPERATOR.md
   seed-noise)? If yes, give their refs. If none, write "none".
3. **OA3, per-era false-crown target. (rev5) ANSWERED: 5%** (owner 2026-11-21). The recommended value is 5%. Confirm
   or set a value; it gates S4 (T14) and sizes `d_margin`.
4. **OA4, pod approval for B1 scoring and the B2 16k memory fit.** Either:
   - (a) use valpriv-4090 **outside lane A duel windows** (no rental; it
     touches an eval pod), or
   - (b) approve renting one pinned-torch pod (torch 2.4.1+cu124 capable,
     ≥ 24 GB), with a cost cap.

   State which, plus the time window.
5. **OA5, forge-mirror read access for B2.** The B2 history-length counts
   (≥ 16k, and the S8 lengths) and first-seen checks can't come from pool
   snapshots: they truncate history to 4096+720 and carry no timestamps.
   Approve running `scripts/laned_w1_measure.py` (or a mirror variant) on
   GitHub Actions or the builder with the `TSFORGE_S3` read credentials. Say
   where.
6. **OA6, pinned pool snapshots for B2's snapshot-diff bound.** Approve
   fetching ≥ 2 consecutive pinned daily `pool` snapshots and one `pool_mkt`
   run of snapshots to a non-box host (GitHub Actions or the builder), and say
   where. They're never fetched to the prod box.
7. **OA7, input plan.** The brief's PLAN START/END block contained only the
   template placeholder. I used the latest local plan (v2 design + the v3
   loop-readiness §3) as commit 1. If you meant a different v2 file, say so
   and the PR is rebuilt on it.
8. **OA8, pod approval for LB6 (WB19).** (rev5) Approve short pinned-torch testnet pods: one 8×H100 (expert-only leg plus one full lane A leg), and one of each cheaper GPU type you want offered to lane B miners (name them). Give a total cost cap.

| OA | Status | Unblocks | Evidence file |
|---|---|---|---|
| OA1 | open | B1 null type 1 (T14, S4 gate) | `evidence/owner/oa1.json` (king pairs) |
| OA2 | open | B1 null type 2b | `evidence/owner/oa2.json` (refs, or `"none"`) |
| OA3 | answered: 5% (D24) | — | — |
| OA4 | open (pod yes + cost cap, or the valpriv-4090 window) | B1 scoring; W1's 16k memory fit | `evidence/owner/oa4.json` |
| OA5 | open (host for `TSFORGE_S3` reads) | W1 history counts, `pool_d` size | `evidence/owner/oa5.json`, then `evidence/w1.json` |
| OA6 | open (non-box host for pinned snapshots) | W1 snapshot-diff bound | `evidence/owner/oa6.json`, then `evidence/w1.json` |
| OA7 | moot: the plan on the branch has been the input since rev3 | — | — |
| OA8 | open (pod yes, GPU types, total cost cap) | WB19 (SB2) | `evidence/owner/oa8.json`, then `evidence/wb19.json` |

### 17.5 Milestones

| Milestone | Condition |
|---|---|
| Build-complete | S2 complete |
| Launch-ready | S0–S5 complete; S3 at 100%; the revert overlay staged |
| Live | S6 complete (lane D) and SB5 complete (lane B) |
| Done | S7 complete for D and B |

S8 is tracked separately.

### 17.6 Agent rules

- Read `status/laned_build.json` (branch `laned/build-state`) before any work,
  and run `scripts/laned/status` after it.
- Only the status script (or the owner, for L7) moves a WP up a level. An agent
  may only set `blocked`, with a reason.
- Roles and the builder's no-edit list are in §3 conventions.
- Appendix A and `LANE_D_PROGRESS.md` are retired in favour of this board
  (Proposed change 21 adopted).
- **(rev6.17, owner 2026-11-23) The loop.** ~~Tests-only PR, then one
  implementation PR per WP (or a ≤ 3-WP disjoint batch).~~ One PR per
  feature: every WP the feature touches, one `laned:<WP>` label each, tests
  and code together, `PATHS` additions in the same PR. CI runs each WP's
  `--combined` (or full Accept), CROSS and the freshness guard; merge only
  when green on a fresh base. Never loosen an existing assertion without
  `laned:tests-correction` and a reason in the PR body. Plan text never
  rides in a WP PR: one plan PR (no `laned:` label) per batch of owner
  decisions.

---

## Appendix A: run status and owner actions (rev6: retired)

**(rev6)** Retired in favour of §17. Run status is the board
(`docs/plans/STATUS.md` and `status/laned_build.json` on branch
`laned/build-state`). The owner inputs OA1–OA8 are in §17.4.

---

## Appendix B: S1 results

**(rev6)** `docs/plans/LANE_D_S1_RESULTS.md` is now a pointer to this appendix
(Proposed change 21). Raw outputs are in
`docs/plans/lane_d_s1/`.

**Evidence rule.** Every number below cites its command, the pool snapshot
sha256, the host and a raw output file under `docs/plans/lane_d_s1/`. There
are no estimates, extrapolations or placeholders. Anything not run says
`not run: <blocker>`; anything below its floor says `below floor` plus the
counts.

**(rev6)** Owner inputs are listed in §17.4 (OA1–OA8).

### B2: W1 measurements

| Measurement | Result | Command | Snapshot sha256 | Host | Raw output |
|---|---|---|---|---|---|
| Do first-seen or ingest timestamps exist in `cascade/pool/`? | **No.** No match for `first_seen`, `ingested_at`, `ingest_time`, `landed_at` or `LastModified` (grep exit 1) | `grep -rnE "first_seen\|ingested_at\|ingest_time\|landed_at\|LastModified" cascade/pool/` at repo `3c67b64` | n/a (code check) | droplet (prod box, read-only grep) | `lane_d_s1/b2_first_seen_grep.txt` |
| pool_mkt cadence | **Scheduled every 6 h (`30 */6 * * *`), but scheduled runs skip while `[marketcast] from_block = 0`, which is the current value on mainnet and testnet.** Today pool_mkt publishes only on manual dispatch | `grep -nE "cron\|schedule" .github/workflows/publish-mkt-pool.yml` plus `publish-mkt-pool.yml:84-95` | n/a (code check) | droplet (read-only) | `lane_d_s1/b2_pool_mkt_cadence.txt` |
| Forward-eligible windows and distinct sources per daily snapshot (6 h margin, h = 64) | **not run: blocked: needs owner (OA6)**, a non-box host to fetch ≥ 2 consecutive pinned `pool` snapshots. Pool records carry no timestamps, so this run can only produce a **snapshot-diff bound (upper bound on eligible counts)**, never measured first-seen | `python -I scripts/laned_w1_measure.py diff …` (script in this PR) | — | — | — |
| Floors met under the snapshot-diff upper bound? | **not run** (depends on the row above) | — | — | — | — |
| Same for pool_mkt at 6 h | **not run: blocked: needs owner (OA6)**, plus pool_mkt isn't published on schedule while MarketCast is off (row 2) | — | — | — | — |
| Series and sources with history ≥ 4k / 8k / 16k (+64), the launch `pool_d` | **not run: blocked: needs owner (OA5).** Not answerable from pool snapshots, which truncate every series to 4096 + 720 (`builder.py:264-266`). Needs the forge mirror (`TSFORGE_S3`) | — | — | — | — |
| Series and sources with history ≥ 32k / 64k / 128k (+64), for S8 | **not run: blocked: needs owner (OA5)** (same reason) | — | — | — | — |
| Launch `pool_d` size | **not run: blocked: needs owner (OA5)** | — | — | — | — |
| 16k-context batch memory fit on valpriv-4090 (24 GB) | **not run: blocked: needs owner (OA4)**, pod approval (valpriv-4090 outside lane A duel windows, or a rented pinned-torch pod) | — | — | — | — |

### B1: null FPR

| Measurement | Result |
|---|---|
| Prior-king pairs (null type 1, primary) | **not run: blocked: needs owner (OA1** pairs, **OA4** pod) |
| Seed replicas (null type 2) | **not run: blocked: needs owner (OA2)** |
| Perturbed king (null type 3, the easy case; no recommendation from it alone) | **not run: blocked: needs owner (OA4)** pod approval |
| Per-entry FPR, null LCB quantiles 50/90/95/99, disjoint/resampled spread ratio, ~~confirmation-duel FPR~~ **(rev6.11)** single-test FPR on 5000-window W5j draws, per-era FPR at k = 1 / 8 / 32, `d_margin` for per-era targets 1/5/10%, at min_clusters 20/30/40 | **not run** (depends on the rows above). The per-era target itself awaits **OA3** |

Other notes from the build:

- `score` doesn't pass a training contract to `evaluate_checkpoint`, so the
  config.json and weights-header pin isn't armed. The code-identity
  byte-check still runs.
- The snapshot sha256 the scripts record is a digest of the unpacked
  directory; the published tar sha256 is recorded as given.
- W1 holds two snapshots in float64, so it must run off the prod box (OA6).

The script (`scripts/null_fpr.py`) and its tests
(`tests/unit/test_null_fpr.py`) are in this PR. CI runs the tests; nothing
was executed on the prod box.

**Repo fact affecting B1 (verified by the script build):**

- `calib_draw` (`cascade/validator/calibration.py:52`, wrapping
  `windows.horizon_draw` at `windows.py:414`) cuts **one window per series**:
  the last 64 points.
- A pinned pool of about 4.6k series (snapshot 9509400: 4,633) therefore
  gives about **4.6k windows at h = 64, not ~12k**. That allows about **4
  disjoint 1000-window sets, not 12**, for the spread check. Fewer sets make
  the disjoint SD noisier; the script reports the set count with the ratio.
- Reaching ~12 disjoint sets would need several snapshots' windows, which
  conflicts with "one window list", or a smaller `--n`.

**Known limitation (B1):** the scoring snapshot isn't forward-only, and fresh
forward windows shift in distribution. **(rev5)** The forward-window re-check
happens on testnet (S3); there's no mainnet shadow (D45).

---

## Appendix C: DEC-CA-0071 amendment rev4 (text)

This is the amendment as drafted in
`decisions/DEC-CA-0071-lane-b-public-suite-targets.md`, on top of PR #482.

### Amendment rev4 (proposed, 2026-11-21)

Status of this amendment: **PROPOSED**. This file carries PR #482's
unmerged text above unchanged, with this amendment on top. The full design
is in `docs/plans/LANE_D_PLAN.md` §12b; this section records the decisions.

**Owner direction (2026-11-21).** Lane B stays **trainer-run**: miners
submit a code-only generator and our trainer trains. Miners may also
(1) **choose which expert rows** are trained, and (2) **target one public
suite**, improving it significantly while regressing nowhere.

1. **Open option (b) is adopted: miner-chosen experts.**
   - The commit declares `declared_rows = {layer: [indices]}`, validated
     against `experts_per_layer = (128, 256, 256, 512, 512, 512, 512)`
     (`cascade/trainer/pyramid_model.py:42`). **(rev5)** Layer keys are MoE
     ordinals 0–6, not block indices. The clamp minimum applies only to
     layers with at least one declared row. Typed refusals:
     `declared_rows_out_of_range`, `declared_rows_bad_layer`,
     `declared_rows_count`.
   - The per-layer clamp becomes fractional (D39): `min_experts_frac =
     0.03125`, `max_experts_frac = 0.1484375`. That gives 128 → 4..19,
     256 → 8..38 and 512 → 16..76, replacing the 256-era `min_experts = 8` /
     `max_experts = 38`.
   - Claims follow the existing merge-acceptance deferral. The earliest
     reveal wins; a deferred claim shows `deferred: rows_held`.
   - ESFT stays the default when nothing is declared, and both modes
     coexist.
   - **Router rows, [DECIDE] D40:** (a) frozen, today's compact-rows
     backend (`cascade/trainer/expert_training.py:112-122`), recommended for
     first arming; or (b) the FlexOlmo unfreeze of the declared rows' router
     entries and bias, with a `traffic_steal_bound` guard.
2. **The static contamination check is a precondition, not an open item.**
   - `suite_targets` arms only after the check ships **and passes on
     testnet**.
   - At intake, before compute, exact and normalised (scale-, offset- and
     rounding-robust) hashes of every GIFT-Eval, BOOM, TIME and FEV test
     window are matched against `.py` numeric literals, `config.json` and
     any non-code bytes. A match is refused as `benchmark_series_embedded`.
   - The hash set is versioned and digest-pinned
     (`[lane_b] contamination_hashset`), refreshed only when a suite
     identity changes.
   - Guard 4 (whole-pool point gain ≥ 0) and the anchor guard stay the
     backstop for procedurally recreated series.
   - **[DECIDE] D38:** `suite_target_max_repo_mb = 8` (recommended) vs keep
     `[generator] max_repo_mb = 128` (`chain.toml:47`). DEC-CA-0028 already
     plans to stop `config.json` carrying bulk data.
3. **The opening gate is replaced ([DECIDE] D41).**
   - The routing-saturation co-gate (`routing_overlap` 0.8, `routing_jaccard`
     0.7) **has never been measured on mainnet**. The driver state shows
     `gate_opened_block = 0`, an empty `routing_history` and
     `lineage_tokens = 0`, because lane B is off.
   - The only measurement, the 2026-11-20 probe, found king-vs-init top-8
     overlap of about 1.1/8.
   - Options: (a) `open_tokens` + a passing lane A king check
     (**recommended**); (b) `open_tokens` alone; (c) keep the co-gate with
     re-measured thresholds.
   - The new key is `[lane_b] open_rule`, overlay-settable after the
     capability release.
4. **Sentinel ([DECIDE] D35).**
   - With the cross-lane rotating sentinel (plan §16), a suite-target entry
     whose target is the current sentinel **defers** to the next era: typed
     `deferred: sentinel_target`, not charged, bounded by
     `sentinel_max_defer_eras` (recommended).
   - The alternative is to exempt lane B's own target from the sentinel.
5. **Routing information for miners** (informational only, never consensus):
   per-king per-expert load, dead experts and per-suite context-only routing
   affinity, in a digest-pinned artefact.
6. **Expert-leg cost (D42, decided):** lane B has its own GPU list, so
   cheaper image-qualified GPUs are allowed, each measured first. The miner
   chooses the leg length, `train_hours`, from 0.5 to 6 h. Steps are fixed at
   admission and signed. The guard is capped by `[lane_b] max_leg_seconds`.
   `[training] max_train_seconds` is not changed.

New `[lane_b]` keys:

- `suite_targets`
- `contamination_hashset`
- `contamination_min_hits`
- `suite_target_max_repo_mb`
- `min_experts_frac`
- `max_experts_frac`
- `router_rows`
- `traffic_steal_bound`
- `open_rule`
- `sentinel_target_rule`
- `sentinel_max_defer_eras`

Each key: drop-when-default in the digest, overlay-settable via
`ALLOWED["lane_b"] = None` (`remainder` stays release-only; **(rev5)** so does
`contamination_hashset`, added to the release-only tuple), loader parse,
round-trip and mutation tests (plan WB10).

---

## Appendix D: related records

- `decisions/DEC-CA-0070-lane-d-open-weights.md`: lane D decision record
  (proposed). Its [DECIDE] table is the same as §6.
- `scripts/null_fpr.py`, `scripts/laned_w1_measure.py` (+ tests): S1
  measurement tools (B1, B2).

