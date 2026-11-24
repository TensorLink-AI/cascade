# Lane D: open weights on the cascade architecture

Lane D lets you train the cascade model **anywhere, on anything, with any
recipe or compute** and submit **only the weights**. The operator only
evaluates. A lane D entry wins by beating the current lane D king (KING_D)
on **forward-only private windows**: forecast targets that first reached our
data after both models were committed, so neither side can have trained on
them. Design record: [`docs/plans/LANE_D_PLAN.md`](plans/LANE_D_PLAN.md)
(DEC-CA-0070). The short CLI reference is
[MINER.md §15](MINER.md#15-open-weights-lane-d-the-cascade-lane-d-cli).

The `chain.toml` template ships lane D **unarmed** (`[lane_d] from_block =
0`). Networks arm it by a signed overlay at an era boundary, never by a
`chain.toml` edit. On mainnet the launch overlay
(`deploy/overlays/launch-20261125.mainnet.toml`) arms it at block 9586800
(about Wed 2026-11-25 22:28 UTC) with a 50% share, and sets
`public_weight = 0.0`: **at launch lane D is scored on the live half only**
(see [How entries are judged](#how-entries-are-judged)). Check
`cascade status --json` (`lanes.d`) and the public queue before you spend
anything.

## Contents

- [What an entry is](#what-an-entry-is)
- [Context is a declared scorer setting](#context-is-a-declared-scorer-setting)
- [Export, verify, submit, ready, commit](#export-verify-submit-ready-commit)
- [Licence and takedown](#licence-and-takedown)
- [How entries are judged](#how-entries-are-judged)
- [The house, the gate, the sentinel](#the-house-the-gate-the-sentinel)
- [Pay and the court](#pay-and-the-court)
- [Lane D FAQ](#lane-d-faq)

## What an entry is

A lane D entry is a standard `cascade_pyramid` checkpoint directory holding
**exactly** these four files and nothing else
(`cascade/lane_d/intake.py` `ALLOWED_FILES`):

| File | Rule |
|---|---|
| `config.json` | The contract's model config, pinned **whole** at `ready` (`cascade/lane_d/entry_config.py`): top-level keys only from `arch`, `base_arch` (= `cascade_pyramid`), `input_transform`, `moe`, `power_ema`, `quantile_levels` (the pyramid's 21 levels), and `moe` keys exactly `CascadeMoEConfig`'s fields; anything else is `409 config_key_not_allowed` / `config_value_not_allowed` / `config_unreadable`. At ingest its `"moe"` object must equal the contract's (`config_mismatch` otherwise) |
| `model.py` | Byte-identical to the repo's `cascade/trainer/pyramid_model.py` (`code_mismatch` otherwise) |
| `forecast_wrapper.py` | Byte-identical to the repo's `cascade/trainer/pyramid_wrapper.py` |
| `weights.safetensors` | BF16 tensors only, the contract's tensor names and shapes, every value finite |

Weights are BF16 only (`[lane_d] dtypes = ["BF16"]`, D14): about 2.5 GB at
the 120m size. An F32 checkpoint (lane A's dtype) is refused with
`dtype_mismatch`. The scorer upcasts to fp32 at load, the same for every
entry. Any other file (`.pt`, `.bin`, pickles, an extra `.py`) is refused with
`file_not_allowed` before anything is downloaded. Nothing in an entry is ever
imported or executed: the safetensors header is parsed by hand and only the
finite-weights check loads tensors.

Lane D never changes `pyramid_model.py`, `pyramid_wrapper.py`,
`base_arch_digest` or any lane A contract term, and it never triggers a
worker-image repin.

## Context is a declared scorer setting

Context is not a checkpoint property. At `ready` you declare the context your
entry is judged at, one of `[lane_d] contexts`. The weights load at any
context because the positional buffers are non-persistent.

**Why `moe.context_length` stays 4096 at every declared context.** The
contract pins `"moe": {"context_length": 4096, ...}` and ingest requires the
entry's `"moe"` to equal it, so a 128k declaration still ships 4096 there.
That is intended, not a cap: the field is positional *metadata*, not a limit
the scored model enforces.

- The pyramid model never reads `context_length` (or `max_patches`): time
  attention uses xPos rotary terms computed per forward from the non-persistent
  `inv_freq` / `zeta` buffers, so there is no learned position table and no
  sequence-length bound in the weights.
- The lane D scorer (`cascade/eval/lane_d_scorer.py` `_load_model`) builds the
  model from the entry's `"moe"` with `context_length` replaced by the
  context the entry is run at (its effective context, up to 131072), and feeds
  each window its own no-pad history up to that context.
- The only reader that would honour 4096 is the pinned `forecast_wrapper.py`
  edge-pad, which lane D scoring never uses.

Pinning the field keeps every entry's `config.json` identical to the
contract's (the `config_mismatch` rule) while the context stays a declared
scorer setting; quality beyond the 4096-point training rows is what T7 and the
arming gates measure.

- **Declarable from launch:** `contexts = [4096, 8192, 16384, 32768, 65536,
  131072]` (4k to 128k). You may declare any of them; a value outside
  `contexts` is refused at submit (`context_not_armed`) and closed at
  judging with the same reason.
- **Armed per settlement, automatically (no overlay):** a variant is armed
  when both gates pass: (1) at least `min_long_series` (300) series from at
  least 20 sources have enough history for it (the pinned
  `laned_readiness/1` document), and (2) its T7 wall time and memory fit the
  settlement budget on the eval host that scores lane D (the pinned
  `laned_t7/1` pod document). The validator computes the armed set at every
  settlement and records it, with the two documents' sha256s, in the signed
  report (`armed_contexts`, `armed_evidence`); satellites verify it there.
- **An unarmed declaration is still valid:** the entry is scored at the
  highest armed variant below its declared context, and at its own context
  from the settlement its variant arms. It is never run above its declared
  context: if no armed variant is at or below it, the entry is not scored
  that settlement. It stays `pending` with the skip reason
  `context_below_armed` and is judged once a variant at or below its
  context arms. The model trains on 4096-point rows, so quality beyond 16k is
  unproven until T7 measures it; in practice 32k–128k pass gate (2) only on
  the 96 GB eval host (stage S8).
- The lane D scorer never pads: each window's history is truncated to the
  largest patch multiple it really has, up to your declared context. A 4k
  model and the same bytes declared at 16k score identically on windows with
  4096 points of history or less. A long context gains only where long
  history exists.
- Lane D scores horizon `horizon = 64` only. There is one lane D board across
  all contexts: king and challenger are each scored at their own declared
  context on identical targets.

## Export, verify, submit, ready, commit

**Use a dedicated hotkey.** A hotkey holds one commitment at a time, so a
lane D commit replaces that hotkey's lane A pointer. Lane D never consumes a
lane A lifetime submission. One hotkey may hold one open entry
(`hotkey_has_open_entry`).

Lane D needs no compute keys: you train on your own hardware and the operator
pays for evaluation. Every command takes `--json`; no flag takes a key, token
or upload URL; the hotkey signs and the coldkey is never opened.

```bash
cascade lane-d export ./trained --out ./entry
cascade lane-d verify ./entry
cascade lane-d submit https://<intake> ./entry --context 4096 --wallet-name W --wallet-hotkey H
cascade lane-d ready https://<intake> --digest <digest> --context 4096 --licence MIT --attest --wallet-name W --wallet-hotkey H
cascade lane-d status https://<intake> --hotkey <ss58>
```

1. `cascade lane-d export` writes the entry directory (the four files only,
   the repo's code bytes, the weights byte-identical) and prints the entry
   digest: the sha256 of the canonical file manifest.
2. `cascade lane-d verify` runs the validators' ingest guards locally. The
   file allowlist runs first; nothing in the entry is executed.
3. `cascade lane-d submit` signs the manifest with your hotkey
   (`POST /v1/lane-d/submit`) and uploads every file straight to the private
   vault through the upload grants the intake returns (`presigned_multipart`,
   the path `decisions/STORAGE-LANE-D.md` chose). Nothing transits the
   operator's box and no grant is printed or logged. It commits nothing.
4. `cascade lane-d ready` attests the licence (`POST /v1/lane-d/ready`). The
   intake checks the object list and sizes against the manifest without
   downloading the weights, records your coldkey and `ready_block`, and
   returns the commitment payload `metro-v1:weights:c<context>:hippius:<vault-ref>`.
   The CLI then commits it on chain.
5. `cascade lane-d status` shows the public queue (`GET /v1/lane-d/queue`);
   it needs no wallet.

The commit reveals immediately (D43): `ready` commits with the chain's
one-block reveal floor, not lane A's boundary-timed reveal. The reveal block
is your entry's commit time T, and your weights are immutable from then on.
That is safe because the commitment shows only the vault ref and digest, the
weights stay private, exact-digest dedup refuses resubmissions
(`duplicate_weights` at the intake, `duplicate_digest` at judging), and only
the uploader's hotkey may commit its digest (`digest_owned`). A miner who
delays the reveal only delays their own forward-data clock.

You must commit within `[lane_d] ready_ttl_blocks` of `ready_block`, or the
entry expires as `ready_expired` and its slot is freed.

There is no Object Lock on the vault (Hippius offers none). Integrity comes
from content addressing: the object path carries the manifest digest
(`lane-d/<hotkey>/<digest>/`), every read re-hashes, and an overwrite fails
`digest_mismatch`, which is the miner's fault.

## Licence and takedown

At `ready` you attest, with `--licence MIT` and `--attest`, that the
checkpoint's weights, code and config, and the training data used to produce
them, are licensed for publication under the MIT licence (exactly MIT: a
modified MIT, such as one with a revenue restriction, is not), and that the
training data contains no outputs of any teacher model released under a
restricted licence. Without `--attest`
the CLI shows the statement and sends nothing. Distillation is allowed, from
Ephemeris or anything else, on your own account, as long as the teacher's
licence permits it.

Kings are published to the Hippius Hub, the R2 public bucket and the HF
dataset `tensorlink-dev/cascade-lane-d-kings`. A licence or rights claim
triggers a takedown: the Hub and HF copies are removed, while the vault, the
R2 copy, the digest and the verdict record stay, and the re-signed lane D
index marks the king `takedown`.

## How entries are judged

**Forward-only windows.** A window is eligible for entry E against king K
only if every target point has **both** a data timestamp **and** a first-seen
time (when our mirror first ingested it) after the wall time of
max(T_E, T_K) plus `[lane_d] forward_margin_hours` (0 h since rev6.14: data
after your submission is fresh; the embargo rules add older data; the 6 h
wait is the separate judging delay, `eval_delay_hours`). The feed's
measured mirror lag must also be within `max_mirror_lag_hours`. A series
without both timestamps is never eligible. Judging waits: nothing of an entry
is drawn, frozen or judged before max(T_E, T_K) plus `eval_delay_hours`
(6 h), measured against the boundary block's on-chain timestamp.

The forward-only rule sets the expected latency, and latency depends on data
frequency. Evidence accumulates over every daily `pool_d` snapshot since T
and freezes at the first settlement that meets the floors, `min_windows`
(1000) and `min_clusters` (20 sources). Five-minute and hourly feeds fill the
floors in days; daily and slower series rarely reach a forward-only set at
h = 64. The 48 h embargo top-up (`embargo_hours`) fills the live half
sooner: at least `min_fresh_window_share` (0.2) of the live windows are fresh,
and the rest may be embargo windows. Judging starts no earlier than
`eval_delay_hours` (6 h) after max(T_E, T_K), and the crown is decided at the
first verdict (FAQ).

**Live half and public half.** The score has a live half (the forward
windows above) and an optional public half: 5000 windows each from
`Salesforce/GiftEvalPretrain` and `nju-zhangsq/RMISC` per settlement, drawn by
the settlement block hash from the whole dataset (multivariate sets are scored
as `(C, L)` panels). Each window's score ratio is clamped to [0.5, 2]
(`window_ratio_floor` = 0.5) and log-averaged into L_live and L_public, and the combined score is
`L = (1 − public_weight) · L_live + public_weight · L_public`. A challenger
must also gain on each half that is scored.

- **At the mainnet launch** the overlay sets `public_weight = 0.0`. The
  verdict is the live half alone; no public windows are drawn or required.
- **Later**, once the public mirror exists, a signed overlay can raise it.
  The template value is `public_weight` = 0.5 (a 50/50 live/public score).
  `cascade status --json` shows the value in force.

**Verdict.** A two-part paired cluster bootstrap of the combined score L
(live clusters = source, public clusters = source and family) with lane A's
quantile head scoring. The challenger clears if the LCB on the 1 − exp(−L)
scale is `≥` the crown bar in force at the settlement (at `alpha = 0.05`),
and the point gain on every scored half is strictly positive. The bar decays with the
current KING_D's tenure (the lane D settlements it has held the crown, the
house included): `d_margin_start` (0.03) at tenure 0, falling linearly to
`d_margin` (0.01) over `d_margin_warmup_settlements` (32 settlements, about
4 days at 3 h each), flat after. A new crown restarts it. The signed report
records the tenure and the bar applied (`crown_bar`) and your gain per half, per
source (live, GiftEvalPretrain, RMISC) and per domain. Several entries freezing in one settlement
form a cohort judged with max-T on their shared window set. When a miner
holds KING_D, each member must also clear the current house on the same set.
An entry is
judged once; a loser is never re-judged on new windows.

**Crowned at the first verdict.** There is no confirmation stage. An entry
that clears the bar, the gate and (when a miner holds KING_D) the house is
crowned at that settlement's verdict. A clear held only by a bench outage
is `bench_hold` and is decided when the sweep runs.

## The house, the gate, the sentinel

**The house** is our own lane A king, always entered in lane D. It is
refreshed whenever lane A's king digest changes, and its T is the block at
which that digest was published. Lane D pays only for weights better than our
own pipeline produces: if the house holds KING_D, the king slot goes unpaid
(it burns, `house_king`). The house is never in the paid court.

**The gate** is a rotating public-benchmark direction check, run once per
crowning candidate. One suite is drawn from `gate_suites` (GIFT-Eval, BOOM,
TIME, FEV, minus the era's sentinel) by the settlement boundary's block hash,
so no miner can know it at commit. On the drawn suite, CRPS and MASE must each
satisfy `LCB ≥ -tau_public` (`tau_public = 0.01`) against both KING_D and the
house, run paired on one host in one sweep. A loss is `gate_regression`; a
sweep that errors, returns partial results or NaN is the miner's
`checkpoint_fault`; a bench host outage holds the crowning (`bench_hold`)
for at most `max_bench_holds` settlements. There is no separate
low-frequency check: the public half covers slow-cadence series once it is
on. Public
bench suites never pay and are never targets.

**The sentinel** is the cross-lane rotating held-out suite: each era one suite
is held out for every lane and never gates. It is swept for each new KING_D
and only published, so the board shows whether lane D generalises.

## Pay and the court

KING_D is paid from lane D's `share` of emissions. The court is KING_D plus
up to `reward_prior_kings` (2) prior **miner** kings, paid by the explicit
`court_weights` `[0.80, 0.15, 0.05]` inside lane D's share: 80% to KING_D,
15% to the newest prior king, 5% to the next. A court with fewer slots splits
over the slots it has (KING_D alone takes all of it; KING_D plus one prior
king take 0.80/0.95 and 0.15/0.95). Prior kings keep being paid with no time
limit. Unpaid share (a house king, an
empty or deregistered slot) burns; the remainder destination is release-only
(`remainder = "burn"`). The shipped default is `share = 0.0` and
`takes_from = "lane_b"`; the activation overlay sets the live split. Check
signed reports and on-chain weights for payment: queue state or a held
(`bench_hold`) clear is not a paid win.

Every settlement the main validator signs a `lane-d-report/1` report (see
[VALIDATOR.md](VALIDATOR.md#lane-d-what-satellites-verify)); satellites
re-derive the verdicts and weights from it. Crowned kings are listed in the
signed `champions/lane_d/index.json` with their context, digest, crown block,
LCB, gate and sentinel numbers and licence.

## Lane D FAQ

**How many entries can one coldkey have?** Lane D has its own caps (shipped
defaults; the deployed overlay can change them):

- `max_open_per_coldkey = 2`: open (ready, not yet judged) entries per
  coldkey. Over the cap, `ready` refuses with `coldkey_cap_open`; this intake
  check is advisory.
- `max_judged_per_coldkey_per_settlement = 2`: entries per coldkey judged in
  one settlement. The validator's queue selection is the authority: an entry
  over it is skipped with `coldkey_cap_settlement`, carried to the next
  settlement and not charged.
- `max_entries_per_settlement = 8`: entries judged per settlement across all
  coldkeys (the same skip reason applies).
- `max_coldkey_unresolved_settlements = 4`: settlements an entry may wait with
  an unresolvable coldkey (`coldkey_unresolved`) before it closes unjudged.
- `max_reveal_delay_days = 14`: an entry still pending after this many days
  closes unjudged (`reveal_delay_exceeded`) and frees its slots.
- `ready_ttl_blocks = 7200`: blocks between `ready` and your on-chain reveal
  before the entry expires (`ready_expired`).
- `max_mirror_lag_hours = 6.0`: the most our mirror may trail a feed's public
  source; windows from a slower feed are never eligible
  (`mirror_lag_exceeded`).

Coldkeys are free, so these caps limit casual flooding, not a determined
Sybil.

**Why was my entry skipped?** The signed report lists every skipped entry
with one typed reason, and the queue shows it:

- `pending_floor`: not yet enough eligible forward windows or clusters.
- `coldkey_cap_settlement`: over a per-settlement cap; carried forward.
- `coldkey_unresolved`: the coldkey recorded at `ready` could not be
  re-checked; alerted, closed after the cap above.
- `coldkey_changed`: the hotkey's coldkey at reveal differs from the one
  recorded at `ready`.
- `king_unscored`: KING_D (or the house, when it is a second bar) has no
  commit time or no forecasts this settlement; the entry waits, uncharged.
- `bench_hold`: the crowning sweep is waiting on the bench host.
- `deregistered`: the hotkey lost its UID; the entry is voided.
- `ready_expired`: not revealed within `ready_ttl_blocks`.
- `reveal_delay_exceeded`: pending longer than `max_reveal_delay_days`.
- `duplicate_digest`: the weights equal a published checkpoint, the house,
  KING_D or another open entry.
- `hotkey_has_open_entry`: the hotkey already holds an open entry.
- `context_not_armed`: the declared context is not in `contexts`.
- `context_below_armed`: the declared context is in `contexts`, but no armed
  variant is at or below it this settlement. The entry stays `pending` and is
  never run above its declaration.
- `before_eval_delay`: the settlement's boundary (its on-chain timestamp) is
  before the entry's judging start, max(T_E, T_K) plus `eval_delay_hours`
  (6 h). The entry stays `pending`; nothing of it is drawn, frozen or judged.
- `boundary_ts_missing`: the validator had no on-chain timestamp for the
  boundary, so nothing was frozen or judged this settlement (fail closed).
- `checkpoint_fault`: the checkpoint failed ingest, scoring or its gate sweep.
- `mirror_lag_exceeded`: the feeds behind the entry's windows lag too much.
- `lane_closed`: the kill switch `until_block` was reached: nothing is
  judged and every open entry closes.

**When will I be crowned?** At the first verdict that clears. There is no
confirmation duel and no second clear: the settlement that first freezes
enough eligible windows for your entry decides it, so the latency is the time
your data frequency needs to fill the floors, not a later settlement on top.
A bench outage can hold the decision (`bench_hold`) for up to
`max_bench_holds` settlements.

**What are the two clocks?** Each target point has a data time (the
timestamp the feed gives it) and a first-seen time (when our mirror first
ingested it). Both must be after max(T_E, T_K) plus the margin. Data time
alone would admit publish-ahead values (day-ahead prices, schedules,
forecast products); first-seen alone would admit values that were public
before T but reached our mirror late. The mirror lag is the per-feed delay
between the source's publish stamp and first-seen; a feed above
`max_mirror_lag_hours` is never eligible. The report carries both clocks and
the mirror lag per window.

**Can I submit F32 weights?** No. Lane D is BF16 only; an F32 file is refused
with `dtype_mismatch`. Convert with safetensors before `export`.

**Do local scores predict a lane D win?** No. The public eval pool is a
lagged reveal of old snapshots, so a model may have seen it. Local scores on
the revealed pool overstate forward gains; only forward-only windows count.

**Can I see a loser's rows?** Every verdict publishes `rows_sha256`, the
sha256 of the exact scored rows. Losers' weights stay private and are removed
from the vault after `loser_ttl_eras`, then held in an owner-only archive for
the dispute window before deletion.
