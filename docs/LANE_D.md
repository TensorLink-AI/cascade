# Lane D: open weights on the cascade architecture

Lane D lets you train the cascade model **anywhere, on anything, with any
recipe or compute** and submit **only the weights**. The operator only
evaluates. A lane D entry wins by beating the current lane D king (KING_D)
on **forward-only private windows**: forecast targets that first reached our
data after both models were committed, so neither side can have trained on
them. Design record: [`docs/plans/LANE_D_PLAN.md`](plans/LANE_D_PLAN.md)
(DEC-CA-0070). The short CLI reference is
[MINER.md §15](MINER.md#15-open-weights-lane-d-the-cascade-lane-d-cli).

Lane D ships **unarmed**. `[lane_d] from_block = 0` means the lane is never
active; it is armed later by a signed network overlay at an era boundary,
never by a `chain.toml` edit. Check `cascade status --json` (`lanes.d`) and
the public queue before you spend anything. A listed feature is not live
availability.

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
| `config.json` | The contract's model config; its `"moe"` object must equal the contract's (`config_mismatch` otherwise) |
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

- **Launch contexts:** `contexts = [4096, 8192, 16384]` (4k, 8k and 16k).
- **Later (S8):** 32768, 65536 and 131072 are armed only at rollout stage
  S8, by overlay through the same `contexts` key, after the T7 measurements
  on the S8 eval host show the longer context fits. The model trains on
  4096-point rows, so quality beyond 16k is unproven until then.
- A context that is not armed is refused at submit (`context_not_armed`) and
  skipped at judging with the same reason.
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
cascade lane-d ready https://<intake> --digest <digest> --context 4096 --licence Apache-2.0 --attest --wallet-name W --wallet-hotkey H
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

At `ready` you attest, with `--licence Apache-2.0` or `--licence OpenMDW-1.1`
and `--attest`, that the checkpoint's weights, code and config, and the
training data used to produce them, are licensed for publication under
Apache-2.0 or OpenMDW-1.1, and that the training data contains no outputs of
any teacher model released under a restricted licence. Without `--attest`
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
max(T_E, T_K) plus `[lane_d] forward_margin_hours` (6 h). The feed's
measured mirror lag must also be within `max_mirror_lag_hours`. A series
without both timestamps is never eligible.

The forward-only rule sets the expected latency, and latency depends on data
frequency. Evidence accumulates over every daily `pool_d` snapshot since T
and freezes at the first settlement that meets the floors, `min_windows`
(1000) and `min_clusters` (20 sources). Five-minute and hourly feeds fill the
floors in days; daily and slower series rarely reach a forward-only set at
h = 64, which is why the low-frequency check below uses the private pool.
Expect about one extra day for the forward margin and then the mandatory
confirmation duel (FAQ).

**Verdict.** A paired cluster bootstrap of the CRPS/MASE geomean ratio
(clusters = source) with lane A's quantile head scoring. The challenger
clears if `LCB ≥ d_margin` (`d_margin = 0.01`, at `alpha = 0.05`) and the
point gain is strictly positive. Several entries freezing in one settlement
form a cohort judged with max-T on their shared window set. When a miner
holds KING_D, each member must also clear the current house on the same set.
An entry is
judged once; a loser is never re-judged on new windows.

**Confirmation.** A candidate that clears must clear again on the next
disjoint eligible window set before it is crowned (`confirm_pending` in
between). If the king changes in between, `confirm_on_king_change =
"restart"` restarts the duel against the new king.

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
for at most `max_bench_holds` settlements. Every crowning candidate also runs
the low-frequency check on the private pool's daily-and-slower slice
(`lowfreq_freqs`): a loss there is `lowfreq_regression`. Public suites never
pay and are never targets.

**The sentinel** is the cross-lane rotating held-out suite: each era one suite
is held out for every lane and never gates. It is swept for each new KING_D
and only published, so the board shows whether lane D generalises.

## Pay and the court

KING_D is paid from lane D's `share` of emissions. The court is KING_D plus
up to `reward_prior_kings` (2) prior **miner** kings, decayed by
`king_decay` (0.5) inside lane D's share. Unpaid share (a house king, an
empty or deregistered slot) burns; the remainder destination is release-only
(`remainder = "burn"`). The shipped default is `share = 0.0` and
`takes_from = "lane_b"`; the activation overlay sets the live split. Check
signed reports and on-chain weights for payment: queue state, a first clear
or `confirm_pending` are not paid wins.

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
- `checkpoint_fault`: the checkpoint failed ingest, scoring or its gate sweep.
- `mirror_lag_exceeded`: the feeds behind the entry's windows lag too much.
- `lane_closed`: the kill switch `until_block` was reached: nothing is
  judged and every open entry closes.

**When will I be crowned?** Not at the first clear. The confirmation duel
means a candidate that clears waits for the next disjoint eligible window set
(the next freeze of new forward windows) and must clear again: about 1–4
days more depending on data frequency, at the earliest a later settlement.
Until then its status is `confirm_pending`. A king change in between restarts
the duel under the default rule.

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
