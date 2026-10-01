---
id: DEC-CA-0049
type: decision
title: "Dethrone bar v2: increment-unit margin 0.4 for a fresh king, decaying to 0.2 over 18000 blocks of tenure — decided on chain by validator readiness (DEC-CA-0045), typed block = owner override"
status: proposed
date: 2026-10-01
tags: [scoring, validator, consensus, margin, activation]
revisit_when: "the first ~10 settlements after the bar applies — record dethrones, LCBs and raw win-rates, and whether 0.2 at the floor still admits raw-tied winners (the increment statistic is noisy; a raw-win-rate co-gate may be the better complement)"
relations: {depends_on: [DEC-CA-0016, DEC-CA-0039, DEC-CA-0045], relates_to: [DEC-CA-0040, DEC-CA-0048]}
---

## Decision

Owner request 2026-10-01: "increase the LCB threshold to 0.4 and have it decay
over 5 rounds to 0.2 — the scaling has changed, the 0.01 threshold is too easy
now", and "it should be a validator auto update as they update their code".

* **Bar.** `[scoring] win_margin_start_v2 = 0.4`, `win_margin_end_v2 = 0.2`,
  `margin_warmup_blocks_v2 = 18000` (5 old 12 h rounds ≈ 2.5 days). The bar
  decays linearly with the king's tenure, counted per settlement on the grid
  in force (18000 / 900 = 20 settlements), exactly like `margin_warmup_blocks`.
  The existing `win_margin_*` keys keep judging every round before the gate,
  so archived receipts replay unchanged.
* **Activation.** `margin_v2_from_block = 0` = decided on chain. A validator
  on a release carrying the bar adds one segment to its single readiness note
  — `margin-v2-<sha256("start|end|blocks")[:8]>` — beside the rollover and
  forfeiture segments. At the first boundary where 51% of eligible validator
  stake has signalled it LOCKS IN (one-way, persisted in
  `activation_margin_v2_state.json`) and the bar applies from the first ERA
  START after the lock-in boundary (owner 2026-10-01: "next clean fresh era";
  `FeatureSpec.align = "era"`). Every settlement of one era is judged under
  one bar — never mid-era — and a lock-in exactly on an era start takes the
  FOLLOWING era start, so the era in which the count crossed finishes on the
  old bar. Worked example (900-block grid, 4 settlements per era = 3600-block
  eras): lock-in at settlement 9187200 + 900 = 9188100 (mid era 2552) ⇒ bar
  from 9190800 (era 2553 start); lock-in at 9187200 (era 2552's own start) ⇒
  also 9190800. Notes naming any other `(lock, act)` pair for this feature are
  inadmissible. The rollover and forfeiture keep their next-boundary rule.
  Changing any of the three values renames the feature (a fresh vote). A typed
  `margin_v2_from_block` is the owner override and the hold-back.
* **Receipts + audit.** Validators stamp `margin_v2_block` (drop-when-default,
  archived signatures survive) from lock-in on; `cascade-audit` replays the
  verdict under it (`apply_receipt_activation`) and checks the block against
  the validators' notes (`margin-v2` check).
* **Guardrails (load-checked).** All three values set or none; floor > 0 (a
  zero floor turns the decay into a term-limit lottery); start >= end; a
  typed block needs the bar.
* **Level fallback keeps the level bar.** The v2 values are priced in
  INCREMENT units. A round the validator must judge in LEVEL units (no init
  baseline: a random-init round or a multi-size duel) is judged at the pre-v2
  level schedule (`ChainConfig.judged_level_params`), never at a 0.4 LEVEL bar
  (40 % of the absolute score = undethroneable). The audit replay applies the
  same rule; receipts still record the unmodified config params.

## Why

Since the increment-unit LCB (DEC-CA-0039) judges every round, the bar is a
fraction of the king's per-round gain over the shared init, not a % of its
absolute score. Recent increment LCBs span about −0.84 … +0.47, so the
inherited 1% → 0.5% bar is effectively "LCB > 0".

Backtest of this week's dethrones (bar at the dethroned king's tenure,
≈2.7 settlements on the old grid ⇒ ≈0.29 under v2):

| Round | Winner | LCB | Bar then | Bar under v2 | Outcome under v2 |
|---|---|---|---|---|---|
| 2026-09-25 19:05 | u105 | 0.2166 | 0.0083 | ≈0.29 | held |
| 2026-09-29 22:05 | u86 | 0.2462 | 0.0083 | ≈0.29 | held |
| 2026-09-30 16:05 | u135 | 0.4667 | 0.0095 | 0.40 | dethrone |

Caveat: u135 cleared 0.4667 at a raw win-rate of 0.517 (Wilcoxon p 0.124) —
the increment statistic divides by a small denominator and is noisy, so even a
0.4 bar can be cleared by a raw-tied challenger. Watch the raw numbers in the
first settlements (revisit condition).

## Blast radius

CONSENSUS (netuid 91 has six external validators). It cannot fork a mixed
fleet: nothing changes until 51% of eligible stake signals, the flip lands at
a future boundary every upgraded node derives identically, and a node that
joins late adopts the block from the notes. A validator that never upgrades
keeps judging at the old bar after the flip and diverges — the same
release-then-activate exposure as every DEC-CA-0045 feature. The trainer and
provisioner do not use the margin and need no change.

## Not mirrored from DEC-CA-0048

* The trainer does not resolve this feature (the forfeiture changes who
  trains the king leg; the margin changes nothing trainer-side).
* `cascade/miner/dashboard.py`'s informational margin line still shows the
  pre-rollover round-counted schedule (it already ignored
  `margin_warmup_blocks`); it does not show the v2 bar.
