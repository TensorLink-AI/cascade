---
id: DEC-CA-0056
type: decision
title: "Lane C: miner-funded fine-tune recipes on the king checkpoint, per-target, judged on a separate per-target eval, paid in per-target slots; emissions A 50 / B 25 / C 25 once armed (50 / 50 / 0 until then)"
status: proposed
date: 2026-10-19
tags: [consensus, emissions, lanes, fine-tuning, v2]
revisit_when: "a second competition gets a forward-only eval (register its kind in COMPETITION_KINDS); or MarketCast (the only competition) cannot supply enough fresh windows for a suite; or lane C's slots stay mostly unpaid for weeks (fewer targets or a lower share)"
relations: {depends_on: [DEC-CA-0050, DEC-CA-0055, DEC-CA-0036], relates_to: [DEC-CA-0019, DEC-CA-0054]}
---

## Decision

Owner (2026-10-19):

1. **What miners compete on.** A fine-tuning recipe (data generator +
   schedule) applied to the CURRENT king checkpoint, declared for one target
   (a domain, frequency or horizon band).
2. **Separate eval.** A fine-tune is judged only on its declared target, on an
   eval separate from lane A's private pool: a private per-target pool
   (refreshed daily like the main pool) decides, paired against the king on
   the same windows; the public GIFT-Eval/BOOM/TIME subsets for that target
   may be added as a guard (not significantly worse), configurable per target.
3. **Funding.** The miner's deposited key pays for the fine-tune leg, the
   same path as challenger legs (DEC-CA-0036); infra faults requeue.
4. **Payout.** Per-target slots like lane B: the best fine-tune on each target
   holds that slot (with a TTL), each slot pays `share / n_targets`; unpaid
   slots burn. A slot moves only to a confirmed qualifier with a strictly
   higher gain LCB on the same king, or to any confirmed qualifier on a newer
   king (see **Scoring hardening**).
5. **Separate product.** A winning fine-tune is published and paid, but never
   becomes the pretraining init; lane A's lineage is untouched.
6. **Emission split.** A 50 / B 25 / C 25 once lane C is armed; until then
   A 50 / B 50 / C 0 (today's DEC-CA-0050 split, unchanged).
7. **Activation.** Ships inert; goes live at a typed `[lane_c] from_block`
   when the owner says (consensus: every validator on the release first).

## Resolved (defaults, owner-tunable)

The former open items, resolved with these defaults (all `[lane_c]` keys,
owner-tunable; the consensus ones change lockstep like `[lane_b]`):

- **Targets** (one emission slot each): the seven lane-B domains
  (`domain:econ_fin`, `domain:energy`, `domain:healthcare`, `domain:nature`,
  `domain:sales`, `domain:transport`, `domain:web_cloudops`) plus
  `horizon:720` and `freq:sub_hourly`. A target spec is `domain:<label>` |
  `freq:<sub_hourly|hourly|daily>` | `horizon:<h>` (h a scored horizon ∪
  `[eval] horizon`, checked once armed).
- **Pool**: one private lane-C pool in the pool bucket under `pool_prefix =
  "pool_c"` — `pool_c/index.json` + `pool_c/snapshots/block-<N>.tar`, the
  lane-A layout and helpers (`prefix` argument). Validators select the
  snapshot by the SETTLEMENT block. A target selects windows: `domain:<d>` by
  metadata `domain`, `freq:<f>` by the frequency class of metadata `freq`
  (seconds/minutes → sub_hourly, H → hourly, D/B → daily), `horizon:<h>`
  every series cut at that horizon. `n_windows = 1000` per target per
  settlement, even-by-domain, seeded `lane-c|<round_id>|<target>`.
- **Base checkpoint**: the ERA KING's trained pointer (the king entry of the
  lane-A settlement that binds the lane-C manifest) — not the promoted init.
  A leg fine-tuned from an older king is stale at a settlement whose king
  differs (the trainer drops / requeues it; `max_base_age_eras = 2` bounds how
  long it waits).
- **Leg budget** (trainer policy): `leg_hours = 1.0`, `legs_per_target = 2`,
  recipe `lr_scale ≤ max_lr_scale = 1.0`, `steps_fraction ≤
  max_steps_fraction = 1.0`.
- **Win rule**: `LCB_α((king − ft)/king) ≥ gain_margin` with `alpha = 0.05`,
  `bootstrap_B = 10000`, `gain_margin = 0.01` (NOT 0: a null fine-tune must
  not win ~α of the time), `min_target_windows = 100`; the same paired
  cluster bootstrap as lane B (`judge_expert_leg` with every window labelled
  with the target, so no domain guards). Optional public guard per target
  (`guard_public = {target = [suites]}`, default none): non-inferiority `LCB
  ≥ −tau_public` (0.01) per suite metric on trainer-signed per-item rows.
  The verdict also needs a strictly positive point gain and at least
  `min_target_clusters = 5` distinct clusters (pool `source`) among the
  target windows. See **Scoring hardening** for the correction, the
  confirmation step and the slot rule.
- **Slot payout**: lane B's schedule in ERAS — `full_rounds = 5` (now
  `full_pay_tiers`, see "Gain-tiered full-pay period 2026-10-20" below),
  `half_life_rounds = 1.0`, `floor = 0.1`, `win_ttl_eras = 28`; `share /
  n_targets × multiplier` per paid slot, `remainder = "burn"`.
- **Split**: `share = 0.25`, `takes_from = "lane_b"`: A = 1 − lane_b.share,
  B = lane_b.share − share, C = share ⇒ A 50 / B 25 / C 25. Lane A's pool
  (and every DEC-CA-0055 rule on it: ladder, cap, slash, pre-launch burn) is
  unchanged.
- **Activation**: `[lane_c] from_block` is a typed ERA-BOUNDARY block at or
  after `[v2] from_block` (load-checked; `share ≤ [lane_b] share`). The era
  starting at it is the first lane-C era; the settlement AT it is still
  pre-lane-C. 0 on mainnet and testnet.

## Scoring hardening (2026-10-19, owner)

Four rules close repeated-attempt and decay-reset gaps. All are lane-C
consensus terms, live only past `[lane_c] from_block` (nothing changes
before it, and lane A is untouched).

- **Bonferroni per target.** At a settlement, target `t`'s `k` judged
  entries (its entries in the signed lane-C manifest plus any deferred
  carries, below) are each tested at `alpha / k`. An auditor derives `k` by
  counting the receipt's judge-phase rows on `t`; each row records `k` and
  `alpha`.
- **Confirmation.** A target's best passer (highest gain LCB, ties to the
  earliest `train_block`, then hotkey) is not a win yet: it becomes PENDING
  in `LaneCState.pending`. At the next lane-C settlement that has a king it
  is re-scored against THAT settlement's king on its fresh windows, at
  uncorrected `alpha` (`phase = "confirm"` rows). Pass ⇒ the win is recorded
  at that settlement's block (its `gain_lcb` is the confirmation LCB). Fail,
  or the king changed (`stage = "stale"`) ⇒ dropped (`outcome = "dropped"`).
  A settlement with no king leaves the pending set untouched. The
  confirmation is against the king only. Public-guard rows are re-checked
  with the stored entry.
- **Cluster floor.** At least `min_target_clusters` (default 5) distinct
  clusters must appear among the target windows, else the verdict is
  `inconclusive` and never passes: 100 windows from one source give one
  observation to the cluster bootstrap. An entry that is inconclusive for
  lack of clusters is DEFERRED once: it joins the next settlement with the
  same king as a candidate, counts in that settlement's `k`, and is dropped if
  it is inconclusive again or the king changed. `cascade-pool publish --lane
  c` warns when a target has fewer distinct sources than the floor.
- **Slot rule** (`cascade.shared.lane_c.win_beats`). A confirmed qualifier
  takes a target's slot:
  * from a holder trained on the SAME king pointer only when its recorded
    `gain_lcb` is STRICTLY greater than the holder's. Otherwise the outcome
    is `below holder LCB`: the slot and its `win_block` (the payout clock)
    stay unchanged. The holder's own hotkey gets no special case. A re-run
    of the same recipe that matches but does not beat the holder's LCB never
    restarts the clock;
  * from a holder trained on an OLDER king, always: that holder no longer
    defends.
  The rule compares LCBs recorded at different settlements, on different
  windows. It is a conservative point comparison, not a paired test; the
  holder is never re-scored. It is a pure function of `(base_pointer,
  gain_lcb, win_block, round_id)` (kings only move forward, so between
  different bases the later win holds; on an exact LCB tie the earlier win
  holds), so an out-of-order catch-up folds to the same holder. The payout
  duration is lane B's (`full_rounds = 5`, halving to `floor = 0.1`,
  `win_ttl_eras = 28`).

The pending set rides the `pre_round` snapshot / rollback and the catch-up
walk exactly like the slots. A prepare judges each catch-up round against a
scratch copy folded in chain order. Receipt rows carry `phase`, `k`, `alpha`,
`clusters`, `outcome`, `base_pointer`, plus `origin_round` (carried /
confirm rows) and the recorded `holder` (`hotkey, gain_lcb, base_pointer,
win_block, round_id`) on passing confirm rows. `check_lane_c` re-derives the
level, the pending choice, the deferral and the slot split from those
values, and `check_lane_c_chain(receipts, cfg)` replays pending continuity
across consecutive receipts.

## Open

- **GPU time lost to a moved king (review L7, partly mitigated).** A
  non-MarketCast leg settles on the king of the settlement that carries it;
  if that is a later era's king the leg is dropped (`base_moved` / `stale`),
  requeued uncharged, but the miner's GPU time is gone. The trainer now
  dispatches a non-mkt leg only while `2 × leg_hours` still fits before the
  era ends (`LaneCDriver._fits_era`, `DISPATCH_MARGIN = 2`); later it waits
  for the next era. It does NOT cover a leg that finishes in time but whose
  settlement waits for a lane-A manifest past the era end (next bullet).
  MarketCast legs are unaffected: their base never moves, so they are never
  dropped `stale` / `base_moved` (`settle` / `_expire_stale` skip them).

- **Lane-C legs ride lane-A settlements.** A lane-C manifest is published
  only beside a lane-A settlement manifest (bound by its `lane_c_digest`),
  exactly like lane B. On a boundary that publishes no lane-A settlement (no
  lane-A challenger), finished lane-C legs wait for a later settlement; they
  are judged against THAT settlement's king, so a king change in between
  makes them stale. Consensus is deliberately unchanged; revisit if legs
  routinely wait or go stale (e.g. publish a king-only settlement when lane-C
  legs are waiting).
- **Manifest provenance (not judged).** The trainer stamps
  `LaneCManifest.base_round_id` = the binding lane-A round id, `pool_block` =
  the settlement boundary, and `LaneCEntry.funded_by = "miner"`. Validators
  accept any value for these three; only `base_pointer` is checked.

## Recipe format

The miner commits `metro-v1:ft:<target>:hippius:<repo>@<digest>` (one
commitment per hotkey). The ref is a generator tree (the challenger
generator format: the data the fine-tune trains on) that ALSO carries
`recipe.json` at its root:

```json
{"target": "domain:energy", "lr_scale": 0.5, "steps_fraction": 0.8,
 "warmup_fraction": 0.1, "schedule": "cosine"}
```

`target` must equal the committed target and be a `[lane_c]` target;
`lr_scale ∈ (0, max_lr_scale]` multiplies the leg's base LR;
`steps_fraction ∈ (0, max_steps_fraction]` is the fraction of the leg's step
budget run; `warmup_fraction ∈ [0, 0.2]` of those steps warm up linearly;
`schedule ∈ {cosine, constant, linear}` after warmup. Unknown keys are
refused. One rule for both sides: `cascade.shared.lane_c.parse_recipe(obj,
cfg, *, target=None) -> Recipe` (raises `ValueError`).

## Interfaces

Shared (validator / audit / shared half — shipped):

- **Config** `cascade.shared.config.lane_c`: `LaneCConfig` (`ChainConfig.lane_c`),
  `parse_lane_c(lc, *, v2_from_block, era_blocks, era_settlements, horizons,
  lane_b_share)`, `parse_target(spec) -> (kind, value)`, `check_target(spec,
  *, horizons=None) -> str`. `LaneCConfig.active(boundary)` (settlement:
  `boundary > from_block`), `.era_active(era_start)` (era: `era_start >=
  from_block`), `.guard_suites(target)`, `.verdict_params()`.
- **Manifest** `cascade.shared.lane_c_manifest`: `lane_c_manifest_key(round_id)
  = "lane_c/round-<id>.json"`; `LaneCEntry{miner_hotkey, miner_uid, target,
  recipe_ref, trained_pointer, train_block, funded_by="", public_rows={}}`
  (`public_rows = {"base": rows, "candidate": rows}`, the
  `pair_public_rows` format, needed only for a guarded target);
  `LaneCManifest{round_id, era_index, created_block, base_pointer,
  base_round_id, pool_block, entries, version=1, signature}`;
  `dump_lane_c_manifest` / `load_lane_c_manifest` / `sign_lane_c_manifest` /
  `verify_lane_c_manifest`; `lane_c_text_digest(text)`;
  `parse_finetune_commit(payload) -> (target, ref) | None`,
  `format_finetune_commit(target, ref)`, `finetune_commit_as_of(history,
  hotkey, block)`.
- **Binding**: `TrainingManifest.lane_c_digest` (signed, drop-when-empty) =
  `lane_c_text_digest(<the exact published lane-C manifest text>)`. The
  trainer publishes `lane_c/round-<id>.json` BEFORE the lane-A manifest of
  that settlement; empty ⇒ no lane-C legs this round.
- **State / payouts** `cascade.shared.lane_c`: `LaneCState` (`slots`,
  `history`, `applied`, `settled_block`, `last_round`, `pending`,
  `last_rows` — the folded round's receipt rows, returned verbatim on a
  crash-replay of that round),
  `TargetWin`, `PendingEntry`, `win_beats`,
  `lane_c_payouts`, `lane_c_pay_record`, `lane_c_shares(lane_b_cfg,
  lane_c_cfg, block)`, `lane_c_vector`, `target_selects`, `freq_class`,
  `target_horizon`, `parse_recipe`. Emission: `cascade.shared.emission.
  lane_parts(..., lane_c_cfg=None, c_wins=())`.
- **Pool**: `cascade.shared.hippius.pool_index_key(prefix)`,
  `pool_snapshot_key(block, prefix)`, `read_pool_index(store, prefix)`,
  `publish_pool_snapshot(..., prefix=)` (default `"pool"` = lane A,
  byte-identical); `cascade.validator.pool.load_bucket_pool(cfg, prefix=)`.
- **Windows**: `cascade.validator.lane_c.lane_c_windows(cfg, window_source,
  target, round_seed, *, block)` — the one draw rule (validator, audit,
  trainer self-judge).
- **Receipts**: `lane_c` (verdict rows: `hotkey, uid, target, stage, passed,
  reason, gain_lcb, gain, n, public_lcbs, trained_pointer, train_block,
  phase, k, alpha, clusters, outcome, base_pointer[, origin_round, holder]`) and
  `lane_c_pay` (`{block, era_blocks, share, slots, wins: [{target, hotkey,
  uid, win_block, age, multiplier}]}`), both drop-when-default; once lane C
  is active `lane_b_pay` also carries lane B's reduced `share` and `slots`.

Trainer half (implements against the above): intake of fine-tune commitments
and recipes (`parse_recipe`), the miner-funded fine-tune legs on the era
king's checkpoint, the lane-C pool publisher (`cascade-pool` with
`prefix = [lane_c] pool_prefix`), the signed lane-C manifest published
before the lane-A manifest with `lane_c_digest` set, and publishing winning
fine-tunes (never as the pretraining init).

## Shipped (inert)

Validator: `cascade.validator.lane_c.LaneCMixin` (on `ValidatorRunner`). At
every settlement where lane C is active it reads the bound manifest, checks
the digest and the pinned trainer signature, judges each entry (stages
`target` → `stale` → `binding` → `fault` → `verdict`), folds the winners into
`<state_db>.lane_c.json` before the handled marker persists, carries the
state in the `pre_round` snapshot (a re-judgement replaces the fold), walks
back over unfolded settlements (catch-up, also at startup) and pays the
3-lane vector. A READ failure (manifest, chain history, checkpoint fetch, the
king's scoring) HOLDS the settlement (`_hold_read_failure`, gate
`lane_c_unverifiable`); a deterministic eval fault loses that entry. Audit:
`check_lane_c` (verdict consistency) and the 3-lane rebuild inside
`check_weights` (`_lane_c_paid_wins`, `_rebuild_lane_vector`). Status:
`status/v2.json` gains `lane_c` once armed; the receipt index gains
`lane_c_verdicts` / `lane_c_pay` and a `lane_c` emission fraction. Before
`from_block` every vector, receipt, manifest and status document is
byte-identical.

## Trainer shipped

Trainer + intake + pool + worker + miner CLI half (inert until `[lane_c]
from_block` is typed; every pre-lane-C path is byte-identical).

- **Intake** (`cascade/funding/intake.py`): `POST /v1/lane-c/fund` — the
  `/v1/fund` identity gate with the signature over
  `canonical_fund_message("lane-c-fund", hotkey, "<target>|<ref>", ts,
  sha256(key))` (binds key digest, target AND ref), a registered hotkey, a
  configured target, and a REVEALED `metro-v1:ft:<target>:hippius:<ref>`
  commitment naming exactly them (`ChainRevealResolver.lane_c`). Queues into a
  SEPARATE lane-C queue (`LaneCQueue`, `lane_c_queue.json` beside the
  challenger queue — derived, never configured, so intake and trainer cannot
  split-brain); the key lands in the shared payer vault (one key per hotkey).
  Also `POST /v1/lane-c/withdraw`, `GET /v1/lane-c/queue`. `cascade-intake
  --lane-c auto|on|off` (auto = on once `from_block` is typed). Lane C never
  consumes the lane-A lifetime submission.
- **Queue** (`cascade/funding/queue.py`): `FundedQueue` semantics verbatim
  (one live entry per hotkey, infra requeue with the bounded attempt budget,
  operator-fault no-burn backoff, TTL expiry, re-commit parked behind a
  flight); the entry's ref field is `"<target>|<ref>"`.
- **Driver** (`cascade/trainer/lane_c_driver.py`, mirrors `LaneBDriver`):
  per tick of a lane-C era (`era_active(era start)`), requeue orphans (open
  legs of a dead process; in-flight queue entries no leg runs), expire open
  legs whose base is > `max_base_age_eras` behind (requeued uncharged), then
  dispatch ≤ `legs_per_target` per target, earliest reveal first, only while
  the hotkey's LATEST commitment is still the funded `metro-v1:ft:` one
  (otherwise the entry ends, class `superseded`) and no lane-A leg of the
  hotkey runs (one leg per hotkey across lanes: pod names, harvest dir and key
  are per hotkey; lane-A intake likewise holds a hotkey with a live lane-C
  leg, and the lane-A pod sweep keeps live lane-C payer pods). A leg fetches
  the ref, validates `recipe.json` with the shared `parse_recipe` against the
  committed target (invalid ⇒ miner fault: the lane-C entry ends, class
  `recipe`), then runs the funded challenger path on the MINER's key
  (`_run_funded_leg(role="finetune", bench=False)`: rent → identity/runtime
  pins → corpus incl. the DEC-CA-0053 shards tier → worker → credential-free
  harvest → ingest verify → teardown; never an operator lane: a capacity
  fallback requeues as `no_capacity`). The base is the CURRENT era king's
  trained pointer. For targets with a public guard the leg carries
  `public_rows {base, candidate}` (base rows cached per king).
- **Fine-tune mode** (`cascade/trainer/finetune.py`, worker `--role finetune
  --recipe-json`): the era contract at `leg_hours` wall (`for_hours`), token
  budget × `steps_fraction`, `base_lr × lr_scale`, recipe warmup, schedule
  cosine→`warmup_cosine` / constant→`warmup_flat` / linear→`warmup_linear`
  (new, fine-tune only); fork-anneal, EMA, power EMA, re-warmup and warm LR
  scale off (the exported weights ARE the fine-tune). The base is staged
  weights-only (its scored `weights.safetensors` + loader files, no optimizer
  / EMA / stable file) ⇒ strict load (no arch change possible) and a FRESH
  optimizer. Repo / pod dir suffix `-c<target>-u<uid>`.
- **Settlement** (`rolling._lane_c_settle`, synchronous: one sign + one put):
  after lane B, BEFORE the lane-A manifest is built, the legs finished on the
  settlement's king go into a signed `LaneCManifest` (`base_pointer` = the
  king pointer of the binding lane-A settlement, `base_round_id` = that
  round id, `pool_block` = the boundary, entries sorted by train block,
  `funded_by = "miner"`) at `lane_c_manifest_key`; the lane-A manifest binds
  `lane_c_text_digest` of the exact text (`lane_c_digest`, passed only when
  set). Write-ahead: the signed text persists before the first byte goes out;
  a crash republishes the SAME bytes for that round, and the lane-A
  `pending_publish` carries `{key, text}` so a lane-A replay re-puts the
  identical lane-C bytes before the manifest that binds them. A failure binds
  nothing (the legs wait); a write-ahead of a round whose lane-A manifest went
  out without it is discarded. Legs fine-tuned from another base or past
  `max_base_age_eras` are dropped and requeued uncharged. Queue entries are
  marked done at publish.
- **Pool** (`cascade-pool publish|reveal --lane c [--prefix P]`): the same
  builder under `[lane_c] pool_prefix`, balanced selection seeded
  `lane-c:block:<N>`, per-target coverage report (warns under
  `min_target_windows`); retired snapshots revealed under `<prefix>/` on the
  HF dataset; gated steps added to `.github/workflows/publish-pool.yml`.
- **Miner CLI** (`cascade lane-c check|commit|fund|status`): local recipe
  validation with the trainer's rule before any commit or spend.
- **Status**: `status/lane_c.json` (queue per status, per-target queued /
  open / finished vs `legs_per_target`, recent settlements and outcomes).
- **Known limit**: like lane B, a lane-C manifest rides a lane-A settlement —
  a boundary with no lane-A challenger publishes no manifest, so finished
  lane-C legs wait for the next boundary that has one (and are dropped +
  requeued if the king moves first).

## Review fixes 2026-10-19

- **One leg per commitment (L5).** The trainer records every published
  leg's commitment (`hotkey|target|ref|commit_block`, driver state `spent`)
  and refuses a re-fund of the SAME commitment with error class `spent` —
  the lane-B rule; before, a re-fund kept its old reveal block at the queue
  front and two hotkeys could starve a target (`legs_per_target = 2`). A
  re-commit (a new reveal block) is a new commitment. An operator-fault
  requeue (`OPERATOR_FAULT_CLASS`) still retrains, uncharged.
- **Exact-identity dedup (L6).** The trainer stamps `LaneCEntry.tree_digest`
  — sha256 over the fetched recipe tree's files (relative path + sha256 of
  the bytes, sorted; dot / `__pycache__` entries skipped), so byte-identical
  trees match whatever repo they were uploaded to. Dropped from the signed
  body when empty. Validators keep a registry (`LaneCState.seen`): a
  candidate whose tree an EARLIER commit holds (seniority = the commitment
  block as of `train_block`, then `train_block`, then hotkey — never a UID)
  is a `duplicate`; when the earlier commit arrives after its copy, the
  copy's slot is evicted (an `{"error": "lane_c_duplicate_evicted"}` receipt
  row) before the round's outcomes are labelled. "Newer base wins" in
  `win_beats` is kept (an old fine-tune is stale against the new king), and
  the holder's OWN re-run on the new king is never its own duplicate — so a
  copied recipe re-run on each new king can no longer take the slot. No
  further slot rule was added. A tree with different bytes (a one-byte edit)
  is a different submission: exact identity only, as DEC-CA-0008.
- **Historical L8, amended 2026-10-26.** The original rule: `lane_parts` and
  `lane_c_shares` both raise `ValueError` for an active lane C without lane
  B, and the loader refuses a `[lane_c] from_block` before lane B is live
  (already implied by the v2 gate; now explicit). The audit reports a
  lane-C payout record without a lane-B one instead of rebuilding a vector.
  The owner-approved rollout is now A → C → B. This restriction remains only
  for legacy C taken from B. C with `takes_from = "lane_a"` may launch before
  B or with B disabled: A/C reserve 75%/25%, then A/B/C reserve 50%/25%/25%.
  Validator weights, receipt replay, audit and discovery use the same split.
- **Slots = ACTIVE targets (L9).** `active_targets(lc, block, mc)`: an
  `mkt:` target counts only once `[marketcast]` is active at the settlement.
  The payout record's `slots`, the vector (`lane_c_vector(slots=)`,
  `lane_parts(c_slots=)`) and the audit (`_lane_c_paid_wins`) all use it, so
  configured-but-inactive MarketCast targets no longer dilute every slot
  (0.25/3 → 0.25/2 in the test). The status dashboard's `slots` still counts
  configured targets (display only).


## Gain-tiered full-pay period 2026-10-20 (owner)

Same rule as DEC-CA-0050's "Gain-tiered full-pay period 2026-10-20" section,
for lane C. CONSENSUS; replaces `[lane_c] full_rounds` in place (refused at
load if still typed); `half_life_rounds`, `floor`, `win_ttl_eras` and
`win_beats` are unchanged.

* **Key.** `[lane_c] full_pay_tiers`, default `[[0.05, 10], [0.02, 5],
  [0.01, 2]]` (UNCALIBRATED placeholders — set from real win LCBs before
  arming), same load checks as lane B's, overlay-settable (every `[lane_c]`
  key is).
* **Which LCB.** The CONFIRMED win's gain LCB — the confirm-phase LCB (at
  `alpha`, on the next settlement's fresh windows) that `TargetWin.gain_lcb`
  records and `win_beats` compares. It is tiered at judge time
  (`LaneCJudgement.full_pay`, `cascade.shared.lane_c.target_full_pay`) and
  `_confirmed_win` records `tier` / `full_eras` / `below_floor` on the
  `TargetWin`; never re-evaluated. A later takeover is still decided by
  `win_beats` alone and restarts the clock with the new win's own tier.
* **MarketCast.** An `mkt:<suite>` win is tiered by `[marketcast]
  full_pay_tiers` (DEC-CA-0064 note), on lane C's TTL and decay.
* **Receipts / audit.** A passing confirm row stamps `full_pay` (`tier`,
  `full_eras`, `below_floor`) and `check_lane_c` re-derives it from the row's
  `gain_lcb`; the slot paid at this settlement must carry the confirmed row's
  `gain_lcb` and `full_pay`. `lane_c_pay` wins carry `gain_lcb`, `tier`,
  `full_eras`, `below_floor`; `_lane_c_paid_wins` checks them exactly as lane
  B's (at-win re-derivation, older wins against `state_post`) and replays the
  multiplier from the recorded `full_eras`.

## MarketCast-only lane C 2026-10-20 (owner)

Owner (2026-10-20): "we will have comps on lane C but currently the only eval
is MarketCast and it updates daily."

* **Lane C is a container for competitions.** A target is
  `<competition>:<value>`; its kind names the competition that judges it. A
  competition is admitted only with a **forward-only eval**: windows that
  close after the entry is trained, so nothing a miner can see before
  submitting is ever scored. The registry is
  `cascade.shared.config.lane_c.COMPETITION_KINDS`
  (`register_competition_kind(kind, eval=..., forward_only=...)`; a kind that
  does not declare `forward_only` as a bool is refused). A future competition
  is added by registering its kind and eval.
* **Today the only competition is MarketCast** (DEC-CA-0064, `mkt:<suite>`,
  forward-only, re-snapshotted daily), registered `forward_only=True`. The
  shipped `[lane_c] targets` are exactly the `[marketcast] suites`
  (`["mkt:returns", "mkt:vol"]`, mainnet and testnet).
* **Generic targets removed.** `domain:` / `freq:` / `horizon:` are
  registered `forward_only=False` and refused everywhere a target is parsed:
  the loader (`[lane_c] targets`, `guard_public`), the commitment parser
  (`metro-v1:ft:` accepts only `mkt:`), `LaneCEntry` (so a signed lane-C
  manifest carrying one is `lane_c_manifest_malformed`), intake and the miner
  CLI. The message is "lane-C competitions need a forward-only eval — only
  MarketCast (mkt:) has one; see DEC-CA-0056". **Why:** a red-team showed
  that their verdict windows — the last horizon of each public series in a
  daily `pool_c` snapshot, revealed ~48 h later — can be scraped from the
  same public feeds and memorised, so a fine-tune could win a slot by
  knowing the answers.
* **No lane-C pool.** `[lane_c] pool_prefix` is refused at load (MarketCast
  reads `[marketcast] pool_prefix`). Deleted with it: the `pool_c` window
  draw (`lane_c_windows`, `target_selects`, `target_horizon`, `freq_class`),
  the validator's king-vs-fine-tune verdict over `pool_c`
  (`judge_lane_c_entry`, `family_sizes`, the L6 recipe-tree identity pass
  and `LaneCState.seen`), the validator's `pool_c` wiring, and
  `cascade-pool publish/reveal --lane c` with its target-coverage report.
  The shared lane-C machinery stays: signed manifest + binding, slots,
  `win_beats`, pending / confirm labels, payouts, TTL, the receipt and audit
  records.
* **MarketCast off.** The loader admits `mkt:` targets for the CONFIGURED
  `[marketcast] suites` (not `target_suites`, which is empty while MarketCast
  is inert), so the shipped config loads with `[marketcast] from_block = 0`.
  `active_targets` then counts none of them: an armed lane C has zero active
  slots and its whole share burns (the vector still sums to 1). The intake,
  commit and recipe surfaces keep using `target_suites`, so nothing can be
  submitted until MarketCast is armed.
* **Lane-C verdict keys** (`alpha`, `bootstrap_B`, `gain_margin`,
  `min_target_*`, `n_windows`, `tau_public`, `min_public_items`) stay parsed
  for the next registered competition; MarketCast judges with its own
  `[marketcast]` keys.

## Concurrent competitions 2026-10-20 (owner)

Owner (2026-10-20): "we might have multiple comps in lane C at once". The
owner did not choose the wind-down rule; the default below is the
recommended one and is configurable per competition.

* **Config.** One table per configured competition,
  `[lane_c.competitions.<kind>]` with `weight` (>= 0), `from_block`,
  `until_block` (0 = no end), `wind_down` (`"finish_full_pay"` |
  `"stop"`) and `wind_down_max_eras` (default 6). Parsed onto
  `LaneCConfig.competitions` (`LaneCCompetition`); an absent table derives
  one weight-1, never-ending competition per target kind. Every `[lane_c]
  targets` kind needs an entry; an entry must name a registered forward-only
  kind ("unknown competition" / the forward-only refusal otherwise).
* **One source of truth for a start.** A kind registers
  `from_block_table`: `mkt` registers `"marketcast"`, so MarketCast's start
  IS `[marketcast] from_block` (it keeps its own arming checks: era
  boundary, >= v2 and lane C, the validator bench source, the owner
  hotkey). `from_block` in `[lane_c.competitions.mkt]` is refused at load. A
  kind without a sibling table (`""`) reads its own `from_block`.
* **Load checks.** `from_block` / `until_block` are era boundaries on the
  grid in force at them; `[lane_c] from_block <= from_block` (so `>= [v2]
  from_block` as before); `until_block > start` (and needs a start);
  `weight >= 0` with a positive total.
* **Emission.** Lane C's share is split by weight first, normalised over
  EVERY configured competition (pending and ended ones included), then
  equally among the competition's targets: a target's slot is `share ×
  weight / Σweight / n_targets` × its multiplier
  (`cascade.shared.lane_c.lane_c_split`). One competition starting or ending
  never changes another's pay: an idle competition's part burns, and a
  configured competition with no targets burns its weight. The vector sums
  to 1 in every state.
* **States** at a settlement `B` (`competition_state`, pure in config):
  `pending` before the start (`mkt`: `MarketCastConfig.active`, i.e. `B >=
  from_block`; own-table kinds the same rule), `active` through `B <=
  until_block` (the settlement AT the end is the last judged one),
  `winding_down` after it while `wind_down = "finish_full_pay"` and fewer
  than `wind_down_max_eras` eras have passed since `until_block`, then
  `ended`.
* **After the end.** No new entries: the intake answers `409
  competition_ended` once the chain block reaches `until_block`, and
  `cascade lane-c commit` refuses. The trainer fails a queued entry whose
  leg could not finish before `until_block` (`block + leg_hours >
  until_block`) with the terminal class `competition_ended`: nothing was
  rented, so nothing is charged, and it is not the miner's fault. No new
  verdicts: from the first settlement after the end, every new entry,
  carried pending entry and backlog candidate of that competition gets an
  `ended` row (dropped, never judged) and the backlog empties.
* **Wind-down.** `finish_full_pay` (default): a win held at the end keeps
  multiplier 1.0 while it is inside its own recorded full-pay period
  (`age < full_eras`) AND the competition is `winding_down`; then it stops
  (no decay tail). `stop`: pay stops at `until_block`. Unpaid parts burn;
  the TTL still applies.
* **Receipts.** `lane_c_pay` carries `competitions` (per competition: its
  config, the resolved `from_block`, `frac`, `state`, `targets`,
  `slot_frac`), `slot_frac` (per slot-bearing target), `slots` (their
  count) and `"wind_down": true` on a win paid in the wind-down. The vector
  is built from `slot_frac` (`lane_parts(c_slot_frac=...)`).
* **Audit.** `_lane_c_paid_wins` (behind `check_weights` / `derive_weights`,
  so satellites too) re-derives `competitions` and `slot_frac` from config
  and rejects any difference, checks the `wind_down` flag and the wind-down
  rule; `check_lane_c` refuses any row other than an uncounted `ended` row
  for a competition past its end.
* **Overlay (DEC-CA-0057).** `[lane_c]` is overlay-settable, so a configured
  competition's `weight` and `until_block` are too
  (`COMPETITION_OVERLAY_KEYS`); an end is never moved to, or away from, a
  block before the overlay's apply block. Release-only: adding or removing
  a competition (a new kind is code), its `from_block` (and `[marketcast]`,
  already release-only) and its wind-down rule (the promise a winner was
  paid under). An overlay restates the whole `competitions` table (a table
  value replaces the base value whole).
* **Shipped.** Mainnet and testnet: `[lane_c.competitions.mkt] weight =
  1.0`, no end. With one competition every slot is `share / n_targets`,
  exactly the previous rule at every block.
