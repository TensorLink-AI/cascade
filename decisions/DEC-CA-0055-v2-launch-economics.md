---
id: DEC-CA-0055
type: decision
title: "v2 launch economics: 95% pre-launch burn, the carried king capped at 5% until a v2 crowning, a 7-day random-init window before the reign clock starts, king-only self-improvement promotions, a lane-A share ladder (new king at a margin-tiered start, 25-80% by its crowning LCB, 50% in level units; +10% per verified warm start up to 95%, a failed check drops to 10%; the last former king 5% for 5 days), and the king funds its own leg (a failing key slashes it to 1%, escalating: >= 6 eras for a first lapse, the rest of the reign for a second)"
status: proposed
date: 2026-10-19
tags: [consensus, emissions, cascade, promotion, v2]
revisit_when: "the share ladder leaves most of lane A burned for weeks (kings rarely pass the check: the margin or the six-mean test is too strict); or a king farms the ladder with checkpoints that pass the check but not the next duel (tighten the check to the duel margin); or the 7-day window shows the from-scratch lineage still improving fast at day 7 (lengthen random_init_eras); or the pre-launch burn's rollout cost validators vtrust (pin it to a typed block)"
relations: {depends_on: [DEC-CA-0013, DEC-CA-0016, DEC-CA-0034, DEC-CA-0039, DEC-CA-0043, DEC-CA-0044, DEC-CA-0050, DEC-CA-0054], relates_to: [DEC-CA-0004, DEC-CA-0015]}
---

## Decision

**Release amendment, 25 October 2026:** mainnet Lane A targets **30 October
2026, 12:00 UTC**, resolved to the first era start at or after that time
using finalized chain timestamps. Burn remains a separate 51%-stake readiness
vote, with this release's `prelaunch-95-r2` signal. The carried king must use a
pinned v2-compliant generator and pass launch qualification; historical source
is not silently padded or exempted from the 8–32-channel rule. King promotions
record their actual fire block, require the full notice window still to remain,
and are announced by hash in the signed manifest chain before taking effect.

Owner (2026-10-19), for the v2 go-live (`[v2] from_block`, DEC-CA-0050):

1. **Pre-launch burn, decided on chain.** It rides the DEC-CA-0045 readiness
   note as a secondary feature (the DEC-CA-0049 pattern): each validator on
   this release signals it, and the boundary after 51% of permit stake has
   signalled, every validator sends 95% of its weight vector to `burn_uid`;
   miners keep their usual split of the other 5% (`[v2] prelaunch_burn =
   0.95`). It ends at the first v2 settlement.
2. **The carried king.** The current (toto2) king's generator keeps training
   as king into v2, but from the v2 block until a king is crowned at a v2
   settlement, lane A (that king and its court) is paid at most 5% of total
   emissions; the rest burns (`[v2] carryover_king_share = 0.05`). Lane B is
   unchanged.
3. **Random-init window, then the reign clock.** Every v2 era before
   `v2_first_era + 14` (7 days) trains from random init; no promotion can
   take effect inside it (`[v2] random_init_eras = 14`). The promotion reign
   clock counts from the window's end. **Check timing:** the king check is
   due at the first settlement whose would-be warm start
   (`min_effective_era` of that boundary) lands at least `[v2]
   reign_warm_start_blocks = 21600` (3 days) after the reign began
   (`cascade.shared.era.v2_check_due`): a passing check's warm start lands 3
   days into the reign — the era of notice is absorbed, not added — so the
   first warm start lands at ~day 10 if the king holds, later while
   challengers keep dethroning. A dethrone during the notice era does not
   cancel an accepted promotion.
4. **Only the king is promoted, and only when it improves.** At each ripe
   reign point the king's latest checkpoint (trained from the current init)
   is checked against that init:
   - the private pool: the increment-unit LCB of king − init clears the
     margin (DEC-CA-0039; LCB is always increment-denominated in v2), and
   - every suite's mean on the trainer-signed bench: GIFT-Eval, BOOM and TIME
     mean CRPS and mean MASE, each strictly lower than the init's (six
     comparisons, all must improve), and
   - the king was not dethroned in the window.
   Pass ⇒ a promotion fires with the king's checkpoint as the ONLY member
   (k = 1 in v2; no challenger checkpoints, no rolling top-k). Fail ⇒ no
   promotion. The reign clock restarts either way.
5. **Lane-A share ladder** (v2 only, of lane A's pool; lane B keeps its 50%
   of total):
   - a king crowned at a v2 settlement starts at a margin-tiered share of
     lane A (see "Margin-tiered start share 2026-10-20" below; 50% when the
     crowning was judged in level units);
   - each passed check: +10%, capped at 95%;
   - a failed check: drops TO 10%, then climbs +10% per later pass;
   - a dethrone: the new king starts at its margin-tiered share;
   - the last former king gets 5% of lane A for 5 days (in blocks) from the
     settlement that dethroned it, then nothing; older kings nothing;
   - everything unpaid burns.
   The carried-king cap (2) applies until the first v2 crowning, in place of
   the ladder.
6. **The king pays for its own leg.** In v2 the era king's leg rents on the
   king's deposited key (the payer vault, DEC-CA-0036), not the operator's.
   When that key fails (rejected, out of funds, revoked) the operator still
   trains the king leg, and the king is slashed to 1% of lane A (ladder state
   unchanged; the rest of its share burns) until it fixes the key: once the
   vault holds a ROTATED key (`/v1/credentials`) that Lium accepts, the slash
   lifts from the next era — no retrain. A fallback caused on our side
   (make-good, capacity, an operator-billed entry, a lost record) is
   `operator_fault`: the operator trains the leg and nobody is slashed. The
   trainer records the funding source on the signed manifest's king entry
   (`king` / `operator` / `operator_fault`); validators slash only on
   `operator`. The slash ESCALATES within a reign — see "Escalating key slash
   2026-10-20 (owner)" below (it replaces "until it fixes the key" for the
   first lapse's minimum and for every later lapse).

7. **The king rotates its key.** A signed `/v1/credentials` deposit (the
   `/v1/fund` identity gate, nothing queued) replaces a hotkey's Lium key in
   the payer vault at any time; it does not use the one lifetime submission.
8. **Seed epochs: the king trains once per warm start.** A seed epoch runs
   from one warm start to the next (`cascade.shared.era.v2_seed_epoch_start`):
   it begins at the first v2 era (random init, covering the 7-day window and
   the first reign) and at each v2 promotion's `effective_era`; every era in
   it shares one training seed set (`v2_era_seed_block`) and one init. The
   king's leg trains once at the epoch start on the king's generator; a
   challenger that dethrones inside the epoch trained with the same seeds and
   init, so its checkpoint IS the king's leg for the rest of the epoch (no
   retrain). A failed check changes no init, so the king does not retrain:
   the same checkpoint is re-checked at the next ripe point on fresh eval.
   Lane-B merges change the init only at a warm start (they ride the
   promotion's effective era); lane-B payouts still start at the settlement
   that judged the win. Eval is unaffected: every settlement draws its own
   windows from its round id and the pool snapshot refreshes daily, so the
   king's one checkpoint is re-scored on fresh data at every settlement.
   Accepted risk: training seeds are public for a whole epoch (the window plus
   the first reign, ~10 days, at launch).
9. **Dethrone bar on v2 settlements** (owner 2026-10-19: "increment margin
   to clear go from 10% to 1%"). `[v2] win_margin_start = 0.10`,
   `win_margin_end = 0.01`, `margin_warmup_blocks = 18000`: a fresh king's
   increment-unit LCB bar decays linearly over 2.5 days of tenure. Applies
   from `[v2] from_block` with no vote (every validator is on the release by
   then); DEC-CA-0049's 0.4 → 0.2 keeps judging toto2 eras, and a round that
   falls back to level units keeps the level bar
   (`ChainConfig.koth_params` / `judged_level_params`). Testnet leaves it
   unset. Changing it later is an overlay (DEC-CA-0057).
10. **King check bar kept separate** (owner 2026-10-19). The king check's
   pool half uses a FIXED `[v2] king_check_margin = 0.2` (increment units,
   king vs init ≈ an LCB of 0.2 % relative improvement; no tenure decay;
   `ChainConfig.king_check_params`), not the dethrone bar above.
11. **Lane-A split is config** (owner 2026-10-19: set the composition without
   validator updates). The ladder numbers are `[v2] ladder_start_tiers`
   (`[[0.50, 8000], [0.25, 6500], [0.10, 5000], [0.01, 2500]]`),
   `ladder_start_level_bp` (5000), `ladder_step_bp` (1000), `ladder_cap_bp` (9500), `ladder_fail_bp` (1000),
   `key_slash_bp` (100), `court_bp` (500), `court_blocks` (36000)
   (`cascade.shared.emission.ladder_params`), all overlay-settable.

## Interfaces

The validator / audit / shared half is shipped; the trainer half implements
exactly this:

- **(a) Manifest `funded_by`.** `TrainedEntry.funded_by: str = ""`
  (`cascade.shared.manifest`, one of `""`, `"king"`, `"operator"`), dropped
  from the canonical (signed) bytes when empty. The trainer sets it on the
  KING entry of every v2-era settlement: `"king"` when the king leg rented on
  the king's own payer-vault key, `"operator"` when the operator's fallback
  trained it. Validators slash the settlement's king (not one crowned at that
  settlement) to 1 % of lane A when it reads `"operator"`; the ladder state is
  unchanged.
- **(b) Bench report `init` entry.** `cascade.shared.bench_report.BENCH_ROLES`
  adds `"init"`; `BenchReport.init_entry()` finds it. Every bench report of a
  v2 settlement carries ONE `role = "init"` entry: the CURRENT init's six
  BenchScores (benched once per seed epoch and repeated), with
  `trained_pointer` = the init checkpoint — the era's `warm_start_ckpt` when
  warm (validators require equality), or, in a random-init era, a published,
  fetchable checkpoint of the random-init model built from the era's seed
  (validators score that pointer as the pool baseline). The report must also
  carry the king pointer's entry. Validators read the PREVIOUS settlement's
  report (`prev_round_id`): the check settlement's own report lands after its
  manifest.
- **(c) Receipt stamps** (`cascade.shared.receipt`, all drop-when-default):
  `prelaunch_burn_block` (the vote-resolved block), `emission`
  (`cascade.shared.emission.emission_record`: `{"boundary",
  "king_since_block", "policy": {prelaunch_burn, lane_a_cap, king_share,
  former_share, king_slash}, "ladder": {"prev_bp", "bp", "event":
  crowned|pass|fail|deferred|none|carried, "court_hotkey",
  "court_until_block", "slashed", "funded_by"}}` — a crowning adds
  `crown_lcb`, `crown_units`, `crown_tier`, `crown_start_bp`,
  `crown_below_floor`, see "Margin-tiered start share" below) whenever a DEC-CA-0055 rule
  shaped the vector, and `king_check` = `{"result": "pass"|"fail"|"deferred",
  "round_id", "block", "king_hotkey", "king_pointer", "reasons": [...],
  "bench_round", "warm_start_era", "lcb", "margin"}` at the settlement the
  check was due. The trainer fires a promotion only on a `"pass"`.
- **(d) PromotionRecord for a v2-lineage generation** (fired at/after `[v2]
  from_block`, under `[v2] king_ladder`): exactly ONE member = the passing
  check's `king_pointer`, `fired_block >=` the check's `block`, `effective_era
  == min_effective_era(round, fired_block)` (no alignment) and `>=
  v2_first_warm_era`; never a checkpoint that is already the live or staged
  init. No ripeness predicate: the pass replaces it. Validators whose clock
  cannot attest (adopted / catching up) skip only the pass requirement.
- **Reign clock (trainer must mirror).** In v2 the clock starts at
  `max(reign start, v2_reign_floor_block)`, restarts at the CHECK settlement's
  boundary on pass AND fail (never on a deferral) and on a dethrone — never at
  the promotion's fire or install. A staged v2 promotion survives a dethrone.
- **(e) Seed epochs** (gated on `[v2] king_ladder` so an already-live v2 —
  testnet — keeps per-era seeds): a v2 era's training seeds derive from
  `v2_era_seed_block(cfg, era, warm_eras)` (warm_eras = effective eras of the
  accepted v2-lineage promotions), and the manifest's era stamp carries THAT
  `seed_block` (so its entry window spans the epoch). One init per epoch: a
  promotion or a lane-B merge changes `warm_start_ckpt` only at a warm start.
  The king entry holds ONE pointer per epoch: the epoch's king leg, or after
  a dethrone inside it the dethroner's trained checkpoint (no retrain); a
  failed check keeps the same pointer.
- **`/v1/credentials`** (`cascade.funding.intake.FundingIntake.credentials`):
  the `/v1/fund` headers; signature over `canonical_fund_message(
  "credentials", hotkey, ref, ts, sha256(lium_key))`; requires a registered
  hotkey whose revealed commitment matches `X-Commit-Ref`; replaces the
  payer-vault key; queues nothing.

## Shipped (inert)

Validator / audit / shared half, inert on mainnet until `[v2] from_block` is
typed or resolved from the configured UTC date, except the pre-launch burn,
which waits for its separate on-chain vote:

- config: `[v2] random_init_eras = 14`, `reign_warm_start_blocks = 21600`,
  `prelaunch_burn = 0.95`, `carryover_king_share = 0.05`, `king_ladder =
  true` (validated in `parse_v2_economics`); `prelaunch_burn_from_block`
  (typed override, 0 = vote). Testnet keeps every key at 0/false: its v2 is
  already live and each rule counts from `[v2] from_block`, so arming one
  would re-judge settled eras.
- vote: DEC-CA-0045 secondary feature `prelaunch-<pct>-<readiness>`
  (`prelaunch-95-r2` for this release),
  aligned to the next settlement boundary after lock-in, persisted in
  `activation_prelaunch_burn_state.json`, stamped as `prelaunch_burn_block`
  and replayed by `apply_receipt_activation`. The 128-byte note budget: with
  every current segment locked at 8-digit blocks the full note would exceed
  the `Raw128` field, so `own_signal_payload` drops the newest segment
  (logged) until it fits — type the already-decided rollover / forfeiture /
  margin-v2 blocks into chain.toml before relying on the burn vote.
- weights: `WeightsMixin._apply_weights` (scored path, resync and re-assert)
  builds every vector through `cascade.shared.emission` (`lane_parts` keeps
  the lanes apart, so the cap / ladder / slash never touch a uid's lane-B
  pay); the audit's `check_weights` rebuilds it from the `emission` record and
  checks the record against the config, the verdict, the king check and the
  manifest king's `funded_by`.
- king check + ladder: `cascade.validator.king_check` (state in
  `ChampionState.king_share_bp / court_* / king_slashed / king_check_pass /
  epoch_init`, carried in the `pre_round` snapshot).
- promotions: `_stage_promotion_record` routes v2 records through
  `_v2_promotion_reason`; `_v2_lineage_size` uses `v2_first_warm_era`.
- era gate: seed-epoch seed block, one king pointer and one init per epoch;
  the audit's `check_era` accepts a stamped seed block only when it names a
  legal epoch start (the first v2 era or `>= v2_first_warm_era`).

Accepted risk: the king check reads a bench report fetched at judge time; a
validator that judges before the previous settlement's report lands defers
where a later one passes/fails (the next settlement re-checks).

## Consensus surface

All of it is derived by every validator from the settlements it judges
(reign clock, increment LCB vs init, the signed bench report) and stamped on
the receipt; the audit re-derives the vector. Rides `[v2] from_block` (no
readiness vote), except the pre-launch burn, which starts at the release.

## Resolved launch coordination

The burn waits for this release's 51% eligible validator stake readiness
threshold. Installing the release alone does not activate it. The dated Lane A
launch is independent; the release amendment above supersedes the original
release-time wording.

## Trainer shipped

All of it is inert until `[v2] from_block` is typed or calendar-resolved, and gated on
`[v2] king_ladder` (the testnet v2 keeps the DEC-CA-0050 top-k path and its
per-era king legs).

- **King-only promotion** (`trainer/runner_promotion._v2_king_check_step`,
  `TrainerPromotion.promote_king`): in the v2 lineage the trainer never runs
  `maybe_promote`; it fires only on a `king_check` with `result = "pass"` read
  off the anchor validator's signed latest receipt — one member (the check's
  `king_pointer`), `fired_block` = the check's `block`,
  `effective_era = min_effective_era(fired_block)` (not fired if that falls
  inside the random-init window; a member already in the live set is not
  re-promoted). A check fires once (`last_king_check`, persisted); `fail`
  restarts the trainer's reign clock at the check's block, `deferred` changes
  nothing.
  The trainer's own clock is only cross-checked against `v2_check_due` (logged,
  never vetoes the validators' pass).
- **Init bench** (`rolling._maybe_bench_init`, `runner_rolling._rolling_bench_init`):
  once per seed epoch on the era's operator king pod — the member checkpoint,
  or the random-init model rebuilt from the epoch's training seed
  (`moe_trainer.build_random_init`) and PUBLISHED (operator identity) so
  validators can score it. It rides every v2 bench report of the epoch as a
  `role = "init"` entry (`trained_pointer` = `warm_start_ckpt`, or the published
  random-init checkpoint; `miner_hotkey = ""`, `miner_uid = -1`); never a
  promotion candidate.
- **King's own key** (`runner_rolling._rolling_train_king`): the v2 king leg
  rents on the king's payer-vault key through the funded challenger path; a key
  failure (absent/expired, auth, out of funds, tamper on its own pod) falls back
  to the operator's pod. Market/infra failures retry on the key. The completed
  leg records `funded_by`; every ladder-era manifest's king entry carries it
  (unknown ⇒ `"operator"`; a dethroner's adopted entry counts as `"king"`). The king's vault TTL is re-stamped at every
  era roll and periodic reconcile while it is king.
- **Seed epochs** (`rolling._new_era_state`, `_carry_epoch_king`): the era stamp's
  `seed_block` is `v2_era_seed_block(cfg, era, warm_eras)` (warm eras = the
  ledger's v2-lineage effective eras), so the train window and seeds follow the
  epoch. Later eras carry the epoch's king leg (or the dethroner's entry; settled
  legs stay across an in-epoch roll), the init bench and the operator king pod;
  lane-B merges apply only at a warm start (`LaneBDriver.init_for(apply_pending=False)`).
- **Carry-overs**: `prev_manifest_sha256` is emitted only from
  `[manifest] hash_link_from_block` (0 = never); an unrecorded leg failure is an
  operator fault (no burned attempt, backs off); the intake skips an entry still
  backing off (`not_before`) before admission/dedup.

## Escalating key slash 2026-10-20 (owner)

Owner: "if it drops to 1% and we pay the key then we want to penalise the king
for longer". CONSENSUS, replaces the item-6 lift rule in place (nothing on
this branch is deployed — no compatibility path):

- A **lapse** opens at a scored settlement whose manifest king entry reads
  `funded_by = "operator"` while no lapse is open, and closes at one reading
  `"king"`. Consecutive operator-paid settlements are ONE lapse;
  `"operator_fault"` (our side) and `""` neither open nor close one.
- **First lapse of a reign:** the king is at `key_slash_bp` for at least
  `[v2] key_slash_min_eras` eras (default 6 = one 3-day round) from the
  boundary that opened it — `slash_until_block = boundary + key_slash_min_eras
  × era_length_blocks(round, boundary)`, exclusive — even if it rotates to a
  working key sooner; after that, until a king-funded leg (today's rule).
- **Second or later lapse in the same reign:** `key_slash_bp` for the rest of
  the reign (until dethroned / forfeited), whatever the key does.
- **A crowning resets it** (dethrone, DEC-CA-0048 forfeiture successor,
  resync demotion: a fresh `ChampionState`). The ladder share
  (`king_share_bp`) and the court are untouched, as before.
- Slashed at a settlement ⇔ operator-paid now, or `boundary <
  slash_until_block`, or lapses ≥ 2. `key_slash_min_eras = 0` is the old
  first-lapse behaviour.

Mechanics: `cascade.shared.emission.key_slash_step` (pure) is run by the
validator in `_v2_settlement_economics` from `ChampionState.king_key_lapses /
king_lapse_open / king_slash_until_block` (drop-when-default; reset on
crowning, incl. `apply_forfeit`). The receipt's `emission.ladder` adds
`key_lapses`, `lapse_open`, `slash_until_block`, `slash_reason`
(`operator_paid` | `min_window` | `repeat_lapse` | `""`), `slash_until` (the
earliest boundary the slash can lift; null = with a king-funded leg / never
this reign) and the prior `prev_key_lapses` / `prev_lapse_open` /
`prev_slash_until_block` (absent at a crowning). The audit
(`audit.checks.key_slash_mismatch`, inside the weights check) replays the step
from the recorded prior; satellites bind that prior to the previous receipt's
`state_post.ladder` (`ladder_prior_mismatch`, continuity), which now carries
the same three fields. `key_slash_min_eras` is validated (int ≥ 0) and is
overlay-settable (DEC-CA-0057 `ALLOWED["v2"]`), like `key_slash_bp` and the
other ladder keys.

Trainer: the failed-key digest that decides "rotated?" is persisted in the
era state (`RollingState.king_failed_keys`, dropped from the file when empty),
so a restart cannot let the same failed key lift the slash. The "key fixed"
probe stays an authenticated `GET /templates`: no Lium balance read exists
anywhere in the code (intake deliberately skips a balance pre-check). An
authenticated but unfunded rotated key therefore closes the lapse, and its
next own-key leg failing opens a SECOND lapse (rest of the reign).
TODO: require balance ≥ the king leg's estimated cost once a Lium balance
endpoint is adopted (no endpoint is invented here).

## Margin-tiered start share 2026-10-20 (owner)

Owner: "consider if the king should get a higher or lower start point to win
the round — if they clear by a 50% margin or 25%, 10% or 1% they start at
different share of emissions", then "ignore tenure". CONSENSUS; replaces the
single `ladder_start_bp` in place (nothing on this branch is deployed — no
compatibility path). The tenure-decaying dethrone bar (`[v2] win_margin_start`
0.10 → `win_margin_end` 0.01 over `margin_warmup_blocks`) is unchanged; the
tiers only set the STARTING ladder share at a crowning.

- **Measure.** The crowned challenger's verdict LCB — the family-wise (max-T)
  corrected increment-unit LCB the dethrone bar was compared against
  (`VerdictRecord.lcb`), never the point estimate.
- **Rule.** `[v2] ladder_start_tiers` = `[lcb_threshold, start_bp]` pairs,
  thresholds strictly descending in (0, 1); the king starts at the start_bp
  of the HIGHEST threshold its LCB meets (`lcb >= threshold`). Defaults
  (PLACEHOLDERS): `[[0.50, 8000], [0.25, 6500], [0.10, 5000], [0.01, 2500]]`.
  start_bp values are integers in `[ladder_fail_bp, ladder_cap_bp]`, and
  non-increasing down the list (checked at load). An LCB below the lowest
  threshold cannot win (the bar's floor is 0.01); if one does (a config
  mismatch) it starts at the lowest tier and the validator logs it
  (`crown_below_floor`).
- **Units.** The tiers apply only to an INCREMENT-unit LCB
  (`cascade.shared.emission.crown_units`: increment margin mode AND the init
  baseline scored AND, for a cohort's joint max-T bound, `[scoring]
  cohort_maxt_increment_from_block` reached). A crowning judged in LEVEL units
  (random-init eras; a level fallback with no init baseline inside a
  warm-start era; a level-unit max-T bound) starts at the fixed `[v2]
  ladder_start_level_bp` (5000). A DEC-CA-0048 forfeiture successor (no duel)
  also reads as `ladder_start_level_bp`.
- **After the start, unchanged:** +`ladder_step_bp` per passed king check up
  to `ladder_cap_bp`, a fail drops to `ladder_fail_bp`, the escalating key
  slash on top, the former king's court, everything unpaid burns.
- **Record + audit.** At a crowning the ladder record stamps `crown_lcb`,
  `crown_units`, `crown_tier` (the threshold used; null in level units),
  `crown_start_bp`, `crown_below_floor`, and `bp` = the start share. The
  weights check (`audit.checks.crown_start_mismatch`) requires `crown_lcb` to
  equal the receipt's verdict LCB (itself replayed by the verdict check),
  re-derives the units from the receipt (`receipt_crown_units`, the same rule
  the verdict replay uses) and re-derives tier and start share from the
  config — never trusted. Satellites run the same check. Both keys are
  overlay-settable (DEC-CA-0057 `ALLOWED["v2"]`) like the other ladder keys.
- **Calibration (before arming).** Run `scripts/bench_seed_noise.py`-style
  seed replicas of one recipe on the private pool and look at the
  distribution of the increment LCB of replica-vs-replica duels: the 0.50
  tier must be rare under noise alone (a lucky reseed must not start at
  80 %), and the lowest tier should sit at `win_margin_end`.
- **Incentives.** A big improvement split into several small wins pays less
  (each crowning restarts at a low tier, and each later win must clear the
  fresh-king bar again), so miners gain by shipping the whole gain at once. A copy of the
  king cannot reach the bar at all (its increment LCB is ~0), so the tiers
  do not open a copy route.
