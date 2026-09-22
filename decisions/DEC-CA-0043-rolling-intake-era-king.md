---
id: DEC-CA-0043
type: decision
title: "Rolling intake + era king: challengers train the moment they are funded, verdicts settle on a 3h grid, the king's leg is trained once per 12h era and cached — every behaviour change block-gated at ONE rollover"
status: proposed
date: 2026-09-20
tags: [funded, rounds, cadence, king, warm-start, consensus, block-gate]
revisit_when: "the retrain-noise measurement (DEC-CA-0036's open gate) puts a fixed generator's seed-to-seed checkpoint spread above ~half the 0.5% margin floor (then the confirmation leg ships in this stack before ROLLOVER); or within-era slips (legs landing after their target settlement) turn out common (multi-member cohorts per manifest); or the slowest external validator's settlement time approaches 1h (reconsider the 3h grid); or an era-king manifest is rejected by a validator that could not fetch a 24h-old seed block hash (archive RPC)"
relations: {builds_on: [DEC-CA-0036, DEC-CA-0038, DEC-CA-0039, DEC-CA-0013, DEC-CA-0004, DEC-CA-0016], revises: [DEC-CA-0013, DEC-CA-0016], related: [DEC-CA-0037, DEC-CA-0044, DEC-CA-0008, DEC-CA-0012]}
---

## Decision

From ONE rollover block (`ROLLOVER`, a multiple of the current 3600 grid,
owner call) the funded pipeline stops being boundary-synchronous:

* **Rolling intake (trainer policy, `[round] rolling_from_block`).** A funded
  challenger's leg starts the moment it is funded, on its payer's pod (or
  the operator-lane fallback), targeting the earliest epoch boundary its
  wall + publish margin can clear. Legs cross boundaries. `funded_field_cap`
  becomes the number of legs in flight. The funded queue gains an
  `in_flight` state carrying the target boundary, the era and the start
  block; `recover_in_round` and the round-entry pod sweep never touch it.
* **Settlements (the epoch grid, 3h from ROLLOVER).** Every boundary is a
  *settlement*: ONE manifest = the era king's entry + every challenger
  harvested, ingest-verified and benched since the last settlement (same
  era). Nothing finished ⇒ no manifest. Manifests are hash-chained
  (`prev_round_id`); validators walk `round-<id>.json` forward from their
  last handled settlement — `latest.json` is only the head hint — so a
  validator that missed settlements catches up through every one,
  dethrones included, before the era-king envelope is checked.
* **Era king (consensus, `[scoring] era_king_from_block`).** An *era* is
  `era_settlements` (4) consecutive settlements sharing one set of training
  seeds, one init and one cached king checkpoint: 12h eras with 3h verdicts
  inside them — today's king-leg cadence, four times the verdict cadence.
  The king's leg is trained ONCE per era on the operator's account and
  cached; the next era's king pre-trains during the last wall + margin of
  the current era. A dethrone does not end the era: the winner's checkpoint
  (trained under the era's seeds and init with the full budget) becomes the
  era king at zero operator cost, and in-flight challengers stay valid
  against it.
* **Tenure and ripeness in blocks (consensus,
  `[scoring] tenure_blocks_from_block`).** `margin_warmup_blocks` and
  `cascade_reign_blocks` replace the round-denominated knobs from the
  rollover, so the 4× faster grid does not 4× the decay and promotion
  clocks in wall time; a king crowned on the old grid keeps its wall-time
  tenure across the switch. The reigning king is anchored ONCE, at its
  first settlement past the gate, at `block − tenure_rounds × old_grid`
  (`era.legacy_king_anchor`) by EVERY validator — one that recorded the
  real crowning block before the gate discards it (the counter is the
  quantity the whole fleet agreed on before the gate; the real block is
  known only to validators that had upgraded before that crowning, and two
  anchors mean two margins at one settlement). The anchor is persisted as
  `king_since_block` with `tenure_anchor_gate` marking it done; every later
  settlement counts blocks from it, and a king crowned after the gate
  counts from its real crowning block. (Re-imputing the anchor each settlement from the counter,
  which keeps advancing on the new grid, grew the tenure `old/new` = 4× per
  settlement — found on testnet 2026-09-21.)

Before ROLLOVER every path is bit-identical to main (receipt bytes,
manifest bytes, verdicts). The audit replays every receipt under its own
block: era context is REQUIRED after the gate and FORBIDDEN before it.

## Era definition (validator-verifiable, zero trainer discretion)

`era_index = start_block // (epoch_blocks × era_settlements)`; era
boundaries are grid boundaries. Era seeds derive from the PREVIOUS era's
start block hash (`seed_block = start_block − era_length`) — known one era
ahead, so the king's leg and late-era challengers pre-train under the next
era's seeds. **A settlement at boundary B belongs to the era containing
B − 1**: an era's settlements are the boundaries in `(start, start + L]`,
its last settlement is the next era's start block. That is what makes the
intake seamless — a leg that cannot clear `start + L` is exactly one whose
start falls inside the next era's pre-train window, so it starts at once
under the next era's seeds; there is no dead zone and a miner never pays
for a leg that lands on the wrong init. An era ends early ONLY on a
contract-digest change.

## Init — one per era, switches land on boundaries, announced a full era ahead

Era n's init is `members_gen(n)[n % k]` where `gen(n)` is the latest
generation whose `PromotionRecord.effective_era ≤ n` (DEC-CA-0013's
rotation re-keyed on the era; at 4 settlements per era members still
rotate every 12h). No generation live ⇒ random init from the era's
training seed. A fired record carries `effective_era ≥ era(fired_block) + 2`
(the first era boundary at least one full era after it fired); validators
reject earlier claims, stage the record, and install it at its first
effective settlement — never earlier. The king persists across the
promotion (DEC-CA-0004) and gets a fresh era leg on the new init. No
manifest ever mixes inits: an era-n manifest trained from another era's
init fails the envelope.

## King identity — never read from the metagraph

Incentive lags a dethrone by validators' set-weights cadence plus a tempo —
longer than a 3h settlement. The trainer derives the era king from the
validators' signed receipt trail and uses the metagraph only as a lagging
cross-check that logs on disagreement. A stale king cannot reach a
settlement manifest by construction.

## Ref binding — verified at train_block

Each entry's ref must be the hotkey's revealed commitment AS OF the entry's
`train_block` (`poll_commitments(include_history=True)`), not the latest
reveal at the boundary; a re-commit between leg start and settlement never
rebinds the running leg — it queues behind it (one live entry per hotkey)
and becomes the hotkey's next entry when the flight settles. The live loop
wires the history provider itself (`run_forever`, from its chain client);
an era manifest judged with no provider FAILS (`era_ref_unverifiable`) —
the check is never silently off.

## Validator envelope (fail closed, from `era_king_from_block`)

`era` stamp present and parseable; `era.start_block` / `seed_block` / `index`
equal the era derived from the boundary; `era.generation` / `member_index`
/ `warm_start_ckpt` equal the ledger's init for that era; every entry's
`train_block` inside `[seed_block, era_end)`; `prev_round_id` = this
validator's last handled settlement; the king entry (when it is the
champion) carries the pointer this validator judged in the crowning
settlement or adopted at the era's first king leg; every entry's ref
verified at its `train_block`. The same-GPU fallback of the GPU gate is
lifted when no GPU is pinned (per-leg SKU choice, PR #294). `round_id`,
the eval window draw, the jittered mix, the scored horizons, the eval-pool
pin and every `*_from_block` gate resolve off the settlement boundary
exactly as today; only the training seeds are the era's
(`receipt.era_base_seed`).

**Gates are pure.** Nothing in the envelope mutates validator state: the
era's king pointer is adopted and a staged promotion installed only when
the settlement is HANDLED (scored, resync-held or rejected), so a transient
after the gate leaves the state untouched and a legitimately re-published
manifest is still judged.

**The chain.** The trainer's first settlement links to `latest.json`'s
`round_id` (the last legacy round — every validator's last handled round);
a trainer that cannot read the root withholds the settlement rather than
publish an unchained one (legs wait a grid step; nothing is marked done or
burned). Validators walk `round-<id>.json` forward from their last handled
settlement; a walk that does not reach it (a hole, or the depth cap)
handles NOTHING — the position is kept, the depth doubles for the next
poll, and an error names the missing round. Judging the oldest collected
manifest instead would fail its chain check, latch past the gap and lose
any dethrone in it. A trainer outage across a boundary publishes that
settlement late, stamped with the boundary it belongs to (`created_block`
= the boundary; validators floor it to the grid), so finished legs are
never thrown away and re-billed.

## Bench at completion, not at publish

A payer pod benches its checkpoint the moment the leg finishes (harvest →
ingest-verify → bench → tear down); it is never held to a boundary. The
top-N operator re-bench runs on the era king's pod. The settlement's bench
report carries the challengers; the era king's numbers land once, in the
era's first report, and consumers fall through to that report for the king
pointer.

## Seniority

"Earliest reveal gets first pick of fitting executors", not strict
serialization: an earlier entry waiting for an H100 under the price cap is
legitimately overtaken by a later one that fits a 4090 now. The rolling
roster records, per rent, the more-senior queued entries that could have
started and did not — a senior that could not start in that pass
(unrevealed, waiting for its pre-train window, held at the cap, dropped by
dedup) is not a pass-over; the tier-0 funded-roster audit WARNs on any
such pass-over instead of on raw reveal order.

## Knobs (all new, all default 0 = off; shipped in chain.toml)

| key | side | meaning from ROLLOVER |
|---|---|---|
| `[round] rolling_from_block` | trainer | rolling intake, cross-boundary legs, settlement manifests, persistent dedup registry, in-flight queue state, bench-at-completion |
| `[round] era_settlements` | trainer | settlements per era (4) |
| `[scoring] era_king_from_block` | consensus | the era envelope above |
| `[scoring] cohort_maxt_increment_from_block` | consensus | #290's gate — set to ROLLOVER |
| `[scoring] tenure_blocks_from_block` + `margin_warmup_blocks` / `cascade_reign_blocks` / `king_resync_max_blocks` | consensus | tenure / ripeness / resync valve in blocks |
| `[round] epoch_blocks = 900, epoch_blocks_prev = 3600, epoch_activation_block = ROLLOVER` | consensus | the 3h grid via the existing scheduled switch |

Loader asserts: every rollover key names ONE block; `ROLLOVER` is a boundary
of both grids AND starts an era (a multiple of `epoch_blocks ×
era_settlements` — the era grid is absolute, `block // era length`);
`era_king_from_block ≥ cohort_maxt_increment_from_block ≥
cohort_maxt_from_block`; `era_settlements ≥ 1` when armed; the rollover
MUST switch the grid (`epoch_blocks_prev` set, `epoch_activation_block =
ROLLOVER` — a rollover on the old grid would run 4-round eras of the old
length, silently); `funded_pods = "rent"` and `funded_king_rent = true`
(legs and the era king are rented just-in-time). `chain.testnet.toml` is
armed at the first testnet era boundary (600 = 150 × 4, with a nominal
300 → 150 switch declared there).

**Per-leg GPU choice is block-gated too.** `[round] funded_sku_per_leg`
(#294, on by default) takes effect at `era_king_from_block` — the block
the validators stop rejecting mixed types — never before: until then every
round locks one type (#295). Deploying this trainer with the key on changes
nothing before ROLLOVER.

The trainer's generation ledger (`era_state.json`) is rebuilt from the
published `promotions/gen-<n>.json` records (they carry `effective_era`) —
the same source validators install from — so a lost state file never
makes the trainer train a generation before its era.

## Accepted risk

* **The era king is scoutable.** Its checkpoint is a public Hub pointer for
  12h; a miner can score it on public data against their candidate and fund
  a leg only when they beat this realization. Eras do not raise the expected
  false-dethrone count (fewer king re-rolls), but they let miners select
  against a weak training draw. The confirmation leg (fresh seeds, operator
  account) is the fix: ship eras with it, or first measure the seed-to-seed
  spread of a fixed generator's checkpoint quality and show it is well
  under the 0.5% margin — the open DEC-CA-0036 gate.
* Byte-exact audit across mixed SKUs is given up — inherited from #294.

## Arming-gate measurements (before ROLLOVER is chosen)

1. The slowest external validator's settlement time (k+2 evals × 3 horizons
   × 1200 windows + poll) — well under 3h; if > ~1h, reconsider the grid.
2. Historical block-hash lookup of a 24h-old `seed_block` against the RPC
   endpoints the six externals actually use — a validator that cannot fetch
   it fails closed on every era-king manifest.
3. Pool snapshot cadence ≥ settlement cadence, or `check_pool_pin`
   tolerates consecutive settlements sharing a pin.
4. The retrain-noise spread above.

## Out of scope (own gates, later)

The confirmation leg — unless the retrain-noise measurement says otherwise
(then it ships in this stack). Multi-member cohorts in one manifest.

## Rollout

Stacked on #290 (increment-unit max-T) and #294 (per-leg SKU, per-SKU latest
safe start). Do not merge until the six external validators run the
release; ROLLOVER is an owner call, one boundary for every key.
