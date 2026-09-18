# All-time top-3 warm-start leaderboard — activation plan (DEC-CA-0044)

Status: **BUILT, mainnet UNARMED** (`[scoring] cascade_alltime_from_block = 0`);
testnet armed at block 1. This is the release-then-activate protocol the
subnet uses for every consensus gate (DEC-CA-0016/0019/0038/0039 precedent).

## What changes at the block

From a round whose epoch boundary is `>= cascade_alltime_from_block`:

| | reign-scoped rule (today) | all-time rule |
|---|---|---|
| member population | this reign's benched checkpoints | all-time top-k, any reign |
| ranking score | uniform geomean of six numbers | suite-weighted: GIFT-Eval 50 · BOOM 25 · TIME 25 |
| when a set changes | ripe reign (`cascade_reign_rounds`) | a better checkpoint entered the top-k |
| pacing | one full reign | `cascade_notice_blocks` (7200 ≈ 24h) announced notice |
| validator provenance | signed report, **this reign only** (attesting) | signed report, **any round** |
| validator floor | uniform geomean, epsilon 5% | weighted score, epsilon 5% |
| validator timing | `is_ripe` (reign rounds) | `is_spaced` (notice blocks since anchor) |

Rotation across members, the single signed manifest pin, the persisting king,
receipts, the record schema and `contract_digest` are unchanged.

## Why every validator must upgrade BEFORE the block

The first all-time record will carry members benched in earlier reigns and
may fire two rounds after the previous generation. A validator on the old
release rejects that record with `warm_start_member_out_of_reign` and/or
`warm_start_promotion_early`, then rejects **every subsequent round** (each
manifest pins a member of the generation it never accepted) until upgraded.
That is a full weight fork for that validator, not a one-round blip.

## Steps

1. **Ship this release with the gate at 0.** Validators restart at leisure;
   nothing changes for them (`cascade_alltime_active` is False everywhere).
   The trainer runs the leaderboard in SHADOW: `promotions/leaderboard.json`
   publishes the board (`rule_active: false`), no announcement fires, the
   legacy reign-scoped selection keeps promoting.
2. **Validate on testnet** (armed at 1): watch one full announce → 24h hold →
   fire cycle; confirm the record publishes before the first manifest pinning
   a new member, `cascade round` / `cascade heat` / `cascade leaderboard`
   show the countdown, `cascade score --warm-start upcoming` resolves, and no
   testnet validator logs a `warm_start_*` rejection.
3. **Confirm every mainnet validator is on the release** (six external
   operators as of 2026-08). Same channel as the margin-decay rollout.
4. **Pick the block**: an epoch boundary at least one full round after the
   last confirmation, announced with the block → UTC time in the release
   note. Set `cascade_alltime_from_block` in `chain.toml`, ship the config
   commit; validators AND the trainer restart onto it before the block (the
   `[scoring]` values are not in `contract_digest`, so the restart order does
   not matter, only that all are done before the block).
5. **First all-time boundary**: the trainer's leaderboard (already
   backfilled in shadow) announces the change if the all-time top-k differs
   from the live set; the record fires 7200 blocks later. Watch validator logs
   for `warm_start_*` reasons on that round and the next.

## Rollback

Set the gate back to 0 (or above the current block) and restart. Any
generation already accepted stays live — the field keeps training from it —
and the next promotion is judged under the reign-scoped rule again. Nothing
in receipts or the audit trail needs rewriting.
