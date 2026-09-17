---
id: DEC-CA-0044
type: decision
title: "Warm-start init is ONE fixed population — the all-time top-3 benched checkpoints ranked by the suite-weighted score (GIFT-Eval 50 : BOOM 25 : TIME 25), rank-insertion replacement, and a 24h announced notice before a set change takes effect"
status: proposed
date: 2026-09-17
tags: [cascade, warm-start, promotion, leaderboard, consensus, dashboard, miner-cli]
revisit_when: "the first two announced changes fire on testnet — confirm the announce → hold → fire cycle publishes the record before any manifest pins a new member and that no validator rejects on warm_start_member_out_of_reign / warm_start_promotion_early; or a mainnet activation block is picked (every external validator on the release first — the rule widens provenance and changes the timing predicate, so a lagging validator rejects every round from the first all-time record); or the board freezes on a lucky-high bench for more than ~6 generations of rounds without a replacement (the winner's-curse case DEC-CA-0017 warned of — then re-bench the board's members on a fresh corpus or add a staleness discount); or a frozen announced set is re-announced twice in a row (the envelope re-check is churning the notice — consider firing the current target after one re-announce)"
relations: {revises: [DEC-CA-0013, DEC-CA-0015, DEC-CA-0017], depends_on: [DEC-CA-0004], relates_to: [DEC-CA-0011, DEC-CA-0014, DEC-CA-0016]}
---

## Decision (owner-directed, 2026-09-17)

The warm-start generation stops being a per-reign pick and becomes **one
fixed population**: the **all-time top `cascade_top_k` (3)** benched duel
checkpoints — king's or challengers', from ANY reign or generation — ranked
by a **suite-weighted score** and still within `cascade_quality_epsilon` of
the best. Rounds keep rotating across the members (`members[epoch % k]`).

1. **Score.** `weighted_cascade_score` (`cascade/validator/cascade.py`): each
   suite collapses to the geomean of its CRPS and MASE; the three suite
   scores combine as a weighted geometric mean under `[scoring]
   cascade_weight_gifteval / _boom / _time` = **0.5 / 0.25 / 0.25**. The three
   suites are orthogonal results (different data and regimes), so they are
   weighted by suite instead of counting six numbers uniformly; GIFT-Eval
   carries half. Uniform thirds are bit-for-bit the old `cascade_score`, so
   archived generations replay unchanged.
2. **Replacement.** The leaderboard is a bounded rank-ordered list
   (`admit_leader`): a strictly better checkpoint is inserted at the rank it
   beats, the members below slide down one place and the last drops out —
   "beat 2nd ⇒ you are 2nd, 2nd becomes 3rd, 3rd is out". Equal never
   displaces (a re-bench of the same artefact cannot churn the set); a
   listed pointer is never re-admitted. The declared set is the top-k within
   the epsilon (`alltime_members`) — a 3rd that trails the best by more than
   5% is not a legal member and is left out, never padded.
3. **24h notice.** A change to the set is never installed on the spot. At the
   boundary the engine ANNOUNCES it (`pending_change`: the next generation's
   frozen member list, announced round/block, `effective_block =
   announced_block + cascade_notice_blocks`; 7200 blocks ≈ 24h) and fires the
   signed `PromotionRecord` at the first boundary at or after the effective
   block. The announcement rides `status/round.json` and `status/heat.json`
   (`warm_start.upcoming`, with a trainer-estimated `effective_at`) and the
   new public `promotions/leaderboard.json` (the whole board, the live
   generation, the announcement). Miners see it on the website's warm-start
   panel (banner + leaderboard table), in `cascade round` / `cascade heat`
   (`upcoming init` lines with a countdown) and `cascade leaderboard`, and can
   train against the exact announced checkpoint with `cascade score <repo>
   --warm-start upcoming`.
   - The announced set is **frozen** for the window: a better checkpoint that
     lands during the notice takes its rank on the board but waits for the
     next change (announced at the boundary after the fire, effective a
     notice later). Restarting the notice on every better arrival would starve
     a compounding lineage (most rounds improve on their init) — measured
     risk, so the window is fixed at announcement.
   - Exception: if the frozen set fell outside the epsilon envelope (a much
     better arrival moved the floor), firing it would be rejected by every
     validator, so the engine re-announces the current target instead.
   - A dethrone inside the window resets the reign anchor; the fire waits
     until `notice_blocks` have passed since the new anchor (the validators'
     spacing check mirrors it). The countdown shown to miners is therefore
     "a schedule, not a promise", like `next_scheduled_init` already is.
4. **Consensus, block-gated** — `[scoring] cascade_alltime_from_block`
   (release-then-activate, like every other gate; testnet armed at 1, mainnet
   0). A record for a round at/after the block is verified under the rule:
   same signature / generation +1 / `cascade_top_k` cap, but
   - **provenance** is ANY trainer-signed bench report (the reign-scope pin
     `warm_start_member_out_of_reign` is off — members legitimately predate
     the reign; an attesting validator's reign log is still the one-sided
     floor source it always was);
   - the **quality floor** is measured with the weighted score (the trainer
     ranks on it, so an honest top-k set must be checked on the same scale);
   - the **timing predicate** is `is_spaced(notice_blocks)` since the last
     reign anchor (dethrone or accepted generation) instead of
     `cascade_reign_rounds` ripeness — the notice period IS the pacing.
   Before the block the reign-scoped rule (DEC-CA-0013/0015/0017) applies
   unchanged, so audit and catch-up validators replay each round under its
   own block. `docs/ALLTIME_INIT_ROLLOUT.md` is the activation protocol.
5. **Trainer state.** The leaderboard accumulates from every signed bench
   report the engine sees, activation or not (shadow before the block; the
   pointer file tags `rule`), and a one-time deploy backfill
   (`_backfill_leaderboard`) walks every published report in the receipt
   index so the population is all-time from the first boundary. Weights are
   re-applied on reload, so a `chain.toml` weight change re-ranks the
   persisted board rather than freezing stale ranks.

## What this supersedes

- DEC-CA-0017 explicitly REJECTED a global all-time top-k pool (winner's-curse
  freeze, Goodhart lock-in, reign-scoped envelope). The owner has now chosen
  exactly that population; the envelope objection is resolved by the
  consensus change above, and the winner's-curse / Goodhart risks are carried
  as this node's revisit conditions rather than blockers. The no-downgrade
  guard is implied under the rule — the board only ever improves — and stays
  explicit only for the legacy path.
- DEC-CA-0015's error-decorrelation selection is retired under the rule: the
  set is a pure score ranking (the diversity now comes from the population
  spanning reigns and generators). The vector cache and `select_members`
  stay for the pre-activation path.
- DEC-CA-0013's "promote after a ripe reign" pacing is replaced by the notice
  period under the rule; the propose-and-verify split, the single manifest
  pin, rotation, and DEC-CA-0004's persisting king are untouched.

## Not changed

Receipts, manifests, the bench-report wire format, `contract_digest`, KOTH
verdicts, margins. The `PromotionRecord` schema is unchanged (record_version
stays 1). `promotions/leaderboard.json` and `warm_start.upcoming` are unsigned
and presentational like the other status docs.
