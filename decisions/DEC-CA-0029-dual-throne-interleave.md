---
id: DEC-CA-0029
type: decision
title: "Dual-throne interleave: x% warm-start / y% random-init rounds, two kings with per-throne courts, geometric payout decay, and per-throne margin decay — Stage-3 design specified now, armed only on divergent-winners evidence"
status: proposed
date: 2026-08-24
tags: [cascade, koth, warm-start, scratch, thrones, emission, consensus, staging]
revisit_when: "DEC-CA-0014 Stage-1 shadows cover ≥2 promotion generations: arm this design only if scratch and warm rounds would repeatedly have crowned DIFFERENT generators (same winners ⇒ this node stays proposed and Stage 1/2 carry the from-scratch signal alone); after any testnet arming, revisit if blended-emission king resolution misidentifies either king for >1 round, if per-throne cadence dilution starves promotions (reign clock not firing within ~2× the single-throne wall-clock), or if either throne's emission share proves too thin to price its security margin"
relations: {depends_on: [DEC-CA-0014, DEC-CA-0013, DEC-CA-0016], relates_to: [DEC-CA-0004, DEC-CA-0012, DEC-CA-0019, DEC-CA-0027]}
---

[[DEC-CA-0014]] Stage 3 named the endgame — interleave random-init and
warm-start rounds as two separate thrones — and deliberately deferred it
behind shadow evidence. This node writes the design down NOW so the arming
decision is a config review, not a design pass, and so the costs are priced
before anyone is tempted by them. Nothing here changes the arming gate:
Stage-1 shadows must show the two regimes crown different generators.

**Round regime from the epoch grid, never negotiated.** Each round's regime
is a pure function of its boundary block: `round_index = boundary //
epoch_blocks`; the round is WARM iff `round_index % b < a` for the consensus
split `a/b` (`[scoring] throne_warm_num / throne_den`; 50/50 = 1/2 = parity).
Block-gated activation (`thrones_from_block`, the [[DEC-CA-0019]] precedent)
so audit replays each round's own rule and externals have one coordinated
boundary. Deterministic — not salted-hash — on purpose: miners see the next
round's regime and may tune their one committed generator for it; that is
the divergent-winners hypothesis made expressible, not a leak. One commit
serves both thrones ([[DEC-CA-0014]]); the same generator may hold both.

**Two ChampionStates, one machine.** `champion["warm"]` / `champion["scratch"]`
— each its own king, tenure, streaks, `former_kings` court. A round duels
ONLY its regime's king and folds into ONLY that throne's state; the other
throne's tenure does not advance (its king was not challenged). Per throne:

- **Margin decay ([[DEC-CA-0016]]) runs on per-throne tenure.** The affine
  2%→0.5%/8-round schedule is reused unchanged per throne — but 8 rounds of
  ONE throne is ~8 days wall-clock at 50/50 on 12h rounds, not 4. The floor
  carries the safety property, so the default is to accept the slower ramp,
  not to halve `margin_warmup_rounds`; re-tune only on replay evidence.
- **Geometric court decay per throne.** Within a throne the court pays
  `king_decay**i` exactly as today (king ∝ 1, prior kings ∝ 0.5, 0.25, …).
- **Warm throne keeps its whole stack:** [[DEC-CA-0013]] promotion (reign
  clock counts WARM rounds only), [[DEC-CA-0017]] no-downgrade guard,
  increment margin ([[DEC-CA-0027]]) when armed. **Scratch throne is the
  from-scratch contract:** no `warm_start_ckpt` (the "random-init forbidden
  once a promotion is live" gate becomes throne-conditional), judged at
  `margin_mode = "level"` — `cascade/eval/koth.py` already carries this
  split (a baseline-less round is judged at level by construction).

**Emission: one blended vector every round, never alternating.** Each round
sets `z · decayed_court(warm) + (1−z) · decayed_court(scratch)` with
`z = [scoring] throne_emission_warm` (start z = a/b). Alternating
winner-take-all per round would be 50/50 only in expectation and would make
on-chain incentive oscillate — and incentive is how the trainer falls back
to identifying the king. Which surfaces the one consensus-critical wrinkle:

**King resolution must stop meaning "highest-incentive UID".** At z = 0.5
the two kings tie by construction. The receipt-trail clock ([[DEC-CA-0013]]
hardening) becomes the PRIMARY resolution for both thrones; receipts gain a
`throne` tag (drop-when-default, so archived signatures survive — the
[[DEC-CA-0012]] convention); the trainer's incentive fallback becomes
throne-aware (highest-incentive within the throne's court) or is dropped.
`_king_uid_to_vote`, king-resync, and `demote_to_trained` all grow the
throne dimension. This is the largest single piece of work in the node.

**Cross-throne coupling is the payoff, not a leak.** The scratch throne's
reign checkpoints enter the WARM throne's promotion candidate pool through
the existing `cascade_quality_epsilon` floor — [[DEC-CA-0014]] Stage 2's
reseed valve stops being an every-M shadow and becomes continuously fed by
a live competition. The moment the lineage stops compounding, the scratch
king's checkpoint clears the floor and reseeds the shared init. Promotion
still pays its owner nothing ([[DEC-CA-0013]]).

**Priced costs (unchanged from DEC-CA-0014's warning, now itemised):**
(1) every rounds-denominated knob is per-throne diluted — `cascade_reign_rounds`
5 becomes ~5 days, `margin_warmup_rounds` 8 becomes ~8 days,
`king_resync_max_rounds`, `dethrone_cp` streaks, `scratch_shadow_every_rounds`
all stretch by 1/x or 1/y in wall-clock; each gets an explicit per-throne
value at arming, never silent inheritance. (2) The emission split divides
each throne's security margin — at 50/50 the cost of buying either throne
halves; z below ~0.3 for either side is rejected out of hand. (3) Receipts,
audit, heat standings, and the website all grow a throne dimension. GPU cost
is UNCHANGED (same rounds, same two lanes per round) — the machinery cost is
consensus surface and state, not compute.

**Staging.** Testnet first at 50/50 (max statistical power for the
divergent-winners question the interleave itself keeps answering); the
mainnet split is re-decided on measurement — x is a knob, not a commitment.
Mainnet arming is a coordinated fleet upgrade (throne schedule, blended
emission, and throne-tagged verdicts are all fleet-consensus; a mixed fleet
forks kings the way a mixed margin forks verdicts, docs/MARGIN_DECAY_ROLLOUT.md
pattern). If Stage-1 shadows show the SAME winners in both regimes, this
node dies and the from-scratch signal stays with Stages 1–2 — that outcome
is cheaper and strictly fine.
