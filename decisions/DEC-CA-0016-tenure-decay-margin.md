---
id: DEC-CA-0016
type: decision
title: "The dethrone margin decays with the king's tenure: reuse the affine warmup schedule in reverse, pick the floor from receipt replay, ship release-then-activate"
status: proposed
date: 2026-08-15
tags: [cascade, koth, margin, tenure, consensus, release]
revisit_when: "The replay harness has run over the full mainnet receipt trail and the owner has picked (end, warmup) — flip status to active at sign-off; after activation, revisit when margin_warmup_rounds of live decayed verdicts exist: if decay-dethrones cluster on rounds the shadow diagnostics (win_rate, boot spread) call fragile, the floor is too low; if long holds still see zero flips and near-misses, the floor is too high; also revisit if a future decision makes tenure reset on warm-start promotion, which would silently defang the decay"
relations: {depends_on: DEC-CA-0012, relates_to: DEC-CA-0013}
---
A flat 2% margin makes an entrenched king cheaper to KEEP than to earn: every
challenger must beat the incumbent by the same bar the incumbent never has to
re-clear, and under warm-start ([[DEC-CA-0013]]) the whole field trains from
the king-shaped init, so the longer a hold, the more the eval landscape tilts
toward it. DECISION: make the margin DECAY with tenure — a fresh king defends
the full `win_margin_start`, an entrenched one progressively less, floored at
`win_margin_end` from `margin_warmup_rounds` of tenure on.

**No new mechanism.** `margin_for_tenure` (cascade/eval/koth.py:79) is already
an affine schedule from start to end over warmup, clamped after; nothing in it
assumes end ≥ start. Setting `end < start` IS the decay. Verified load-bearing
facts: `ChampionState.tenure_rounds` does not reset on a cascade warm-start
promotion — only on a real dethrone or king-resync (live proof: tenure=9 at
the r19 verdict while the reign clock had fired at 5) — so decay deepens
across a whole hold; receipts already record `king_tenure_rounds` and the full
`verdict.params`, so `cascade-audit` replays archived rounds unchanged and
there is NO receipt format change; the trainer never reads the margin.

**Evidence before schedule.** `scripts/replay_margin_decay.py` re-judges the
signed receipt trail under the candidate grid (start=0.02, end ∈ {0.005,
0.008, 0.010}, warmup ∈ {8, 12, 16}) from recorded tenure + recorded LCBs
(`margin_for_tenure` is pure; the bootstrap quantile is untouched by a margin
change, so recorded LCBs are exactly the right statistic). All flips are
loss→win by construction; the report also counts near-misses within 50bp of
each candidate margin, and is explicit that it is FIRST-ORDER (a flip would
have rewritten the subsequent throne history — counts measure how binding the
flat margin was, not an alternate timeline). The owner's working pick
(2026-08-15): end=0.010, warmup=12 — the grid's most conservative floor, half
the fresh-king bar, reached in ~6 days of 12h rounds; confirmed against the
replay report before the mainnet release.

**HARD GUARDRAIL: the floor stays > 0.** The margin is the improvement bar
above the LCB noise gate; at 0 any statistically nonzero improvement
dethrones, and `bootstrap_alpha` alone does not price repeated attempts
across rounds. The harness refuses `end <= 0` and the config review must too.

**Rollout is release-then-activate, never a local flip** (netuid 91 has 6
external validators; the margin is computed per-validator at verdict time).
During any config-mixed window, a round whose decided LCB lands in
`[margin_decayed, margin_flat)` splits the fleet's verdicts, and champion
state has NO automatic reconvergence — divergent kings mean divergent weight
vectors and a forked receipt trail until a both-branch dethrone or a manual
resync. Full protocol, restart-window timing, monitoring via per-validator
receipt comparison, and the rollback boundary (safe only until the first
genuine decay-dethrone) live in docs/MARGIN_DECAY_ROLLOUT.md. Testnet first:
chain.testnet.toml carries the candidate schedule; one full ramp (tenure past
warmup, margin walking down on receipts, promotion NOT resetting it, audit
replay clean) gates the mainnet release.

Interactions checked: cohort duel ([[DEC-CA-0012]]) corrects the QUANTILE
while the margin stays one flat bar per round, so decay lowers it identically
for every cohort member — no multiplicity interaction; warm-start promotion
and the reign clock run on independent counters (a dethrone resets both);
gift gate is AND-ed after the margin decision; inconclusive rounds still
advance tenure (a hold is lengthening even when a round makes no decision).
