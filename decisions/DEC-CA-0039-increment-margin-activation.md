---
id: DEC-CA-0039
type: decision
title: "Arm the increment margin, block-gated: a maturing lineage's dethrone bar prices a fraction of the per-round increment, not the king's absolute score"
status: accepted
date: 2026-09-06
tags: [scoring, koth, margin, warm-start, consensus]
revisit_when: "dethrones become churny (a king changes hands on sub-noise increments — raise margin_increment_floor); or the increment unit collapses toward the floor for many consecutive rounds (the lineage has genuinely converged — that is a basin-escape / reseed question, DEC-CA-0014, not a margin one); or a random-init reseed round is audited and the level fallback is shown to mis-replay"
relations: {arms: DEC-CA-0027, related: [DEC-CA-0016, DEC-CA-0034, DEC-CA-0038, DEC-CA-0014]}
---

Under `margin_mode = "level"` the dethrone bar is a fixed fraction of the
king's ABSOLUTE geomean (`win_margin_start` 1% → `win_margin_end` 0.5% floor
over 8 rounds of tenure). As the warm-start lineage matures (gen 5+,
~150k steps/generation), per-round improvements shrink toward the noise band,
and once a real gain is below the 0.5% floor the king becomes structurally
undethroneable however genuine the edge — the "signal shrinks as training
persists" failure.

Evidence (receipt replay on the recent mainnet window, blocks 8942400–9007200,
init scored every round under `init_gate_mode = "shadow"`): of 17 competitive
rounds, 2 flip from HELD to DETHRONE under increment — e.g. block 9000000,
level LCB 0.0046 held by 0.0004 against the 0.5% floor, the same edge = 0.46
of the round's actual increment. Crucially, every round with a genuinely worse
challenger (negative level LCB) also held under increment — it opened no false
dethrones; it flipped only real-but-small edges the fixed floor blocked.

DECISION: arm the increment margin (built as DEC-CA-0027, `margin_mode =
"increment"`; the bar prices a fraction of the floored mean improvement over
the shared init, so it tracks the shrinking increment). BLOCK-GATED via
`[scoring] increment_from_block` (release-then-activate, the
`margin_activation_block` shape): `koth_params(block)` resolves the mode from
the round's epoch block through `effective_margin_mode`, so the validator and
the audit derive the same rule per round and `cascade-audit` replays each
round under its own — a random-init reseed round with no baseline still falls
back to level in `evaluate_round` and replays correctly (the DEC-CA-0034
"params AND row presence" machinery). `margin_mode` stays `"level"` as the
base; `0` = no scheduled flip. No receipt change (mode already rides
`VerdictRecord.params`, drop-when-default) — verified: every pre-gate mainnet
receipt still audits clean under the armed config.

ARMED at mainnet `increment_from_block = 9046800` (Fri 2026-09-11 20:30 UTC, the
evening funded round) — the SAME block as the cohort max-T (DEC-CA-0038) and the
funded go-live, so validators get one coordinated upgrade window for all of them. The two corrections are complementary and stack: increment fixes the
shrinking-signal (level vs increment), max-T fixes the large-field multiplicity
penalty (α/k vs the real correlation). Testnet armed at 1.

NOT a substitute for basin escape (DEC-CA-0014): increment keeps INCREMENTAL
competition alive as the lineage converges; if the lineage has genuinely
plateaued (from-scratch benches ≈ the mature king), the answer is a reseed,
which needs the scratch-shadow diagnostic armed first — a separate track.
