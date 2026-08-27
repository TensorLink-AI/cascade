---
id: DEC-CA-0034
type: decision
title: "Init-baseline: a null baseline for KOTH — heat shadow row + duel-side dethrone floor (gift-gate shaped), shipped off"
status: proposed
date: 2026-08-26
tags: [cascade, koth, verdict, baseline, warm-start, consensus, receipts, observability]
revisit_when: "Shadow logs from a few rounds exist (pick enforce tolerance from data, never from principle); or the DEC-CA-0033 recipe arms and entrants start beating the init (then strict enforce becomes meaningful); or a zero-train/min-tokens trainer guard lands (the floor does NOT catch a pristine-init upload — it scores EQUAL, not worse)"
relations: {depends_on: DEC-CA-0013, informs: DEC-CA-0017, motivated_by: DEC-CA-0033}
---
Measured 2026-08-26: the raw warm-start init outscored every trained entrant
on the post-mix private windows (init 0.208 vs entrants 0.231–0.252 in r42)
— and nothing could see it, because KOTH compares entrants to each other and
to the king, never to "do nothing". A challenger could take the throne while
being worse than the untouched init.

DECISION — two mechanisms, deliberately split (owner 2026-08-26: "we don't
want to filter everyone out of heat; the gate is the duel's"):

1. **Heat shadow row** (`[round] init_gate_mode = "off" | "shadow"`,
   trainer-only): the round's init is scored on the same heat slice as every
   entrant and published as `HeatResult.init_baseline` (drop-when-None; the
   standings JSON shape is frozen for pre-gate rounds). It NEVER filters the
   heat — heat checkpoints are short-budget models on a small slice, and a
   heat filter could empty a round on noise. There is no heat enforce mode.
2. **Duel floor** (`[scoring] init_gate_mode = "off" | "shadow" | "enforce"`
   + `init_gate_tolerance`): the validator scores the init on the verdict
   windows (the increment-margin baseline rows, reused) and, under enforce,
   a challenger whose observed geomean is worse than
   `init × (1 + tolerance)` cannot win the round. Gift-gate shape: it can
   only BLOCK a dethrone, never grant one; king retention untouched — the
   worst case of over-tight enforcement is "king retains", exactly an
   empty-field round. Consensus-relevant ([scoring] is fleet-consensus):
   release-then-activate across all six external validators, shadow first,
   never straight to enforce.

Receipt/audit safety: `VerdictRecord.init_baseline_geomean` /
`init_floor_passed` drop from the canonical body when None (archived
signatures survive); the recorded-params drop table gains the two keys at
their defaults. The audit derives the judged margin rule as "params say
increment AND baseline rows exist" — baseline rows alone no longer imply
the increment margin, since a level round with the floor on records them
too; the floor replays through the recorded `init_gate_mode` param.

Explicitly out of scope: the pristine-init exploit (a 0-step submission
scores EQUAL to the init and passes any worse-than floor) — that needs a
trainer-side zero-train guard (min tokens / weights-identity), tracked
separately. The baseline scored is the init checkpoint's SCORED face, the
same artifact form challengers are scored on (see the two-face note in
`cascade/eval/koth.py` once a finished-form mechanism arms).
