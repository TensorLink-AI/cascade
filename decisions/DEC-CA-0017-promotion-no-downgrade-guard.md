---
id: DEC-CA-0017
type: decision
title: "Promotion never ratchets the shared init downhill: a ripe reign with only worse-benching candidates holds the live generation"
status: active
date: 2026-08-15
tags: [cascade, warm-start, promotion, trainer, basin]
revisit_when: "a hold persists for a full extra reign-length or more (the guard is starving rotation diversity — consider firing on equal-within-epsilon rather than equal-or-better, or feeding held reigns into the DEC-CA-0014 shadow read as evidence the lineage has plateaued); or Stage-2 reseed valve arms (a scratch candidate clearing the quality floor should also clear this guard by construction — verify the two compose); or bench noise is measured large enough that a strictly-worse hold is mostly noise (then compare with a noise-sized epsilon)"
relations: {depends_on: DEC-CA-0013, relates_to: DEC-CA-0014}
---
`maybe_promote` used to fire UNCONDITIONALLY on a ripe reign clock: select the
best of this reign's benched candidates and install them as the next
generation — with no comparison against the generation the field is currently
training from. A reign whose rounds all benched WORSE than their own init (a
bad patch of rounds, an off era of data) would still promote, and every
subsequent round would warm-start from the worse checkpoint: the ratchet
could slip backwards, compounding exactly the basin problem [[DEC-CA-0014]]
exists to escape.

DECISION: a ripe clock says a promotion MAY fire, never that it must. If the
best candidate of the reign benches worse than the live generation's best
member (same `cascade_score` scale, lower better), the trainer HOLDS: the
live generation keeps training, the clock stays ripe, candidates keep
accumulating, and the promotion fires the first round a candidate at least
MATCHES the incumbent init's bench (equal is not a downgrade — it still
rotates fresh diversity in). Members without a finite recorded score (legacy
pointer adoptions carry `score = NaN`) cannot anchor the comparison and never
wedge a firing.

This is pure trainer policy under [[DEC-CA-0013]]'s propose-and-verify split —
declining to declare a new generation is always envelope-legal (validators
verify records when declared; the existing member set stays live) — so it
ships trainer-side with zero consensus impact.

What this deliberately is NOT: a global all-time top-k candidate pool
("promote the best checkpoint ever seen"). That was considered and rejected:
the validator envelope pins member provenance to the current reign (a
consensus release to change), the all-time maximum of a noisy bench is a
winner's-curse order statistic that would freeze the ratchet on a lucky-high
measurement, and a benchmark-overfit checkpoint — the very basin being
guarded against — would become immortal. Basin ESCAPE stays with
[[DEC-CA-0014]]'s staged path (shadow scratch control → reseed valve); this
guard only ensures promotion itself never walks INTO a worse basin.
