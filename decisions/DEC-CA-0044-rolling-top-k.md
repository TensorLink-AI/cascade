---
id: DEC-CA-0044
type: decision
title: "Rolling top-k promotion: live members compete with the reign's candidates for the same k slots; a candidate enters only by beating the worst live member, and a promotion fires only when membership changes"
status: accepted
date: 2026-09-22
tags: [cascade, warm-start, promotion, trainer, consensus, block-gate]
revisit_when: "a generation holds for several ripe reigns because no candidate beats the worst member (the lineage has plateaued — a DEC-CA-0014 reseed-valve question, not a reason to loosen admission); or bench noise is measured large enough that 'strictly better than the worst member' is mostly noise (then admit on a noise-sized margin); or cascade_top_k moves off 3"
relations: {supersedes: [DEC-CA-0017], builds_on: [DEC-CA-0013, DEC-CA-0015, DEC-CA-0004], related: [DEC-CA-0043, DEC-CA-0014]}
---
The warm-start member set is a ROLLING top-k (`cascade/trainer/promotion.py`).
On a ripe reign clock the live generation's members compete with the reign's
benched candidates for the same `cascade_top_k` (3) slots. A candidate is
admitted only when a slot is free or it benches STRICTLY better (lower
`cascade_score`) than the worst scored live member (`admit_candidates`);
members without a finite score hold no slot. The carried members plus the
admitted candidates are then re-selected under the unchanged envelope: every
member within `cascade_quality_epsilon` of the pool's best, the best
anchoring the set, the remaining slots by measured error decorrelation
([[DEC-CA-0015]], structural spacing as fallback). A much better entrant can
therefore push carried members off the frontier and shrink the generation.

A promotion fires only when the re-selected membership differs from the live
one; otherwise the generation holds, the clock stays ripe and candidates keep
accumulating. This subsumes the best-vs-best no-downgrade guard
([[DEC-CA-0017]]): a candidate that beats nobody enters nothing, so the shared
init never ratchets downhill, and unlike that guard a worse-than-best
candidate can still fill a free slot or replace the worst member. Carried
members keep their original `source_round` and score; the king persists
through a promotion ([[DEC-CA-0004]]) and the trainer still proposes while
validators verify the envelope ([[DEC-CA-0013]]).

Consensus gate: a carried member's `source_round` predates the reign it
carries into, so validators exempt members of the generation they already
accepted (live or staged) from the reign-scope provenance check; quality is
still re-checked for every member. Both the trainer's rolling selection and
the validator's exemption switch on at `[scoring] era_king_from_block` (the
[[DEC-CA-0043]] rollover); before it the trainer selects from the reign's
candidates alone under the [[DEC-CA-0017]] guard and validators scope every
member, so a validator on the previous release never rejects a record its
peers accept.
