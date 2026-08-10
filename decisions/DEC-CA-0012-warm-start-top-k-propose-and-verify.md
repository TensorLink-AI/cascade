---
id: DEC-CA-0012
type: decision
title: "Warm-start promotes a top-k member set; the trainer selects, validators verify an envelope"
status: active
date: 2026-08-10
tags: [cascade, warm-start, promotion, trainer, consensus, diversity]
revisit_when: "one full testnet promotion cycle (promote a multi-member set → rotate → next generation selects across lineages) shows whether structural diversity picks genuinely distinct checkpoints — or per-window error-decorrelation shadow data argues for a behavioral selection policy; also revisit if any evidence appears that the miner whose generator data shaped a promoted init gains a subsequent-round edge"
relations: {}
---
Cascade warm-start promotion changes from "every validator re-derives one
winner and installs it" to **propose-and-verify over a top-k member set**.

WHY a set: promoting the single lowest-geomean checkpoint funnels the whole
field down one trajectory and Goodharts the six-number bench aggregate. A
k-member set trains k parallel lineages at ZERO extra GPU cost — every round
still trains from ONE shared init (the controlled-experiment invariant,
DEC-CA-0004, is untouched: the manifest carries a single `warm_start_ckpt`),
but successive rounds rotate across members, and the next generation's
selection prunes across lineages. Reign checkpoints are same-init same-step
siblings (~150k steps; each round trains FRESH from the shared init), so
depth accrues per GENERATION, not within a reign — rotation costs samples per
lineage, not model depth. Keep k small (2–3).

WHY the trainer selects (the consensus insight): the six numbers selection
runs on are produced and signed BY the trainer; fleet re-derivation only ever
re-checked the trainer's arithmetic on the trainer's own data. Selection
authority moves to the trainer (`cascade/trainer/promotion.py`), which fires a
signed `PromotionRecord` (`promotions/gen-<n>.json` + locator index,
`cascade/shared/promotion.py`, bench-report signing conventions). Validators
verify a small deterministic ENVELOPE instead (`_check_warm_start` /
`_verify_promotion`):

- **provenance** — every member has trainer-signed bench numbers (reign log,
  or fetched on demand from the member's `source_round` report; fails CLOSED);
- **quality floor** — every member within `cascade_quality_epsilon` of the
  best verifiable reign score (diversity is never bought with a worse init);
- **ripeness** — a new generation only after the block-anchored reign clock
  survived `cascade_reign_rounds` (enforced on the +1 transition when the
  validator's clock can attest; a bootstrap/catch-up clock skips, it only
  measures its own uptime);
- **set cap + monotonic generation** — ≤ `cascade_top_k` members; a replayed
  old record can never roll the live set back.

A declared init that is ANY live member passes — so rotation, and later
ADAPTIVE allocation (dropping a losing lineage mid-generation), are pure
trainer policy, not consensus. The selection policy itself (v1: quality gate →
geomean-best anchor → prefer different generator → max round spacing among
same-generator siblings) can evolve — eval-profile dispersion, per-window
error decorrelation — without fleet lockstep. `[scoring] cascade_top_k` and
`cascade_quality_epsilon` are the only new consensus values (fleet-identical,
like `cascade_reign_rounds`).

CANDIDATE POOL includes CHALLENGER checkpoints, not just the king's: the
trainer already benches both duel sides every round (BenchReport roles), and
different generators are genuinely different data distributions — the deepest
diversity available. The owner of a promoted checkpoint earns NOTHING from
promotion (explicit design decision): promotion moves the shared floor for
everyone at once, so there is no payout to steer and no promotion-seeking
incentive; the residual question (does the generator whose data shaped the
init gain a later edge?) is a revisit condition, not a blocker.

MIGRATION (mainnet armed 2026-08-05 on the old single-winner mechanism): both
sides grandfather the legacy `warm_start_init.json` single pointer as
generation 1 (trainer `TrainerPromotion.load`, validator
`_adopt_legacy_warm_start`), so an armed fleet upgrades without a
`warm_start_mismatch` round. The validator no longer writes the pointer file —
the trainer's engine owns it. Trainer + validators deploy together. Known
bounded gap: the grandfathered generation has no published record, so a
validator with NO local state joining during the grandfather window rejects
rounds until the first real promotion publishes (≤ one reign; the old
mechanism was worse — such a validator rejected until its OWN clock fired).

HARDENING (two post-review passes, same change): the trainer's reign clock
keys off the signed RECEIPT trail's verdict king (sticky across fetch blips;
falling back to on-chain incentive, which lags a dethrone 1-2 epochs and
would fire promotions every validator judges premature); a fired record
persists as `pending_record` until its publish is confirmed, and is flushed
again right before the manifest publishes (a store outage at fire time heals
within the round); the validator's pending-bench queue carries the reign
ANCHOR, not just the king, so a promotion's re-crown (same king) drops stale
entries; a DETHRONE round's checkpoints are recorded in NEITHER log (the
trainer wipes them at its re-crown — a validator logging them would hold a
stricter quality floor than the trainer selected against, wedging every
honest promotion; the floor's one-sidedness invariant requires validator log
to be a subset of the trainer pool); timing attestation is `clock_observed`
(anchor from a watched verdict/acceptance) rather than `generation >= 1`, so
a genesis validator attests even the FIRST promotion while adopted or
re-anchored clocks stay permissive; and an attesting validator pins fetched
member provenance to the current reign (`report.created_block >=
reign_start`), closing the any-historical-report loophole.

Kept from the prior design: the manifest's single per-round pin (no receipt
or audit change — cascade-audit still re-derives from `warm_start_ckpt`,
DEC-CA-0009's signed-trail constraint respected); random-init fallback still
forbidden once a promotion is live; king persists on promotion (DEC-CA-0004);
throne never vacated. What the fleet loses: attesting "the BEST checkpoint was
promoted" (that attestation was ceremonial — the trainer authored "best");
what it keeps is the part with teeth: quality floor, provenance, timing, and
within-round init sharing (structural, one manifest field). See
[[DEC-CA-0004]], [[DEC-CA-0005]], [[DEC-CA-0006]] (shadow-first precedent),
[[DEC-CA-0009]].
