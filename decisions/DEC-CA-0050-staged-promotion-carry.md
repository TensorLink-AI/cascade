---
id: DEC-CA-0050
type: decision
title: "Staged-promotion carry: a dethrone between staging and effective_era keeps the staged promotion, and a prior-reign record is never timed on the new king's clock — decided on chain (DEC-CA-0045), typed block = owner override"
status: proposed
date: 2026-10-14
tags: [cascade, validator, consensus, promotion, activation, incident]
revisit_when: "the first promotion staged after lock-in survives a dethrone end to end on mainnet (staged → dethrone → installed at effective_era with no era_init_mismatch); once promotion-carry locks in, pin it too in the following release; any future feature must check the 128-byte note budget (pin decided features first)"
relations: {depends_on: [DEC-CA-0013, DEC-CA-0043, DEC-CA-0045], relates_to: [DEC-CA-0004, DEC-CA-0048, DEC-CA-0049]}
---

## Incident (2026-10-14, netuid 91)

```
04:05:49 cascade: promotion generation=14 staged for era 2577 (3 member(s))
16:13:57 cascade: reign clock reset for new king 5HCFGiKHeaVD at block 9275400 (observed)
19:07:08 WARNING cascade: promotion record gen=14 fired before the reign clock ripened (block 9271800); ignored
01:06:35 WARNING rejecting manifest round=…: era_init_mismatch: era 2577 declares generation 14 member 0, the ledger names generation 13 member 0
```

1. `crown()` rebuilt `CascadeState` on the dethrone carrying
   `generation`/`members` but not the DEC-CA-0043 `pending_*` fields, so the
   staged gen 14 was erased.
2. The era gate's refresh then re-verified the published record, judging
   ripeness at `fired_block` 9271800 against the NEW reign's anchor 9275400 —
   negative elapsed time, "not ripe", ignored. The trainer (correctly) kept
   gen 14, so every era-2577+ manifest fails `era_init_mismatch`, every era,
   until fixed.

## Decision

From `[scoring] promotion_carry_from_block` (judged per settlement / dethrone
from that round's own epoch boundary block):

* **Carry.** `crown(…, carry_pending=True)` keeps `pending_generation`,
  `pending_members`, `pending_effective_era`, and records the outgoing reign's
  anchor as `prev_reign_start_block` / `prev_clock_observed` (observed only
  when the outgoing clock was itself anchored and observed). Both new state
  fields are drop-when-default in `cascade_state.json`, so existing files
  and a never-carried state serialise byte-identically.
* **Prior-reign timing.** In the era gate's staging check
  (`_refresh_pending_promotion`), a record with `fired_block <
  reign_start_block` fired under an earlier reign. It is timed against the
  previous reign's anchor when `prev_reign_start_block <= fired_block` and
  `prev_clock_observed`; otherwise the timing check is SKIPPED (the existing
  non-attesting/bootstrap rule). When attested, member provenance is
  reign-scoped to that previous reign. In both cases the CURRENT reign's
  checkpoint log is left out of the member lookup and the quality floor —
  those checkpoints were never the record's candidates, and the new king's
  (often better) checkpoints would otherwise reject an honest promotion.
  Signature, `cascade_top_k`, `min_effective_era`, provenance and quality
  still apply. Records fired within the current reign are judged exactly as
  before.
* **Late re-stage.** No change needed: `_era_ledger` and
  `_install_era_promotion` already use `era.index >= pending_effective_era`,
  so a promotion re-staged after its effective era began is in force at, and
  installed by, the next handled settlement of any era at or past it.
* **Activation.** `promotion_carry_from_block = 0` = decided on chain: a
  fourth readiness-note segment `promotion-carry` (fixed name — the rule has
  no tunable values), present when `[activation]` is on and
  `cascade_enabled`. 51% of eligible stake locks in; the rule applies from
  the NEXT SETTLEMENT BOUNDARY after the lock-in boundary (`align =
  "boundary"`) — the rule is evaluated per round from the round's own block,
  so it needs no era alignment. Persisted in
  `activation_promotion_carry_state.json`; receipts stamp
  `promotion_carry_block` (drop-when-default); the audit's
  `promotion-carry` check verifies it against the notes and
  `apply_receipt_activation` replays it. A typed block is the owner override.
* **Before the gate everything is bit-identical** (the incident replays
  exactly in `tests/unit/test_promotion_carry.py`), so upgraded and
  non-upgraded validators agree until lock-in.

## The 128-byte note budget: pinned lock-ins

The plain commitment field is `Raw0..Raw128`. Mainnet's note before this
release, with the rollover, forfeiture and margin-v2 segments all locked in at
7-digit blocks, was 116 bytes; a `promotion-carry` segment adds 20 (unlocked)
to 32 (locked) bytes — 136–148, unwritable.

Owner decision 2026-10-14: the SAME release PINS the three decided lock-ins
in chain.toml (`[[activation.pinned]]`, one table per feature: `feature`,
`lock_block`, `activation_block` — rolling-era-king 9133200/9136800,
forfeit-42b1a486 9165600/9166500, margin-v2-d30eb443 9190800/9194400).
Typing the blocks into the feature keys was rejected: it zeroes the
`resolved_*` markers, so new receipts would lose `activation_block` /
`margin_v2_block` and an auditor on the old chain.toml could no longer replay
them. A pin instead:

* is armed AT LOAD through the existing `apply_activation` /
  `apply_forfeit_activation` / `apply_margin_v2_activation`
  (`apply_pinned_activations`, rollover first), so the effective config, the
  `resolved_*` markers, the receipt stamps and the audit replay are
  field-for-field what a validator that tallied the lock-in runs with
  (`test_pinned_config_is_field_for_field_the_tallied_lock_in`,
  `test_pinned_receipt_stamps_and_audit_match_the_tallied_lock_in`);
* is validated: the pair must be the one pair a lock-in could produce
  (`valid_pair`, feature alignment), a pin plus a typed block for the same
  feature is refused, a pin naming a feature the config no longer decides
  (edited forfeit list / v2 bar) is refused as stale, activation must be on;
* is never tallied again and never reads the chain (`resolve_activation`
  short-circuits; a persisted lock-in on the same pair is kept byte-for-byte,
  otherwise the record is the pin with `source="pinned"` — the source tag is
  in no signed byte);
* carries no note segment. The note becomes the plain primary head plus
  promotion-carry: `cascade-ready:1:rolling-era-king:0:0:promotion-carry:0:0`
  (56 bytes), `…:promotion-carry:<lock>:<act>` after lock-in (68 bytes at
  7-digit blocks, 70 at 8). The head's `0:0` is never tallied by a node that
  runs the pin;
* is accepted by the audit as the agreement (`activation` / `margin-v2` /
  `promotion-carry` checks PASS when the stamp equals the pin — the notes no
  longer carry these lock-ins).

`own_signal_payload` still drops the promotion-carry segment rather than fail
the write if a note would ever exceed 128 bytes (safety net).

**Old-release validators.** They parse the new notes (multi-segment parsing
predates this release) and keep running from their PERSISTED lock-ins (the
resolver returns before any tally once locked). An old-release node that LOSES
its activation state cannot catch up: the notes no longer name the three
lock-ins (it would re-tally the rollover from the `0:0` heads at a later
boundary, and never arm the forfeiture or the v2 bar). Operators must not wipe
activation state; upgrade instead. Old notes never count toward
promotion-carry (no such segment), so the 51% must come from upgraded stake.

## Trainer

No change. The trainer authors the record and already keeps the staged
generation across a dethrone; it never needed to agree on this rule. It does
not read this feature.

## Blast radius

CONSENSUS (netuid 91 has six external validators). Inert until 51% stake
signals (or the owner types a block); the flip lands at a boundary every
upgraded node derives identically. A validator that never upgrades keeps
rejecting gen-N+1 manifests after a dethrone-during-staging and diverges.

## Recovery of the 2026-10-14 wedge once active

Validator ledger gen 13, nothing staged, no previous anchor (the old crown
erased it), reign anchored 9275400 (observed), record gen 14 fired 9271800
for effective era 2577. At the first settlement at/after the activation
block, the era gate refreshes: 14 = 13 + 1; signed; members ≤ k; 2577 ≥
min_effective_era(9271800) = 2577; 9271800 < 9275400 ⇒ prior reign; no
previous anchor ⇒ timing skipped; members verified from their signed bench
reports (no reign scope, no current-log floor) ⇒ staged; the ledger for era
≥ 2577 names gen 14 ⇒ the manifest is accepted and gen 14 installed when the
settlement is handled. Settlements rejected before activation stay rejected
(already handled).
