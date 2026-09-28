---
id: DEC-CA-0048
type: decision
title: "King forfeiture + admission denylist: a listed hotkey leaves the throne and the court at a set block (consensus, release-then-activate); the trainer refuses its admissions and never trains it as king"
status: proposed
date: 2026-09-28
tags: [scoring, validator, trainer, consensus, governance]
revisit_when: "the first use — record the evidence standard applied and whether the successor rule (named successor; else most recent eligible former king, else vacant) produced the intended throne"
relations: {depends_on: [DEC-CA-0004, DEC-CA-0043, DEC-CA-0045], relates_to: [DEC-CA-0008, DEC-CA-0016]}
---

## Decision (mechanism only — arming is an owner governance act with a written evidence standard)

Two knobs, both inert by default:

* **`[scoring] forfeit_hotkeys = [...]`, `forfeit_from_block = <an era's FIRST settlement boundary>`,
  `forfeit_successor_hotkey = "<hotkey>"`** — CONSENSUS. The poll the chain reaches the block (block
  clock, `_forfeit_on_block`; no manifest needed — the weights move at the block itself, re-pushed as
  soon as `weights_rate_limit` allows) and again at any settlement whose boundary reaches it
  (`_apply_forfeiture`, idempotent), every validator strips the listed hotkeys from the throne and the court
  (`state.apply_forfeit`): a forfeited king abdicates to the NAMED successor (crowned fresh: tenure and
  streaks reset, `king_since_block` = the forfeiture boundary, era king pointer cleared so the
  successor's first king leg is adopted) — or, with no successor named, to the most recent former king
  that is not itself forfeited, else the throne is VACANT until the next duel decides one. Forfeited
  former kings leave the payout court; the forfeited king is never retired into it. The resync safety
  valve never re-adopts a forfeited trained king. `_reward_uids` filters the list defensively.
  **Trainer:** an era whose LAST settlement reaches the block trains the SUCCESSOR's king leg
  (`_forfeit_switch`, its revealed generator resolved as of the block, uid from the metagraph); an
  already-running era switches too (king leg restarts, the old king pod retired; the validators hold
  that era's settlements until the new leg lands); with no successor named
  no king leg is trained (vacant) until receipts name one. Set the block to a settlement boundary;
  a mid-era block holds that era's settlements (successor crowned, no duel) until the retrained king
  leg lands — a few hours, not an era. Release-then-activate: every
  external validator installs the release before the block, or weights fork on that boundary.
* **Decided on chain (owner 2026-09-28: "when 51% of validator stake rolls over like our last major
  update").** With the list (and successor) shipped and `forfeit_from_block = 0`, the block is resolved
  by the DEC-CA-0045 machinery: each validator's readiness note gains a second segment
  `forfeit-<sha256("<sorted hotkeys>|<successor>")[:8]>:<lock>:<act>` (one plain commitment per hotkey,
  so it rides in the same note — `cascade-ready:1:<f1>:<l1>:<a1>:<f2>:<l2>:<a2>`; a pre-segment parser
  reads the extended note as malformed and counts it as NOT signed, acceptable because forfeiture is
  itself a consensus change every validator installs; the note is byte-identical to today's while no
  forfeiture is configured). Same tally, threshold and one-way lock-in, its own record
  (`activation_forfeit_state.json`); the resolved rollover — the boundary right after the one where the
  count crossed — is written as-is into `forfeit_from_block` (`forfeit_block_for`; owner 2026-09-28: no
  era of notice, the changeover follows the validators' upgrades as closely as the grid allows)
  (`apply_forfeit_activation`; `activation.resolved_forfeit_block` marks it as resolved). A typed
  `forfeit_from_block` is the owner override. The trainer resolves the same tally (never signals) so it
  hands the king leg over at the block the validators apply. An edited list or successor is a new
  feature name = a fresh vote. Receipts do NOT yet stamp the forfeit block (audit replays a typed block
  only) — follow-up.
* **`[round] blocked_hotkeys = [...]`, `blocked_from_block`** — trainer-side, not consensus. Listed
  hotkeys are refused at rolling admission (`failed [blocked]`, NO burn — the fee is policy, not code),
  never rent a pod.

## Why the throne cannot be stripped from the trainer alone

DEC-CA-0004: the throne is never vacated; the king changes only by losing a duel, and validators hold
the king in their own state — a trainer manifest naming a different king is resynced back. So removing
a king is a validator rule by construction. This node gives it a block-gated shape identical to every
other consensus constant (DEC-CA-0016/0019/0038/0039).

## Not decided here

The evidence standard for listing a hotkey. Similarity is information, not proof (DEC-CA-0008,
2026-09-28 owner position); a forfeiture must cite what WAS established (precedence, non-publication,
structural identity, a route) and what was not.
