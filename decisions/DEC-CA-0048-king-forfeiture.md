---
id: DEC-CA-0048
type: decision
title: "King forfeiture + admission denylist: a listed hotkey leaves the throne and the court at a set block (consensus, release-then-activate); the trainer refuses its admissions and never trains it as king"
status: proposed
date: 2026-09-28
tags: [scoring, validator, trainer, consensus, governance]
revisit_when: "the first use — record the evidence standard applied and whether the successor rule (most recent eligible former king, else vacant) produced the intended throne"
relations: {depends_on: [DEC-CA-0004, DEC-CA-0043, DEC-CA-0045], relates_to: [DEC-CA-0008, DEC-CA-0016]}
---

## Decision (mechanism only — arming is an owner governance act with a written evidence standard)

Two knobs, both inert by default:

* **`[scoring] forfeit_hotkeys = [...]`, `forfeit_from_block = <settlement boundary>`** — CONSENSUS.
  At the first settlement whose boundary reaches the block, every validator strips the listed hotkeys
  from the throne and the court (`state.apply_forfeit`): a forfeited king abdicates to the most recent
  former king that is not itself forfeited (tenure and streaks reset), or the throne is VACANT until the
  next duel decides one; forfeited former kings leave the payout court. The manifest that still names
  the forfeited king reads as a stale-king manifest and takes the resync path, so no duel is judged
  against a forfeited king's checkpoint. `_reward_uids` filters the list defensively. The trainer follows
  the validators' receipts to the successor (`_receipt_king`), and refuses to launch a king leg for a
  forfeited hotkey. Release-then-activate: every external validator installs the release before the
  block, or weights fork on that boundary.
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
