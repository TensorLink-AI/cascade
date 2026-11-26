---
id: DEC-CA-0072
type: decision
title: "Post-gate kings keep their recorded crowning block as the tenure anchor ([scoring] crowning_anchor_from_block, release-then-activate)"
status: accepted
date: 2026-11-20
tags: [validator, consensus, tenure, audit]
revisit_when: "the first dethrone after activation — check the new king's emission.king_since_block equals the crowning settlement's boundary and the ladder reign start"
relations: {depends_on: [DEC-CA-0043, DEC-CA-0055, DEC-CA-0057], relates_to: [DEC-CA-0045]}
---

## Problem

DEC-CA-0043 re-anchors the reigning king ONCE at the first settlement past
`tenure_blocks_from_block`, imputing `king_since_block` from the
`tenure_rounds` counter, and marks it done in `tenure_anchor_gate`. A dethrone
(and a forfeit or demotion) builds a fresh `ChampionState`, which resets
`tenure_anchor_gate` to 0. So the re-anchor fired again at every NEW king's
first settlement, imputing from a counter of 0: the anchor became that
settlement's own block, overwriting the real crowning block the dethrone had
just recorded.

Mainnet: 5G6j crowned at boundary 9530100 (the ladder's `reign_start_block`);
recorded `king_since_block` 9531900. Every post-gate king's tenure, and so its
margin decay, counts one handled settlement late. All validators run the same
code, so they agree (no fork). The #469 receipt audit flags it ("reign start
… precedes the crowning"); receipt-replay has been red since the first king
check recorded under that audit (round 8191231874266632009).

## Decision

* From `[scoring] crowning_anchor_from_block` (0 = off), a king whose
  recorded `king_since_block` is at or after `tenure_blocks_from_block` keeps
  it: the validator marks `tenure_anchor_gate` and does not re-impute. A
  post-gate crowning block is written by the dethrone/forfeit/demotion path
  that every validator runs identically, so it is already a consensus value.
* Kings crowned before the gate (or with no recorded block) keep the
  DEC-CA-0043 counter re-anchor unchanged.
* Not retroactive: a king already re-anchored before activation keeps its
  stored anchor (its `tenure_anchor_gate` is set; nothing revisits it).
* Audit: "reign start precedes the crowning" FAILs only when the recorded
  crowning block is at or after the activation block. Before it, the offset
  is the known pre-fix behaviour, identical on every validator.

## Rollout

Release with the knob at 0 (no behaviour change), then activate via a
netconfig overlay (DEC-CA-0057) at a settlement boundary after the external
validators have upgraded. It only changes behaviour at a dethrone's first
settlement, so the activation block needs no era alignment.
