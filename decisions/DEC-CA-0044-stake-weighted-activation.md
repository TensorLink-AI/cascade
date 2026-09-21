---
id: DEC-CA-0044
type: decision
title: "Stake-weighted activation: validators signal readiness on chain, the DEC-CA-0043 rollover locks in at the first boundary where 51% of eligible validator stake has signalled and flips at the next boundary — no typed-in block"
status: proposed
date: 2026-09-21
tags: [consensus, block-gate, rollout, validators, stake, rolling, era-king]
revisit_when: "the owner validator's own share of permit-holding stake on netuid 91 is over the threshold (then 51% is one operator deciding — raise the threshold, exclude the owner hotkey from the denominator, or accept it as the typed-in block with extra steps); or a lock-in fires while under half the external validators are upgraded and the ones left behind hold weight for more than one round (raise the lock-in-to-rollover delay from one boundary to one era); or a second feature needs its own signal (the note format carries one feature — generalise to a list before then); or the public finney endpoint stops serving metagraph reads at a boundary block a few minutes old (the live-view fallback then decides more often than the as-of read)"
relations: {builds_on: [DEC-CA-0043, DEC-CA-0019, DEC-CA-0016], related: [DEC-CA-0038, DEC-CA-0039, DEC-CA-0036]}
---

## Decision

The DEC-CA-0043 rollover (rolling intake, the era king, tenure in blocks,
the 3600 → 900 grid switch, the increment-unit max-T) is no longer typed
into `chain.toml` as a block the fleet must hit. The validators decide it
on chain:

* **Signal.** A validator on a release carrying the feature posts ONE plain
  on-chain commitment from its hotkey at startup —
  `cascade-ready:1:rolling-era-king:0:0`. The plain store is unused by
  miners (they commit through the timelock reveal store), and the reserved
  prefix is dropped by every field builder.
* **Tally.** At every boundary of the grid in force, every node reads the
  metagraph and the plain commitments AS OF that boundary block and sums
  the stake of permit-holding validators (positive stake; optionally only
  those that set weights within `dormant_after_blocks`) that have posted
  the note. Same block, same state, same number on every node.
* **Lock-in at 51%.** The first boundary where the signed share reaches
  `[activation] threshold` (0.51) locks in. Lock-in is one-way: a node that
  has seen it persists the block (`activation_state.json`) and never
  re-evaluates, so stake drifting back under the line changes nothing.
* **Rollover = the next boundary.** The round in which the count crossed
  finishes on the old rules; the one after starts on the new. That is the
  clean restart point for the trainer: restart it at the lock-in boundary
  (nothing in flight) and it has the whole round to pre-train the first
  era king. A trainer restarted late is not a failure — the first era's
  king leg trains inside the era and settlements wait for it (DEC-CA-0043's
  existing behaviour).
* **Agreement.** A validator that has locked in rewrites its note as
  `cascade-ready:1:<feature>:<lock_block>:<rollover>`. A node that restarts,
  joins late, or whose endpoint cannot serve the boundary block adopts the
  block that validators holding the threshold of eligible stake all name.
  No archive node, no trainer, no receipts bucket in the decision path.
* **Apply.** The resolved block is written into every DEC-CA-0043 key of the
  loaded config exactly as the owner would have typed it, through the
  loader's own alignment rule (`check_rollover_alignment`) — a runtime
  rollover can never reach a state the loader would refuse. The typed-in
  keys always win over the resolved block: the owner override, and the
  hold-back.
* **Record.** The validator stamps the resolved block on every receipt from
  lock-in on (`activation_block`, drop-when-default, so archived signatures
  survive); `cascade-audit` replays each round under it and verifies the
  block against the validators' current notes (`activation` check).
* **Readers.** The trainer and the provisioner READ the decision (notes,
  then the boundary tally) and arm themselves; they never signal. The
  trainer checks every tick, so a trainer upgraded early arms live; the
  provisioner resolves at startup.

Mainnet ships `[activation] feature = "rolling-era-king"`, `threshold = 0.51`,
`epoch_blocks_after = 900`, every DEC-CA-0043 key at 0: the release IS the
rollout, and the block is whatever boundary the validators reach 51% at,
plus one. Testnet keeps its typed-in rollover (600), so the mechanism is
inert there until the keys are zeroed for a validation cycle.

## Why 51% and why the next boundary

51% is the owner's call: the smallest majority that cannot be outvoted by
the validators left behind. Two consequences are accepted and mitigated
rather than designed away: (1) the flip is fragile in stake terms, so
lock-in is one-way and taken at boundaries only; (2) up to 49% of stake
gets one round of warning, so the note, the log lines, the state file and
`status/chain.json` all say "locked in, rollover at R" from the moment it
is visible, and a validator on the old release at R falls out of consensus
exactly as it would under a typed-in block. A longer delay (one era) was
considered and rejected for now: the owner wants the switch at the next
round so the trainer restart is clean, and the extra 21 hours buy nothing
for a validator that did not upgrade in the days the release was out.

## Rejected

* **The trainer counts and writes the tally into the settlement manifest.**
  The trainer may still be on the old release when the count needs to
  happen (it is restarted at the lock-in boundary, not before), and it
  would make the owner's process the oracle for a validator decision.
  Validators count; the trainer follows.
* **Carrying the decision in receipts.** The receipts bucket is
  owner-written and last-writer-wins; a validator reading another's
  receipt to learn the block trusts the bucket, not the chain. Notes on
  chain are the carrier; receipts only record.
* **Activating at the lock-in boundary itself.** The round at that boundary
  was built and judged on the old rules; flipping mid-boundary means two
  rule sets for one round.
* **A per-feature `*_from_block` resolved independently.** Every
  DEC-CA-0043 key must name ONE block (the loader's rule); the resolver
  writes all of them from one lock-in.
