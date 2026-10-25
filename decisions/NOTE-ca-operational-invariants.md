---
id: NOTE-ca-operational-invariants
type: note
title: "Operational invariants (hard-learned)"
status: active
date: 2026-07-14
tags: [operations, deployment]
relations: {}
---
Hard-learned operational rules for running cascade; violating any of these
has bitten us before.

- Only the LOCKED `[training]` terms (`arch_preset`, `base_arch_digest`,
  `expected_gpu`, `train_seed_salt`, per-size pins) still force a
  coordinated validator restart: since [[DEC-CA-0036]]'s declared-contract
  gating (`[scoring] declared_contract_from_block`) every other `[training]`
  edit is trainer-declared and ships with the trainer + worker image.
- A validator restart is for an ACTUAL SCORING CHANGE. `[scoring]`, `[eval]`,
  and the cascade envelope decide verdicts: two validators disagreeing there
  FORK, so those change lockstep at an announced block or through the
  on-chain activation of [[DEC-CA-0045]]. Do not reach for the declared
  contract path to dodge that window — it covers admission, not scoring.
- Pods are rsync'd trees, not git checkouts; `uv sync` needs `--all-extras`
  (torch lives behind the `train` extra).
- The bittensor SDK pin ([[DEC-CA-0051]], `bittensor==11.3.0`) is
  consensus-adjacent: the reveal encoding, the activation tally's stake
  source and the u16 weight quantisation must stay bit-identical across the
  fleet. Never seal a miner reveal with `bittensor.timelock.encrypt` (raw
  UTF-8 — the miner is silently skipped every round); never read stake from
  `neurons.all` (a different `total_stake`).
- A UID is not a seniority claim: Bittensor recycles deregistered UIDs to new
  registrants. Anything deciding "who was here first" needs a block number
  (commit or reveal), never a UID.
- The orchestrator holds the private eval pool and the trainer's wallet.
  Never run generator code there without `sandbox_mode = "container"` or
  `sandbox_strict = true` — the subprocess sandbox shares their uid and
  filesystem, so netns is the only real boundary.
- `[v2] from_block` moves the lane-A contract AT the block (era start), not at
  the release: trainer + every validator must run the v2 release BEFORE it.
  Never type a block that is not an era boundary (the loader refuses it) or
  one already passed (eras between the block and the deploy would be judged
  under the wrong contract).
