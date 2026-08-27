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

- `[training]` edits change `contract_digest` → the VALIDATOR must restart
  too, or it rejects every manifest (`contract_digest_mismatch`). This is
  also why the [[DEC-CA-0001]] alternative needs a coordinated restart at a
  boundary. NARROWED by [[DEC-CA-0029]] once `[scoring]
  declared_contract_from_block` is armed: only the LOCKED terms
  (`base_arch_digest`, `arch_preset`, `expected_gpu`, per-size pins) still
  force a fleet restart — every other `[training]` edit is trainer-declared
  and ships alone. Until it is armed (it ships at 0), the rule above holds
  as written.
- A validator restart is for an ACTUAL SCORING CHANGE. `[scoring]`, `[eval]`,
  and the cascade envelope decide verdicts: two validators disagreeing there
  FORK, so those stay lockstep at an announced boundary, forever. Do not
  reach for the [[DEC-CA-0029]] declared path to dodge that window — it
  covers admission, not scoring.
- Pods are rsync'd trees, not git checkouts; `uv sync` needs `--all-extras`
  (torch lives behind the `train` extra).
- Never restart the provisioner inside its pre-boundary trigger window.
