# CLAUDE.md — cascade

Working notes for AI-assisted sessions on this repo. Keep entries short; each
records a DECISION and its revisit condition, not general documentation.
Decisions now live as graph nodes in `decisions/` — the node is canonical;
this file keeps a one-line pointer per decision so the summary stays
in-context.

## Design decisions

- **DEC-CA-0001** — Throughput policy: "wall is the law". `ref_throughput` (185k)
  is calibrated to a well-fed trainer, not the median miner pipeline (~80k);
  generator throughput is a compute multiplier and mass `deadline_hit`s are
  intentional — do NOT "fix" them by loosening the wall.
  (`decisions/DEC-CA-0001-throughput-wall-is-the-law.md`)
- **DEC-CA-0002** — Mainnet home is netuid 91 (decided 2026-07-14). `chain.toml`
  ships with the mainnet values baked in (netuid 91, L40S pin, worker-image
  digest, `pool_bucket`).
  (`decisions/DEC-CA-0002-mainnet-netuid-91.md`)
- **DEC-CA-0003** — Provisioner rules of escalation: an EMPTY stage walks the
  SKU ladder under a 30-min wall-clock deadline, a below-50% fleet gets one
  same-SKU top-up, failed stages retry on a 15-min cooldown while their
  window lasts, the final rents JIT at the heat_complete marker (mainnet),
  and the heat ladder's floor is 2× pods (no 1× singles).
  (`decisions/DEC-CA-0003-provisioner-rules-of-escalation.md`)

New decisions get the next `DEC-CA-####` node in `decisions/` plus a one-line
pointer here. Put the revisit condition in the node's `revisit_when:` key.

## Operational invariants (hard-learned)

Canonical node: `decisions/NOTE-ca-operational-invariants.md`.

- `[training]` edits change `contract_digest` → the VALIDATOR must restart too,
  or it rejects every manifest (`contract_digest_mismatch`).
- Pods are rsync'd trees, not git checkouts; `uv sync` needs `--all-extras`
  (torch lives behind the `train` extra).
- Never restart the provisioner inside its pre-boundary trigger window.

## TensorLink graph

This repo is a spoke of the company strategy graph (`TensorLink-AI/strategy`).
Node ID prefix for this repo: **CA**. Decisions live in `decisions/` as
`DEC-CA-####` nodes (frontmatter per `strategy/knowledge/schema.md`).
Cross-repo edges use namespaced targets, e.g. `ME:EV-0021`, `CO:OQ-C1`.
