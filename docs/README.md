# Cascade docs: start here

These guides describe the live system. Where a guide and the effective
`chain.toml` (or `cascade status --json`, signed receipts) disagree, the
configuration and signed records win. Protocol rationale is in
[decisions/](../decisions/); superseded docs are in [archive/](archive/README.md).

## Miners

| Doc | Read it for |
|---|---|
| [MINER.md](MINER.md) | The full miner reference: install, write, verify, submit, fund, verdicts, rewards, every lane. |
| [MINER_LANE_A.md](MINER_LANE_A.md) | Lane A brief: synthetic-data generators for the king model. |
| [MINER_LANE_B.md](MINER_LANE_B.md) | Lane B brief: domain-expert generators. |
| [MINER_LANE_C_MARKETCAST.md](MINER_LANE_C_MARKETCAST.md) | Lane C brief: the MarketCast fine-tune competition. |
| [MARKETCAST.md](MARKETCAST.md) | MarketCast step by step: the recipe, gates and verdicts. |
| [MINER_LANE_D.md](MINER_LANE_D.md) | Lane D brief: open weights. |
| [LANE_D.md](LANE_D.md) | Lane D in full: forward-only windows, KING_D, the `cascade lane-d` CLI. |
| [INTERFACE.md](INTERFACE.md) | The generator contract (`generate()`, channels, groups, sandbox rules). |
| [MODEL.md](MODEL.md) | The model the generators train: Cascade MoE Base sizes and recipe. |
| [DEMO_LOCAL_ITERATE.md](DEMO_LOCAL_ITERATE.md) | An offline loop on the small 17.58M preset and a retired eval pool. |
| [AGENT_MINING.md](AGENT_MINING.md) | Mining with a coding agent: local work, JSON output, authorization. |

## Validators

| Doc | Read it for |
|---|---|
| [VALIDATOR.md](VALIDATOR.md) | Running a satellite or full-evaluator validator and setting weights. |
| [AUDIT.md](AUDIT.md) | `cascade-audit`: checking signatures, decisions and reward accounting. |

## Operator

| Doc | Read it for |
|---|---|
| [INCIDENT_RULES.md](INCIDENT_RULES.md) | The incident lint rules CI enforces, and how to add one after an incident. |
| [EVAL_POOL.md](EVAL_POOL.md) | Building the private eval pool with `cascade-pool`. |
| [LANE_C_COMPETITIONS.md](LANE_C_COMPETITIONS.md) | Adding a new lane C competition (code plus config). |
| [V2_DIRECT_MOE.md](V2_DIRECT_MOE.md), [V2_DIRECT_MOE_OPERATIONS.md](V2_DIRECT_MOE_OPERATIONS.md) | The separate direct-MoE protocol. Built but refused for live activation in this release. |

## Design

| Doc | Read it for |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the rolling A/B/C(/D) system is implemented, component by component. |
| [plans/LANE_D_PLAN.md](plans/LANE_D_PLAN.md) | The lane D build plan; the build tooling and tests read it. |
| [notes/S0A_DEPLOYED_EDITS.md](notes/S0A_DEPLOYED_EDITS.md) | Owner edits for the deployed `chain.toml` (parsed by `tests/accept/r6x`). |
