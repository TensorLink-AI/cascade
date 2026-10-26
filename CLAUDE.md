# Cascade coding instructions

Read [AGENTS.md](AGENTS.md) for repository and miner instructions.
For local A/B/C experiments, use [the agent walkthrough](docs/AGENT_MINING.md).
For cheaper generator iteration, use released smaller models with `cascade score`
or `cascade iterate`, then recheck promising ideas on the live model. The implemented 2M, 7M, 28M and 120M
[presets](deploy/presets/README.md) have unmeasured GPU performance; 56M remains unspecified;
the walkthrough explains configuration and checkpoint compatibility.
The [architecture](docs/ARCHITECTURE.md), [submission interface](docs/INTERFACE.md)
and [operator runbook](docs/OPERATOR.md) describe the implementation and operation.
Protocol rationale is retained in [decisions/](decisions/); historical decisions
are not evidence that a feature is active. Check effective configuration.
