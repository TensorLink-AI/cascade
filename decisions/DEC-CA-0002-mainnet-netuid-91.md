---
id: DEC-CA-0002
type: decision
title: "Mainnet home: Bittensor netuid 91"
status: active
date: 2026-07-14
tags: [chain, launch, operations]
revisit_when: "a mainnet re-home or netuid change is considered"
relations: {}
---
cascade runs on Bittensor mainnet netuid 91 (decided 2026-07-14). The shipped
`chain.toml` bakes in the mainnet values (`netuid = 91`, the L40S GPU pin, the
worker-image digest, `pool_bucket`); the remaining operator inputs are set at
deploy time (`trainer_hotkey`, `commit_floor_block`, `base_arch_digest`,
`ref_throughput_tokens_per_s`, Hippius `[storage]` credentials, `[eval]
window_pool`, and the gift-gate mode). Throughput policy for the contract is
[[DEC-CA-0001]].
