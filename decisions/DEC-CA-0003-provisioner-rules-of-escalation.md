---
id: DEC-CA-0003
type: decision
title: "Provisioner rules of escalation: deadline-bounded ladder walking, 2x floor"
status: active
date: 2026-07-23
tags: [provisioner, operations, cost]
revisit_when: "heats regularly degrade to local training, or the trigger margin / heat window math changes materially"
relations: {}
---
When a heat/final rental fails, the provisioner escalates by three rules,
cheapest signal first (`ProvisionerLoop._rent_stage_escalating`):

1. a dud pod gets ONE same-rung replacement, its machine excluded;
2. a stage that comes up EMPTY (failed launch, or every pod + replacement a
   dud) re-enters the SKU ladder at the next (candidate × provider) rung —
   capacity re-probed at escalation time, each rung re-checked against the
   round budget;
3. a stage PARTIAL below `min_viable_fleet` (0.5) of its slot demand gets ONE
   same-candidate top-up batch — never a different SKU, preserving the
   stage-never-mixes-candidates fairness invariant.

Escalation is bounded by wall clock (`escalate_deadline_s`, 30 min), not
attempt count. The bound is NOT because degrading is acceptable — the
orchestrator is CPU-only, so trainer-local training is effectively a lost
round, and a locally-trained final can never pass the validator's
`expected_gpu` pin regardless — but because the rent path blocks the service
loop: while it escalates, teardown/heartbeat/reaper ticks starve (the
starvation class behind the 2026-07-14 lost window). The round-level
rent-once latch stays — a failed round never retries within the round. The
eval stage deliberately does not escalate (one pod; the validator's local
CPU evals are genuinely viable, unlike trainer-local training).

Same decision, config side: the heat ladder's floor is 2× pods — no 1× rungs.
A single-GPU pod pays the full bootstrap cost (rsync + `uv sync`) for one
lane, and a singles fleet burns the whole `max_pods` cap on flaky boxes.
