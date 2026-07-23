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
attempt count: the heat window shrinks in real time, so late pods are worth
less than the trainer's local fallback. The round-level rent-once latch stays —
a failed round never retries within the round. The eval stage deliberately
does not escalate (one pod; local validator evals are a cheap fallback).

Same decision, config side: the heat ladder's floor is 2× pods — no 1× rungs.
A single-GPU pod pays the full bootstrap cost (rsync + `uv sync`) for one
lane, and a singles fleet burns the whole `max_pods` cap on flaky boxes.
