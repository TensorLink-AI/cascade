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

4. a stage that rented NOTHING re-attempts the full pick→budget→rent
   pipeline every `rent_retry_cooldown_s` (15 min) for as long as the stage
   can still matter: the heat while one serial screening wave fits its
   remaining window (the fleet re-sizes to that window), the final while
   full duel hours + boot margin remain (`_maybe_retry_stages`);
5. the FINAL fleet is stage-phased (`final_rent_on = "heat_complete"` on
   mainnet): rented just-in-time at the trainer's `heat_complete.json`
   marker, sized off the marker's ACTUAL finalist list — unless the primary
   L40S rung probes scarce at the margin, in which case early rental is the
   exception that locks capacity (`_maybe_rent_final_jit`). The trainer
   re-reads hosts before the duel (`_reload_remote_hosts(require_stage=
   "final")`, waiting `--hosts-wait-seconds`), so JIT pods land inside its
   patience.

Escalation is bounded by wall clock (`escalate_deadline_s`, 30 min), not
attempt count. The bound is NOT because degrading is acceptable — the
orchestrator is CPU-only, so trainer-local training is effectively a lost
round, and a locally-trained final can never pass the validator's
`expected_gpu` pin regardless — but because the rent path blocks the service
loop: while it escalates, teardown/heartbeat/reaper ticks starve (the
starvation class behind the 2026-07-14 lost window). The between-attempt
waiting of rules 4-5 costs nothing: the loop keeps cycling and only the
attempts themselves block. The rent-once latch still guards `plan_fn` and
the 30s poll cadence. The eval stage deliberately does not escalate (one
pod; the validator's local CPU evals are genuinely viable, unlike
trainer-local training).

Same decision, config side: the heat ladder's floor is 2× pods — no 1× rungs.
A single-GPU pod pays the full bootstrap cost (rsync + `uv sync`) for one
lane, and a singles fleet burns the whole `max_pods` cap on flaky boxes.
