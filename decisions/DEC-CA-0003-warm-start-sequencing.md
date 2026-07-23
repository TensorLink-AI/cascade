---
id: DEC-CA-0003
type: decision
title: "Cascade warm-start: revert on testnet, sequence reign-clock determinism before consumption"
status: active
date: 2026-07-21
tags: [cascade, warm-start, trainer, determinism, audit]
revisit_when: "block-anchored reign clock + manifest-derived reign log land and survive a full testnet cascade with no trainer/validator king divergence — then implement trainer consumption (Problem 1) and re-evaluate arming"
relations: {}
---
Cascade warm-start (`[scoring] cascade_enabled`) is HALF-BUILT: the validator
half works end-to-end (bench king each round → track reign → select best
checkpoint → write `warm_start_init_path` → vacate throne; verified live on
testnet 2026-07-20), but nothing in `cascade/trainer/` reads
`warm_start_init_path` — `toto2_trainer.py` is hardcoded to random init.
Enabling it today buys throne-rotation churn plus per-round king-bench GPU
cost with ZERO warm-start benefit, and the 07-20 live test surfaced a real
failure: stale wall-clock `cascade_state` fired immediately, and the trainer
(keyed to on-chain `highest_incentive_hotkey`) diverged from the validator's
champion for hours of rejected rounds. On mainnet 24h rounds that window is
proportionally longer.

DECISION — three parts, in order:
1. **Revert testnet now**: `cascade_enabled = false`. Repo
   `chain.testnet.toml` already says false (the 07-20 flip was made on the
   rsync'd L40S host, never committed) — the revert is purely operational:
   flip on-host + restart trainer AND validator. `[scoring]` keys are not in
   `contract_digest`, but both processes read the flag.
2. **Sequence Problem 2 before Problem 1.** First make the reign clock
   deterministic and the handoff synchronized: (a) block-anchor the reign
   clock (`reign_blocks` from the crowning block, replacing wall-clock
   `reign_start` in `cascade.py crown()`/`should_cascade()`) — every
   validator fires the same round and stale-state immediate-fire dies;
   (b) reconstruct the reign log from signed manifest history for
   downtime-robust winner selection; (c) key the trainer's vacate off the
   `warm_start_init` pointer / promotion boundary instead of on-chain
   incentive, killing the re-sync lag. Only THEN implement trainer
   consumption: fetch the content-addressed init, `load_state_dict` it
   (seam at `toto2_trainer.py:575`), record the init digest in the
   manifest/receipt and fold it into the round's reproducibility contract so
   cascade-audit re-derives from the pinned checkpoint (fetch failure must
   NOT silently fall back to random init — that breaks byte-exact audit).
3. **Mainnet stays unarmed** until both land and a full testnet cascade
   completes cleanly end-to-end (promotion → warm-started round → validator
   pin check → audit re-derivation from the pinned init).

Rationale for the sequencing: consumption without a deterministic reign clock
means fleet validators can disagree on WHICH init a round trains from —
warm-start must happen in lockstep before the trainer actually initializes
from it, or the audit contract and fairness (king+challenger sharing one
pointer) are both undermined. Testnet ops note: the king bench additionally
needs `bench_data` + benchmarks venv synced and the optional `time` extra
(skipped by plain `uv sync --frozen`) — fold into host setup before
re-arming. See [[DEC-CA-0002]] and [[NOTE-ca-operational-invariants]].
