---
id: DEC-CA-0036
type: decision
title: "Miner-funded challenger legs under elastic-cadence rounds; king and confirmation legs stay operator-funded"
status: proposed
date: 2026-08-28
tags: [funding, provisioner, round, cadence, intake, security, cost]
revisit_when: "the funded queue routinely overflows max cadence × finalist_cap for days (the k-cap or the grid is the wrong bottleneck); or shadow mode shows most of the field funding but the screen still deciding outcomes (retire the heat formally); or scratch-shadow retrain-noise measurements put same-generator retrain variance above the decayed margin floor (the confirmation gate needs the un-decayed margin, or the floor moves); or Lium reliably lacks the pinned final SKU so funded entries live in capacity-requeue purgatory (multi-provider BYOK or per-size pins, DEC-CA-0027)"
relations: {builds_on: [DEC-CA-0003, DEC-CA-0008, DEC-CA-0012, DEC-CA-0016], external: "PRISM (subnet 100), tensorlink-dev/base: prism-lium-payer / lium-rent-pool / prism-challenge"}
---

Challenger training legs are billed to the **submitting miner's own Lium API
key**; the operator keeps every other kind of control — the rental, the pinned
image and SKU, the orchestration, the eval. Costs scale with the field and
land on the miners creating them. The pattern is PRISM's (subnet 100): miner
sends `X-Lium-Api-Key` with an authenticated request, operator re-executes on
a pod rented with that key, fail-closed. Cascade keeps its ROUND structure —
the round model is what makes duels paired (one king leg, one draw, one
cohort) — and gets demand-responsiveness from **elastic cadence** instead of
per-submission pipelines.

## The shape

1. **Funding channel.** One new HTTP surface (`cascade-intake`): a
   hotkey-signed POST binding the miner's Lium key to their *revealed* commit
   ref. Keys live in a memory-first, TTL-sealed vault (0600 files so a
   restart can still tear down; never in any shared store, never logged —
   the queue, standings, and manifests carry hotkeys and refs only). An API
   key can never ride the chain commit: a reveal is public forever.
2. **Funded queue = the demand signal.** Entries order by reveal block
   (earliest-commit seniority, per [[DEC-CA-0008]] and the UID invariant),
   one live entry per hotkey. `[round] funded_mode = "required"` makes the
   queue the round's field: at most `finalist_cap` funded entries enter,
   which trips the heat's fits-the-cap fast path — **no GPU screen runs;
   every funded entrant duels**. The screen thereby retires operationally
   without touching consensus or [[DEC-CA-0012]]'s statistics: per-round k
   stays capped where the α/k correction was sized, and demand raises the
   NUMBER of rounds, never k.
3. **Elastic cadence rides existing machinery.** Validators never schedule
   rounds — they poll `read_latest_manifest` and score what appears — so
   scale-DOWN is pure trainer policy (`skip_unfunded_rounds`: an unfunded
   boundary runs nothing; the king holds, consensus-invisibly). Scale-UP is
   the one consensus-touching move: set the epoch grid to the maximum
   cadence via the existing scheduled `epoch_blocks` switch
   (`epoch_blocks_prev` / `epoch_activation_block` — built for exactly
   this), then skip empty boundaries. 1–4 rounds/day = a 6h grid plus the
   floor.
4. **Per-payer provisioning.** One `LiumProvider` per payer key (PRISM's
   per-submission backend): the key enters only the rent subprocess's env,
   never argv or our environ; pods are ledgered with `payer_hotkey` because
   teardown and orphan-reap on a miner's account REQUIRE the miner's key
   (the operator's key cannot even see those pods). Rent failures classify
   auth / rate-limited / no-capacity / infra; **an infra fault never burns
   an entry** — it requeues on the miner's kept key (bounded attempts);
   sold-out waits unboundedly; auth is the miner's to fix. A rent is never
   auto-retried in-place: each attempt spends the miner's budget.
5. **The king's leg (and the reference) stay operator-funded, every round.**
   Owner decision 2026-08-28. And on a provisional dethrone, the operator
   **retrains the winner's generator on the operator's own account** — same
   init, contract, seeds — and the crown stands only if the operator-produced
   checkpoint clears the margin again. A Lium account owner can always
   console into their own pod, so nothing trained on a miner-funded pod is
   ever the artifact consensus anchors on; the miner-funded leg is a paid
   screen, the confirmation leg is the binding one. Operator GPU cost is
   O(rounds + dethrones), not O(field).

## Consequences owned

- **No shared operator credentials on miner-funded pods** (HIPPIUS_*,
  HF_TOKEN, WANDB_*): stdin-piped env is readable by root on the box, and
  the payer is root-adjacent by account ownership. Funded legs get scoped
  per-pod credentials or push artifacts through the orchestrator.
- **Tenure decay ([[DEC-CA-0016]]) must re-denominate to wall-clock/blocks**
  before "required" arms: with elastic cadence, tenure-in-rounds is
  demand-coupled — an attacker could fund junk entries to raise cadence and
  grind the margin to its floor on demand.
- **No balance pre-check** (PRISM's choice too): a key is validated by use;
  underfunded surfaces as a classified rent failure that requeues.
- **Dedup stays pre-duel.** Anyone can now pay in, so [[DEC-CA-0008]]'s
  content screen matters more, not less.
- **Registration economics can relax**: the compute funding is the per-entry
  cost, so `one_submission_per_hotkey`'s lifetime bar can move toward
  PRISM's one-live-entry-per-hotkey (the queue already enforces that shape).

## Built (this node's landing, all inert at shipped defaults)

`cascade/funding/` (vault, fault taxonomy, queue, intake + `cascade-intake`),
per-payer `LiumProvider` + `payer_hotkey` ledger attribution + funded
rent/teardown/reconcile (`cascade/provision/funded.py`), `[round]
funded_mode/funded_queue_path/skip_unfunded_rounds/max_rounds_per_day`
(digest-inert; `"off"` shipped), the trainer's funded field filter +
settle marking + boundary skip, and `cascade fund` / `fund --withdraw`
miner commands. Arming order: intake deployed → shadow on testnet →
required on testnet (with the grid change) → mainnet release-then-activate.
NOT built here: scoped pod credentials, the confirmation-leg wiring, the
tenure re-denomination, and the retrain-noise measurement — each gates
arming, none gates landing.
