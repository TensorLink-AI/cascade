---
id: DEC-CA-0058
type: decision
title: "One owner main validator scores; satellite validators replay its signed receipts (verdict, ladder, lanes, weights) from the recorded rows and set the vector that replay produces — no model, no pool, no torch"
status: proposed
date: 2026-10-19
tags: [consensus, validator, operations, audit]
revisit_when: "a satellite replay ever disagrees with the main validator's receipt (investigate before anything else); or the main validator's silence burns an era (add a standby main); or fabricated score rows become a live concern (add a seeded CPU spot-check to satellites)"
relations: {depends_on: [DEC-CA-0057, DEC-CA-0052, DEC-CA-0043], relates_to: [DEC-CA-0055, DEC-CA-0056]}
---

## Decision

Owner (2026-10-19):

1. **Main validator.** The owner's validator runs the full path (gates,
   CPU scoring, verdicts, lanes A/B/C, king check, emission) and publishes
   its signed receipt per settlement, as every validator does today. Its
   hotkey is also the DEC-CA-0057 config signer (`[netconfig] owner_hotkey`).
2. **Satellites replay, never evaluate.** A satellite reads the main
   validator's receipt for each settlement, oldest first, and checks:
   - the main validator's signature on it;
   - the trainer's signature on the manifest it names (by digest);
   - continuity — its prior state (king, tenure, ladder, court, lane slots) is
     the previous receipt's posterior;
   - the verdict, cohort correction, bench gate, king check, ladder, lane-B/C
     payouts and the weight vector, re-derived from the recorded rows with the
     same code `cascade-audit` uses (`check_verdict`, `check_weights` and the
     emission/ladder checks).

   All checks pass ⇒ it sets the vector it derived. Pure numpy over the
   receipt; no checkpoint, no eval pool, no torch.
3. **What it does not catch.** Fabricated score rows. Satellites (and tier-0
   `cascade-audit`) only check signatures and sha256 pins. Re-scoring is a
   separate, report-only audit, never consensus: `cascade-audit direct-moe
   --reproduce` for the direct-MoE lane (DEC-CA-0052) and, for the lane-A
   bench rows of a validator-sourced settlement, `cascade-audit bench <id>
   --reproduce` (DEC-CA-0063: re-sweeps sampled items from the public
   checkpoints, within a documented tolerance).
4. **Disagreement.** Any failed check ⇒ the satellite burns (endorses
   nobody, stays active, like `--force-burn`) and logs the failing check
   until a receipt replays cleanly.
5. **Main down.** The satellite keeps re-asserting the last verified vector
   so it stays active; after `stale_after_blocks` (default one era) with no
   new receipt it burns until the main validator publishes again.
6. **Install.** Satellites install without the `train` extra; the satellite
   entry point must not import torch.

## Shipped

Shipped inert: a satellite needs `[netconfig] owner_hotkey`, which is empty on
mainnet and testnet. The main validator's receipt stamps are also written only
once that key is set, so every receipt until then keeps its exact bytes.

- `cascade-satellite` (`cascade/validator/satellite.py`), also reachable as
  `cascade-validator --satellite`. Install with `pip install -e '.[satellite]'`
  (`[hippius,chain]`, no `[train]`). A subprocess test runs one full replay
  with `torch` blocked.
- Replay per receipt, oldest first. The satellite checks the main signature,
  then the trainer manifest: signature, published-text sha256 and equality
  with the embedded copy. Next comes continuity: the trainer's hash link, the
  prior/posterior state digests, and `check_lane_c_chain`. Then it runs
  `cascade.audit.main.audit_receipt` tier 0 under the receipt's DEC-CA-0045
  blocks and owner-signed overlay. That includes the bench gate replayed from
  the trainer-signed report — or, on a settlement under `[v2] bench_source =
  "validator"` (DEC-CA-0063), from the MAIN validator's signed report the
  receipt binds (`bench-source`, `bench-coverage`) — and the DEC-CA-0055 king
  check re-derived by `checks.check_king_check` rather than trusted: due-ness
  from the recorded reign clock, the pool half replayed on the recorded rows,
  and the bench half against the previous settlement's trainer-signed report
  (under the validator source: against THIS settlement's validator report,
  under the bench-gate rule in force — DEC-CA-0061). Last, it
  rebuilds the vector with
  `checks.derive_weights` and compares it uid for uid. At each push it checks
  reward UIDs against its own metagraph.
- Satellite rules:
  - **Burn on failure.** Any failed check burns the satellite, which stays
    active and logs the check at every push. A failed receipt whose main
    signature verifies still becomes the continuity anchor, so one clean
    receipt recovers.
  - **Retry when a check cannot run.** Examples: storage down, the bench
    report or overlay unreadable, no verifier. The receipt is retried and the
    standing vector is kept, so the stale clock still runs.
  - **Hold, then burn, when main is silent.** The satellite re-asserts the
    last verified vector until `[satellite] stale_after_blocks` (3600) past
    its boundary, then burns.
- Readiness notes: the satellite inherits `ActivationMixin` unchanged. It
  writes the same DEC-CA-0045 note, including the DEC-CA-0057 `cfg-`
  segment, from its own records.
- Owner decisions applied: satellites replay the main validator's rows and
  never evaluate. Hold then burn after `stale_after_blocks`. The main
  validator is `[netconfig] owner_hotkey`.

## Interfaces

- **Receipt fields** (`RoundReceipt`, drop-when-default, written by the full
  validator only while `[netconfig] owner_hotkey` is set):
  - `manifest_sha256`: sha256 of the exact published
    `manifests/round-<id>.json` text (the validator's
    `last_handled_manifest_sha`).
  - `state_prior_sha256`: `state_digest` of the settlement state when the
    settlement started (`ReceiptMixin._capture_settlement_prior`, called
    right after the pre-round snapshot in `_handle_manifest_inner`).
  - `state_post`: that state when the receipt is built.
- **`cascade/shared/settlement_state.py`**:
  - `settlement_state(champion, lane_b=, lane_c=)` returns `{king,
    king_since_block, tenure_rounds, former_kings, ladder{bp, court_hotkey,
    court_until_block, slashed}, lane_b{domain: [hotkey, uid, merge_block,
    round_id]}, lane_c{slots, pending}}`. The lane keys are present only when
    the lane is configured.
  - `state_digest(state)` and `paid_hotkeys(state)`.
- **King-check evidence** (DEC-CA-0055 receipts only, so a receipt with no
  king check keeps its bytes):
  - `RoundReceipt.king_check_scores` (drop-when-empty): the pool rows the
    check scored itself and the receipt does not already carry, role `king`
    (no duel at the settlement) and/or `init` (the verdict scored no
    `baseline` rows).
  - New keys in the `king_check` record: `tenure_rounds`, and `bench`
    (`{king, init}` suite means plus `init_pointer`).
  - `emission.ladder.reign_start_block`: the reign clock `v2_check_due`
    read, present when the validator's cascade view holds the champion.
- **`cascade.audit.checks`**:
  - `derive_weights(receipt, cfg) -> (vector | None, why)` (new).
  - `check_king_check(receipt, cfg, bench=None)`, run by `run_tier0`
    (`king_bench=`). Its rows are `king-check`, plus `king-check-bench`
    (a WARN, in `UNVERIFIABLE_WARN`) when the signed report could not be
    checked. `cascade.audit.main.fetch_king_check_report` reads that report.
  - `_lane_vector_want` (extracted from `_rebuild_lane_vector`; the check's
    behaviour is unchanged).
- **`[satellite]`** (`cascade/shared/config/satellite.py`, read from
  `cfg.raw`; the main loader is untouched): `stale_after_blocks = 3600`,
  `poll_seconds = 60`, `max_walk = 64`.
- **Module API** (`cascade.validator.satellite`):
  - `replay_receipt(receipt, cfg, reader=, prev=) -> Replay`, which raises
    `Transient` when a check cannot run.
  - `check_reward_uids`, `fit_vector`, `pin_main_validator`.
  - `SatelliteRunner` (`sync` / `desired` / `push` / `poll` / `status`) and
    `build_satellite`.
  - The check names are `main-signature`, `manifest-signature`,
    `manifest-fetch`, `manifest-digest`, `netconfig`, `continuity`,
    `lane_c_chain`, every audit row name, `weights-derive`, `reward-uid` and
    `replay`.
- **State**: `<state-dir>/satellite_state.json` (anchor, last verified
  vector and its attribution, failing check), `activation*_state.json`,
  `netconfig_state.json`, and the status file
  `<state-dir>/status/satellite-<hotkey>.json` (`mode`,
  `last_verified_round`, `last_failing_check`).
- **Known limits**:
  - **Main restarted mid-settlement.** If the main validator restarts in the
    middle of a settlement that folds a lane-B win, the replay starts from
    the post-fold lane-B state. That one receipt's prior digest then
    mismatches and the satellite burns for one settlement. The lane-B fold is
    idempotent and is not part of the pre-round snapshot.
  - **Reign clock.** The recorded `reign_start_block` is the main
    validator's own value. The audit bounds it by the crowning block and
    the settlement block, but does not rebuild it across receipts (a
    restart at each pass/fail check or accepted promotion).
  - **Forfeiture between settlements.** A DEC-CA-0048 block-clock forfeiture
    between two settlements is accepted, logged, at the first settlement past
    the gate.

## Open

- Defaults: `stale_after_blocks = 3600` (one era).
