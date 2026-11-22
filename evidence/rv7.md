# RV7: lane D review of W5e, W5c, W5b, W7a, W7c, W-S2, W-S3 and the RV6 fixes

Covers: W5e, W5c, W5b, W7a, W7c, W-S2, W-S3, W18, WB18, WB11, WB13, W0c

Reviewer: RV7 (fresh context, read-only on main). Reviewed at main
`4ae3f41` (#649). In scope:

- W5e #645 (verdict, confirmation duel, gate wiring, fail closed);
- W5c #641 (house record + audit);
- W5b #639 (one snapshot per window);
- W7a #637 (schema) and #649 (audit re-derives the gate);
- W7c #630 (lost / rolled-back state);
- W-S2 #626 (lane B excludes the era sentinel: a CONSENSUS change);
- W-S3 #627 (sentinel board, non-gating);
- RV6 fixes: W18 #631, WB18 #633, WB11 #640, WB13 #643;
- merge tooling (`scripts/laned` freshness, #602).

The spec is docs/plans/LANE_D_PLAN.md (§1.3 verdict and D20, §1.3b gate,
§1.5 consensus and state, §3.0 rows, §3.2 invariants, §16, §17) plus each
WP's `tests/accept/<wp>/`. I read the code, not the PR bodies.

Known open items, not re-reported: W3b5 (being built); the validator
production call site (bench, block hash, sentinel, W3b3/W-S3 calls;
unassigned); the W3a owner probe; the W18 ledger producer (W17); the
payer-key TTL contract (#636); the WB13 413 limits as miner policy; the
synchronous bench sweep blocking weight reasserts.

Checks run, one directory at a time at `nice -n 19`, with
`env PYTHONPATH=<wt> python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -m 'not torch'`:

- w5e 30 passed; w7a 246; w7c 81; w5b 25; w5c 41; w_s2 23; w_s3 57;
  w18 98 passed, 1 skipped; wb18 28; wb11 131 passed, 2 skipped; wb13 77.
- `stub_scan.py` over the w5e, w7a, w7c, w5b, w5c, w_s2 and w_s3 `PATHS`:
  0 findings.
- CI on main: `66b039e` laned-accept is red only on #646 (W7a tests ahead
  of #649, expected). The `4ae3f41` runs were still in progress.
- Scratch reproductions against the worktree, no torch, for B1 and B2 below.
  They use the W5e fixtures (`tests/accept/w5e/helpers.py`).

## RV6 BLOCKING items: re-check

- **B1 W18 (abort writes no log): resolved by #631.** A `StorageFailure`
  stops the run only after the complete action log is written
  (`SweepAborted`). Unprocessed objects get `run_aborted` keeps.
- **B2 W18 (unreadable lock = no lock): resolved by #631.** Only
  `NoSuchObjectLockConfiguration` means unlocked. Any other answer is
  `lock_unknown` (a keep, no delete). Legal holds are read per version.
- **B3 W7c (lost state = genesis): resolved by #630.** `load_prior_state`
  refuses `state_missing` / `state_digest_mismatch` once a report is
  persisted. This makes B1 below a permanent wedge, not a one-boundary one.
- **B4 WB18 (deferred entry expires): resolved by #633.** Deferred funded
  hotkeys (`sentinel_target` and WB15 `rows_held`) are `touch`ed every tick.
- **B5 WB11 (benign repos refused): resolved by #640.** Low-information
  windows are in neither set and never counted at lookup. Still owed before
  the owner pin: the false-positive rate measured on real generators.
- **B6 WB13 (scan stall): resolved by #643.** It has a window budget, a 20 s
  deadline, archive caps, a (ref, pin) LRU cache and a typed 413. Residual:
  note 9.
- **B7 WB13 (packaging evasions): resolved by #643.** It decodes str/bytes
  constants, JSON strings, zip/`.npz`/`.npy`, all eight f64 offsets in both
  byte orders, and delimited text.
- **B8 WB11 (lookup size): partly resolved.** `load_lookup` exists and
  `verify` enforces a budget, but the intake doesn't use it. See B3.
- **B9 W0c (stale-base merge): open, unchanged.** `scripts/laned` and
  `laned-accept.yml` haven't changed since #602. This batch didn't trip it:
  for #637, #639, #641, #643, #645 and #649, no interacting PR merged between
  the job's start and the merge. The guard is still a job-time check, not a
  merge precondition.
- **B10 W3b2: not in scope, not re-checked.**

## BLOCKING

- [ ] **B1 W5e × W7c: a crash between the report write and the state write in a gated settlement wedges lane D for good.**
  - **Where.**
    - `cascade/validator/lane_d.py:433` puts the bench `session` id into
      every `gate.results[]` item, which is part of the signed body.
    - Recovery (`:722`, `:901`) re-runs `_run_gate`, so it sweeps the bench
      again in a new session.
    - `:989-991`: the recomputed body differs from the persisted one, so the
      call fails with `report_conflict`.
    - `lane_d_state.py:272-276`: at every later boundary the orphaned report
      sits "ahead" of the state, so the call fails with
      `state_digest_mismatch`.
  - **Reproduced.**
    - X first-clears at b1. At b2 its confirmation clears and the gate
      passes. `save_state` raises once, after the report has landed.
    - Re-call b2 with a new `SweepSession` (a restarted validator can't
      reuse the dead process's session): `report_conflict`.
    - Call b3: `state_digest_mismatch`.
    - Re-calling with the same session id recovers. But the same wedge
      follows from any bench difference: a host failure that turns `pass`
      into `bench_hold`, or GPU non-determinism in the re-swept rows.
  - **Way out.** None without hand-deleting a persisted report (forbidden).
    `restore_state` doesn't help: the b2 report still sits ahead of any b1
    backup.
  - **Wrong impl the tests pass.** Main's own code. No w7c restart test
    gates anything: every crash test is uncrowned or ungated, and the W5e
    tests never crash.
  - **Fix.**
    - On recovery (`persisted is not None`), take the gate outcomes from the
      persisted report's `gate` record and never re-sweep. First check that
      its draw re-derives and that its results cover exactly the recomputed
      candidates. Alternatively, write the gate results ahead before the
      report.
    - Add a typed operator path to retire a persisted report whose receipt
      was never returned (pinned to the expected prior digest).
    - Add tests: a crash in a crowning settlement recovers to the identical
      report with a new session and with a flaky host, and b+1 then settles.

- [ ] **B2 W7a #649: `check_lane_d` (and `verify_report`) pass crowns that never cleared, or that skipped the mandatory confirmation.**
  - **(a) The held path crowns any listed entry.**
    - `cascade/audit/lane_d.py:378-398` treats any gate result whose digest
      is a listed entry with no verdict as a "held candidate".
    - A `pass` on it is crownable. Nothing shows that the entry ever cleared
      a confirmation or was `bench_hold`.
    - **Reproduced:** entry y, `pending`/`pending_floor`, never judged. A
      gate `pass` for y plus `crowned = y` (validator key re-signs) gives
      `verify_report` True and `check_lane_d` PASS.
  - **(b) `stage` is never bound.**
    - `:216-223` reads `cohort["stage"]` from the report. Nothing checks
      that a `confirm` verdict has a prior first clear:
      `confirmation.first_set_sha256` can be null and never has to differ
      from this set.
    - **Reproduced:** X's first duel at b1, relabelled `confirm` and crowned
      with a gate `pass`, gets PASS. That's a crown in one settlement, which
      skips §1.3's mandatory confirmation duel (the D20 / A2 false-crown
      bound).
  - **(c) Also taken on trust, not reproduced:**
    - `report["seed"]` is never checked against
      `lane-d/1:{boundary}:{pool.sha256}`, so the bootstrap seed can be
      ground;
    - stage-`first` cohorts aren't bound to `king_before`;
    - `gate.sentinel` isn't checked against the era's W-S1 record, and the
      stated sentinel steers which suite is drawn.
  - **Why it's blocking.** §1.5 says satellites and the audit re-derive
    every verdict from signed fields. #649 says it refuses crowns without a
    passing gate, but the crown itself needs no derivation.
  - **Wrong impl the tests pass.** The current audit: the w7a audit tests
    doctor only gate fields, never stage or held provenance.
  - **Fix.**
    - Sign the provenance:
      - per `confirm` verdict, a non-null `first_set_sha256` that differs
        from this cohort's `set_sha256`, and `first_king_digest` consistent
        with the rule and `king_before`;
      - per held result, its hold record: `boundary` below the current one,
        `set_sha256`, `lcb` and `king_d`. The entry must read
        `confirm_pending` / `bench_hold`, or `crowned` when it passes.
    - The audit checks these fields. W9 cross-checks them against the
      previous report in the receipt chain.
    - Pin the seed. Add one doctored-report test per case.

- [ ] **B3 WB13 / WB11 (RV6 B8, blocks the SB3 pin, not the live intake): the intake still loads the hash set as the full dict.**
  - **Where.** `cascade/funding/main.py:551-554` calls `load_pinned_hashset`.
    That calls `load_artefact`, the full JSON dict, for EVERY
    `hashset-v<n>` file in the directory until one matches
    (`cascade/interface/contamination_scan.py:539-544`).
  - `load_lookup` (#640, about 64 B per window) has no production caller.
    The #643 body says why: `load_lookup` wasn't on main yet when #643 was
    built. `contamination.py:34` claims the intake uses it.
  - **Impact.** It's the same multi-GB load on the 7 GB box that RV6 B8
    measured against. The `verify` budget bounds `load_lookup`, not the path
    that actually runs.
  - **Fix.**
    - `load_pinned_hashset` selects the file by name/version, then calls
      `load_lookup(p, expect_sha256=pin)`.
    - Add a test that the intake's hash set is a `ContaminationLookup` and
      that non-matching files are never parsed whole.

## W-S2 (consensus) assessment

- **Gating is correct.** Every new path returns the unarmed call when
  `[v2] sentinel_from_block <= 0` or the settlement's era isn't armed:
  - `sentinel_pairing_kw(None)` is `{"require": require}`;
  - the validator's `_lane_b_sentinel_kw` returns `{}` with no chain read;
  - the trainer's `_settlement_sentinel` returns `None` and calls
    `ops.judge` with three arguments.
- **Both sides draw the same era.**
  - The validator uses `settlement_era_start(_epoch_start_block(manifest))`.
  - The trainer uses `settlement_era_start(block)`.
  - These are equal: the current settlement's `created_block` floors to its
    boundary, and a late one is stamped `created_block = boundary`
    (`cascade/trainer/rolling.py:3051`).
- **Suite names match.** `SENTINEL_SUITES` and `require_public` both use
  `gift`, `boom`, `time` and `fev`, as do the `item_rows_from_report` keys.
- **Read failures never guess a sentinel.** They hold the settlement
  (`LaneBTransient` → `StorageError`; trainer `OSError`).
- **Release-then-activate is possible:** ship W-S1 and W-S2 in one capability
  release, then arm with the overlay. See note 10 for the shared arm key.

## Invariant 1 / lane A

- **`settle_lane_d`.** It still returns `None`/`None`/`None` before any
  read when unarmed.
- **W-S3.** `status/v2.json` gains `sentinel` only when a board is passed,
  and `lane_a.sentinel` only when the receipt carries a W-S1 record.
  `build_sentinel_board` returns `None` unarmed. There's no production
  caller yet.
- **W-S2.** Lane B code only. `judge_acceptance`'s live callers pass no
  sentinel (both sides are unchanged).
- **Lane A.** No in-scope PR changes lane A scoring, the receipt, emission
  or `benchmarks.py`. receipt-replay is green on every in-scope PR.

## Non-blocking notes

1. **W5e: no pinned suite identity.** `_run_gate` calls `paired_gate`
   without `pinned_identity` (`lane_d.py:465`), so suite identity is
   checked by majority version only. Thread the W3b4 pin through.
2. **W5e: the sentinel is the caller's word.** `settle_lane_d` can't check
   that `sentinel` matches the era's W-S1 draw, because it only sees the
   lane D cfg. With W-S1 armed and `sentinel=None`, the gate can draw the
   held-out suite (D35 fails open). Neither side detects this (B2(c)). Add
   it to the unassigned call-site WP's acceptance.
3. **A10 fails open until W3b5.** A confirming clear with slice windows is
   crowned with `lowfreq.result = "not_candidate"` (`lane_d.py:651-667`).
   Either hold it, as the gate does, or record "arming waits on W3b5" in
   §17.
4. **Holds never expire and are uncapped.** `_held_candidates` keeps a hold
   with no TTL, and held candidates sit outside
   `max_entries_per_settlement`. Each one costs a bench sweep every
   settlement while the bench is down or the call is the legacy one. Bound
   it with W5h's reveal-delay close or a hold TTL.
5. **Invariant 11: a known digest is ignored silently.** An entry with an
   already-known digest under a new hotkey (`lane_d.py:312-313`, e.g. a
   copy of a crowned king's public weights) is dropped with no typed skip.
6. **D20 is unclear on the house bar.** Under `restart`, a house refresh
   while a miner holds KING_D keeps the first clear, even though the house
   bar changed (change 3). D20 lists "house refresh" as a king change.
   State which reading is intended, and test it.
7. **Hash normalisation mismatch.** `lane_d_gate._hash_hex` accepts
   upper-case and padded hashes but records them raw. The audit's
   `_gate_draw` requires strict lower-case, so an honest report would FAIL.
   Normalise the hash in `gate_record`.
8. **An uncaught draw error.** `gate_record` raises when no gate suite is
   eligible once the sentinel is removed (e.g. `gate_weights` positive only
   on the drawn sentinel). Nothing catches that, so every crowning
   settlement in that era raises. Refuse it at parse time, or make it a
   `bench_hold`.
9. **WB13: a load-dependent refusal becomes permanent.** A deadline-driven
   `budget_exceeded` depends on load, but it's cached by (ref, pin), so a
   transient becomes a sticky 413 until the LRU evicts it. Cache only
   window-budget overruns. The scan still runs on the handler thread, up to
   20 s each.
10. **W-S2 shares its arm key with W-S1.** Both arm on `[v2]
    sentinel_from_block`. A release carrying W-S1 but not W-S2, armed
    while lane B is active, diverges on sentinel-only regressions. Put "W-S2
    in the same capability release" on the arming checklist (§14.5).
11. **W7c (RV6 note 5(c)) is still open.** The recovery and re-call paths
    return the persisted signature unverified.

Verdict: NOT ACCEPTED. There are 3 BLOCKING items:

- W5e × W7c: 1 (B1, a permanent wedge on crash recovery);
- W7a audit: 1 (B2, forged crowns and a skipped confirmation pass);
- WB13 / WB11: 1 (B3, RV6 B8 still unwired; it blocks the SB3 pin only).

RV6 B9 (W0c stale-base) stays open, and B10 wasn't re-checked. Of the RV6
items, B1–B7 are resolved and B8 is partly resolved. W-S2's gating is
correct, and W-S3 has no BLOCKING item.
