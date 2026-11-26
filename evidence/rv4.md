# RV4: lane D review of W3b1, W7a, WB14, W5b, W3a

Covers: W3b1, W7a, WB14, W5b, W3a

Reviewer: RV4 (fresh context, read-only on main). Reviewed at main
`551a6cd` (#571). In scope:

- W3b1 #559 (tests #553);
- W7a #554 (tests #541, #550, #564);
- WB14 #556 (tests #543);
- W5b #563 (tests #555);
- W3a #561 (tests #557);
- the RV3 fixes merged since: #560 (W1.5 fixtures), #562 (W5a tests),
  #564 (W7a tests), #565 (W5d tests) and #571 (the W5d T7 kit).

The spec is docs/plans/LANE_D_PLAN.md: the §3.0 rows, §3.1, §3.2, §1.3,
§1.3b, §1.5, §14.3, §16 and §17, plus each WP's `tests/accept/<wp>/`. I read
the code itself, not the PR bodies.

Known open items, not re-reported:

- #568 (W5a fail-closed impl);
- the W7a schema enum fix (`lane_closed`, `unlisted_feed`);
- the W7b/W1.5 `verify_report` conflict;
- W3b2 gate wiring into `settle_lane_d` (W5e).

Checks run, one directory at a time at `nice -n 19`, with
`env PYTHONPATH=<wt> python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -m 'not torch'`:

- w3b1: 60 passed. wb14: 77 passed. w5b: 21 passed. w3a: 77 passed,
  4 skipped (`$LANED_EVIDENCE` unset). w1_5: 15 passed. w5d: 115 passed,
  2 skipped.
- w7a: 5 failed. All 5 are the known enum fix (`lane_closed`,
  `unlisted_feed`).
- w5a: 10 failed. All 10 are the known #568.
- So W7a and W5a are red on main and sit at L2, not L3.
- **Reviewer pitfall (new).** `/root/cascade-v2/.venv` has an editable
  `cascade` install that wins over pytest's rootdir insertion. Without
  `PYTHONPATH=<wt>` a run tests `/root/cascade-v2/cascade`, not the
  worktree: w3b1 then shows 52 `ModuleNotFoundError`s, and the other
  directories pass or fail against the wrong code. Check with
  `python -P -c "import cascade; print(cascade.__file__)"`.
- `stub_scan.py --root <wt> --paths-file tests/accept/<wp>/PATHS`:
  0 findings for w3b1, w7a, wb14, w5b, w3a and w5d (1, 4, 4, 3, 2 and 2
  files).
- A reproduction of BLOCKING item 2 (below), run from a scratch script
  against the worktree with the W5b test helpers.

## RV3 BLOCKING items: re-check

- **RV3 W6b/W9 (forward-only fails open without `frontier` / `feeds`):
  still open.** The tests-only PR #562 is merged. The implementation is #568
  (known), and the w5a tests are red on main until it lands. RV3 also asked
  for amendments to the W6b and W9 rows. Those are plan PRs, and I don't see
  them on main.
- **RV3 W5d (the T7 kit certifies an unexercised context): resolved by
  #571.**
  - `measure_variant` refuses a history shorter than the variant
    (`ShortHistoryError`) before torch loads. It also requires
    `batch_plan == [variant]` (`cascade/eval/lane_d_t7.py:220-234`).
  - Rows record `input_len`, `history_min`, `history_max` and
    `peak_mem_source`.
  - `build_evidence` rebuilds every handed-in row through `variant_row`
    (`_recheck_row`), so a hand-edited row gets every check. Pod evidence
    accepts only `cuda_max_allocated`.
  - The CUDA peak is reset per variant (`:242`).

## BLOCKING

- [ ] BLOCKING W7a: the closed `lane-d-report/1` schema has nowhere to put the drawn gate suite, the era's sentinel or the gate weights, so W3b2/W5e can't sign them without a schema bump.
  - **The plan.**
    - §1.5: the signed report carries "the drawn suite".
    - §14.3 rule 2: "every value a satellite needs (… seed, suite name …)
      is in the signed report and cross-checked against the overlay".
    - §16 rule 1: the drawn sentinel is "a structured field" in the lane D
      report, which satellites verify from the block hash.
    - W3b1's own module (`cascade/validator/lane_d_gate.py:21-22, 137-155`)
      says `gate_record` "is the `gate` field of the lane D report (W7a)".
  - **The schema.**
    - Top-level `additionalProperties: false`
      (`cascade/validator/schemas/lane_d_report_1.json:10`).
    - No `gate` or `sentinel` property, and no gate result (CRPS/MASE LCB
      vs KING_D and vs the house, §1.3b).
    - `params` is closed (`:76`) and has `gate_suites` but not
      `gate_weights`. It also lacks `max_source_share` (W5b's cap) and
      `min_long_series`.
  - **Failure scenario.** W5e wires `gate_record(...)` into the report as
    `report["gate"]`. `validate_report` then refuses every armed report with
    `unknown_field at gate`. The builder's other choice is to put the gate
    under `extensions`, which the schema itself calls "never consensus" (so
    satellites must not read it). Either way satellites can't verify:
    - which suite gated a crowning, or that it wasn't the sentinel;
    - the `max_source_share` drop.

    Adding the fields later is the `lane-d-report/2` bump that §14.3 rule 3
    says v1 must anticipate.
  - **Fix (W7a: tests-only PR first, then the schema and `_params`).**
    - Add a required top-level `gate`: `null` when no candidate reached the
      gate, else `{suite, block, block_hash, sentinel, eligible}`, matching
      `gate_record`.
    - Add per-verdict gate results (suite, opponent `king` | `house`,
      metric, `lcb`, `passed`).
    - Add `sentinel` (`{suite, block, block_hash}` or `null` when W-S1 is
      unarmed).
    - Add `gate_weights`, `max_source_share` and `min_long_series` to
      `params`. Have `cascade/audit/lane_d.py:_expected_params` (`:40`)
      derive them from cfg.

- [ ] BLOCKING W5b: a window admitted by an older snapshot is scored from the newest snapshot's version, which §1.3 may refuse (invariant 4).
  - **The code.**
    - `_eligibility` (`cascade/validator/lane_d.py:333`) admits a window id
      if *any* loaded snapshot admits it.
    - `_window_index` (`:322`) resolves every id to the *newest* snapshot
      holding it.
    - `_cohort_set` then scores `index[wid][0]` (`:371`), and the report
      takes `mirror_lag_hours` from that newest snapshot's feeds (`:569`).
    - A window id (`<series_id>:<target_start_ts>:h64`) is stable across
      snapshots, but its `first_seen_ts`, values, `frontier` and its feed's
      measured lag are not.
  - **Reproduced.** I used the W5b helpers, with 4 sources × 6 windows and
    the floors met:
    - snapshot A has every window eligible;
    - snapshot B (newest) has the same records, except one window's
      `first_seen_ts` is moved before the bound;
    - `eligible_windows` on B gives that window `publish_ahead`.

    `settle_lane_d(pool_dir=B, prior_pool_dirs=[A])` judges all 24 windows,
    including the refused one. The signed report lists it with
    `min_first_seen_ts = 1790018000` < bound `1790021600`.
  - **What else hits it.** Any feed whose measured lag rises above
    `max_mirror_lag_hours` in a later snapshot gives a report window with
    `mirror_lag_hours` over the cap. So does a source revision of a
    window's values: it is scored on values that no admitting snapshot
    held. A satellite or the W9 audit checking "declared timestamps against
    declared T values" (§1.5) finds the main validator's own report in
    breach of forward-only, after the crowning.
  - **Tests too weak.** No W5b test gives one window id two different
    versions across snapshots. A wrong implementation (score the newest
    version, admit by any snapshot) passes all 21.
  - **Fix (W5b: tests-only PR first).** Score each window from a snapshot
    that admits it for this cohort's `max(T)`, for example the newest
    admitting one. Carry that snapshot's id with the window. Or refuse a
    window whose newest version is refused. Then add the two-version test.

- [ ] BLOCKING W5b: the signed report binds only the newest snapshot, though W5b judges windows from the prior ones.
  - **The code.** `report["pool"]` (`cascade/validator/lane_d.py:637`) is
    `{snapshot, sha256, first_seen_roots}` of `pools[-1]` only. Windows
    frozen from `prior_pool_dirs` carry no snapshot reference in `windows`
    or `cohorts`. Meanwhile `cohorts[].first_seen_roots` is the union over
    every loaded snapshot (`:534`), so the two root lists in one report
    disagree.
  - **Failure scenario.** An entry freezes on windows from snapshots
    D-3…D. Later the crowned king's snapshot is revealed for re-scoring
    (§1.5), or a loser disputes under D23(ii) ("verify targets, window
    ids"). The report names only snapshot D's sha256, so nothing signed says
    which bytes the D-3…D-1 windows were scored on. A validator with a
    different `prior_pool_dirs` retention judges a different union, and the
    report doesn't show it.
  - **Fix (W5b + W7a schema).** `pool` becomes a list of
    `{snapshot, sha256, first_seen_roots}` for every snapshot that
    contributed a judged window. Each `windows[]` row names its snapshot
    (from BLOCKING item 2's fix). The tests-only PR asserts a two-snapshot
    union report lists both sha256s.

- [ ] BLOCKING W3a: on a lock-enabled bucket with a default retention rule, the live probe COMPLIANCE-locks about 11 GB of test objects it can't delete.
  - **The code.**
    - The probe's only bucket-lock read is
      `bucket_lock_enabled` (`cascade/laned/storage_probe.py:454-459`). It
      checks `ObjectLockEnabled == "Enabled"` and ignores
      `ObjectLockConfiguration.Rule.DefaultRetention`.
    - The bucket must be lock-enabled for the `object_lock_multipart` and
      `compliance_retention_unshortenable` probes.
    - The probe then writes a 2.5 GiB source, a 2.5 GiB CopyObject, a
      2.5 GiB UploadPartCopy destination, and 512 MiB + 2.5 GiB single PUTs
      (`LIVE_DEFAULTS`, `:80-82`).
    - `cleanup` (`:1091`) deletes versions and reports anything left as
      `remaining_unlocked`.
  - **Failure scenario.** The owner points `--bucket` at a lock-enabled
    bucket provisioned the way a vault would be, with
    `DefaultRetention {COMPLIANCE, N days}`. Its name isn't in chain.toml
    yet, so `production_buckets` doesn't refuse it. Every object the probe
    writes inherits the default lock:
    - about 11 GB can't be deleted for N days, and is billed for them;
    - the run ends `complete: false` with `remaining_unlocked` non-empty;
    - this breaks the probe's own promise (module docstring, `:29-31`, and
      the `--confirm-live` text, `:1255`): only two objects of
      ≤ 4096 bytes are locked, for ≤ 1 day.
  - **Fix (W3a).** Before any write, read the lock configuration and refuse
    (`UsageError`, exit 2) when `Rule.DefaultRetention` is present. Record
    the read in the evidence. Add a fixture-model field `default_retention`
    and a test that the probe refuses it with no object written. The tests
    go first, under rule 1.

- [ ] BLOCKING W3a: L4 can be reached with fixture evidence relabelled as live, and the verifier never checks the live resource.
  - **The plan.**
    - §17.1 L4: "evidence JSON present, **and the verifier passes against
      the live resource**".
    - §3: "the evidence verifier cross-checks against the live resource
      where it can".
    - This is the same class as RV2's R10 item.
  - **The code.** `tests/accept/w3a/helpers.py:188-200` (`as_live_evidence`)
    turns a `--dry-run --fixture` run into evidence the verifier accepts. It
    takes 4 edits: `source: "live"`, `git_dirty: false`, rename the
    `fixture-` request ids, and restate `passed`. The repo's own tests use
    it to show that the verifier (`evidence_problems`, `:700`) accepts that
    evidence. Nothing in the verifier contacts Hippius. The checks are all
    self-consistency of the recorded observations: sizes, statuses, ids not
    `fixture-`.
  - **Failure scenario.** A fixture model that says
    `presigned_upload_part: pass` is relabelled, and its sha256 is put in
    `decisions/STORAGE-LANE-D.md`. W3a reaches L4, and W3, W6d, W10, W13
    and W18 build `presigned_multipart` against storage that may not
    support it. S1 ("W3a at L4") passes on no live run.
  - **Fix (W3a; tests first).** Add a `storage_probe verify --evidence`
    mode that the owner runs with `HIPPIUS_*` set. It HEADs the recorded
    COMPLIANCE-locked objects, checks the `VersionId`, `ETag`, mode and
    retain-until that the evidence records, and lists the run root to
    confirm cleanup.
    - Make the default lock retention long enough to verify after the run
      (e.g. 3 days, still ≤ 4 KiB per object).
    - The L4 test requires a `verify` record (time, request ids) written
      by that mode, and refuses evidence whose locked objects' retain-until
      had passed before `verify` ran.

## Non-blocking notes

1. **W3b1: the sentinel removal depends on name equality across two
   tables.** `gate_suite_for` removes the sentinel only if it is spelled the
   same as a `gate_suites` entry. "A sentinel outside `gate_suites` removes
   nothing" is a pinned rule (`lane_d_gate.py:108-109`). `[lane_d]
   gate_suites` is overlay-settable and isn't validated against
   `[v2] sentinel_suites` (`cascade/shared/config/lane_d.py:255`).
   - An overlay with `gate_suites = ["gift_eval", …]` makes the era's
     sentinel a drawable gate suite, with no error.
   - Recommendation for W2: refuse a `gate_suites` entry that isn't in
     `[v2] sentinel_suites` (or `SENTINEL_SUITES`).
2. **W3b1 is otherwise sound.**
   - The draw uses exact `Fraction` arithmetic over the tagged sha256, with
     no `hash()` and no float sum.
   - The domain tag is separate from W-S1's.
   - Scale invariance holds, and the refusals are typed.
   - `gate_record` stores the hash as given, while the draw lowercases it.
     That's harmless, because replay lowercases too.
3. **W5b: `until_block` closes entries only at the exact boundary.**
   `_closing` (`lane_d.py:469-473`) requires `boundary == until_block`, and
   `active()` is false for every later boundary.
   - If the main validator misses that one settlement (downtime, or an
     overlay whose `until_block` is adopted after its boundary), every open
     entry stays `pending` in `lane_d_state.json` forever and is never
     closed with `lane_closed`.
   - Pay isn't affected, since the lane is off. The status board and the
     T18 "close unjudged" record are.
   - Suggest closing at the first settled boundary ≥ `until_block` that
     still has open entries.
4. **W5b: accumulation depends on snapshot retention, and nothing pins
   it.**
   - Only `_select`-chosen entries advance their `frozen_set`.
   - `_cohort_set` drops frozen windows that are no longer in any loaded
     pool (`wid not in index`).
   - If `prior_pool_dirs` keeps fewer snapshots than an entry needs to
     freeze plus be deferred, its frozen set shrinks below the floors. It
     never refreshes, because frozen sets are immutable, so it sits
     `pending_floor` until `reveal_delay_exceeded`.
   - W5e (the caller) and W6d (retention) should pin: "retain every
     snapshot since the oldest open entry's T".
5. **W5b: `max_source_share` versus `min_clusters`.** With the cap at 0.25,
   a union with fewer than 4 sources gets `Q = 0` and keeps nothing
   (`lane_d_frozen.py:55-62`). That's correct as written, and harmless at
   `min_clusters = 20`. An overlay pairing `min_clusters < 4` with
   `max_source_share = 0.25` would make freezing impossible. W2 could refuse
   `max_source_share * min_clusters < 1`.
6. **W5b: a frozen but not yet judged set is discarded when the comparison
   king's T changes** (`_advance_member`, `lane_d.py:381-394`). That
   fits D20 (a) "re-freezes against the new king". It contradicts
   `lane_d_frozen`'s docstring, "once frozen it never changes". Say so in
   the docstring, or in the README contract.
7. **W7a: `validate_report` has no caller on main.** `settle_lane_d` signs
   without validating, and `cascade/audit/lane_d.py` doesn't validate
   either. Today a schema-invalid report is signed and published; the
   `lane_closed` report at `until_block` is one. The test
   `test_e2e_until_block_closing_report_validates` catches that specific
   case. Recommend that `settle_lane_d` calls `validate_report` before
   `sign_report` (fail closed), and that W9 calls it first.
8. **W7a: `extensions` content isn't checked for finite numbers.**
   `_Validator.check` descends only where the schema declares
   `properties` or `items`. A NaN inside `extensions` passes
   `validate_report`, and then `canonical_json(allow_nan=False)` raises at
   signing. Validation passes while signing fails. A recursive finite-number
   walk in `validate_report` closes it.
9. **WB14: declared rows are stored but nothing consumes them yet.**
   - `declared_rows` reaches `LaneBQueue` entries, but no trainer path
     reads it. That's WB15's job, and §3 orders it after WB14.
   - Lane B is off (`[lane_b] from_block = 0`), so nothing is live. But if
     lane B arms before WB15 lands, the intake accepts, signs and stores a
     declaration that ESFT then silently overrides, and the miner pays for
     a leg that ignored its rows.
   - Suggest that the intake refuses the header with a typed
     `declared_rows_unsupported` until WB15 is at L3, or that WB15 ships in
     the same release as WB14.
10. **WB14: drive-by edit in `cascade/funding/main.py`.** It turns
    `_DeadlineCachedPoll` into an `ABC` with `@abstractmethod`, replacing
    `raise NotImplementedError` (probably for the stub scan). Both
    subclasses implement `_poll` and `_fold` (`:159`, `:255`), so it's
    harmless. It is a funding-service change that isn't named in the WB14
    row.
11. **WB14 is otherwise sound.**
    - Keys are MoE ordinals 0–6, and block indices are refused.
    - The clamp applies only to layers with rows.
    - It is validated against `experts_per_layer`, read by AST without
      torch. The presets don't override `experts_per_layer`, so the class
      default is the live geometry.
    - The rows digest is bound into the signed ref, and the field is
      byte-identical when absent.
    - The queue keeps a running leg's rows, and default fields are
      stripped on save.
12. **W3a: the decision verifier and the probe each compute the supported
    paths.** `supported_paths` (`helpers.py`) and `recommended_path`
    (`storage_probe.py:1114`) agree today. If a fourth path is added later,
    a single shared table would keep them from drifting.
13. **Invariant 1 / lane A untouched.**
    - None of the in-scope PRs touch lane A code (`cascade/eval/`
      lane A paths, `shared/emission.py`, the receipt, or the trainer
      apart from the new `cascade/trainer/declared_rows.py`).
    - The `cascade/trainer/` file is torch-free. It is imported only by the
      funding intake, and only when the header is present.
    - W5b's `settle_lane_d` returns before any work when it isn't armed.
14. **Invariant 12.** `cascade/audit/lane_d.py` still imports only
    `eval.bootstrap`, `audit.checks` and `shared.sr25519`. It gets no
    `lane_d_forward`, `lane_d_frozen` or `lane_d_gate`. Future work: W9's
    re-derivation of BLOCKING item 2 (which snapshot a window was scored
    from) must not import `lane_d_frozen`.

Verdict: NOT ACCEPTED. There are 5 BLOCKING items: W7a (1), W5b (2) and
W3a (2). W3b1 and WB14 have no BLOCKING item and may reach L5 once they are
at L3. Of the RV3 BLOCKING items, W5d is resolved; W6b/W9 waits on #568
and the row amendments.
