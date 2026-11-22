# RV6: lane D review of W18, WB18, W7c, WB17, WB13, WB11 schema 2 and the RV5 fixes

Covers: W18, WB18, W7c, WB17, WB13, WB11, W3b3, W3b2, W0c

Reviewer: RV6 (fresh context, read-only on main). Reviewed at main
`c27f71c` (#621). In scope:

- W18 #601 (tests #597);
- WB18 #605 (tests #600);
- W7c #618 (tests #608);
- WB17 #620 (tests #616);
- WB13 #619 (tests #615);
- WB11 schema 2 #621 (tests #617; original #596/#590);
- RV5 fixes: W3b3 #599 (+#613), W3b2 #607 (+#604), W0c #602.

The spec is docs/plans/LANE_D_PLAN.md: the §3.0 rows, §3.1, §3.2, §12b.1,
§12b.4, §12b.5, §16 and §17, plus each WP's `tests/accept/<wp>/`. I read the
code, not the PR bodies.

Known open items, not re-reported: the W7a schema impl (blocked on the owner
edit); the W5b/W5c/W5e/W3b5 impls waiting on it; the W18 vault-ledger
producer (unassigned); the W3b2 pointer history (unassigned); the W3b3
sentinel call site (unassigned); the §12b.1 wording vs WB11 schema 2. The
"pinned `FundingIntake` without both kwargs" and "WB13 scan budget" items are
assessed below, as asked.

Checks run, one directory at a time at `nice -n 19`, with
`env PYTHONPATH=<wt> python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -m 'not torch'`:

- w18: 52 passed, 1 skipped (`$LANED_EVIDENCE` unset). w7c: 63 passed.
  wb18: 19 passed. wb17: 51 passed, 1 skipped. wb13: 43 passed. wb11: 103
  passed, 2 skipped. w3b3: 42 passed (the RV5 ImportError is fixed). w3b2:
  150 passed, 2 skipped. w0c selftests: 80 passed.
- `stub_scan.py` over each in-scope `PATHS`: 0 findings.
- CI on main: the push runs up to `cac53f5` are red only on the known W5b
  (3), W5c house-report (25, the RV5 item-4 tests awaiting the impl) and W7a
  tests. The `c27f71c` run was still in progress at review time.
- Scratch reproductions against the worktree, no torch, for every BLOCKING
  item below except B9 (a timeline read from GitHub).

## RV5 BLOCKING items: re-check

- **W3b3 (sentinel doesn't import): resolved by #599.** It uses
  `classify_suite`, with no private cross-module imports. RV5 note 5 is
  partly open (note 7 below).
- **W0c (stale-base merge): open.** See B9.
- **W0c (verification WPs not cross-eligible): resolved by #602.**
  `built_wps_at` (`scripts/laned/accept:749-772`) includes declared
  verification WPs.
- **W0c (verification mode open to any WP): resolved, with a residual.**
  Verification mode now needs the WP to be listed in
  `scripts/laned/verification_wps`, read at the base. Only W0b is listed.
  The residual is note 8.
- **W5c (house comparison unverifiable): open.** The tests are on main and
  red (25 in `test_w5c_house_report.py`), and the impl waits on W7a. Known.
- **W3b2 (L4 evidence fabricable): partly resolved.** See B10.

## BLOCKING

- [ ] **B1 W18: a storage error mid-run aborts the run after deletions and writes no action log.**
  - **Where.** `cascade/laned/vault_sweeper.py:817`, `:882`, `:889`,
    `:936-937`. `locked_until`, `archived_at` and `list_keys` are called
    outside any try. A `StoreError` escapes `sweep()`, so the log write
    (`:957`) never runs. The CLI prints `storage_failed` and exits 1.
  - **Reproduced.** A 503 SlowDown on the second due entry's retention and
    HEAD reads. By then the first entry's vault objects were already deleted,
    and `logs/` was empty.
  - **Impact.** The only record of a miner-data deletion is lost. After the
    archive copies expire, nothing shows those objects existed. This breaks
    the W18 README rule that every listed object gets a record.
  - **Fix.** Catch errors per key and record `keep`/`failed` with a typed
    reason (add `lock_unreadable`). Write the log in a `finally`, aborts
    included. Add a test: an error on entry 2 still leaves a complete log on
    disk that shows entry 1's deletes.

- [ ] **B2 W18: an unreadable lock is treated as no lock.**
  - **Where.** `vault_sweeper.py:86-87` (`_NO_LOCK_CODES` includes
    `InvalidRequest`, a generic S3 400) and `:605-606`. On that code,
    `_version_retain_until` returns `None` without the HEAD fallback.
  - **Reproduced.** Two COMPLIANCE-locked versions whose
    `GetObjectRetention` answered `InvalidRequest` were archived and then
    sent `delete_object`. Only the server's lock enforcement stopped the data
    loss.
  - **Wrong impl the tests pass.** `except Exception: return None` around
    `get_object_retention`. The fake only ever raises
    `NoSuchObjectLockConfiguration`.
  - **Fix.** Keep only the two no-lock codes. On any other error check HEAD,
    and if that fails too, keep the object as `object_lock` (unknown). Add
    tests for `InvalidRequest`, 403 and 503.

- [ ] **B3 W7c: a lost state file silently restarts lane D from genesis (invariant 5).**
  - **Where.** `cascade/validator/lane_d.py:528` calls
    `load_state(state_path, house)` with no `expect_sha256`, and
    `lane_d_state.py:169-171` maps a missing file to genesis. Nothing checks
    for earlier reports.
  - **Reproduced.** Settle b1, then b2 (X crowned). Delete the state file and
    settle b3. b3's report shows `king_before` = the house and
    `crowned = None`, and X is judged again. That is a silent dethrone plus a
    re-score, and this validator now diverges from the others. Only
    `verify_receipt_chain` would flag it (`chain_break`), and nothing calls it
    in production yet.
  - **Trigger.** A redeploy with the wrong state dir, or a restore that
    misses the file.
  - **Wrong impl the tests pass.** Main's own behaviour: no test removes the
    state file while `<stem>_reports/` holds reports.
  - **Fix.** Refuse with a typed `state_missing` when the file is absent but
    the report directory already holds a report, or when the caller passes
    the last published posterior. Thread `expect_sha256` from the receipt
    once W8/W14 are wired.

- [ ] **B4 WB18: a sentinel-deferred funded entry can expire, so the deferral isn't uncharged (§12b.5).**
  - **Where.** `cascade/trainer/lane_b_driver.py:700` runs
    `fq.expire_stale()` on every tick. The age is measured from `active_at`
    (`cascade/funding/queue.py:999-1001`), with TTL
    `[round] funded_entry_ttl_hours = 36`. The hold (`lane_b_driver.py:512-541`,
    `:757-766`) only drops the entry from `eligible`. It never refreshes it.
  - **Arithmetic.** `sentinel_max_defer_eras = 4` at 12 h per era is 48 h,
    which is longer than the 36 h TTL.
  - **Scenario.** A funded `suite:gift` entry is deferred for three eras (a
    1-in-16 chance after the first deferral, higher if it had already
    queued). It moves to `failed`/`funding_expired`, and the miner pays
    again.
  - **Wrong impl the tests pass.** This one: no wb18 test ages the funded
    queue.
  - **Fix.** Refresh held entries' liveness on each tick, or exempt
    `deferred: sentinel_target` from `expire_stale`. Add a test: an entry
    held for `sentinel_max_defer_eras` eras, with the clock past the TTL, is
    still `queued`. WB15's `rows_held` deferral has the same flaw.

- [ ] **B5 WB13 / WB11: benign repos are refused (false positives).**
  - **Where.** `cascade/eval/contamination.py:259-266` hashes every finite
    32-point window into the exact set, constant windows included. The scan
    (`cascade/interface/contamination_scan.py:248-249`) decodes any non-code
    file from offset 0 as f64/f32/i64/i32. It treats all of a `.py` file's
    numeric constants as one stream.
  - **Reproduced.** I built a hash set from suites with a 40-zero run (real
    GIFT/BOOM has these: solar at night, sparse counters) and one sparse
    spike series. Each of these benign inputs came back `embedded`:
    - a `.npy` holding 1 KB of zeros;
    - a `.py` with 32 `x = 0` lines;
    - three one-hot 32-element list literals (3 normalised hits).
  - **Tests miss it.** TB12's clean repo is a toy, and `test_int_bytes` pins
    a flat window as a hit.
  - **Fix (WB11).** Drop low-information windows from both sets: constant
    windows, and windows with ≤ k distinct values or a degenerate z pattern.
    Add a WB13 test: a realistic benign repo is clean against a hash set
    built from suites with zero runs and spikes. Measure the false-positive
    rate on real generators before the owner pin.

- [ ] **B6 WB13: one crafted file can stall the live funding intake for hours (goes beyond the known "scan budget" item).**
  - **Where.** `contamination.py:147-189` and `intake.py:1323`.
  - **Cost.** A window whose 32 z-scores all sit within TOLERANCE of a bin
    edge costs 2 × 4096 sha256 probes. I measured 0.019 s per window. 1 MB
    of crafted f64 is about 131k windows, or about 2,500 s of CPU. At the
    8 MB D38 cap that's about 5.5 h. Non-`suite:` targets have no intake
    cap at all.
  - **Why it hurts.** The scan runs synchronously in the
    `ThreadingHTTPServer` handler. The connection deadline closes the socket
    but not the thread. Short hashlib calls hold the GIL, so lane A fund
    requests starve on the 4-core box the validator shares. There's no
    result cache, so every repeat fund re-scans.
  - **Fix.**
    - A per-request probe or wall budget that fails closed with a typed
      503/413.
    - Cap probes per window (e.g. 2^4) and windows per tree.
    - Cache the result by (ref, pin).
    - Scan off the handler thread on a bounded worker.
    - Add a byte cap for every lane B tree.
    - Add a test that an adversarial edge-window file finishes within
      budget.

- [ ] **B7 WB13: the LB1 arming gate is evaded by trivial packaging of an exact benchmark copy.**
  - **Where.** `contamination_scan.py:139-149` (`_number` drops str and
    bytes), `:200-217` and `:248-251`.
  - **Reproduced clean** (each is a one-line change by the miner):
    - a str literal `"1.2,3.4,…"`;
    - a bytes literal read back with `np.frombuffer`;
    - a `np.savez_compressed` `.npz` (`*.npz` is in the fetch allow-list,
      `hippius.py:208-212`);
    - raw f64 at a 4-byte offset, or big-endian;
    - a one-line CSV;
    - delta encoding, or interleaving with zeros.
  - **Spec.** §12b.1 asks for "any non-code bytes (decoded as
    float32/float64/int arrays)". Guard 4 is the backstop for procedural
    regeneration, not for shipping the bytes in a container.
  - **Fix.**
    - Decode str/bytes constants: numeric tokens, base64/hex, `frombuffer`
      candidates.
    - Inflate `.npz`/zip/gzip under size and ratio caps, and parse `.npy`
      and `.safetensors` headers.
    - Decode at every alignment and in both endiannesses, and split text on
      any delimiter.
    - Add one test per form.
    - Record the residual limit in the decision register: an outlier every
      31 points, or noise above about 0.03 σ, evades by design.

- [ ] **B8 WB11 owner pin (L4): the artefact's size and load RSS at real-suite scale are unmeasured, and the intake loads it whole on the 7 GB prod box.**
  - **Where.** `contamination.py:268-281` stores each window twice as a
    hex-keyed JSON dict, and `funding/main.py:554` loads it into memory.
  - **Estimate, not a measurement.** Tens of millions of test-window offsets
    puts this in the multi-GB range, enough to OOM every service on the box.
  - **Fix.** The pin's verify step records `n_exact`/`n_normalised`, file
    size and load RSS, and refuses a pin over a stated budget. Consider a
    sorted binary layout of truncated (64/128-bit) hashes.

- [ ] **B9 W0c #602: the stale-base guard runs when the job runs, not at merge, and this batch bypassed it.**
  - **Where.** The `freshness` step in `.github/workflows/laned-accept.yml`
    compares against `origin/main` at job time. "Re-run before merging" is
    a comment ("the orchestrator enforces it"). `main` still has no branch
    protection (protection 404, rulesets `[]`).
  - **It happened.**
    - #621 (WB11 schema 2, `cascade/eval/contamination.py`) ran its PR job on
      base `19088de` (done 07:49).
    - #620 merged at 07:59, then #619 (WB13) at 08:02:55. WB13 imports
      `contamination.py` (`funding/intake.py:492`, `contamination_scan.py`).
    - #621 merged at 08:03:32, 37 s after #619, with no re-run.
    - wb13 happens to pass on `c27f71c` (43 passed), but this is the RV5
      #592/#593 sequence again.
  - **Fix.** Make it a mechanical precondition of the merge step: refuse to
    merge when the head the run tested isn't the current merge with
    `origin/main`. Alternatively, the owner turns on "require branches up to
    date" plus the laned-accept required check. Add a selftest: a PR green
    on an older base is refused after an interacting PR merges.

- [ ] **B10 W3b2 #607: L4 pod evidence can still be fabricated offline.**
  - **Fixed.** Item ids are bound to a pinned `suite_identity`, and there is
    a real read-only `verify_live` (`cascade/laned/bench_evidence.py`).
  - **Gap.** The L4 check `verified_live_problems`
    (`tests/accept/w3b2/evidence_lib.py:318`) checks the unsigned
    `verified_live` record for internal consistency only. The test helper
    `record_for` (`:447-462`) builds a passing record by hand. A forger can
    do the same with a real main `git_sha`, a block hash that draws the
    right suite, and fabricated rows on the pinned items.
  - **Honest L4 isn't reachable either.** The real CLI always refuses with
    `NoPointerSource` (the pointer history is the known unassigned item).
  - **Fix.** The owner's verify signs the record with an operator hotkey, or
    CI re-runs `verify_live` with read-only adapters at L4 time. Until a
    pointer source exists, record W3b2 L4 as blocked in
    `evidence/blocked/w3b2.md`.

## Assessed known items

- **Pinned `FundingIntake` without both kwargs skips the scan: LOW.**
  - It fails open with a warning (`intake.py:482-490`, `:1258`).
  - It isn't reachable in production: `funding/main.py:602-603` always
    passes both, and no other `FundingIntake(` caller exists in `cascade/`,
    `scripts/` or `deploy/`.
  - One-line fix: pinned but unwired should answer 503
    `contamination_hashset_unavailable`, not skip.
- **WB13 scan budget.** This is subsumed by B6: the cost is adversarial,
  not just repo size. Also, the D38 size check runs after the full download
  (`intake.py:1314-1321`), so it doesn't bound fetch cost.

## Invariant 1 / lane A

- **WB13.** Unarmed, it is byte-identical in consensus and in responses.
  - `funding/main.py:548-564` builds the hash set and fetcher only when
    `lane_b_queue` exists and the pin is set.
  - `intake.py:1258` skips the scan when the pin is `""`.
  - The queue writes `contamination` only when it is set
    (`queue.py:380-386`).
  - No scan module is imported and no fetch happens. Mainnet's `lane_b_fund`
    answers `lane_b_disabled`.
- **WB17 / WB18.** Unarmed, trainer behaviour is unchanged.
  - The co-gate keeps the same `may_open` expression.
  - `_sentinel_target` returns `None` before any chain read unless
    `suite_targets` is on and `sentinel_from_block > 0`.
  - `targets` and the `s-` domain refusal change only when `suite_targets`
    is true.
  - New keys default to the unarmed behaviour.
  - Not byte-identical: `status/lane_b.json` now always carries `open_rule`,
    `gate_rule` and `gate_closed_reason`, and the driver state file gains
    `gate_rule`, `gate_evidence`, `gate_closed_reason` and
    `sentinel_deferrals`. The driver is built whenever `[v2] from_block > 0`,
    so mainnet's files change while lane B is off. It isn't consensus. This
    is the same class as RV5 note 6. Either gate the keys on `lb.active` or
    record it as accepted.
- **W7c, W18.** No production caller. Unarmed, `settle_lane_d` returns
  `receipt = None` and writes nothing. #601 adds one new file.
- **Lane A.** No in-scope PR touches lane A scoring, the receipt, emission or
  `benchmarks.py`. `expert_lane.py` (#620) gains only the new
  `open_rule_verdict` function.

## Non-blocking notes

1. **WB17: `tokens_king_check` opens on any past pass, not this king's.**
   Fix this before an overlay selects that rule.
   - `_receipt_king_check` (`runner_promotion.py:386-426`) takes the newest
     receipt with a `king_check`, up to 64 receipts back.
   - `open_rule_verdict` checks only `result == "pass"`. It never compares
     `king_pointer` with the current king, or checks age.
   - After a dethrone, the old king's pass opens the one-way latch.
   - Wrong impl the tests pass: one that ignores king identity.
   - Fix: require the current `king_pointer` and a boundary within one era,
     and add that test.
2. **WB18: the defer counter can count one era twice.**
   - `era_index` comes from `cur.index` (`rolling.py:1004`), but the
     sentinel comes from `era_start_block(block)` (`lane_b_driver.py:493`).
   - While the era roll is held, era k+1's draw is counted under k, then
     again after the roll. The bound is reached an era early.
   - Fix: key the count by the era-start block.
3. **WB18: armed suite targets add four slots that nothing can judge.**
   - `suite_targets = true` adds `suite:{gift,boom,time,fev}` to `targets`
     and the emission slots.
   - Nothing judges a suite target: `affinity_contexts`
     (`validator/expert_lane.py:93-113`) has no `suite:` branch, and W-S2 is
     unbuilt.
   - File a WP before SB3, or arming dilutes every domain slot.
4. **W18.**
   - (a) In live mode `--now` is caller-chosen (`vault_sweeper.py:992`,
     `:997-1006`). A future `--now` expires archives inside the 14-day
     dispute window, and on fs it deletes not-yet-expired locks. Refuse a
     skewed `--now` under `--live`.
   - (b) The ledger (`laned_vault_ledger/1`) is unsigned and not bound to
     `lane_d_state.json`. It also has no status for the house, a prior king
     in the paid court, or `confirm_pending`. List these as keep reasons in
     the ledger contract when the producer is assigned (with the known item).
   - (c) Every version of a key is deleted, but only the current one is
     archived and verified (`:705-708`). Either archive per version or
     record that as accepted.
   - (d) L4 evidence (`tests/accept/w18/helpers.py:787-835`) is
     self-consistency only, and an empty `actions` list passes. This is the
     same class as B10. Require a signed or re-verified record with at least
     one archive-then-delete pair.
5. **W7c.**
   - (a) Recovery depends on identical inputs. A changed `uid_of`, entries,
     house or pool gives `report_conflict`, and that boundary wedges (fail
     closed). W19 must pass inputs pinned to the boundary block; add that to
     the W19 row.
   - (b) `verify_receipt_chain` (`lane_d_state.py:302-320`) doesn't anchor
     the first prior to `state_digest(genesis(house))`. Add a
     `genesis_sha256` argument.
   - (c) The recovery and re-call paths (`lane_d.py:727`, `:771`) return
     the persisted report's signature without verifying it. Verify it
     against `pinned_report_signer`, or re-sign.
   - (d) The receipt embedding of prior/posterior is deferred to W8/W14 by
     the README, but neither row names it. Give it an owner.
   - (e) Otherwise sound:
     - canonical bytes are exact;
     - duplicate keys are refused at any depth (`object_pairs_hook`), which
       closes RV5 note 3(c);
     - the write is atomic (tmp, fsync, `os.replace`, dir fsync), and the
       report lands before the state;
     - `report_sha256` hashes `canonical_body`, not the randomized
       signature (RV5 note 3(b));
     - the re-call path re-scores nothing, and an older boundary gives
       `stale_boundary`;
     - runs are byte-identical across hash seeds.
6. **WB11/WB13.**
   - (a) `MAX_EDGE_VALUES = 12` (`contamination.py:169`) is a stated
     false-negative bound: a 13th near-edge value isn't probed. Document it
     next to TOLERANCE.
   - (b) The repo-wide `.py` constant stream builds windows that straddle
     unrelated constants. That adds false positives (B5) and no detection.
     Prefer per-literal runs plus the scalar-assignment case.
   - (c) Sound:
     - no miner code runs: `ast.parse`, `json.loads` and bytes only;
     - symlinks aren't followed, and RecursionError/MemoryError fall back
       to byte decoding;
     - refs are content-pinned;
     - it fails closed on a missing or mismatched artefact, a fetch error
       or a scan exception (503, nothing queued);
     - the refusal is typed `benchmark_series_embedded` with the receipt
       shape the row asks for;
     - schema-1 artefacts are refused;
     - the build is deterministic: pure-Python IEEE, no `hash()`, canonical
       JSON, gzip mtime 0, and `verify --input` byte-compares.
   - (d) The plan's `.json.zst` artefact name is refused by the loader
     (`:329-330`). Align the name in the plan or the loader.
7. **W3b3 #599: RV5 note 5 is partly open.** `era_block` isn't checked
   against the era containing `report["boundary"]`, and no role's
   `suite_identity` is recorded (`validator/lane_d_sentinel.py:206-256`).
8. **W0c #602 residual: `scripts/laned/verification_wps` is loop-editable.**
   A W0c PR, which the lane D loop auto-merges, can add a WP to it. RV5
   asked for the owner-merged plan row to declare verification WPs. Either
   read a `verification` token from the §3.0 row at the base, or have the
   owner confirm the file as owner-only.

Verdict: NOT ACCEPTED. There are 10 BLOCKING items:

- W18: 2 (B1, B2);
- W7c: 1 (B3);
- WB18: 1 (B4);
- WB13/WB11: 4 (B5–B8), all of which block arming `suite_targets` (SB3),
  not the unarmed live intake;
- W0c: 1 (B9, the RV5 stale-base item, still open);
- W3b2: 1 (B10, the RV5 evidence item, partly fixed).

WB17 has no BLOCKING item and may reach L5 once at L3. Note 1 must be fixed
before an overlay selects `tokens_king_check`. Of the RV5 items, W3b3's and
two of W0c's three are resolved. W5c's stays open as known.
