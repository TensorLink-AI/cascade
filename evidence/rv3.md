# RV3: lane D review of W5a, W4, W8, W-S1, W5d, R5, R10

Covers: W5a, W4, W8, W-S1, W5d, R5, R10

Reviewer: RV3 (fresh context, read-only on main). Reviewed at main
`e617901` (W5d #552). In scope: W5a #544, W4 #548, W8 #549, W-S1 #545,
W5d #552, and the RV2 fixes R5 #551 and R10 #533. The spec is
docs/plans/LANE_D_PLAN.md: the §3.0 rows, §3.1, §3.2, §16, §17, the
accepted "Proposed design changes" (rev5: 1-17, 20, 21 accepted) and each
WP's `tests/accept/<wp>/`. I read the code itself, not the PR bodies.

Checks run, single directories at `nice -n 19`, with
`python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -m 'not torch'`:

- w5a 30 passed. w8 43 passed. w4 62 passed (7 torch deselected).
  w5d 45 passed, 2 skipped (`$LANED_EVIDENCE` unset), 5 torch deselected.
  w_s1 133 passed. r5 124 passed, 1 skipped (no shellcheck on this box).
  r10 70 passed, 1 skipped (the live T25 check, `$LANED_EVIDENCE` unset).
- `stub_scan.py --root <wt> --paths-file tests/accept/<wp>/PATHS` for w5a,
  w4, w8, w_s1, w5d, r5 and r10: 0 findings each (2, 4, 3, 16, 2, 7 and 2
  files). Reviewer pitfall: without `--root` the scan resolves PATHS
  against the cwd and reports "0 finding(s) in 0 file(s)" for a worktree.
  `accept` runs from the repo root, so CI is unaffected.
- `ls` of `/root/cascade` and `/root/cascade/_train_work` (names only, no
  file opened) for the R5 snapshot check.

## RV2 BLOCKING items: re-check

- RV2 R5 (sqlite3 snapshot, missing `activation_state.json`) is resolved.
  `deploy/laned/lib.sh:32-36` lists `activation_state.json` and the three
  `validator_state.db.*.json` sidecars. Every file is copied with `cp -a`
  (`lib.sh:356-366`), and its `.bak` is copied too (`laned_snapshot_bak`,
  `:345-350`). No `sqlite3` is left under `deploy/laned/`. The r5 README
  (`:132-135`) and `helpers.path_without_sqlite3` pin that.
- RV2 R10 (hand-written `evidence/t25.json` reaches L4) is resolved.
  `tests/accept/r10/test_r10_live_evidence.py:50-66` runs the real
  `python -P -m cascade.laned.restore verify-evidence` on
  `$LANED_EVIDENCE` and asserts exit 0 and `OK T25 evidence`. It skips only
  when the variable is unset. The self-tests check that a bare
  `{"wp", "passed"}` file fails.

## BLOCKING

- [ ] BLOCKING W6b/W9 (fix rows in W5a, W6b and W9): forward-only (invariant 4) fails open when a series has no `frontier` key, and production can hit it.
  - **The code.** `cascade/validator/lane_d_forward.py:111-116` returns
    *eligible* once the two clocks pass, whenever `w.get("frontier") is None`.
    That skips rule 5 (`absent_from_s0`) and rule 6 (`backfill`).
    `cascade/validator/lane_d.py:95-100` (`_parse_frontier`) maps both an
    absent key and an explicit JSON `null` to `None`. So `"frontier": null`
    fails open as well. Only `[]` is refused.
  - **The feeds block fails open the same way.** A missing top-level `feeds`
    block, or a feed that isn't listed, means "lag unmeasured, not a
    forecast product" (`lane_d.py:72-75`, `lane_d_forward.py:97-102`). That
    silently turns off `over_lag` and `forecast_product`.
  - **The spec it breaks.** Accepted change 1(a) makes a window eligible only
    if "the series is present in S0 with a recorded `end_ts`". A keyless
    series has no evidence of being in S0, yet it is admitted.
  - **Why production can hit it.** The safety argument is only a docstring
    (`lane_d_forward.py:28-34`: "W6b must always write `frontier` … the audit
    (W9) refuses a production snapshot whose series lack it"). No plan row
    makes it binding:
    - the W6b row (§3.0, plan line 1501) names only "per-series `end_ts` /
      `first_target_ts`";
    - the W9 row (line 1516) names only "both-clock eligibility";
    - accepted change 2 assigns the feed metadata to W6b, but the row
      doesn't list `feeds`;
    - `load_pool` accepts any `pool_d/1` snapshot with no frontier
      (`lane_d.py:118-119`).

    So a W6b builder that meets its own row, or one that drops `frontier`
    for some series (a newcomer series, a source with no prior snapshot, a
    `null` from a serializer), feeds the main validator windows that contain
    backfilled or pre-T data. The settlement crowns on them. W9 would flag
    it only after the fact, and only if its row is amended.
  - **The tests pin the fail-open.**
    `tests/accept/w5a/test_w5a_eligibility.py:248-254`
    (`test_series_without_frontier_key_is_judged_on_the_two_clocks`) and the
    `"legacy"` case in `:79-90` assert that a keyless series is eligible.
  - **Fix.**
    - **W5a, tests-only PR first (rule 1), then the implementation.** A
      series whose `frontier` is absent or `null` is refused with a new
      typed reason (`missing_frontier`, or `absent_from_s0`). A feed with
      no `feeds` entry is refused as `over_lag` with "unmeasured", unless
      the snapshot declares the feed `mirror_lag_hours: null` explicitly.
      Replace the `legacy` test with one asserting the refusal. If the W1.5
      fixture must keep working, regenerate it with `frontier`. Don't keep a
      keyless path in `cascade/validator/`.
    - **W6b row, add:** "writes `frontier` (a list, possibly empty) for
      every series and a `feeds` entry (`mirror_lag_hours` number or
      `null`, `forecast_product`) for every feed in the snapshot. The
      builder's tests assert that no series or feed lacks them."
    - **W9 row, add:** "refuses a judged snapshot in which any series lacks
      `frontier` or any feed lacks a `feeds` entry. It re-derives S0 and
      the backfill rule from the snapshot without importing
      `cascade.validator.lane_d_forward` (invariant 12)."

- [ ] BLOCKING W5d: the T7 kit can certify a context variant that it never exercised.
  - **What the kit records.** `cascade/eval/lane_d_t7.py:147-170`
    (`measure_variant`) times `forecast_quantiles` at `context=variant` on
    whatever histories it is given. The no-pad plan truncates each history
    to `min(n, context)` (`lane_d_scorer.py:95-97`). So a "16384" or
    "131072" row timed on short histories measures a model that runs at
    ≤ 4096 tokens. Nothing in the row, `variant_row` (`:99-110`), or the
    evidence (`build_evidence`, `:113-136`) records the input lengths that
    actually ran.
  - **The verifier can't catch it.** `evidence_problems`
    (`tests/accept/w5d/helpers.py:249-292`) has nothing to check. The repo's
    own tiny-model test times `context=8192` on histories of 64/900/4096/6000
    points (`tests/accept/w5d/test_w5d_t7_evidence.py:274-290`), so no window
    reaches 8192.
  - **The memory number is wrong on CPU.** It is `ru_maxrss` (`:139-144`),
    the peak over the whole process, so every row after the largest one
    inherits that peak.
  - **Why it matters.** T7 decides which `[lane_d] contexts` are armed
    (≤ 50% of a settlement's wall time, §1.6). An under-measured 8k/16k row
    could arm a context that overruns settlements.
  - **Fix (W5d PATHS, `cascade/eval/lane_d_t7.py`, with the tests-only PR
    first for `tests/accept/w5d/`):**
    - `measure_variant` refuses unless every history has at least `variant`
      points.
    - The row records `input_len` (the `batch_plan` key that ran) and
      `n_windows_at_variant`.
    - On CPU, measure peak memory per variant in a fresh subprocess, or
      label it `process_peak_rss` and refuse it as a budget input.
    - `evidence_problems` requires `input_len == variant` on every row.

## Non-blocking notes

1. **W-S1 (unarmed ⇒ byte-identical): confirmed.** With
   `sentinel_from_block = 0`:
   - The validator draws nothing. `_draw_settlement_sentinel` returns before
     any `client.block_hash` call (`cascade/validator/loop.py:892-908`).
   - The gate and the king check get `{}` (`loop.py:915-926`,
     `evaluation.py:26-30`, `king_check.py:420-424`).
   - The receipt field is drop-when-default (`receipt.py:727-728`).
     `_receipt_sentinel` reads no chain (`receipts.py:151-173`).
   - The audit adds no row: `check_sentinel` runs only when
     `_sentinel_in_play` (`audit/checks.py:4228-4230`).
   - The satellite returns before any read (`satellite.py:203-205`).
   - `evaluate_bench_gate(sentinel=None)` takes the untouched path
     (`bench_gate.py:244-270`).

   The goldens from main `0bad8b0` (gate detail, king-check detail, scored
   and rejected receipt bodies) pin this, and the suite passes. One small
   point: the validator and the receipt compute `settlement_era_start`
   before the armed check. That is pure arithmetic
   (`era.settlement_contract_block`), but checking
   `sentinel_from_block(cfg) == 0` first would make "no new code runs while
   unarmed" literal.
2. **W-S1 (mixed fleet).** This is the same pattern as RV1 note 10, with one
   more key family:
   - `ALLOWED["v2"]` now admits the four `sentinel_*` keys
     (`netconfig.py:111-112`). A pre-W-S1 node refuses an overlay that
     carries them, and it refuses the whole overlay. So arming by overlay
     before the fleet upgrades splits the fleet.
   - A pre-W-S1 satellite still verifies an armed receipt's signature,
     because it uses the received body (`receipt.py:1220`). But its replay
     re-runs the gate on all four suites. The detail then mismatches, and
     every armed settlement with a gate or king check fails replay there.

   The R7 publish wrapper should refuse a `sentinel_from_block` overlay
   unless R4 shows the `cfg-` stake ≥ 2/3 **and** the known satellites
   report the W-S1 release.
3. **W-S1 (audit independence).** The audit and the satellite re-derive
   the draw with the producer's own `sentinel_for_era` / `verify_record`
   (`audit/checks.py:1521`, `satellite.py:200`). They replay the gate
   through the producer's `evaluate_bench_gate` / `king_check_gate`
   (`checks.py:1667-1672`, `:2190-2194`). That is the existing lane A
   pattern: the DEC-CA-0061 gate is shared consensus code, and lane D
   invariant 12 doesn't cover it. It does mean a wrong rule in `sentinel.py`
   can't be caught by replay. The TS5 golden draw vectors in
   `tests/accept/w_s1/` are the only independent anchor, so keep them
   hand-pinned.
4. **W-S1 (trust when the hash is unreadable).** When the satellite can't
   read the era-start hash (`satellite.py:206-212`) or the audit has no
   client (`checks.py:1546-1553`), the recorded suite is replayed. A main
   validator could then pick the held-out suite, unchecked, for any
   observer that can't read the chain. That is acceptable under TS1, but
   the audit row says so only in its detail text. Consider WARN rather
   than OK for "recorded suite replayed".
5. **W5a (T8 wording vs §1.3).** The §4 T8 row (plan line 1715) swaps the
   labels. It says "data-time-only pass but first-seen fail (late mirror);
   first-seen-only pass but data-time fail (publish-ahead)". §1.3, the code
   (`lane_d_forward.py:103-110`) and the tests
   (`test_w5a_eligibility.py:100-110`, whose docstring notes the swap) all
   use the opposite, correct mapping:
   - first-seen fails ⇒ `publish_ahead`;
   - data time fails ⇒ `late_mirror`.

   Both cases are refused either way, so only the reason label is at stake.
   Fix the T8 row text (an owner plan edit) so W9's independent
   re-derivation doesn't copy the wrong labels. Also,
   `tests/accept/w5a/README.md:54-55` says "rule 5 only" / "rule 4 only",
   but the module docstring numbers both as rule 4.
6. **W5a (removed `forward_eligible`).** There are no stale callers. The only
   reference left is a comment in `tests/accept/w5a/helpers.py:126`.
   `_window_set` now pairs pool windows with verdicts using
   `zip(strict=True)` (`lane_d.py:268-272`), so the two can't misalign.
   `settle_lane_d` still has no production caller. W7c wires it.
7. **W5a (unmeasured lag).** Even with the BLOCKING fix, "listed with
   `mirror_lag_hours: null` ⇒ not a refusal" (`lane_d_forward.py:100-102`)
   is a policy choice. §1.3 says "a feed above the limit is never eligible"
   and is silent on unmeasured feeds. The forecast-product and publish-ahead
   cases are still caught by the first-seen clock. Record the choice in
   the decision register.
8. **W8 (unarmed and `share = 0`): confirmed byte-identical.** With
   `lane_d_cfg=None`, `pool_a = 1.0 - b - c - 0.0` (`emission.py:370-371`),
   which is bit-equal to the old `1.0 - b - c`. With `takes_from =
   lane_a`, the old `(1-b-c) - min(want, max(0, 1-b-c))` and the new
   `((1-b)-c)-d` are the same float operations. `lane_d_vector(share=0)`
   returns zeros (`:410-412`).
9. **W8 (takes_from = lane_b / lane_c is NOT bit-identical to the removed
   helper).** The old `lane_d_allocation` kept `pool_a = 1 - b - c` computed
   from the original `b`. The new code computes `1 - (b-d) - c - d`, which
   can differ in the last ulp. Mainnet uses `lane_a` and nothing is armed,
   so there's no live effect. But the commit message's "same float
   operations" claim holds only for `lane_a`. W9 must use the new formula.
   Otherwise, compute pool A once from the untouched shares.
10. **W8 (wiring trap for W7c/W9).** No production caller passes
    `lane_d_cfg` yet (`validator/weights.py:517-524`,
    `audit/checks.py:2834-2841`, `:3660`). Both callers pass the second
    element to `emission_vector` only when `pay is not None or pay_c is not
    None`. If lane D pays while B and C are off, lane D's part would be
    dropped, and `set_weights` normalisation would hand it to lane A. The WP
    that wires lane D must add the lane D pay record to that condition at
    both sites.
11. **W8 / invariant 12 (for W9).** Lane D's share and court math now lives
    in the shared `lane_pool_shares` / `lane_parts` / `lane_d_vector`, and
    the lane A/B/C audit already imports `lane_parts`. The W9 row requires
    a `derive_weights` that is "independent of producer code". Either W9
    re-derives lane D's carve and court vector without these functions, or
    the plan states that they are shared consensus code exempt from
    invariant 12. Decide before W9's tests are written.
12. **W4 (invariant 3): no miner bytes execute.** Steps 1-7 are stdlib only.
    The header is parsed by hand (`lane_d_ingest.py:250-302`), with offsets,
    contiguity and exact coverage checked. The code files must be
    byte-equal to the repo's (`:223-228`). Step 8 opens the file with
    `safetensors.safe_open(framework="pt")` (`checkpoint_guard.py:186-195`),
    which doesn't unpickle. The scorer imports the repo's `pyramid_model`,
    never the checkpoint's (`lane_d_scorer.py:117-131`). The fresh
    interpreter test confirms that torch never loads before step 8. One
    residual: `_read_header` uses `json.loads`, where a duplicated key means
    the last one wins. Refusing duplicate keys (`object_pairs_hook`) would
    rule out any disagreement with the Rust safetensors parser.
13. **W4 (presigned credentials): no leak found.**
    - The request carries only `Accept-Encoding` (`storage.py:131-145`).
    - No environment credential is read.
    - Redirects are limited to http(s) (`:116-123`).
    - Every message is `_redact`ed (the query string is dropped), and the
      urllib errors are raised `from None`.

    Note that `build_opener` still installs the default `ProxyHandler`, so
    an `https_proxy` on the pod sees the full signed URL. Tell operators.
14. **W4 (partial files).** `fetch_presigned` writes `.<name>.part-<pid>` and
    unlinks it on any exception (`storage.py:196-205`). `_staged` publishes
    the directory by `os.replace` only after every check passes
    (`lane_d_ingest.py:356-384`). Two gaps remain:
    - A SIGKILL leaves `.<digest>.staging-<pid>` behind. Only a later run
      with the same pid cleans it, so sweep `.*.staging-*` at intake start.
    - When `final` exists, it is `rmtree`'d before the replace. Readers see
      no directory for a moment. Rename the old one aside instead.
15. **W4 (storage transport memory).** `ingest_entry` reads each object
    whole with `storage.get()` (`:421`) before the size comparison. The
    earlier `check_object_list` size check bounds it, but only up to the
    time between the size check and the read. On the 7 GB box, a
    weights-sized `get` is a real allocation. Stream the read, or cap it.
16. **W4 (dead and uncalled code).** `checkpoint_guard.verify_lane_d_checkpoint`
    (W1.5) now has no caller: lane D ingest uses its own pipeline. It sits
    in a CONSENSUS-listed file, so delete it, or have ingest call it.
    `ingest_entry`, `ingest_presigned`, `expectation_for`, `check_handshake`
    and `with_handshake` have no production caller yet (W7c/intake).
17. **W4 (handshake scope).** The hello is self-reported
    (`lane_d_handshake.py:123-127`). It catches a stale or hot-patched host
    that reports honestly (T19, T28). It is not attestation. Also,
    `package_hash` hashes every non-bytecode file under `cascade/`
    (`:61-86`), so a stray file (an editor swap file, `.DS_Store`) on the
    eval host fails the handshake. That fails closed, which is the safe
    direction. Note it in the eval host runbook.
18. **W5d (no-pad vs lane A's `pyramid_wrapper`).**
    - **Where they diverge.** Lane A edge-pads a short history with
      `row[0]` to `context_length` (`trainer/pyramid_wrapper.py:29-39`).
      Lane D drops the `n % 32` oldest points and runs at the real length
      (`lane_d_scorer.py:95-114`). The rest of the forward
      (standardisation, mask, `start = length//ps - 1`, the slice of loc and
      scale) matches line for line (`lane_d_scorer.py:134-157` vs
      `pyramid_wrapper.py:41-57`).
    - **What the parity test covers.**
      `test_house_at_4096_equals_its_lane_a_per_window_crps` uses only
      histories of ≥ 4096 points (`test_w5d_scorer.py:322-323`). The W5d
      row's "the house scored by the lane D scorer at 4096 equals its lane A
      per-window CRPS" holds only on such windows. On shorter windows the
      house scores differently from its lane A score, by design.
    - **What to do.** Qualify the row text. Make sure W6b's history
      extension (16384 + 64) keeps short-history windows rare, or report
      their count.
19. **W5d (torch guards and goldens): no defect.**
    - No golden is computed under a `sys.modules["torch"] = None` guard in
      W4, WB14 or W5d.
    - W5d's `no_torch` fixture (`test_w5d_scorer.py:45-49`) and the T7
      `short_window_check` test (`test_w5d_t7_evidence.py:270-275`) wrap
      only refusals that raise `ValueError` before any numpy or scipy
      scoring runs.
    - WB14 uses a `meta_path` blocker that raises `ModuleNotFoundError`
      (`tests/accept/wb14/helpers.py:50-67`). That is the same as torch not
      being installed, so scipy's probe isn't broken.
    - W4 checks torch absence in a fresh subprocess
      (`test_w4_ingest.py:239-276`).
    - W-S1's README and helpers warn against recomputing goldens under the
      guard.
20. **W5d (the scorer trusts its caller).** `forecast_quantiles` reads
    `config.json["moe"]` and the weights from `checkpoint_dir` with no check
    that the directory came from ingest (`lane_d_scorer.py:123-130`). That's
    fine while the only callers sit behind ingest. W7c should pass only
    `ingest_*` return paths.
21. **R5 (snapshot vs the live box layout, names only).**
    - **Present in `/root/cascade` and snapshotted:**
      - `validator_state.db` (+`.bak`, +`.settlement.json`);
      - `cascade_state.json` (+`.bak`);
      - `warm_start_init.json`;
      - `activation_state.json` (+`.bak`);
      - five `activation_*_state.json` (+ their `.bak` where one exists).
    - **Absent today, reported as absent:** `lane_d_state.json`,
      `netconfig_state.json`, `satellite_state.json`, `decided_votes.toml`,
      the `.lane_b`/`.lane_c` sidecars.
    - **Not snapshotted, although R5 restarts the trainer:** the trainer's
      state under `/root/cascade/_train_work/`. That includes:
      - `era_state.json` and `trainer_submissions.json` (the submissions db
        CLAUDE.md says to snapshot before touching);
      - `funded_queue.json`, `trainer_promotion.json` and
        `champion_publisher.json`;
      - `lane_b_driver.json`, `coldkey_identities.json` and
        `published_settlements.json`;
      - its own `activation_*` and `netconfig` files.
    - **Also not snapshotted:** `provisioner_state.json`,
      `pod_monitor_state.json`, and the deployed `chain.toml`, which carries
      live overrides, together with `hosts.toml` / `provision.toml`.
    - **Why this is a note.** §10.8's list names only validator files, so
      R5 matches the plan as written. The capability release changes no
      trainer behaviour, so a rollback is a forward-fix.
    - **Recommendation for §10.8 and R5 before S5.** When the trainer is in
      the restart set, add the `_train_work` JSON state and the deployed
      config files to the snapshot.
22. **R5 / R10 (two inventories).** R5's `LANED_STATE_FIXED` and
    `cascade/laned/restore.py` (`ACTIVATION_MAIN`, `ACTIVATION_GLOB`,
    `activation_paths`) still list the state files separately. R10's bundle
    still leaves out the `.lane_b.json` / `.lane_c.json` sidecars (RV2
    note 12, still open). One shared inventory module would stop the drift
    that caused the RV2 R5 item.
23. **Invariant 12 overall.** No in-scope audit or satellite code imports
    lane D producer eligibility, verdict or weight code:
    - `cascade/audit/` and `satellite.py` don't import `lane_d_forward`,
      `lane_d`, `lane_d_scorer` or `lane_d_ingest`;
    - the W-S1 additions import only `shared.sentinel`, `eval.bench_gate`
      and the existing `king_check_gate` (note 3).

    The open risks are forward-looking: W9 must not import `lane_d_forward`
    (BLOCKING item 1) or reuse the shared lane D weight math without a plan
    exemption (note 11).
24. **Invariant 10.** W-S1 pairs its unarmed goldens with armed tests whose
    effect is non-empty: the sentinel suite is dropped from `suites`, the
    receipt sha differs, the ratchet blocks. W8 pairs the `lane_d_cfg=None`
    identity with armed carve tests. W5a, W4 and W5d have no
    unarmed-identity tests, so the pairing rule doesn't apply to them.
