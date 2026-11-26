# RV5: lane D review of W3b2, W5c, W7b, W3b4, W3b3, WB15, W0b

Covers: W3b2, W5c, W7b, W3b4, W3b3, WB15, W0b

Reviewer: RV5 (fresh context, read-only on main). Reviewed at main
`214add8` (#595). In scope:

- W3b2 #574 (tests #569);
- W5c #579 (tests #567);
- W7b #583 (tests #566, #573, #585; W1.5 test fix #582);
- W3b4 #592 (tests #588);
- W3b3 #593 (tests #587);
- WB15 #595 (tests #589);
- W0b #575 (verification WP, tests only);
- RV4 fixes merged since: W3a #578/#580, R8 #584, W5d #571, W5a #568;
- loop tooling: W0c #570 (verification mode) and #576 (cross-WP step, KNOWN).

The spec is docs/plans/LANE_D_PLAN.md: the §3.0 rows, §3.1, §3.2, §1.3,
§1.3b, §1.4, §1.5, §12b.2, §16 and §17, plus each WP's `tests/accept/<wp>/`.
I read the code, not the PR bodies.

Known open items, not re-reported: the W7a schema impl (blocked on the owner
edit to `cascade/audit/lane_d.py`); W5b RV4 impl held on `laned/w5b-rv4-impl`;
W5e and W3b5 unbuilt; W3b3 not called from the validator post-settlement
path; WB15 `routing_stale` vs declared legs. The "settle_lane_d without gate
kwargs" item is assessed below, as asked.

Checks run, one directory at a time at `nice -n 19`, with
`env PYTHONPATH=<wt> python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -m 'not torch'`
(`cascade.__file__` checked to be the worktree):

- w3b2: 84 passed, 2 skipped (`$LANED_EVIDENCE` unset). w3b4: 128 passed.
  w5c: 15 passed. w7b: 95 passed. wb15: 18 passed. w0b: 3 passed, 13
  deselected (torch). w3a: 124 passed, 6 skipped. w1_5: 15 passed, 10
  deselected.
- **w3b3: 41 failed, 1 passed** (BLOCKING item 1).
- w5b: 3 failed (`test_w5b_versions.py`, the RV4 tests from #581; the impl is
  the known held branch). So W5b is red on main and sits at L2.
- CI on main agrees. The laned-accept push runs at `885d37a` (#593) and
  `214add8` (#595) record `W3b3 accept push failure`, so the board holds W3b3
  at L2.
- `stub_scan.py` over each in-scope `PATHS`: 0 findings (w3b2 1 file, w3b4 2,
  w3b3 1, w5c 2, w7b 2, wb15 3, w0b 1).
- Two scratch reproductions against the worktree: the W5c audit mismatch
  (item 4) and the cross-step eligibility of W0b (item 3).

## RV4 BLOCKING items: re-check

- **W7a (no `gate` / `sentinel` / `gate_weights` in the schema): open.**
  This is the known W7a item. W5c now makes it worse: see item 4.
- **W5b (window scored from the newest snapshot's version): open.** The
  tests (#581) are on main and red. The impl is the held branch.
- **W5b (report binds only the newest snapshot): open.** Same as above.
- **W3a (default-retention bucket): resolved by #578/#580.**
  - `bucket_lock_preflight` (`cascade/laned/storage_probe.py:324-363`) reads
    `GetObjectLockConfiguration` before any write.
  - It refuses a `DefaultRetention` with `bucket_default_retention`, and an
    unreadable answer with `bucket_lock_unreadable` (fail closed).
- **W3a (L4 from relabelled fixture evidence): resolved as RV4 asked, with a
  residual risk** (note 9).
  - A read-only `storage_probe verify` (`:1406-1445`) re-reads every locked
    version, checking HEAD, retention and the GET sha256, and lists the run
    root.
  - The L4 test requires its `verified_live` record.

## BLOCKING

- [ ] BLOCKING W3b3: `cascade/validator/lane_d_sentinel.py` doesn't import on main, so the sentinel step can't run.
  - **The code.** `lane_d_sentinel.py:36` reads
    `from cascade.validator.lane_d_bench import METRICS, _suite_rows`.
    W3b4 (#592, merged 05:34) rewrote `lane_d_bench.py`, and `_suite_rows`
    no longer exists. W3b3 (#593) was built and checked on base `089de36`,
    which is before #592, and merged at 05:43 without a re-run.
  - **Result.** `ImportError: cannot import name '_suite_rows'`. 41 of the 42
    W3b3 tests fail, locally and on the main push runs. No production code
    imports the module, so nothing live breaks. But W3b3 is at L2 with
    unrunnable code, and the §16 "swept and published for every new king"
    rule has no working implementation.
  - **Fix (W3b3 impl PR).** Rebuild `_model` on W3b4's public surface.
    `classify_suite` / `paired_gate`'s `_classify` row extraction gives the
    per-role rows and fault codes, so the sentinel and the gate classify
    faults identically.
    - Don't import private (`_`-prefixed) names across WP modules.
    - Also see note 5: the record should carry paired numbers and the suite
      identity.

- [ ] BLOCKING W0c (#576 / workflow): a laned PR can merge green on a stale base and break a built WP. #593 did exactly that.
  - **The gap.**
    - The PR job's dependent-WP step (`accept --cross --base HEAD^1`) runs
      once, against main as of the run.
    - `main` has no branch protection and no rulesets
      (`gh api …/branches/main/protection` gives 404, rulesets `[]`). So
      nothing requires the branch to be up to date or re-runs checks after
      main moves.
    - The loop merges on a green badge.
  - **Failure scenario (happened).** #592 changed `lane_d_bench.py` while
    W3b3 was at L1, so W3b3 wasn't in #592's cross set. #593's own Accept
    had passed against the pre-#592 base, and it merged 9 minutes later.
    Main went red, and #595 then merged on top (W3b3 was KNOWN-red).
  - **Fix (W0c).** Either:
    - make the merge step (or a required check) re-run `check-pr`, the WP's
      Accept and `--cross` on the PR merged into the *current* main
      immediately before merging, and refuse when the base it ran on isn't
      `origin/main`; or
    - turn on "require branches to be up to date" plus the laned-accept
      required check for `laned:*` PRs.

    In both cases, add a selftest where a second PR, green on an older base,
    is refused after the first one merges.

- [ ] BLOCKING W0c (#576): the cross-WP step never re-accepts a verification WP, so a PR can break W0b and still merge.
  - **The code.** `built_wps_at` (`scripts/laned/accept:725-740`) keeps a
    WP only when `lanedlib.impl_commit_index` finds an implementation
    commit. A verification WP (#570) has none: its L2 commit comes from
    `verification_commit_index`, which the board uses but `--cross` does
    not.
  - **Reproduced.** I loaded `scripts/laned/accept` and called
    `built_wps_at(<wt>, "HEAD")`. W0b is **not** cross-eligible; W5d, W3b3,
    W3b4, W5c, W7b and WB15 are.
  - **Failure scenario.** A W5d follow-up changes the lane D scorer's
    forward (for example, it reintroduces edge padding). W0b's PATHS is
    exactly `cascade/eval/lane_d_scorer.py`, so its context-equivalence
    tests would fail. They don't run on the PR. The PR merges, and W0b drops
    from L3 to L2 only on the next push run.
  - **Fix.** `built_wps_at` uses the board's L2 rule: an implementation
    commit **or** a verification commit (tests-tree match against the
    ledger, or, without the ledger, the first tests-only commit after PATHS
    for a WP whose non-test PATHS exist). Add a selftest with a verification
    WP in the cross set.

- [ ] BLOCKING W0c (#570): verification mode isn't limited to verification WPs, so any WP can reach L2 with no implementation.
  - **The code.** `cmd_verification` (`scripts/laned/accept:366-395`) gives
    `MODE verification` (L1 and L2 in one merge) to *any* tests-only PR
    when:
    - the base has no L2 commit for the WP;
    - every non-test PATHS entry already exists on the base;
    - the Accept passes on the base.

    It never checks that the plan row calls the WP a verification WP. It
    also skips §17.1 L1's "tests fail on main before L2" for every WP that
    takes this path.
  - **Failure scenario.** W5e's PATHS are all existing files
    (`lane_d.py`, `lane_d_frozen.py`, `lane_d_report.py`, the schema). A
    W5e (or W5f, W5g, W17 …) tests-only PR whose tests happen to pass on main
    goes straight to L2. That could be tests that only re-assert W5b/W5c
    behaviour, or tests whose gate assertions are guarded by
    `if report.get("gate")`. The push run then gives L3 and the review
    gives L5, with no implementation PR ever opened. The level board then
    reports a built gate wiring that doesn't exist, which is the "faked
    completion" §3.1 rule 8 forbids.
  - **Fix.** Allow verification mode only for WPs that the owner-merged
    plan marks as verification WPs. For example, a `verification` token in
    the §3.0 row, read with `plan_at(base)` the way `wp_ids_at` reads the
    table. Put W0b in that list. The test author must not be able to
    self-declare it (that rules out a marker under `tests/accept/`). Add a
    selftest: a non-verification WP whose tests pass on base gets
    `expect-fail` semantics and fails (nothing fails for a right reason).

- [ ] BLOCKING W5c: change-3 reports can't be verified. The in-repo audit FAILs every report where a miner holds KING_D, and nothing signed records the house comparison.
  - **The code.**
    - When a miner holds KING_D, `settle_lane_d` scores the house as a
      second bar and writes its rows (`cascade/validator/lane_d.py:554,
      580`).
    - It sets `cleared = cleared_vs_king and cleared_vs_house`
      (`:604-607`).
    - The verdict still reports only the vs-KING_D `lcb` / `gain`. The
      cohort doesn't name the house bar, and there is no house LCB or
      house gain in the report.
  - **Reproduced** with W5c's own `_duel` fixture, then `check_lane_d(report,
    cfg, signer_hotkey=signer)`:
    - house `bad`, Y `good` (Y crowned-path `confirm_pending`): `FAIL '72
      rows; the cohorts account for 48'`, because
      `cascade/audit/lane_d.py:190` counts `1 + members` rows per window.
    - house `sharp`, Y `good`: Y has `lcb = 0.4684 ≥ d_margin` and a
      positive gain, but `cleared = False`. Once the row count is fixed,
      `_cohort_problems` (`audit/lane_d.py:116-145`) re-derives
      `cleared = True` from the KING_D rows. It reports the honest report as
      tampered.
    - `validate_report` and `verify_report` both pass, so a satellite that
      checks only the signature accepts a verdict it can't re-derive.
  - **Failure scenario.** From the first settlement after a miner is
    crowned, every armed report fails `cascade-audit` (W1.5's "audit
    passes" no longer holds in this state). A validator could also flip a
    vs-house loss to a win, or the reverse. The signed rows would agree with
    either verdict, because no signed field says which digest was the house
    bar or what its LCB was (§1.5, §14.3 rule 2: "every value a satellite
    needs … is in the signed report").
  - **Why the tests miss it.** No W5c test runs `check_lane_d` or re-derives
    `cleared` from the report alone. A wrong implementation that ANDs with
    a random bit, or that uses a different seed for the house comparison,
    passes all 15.
  - **Fix (W5c + W7a schema + W9 audit, tests-only PRs first).**
    - Each cohort carries `bars: [{digest, role: "king" | "house"}]`.
    - Each verdict carries `vs_house: {lcb, gain, cleared}`, or `null` when
      the house holds KING_D.
    - The audit counts `len(bars) + len(members)` rows per window and
      re-derives both judgements.
    - Add a W5c test that every report `settle_lane_d` produces in the
      house scenarios passes `check_lane_d`.

- [ ] BLOCKING W3b2: W3b2's L4 pod evidence can be fabricated offline. It's the class of RV4's W3a item.
  - **The code.** `evidence_problems`
    (`tests/accept/w3b2/test_w3b2_evidence.py:51-120`) checks only internal
    consistency:
    - `source == "pod"`;
    - 40-hex `git_sha`, `git_dirty: false`;
    - one session string;
    - 64-hex digests;
    - ≥ 10 `ok` rows;
    - the W3b1 draw matches `draw.block_hash`;
    - `gate` equals `paired_gate` recomputed from the recorded rows.
  - **Failure scenario.** Take `helpers.triple()`'s rows (or any synthetic
    rows with ≥ 10 items), set `source: "pod"`, pick any block hash, and
    write `gate = paired_gate(...)`. The verifier accepts it, and W3b2
    reaches L4 with no pod run.
    - Nothing ties the item ids to the real suite: no check against a
      pinned `suite_identity`, which W3b4 now provides.
    - Nothing ties the digests to real published checkpoints (KING_D / the
      house), the block hash to the chain, or the host to an R12 rental.
  - **Fix (W3b2 tests-correction PR).** The verifier requires:
    - each role's `suite_identity(suite, entry)` to equal an identity pinned
      in the repo for the real suite revision;
    - `king_d` and `house` digests that resolve in the published champions
      index or king manifest;
    - `draw.block` / `block_hash` that match the chain (the owner records
      them, and they can be checked later);
    - R12 on-rent evidence for `host`.

    Where a live cross-check can't run in CI, the owner-run step must write
    a signed record (as in note 9).

## Assessed: `settle_lane_d` without gate kwargs (known item)

- **Reachable in production today?** No. `settle_lane_d`
  (`cascade/validator/lane_d.py:487-489`) takes no gate parameters. No
  module outside the lane D package and the tests calls it: a grep of
  `cascade/` finds only its own module, and the validator loop isn't wired
  (W19 and the W3b3 wiring are unassigned). `[lane_d] from_block` is 0.
- **But** every crowning on main is ungated, and the W5e test README pins
  that ("The W5b/W5c/W7a fixtures omit them and still expect a crown"). It
  also leaves "what a `bench=None` call does" unpinned. If W19 wires the
  call before W5e, or W5e keeps a no-gate default, an armed validator crowns
  with no §1.3b gate at all.
- **Must fail closed (BLOCKING on W5e's spec before its impl merges):**
  - an armed settlement that reaches a `confirm`-stage clear with
    `bench`/`boundary_block_hash` missing must produce `bench_hold` (no
    crown, not lost);
  - a report with `crowned` set and `gate: null` must be refused by
    `validate_report` (once W7a's `gate` exists), by W9 and by satellites;
  - the W5b/W5c/W7a fixtures that expect an ungated crown move to an
    explicit gated helper by tests-correction PRs.

## Non-blocking notes

1. **W3b2: a digest the host omits is blamed on the miner.**
   `SweepSession.paired` silently `continue`s past a digest the host didn't
   return (`lane_d_bench.py:520-523`). The candidate's record then has no
   suite, which `_status_defect` classifies as `absent` and therefore
   `checkpoint_fault` (entry `closed`, W5e table).
   - Within a session, KING_D and the house are cached, so the second
     candidate's host call is `[candidate]` alone. A host adapter that maps
     a pod reset to an empty answer turns an infra failure into the miner's
     fault.
   - The tests pin this (`test_host_returning_nothing_for_the_candidate_alone_is_its_fault`),
     so it's a spec choice. The R12/host adapter contract should state that
     a per-digest failure is reported as a `status` (`timeout`/`error` with
     detail), never omitted. Alternatively, `SweepSession` could mark an
     omitted requested digest as `timeout`.
2. **W5c: the cohort's `t_max` ignores the house's T** (`lane_d.py:372,
   609`). The house is now a bar, but the eligibility bound is
   max(T over the members and KING_D).
   - A house refreshed after the challenger's T is compared on windows that
     predate the house's T. Generator-embedded data could then favour the
     house.
   - This fails against the challenger (a wrongful loss), never open. Decide
     whether `T_house` enters `t_max` when the house is a bar (§1.3, D46).
3. **W7b is sound for what it claims.**
   - The canonical body is the report minus `signature`, with `signer`
     inside it: sorted keys, compact, ASCII, `allow_nan=False`.
   - The signature has one spelling (128 lower-case hex, enforced in the
     schema and in `verify_report`).
   - `verify_report` validates first and never raises on bad input, except
     `SignatureVerifierUnavailable` (DEC-CA-0051).
   - `pinned_report_signer` refuses a blank pin and never falls back to the
     self-declared signer. Mainnet and testnet pin different hotkeys
     (`chain.toml:1012`, `chain.testnet.toml:545`), so the missing netuid in
     the body isn't a cross-network replay path today.

   Gaps for the consumers (W7c/W7d):
   - (a) `verify_report` doesn't bind an expected `boundary`, so a valid old
     report replays at a later settlement unless the caller checks
     `report["boundary"]`.
   - (b) sr25519 signatures are randomized. Any digest W7c binds into
     receipts or `last_report` (invariant 5, restart idempotency) must hash
     `canonical_body`, not the signed bytes.
   - (c) `json.loads` keeps the last of duplicate keys, so the signed meaning
     is parser-dependent. Refuse duplicate keys when loading
     (`object_pairs_hook`).
4. **W3b4 is sound.**
   - The identity is (suite, version, sha256 of the sorted unique ids), with
     no `hash()` involved.
   - Scope-skips are exempt by recomputing coverage with
     `suite_incomplete`, never by trusting the flag.
   - A version or item-set mismatch on any role is an infra hold; a defect on
     a reference is never the miner's fault; a timeout outranks a candidate
     fault.
   - With no pin, the version majority rule is order-independent.
5. **W3b3 (when rebuilt): the record isn't paired.**
   - `_model` takes each role's geomean over its own items, so a partial
     reference and a full king yield comparable-looking numbers on different
     sets. The "sentinel falls for 3 kings" alert (§1.3b) needs paired
     ratios on the shared items, the way `_sentinel_record` in
     `lane_d_bench` already does it.
   - Record each role's `suite_identity`.
   - Check that `era_block` is the era containing `report["boundary"]`.
     Otherwise a wrong era silently records another era's sentinel.
6. **WB15: deferred claims hold nothing, so a later ESFT entry can take rows
   an earlier declared claim waits for.**
   - This follows the WB15 README (rule 6), but it weakens §12b.2's "claim
     order: earliest reveal". Example: declared D (reveal 100) is deferred
     behind E1, and in the same tick ESFT E2 (reveal 200) is planned over
     some of D's other rows. When E1 settles, D waits on E2, and so on.
   - The owner should confirm the intent: either deferred claims reserve
     rows against later-revealed entries, or the plan text says starvation
     is accepted.
   - Separately, `status()` now always emits `"deferred": []`. The driver is
     built whenever `[v2] from_block > 0` (`trainer/main.py:347`), so
     mainnet's `status/lane_b.json` gains the key while lane B is unarmed.
     It isn't consensus, but it isn't byte-identical either.
7. **WB15 is otherwise sound.**
   - The claim order is (funded `reveal_block`, hotkey).
   - Holders are open legs, finished-but-unsettled legs and pending merges,
     in either mode.
   - Deferral is uncharged: `mark_in_flight` is reached only past the
     overlap check.
   - ESFT excludes `owned_rows()` ∪ `_held_rows()`, including legs declared
     this tick.
   - Lane B isn't armed, and `tick` returns before planning when it's
     inactive.
8. **W0b is sound as a verification WP.** It compares quantile-head outputs
   at the 9-level grid against lane A's `load_quantile_forecaster` path, per
   window and batched, at `ATOL = 1e-4`. An edge-pad control shows the
   tolerance discriminates. The strict-load sweep spies on, and doesn't
   replace, `load_state_dict` at every `SCORER_CONTEXTS` value, with
   extra-key and missing-key negatives. The torch tests run in CI (green
   push runs on main).
9. **W3a residual: `verified_live` is self-attested.**
   `verified_live.body_sha256` binds the record to the evidence body, but it
   doesn't authenticate it. Anyone can write a plausible record, and the
   L4 test can't tell. The locked objects stay readable for the lock period,
   so a later re-verification is possible. Either have the owner's run sign
   the record (an operator hotkey), or let CI run `storage_probe verify`
   with read-only credentials at L4 time.
10. **W0c #576: KNOWN lets PRs merge on a red main.** #595 merged while W3b3
    was KNOWN-red. Comparing per test id is right for regressions. But the
    loop should not merge into a main whose last push run is red for a WP
    the PR doesn't fix, or it should at least file the red WP as blocked.
11. **R8 #584 and W5a #568 are sound.**
    - `evidence/` is its own no-restart AREA.
    - W5a fails closed: an absent or `null` `frontier` is `absent_from_s0`,
      and an unlisted feed or missing `feeds` block is `unlisted_feed`.
    - A listed feed with `mirror_lag_hours: null` still isn't a refusal.
      That's the policy choice RV3 noted, and it belongs in the decision
      register.
12. **Invariant 1 / lane A untouched.**
    - None of the in-scope PRs touch lane A scoring, the receipt, emission
      or `cascade/eval/benchmarks.py`. `suite_identity.py` is new, and only
      lane D code imports it.
    - `settle_lane_d` returns before any work when it isn't armed.
    - The WB15 trainer change runs only under `lb.active(block)`, apart from
      the status key in note 6.
13. **Invariant 12.** `cascade/audit/lane_d.py` still imports no producer
    eligibility, verdict or gate module. The W5c audit fix (item 4) must
    re-derive the house judgement with `eval.bootstrap`, not by importing
    `judge_cohort`.

Verdict: NOT ACCEPTED. There are 6 BLOCKING items: W3b3 (1), W0c loop
tooling (3: stale-base merge, verification WPs outside the cross step,
verification mode open to any WP), W5c (1) and W3b2 (1). There is also one
fail-closed requirement on W5e's spec (assessed above). W3b4, W7b, WB15
and W0b have no BLOCKING item and may reach L5 once at L3. Of the RV4
items, W3a's two are resolved (with residual note 9), and W7a's and
W5b's three stay open as known.
