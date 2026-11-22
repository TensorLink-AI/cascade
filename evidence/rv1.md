# RV1: lane D review of W0c, W0, R6, R2, R1, W2

Covers: W0c, W0, R6, R2, R1, W2

Reviewer: RV1 (fresh context, read-only on main). Reviewed at main
`86923de` (W2 #502). R8 #503 landed after that and is out of scope. The spec
is docs/plans/LANE_D_PLAN.md: the §3.0 rows, §3.1, §3.2, §17 and each WP's
`tests/accept/<wp>/`. I read the code itself, not the PR bodies.

Checks run, all single files at low priority:

- `tests/accept/{w0,w2,r1,r2,r6}/`, run with
  `python -P -m pytest --rootdir <wt> --noconftest`: 15, 572, 62, 40 and 43
  passed. Without `-P` the interpreter imports `cascade` from the cwd
  checkout, not the worktree, so R6 then fails with
  `No module named 'cascade.laned'`. That is a reviewer pitfall, not a defect.
- `scripts/laned/unread_keys.py` narrowed to
  `--source cascade/shared/config/lane_d.py` over both shipped files: 0 of 86
  keys unread.
- `scripts/laned/status` at origin/main, using the ledger from
  `laned/build-state:status/ci_runs.json`: W0c, W0, R1, R2, R6 and W2 are at L3.
  Invariants covered: 1, 9 and 10. W0 shows `evidence/w0.json: evidence file
  missing`.

## BLOCKING

- [ ] BLOCKING W0c: the §17.3 invariant counter counts fixture strings as
  tests. `scripts/laned/status:432-440` regex-scans the raw text of every
  `.py` under `tests/accept/<wp>/` for `def test_inv<n>_`. `tests/accept/w0c/helpers.py:131-133`
  holds the fixture literals `"def test_inv9_share_zero():\n    assert True\n"`
  and `"def test_inv1_unarmed(): ..."`. As a result, the live board reported
  invariants 1 and 9 as covered at b366e9b, when no WP at L3 had a real
  invariant 1 or 9 test. Today invariant 9 is still covered only by that
  string. The S2 coverage figure (§17.3 "S2 needs 100%") can therefore be
  reached with `assert True` stubs.
  **Proposed W0c row addition:** "(v) the §17.3 invariant counter counts only
  test functions found by AST (`ast.FunctionDef` named `test_inv<n>_*`) in
  collected `test_*.py` files of WPs at ≥ L3. String literals and helper or
  fixture modules never count. A self-test plants a `def test_inv9_` string
  literal in a helper and asserts invariant 9 stays uncovered. S2's
  requirements include invariants 1–12 = 12/12."

- [ ] BLOCKING W0c: L4 evidence is self-declared and never verified. For an
  external WP, `lanedlib.check_json_evidence` (`scripts/laned/lanedlib.py:115-129`)
  checks only `"wp" == <WP>` and `"passed": true`. The status script uses that
  check for L4 (`status:219-225`) and for every execution-stage gate
  (`status:405-411`). The push-to-main job runs `accept "$wp"` without
  `--evidence` (`.github/workflows/laned-accept.yml:204`), and `accept` never
  passes the evidence path to the WP's pytest run (`accept:109-138`). So no
  per-WP verifier can ever inspect the evidence or cross-check it against the
  live resource. One hand-written `{"wp": "R6x", "passed": true}` moves R6x to
  L4 and completes S0a. That conflicts with §3 ("Accept verifies evidence
  JSON written by the real run (endpoint, request ids, object digests, block
  numbers)"), §17.1 L4 ("the verifier passes against the live resource") and
  §3.1 rule 8.
  **Proposed W0c row addition:** "(vi) for an external WP, `accept <WP>
  --evidence F` exports `LANED_EVIDENCE=F` to the WP's pytest run, and the
  WP's `tests/accept/<wp>/` own the content verifier. The push-to-main job
  passes `--evidence <file from the Accept cell>` for every external WP that
  has the file at the commit. L4 (and an execution-stage gate) needs that
  push/accept run with evidence to be green on main, not only the
  `wp`/`passed` fields. A self-test asserts that a bare
  `{"wp": ..., "passed": true}` does not reach L4 for a fixture WP whose
  verifier test requires a field."

- [ ] BLOCKING W0c: ci-only WPs never reach L4 until reviewed, which
  contradicts §17.1 and W0c's own spec. §17.1 says L4 is "n/a for ci-only
  WPs, which pass L4 automatically". `tests/accept/w0c/README.md` (Levels
  table, L4) says "it is a ci-only WP, **or** its evidence file exists". But
  `status:258-261` returns 3 for an unreviewed ci-only WP. Stage checks that
  name a ci-only WP at L4 then fail: S1 requires "W1, W0b and W3a at L4" and
  W0b is `Runs in: ci` (`status:389`). So S1 cannot complete until an RV
  covers W0b, which is not an S1 condition. No fixture pins an unreviewed
  ci-only WP, which is why the tests miss it.
  **Proposed W0c row addition:** "(vii) a ci-only WP at L3 with no external
  evidence is reported L4 (§17.1 'pass L4 automatically'). The status fixture
  adds an unreviewed ci-only WP expected at L4, and a stage that requires a
  ci-only WP at L4 is satisfied by it."

## Non-blocking notes

1. W0c / plan: W0 can never reach L4 or L5. Its Runs-in cell `ci + owner merge`
   contains `owner`, so `status` treats it as external (README pins this:
   "W0 | L3 | external"). Its Accept cell names no `--evidence`, so the default
   `evidence/w0.json` is required, and no WP produces that file. S2 ("every
   build WP at L5") therefore cannot complete. Proposed plan row change: W0
   `Runs in` = `ci`, with the owner merge stated in What (every WP is
   owner-merged under rule 6).
2. W0c / plan: the S2 "build" set (`status:362`) is every row in the two non-lane-B
   build tables. That includes owner execution WPs (R2x, R6x, R7t, R7m, R9,
   R13, R14, R15, R16), while the §3.0 Review-WPs text counts "W/R build WPs,
   not owner execution WPs". R9 is the S2b release, so the Build-complete
   milestone waits on S2b. Clarify "build WP" for S2 in §17.2, or tag the
   execution-type R rows.
3. W0c: S2's requirements (`status:390`) don't check the §17.3 invariant
   coverage, and they don't check that `T-rows`/orphans are clean. Only
   "every build WP at L5" plus the gate file are checked.
4. W0c: L5 is sticky. `reviewed()` (`status:227-240`) accepts any `rv<n>.md`
   that covers the WP, even when the WP's PATHS changed after the review
   commit. Consider requiring that the review commit descends from the WP's
   last commit touching its PATHS.
5. W0c: L2 detection (`status:205-217`) takes the first commit after the PATHS
   commit whose files all sit inside the WP's PATHS. PATHS overlap: R6's
   `cascade/laned/` covers R1's and R2's files, and `chain.toml` is shared by
   W2 and later WB10/R3. So another WP's implementation can be picked as the
   L2 commit. Restricting to PRs labelled `laned:<WP>` (the ledger has the
   label) would remove the ambiguity.
6. W0c: L1 accepts any successful expect-fail run for the WP, including one on
   a PR that never merged, or on tests that were later replaced. Pin it to an
   ancestor of the PATHS commit on main.
7. W0c: the invariant 10 check (`accept:63-78`) is name-based only. An "unchanged
   when unarmed" test that doesn't use the `unchanged_when_unarmed` token
   escapes the pairing rule. That is acceptable as a convention, but test
   authors should be told in the W0c README.
8. W0c: `unread_keys.py` gathers reads into one global string set over all
   sources (`unread_keys.py:34-66`). With `--source cascade/shared/config`
   (`tests/accept/w2/helpers.py:394`), generic key names such as `share`,
   `alpha`, `horizon` and `floor` are "read" by the lane B/C parsers, so the
   W2 check could not fail for them. Today `lane_d.py` alone reads all 86 keys
   (checked), so nothing is wrong now. Consider scoping reads per table, or
   running it with `--source cascade/shared/config/lane_d.py`.
9. W0c: `accept RVn --evidence` checks only for open BLOCKING lines. It does
   not check the §3.0 Review rule "each finding is either fixed or filed as a
   WP".
10. W2 (consensus, overlay): with `ALLOWED["lane_d"] = None`, an overlay can set
    `from_block` once DEC-CA-0057 adoption reaches its 51% permit-stake
    lock-in (`netconfig.py:83-90`). The plan's stake gate for arming is
    "observed `cfg-` stake ≥ 2/3 before `effective_block`" (§5 row 11, X6a,
    D10). That gate is operational only: no code refuses a `[lane_d]
    from_block` overlay below 2/3. Old-release nodes refuse the overlay
    whole (§14.1), so arming early splits the fleet. Recommend that R3 (lint)
    or R7 (publish wrapper) refuse a `from_block`-setting `[lane_d]` overlay
    unless R4's tracker output shows ≥ 2/3, and that the plan names that
    check. The retroactive guard (`netconfig.py:464-472`) does cover
    `from_block`/`until_block`, because `lane_d` is now in `ALLOWED`.
11. W2 (consensus, unarmed): I found no digest that moves. No receipt or
    manifest digest folds `cfg.raw`; the receipt carries only
    `netconfig_digest` of an overlay body. The test fingerprint covers
    `contract_digest`, `numerics_contract_digest`, the v2 digests and
    `koth_params` at three blocks. `overlay_policy()` output changes (lane_d
    `*`), but by design and outside consensus. The inv1 test compares the new
    code with itself (absent vs. shipped vs. explicit table), not with a
    pre-W2 golden digest. A golden pin would make "matches today's" literal.
12. W2 (`[lane_c] remainder` release-only, `netconfig.py:213-215`): safe today.
    `LANE_C_REMAINDERS = ("burn",)` (`config/lane_c.py:42`), so the default
    is the only legal value, and plan §14.1 records that no overlay has ever
    been published. The residual risk is in a mixed fleet: an overlay that
    merely restates `[lane_c] remainder = "burn"` is now refused whole by new
    nodes and accepted by old ones. R3/R7 overlay bodies must never carry
    it, and the R3 lint should refuse it explicitly.
13. W2: the stricter validators now apply to the deployed file's `[lane_d]`.
    Examples: `alpha` must be in (0, 1), `gate_weights` is validated, and
    `takes_from` is no longer `str()`-coerced. Run the deployed
    `/root/cascade/chain.toml` through `load_chain_config` (R11/R6x) before
    S2b, so a legacy value doesn't crash-loop a validator at upgrade.
14. W2: there are two era-boundary functions. The `[lane_d]` load check uses the
    loader lambda (`loader.py:744-747`: `epoch_blocks_prev`/`epoch_blocks` ×
    `era_settlements`). `validate_overlay` uses `netconfig.is_era_start`
    (`era_length_blocks`). The pattern is the same as lane C, but nothing
    asserts the two agree. R1's golden vectors could include a `[lane_d]
    from_block` case.
15. W2: `LaneDConfig.active()` (`lane_d.py:106-113`) has no production caller yet.
    That is expected until W1.5/W8, but it means "real active()" is exercised
    only by tests. W1.5's "`active()` is not a constant" assertion is the
    first real consumer.
16. W2: legacy keys are kept as the row requires, but `chain.toml` still annotates
    `targets`/`teacher_ref`/… with `# TBD`, and the legacy `max_context_length`
    is capped at 32768 (`lane_d.py:270-272`), which conflicts in spirit with the
    new 131072 `contexts` cap. Both are dead under DEC-CA-0070. Plan a later
    release that drops them, so they can't be mistaken for live knobs.
17. R6: the classifier rules read live `netconfig`/`decided` code by design.
    The `validate_overlay` self-check on the overlay body is therefore circular
    for allow-list errors: if `check_overrides` let a consensus key through,
    both would agree. The only independent anchor is the hand-pinned
    `PINNED` table in `tests/accept/r6/test_dedrift.py:293-317`. That is
    adequate, but keep it hand-maintained (never generated from `ALLOWED`).
    The 21-marker fixture is present (`fixtures/deployed_chain.toml`: 21).
18. R6: PATHS is the whole `cascade/laned/` package, so R6's stub scan and
    expect-fail cover R1's and R2's files (see note 5).
19. R1: the two computations are independent in implementation but not in
    input or rules. Both read the same operator-supplied grid JSON (`round`
    vs `segments`), and `SegmentWalk` re-encodes the same rules as the
    shared code: `+2` eras for min-effective (`blocks.py:271-273`),
    settlement judged at `boundary - 1` (`:289-297`) and the reveal `max(1,
    …)` (`:324-331`). That catches arithmetic and grid-in-force bugs, but not
    a wrong rule in `cascade.shared`. The golden vectors with past
    off-by-ones are the real anchor. `SharedComputation` models one grid
    change only (`epoch_blocks_prev`/`epoch_activation_block`), so a
    multi-change history is refused (exit 3) rather than answered. That is
    safe.
20. R1: the `pyproject.toml` `[project.scripts]` line matches
    `deploy_gate.WORKER_IMAGE` (`^pyproject\.toml$`, `scripts/deploy_gate.py:49-50`),
    so the next release is classified as touching the worker image even
    though no dependency changed (`uv.lock` is untouched). Say so in the
    release notes, so it doesn't trigger a spurious repin.
21. R2: the trim loop is re-implemented (`notes.py:141-145`) rather than calling
    `activation.own_signal_payload`. The cross-check tests against the real
    builder (`tests/accept/r2/test_r2_notes.py:349-393`) pin the
    equivalence, and the 2026-11-13 crash-loop fixture reproduces the
    refusal. I found no defect.
22. W0: DEC-CA-0070 front matter is valid, it supersedes DEC-CA-0062, and every
    register decision is matched by the tests. I found no defect.
23. Invariant 12: no audit or satellite-replay code is in scope yet (W7d/W9
    aren't built). `cascade/laned/*` imports only `cascade.shared`
    config/netconfig/activation/era/decided, and no producer eligibility,
    verdict or weight code. Invariant 10: W2 pairs
    `test_inv1_lane_d_config_unchanged_when_unarmed` with a meaningful armed
    test (`tests/accept/w2/test_w2_active.py:68-110`). R1, R2 and R6 have no
    unarmed tests. There are no stubs, `TODO`s or mock imports in the
    reviewed non-test paths.
