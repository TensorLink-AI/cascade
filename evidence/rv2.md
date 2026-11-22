# RV2: lane D review of R8, R10, WB21, R5, R11, R3, W2b, R4, WB10

Covers: R8, R10, WB21, R5, R11, R3, W2b, R4, WB10

Reviewer: RV2 (fresh context, read-only on main). Reviewed at main
`ecc318a` (WB10 #526). The spec is docs/plans/LANE_D_PLAN.md: the §3.0 rows,
§3.1, §3.2, §17, each WP's `tests/accept/<wp>/` and DEC-CA-0071. I read the
code itself, not the PR bodies. I also re-checked the W0c fixes (#507, #518)
against RV1's three BLOCKING items.

Checks run, all single files at `nice -n 19`, using
`python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider`:

- wb10: 4 + 67 + 49 + 3 + 220 passed. w2b: 211 + 12 + 20 + 4 + 4 passed.
  r3: 36 passed. r4: 13 + 60 passed. r8: 34 passed. r11: 37 passed.
  r10: 9 + 35 + 8 + 5 + 5 passed. r5: 32 + 41 + 18 passed, plus 3 passed and
  1 skipped (shellcheck is not installed on this box). wb21: 19 passed.
- `stub_scan.py --paths-file tests/accept/<wp>/PATHS` for r8, r10, r5, r11,
  r3, w2b, r4 and wb10: 0 findings each.
- `unread_keys.py --source cascade/shared/config/lane_d_ops.py` over both
  shipped files: all 8 `[lane_d_ops]` keys are read. All 15 WB10 `[lane_b]`
  keys are read in `config/lane_b.py:124-160`.
- I loaded the deployed `/root/cascade/chain.toml` with this tree's
  `load_chain_config` (a read, no torch). It loads. `lane_b.funded_pod_skus`
  resolves to the deployed `[round]` list `('8xH100', '8xH200')`.
- `overlay_lint.is_release_only('lane_b', 'contamination_hashset', …)`
  returns `False`, and `('lane_c', 'remainder', 'burn')` returns `True`
  (see note 1).

## RV1 BLOCKING items: re-check

- RV1 item 1 (the invariant counter counted fixture strings) is resolved.
  `scripts/laned/status:145-181` counts only module-level `test_inv<n>_` defs
  and `Test*` methods, found by AST, in `test_*.py` files. It counts them only
  for WPs at L3 or above (`status:558-564`). The self-test at
  `status:714-744` plants string literals, helper-module defs and nested defs,
  and asserts that none of them count.
- RV1 item 2 (L4 evidence was self-declared) is resolved in the tooling, but
  not yet in practice. `accept` exports `LANED_EVIDENCE` (`accept:101-108`).
  The push job passes `--evidence` and records `--evidence-sha`
  (`laned-accept.yml:225-231`). L4 needs a green run that verified the
  current blob (`status:120-142`, `:310-324`). However, **no
  `tests/accept/*/` file reads `LANED_EVIDENCE`** (grep over the whole tree).
  So for every external WP the content check is still just `wp` plus
  `passed`. See the R10 BLOCKING item and note 14.
- RV1 item 3 (ci-only WPs never reached L4) is resolved. `status:397-403`
  returns 4 for an accepted, unreviewed WP. `evidence_ok` returns
  `(True, "ci-only")` for a ci-only WP (`status:317-318`). The fixture pins
  C1 at L4 (`status:816-818`).

## BLOCKING

- [ ] BLOCKING R5: the `--execute` state snapshot cannot succeed on the live box, and it leaves out the main activation tally.
  `deploy/laned/lib.sh:346` copies `validator_state.db` with
  `sqlite3 "$path" ".backup …"`. On the deployment that file is JSON text:
  `file /root/cascade/validator_state.db` says "JSON text data", and the
  validator writes it with `write_text_atomic`
  (`cascade/validator/loop.py:1622`). `sqlite3` is also not installed on the
  box (`which sqlite3` finds nothing). The r5 tests stub `sqlite3`
  (`tests/accept/r5/helpers.py:33`), so they can't see this. Every `--execute`
  run therefore dies at `laned_emit_snapshot` (`lib.sh:501`, exit 3) after
  `deploy_gate check`. It fails closed, but S5 (§5 row 10, "Snapshot all
  state (§10.8)") can never run. There is also a second defect: the glob
  `"$ROOT"/activation_*_state.json` (`lib.sh:350`) does not match
  `activation_state.json`, the DEC-CA-0045 main tally. That file is live in
  `/root/cascade`, and it is not in `LANED_STATE_FIXED` (`lib.sh:28-29`).
  `cascade/shared/decided.py:327-328` and `cascade/laned/restore.py:111-116`
  both add it to the glob explicitly. The `.bak` siblings that the loaders
  fall back to (`validator_state.db.bak`, `cascade_state.json.bak`,
  `activation_*.json.bak`) and the `<state_db>.lane_b.json` /
  `.lane_c.json` sidecars (`cascade/validator/lane_b.py:54`,
  `lane_c.py:279`) are not copied either.
  **Proposed fix (R5 PATHS, `deploy/laned/lib.sh`):** copy
  `validator_state.db` with `cp -a`, like the other JSON state files. Add
  `activation_state.json` to `LANED_STATE_FIXED`, or glob both forms. Copy
  each fixed file's `.bak` and the `<state_db>.lane_b.json` /
  `.lane_c.json` sidecars when they exist. A better option is to call
  `python -P -m cascade.laned.restore snapshot` so that the two state
  inventories can't drift. The r5 test author should drop `sqlite3` from
  the allowed copy tools in `tests/accept/r5/README.md:139`. They should also
  add a fixture with a JSON `validator_state.db` and an
  `activation_state.json`, and assert that both are copied.

- [ ] BLOCKING R10: L4 can be reached with a hand-written `evidence/t25.json`, because nothing in the R10 tests verifies the evidence.
  The R10 Accept cell is
  `scripts/laned/accept R10 … --evidence evidence/t25.json`, and Runs-in
  includes `testnet`, so `status` treats R10 as external
  (`status:47`, `:235`). `cascade/laned/restore.py:508-585` implements the
  content verifier (`verify-evidence`: schema, mainnet source ids, release
  head, clean start, verdict equality, bundle sha cross-check). But no test
  under `tests/accept/r10/` reads `LANED_EVIDENCE`.
  `test_r10_evidence.py` exercises the verifier only on `tmp_path`
  fixtures. So the push-to-main run with `--evidence evidence/t25.json`
  checks only `"wp": "R10"` and `"passed": true` (`lanedlib.py:116-129`).
  A bare `{"wp": "R10", "passed": true}` would put R10 at L4. That is the
  exact gap RV1 item 2 was meant to close.
  **Proposed fix (R10's `tests/accept/r10/`, a tests-only PR):** add
  `test_r10_live_t25_evidence`. When `LANED_EVIDENCE` is set, it runs
  `python -P -m cascade.laned.restore verify-evidence --evidence
  $LANED_EVIDENCE` (plus `--bundle` when the evidence names one) and asserts
  exit 0. When the variable is unset, it skips with a reason (an ordinary PR
  run). Also add a fixture case where a bare `{"wp", "passed"}` file set as
  `LANED_EVIDENCE` makes that test fail.

## Non-blocking notes

1. **R3 / WB10: the release-only detection is brittle, and already misses one
   key. My verdict is not BLOCKING.** `overlay_lint.is_release_only`
   (`cascade/laned/overlay_lint.py:142-151`) decides by looking for
   `_BURN_MARKER = "burn destination is release-only"` (`:47`) in the
   `check_overrides` error text. WB10, which merged after R3, refuses
   `[lane_b] contamination_hashset` with a different message,
   `"data pin is release-only"` (`netconfig.py:227-229`). So
   `is_release_only` returns `False` for it (checked). This is not BLOCKING
   because `rule_invalid` runs `validate_overlay`, which calls
   `build_config`, which calls `check_overrides` (`netconfig.py:371`). That
   refuses any check_overrides-refused key, whatever the message, so no
   release-only key can pass the lint. The defect is that the key is
   reported as `invalid` rather than `release-only`, and the "even restated"
   rule now silently depends on message wording.
   **Proposed fix:** add a named constant in `cascade/shared/netconfig.py`:
   `RELEASE_ONLY_KEYS: dict[tuple[str, str], str] = {("scoring", "burn_uid"): "burn destination", ("lane_b", "remainder"): …, ("lane_c", "remainder"): …, ("lane_d", "remainder"): …, ("lane_b", "contamination_hashset"): "data pin"}`.
   `check_overrides` (`:224-229`) would iterate it, and `overlay_lint`
   would use `(table, key) in N.RELEASE_ONLY_KEYS` with no
   `check_overrides` call. This fits the netconfig.py PATHS of W2, W2b or
   WB10 plus `cascade/laned/` for R3. R3's tests should add a
   `("lane_b", "contamination_hashset", "sha256:…")` case to
   `test_r3_release_only_refused_in_lane_d_overlay`
   (`tests/accept/r3/test_r3_lint.py:141-153`). This also matches the WB10
   row's wording ("added to the `netconfig.py:210` tuple"): the builder
   used a separate `elif` instead.
2. **WB10 (consensus, mixed fleet):** `contamination_hashset` is now refused
   by upgraded nodes. Pre-WB10 nodes accept it silently: they have
   `ALLOWED["lane_b"] = None`, and `parse_lane_b` has no unknown-key
   refusal (`config/v2.py:762-766`). So an overlay that carries it, even as
   `""`, splits a mixed fleet in the same way as RV1 note 12. Plan §14.1
   records that no overlay has been published, and R3 refuses it (as
   `invalid`), so there is no live risk. R7's generated bodies must never
   include it. Overlays that set the other 14 keys are accepted by both old
   and new nodes, but an out-of-range value (for example
   `guard_factor = 0.9`) is refused only by new nodes. R3's `rule_invalid`
   runs on new code, so the lint catches that too. Nothing reads the keys
   yet, so an accepted value has no behavioural effect on either side.
3. **WB10 (unarmed digests):** I found no digest that moves. No
   receipt, manifest or contract digest folds `LaneBConfig` or `cfg.raw`. I
   grepped `asdict`/`fields` and every receipt/manifest `lane_b` use. As a
   result, "drop-when-default in the digest" (DEC-CA-0071 §"New keys", WB10
   row) is vacuous today: there is no fold for it to drop from.
   `test_inv1_lane_b_config_unchanged_when_unarmed`
   (`tests/accept/wb10/test_wb10_invariants.py:37-50`) compares new code
   with itself (shipped vs. keys dropped vs. explicit defaults), the same
   pattern as RV1 note 11. The WP that first folds a key (suite-target
   commitment, D42 leg contract) must add the drop-when-default fold and a
   golden pre-arming digest pin.
4. **WB10 (dead knob):** `traffic_steal_bound` (`config/lane_b.py:40, 138`)
   is the guard for D40 option (b). DEC-CA-0071 records (b) as **not
   adopted**, and `LANE_B_ROUTER_ROWS = ("frozen",)` makes the key
   unreachable. The WB10 row lists it, so shipping it matches the row. Mark
   it as inert in the `chain.toml` comment so it isn't mistaken for a live
   guard.
5. **WB10 (D42 GPU list):** the repo `chain.toml` `[lane_b] funded_pod_skus`
   (`chain.toml:1466`) restates the repo `[round]` list of four SKUs. The
   deployed `[round]` has two (`8xH100`, `8xH200`). When the next deploy
   syncs the file, lane B's list will include `4xH200`/`4xH100` while lane A
   doesn't. That is harmless while unarmed. DEC-CA-0071 D42 requires each
   lane B GPU to be image-qualified and measured first, so diff the list
   against the deployed `[round]` and `[v2] funded_sku_wall_*` before any
   arming overlay. `loader.py:358` duplicates `[generator] max_repo_mb`'s
   2048 fallback from `:422`. The two agree, but consider passing the
   parsed `GeneratorConfig` value instead.
6. **WB10 / DEC-CA-0071:** the DEC says that "a domain named `s-*` is
   refused at load" (§1 Codes), and that `max_leg_hours`, `max_leg_seconds`
   and `guard_factor` are consensus. WB10 implements neither the `s-*`
   refusal nor any consensus fold. That is fine as long as a later WP (WB11
   or later) owns them. Check that the lane B rows name both.
7. **W2b (consensus):** there is no mixed-fleet change. Before W2b, an
   overlay with `[lane_d_ops]` or `[generation]` was already refused (neither
   table is in `ALLOWED`, so "not on the DEC-CA-0057 allow-list"). Now it is
   refused as operator-only (`netconfig.py:138-141, 213-215`). Old and new
   nodes refuse the same bodies, and only the message differs.
   `overlay_policy()["operator_only"]` has one consumer,
   `cascade/laned/dedrift.py:117`, which now classifies `lane_d_ops` as
   operator-only, as intended. No digest or receipt reads the table.
8. **W2b (unread in production):** `lane_d_ops_config` / `LaneDOpsConfig`
   (`config/lane_d_ops.py:151-154`) has no production caller yet. It is
   expected to have none until the intake and validator WPs, but the 8 keys
   are parsed and never consumed. The satellite-import test
   (`tests/accept/w2b/test_w2b_satellite_imports.py`) does check the real
   import closure, and it fails when it should (the walker asserts that it
   reaches loader/netconfig/receipt).
9. **R8 (today's deploy flow):** no refusal changes for trainer/validator.
   `CONSENSUS` is unchanged. The new AREAs only add notes, and `classify`
   (`scripts/deploy_gate.py:98-104`) assigns every matching area. `scoring`
   now also applies to `intake-laned`/`eval-host` (`:91-93, 262-265`), and
   `verify` for trainer/validator keeps the same marker and process match.
   The visible change is that `worker_image` now covers
   `cascade/{direct,…,validator}/**/*.py` (`:59-67`). The "rebuild + repin
   only at a seed-epoch boundary" note (`:219-221`) therefore fires on
   almost every trainer/validator deploy. Tell the owner this is
   expected, so it doesn't prompt an unneeded mid-epoch repin.
10. **R8 (comment is wrong):** `deploy_gate.py:55-58` justifies
    `ON_POD_CODE` with "worker_code_fingerprint hashes every
    cascade/**/*.py, so over-covering here is the safe direction". But
    `ON_POD_CODE` leaves out `cascade/laned/`, `cascade/lane_d/` and
    `cascade/website/`, and `cascade/provision/codeprint.py:15-16` does hash
    them. An image rebuilt for a lane D-only release therefore gets a new
    fingerprint with no `worker_image` note. Functionally that is fine,
    because that code never runs on a pod. Fix the comment, or add a
    separate "image fingerprint changes on rebuild" note for any
    `^cascade/.*\.py$`. Separately, the test allow-list `(^|/)[^/]+\.md$`
    (`tests/accept/r8/test_r8_deploy_gate.py:34`) lets any `.md` anywhere
    pass unclassified. That is acceptable today.
11. **R5 (other):**
    - A real run needs `--execute` (`lib.sh:185`).
    - `LANED_NOW` is refused with `--execute` (`:230`).
    - `window.py` fails closed on unreadable state (`window.py:44-53`, an
      `in_flight`/`harvest` refusal) and on any non-0/1 exit (`lib.sh:273-276`).
    - Every ssh and rsync to the eval host pins `UserKnownHostsFile` with
      `StrictHostKeyChecking=yes` and `BatchMode=yes` (`:456-459`), and an
      empty known-hosts file is refused (`:217`).
    - No key names appear in the scripts. `.env` is sourced only inside the
      launch subshell and is never printed.

    Still open:
    - The box restart reuses one `--venv` for both release and previous and
      never runs `uv sync --frozen` on the box (`:418-447`). A release that
      changes `uv.lock` runs on the old environment. Refuse when the
      release's `uv.lock` differs from `--previous-ref`'s, or add a sync step.
    - §5 row 10 says "lane D intake (systemd unit with the code-path guard)",
      but R5 launches it with `systemd-run --scope` and `nohup` like the
      others (`:418-447`).
    - `KNOWN_HOSTS` is interpolated unquoted into `ssh_e` (`:456`), which
      breaks on a path with spaces.
    - The shellcheck test skips on this box. Confirm that the CI runner has
      shellcheck, otherwise that Accept item is vacuous.
12. **R10 (safety):** R10 never writes the source. `snapshot` only reads, and
    it refuses `--out` inside `--source` (`restore.py:195-196`). `restore`
    refuses a target whose netuid **or** network_id equals the bundle's
    source (`:350-356`), and a target that overlaps the bundle or the recorded
    source dir (`:358-363`). One gap remains: the same-network check trusts
    the bundle's self-recorded `source`, and the overlap check covers only the
    snapshot's own source dir. A bundle taken from a copy, restored with
    `--force-target` into the live `/root/cascade` under
    `chain.testnet.toml`, would overwrite live mainnet state. Refuse a target
    dir that holds a running validator's state lock
    (`validator_state.db.lock`), or `$CASCADE_ROOT`. Also, the bundle
    leaves out the `<state_db>.lane_b.json` / `.lane_c.json` sidecars. Those
    are validator state once lane B or lane C is armed, and T25 should
    restore them.
13. **R11:** the gates are the services' own functions (`startup_gates.py:168-302`,
    `GATES` in `:306-338`), and the scanner tests fail when `main.py` gains
    a gate. T26 runs the shipped files at pinned heads. It runs them without a
    `--state-dir`, so the activation-note gates see no persisted records. RV1
    note 13 asked for the **deployed** file plus the live state copy to go
    through it before S2b. I checked that the deployed file loads, but I did
    not run R11 against `/root/cascade` (scope rule). Do that as an owner
    step: `python -P -m cascade.laned.startup_gates --config <copy of
    deployed chain.toml> --state-dir <copy of state> --role validator
    --head-block <live>`.
14. **W0c / plan (evidence for other external WPs):** R5 (`ci (scripts) +
    owner (execute)`) and WB21 (`ci + owner`) are external
    (`status:47`), and their Accept cells name no `--evidence`. They
    therefore need `evidence/r5.json` / `evidence/wb21.json`, which nothing
    produces or verifies. This is the same pattern as RV1 note 1 for W0.
    Either define the evidence and its verifier test in each WP's tests, or
    change Runs-in to `ci` and state the owner step in What. Consider also
    having `status` refuse L4 for an external WP whose `tests/accept/<wp>/`
    never mentions `LANED_EVIDENCE`. That would make the R10-class gap
    impossible to reintroduce.
15. **R3 / R4 (RV1 note 10, the stake gate):** R4 now produces the exact
    stake share (`upgrade_tracker.py:174-211`), but nothing consumes it. R3's
    `unsafe-fleet` rule checks only the `overlay_keys` that the policy file
    declares (`overlay_lint.py:212-228`), not stake. A `[lane_d] from_block`
    overlay can therefore still pass the lint while upgraded stake is below
    2/3. R7's publish wrapper, which is not on main yet, should require an R4
    report with `meets_threshold: true` for the same `cfg-` name. R4's
    `DEFAULT_THRESHOLD = "0.6667"` (`:38`) is slightly stricter than 2/3.
    That is the safe direction, but the report should say so.
16. **R3 (test nit):** `test_r3_t21_kill_switch_passes_on_upgraded_fleet`
    (`tests/accept/r3/test_r3_lint.py:52-60`) ends with a stray second
    `assert_ok(*lint(...))` without policies. That second call repeats the
    previous test.
17. **R4:** the grammar comes from `activation.parse_signals`/`signal_for` and the
    eligible/signed sets from `activation.tally`. Only the `Fraction`
    arithmetic is local, and parsing stakes with `parse_float=Fraction`
    keeps decimal stakes exact. I found no defect.
18. **WB21:** DEC-CA-0071 front matter is `status: accepted` and the
    amendment is marked ACCEPTED (2026-11-21). The new-key list matches
    WB10's 11 plus the 4 D42 keys. I found no defect, apart from the
    evidence point in note 14.
19. **Invariants 10 and 12; no publish path:**
    - Invariant 10: WB10 pairs
      `test_inv1_lane_b_config_unchanged_when_unarmed` with
      `…_armed_nonempty_lane_d_effect` (`test_wb10_invariants.py:37, 53`).
      The armed test is meaningful: each override takes effect, and the
      lane A digests, `[round]` and `max_train_seconds` stay unmoved. No
      other in-scope WP has unarmed tests.
    - Invariant 12: no in-scope code is audit or satellite-replay code.
      `startup_gates` imports the validator/trainer runners by design (it
      runs the real gates), and the W2b test confirms that the satellite
      path doesn't import `lane_d_ops`.
    - No publish path: there is no publish or chain-write code in
      `cascade/laned/` or `deploy/laned/` (grep for `netconfig-publish`,
      `set_weights`, `set_reveal_commitment`, uploads). The R7
      confirm-token path has no implementation on main yet, so there is
      nothing to check.
