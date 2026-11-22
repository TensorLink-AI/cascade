# W5e acceptance tests: verdict, confirmation duel, D20, `rows_sha256`, gate wiring

W5e (plan rev6 §3.0 W5e row, §1.3 "Verdict", §1.3b, §1.5, D20, A2, A9, T3,
T4, T9b, T17) pins how a judged cohort becomes a crown. The orchestrator
also assigned W5e the wiring of the W3b gate (`lane_d_gate`, `lane_d_bench`)
into `settle_lane_d`, which W3b2's README left unassigned. These tests are
the spec. The builder may touch only the paths in `PATHS`. The W1.5, W5a,
W5b, W5c, W5d, W7a and W3b tests must stay green.

## Contract (`settle_lane_d`, `cascade/validator/lane_d.py`)

### Verdict

- A member clears iff `LCB_α ≥ d_margin` (inclusive) **and** its point gain
  is strictly positive. The LCB is the paired cluster bootstrap (clusters =
  source), max-T over the cohort, and doesn't depend on `d_margin`.
- Each verdict's `lcb`, `gain` and `cleared` equal `judge_cohort` recomputed
  from the report's own rows over `cohorts[].window_ids`, with the members in
  cohort order and `seed = report["seed"]`.
- **Max-T crowns the best.** When several members clear their confirmation
  in one settlement, the crown goes to the one with the highest LCB.

### Mandatory confirmation duel

- A first clear is never crowned. The verdict has `stage: "first"` and
  outcome `confirm_pending`, and `crowned` is `null`. The entry is
  `confirm_pending` / `confirm_pending` in state and report. Its
  `confirm` is `{first_set_sha256 (= the cohort's set_sha256), king_digest,
  rule, boundary}`.
- The confirmation runs on the **next disjoint frozen set**. It shares no
  window with the first set, even when the earlier snapshot is still
  retained in `prior_pool_dirs`. The verdict has `stage: "confirm"`, and
  `confirmation.first_set_sha256` / `first_king_digest` point back to the
  first clear.
- Clearing again crowns (outcome `crowned`, `report.crowned`, state `king`,
  pay slot 0 = the miner's UID). Failing is `lost` / `lost_confirmation`, and
  the king is unchanged.
- `confirm_pending` persists. A settlement with no fresh window leaves the
  entry unjudged, still `confirm_pending` with the same `confirm` record. A
  restart (a fresh call on a copy of the state file) gives the same
  confirmation verdict and `rows_sha256`.

### D20 (`confirm_on_king_change`)

- **`"restart"` (default), T9b.** After a house refresh moves KING_D, the
  entry is re-judged at `stage: "first"` against the new king on a fresh
  set. The first clear is discarded (`confirmation.first_set_sha256` is
  `null`), and a new clear records the new king and boundary in `confirm`.
  There's no crown that settlement.
- **`"original"`.** After a house refresh, the confirmation still runs
  (`stage: "confirm"`, `king_digest` = the first-clear king H0), and it
  crowns when it clears. **On main this fails.** H0 is no longer the king,
  the house or a prior king, so its T is unknown and the entry is skipped as
  `king_unscored`. The `confirm` record must carry what the confirmation
  needs.

### `rows_sha256` (A9)

- `rows_sha256 == sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
  allow_nan=False).encode())`, which is `lane_d_report.rows_sha256(rows)`.
- The rows are window-major in the cohort's `window_ids` order. Every model
  the cohort is scored with has a row on every window. The values are finite.
- The digest is bound into the state's `last_report`, and any change to a
  row changes it.

### Determinism

- Permuting `entries` gives identical `cohorts`, `verdicts`, `rows`,
  `rows_sha256`, `judged`, `entries`, `seed` and `crowned`. So do two
  subprocesses with different `PYTHONHASHSEED`s.

### Gate wiring (§1.3b)

`settle_lane_d` gains three keywords:

| keyword | meaning |
|---|---|
| `boundary_block_hash` | the settlement boundary's block hash (`0x` + 64 hex); the gate draw's input |
| `sentinel` | the era's §16 sentinel suite (`str`), or `None` when W-S1 is unarmed |
| `bench` | the bench host session, a `lane_d_bench.SweepSession` |

On main the tests fail on an AssertionError in `helpers.settle_gated`,
which checks the signature before it calls. What happens when they are
omitted or `None` is pinned by "Fail closed" below.

- **Who is gated:** every verdict that cleared at `stage: "confirm"` (a
  crowning candidate), nothing else. First clears and losers are never swept.
  A settlement with no crowning candidate makes no host call, and
  `report.get("gate")` is `None`.
- **The draw:** one suite per settlement,
  `gate_suite_for(boundary_block_hash, gate_suites, gate_weights, sentinel)`.
  The report's `gate` record holds at least
  `gate_record(cfg, boundary, boundary_block_hash, sentinel=sentinel)`
  (`suite`, `block`, `block_hash`, `sentinel`, `eligible`). Changing only the
  hash changes the drawn suite, and so the outcome.
- **The sweep:** the candidate, KING_D and the house are swept through
  `bench.paired(...)` in the settlement's session, for the drawn suite and,
  when a sentinel is given, the sentinel suite as well. The decision is
  `lane_d_bench.paired_gate` on those records. The sentinel is recorded and
  never gates: a regression on the sentinel alone still crowns.
- **Outcomes:**

  | `paired_gate` outcome | entry (state and report) | crown |
  |---|---|---|
  | `pass` | `crowned` | yes, best LCB among the passing candidates |
  | `gate_regression` | `lost` / `gate_regression`; verdict outcome `lost` | no |
  | `checkpoint_fault` | `closed` / `checkpoint_fault` (the miner's fault) | no |
  | `bench_hold` | open (`hold` or `confirm_pending`) / `bench_hold` | no; not lost |

- **Several candidates:** each is gated on the same draw. A regression by the
  max-T best doesn't block a passing runner-up, which is crowned.
- **A miner KING_D:** the house is still the second reference (T3). A
  candidate that beats KING_D on the drawn suite but regresses against the
  house is `gate_regression`.
- **`bench_hold` keeps the candidate's chance.** At the next settlement,
  with a healthy bench and no fresh windows, the held candidate is re-gated
  on the new boundary's draw and crowned. It is not re-duelled: it has no
  rows in that report. Counting holds against `max_bench_holds` is W5g's.

### Fail closed (RV5, plan §1.3b, §3 invariants)

"Armed" is `cfg.active(boundary)`: `[lane_d] from_block > 0` and the
settlement's boundary is after `from_block` (and not past `until_block`).

- **Unarmed: unchanged.** The legacy call (no gate keywords) returns
  `report=None, pay=None` and writes no state, with `from_block = 0` or at
  `boundary == from_block`. Unarmed nothing is judged, so nothing can crown.
- **Armed, no crowning candidate: unchanged.** The legacy call stays valid
  for a settlement where no member clears at `stage: "confirm"` (first
  clears, losers, pending floors). The verdict tests use it that way.
- **Armed, a crowning candidate, no gate inputs: `bench_hold`.** A member
  that clears its confirmation would crown, but the settlement has no `bench`
  session or no `boundary_block_hash`. That covers the legacy call without
  the keywords, `bench=None`, `boundary_block_hash=None`, and both. The §1.3b
  gate can't run, so the member is not crowned:
  - `report.crowned` is `null`, `king_after == king_before`, the state king
    is unchanged and the king slot still burns;
  - the entry (state and report) is `bench_hold` with status `hold` or
    `confirm_pending`: open, not `lost` or `closed`, not the miner's fault;
  - the verdict outcome is neither `crowned` nor `lost`;
  - if a `gate` record is written, none of its results is `pass`;
  - not charged: at the next settlement, with the inputs back and no fresh
    window, the held clear is re-gated on the new draw and crowned without a
    new duel (no rows for it).
- **A crowned report needs a gate record.** `validate_report` must raise
  `ReportSchemaError` on a `lane-d-report/1` report whose `crowned` is
  non-null and whose `gate` is absent or `null`, even when it's correctly
  re-signed. `verify_report`, which validates first, therefore returns
  `False`. A report with `crowned: null` and no gate still validates. In the
  schema this means: `gate` stays optional (W7a RV4), and a conditional
  makes `gate` required and an object whenever `crowned` is a string. W7a's
  RV4 schema change is pending, so whichever of W5e and W7a lands second
  carries the rule. Its `results` must hold a `pass` for the crowned digest
  (W7a pins that, `test_w7a_gate_record.py`).
- **The audit must refuse it too (W7a/W9, not pinned here).**
  `cascade/audit/lane_d.py` isn't in W5e's `PATHS` (the owner edit is
  pending, RV4). `check_lane_d` must FAIL an armed report whose `crowned` is
  set and whose `gate` is missing or `null`, and one whose `gate.results`
  has no `pass` for the crowned digest. Satellites (W7d) must refuse it the
  same way.

#### Fixtures that crown while armed without gate keywords

Every crowning fixture is armed: unarmed, `settle_lane_d` returns nothing.
On main these settle with the legacy call and expect a crown. Under the rule
above they become `bench_hold`, so their owners must pass the gate inputs
(a block hash and a passing bench session, as `helpers.settle_crowning`
does). This PR doesn't edit them. The list comes from running each
directory on main with a spy on `settle_lane_d` and `validate_report`.

| WP | fixture (call site) | tests that crown ungated |
|---|---|---|
| W5c | `tests/accept/w5c/helpers.py` `settle` | `test_w5c_house.py`: the five `test_house_change3_*` tests |
| W7a | `tests/accept/w7a/helpers.py` `settle` / `two_settlements` (BOUNDARY_2 crowns) | `test_w7a_validate.py::test_settle_reports_validate`, `test_w7a_report_fields.py::test_e2e_settle_reports_validate_and_state_quantile_head`, `test_w7a_snapshots.py::test_e2e_report_lists_its_snapshots_and_each_windows_version`, `test_w7a_gate_params.py::test_params_carry_the_parsed_gate_and_source_knobs`, `test_w7a_gate_record.py::test_fixture_gate_is_the_w3b1_draw_and_the_w3b2_outcomes`, `test_w7a_inv11_reasons.py::test_inv11_house_king_slot_burns_with_house_king`. In all, 135 W7a tests pass the crowned BOUNDARY_2 report, with no gate, to `validate_report`, which must then refuse it. Those include `test_w7a_gate_record.py::test_gate_is_optional_reports_without_it_validate` and `::test_no_gated_candidate_means_no_gate_key`, which must use an uncrowned report. |
| W5b | `tests/accept/w5b/helpers.py` (armed, `from_block = ARM_BLOCK`) | none: no W5b test crowns |
| W5a | `tests/accept/w5a/test_w5a_settle.py` `_settle` | `test_e2e_skeleton_refuses_trap_windows_and_still_pays_the_challenger` |
| W1.5 | `tests/accept/w1_5/test_w1_5_e2e.py` `_settle` / `_settle_both` | the seven `[synthetic]` e2e/inv tests that crown, including `test_e2e_audit_passes` and `test_inv9_share_zero_vector_armed_nonempty_lane_d_effect`, which also validate the crowned report |
| W3b3 | `tests/accept/w3b3/helpers.py` `_settle` | `test_w3b3_never_gates.py` (2), `test_w3b3_publish.py::test_the_record_is_published_once_under_its_key`, `test_w3b3_record.py::test_fixture_chain_crowns_y_over_k1_and_k1_over_the_house` |
| W3b5 | `tests/accept/w3b5/helpers.py` `settle` | every test in `test_w3b5_lowfreq.py` that reaches a confirming clear (about 20) |
| W5e | `helpers.settle` | corrected here: the four crowning verdict tests use `helpers.settle_crowning` |

W5d, W7b and W8 don't crown through `settle_lane_d`.

## Not pinned

- The per-verdict gate result fields and the `gate` record's extra keys.
  The W7a RV4 schema change defines them. `validate_report` isn't called on
  gated reports, because the `lane-d-report/1` schema on main has no `gate`
  field (RV4 BLOCKING W7a). Only the refusal of a crowned report without a
  gate is pinned (above). `lane_d_report.py` and the schema file are in
  `PATHS` for the `gate_regression` entry reason, the record the wiring
  needs and that refusal.
- Whether `bench_hold` uses status `hold` or `confirm_pending`, whether a
  held settlement without a block hash writes a `gate` record, the gate's
  bootstrap seed string, and what happens to a confirmed non-best member.
- Whether a hold for a missing gate input counts against `max_bench_holds`
  (W5g), and the low-frequency check (W3b5).

## Fixtures

- Each snapshot has 24 fresh windows over 4 sources (`min_windows` 24,
  `min_clusters` 4), all eligible for every entry, and no two snapshots share
  a window. Every series names a listed feed and carries a `frontier`.
- Forecast qualities are numpy quantiles. X is `sharp` and the house is
  `bad`, so X clears both duels.
- The bench pod is the one external service. `helpers.BenchHost` returns
  recorded per-item CRPS/MASE (a factor below 1 = better than the
  references) through a real `SweepSession`. There's no GPU, torch or
  checkpoint.

## Tests

| Rule | Test |
|---|---|
| first clear → `confirm_pending`, `confirm` record | `test_a_first_clear_is_not_crowned_and_goes_confirm_pending` |
| confirmation on the next disjoint set crowns | `test_confirmation_runs_on_the_next_disjoint_set_and_crowns` |
| failed confirmation → `lost_confirmation` | `test_a_clear_on_the_confirmation_set_fails_and_is_lost_confirmation` |
| `confirm_pending` persists; restart | `test_confirm_pending_persists_until_a_fresh_set_freezes_and_across_restarts` |
| `d_margin` inclusive | `test_d_margin_is_an_inclusive_bar_on_the_lcb` |
| max-T crowns the best | `test_max_t_crowns_the_best_of_a_confirming_cohort` |
| clear iff LCB ≥ d_margin and gain > 0 | `test_every_verdict_clears_iff_lcb_reaches_d_margin_and_gain_is_positive` |
| D20 restart (T9b) | `test_t9b_d20_restart_house_refresh_discards_the_first_clear` |
| D20 original | `test_t9b_d20_original_confirms_against_the_king_of_the_first_clear` |
| `rows_sha256` | `test_rows_sha256_is_the_canonical_digest_of_the_ordered_rows` |
| determinism (input order, hash seed) | `test_verdicts_are_deterministic_in_entry_input_order`, `test_verdicts_are_independent_of_the_hash_seed` |
| gate pass crowns, draw recorded | `test_gate_pass_crowns_and_records_the_draw` |
| first clears not gated | `test_first_clears_are_not_gated` |
| `gate_regression` (T4) | `test_t4_gate_regression_is_lost_with_a_typed_reason` |
| `checkpoint_fault` (T17) | `test_t17_candidate_fault_on_the_drawn_suite_is_checkpoint_fault` |
| `bench_hold` keeps the chance | `test_reference_fault_is_bench_hold_and_the_clear_survives` |
| sentinel swept, never gates | `test_the_sentinel_is_swept_and_never_gates` |
| draw follows the block hash | `test_the_draw_follows_the_boundary_block_hash` |
| losers and first clears not swept | `test_losers_and_first_clears_are_never_swept` |
| every confirming clear gated | `test_every_confirming_clear_is_gated_and_the_best_passing_one_crowns` |
| miner king: the house still bars (T3) | `test_t3_with_a_miner_king_the_house_still_bars_the_gate` |
| armed crowning without bench / block hash → `bench_hold` (RV5) | `test_armed_crowning_without_bench_or_block_hash_is_bench_hold[legacy_call, bench_none, hash_none, both_none]` |
| unarmed legacy call unchanged | `test_unarmed_legacy_call_is_unchanged` |
| crowned report without a gate refused | `test_a_crowned_report_without_a_gate_record_is_refused[absent, null]` |
| uncrowned report needs no gate | `test_an_uncrowned_report_needs_no_gate_record` |

## On main

- 21 tests fail on an AssertionError:
  - the 10 gate tests and the 3 gated fail-closed tests (bench_none,
    hash_none, both_none), plus the 2 refusal tests and the uncrowned-report
    test, because the gate keywords are absent;
  - the 4 crowning verdict tests (RV5: they now crown through
    `settle_crowning`). One of them is D20 `"original"`, which also asserts
    the confirmation isn't skipped as `king_unscored`;
  - `[legacy_call]`, because main crowns without a gate.
- The other 9 pass: the 8 non-crowning verdict tests (W5b and W5c built
  them, and W5e owns them) and the unarmed legacy test.
