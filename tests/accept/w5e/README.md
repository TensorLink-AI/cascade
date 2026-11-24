# W5e acceptance tests: verdict, crown at the first verdict, `rows_sha256`, gate wiring

W5e (plan rev6 §3.0 W5e row, §1.3 "Verdict", §1.3b, §1.5, A2, A9, T3, T4,
T17; **rev6.11 (b)**: the confirmation duel is DROPPED, owner 2026-11-23)
pins how a judged cohort becomes a crown. The orchestrator also assigned W5e
the wiring of the W3b gate (`lane_d_gate`, `lane_d_bench`) into
`settle_lane_d`. These tests are the spec. The builder may touch only the
paths in `PATHS`. The W1.5, W5a, W5b, W5c, W5d, W7a and W3b tests must stay
green.

## Contract (`settle_lane_d`, `cascade/validator/lane_d.py`)

### Verdict

- **(rev6.23)** The bar is the crown bar in force at the settlement
  (`crown_bar.bar`): `d_margin_start` at KING_D's tenure 0 (the house's
  too), linear to `d_margin` at `d_margin_warmup_settlements`, flat after;
  a crown restarts the tenure (`test_w5e_crown_bar.py`). Below, `d_margin`
  means that bar.
- A member clears iff `LCB_α ≥ d_margin` (inclusive) **and** its point gain
  is strictly positive. The LCB is the paired cluster bootstrap (clusters =
  source), max-T over the cohort, and doesn't depend on `d_margin`.
- Each verdict's `lcb`, `gain` and `cleared` equal `judge_cohort` recomputed
  from the report's own rows over `cohorts[].window_ids`, with the members in
  cohort order and `seed = report["seed"]`.
- **Max-T crowns the best.** When several members clear in one settlement,
  the crown goes to the passing one with the highest LCB.

### Crown at the first verdict (rev6.11 (b))

- A first clear (max-T LCB against KING_D on the frozen set) that passes A10
  (W3b5) and the §1.3b gate is **crowned at that settlement**: verdict
  outcome `crowned`, `report.crowned`, state `king`, entry `crowned`, pay
  slot 0 = the miner's UID.
- **No confirm stage.** A verdict's and a cohort's `stage` is `first` or
  absent (W7a decides); no verdict carries a `confirmation` block; no state
  entry carries a `confirm` record (absent or `null`). `confirm_pending`
  is in none of `lane_d.OPEN_STATUSES`, `lane_d_queue.OPEN_STATUSES` and
  `lane_d_status.OPEN_STATUSES`, and neither `confirm_pending` nor
  `lost_confirmation` appears in any report or state the settlement writes.
- A verdict that doesn't clear is `lost` / `lost_duel`.
- A crowned king is never judged again on a later set; there is no second
  set and no D20 king-change rule: a house refresh after a crown leaves the
  king in place.
- **`confirm_on_king_change` is a legacy key, parsed and ignored.** Both
  values give the same `cohorts`, `verdicts`, `rows_sha256`, `entries`,
  `crowned` and `params`, and `params` no longer carries it.
- **Restart-safe:** a fresh call on a copy of the prior state file gives the
  same verdicts, crown and `rows_sha256`.

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

- **Who is gated:** every verdict that cleared (rev6.11: a first clear is the
  crowning candidate) and didn't fail A10, plus every held candidate.
  Losers are never swept.
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
  | `bench_hold` | open (`hold`) / `bench_hold` | no; not lost |

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
  for a settlement where no member clears (losers, pending floors). The
  verdict tests use it that way.
- **Armed, a crowning candidate, no gate inputs: `bench_hold`.** A member
  that clears at its first verdict would crown, but the settlement has no `bench`
  session or no `boundary_block_hash`. That covers the legacy call without
  the keywords, `bench=None`, `boundary_block_hash=None`, and both. The §1.3b
  gate can't run, so the member is not crowned:
  - `report.crowned` is `null`, `king_after == king_before`, the state king
    is unchanged and the king slot still burns;
  - the entry (state and report) is `hold` / `bench_hold`: open, not
    `lost` or `closed`, not the miner's fault (rev6.11: no `confirm_pending`);
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

- The per-verdict gate result fields and the `gate` record's extra keys
  (W7a). Only the refusal of a crowned report without a gate is pinned.
- The outcome name of a held verdict (anything but `crowned` / `lost`) and
  of a passing max-T runner-up (anything but `crowned`): W7a's enum.
- Whether a held settlement without a block hash writes a `gate` record,
  the gate's bootstrap seed string, and what happens to a passing non-best
  member.
- What happens to a hold when KING_D changes before the re-gate (W7a's audit
  pins that a re-gated `held_from.king_d` must equal `king_before`).
- Whether a hold for a missing gate input counts against `max_bench_holds`
  (W5g), and the low-frequency check (W3b5).

## Fixtures

- Each snapshot has 24 fresh windows over 4 sources (`min_windows` 24,
  `min_clusters` 4), all eligible for every entry, and no two snapshots share
  a window. Every series names a listed feed and carries a `frontier`.
  **(rev6.10)** Every listed feed pins `embargo_hours: 0` (W6b), the rev3
  rule, so the fixtures keep their verdicts under the embargo-aware W5a,
  which fails a feed without it closed.
- Forecast qualities are numpy quantiles. X is `sharp` and the house is
  `bad`, so X clears.
- The bench pod is the one external service. `helpers.BenchHost` returns
  recorded per-item CRPS/MASE (a factor below 1 = better than the
  references) through a real `SweepSession`. There's no GPU, torch or
  checkpoint.

## Tests

| Rule | Test |
|---|---|
| first clear crowns at the first verdict | `test_a_first_clear_is_crowned_at_the_first_verdict` |
| no confirm stage / status anywhere | `test_no_confirm_stage_or_status_exists` |
| no second set | `test_the_crowned_king_is_not_judged_again_on_a_second_set` |
| a loss is `lost_duel` | `test_a_failed_first_verdict_is_lost_duel`, `test_d_margin_is_an_inclusive_bar_on_the_lcb` |
| restart safety | `test_the_first_verdict_is_restart_safe` |
| max-T crowns the best | `test_max_t_crowns_the_best_of_a_clearing_cohort_at_the_first_verdict` |
| clear iff LCB ≥ d_margin and gain > 0 | `test_every_verdict_clears_iff_lcb_reaches_d_margin_and_gain_is_positive` |
| `confirm_on_king_change` parse-only | `test_confirm_on_king_change_is_parsed_and_ignored` |
| no D20 | `test_a_house_refresh_never_discards_or_restarts_a_crown` |
| `rows_sha256` | `test_rows_sha256_is_the_canonical_digest_of_the_ordered_rows` |
| determinism (input order, hash seed) | `test_verdicts_are_deterministic_in_entry_input_order`, `test_verdicts_are_independent_of_the_hash_seed` |
| gate pass crowns at the first verdict, draw recorded | `test_gate_pass_crowns_at_the_first_verdict_and_records_the_draw` |
| a first clear is gated | `test_a_first_clear_is_gated_at_the_settlement_it_clears` |
| `gate_regression` (T4) | `test_t4_gate_regression_is_lost_with_a_typed_reason` |
| `checkpoint_fault` (T17) | `test_t17_candidate_fault_on_the_drawn_suite_is_checkpoint_fault` |
| `bench_hold` = a first clear the gate couldn't decide; re-gated, not re-duelled (W5g follow-up) | `test_reference_fault_is_bench_hold_and_the_first_clear_survives` |
| sentinel swept, never gates | `test_the_sentinel_is_swept_and_never_gates` |
| draw follows the block hash | `test_the_draw_follows_the_boundary_block_hash` |
| losers never swept; no clear, no gate | `test_losers_are_never_swept`, `test_no_clear_means_no_sweep_and_no_gate_record` |
| every first clear gated, best passing crowns | `test_every_first_clear_is_gated_and_the_best_passing_one_crowns` |
| miner king: the house still bars (T3) | `test_t3_with_a_miner_king_the_house_still_bars_the_gate` |
| armed crowning without bench / block hash → `hold` / `bench_hold` (RV5) | `test_armed_crowning_without_bench_or_block_hash_is_bench_hold[legacy_call, bench_none, hash_none, both_none]` |
| unarmed legacy call unchanged | `test_unarmed_legacy_call_is_unchanged` |
| crowned report without a gate refused | `test_a_crowned_report_without_a_gate_record_is_refused[absent, null]` |
| uncrowned report needs no gate | `test_an_uncrowned_report_needs_no_gate_record` |

## On main

### Correction (rev6.11 (b)): the confirmation duel is dropped

Tests-correction PR (w0c `--tests-correction`). Every W5e test file changes
(`helpers.py` is in each file's closure). On main (the confirm stage still
built) 23 tests fail on an `AssertionError`: a first clear is
`confirm_pending`, never gated, never crowned; `OPEN_STATUSES` still holds
`confirm_pending`; `params` still echoes `confirm_on_king_change`. The other
9 pass (the non-crowning verdict, determinism, rows and unarmed tests).
A scratch prototype of the contract (stage removed, bench_hold status
`hold`) passed all 32, plus every W7a test of the companion correction.

`PATHS` gains `cascade/validator/lane_d_queue.py` and
`cascade/validator/lane_d_status.py` (their `OPEN_STATUSES`).

**Other WPs whose fixtures expect the confirm stage** (corrections needed
before the W5e implementation can pass CROSS; not edited here): W3b3, W3b5,
W5c, W5f, W5g, W1.5 (crowning e2e chains at boundary 2), W7c, W7d, W9.

## PATHS correction (orchestrator 2026-11-23): audit and replay

Plan rev6.11 (b) ("W5e: remove the stage") counts the confirm stage in
`cascade/validator/lane_d.py`, `cascade/audit/lane_d.py` and
`cascade/satellite/lane_d_replay.py`. Dropping it from the producer alone
leaves the audit and the replay expecting a confirm-stage crown and the
`confirm_on_king_change` config echo. That fails 124 honest-chain tests
that pass on main (W5c, W5f, W7a, W7d, W8, W8b, W9) and the W1.5 e2e audit.
So W5e's PATHS add those two files (only those): W5e removes the stage from
the producer, the audit and the replay together. The audit and replay
rules W7a / W9 / W7d add on top (embargo, draw and fresh-share fields,
`confirmation_stage_present`) stay those WPs' work. No test changed.

## Multichannel windows (rev6.18 F2): `test_w5e_multivariate.py`

Plan rev6.18 A4 (DEC-CA-0041). Fixture: `tests/accept/w7a/mv.py`.

- `settle_lane_d(..., max_channels=)` takes `[generator] max_channels` (the
  wiring passes `cfg.generator.max_channels`, `lane_d_wiring._max_channels`;
  `None`, the legacy call, scores no multichannel window).
- Each multichannel pool window is reduced to its pick before anything is
  judged (`cascade.eval.lane_d_channels`; README w5d): empty channels drop,
  a wider series keeps `max_channels` channels by the boundary block hash;
  without a block hash an over-cap series leaves the pool. The pick of an id
  is its first version's in pin order; a version that cannot carry it leaves
  the pool.
- A multichannel window is ONE row per model: `crps`/`mase` are the
  channel means of the per-channel univariate scores (zero-target channels
  masked from `crps` only); it is one resampling unit in its source's
  cluster (evidence counts windows and sources, never channels).
- A forecast's expected shape is `(k, horizon, 9)` for a k-channel window
  (k >= 2), else `(horizon, 9)`; a mismatch is a malformed forecast.
- Univariate results are unchanged: the same chain stored 1-D or as `(1, L)`
  gives the same rows, point gains and crown; a univariate settlement's
  report body is the same with or without `max_channels`.
- The inputs source (`LaneDInputsSource._windows(pools, boundary)`) sends the
  eval host only the picked channels, reading the boundary block hash only
  when a series is wider than the cap (`channel_block_hash`); the wiring
  refuses inputs picked by another hash (`inputs_unavailable`).

On main these tests fail on the `max_channels` assertion of `mv.settle_mv`,
`cannot import name '_max_channels'` (`lane_d_wiring`, in PATHS) or an
assertion wrapping the old `_windows` signature.
