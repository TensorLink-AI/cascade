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
which checks the signature before it calls. The tests don't pin the
behaviour when these keywords are omitted. The W5b/W5c/W7a fixtures omit
them and still expect a crown.

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

## Not pinned

- The per-verdict gate result fields and the `gate` record's extra keys.
  The W7a RV4 schema change defines them. `validate_report` isn't called on
  gated reports, because the `lane-d-report/1` schema on main has no `gate`
  field (RV4 BLOCKING W7a). `lane_d_report.py` and the schema file are in
  `PATHS` only for the `gate_regression` entry reason and the record the
  wiring needs.
- What a `bench=None` call does, whether `bench_hold` uses status `hold` or
  `confirm_pending`, the gate's bootstrap seed string, and what happens to
  a confirmed non-best member.
- The low-frequency check (W3b5) and `max_bench_holds` (W5g).

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

## On main

- 11 tests fail on an AssertionError: the 10 gate tests (the gate keywords
  are absent) and D20 `"original"` (`king_unscored`).
- The other 11 pass. W5b and W5c built those rules, and W5e owns them.
