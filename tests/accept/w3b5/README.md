# W3b5 acceptance tests: always-run low-frequency non-inferiority check (A10)

W3b5 (plan rev6 §3.0 W3b5 row, §1.3b "Always-run low-frequency check (A10)",
§1.5, §1.7, §9, §4 T4b) adds the low-frequency check to the lane D
settlement. These tests are the spec. The builder may touch only the paths
in `PATHS`. The W1.5, W5a–W5e, W7a and W7b tests must stay green.

Accept: `scripts/laned/accept W3b5` (CI). T4b is
`test_t4b_a_model_that_regresses_only_on_low_frequency_series_is_blocked`.

## The plan text pinned

W3b5 row: "**(rev3)** Always-run low-frequency private-pool non-inferiority
check (`lowfreq_freqs`, `tau_public`; `lowfreq_regression`) (A10) | W5d | ci |
`scripts/laned/accept W3b5` (T4b)".

T4b: "**(rev3)** A model that regresses only on low-frequency series |
Blocked by `lowfreq_regression`".

§1.3b A10: every crowning candidate runs "a paired non-inferiority check on
a low-frequency slice of the private pool". The slice is `[lane_d]
lowfreq_freqs`. The test is CRPS and MASE `LCB_α ≥ −tau_public` against both
KING_D and the house, on the same window set, scored by the lane D scorer.
On failure the reason is `lowfreq_regression` and the candidate isn't
crowned. §1.7: "Daily-or-slower series almost never qualify, which is why
A10's low-frequency check uses the private pool instead."

## Contract (`settle_lane_d`, `cascade/validator/lane_d.py`)

### Who is checked: always, every crowning candidate

- A **crowning candidate** is a verdict that cleared at `stage: "confirm"`
  (W7a's definition). Each one is checked, in every settlement, whatever
  else happens. It doesn't depend on the gate draw or on the W3b gate being
  wired: on a main without W5e's gate keywords these tests make the legacy
  call, and once `settle_lane_d` accepts them they pass a healthy bench (see
  "Fixtures", RV5).
- First clears (`stage: "first"`) and losers aren't checked. Their
  `lowfreq` block stays W7a's `not_candidate` with `n_windows: 0`. A first
  clear still goes `confirm_pending`, and a confirmation loser is still
  `lost_confirmation`, even when it would regress on the slice.
- When several candidates confirm in one settlement, each is checked on its
  own. A regression by the max-T best doesn't block a passing runner-up,
  which is crowned (the same rule W5e pins for the gate).

### The slice: the private pool at `lowfreq_freqs`

- The slice is the private pool's windows whose `freq` is in
  `[lane_d] lowfreq_freqs`. It is **not** limited to the forward-only
  frozen set (§1.7). In the fixtures every low-frequency series is years
  old, so the §1.3 rule refuses it (`before_margin`). It is in no cohort's
  `window_ids` and has no `rows`, and it still makes up the whole slice.
- `freq` values are matched exactly against `lowfreq_freqs`. Weekly series
  are in the slice under `["W"]` and outside it under `["D"]`.
- `n_windows` is the slice size. Every fixture slice window has a finite
  target and forecasts from the candidate, KING_D and the house.
- The candidate, KING_D and the house are all scored on the same slice
  windows with the lane D scorer (`score_window`, per window CRPS and MASE).

### The test: CRPS and MASE, against KING_D and the house

- For each reference (`king_d`, `house`) and each metric (`crps`, `mase`),
  the check takes a paired LCB_α (`[lane_d] alpha`) of the candidate's
  relative improvement over the reference on the slice. The repo's paired
  bootstrap is reused, for example `cascade.eval.expert_lane.paired_geomean_lcb`
  as `lane_d_bench` uses it for the drawn suite. The bootstrap form and seed
  string aren't pinned; they must be deterministic.
- `pass` iff all four LCBs are `≥ −tau_public`. Otherwise it's `fail`.
- When the house holds KING_D, both references are the house, and
  `lcb.house == lcb.king_d`.
- **The fixtures make the expected LCB exact.** Every low-frequency quantile
  sits above the target by a per-level offset times one scalar per window,
  so pinball loss and MASE are linear in the offsets. A candidate's CRPS and
  MASE ratios to a reference are then the same on every window, and any
  paired bootstrap of the relative improvement is degenerate at
  `1 − ratio`. The tests compare to that value within 1e-6
  (`helpers.expected_lcbs`). The jittered fixtures (5% per window) keep
  every window's ratio on one side of `−tau_public`, so their outcome also
  holds for any bootstrap.
- Each metric must pass on its own. A MASE-only regression (better CRPS) and
  a CRPS-only regression (better MASE) both fail.
- `tau_public` comes from config. A 3% regression fails at the default 0.01
  and passes at 0.05. A 0.5% regression passes at 0.01, and a 2% regression
  fails.

### Effect

| result | entry (state and report) | verdict `outcome` | crown |
|---|---|---|---|
| `pass` | `crowned` (when it's the best passing candidate) | `crowned` | yes |
| `fail` | `lost` / `lowfreq_regression` | `lost` | no; KING_D, `king_after` and the pay court unchanged |
| `no_windows` | unchanged by A10 | unchanged | not blocked |

A `lowfreq_regression` entry is judged, so it's in `judged` and not in
`skipped`. It is never judged again.

### Too few low-frequency windows: the floor is one window

The plan sets no floor for the A10 slice. Its floors (`min_windows`,
`min_clusters`) belong to the forward-only frozen sets in §1.3, and A10 is
"always-run". So the tests pin:

- **An empty slice** is `no_windows` (W7a's enum value, used here as the
  insufficient-data state). It **does not block**. The plan blocks only "on
  failure" of the test, and with no window there is no test. A floor that
  blocks would refuse every candidate whenever the private pool has no
  low-frequency series.
- **One window or more** is evaluated, `pass` or `fail`. A single
  regressing window blocks. Treating small slices as `no_windows` would let
  a regression through with a recorded skip. Raising the floor is a plan
  change, not a W3b5 build choice.

The report enum has no `insufficient` value, and these tests don't add one.

### The report's `verdicts[].lowfreq` block

- `result`, `freqs` (= `lowfreq_freqs`), `n_windows` and `tau_public`, as
  W7a pins them.
- For `pass` and `fail`, the block also carries
  `lcb = {"king_d": {"crps", "mase"}, "house": {"crps", "mase"}}`, as finite
  floats. `result == "fail"` iff some value is `< −tau_public`.
- For `no_windows` and `not_candidate`, the block is exactly W7a's four keys,
  with no `lcb` (W7a's `test_e2e_lowfreq_result` compares the whole dict).
- So the schema file gains `lcb` as an **optional** property of `lowfreq`,
  still closed (`additionalProperties: false`). `validate_report` must
  accept the pass and fail reports.

### Determinism

- Settling boundary(2) twice, from copies of the same boundary(1) state,
  gives identical verdicts (including `lowfreq`), `rows_sha256`, `crowned`,
  `entries`, `king_after`, `judged`, `skipped` and entry state. The slice is
  jittered, so the bootstrap is not degenerate.
- Permuting the `entries` input gives identical `verdicts`, `rows_sha256`,
  `crowned`, `entries` and `judged`.

## Not pinned

- The bootstrap form (per-window or cluster, uncorrected or max-T) and its
  seed, beyond determinism and the exact degenerate value.
- Whether prior snapshots in `prior_pool_dirs` add slice windows. Every
  fixture slice is in the newest snapshot only.
- What a slice window missing a forecast or a usable target does.
- Whether the report publishes the slice's per-window rows. Satellite replay
  of A10 belongs to W7d/W9.
- The interaction with the W3b gate. Both must pass to crown, but W5e owns
  the gate wiring.

## Fixtures

- Each snapshot has 24 fresh hourly windows over 4 sources (`min_windows`
  24, `min_clusters` 4), all eligible for every entry, plus the
  low-frequency series (`D` or `W`, over 2 sources).
- Hourly quality comes from numpy quantiles. X is `sharp`, Z and Y are
  `good` and the house is `bad`, so X clears both duels.
- The miner-king tests crown Y at boundary(2) (non-inferior on its own
  slice), then judge X (revealed a day later) against Y and the house at
  boundaries 3 and 4.
- No service is mocked. No GPU, torch or checkpoint is used.
- **Gate inputs (W5e fail-closed, RV5).** Once W5e lands
  (`tests/accept/w5e/README.md` "Fail closed"), an armed settlement that
  would crown without a `bench` session and a `boundary_block_hash` is
  `bench_hold`, not a crown, so every test here that reaches a confirming
  clear would stop crowning for a reason that isn't A10. `helpers.settle`
  therefore passes `boundary_block_hash` (`helpers.gate_block_hash`),
  `sentinel=None` and a healthy `bench` (`lane_d_bench.SweepSession` over
  `helpers.HealthyBench`) whenever `settle_lane_d` accepts those keywords
  (`inspect.signature`), and makes the legacy call otherwise. The bench is
  the one external service: every digest but the house returns the same
  recorded rows (a tie with a miner KING_D never regresses) and the house is
  10% worse, so the §1.3b gate passes every crowning candidate on any drawn
  suite and only the A10 check decides. No assertion changed; on main the
  same 17 fail and 5 pass.

## Tests

| Rule | Test |
|---|---|
| T4b: regression only on low frequency blocks | `test_t4b_a_model_that_regresses_only_on_low_frequency_series_is_blocked` |
| non-inferior crowns, `pass` + LCBs | `test_a_candidate_not_worse_on_low_frequency_series_is_crowned_with_pass` |
| inside / beyond `tau_public` | `test_a_regression_within_tau_public_passes_non_inferiority`, `test_a_regression_just_beyond_tau_public_fails` |
| `tau_public` from config | `test_tau_public_is_read_from_config` |
| CRPS and MASE each | `test_a_mase_only_regression_is_blocked`, `test_a_crps_only_regression_is_blocked` |
| slice = `lowfreq_freqs` | `test_the_slice_follows_lowfreq_freqs`, `test_series_outside_lowfreq_freqs_are_not_in_the_slice` |
| slice = private pool, not forward-only | `test_the_slice_is_the_private_pool_not_the_forward_only_set` |
| floor = one window; empty = `no_windows`, non-blocking | `test_a_single_low_frequency_window_is_still_checked`, `test_an_empty_slice_is_no_windows_and_never_blocks` |
| only crowning candidates | `test_first_clears_are_not_checked_and_go_confirm_pending`, `test_a_confirmation_loser_is_lost_confirmation_not_lowfreq_regression` |
| blocked is final | `test_a_blocked_entry_is_not_judged_again` |
| every candidate; runner-up crowns | `test_every_crowning_candidate_is_checked_and_a_passing_runner_up_crowns` |
| miner KING_D: both references | `test_with_a_miner_king_a_regression_against_the_house_alone_blocks`, `test_with_a_miner_king_a_regression_against_king_d_alone_blocks`, `test_with_a_miner_king_non_inferior_on_both_crowns` |
| determinism | `test_the_check_is_deterministic_across_a_restart[pass/fail]`, `test_the_check_is_independent_of_entry_input_order` |

## On main

- 17 tests fail on an AssertionError. Main counts the slice inside the
  frozen set (0 here), reports `no_windows` and crowns regardless.
- 5 pass: the empty slice, series outside `lowfreq_freqs`, first clears,
  confirmation losers and input-order determinism. W7a and W5e built those
  rules, and W3b5 owns them as its assertions.

## Correction (plan rev6.10 (a) / rev6.12 (a)): fixture feeds pin `embargo_hours`

W5a (tests in #758) makes forward eligibility embargo-aware and fails a
listed feed whose `embargo_hours` is missing closed as `unlisted_feed`. This
WP's pool fixture wrote feeds without the key, so once the W5a
implementation lands every fixture window would be refused. Every fixture
feed now pins `embargo_hours: 0.0` (e = 0, the rev3 rule byte for byte, so
every verdict is unchanged); on main the key is ignored. No assertion
changed; every test passes on main and with the W5a implementation.

## Correction (plan rev6.11 (b)): crown at the first verdict

The confirmation duel is dropped (owner 2026-11-23): a crown is decided at
the first verdict, first clear (max-T LCB vs KING_D) ∧ A10 ∧ the §1.3b gate.
There is no `confirm` stage, no `confirm_pending` status or reason and no
`lost_confirmation`; a held first clear is `hold` / `bench_hold`. This WP's
crowning chains move to that rule with nothing weakened. Boundary(1) lists no entry; the candidate is judged, A10-checked and gated at its first verdict at boundary(2) (the miner-king chain: Y at boundary(2), X at boundary(4)). The first-clear test now pins that a first clear is checked at once, and the loser test pins `lost_duel`. On main the
old code still runs the confirmation stage, so the changed tests fail on an
assertion until the W5e implementation lands.

## REMOVED (plan rev6.18 A7, F4; owner 2026-11-23 DECIDED)

Owner decision (rev6.18, verbatim): "The low-frequency check A10 is
removed." This is a correction of the whole suite above, which is kept for
history only: `test_w3b5_lowfreq.py` and its `helpers.py` are deleted with
the check. `test_w3b5_removed.py` pins the removal:

- `cascade/validator/lane_d_lowfreq.py` is gone and `settle_lane_d` runs and
  records no A10 check (no `lowfreq` block, no `lowfreq_regression`);
- the `lane-d-report/1` schema has no `lowfreq` / `lowfreq_lcbs` /
  `lowfreq_result` definitions, no verdict `lowfreq`, no
  `lowfreq_regression` entry reason and no `params.lowfreq_freqs`;
- `lane_d_report` exports no `LOWFREQ_RESULTS` and has no `_check_lowfreq`;
- the audit (`cascade/audit/lane_d.py`, chain mode `lane_d_chain.py`) has no
  lowfreq rule;
- `[lane_d] lowfreq_freqs` is retired: not parsed, listed in
  `RETIRED_KEYS["lane_d"]` (a deployed file that carries it loads), and gone
  from both shipped files.

Slow cadences are now covered by the public half (rev6.18 A1-A3:
GiftEvalPretrain + RMISC, 50% of the combined score). `tau_public` stays: the
§1.3b gate uses it.
