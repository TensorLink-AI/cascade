# W5c acceptance tests: cohort intersection, latest-T deferral, the house bar

W5c (plan rev6 §3.0 W5c row, §1.3 "Cohorts (several entries in one
settlement)", T9, and the accepted Proposed design change 3) pins how the
entries frozen in one settlement are judged together. These tests are the
spec. The builder may touch only the paths in `PATHS`. The W1.5, W5a and W5b
tests must stay green.

W5b already judges a cohort on the intersection of its members' frozen sets
and keeps a latest-T deferral (see `tests/accept/w5b/README.md`). W5c owns
those rules as its own assertions below. They pass on main. W5c's new
behaviour is change 3, and those tests fail on main.

## Contract (`settle_lane_d`, `cascade/validator/lane_d.py`)

### Cohort formation

- A cohort is every open entry, judged against the same KING_D at the same
  stage, whose `frozen_set` is frozen at this settlement. That includes an
  entry frozen and deferred at an earlier settlement.
- An entry that is still unfrozen is not a member. Its T does not enter the
  cohort's max-T, and it does not shrink the intersection. It stays
  `pending` / `pending_floor`.
- `cohorts[].members` is ordered by T (`reveal_ts`) ascending, with ties
  broken by digest. It is not digest order and not input order.
- `cohorts[].t_max` is max(T over the members and the king).

### The judged set

- The set is the intersection of the members' frozen sets that is eligible
  (§1.3, W5a) for the cohort's `t_max`. It also needs a usable target and a
  usable forecast from the king and every member.
- A window that is eligible for a member's own T but not for `t_max` is
  excluded, even though it is in that member's frozen set.
- Every member and the king are scored on exactly this set
  (`rows`, `cohorts[].window_ids`, `set_sha256 = sha256("\n".join(sorted ids))`).

### Floors after the intersection: latest-T deferral

- If the set misses `min_windows` or `min_clusters`, the member with the
  latest T is deferred. Then `t_max` and the set are recomputed for the
  remaining members, and this repeats until the set meets the floors or no
  member is left.
- A deferred entry is not judged: no rows, no verdict, not in `judged`, and
  no cohort if every member is deferred. It gets one `skipped` record with
  the typed reason `pending_floor` (in `ENTRY_REASONS`), and its state stays
  `pending` / `pending_floor` with `used_windows == []`. Its `frozen_set`
  stays frozen and unchanged.
- At the next settlement it is a cohort member again, judged on its
  unchanged frozen set: the cohort's `set_sha256` equals the frozen set's.

### Verdict: max-T against KING_D

- Each verdict's `lcb`, `gain` and `cleared` equal
  `judge_cohort(king_scores, member_scores, clusters=sources, cfg, seed=report["seed"])`.
  This is computed from the report's own rows over `cohorts[].window_ids` in
  order, with the members in cohort order. It is the joint `cohort_maxt_lcbs`
  over the cohort against KING_D (`verdicts[].king_digest`).

### Proposed change 3 (accepted): the house bar

- When a miner holds KING_D and the house is a different model, a miner
  challenger is also compared with the current house on the same set. It
  clears only if it clears `d_margin` against KING_D (as above) **and**
  against the house.
- **The house ranks first on ties.** An exact copy of the house's forecasts
  does not clear. It ends `lost` / `lost_duel`, with `cleared: false` and
  outcome `lost`, and the king is unchanged.
- A challenger that beats KING_D but does not beat the house by `d_margin`
  also ends `lost` / `lost_duel`.
- A challenger that beats both clears as before (`confirm_pending`).
- The house is scored on the cohort's set. Its `rows` cover exactly
  `cohorts[].window_ids`, next to the king's and the members'.
- The verdict's `king_digest` stays KING_D.

### The house judgement in the report (RV5)

§1.5 and §14.3 rules 1-2: a satellite re-derives every verdict from signed
fields alone. So when a miner holds KING_D and the house is a second bar,
the signed report records the house judgement.

New report fields (for the W7a schema owner: add exactly these to
`lane-d-report/1`; both are optional keys, present only in a cohort with a
house bar):

- `cohorts[].house_digest` (string, 64-hex): the house digest the cohort's
  members were also judged against. It equals `report["house"]["digest"]`
  and differs from `cohorts[].king_digest`.
- `verdicts[].house` (object, exactly `digest`, `lcb`, `gain`, `cleared`):
  `digest` equals the cohort's `house_digest`; `lcb` / `gain` / `cleared`
  equal `judge_cohort(house_scores, member_scores, clusters=sources, cfg,
  seed=report["seed"])` on the report's own rows over
  `cohorts[].window_ids`, members in cohort order (the same max-T rule as the
  KING_D judgement, the house as the bar).

Rules:

- `verdicts[].lcb` / `gain` stay the KING_D judgement, and
  `verdicts[].cleared == (lcb >= d_margin and gain > 0) and house.cleared`.
  An exact copy of the house has `house.gain == 0.0` and `house.cleared ==
  false`.
- Rows per cohort = `len(window_ids) * (2 + len(members))` with a house bar
  (KING_D, house, members), `len(window_ids) * (1 + len(members))` without.
- The two-bar rule is for **miner** cohorts. At a refresh boundary the new
  house is also a challenger (§1.4: "Refresh: it's refreshed whenever that
  digest changes ... It enters at that settlement"; §2 item 2: "The house
  can retake KING_D when a better lane A king beats it on forward
  windows"), judged in its own cohort: W5f's contract
  (`tests/accept/w5f/README.md` §4: `members == [house]`, `king_digest` =
  KING_D, no `house_digest`). `test_house_report_rows_account_for_the_house`
  skips that cohort (the one whose members contain the house digest) for the
  bars check, asserts only its W5f shape when present (it is absent before
  W5f), and requires at least one miner cohort to have been checked.
- **When the house holds KING_D, neither key appears** (not `null`, not a
  copy of the KING_D judgement). Why absent:
  - there is one bar, so a copy would be a second signed reading of the same
    number that every satellite must cross-check for no information;
  - whether a house bar applies is itself re-derivable
    (`cohorts[].king_digest != report["house"]["digest"]`), so an omission
    can't hide anything: the audit requires the keys exactly when it applies;
  - a house-KING_D report stays exactly today's shape, so it stays valid
    under the current schema (`additionalProperties: false` on cohorts and
    verdicts) and the W1.5 / W8 `verify_report` tests stay green.
- The audit (`cascade.audit.lane_d.check_lane_d`, now in `PATHS`) counts
  the house's rows, re-derives both judgements from the rows, and returns a
  `FAIL` `CheckResult` (never an exception) when a house judgement is
  omitted, nulled, names another digest, disagrees with the rows, is flipped
  either way, has its rows dropped, or appears where the house holds KING_D.
  The tampered reports are re-signed by the signer, so the signature alone
  never decides.
- **Schema validation of the new keys belongs to W7a** (its schema fix is
  pending). These tests do not call `validate_report` / `verify_report` on a
  report that carries them. Until W7a adds them, a miner-KING_D report
  fails `validate_report` with `unknown_field`: land the W5c implementation
  with, or after, the W7a schema change.

### Determinism

- The same inputs, in any input order of `entries`, give identical
  `cohorts`, `verdicts`, `rows`, `rows_sha256`, `judged` and `entries`, the
  same `skipped` multiset, and the same state, apart from `last_report`
  (the signature is randomized).

## Fixtures

- Each pool has four sources and splits evenly over them, so the 0.25
  source cap never drops a window. The floors are lowered to 24 windows and
  4 sources. Every series names a feed listed in `feeds` and carries a
  `frontier` (W5a fails closed).
- The cohort pool has three §1.3 bands, a day apart. Band A (8 windows) is
  eligible for T_A only, band B (8) for T_A and T_B, and band C (24) for all
  three. A deferral is forced by leaving forecasts absent (not malformed) for
  some of a member's windows.
- The house tests run real settlements. K1 first clears against the genesis
  house, then confirms on a disjoint snapshot and is crowned. Then the house
  refreshes to H1 and challenger Y is judged. No state is hand-written.
- They use numpy quantile forecasts: no torch, no checkpoint, no signature
  verification.
- **Gate inputs (W5e fail-closed, RV5).** Once W5e wires the §1.3b gate,
  an armed settlement that would crown without a bench session and a
  boundary block hash is `bench_hold`, not a crown (see
  `tests/accept/w5e/README.md`). `helpers.settle` therefore passes
  `boundary_block_hash`, `sentinel=None` and a healthy `bench`
  (`lane_d_bench.SweepSession` over `helpers.HealthyBench`) whenever
  `settle_lane_d` accepts those keywords (`inspect.signature`), and makes
  the legacy call otherwise, so the tests pass both on today's main and
  after W5e. The bench is the one external service: every digest but the
  current house returns the same recorded rows (a tie with a miner KING_D
  never regresses) and the house is 10% worse, so every crowning candidate
  passes the gate on any drawn suite and the duel alone decides each
  outcome. Only the house tests and the house-report tests crown (K1 at
  BOUNDARY_2). The house-report tests run `check_lane_d` on that crowned
  report: on main it carries no `gate`; the gate record arrives with W5e
  (written by the gated settlement, a `pass` for the crowned digest). No
  assertion changed.

## Tests

| Rule | Test |
|---|---|
| fixture sanity (bands, frozen sets) | `test_cohort_fixture_bands_and_frozen_sets` |
| cohort = frozen entries, T order, `t_max` | `test_t9_cohort_is_every_entry_frozen_this_settlement_ordered_by_t` |
| intersection eligible for max-T (exact ids) | `test_t9_cohort_is_judged_on_the_intersection_eligible_for_max_t` |
| unfrozen entry is not a member | `test_an_unfrozen_entry_is_not_a_cohort_member` |
| latest-T deferral, typed reason, frozen set kept | `test_t9_latest_t_member_is_deferred_when_the_intersection_misses_the_floors` |
| deferred entry judged next settlement on its frozen set | `test_deferred_entry_is_judged_next_settlement_on_its_frozen_set` |
| repeated deferral, latest T first | `test_deferral_repeats_latest_t_first_until_the_floors_hold` |
| lone member short after the intersection | `test_a_lone_member_short_after_the_intersection_is_deferred_not_judged` |
| max-T against KING_D on the shared set | `test_verdicts_are_cohort_max_t_against_king_d_on_the_shared_set` |
| determinism | `test_cohort_and_deferral_are_deterministic_in_input_order` |
| change 3: tie with the house loses | `test_house_change3_copy_of_the_house_ties_and_loses` |
| change 3: beats king, not house | `test_house_change3_challenger_short_of_d_margin_vs_house_loses` |
| change 3 control: weak house | `test_house_change3_control_beating_king_and_a_weak_house_clears` |
| change 3: beats both | `test_house_change3_challenger_clearing_king_and_house_clears` |
| change 3: house scored on the same set | `test_house_change3_house_is_scored_on_the_cohorts_set` |
| RV5: cohort names the house bar, verdict carries `house` | `test_house_report_names_the_house_the_cohort_was_judged_against` |
| RV5: `house` + `cleared` re-derived from the rows and seed | `test_house_report_verdict_rederives_from_the_reports_rows_alone` |
| RV5: rows = windows x (KING_D + house + members) | `test_house_report_rows_account_for_the_house` |
| RV5: house holds KING_D, no house keys | `test_house_holds_king_d_report_carries_no_house_judgement` |
| RV5: audit PASSes every W5c fixture report | `test_audit_passes_every_report_the_w5c_fixtures_produce` |
| RV5: audit FAILs a doctored house judgement (10 kinds) | `test_audit_fails_a_doctored_house_judgement` |
| RV5: audit FAILs a vs-house loss flipped to a win | `test_audit_fails_a_vs_house_loss_flipped_to_a_win` |
| RV5: audit FAILs house keys on a house-KING_D report | `test_audit_fails_a_house_judgement_where_the_house_holds_king_d` |

The RV5 tests are in `test_w5c_house_report.py`. Its house scenarios are
the `test_w5c_house.py` chain (four: copy of a sharp house, good vs a sharp
house, good vs a bad house, sharp vs a fair house), each built once per
module; every BOUNDARY_1/2/3 report of each is audited.

## On main

- The change 3 tests in `test_w5c_house.py` pass (W5c #579).
- In `test_w5c_house_report.py`, 25 tests fail on an `AssertionError`: main
  writes no `house_digest` / `house`, and `check_lane_d` FAILs every
  miner-KING_D report ("72 rows; the cohorts account for 48"), so each
  tamper test fails its "the honest report must pass first" precondition.
  `test_house_holds_king_d_report_carries_no_house_judgement` passes (that
  shape is today's).
- Every other test passes.
