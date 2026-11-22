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
- The verdict's `king_digest` stays KING_D. These tests don't pin how the
  house comparison is reported beyond the outcome and the house rows.

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

## On main

- The three change 3 tests (tie, short vs house, house rows) fail on an
  `AssertionError`. Main compares challengers with KING_D only, so Y goes
  `confirm_pending`, and the house has no rows.
- Every other test passes. W5b already implements those rules, and W5c now
  owns them.
