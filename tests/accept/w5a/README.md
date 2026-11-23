# W5a acceptance tests: forward eligibility (embargo-aware since rev6.10)

W5a (plan rev6 §3.0 W5a row, §1.3, Proposed design changes 1 and 2, which
the owner ACCEPTED in rev5, T8, §3.2 invariant 4) replaces W1.5's thin
two-clock check with the full §1.3 rule. It adds typed refusal reasons and
puts the rule on `settle_lane_d`'s code path. These tests are the spec. The
builder may touch only the paths in `PATHS`. W1.5's skeleton must stay
green: `make laned-e2e` keeps passing.

## The rule

The bound for an entry E and the king K it is judged against is

`bound = max(T_E, T_K) + 3600 · cfg.forward_margin_hours`

Both T values are unix seconds (`reveal_ts`; the house's `reveal_ts` for the
house). A window is eligible only if all of these hold:

1. **Clocks recorded.** `target_ts` and `first_seen_ts` each have
   `cfg.horizon` points, and `end_ts` is set and equals `target_ts[-1]`.
1a. **Feed listed (RV3 BLOCKING 1).** The window's feed (its `feed`, else
   its `source`) has an entry in the pool's top-level `feeds` block. An
   unlisted feed, or a pool with no `feeds` block, is `unlisted_feed`: without
   the entry neither rule 2 nor rule 3 can be judged, so the window fails
   closed instead of silently switching both off.
2. **Not a forecast product.** The window's feed is not marked
   `forecast_product`.
3. **Mirror lag within the limit.** The feed's measured
   `mirror_lag_hours ≤ cfg.max_mirror_lag_hours`. A feed the snapshot LISTS
   with `mirror_lag_hours: null` is declared unmeasured and is not a refusal
   (W6fs measures lag only "where the feed provides" a publish stamp; RV3
   note 7: this is a policy choice for the decision register).
4. **Data time after the bound.** `min(target_ts) > bound`, strictly, so
   every target point counts.
5. **First-seen after the bound.** `min(first_seen_ts) > bound`, strictly.
6. **Present in S0 (change 1a).** S0 is the earliest `frontier` record whose
   `as_of_ts > bound`, that is, the first snapshot after the bound. If no
   such record exists, the series is `absent_from_s0`. A series absent from
   S0 starts its clock at first appearance: once a later snapshot holds it,
   that snapshot is its S0.
7. **After S0's frontier (change 1c).** `min(target_ts) > end_ts(S0)`.
   Otherwise the window is a `backfill`.

**Fail closed on a missing frontier (RV3 BLOCKING 1).** Rules 6 and 7 apply
to every series. A series whose `frontier` is absent or `null` has no
evidence of being in S0 (change 1(a): "present in S0 with a recorded
`end_ts`"), so it is `absent_from_s0`, exactly like an empty list. There is
no keyless two-clock path. The W1.5 fixture carries `frontier` and `feeds`
since #560, so the skeleton stays green.

## Reasons

`REFUSAL_REASONS` is exactly this tuple, in precedence order. When several
rules fail, the reason reported is the first failing one in this order:

| reason | fails |
|---|---|
| `missing_clock` | rule 1 |
| `unlisted_feed` | rule 1a (feed not in `feeds`, or no `feeds` block) |
| `forecast_product` | rule 2 |
| `over_lag` | rule 3 |
| `before_margin` | rules 4 and 5 both |
| `publish_ahead` | rule 5 only: future-stamped values that the mirror saw before the bound |
| `late_mirror` | rule 4 only: values public before T that reached the mirror after the bound |
| `absent_from_s0` | rule 6, including a `frontier` that is absent or `null` |
| `backfill` | rule 7 |

**Note on T8's wording.** T8's parentheticals swap the clocks: "data-time-only
pass but first-seen fail (late mirror)". The tests pin §1.3's physics instead:
"data time alone admits … publish-ahead feeds … first-seen alone admits values
… [that] reached our mirror late". The case names are T8's.

## Module contract

### `cascade/validator/lane_d_forward.py` (new)

- `REFUSAL_REASONS`: the tuple above. Members may be a `str` enum; the tests
  compare them with `str(...)` and `==`.
- `eligible_windows(pool, *, t_entry, t_king, cfg) -> iterable`:
  - `pool` is what `cascade.validator.lane_d.load_pool(pool_dir, horizon)`
    returns;
  - it yields one verdict per pool window, in pool order, each with
    `.window_id`, `.eligible: bool` and `.reason` (`None` iff eligible,
    otherwise one of `REFUSAL_REASONS`);
  - it reads `cfg.forward_margin_hours`, `cfg.max_mirror_lag_hours` and
    `cfg.horizon`;
  - swapping `t_entry` and `t_king` gives identical verdicts, because only
    their max counts.

### `cascade/validator/lane_d.py`

- `load_pool` parses the pool_d/1 additions below. How it represents them
  internally is free.
- `settle_lane_d` builds every cohort's window set from `eligible_windows`
  verdicts. The module calls `eligible_windows`; the AST test checks this.
- **Invariant 4:** no report row of a member is on a window that fails the
  rule for max(T_member, T_king).
- **The v2 snapshot-only rule is refused.** Neither module references
  `laned_w1_measure`, `snapshot_diff`, `BOUND_LABEL`, `APPEARED` or
  `CARRIED`. That rule survives only as W1's report-only upper bound.

### pool_d/1 additions (`metadata.json`)

`load_pool` still parses a snapshot without them, but since RV3 their
absence fails closed: a window is eligible only with both.

- Top level: `"feeds": {"<feed>": {"mirror_lag_hours": float | null,
  "forecast_product": bool}}`, one entry for every feed the series name.
- Per series:
  - `"feed"`: defaults to the series' `source`;
  - `"frontier"`: `[[as_of_ts, end_ts], ...]`, one pair for each EARLIER
    pool_d snapshot that held the series (possibly empty). `as_of_ts` is that
    snapshot's build (wall) time and `end_ts` is the series' `end_ts` in it.

## Tests

| Row / invariant | Test |
|---|---|
| T8 late mirror, publish-ahead, over-lag, forecast product, backfill | `test_t8_*` in `test_w5a_eligibility.py`. Each fixture fails exactly one rule, and the timelines stay physically consistent (a frontier's `end_ts` never exceeds its `as_of_ts`) |
| T8 entry and king T in both orders | `test_t8_entry_and_king_t_in_both_orders_uses_the_max[entry_later / king_later]` |
| change 1 (S0) | `test_t8_series_absent_from_s0_is_refused`, `test_absent_series_starts_its_clock_at_first_appearance`, `test_t8_backfilled_partition_is_never_eligible` |
| strictness, every point, config reads, feeds, missing clocks, precedence | the remaining tests in `test_w5a_eligibility.py` |
| v2 rule refused | `test_v2_snapshot_rule_is_not_on_the_lane_d_code_path` (AST) and `test_v2_snapshot_rule_admits_what_w5a_refuses`. The second runs the real `scripts/laned_w1_measure.py` helpers to show that the v2 rule admits a newcomer series with pre-T data; W5a refuses it as `late_mirror` |
| invariant 4 (property) | `test_inv4_settle_never_scores_an_ineligible_window`. It uses 8 seeded random fixtures: random `T_K` on either side of `T_E`, one or two entries, margin 0/6/12 h, lag limit 3/6/9 h, a random lag for the over-lag feed, and every kind mixed in at shifted bounds. Each member row is checked against an oracle in `helpers.py`, which re-derives the rule independently. `eligible_windows` must match the oracle exactly, and every trap reason must occur |
| invariant 10 pair | `test_inv4_window_set_unchanged_when_unarmed` (passes on main) + `test_inv4_window_set_armed_nonempty_lane_d_effect` (exactly the 48 eligible windows are scored and no trap) |
| W1.5 skeleton +1 assertion | `test_e2e_skeleton_refuses_trap_windows_and_still_pays_the_challenger`. Two settlements run on pools carrying traps of every kind, on which the house is exact and the challenger 1000 sd off. The traps are refused, so the challenger clears, confirms, is crowned and gets lane D weight (`lane_d_vector`) |

## On main

- The unit tests fail with `ModuleNotFoundError:
  cascade.validator.lane_d_forward`, which is in PATHS.
- The armed settlement tests fail on an assertion: main scores the
  clock-passing traps (`absent_from_s0`, `backfill`, `over_lag`,
  `forecast_product`).
- The unarmed test passes.

### RV3 correction (tests-only PR, w0c tests-correction mode)

These tests are new or changed, and they fail on main with an
`AssertionError` until the W5a impl fix lands:

- `test_refusal_reasons_are_the_typed_enum_in_precedence_order`:
  `unlisted_feed` is now in the enum;
- `test_one_verdict_per_window_in_pool_order`: a keyless series is refused;
- `test_unlisted_feed_is_refused`, `test_pool_without_feeds_block_is_refused`,
  `test_series_without_frontier_key_is_refused` (this replaces the old
  keyless-eligible pin) and `test_series_with_null_frontier_is_refused`;
- `test_reason_precedence`: an unlisted feed beats the clock rules;
- the three armed settle tests: the oracle fails closed, and the traps now
  include `no_frontier`, `null_frontier` and `unlisted_feed`.

`test_unmeasured_lag_is_not_a_refusal` is renamed to
`test_listed_feed_with_null_lag_is_not_a_refusal`, and passes. Every other
test is unchanged and passes.

Locally (bittensor 10.5, no torch) the W5a tests need neither torch nor
`bittensor.sp_core`, because they never verify a signature.

### RV5 correction: the crowning fixture passes the gate inputs (W5e fail-closed)

Once W5e lands (`tests/accept/w5e/README.md` "Fail closed"), an armed
settlement that would crown without a `bench` session and a
`boundary_block_hash` is `bench_hold`, not a crown.
`test_e2e_skeleton_refuses_trap_windows_and_still_pays_the_challenger`
crowns at BOUNDARY_2 through `_settle`, so `_settle` now passes
`boundary_block_hash` (`helpers.gate_block_hash`), `sentinel=None` and a
healthy `bench` (`lane_d_bench.SweepSession` over `helpers.HealthyBench`)
whenever `settle_lane_d` accepts those keywords (`inspect.signature`), and
makes the legacy call otherwise. The tests pass both on today's main and
after W5e. The bench is the one external service: every digest but the
house returns the same recorded rows and the house is 10% worse, so the
crowning candidate passes on any drawn suite and the duel alone decides.
No assertion changed; every test still passes on main.

## Embargo (rev6.10 (a), rev6.12 (a)/(b); tests-correction)

The W5a row (rev6.10, rev6.11, rev6.12) makes the rule embargo-aware. Let
`b = max(T_E, T_K) + 3600 · forward_margin_hours` (the bound above) and, per
series, **for every pool `freq`** (rev6.12 (a) reverses rev6.11's sub-hourly
zero):

    e = min(feeds[feed].embargo_hours, cfg.embargo_hours)

- **Fail closed:** a listed feed whose `embargo_hours` is missing, `null`, a
  bool, a string or negative makes the window `unlisted_feed` (same slot in
  the precedence order: after `missing_clock`). `0` is valid.
- **The rule:** rules 4 and 5 compare against the **feed bound** `b − e`:
  every target point needs `data_ts > b − e` and `first_seen > b − e`
  (`first_seen + e > b`), strictly; the labels keep their meaning
  (`before_margin`, `publish_ahead`, `late_mirror`). S0 is the earliest
  frontier record with `as_of_ts > b − e`; backfill is judged against it.
- **e = 0 is byte-identical to the rev3 rule** (the test diffs the
  producer's serialised verdicts against `helpers.oracle_reason`, with the
  feeds at 0 and with `cfg.embargo_hours` at 0).
- **Mixed targets:** a window may straddle `b`; it passes when every point
  passes the rule above.
- **Verdict fields:** an eligible verdict carries `embargo_hours` (the
  effective e, a float) and `embargo_window` (rev6.12 terms: `False` =
  **fresh**, the window also passes at e = 0 on every point; `True` =
  **embargo** window, a mixed one included). W5j's fresh-first draw and
  `min_fresh_window_share` read the flag; W7a reports it. Refused verdicts
  may carry anything there.
- `cfg.embargo_hours` is W2e's key (default 48). The tests never read it:
  `helpers.with_embargo` hands the producer a config that has it, a
  `dataclasses.replace` once W2e lands.
- The per-feed value is pinned in the snapshot (`feeds[feed].embargo_hours`,
  W6b); `load_pool` must carry it through to `eligible_windows` unchanged
  (a non-numeric value is the window's refusal, never a pool error).

Fixtures: every listed feed now pins `embargo_hours` (`helpers.feeds_block`,
default 0), so every pre-embargo test keeps its verdicts.

| Rule | Test (`test_w5a_embargo.py` unless noted) |
|---|---|
| inside the embargo eligible at 48, rev3 refusal at 0 | `test_window_inside_the_embargo_is_eligible_at_e48`, `test_same_window_at_e0_is_the_rev3_refusal` |
| every cadence gets it | `test_every_cadence_gets_the_embargo[...]` |
| mixed target | `test_a_target_straddling_b_is_eligible_at_e48_only` |
| catalog only shortens; cfg read | `test_catalog_value_can_only_shorten` |
| strict feed bound, every point | `test_feed_bound_is_strict_on_both_clocks`, `test_every_target_point_counts_against_the_feed_bound` |
| S0 / backfill at `b − e` | `test_s0_and_backfill_use_the_feed_bound` |
| bad feed value fails closed | `test_bad_feed_embargo_fails_closed_as_unlisted_feed`, `test_bad_feed_embargo_ranks_after_missing_clock`, `test_zero_is_a_valid_feed_embargo` |
| verdict fields | `test_verdict_carries_effective_embargo_and_embargo_window_flag` |
| e = 0 byte-identical | `test_e0_is_byte_identical_to_the_rev3_verdicts[feed_zero, config_zero]` |
| property vs the oracle | `test_embargo_rule_matches_the_oracle_over_random_fixtures` |
| through `settle_lane_d` | `test_w5a_embargo_settle.py::test_settle_judges_embargo_windows_of_every_cadence` |

### On main (tests-correction)

The new embargo tests fail on an `AssertionError` (main refuses embargo
windows, admits invalid feed values, and its verdicts have no
`embargo_hours` / `embargo_window`); the byte-identity tests pass. The
other files change only through `helpers.py` (feeds pin
`embargo_hours: 0`) and the custom feed dicts in
`test_w5a_eligibility.py`, and still pass. `test_w5a_settle.py`'s e2e test
no longer requires rows at BOUNDARY_2: rev6.11 (b) crowns at the first
verdict, so BOUNDARY_2 may judge nothing. A scratch prototype passed every
W5a test.

**Blast radius (flag for the coordinator):** the fail-closed rule refuses
every window of a feed without `embargo_hours`. The pool fixtures of W1.5,
W3b3, W3b5, W5b, W5c, W5f, W5g, W7b, W7c, W7d and W9 write `feeds` without
it, so they need the same one-line fixture fix before the W5a
implementation can pass CROSS. W5e's and W7a's are fixed in this PR.
