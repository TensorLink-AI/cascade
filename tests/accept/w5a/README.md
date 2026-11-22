# W5a acceptance tests: forward eligibility

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
2. **Not a forecast product.** The window's feed is not marked
   `forecast_product`.
3. **Mirror lag within the limit.** The feed's measured
   `mirror_lag_hours ≤ cfg.max_mirror_lag_hours`. An unmeasured lag (the feed
   isn't listed, or its lag is `null`) is not a refusal. W6fs measures lag
   only "where the feed provides" a publish stamp.
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

Rules 6 and 7 apply only to a series whose record has a `frontier` key.
Without the key (pool_d/1 snapshots built before W6b, which includes the W1.5
fixture), the series is judged on rules 1–5. With an empty list, it is
`absent_from_s0`.

## Reasons

`REFUSAL_REASONS` is exactly this tuple, in precedence order. When several
rules fail, the reason reported is the first failing one in this order:

| reason | fails |
|---|---|
| `missing_clock` | rule 1 |
| `forecast_product` | rule 2 |
| `over_lag` | rule 3 |
| `before_margin` | rules 4 and 5 both |
| `publish_ahead` | rule 5 only: future-stamped values that the mirror saw before the bound |
| `late_mirror` | rule 4 only: values public before T that reached the mirror after the bound |
| `absent_from_s0` | rule 6 |
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

These are optional, so W1.5's fixture format stays valid.

- Top level: `"feeds": {"<feed>": {"mirror_lag_hours": float | null,
  "forecast_product": bool}}`.
- Per series:
  - `"feed"`: defaults to the series' `source`;
  - `"frontier"`: `[[as_of_ts, end_ts], ...]`, one pair for each EARLIER
    pool_d snapshot that held the series. `as_of_ts` is that snapshot's build
    (wall) time and `end_ts` is the series' `end_ts` in it.

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

Locally (bittensor 10.5, no torch) the W5a tests need neither torch nor
`bittensor.sp_core`, because they never verify a signature.
