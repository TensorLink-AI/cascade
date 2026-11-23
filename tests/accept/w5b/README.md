# W5b acceptance tests: union across snapshots, freeze at floors, source cap

W5b (plan rev6 §3.0 W5b row, §1.3 "Evidence accumulates, then freezes" and
"Cohorts", T18, T18e, §17.4 D5 / D29 / B14) turns the per-settlement window
set into a frozen set. An entry's set is the union of the windows W5a's
`eligible_windows` admits over every `pool_d` snapshot since T. It freezes at
the first settlement where it meets the floors, and once frozen it never
changes. **(rev5)** One source may hold at most `[lane_d] max_source_share`
of a frozen set. The excess is dropped deterministically, seeded by the
settlement, and the set then re-checks its floors. These tests are the spec.
The builder may touch only the paths in `PATHS`. W1.5's skeleton and W5a's
tests must stay green.

## Module contract

### `cascade/validator/lane_d_frozen.py` (new)

`cap_source_share(windows, *, max_share: float, seed: str) -> list[dict]`

- `windows` is an iterable of dicts. Only `window_id` and `source` are read,
  so `load_pool` windows and bare `{"window_id", "source"}` dicts both work.
- It returns the kept windows sorted by `window_id`, with no duplicates.
- **The cap.** It drops the fewest windows such that no source holds more
  than `max_share` of the kept set. Each source keeps `min(n_s, Q)` of its
  `n_s` windows, where `Q` is the largest integer `≤ max n_s` with
  `Q ≤ max_share · Σ_s min(n_s, Q)`. A source with `n_s ≤ Q` keeps every
  window. Exactly `max_share` is allowed: the rule bites only on "more
  than". With too few sources (say 3 at 0.25) the result is empty. The tests
  use dyadic shares, so `Q ≤ share · total` compares exactly.
- **Which windows go.** The windows a capped source drops are a seeded
  choice, so the same inputs give the same kept list, whatever the input
  order. A different `seed` gives a different choice. The RNG is free, but it
  must be a deterministic function of `seed` and the window ids.

`advance_frozen_set(progress, pools, *, t_entry, t_king, cfg, seed, exclude=()) -> dict`

- `progress` is `None` for a new entry, or a previous return value. That
  includes one that went through `json.dumps`/`json.loads`, because it lives
  in `lane_d_state.json`. The input is never mutated.
- `pools` is a list of `load_pool` snapshots. Several pools in one call give
  the same pinned keys (below) as the same pools fed one call at a time, in
  any order, while the set stays unfrozen.
- The return value is plain JSON (`allow_nan=False`) and holds at least:
  - `frozen: bool`;
  - `window_ids`: a sorted list;
  - `set_sha256`: `sha256("\n".join(window_ids))` (the repo's
    `cohorts[].set_sha256` convention) when frozen, `None` otherwise.
  Other keys, such as each id's source, are free.
- **Union.** Each call adds the windows of every pool that
  `eligible_windows(pool, t_entry=…, t_king=…, cfg=…)` marks eligible. Ids in
  `exclude` are skipped; that is how the confirmation set stays disjoint.
  Windows are deduplicated by `window_id`, so the same window in two
  snapshots, or a snapshot read twice, counts once.
- **Freeze.** After the union, the cap (`cfg.max_source_share`, `seed`) runs
  on it. If the capped set meets `cfg.min_windows` and `cfg.min_clusters`
  (distinct sources), the result is frozen: `window_ids` is the capped set.
- **Decision: floors re-checked after the drop.** If the capped set misses a
  floor, the result is not frozen, and `window_ids` keeps the **uncapped**
  union. The entry keeps accumulating, and the cap runs again at the next
  check, with that settlement's seed. So windows dropped at a failed check
  are not lost.
- **Frozen is final.** Given a frozen `progress`, it returns a value equal to
  `progress`, whatever the pools or seed.

### `cascade/validator/lane_d.py`

- `settle_lane_d` gains the keyword `prior_pool_dirs` (a sequence of paths,
  default empty). These are the earlier retained `pool_d` snapshots. They
  feed unions and supply the history and targets of windows frozen from
  them. `pool_dir` stays the newest snapshot. Duplicates in the list are
  harmless.
- Each open entry's progress lives in `state["entries"][digest]["frozen_set"]`.
  While the entry is unfrozen this is the `advance_frozen_set` record above,
  and the entry stays `pending` / `pending_floor` with no rows.
- An entry is judged only once its set is frozen. A cohort is judged on the
  intersection of its members' frozen sets that also has forecasts, a subset
  of each frozen set (§1.3 Cohorts). The existing latest-T deferral keeps
  working.
- The cap's seed is derived from the settlement (its boundary). The same
  settlement gives the same drop, and another settlement gives another.
- After a first clear, the confirmation accumulates a fresh set that
  excludes `used_windows` (D20 disjointness, unchanged). Its
  `confirm.first_set_sha256` equals the judged set's `set_sha256`.
- **T18 carry.** Nothing resets at an era boundary. An entry readied in an
  era's last settlement keeps its union into the next era.
- **T18 `until_block`.** At the settlement whose boundary is `until_block`
  (the last active one, `cfg.active(until_block)` is true), nothing is
  judged: `cohorts` is `[]` and no entry has rows. Every open entry
  (`pending` / `confirm_pending`, including ones registered at that
  settlement) closes with status `closed`, reason `lane_closed`, in state and
  in the report's `entries`. None is crowned or put in the court. After
  `until_block` the lane is inactive: `report` and `pay` are `None`, and the
  state file is untouched (existing behaviour).

### (RV4) One version per window; the report names every judged snapshot

A window id is stable across snapshots, but its `first_seen_ts`, values,
`frontier` and its feed's measured `mirror_lag_hours` are not. RV4 found
main admitting a window if *any* snapshot admits it and then scoring the
*newest* snapshot's version, which §1.3 may refuse.

- **Decision: the earliest admitting snapshot's version.** A judged window
  is scored, and its `windows[]` row (`min_target_ts`, `min_first_seen_ts`,
  `mirror_lag_hours`, `source`, `feed`) taken, from one snapshot: the
  earliest loaded snapshot (oldest first, `pool_dir` last) whose
  `eligible_windows` admits it for the cohort's max(T). Admission from one
  snapshot is never combined with clocks, history, targets or lag from
  another.
  - Why earliest: §1.3 grows the union snapshot by snapshot, so a window
    enters the entry's evidence at its first admitting snapshot, and that is
    the version the set accumulated. Scoring it means a later snapshot can
    never change what an already-collected window is scored on. A later
    revision of values, clocks or lag (newer `pool_d` bytes) can't move a
    verdict between freeze and judgment. Invariant 4 needs the scored version
    to pass §1.3. The admitting version does, by construction, so a §1.5
    audit of the report's declared timestamps passes.
  - A window that no loaded snapshot admits is never judged. It isn't in
    `windows`, `rows`, `cohorts[].window_ids` or the entry's
    `frozen_set.window_ids`.
  - No judged window has `min_first_seen_ts < bound`, `min_target_ts <
    bound`, or `mirror_lag_hours > max_mirror_lag_hours`.
  - Not pinned: a window refused by an older snapshot but admitted by a
    newer one. The rule above scores the newer version, and refusing such a
    window is also allowed.
- **The report names every judged snapshot.** Two new fields, so the
  `lane-d-report/1` schema (W7a) has to add them. The pending W7a RV4
  correction doesn't define them yet, so **the W7a schema owner needs to
  add**:
  - `pool.snapshots`: a list of `{snapshot, sha256, first_seen_roots}`.
    It has one entry per snapshot that supplied the version of at least one
    judged window. It may also list other loaded snapshots, but only loaded
    ones, each once. `pool.snapshot` / `pool.sha256` / `pool.horizon` stay
    as they are (the newest snapshot; the seed uses it).
  - `windows[].snapshot_sha256`: the `sha256` of the snapshot whose
    version was scored. It is always one of `pool.snapshots[].sha256`.
  - The two root lists agree. `pool.first_seen_roots` is the sorted union
    of `pool.snapshots[].first_seen_roots`, and every
    `cohorts[].first_seen_roots` equals it.

  Until the schema has them, the tests assert on the settlement result
  (`settle_lane_d(...).report`), not on `validate_report`.

### `cascade/validator/lane_d_state.py`

- `ENTRY_REASONS` gains `"lane_closed"` (invariant 11: a typed reason).

## Tests

| Row / rule | Test |
|---|---|
| T18e minimum-drop cap, only the dominant source loses windows | `test_t18e_excess_of_a_dominant_source_is_dropped` (40 + 4×10 at 0.25 ⇒ 13 + 4×10) |
| cap is a no-op within the share (including exactly 0.25) | `test_t18e_a_set_within_the_share_is_unchanged` |
| deterministic drop: same inputs, any order ⇒ same drop | `test_t18e_same_inputs_give_the_same_drop` |
| seeded drop: another seed ⇒ another drop | `test_t18e_a_different_seed_gives_a_different_drop` |
| cap matches the oracle (60 random cases, shares 1/8–1/2) | `test_t18e_cap_matches_the_minimum_drop_oracle` |
| union across snapshots, freeze at floors, set digest | `test_union_accumulates_across_snapshots_until_the_floors` |
| dedup by window id | `test_union_dedups_by_window_id_across_snapshots` |
| only §1.3-eligible windows join; max(T_E, T_K) | `test_union_counts_only_forward_eligible_windows` |
| frozen set never changes; input not mutated | `test_frozen_set_never_changes_after_freeze` |
| confirmation disjointness (`exclude`) | `test_union_skips_excluded_windows` |
| persistence: JSON round trip continues identically | `test_union_progress_round_trips_through_json` |
| T18e floors re-checked after the drop | `test_t18e_floors_are_rechecked_after_the_cap` |
| config reads (`min_windows`, `min_clusters`, `max_source_share`) | `test_union_reads_the_floors_from_config` |
| W1.5 skeleton + union end to end | `test_e2e_union_freezes_across_two_settlements` |
| T18 carry across eras | `test_t18_entry_readied_in_an_eras_last_settlement_carries_across_eras` |
| T18 `until_block` | `test_t18_until_block_closes_pending_entries_unjudged_and_unpaid` |
| T18e through the settlement (seeded by the settlement) | `test_t18e_settlement_caps_a_dominant_source_deterministically` |
| invariant 5 flavour: replay from the same state | `test_inv5_replaying_a_settlement_from_the_same_state_is_identical` |
| invariant 10 pair | `test_union_unchanged_when_unarmed` + `test_union_armed_nonempty_lane_d_effect` |
| (RV4) invariant 4: a two-version window is scored from its earliest admitting snapshot | `test_a_two_version_window_is_scored_from_its_earliest_admitting_snapshot` |
| (RV4) refused in every snapshot ⇒ never judged | `test_a_window_refused_in_every_snapshot_is_never_judged` |
| (RV4) report lists both snapshots of a union; root lists agree | `test_report_names_every_snapshot_of_a_two_snapshot_union` |
| (RV4) report names the admitting snapshot of a two-version window | `test_report_names_the_admitting_snapshot_of_a_two_version_window` |

The fixtures use numpy quantile forecasts, as W5a's do: no torch, no
checkpoint, and no signature verification. The floors are lowered to 24
windows / 4 sources. The recommended 1000 / 20 (D5) are W2 defaults; the
code reads them from config.

## On main

- Every test except the `until_block` test and the fixture sanity check
  imports `cascade.validator.lane_d_frozen` first. It fails with
  `ModuleNotFoundError`, a path in PATHS.
- The `until_block` test fails on an assertion: main judges the entry at the
  closing settlement.
- `test_union_window_record_sanity` passes. It guards the fixture only.
- (RV4, `test_w5b_versions.py`, after W5b was built) Three tests fail on an
  assertion against the built W5b. The two-version scoring test sees
  `min_first_seen_ts` 1790018000, which is below the bound 1790021600, so
  main scores the newest version. The two report tests find no
  `pool.snapshots`. The "refused in every snapshot" test passes. It keeps a
  fix honest.

## Correction (plan rev6.10 (a) / rev6.12 (a)): fixture feeds pin `embargo_hours`

W5a (tests in #758) makes forward eligibility embargo-aware and fails a
listed feed whose `embargo_hours` is missing closed as `unlisted_feed`. This
WP's pool fixture wrote feeds without the key, so once the W5a
implementation lands every fixture window would be refused. Every fixture
feed now pins `embargo_hours: 0.0` (e = 0, the rev3 rule byte for byte, so
every verdict is unchanged); on main the key is ignored. No assertion
changed; every test passes on main and with the W5a implementation.
