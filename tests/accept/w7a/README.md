# W7a acceptance tests: the `lane-d-report/1` schema

W7a (plan rev6 §3.0 W7a row) turns W1.5's thin lane D report into a
versioned, closed schema. It covers §1.5 "(rev3) The report also carries",
the §14.3 forward fields, B7 (`scoring_method`), B12 (first-seen Merkle roots)
and §3.2 invariant 11. These tests are the spec. The builder may touch only
the paths in `PATHS`. W1.5's skeleton must stay green (`make laned-e2e`).

The repo has no JSON-schema library, and adding one would be a dependency
change. So the contract is a **schema file** (the published document
satellites read) plus a **stdlib validator** in `lane_d_report.py` that
enforces it. The tests pin both and tie them together.

## Decisions pinned here

- **Every report object is closed.** An unknown field anywhere is refused
  (`unknown_field`), so a satellite can never silently ignore a consensus
  field it doesn't know. The one exception is the optional top-level
  `extensions` object. It is open, has no `properties` in the schema, and is
  **never consensus**: satellites must not read verdict, eligibility or
  weight inputs from it. Anything consensus needs a schema bump (§14.3 item
  3: "a schema bump is a known, rare release").
- **An unknown or absent `schema` is refused first** (`unknown_schema`),
  before any other check.
- **`scoring_method` must be exactly `"quantile_head"`.** Absent →
  `missing_field`; any other value → `bad_value` (B7).
- **Typed reasons are never null.** Every entry carries `reason` from
  `ENTRY_REASONS`, including `pending` and `hold` (invariant 11). A court slot
  carries `burn_reason`, which is `null` iff `uid` is an int.
- **Pools without the new metadata stay valid.** A pool_d/1 snapshot without
  `feeds`, per-series `feed` or `first_seen_roots` (the W1.5 format) gives
  `feed = source`, `mirror_lag_hours = null` and `first_seen_roots = []`.
  W6fs makes the roots mandatory and refuses a snapshot whose records
  postdate its root. That isn't W7a's job.

## Module contract: `cascade/validator/lane_d_report.py`

- `REPORT_SCHEMA == "lane-d-report/1"` (unchanged).
- `SCHEMA_PATH`: the path of `cascade/validator/schemas/lane_d_report_1.json`.
  `load_schema() -> dict` returns the parsed file.
- `validate_report(report) -> None` raises `ReportSchemaError` on the first
  defect. It checks structure only. The signature is `verify_report`'s job
  (W7b).
- `ReportSchemaError(ValueError)` has `.code`, one of `SCHEMA_ERROR_CODES`,
  and `.field`, the path of the offending field:
  - dotted keys with `[i]` list indices, for example
    `windows[0].mirror_lag_hours`, `pay.court[0].burn_reason` or
    `cohorts[0].refused.before_margin`;
  - for a missing field, the path the field should have had;
  - for `not_object`, any value.
- `SCHEMA_ERROR_CODES` holds exactly these codes:

  | code | when |
  |---|---|
  | `not_object` | the report isn't a JSON object |
  | `unknown_schema` | `schema` is absent or isn't `"lane-d-report/1"` (checked first) |
  | `missing_field` | a required field is absent |
  | `unknown_field` | a field the schema doesn't define (outside `extensions`) |
  | `bad_type` | wrong JSON type, `null` where not allowed, or a bool where an int or number is required |
  | `bad_value` | right type, but outside the const, enum, format or range. Covers `scoring_method`, enums, 64-hex digests and roots, non-finite numbers, and a court slot whose `uid`/`burn_reason` pair is inconsistent |
  | `rows_sha256_mismatch` | `rows_sha256` isn't the canonical digest of `rows`. Checked only after every structural check passes |

- The enum tuples below are exported under these names and equal the schema
  file's `$defs.<name>.enum`. Equality is compared as sets.

## The schema file: `cascade/validator/schemas/lane_d_report_1.json`

- JSON Schema draft 2020-12: `"$schema":
  "https://json-schema.org/draft/2020-12/schema"`, `"type": "object"`.
- `properties.schema.const == "lane-d-report/1"` and
  `properties.scoring_method.const == "quantile_head"`.
- Top-level `required` is exactly the list below, and `properties` is that
  list plus `extensions`, with `additionalProperties: false`.
- Every object schema that has `properties` also has
  `additionalProperties: false`. At least the ten nested report objects are
  described.
- `$defs` holds the enums: `entry_status`, `entry_reason`, `skip_reason`,
  `burn_reason`, `window_refusal_reason`, `confirmation_stage`,
  `confirm_rule`, `lowfreq_result` and `verdict_outcome`.

## Fields

Top level (all required): `schema`, `boundary`, `signer`, `signature`,
`scoring_method`, `params`, `seed`, `pool`, `house`, `king_before`,
`king_after`, `crowned`, `prior_kings`, `entries`, `judged`, `skipped`,
`cohorts`, `windows`, `verdicts`, `rows`, `rows_sha256`, `pay`. Optional:
`extensions`.

| object | required fields (the tests check these; more are the builder's choice, but they must be in the schema) |
|---|---|
| `params` | `share`, `takes_from`, `remainder`, `horizon`, `horizons` (§14.3 horizon list, `[horizon]` today), `contexts`, `forward_margin_hours`, `max_mirror_lag_hours`, `min_windows`, `min_clusters`, `d_margin`, `alpha`, `bootstrap_B`, `king_decay`, `reward_prior_kings`, `confirm_on_king_change`, `max_entries_per_settlement`, `max_judged_per_coldkey_per_settlement`, `scoring_levels`, `gate_suites` (§14.3 suite list), `tau_public`, `lowfreq_freqs`. Each equals the parsed `[lane_d]` value (lists as JSON lists) |
| `pool` | `snapshot`, `sha256`, `horizon`, `first_seen_roots`: the B12 daily Merkle roots the snapshot relies on, sorted, from pool metadata `first_seen_roots` |
| `entries[]` | `digest`, `hotkey`, `coldkey` (A8: as recorded at `ready`, `null` if unresolved), `context` (§14.3 per-entry context), `ready_block`, `reveal_block`, `reveal_ts`, `status`, `reason` |
| `judged` | the digests judged at this settlement, equal to the verdicts' digests |
| `skipped[]` | `digest`, `reason` ∈ `SKIP_REASONS`. One per entry considered this settlement and not judged, **including entries voided this settlement** (`deregistered`, `coldkey_unresolved`, `ready_expired`, …). Disjoint from `judged` |
| `cohorts[]` | `index`, `king_digest`, `king_reveal_ts`, `stage`, `members`, `t_max`, `window_ids`, `set_sha256`, `first_seen_roots` (the roots the frozen set relies on: the snapshot's roots today, a union once W5b unions snapshots), `refused`: `{reason: count}` of the pool windows refused by the §1.3 eligibility rule, keys from `WINDOW_REFUSAL_REASONS`, counts ≥ 1. Windows dropped for other reasons (already used, no forecast) aren't counted |
| `windows[]` | `window_id`, `source`, `feed` (series `feed`, default `source`), `min_target_ts` and `min_first_seen_ts` (both clocks, A1), `mirror_lag_hours` (the feed's `feeds[feed].mirror_lag_hours`, `null` when unmeasured or unlisted) |
| `verdicts[]` | `digest`, `king_digest`, `stage`, `cohort`, `lcb`, `gain`, `cleared`, `outcome`, `confirmation`, `lowfreq` |
| `verdicts[].confirmation` | `stage` (`first`/`confirm`), `rule` (= `confirm_on_king_change`), `first_set_sha256` and `first_king_digest`: `null` on stage `first`; on `confirm`, the first clear's `set_sha256` and king (A2, D20) |
| `verdicts[].lowfreq` | `result`, `freqs` (= `lowfreq_freqs`), `n_windows`, `tau_public` (A10). The candidate is a verdict that cleared at stage `confirm`: `pass` or `fail` (`fail` means the entry isn't crowned, `lowfreq_regression`), or `no_windows` when the set holds no window at those frequencies. Every other verdict is `not_candidate` with `n_windows = 0`. The test fixture is hourly, so the crowned candidate is `no_windows`. Computing pass and fail is W3b5's job |
| `rows[]` | `window_id`, `digest`, `cohort`, `crps`, `mase`, all finite |
| `pay` | `share`, `court[]`: `digest`, `hotkey`, `uid`, `burn_reason` |

`rows_sha256` is sha256 over `json.dumps(rows, sort_keys=True,
separators=(",", ":"), allow_nan=False)` (A9). The tests re-derive it
independently.

## Enums (exact sets)

| name | values |
|---|---|
| `ENTRY_STATUSES` | `pending`, `hold`, `confirm_pending`, `crowned`, `lost`, `closed` |
| `SKIP_REASONS` | `pending_floor`, `coldkey_cap_settlement`, `coldkey_unresolved`, `coldkey_changed`, `king_unscored`, `bench_hold`, `deregistered`, `ready_expired`, `reveal_delay_exceeded`, `duplicate_digest`, `hotkey_has_open_entry`, `context_not_armed`, `checkpoint_fault`, `mirror_lag_exceeded` |
| `ENTRY_REASONS` | `SKIP_REASONS` plus `confirm_pending`, `lost_duel`, `lost_confirmation`, `lowfreq_regression`, `crowned` |
| `BURN_REASONS` | `house_king` (D21: the house holds KING_D), `deregistered` (B8: a court member lost its UID) |
| `WINDOW_REFUSAL_REASONS` | W5a's `REFUSAL_REASONS`: `missing_clock`, `forecast_product`, `over_lag`, `before_margin`, `publish_ahead`, `late_mirror`, `absent_from_s0`, `backfill` |
| `CONFIRMATION_STAGES` | `first`, `confirm` |
| `CONFIRM_RULES` | `restart`, `original` |
| `LOWFREQ_RESULTS` | `pass`, `fail`, `no_windows`, `not_candidate` |
| `VERDICT_OUTCOMES` | `lost`, `confirm_pending`, `confirmed`, `crowned` |

The §1.5 state list, W1.5's reasons and the W5h reasons named in the plan
are all present. Which reason goes with which status beyond the cases
below belongs to W5g and W5h.

## Tests

| Row / invariant | Test |
|---|---|
| versioned schema file, closed, enums | `test_w7a_schema_file.py` |
| unknown version refused, `scoring_method`, unknown fields, types, enums, rows digest | `test_w7a_validate.py`. Each case changes one thing in a real `settle_lane_d` report, and row mutations are re-sealed |
| a W1.5 `settle_lane_d` report validates (synthetic kind, no torch) | `test_e2e_settle_reports_validate_and_state_quantile_head`, `test_e2e_legacy_pool_format_report_validates` |
| §14.3 forward fields, both clocks, mirror lag, roots, judged set, coldkey, confirmation, low-frequency, burn | the other `test_e2e_*` tests in `test_w7a_report_fields.py`. Two settlements: the first clear (house holds the slot, `house_king`), then the confirmation that crowns. One `before_margin` trap window per pool |
| invariant 11 (exact reasons) | `test_inv11_*` in `test_w7a_inv11_reasons.py`. Covers the coldkey cap (`coldkey_cap_settlement`, the entry stays `pending` with that reason), `deregistered`, `coldkey_unresolved` (`coldkey: null`), `ready_expired`, a floor miss (`pending_floor`, no rows, `rows_sha256` of `[]`), a house-held slot (`house_king`) and a crowned king that deregisters (`deregistered`) |

## On main

- The schema-file tests fail on `assert` (there is no file).
- Everything else fails at `ImportError: cannot import name …` from
  `cascade.validator.lane_d_report`, which is in PATHS.
- `test_inv11_fixture_blocks_are_consistent` passes. It checks the fixture.

The synthetic chain was probed on main: its settlements judge, confirm and
crown as the tests expect, and today's report lacks exactly the W7a fields.
The tests need neither torch nor `bittensor.sp_core`, because they never
verify a signature. Run them locally with `-m "not torch"`.
