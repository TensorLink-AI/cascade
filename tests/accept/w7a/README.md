# W7a acceptance tests: the `lane-d-report/1` schema

W7a (plan rev6 §3.0 W7a row) turns W1.5's thin lane D report into a
versioned, closed schema. It covers §1.5 "(rev3) The report also carries",
the §14.3 forward fields, B7 (`scoring_method`), B12 (first-seen Merkle roots)
and §3.2 invariant 11. These tests are the spec. The builder may touch only
the paths in `PATHS`. W1.5's skeleton must stay green (`make laned-e2e`).

`PATHS` includes the lane D audit, `cascade/audit/lane_d.py`. W1.5's
`check_lane_d` requires the report's `params` to equal its own
`_expected_params(cfg)` exactly. So when W7a adds the five new `params`
fields, `horizons`, `max_mirror_lag_hours`, `gate_suites`, `tau_public` and
`lowfreq_freqs`, and RV4's `gate_weights`, `max_source_share` and
`min_long_series`, the audit must derive them from config too. Otherwise the
W1.5 audit tests fail. The audit derives them itself. It never imports them
from producer code (invariant 12).

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
- **Pools without the new metadata still give a valid report.** A pool_d/1
  snapshot without `feeds`, per-series `feed`, `frontier` or
  `first_seen_roots` (the W1.5 format) gives `feed = source` and
  `first_seen_roots = []`. W5a fails closed (RV3): a feed not listed in
  `feeds` (or no `feeds` block) is refused `unlisted_feed`, and a series
  without `frontier` is `absent_from_s0`. So such a pool judges nothing:
  `cohorts`, `windows`, `rows`, `judged` and `verdicts` are `[]`, the entry
  stays `pending` / `pending_floor`, and the report still validates. The rich
  fixture lists every feed and gives every series a `frontier`. W6fs makes
  the roots mandatory and refuses a snapshot whose records postdate its
  root. That isn't W7a's job.
- **The gate record is the optional top-level `gate`** (RV4). It is present
  iff at least one candidate reached the §1.3b gate at this boundary, and
  absent otherwise (lane closed, nothing judged, no crowning). `null` is
  refused `bad_type`: a signed body has one way to say "no gate". It's
  optional, not required-nullable, because every report without a gated
  candidate (today: all of them, until W5e wires the gate) must keep
  validating unchanged. See "The gate record" below.
- **`gate_weights`, `max_source_share` and `min_long_series` are required
  `params`** (RV4), like every other `params` field. §14.3 item 2: every
  value a satellite needs is in the signed report and cross-checked against
  `config_at(boundary)`. Satellites need `gate_weights` to re-derive the gate
  draw (W3b1), `max_source_share` to re-check W5b's T18e drop and
  `min_long_series` to check a context variant's arming (§1.7). An optional
  field would let a report omit a value the cross-check needs.
- **`pool.snapshots` and `windows[].snapshot_sha256` are required** (RV4,
  for W5b #581). W5b judges windows from older snapshots of a union, so the
  signed report must bind every snapshot a scored version came from
  (invariant 4), and satellites must be able to check it from the report
  alone (§14.3 item 2). Every report has at least one loaded snapshot, the
  newest, so there is no "absent" case to express. An optional field would
  let a report silently drop the binding. The cost is that every producer
  emits them, which is trivial for one snapshot (`[newest]`, every window
  from it). See "Snapshots" below.
- **`lane_closed` is a typed reason** (W5b, T18). At `[lane_d] until_block`
  every open entry closes with status `closed` and reason `lane_closed`, in
  `entries[].reason` and `skipped[].reason`, and that report validates.

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
  list plus `extensions` and `gate`, with `additionalProperties: false`.
- Every object schema that has `properties` also has
  `additionalProperties: false`. At least the ten nested report objects are
  described.
- `$defs` holds the enums: `entry_status`, `entry_reason`, `skip_reason`,
  `burn_reason`, `window_refusal_reason`, `confirmation_stage`,
  `confirm_rule`, `lowfreq_result`, `verdict_outcome` and (RV4) `gate_outcome`
  and `gate_ref`.
- (RV4) `properties.gate` describes the closed gate record; `gate` is not
  in `required`.

## Fields

Top level (all required): `schema`, `boundary`, `signer`, `signature`,
`scoring_method`, `params`, `seed`, `pool`, `house`, `king_before`,
`king_after`, `crowned`, `prior_kings`, `entries`, `judged`, `skipped`,
`cohorts`, `windows`, `verdicts`, `rows`, `rows_sha256`, `pay`. Optional:
`extensions`, `gate` (RV4).

| object | required fields (the tests check these; more are the builder's choice, but they must be in the schema) |
|---|---|
| `params` | `share`, `takes_from`, `remainder`, `horizon`, `horizons` (§14.3 horizon list, `[horizon]` today), `contexts`, `forward_margin_hours`, `max_mirror_lag_hours`, `min_windows`, `min_clusters`, `d_margin`, `alpha`, `bootstrap_B`, `king_decay`, `reward_prior_kings`, `confirm_on_king_change`, `max_entries_per_settlement`, `max_judged_per_coldkey_per_settlement`, `scoring_levels`, `gate_suites` (§14.3 suite list), `tau_public`, `lowfreq_freqs`, **(RV4)** `gate_weights` (`{suite: weight}` as parsed: keys ⊆ `gate_suites`, a suite absent from the table is absent here; finite numbers ≥ 0), `max_source_share` (number in (0, 1]), `min_long_series` (int ≥ 0). Each equals the parsed `[lane_d]` value (lists as JSON lists) |
| `pool` | `snapshot`, `sha256`, `horizon` (the newest snapshot's), `first_seen_roots`: the B12 daily Merkle roots the snapshots rely on, sorted, from pool metadata `first_seen_roots` (the sorted union of `snapshots[].first_seen_roots`), **(RV4)** `snapshots` |
| `pool.snapshots[]` | **(RV4)** `snapshot`, `sha256` (64-hex), `first_seen_roots` (64-hex list) |
| `entries[]` | `digest`, `hotkey`, `coldkey` (A8: as recorded at `ready`, `null` if unresolved), `context` (§14.3 per-entry context), `ready_block`, `reveal_block`, `reveal_ts`, `status`, `reason` |
| `judged` | the digests judged at this settlement, equal to the verdicts' digests |
| `skipped[]` | `digest`, `reason` ∈ `SKIP_REASONS`. One per entry considered this settlement and not judged, **including entries voided this settlement** (`deregistered`, `coldkey_unresolved`, `ready_expired`, …). Disjoint from `judged` |
| `cohorts[]` | `index`, `king_digest`, `king_reveal_ts`, `stage`, `members`, `t_max`, `window_ids`, `set_sha256`, `first_seen_roots` (the roots the frozen set relies on: the snapshot's roots today, a union once W5b unions snapshots), `refused`: `{reason: count}` of the pool windows refused by the §1.3 eligibility rule, keys from `WINDOW_REFUSAL_REASONS`, counts ≥ 1. Windows dropped for other reasons (already used, no forecast) aren't counted |
| `windows[]` | `window_id`, `source`, `feed` (series `feed`, default `source`), `min_target_ts` and `min_first_seen_ts` (both clocks, A1), `mirror_lag_hours` (the feed's `feeds[feed].mirror_lag_hours`, `null` when unmeasured; an unlisted feed's window is refused `unlisted_feed`, so it never appears here), **(RV4)** `snapshot_sha256` (64-hex: the snapshot whose version was scored) |
| `verdicts[]` | `digest`, `king_digest`, `stage`, `cohort`, `lcb`, `gain`, `cleared`, `outcome`, `confirmation`, `lowfreq` |
| `verdicts[].confirmation` | `stage` (`first`/`confirm`), `rule` (= `confirm_on_king_change`), `first_set_sha256` and `first_king_digest`: `null` on stage `first`; on `confirm`, the first clear's `set_sha256` and king (A2, D20) |
| `verdicts[].lowfreq` | `result`, `freqs` (= `lowfreq_freqs`), `n_windows`, `tau_public` (A10). The candidate is a verdict that cleared at stage `confirm`: `pass` or `fail` (`fail` means the entry isn't crowned, `lowfreq_regression`), or `no_windows` when the set holds no window at those frequencies. Every other verdict is `not_candidate` with `n_windows = 0`. The test fixture is hourly, so the crowned candidate is `no_windows`. Computing pass and fail is W3b5's job |
| `rows[]` | `window_id`, `digest`, `cohort`, `crps`, `mase`, all finite |
| `pay` | `share`, `court[]`: `digest`, `hotkey`, `uid`, `burn_reason` |

### Snapshots (RV4, for W5b #581)

| rule | refused as |
|---|---|
| `pool.snapshots` is a list of closed `{snapshot, sha256, first_seen_roots}` objects | `bad_type` at `pool.snapshots`; an extra key is `unknown_field` at `pool.snapshots[i].<key>` |
| `sha256` and each root are 64 lowercase hex | `bad_value` at `pool.snapshots[i].sha256` / `.first_seen_roots[j]` |
| each snapshot is listed once, by name and by sha256 (the name is checked first) | `bad_value` at the later item's `snapshot` or `sha256` |
| the newest snapshot (`pool.snapshot` with `pool.sha256`) is listed | `bad_value` at `pool.snapshots` |
| `windows[i].snapshot_sha256` is one of `pool.snapshots[].sha256` | `bad_value` at `windows[i].snapshot_sha256` |

Cross-field rules run after the structural checks. The producer also keeps
`pool.first_seen_roots` equal to the sorted union of the listed roots (the
e2e test checks it; W5b #581 pins the union across two snapshots). Which
snapshot a window's version comes from is W5b's rule, not W7a's. The fixture
for the validator tests is `helpers.with_snapshots`, spliced into a real
report: one snapshot, or snapshot 1 added as an older member that scores
the first window.

### The gate record (RV4)

`gate` is W3b1's `gate_record(cfg, boundary, block_hash, sentinel=...)` plus
`results`, so W5e writes it without a schema bump (§1.5 "the drawn suite",
§14.3 item 2, §16 rule 1: the sentinel is a structured field). Every object
is closed (`unknown_field`).

| field | rule |
|---|---|
| `suite` | non-empty string, the drawn gate suite; must be one of `eligible` (`bad_value` at `gate.suite`) |
| `block` | int ≥ 0, the settlement boundary the draw used; must equal the report's `boundary` (`bad_value` at `gate.block`) |
| `block_hash` | that boundary's hash, `^0x[0-9a-f]{64}$` (lowercase, as substrate returns it; W5e lowercases) |
| `sentinel` | the era's §16 sentinel suite (non-empty string), or `null` when W-S1 is unarmed |
| `eligible` | the suites the draw ran over (non-empty strings) |
| `results[]` | non-empty (`bad_value` at `gate.results` when empty), one per candidate digest swept at this boundary (a repeated digest is `bad_value` at its `results[i].digest`) |
| `results[].digest` | 64-hex digest |
| `results[].outcome` | ∈ `GATE_OUTCOMES` (`bad_value` otherwise) |
| `results[].lcb` | `{king_d: {crps, mase}, house: {crps, mase}}`, the four LCBs, finite numbers, when `outcome` is `pass` or `gate_regression`; `null` on `checkpoint_fault` / `bench_hold`. The wrong one for the outcome is `bad_value` at `results[i].lcb` |
| `results[].blocked_by` | list of `GATE_REFS`; non-empty iff `outcome == "gate_regression"`, else `[]` (`bad_value` at `results[i].blocked_by`) |
| `results[].session` | non-empty string, the bench host session (W3b2) |

`paired_gate`'s `reason` string is not carried (§14.3 item 4: no free text
in anything replayed), and `alpha`, `tau_public`, `B` and `seed` are already
in `params` / `seed`. The fixture gate is built from the real W3b1 draw and
the real W3b2 gate (`helpers.build_gate`) and spliced into the crowning
settlement's report, since W5e's wiring doesn't exist yet. Which candidates
are swept, and how an outcome maps to an entry's status and reason, belongs
to W5e and W5g.

`rows_sha256` is sha256 over `json.dumps(rows, sort_keys=True,
separators=(",", ":"), allow_nan=False)` (A9). The tests re-derive it
independently.

## Enums (exact sets)

| name | values |
|---|---|
| `ENTRY_STATUSES` | `pending`, `hold`, `confirm_pending`, `crowned`, `lost`, `closed` |
| `SKIP_REASONS` | `pending_floor`, `coldkey_cap_settlement`, `coldkey_unresolved`, `coldkey_changed`, `king_unscored`, `bench_hold`, `deregistered`, `ready_expired`, `reveal_delay_exceeded`, `duplicate_digest`, `hotkey_has_open_entry`, `context_not_armed`, `checkpoint_fault`, `mirror_lag_exceeded`, `lane_closed` (W5b: closed at `until_block`) |
| `ENTRY_REASONS` | `SKIP_REASONS` plus `confirm_pending`, `lost_duel`, `lost_confirmation`, `lowfreq_regression`, `gate_regression` (RV4: a judged loss on the §1.3b gate, like `lowfreq_regression`, so not a skip), `crowned` |
| `BURN_REASONS` | `house_king` (D21: the house holds KING_D), `deregistered` (B8: a court member lost its UID) |
| `WINDOW_REFUSAL_REASONS` | W5a's `REFUSAL_REASONS`, in precedence order: `missing_clock`, `unlisted_feed` (W5a fail-closed), `forecast_product`, `over_lag`, `before_margin`, `publish_ahead`, `late_mirror`, `absent_from_s0`, `backfill` |
| `CONFIRMATION_STAGES` | `first`, `confirm` |
| `CONFIRM_RULES` | `restart`, `original` |
| `LOWFREQ_RESULTS` | `pass`, `fail`, `no_windows`, `not_candidate` |
| `VERDICT_OUTCOMES` | `lost`, `confirm_pending`, `confirmed`, `crowned` |
| `GATE_OUTCOMES` | W3b2's `lane_d_bench.OUTCOMES`: `pass`, `gate_regression`, `checkpoint_fault`, `bench_hold` (`$defs.gate_outcome`) |
| `GATE_REFS` | W3b2's `lane_d_bench.REFS`: `king_d`, `house` (`$defs.gate_ref`) |

`checkpoint_fault` and `bench_hold` are W3b2 outcomes and also
`SKIP_REASONS` (W3b2 README: "Both strings are members of
`lane_d_report.SKIP_REASONS`"): a faulty candidate is refused, a held one is
not judged this settlement.

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
| the W5b `until_block` closing settlement's report validates (`lane_closed`) | `test_e2e_until_block_closing_report_validates` in `test_w7a_lane_closed.py` (W5b's fixture, imported from `tests/accept/w5b/helpers.py`) |
| RV4 gate record: optional, closed, outcomes, LCBs, consistency | `test_w7a_gate_record.py` |
| RV4 `pool.snapshots`, `windows[].snapshot_sha256` | `test_w7a_snapshots.py` |
| RV4 `params.gate_weights`, `max_source_share`, `min_long_series` | `test_w7a_gate_params.py` (default config, plus one settlement under a non-default overlay), `test_every_params_field_is_required`, `test_e2e_params_carry_every_satellite_value` |
| invariant 11 (exact reasons) | `test_inv11_*` in `test_w7a_inv11_reasons.py`. Covers the coldkey cap (`coldkey_cap_settlement`, the entry stays `pending` with that reason), `deregistered`, `coldkey_unresolved` (`coldkey: null`), `ready_expired`, a floor miss (`pending_floor`, no rows, `rows_sha256` of `[]`), a house-held slot (`house_king`) and a crowned king that deregisters (`deregistered`) |

## On main

- The schema-file tests fail on `assert` (there is no file).
- Everything else fails at `ImportError: cannot import name …` from
  `cascade.validator.lane_d_report`, which is in PATHS.
- `test_inv11_fixture_blocks_are_consistent` passes. It checks the fixture.

### Correction: `lane_closed`, `unlisted_feed`, fail-closed fixtures

With W7a built on main, the corrected tests fail on `assert` only, pending
the implementation follow-ups:

- `test_schema_enums_match_the_module_tuples_exactly[entry_reason]` and
  `[skip_reason]`: the schema file and module lack `lane_closed` (W7a).
- `test_schema_enums_match_the_module_tuples_exactly[window_refusal_reason]`:
  W5a's `REFUSAL_REASONS` lacks `unlisted_feed` (W5a fail-closed, then the
  schema file's `window_refusal_reason` enum).
- `test_e2e_until_block_closing_report_validates`: `validate_report` refuses
  `lane_closed` (`bad_value`).
- `test_e2e_legacy_pool_format_report_validates`: main's W5a still admits an
  unlisted feed's windows on the two clocks.

Everything else passes on main and under W5a's fail-closed rule.

The synthetic chain was probed on main: its settlements judge, confirm and
crown as the tests expect, and today's report lacks exactly the W7a fields.
The tests need neither torch nor `bittensor.sp_core`, because they never
verify a signature. Run them locally with `-m "not torch"`.

### Correction (RV4): the gate record and gate params

RV4 BLOCKING W7a: the closed schema had nowhere for the drawn gate suite,
the sentinel, the gate results or the gate weights (nor, for W5b #581, the
snapshots a union scores from), so W5e couldn't sign
them without a `lane-d-report/2` bump. On main these fail on `assert` only:

- every test in `test_w7a_gate_record.py` that inserts a `gate`: main
  refuses it as `unknown_field at gate`, not the defect under test;
- every test in `test_w7a_gate_params.py`, `test_every_params_field_is_required`
  for the three new fields and `test_e2e_params_carry_every_satellite_value`:
  the report lacks them;
- `test_schema_file_top_level_fields_are_exact_and_closed` (no `gate`
  property) and `test_schema_enums_match_the_module_tuples_exactly` for
  `entry_reason` (`gate_regression`), `gate_outcome` and `gate_ref`.

- every test in `test_w7a_snapshots.py` (the fields are absent, and the
  closed schema refuses them as `unknown_field`).

`test_gate_is_optional_reports_without_it_validate` and the gate fixture
check pass on main. A valid report is now checked through
`helpers.assert_valid`, so a refusal is an `AssertionError` naming the code
and field.

### Correction (RV5): crowning fixtures pass the gate inputs (W5e fail-closed)

Once W5e lands (`tests/accept/w5e/README.md` "Fail closed"), an armed
settlement that would crown without a `bench` session and a
`boundary_block_hash` is `bench_hold`, not a crown, and `validate_report`
refuses a crowned report whose `gate` is absent or `null`.

- `helpers.settle` passes `boundary_block_hash` (`helpers.gate_block_hash`),
  `sentinel=None` and a healthy `bench` (`lane_d_bench.SweepSession` over
  `helpers.HealthyBench`) whenever `settle_lane_d` accepts those keywords
  (`inspect.signature`), and makes the legacy call otherwise. So
  `two_settlements` still crowns at BOUNDARY_2 both on today's main and
  after W5e. The bench is the one external service: every digest but the
  house returns the same recorded rows and the house is 10% worse, so the
  crowning candidate passes on any drawn suite and the duel alone decides.
- Every test that validates the crowned BOUNDARY_2 report keeps its
  meaning. On main that report has no `gate`; after W5e it carries the
  gate record W5e writes (a `pass` for the crowned digest), which the RV4
  schema above accepts. The gate record arrives with W5e: these tests don't
  splice one in. `_gated` still replaces it with `helpers.build_gate`.
- `test_gate_is_optional_reports_without_it_validate` and
  `test_no_gated_candidate_means_no_gate_key` now use uncrowned reports
  (the BOUNDARY_1 first clear and both W1.5-format settlements): without a
  gate only an uncrowned report validates, and `gate: null` on it is still
  `bad_type at gate`. Their pass/fail on main is unchanged (the first
  passes, the second fails on `unknown_field at gate`, as before).
