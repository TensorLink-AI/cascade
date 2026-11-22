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
| W5e gate in the audit: `check_lane_d` re-derives the draw, each result's outcome from its LCBs, the verdicts and the crown; refuses crowns without a passing gate | `test_w7a_audit_gate.py` (see "Correction (W5e gate)") |
| RV7 B2: held-crown provenance (`held_from`), the confirmation's first clear, the seed, `king_before`, the era sentinel, W3b5 lowfreq fails; chain mode | `test_w7a_audit_provenance.py` (see "Correction (RV7 B2)") |
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

### Correction (W5e gate): the audit re-derives the gate

W5e wired the §1.3b gate into `settle_lane_d` (`tests/accept/w5e/README.md`
"Gate wiring", "Fail closed"). A cleared confirmation is no longer always
`confirmed` / `crowned`, and the W5e README leaves the audit's half of the
fail-closed rule to W7a ("The audit must refuse it too"). `check_lane_d`
(`cascade/audit/lane_d.py`, in `PATHS`) today treats the highest-LCB
`confirmed`/`crowned` verdict as the rightful crown and requires every
cleared confirmation to read `confirmed`/`crowned`. That rejects honest
gated reports and accepts crowns the gate never passed. §14.3 items 1-2:
satellites re-derive every verdict from signed fields, so the audit must
re-derive the gate's too, from the report and `cfg` alone (invariant 12:
no import of `cascade.validator.lane_d*`).

`check_lane_d` must re-derive:

| rule | from |
|---|---|
| `gate`, when present, is a dict; `gate.block == boundary`; `gate.suite` and `gate.eligible` are the draw `gate_suite_for(gate.block_hash, gate_suites, gate_weights, gate.sentinel)` over `cfg` (the sentinel is never the suite) | W3b1 |
| a result with LCBs is `pass` iff all four (`king_d`/`house` x `crps`/`mase`) are `>= -tau_public`, else `gate_regression`, with `blocked_by` = the refs that have an LCB below `-tau_public` | W3b2 `paired_gate` |
| every cleared `stage: "confirm"` verdict has a gate result when `gate` is present; `pass` -> `crowned` or `confirmed`; `gate_regression` / `checkpoint_fault` -> `lost`; `bench_hold` -> `confirmed`, not crowned | W5e outcomes table |
| without `gate` (the gate could not run, fail closed) a cleared confirmation reads `confirmed` and nothing crowns | W5e "Fail closed" |
| the crown is the highest-LCB cleared confirmation whose result is `pass`; a `pass` result with no verdict in the report is a held candidate re-gated at this settlement and may be the crown; no `pass` -> `crowned` is `null` and the king is unchanged | W5e "Several candidates", "`bench_hold` keeps the candidate's chance" |

A crowned report whose `gate` is absent or `null`, or whose results hold no
`pass` for the crowned digest, FAILs. Every FAIL is a typed `CheckResult`.

Tests: `test_w7a_audit_gate.py`. The reports come from real
`settle_lane_d` calls with the W5e gate inputs, built with W5e's fixtures
(`tests/accept/w5e/helpers.py`: `settle_gated`, `BenchHost`,
`bench_table`), plus W7a's own `two_settlements`. Doctored reports are
re-signed by the validator's key, and each doctored case first requires
the honest report it starts from to PASS.

| case | expected |
|---|---|
| gate `pass` crown (with and without a §16 sentinel); W7a's gated chain | PASS |
| cleared confirmation `lost` via `gate_regression`; `closed` via `checkpoint_fault`; `bench_hold` (confirmed, not crowned) | PASS |
| passing runner-up crowned over a regressing max-T best | PASS |
| held candidate re-gated and crowned at a later settlement, no verdict for it | PASS |
| ungated (legacy call) reports: a first clear and a loser; an armed crowning candidate held (fail closed, no `gate`) | PASS |
| crowned with `gate` absent / `null` / no result for the crowned digest / its result `bench_hold` or `gate_regression` | FAIL |
| `gate.suite` not the draw; `block_hash` that draws another suite; `block != boundary`; the sentinel stated as the suite | FAIL |
| `pass` with an LCB below `-tau_public`; `gate_regression` with every LCB clear; `blocked_by` naming the wrong ref | FAIL |
| a `gate_regression` verdict left `confirmed`; a `pass` left uncrowned; a `pass` doctored onto the regressing best (re-derived crown differs); a cleared confirmation missing from `gate.results` | FAIL |
| a held re-gated crown whose result is `bench_hold` or whose `gate` is gone; an ungated cleared confirmation stated `crowned` | FAIL |

On main (W5e built, the audit not gate-aware) 24 of the 28 tests fail on
an `assert`: the honest gated outcomes FAIL the audit ("outcome lost not in
('confirmed', 'crowned')", "crowned None; re-derived ...", "the king changed
without a confirmed verdict"), and the doctored crowns PASS it because it
ignores `gate`. Four pass on main: the two honest gate-pass crowns, W7a's
gated chain and the ungated no-crown reports. Every other W7a test file is
unchanged.

### Correction (RV7 B2): the crown's provenance, not only its gate

RV7 B2 (`evidence/rv7.md`): #649's `check_lane_d` and `verify_report` PASS
two forged crowns once the validator key re-signs them:

- (a) any listed entry with a gate `pass` and no verdict is taken as a held
  candidate, so an entry still `pending` / `pending_floor` that was never
  judged can be crowned;
- (b) `cohorts[].stage` is taken on trust, so a first duel relabelled
  `confirm` crowns in ONE settlement. That skips §1.3's mandatory
  confirmation duel (the D20 / A2 false-crown bound).

It also takes on trust (c) `report.seed`, the KING_D of a stage-`first`
cohort and `gate.sentinel`. W3b5 (#651) adds (d): a cleared confirmation
whose A10 `lowfreq` block reads `fail` is `lost`, so the audit must expect
that instead of a gate outcome.

§1.5 / §14.3 items 1-2: satellites re-derive every verdict from signed
fields. A crown therefore needs signed provenance, and the audit checks it.

#### New report field (W7a implementer: schema, producer, validator)

`gate.results[].held_from`: optional and closed. It's present on a result
**iff** that result is a held candidate re-gated without a verdict at this
settlement. It's absent (never `null`, which is `bad_type`) on a result
whose candidate has a verdict here. All five keys are required:

| key | value |
|---|---|
| `boundary` | int: the settlement where the entry's confirmation cleared and the gate first held it (`bench_hold`, or no gate record: the fail-closed legacy call). It's the entry's `gate_hold.boundary`, which keeps its first record across later holds |
| `report_sha256` | 64-hex: that settlement's report body digest (`lane_d_state.report_digest`: sha256 over the canonical body without `signature`) |
| `set_sha256` | 64-hex: that confirmation cohort's `set_sha256` (`gate_hold.set_sha256`) |
| `lcb` | number: that confirmation verdict's KING_D `lcb` (`gate_hold.lcb`) |
| `king_d` | 64-hex: the KING_D the hold was taken under (`gate_hold.king_d`) |

Producer: when `gate_hold` is first written it doesn't know the report
digest yet. Store it in `gate_hold.report_sha256` once `report_digest` is
computed and before `save_state`. The W7c state parser must accept the key.
Then copy the hold into the held candidate's gate result. The schema file
describes `held_from` as a `$defs` object with
`additionalProperties: false`. No other report field changes:
`confirmation.first_set_sha256` / `first_king_digest` already exist, and
the stage `confirm` rules below apply to them.

#### `check_lane_d` contract

`check_lane_d(report, cfg, *, signer_hotkey, prior_reports=None, era_sentinel=<unset>)`.
Both new keywords are optional, so every existing caller keeps its
behaviour.

These single-report rules always apply (FAIL on breach):

| rule | RV7 |
|---|---|
| `seed == f"lane-d/1:{boundary}:{pool.sha256}"` | (c) |
| a stage-`first` cohort's `king_digest == king_before.digest` | (c) |
| a stage-`confirm` verdict has a 64-hex `first_set_sha256` that differs from its cohort's `set_sha256`, `first_king_digest ==` its cohort's `king_digest`, and `confirmation.stage == "confirm"`. Under `confirm_on_king_change = "restart"`, its cohort's king is `king_before.digest` | (b) |
| a held result (gate result, no verdict, a listed entry) carries `held_from` with exactly the five keys, `boundary <` the report's boundary, a finite `lcb >= d_margin`, and, under `restart`, `king_d == king_before.digest` (a king change drops a hold). A verdict-backed result carries no `held_from` | (a) |
| a cleared `confirm` verdict whose `lowfreq.result` is `pass`/`fail` has `lowfreq.lcb`. Its result is re-derived: `fail` iff some LCB `< -tau_public`. A `fail` must read `lost`, needs no gate result (one is a FAIL), and is never in the passing set. `no_windows` / `pass` go to the gate as before | (d) |

**Chain mode** (`prior_reports` given): this is what W9 runs. The list holds
the earlier signed reports the auditor has, in any order. A prior report
counts as evidence only if its signature verifies under `signer_hotkey`. In
chain mode, each of the following must be shown, or the check FAILs:

- **Each confirm verdict's first clear.** Some earlier verified report must
  have a stage-`first` cohort with `set_sha256 == first_set_sha256` and
  `king_digest == first_king_digest` that lists the digest as a member. Its
  verdict there must be cleared and `confirm_pending`, and its `window_ids`
  must be disjoint from the confirming cohort's.
- **Each `held_from`.** The verified report at `held_from.boundary` must
  exist, and its body digest must equal `report_sha256`. In it, the digest's
  verdict must be stage `confirm`, cleared and `confirmed`, with
  `lcb == held_from.lcb` and `king_digest == held_from.king_d`. Its cohort's
  `set_sha256` must equal `held_from.set_sha256`. Its gate record must be
  absent or hold a `bench_hold` result for the digest.

A structurally valid but invented `held_from` or `first_set_sha256` can only
be caught here, which is why W9 must always audit with the chain.

**`era_sentinel`** is the era's W-S1 receipt record, `{"suite", "block",
"block_hash"}`, or `None` when W-S1 is unarmed for the settlement's era. When
it's given and the report has a `gate`, `gate.sentinel` must equal
`era_sentinel["suite"]` (or be `null` for `None`). Re-deriving the record
from its block hash is the receipt audit's job (W-S1). Left unset, nothing
about the sentinel is checked (today's behaviour).

#### Tests (`test_w7a_audit_provenance.py`)

The reports come from real `settle_lane_d` calls. They reuse the W5e
scenarios from `test_w7a_audit_gate._build` and add two:

- `pending`: RV7's reproducer, with Y `pending_floor`.
- `challenger`: Y's first duel against a miner KING_D with the house bar.

W3b5's T4b chain supplies a lowfreq `fail`, and W7a's `two_settlements` is
used too. Each doctored case first requires its honest report to PASS.

| case | expected |
|---|---|
| the hold run's crown carries `held_from` naming the boundary(2) hold (boundary, digest, set, lcb, king); verdict-backed results carry none; the schema describes it closed, optional, `null` / extra / missing keys refused | producer + schema |
| honest held crown alone and in chain mode; every honest `s2` (pass, sentinel, regression, fault, hold, runner-up, W7a's chain) in chain mode; an honest lowfreq `fail` alone and in chain mode | PASS |
| RV7 (a): a never-judged entry crowned as held (no `held_from`); the same with an invented `held_from` (chain `[]`) | FAIL |
| `held_from` absent, `boundary` not earlier, `lcb < d_margin`, `king_d` not `king_before`; `held_from` on a verdict-backed result | FAIL |
| chain mode: `held_from` naming the first-clear report, wrong `report_sha256` / `set_sha256` / `lcb`, hold report missing, hold report's signature invalid | FAIL |
| RV7 (b): first duel relabelled `confirm` (null first set); the same with an invented first set (chain `[]`); a confirm verdict with first set null / equal to its own set, first king null / not the cohort king | FAIL |
| chain mode: first clear's report absent; `first_set_sha256` naming no first-stage cohort; the first clear's report doesn't verify | FAIL |
| (c): a ground seed with every LCB re-run under it (both settlements); a first duel relabelled as judged against the house instead of `king_before`; `gate.sentinel` against `era_sentinel` (null in an armed era, stated in an unarmed one, another suite, a re-stated self-consistent draw) | FAIL (the matching honest era records PASS) |
| (d): a lowfreq `fail` doctored to crowned (with a forged gate `pass`), to crowned with `result: "pass"`, to `result: "pass"` still `lost`, to `confirmed` | FAIL |

The (d) tests skip ("W3b5 (#651) not on this base") while the producer
writes no `verdicts[].lowfreq.lcb`, which means until #651 lands.

**On main** (`4ae3f41`): 29 fail, 2 pass and 5 skip (d).

- Every doctored single-report case fails on an `assert` because main's
  audit PASSes the forgery.
- The `held_from` producer and schema tests fail on an `assert` because the
  field is absent.
- The chain-mode and `era_sentinel` cases fail on an `assert` that
  `check_lane_d` takes no `prior_reports` / `era_sentinel`.
- `test_results_with_a_verdict_carry_no_held_from` and
  `test_held_from_on_a_verdict_backed_result_fails_the_audit` pass on main.
  The second passes for an incidental reason: main's closed `_RESULT_KEYS`
  refuses the extra key.

Once #651 lands on main without this fix, the five (d) cases fail on an
`assert`. The honest lowfreq `fail` FAILs main's audit with "outcome lost
!= re-derived confirmed".

A scratch prototype of this contract (not committed) passed every W7a test
with and without #651 merged, plus the w3b5, w5b, w5c, w5e, w7c, w8 and w1_5
suites. That shows the contract can be built and that every honest fixture
report PASSes.

`test_w7a_audit_gate.py` and every other W7a test file are unchanged.
