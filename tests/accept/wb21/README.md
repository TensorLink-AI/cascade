# WB21 acceptance tests: the contract they pin

WB21 (plan §3.0, lane B rows) is the **DEC-CA-0071 amendment**: "this PR
drafts it; owner accepts". It runs in `ci + owner`. It owns no T-rows or
invariants. The builder may touch only the path in `PATHS`:

- `decisions/DEC-CA-0071-lane-b-public-suite-targets.md`

The tests read the repo directly and use only the stdlib. Everything the
record must say is parsed from `docs/plans/LANE_D_PLAN.md` at test time, so an
owner-merged plan change flows through:

- **Appendix C** ("DEC-CA-0071 amendment rev4 (text)"), the `### Amendment
  rev4` subsection: its numbered items, its `New `[lane_b]` keys:` list and
  its `Each key:` paragraph;
- the **§17.4 decision register** rows for **D35–D42 and D42b**;
- **§12b.2** (MoE-ordinal `declared_rows` keys) and **§12b.6** (D42 / D42b).

## Front-matter rules and wording scan (same as W0)

The front-matter parser, the rules and the undecided-wording scan are copied
unchanged from W0's tests (`tests/accept/w0/`, PR #489), so the two decision
WPs agree. In short:

- **Floor:** `id`, `type`, `title`, `status`, `date` and `tags` are present.
  The only other allowed keys are `revisit_when`, `relations`, `note` and
  `amended`. `id` equals the file-name prefix, `type: decision`, `date` is an
  ISO date and `tags` is a non-empty flow list.
- **Strict (DEC-CA-0071):** the floor plus a non-empty `revisit_when` and a
  `relations` flow mapping whose targets are `DEC-CA-NNNN` ids with a file in
  `decisions/`.
- **Undecided wording** (whole file, front matter included, per line):
  `[DECIDE]`; `S6a`/`S6b`/`S6c`; `pending`; `proposed` unless followed by
  `change(s)` / `design change(s)`; `shadow*` unless negated in the same
  clause; an `Open decisions` heading; `none is/are adopted`.

## Where the record's text is looked for

- **The amendment section** is the body under the first heading (any level)
  whose text starts `Amendment rev4`, up to the next heading of the same or a
  higher level. The plan's heading says `(proposed, …)`; the record's must not,
  because of the wording scan.
- **Numbered items** are top-level `N. ` lines plus the following blank or
  indented lines. Item *n* in the record is compared with item *n* in
  Appendix C.
- **Register tables** are any pipe table in the record whose first three
  header cells are `D`, `Status`, `Option chosen` (case-insensitive), as in
  W0. A D-number may appear in only one row.
- Comparisons collapse whitespace and drop `**`.

## What each test pins

| test | pins |
|---|---|
| `test_dec71_front_matter_valid` | DEC-CA-0071 passes the strict rules |
| `test_dec71_id_unique` | only this file claims `id: DEC-CA-0071` |
| `test_dec71_status_is_decided` | `status` is `accepted` or `active`, never `proposed` |
| `test_dec71_amendment_items_match_appendix_c` | for each Appendix C item *n*, the record's amendment item *n* exists and contains (case-insensitive) the item's bold lead, with any parenthetical holding `[DECIDE]` removed (so `Sentinel ([DECIDE] D35).` → `Sentinel`); every inline-code span of the plan item; and every D-number the plan item names. The record has no items the plan lacks |
| `test_dec71_amendment_lists_new_lane_b_keys` | the amendment has a `New `[lane_b]` keys:` bullet list with every key of the plan's list |
| `test_dec71_amendment_key_rules_match_appendix_c` | the amendment's `Each key:` paragraph has every code span of the plan's (`ALLOWED["lane_b"] = None`, `remainder`, `contamination_hashset`) and states `contamination_hashset` is release-only (rev5) |
| `test_dec71_owner_direction_recorded` | the amendment carries the plan's `Owner direction (…)` lead |
| `test_dec71_records_lane_b_decisions` | a register row for each of D35–D42 and D42b; no D-number the plan register lacks |
| `test_dec71_choices_match_plan_register` | each row's `Status` and `Option chosen` equal the plan's, as do any other register columns the record carries (`Class`, `Decided`); every status is `decided`, `withdrawn` or `deferred-to-stage` |
| `test_dec71_declared_rows_keys_are_moe_ordinals` | the amendment says `declared_rows` keys are MoE ordinals 0–6, not block indices, and that the clamp minimum applies only to layers with at least one declared row |
| `test_dec71_d42_gpu_list_and_leg_hours_rules` | the amendment states the D42 / D42b rules (`D42_RULES` in the test): `[lane_b] funded_pod_skus` separate from `[round]`; GPU types qualified and measured first; `train_hours` from 0.5 to `max_leg_hours` 6 in 0.5 h steps; `guard_factor`; the guard capped by `[lane_b] max_leg_seconds`; `[training] max_train_seconds` not changed; D42b (a) wall/real hours on the chosen GPU via `measured_steps_per_s`; steps fixed at admission and signed. Appendix C item 6 alone isn't enough: the §12b.6 rules must be carried too |
| `test_dec71_has_no_undecided_wording` | the whole file passes W0's scan |
| `test_dec71_numbering_consistent` | the file never mentions `DEC-CA-0069` (the record was renumbered 0071, plan A7) and its `id` is `DEC-CA-0071` |

Sanity tests (pass on main): `test_rules_hold_for_existing_decided_records`
(the floor accepts every accepted/active record on main),
`test_plan_register_has_wb21_decisions_decided`, `test_plan_appendix_c_parses`
(≥ 6 items with leads, ≥ 11 new keys), `test_plan_states_every_d42_rule` and
`test_plan_states_every_moe_ordinal_rule` (every rule pattern matches the
plan's own §12b.6 / §12b.2 + Appendix C text, so the patterns are the plan's
wording), and `test_undecided_wording_scanner_accepts_register_text`.

## Failure on main (expected, for the right reason: assertions only)

The record on main is the PR #482 text plus the rev4 amendment as first
drafted: `status: proposed`, `## Decision (proposed …)`, `## Amendment rev4
(proposed, …)`, four `[DECIDE]` markers and "pod approval pending". Item 6 is
still the pre-D42 "measured before deciding" text, so it lacks the
`Expert-leg cost (D42, decided)` lead, `train_hours`, `[lane_b]
max_leg_seconds` and `[training] max_train_seconds`. The `Each key:`
paragraph lacks the rev5 `contamination_hashset` release-only rule. There is
no D register table, the rev5 MoE-ordinal sentences are absent and none of
the D42 rules is stated. The front-matter-valid, id-unique, new-keys,
owner-direction, numbering and sanity tests pass on main.
