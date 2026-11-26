# W0 acceptance tests: the contract they pin

W0 (plan §3.0) finalises **DEC-CA-0070** (lane D open weights). It supersedes
DEC-CA-0062's recipe design and records **D1–D43, D45–D48 and D42b**, with
**D44 withdrawn**. It runs in `ci + owner merge`. The builder may touch only
the paths in `PATHS`:

- `decisions/DEC-CA-0070-lane-d-open-weights.md`
- `decisions/DEC-CA-0062-lane-d-distillation-real-data.md` (the back-link)

The tests read the repo directly and use only the stdlib. Every expected
choice comes from parsing the **§17.4 decision register** in
`docs/plans/LANE_D_PLAN.md` on main at test time. They are never copied
into the tests, so a register change made by an owner-merged plan PR flows
through.

## Front-matter rules (there is no decision validator in the repo)

These rules come from the records on main. The `test_rules_hold_for_existing_*`
tests check that every accepted/active and superseded record on main obeys the
floor below.

- The first line is `---` and the front matter is closed by a `---` line. Each
  line is `key: value`. The value is a double-quoted string, a flow list
  `[a, b]`, a flow mapping `{k: [..]}`, a bare scalar, or a `>-` / `>` / `|`
  block scalar.
- **Floor (every record):** `id`, `type`, `title`, `status`, `date` and `tags`
  are present. No keys other than those, `revisit_when`, `relations`, `note`
  and `amended` are allowed. `id` equals the `DEC-CA-NNNN` prefix of the file
  name, `type: decision`, `status` is one of `proposed | accepted | active |
  superseded`, `date` is an ISO date and `tags` is a non-empty flow list.
- **Strict (DEC-CA-0070 and DEC-CA-0062):** the floor, plus a non-empty
  `revisit_when` and a `relations` flow mapping. Every relation target is a
  `DEC-CA-NNNN` id with a file in `decisions/`.

## What each test pins

| test | pins |
|---|---|
| `test_dec70_front_matter_valid` | DEC-CA-0070 passes the strict rules |
| `test_dec70_id_unique` | only this file claims `id: DEC-CA-0070` |
| `test_dec70_status_is_decided` | `status` is `accepted` or `active` (the decided values on main), never `proposed` |
| `test_dec70_supersedes_dec62` | `relations` has `supersedes: [DEC-CA-0062]`. That is the full-supersession key the repo uses, e.g. DEC-CA-0043 and DEC-CA-0044. No other `supersedes*` kind (such as the draft's `supersedes_on_acceptance`) may name DEC-CA-0062 |
| `test_dec70_records_every_decision` | the body has a register table `\| D \| Status \| Option chosen \| …` with one row per D-number in the W0 row ∪ the plan register, and no D-number the plan lacks |
| `test_dec70_d44_withdrawn` | D44's row has status `withdrawn` |
| `test_dec70_choices_match_plan_register` | for every plan register row, the record's `Status` and `Option chosen` cells equal the plan's. So does any other register column the record carries (`Class`, `Decided`). Comparison collapses whitespace and drops `**`. Every status is `decided`, `withdrawn` or `deferred-to-stage` |
| `test_dec70_has_no_undecided_wording` | the whole file (front matter included) is free of undecided wording (see below) |
| `test_dec62_marked_superseded_by_dec70` | DEC-CA-0062 passes the strict rules, has `status: superseded` and has a `note` starting `Superseded by DEC-CA-0070`. That matches how DEC-CA-0003/0005/0006/0011/0017/0037 carry their back-links |
| `test_plan_register_covers_w0_and_nothing_is_open` (sanity) | the plan register lists every W0 D-number, none is `open`, and D44 is `withdrawn` |
| `test_undecided_wording_scanner_accepts_register_text` (sanity) | the plan's own register cells pass the wording scan, so a faithful copy can pass |

The record may contain more than one register table (for example split by
plan section). A D-number may appear in only one row. Rows are matched by the
first three header cells `D`, `Status` and `Option chosen` (case-insensitive).
Prose outside the table is free-form.

## Undecided wording (flagged per line)

- `[DECIDE]` anywhere.
- `S6a`, `S6b` or `S6c`. The rev6 stages are S0–S8 plus SB0–SB5, and these
  sub-stages no longer exist.
- `pending`, any case.
- `proposed`, any case, unless it is followed by `change(s)` or `design
  change(s)`. That allows the plan's "Proposed change 22" references.
- `shadow*`, unless the same clause (the 60 characters before the match, cut at
  the last `.`, `;`, `!` or `?`) holds a negation: `no`, `not`, `never`,
  `without`, `nor`, `instead of` or `replace(s|d)`. So D45's "No shadow modes"
  passes.
- A heading containing `Open decisions`.
- `none is adopted` / `none are adopted`.

## Failure on main (expected, for the right reason)

The draft has `status: proposed` and `supersedes_on_acceptance: [DEC-CA-0062]`.
Its only D table is the `# | Decision | Default / recommendation` defaults
table, with no `Status` / `Option chosen` register, so all 50 D-numbers are
missing. It carries `[DECIDE]`, S6a/S6b/S6c, "pending" and non-negated
"shadow" wording. DEC-CA-0062 is still `status: proposed` with no back-link.
The front-matter-valid, id-unique, record-exists and sanity tests pass on
main.
