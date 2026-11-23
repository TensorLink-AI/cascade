# W12a acceptance tests: lane D docs and agent surfaces

W12a (plan rev6.4 §3.0 W12a row, §11.2 and its "§11.2 → WPs" row) writes the
lane D text surfaces. These tests are the spec. The builder may touch only
the docs in `PATHS`. They are doc **contract** tests, not prose review.
W12a depends on W10: the CLI the docs describe must be on main first.

The W12a row, verbatim:

> Docs and agent surfaces (§11.2 text surfaces, website; **rev3:** LANE_D.md
> FAQ on coldkey caps, confirmation latency, both clocks, BF16; VALIDATOR.md
> on what satellites verify for the new report fields; OPERATOR.md on launch
> eval host, S8, first-seen recorder) | W10 | ci | `scripts/laned/accept W12a`
> (every §11.2 surface mentions lane D; links resolve; `llms.txt` lane list
> matches the status fixture)

## The code is the truth

The tests hardcode no list of keys, defaults, file names or reason codes.
`helpers.py` reads them from main on every run, with torch-free imports:

| fact | source |
|---|---|
| `[lane_d]` keys and defaults | `cascade/shared/config/lane_d.py` `LaneDConfig` fields |
| checkpoint files | `cascade/lane_d/intake.py` `ALLOWED_FILES` |
| typed skip reasons | `cascade/validator/lane_d_report.py` `SKIP_REASONS` |
| report fields | `cascade/validator/schemas/lane_d_report_1.json` (top level and `entries` / `verdicts` / `windows`) |
| `cascade lane-d` commands | the real parser once W10 is on main, else the plan's five |
| the status fixture's lane list | the `lanes` keys of `live_status.status_snapshot({})`, upper-cased (`cascade status --json`; a/b/c on main, d once W12b lands) |

The parsers come from WB20 (`tests/accept/wb20/helpers.py`): sections, spans,
`has_token`, paragraphs, TOML default equality, links with the
`check_docs.py` slug algorithm, and leaks.

## Definitions

- **Lane D section:** a heading (outside fenced code) matching `lane[ -]?d`
  (case-insensitive), up to the next heading of the same or a higher level.
- **Mention:** `lane[ _-]?d` anywhere in the file.
- **Token:** WB20's `has_token`: an inline code span, the whole span or a
  whole word inside it, or a whole word on a fenced line.
- **Paragraph:** a blank-line separated block. A list or a table counts as
  one block. "A paragraph with X and Y" means one block holds both
  (case-insensitive).
- **Negation:** `no`, `not`, `never`, `cannot`, `can't`, `without`, `nor`
  or `neither`.
- **Key with default:** one line has a span naming the key (`` `k` ``,
  `` `[lane_d] k` `` or `` `k = v` ``) and the `LaneDConfig` default, either
  in `` `k = v` `` or as another span on that line (TOML value equality, so
  `6` equals `6.0`).

## Every surface (`test_w12a_surfaces.py`)

- **Mentions lane D:**
  - `docs/LANE_D.md`, INTERFACE, MINER, AGENT_MINING, VALIDATOR, OPERATOR and
    ARCHITECTURE;
  - `AGENTS.md`, `agent.md`, `README.md` and `CHANGES.md`;
  - both `llms.txt` (repo and `cascade/website/`) and the website
    `index.html`.
- **Has a lane D section:** the nine Markdown text surfaces (LANE_D, the six
  guides, AGENTS, agent).
- **Links resolve** in those nine plus `llms.txt` and `README.md`, anchors
  included. `CHANGES.md` is exempt: it cites absolute `/tmp` logs of old runs.
- **No secret, private host or non-documentation IPv4** in LANE_D.md, or in
  the lane D sections of the others. These are WB20's patterns.
- `python -P scripts/check_docs.py` exits 0.
- **The `llms.txt` lane list matches the status fixture.** The first-column
  letters of the repo `llms.txt` tables headed `Lane` equal the status
  snapshot's lanes. If W12b is on main first, the table gains its D row. If
  not, it gains the row in the PR that follows W12b.
- Both `llms.txt` link `LANE_D.md` and drop the stale "Planned order A → C →
  B". `README.md` links `docs/LANE_D.md`. `CHANGES.md` has a release-note
  paragraph that names lane D and either `LANE_D.md` or `lane-d`.
- **Stale guards** (pass on main):
  - every `` [lane_d] <key> `` on any surface is a `LaneDConfig` field;
  - every `cascade lane-d <cmd>` is a real command.

## `docs/LANE_D.md` (new; `test_w12a_lane_d_md.py`)

- **Title:** the `# ` title names lane D.
- **Checkpoint format:** every `ALLOWED_FILES` name as a token, plus
  `pyramid_model.py` and `pyramid_wrapper.py`. A paragraph says BF16 only and
  that F32 is refused.
- **Contexts:** every default context (`4096`/`4k`, `8192`, `16384`), and a
  paragraph naming S8 and context.
- **Commands:** `cascade lane-d export|verify|submit|ready|status`, each in a
  span or code block.
- **Hotkey:** "dedicated hotkey".
- **Licence:** a paragraph naming `Apache-2.0`, `OpenMDW`, "training data"
  and "teacher" (the B13 attestation). "takedown".
- **D43:** a paragraph with "reveal", "immediate(ly)" and "commit".
- **Latency:** a paragraph with "forward-only", "latency" and
  hourly/daily/frequency.
- **Pay:** "house", "gate", "sentinel" and "court" are explained.
- **A FAQ section** (a heading with `FAQ` or "frequently asked") with:
  - each of `max_open_per_coldkey`, `max_judged_per_coldkey_per_settlement`,
    `max_entries_per_settlement`, `max_coldkey_unresolved_settlements`,
    `max_reveal_delay_days`, `ready_ttl_blocks` and `max_mirror_lag_hours`
    **with its default**;
  - every `SKIP_REASONS` value as a token;
  - a paragraph on the confirmation duel and its latency ("confirmation" plus
    latency/settlement/later);
  - a paragraph on both clocks ("first-seen", "data time" or "data-time",
    "mirror lag");
  - "BF16".
- Every `` `[lane_d] k` `` span names a real key.

## The guides (`test_w12a_guides.py`)

| surface | requirement (in its lane D section unless noted) |
|---|---|
| INTERFACE | every `ALLOWED_FILES` token and "BF16" |
| MINER | a `Lane` table with a `D` row (anywhere in the file); a link to `LANE_D.md`; a paragraph with "compute key(s)" and a negation |
| AGENT_MINING | a paragraph with "overstate", "forward" and revealed/local |
| AGENTS.md, agent.md | a paragraph with "human" and "authoris/authoriz…" |
| VALIDATOR | the tokens `rows_sha256`, `judged`, `skipped`, `coldkey`, `confirmation`, `lowfreq`, `min_target_ts`, `min_first_seen_ts` and `mirror_lag_hours`, each self-checked against the schema; a paragraph with "satellite(s)", "target" and a negation (what satellites cannot verify) |
| OPERATOR | "valeval-laned", "S8", "first-seen", "kill switch", "Blackwell", "vault", "retention", and the tokens `cascade-laned-intake` and `lane_d_coldkey_unresolved`; a paragraph with "eval host", "pinned" and "torch" |
| ARCHITECTURE | the tokens `cascade-laned-intake` and `lane_d_state.json`; "signed" and "report" |

The OPERATOR lane D section already exists on main (`cascade-laned overlay`:
lane D bring-up kit) and holds "kill switch". The other requirements extend
it.

## Not pinned

- Prose quality and section order.
- The website's HTML beyond a lane D mention.
- The status JSON fields (W12b) and `status/lane_d.json` (W17).
- DEC-CA-0070 (W0).

## Invariants

None owned, no `unchanged_when_unarmed` names.

## Failure on main (100 collected: 73 fail, 27 pass)

`docs/LANE_D.md` doesn't exist, the other guides have no lane D section, and
the `llms.txt` / README links are missing. Every failure is an
`AssertionError`.

PATHS are docs only, so no failure can be a PATHS import error. The link,
leak, check_docs, lane-list and stale tests pass on main and must stay
green.

A scratch `LANE_D.md` written to the contract passed all 29 LANE_D.md tests,
which checks the parsers.

## Correction (plan rev6.11 (b)): crown at the first verdict

The confirmation duel is dropped (owner 2026-11-23): a crown is decided at
the first verdict, first clear (max-T LCB vs KING_D) ∧ A10 ∧ the §1.3b gate.
There is no `confirm` stage, no `confirm_pending` status or reason and no
`lost_confirmation`; a held first clear is `hold` / `bench_hold`. This WP's
crowning chains move to that rule with nothing weakened. The VALIDATOR guide check no longer requires the removed `confirmation` report field (W7a makes a verdict carrying it `unknown_field`). On main the
old code still runs the confirmation stage, so the changed tests fail on an
assertion until the W5e implementation lands.
