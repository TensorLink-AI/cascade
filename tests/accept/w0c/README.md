# W0c acceptance tests: the contract they pin

W0c builds the lane D loop tooling (plan §3.0 W0c row, §3 conventions, §3.1
build rules, invariant 10, §17). These tests are the spec. The builder may
touch only the paths in `PATHS`. Every tool is driven as a subprocess,
`python scripts/laned/<tool> ...`, so the tools must be Python 3.11 scripts
(stdlib only is enough). `accept` and `status` have no `.py` suffix.

All fixture repos are real git repos built in `tmp_path`. The only mocked
service is GitHub Actions: its results reach `status` as a JSON file.

## Common conventions

- **WP directory:** `tests/accept/<wp>/`, where `<wp>` is the WP id
  lowercased with `.` and `-` replaced by `_` (`W1.5` → `w1_5`,
  `W-S1` → `w_s1`, `RV1` → `rv1`).
- **`PATHS`:** one entry per line. Blank lines and lines starting with `#`
  are ignored. An entry ending in `/` matches everything under that
  directory. An entry containing `*`, `?` or `[` is matched with
  `fnmatch.fnmatchcase` against the repo-relative path. Any other entry
  matches that exact file.
- **`--root DIR`:** every tool takes it as the repo to act on. Relative paths
  given on the command line (`--evidence`, `--config`, `--source`,
  `--patterns`, scan paths) resolve against `--root`.
- **Pytest command:** `accept` runs the tests with `uv run --frozen pytest`
  by default. When the env var `LANED_PYTEST_CMD` is set, it is
  shlex-split and used in its place. The tests set it to
  `python -m pytest -p no:cacheprovider`. The command always runs with
  cwd = `--root`.
- **Exit codes:** 0 means pass. Any non-zero code means fail. Usage errors
  may use 2.

## `stub_scan.py` (build rule 3)

`stub_scan.py --root R [PATH_OR_GLOB ...] [--paths-file FILE]` scans the
listed paths. Directories are recursed. `--paths-file` reads a `PATHS` file.
Files whose first path component is `tests` are exempt.

It prints one line per violation, `<relpath>:<line>: <rule-id> ...`, and
exits 1 if there are any. The rule ids are:

| rule id | flags |
|---|---|
| `todo` / `fixme` | the word `TODO` / `FIXME` anywhere in a scanned file |
| `pass-body` | a function whose body, after an optional docstring, is only `pass` |
| `ellipsis-body` | a function whose body, after an optional docstring, is only `...` |
| `not-implemented` | any use of `NotImplementedError` |
| `sample-value` | a string constant whose whole value (stripped, case-insensitive) is one of `sample`, `example`, `dummy`, `placeholder`, `foo`, `bar`, `baz`, `changeme`, `tbd`, `xxx`; or that contains `lorem ipsum` or `deadbeef`; or whose hex part (after an optional `<algo>:` prefix) is 8 or more copies of one character (for example `sha256:000…0`). Words inside longer text ("for example, …") are not flagged. |
| `mock-import` | `import mock`, `from mock import …`, `unittest.mock` in any import form, `pytest_mock` |

Note that W0c's own Accept runs this scanner over `scripts/laned/`, so the
scanner's source must not trip its own rules. For example, it can build the
literal words by concatenation.

## `unread_keys.py` (build rule 4)

`unread_keys.py --root R --config FILE [--config ...] --source DIR [--source ...]`

For every key of the `[lane_d]` and `[lane_d_ops]` tables in each config,
some `.py` file under a `--source` directory must read it. A read is one of
these uses of the key as a string constant:

- the slice of a Load-context subscript (`raw["k"]`);
- the first argument of `.get(`, `.pop(` or `.setdefault(`;
- the left operand of an `in` comparison (`"k" in raw`).

These don't count as reads: mentions in an `ALLOWED` tuple, a docstring or a
comment, and dict-literal keys. Each key that isn't read prints
`unread <table>.<key>` and the script exits 1. Other tables are ignored.

## `follower_patterns.py` (build rule 5)

`follower_patterns.py --root R --source DIR [...] --patterns FILE [...]`

**What counts as a lane D log call.** It is a call
`<recv>.info|warning|error|critical|exception(msg, ...)` in a `.py` file whose
repo-relative path contains `lane_d`, where the receiver's last name segment
contains `log` (case-insensitive). `debug` calls are excluded.

**The message must be literal and tagged.** The message must be a string
constant, an f-string or a `%`-format with a literal left side. Its literal
start must be an event tag matching `lane_d_[a-z0-9_]+`, for example
`log.info("lane_d_crowned hotkey=%s", hk)`. If it isn't, the script prints
`untagged <relpath>:<line>`. A non-literal message is also `untagged`.

**The tag must have a follower pattern.** It does if it occurs as a whole
word inside a string literal of a patterns file (cascade-ops
`cascade_ops/events.py` in production). Comments don't count. If no literal
has it, the script prints `uncovered <relpath>:<line> <tag>`.

The script exits 1 on any of these findings.

## `accept`

### `accept <WP> [--evidence FILE] [--root R]`

It runs these steps in order and stops at the first failure, printing
`FAIL <step>`:

| step | what it does |
|---|---|
| `invariant10` | the pairing check below |
| `pytest` | the pytest command over `tests/accept/<wp>/` |
| `stub_scan` | `stub_scan.py` over the WP's `PATHS` |
| `e2e` | `make laned-e2e`, only if the repo's Makefile defines that target (from W1.5 on) |
| `evidence` | only with `--evidence`, see below |

**The `evidence` step.**

- A `.json` file must exist, parse as a JSON object, have `"wp"` equal to
  `<WP>` and have `"passed": true`.
- A `.md` review file (RVn) fails if it has any open blocking line,
  `- [ ] BLOCKING ...`. Closed items, `- [x] BLOCKING ...`, are fine.

### Invariant 10 pairing convention

Look at every test function or method in `tests/accept/<wp>/**/test_*.py`
whose name contains `unchanged_when_unarmed`. Replace that substring with
`armed_nonempty_lane_d_effect`. A test with the resulting name must exist
somewhere in the same WP directory, in any file or class. Some examples:

- `test_receipt_unchanged_when_unarmed` pairs with
  `test_receipt_armed_nonempty_lane_d_effect`;
- `test_manifest_armed_nonempty_lane_d_effect` doesn't pair with it.

Each unpaired name is printed and the run fails with `FAIL invariant10`.
The check also runs in `--expect-fail` mode.

### `accept <WP> --expect-fail [--root R]`

This mode is for tests-only PRs. It runs the invariant 10 check and the WP's
tests, then passes only if all three of these hold:

- at least one test is collected;
- at least one test fails for a right reason;
- no failure or error has a wrong reason.

Passing tests are allowed. For example, an "unchanged when unarmed" test
legitimately passes on main.

Right reasons:

- an `AssertionError`;
- a `ModuleNotFoundError`, `ImportError` ("cannot import name") or
  `AttributeError` ("module 'x' has no attribute") for a module whose path
  (`a/b.py` or `a/b/`, for module `a.b`) is matched by the WP's `PATHS`.
  This also counts at collection time.

Wrong reasons:

- a syntax error;
- a fixture error;
- an import error or attribute error for anything not in `PATHS`;
- nothing collected;
- all tests passing.

### `accept <WP> --check-pr --base REF --head REF [--pr-wps WP,...] [--correction] [--root R]`

It diffs `git diff --name-only BASE...HEAD` (plan rev6.17, owner
2026-11-23: one PR per feature; tests + code together; PATHS ride along):

- ~~**Tests and other paths together.** If the diff touches `tests/accept/`
  and any other path, the check fails with `FAIL rule1` (rule 1).~~
  (struck by rev6.17)
- **Tests and code together (FEATURE rule, rev6.17).** The diff may change
  `tests/accept/<wp>/` (its `PATHS` included) of every labelled WP together
  with code. Every code path must match some labelled WP's scope: its `PATHS`
  on the base plus the entries this PR adds (a removed entry still counts
  from the base). Otherwise `FAIL paths` names the offending paths. It
  prints `PATHS-ADDED <WP> <entry>` / `PATHS-REMOVED <WP> <entry>`, and
  `MODE combined` for a WP whose tests dir is in the diff, else
  `MODE implementation`.
- **Weakened tests.** A combined PR that deletes a test module, removes a
  test (or asserting) function, has fewer assertions in a function or
  helper, adds skip/xfail markers, or removes a `PATHS` entry of a WP fails
  with `FAIL weakened` (one `weakened <WP>: ...` line each) unless the PR
  also carries the flag label `laned:tests-correction` (`--correction`).
- **Tests only.** If the diff only touches `tests/accept/<wp>/`, it prints
  `MODE tests-only` (unbuilt WP) or `MODE tests-correction` (built) and
  exits 0. With one label, touching another WP's test directory fails.
- **Implementation.** Otherwise every changed path must match the WP's
  `PATHS`. If they all do, it prints `MODE implementation` and exits 0. If
  not, it prints `FAIL paths` with the offending paths. `scripts/laned/` is
  outside every WP's `PATHS` except W0c's.
- **Several labels (`--pr-wps`).** Any number of `laned:<WP>` labels
  (rev6.17: no 3-WP cap, overlapping `PATHS` and in-batch Depends allowed;
  `scripts/laned/coupled_batches` is no longer needed). It prints
  `BATCH <WPs>`, `SCOPE <WP> <n> file(s)` and `WPMODE <WP> <mode>` per WP
  and the asked WP's `MODE`. Refusals: `batch-non-wp-label`,
  `batch-duplicate-label`, `batch-stray-tests`, `batch-unlabelled-tests`,
  `batch-no-tests` (a WP with no `PATHS` + test file at HEAD),
  `batch-out-of-scope`, `batch-label-without-changes`,
  `batch-paths-too-broad`, `batch-paths-foreign-tests`.

### `accept <WP> --combined --base REF --head REF [--correction] [--root R]`

The run for `MODE combined`. The WP's NEW tests (test functions absent on
the base) and CHANGED tests (AST differs, or a helper / conftest / fixture
in their closure changed), taken from HEAD, run against the base's code (a
temporary base worktree whose `tests/accept/<wp>/` is HEAD's), classified as
in `--expect-fail`:

- every new test must fail for a right reason; one that passes on the base
  proves nothing and fails the run, unless the WP is listed in
  `scripts/laned/verification_wps` at the base;
- no new or changed test may fail for a wrong reason;
- at least one must fail for a right reason (unless verification WP or
  `--correction`).

Then the full Accept must pass on HEAD. It prints `TESTS-TREE <sha>`,
`LEDGER-MODE combined` and `PASS <WP> combined`.

### `accept --gate <STAGE> [--root R]` (§17.2)

It verifies `evidence/gates/<stage lowercased>.json` in the working tree. The
file must exist and parse, its `"stage"` must equal `<STAGE>`, and its
`"checks"` must be a non-empty list in which every item has
`"passed": true`. A top-level `"passed": true` isn't enough on its own.

## `status` (§17)

`status --root R --ref REF --ci-results FILE --out DIR [--release-tag-glob GLOB] [--push REMOTE]`

`REF` defaults to `origin/main` in production. Every input is read from the
ref's tree (`git show REF:path`), never from the working tree. These are the
plan file, `tests/accept/*`, `evidence/*` and the release tags.

**Outputs.** It writes `DIR/docs/plans/STATUS.md` and
`DIR/status/laned_build.json`. With `--push REMOTE`, it commits both files to
branch `laned/build-state` on `REMOTE` without force, and never touches
`main`, locally or remotely. Without `--push`, nothing is pushed. In
production the workflow passes `origin`.

### Plan parsing (`docs/plans/LANE_D_PLAN.md` at REF)

- **WP rows:** any Markdown table whose first header cell is `WP` and which
  has a `Runs in` column. `Depends` and `Accept` are found by header name.
- **External WPs:** a WP is external if its `Runs in` cell contains the word
  `builder`, `pod`, `testnet` or `owner`.
- **Parked WPs:** rows under a `## ` heading containing `parked` are
  `parked`.
- **Decisions:** the table headed `| D | Status | Option chosen | Class | Decided |`.
- **Owner inputs:** the table headed `| OA | Status | Unblocks | Evidence file |`.
- **T-rows:** the first-column `T<n>` ids of tables headed `| # | Case | Expected |`.
  Their owners come from the table headed `| Row | WPs |`.

WP ids are matched as whole tokens, so `W0` never matches `W0c`.

### CI results (`--ci-results`, a mock of the GitHub Actions API)

```json
{"runs": [{"workflow": "laned-accept", "event": "push|pull_request",
           "mode": "accept|expect-fail|combined|...", "wp": "W1", "head_sha": "<sha>",
           "conclusion": "success|failure", "created_at": "<ISO 8601>",
           "pr": 123}]}
```

`pr` (rev6.17) is recorded on every `pull_request` row (`ci_ledger.py record
--pr N`): the row also tells the LABELLED L2 rule that PR N carried
`laned:<wp>`.

### Levels (each needs all the levels below it)

| level | condition |
|---|---|
| L0 | the WP is listed |
| L1 | `tests/accept/<wp>/PATHS` exists at REF **and** there is a successful `expect-fail` run for the WP (its sha need not be on main), **or** (rev6.17) the WP's last ledger row for PR N is a green `combined` run and the squash merge `(#N)` changes `tests/accept/<wp>/` |
| L2 | LABELLED rule (rev6.17): a squash merge `... (#N)` whose PR labels are known (`$LANED_PR_LABELS`, the workflow's `gh pr list --state merged --json number,labels`, plus ledger rows with `pr`) is the L2 commit iff PR N carried `laned:<WP>` and it changes at least one non-test path (outside `tests/`) in the WP's `PATHS`; a known PR without the label never is. Otherwise (no PR number, labels unknown) the git-only rule: an implementation commit on REF's first-parent history that descends from the commit that added the WP's `PATHS`, touches at least one path outside `tests/accept/`, and has every changed path in `PATHS`. The first such commit is "the L2 commit". |
| L3 | the latest `push`/`accept` run for the WP (ordered by the position of `head_sha` in REF's history, then by `created_at`) whose sha is the L2 commit or a later one on REF has `conclusion: success`. Runs on earlier commits are ignored. A later red run drops the WP back to L2. |
| L4 | it is a ci-only WP (plan §17.1: "n/a for ci-only WPs, which pass L4 automatically"), **or** its evidence file exists at REF and the evidence check passes. The evidence file is `--evidence <file>` from the Accept cell, else `evidence/<wp>.json`. So an accepted ci-only WP that no RV covers yet is at L4, not L3. |
| L5 | some `evidence/rv<n>.md` has a `Covers:` line naming the WP and no open `- [ ] BLOCKING <WP>` line |
| L6 | the L2 commit is an ancestor of a tag matching `--release-tag-glob` |
| L7 | `evidence/l7/<wp>.json` has `activation_block` (int), `monitor == "green"` and `era_range` `[start, end]` with `end >= start` |

### Side states (`side_state`, otherwise `null`)

- `parked`;
- `blocked:<reason>` from `evidence/blocked/<wp>.md`, whose first line is
  `Reason: <owner|external|dependency|Dnn>`;
- `blocked:Dnn` when the WP's `Depends` cell names a decision that is `open`
  with class `S`;
- `blocked:owner` when an `open` owner-input row's `Unblocks` cell names the
  WP and its evidence file is absent at REF;
- `blocked:dependency` when a WP named in `Depends` is below L2 and the WP
  itself is below L2.

### `status/laned_build.json`

```json
{
  "schema": "laned_build/1", "ref": "<REF as given>", "commit": "<sha of REF>",
  "generated_at": "<ISO 8601>",
  "wps": {"<WP>": {"level": "L0".."L7", "side_state": null | "parked" | "blocked:<reason>", ...}},
  "stages": {"S0": {"state": "complete|incomplete", ...}, ...},
  "milestones": {"Build-complete": "complete|incomplete", "Launch-ready": ..., "Live": ..., "Done": ...},
  "decisions": {"D1": {"status": "...", "option": "...", "class": "O|S|...", "decided": "...", "blocking": false}},
  "owner_inputs": {"OA1": {"status": "...", "evidence": "evidence/owner/oa1.json", "present": false}},
  "counters": {
    "t_rows": {"passed": 0, "total": 0},
    "invariants": {"covered": [1, 2], "total": 12},
    "orphans": ["T3"],
    "blocked": [{"reason": "owner", "count": 1, "wps": ["W2"]}]
  }
}
```

The `stages` keys are S0, S0a, S0b, S0c, S1, S2, S2b, S3, S4, S5, S6,
SB0–SB5, S7 and S8, with the §17.2 requirements encoded in `scripts/laned/`.
A stage is complete when its WP requirements hold **and** its gate evidence
passes the `--gate` check.

- **T-rows passed:** the T ids in `evidence/s3_matrix.json` `rows` with
  `"passed": true`.
- **Invariant n is covered:** a test named `test_inv<n>_*` exists in a WP
  that is at L3 or above.
- **Orphan:** a T-row with no owner row, or one that maps only to WP ids that
  aren't listed.
- **`blocked`:** one entry per `blocked:` reason. `parked` isn't counted, and
  `wps` is sorted.

`STATUS.md` has a table headed `| WP | Level | State |` with one row per WP:
`| <WP> | L<n> | <side_state or -> |`. It also has a line containing `T-rows`
and `<passed>/<total>`.

### Status fixture: why each expected level holds

| WP | expected | why |
|---|---|---|
| W0c | L7 | Clean implementation, green push runs after the merge (a red run before the merge is ignored), ci-only, covered by RV1 with only a closed BLOCKING item, before tag `cap-1`, L7 evidence |
| W0 | L3 | Runs in `ci + owner merge`, so it is external, and `evidence/w0.json` is absent (an uncommitted copy in the working tree is ignored) |
| W1 | L2 | At the `evidence` commit it was accepted and, being ci-only and not covered by any RV, at L4. A red push run on the later `docs` commit dropped it back to L2. On a branch off `evidence` that adds an RV2 covering W1, it is L5 if RV2's BLOCKING items for W1 are all closed and L4 if one is open |
| W2 | L1, `blocked:owner` | expect-fail passed, no implementation, and OA9 is open with no evidence |
| W3 | L0 | Its expect-fail run failed |
| W4 | L1 | Its implementation commit also edited `README.md`, which isn't in its `PATHS` |
| W5 | L0, `blocked:dependency` | It depends on W2, which is at L1 |
| W6 | L0, `blocked:D99` | It depends on the open structural D99 (`null` in the D99-decided variant) |
| W7 | L1, `blocked:external` | `evidence/blocked/w7.md` |
| R6x | L4 | External evidence `evidence/s0a_diff.json` passes, but RV1 has an open BLOCKING R6x |
| R2x | L3 | Its evidence has `passed: false` |
| WB1 | L0, `parked` | It sits under the `(parked)` §12 heading |

## Workflow `.github/workflows/laned-accept.yml`

It must trigger on `pull_request` with `types` including `labeled` and
`synchronize`, and on `push` to `main`. It must run `scripts/laned/accept`
with `--check-pr` and `--expect-fail`, select WPs from `laned:<WP>` labels,
and run `scripts/laned/status` on push to main.

Rev6.17: `MODE combined` runs `accept <WP> --combined`; `laned:tests-correction`
is a flag label (never a WP) passed as `--correction`; every PR ledger row
carries `--pr`; both jobs write the merged PRs' labels to
`$LANED_PR_LABELS` (`gh pr list`, fails closed). The tooling still comes
from the base branch, so a PR that changes the rules runs under the old ones.

## Loop brief (orchestrator-facing, rev6.17)

- One PR per feature: put every WP the feature touches on one PR, one
  `laned:<WP>` label each, tests and code together, `PATHS` additions in the
  same PR. No 3-WP cap, no disjoint-`PATHS` or dependency split, no
  `coupled_batches` declaration.
- Each new test must fail on the base for a right reason and pass on the
  head (`--combined`); a test that already passes on the base belongs to a
  verification WP or is not new evidence.
- Never delete or loosen an existing assertion (or drop a `PATHS` entry)
  without the `laned:tests-correction` label and a reason in the PR body.
- Every WP whose code a PR changes must carry its label: an unlabelled WP
  gets no L2 from the merge, however broad its `PATHS`.
- Tests-only and tests-correction PRs still work; they are optional.
- Plan text changes: one plan PR per batch of decisions (no `laned:` label;
  `docs/plans/` is PROTECTED from every WP PR).
