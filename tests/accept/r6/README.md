# R6 acceptance tests: the contract they pin

R6 is the S0a de-drift classifier (plan §3.0 R6 row, §5 row 2 "S0a De-drift",
§14.1, §14.2, §14.4 item 1). It reads a deployed `chain.toml` and the release
(`main`) `chain.toml`, classifies every `LIVE OVERRIDE` marker in the deployed
file as **main**, **overlay** or **operator-only**, and emits the main-PR diff
and the overlay body. These tests are the spec. The builder may touch only
the paths in `PATHS`. Conventions (`PATHS` format, directory naming, the
`accept` runner, expect-fail reasons) are W0c's (`tests/accept/w0c/README.md`).

R6 owns no T-rows and no invariants. R6x (the owner's execution against the
real deployed file) is a separate WP and is not tested here.

## Where it lives

- Module `cascade/laned/dedrift.py` (package `cascade/laned/`). Python 3.11;
  it may import the real `cascade.shared` code (it must, see the rules
  below) and the stdlib. No torch, no network, no chain, no storage.
- CLI: `python -m cascade.laned.dedrift ...`. A `cascade-laned dedrift`
  console alias is out of scope here (it needs `pyproject.toml`, which is not
  in `PATHS`).
- Python API: `classify(table: str, key: str) -> tuple[str, str]`, returning
  `(class, reason)` for one key. The CLI uses the same function.

## CLI

```
python -m cascade.laned.dedrift --deployed F --main G --out DIR --effective-block N
```

- `--deployed F`: the deployed `chain.toml` (read only; never modified).
- `--main G`: the release `chain.toml` the PR targets (read only).
- `--out DIR`: created if missing. Three files are written:
  `classification.json`, `main.diff`, `overlay.json`.
- `--effective-block N`: the overlay body's `effective_block`.
- Exit 0 on success. Non-zero when:
  - a marker cannot be attached to a key (see "Markers");
  - the overlay body fails `cascade.shared.netconfig.validate_overlay` on the
    patched main (for example `N` is not an era start; the message then
    contains `era start`, as `validate_overlay`'s does);
  - a file does not parse as TOML. Usage errors may use 2.

## Markers

A marker is a line of the deployed file containing the text `LIVE OVERRIDE`
inside a TOML comment (after `#`). Three shapes occur in the real file and
in the fixture:

1. Inline: `key = value   # LIVE OVERRIDE 2026-10-31: ...`.
2. Inline, mid-comment: `key = value   # ... Prior LIVE OVERRIDE ...`.
3. Own-line: a comment line with the marker, optionally followed by more
   comment or blank lines, then the `key = value` it applies to (the value
   may span lines, e.g. a multi-line array).

An inline marker attaches to the key on its line. An own-line marker attaches
to the next `key = value` line. If a table header or the end of the file
comes first, the marker is dangling and the run fails with a message
(stderr or stdout) containing `LIVE OVERRIDE`. The table is the most
recent `[table]` header (dotted name for nested tables). Values are always
taken from the parsed TOML (`tomllib`), never from the text.

## `classification.json`

```json
{
  "schema": "cascade-dedrift/1",
  "deployed_sha256": "<sha256 hex of the --deployed file bytes>",
  "main_sha256": "<sha256 hex of the --main file bytes>",
  "markers": [
    {"line": 276, "table": "round", "key": "blocked_hotkeys",
     "deployed": ["5Grw...", "5FHn..."], "main": null, "drift": true,
     "class": "overlay", "reason": "overlay_key"}
  ],
  "counts": {"main": 13, "overlay": 1, "operator-only": 7}
}
```

- Top-level keys are exactly these five. Per-marker keys are exactly these
  eight.
- `line` is the 1-based line of the marker (for an own-line marker, the
  comment line, not the key line). Markers are sorted by `line`.
- `deployed` is the value in the deployed file; `main` is the value in the
  main file, or `null` when main does not carry the key. Both are the TOML
  values as JSON.
- `drift` is `deployed != main` (a key missing from main is drift).
- `counts` counts every marker by class (drifting or not); all three keys
  are always present.
- No absolute path, timestamp or other run-dependent value appears in any
  output. **Same input bytes ⇒ byte-identical outputs**, wherever the inputs
  and `--out` live.

## Classification rules (first match wins)

Every rule is read from the real code, not a copy of it.

| # | When `(table, key)` is | class | reason |
|---|---|---|---|
| 1 | `table` is listed in `netconfig.overlay_policy()["operator_only"]` (today `generation`; `cascade/shared/config/generation.py` docstring: operator-side, never in `contract_digest`) | operator-only | `operator_only_table` |
| 2 | `table == "round"` and `key` is in `cascade.shared.config.reliability.DEFAULTS` (trainer-only recovery policy; `netconfig.py:118` "satellites can ignore their values") | operator-only | `trainer_only_policy` |
| 3 | `("eval", "gift_gate_data_dir")`: a path on the validator host (`config/scoring.py:43-47`) | operator-only | `host_path` |
| 4 | `("eval", "calib_horizons")`: log-only telemetry that never feeds the verdict (`config/scoring.py:56-66`) | operator-only | `log_only` |
| 5 | a decided block: a key `netconfig.effective_raw` writes from a resolved activation (`round.rolling_from_block`, `round.epoch_blocks`, `round.epoch_blocks_prev`, `round.epoch_activation_block`, `scoring.era_king_from_block`, `scoring.tenure_blocks_from_block`, `scoring.cohort_maxt_from_block`, `scoring.cohort_maxt_increment_from_block`, `scoring.forfeit_from_block`, `scoring.margin_v2_from_block`, `v2.prelaunch_burn_from_block`, `v2.image_decoupled_from_block`, `eval.shared_ladder_from_block`), a key in `decided.TYPED_KEY_BY_PREFIX`, or `v2.from_block` (the resolved `launch_at_utc` block, typed per the runbook) | main | `decided_block` |
| 6 | a pin in `cascade.shared.decided.PIN_KEYS` (e.g. `round.funded_pod_image`, `round.worker_code_fingerprint`) | main | `decided_pin` |
| 7 | `table` is in `netconfig.REFUSED_TABLES` (`training`, `manifest`, `storage`, `chain`, `netconfig`, `activation`) | main | `release_only_table` |
| 8 | `netconfig.check_overrides({table: {key: value}})` refuses it: the release-only tuple (`scoring.burn_uid`, `lane_b.remainder`, `netconfig.py:210`), a table not in `ALLOWED`, a key not in a table's allow-list (`netconfig.py:83-118`), a nested `v2` table | main | `release_only_key` |
| 9 | otherwise (overlay-settable per `ALLOWED`) | overlay | `overlay_key` |

Rules 5 and 6 come before the allow-list: decided blocks are typed in main
(§5 S0a; `decided.py` docstring), even where `ALLOWED` would let an overlay
set them. Rule 8 follows `ALLOWED` and the release-only tuple as they are
when the tool runs, so a later release that widens or narrows them changes
the classification without editing R6.

## `main.diff`

A unified diff of `chain.toml` with headers `--- a/chain.toml` and
`+++ b/chain.toml` that `git apply` applies cleanly to the `--main` file
(saved as `chain.toml`). Applying it gives a file whose parsed TOML equals
main's, except that every marker with `class == "main"` **and** `drift`
has its deployed value. Nothing else changes value. The patched file:

- loads with `cascade.shared.config.parse_chain_config`;
- contains no `LIVE OVERRIDE` text (the markers stay out of main).

With no main-class drift, `main.diff` is an empty file.

## `overlay.json`

A DEC-CA-0057 overlay body (`cascade-netconfig/2`), the output of
`netconfig.next_overlay(...)` with `previous=None` (so `seq` 1, `prev` null),
`effective_block` = `--effective-block`, `network` =
`netconfig.network_identity` of the main config, and `overrides` exactly
`{table: {key: deployed}}` for every marker with `class == "overlay"` **and**
`drift`. With none, `overrides` is `{}`. The body passes `check_body`, and
`validate_overlay(parse_chain_config(patched main), body)` succeeds. It never
contains a release-only table or key, or any operator-only or main-class key.

## The fixture

- `fixtures/main_chain.toml`: a frozen copy of the repo's `chain.toml` at
  main `714a823` (public; frozen so main's own S0a PR doesn't move the
  tests).
- `fixtures/deployed_chain.toml`: that copy with exactly 21 `LIVE OVERRIDE`
  markers on the 21 table/key pairs that carry one in the real deployed
  file, in the real marker shapes. **Values are neutral**: hotkeys are the
  public Substrate dev keys (Alice, Bob), and no value is copied from the
  deployed host.
- `fixtures/expected_classification.json`: the expected `markers` list.

| line | marker | class / reason | drift |
|---|---|---|---|
| 276 | `[round] blocked_hotkeys` (own-line, multi-line array; absent in main) | overlay / overlay_key | yes |
| 283 | `[round] ops_max_active_per_coldkey` (mid-comment) | operator-only / trainer_only_policy | no |
| 292, 297, 298, 317 | `[round] epoch_blocks`, `epoch_blocks_prev`, `epoch_activation_block`, `rolling_from_block` | main / decided_block | no |
| 366 | `[round] funded_pod_image` | main / decided_pin | no |
| 397, 437, 470 | `[round] funded_pod_skus`, `funded_max_price_per_hour` (mid-comment), `funded_operator_fallback` | main / release_only_key | yes |
| 642 | `[eval] gift_gate_data_dir` | operator-only / host_path | yes |
| 643 | `[eval] calib_horizons` (absent in main) | operator-only / log_only | yes |
| 705, 758, 763 | `[scoring] cohort_maxt_increment_from_block`, `era_king_from_block`, `tenure_blocks_from_block` | main / decided_block | no |
| 983 | `[storage] hf_backup_repo` | main / release_only_table | yes |
| 1156 | `[v2] from_block` (own-line block of 3 comment lines) | main / decided_block | no |
| 1553, 1555, 1558, 1566 | `[generation] providers`, `fallback_vcpus` (absent in main), `max_parallel_jobs`, `vm_image` | operator-only / operator_only_table | yes |

Tests also derive variants in `tmp_path`: a 22nd marker on the release-only
`[scoring] burn_uid` (must go to main, never the overlay), a dangling marker
(must fail), and deployed == main (no markers, empty diff, empty overrides).

## Running

```
nice -n 19 env PYTHONPATH= python -m pytest --noconftest -p no:cacheprovider -q tests/accept/r6/
```

from the repo root. The CLI is run as a subprocess with cwd = repo root, so
it imports this checkout's `cascade`. Nothing heavy: no torch, no models.

## Failure on main

`cascade/laned/` does not exist on main, so every CLI test fails its
`returncode == 0` assertion and the API tests fail with
`ModuleNotFoundError: No module named 'cascade.laned'`, a module under
`PATHS`. Both are right reasons under W0c's expect-fail rule.
