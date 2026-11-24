# WB24 acceptance tests: the contract they pin

WB24 (plan §3.0, lane B track, rev6.3) is the **lane B miner CLI**. It runs
in `ci`. The builder may touch only the paths in `PATHS`:

- `cascade/miner/lane_b.py` (the `lane-b` subcommands)
- `cascade/miner/submission_status.py` (the shared lane status view)
- `cascade/miner/funding.py` (the shared signing and pre-send checks)

The docs (`docs/MINER.md`, `docs/AGENT_MINING.md`, `docs/INTERFACE.md`) are
WB20's PATHS and are not part of WB24. Updating the "hand-built request"
text is a separate docs change.

## How the tests run

- **In process:** `cascade.miner.cli.main(argv)` with `--json` for the
  machine contract (`cascade-cli/1`: `ok`, `exit_code`, `data`,
  `error.code`) and without it for the human lines. A `SystemExit` from
  argparse (an unknown flag or subcommand on main) is returned as exit 2 with
  `error.code == "usage"`. Tests that pin a refusal require a code other than
  `usage`, so an unknown flag never passes as a refusal.
- **No network.** A socket guard refuses every connection except to the
  loopback fake the test started, and records every attempt. The fake intake
  is either a recorder (`FakeIntake`, which records each request exactly as
  received and answers canned JSON) or the real `FundingIntake` behind it
  (`intake.real`). With the real intake, the CLI's request is checked by the
  intake's own verifier: real lane B queue, `require_signature=True`, a
  stand-in sr25519 verifier, reveal oracle `700`.
- **Wallet and keys.** `cascade.shared.chain.open_wallet` is a stand-in
  whose hotkey signs deterministically (`sha256(hotkey + message)`) and
  records every message it signs. No keyfile is read. The provider key is the
  fake `FAKE_KEY` in `$WB24_LIUM_KEY`. `CASCADE_NO_DOTENV=1` is set, so no
  `.env` is loaded.
- **chain.toml.** The tests use copies of the shipped file with edits to
  `[lane_b]` (pin, `suite_targets`, clamp fractions, `contamination_min_hits`)
  and `[storage]` (status tests only: `s3_endpoint` points at the loopback
  recorder, `manifest_bucket = "wb24-manifests"`).
  `cascade.shared.config.loader.DEFAULT_CHAIN_TOML` is redirected to a copy,
  so a command that loads the default never reads a deployed file.
- **Torch** is blocked while the tests run (the miner CLI is torch-free).
  `experts_per_layer` is read from `pyramid_model.py` by AST.

## `lane-b fund --declared-rows <file>` (WB14)

| test | pins |
|---|---|
| `test_declared_rows_round_trip_through_intake` | a valid file, written in a non-canonical form (unsorted keys and rows, whitespace), is sent as `X-Lane-B-Declared-Rows: canonical_declared_rows(parse_declared_rows(file))`. The real intake answers 202 and stores exactly those parsed rows on the queue entry. The hotkey signs `canonical_fund_message("lane-b-fund", hotkey, lane_b_signed_ref(targets, ref, rows), ts, sha256(key))`, the message with the `\|rows=<sha256>` tag that the intake verifies. No key is printed |
| `test_declared_rows_signature_binds_the_rows` | the captured request, with the rows header swapped for another valid declaration, gets `400 bad_signature` at a fresh intake. Unmodified, it gets 202 |
| `test_declared_rows_refused_locally_with_intake_codes[code]` | for each of the four `DECLARED_ROWS_CODES`, a file that the intake's parser and the real intake (`400 <code>`) both refuse with `code` makes the CLI exit 2 with `error.code == code`. Nothing is sent, nothing is queued and no key is printed |
| `test_declared_rows_clamp_read_from_chain_toml` | the local clamp is the `[lane_b] min_experts_frac` / `max_experts_frac` of `--chain-toml`. Rows that are valid under the shipped fractions are refused as `declared_rows_count` when `min_experts_frac` is doubled |
| `test_declared_rows_unreadable_file_refused_locally` | a missing file gives exit 2 with a typed code other than `usage`. Nothing is sent |

## `lane-b fund --target suite:<s>` (WB18)

| test | pins |
|---|---|
| `test_target_is_an_explicit_flag` | `--target` is an option of `lane-b fund` in its own right, and `--targets` stays. On main, `--target` reaches `--targets` only through argparse prefix abbreviation |
| `test_target_suite_round_trip_through_intake` | with `suite_targets = true` and a pin, `--target suite:gift` (no `--targets`) sends `X-Lane-B-Targets: suite:gift`. It signs `lane_b_signed_ref(("suite:gift",), ref)`, and the real intake queues `(("suite:gift",), ref)` |
| `test_target_suite_with_declared_rows` | `--target suite:boom` with `--declared-rows` is accepted end to end, and the rows are stored |
| `test_target_suite_refused_locally[...]` | `suite:nope` (armed) and `suite:gift` (suite targets off) give exit 2 with `bad_targets`. Nothing is sent |

The round-trip and refusal tests pass on main because argparse abbreviates
`--target` to `--targets`. They are regression guards for the explicit flag.

## `lane-b fund --train-hours` (WB23, not built)

WB23 (`train_hours` on the funding request) is not on main. The row says
`--train-hours` comes with WB23 "once built", so until then the flag is pinned
as **refused locally**. `test_train_hours_refused_locally_until_wb23` checks
that `--train-hours 2` exits 2 with a typed code other than `usage`, that the
message names `--train-hours`, that nothing is sent and that no key is
printed. When WB23 is built, this test changes in a tests-correction PR.

## Intake replies: the WB13 contamination codes

| test | pins |
|---|---|
| `test_fund_surfaces_contamination_refusals[status-code]` | each WB13 refusal (`400 benchmark_series_embedded`, `413 contamination_budget_exceeded`, `413 repo_too_large`, `503 contamination_hashset_unavailable` / `repo_unavailable` / `contamination_scan_failed`) gives a non-zero exit, `error.code == code`, and the code appears on stderr. This passes on main as a regression guard |
| `test_fund_shows_contamination_receipt[code]` | when the refusal carries the `contamination` receipt, the JSON document carries the receipt (some dict in it equals the receipt). The human output names the receipt's `result`, and its `window_id` when there is one. The canned message names neither |

## `lane-b status` (WB15, WB17, WB18 reasons)

`status/lane_b.json` is read from the configured public bucket
(`<[storage] s3_endpoint>/<manifest_bucket>/status/lane_b.json`, for example
through `cascade.miner.dashboard.fetch_public_json`). The intake queue rows
carry no reason codes. The deferral rows are built with the producers
(`row_claims.deferred_row`, `row_claims.sentinel_row`).

| test | pins |
|---|---|
| `test_status_shows_deferral_reason[rows_held\|sentinel_target]` | with `--hotkey H`, the JSON `data` contains a dict with `deferred == reason` and `hotkey == H`, and the human stdout names the reason. The document is fetched from the bucket path above |
| `test_status_shows_gate_closed_reason[reason]` | for each `gate_closed_reason` value (`tokens_short`, `king_check_not_passed`, `king_check_unavailable`), the JSON `data` contains a dict with `gate_closed_reason == reason`, and the human stdout names it |
| `test_status_shows_reasons_without_hotkey_filter` | without `--hotkey`, the gate reason and a `rows_held` row are still named |
| `test_status_output_unchanged_when_unarmed` | with no status document (404), the exit is 0, `data` keeps today's keys (`queued_depth`, `entries`, `as_of`, `submissions`, `freshness`) and no reason code is shown |
| `test_status_output_armed_nonempty_lane_d_effect` | with a document, today's keys stay and the reasons are shown |
| `test_status_open_gate_shows_no_gate_reason` | an empty `gate_closed_reason` shows no gate reason |

## `lane-b verify-repo <dir> [--chain-toml F]` (WB13, local)

The tests write a tiny seeded WB11 artefact as
`benchmarks/contamination/hashset-v1.json` under `tmp_path` and wrap
`cascade.interface.contamination_scan.load_pinned_hashset` so that a call
with the chain.toml pin loads it from there. They also wrap `scan_repo` to
record its tree. **Seam:** the command must look these two functions up on
the `contamination_scan` module when it runs, for example with a
function-local import (as the intake does), not through a name bound at
import time.

| test | pins |
|---|---|
| `test_verify_repo_clean` | exit 0, and `data.contamination` equals `scan_repo(dir, pinned set, min_hits=[lane_b] contamination_min_hits)`. `load_pinned_hashset` is called with the chain.toml pin and `scan_repo` with the given dir. The human output says `clean` |
| `test_verify_repo_embedded_refused` | exit 1, `error.code == "benchmark_series_embedded"` (the intake's code), `data.contamination` is the receipt, and the human output names the `window_id` |
| `test_verify_repo_budget_exceeded_refused` | a tree with an archive of more than 1024 members gives exit 1, `error.code == "contamination_budget_exceeded"` and the receipt |
| `test_verify_repo_uses_chain_toml_min_hits` | with `contamination_min_hits = {exact = 1000, normalised = 1000}`, the embedding repo scans clean (exit 0, same receipt) |
| `test_verify_repo_never_executes_the_repo` | every module of the tree (`generator.py`, `__init__.py`, `setup.py`, `conftest.py`, `sitecustomize.py`, `usercustomize.py`, `__main__.py`, `pkg/__init__.py`) writes a marker and raises at its top level, and a pickle sits beside them. After the command, no marker exists, no repo module is in `sys.modules`, `sys.path` and the cwd are unchanged, and the receipt equals the reference scan |
| `test_verify_repo_fails_closed_without_the_pinned_hashset[pin]` | unpinned (`""`), or a pin that no artefact matches, gives a non-zero exit, a typed code other than `usage`, and never a `clean` receipt |
| `test_verify_repo_not_a_directory` | exit 2 with a typed code other than `usage` |

In every verify-repo test, no connection is attempted (it never uploads), no
wallet is opened and the key is never printed.

## Never prints keys

Every fund, status and verify-repo test checks stdout and stderr for
`FAKE_KEY`. `test_fund_never_prints_key_even_if_intake_echoes_it` pins that
an intake reply whose message echoes the key is not printed verbatim, in
`--json` or in human mode.

## Unarmed defaults (golden from main)

`test_fund_request_bytes_unchanged_when_unarmed[fund|fund_gpu|withdraw]`
pins today's request bytes with a fixed `time.time` (`1763000000`), the
stand-in signer and the fake key. The digest is the sha256 of
`Recorded.canonical()`: method, path, body and every header except the
transport headers (`Host`, `User-Agent`, `Accept-Encoding`, `Connection`,
`Content-Length`), sorted by lower-cased name. The goldens were computed on
main at `30a0495`:

- `fund` (`--targets energy --label wb24-miner`): `6d364dcc…cecedb`
- `fund_gpu` (`--gpu 8xH100 --max-price 30`): `8f5ef514…fcbc8fa`
- `withdraw`: `d4214192…b8d605c`

The paired `test_fund_request_bytes_armed_nonempty_lane_d_effect` checks
that `--declared-rows` adds exactly `X-Lane-B-Declared-Rows` (the canonical
rows) and changes only `X-Signature`.

## Sanity tests (pass on main)

`test_codes_match_declared_rows_module`, `test_intake_contamination_codes_exist`,
`test_plan_row_names_the_flags`, `test_reason_codes_match_producers` and
`test_fixtures_scan_as_intended` (the clean, embedded, budget and exec-trap
fixtures give the intended results under the real scanner, and the trap
never fires).

## Failure on main (expected, for the right reason: assertions only)

`lane-b fund` has no `--declared-rows` or `--train-hours` and no explicit
`--target`, so argparse exits 2 with `usage`. `lane-b verify-repo` does not
exist (`usage`). `lane-b status` does not read `status/lane_b.json`, so no
reason is in `data` or in the output. A refusal's contamination receipt is
not carried, and an echoed key is printed. The golden, suite-target
(through abbreviation), contamination-code, unarmed-status and sanity tests
pass on main.

## rev6.22: repeatable `--target` (`test_wb24_multi_target.py`)

`lane-b fund --target T` may be repeated (one target per flag, any configured
target, suites mixed with domains/horizons); it sends the same slot-ordered
`X-Lane-B-Targets` as `--targets T1,T2`. More than `max_domains_per_leg` (4),
a repeat, a comma inside one flag, or an unknown target is refused locally as
`bad_targets` with nothing sent (one test with the four-target case, so it
fails on BASE, whose cap is 3 and whose `--target` takes one suite).
