# W7c acceptance tests: `lane_d_state.json` + receipt prior/posterior digest binding

W7c (plan rev6 §3.0 W7c row, depends on W7b) pins the lane D state file and
the digests a settlement's receipt binds (§1.5 "State": "Its digest is in the
receipt's prior and posterior digests (DEC-CA-0058), and the report digest is
bound into the receipt as the bench source is"), and invariant 5 / T11
("Restarts are idempotent. No re-score, no double crown."). It also closes the
RV5 W7b gaps left to W7c: (a) `verify_report` binds no boundary, so an old
report must be refused by the receipt; (b) sr25519 signatures are randomized,
so every bound digest hashes the canonical body, never signature bytes;
(c) duplicate JSON keys are refused, never last-wins.

These tests are the spec. The builder may touch only the paths in `PATHS`.
The W1.5, W5a–W5e, W7a and W7b tests must stay green.

Every state, report and receipt comes from real `settle_lane_d` runs on a
synthetic chain (numpy only, no torch): challenger X (sharp) and the genesis
house H0 (bad). boundary(1) is X's first clear (`confirm_pending`),
boundary(2) confirms and crowns X on a disjoint snapshot, boundary(3) judges
nothing. Whenever `settle_lane_d` takes the W5e gate keywords
(`boundary_block_hash`, `sentinel`, `bench`), the fixture passes them with a
passing bench session (an armed crowning without them fails closed, W5e
RV5). The bench pod is the one mocked external service. The signer is a
deterministic test keypair.

## Contract: `cascade/validator/lane_d_state.py`

### Canonical bytes and the digest

- `canonical_state(state) -> bytes` is
  `json.dumps(state, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
  allow_nan=False).encode("utf-8")`. A non-finite number raises `ValueError`.
- `state_digest(state) -> str` is the sha256 hex of `canonical_state(state)`.
  Key insertion order never changes it.
- The file is exactly `canonical_state(state)`: no indentation, no trailing
  newline. So `sha256(file bytes) == state_digest(state)`.
- The file carries `"schema": STATE_SCHEMA` (the value isn't pinned).

### Atomic write: `save_state(path, state)`

The bytes are written to a temp file **in the same directory**, fsynced, then
renamed onto `path` with `os.replace` (as `cascade.shared.atomic.
write_text_atomic` does). A crash at the rename leaves the previous file
byte-identical.

### Strict load: `load_state(path, house, *, expect_sha256=None) -> dict`

- A missing file is `genesis(house)` here. `settle_lane_d` adds the
  lost-state check (below): genesis is only ever settled from when no prior
  settlement artefact exists.
- A path that exists but can't be read as a file (a directory, an `OSError`
  on read) is `state_unreadable`, never genesis.
- Every refusal is `LaneDStateError(ValueError)` with `.code` in
  `STATE_ERROR_CODES`:

  | code | when |
  |---|---|
  | `bad_json` | not UTF-8, not JSON, or not a JSON object (empty and truncated files included) |
  | `state_missing` | the state file (or a restore backup) is absent although a prior settlement was persisted (RV6 B3, below) |
  | `state_unreadable` | the path exists but can't be read as a file |
  | `duplicate_key` | any object, at any depth, repeats a key (checked while parsing, so `json.loads` never silently keeps the last one) |
  | `not_canonical` | the bytes aren't `canonical_state` of what they parse to (any re-indent, inner space, key reorder or trailing newline) |
  | `unknown_schema` | `schema` isn't `STATE_SCHEMA` |
  | `state_digest_mismatch` | `expect_sha256` is given and the file's digest differs (for a missing file: the genesis digest) |
  | `stale_boundary` | see `settle_lane_d` below |
  | `report_conflict` | see `settle_lane_d` below |

### The receipt record

`RECEIPT_SCHEMA` (the value isn't pinned). An armed settlement's receipt is
exactly these keys:

```json
{"schema": "<RECEIPT_SCHEMA>", "boundary": <int>,
 "prior_state_sha256": "<64 hex>", "posterior_state_sha256": "<64 hex>",
 "report_sha256": "<64 hex>"}
```

- `prior_state_sha256` is `state_digest` of the state **as loaded**, before
  anything is applied (genesis included). That's before `refresh_house`.
- `posterior_state_sha256` is `state_digest` of the state written, which is
  the sha256 of the file bytes.
- `report_sha256` is the sha256 of `lane_d_report.canonical_body(report)`.
  That's the body without `signature`, so a re-signed report still matches.
- `boundary` equals `report["boundary"]` and the state's `boundary`.

The state's `last_report` holds at least `boundary`, `rows_sha256`,
`report_sha256` and `prior_state_sha256`, and **no signature**. The
signature's hex never appears in the state file, so two runs of the same
settlement give byte-identical states.

### Verifiers

Both raise `LaneDReceiptError(ValueError)` with `.code` in
`RECEIPT_ERROR_CODES`, and return normally otherwise.

`verify_receipt(receipt, *, report, state=None)` checks, in this order:

1. structure (below);
2. `report["boundary"] == receipt["boundary"]`, else `boundary_mismatch`. A
   valid old report replayed at a later settlement is refused here;
3. the sha256 of `canonical_body(report)` equals `report_sha256`, else
   `report_mismatch` (an old report relabelled to the new boundary and
   re-signed; a different judgement at the same boundary);
4. when `state` is given, `state_digest(state) == posterior_state_sha256`,
   else `state_mismatch`.

`verify_receipt_chain(receipts)` checks the structure of every receipt first.
Then boundaries must strictly increase (`boundary_order`, covering reversed or
repeated receipts). Then every `prior_state_sha256` must equal the previous
receipt's `posterior_state_sha256` (`chain_break`). A dropped receipt (a
missing link) and a hand-edited state file between two settlements are both
`chain_break`.

Structure, per receipt:

| code | when |
|---|---|
| `not_object` | not a dict |
| `missing_field` | a key of the record is absent (`schema` included) |
| `unknown_field` | a key outside the record |
| `unknown_schema` | `schema` isn't `RECEIPT_SCHEMA` |
| `bad_value` | `boundary` isn't an `int` (`bool`, `float` and `str` refused), or a digest isn't 64 lower-case hex |

### Report files

- `report_path(state_path, boundary) -> Path` is where the settlement's
  signed report is persisted. It sits under `state_path`'s directory, so it's
  snapshotted with the state, and it differs per boundary.
- `load_report_file(path) -> dict` parses strictly. Duplicate keys at any
  depth raise `LaneDStateError("duplicate_key")`. Any layout of a clean report
  loads equal to it.

## Contract: `settle_lane_d` (`cascade/validator/lane_d.py`)

- `LaneDSettlement` gains `receipt`. It's `None` unarmed. Unarmed, the
  settlement also writes no file (invariant 1).
- The state is read through the strict `load_state`, so a defective file is
  refused with its code before anything is written.
- **Write order:** the signed report is written atomically to
  `report_path(state_path, boundary)` **before** the state file.
- **Recovery (state boundary < boundary, report file present):** this is a
  crash after the report landed and before the state did. The settlement is
  recomputed, and the recomputed canonical body must equal the persisted
  file's body. If it does, the file is kept byte-for-byte (its signature too,
  even a re-signed one), the persisted report is returned and the state is
  written. If the body differs, it's `LaneDStateError("report_conflict")`
  and nothing is written. If the file has duplicate keys, it's
  `duplicate_key`.
- **Re-call (state boundary == boundary):** the boundary is already settled.
  The persisted report is returned with the same `pay` and `receipt`, and
  nothing is re-scored or re-gated: no forecast is read and no bench call is
  made. No file changes. If the persisted report's body doesn't match the
  state's `last_report.report_sha256`, it's `report_conflict`.
- **Older boundary (state boundary > boundary):**
  `LaneDStateError("stale_boundary")`, and nothing changes.

### Lost state fails closed (RV6 B3, invariant 5)

A *persisted report* is a file `lane_d_report_<int>.json` in
`report_path(state_path, b).parent`. Every refusal below happens before any
forecast is read or bench call is made, and leaves the whole state directory
byte-identical (no report, no state, no crown, no pay).

- **Missing state.** The state file is absent and a persisted report exists
  at any boundary other than the one being settled:
  `LaneDStateError("state_missing")`. This holds whatever
  `expect_prior_sha256` says (an expected genesis digest never licenses it).
  The one exception is the README "Recovery" of the *first* settlement:
  the only persisted report is this boundary's, so genesis is the true prior.
- **Unreadable state.** Empty or truncated bytes are `bad_json`; a path that
  can't be read as a file is `state_unreadable`. Neither is genesis.
- **Rolled-back state.** The state's `boundary` is below a persisted report's
  boundary, and that report isn't the one this call recovers (state at b1,
  reports b1 and b2, settling b3): `state_digest_mismatch`. State at b1 with
  b2's report, settling b2, is the normal recovery path (it can't be told
  apart from a crash between the two writes) and ends on the clean run's
  bytes.
- **The caller's expected prior.** `settle_lane_d(..., expect_prior_sha256=
  None)`: when given, the digest of the state as loaded (the genesis digest
  only when genesis is allowed) must equal it, else `state_digest_mismatch`.
  W8/W14 thread the last published receipt's `posterior_state_sha256` here.
- **Genesis** is settled from only when no persisted report exists (an empty
  `<stem>_reports/` directory counts as none).

### Explicit restore: `restore_state(state_path, backup_path, *, expect_sha256) -> dict`

The only way back from `state_missing` (a rebuild from the report chain isn't
defined: reports don't carry `used_windows`, `frozen_set` or `confirm`). The
operator passes a backup (§10.8 snapshot) and the last published receipt's
`posterior_state_sha256`. In order:

1. the backup is read with the strict load (`state_missing` if absent,
   `state_unreadable`, `bad_json`, `duplicate_key`, `not_canonical`,
   `unknown_schema`);
2. the sha256 of the backup bytes equals `expect_sha256`, else
   `state_digest_mismatch`;
3. when the backup's `boundary` is > 0, its `last_report.report_sha256`
   equals the body digest of the persisted report at that boundary (and that
   file exists), else `report_conflict`.

On success the backup bytes are written atomically to `state_path` (byte for
byte) and the parsed state is returned; persisted reports are never touched.
On refusal nothing is written. The next settlement then continues the chain:
its receipt equals the clean run's, and its prior is the restored posterior.
An operator CLI wrapping it (e.g. `python -P -m cascade.validator.lane_d_state
restore --state <path> --backup <path> --expect-sha256 <hex>`) is the
implementer's choice and isn't pinned.
- **Determinism:** two runs of the same chain on the same pool bytes give
  byte-identical state files and equal receipts at every boundary, under any
  `PYTHONHASHSEED`. A crash at either rename, followed by recovery, ends on
  the clean run's state bytes and receipts. The crown happens once: KING_D is
  X, X isn't in `prior_kings`, and the next settlement's prior is the
  recovered posterior.

## Not pinned

- The values of `STATE_SCHEMA` and `RECEIPT_SCHEMA`.
- How a lane A `RoundReceipt` embeds the lane D receipt record. That wiring
  belongs to emission (W8) and the absent state (W14). W7c pins the record
  and its verifiers.
- The signature check of a report. That's W7b's `verify_report` with
  `pinned_report_signer`, and `verify_receipt` doesn't call it.
- What a re-call does when the persisted report file is deleted, and what
  `verify_receipt_chain([])` does.
- Whether `restore_state` refuses to overwrite an existing live state file,
  and what `expect_prior_sha256` does on a re-call at a settled boundary.
- Whether the lost-state check lives in `load_state` (e.g. behind a keyword)
  or in `settle_lane_d`; `load_state` on a fresh path with no reports stays
  genesis.
- Temp files left behind by a crash.
- An optional up-front refusal: a settlement given a hand-edited
  (still-canonical) state may raise `state_digest_mismatch` instead of
  producing a receipt that breaks the chain. Either is accepted.
- **The vault ledger (`laned_vault_ledger/1`, W18).** W7c does not produce
  it. The plan's W7c row is the state file and receipt binding only. The
  ledger's statuses (`submitted`, `judging`, `king`, ...) and per-entry
  `closed_era` aren't in `lane_d_state.json`, so mapping one onto the other
  belongs with the sweeper's live wiring, not here.

## Why each test fails on main

| test | failure on main |
|---|---|
| most of `test_w7c_state.py`, `test_w7c_receipt.py` and `test_w7c_restart.py` | `ImportError` for a new name in `lane_d_state.py` (in PATHS): `LaneDStateError`, `LaneDReceiptError`, `state_digest`, `canonical_state`, `report_path`, `load_report_file`, `verify_receipt`, `verify_receipt_chain`, `RECEIPT_SCHEMA`, `*_ERROR_CODES` |
| `test_state_file_is_exactly_the_canonical_bytes_of_the_state`, `test_save_state_writes_canonical_bytes_via_fsync_then_rename` | assert: `save_state` writes `indent=1` |
| `test_receipt_unchanged_when_unarmed`, `test_state_binds_the_report_body_and_the_prior_never_the_signature`, `test_inv5_t11_no_double_crown_across_repeated_restarts`, `test_inv5_receipts_are_independent_of_the_hash_seed` | assert: `LaneDSettlement` has no `receipt`, a re-call is refused, and `last_report` carries the randomized signature |
| `test_w7c_lost_state.py` (RV6 B3; missing/rolled-back state, expected prior) | assert: main settles from genesis (`DID NOT RAISE`; the reproduction shows `king_before` = the house and X re-judged) or rolls back silently |
| `test_w7c_lost_state.py` (directory at the state path) | `IsADirectoryError` instead of `LaneDStateError("state_unreadable")` |
| `test_w7c_lost_state.py` (`expect_prior_sha256`) | `TypeError`: `settle_lane_d` has no such keyword |
| `test_w7c_lost_state.py` (restore) | `ImportError`: no `restore_state` |
| both `test_error_codes_are_exported` / `test_rv6_b3_error_codes_include_the_lost_state_codes` | assert: `STATE_ERROR_CODES` lacks `state_missing`, `state_unreadable` |

`test_a_crash_at_the_rename_leaves_the_previous_state_intact` passes on main,
and so do the lost-state tests that pin what must keep working (genesis with
no artefacts, first-settlement recovery, recovery from b1 state with only
b2's report, empty/truncated state as `bad_json`).
The atomic write is already there, and W7c keeps it.
