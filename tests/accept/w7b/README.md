# W7b acceptance tests: lane D report signing

W7b (plan rev6 §3.0 W7b row: "Report signing", depends on W7a) pins how the
main validator signs the `lane-d-report/1` report and how a satellite
verifies it (§1.5: "the main validator signs the lane D report"; "Satellites
... check the signature, the structural rules ..."; §14.3 items 1 and 3).
These tests are the spec. The builder may touch only the paths in `PATHS`.
W1.5's skeleton and W7a's schema tests must stay green, with the one
exception described under "Conflict with W1.5" below.

Every case starts from a real report that `settle_lane_d` produced and signed
on the synthetic chain (no torch). The signing key is a fresh sr25519 test
keypair whose secret lives only in the test's `tmp_path`. It stands in for
the main validator hotkey. No real wallet is ever touched.

## Contract: `cascade/validator/lane_d_report.py`

### Canonical body: the exact signed bytes

`canonical_body(report) -> bytes` is the report **minus its `signature`
field**, serialised as
`json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
allow_nan=False).encode("utf-8")`.

- This is the MarketCast report pattern (`cascade/shared/marketcast_report.py`
  `canonical_body`).
- `signer` is part of the body, so the stated signer is itself signed.
- Every other top-level field is in the body, including `schema`,
  `scoring_method`, `params`, `rows` and `rows_sha256`, and so is the
  optional `extensions` map. `extensions` is never consensus, but it is still
  signed.
- A non-finite number makes `canonical_body` raise `ValueError`.
- No domain prefix is added: `schema` (`"lane-d-report/1"`) is inside the
  body, and that already separates a lane D report from any other artifact
  the same hotkey signs.

### Signature format and version

The schema is closed, and W7a pins its top-level field set exactly. So there
is **no separate signature-version field**. The version is the report's
`schema`: `lane-d-report/1` means the scheme below. Any change to the scheme
is a schema bump (§14.3 item 3), and an unknown schema is refused before
anything else.

- `SIGNATURE_SCHEME == "sr25519"`: schnorrkel sr25519 with the substrate
  signing context, through `cascade.shared.sr25519` (the repo's single
  verifier, `verify_hotkey`).
- `SIGNATURE_PATTERN == r"^[0-9a-f]{128}$"`: the 64 signature bytes as
  lower-case hex, with no `0x`, no whitespace and no upper case.
- The schema file states the same encoding:
  `properties.signature.pattern == SIGNATURE_PATTERN`. So `validate_report`
  refuses any other encoding with `bad_value` at `signature`.
- `verify_report` refuses every other encoding too, even when it decodes to
  the genuine signature bytes. Without this rule, `bytes.fromhex` accepts
  upper case and whitespace, so one signature would have many valid
  spellings in a signed artifact.

### `sign_report(report, signer) -> dict`

- `signer` is a keypair, or a wallet whose `hotkey` is one. In production it
  is the main validator hotkey. The tests pass in a tmp keypair.
- It returns a copy and never mutates its input. Any old `signature` is
  dropped.
- It stamps `signer = <ss58>`, signs `canonical_body` of the stamped copy and
  stores the signature in `SIGNATURE_PATTERN` form.

### `verify_report(report, signer_ss58) -> bool`

It returns `True` only when all of these hold:

1. `report` is a dict that passes `validate_report`. This includes the
   signature encoding and `rows_sha256`.
2. `signer_ss58` is non-empty and `report["signer"] == signer_ss58`.
3. `signature` is the sr25519 signature of `signer_ss58` over
   `canonical_body(report)`.

Every other input returns `False`. That includes a non-dict, a missing,
`null`, empty or non-string signature, a NaN anywhere, the wrong signer, any
field mutation, a signature over a non-canonical encoding of the body
(default separators, indentation, unsorted keys, a trailing newline, or a
body that still carries a `signature` key) and a report that fails
`validate_report`. It never raises for these, and it never mutates its input.

**Order: validate first, then verify.** A report failing `validate_report`
returns `False` without reaching the sr25519 verifier. So it is refused even
when the verifier is unavailable. A *valid* report whose signature can't be
checked still raises `SignatureVerifierUnavailable` (DEC-CA-0051: never a
silent `False`). `validate_report` itself stays structural only. Both refuse
a schema-invalid report: `validate_report` raises, and `verify_report`
returns `False`.

JSON round trips verify. A report dumped with any key order or indentation
and loaded back still verifies, because the body is re-canonicalised.

### Pinned signer: `pinned_report_signer(cfg) -> str`

- It returns `cfg.manifest.validator_hotkey` (`chain.toml [manifest]
  validator_hotkey`, the main validator receipt signer), stripped.
- An empty or blank pin raises `ValueError`. Unlike the receipt audit, there
  is no fallback to the report's self-declared `signer`.
- Callers verify with `verify_report(report, pinned_report_signer(cfg))`. A
  report that another key signed is refused, even when that signature is
  valid, and even when its `signer` field is rewritten to the pinned hotkey.

## Conflict with W1.5 (needs a W1.5 tests-correction)

`tests/accept/w1_5/test_w1_5_e2e.py::test_e2e_tampered_row_fails` asserts
that `verify_report(resigned, ...)` is **true** for a report whose row was
tampered and then re-signed. That report's `rows_sha256` no longer matches
its rows. W7b's contract (validate before verify) makes that call return
`False`. The test's real claim, `check_lane_d(...) == FAIL`, is unaffected.

An implementation PR can't touch `tests/accept/` (rule 1). So the fix is a
W1.5 tests-correction PR that changes that one line to
`assert not verify_report(resigned, signer.ss58_address)`. It has to be
sequenced with the W7b implementation. The tests-correction mode allows the
changed file to fail for a right reason until W7b lands.

## Why each test fails on main

| test | failure on main |
|---|---|
| `test_signature_format_constants_are_exported`, all of `test_w7b_pinned_signer.py` | `ImportError` for a new name in `lane_d_report.py` (in PATHS) |
| `test_schema_file_pins_the_signature_format` | assert: the schema file has no `signature.pattern` |
| `test_validate_report_refuses_a_non_canonical_signature_encoding[*]` | assert: `validate_report` accepts any non-empty string |
| `test_non_canonical_signature_encoding_is_refused[upper_case, leading_space, trailing_newline, spaced_bytes]` | assert: `bytes.fromhex` accepts them, so `verify_report` returns `True` |
| `test_signed_report_failing_validate_report_is_refused[*]` | assert: `verify_report` checks only the signature |
| `test_validate_runs_before_the_signature_check` | assert: the invalid report reaches the verifier, which raises |

The other tests pass on main. They pin behaviour W7b must keep: the
canonical body bytes, the mutation and wrong-signer refusals, the
non-canonical body refusals and the round trips.
