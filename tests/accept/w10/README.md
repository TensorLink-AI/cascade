# W10 acceptance tests: the lane D miner CLI

W10 (plan rev6.4 §3.0 W10 row; §1.1 checkpoint format; §1.2 "Flow" steps
1–4 and D43; §10.2 "Miner kit"; §11.2 B13 licence attestation) builds
`cascade lane-d export | verify | submit | ready | status`. These tests are the
spec. The builder may touch only `PATHS`: `cascade/miner/lane_d.py` (new) and
the one registration line in `cascade/miner/cli.py` `build_parser`, plus its
module docstring.

The W10 row, verbatim:

> Miner CLI `cascade lane-d export | verify | submit | ready | status`, **W3a-chosen
> upload path only**. **(rev5, D43)** The commit reveals immediately
> (`blocks_until_reveal = min_blocks`), never at boundary − margin; a test
> asserts the lane D commit path never uses the lane A boundary timing

W10 depends on W3a (the decision record), W3 (the intake) and W4 (the ingest
guards). The builder starts only once all three are on main.

## Fakes (external services only)

- **Chain.** `ChainClient.from_config` is patched to return
  `helpers.FakeChain`. The CLI may use only `current_block()`,
  `wallet().hotkey` (sign, `ss58_address`), `hotkey_ss58()` and
  `commit_submission(payload, blocks_until_reveal)`. Reading
  `wallet().coldkey` or `coldkeypub` is recorded and fails the run. The
  CLI never touches the coldkey.
- **Intake and vault.** `helpers.FakeIntake` is a real HTTP server on
  127.0.0.1. It speaks W3's wire contract (`tests/accept/w3/README.md`
  "Ownership", "Submit", "Ready") and accepts the grant PUTs.
- **Network.** Every non-loopback `connect` in the test process is refused
  and recorded. A test fails if one was attempted.
- **Keys.** Deterministic `//w10-…` test keypairs, never a wallet on disk.

The CLI runs in-process (`cascade.miner.cli.main([... , "--json"])`) with
`CASCADE_NO_DOTENV=1` and cwd = `tmp_path`, so no `.env` is ever loaded.

## Upload path (parameterised on the W3a decision)

`decisions/STORAGE-LANE-D.md` names the one path on its `Chosen path:` line.
Its `| Path id |` table lists `presigned_multipart`, `single_presigned_put`
and `upload_relay`. **The tests are parameterised on the chosen id. None is
hardcoded.** `helpers.FakeIntake` mints grants for whichever id it is given,
and the upload tests run with `require_chosen_path()`. Rev6.4 expects
`presigned_multipart`, because Hippius has no Object Lock and the probe re-runs
`--no-object-lock`. That expectation is not assumed.

- `cascade.miner.lane_d.UPLOAD_PATH_IDS` equals the record's table, and
  `UPLOAD_PATH` equals the record's choice. While the record is unfilled,
  `test_the_cli_offers_only_the_decision_records_choice` and every upload
  test fail with that message. The builder records `evidence/blocked/w10.md`
  (`Reason: owner`, W3a) and never picks a path.
- `submit` sends `X-Upload-Path: <UPLOAD_PATH>`. If the answer's
  `upload_path` is any other id, the CLI refuses it locally with
  **`upload_path_not_offered`** (exit 1), and no PUT is sent.

**The grant format** is the consumer side of W3's minter output. W3 leaves
`grants` to the minter, and W10 pins it:

```json
{"files": {"<name>": {"requests": [
  {"method": "PUT", "url": "<presigned or relay URL>", "offset": 0, "length": 1024,
   "headers": {"<name>": "<value>"}}]}}}
```

- **`presigned_multipart`:** one request per part, in order.
- **`single_presigned_put`:** one request per file.
- **`upload_relay`:** one request per file, carrying a ticket header.

The CLI PUTs `file[offset:offset+length]` to `url` with the given headers.
Before sending anything it checks two things:

- the grant files are exactly the manifest's, and each file's requests cover
  `[0, size)` contiguously in order;
- every URL is `https`, or `http` to loopback.

Anything else is refused locally, **`bad_grants`** or **`insecure_grant`**
(exit 1), with no PUT and no connect. If W3's minter chose another shape, a
tests-correction aligns the two. The builder doesn't adapt silently.

## The command group

- `build_parser()` has a `lane-d` group. Its children are exactly `export`,
  `verify`, `submit`, `ready` and `status`, plus an optional shared `test`
  child like lanes B and C. Every child takes `--json`.
- `submit` and `ready` take `--wallet-name`, `--wallet-hotkey`,
  `--wallet-path`, `--network` and `--chain-toml`. `status` takes no wallet.
- No lane-d flag takes a secret, token, password, API key, presigned URL or
  grant (§1.2: never on a command line).
- **W9 invariant 12.** `cascade.audit.checks`' static import closure reaches
  `cascade.miner.cli` (via `validator.marketcast` → `trainer` →
  `miner.live_status` → `miner.cli`). So `cascade/miner/lane_d.py` must not
  statically import, even inside a function, any module W9 forbids:
  `cascade.lane_d*`, `cascade.eval.lane_d_*`, `cascade.validator.lane_d*`,
  `cascade.laned` or `torch`. It loads W3/W4 helpers at call time, for
  example with `importlib.import_module`. A scratch control that used
  `from ..lane_d.intake import …` broke
  `test_inv12_audit_import_closure_reaches_no_lane_d_producer_module`.

## `export <src> --out <dir>`

`src` is a trained cascade_pyramid checkpoint directory: `config.json` and
`weights.safetensors`, optionally code files and other training leftovers.

- `out` gets **exactly** the four allowed files. `model.py` and
  `forecast_wrapper.py` are the repo's `cascade/trainer/pyramid_model.py` and
  `pyramid_wrapper.py` bytes. `config.json` is the source's (JSON-equal).
  BF16 weights are copied byte-identical. Leftovers (`optimizer.pt`, notes)
  never reach `out`.
- The result `data` has `digest` (sha256 of the canonical manifest of
  `out`, `intake.manifest_digest`) and `files`.
- A non-empty `out`, or a source without weights, is refused with a typed
  code, and `out` is left as it was.
- A source `model.py` that is not the repo's bytes is **`code_mismatch`**
  (exit 1), and nothing is written.
- **torch:** F32 weights (lane A's dtype) are either converted, so every
  tensor in the written header is BF16 with the same names, or refused with
  nothing written (D14: BF16 only).

## `verify <dir>`

It runs W4's ingest guards locally, and the error codes are W4's
`IngestCode` values (exit 1).

- A disallowed file (`evil.py`, `pkg/__init__.py`, `model.pt`,
  `pytorch_model.bin`) is **`file_not_allowed`**. That is checked in a
  subprocess with **torch unimportable**, and the file's planted marker is
  never created: the allowlist runs before anything that needs torch, and
  nothing in the entry is imported or executed (invariant 3).
- A missing directory is a typed error.
- **torch:** an edited `model.py` is `code_mismatch`, and an unparseable
  config is an `IngestCode` refusal.

## `submit <intake> <dir> --context C`

1. **Local checks, before any request.** The directory's files are exactly
   `ALLOWED_FILES`, else **`file_not_allowed`** (extra `.py`, a pickle,
   missing weights). `C` is in the loaded `[lane_d] contexts`, else
   **`context_not_armed`**. Plain `http` to a non-loopback intake is refused,
   as lane A's `_intake_transport_ok` does.
2. **The signed submit, W3's contract.** The headers are `X-Miner-Hotkey`
   (the ChainClient wallet's hotkey), `X-Content-Digest` (the manifest
   digest), `X-Context-Length`, `X-Timestamp` (unix seconds, within ±300 s)
   and `X-Signature`: the hex sr25519 signature over
   `cascade-lane-d-submit:v1:<hotkey>:<digest>:<context>:<timestamp>:<sha256(body)>`.
   The body is the canonical manifest (`intake.canonical_manifest`).
3. **The upload** through the grants (above). The bytes the vault receives
   reassemble every file exactly.
4. **The result** `data` has `digest`, `hotkey`, `context`, `upload_path` and
   `uploaded_bytes` (the sum of the four sizes), and never `grants`. `submit`
   commits nothing.

Errors:

| case | exit, `error.code` |
|---|---|
| intake 4xx | 1, the intake's own `code` (e.g. `duplicate_weights`, `hotkey_entry_open`, `not_registered`), nothing uploaded |
| intake 5xx, intake unreachable | 3 |
| a grant PUT fails | 4, `upload_failed` |

## `ready <intake> --digest D --context C --licence L --attest`

- `L` is one of `cascade.miner.lane_d.LICENCES == ("MIT",)`. Any other value is refused before any request.
- Without `--attest`, nothing is sent and nothing committed
  (**`licence_attestation_missing`**). The statement is shown to the miner.
- The body is `{"licence": L, "attestation":
  cascade.miner.lane_d.LICENCE_ATTESTATION}`, signed like submit with
  action `ready`. The statement is at least 80 characters and names both
  licences, "training data" and "teacher". Once W3 is on main it equals
  `cascade.lane_d.service.LICENCE_ATTESTATION`. That test skips until then.
- **The commit payload is checked before signing anything on chain.** It
  must start `metro-v1:weights:c<C>:hippius:` and contain `D`. Any other
  payload is **`bad_commit_payload`** (exit 1) and is never committed.
- An intake refusal commits nothing: 4xx is exit 1 with the intake's code,
  5xx is exit 3. A `ChainError` from the commit is exit 3.
- The result `data` has `digest`, `coldkey_at_ready`, `ready_block`,
  `ready_expires_block`, `commit_payload`, `blocks_until_reveal` and
  `commit_block` (the `current_block()` at commit).

### The commit (D43)

- `ready` commits the returned `commit_payload` exactly once, through
  `commit_submission(payload, blocks_until_reveal=min_blocks)`. Here
  `min_blocks` is the default of `blocks_until_boundary_reveal`'s
  `min_blocks` (today 1, "reveal now"). That holds at every block, including
  inside lane A's reveal margin.
- During the tests, `cascade.shared.chain.blocks_until_boundary_reveal` and
  `cascade.miner.cli._resolve_blocks_until_reveal` are tripwires, and a call
  to either fails the test.
- Statically, `cascade/miner/lane_d.py` never names
  `blocks_until_boundary_reveal`, `_resolve_blocks_until_reveal`,
  `reveal_margin_blocks` or `next_epoch`.

## `status <intake> [--hotkey H]`

It sends `GET /v1/lane-d/queue` and nothing else. `data.entries` is the
intake's rows verbatim, or only `H`'s with `--hotkey`. No `ChainClient` is
built. Unreachable or 5xx is exit 3.

## Secrets (every flow)

stdout, stderr and every log record never contain any of these:

- a grant URL's signature (`X-Amz-Signature`, the fake's grant secret);
- a relay ticket;
- the request signatures;
- the hotkey's private key, seed or mnemonic;
- a refused grant host.

## Not pinned

- How the CLI streams a 2.5 GB file. T7c on testnet measures it, and the CLI
  must not hold the whole file in RAM.
- Retries.
- The `test` child.
- A dedicated-hotkey warning for a hotkey that already holds a lane A
  pointer.
- The human text.

## Invariants

None owned, no `unchanged_when_unarmed` names.

## Failure on main (66 collected: 3 torch-marked; of the 63 others, 62 fail, 1 passes)

| tests | on main |
|---|---|
| every test that runs the CLI or imports `cascade.miner.lane_d` | `ModuleNotFoundError: cascade.miner.lane_d` / `cannot import name 'lane_d' from 'cascade.miner'` (in PATHS) |
| the upload, ready and payload tests | assert: the decision record has no `Chosen path:` (W3a, owner). After the owner fills it: the module error above |
| the static tests and the torch-free `verify` subprocess tests | assert: `cascade/miner/lane_d.py` does not exist |
| `test_the_decision_record_lists_the_three_path_ids` | passes |

A scratch control passed every non-torch test: a minimal `lane_d.py` in a
throwaway worktree with the record filled as `presigned_multipart`. W9 and
W5g stayed green with it.
