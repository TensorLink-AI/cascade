# W4 acceptance tests: the ingest-guard contract

W4 (plan rev6 §3.0) deepens the walking skeleton's ingest (W1.5) into the full
lane D ingest guards: presigned fetch, sha256, `verify_checkpoint`, an exact
file allowlist, the lane-D-only BF16 dtype path, `verify_weights_finite` and
the worker handshake (git head + package hash). Accept:
`scripts/laned/accept W4`, where the T5 malformed set is refused with exact
codes.

These tests are the spec. The builder may touch only the paths in `PATHS`.
W1.5's skeleton (`make laned-e2e`, `ingest_entry(storage, entry, dest_dir,
*, contract)`) must stay green.

## What is pinned

| Row / invariant | Test |
|---|---|
| T5: each malformed case refused at ingest with its exact `IngestCode` | `test_t5_malformed_refused_with_exact_code_storage[case]`, `..._presigned[case]`, `test_t5_non_finite_refused_with_exact_code[case]` (torch) |
| T5 positive control: the same fixture, unmodified, ingests | `test_t5_fixture_positive_control_ingests_on_both_transports` (torch) |
| Invariant 3: no miner bytes execute; only repo code imports; the allowlist is exact | every T5 case (planted marker never created, no module from the tmp dir in `sys.modules`); `test_inv3_malformed_set_refused_in_fresh_interpreter_without_torch` |
| Allowlist and size budget refuse before any fetch | the `manifest_only` / `no_weights` fetch policies of the T5 cases |
| sha256 / content addressing | `test_storage_stray_object_and_overwrite_refused`, `test_presigned_overwritten_object_refused`, `test_presigned_manifest_must_hash_to_digest`, `test_fetch_presigned_sha256_mismatch_refused` |
| Presigned GETs only, no credentials on pods (§1.2) | `test_fetch_presigned_needs_and_sends_no_credentials`, `test_presigned_forged_url_fails_closed`, `test_fetch_presigned_rejected_signature_fails_closed`, `test_fetch_presigned_refuses_non_http_urls` |
| Oversize refused before a full download | `test_fetch_presigned_oversize_declared_length_refused_before_download`, `test_fetch_presigned_unbounded_stream_stops_at_size`, T5 `oversize` |
| Lane D expectation = lane A's spec with BF16 | `test_lane_d_expectation_is_lane_a_spec_with_bf16`, `test_real_tiny_checkpoint_ingest_and_non_finite` (torch) |
| Invariant 1 pair: lane A's F32 guard unchanged | `test_inv1_lane_a_f32_guard_unchanged_when_unarmed` + `test_inv1_lane_a_f32_guard_armed_nonempty_lane_d_effect` (torch) |
| Worker handshake, T19 (stale code ⇒ CRIT, nothing scored) | `test_t19_stale_worker_is_crit_and_nothing_scored`, `test_handshake_*`, `test_package_hash_tracks_source_bytes_only`, `test_local_identity_*`, `test_worker_hello_carries_the_identity` |
| T28: fails closed when mismatched, works in the documented order | `test_t28_documented_upgrade_order` |

## T5 cases

The fixture is a lane D checkpoint written directly, with no torch. It holds
the repo's real `cascade/trainer/pyramid_model.py` and `pyramid_wrapper.py`
bytes, a small `config.json` (`{"moe": {...}, "base_arch", "quantile_levels"}`)
and a BF16 `weights.safetensors` whose header is written by hand. The tests
check it against a matching `LaneDExpectation`. Each case changes exactly one
thing:

| case | change | exact code | fetch policy |
|---|---|---|---|
| `extra_py` | `evil.py` (would `os.mkdir(marker)` on import) | `file_not_allowed` | manifest only |
| `nested_py` | `pkg/__init__.py` (same payload) | `file_not_allowed` | manifest only |
| `torch_pt` | `model.pt`, a pickle that would `os.mkdir(marker)` on load | `file_not_allowed` | manifest only |
| `pickle_bin` | `pytorch_model.bin` (same pickle) | `file_not_allowed` | manifest only |
| `pickle_pkl` | `state.pkl` (same pickle) | `file_not_allowed` | manifest only |
| `edited_model_py` | `model.py` = payload + repo bytes | `code_mismatch` | — |
| `edited_wrapper_py` | `forecast_wrapper.py` = payload + repo bytes | `code_mismatch` | — |
| `wrong_config` | `moe.d_model` 16 → 32 | `config_mismatch` | — |
| `unparseable_config` | truncated JSON | `config_mismatch` | — |
| `pickle_as_weights` | `weights.safetensors` holds the pickle | `bad_header` | — |
| `bad_data_offsets` | one tensor's `data_offsets` disagree with dtype × shape | `bad_header` | — |
| `f32_dtype` | every tensor F32 (lane A's dtype) | `dtype_mismatch` | — |
| `wrong_shape` | `head.weight` (8,16) → (16,8), same bytes | `shape_mismatch` | — |
| `extra_tensor` / `missing_tensor` | a tensor added / removed | `tensor_set_mismatch` | — |
| `oversize` | weights padded past `max_weights_bytes` | `oversize` | weights never fetched |
| `nan_weights` / `inf_weights` (torch) | first BF16 element NaN / +Inf | `non_finite` | — |

The fetch policies:

- **Manifest only.** No object other than `manifest.json` is fetched: no
  storage `get` and no HTTP GET.
- **Weights never fetched.** `weights.safetensors` is not fetched, because the
  budget is checked against the manifest's declared size.

Every refused case also asserts:

- the marker was never created;
- no module was imported from the tmp dir;
- `dest_dir` is left empty;
- a log record at WARNING or above starts with `lane_d_ingest_refused` and
  contains the code value.

## Module contract

### `cascade/eval/lane_d_ingest.py`

- **`IngestCode`** is an `enum.StrEnum` holding at least these members and
  values: `DIGEST_MISMATCH`, `FILE_NOT_ALLOWED`, `STRAY_OBJECT`,
  `MISSING_OBJECT`, `SIZE_MISMATCH`, `MISSING_MANIFEST`, `BAD_MANIFEST`,
  `BAD_IDENTITY`, `OVERSIZE`, `CODE_MISMATCH`, `CONFIG_MISMATCH`,
  `BAD_HEADER`, `DTYPE_MISMATCH`, `SHAPE_MISMATCH`, `TENSOR_SET_MISMATCH`,
  `NON_FINITE` and `FETCH_FAILED`. Each value is the member's name in lower
  case.
- **`INGEST_CODES`** is the tuple of all `IngestCode` values.
- **`IngestError(code, detail="")`** has `.code`, an `IngestCode` member.
  Every refusal is an `IngestError`. W1.5's `checkpoint_tampered` and
  `checkpoint_diverged` are split into the specific codes above.
- **`LaneDExpectation(*, code, config, tensors, max_weights_bytes)`** is what
  an entry must match:
  - `code`: `{"model.py": bytes, "forecast_wrapper.py": bytes}`;
  - `config`: the `"moe"` object `config.json` must equal;
  - `tensors`: `{name: (dtype, shape tuple)}`;
  - `max_weights_bytes`: an int.
- **`expectation_for(contract, *, dtype="BF16") -> LaneDExpectation`**
  (needs torch) is the contract's expectation:
  - `code` is `expected_checkpoint_code("cascade_pyramid")`;
  - `config` is the contract's `CascadeMoEConfig.from_contract(...).to_dict()`;
  - `tensors` is lane A's pinned state spec with every F32 replaced by
    `dtype`;
  - `max_weights_bytes` lies in `[F32 file size, 3 × F32 file size]`, so lane
    A's F32 is refused as `dtype_mismatch`, not `oversize`.
- **`ingest_entry(storage, entry, dest_dir, *, contract, expect=None)`** is
  W1.5's storage path. `ingest_presigned(grant, dest_dir, *, contract,
  expect=None)` is the pod path. `grant = {"hotkey", "digest",
  "manifest_url", "files": {name: url}}`, where every URL is a presigned GET.
  - Both return `dest_dir/<digest>`, holding exactly the allowlisted files
    byte for byte.
  - With `expect` given, `contract` is unused and may be `None`. With `expect`
    `None`, they use `expectation_for(contract)`.
- **Guard order**, failing closed at the first problem:
  1. The manifest hashes to the digest (`digest_mismatch`).
  2. The manifest's names are exactly the allowlist (`file_not_allowed`).
     Storage only: no stray object exists under the prefix (`stray_object`).
  3. The declared weights size is within `max_weights_bytes` (`oversize`).
  4. Fetch: every object is size-capped at its manifest size and re-hashed
     (`oversize` / `size_mismatch` / `digest_mismatch` / `fetch_failed`).
  5. The code bytes match exactly (`code_mismatch`).
  6. The config matches (`config_mismatch`).
  7. The safetensors header is checked: parse and offsets (`bad_header`), the
     tensor name set (`tensor_set_mismatch`), dtype (`dtype_mismatch`), shape
     (`shape_mismatch`).
  8. The finite check, `verify_weights_finite` or an equivalent
     (`non_finite`).
- **No torch before step 8.** Steps 1–7 import no torch. A fresh interpreter
  that refuses the whole non-torch T5 set must not have `torch` in
  `sys.modules`, so the production box can run them. Nothing from the
  checkpoint is ever imported, unpickled or executed.

### `cascade/eval/checkpoint_guard.py`

- Lane A's `verify_checkpoint(dir, contract)` is unchanged: F32 only, the same
  `CheckpointReport`, and checkpoint bytes untouched.
- A BF16 checkpoint is still refused by lane A with `CheckpointTampered`.
- The BF16 path is lane D only.

### `cascade/lane_d/storage.py`

**`fetch_presigned(url, dest, *, sha256, size, timeout_s=...) -> Path`**
streams one presigned GET to `dest` (creating parents) with no
`Authorization` header. It reads no credential: the URL alone suffices, and
no credential env value is ever sent. Failures raise **`FetchError`**, whose
`.code` is one of:

| code | when |
|---|---|
| `oversize` | a `Content-Length` over `size` (before reading the body), or more than `size` bytes streamed (stop reading at once) |
| `size_mismatch` | the body is shorter than `size` |
| `digest_mismatch` | the sha256 is wrong |
| `fetch_failed` | an HTTP error (403 included), a non-`http(s)` URL, or a timeout |

On any failure, neither `dest` nor a partial file is left in its directory.
The package stays torch-free.

### `cascade/eval/lane_d_handshake.py` (new)

- **`HANDSHAKE_SCHEMA`** is `"lane-d-handshake/1"`.
- **`CodeIdentity(git_head, package_hash)`** is a frozen dataclass.
- **`package_hash(root) -> str`** is 64 hex over the source files under
  `<root>/cascade/`. `root` is the directory that contains the package. The
  hash depends only on relative paths and bytes:
  - not on location or mtimes;
  - not on `__pycache__` or `.pyc` files;
  - any `.py` edit, addition or rename changes it.
- **`local_identity(root=None)`** returns a `CodeIdentity`. `root` defaults to
  the directory containing the imported `cascade` package. `git_head` is the
  full 40-hex `HEAD` when `root` is itself the top of a git checkout, with no
  `+dirty` suffix (the package hash catches edits). Otherwise it is `""`.
- **`worker_hello(root=None)`** returns `{"schema", "git_head",
  "package_hash"}`.
- **`check_handshake(hello, *, expected=None) -> CodeIdentity`** compares the
  hello with `expected` (default `local_identity()`).
  - On a match it logs INFO `lane_d_handshake_ok ...` and returns the
    identity.
  - Otherwise it logs CRITICAL `lane_d_handshake_mismatch ...`, naming the
    fields, and raises `HandshakeMismatch` (a `RuntimeError`) whose `.fields`
    is the sorted tuple of mismatching fields among `git_head`,
    `package_hash` and `schema`.
  - It fails closed in every unclear case: an empty `git_head` or
    `package_hash` on either side is a mismatch on that field, and so is a
    non-dict, a wrong schema or a missing key.
- **`with_handshake(hello, work, *, expected=None)`** checks first. It calls
  `work()` only on a match and returns its result, so on a mismatch nothing
  is scored (T19).
- **T28** follows the documented order (§10.5 S5: eval host, then the
  validator). Every mixed step fails closed, the end state works, and the
  reverse order also fails closed until both sides match.
- The `lane_d_handshake_*` tags need cascade-ops follower patterns (W11, build
  rule 5).

## Fixtures and mocks

- The only mocked service is the vault bucket. `helpers.VaultServer` is a
  localhost presigned-URL server: an HMAC signature in the query, 403 on a
  bad one. It records every request and the bytes it sends.
- The storage path uses the real `FilesystemStorage`, subclassed only to
  record `get` calls.
- Proxy env vars are cleared for localhost.
- Credential env vars are only ever deleted or set to sentinel strings.

## Failure on main and the `torch` marker

- On main, every test fails inside its body. The failure is a `cannot import
  name` from `cascade.eval.lane_d_ingest` / `cascade.lane_d.storage`, or `No
  module named cascade.eval.lane_d_handshake`.
- `test_inv1_lane_a_f32_guard_unchanged_when_unarmed` passes on main (lane A
  is already unchanged).
- `test_w4_torch.py` is marked `torch`. The production box must never run
  it: run `-m "not torch"` locally (also inside `LANED_PYTEST_CMD`). CI
  installs torch.
