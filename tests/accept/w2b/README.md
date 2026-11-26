# W2b acceptance tests: the `[lane_d_ops]` contract

W2b (plan rev6 §3.0 row W2b, §14.2 "main-validator-only operations") adds a
**non-consensus** `[lane_d_ops]` table: the lane D vault bucket, the `pool_d`
location, the eval host and the intake/dispatch queue policy. Only our main
validator and the lane D intake read it; satellites never do; it is never in a
receipt, a digest or an overlay. These tests are the spec; the builder may
touch only the paths in `PATHS`.

Conventions follow `tests/accept/w0c/README.md`: no PATHS module is imported at
module level (every `cascade.*` import is inside a function a test body
calls), so on main every failure is an `AssertionError` or a missing
`cascade.shared.config.lane_d_ops` (which is in `PATHS`). Nothing is mocked.
The receipt test reuses the real fixture `tests/unit/receipt_fixture.py`.

## Interface pinned

- Module **`cascade/shared/config/lane_d_ops.py`** defines a frozen dataclass
  `LaneDOpsConfig` (one attribute per key, same name; equality compares every
  key) and `lane_d_ops_config(cfg) -> LaneDOpsConfig`, which reads
  `cfg.raw["lane_d_ops"]` (absent table = every default) and `cfg.raw["storage"]`
  for the bucket cross-check, the same pattern as `[satellite]`
  (`cascade/shared/config/satellite.py`).
- A refusal is a `ValueError` (`TypeError`/`KeyError` also count) raised by
  `lane_d_ops_config` **or** by the chain loader; the tests accept either.
- The module is **not** re-exported from `cascade/shared/config/__init__.py`
  (not in `PATHS`) and `loader.py` does not import it at load time. The
  parsed chain config (`ChainConfig` minus `raw`) does not change with it.

## Keys and defaults (key names are W2b's choice)

| key | type | default | validation |
|---|---|---|---|
| `vault_bucket` | str | `"cascade-lane-d-vault"` (§1.2) | S3 bucket name `^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$`; never `[storage] manifest_bucket`, `logs_bucket` or `pool_bucket` |
| `vault_prefix` | str | `"lane-d"` (§11.1 `lane-d/<hotkey>/<digest>/`) | non-empty, no leading `/`, no `..` segment, no whitespace |
| `pool_d_bucket` | str | builder picks | bucket name as above; never a shared `[storage]` bucket; never equal to `vault_bucket` (the vault is dedicated) |
| `pool_d_prefix` | str | builder picks | prefix rule as `vault_prefix` |
| `eval_host` | str | `""` (unset until the S4 bring-up, D11) | `""` or a host / inventory name: no `@`, `:`-userinfo, scheme, `/`, whitespace or leading `-` |
| `queue_max_inflight` | int | builder picks, > 0 | int >= 1 (bools, floats, strings refused): forecast jobs in flight on the eval host |
| `queue_max_pending` | int | builder picks, > 0 | int >= 1: the intake's global backpressure cap on pending entries (advisory; the consensus caps stay in `[lane_d]`) |
| `queue_poll_seconds` | int | builder picks, > 0 | int >= 1: the ready-queue poll interval |

Exact refused/accepted values: `helpers.REFUSED` / `helpers.ACCEPTED`.
**Defaults assumed** (the plan states none): the builder picks the four
"builder picks" values and lists them in the PR.

**Unknown keys are refused**, naming the key. **Secret-shaped keys are refused
as such**: any key named `secret`/`token`/`password`/`*_secret`/`*_token`/
`*_key`/`*_password` (`helpers.SECRET_KEYS`) is refused whatever its value,
with a message that names the key and says `secret` or `credential`, and never
echoes the value. The table carries names and locations only; credentials are
env-only.

## Shipped files

Both `chain.toml` and `chain.testnet.toml` carry `[lane_d_ops]` with exactly the
eight keys, `eval_host = ""` (no live host in a public file). Mainnet's
`vault_bucket`/`vault_prefix` are the defaults. Testnet's vault location
(`vault_bucket`, `vault_prefix`) and `pool_d` location (`pool_d_bucket`,
`pool_d_prefix`) each differ from mainnet's (§10.3: separate stack, testnet
prefix).

## What each file pins

| file | pins |
|---|---|
| `test_w2b_keys.py` | module/class/function exist; per-key default; absent table = valid defaults; shipped files carry every key and nothing else; testnet locations differ; **mutation** per key; **round-trip** (load → dump the parsed fields as TOML → `load_chain_config` → equal) for two full non-default tables and for the shipped defaults; refusals; the shared-bucket and vault≠pool_d cross-checks; unknown and secret-shaped keys |
| `test_w2b_non_consensus.py` | every key at value A, at value B, the shipped table and no table give equal parsed configs (minus `raw`), equal `lane_d`, equal contract / numerics / v2 contract digests and `koth_params` digests (lane D unarmed and armed), and byte-identical scored and rejected receipts built by `tests/unit/receipt_fixture.py`; receipt/manifest modules never name the table |
| `test_w2b_overlay.py` | `lane_d_ops` not in `netconfig.ALLOWED`; `overlay_policy()["operator_only"]` lists `lane_d_ops` (and still `generation`); `check_overrides` refuses every key naming `[lane_d_ops]`; an overlay mixing `[lane_d]` and `[lane_d_ops]` is refused whole by `check_overrides` and `validate_overlay` while the `[lane_d]` half alone validates; R6 `dedrift.classify` reports `("operator-only", "operator_only_table")` |
| `test_w2b_satellite_imports.py` | **the Accept property**: the static import graph from `cascade/validator/satellite.py` never reaches the module (below); no reached module names it as a string; satellite.py and `config/satellite.py` never mention `lane_d_ops`; the config package and loader don't import it |
| `test_w2b_unread_keys.py` | `scripts/laned/unread_keys.py --source cascade/shared/config` passes for a table of every key and for both shipped files, and fails on a planted unread key |

## The satellite code path (static)

It is satellite.py's own imports, **module level and inside its functions**,
plus their parent packages, plus the **load-time** import closure of all of
them (function bodies and `if TYPE_CHECKING:` blocks of other modules are not
followed; a `from pkg import x` where `pkg/x` is a module counts as importing
it). `cascade/validator/__init__.py` imports `cascade.validator.loop`, so the
main validator is on this path at load time: **the main validator and any
module on that path may read `[lane_d_ops]` only through a function-local
import**, never a module-level one. The intake (W3) is a separate service.

## Invariants

None owned; no `unchanged_when_unarmed` names, so no invariant-10 pairing.
The non-consensus tests pass on the shipped config once the module exists;
on main they fail at the missing module.
