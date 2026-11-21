# R10 acceptance tests: the contract they pin

R10 is the upgrade-from-mainnet-state harness for T25 (plan §3.0 R10 row,
§4 T25, §10.3 testnet stack, §10.8 snapshots and rollback). T25 says:
upgrade from a **copy of mainnet state** (`validator_state.db`,
`activation_*`, `netconfig_state`, `cascade_state.json`) and expect a
**clean start and an identical next verdict**.

The CI part of R10 is a CLI. These tests are its spec. The builder may touch
only the paths in `PATHS`: `cascade/laned/restore.py` and the shared package
marker `cascade/laned/__init__.py` (R1, R2 and R6 add their own modules to
the same package). The common conventions (PATHS syntax, exit codes, the
`--expect-fail` rules) are in `tests/accept/w0c/README.md`.

Every test drives the tool as a subprocess,
`python -P -m cascade.laned.restore <command> ...`, with `cwd` and
`PYTHONPATH` set to the repo root. Some tests also call the real state
loaders in-process (`cascade.validator.state.loads`,
`cascade.validator.cascade.load_state`, `cascade.shared.netconfig.load_state`,
`cascade.shared.activation.ActivationStore`) to cross-check what the tool
restored. Nothing is mocked. The fixture state is written in `tmp_path` by the
real writers (`state.dumps`, `cascade.dumps`, `NetConfigState.to_json`,
`ActivationStore.save`), so it has the real on-disk formats. The tests never
read real node state. The configs are the repo's own `chain.toml` (mainnet,
netuid 91) and `chain.testnet.toml` (testnet, netuid 259), loaded with the
real `load_chain_config`.

## What the state looks like (read from code)

All of the state files are JSON text, even `validator_state.db`. Each loader
reads `<file>`, falls back to `<file>.bak` when the main file is torn, and
raises (`ValidatorStateUnreadable`, `CascadeStateUnreadable`,
`NetConfigStateUnreadable`, `ActivationStateUnreadable`) when the file exists
but neither copy parses.

| role | source file name | where the target config puts it |
|---|---|---|
| `validator_state` | `[validator] state_db_path` (`validator_state.db`) | the target's `state_db_path` (`validator_state.testnet.db`) |
| `cascade_state` | `[validator] cascade_state_db_path` (`cascade_state.json`) | the target's `cascade_state_db_path` (`cascade_state.testnet.json`) |
| `warm_start_init` | `[validator] warm_start_init_path` (`warm_start_init.json`) | the target's `warm_start_init_path` (`warm_start_init.testnet.json`) |
| `netconfig_state` | `netconfig_state.json` beside `state_db_path` | `netconfig_state.json` beside the target's `state_db_path` |
| `activation` | every `activation_state.json` and `activation_*_state.json` beside `state_db_path` | same name beside the target's `state_db_path` |

`validator_state` is required. The other roles are copied when present. A
`<file>.bak` sibling of any of them travels with it and lands at
`<target path>.bak`. Relative config paths resolve against the node dir
(`--source` or `--target`).

## Commands

### `snapshot --source DIR --config TOML --out DIR [--block N]`

`--config` is the source node's config (mainnet). The command:

- creates one new bundle dir under `--out` whose name matches
  `state-YYYYMMDDTHHMMSSZ` (UTC);
- copies the files in the table above into the bundle, **and nothing else**.
  Key material (`coldkey`, `hotkeys/`, `wallets/`), `.env` files, configs
  (`chain.toml`), logs, the trainer submissions db and the settlement
  journal are never bundled;
- writes `manifest.json` in the bundle:

  ```json
  {"schema": "laned_state_bundle/1",
   "created_at": "<ISO 8601 UTC>",
   "source": {"dir": "<abs>", "netuid": 91, "network_id": "bittensor:finney",
              "block": 9876543},
   "files": [{"role": "validator_state", "source_name": "validator_state.db",
              "path": "<relative to the bundle dir>", "sha256": "<hex>",
              "bytes": 123}]}
  ```

  `source.block` is `--block` (`null` without it). `netuid` and `network_id`
  come from the source config;
- prints `BUNDLE <abs bundle dir> sha256=<hex>`, where the hex is the sha256
  of the `manifest.json` bytes. This is **the bundle sha256** that T25
  evidence records;
- never writes to `--source`;
- **refuses** (non-zero exit, message containing `settlement`, no bundle
  created) when `<state_db_path>.settlement.json` exists with
  `"phase": "prepared"`. That means a settlement is in flight, and §10.8
  forbids restoring state across a settlement. A `committed` journal is
  fine, and it isn't bundled either.

### `verify --bundle DIR`

Exits 0 only if every manifest entry's file exists with the listed sha256 and
the bundle holds no file the manifest doesn't list (other than
`manifest.json`). Otherwise it exits non-zero and names the offending file.

### `restore --bundle DIR --target DIR --config TOML [--force-target] [--start-cmd]`

`--config` is the target node's config (testnet).

- It verifies the bundle first. A bundle that doesn't verify is refused, and
  nothing is written to the target.
- It refuses a target dir that exists and isn't empty, unless
  `--force-target` is given. A missing target dir is created.
- It refuses a target config whose `[subnet] netuid` or
  `[netconfig] network_id` equals the bundle's source. A mainnet copy is never
  restored onto mainnet.
- It places each file where the target config says (the table above). **No
  byte of any state file is rewritten:** every restored file has the sha256
  that the manifest lists.
- It writes `laned_restore.json` in the target dir:

  ```json
  {"schema": "laned_state_restore/1",
   "bundle_sha256": "<sha256 of the bundle's manifest.json>",
   "source": {"netuid": 91, "network_id": "bittensor:finney", "block": 9876543},
   "target": {"dir": "<abs>", "netuid": 259, "network_id": "bittensor:test"},
   "files": [{"role": "...", "source_name": "...", "target_path": "<relative to target>",
              "sha256": "<hex>"}]}
  ```

  The target dir holds the restored files and `laned_restore.json`, and
  nothing else.
- With `--start-cmd` it also prints the release start command without running
  it, as two lines: `START_CWD <abs target dir>` and `START <command>`.
  `<command>` is a shell-quoted argv. It's either `<python> -P -m
  cascade.validator.main ...` or the `cascade-validator` entry point, and it
  carries `--chain-toml <abs config>` and `--network <net>`, where `<net>`
  is the part of the target `network_id` after `bittensor:` (`test`).
  Nothing is started.

**Which network-identity fields are rewritten.** Only the node's identity
changes: the file placement (the target config's file names) and the
`netuid` / `network_id` recorded in `laned_restore.json` (91 /
`bittensor:finney` to 259 / `bittensor:test`). The contents of the state
files are never rewritten:

- the adopted overlays in `netconfig_state.json` are owner-signed envelopes
  whose body binds `network: {chain, netuid}`, so rewriting them would break
  the signature and the digest chain;
- hotkeys are SS58 and the same on every chain;
- uids such as `king_uid` are restored as-is, since a T25 verdict is judged
  by hotkey;
- keys never enter a bundle.

### `check --target DIR --config TOML`

This runs before the release starts. It loads every restored file through the
real loaders, at the paths the target config names. It exits 0 and prints
`CLEAN START` only if all of these hold:

- `laned_restore.json` exists and every file it lists still has its sha256;
- the validator state is present;
- no loader raises;
- no loader resumed from a `.bak`, because a torn main file isn't a clean
  copy;
- `warm_start_init` (when present) parses as a JSON object, which is how the
  validator reads it.

Otherwise it exits non-zero and names the offending target file. On a `.bak`
fallback, the message contains `.bak`.

### `verify-evidence --evidence FILE [--bundle DIR]`

This checks the T25 testnet evidence (`evidence/t25.json`; `accept R10
--evidence evidence/t25.json` also needs `"wp": "R10"` and
`"passed": true`). It exits 0 only for:

```json
{"schema": "laned_t25/1", "wp": "R10", "row": "T25", "passed": true,
 "source_network_id": "bittensor:finney", "source_netuid": 91,
 "source_block": 9876543,
 "bundle_sha256": "<64 lowercase hex>",
 "target_node": "testnet-ctl", "target_netuid": 259,
 "release_head": "<40 lowercase hex git sha>",
 "clean_start": true,
 "first_verdict": {"round_id": "...", "outcome": "...", "king_hotkey": "..."},
 "mainnet_verdict": {"round_id": "...", "outcome": "...", "king_hotkey": "..."},
 "verdict_identical": true}
```

- Every key is required. `source_block` is an int > 0. `target_node` and the
  verdict fields are non-empty strings. `first_verdict.round_id` is the
  "first verdict id".
- `verdict_identical` must equal `first_verdict == mainnet_verdict`. The
  claim is cross-checked, never trusted.
- `passed` must be true, and `passed` true needs `clean_start` and
  `verdict_identical`.
- With `--bundle`, `bundle_sha256` must equal the sha256 of that bundle's
  `manifest.json`.

## Test files

| file | pins |
|---|---|
| `test_r10_snapshot.py` | the bundle layout, the manifest and its sha256 line, the allowlist (no keys, env, configs or logs), the `.bak` siblings, no writes to the source, the in-flight settlement refusal |
| `test_r10_verify.py` | an intact bundle verifies; tampered, missing and extra files and a tampered manifest are refused |
| `test_r10_restore.py` | byte-identical placement by the target config, `laned_restore.json`, refusing a non-empty target, `--force-target`, refusing a tampered bundle with nothing written, refusing a mainnet target, the `--start-cmd` dry-run |
| `test_r10_check.py` | `CLEAN START` and real-loader equality with the source; refusals for an unparseable file of each role, a `.bak` fallback, post-restore drift, a missing restore record, and missing validator state |
| `test_r10_evidence.py` | valid T25 evidence passes; each schema or consistency break is refused; the `--bundle` sha cross-check |
