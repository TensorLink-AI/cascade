# R7 acceptance tests: the contract they pin

R7 builds the S0c overlay bring-up artefacts (plan §3.0 row R7, §5 row 4
"S0c Overlay bring-up", §5 "Kill switch and latency", §14.1, §14.4 item 2).
These tests are the spec. The builder may touch only the paths in `PATHS`.
R7 owns no invariants. It shares T21 (with R3, W2) and T27 (with W14). R7
pins only their CI part: generating a kill-switch overlay body and the
wrapper's dry run for it. The testnet and mainnet drills are R7t and R7m.
The common conventions (PATHS syntax, exit codes, expect-fail right reasons)
are in `tests/accept/w0c/README.md`.

Nothing here touches a chain, a store or a wallet. Every program a real
publish could reach (`cascade-validator`, `btcli`, `ssh`, `scp`, `aws`,
`s3cmd`, `curl`, `wget`) is a shim first on `PATH`. A shim records its argv
and exits 97, so a test fails if anything calls one. `HIPPIUS_*`, `HF_*`,
`WANDB_*`, `BT_*`, `AWS_*`, `R2_*` and `CLOUDFLARE_*` are removed from the
environment, and `HOME` points into `tmp_path`.

## Where it lives

- `cascade/laned/overlay.py`: an `add_parser(subparsers)` module, as R1's
  `cli.py` docstring describes. It registers `overlay` with the four
  subcommands below.
- `cascade/laned/cli.py`: add **only** `"overlay": "cascade.laned.overlay"`
  to `COMMANDS`. R1's `"blocks"` entry must stay.
- `docs/OPERATOR.md` (optional): **only** a short `cascade-laned overlay`
  section under "Network config overlays". Every `--flag` in its snippets
  must be one the CLI accepts, because `scripts/check_docs.py` refuses
  unknown options.

The tests drive it as a subprocess with cwd and `PYTHONPATH` set to the repo:
`python -P -m cascade.laned.cli overlay <command> ...`. It needs nothing
outside the stdlib and `cascade.shared`.

## The three generators

```
cascade-laned overlay benign  --config BASE --grid GRID --after BLOCK --out DIR
cascade-laned overlay refused --config BASE --grid GRID --after BLOCK --out DIR
cascade-laned overlay kill    --config BASE --grid GRID --after BLOCK --out DIR \
    --previous BODY.json
```

- `BASE` is a chain.toml, and the tests use copies of the shipped
  `chain.toml` and `chain.testnet.toml`. In production it is the
  **deployed** file, because overlays merge onto each node's own base (§5
  S0a). The tests also use a mainnet base with lane D armed in the file
  itself (`from_block = 9720000`, `share = 0.25`, `takes_from = "lane_a"`).
  So the generator must restate the given base, not repo defaults.
- `GRID` is R1's `laned_grid/1` live-grid JSON.
- **`effective_block`** must equal R1's answer to
  `cascade-laned blocks next-era-start <BLOCK> --grid GRID --config BASE`.
  A base whose `[round]` drifts from the grid is refused: the run exits
  non-zero and writes neither file. In the tests that is `chain.testnet.toml`
  against the mainnet grid.
- On success the run exits 0 and writes two files:
  - `DIR/body.json`: the overlay body, `{protocol, network, seq, prev,
    effective_block, overrides}` plus an optional `comment`, exactly the
    `netconfig.next_overlay` shape. `network` is
    `network_identity(base)`.
  - `DIR/overrides.toml`: the TOML file `cascade-validator
    --netconfig-overlay` takes. `tomllib` must parse it to exactly
    `body["overrides"]`.
  
  It also prints one JSON object on stdout with `seq`, `effective_block`
  and `digest` (`body_digest(body)`).

### `benign`: the seq-1 no-op (§5 S0c, §14.4 items 2 and 3)

- `seq` 1 and `prev` null.
- `overrides` is non-empty. Every table is overlay-settable, and no entry is
  refused by `check_overrides`.
- Every key is **typed in the base** with the **same value**. The body
  restates current values only, and which keys is the builder's choice.
- `validate_overlay(base, body)` passes. The config it builds equals the
  base on every parsed field, ignoring `raw` and the overlay stamp
  (`netconfig.seq/digest/applied_block`). `plan_overlay(...).diff` is empty.

### `refused`: the deliberately refused body (§14.1, §14.4 item 2)

- It has the same shape, with `seq` 1, `prev` null and the computed
  `effective_block`. Only its overrides are defective.
- `validate_overlay` raises `NetConfigError` naming `release-only`, and so
  do `check_overrides` and `build_config`. `plan_overlay` is not `ok`.
- **Refused whole:** the body carries at least one refused entry and at least
  one valid one. The refused entry can be a release-only key such as
  `[lane_d] remainder`, a `REFUSED_TABLES` table or an unknown table. With
  the refused entries stripped, the rest validates and changes nothing. So
  the refusal throws away keys that would otherwise apply.
- The stdout JSON also carries `refusal`, a string containing `release-only`.
- Note: `cascade-validator --netconfig-publish` never publishes an invalid
  plan (`main.py` returns 1 when `plan.ok` is false), and `sign_overlay` runs
  `check_body`. So R7 only *generates* this body. Getting it onto testnet
  is R7t's problem.

### `kill`: the revert overlay (§5 kill switch, T21, T27 CI half)

- `--previous` is the body in force, seq N. The kill body is seq N+1, with
  `prev = body_digest(previous)`.
- It has the same tables as the previous overlay, and every table other than
  `[lane_d]` is restated identically (§14.5: lane A keys stay byte-identical).
  It keeps every `[lane_d]` key of the previous overlay with the same value,
  except `until_block` / `share`, and it may add only those two keys.
- `validate_overlay(base, body)` passes and `plan_overlay(base, overrides,
  eff, previous)` is `ok`, which also means no retroactivity. In the built
  config, either `lane_d.until_block == effective_block` or
  `lane_d.share == 0.0`. Compared with the config the previous overlay
  builds, **only `lane_d` differs**.
- T21: at the first settlement of the stated era (`effective_block + 900`),
  lane D is no longer paid: `not (active(b) and share > 0)`. Before the kill,
  it was.
- The kill is refused (non-zero, no `body.json`) when lane D is not armed in
  the config the previous overlay builds (`from_block == 0`).

## The runbook wrapper: `publish` (Approve-tier, §3 "Owner steps")

```
cascade-laned overlay publish --config BASE --overrides FILE --effective-block N \
    [--previous BODY.json] [--comment TEXT] [--network NET] \
    [--wallet-name NAME] [--wallet-hotkey NAME] [--execute --confirm TOKEN]
```

It makes no network, chain or storage call. `--previous` stands in for the
latest published body; at publish time `cascade-validator` re-derives
seq/prev from the store and prints its own plan.

**Dry run (the default).** It exits 0 when the plan is valid and 1 when it
is refused. Two runs with the same inputs print the same stdout. The output
has these lines:

| line | content |
|---|---|
| `DRY RUN...` | at least one line starting `DRY RUN` |
| `SEQ <n>` | `1`, or the previous seq + 1 |
| `PREV <hex>` | the full `body_digest(previous)`, or `-` with no previous |
| `EFFECTIVE_BLOCK <n>` | `N` |
| `DIGEST <hex>` | `body_digest(next_overlay(overrides, N, previous, network=network_identity(base), comment=TEXT))` |
| `NOTE <cfg-xxxxxxxx>` | `cfg_feature_name(body)` |
| `DIFF <line>` | one per `diff_tables(base.raw, merge_raw(base.raw, overrides))` line, in order. `DIFF (none)` when there are none |
| `CHECK VALID` / `CHECK REFUSED <reason>` | `plan_overlay(base, overrides, N, previous).ok`. The reason carries the errors (`release-only` for a release-only key) |
| `COMMAND <shell-quoted argv>` | only when valid (see below) |
| `CONFIRM <token>` | only when valid: one whitespace-free token that changes when the plan changes (different effective block ⇒ different token) |

The output also mentions `--execute`, so the owner knows how to run it.
Neither stdout nor stderr mentions `coldkey`, `BT_WALLET_PASSWORD` or
`.bittensor/wallets`.

**The printed command.** `shlex.split` of the `COMMAND` text is
`cascade-validator` followed by `--flag value` pairs and the bare
`--netconfig-publish`, with:

- `--chain-toml BASE`, `--netconfig-overlay FILE` and
  `--netconfig-effective-block N`, with paths exactly as given;
- `--netconfig-publish`;
- only when given: `--network NET`, `--netconfig-comment TEXT`,
  `--wallet-name NAME`, `--wallet-hotkey NAME` (names forwarded verbatim,
  never invented: no `--wallet*` flag without them, never `--wallet-path`);
- no other flag, and no flag twice.

**Execute.** `--execute` runs the command only with `--confirm` equal to the
token the dry run printed for the same inputs. These all exit non-zero and
call nothing: `--execute` without `--confirm`, with a wrong token (the output
then mentions `confirm`), or on a refused plan. The tests never pass a
correct token. When execute is reached, it resolves `cascade-validator` from
`PATH`.

**Keys.** `cascade/laned/overlay.py` contains none of these
(case-insensitive): `coldkey`, `BT_WALLET_PASSWORD`, `.bittensor/wallets`,
`keyfile`, `import bittensor`, `bt.Wallet`, `sign_overlay`,
`publish_overlay`, `open_manifest_store`, `put_text`. Only
`cascade-validator` signs and publishes, with the owner's own wallet.

## Fixture blocks

- Mainnet grid (R1 README): `--after 9800001` gives 9802800.
- Testnet grid (`chain.testnet.toml` `[round]`: 150/300/600, 4 per era):
  `--after 6000001` gives 6000600.
- Activation (seq N): `[lane_d] from_block = 9720000, share = 0.25,
  takes_from = "lane_a"` at `effective_block` 9712800.

`test_r7_fixtures_are_valid_on_main` passes on main. It proves these
fixtures validate with the real `netconfig` code. A hand-built kill body
changes only `lane_d`.

## Expected on main

The 17 tests that drive `overlay` fail with `AssertionError`, because
`cascade-laned` rejects the unknown `overlay` command with exit 2 and has no
`COMMANDS["overlay"]`. These tests pass on main: the fixture-sanity tests,
the drift and unarmed-kill refusals, and the `--execute` refusals. They only
assert a non-zero exit and no shim call.
