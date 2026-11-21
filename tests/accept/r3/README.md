# R3 acceptance tests: the contract they pin

R3 is the overlay lint (plan §3.0 R3 row, §5 S6 and kill switch, §14.2,
§14.5). It refuses any `[lane_d]` overlay that touches a non-`lane_d` key
(except a byte-identical restatement of seq N-1) or a release-only burn
destination (`remainder`). It also owns the CI-checkable parts of T21 and
T24. These tests are the spec. The builder may touch only the paths in
`PATHS`. Conventions (`PATHS` format, directory naming, the `accept` runner,
expect-fail reasons) are W0c's (`tests/accept/w0c/README.md`).

R3 depends on W2 (`ALLOWED["lane_d"] = None`, `[lane_d]` in the loader and
in `chain.toml`), which is on main.

## Where it lives

- Module `cascade/laned/overlay_lint.py` in the shared lane D tooling
  package (`cascade/laned/__init__.py` already exists and isn't in R3's
  `PATHS`). Python 3.11. It may import the real `cascade.shared` code and the
  stdlib. No torch, no network, no chain, no storage. Inputs are read only.
- CLI: `python -P -m cascade.laned.overlay_lint --overlay F [--previous G] --base B [--policies P]`.
- Python API: `main(argv: list[str]) -> int`. It parses `argv` like the
  CLI, prints the report to stdout and returns the exit code. The CLI is
  `sys.exit(main(sys.argv[1:]))`.

## Inputs

- `--overlay F`: the DEC-CA-0057 body of seq N as JSON (`{protocol, network,
  seq, prev, effective_block, overrides[, comment]}`), the unsigned body that
  `cascade.shared.netconfig.next_overlay` builds. It isn't a signed envelope.
- `--previous G`: the body of seq N-1. It's required when the overlay's
  `seq > 1` and must be absent at `seq == 1`. The lint doesn't require it to
  pass today's `check_overrides`: it may have been published under an
  older release (for example one that let `[lane_c] remainder` through).
- `--base B`: the `chain.toml` the overlay is validated against.
- `--policies P` (optional): the node policies of the target fleet (T24):

  ```json
  {"nodes": [{"name": "trainer-main", "policy": <the JSON of cascade-validator --netconfig-policy>}]}
  ```

  `policy` is exactly what `cascade.shared.netconfig.overlay_policy()`
  returns on that node. Its `overlay_keys` lists the overlay tables that
  node's release accepts. Without `--policies` no fleet check runs.

## Rules

An overlay is a **`[lane_d]` overlay** when its `overrides` has a `lane_d`
table. That's the row's literal reading. The reason (§14.1, §14.5) is
presence, not change: a node without `lane_d` in its allow-list refuses
the **whole** overlay. Any non-`lane_d` change carried by such an overlay
then splits the fleet into nodes that took it and nodes that didn't.

| rule | refuses |
|---|---|
| `non-lane-d` | In a `[lane_d]` overlay, any non-`lane_d` `(table, key)` whose canonical JSON value differs from seq N-1's: a changed value, a key seq N-1 doesn't set (even when it equals the `chain.toml` value, because each node's base can drift, §5 S0a), or a key seq N-1 sets that seq N drops (overlays are snapshots, so a dropped key reverts to the base). At seq 1 there's nothing to restate, so every non-`lane_d` key is refused. |
| `release-only` | Any overlay (`[lane_d]` or not) that sets `[lane_d] remainder`, `[lane_c] remainder`, `[lane_b] remainder` or `[scoring] burn_uid`, **even as a byte-identical restatement** of seq N-1. RV1: a restated `[lane_c] remainder` is accepted by pre-capability nodes and refused by upgraded ones, so it would split a mixed fleet. |
| `chain` | `seq != previous.seq + 1`, `prev != body_digest(previous)`, `seq > 1` without `--previous`, or `seq == 1` with `--previous`. |
| `invalid` | Anything `cascade.shared.netconfig.validate_overlay(<base parsed>, overlay)` refuses. The line carries its message, for example `... is not an era start ...`. |
| `unsafe-fleet` | With `--policies`, any node whose `policy.overlay_keys` lacks a table the overlay sets. The node refuses that overlay whole (T24). |

"Byte-identical" compares the canonical JSON encoding that
`netconfig.body_digest` uses (`json.dumps(v, sort_keys=True,
separators=(",", ":"))`). So `300` and `300.0` differ, and dict key order
doesn't matter.

Overlays without a `[lane_d]` table (lane A tuning) aren't subject to
`non-lane-d`. The other rules still apply. A lane A overlay is safe on a
mixed fleet (T24: lane A stays safe while the trainer waits for S5).

Not pinned: an overlay that withdraws a `[lane_d]` table seq N-1 had.

### The kill switch (T21)

Seq N-1 is the S6 activation overlay: `[lane_d] from_block, share = 0.25,
takes_from = "lane_a"` plus lane A keys restated from seq N-2. Seq N
restates the lane A keys byte-identically and either adds `until_block` (an
era boundary after `from_block`) or sets `share = 0`. Both shapes pass, also
against a fleet whose every policy lists `lane_d`. No release is needed.

## Output and exit codes

- stdout has one line per finding, `REFUSED <rule> ...`, sorted as plain
  strings. A `non-lane-d` or `release-only` line starts with `REFUSED <rule>
  [<table>] <key>`. A `chain` line starts `REFUSED chain`. An `invalid` line
  starts `REFUSED invalid` and contains the `NetConfigError` text. An
  `unsafe-fleet` line starts `REFUSED unsafe-fleet <node name>` and contains
  `[<table>]`.
- The last line is `RESULT OK` when there are no findings, otherwise
  `RESULT REFUSED <n>`, where `<n>` is the number of `REFUSED` lines.
- Exit 0 means `RESULT OK`. Exit 1 means refused. Exit 2 means a usage error
  or unreadable input: a missing required argument, a missing file, or a file
  that isn't a JSON object.
- Deterministic: the same inputs always give byte-identical stdout,
  whatever the key order in the JSON files. Nothing is timestamped.

## Test files

- `helpers.py`: fixture bodies built with the real `netconfig.next_overlay`
  against the shipped `chain.toml`. The module under test is imported only
  inside test bodies (`lint_module()`).
- `test_r3_lint.py`: everything above, in-process through `main(argv)`, plus
  subprocess runs of `python -P -m cascade.laned.overlay_lint`.

Owned T-rows (CI part only): T21 (with R7, W2) and T24 (with W2).
