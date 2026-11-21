# R1 acceptance tests: the contract they pin

R1 builds the block-arithmetic tool `cascade-laned blocks` (plan §3.0 row R1,
§5 "Block arithmetic", §13). These tests are the spec. The builder may touch
only the paths in `PATHS`. R1 owns no T-rows and no invariants.

## Where it lives

- `cascade/lane_d/blocks.py`: the arithmetic and the `blocks` subcommands.
- `cascade/lane_d/cli.py`: `main(argv=None) -> int`, which dispatches
  `cascade-laned <command>`. Here the only command is `blocks`. It must run as
  `python -m cascade.lane_d.cli` (an `if __name__ == "__main__"` guard calling
  `sys.exit(main())`).
- `cascade/lane_d/__init__.py`.
- `pyproject.toml`: add **only** `cascade-laned = "cascade.lane_d.cli:main"`
  under `[project.scripts]`. Don't touch dependencies, the torch pin or
  `uv.lock`.

The tests drive it as a subprocess: `python -m cascade.lane_d.cli blocks
<query> <args> --grid FILE [--config FILE]`, with cwd and `PYTHONPATH` set to
the repo. The tool must not need a chain connection, torch, or anything
outside the stdlib and `cascade.shared`.

## The live grid (`--grid FILE`, required)

The grid is a JSON object:

```json
{"schema": "laned_grid/1",
 "round": {"epoch_blocks": 900, "epoch_blocks_prev": 3600,
           "epoch_activation_block": 9136800, "era_settlements": 4,
           "reveal_margin_blocks": 25},
 "segments": [{"from_block": 0,       "epoch_blocks": 3600, "era_blocks": 14400},
              {"from_block": 9136800, "epoch_blocks": 900,  "era_blocks": 3600}]}
```

- `round` holds the `[round]` grid keys as resolved live. `reveal_margin_blocks`
  is optional.
- `segments` is the grid as observed on the chain. It's sorted, and the first
  segment has `from_block` 0.
- These are invalid and exit 2 with nothing on stdout: a missing `round` or
  `segments`, a `schema` other than `laned_grid/1`, a first segment not
  starting at 0, or a non-integer value (booleans and strings are refused).

## The two computations

Every answer is computed twice, and both must agree:

1. **The shared code** on `round`: `era_index_of`, `era_start_block`,
   `next_era_start`, `settlement_era` and `min_effective_era` from
   `cascade/shared/era.py`; `is_era_start` and `apply_block_for` from
   `cascade/shared/netconfig.py`; `activation_block_for` from
   `cascade/shared/activation.py`; `blocks_until_boundary_reveal` from
   `cascade/shared/chain.py`; and `effective_epoch_blocks`, always at the block
   being asked about, never the head.
2. **An independent walk over `segments`**. It finds the segment containing the
   block, and that segment's `epoch_blocks` and `era_blocks`, using absolute
   grid arithmetic (`block // length`). It must not call the functions in 1.

If they disagree, the tool exits **3**, prints a message containing
`mismatch` on stderr and prints nothing on stdout. Every successful answer
carries `"cross_checked": true`.

The tests feed four mismatched grids:

- `epoch_activation_block` set to the lock-in block 9133200 instead of the
  rollover 9136800;
- the real pre-8726400 7200-block history, which `chain.toml` can't express
  because it keeps only the last switch;
- `era_settlements` set to 3 against 3600-block eras.

They also check that the same queries succeed on the consistent grid.

**Known seam quirk (not pinned):** for blocks before `epoch_activation_block`,
`era.next_era_start` uses the old era length, so it skips the rollover era
start. Don't rely on either behaviour there.

## `--config FILE` (optional)

This is a `chain.toml`. Its `[round]` keys `epoch_blocks`,
`epoch_blocks_prev`, `epoch_activation_block`, `era_settlements` and
`reveal_margin_blocks` (when present in both) must equal `grid.round`.
Otherwise the tool exits **3** with `drift` on stderr and nothing on stdout.
The repo `chain.toml` must parse and agree with the live fixture.

## Subcommands

Each one prints exactly one JSON object on stdout and exits 0. Every object
includes `"query"` (the subcommand name) and `"cross_checked": true`.

| subcommand | output keys (besides `query`, `cross_checked`) |
|---|---|
| `era-start <block>` | `block`, `epoch_blocks`, `era_blocks` (both in force at the block), `era_index`, `era_start`, `is_era_start` (false for block 0) |
| `next-era-start <block>` | `block`, `next_era_start` (strictly after), `next_era_index` |
| `settlement <boundary>` | `boundary`, `era_index` (the era containing the boundary), `settlement_era` and `settlement_era_start` (the era containing boundary − 1, the one the settlement is judged in), `is_era_start`. A block that isn't a boundary of the grid in force exits 2 |
| `activation --lock-block L --align boundary\|era\|epoch` | `lock_block`, `align`, `activation_block` (`activation_block_for`) |
| `effective-block --effective-block E --lock-block L` | `effective_block`, `lock_block`, `apply_block` (`apply_block_for`: E if L < E, else the first era start strictly after L). An E that isn't an era start exits 2 |
| `min-effective-era <block>` | `block`, `min_effective_era`, `effective_block` (= era × era length) |
| `promotion-fire --check-boundary B --fire-block F` | `check_boundary`, `fire_block`, `effective_era` (= min_effective_era(B)), `min_effective_era_at_fire`, `fires` (effective_era ≥ min_effective_era_at_fire, otherwise "arrived too late", `runner_promotion.py:505`) |
| `reveal <block> [--margin M] [--next-epoch]` | `block`, `margin_blocks` (`--margin`, else `round.reveal_margin_blocks`, else exit 2), `blocks_until_reveal`, `reveal_block` (= block + delay), `boundary` (`(reveal_block // epoch + 1) * epoch`, as in `miner/cli.py:921`) |

## Exit codes

- 0: answered.
- 2: usage error or invalid input (a missing `--grid`, a negative block, a bad
  fixture, not a boundary, not an era start).
- 3: cross-check refusal (`mismatch` or `drift`).

## Golden vectors

Each one cites its source in the test file. Most are real blocks from the
OPSLOG or `chain.toml`. `test_golden_vectors_agree_with_shared_code` re-derives
them all from the shared code, which is why that test already passes on main.
The past off-by-one cases pinned are:

- A lock-in is aligned on the grid in force at the lock block, not the head:
  9133200 → 9136800, never 9134100.
- 9133200 is not an era start on the old 14400-block era grid.
- A lock-in exactly on an era start takes the *next* one: 9190800 → 9194400
  with `era` align. 9486000 → 9493200 with `epoch` align, which is two eras
  later.
- An overlay lock-in exactly at `effective_block` slips a whole era:
  9561600 → 9565200.
- An era-start boundary is the previous era's last settlement: 9464400 has
  `era_index` 2629 but `settlement_era` 2628, and 9547200 has 2652 against
  2651.
- `min_effective_era` is e + 2. A king check fired one era late "arrived too
  late": 9468900 checked, then fired at 9471600. Only an era's 1st and 2nd
  settlement checks can fire.
- A reveal on a boundary targets the next boundary. Inside the margin it
  floors to 1.

## Failure on main

The module `cascade.lane_d` doesn't exist, so every subprocess exits 1 and
every tool test fails with an `AssertionError`. The pyproject test fails on the
missing entry point, also with an `AssertionError`. Only
`test_golden_vectors_agree_with_shared_code` passes. There are no syntax,
fixture or import errors in the test process.
