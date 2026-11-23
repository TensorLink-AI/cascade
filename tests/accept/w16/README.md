# W16 acceptance tests: the reference fine-tune script (miner kit)

W16 (plan §3.0 W16 row, §10.2 "Miner kit", §4 T3) ships the miner kit's
reference fine-tune script: it starts from a published checkpoint (the latest
public KING_A, or a lane D king/house), fine-tunes it on the miner's own
**public** data on the miner's GPU, and writes a lane D entry in the exact
W4 format, ready for W10's `cascade lane-d verify | submit`. Accept:
`scripts/laned/accept W16`; CI runs a CPU tiny run.

These tests are the spec. The builder may touch only `PATHS`:
`cascade/miner/lane_d_finetune.py`. Conventions follow
`tests/accept/w0c/README.md`. W10 (the `cascade lane-d` CLI) isn't built;
nothing here depends on it. The script is a standalone module; wiring it into
the W10 CLI is W10's call.

## Test split

| File | Runs | What |
|---|---|---|
| `test_w16_cli.py` | box + CI | CLI, config and refusal contract. Every subprocess runs with `import torch` blocked (a sandbox `torch` package that raises); the in-process tests only call `parse_args`. |
| `test_w16_tiny_run_torch.py` | CI only (`torch` mark) | A real CPU fine-tune of the smallest pyramid size (`cascade-moe-v2-2m`, via `scripts/pyramid_config.py`) from a random-init base written by the real trainer's `save_checkpoint`. |

On main every test fails with a `ModuleNotFoundError` for
`cascade.miner.lane_d_finetune` (in `PATHS`) or an `AssertionError` on a
subprocess exit code.

## Module contract: `cascade/miner/lane_d_finetune.py`

Run as `python -P -m cascade.miner.lane_d_finetune ARGS`. Importing the
module, `--help`, `--dry-run` and every typed refusal must work **without
torch**: torch (and the trainer modules that need it) is imported only after
every input check below has passed.

- **`parse_args(argv) -> config`**, an object with these attributes
  (argparse; bad values exit with code 2 via `SystemExit`):

  | flag | attribute | type / rule | default |
  |---|---|---|---|
  | `--base DIR` | `base` | `Path`, required | — |
  | `--data FILE` | `data` | `Path`, required | — |
  | `--out DIR` | `out` | `Path`, required | — |
  | `--steps N` | `steps` | int ≥ 1 | any int ≥ 1 |
  | `--lr F` | `lr` | float, finite, > 0 | any such float |
  | `--seed N` | `seed` | int | `0` |
  | `--batch-size N` | `batch_size` | int ≥ 1 | any int ≥ 1 |
  | `--device D` | `device` | one of `auto`, `cpu`, `cuda` | `auto` |
  | `--context-length N` | `context_length` | int ≥ 1, or `None` = the base config's | `None` |
  | `--dry-run` | `dry_run` | flag | `False` |

- **`main(argv=None) -> int`** is the entry point (`__main__` exits with it).
- **`REFUSAL_CODES`**: exactly, in check order, `("base_files", "base_code",
  "base_config", "base_header", "data_missing", "data_bad", "data_empty",
  "out_not_empty", "torch_unavailable")`.
- **`FinetuneRefused(code, detail="")`**: an `Exception` with `.code`.

## Inputs

- **`--base`**: a checkpoint directory holding exactly the four lane D files
  `config.json`, `forecast_wrapper.py`, `model.py`, `weights.safetensors`
  (regular files, no subdirectories). Nothing in it is ever imported or
  executed: the code files are compared as bytes.
- **`--data`**: a JSONL file, one JSON object per non-blank line, each with
  `"target"`: a list of finite numbers (no bools, strings, `NaN` or
  `Infinity`). The script reads only this file (public data, plan §4 T3).
- **`--out`**: must not exist, or be an empty directory.

## Refusals

Each check runs in this order, before torch is imported, in both normal and
`--dry-run` mode; the first failure is reported and nothing else runs. A
refusal prints exactly one stderr line
`lane_d_finetune_refused <code>: <detail>` (no traceback), exits **3**, and
leaves `--out` as it was (not created; an existing directory untouched). The
base is never modified.

| code | when |
|---|---|
| `base_files` | the base's entries are not exactly the four files (extra `.py`, `.pt`, a subdirectory, a missing file) |
| `base_code` | `model.py` / `forecast_wrapper.py` differ from the repo's `cascade/trainer/pyramid_model.py` / `pyramid_wrapper.py` bytes (lane D is `cascade_pyramid` only) |
| `base_config` | `config.json` is not a JSON object, has no `"moe"` object, or `"base_arch"` ≠ `"cascade_pyramid"` |
| `base_header` | `weights.safetensors` has no parseable header, offsets that disagree with dtype × shape, or an unknown dtype (read by hand, no torch) |
| `data_missing` | `--data` doesn't exist (the detail names the path) |
| `data_bad` | a non-blank line isn't a JSON object with a finite numeric `"target"` list |
| `data_empty` | no series (empty or blank-only file) |
| `out_not_empty` | `--out` exists and isn't an empty directory (including `--out` = `--base`) |
| `torch_unavailable` | every input is valid, it isn't a dry run, and torch can't be imported |

## `--dry-run`

Runs every input check and prints one JSON object as the last stdout line,
then exits 0 without importing torch for training and without creating
`--out`:

`{"base", "data", "out"}` (resolved absolute paths as strings),
`"base_arch"` (from `config.json`), `"n_tensors"` (in the base header),
`"n_series"`, `"steps"`, `"lr"`, `"seed"`, `"batch_size"`, `"device"`,
`"context_length"` (as given), `"out_dtype": "BF16"` and `"files"` (the
sorted four file names).

## A run (CI tiny run, `torch`)

With `--device cpu`, 3 steps, `--lr 0.01`, `--context-length 256` on six
1024-point series:

- exit 0; `--out` holds exactly the four files (no logs, no temp files);
- `model.py`, `forecast_wrapper.py` and the parsed `config.json` equal the
  base's; the base is unchanged;
- every floating tensor is BF16 with the base's names and shapes;
- the weights moved (some tensor differs from the base cast to BF16) and are
  all finite;
- the W4 ingest (`ingest_entry` with `expectation_for(contract)`, via the fs
  vault) accepts it byte for byte;
- the last stdout line is a JSON summary `{"out", "steps", "final_loss",
  "digest"}`: `final_loss` a finite float and `digest` =
  `manifest_digest(checkpoint_manifest(out))` (the entry digest W10 submits);
- two runs with the same seed on CPU are byte-identical;
- a BF16 base (a lane D king or house) works the same way.
