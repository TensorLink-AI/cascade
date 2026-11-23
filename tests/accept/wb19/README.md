# WB19 acceptance tests: the contract they pin

WB19 (plan §3.0 lane B rows, §12b.6 LB6, D42 / D42b, §17.4 OA8) measures an
expert-only lane B leg **per candidate GPU type**: expert-only step rate, leg
overhead split from training, peak memory at the contract micro-batching,
image qualification (driver + pinned torch). It runs in
`pod (owner yes, OA8)`; its Accept is
`scripts/laned/accept WB19 --evidence evidence/wb19.json`.

**Owner decision 2026-11-23 ("Yes skip it"; plan rev6.5 (a)): WB19 does NOT
run the full lane A leg on 8×H100.** Only the short per-GPU expert legs run
(cost cap $20). The measured `[round] funded_sku_wall_seconds` in the shipped
`chain.toml` stands for the full-leg wall. The full leg is therefore
**optional** in the evidence: none is required, and `lane_a_full_leg` may be
`null`. `--mode full` stays in the tool contract (harmless, unused), and if a
full-mode run IS present it is validated exactly as before.

WB22 and WB23 consume the result: WB22 is "the trainer-side per-GPU step-rate
table from WB19"; WB23 fixes `steps = floor(train_hours × 3600 ×
measured_steps_per_s[gpu])` at admission (D42b (a)) and a guard of
`train_hours × guard_factor` plus the **measured overhead**.

The builder may touch only the paths in `PATHS`:

- `scripts/laned_wb19_measure.py`, the measurement tool (`run`, `combine`);
- `cascade/funding/step_rates.py`, the trainer-side table reader.

The CI part is the tool and the reader. The pod run itself is owner/pod
evidence (`evidence/wb19.json`) and is never mocked.

Some rules for the tests themselves:

- No PATHS module is imported at module level, in a fixture or in
  `helpers.py`. The tool is loaded by path (`helpers.load_tool`) and the
  reader imported inside test bodies.
- No test needs torch or a GPU. Every test runs with `--noconftest -m 'not
  torch'`. The pod's leg result reaches `finish_record` as a fixture
  (`helpers.good_result`), the only stand-in, for an external GPU run.
- The contract digest and image pin are re-derived **independently** in
  `helpers.py` from the shipped `chain.toml` (the lane B driver's
  derivation, `lane_b_driver.py`), never read back from the tool.

## `run`

```
python -P scripts/laned_wb19_measure.py run --chain-toml FILE --contract-block N \
    --sku SKU --mode expert|full --out FILE [--steps N] [--selection-json JSON] \
    [--candidates OA8.json] [--expect-contract-digest HEX] \
    [--dry-run --gpu-fixture FILE]
```

In-process the same entry is `main(argv, *, env=None, fs_root="/") -> int`
(`env` replaces `os.environ`, `fs_root` is where the prod markers are looked
for). The CLI always uses the real environment and `/`. Tests drive `main`
with a clean throwaway `fs_root`, so they run anywhere.

### The leg contract (determinism)

- `--mode expert`: `cfg.throne_contracts_at(N)[0].for_hours([lane_b]
  leg_hours, guard_factor=[round] heat_guard_factor,
  guard_floor_seconds=[round] heat_guard_floor_seconds)`, exactly the lane B
  driver's derivation.
- `--mode full`: `cfg.throne_contracts_at(N)[0]` unreduced (the lane A leg).
- `contract.digest` is `cascade.shared.manifest.contract_digest` of that
  contract. The leg runs **under it unchanged**. There is no flag that
  changes `batch_size` or `micro_batches` (`--micro-batches` is a usage
  error), because a different accumulation changes numerics (§12b.6 D42 1(iii)).
- `--expect-contract-digest HEX` refuses `wrong_contract` when the derived
  digest differs. Pass it on the pod, so a stale baked `chain.toml` (the
  image-rotation trap) refuses before any spend.
- An expert leg stops after `--steps` optimizer steps (≥ 1). A full leg
  always runs the contract's `training_steps`, and `--steps` with
  `--mode full` is `bad_steps`.
- `--selection-json` is required for `--mode expert`. It has the worker's
  shape (`worker.py --selection-json`): a non-empty JSON object, MoE layer
  ordinal (decimal int string) to a non-empty list of distinct non-negative
  ints. Anything else is `bad_selection`. The record stores it canonical:
  str keys, rows sorted ascending. `full` stores `null`.
- **`leg_config_digest`** = sha256 of the canonical JSON (`sort_keys`,
  separators `(",", ":")`, `allow_nan=False`) of
  `{"schema": "laned_wb19_leg/1", "mode", "sku", "contract_digest",
  "image_digest", "steps", "selection"}`. Selection order never moves it.
  `steps`, `sku`, `selection`, `mode`, contract or image always do.

### Candidates and the GPU present

- **Candidates** = `[lane_b] funded_pod_skus` of `--chain-toml` ∪ the
  `gpu_types` list of `--candidates` (the owner's OA8 record,
  `evidence/owner/oa8.json`, shape `{"oa": "OA8", "status", "gpu_types":
  [SKU, …], "cost_cap_usd"}`). `--sku` matches case-insensitively, and the
  record stores the catalog's spelling (`GpuCatalog.canonical`). Not a
  candidate is `gpu_not_candidate`.
- **The SKU is a layout** (`cascade.provision.layout.split_layout`):
  `8xH100` is 8 × `H100`, and `RTX4090` is 1 GPU. The visible GPUs must be
  exactly that count, all with one name, and the base type must equal
  (case-insensitive) the concatenation of one or more consecutive
  whitespace-separated tokens of that name. So `H100` matches "NVIDIA H100
  80GB HBM3", `RTX4090` matches "NVIDIA GeForce RTX 4090" and `A6000`
  matches "NVIDIA RTX A6000", but `L40` never matches "NVIDIA L40S". Any
  other case is `gpu_mismatch`.
- **GPU facts.** On a pod they come from `nvidia-smi`, resolved on the env's
  `PATH`. No `nvidia-smi` means `no_gpu`, before torch is imported. In a dry
  run they come from `--gpu-fixture`, a JSON
  `{"gpus": [{"name", "memory_total_bytes"}, …], "driver", "cuda"}`.
  `--gpu-fixture` without `--dry-run` is `fixture_not_allowed`: a pod record
  never carries fixture GPU facts.
- **Image.** `CASCADE_TRAIN_IMAGE_DIGEST` (`trainer.contract.
  TRAIN_IMAGE_DIGEST_ENV`, a bare digest or a full `…@sha256:` ref) must
  carry the leg contract's `train_image_digest`. Unset, malformed or
  different is `wrong_image`. This applies in every mode.

### Refusals

`RUN_REFUSALS` = `prod_host`, `wrong_contract`, `gpu_not_candidate`,
`wrong_image`, `no_gpu`, `gpu_mismatch`, `bad_selection`, `bad_steps`,
`fixture_not_allowed`, `leg_failed`, `non_finite`.

A refusal exits **2**. The last non-empty stderr line is
`REFUSED <code>: <message>`. `--out` isn't written: an existing file stays
byte-identical.

Order of checks (the ones tests pin): `prod_host` first, whatever else is
wrong; then argument checks (`bad_steps`, `bad_selection`,
`fixture_not_allowed`); then `wrong_contract`; then `gpu_not_candidate`;
then `wrong_image`; then the GPU probe (`no_gpu`, `gpu_mismatch`). All of
these come before torch is imported or anything is trained.

### Never on the prod box

`prod_reasons(env, fs_root) -> list[str]` is non-empty (each reason names
the marker) when `env["CASCADE_HOST_ROLE"]` stripped and lowercased is
`prod`, or any of `PROD_MARKERS` exists under `fs_root`:

```
etc/cascade-ops.env
etc/systemd/system/cascade-ops-follow.service
etc/systemd/system/cascade-intake.service
root/cascade/validator_state.db
```

The guard only stats them and **never reads them** (`cascade-ops.env` holds
alert secrets). A stat that raises an `OSError` (a GitHub runner's user may
not stat `/root`) counts as absent and never crashes the tool. `run`, dry run included, refuses `prod_host` on any reason.
`combine` is pure JSON assembly and isn't guarded. On the prod box the
subprocess tests in `test_wb19_cli.py` that need a non-prod host skip, and
`test_cli_refuses_on_this_prod_box` runs instead.

### Secrets

The tool never prints or records an env value other than the image digest:
no `HIPPIUS_*`, `HF_TOKEN`, `WANDB_API_KEY`, `LIUM_API_KEY`,
`BT_WALLET_PASSWORD`, `R2_*`, and no env dump in the record. Tests plant
sentinel values and grep stdout, stderr and the record.

### The run record (`laned_wb19_run/1`)

```json
{"schema": "laned_wb19_run/1", "wp": "WB19", "source": "pod" | "dry_run",
 "mode": "expert" | "full", "sku": "8xH100", "contract_block": 9399600,
 "gpu": {"name": "NVIDIA H100 80GB HBM3", "count": 8, "driver": "…", "cuda": "…",
         "memory_total_bytes": 85899345920},
 "torch": "2.4.1+cu124" | null, "image_digest": "sha256:<64>",
 "contract": {"digest": "<64>", "training_steps": 23841, "max_train_seconds": 10800,
              "micro_batches": 8, "batch_size": 512, "context_length": 4096,
              "leg_hours": 3.0 | null},
 "selection": {"0": [1, 4]} | null, "steps": 500, "leg_config_digest": "<64>",
 "wall": {"total_seconds": …, "train_seconds": …,
          "overhead_seconds": {"boot": …, "corpus": …, "harvest": …}} | null,
 "steps_per_s": … | null, "tokens_per_s": … | null, "peak_memory_bytes": … | null,
 "finite": true | null, "tool": {"git_sha": "<40 hex or empty>", "git_dirty": false},
 "body_sha256": "<64>"}
```

- **`body_sha256`** is the sha256 of the canonical JSON (as above) of the
  record without `body_sha256`. It is exposed as `body_sha256(record)`.
  Records are hashed, not signed: no key ever goes to a pod.
- **A dry run** (`--dry-run --gpu-fixture`, CPU only, writes `source:
  "dry_run"`) fills the plan (`steps` = the steps it would run) and leaves
  every measurement `null`. It carries no clock or host noise, so two dry
  runs with the same arguments (any `PYTHONHASHSEED`) are byte-identical. Its
  last stdout line is JSON with `leg_config_digest`, `contract_digest`,
  `sku` and `steps`.
- **`wall`** is measured inside the tool:
  - `boot`: tool start to model ready, covering imports, CUDA init and
    init/warm-start load;
  - `corpus`: generator start-up and corpus fill before the first step;
  - `train_seconds`: first step to last step;
  - `harvest`: last step to checkpoint written and digested;
  - `total_seconds` ≥ the sum of the four.

  Pod boot and image pull happen before the tool starts and are outside it.
- **Rates.** `steps_per_s = steps / train_seconds`. `tokens_per_s = steps ×
  batch_size × context_length / train_seconds`, the contract's point-pass
  unit (`tokens_for_hours`).
- **`peak_memory_bytes`** is the max over devices of the leg's peak
  allocation (int > 0 on a pod). It must stay below `memory_total_bytes`
  (the `memory` check).
- **Torch is imported only inside the real leg path.** Loading the module
  and every dry run or refusal stay torch-free. A planted `torch` that raises
  on import doesn't break a dry run.

### `finish_record(plan, result) -> dict`

This is how the real path turns the plan plus the leg's result into the
record. `result` = `{"ok", "steps", "error", "deadline_hit", "final_loss",
"torch", "peak_memory_bytes", "wall"}`. The function keeps every plan field
(source included), fills the measurements, recomputes `body_sha256` and
raises `MeasureRefused` (`.code`, a `RUN_REFUSALS` member):

- `leg_failed`: `ok` is false, `deadline_hit` is true, or `steps` differs
  from the plan's. It wins over `non_finite`;
- `non_finite`: a NaN or inf `final_loss` or wall value, or `train_seconds ≤ 0`.

## `combine`

```
python -P scripts/laned_wb19_measure.py combine --chain-toml FILE --runs R1 [R2 …] \
    --out evidence/wb19.json [--candidates evidence/owner/oa8.json] [--table-out FILE]
```

`COMBINE_REFUSALS` = `bad_run`, `not_pod`, `wrong_image`, `wrong_contract`,
`mixed_block`, `duplicate_sku`. A refusal follows the same rules as for
`run`: exit 2, a `REFUSED <code>:` line, nothing written.

- `bad_run`: an unparseable record, a wrong schema, a missing field, or a
  body hash that doesn't match (a hand edit).
- `not_pod`: a `dry_run` record.
- `mixed_block`: the runs don't share one `contract_block`.
- `wrong_image` / `wrong_contract`: a record's image or contract digest isn't
  the one `--chain-toml` derives at its block and mode.
- `duplicate_sku`: two expert runs for one SKU.

Otherwise it writes `laned_wb19/1` and exits 0:

```json
{"schema": "laned_wb19/1", "wp": "WB19", "passed": true,
 "checks": [{"name": "coverage", "passed": true, "detail": "…"},
            {"name": "full_leg", "passed": true, "detail": "…"},
            {"name": "memory", "passed": true, "detail": "…"}, …],
 "runs": [<records verbatim>],
 "table": {"schema": "laned_wb19_table/1", "image_digest": "sha256:<64>",
           "contract_digest": "<expert leg contract digest>", "contract_block": N,
           "rows": {"8xH100": {"gpu_name", "gpu_count", "driver", "cuda", "steps_per_s",
                               "tokens_per_s", "overhead_seconds", "peak_memory_bytes",
                               "memory_total_bytes", "run_body_sha256"}, …}},
 "lane_a_full_leg": {"sku": "8xH100", "train_seconds", "total_seconds", "overhead_seconds",
                     "funded_sku_wall_seconds": 8801} | null (null: no full run),
 "oa8_sha256": "<sha256 of the --candidates file bytes>" | null,
 "body_sha256": "<64>"}
```

- **`table.rows`** has one row per expert run (never the full leg).
  `overhead_seconds` there is `boot + corpus + harvest`.
- **`passed`** is true only when every check passes:
  - `coverage`: every candidate has an expert run. The detail names the
    missing SKUs;
  - `full_leg`: at most one full run on `8xH100` (zero passes: the full
    leg is optional since the owner skipped it 2026-11-23; two fail);
  - `memory`: every expert run has `peak < memory_total`. The detail names
    the failing SKUs.
- **`lane_a_full_leg`** is `null` when there is no full run. When there is
  one, **`lane_a_full_leg.funded_sku_wall_seconds`** is
  `[round] funded_sku_wall_seconds["8xH100"]` of `--chain-toml`. Put next to
  `train_seconds` / `total_seconds`, it answers the plan's question: is the
  measured full-leg wall training-only?
- **`--table-out`** writes `table` alone as its own file.

## The table reader (`cascade/funding/step_rates.py`, for WB22 / WB23)

The reader is stdlib only. Importing it never imports torch.

- `TABLE_SCHEMA = "laned_wb19_table/1"`. `StepRateTableError(ValueError)`
  has `.code`.
- `StepRate` is a frozen dataclass: `sku, gpu_name, gpu_count, steps_per_s,
  tokens_per_s, overhead_seconds, peak_memory_bytes, memory_total_bytes`.
- `load_step_rate_table(path, *, image_digest=None) -> dict[str, StepRate]`
  reads either a bare table or a `laned_wb19/1` evidence file (its `table`).
  - Evidence with `passed` not true is refused with `not_passed`.
  - Refused with `bad_table`: unparseable JSON, a wrong schema, no or empty
    `rows`, an `image_digest` that isn't `sha256:<64 hex>`, a rate that is
    non-numeric, ≤ 0 or non-finite, or a missing or negative
    `overhead_seconds`.
  - `image_digest` given and different from the table's is `wrong_image`. A
    repinned image makes the table stale, as with
    `funded_sku_wall_measured_image`.
- `rate_for(table, sku) -> StepRate | None` is case-insensitive and ignores
  surrounding whitespace.
- `steps_for_hours(rate, hours) -> int` = `floor(hours × 3600 ×
  rate.steps_per_s)` (D42b (a)). Hours that are ≤ 0 or non-finite raise
  `ValueError`.

## Pod evidence (`test_wb19_evidence.py`)

`accept WB19 --evidence evidence/wb19.json` exports `$LANED_EVIDENCE`. The
`test_evidence_*` tests then run `evidence_problems`, which re-checks against
the **shipped** `chain.toml`:

- **Every run:**
  - `source: "pod"` and a valid body hash;
  - `contract_block` ≥ `[v2] from_block`;
  - the contract digest re-derived at its block and mode, `micro_batches`
    unchanged, and the image equal to the pin;
  - a full leg runs exactly the contract's `training_steps`;
  - `leg_config_digest` re-derived;
  - GPU count and name match the SKU (the token rule above), with a
    non-empty driver and CUDA;
  - `torch == "2.4.1+cu124"` (image qualification: the pinned torch ran on
    this driver) and `finite`;
  - the wall split and both rates are consistent;
  - `peak_memory_bytes > 0`;
  - `tool.git_sha` is a clean commit on this repo's history that contains
    the tool.
- **The table** is re-derived from the runs.
- **Coverage:** every `[lane_b] funded_pod_skus` SKU (a GPU type enters that
  list only after measurement, D42 1) and every `gpu_types` entry of the
  committed `evidence/owner/oa8.json` (required, with `oa8_sha256` binding its
  bytes).
- **The full leg (optional, owner 2026-11-23):** none is required. With no
  full run, `lane_a_full_leg` must be `null` (a value with no run behind it is
  refused). With one full `8xH100` leg, `lane_a_full_leg` must match it, as
  before. More than one is refused.
- **Memory:** every row fits memory.
- **The reader:** the WB22 reader loads the file, and its rates match.

Without `$LANED_EVIDENCE` these skip.

**What offline checks cannot prove:** a record built with the real tool and
a made-up leg result is self-consistent. `test_verifier_arithmetic_holds_on_
a_combine_body` shows that such a body passes the arithmetic checks. The
binding to a real pod is the owner's: the OA8 approval file (owner-written,
hashed into the evidence) and the owner-committed evidence PR. The
`test_verifier_*` tests prove that hand edits are refused: an edited rate
with no rehash, a dry-run source, the wrong torch, a GPU count that doesn't
match the SKU, a changed `micro_batches`, out-of-memory, an inconsistent
`tokens_per_s`, a table not derived from the runs, a `lane_a_full_leg` that
doesn't match its full run or has none behind it, two full legs and an
unknown tool commit. Evidence without a full leg passes.

## The pod command (operator, after OA8; testnet pod, pinned image)

On each rented pod (one per candidate GPU type), from the image's checkout,
with the **deployed** `chain.toml` copied over as `chain.deployed.toml`:

```
CASCADE_TRAIN_IMAGE_DIGEST=sha256:<pinned> python -P scripts/laned_wb19_measure.py run \
  --chain-toml chain.deployed.toml --contract-block <era start ≥ [v2] from_block> \
  --sku <SKU> --mode expert --steps 500 --selection-json '<{"layer": [rows]}>' \
  --candidates evidence/owner/oa8.json --expect-contract-digest <digest from a dry run> \
  --out wb19-<SKU>.json
```

No `--mode full` run (owner skipped the full lane A leg 2026-11-23). Then,
anywhere: `combine --chain-toml chain.toml --runs wb19-*.json --candidates
evidence/owner/oa8.json --out evidence/wb19.json`. The owner commits
`evidence/wb19.json`.

## Failure on main (expected, for the right reason)

Neither PATHS file exists on main. `scripts/laned/accept WB19
--expect-fail` gives 136 collected, 1 passed (`test_name_rule_examples`,
the test's own name rule), 5 skipped (the 2 `test_evidence_*` without
`$LANED_EVIDENCE`, and on the prod box the 3 non-prod CLI tests) and
130 right-reason failures:

- `AssertionError` "scripts/laned_wb19_measure.py is missing (WB19 builds
  it)" (in-process and CLI);
- `ModuleNotFoundError` for `cascade.funding.step_rates`, a PATHS module.

There are 0 wrong-reason failures. On CI (no prod marker) the 3 CLI tests
run and fail on the missing tool as well, and
`test_cli_refuses_on_this_prod_box` skips.

Run locally (prod box, tests only, no torch):

```
nice -n 19 env PYTHONPATH=<wt> python -P -m pytest --rootdir <wt> --noconftest \
  -p no:cacheprovider -m 'not torch' <wt>/tests/accept/wb19
```
