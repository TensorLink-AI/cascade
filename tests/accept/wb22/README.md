# WB22 acceptance tests: lane B's own GPU list (D42; TB23, TB24)

Plan sources: the WB22 row (§3.0 lane B track), §12b.6 LB6 D42 item 1, rev6.8
(WB25: `gpu_n`), T-rows TB23 and TB24 (§4). Owner decisions from 2026-11-23:
no single 1×RTX4090 ("too slow"), "use common sense", and "for lane B I said
to test 8x4090".

The builder may touch only the paths in `PATHS`. Lane A stays byte-identical.
There is no dependency, `[training]` or contract-digest change. Nothing here
needs torch or a GPU.

## The shipped list (`chain.toml`, mainnet)

```toml
[lane_b]
funded_pod_skus = ["8xH200", "8xH100", "4xH200", "4xH100", "8xRTX4090"]
```

- Rationale: these are the data-centre multi-GPU layouts that lane A's funded
  ladder already rents (`[round] funded_pod_skus`), plus `8xRTX4090`, which
  the owner asked for on lane B only. No layout has a single GPU. WB19
  measured 0.055 steps/s on one GPU, and WB25 makes an expert leg use every
  GPU of the layout. `8xRTX4090` uses the layout spelling the rental code
  splits (`cascade.provision.layout.split_layout("8xRTX4090") == ("RTX4090",
  8)`; Lium is queried with `--gpu RTX4090` at count 8).
- The comment above the key names the owner decision (`owner 2026-11-23`)
  and `8xRTX4090`.
- `[round] funded_pod_skus` is unchanged: `8xRTX4090` never becomes a lane A
  type.
- `chain.testnet.toml` is unchanged. Its `[lane_b]` list stays its `[round]`
  list.
- The parser default (key absent = the same file's `[round]` list, WB10) is
  unchanged. The shipped value moves no contract, lane A or lane B consensus
  digest (WB10's tests-correction #774 lets the shipped value differ).

## `cascade/funding/lane_b_gpus.py` (stdlib only, never imports torch)

| name | contract |
|---|---|
| `STEP_RATES_ENV` | `"CASCADE_LANE_B_STEP_RATES"` |
| `DEFAULT_STEP_RATES` | `"evidence/wb19.json"` (repo-relative; WB19's evidence) |
| `UNMEASURED` | `"gpu_rate_unmeasured"`, the typed refusal code |
| `step_rates_path(env=None) -> Path` | `env[STEP_RATES_ENV]` when set and non-empty, else `<repo>/evidence/wb19.json`. `env` None means `os.environ` |
| `lane_b_image_pin(cfg, block=None) -> str` | `sha256:<64>` taken from `cfg.throne_contracts_at(b)[0].train_image_digest`. `b` is `block`, or `cfg.training.v2_from_block` when `block` is None |
| `load_lane_b_rates(path, *, image_digest) -> dict` | a missing file gives `{}` (nothing measured). Otherwise `step_rates.load_step_rate_table(path, image_digest=image_digest)`, which raises its `StepRateTableError` (`bad_table`, `not_passed`, `wrong_image`) |
| `LaneBGpus` | frozen dataclass `(skus: tuple[str, ...], rates: dict \| None, default_max_price_per_hour: float = 0.0)`. `rates` None means no table is wired (dev/unit only), and the rate gate is then off |
| `.catalog` | property: a `GpuCatalog(skus=skus, walls=(), wall_cap_seconds=0.0, default_max_price_per_hour=...)` |
| `.rate(sku) -> StepRate \| None` | `step_rates.rate_for(rates, sku)`, or None when `rates` is None |
| `.measured(sku) -> bool` | True when `rates` is None or the SKU has a rate |
| `.rent_skus() -> tuple` | the measured `skus`, in list order |
| `.refusal(choice) -> dict \| None` | `{"code": UNMEASURED, "message": …}` when a chosen type is unmeasured, or when there is no choice and `rent_skus()` is empty. Otherwise None |
| `lane_b_gpus(cfg, rates) -> LaneBGpus` | `skus = cfg.lane_b.funded_pod_skus`, `default_max_price_per_hour = cfg.round.funded_max_price_per_hour` |
| `lane_b_gpus_from_env(cfg, *, env=None, block=None) -> LaneBGpus` | loads `step_rates_path(env)` against `lane_b_image_pin(cfg, block)`. A table error is logged and gives `rates = {}` (fail closed: everything is unmeasured, never a crash) |
| `rent_scope(runner, gpus)` | context manager. It sets `runner.__dict__["_leg_local_obj"].lane_b_rent_skus = gpus.rent_skus()` (creating the `threading.local` if needed) and restores the previous value on exit, even on an exception. Thread-local, so other legs' threads never see it |
| `exact_rederive_ok(*, recorded_gpu_name, recorded_gpu_n, local_gpu_name, local_gpu_n) -> bool` | the audit pin: True only when the recorded name is non-empty and equals the local name, and the GPU counts are equal. A recorded `gpu_n` ≤ 0 (no record: a pre-WB25 one-GPU leg) counts as 1 |

## Intake (TB23, TB24)

- `FundingIntake(..., lane_b_gpus: LaneBGpus | None = None)`.
- `POST /v1/lane-b/fund` with `lane_b_gpus` wired:
  - the GPU headers are parsed against `lane_b_gpus.catalog`. A type outside
    lane B's list is `unknown_gpu`, as on `/v1/fund`;
  - then `lane_b_gpus.refusal(choice)`. The result is 400
    `gpu_rate_unmeasured`, with nothing queued and no key vaulted (no rental
    can follow);
  - a chosen, measured type is queued with the catalog's spelling
    (`entry.gpu`). The 202 reply carries `gpu` and `steps_per_s` (the
    measured rate).
- `lane_b_gpus` None: today's behaviour (lane A's `gpu_catalog`, no rate
  gate). Only unit tests construct it that way.
- `/v1/fund` (lane A) is unchanged: it uses the `[round]` catalog, so
  `8xRTX4090` there is `unknown_gpu` (TB23's "a lane A leg requesting the
  same type is still refused").
- `cascade/funding/main.py` has `gpu_catalogs(cfg, *, env=None) ->
  (GpuCatalog, LaneBGpus)`: `GpuCatalog.from_config(cfg)` and
  `lane_b_gpus_from_env(cfg, env=env)`. The production intake is built with
  `lane_b_gpus=` from it.

## Trainer (TB23)

- `_FundedCapacityMixin` (`runner_capacity.py`), with a `lane_b_rent_skus`
  value (not None) on the thread-local:
  - `_funded_skus_for_rent()` returns that tuple;
  - `_sku_per_leg_active()` is True;
  - `_chosen_gpu_rent(choice)` resolves the choice against that tuple. The
    miner's $/h cap replaces the operator caps, as today. A type outside
    the tuple gives None (the default path).
  - Without the value (every lane A leg), all three behave exactly as on main.
- `RunnerLaneBOps._dispatch_funded` (`lane_b_driver.py`):
  - it builds `lane_b_gpus_from_env(cfg, block=era.start_block)`;
  - an empty `rent_skus()` is a `LegFailure` with `error_class
    "no_capacity"`, `burn=False` and `miner_fault=False`, and
    `_run_funded_leg` is never called;
  - otherwise `_run_funded_leg` runs inside `rent_scope`. Afterwards, `gpu_n
    = layout_gpus(funded_pod_sku)` of the leg's thread-local (0 when unset).
    `done` is called with `gpu_n=` as a keyword.
- `runner_funded_legs.py` sets `_loc.funded_pod_sku` from the rented pod's
  `sku`, right after `_rent_funded_host` (a source check).
- `LaneBDriver._leg_done(..., gpu_n=0)` stores `gpu_n` on
  `FinishedExpertLeg` (new field, default 0). `settle` passes it to
  `ExpertEntry(gpu_n=...)`.
- `ExpertEntry.gpu_n: int = 0`:
  - it is dropped from `ExpertManifest.canonical_body()` when 0, so an entry
    without it keeps main's bytes;
  - it survives `to_json` / `load_expert_manifest` when > 0;
  - a negative value or a bool raises `ValueError`.

## Failure on main (expected, right reasons only)

`cascade.funding.lane_b_gpus` doesn't exist (`ModuleNotFoundError` for a
PATHS module). `main.gpu_catalogs` is missing (`cannot import name … from
cascade.funding.main`, a PATHS module). The config, intake keyword,
thread-local, `gpu_n` and source checks fail with `AssertionError`. A few
guards pass on main: lane A's list unchanged, testnet unchanged, and the
parser default.

Run locally (tests only, no torch):

```
nice -n 19 env PYTHONPATH=<wt> python -P -m pytest --rootdir <wt> \
  -p no:cacheprovider -m 'not torch' <wt>/tests/accept/wb22
```
