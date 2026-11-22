# W1.5 acceptance tests: the walking-skeleton contract

W1.5 (plan rev6 §3.0) lays a thin real version of every lane D stage, end to
end, and the CI target `make laned-e2e` that runs it:

fixture `pool_d` (with `end_ts`) → tiny checkpoint → intake (filesystem
adapter behind the real storage interface) → ingest guards → scorer → signed
report → `lane_d_state.json` → emission → `check_lane_d`.

These tests are the spec. The builder may touch only the paths in `PATHS`.
W3, W4, W5a–h, W7a–c, W8 and W9 own the full versions later. Each of them
must keep this skeleton green (row: "every later WP keeps it green and adds
≥ 1 assertion").

## What is pinned

| Row / invariant | Test |
|---|---|
| `make laned-e2e` exists | `test_makefile_defines_laned_e2e`: top-level `Makefile`; target `laned-e2e:`; `make -n laned-e2e` shows `uv run --frozen`, `pytest` and `tests/accept/w1_5` |
| non-zero lane D weight for the challenger | `test_e2e_challenger_gets_nonzero_lane_d_weight` |
| audit passes | `test_e2e_audit_passes` |
| one tampered row fails | `test_e2e_tampered_row_fails`: a tampered row fails `check_lane_d`, both as published (stale signature) and re-signed by the same key (stale `rows_sha256`); `verify_report` returns False for both, since it validates the report before checking the signature (W7b) |
| unarmed byte-identical (inv. 1) | `test_inv1_weights_unchanged_when_unarmed` + `test_inv1_weights_armed_nonempty_lane_d_effect` |
| `share = 0`: weights unchanged, valid report (inv. 9) | `test_inv9_share_zero_vector_unchanged_when_unarmed` + `test_inv9_share_zero_vector_armed_nonempty_lane_d_effect` |
| T13, inv. 2: lane A verdicts and court identical | `test_inv2_t13_lane_a_court_unchanged_when_unarmed` + `test_inv2_t13_lane_a_court_armed_nonempty_lane_d_effect` |
| `LaneDConfig.active()` is not a constant | `test_lane_d_config_active_is_not_a_constant` (already passes on main since W2) |
| storage interface, intake, emission part | `test_storage_interface_and_filesystem_adapter`, `test_intake_ready_reads_through_storage`, `test_lane_d_emission_part_taken_from_lane_a` |

Scope of T13 and invariant 2 at skeleton level:

- Lane A's court is the set of UIDs its part pays. That set and its relative
  split are identical armed and unarmed.
- Lane A's share-of-total is `1 − share` (D1/D48: lane D takes 0.25 of the
  total from lane A's pot).
- Lane A's state files (`cascade_state.json`, `validator_state.db`, where its
  verdicts live) in the same state dir are byte-identical after armed lane D
  settlements.
- The receipt half of invariant 2 belongs to W7c, W8 and W9.

## Two fixture kinds, one box-side pipeline

`test_w1_5_e2e.py`'s `chain_kind` fixture is parametrised, and every pipeline
test runs on both kinds. The fixture holds only the kind and a scratch dir: it
imports no `PATHS` module and builds nothing. Each test body calls `_chain()`,
which first imports the box-side W1.5 modules (on main the test fails there,
inside the body). It then builds the chain and the test signer once per kind
and reuses them:

- **`model`** (marked `torch`; CI installs torch with `uv sync --all-extras`).
  This is the real chain.
  - Two random-init checkpoints are built with the repo's model code at the
    smallest size (`cascade-moe-v2-2m`, via `scripts/pyramid_config.py`
    `local_config("2m")` and `pyramid_trainer.save_checkpoint`).
  - The **house** is F32, like a lane A king, with its head's median-channel
    bias raised by +3. It therefore forecasts `loc + 3·scale`.
  - The **challenger** is BF16 (D14).
  - The challenger is uploaded through intake. Its local copy is deleted, it
    is readied and ingested from storage, and both models are forecast by the
    lane D scorer.
  - Targets are the challenger's median plus 0.05·sd noise, so the
    challenger must win.
- **`synthetic`** (no torch). This is the same box-side pipeline, but the eval
  host's `(horizon, 9)` quantile forecasts are given directly. The eval host
  is a pod, so it counts as an external service on this path. The challenger
  is centred on the targets and the house is 3 sd off.

Settlements run at `BOUNDARY_1 = from_block + 900` on snapshot 1 and at
`BOUNDARY_2 = from_block + 1800` on snapshot 2. The two snapshots' window
sets are disjoint and every window is forward-eligible. With two settlements,
a later mandatory confirmation duel (W5e) still crowns inside the skeleton.
The weight assertions read the second settlement.

The armed config is the real W2 parse:

- `from_block = 2700·3600`, `share = 0.25`, `takes_from = "lane_a"`;
- floors lowered so 48 windows meet them: `min_windows = 24`,
  `min_clusters = 4`, `bootstrap_B = 200`.

"Unarmed" is the same table with `from_block = 0`. The fixture metagraph:

| Item | Value |
|---|---|
| `n_uids` | 8 |
| `burn_uid` | 0 |
| lane A court (king, prior king) | `[3, 4]`, lane A `king_decay` 0.5 |
| challenger UID | 6 |
| the house | no UID, never paid |

## Module contract (thin real versions; the names are what the tests import)

### `cascade/lane_d/storage.py`

- `LaneDStorage` is an `abc.ABC` with abstract `put(key: str, data: bytes)`,
  `get(key) -> bytes`, `list(prefix) -> list[str]` (sorted full keys) and
  `size(key) -> int`. A missing key raises `KeyError`.
- `FilesystemStorage(root)` is the CI adapter. Keys are `/`-separated paths
  under `root`.

### `cascade/lane_d/intake.py`

- `checkpoint_manifest(dir) -> {"files": {name: {"sha256", "size"}}}`.
- `manifest_digest(manifest) -> str`: sha256 hex, the entry digest.
- `upload_checkpoint(storage, hotkey, dir) -> digest` writes every file under
  `lane-d/<hotkey>/<digest>/<name>`. This is the miner's upload, done here
  through the adapter.
- `ready(storage, *, hotkey, digest, context, coldkey, ready_block, cfg) ->
  dict`. It reads only the storage, and returns at least
  `{"hotkey", "digest", "context", "coldkey_at_ready", "ready_block"}`.

### `cascade/eval/lane_d_ingest.py`

`ingest_entry(storage, entry, dest_dir, *, contract) -> path`:

- fetches the entry's objects;
- re-hashes them against the manifest;
- runs the checkpoint guards, including the lane-D-only BF16 path (lane A's
  F32 guard is unchanged) and `verify_weights_finite`;
- returns the verified checkpoint directory, which holds `weights.safetensors`.

### `cascade/eval/lane_d_scorer.py`

- `SCORING_LEVELS == (0.1, …, 0.9)`.
- `forecast_quantiles(checkpoint_dir, histories, *, context, horizon) ->
  array (n, horizon, 9)`, all values finite.
- It uses the quantile head at the 9-level grid, with no padding (§1.1) and
  fp32 scoring. It loads F32 and BF16 checkpoints alike.

### `cascade/validator/lane_d.py`

```
settle_lane_d(*, cfg, boundary, pool_dir, entries, house, forecasts,
              uid_of, state_path, signer) -> object with .report, .pay
```

The arguments:

| Argument | Meaning |
|---|---|
| `entries` | every revealed entry: the `ready` record plus `reveal_block` and `reveal_ts` (unix s). Entries already judged are skipped using the state |
| `house` | `{"digest", "reveal_ts"}` |
| `forecasts` | `{digest: {window_id: array (horizon, 9)}}`, for every window of the snapshot |
| `uid_of` | `{hotkey: uid}` |
| `signer` | an sr25519 keypair |

The results:

- **Unarmed** (`not cfg.active(boundary)`): `.report is None` and
  `.pay is None`. Nothing is written to the state dir.
- **Armed**:
  - `.report` is the signed report (a plain JSON dict);
  - `.pay` is `{"share": cfg.share, "court_uids": [king first …]}`. A `None`
    entry means the house holds the slot, which burns (D21);
  - `state_path` (`lane_d_state.json`) is written with `["king"]["digest"]`
    and `["house"]["digest"]`.

### `cascade/validator/lane_d_report.py`

- `REPORT_SCHEMA == "lane-d-report/1"`.
- `sign_report(report, signer) -> report` and
  `verify_report(report, signer_ss58) -> bool`. These use the repo's sr25519
  helpers, the MarketCast report pattern.
- Report fields pinned:
  - `schema`, `boundary` and `signature`;
  - `params.share`, which equals `cfg.share` (§14.3);
  - `rows`: `[{"window_id", "digest", "crps", "mase"}]`, both models'
    digests present on a judged settlement;
  - `rows_sha256`: 64 hex characters, over the rows.

### `cascade/validator/lane_d_state.py`

Reads and writes `lane_d_state.json`. Its internals are free.

### `cascade/audit/lane_d.py`

`check_lane_d(report, cfg, *, signer_hotkey) -> cascade.audit.checks.CheckResult`
returns `PASS` on a genuine report, and `FAIL` when the signature,
`rows_sha256` or the verdict re-derived from the rows doesn't match.

### `cascade/shared/emission.py`

- `lane_d_vector(n_uids, *, share, court_uids, king_decay, burn_uid) ->
  list[float]` sums to `share`. A sole court member gets all of it, and a
  `None` (house) slot burns.
- `lane_parts(…, lane_d_cfg=None, d_court_uids=())`:
  - with `lane_d_cfg` (passed only when lane D pays at the settlement, as for
    lane C), lane D's `share` is taken from lane A's pool (`takes_from =
    "lane_a"`) and its part is added to the second element;
  - with `None`, the result is exactly today's.

`cascade/shared/lane_c.py` is in `PATHS` in case `lane_pool_shares` changes
(W8/D48).

### `Makefile`

Target `laned-e2e` runs `uv run --frozen pytest … tests/accept/w1_5…`.

### `scripts/deploy_gate.py`

R8's every-tracked-path-classifies test needs the new top-level `Makefile` in
an AREA (for example `ci`).

### `pool_d` fixture format (`pool_d/1`, one directory per snapshot)

- `metadata.json`: `{"schema": "pool_d/1", "snapshot", "horizon", "feeds":
  {feed: {"mirror_lag_hours", "forecast_product"}}, "series":
  [{"series_id", "source", "feed", "freq", "seasonal_period",
  "first_target_ts", "end_ts", "target_ts": [h], "first_seen_ts": [h],
  "frontier": [[as_of_ts, end_ts]]}]}`. Clusters are `source`.
- RV3 BLOCKING 1 (accepted change 1(a)): forward-only fails closed on a
  series with no `frontier` (absent or `null`) and on a feed missing from
  `feeds`. So every fixture series carries one frontier record (an earlier
  snapshot taken 3 h before the snapshot's first target, holding data up to
  1 h before that) and a `feed` listed in `feeds` with a measured
  `mirror_lag_hours` of 1.0 and `forecast_product = false`. S0 lies after
  every bound the fixture uses and before every target, so every window stays
  eligible. `test_fixture_pool_d_carries_end_ts` pins this.
- `hist.npz` and `tgt.npz`: one float64 array per `series_id`.
- Window id: `<series_id>:<target_ts[0]>:h64` (§1.3).

## Failure on main and the `torch` marker

- On main, every pipeline test fails at its first import of a `PATHS` module
  (`cascade.validator.lane_d`, `cascade.audit.lane_d`, `cascade.lane_d`,
  `cascade.eval.lane_d_ingest`) or with `cannot import name 'lane_d_vector'`.
- The Makefile test fails on an assertion.
- The `active()` test and the fixture-format test pass.
- Tests marked `torch` (the `model` chain) need torch. The production box must
  never run them, so locally run `-m "not torch"`.
- `torch` is not registered in pyproject, so pytest warns about an unknown
  mark.
- Signature verification needs bittensor ≥ 11 (`bittensor.sp_core`), which
  `uv.lock` pins.

## RV5 correction: the crowning fixture passes the gate inputs (W5e fail-closed)

Once W5e lands (`tests/accept/w5e/README.md` "Fail closed"), an armed
settlement that would crown without a `bench` session and a
`boundary_block_hash` is `bench_hold`, not a crown, and `validate_report`
(so `verify_report`) refuses a crowned report without a `gate` record.
The seven `[synthetic]` e2e/inv tests that crown at BOUNDARY_2 go through
`_settle` / `_settle_both`, so `_settle` now passes `boundary_block_hash`
(`helpers.gate_block_hash`), `sentinel=None` and a healthy `bench`
(`lane_d_bench.SweepSession` over `helpers.HealthyBench`) whenever
`settle_lane_d` accepts those keywords (`inspect.signature`), and makes the
legacy call otherwise. The tests pass both on today's main and after W5e.

- The bench is the one external service: every digest but the house
  returns the same recorded rows and the house is 10% worse, so the
  crowning candidate passes on any drawn suite and the duel alone decides.
- `test_e2e_audit_passes` and `test_inv9_share_zero_vector_armed_nonempty_lane_d_effect`
  still verify and audit the crowned report. On main it carries no `gate`;
  the gate record arrives with W5e (written by the gated settlement, a
  `pass` for the crowned digest), which W7a's RV4 schema accepts. These
  tests don't splice one in, and no assertion changed.
