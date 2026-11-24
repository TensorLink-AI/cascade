# W8c acceptance tests: the production `LaneDInputs` source

W8c (plan rev6.7, W8c row; owner 2026-11-23, "Yes w8c") depends on W8b, W5d,
W15, W4, W3, W6d and W5f. These tests are the spec. The builder may touch only
the paths in `PATHS`. The W8b, W19, W15, W5d, W5f, W4 and W9 suites must stay
green.

## What the plan says

The W8c row, verbatim:

> | W8c | **(rev6.7, owner 2026-11-23)** Production `LaneDInputs` source: W8b's step calls `settle_lane_d` with a `LaneDInputs` from its `inputs` source, but nothing in production supplies one, so an armed lane D refuses `inputs_unavailable` at every settlement. W8c builds a `LaneDInputs` per armed boundary from signed or pinned sources only: (1) the queued entries from the W3 intake state and vault (`cascade/lane_d/`), revealed by the boundary; (2) each entry's, KING_D's and the house's forecasts on the boundary's frozen windows, computed on the lane D eval host by the W5d scorer (`cascade/eval/lane_d_scorer.py`) over checkpoints fetched through the W15 prefetch cache (`cascade/eval/lane_d_cache.py`) and W4 ingest, behind the W4 handshake; (3) the pinned `pool_d` snapshot(s) (W6d), each re-hashed against its pin, and their windows (W5a/W5b); (4) the house record from the W5f signed pointer, and the bench `SweepSession` (W3b*) for W8b's `bench`. **Deterministic:** the same boundary and sources give the same `LaneDInputs`. **Fail-closed:** each missing or unverifiable input is its own typed code (no inputs, no settlement this tick; W8b retries), never a guessed or default input. **Off the poll thread:** it runs inside the W19 worker's step; its chain reader is built on that thread. **No new chain reads beyond `ChainClient`.** **Unarmed:** no source is built and nothing is read (invariant 1); nothing reachable from `cascade.audit` imports it (invariant 12). Wired into `cascade/validator/main.py` through `build_lane_d_step`'s `inputs` and `bench` parameters | W8b, W5d, W15, W4, W3, W6d (`pool_d` is owner/forge data; tests use fixtures), W5f | ci + testnet | `scripts/laned/accept W8c` |

## Today's code (main @ 7d0e303)

- `cascade/validator/lane_d_wiring.py` (W8b): `LaneDInputs(pool_dir, entries,
  forecasts, uid_of, house, prior_pool_dirs=())`; `build_lane_d_step(cfg, *,
  state_dir, signer, chain_factory, store, inputs=None, bench=None)`. If
  `inputs` is `None`, every armed call is `inputs_unavailable`. If `inputs`
  raises, the step raises `LaneDWiringError("inputs_unavailable")` with the
  error chained as `__cause__`.
- `cascade/validator/main.py` calls `build_lane_d_step(...)` with no `inputs`
  and no `bench`, so production lane D cannot settle.
- `settle_lane_d` (`cascade/validator/lane_d.py`) takes `entries`
  (`{hotkey, digest, context, coldkey_at_ready, ready_block, reveal_block,
  reveal_ts}`), `forecasts` (`{digest: {window_id: (horizon, 9)}}`), `uid_of`,
  `house` (`{digest, reveal_ts}`, W5f `check_house`) and the `pool_d/1`
  snapshot dirs (`load_pool`). A chosen entry whose forecast on a pool window
  is malformed (`lane_d_lowfreq.forecast_ok` is `False`) is closed as
  `checkpoint_fault`. A missing bar (KING_D or house) forecast is
  `king_unscored`.
- The intake (`cascade/lane_d/service.py`) persists `lane_d_intake_queue.json`
  (`lane-d-intake-queue/1`, one row per hotkey with `state` in
  `submitted/ready/revealed/ready_expired/closed` and `reveal_block`).
- W15 `PrefetchCache.get/put/ensure`, `prefetch_grant` (fills through W4
  `ingest_presigned`); W5d `forecast_quantiles`; W4 `check_handshake` /
  `CodeIdentity`; W5f `load_house_pointer`; W7c `load_state`; W3b2
  `SweepSession`; W19 `LaneDWorker`.

## Contract

### `cascade/validator/lane_d_inputs.py` (new, box side)

```
INPUTS_ERROR_CODES ⊇ {"not_active", "chain_unavailable", "intake_unreadable",
    "reveal_unreadable", "metagraph_unreadable", "state_unreadable",
    "house_unavailable", "pool_unavailable", "pool_digest_mismatch",
    "handshake_mismatch", "forecasts_unavailable", "forecast_malformed"}
LaneDInputsError(RuntimeError)       .code in INPUTS_ERROR_CODES (an unknown code: ValueError)
build_lane_d_inputs(cfg, *, state_dir, intake_state_dir, chain_factory, pool, eval_host,
                    house_pointer, house_signer_ss58, bench_host=None,
                    expected_identity=None) -> source | None
source(boundary) -> LaneDInputs      # W8b's `inputs`
source.bench(boundary) -> SweepSession | None   # W8b's `bench`
```

- `cfg` is the full `ChainConfig`. **Unarmed** (`[lane_d] from_block = 0`):
  returns `None`. It calls, reads and creates nothing (no factory, pool, eval
  host or bench host use; neither dir is created).
- **Lazy.** Building an armed source reads nothing. Every read happens inside
  `source(boundary)`, on the calling thread.
- `state_dir`: the W8b step's dir. The source reads `state_dir /
  "lane_d_state.json"` through `lane_d_state.load_state(path, house)`
  (genesis when absent) and never writes it.
- `intake_state_dir`: the intake's state dir, holding
  `cascade.lane_d.service.QUEUE_FILENAME`.
- `chain_factory()`: the source's **own** read-only reader, built on the
  calling thread at the first armed call (never the poll client). It uses
  `ChainClient` method names only: `block_timestamp(block) -> float` and
  `registered_hotkeys() -> list[str]` (uid order).
- `pool(boundary) -> [{"dir", "sha256"}]`: the W6d pins, oldest first.
- `eval_host`: `hello() -> dict` (W4 `worker_hello` shape) and
  `forecast(digest, *, context, horizon, windows) -> array (n, horizon, 9)`.
  It raises `LaneDForecastError` (below) or any other exception. The
  transport (SSH, HTTP, ...) is the builder's and is not pinned.
- `house_pointer`: the W5f pointer file. `house_signer_ss58`: its pinned
  signer.
- `bench_host(digests, suites)`: the bench pod's sweep. `None` means
  `source.bench(b)` returns `None`, so W8b passes `bench=None` and a crowning
  candidate is `bench_hold`.
- `expected_identity`: the W4 `CodeIdentity` the eval host must match
  (`None` means `local_identity()`).

**`source(boundary)`** (armed, `cfg.lane_d.active(boundary)`) returns a
`LaneDInputs` with:

- **entries**: every intake row with `state == "revealed"` and
  `reveal_block <= boundary` (no other state; an absent queue file is no
  entries), as exactly `{hotkey, digest, context, coldkey_at_ready,
  ready_block, reveal_block, reveal_ts}`, where `reveal_ts =
  int(reader.block_timestamp(reveal_block))`. Sorted by `(reveal_ts,
  digest)`.
- **pool_dir** is the newest pin's `dir`. **prior_pool_dirs** is the other
  pins' dirs in pin order. Each pin is re-hashed: `lane_d.load_pool(dir,
  cfg.lane_d.horizon)["sha256"]` must equal its `sha256`.
- **house**: `load_house_pointer(house_pointer,
  signer_ss58=house_signer_ss58)`.
- **uid_of**: `{hotkey: uid for uid, hotkey in
  enumerate(reader.registered_hotkeys())}`.
- **forecasts**: first `check_handshake(eval_host.hello(),
  expected=expected_identity)`. Then one `eval_host.forecast` call per
  digest, in sorted digest order, over D = the entries' digests ∪ the
  state's open entries (`status` in `lane_d.OPEN_STATUSES`) ∪ the state's
  KING_D digest ∪ the state's house digest ∪ the pointer's house digest.
  `windows` is every pinned window with a non-empty `window_id`, unique by
  id (the first pin listing it, oldest first), sorted by id, sent as exactly
  `{"window_id", "history"}`. No target leaves the box (D12). `horizon` is
  `cfg.lane_d.horizon`. `context`: an entry's (or a miner KING_D's) declared
  context; the house (and a house KING_D) at a context in
  `cfg.lane_d.contexts`. The result is `forecasts[digest] = {window_id:
  array[i]}`.

**Fail closed.** Each refusal is a `LaneDInputsError` and writes nothing
(the state, intake and house dirs are byte-identical afterwards). W8b turns
it into `inputs_unavailable` with this error as `__cause__`, and the next
tick with the input back settles normally:

| Input | Code |
|---|---|
| a boundary where lane D is not active (touches nothing) | `not_active` |
| `chain_factory()` raises | `chain_unavailable` |
| the queue file is not JSON, not `lane-d-intake-queue/1`, or has a malformed row | `intake_unreadable` |
| `block_timestamp` raises | `reveal_unreadable` |
| `registered_hotkeys` raises | `metagraph_unreadable` |
| `lane_d_state.json` refused by `load_state` | `state_unreadable` |
| the pointer is missing, forged or wrongly signed (the `LaneDHouseError` chained); no forecast is requested | `house_unavailable` |
| `pool()` raises, returns no pin, or a pin's dir doesn't load; no forecast is requested | `pool_unavailable` |
| a pin's bytes don't hash to its `sha256`; no forecast is requested | `pool_digest_mismatch` |
| the hello doesn't match (W4, T19); no forecast is requested | `handshake_mismatch` |
| `forecast` raises anything but a miner-fault `LaneDForecastError` for an entry, or raises anything for KING_D / the house | `forecasts_unavailable` |
| a KING_D / house array that isn't `(n, horizon, 9)` finite | `forecast_malformed` |

**A miner's own fault is not an infrastructure failure.** If `forecast`
raises a `LaneDForecastError` whose code is in `MINER_FAULT_CODES`
(`checkpoint_refused`, `forecast_malformed`) for an entry digest, or returns
an array that isn't `(n, horizon, 9)` finite, the source does not refuse.
It records that digest's forecasts so that `forecast_ok` is `False` on every
window, and `settle_lane_d` closes the entry as `checkpoint_fault`. One bad
checkpoint never blocks lane D.

**Deterministic.** Two fresh sources over the same sources give equal
`LaneDInputs` and the same `eval_host.forecast` call sequence. Reading writes
nothing.

**`source.bench(boundary)`** returns `SweepSession(<session id>,
bench_host)`. The id is a deterministic function of the boundary: equal for
two sources at one boundary, different across boundaries.

### `cascade/eval/lane_d_forecasts.py` (new, eval-host side)

```
FORECAST_ERROR_CODES ⊇ {"checkpoint_unavailable", "checkpoint_refused", "forecast_failed",
                        "forecast_malformed"}
MINER_FAULT_CODES == ("checkpoint_refused", "forecast_malformed")   # any order
LaneDForecastError(RuntimeError)     .code in FORECAST_ERROR_CODES (unknown: ValueError)
forecast_entry(cache, digest, windows, *, context, horizon, contexts, grant=None,
               contract=None, expect=None, device="cpu", forecast=None) -> ndarray (n, horizon, 9)
```

- `cache` is a W15 `PrefetchCache`. A verified hit (`cache.get`) is scored
  in place and makes no vault request. On a miss with `grant`, it calls
  `prefetch_grant(cache, grant, contract=, expect=)` (W4 ingest guards run
  before admission). On a miss without `grant`, it raises
  `checkpoint_unavailable` and nothing is scored.
- An `IngestError` (chained) with code `fetch_failed` is
  `checkpoint_unavailable`. Any other `IngestError` (the miner's bytes) is
  `checkpoint_refused`. Nothing is admitted.
- The scorer: `forecast(checkpoint_dir, [w["history"] for w in windows],
  context=, horizon=, contexts=, device=)`. With `forecast=None` it is
  resolved **at call time** as
  `cascade.eval.lane_d_scorer.forecast_quantiles`. A `ValueError` from it, or
  a result that isn't `(len(windows), horizon, 9)` finite, is
  `forecast_malformed`. Any other exception is `forecast_failed`.
- Torch-free at import (the scorer imports torch lazily).

### The entry point (`cascade/validator/main.py`)

`main()` calls `build_lane_d_inputs(...)` (its `chain_factory` is not the
poll `client`) before `build_lane_d_step(...)`, and passes both `inputs=`
and `bench=` (not the constant `None`) to `build_lane_d_step` (an AST check).
Unarmed, `build_lane_d_inputs` returns `None` and `build_lane_d_step`
returns `None`. Handling a `None` source when building `bench` is the
builder's job.

### Armed contexts (rev6.6; tests-correction 2026-11-23, `test_w8c_armed.py`)

The plan's rev6.6 section and §1.1 context bullet: "The main validator
computes the armed set per settlement and records it, with the evidence it
used, in the signed report". W5d (`armed_record`, `effective_context`, #744)
and W7a (`settle_lane_d(..., armed=)`, `check_armed`) are merged; this
source is what computes the set in production. Until this correction the
source treated `cfg.lane_d.contexts` (the DECLARABLE set) as armed and never
passed `armed=`, so no production report carried `armed_contexts`.

```
build_lane_d_inputs(cfg, *, ..., readiness, t7) -> source | None   # both REQUIRED keyword-only
readiness() -> bytes      # the pinned laned_readiness/1 document's bytes, read at every call
t7() -> bytes             # the pinned laned_t7/1 document's bytes, read at every call
LaneDInputs.armed         # new field on W8b's dataclass, default None (W8b unchanged)
DocumentFile(path)        # production source: () -> the file's bytes, re-read at every call
SSHEvalHost.set_armed(contexts)   # the request's "contexts" becomes the armed set
```

- **Unarmed / at build:** neither document source is called.
- **`source(boundary)`** reads both documents once, on the calling thread,
  after the pools and before the eval host is contacted, and computes
  `armed = lane_d_scorer.armed_record(cfg.lane_d.contexts, readiness=<bytes>,
  t7=<bytes>, min_long_series=cfg.lane_d.min_long_series)`. The returned
  `LaneDInputs.armed` is exactly that record; W8b's step passes it to
  `settle_lane_d(..., armed=inputs.armed)` (`getattr(inputs, "armed", None)`,
  so an inputs object without the field settles as before), and the signed
  report then carries `armed_contexts` / `armed_evidence` (W7a).
- **Forecast contexts:** each digest is forecast at
  `effective_context(declared, armed["armed_contexts"])`: an entry or a
  miner KING_D at its declared context when armed, else the highest armed
  variant below it, ~~else the lowest armed one~~ **(plan rev6.15, owner
  2026-11-23)** else **not forecast at all** (`effective_context` is
  `None`: no armed variant ≤ the declared context; the settlement skips the
  entry `context_below_armed`, and a miner KING_D without a forecast makes
  its cohort `king_unscored`; no new input error, the tick is not refused);
  the house (and a house KING_D) as declared 4096 (the trained length). The
  house declares no context of its own, so the rev6.15 skip doesn't apply
  to it: when no armed variant is ≤ 4096 it runs at the lowest armed one,
  as before. The skip is deterministic: the same documents give the same
  eval-host calls. The entry rows keep their
  declared `context` (settle_lane_d records it). An entry whose declared
  context is not in `cfg.lane_d.contexts` is not forecast (settle_lane_d
  closes it at registration, `context_not_armed`).
- **The eval host:** when it has `set_armed`, the source calls
  `set_armed(armed_contexts)` once per settlement, after the handshake and
  before the first forecast. `SSHEvalHost` then sends that list as the
  request's `contexts` (the eval host's `declared_context` gate); before any
  `set_armed` it sends `cfg.lane_d.contexts` as today.
- **The inputs TOML** gains two required keys, `readiness` and `t7` (absolute
  paths of the two documents); `load_inputs_services` returns
  `readiness=DocumentFile(...)`, `t7=DocumentFile(...)`. The documents are
  re-read per settlement, so a readiness update arms a variant at the next
  settlement without a restart; the report pins the bytes used by sha256.

**Fail closed** (each a `LaneDInputsError`, nothing written, no hello and no
forecast; W8b reports `inputs_unavailable` with it as `__cause__`):

| Input | Code |
|---|---|
| `readiness()` raises (absent, unreadable) | `readiness_unavailable` |
| `t7()` raises (absent, unreadable) | `t7_unavailable` |
| `armed_record` refuses a document (not JSON, wrong schema, bad counts, a T7 row whose stated `fits` disagrees with its numbers) | `armed_invalid` |
| no variant passes both gates (nothing can be scored) | `no_armed_context` |

Consensus: deterministic in (boundary, sources, document bytes); lane A is
untouched; unarmed is byte-identical. Satellites verify the recorded record
(W7d `armed_*` checks) and do not re-read the documents.

Builder PATHS: `cascade/validator/lane_d_wiring.py` is added (the
`LaneDInputs.armed` field and the `armed=` pass-through only).

### Invariants

- **1:** above (unarmed: `None`, nothing used, nothing created).
- **12:** `cascade.validator.lane_d_inputs` and
  `cascade.eval.lane_d_forecasts` exist and are outside the audit's static
  import closure (`tests/accept/w9/test_w9_independence._closure`).
  `loop.py`, `weights.py` and `lane_d_wiring.py` don't import them.
- **Off-thread:** called from another thread (W19's `LaneDWorker` running
  W8b's step), the factory, every reader call and every eval-host request
  run on that thread, never on the constructing thread.

## Fixtures and services

W8b's fixture (`tests/accept/w8b/helpers.py`, W7a's synthetic chain, numpy
only): X is revealed at `REVEAL_BLOCK`, is listed from B1 and is crowned at
its first verdict at B1 (uid 6; plan rev6.11 (b)). The intake queue file is written in the real format. The house
pointer is a real W5f pointer signed by a test key. Fakes stand in for the
reader, the W6d pin list, the eval host (W7a's forecasts), the bench pod
(W8b's) and the bucket. The eval-host tests use a real `PrefetchCache` with
W15's entry files, a stubbed scorer and a stubbed `ingest_presigned`. No
chain, storage, pod, torch, GPU or real key; everything lives under
`tmp_path`.

## Readings chosen (the row is silent; flagged)

- **"Frozen windows."** The source can't know a boundary's frozen sets
  before `settle_lane_d` computes them. It forecasts every needed digest on
  every pinned window, and the settlement uses the windows it freezes. The
  cost is bounded by `pool_d` size × (judged entries + 2). Restricting to
  the eligible windows is a later optimisation, and only safe if the
  settlement is byte-identical.
- **The house's context.** The pointer carries none. Any context in
  `[lane_d] contexts` passes here. 4096 (the trained length) is the obvious
  choice. The ~~rev6.6 "highest armed variant"~~ **(rev6.15)** "highest
  armed variant ≤ declared, else skip" rule for unarmed declared contexts is
  W5d's `effective_context`; this WP only applies it (and skips `None`).
- **The metagraph** is read at head (`registered_hotkeys`), as the intake's
  registration gate does. A boundary-pinned metagraph would need a public
  `ChainClient` read at a block (owner call; not added here).
- **A miner fault becomes malformed forecasts.** This is the settlement's own
  `checkpoint_fault` path. It is not a guessed input: the entry is closed,
  never scored.
- **An absent intake queue** is an empty queue (the intake's own reading).
  An unreadable one is refused.
- The eval-host transport, the grants a cache miss uses, and the testnet half
  of `Runs in` are not covered here.

## Failure on main

Every source and eval-host test fails with `ModuleNotFoundError` for
`cascade.validator.lane_d_inputs` or `cascade.eval.lane_d_forecasts` (both in
`PATHS`). The torch-free import check fails the same way, inside its
subprocess. The invariant 12 check and the `main()` AST check fail on
assertions. `test_inv12_nothing_on_the_weight_path_imports_the_source` passes
on main. Run locally with `-m "not torch"` (no test here needs torch).

## Correction (plan rev6.11 (b)): crown at the first verdict

The confirmation duel is dropped (owner 2026-11-23): a crown is decided at
the first verdict, first clear (max-T LCB vs KING_D) ∧ A10 ∧ the §1.3b gate.
The source lists X from B1 (its reveal is before B1), so X is crowned at B1.
`test_step_with_the_source_settles_like_the_hand_built_inputs` keeps its
report-for-report comparison with W8b's hand-built inputs at B1..B3 and now
reads the crown from the B1 report (with the armed set), asserts B2 and B3
crown nothing and that uid 6 is KING_D's court slot at B1, B2 and B3 (it
checked B2 only). Nothing is weakened. On main the old code still runs the
confirmation stage (X is crowned at B2), so the changed test fails on an
assertion until the W5e implementation lands.

## Public half (rev6.18 F4)

At `public_weight > 0` the source draws the public sources from the box's
`public_indexes` (inputs file table `public_indexes = {source = dir}`) and
the boundary block hash, asks the eval host (`public_scores`: SSH `python -m
cascade.eval.lane_d_forecasts public --public-root R`, R from the inputs
file's `eval_public_root`) for every needed digest's per-window scores on
the drawn series it reads from its own mirror, and returns
`LaneDInputs.public_indexes` / `public_scores`. `public_unavailable`: no
indexes, the host fails, or the read-time drops differ between digests. A
miner's own fault is NaN on every kept public window (`checkpoint_fault` in
the settlement). `test_w8c_public.py` pins it, including the host end
(`public_scores_entry`). The fault tests import `_forecast_ok` from
`cascade.validator.lane_d` (A7 deletes `lane_d_lowfreq`).

Hippius mirror (owner 2026-11-24, #894): `eval_public_root` is optional.
Without it the box sends `public` with no `--public-root` and the host reads
the drawn series from the Hippius mirror of the release's
`deploy/laned/public_mirror.toml` (`public_scores_entry(..., mirror=)`,
W6p `PublicMirror`; volume cache, then Hippius, then R2). Optional inputs
keys `eval_public_cache_dir` / `eval_public_cache_budget_bytes` become
`--public-cache-dir` / `--public-cache-budget-bytes`. With it, the local
mirror path is unchanged. `test_w8c_public_mirror.py`.

## rev6.14 (a): judging start vs eligibility boundary (owner 2026-11-23)

Option A (orchestrator 2026-11-23): the source reads the boundary block's on-chain timestamp with its chain reader and returns it as `LaneDInputs.boundary_ts` (W8b passes it to `settle_lane_d`); unreadable is `boundary_ts_unreadable`, never a guessed time. The fake reader answers a boundary block with W8b's fixture clock. `test_w8c_rev614.py`.
