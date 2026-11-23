# W3b6a acceptance tests: the lane D bench sweep transport

W3b6a (plan rev6.13, "W3b6 spec" and the W3b6a row; owner 2026-11-23
"build this") depends on W3b2, W8c, W15 and W4. These tests are the spec.
The builder may touch only the paths in `PATHS`. The W3b2, W8c, W8b, W15
and W4 suites must stay green.

## The gap

W8b's step passes `bench=` (W8c's `source.bench`). `load_inputs_services`
returns `bench_host=None`, though: "no lane D bench host transport exists
yet". So an armed lane D holds every crowning candidate (`bench_hold`) and
can't crown. A lane D checkpoint can't reach the bench pod the way lane A's
does (box `scp`), because §10.2 keeps weights off the box.

## Contract

### `cascade/eval/lane_d_bench_sweep.py` (new, bench-host side)

```
REQUEST_SCHEMA = "lane-d-bench-request/1"
REPLY_SCHEMA = "lane-d-bench-reply/1"
LaneDBenchSweepError(RuntimeError)   .code: "checkpoint_unavailable" | "sweep_failed"
serve_sweep(request, *, cache, work_dir, contract=None, expect=None, sidecar=None,
            fingerprint=None) -> reply
python -m cascade.eval.lane_d_bench_sweep sweep --cache-dir D --budget-bytes N
        [--chain-toml F] [--work-dir W]     # request JSON on stdin, reply JSON line on stdout
```

- **request** = `{"schema", "digests": [str], "suites": [gate keys], "contexts":
  {digest: int}, "grants": {digest: grant | null}}`. Gate keys are `gift`,
  `boom`, `time` and `fev` (`benchmarks.ITEM_SUITE_KEY` values). Every
  digest needs a context (a scorer context) and a `grants` entry. Anything
  else is a `ValueError` and nothing is fetched or swept.
- **Per digest, in request order**, the cache follows the eval host's rule
  (`lane_d_forecasts._checkpoint`): a verified hit is used as is, and a miss
  with a grant runs W15 `prefetch_grant` (W4 `ingest_presigned` guards).
  - **Infrastructure fails the whole sweep** with `LaneDBenchSweepError(
    "checkpoint_unavailable")` and nothing is swept: a miss without a grant,
    a fetch failure (`fetch_failed`), or a vault / disk / `over_budget`
    fault.
  - **The miner's bytes:** a W4 refusal (any other `IngestError`, a fill
    that doesn't hash) makes that digest's rows `{suite: {"status":
    "error", "rows": []}}` for every requested suite. The digest isn't
    swept, and the others are.
- **Materialised** into `work_dir / f"{digest}-c{context}"`:
  `forecast_wrapper.py`, `model.py` and `weights.safetensors` byte-identical
  to the cache entry, and `config.json` equal to the entry's with
  `context_length` = the request's context. The dir passes
  `verify_checkpoint_code` (`cascade_pyramid`). The cache entry itself is
  never modified.
- **One paired pass:** `sidecar(dirs, suites)` is called once, with every
  materialised dir in request order and the sidecar suite names
  (`gift-eval`, `boom`, …). It returns one report or `None` per dir. Default:
  `benchmarks._invoke_sidecar_many` on the repo's `benchmarks/` project.
  Each report becomes `benchmarks.item_rows_from_report(report)`. A `None`
  is that digest's `error` rows. All `None` (the pass failed) is
  `LaneDBenchSweepError("sweep_failed")`.
- **reply** = `{"schema": REPLY_SCHEMA, "rows": {digest: rows}, "fingerprint":
  {...}}`. It is strict JSON (no NaN). `fingerprint` is the given dict, else
  probed (GPU model, driver, sidecar torch; the builder's choice of probe).
  Deterministic in (request, cache, sidecar).
- Torch-free at import.

### `cascade/validator/lane_d_bench_host.py` (new, box side)

```
SSHBenchHost(hosts_path, *, cache_dir, budget_bytes, grants=None, timeout_s=..., run=None)
host.set_contexts({digest: context})      # the source, per settlement
host(digests, suites) -> {digest: {suite: entry}}    # W3b2's Sweep
LaneDBenchHostError(RuntimeError)    # every refusal below (the gate holds)
```

- At every call it resolves the first `[[host]]` of `hosts_path`
  (`trainer.remote.load_hosts`) with `stage` in `final` / `any`, so a
  `bench_lease`-published file works. A missing file, or no such host, raises
  and runs nothing.
- A digest with no context set raises before anything runs.
- It sends the request above on stdin: `grants(digest)` per digest (`None`
  without `grants`), contexts and suites. No weights, histories or targets
  leave the box. It runs `<remote_python> -m cascade.eval.lane_d_bench_sweep
  sweep --cache-dir … --budget-bytes …` in `host.workdir` over SSH
  (`trainer.remote.build_ssh_argv`), through `run` (default
  `subprocess.run`).
- It reads the **last** stdout line. It raises on: a non-zero exit, a line
  that isn't JSON, another schema, `rows` not a mapping, a digest it didn't
  ask for, or an entry / suite entry that isn't a mapping. It never guesses
  a row.
- Any raise reaches `SweepSession.paired`, and `lane_d._run_gate` turns it
  into `bench_hold`.

### `cascade/validator/lane_d_inputs.py`

- `INPUTS_FILE_KEYS` gains the optional `bench_hosts` (a path). Absent, the
  returned `bench_host` is `None` (today's behaviour). Present, it is
  `SSHBenchHost(bench_hosts, cache_dir=eval_cache_dir,
  budget_bytes=eval_cache_budget_bytes, grants=<the W8c VaultGetGrants>)`.
- `source(boundary)` calls `bench_host.set_contexts(need)` when the bench
  host has it: once per settlement, with the same `{digest: effective
  context}` map it forecasts at, before returning.

## Consensus

The transport moves rows only. What decides is `paired_gate` on the three
records (W3b2), unchanged, and every fault is a hold or the miner's typed
`checkpoint_fault`. The bench host's GPU numbers aren't bit-reproducible:
W3b6b publishes and binds them so satellites verify instead of re-sweeping.
Lane A is untouched (no lane A module changes). Unarmed, no source exists,
so nothing runs.

## Fixtures

The cache entries carry the release's real pinned `cascade_pyramid` code
bytes, so `verify_checkpoint_code` passes. The sidecar, the SSH runner and
W4's `ingest_presigned` are fakes, and W8c's rig supplies the source. No
chain, storage, pod, torch or GPU.

## Failure on main

Every pod-side and box-side test fails with `ModuleNotFoundError` for
`cascade.eval.lane_d_bench_sweep` or `cascade.validator.lane_d_bench_host`
(both in `PATHS`). The torch-free import check fails the same way inside its
subprocess. The inputs-file tests fail on the `bench_hosts` assertions. The
source test fails on the `set_contexts` assertion, or on W8c's own
armed-docs assertion while the W8c armed-contexts implementation is unmerged.
