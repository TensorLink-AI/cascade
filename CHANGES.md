# Group packing — review changes

Branch: `feat/group-packing`, based on `cascade-v2` commit `4f10049`.
Local commits only; no push, PR, deployment, paid run or standalone training job.
The only model updates executed are isolated CPU unit-test fixtures.

## Files and behavior

| Files | Change |
|---|---|
| `cascade/miner/verify.py`, `verify_worker.py`; `cascade/trainer/sandbox.py`, `sandbox_container.py` | Shard-aware full/quick verification uses the configured cores per child, affinity and BLAS caps in subprocess/container children. Caller environment and legacy CPU defaults stay unchanged. The quick check still has its 30-second local timeout. |
| `cascade/interface/generator.py` | Contract-gated groups, interface version 2, strict integer/shape/canonical-label checks before uint8 conversion; explicit `None` collapses to absence. `group_id` stays reserved. |
| `cascade/shared/config/chain_config.py`, `sections.py`, `training.py`, `validators.py`; `cascade/miner/verify.py` | Bind field acceptance and interface version to the effective contract; reject groups on backends without a consumer; offline launch-intersection checks keep only shared fields. |
| `cascade/generation/shards.py`, `tier.py`; `cascade/trainer/stream.py` | Schema-2 nullable group bytes, group-preserving tier and stream digests, recheck contract field acceptance. Schema-1 bytes stay unchanged. |
| `cascade/trainer/toto2_trainer.py` | Time crops preserve groups; equal-width batches emit int64 labels. Bare rows in mixed batches get one all-zero group. There is no channel padding. |
| `cascade/trainer/pyramid_model.py` | Group equality masks for variate SDPA; one independent shared copula latent per group. Omission/all-zero labels take the original unmasked/RNG path. |
| `cascade/trainer/pyramid_trainer.py` | Contracted dict batches, micro-batch group slices, replay hashes include labels; unchanged marginal/energy objective definitions and budgeting. |
| `cascade/trainer/channel_stats.py` | Partner comparisons stay within groups; dedup distinguishes grouping; singleton groups count as unpartnered. Grouped run telemetry reaches both result and done log. |
| `cascade/miner/templates/generator/generator.py`, `config.json` | Optional seeded packing with independent group drivers. Example size-bin weights 30/25/25/15/5; disabled by default. Last group is clipped to remaining channels. |
| `chain.toml`, `chain.testnet.toml`; `deploy/presets/cascade-moe-v2-{2m,7m,28m,120m}.toml`; `scripts/pyramid_config.py`; `cascade/website/data/benchmark-models.json` | Arm groups on new pyramid contracts and repin their model-source identities. Historical training sections stay unchanged. |
| `decisions/DEC-CA-0065-generator-group-packing.md`; `docs/INTERFACE.md`, `MODEL.md`, `ARCHITECTURE.md`, `MINER.md`; `AGENTS.md`, `agent.md`, both `llms.txt` files | Field semantics, agent guidance, release/lineage and audit limits. DEC-CA-0024's group-batching deferral is superseded; per-series `group_id` is untouched. |
| `tests/unit/test_verify_cpu_profile.py`, `test_pyramid_model.py`, `test_accepted_fields.py`, `test_record_carrier.py`, `test_shards_stream.py`, `test_v2_generator_channels.py`, `test_batch_denomination.py`, `test_channel_stats.py`, `test_digest_golden.py`, `test_pyramid_training.py` | Real sandbox pinning/record transport, masking/gradient isolation, sampling, tier/batch/trainer integration, schema bytes, digest goldens and source pins. |

## Digest changes

The optional groups vector uses the existing extended-record framing. Empty
`accepted_fields` still drops from `contract_digest`; historical values-only
corpus/stream/dedup/contract goldens are unchanged. Explicit labels participate
in the digest, including all-zero labels. Model parameters and tensor geometry
do not change, but source bytes do: the checkpoint guard requires a new release
and matching lineage. Rebuild/qualify CPU and worker images and a matching base;
do not relabel old Q21 checkpoints or bypass the guard.

Architecture digests (`base_arch_digest`), old → new:

- `chain.toml` (same-size mainnet/testnet pins share it):
  `66b8c5d8a426a24e52cd2c0642c6fd39cc3c8a09feeca1fc297fc92fba3ed51d` → `b19f700765825b3a30b26f83b9677b7fcacd65ab300e43e1eef2a6e255135228`
- `chain.testnet.toml` (same-size mainnet/testnet pins share it):
  `6aed15d788d32b849f574443181a423058db742637a75df1fdae4bcc6b0f865d` → `f836f54c7c5bf21ac2600f00f9b28b45461f67727e1a87ea268022ba326ff9e8`
- `deploy/presets/cascade-moe-v2-28m.toml` (same-size mainnet/testnet pins share it):
  `69dd4b8639e79b6ffa1378f86f608ec209e22d69c3d4cc19717894be08fdcb02` → `40150906d22d2725b2aadcb0a5d1d15ba81c8d8a05bd583babeb2165e42d1dec`
- `deploy/presets/cascade-moe-v2-7m.toml` (same-size mainnet/testnet pins share it):
  `9d1e6363c84ee3fa7c40293f10809aa92a5dc3044a84fc48796354729aa7eb7b` → `7a8cc0e9a66b54fa68344e8aface33f107b9fcd4743a93fbb7a99dd7b3372b74`

Contract digests, old → new (new model source pin plus `accepted_fields=["groups"]`):

- `chain.toml`:
  `0489d1fdffa5a6ead35b72c7232edbf456c27d8224aec33566ef00115ce1a4b5` → `e4443da58dca2d1eb86173531488c707bf6f804c77ce6936946d75a3f4cee293`
- `chain.testnet.toml`:
  `0b6b33e98738289ef2687e948a599e2a0ecfec380a16a30fcd623e9ac4495040` → `5fa9ab91161751e305a3c42aaba9bc8c4e583f64b91d14e4b6adb7dbddd77bd3`
- `deploy/presets/cascade-moe-v2-2m.toml`:
  `20e701cf9f343bf30d52ac2021ab337b825c9987608d3b565a34abac3902408c` → `d5375e0ec1a9a606f48eb8a77d59417df5510fb7314746e42a74ceb6a96e2b13`
- `deploy/presets/cascade-moe-v2-120m.toml`:
  `951fe988aa8215521f0656a27338bf31ca7c015f52e758b5d2cd0ae63c61c2a5` → `d229ac60cd69de47bdc0f72e7c37dc7a0d51cf0a2af7014101d6068c6dfc7067`
- `deploy/presets/cascade-moe-v2-28m.toml`:
  `9e7d811763be466c395d63dd412897a788913321cda2095e703316f2915286ef` → `67470dbc20542f639f79d1d397af7b4390ac802f600e8d58c8927f6e8f2c67d0`
- `deploy/presets/cascade-moe-v2-7m.toml`:
  `0b427410bc55218f5ce23e1b7341e0d5852cf5d81c3e46f5cdf257b4db00efe1` → `0ff8943f19830b874b07b5e5b0ab3c716ae74c5ef2602769ac4c1770210f395c`

## Verification

Full unit suite: **4,707 passed, 4 skipped** (7 warnings), 469.92 seconds.
Command: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/unit -q -o tmp_path_retention_policy=failed -o tmp_path_retention_count=1`.
Final focused regressions: **181 passed**, including real streaming sandbox
label transport. Deadline-helper/catalog rerun: **17 passed**, assertions
unchanged. Docs checker: **19 guides, 350 CLI snippets, zero issues**.
Ruff **0.15.20: passed**. No standalone training/GPU jobs were run.
Full output: [/tmp/group-packing-final-unit-all.log](/tmp/group-packing-final-unit-all.log)
and [/tmp/group-packing-final-focused.log](/tmp/group-packing-final-focused.log).

The pre-change values-only model forward/head/sample byte hashes and both
loss scalars matched after the model edit on this CPU runtime. An independent
old unmasked-attention reference, exact no-groups/all-zero losses and RNG-state
checks, and the original schema-1 writer comparison are in the tests. The
original starter and new disabled-packing starter produced byte-identical
values for 3 seeds × 5 yields. Historical Toto2/MoE model-source bytes match
`4f10049` exactly.

Complete run logs live in `/tmp/group-packing-*.log`. Earlier failures were in
new test fixtures: duplicate corpus rows violated the existing dedup gate;
a consumer stopped before legacy deferred stream accounting updated its last
row; a transport fixture omitted requirements and assumed the raw sandbox
applied the consumer's point budget. The fixtures now use valid seeded draws,
the existing eager-accounting protocol for that test, complete repo layout and
a bounded stream consumer. No production assertion was weakened. The first full suite also found a stale dashboard architecture catalog pin
(fixed by refreshing architecture metadata only) and an existing wall-clock
deadline assertion failed under concurrent CPU load. That timing assertion
remains unchanged; the final full run executes alone with controlled root
thread counts. Full output covering all development failure cases is retained in
[/tmp/group-packing-test-failures.txt](/tmp/group-packing-test-failures.txt).

## Review and integration limits

- No channel padding is introduced; buckets preserve exact C, and each label
  always has at least one channel. Groups do not connect different examples.
- Ordinary validators do not re-execute generators at settlement. Local verify
  re-derives both draws; tier-1 shard audits re-derive stored hashes and consumed
  stream digests including labels. Complete seeded corpus regeneration remains
  a separate operator audit, as before.
- The group-size example is a miner choice, not a required corpus distribution.
  Routing auxiliaries still summarize the full batch; isolation of model
  outputs does not mean independent optimization per group.
- Real GPU SDPA behavior, peak memory/throughput, production container images,
  and baseline qualification need an integration run. CPU tests do not establish
  those results. No training/GPU jobs or live state changes were run here.


## Follow-up: 32-channel generators and stronger private evidence

This separate follow-up to DEC-CA-0065 implements DEC-CA-0066. Review it as a
policy/eval change, separately from the group-packing architecture change.

- `chain.toml`, `chain.testnet.toml`, the generator template and current miner/
  agent/operator docs: require exactly **32 synthetic channels** per yield;
  groups may be any sizes inside it. Historical v1 keeps its earlier floor.
  Cached 8-channel corpora are rejected for new 32-channel legs. Champions are
  untouched; the launch baseline must now qualify at 32 channels.
- `cascade/pool/cli.py`, `builder.py`, `sources/*.py`: new pool builds default
  to **800/domain**, **400/domain-frequency**, up to **32 native eval channels**
  and source-family clustering. Narrow real data is retained; no padding or
  duplicate channels. Existing explicit cron flags still need updating.
- `cascade/pool/families.py`: curated catalog families, conservative crypto/FX/
  sensor/origin fallback keys and transitive clean/noisy-pair grouping. New
  metadata records the family in `source` and keeps feed provenance separately.
  Unresolved families are dropped with a visible reason. Historical snapshots
  and the legacy producer mode keep their recorded feed semantics.
- `cascade/shared/config/{v2.py,chain_config.py}`, `shared/v2_status.py`,
  `validator/{expert_lane.py,windows.py,pool.py}`: mainnet v2 verdicts request
  **4,000 windows**. The resolver preserves the old-era boundary and replay;
  testnet remains at its small budget. The single-horizon path no longer
  silently shrinks v2 evidence to the old jitter target or bag. Ladder rungs
  split the budget equally; actual eligible capacity can reduce it. Discovery
  reports the effective budget. No change to public benchmark definitions.
- `scripts/check_docs.py` renders the fixed channel count as 32. New and updated
  tests cover cached-width rejection, native 32-channel extraction, both cap
  bottlenecks, source-family transport through scoring/bootstrapping, historical
  draws, snapshot delegation and invalid window-budget configuration. See
  `test_eval_source_families.py`, `test_v2_verdict_budget.py` and the updated
  generator, forge, pool, admission and record integration tests.

No additional model-source/architecture or training-contract digest changes
occur in this follow-up. Generator policy, future pool content and the v2
private-eval configuration do change. The family rule is conservative rather
than proof of independence: more correlated rows do not become independent
samples, and the corrected confidence bound may widen.

Final full unit suite: **4,723 passed, 4 skipped** (7 warnings), 468.03 seconds;
[/tmp/cascade-c32-pool-final-unit-all.log](/tmp/cascade-c32-pool-final-unit-all.log).
Focused checks: **215 passed**; final CPU-profile/CLI retest: **5 passed**.
Docs: **19 guides, 350 snippets, zero issues**. Ruff **0.15.20: passed**.
The first full follow-up run exposed one stale fixture using the old 8-channel
example; the test now exercises the current 32-channel starter. Its acceptance
assertions remain intact. A portability regression also covers hosts without
Linux's affinity API: pre-import BLAS limits still apply without failing spawn.
No changes were made to pinned model-source bytes after the group-packing
commit. Failure-case outputs are retained at
[/tmp/cascade-c32-pool-test-failures.txt](/tmp/cascade-c32-pool-test-failures.txt).

No scrape, live pool rebuild/publication, cron edit, chain write, training job
or GPU job was performed. The estimated ~4,900-series pool remains a projection;
verify the actual eligible count, family coverage, cluster floors and future
snapshot activation in an authorized rebuild. Production 32-channel GPU
memory/throughput and signed image/baseline qualification remain outstanding.


## CPU generation observability and compressed private storage

- `generation/progress.py`, `tier.py`, `cli.py`, `transfer.py`, `stage.py`:
  bounded generation/hash/upload counters, scoped progress PUT, nonce-checked
  polling and parent-owned terminal state. Local snapshots are atomic;
  W&B and progress uploads run outside the generation work path. Upload
  observer failures do not change a successful transfer or corpus digest.
- `trainer/runner_generation.py`, `wandb_sink.py`: durable recent progress,
  exact queue-attempt bindings where available, restart discovery and a
  separate CPU W&B run. Late CPU replies cannot overwrite a new attempt or a
  terminal result. Independent W&B runs use `create_new`, including trainer
  events, so parallel logging does not close other active runs.
- `shared/v2_status.py`, `miner/live_status.py`, `submission_status.py`:
  public counters and CLI generation reports, with independent freshness,
  historical labels and explicit `generator_ref_only` fallback. A report is
  observability, never a training verdict or reward record.
- `generation/shards.py`, `shared/config/generation.py`, `chain.toml`:
  explicit schema-3 typed float32 byte-stream split plus Zstandard level 3;
  mainnet opts in. Legacy defaults preserve existing schema-1/2 bytes.
  Private corpus storage can use an explicit R2 endpoint/bucket and separate
  environment-named credentials. Mainnet still selects `primary`; no account
  or bucket was configured or accessed.
- `generation/dataset.py`, `lifecycle.py`, `trainer/loop.py`:
  anchored-receipt-bound observed king publication and conservative expiry.
  King marks precede public copying; failed publication retries safely.
  Active, crowned and unresolved corpora are retained. Upkeep is single-flight
  even if the outer deadline expires. This is not complete historical receipt
  reconciliation or orphan-upload cleanup; unmatched B/C corpora are retained.
- `pyproject.toml`, `uv.lock`: W&B floor raised to the already locked 0.28
  release, which supports independent simultaneous runs. Dependency resolution
  check passed offline; no package versions changed in the lock.
- `test_generation_observability.py`, `test_generation_storage.py`:
  lossless bytes/digests/groups, legacy writer equivalence, multiple Arrow
  batches, exact/stale/historical attempt reporting, scoped polling, observer
  failure isolation, W&B concurrency, private R2 routing, invalid config,
  receipt bindings, failed public copies and protected expiry.
- Operator, architecture, miner and agent entry-point guides explain the
  storage format, counters, privacy, disk requirement and deployment checks.

### Compression measurements

Local CPU fixtures: 16 series of 32 channels × 8,064 points each (16.5 MB raw),
three repetitions per method; readback checked exact stored float32 bytes.
These are local Parquet measurements, not production-VM or end-to-end claims.
No upload, fsync, training or GPU job was included.

| Fixture | Existing Zstd bytes / raw | Float32 BSS + Zstd bytes / raw | Smaller than existing |
|---|---:|---:|---:|
| White noise | 0.924 | 0.856 | 7.4% |
| Random walk | 0.916 | 0.759 | 17.1% |
| Trend and seasonality | 0.898 | 0.756 | 15.8% |

Median write times were 0.078–0.106 seconds for legacy and 0.119–0.157
seconds for BSS; read times 0.029–0.030 versus 0.037–0.051 seconds. Savings
cost some local CPU time and are generator-dependent. Raw evidence:
`/tmp/cascade-corpus-compression-benchmark.json` and its adjacent script/log.
50B float32 points are 200 GB raw. The current CPU path retains the full corpus
until hashing completes; size local disk with working-space overhead.

### Identity and deployment

No model-source or contract pins moved. Logical corpus digests and float32
values are identical between encodings; schema-3 Parquet file hashes differ.
An absent encoding setting still selects the legacy byte-identical writer.
Rebuild/qualify CPU and GPU reader images before enabling schema 3. A real
private-bucket upload/readback/retry run and a public W&B access check remain
unexecuted. Publication and cleanup were tested only with isolated stores and
synthetic receipts; no live data, keys, provider rentals or paid jobs were used.
No push or PR was performed.

### Verification for this follow-up

The first complete run reported **1 failed, 4,749 passed, 4 skipped**.
Failure: `test_backlog_recovery.py::test_cpu_vm_admission_waits_before_provisioning`.
The completion logger unconditionally read `.index` from a placeholder result
when observability was disabled. It now reads completion metadata only with
an observer and catches metric-extraction failures without failing successful
work. The admission assertions were preserved; a new regression explicitly
covers unreadable completion metrics. Full original failure output:
`/tmp/cascade-generation-unit-initial-failure.log`.

Latest focused coverage: **175 passed** in 9.12 seconds, including admission,
shards, status, storage, W&B and the new failure-isolation case. Log:
`/tmp/cascade-generation-final-focused.log`. Ruff, docs and offline lock checks
pass. Final complete suite: **4,754 passed, 4 skipped** (7 existing NVML
warnings), 454.62 seconds. Command:
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m pytest tests/unit -q -o tmp_path_retention_policy=failed -o tmp_path_retention_count=1`.
Full output: `/tmp/cascade-generation-final-unit-all.log`.
Docs: 19 current guides, 353 CLI snippets, 30 miner commands and 13 console
entrypoints, 0 issues (`/tmp/cascade-generation-final-docs.log`). Ruff:
`/tmp/cascade-generation-final-ruff.log`. Offline frozen-lock check:
`/tmp/cascade-generation-lock-check.log`.

## Final pass over live-test changes (`fc4fa27`)

Reviewed the four commits after `251a6a5`, including the new live-testnet
harness, its 32-channel fixtures, structured verify rejection and v2 king
harvest fix. Remote CI for `fc4fa27` passed. Additional local failure scenarios
confirmed defects outside that coverage; no live stage or rental was run.

### Findings and fixes

| Severity | Trigger and impact | Location and fix | Evidence |
|---|---|---|---|
| High | Carried king has no key; its operator-funded CPU build entered ordinary admission and could lose priority behind miner work. | `runner_rolling.py`: preserve king priority while preparing its fallback corpus, restoring thread-local state on failure. Reward/funding rules are unchanged. | Real king dispatch control flow with missing/working keys, operator-side failure and CPU fault. |
| High | Concurrent king/backup or lane harvests shared a seed/hotkey directory; one could overwrite or clean up another checkpoint. Pending upload records also shared one key. | `runner_common.py`, `loop.py`, `runner_funded_legs.py`: bind staging and pending records to source/role/size/init/contract/suffix; cleanup removes only that job. Matching legacy records stay readable; ambiguous records are held. Pyramid retries require contract binding. | Two actual concurrent harvest calls with a barrier retain distinct primary/backup bytes. Independent pending records survive another job's cleanup. |
| High | Live harness proceeded to paid legs after failed preflight/admission and could lose submit intent across a crash or state reset. | `tests/e2e_testnet/run.py`: preflight before paid work, block on failures, test-network-only flag, strict args, atomic state/report, preserve unresolved submit intent and require resume. | Reproduced paid-stage execution after a failed preflight; regression now prevents the call. Unresolved attempts never auto-resubmit. |
| High | Harness could accept rejected rounds or eligibility as settlement success, omitted signature verification and compared only reward UIDs. | Require anchored signed scored receipts containing the exact submitted challenger ref; compare normalized weight values within bounded u16 rounding. Search indexed receipt history; explicitly skip current-weight checks after a newer receipt. | Reproduced rejection accepted as success; tests cover wrong ref, eligibility-only, signer/round mismatch, different values on the same UIDs and non-finite weights. |
| Medium | CLI response moved to `data.submissions[]`; harness kept polling obsolete top-level fields and could wait for six hours after completion. | Parse the exact lane/hotkey/ref view and require its own freshness. | Reproduced `done` reported as unknown; current, stale and wrong-attempt evidence tested. |
| Medium | Failed Lium listing looked like an empty account; Vast/Runpod inventory was absent and network prefix hardcoded. | Failed/malformed listings remain unverified; inspect all four providers with the configured testnet netuid. | Reproduced empty inventory on nonzero CLI exit; provider adapters and prefix verified with isolated fakes. |
| Medium | Short literal `exec` payload bypassed the packed-source size threshold; the harness treated it as a known warning. | `static_guard.py`: inspect literal code evaluators even below the arbitrary-string threshold, keeping data strings and expression literals compatible. The fixture is now a required rejection. | Literal blocked import rejected in scan/reject modes; ordinary short strings and `unpack=False` behavior retained. Runtime-assembled code still needs the sandbox. |
| Medium | Nondeterminism fixture imported blocked `os`, so rejection did not test determinism. | Use an unseeded NumPy generator without that blocked import. | Fixture now reaches the intended deterministic-output check. |
| Documentation | Operator guide still required B before C, and a TOML comment said king training always used operator funding. | Correct A → C → B, first-king funding/recovery, per-job harvest identity, optional miner packing and Targon deployment coverage. | Docs checker and source-backed control-flow review. |

The initial four harness reproductions all failed on the reviewed remote code:
`/tmp/cascade-live-review-reproductions.log`. A first new king test fixture used
a read-only property incorrectly; the fixture was corrected without changing
its funding or admission assertions. The confirmed-defect reproduction output remains preserved at the first log
above; corrected king fixtures pass in the final focused and complete runs.

### Funding and packing rules verified

CPU generation is operator-paid. Mainnet v2 tries the current king's deposited
provider key for its GPU leg; absent/unfunded/revoked credentials fall back to
the operator. No new challenger fund request is required. A key-caused fallback
retains `funded_by="operator"` and the existing 1%-of-A key-lapse policy,
including for the carried king. Operator/provider recovery uses
`operator_fault` without opening a lapse. No first-leg grace exception was
introduced. If the operator also lacks funds/capacity, king retries and its
dependent settlements wait. Testnet's default king_ladder=false exercises
operator funding, not mainnet key penalties.

Miners choose any canonical optional grouping of exactly 32 channels. The
30/25/25/15/5 mix is an example of draw weights, not a required or calibrated
realized distribution. Mainnet's 512 sequence budget gives 16 examples × 32
channels, split into eight micro-batches; series-count equality is not a
measured GPU-cost guarantee. Existing model masks and group copula behavior
were rechecked; no model source/architecture pin changed.

### Targon scope and remaining launch checks

- `--private-eval-hosts` routes A private verdicts, king checks and B checks to
  the prepared SSH GPU host; histories cross, targets/scoring stay local.
- `--eval-hosts` routes all four public suites: GIFT-Eval, BOOM, TIME and FEV.
  Configured private inference holds if its host is missing; public benchmarks
  retain their documented local-device fallback.
- MarketCast C uses the main validator's own `--device`, not either hosts flag.
  Targon GPU provisioning is manual; CPU generation has its own automatic
  provider. The running service flags/hosts were not inspected or changed.
- A real Q21 pyramid checkpoint, made from initialized CPU fixture weights
  without training, passed the private request/response protocol and exact
  sequential/parallel qualification. GPU numerical/runtime qualification on
  the actual Targon hardware remains required.
- Checked-in launch image/source pins and the v2 baseline proof still require
  the documented deployment qualification. These code edits change the worker
  source fingerprint and verifier identity: rebuild/repin the deployed image
  and requalify the baseline; do not substitute invented registry digests.
- Default streaming testnet does not exercise the production CPU shard/R2
  path, public W&B access or mainnet king_ladder behavior. Those require a
  separately configured authorized integration run. GPU memory/throughput for
  the production 1.23B model remains a hardware measurement.

Focused checks: 129 harvest/recovery/king tests passed, and 92 final
packing/private-Q21/harvest/harness checks passed. Ruff and docs passed.
Final complete unit suite: **4,777 passed, 4 skipped**, 8 NVML warnings,
527.61 seconds. Command:
`PYTHONPATH=/tmp/cascade-v2-live-review-20261027 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/tensorlink/cascade-v2/.venv/bin/python -m pytest tests/unit -q -o tmp_path_retention_policy=failed -o tmp_path_retention_count=1`.
Logs: `/tmp/cascade-live-review-full-unit.log`,
`/tmp/cascade-live-review-final-focused.log`,
`/tmp/cascade-live-review-harvest-focused.log`,
`/tmp/cascade-live-review-ruff.log`, `/tmp/cascade-live-review-docs.log`.
The docs checker covered 19 guides, 353 CLI snippets, 30 miner commands and
13 console entrypoints, with zero issues.
No push, deployment, chain submission, provider rental or live-state change
was performed in this pass.
