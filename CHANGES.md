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
