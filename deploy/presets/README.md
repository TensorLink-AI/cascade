# Model presets

The existing [Cascade MoE 28M preset](cascade-moe-28m.toml) belongs to the
currently implemented model. Its source digest and historical checkpoints
remain valid for replay.

The next model family has five reserved variants. These files are placeholders
for the model definitions you will provide. Sizes are **nominal active parameter
targets**, not measured counts. Total parameter counts and expert layouts have
not been assigned.

The smaller variants are intended for miners to improve generators with less
GPU memory and shorter local experiments. Once the models are populated, compare
candidate and baseline on the same small variant, budget, seeds and held-out
pool. Test promising generators again on the effective competition model before
proposing a submission. Improvement on a small model does not establish a live
win, and a checkpoint from a different architecture is not a compatible warm
start. Adding these variants does not require every live entry to train all five.

| Target active parameters | Reserved architecture ID | Placeholder |
|---|---|---|
| 2M | `cascade-moe-v2-2m` | [2M](pending/cascade-moe-2m.toml) |
| 7M | `cascade-moe-v2-7m` | [7M](pending/cascade-moe-7m.toml) |
| 28M | `cascade-moe-v2-28m` | [28M](pending/cascade-moe-28m.toml) |
| 56M | `cascade-moe-v2-56m` | [56M](pending/cascade-moe-56m.toml) |
| ~120M | `cascade-moe-v2-120m` | [~120M](pending/cascade-moe-120m.toml) |

Each placeholder deliberately omits tensor dimensions, source and checkpoint
digests, training recipes and throughput estimates. Configuration loading
rejects its metadata in both primary and additional-size contracts, including
a future `[v2.training]` table. These files do not change the active model,
enable additional training jobs or make any performance claim.

To populate a variant, supply its model implementation, exact configuration,
checkpoint layout and source identity; qualify initialization, training,
evaluation and Lane B's expert selection/merge behavior. Measure the reference
throughput and memory requirements, then supply the complete training contract.
Remove `implementation_status` and `target_active_parameters` from that contract
only after qualification, and release that ID from `UNIMPLEMENTED_ARCH_PRESETS`
in `cascade/shared/config/training.py`. Selecting a reserved name alone is also
blocked, so it cannot label the old backend as a new model.
Activate it through a reviewed contract release and
new lineage. Keep the old implementations and pins available for replay.
