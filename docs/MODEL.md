# Cascade MoE Base: model and training recipe

**Cascade MoE Base - 1.23B** is the v2 candidate, with 1.234B total
parameters and 111.65M active per token. Smaller variants share its structure so
miners can test generators with less local compute. Architecture availability
and deployment readiness are separate: consult fresh `cascade status --json`.
The worker image, CPU generation image, initial generator and GPU behavior need
qualification before this candidate can run live.

## Sizes

| Model (total parameters) | Width / heads | Dense FFN | Expert / shared hidden | Total parameters | Active per token |
|---|---|---:|---|---:|---:|
| Cascade MoE Base - 17.53M | 128 / 2 | 276 | 23 / 92 | 17,528,416 | 2,031,200 |
| Cascade MoE Base - 68.92M | 256 / 4 | 552 | 46 / 184 | 68,919,520 | 6,930,656 |
| Cascade MoE Base - 273.64M | 512 / 8 | 1,104 | 92 / 368 | 273,642,976 | 25,687,520 |
| Cascade MoE Base - 1.23B | 1,088 / 17 | 2,346 | 196 / 784 | 1,234,189,344 | 111,651,872 |

Model names use total parameter counts, rounded in millions (M) or billions (B).
These counts include routers, shared experts and output heads. Active counts
subtract the unselected routed expert matrices; they are not GPU memory
estimates. [Preset instructions](../deploy/presets/README.md) explain smaller-model
experiments. No 56M definition has been supplied.

## Architecture

All sizes have ten blocks and 32-point patches. Training windows have at most
4,096 points per channel. V2 generators emit 8–32 channels; the forecaster also
accepts individual real series during evaluation. Channels stay together through
attention so the model can learn their relationships.

| Block | Attention axis | Feed-forward layer |
|---|---|---|
| 0 | Causal time | Dense GELU |
| 1, 2, 3 | Causal time | 128, 256, 256 routed experts |
| 4 | Across channels | Removed |
| 5, 6, 7, 8 | Causal time | 512 experts each |
| 9 | Across channels | Removed |

Each MoE layer selects 8 experts. Experts form 8 contiguous groups: rank each
group by its top two selection scores, retain four groups, then select the top
8 experts within them. Affinity is `sqrt(softplus(router_logits))`; logits are
computed in float32. A checkpointed bias affects selection only. The selected
unbiased affinities are normalized into mixing weights. An always-on shared
expert is added. No token is dropped or redirected to fit a capacity limit.
Sparse dispatch falls back to processing occupied experts individually when a
padded batch would allocate excessive memory under an imbalanced router.

The patch stem adds a linear projection of `[values, missing flags]` to a causal
convolution branch. That branch uses `[visible values, observed flags]`, two
sequential 128-channel convolutions (kernel 7, dilations 1 and 4), GELU, patch
average pooling and a projection. A residual SiLU MLP and a zero-initialized
three-role embedding complete the stem; this recipe uses target role zero.

Every block uses affine-free LayerNorm with epsilon `1e-4`. Time attention has
xPos positions, per-dimension query scaling, a `1/head_dim` logit scale and a
zero-initialized gate `2 * sigmoid(gate(h))`. Cross-channel attention has no
position or gate. Its entire FFN sublayer and FFN residual attenuation are absent.
Scaled residual coefficients follow the fixed 128-patch, depth-ten formula in
the [machine-readable specification](../deploy/presets/pyramid-ladder.json).

## Masking, normalization and targets

CPM hides contiguous spans independently per channel. Draw a masked fraction
uniformly in `[0, 0.4)`, then spans of 1–16 patches until it is reached, with a
bounded number of placements. Hidden input points are zeroed and excluded from
causal location/scale statistics. Statistics use float64, an eight-observation
minimum and a `1e-5` scale floor. Binary visible rows use location zero and scale
one. Input values are transformed by `asinh` and clamped to ±64.

The output at patch `p` predicts the 32 points of patch `p+1`, using patch `p`'s
last location/scale as its anchor. Supervision includes both observed and masked
targets. This recipe accepts values only; generator record masks and covariate
roles are refused rather than silently ignored.

## Unified output head

Final LayerNorm and a residual SiLU MLP feed one linear head with 11 values per
future point: median, eight spacing logits, a diagonal driver and one factor.
Positive `softplus(logit) + 1e-4` spacings produce monotone quantiles at levels
0.1 through 0.9, centered at the median. Quantiles are in linear standardized
units. Only the pinball loss transforms them with `asinh`; inference rescales
them directly, without a second `sinh`.

Copula drivers use the first eight DCT modes. Diagonal scales have constant RMS
across each 32-point patch; factor magnitudes are equalized while preserving
their patch energy. Independent temporal noise has covariance
`0.99 * exp(-0.5 * ((t-t')/8)^2) + 0.01 * I`. A rank-one factor draw is shared
across channels. Drivers are normalized to unit marginal variance and mapped
through a linear spline in normal-score coordinates, with linear tails.
The validator wrapper uses these joint samples and a deterministic window seed.
Benchmark quantiles come directly from the monotone head. Neither path sorts
quantiles or clips sampled paths to conceal a failed forecast.

## Optimizer and schedule

Master weights stay float32; CUDA model forwards use bfloat16 autocast. Router,
head calculations and losses use float32; statistics use float64.

NorMuon covers 2-D matrices and each routed expert matrix separately, excluding
the patch linear, head and routers. It uses Nesterov momentum 0.96, row second
moment 0.999, five bfloat16 Newton–Schulz iterations with coefficients
`(3.4445, -4.775, 2.0315)`, norm restoration, a matrix shape factor and decoupled
weight decay `2e-8`. AdamW covers the remaining parameters, with betas
`(0.91, 0.972)`, epsilon `1e-8` and decay 0.05; routers have no decay.

At width `d`, the matrix LR is `0.002125 * (d/512)^0.42`. The head AdamW LR is
`0.000425 * 512/d`; other AdamW groups use 0.000425. These rates already include
the research recipe's 0.85 multiplier. Slow-router settings, when enabled, act
on the router group and selection-bias update; expert-subset legs freeze both.

The supplied full recipe has 25,000 trunk updates and 2,500 cooldown updates,
with an 80-update warmup and a linear cooldown ending at zero. Its global batch
is 64 groups × 8 channels × 4,096 points: 512 channel rows. Competition contracts
count channel rows (`batch_denomination = "sequences"`) so wider groups reduce
the number of groups per update. `training_steps` sets an explicit maximum;
`ref_throughput_tokens_per_s = 0` records that throughput is unmeasured. The
point budget and wall limit remain enforced. A short or exhausted corpus may
end earlier; metrics distinguish incomplete training, exhaustion and a deadline.
Mainnet keeps the existing 50B CPU corpus cap and therefore uses at most 23,841
updates (49,998,200,832 points at the reference geometry), with an 80-update warmup
and a final linear cooldown. Local presets retain the full 27,500-update research
recipe. Testnet uses 16 updates on Cascade MoE Base - 17.53M. Shorter local `--train-hours` runs scale the
explicit update budget and schedule.

The marginal objective is ten times the mean pinball loss, summed over all nine
levels in asinh coordinates. The joint objective is ten times the fair energy
score from four copula samples per next-patch 32-vector, divided by `sqrt(32)`.
Two gradients are accumulated separately. The joint gradient receives weight
`min(0.1, 0.1 * norm(marginal_gradient) / norm(joint_gradient))`; global norm
clipping at 7 follows their combination. Router auxiliaries average the
`1e-4` balance and `1e-3` z-loss terms across MoE layers. Selection biases update
proportionally to global-batch load error and are re-centered after each step.

The scored checkpoint is the cooldown endpoint. The pre-cooldown weights and
named optimizer state form the warm-start lineage. Recovery replays the exact
input prefix, checks its digest and runtime, restores RNG/optimizer state and
continues without adding updates. An expert leg starts from scored king weights;
only its selected rows can change or merge.

## Implementation and qualification

| Source | Role |
|---|---|
| `cascade/trainer/pyramid_model.py` | Self-contained architecture, normalization, routing, monotone head and copula |
| `cascade/trainer/pyramid_trainer.py` | Dense CPM losses, gradient cap, scheduling, checkpoint export and recovery |
| `cascade/trainer/pyramid_optimizer.py` | NorMuon and width-corrected parameter groups |
| `cascade/trainer/pyramid_wrapper.py` | Standalone quantile and joint-sample inference |
| `cascade/trainer/expert_training.py` | Compact selected-expert training |
| `cascade/eval/checkpoint_guard.py` | Trusted code, config and tensor checks |
| `scripts/pyramid_config.py` | Complete offline miner configuration |

The old Toto2 and flat-expert MoE sources remain byte-identical for replay.
This port uses a single device with configurable gradient accumulation. The
research run's four-GPU expert-parallel transport and immutable research data
loaders are not imported. Training uses miner-generated data. GPU memory,
throughput, multi-GPU operation and full public-suite results must be measured
on the qualified production image before launch; parameter counts and CPU
regression tests do not establish those results.
