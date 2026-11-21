---
id: DEC-CA-0069
type: decision
title: "MarketCast recipe code hooks: a recipe may ship train.py (any loss, masking, parameter groups, curriculum) run by the MarketCast leg in a sealed child; the verdict, report and audit are unchanged"
status: proposed
date: 2026-11-16
tags: [marketcast, lanes, fine-tuning, recipe, sandbox, security]
revisit_when: "the sealed-child red team (§Qualification) finds an escape; or a hook pattern turns out to need data the info dict does not carry; or a MarketCast leg ever runs where operator credentials are present without the sealed child"
relations: {amends: [DEC-CA-0064], uses: [DEC-CA-0056, DEC-CA-0057]}
---

## Problem

A MarketCast recipe (DEC-CA-0064) is declarative. A miner chooses data, a mix,
horizons, an optimizer, a trainable scope and five augmentations. The training
itself is fixed:

- the loss is knot-weighted pinball plus one soft coverage penalty
  (`marketcast_leg.quantile_loss`);
- there is no masking at all;
- parameters form one AdamW group;
- one schedule runs the whole leg, with no stages;
- the trainable set is one of four scopes.

The fixed loss also differs from the metric. Training normalises each sample
by the mean |Δ| of its context. The verdict scores each suite's own CRPS:
`ohlcv` divided by the last context value, the other suites in their own units.
A miner cannot train on what is scored.

Owner, 2026-11-16: "we need the recipe — that's the value, the elegant way to
do it" and "we need unlimited flexibility on losses etc". The recipe stays the
unit a miner submits and the trainer runs. What it can express must not be
bounded by a menu we maintain.

## Decision

A MarketCast recipe tree may ship **`train.py`** next to `marketcast.json`
(and the optional `generator.py`). It defines any of the hooks below. Any hook
it leaves out keeps today's built-in behaviour, so a recipe without
`train.py` trains byte-identically to DEC-CA-0064.

```python
# train.py — every function optional; plain Python + torch + numpy
def param_groups(model, info) -> list[dict]:     # torch param groups: params + any opt kwargs
def optimizer(groups, info) -> torch.optim.Optimizer
def mask(context, target, rng, info) -> tuple[context, target]   # batch tensors in, out
def loss(pred, target, context, info) -> torch.Tensor            # scalar
def on_step(step, model, optimizer, info) -> None  # schedules, unfreezing, stages, curriculum
def on_end(model, info) -> None                    # e.g. EMA / SWA swap-in before export
```

`info` is a plain dict the leg fills. It holds:

- the batch's `suite`, `horizon`, `steps`, `freq_minutes`, the `levels`
  (knots) and the per-sample `synthetic` flags;
- the recipe's parsed JSON;
- `step` and `total_steps`;
- the eval's own loss helpers (`info["metrics"]`): the per-suite scored CRPS
  and its normalisation, from `cascade/eval/marketcast.py`, so a miner can
  train on exactly what the verdict scores.

The model is the pinned Timestral-28M module. Hooks may add trainable state
(adapters, heads used only in training), but the **exported checkpoint must
still have the base's exact tensor names, shapes and dtypes**. That is the
existing harvest rule (`same_structure`), so whatever a hook adds is either
merged into those weights in `on_end` or dropped.

### Where the code runs

The leg splits into two processes on the pod:

- **The trusted parent** (today's `cascade-train-worker --role mkt_finetune`):
  - loads and checks the base;
  - builds the training snapshot;
  - streams generator series from the existing generator sandbox;
  - after the child exits, checks and exports the weights and writes
    `marketcast_meta.json`.
- **The sealed child** runs the whole optimisation loop with the miner's
  hooks:
  - It reuses `cascade/trainer/sandbox.py`: the env allowlist, the network
    namespace (`sandbox_strict` refuses without one), rlimits, a memory scope
    and the `stream_gpu` GPU profile.
  - It sees only a private working directory holding the base weights, the
    training snapshot, the batch stream and its output directory.
  - Its wall clock is the leg's wall (the recipe's `leg_hours`).
  - Only `float32` safetensors and a size-capped JSON summary cross back, via
    files the parent reads with no unpickling.

The parent never imports `train.py`, and miner code never runs in the parent.

### Trust model (why this is safe)

- **The verdict does not depend on how the weights were made.** MarketCast
  scores the harvested weights paired against the base, on windows that start
  after the trainer-stamped `train_time` (forward-only, DEC-CA-0064). Training
  code can only produce weights, and bad weights lose. There is no target to
  leak, because the targets did not exist yet.
- **The trainer stamps everything the verdict binds**, never the pod:
  `train_time`, the commitment binding, `coldkey`, and the checkpoint sha256
  re-derived at harvest.
- **A payer pod is already the miner's hardware.** It is rented on the miner's
  Lium key (`runner_funded_pods`). It is `isolated = true` with
  `forward_env = ()`, and on the orchestrator-harvest path "nothing at all
  reaches the pod's environment". So code that runs only on the miner's pod
  gains nothing it did not have.
- **The sealed child is what makes the remaining cases safe:**
  - operator-billed recovery legs (`funded_operator_fallback`) run on OUR
    lanes;
  - the minted-credential path for a payer pod (a per-pod Hub robot) must
    never be readable by miner code.

  The child holds no credential, has no network and has its own filesystem
  view.

### Audit

`cascade-audit marketcast --reproduce` re-scores the published weights (base
and fine-tune quantiles, bar samples). It never retrains, so it is unchanged.
The recipe tree, `train.py` included, is published with the entry as today,
so anyone can re-run the training. A re-run is reproducible only as far as
the miner's code is deterministic. The leg still seeds torch, numpy and the
sampler, and still sets deterministic torch. Exact re-training was never part
of consensus.

### Bounds that stay

| Bound | Why |
|---|---|
| `leg_hours` ≤ `max_leg_hours` (miner-chosen wall) | the miner's pod bill; pod reliability over long runs |
| exact base structure + finite weights at harvest | the verdict compares like with like; unchanged |
| `train.py` ≤ 256 KiB, UTF-8, tree-size cap as today | no smuggled payloads; reviewable recipes |
| child memory scope + GPU profile | one leg cannot starve the pod's parent |
| at most `max_open_legs_per_coldkey` open legs | queue fairness; unchanged |

What a recipe declares for the data stays declarative, and real data still
comes only from the trainer's fetch, clipped at `train_time`. The parent
streams batches to the child, and `mask` / `loss` see those batches. A hook
cannot fetch, because the child has no network.

**Embargo-gap data (DEC-CA-0064 addendum 2026-11-19).** The `4hx42` / `1dx7`
horizons are judged on the embargo gap before the submission — days the
miner has seen, which only the leg has not. A hook must therefore never be
a channel for data: `train.py` is code under the size cap, but constants
in it (or in files beside it) could embed gap bars and train on them,
turning those cells into a memory test. Before hooks arm: refuse numeric
blobs above a small size in the tree, keep the child's inputs to the
parent's batches, and treat a hook-trained entry's gap cells as suspect if
a review finds embedded series (or judge hooked entries on forward-only
cells alone).

## What changes

- `cascade/shared/marketcast_recipe.py`:
  - `has_train_hooks(tree)`;
  - a static check of `train.py` (size, UTF-8, must parse, defines only known
    hook names at module level, no top-level side effects beyond imports and
    definitions) — a fast miner-side error, NOT the security boundary;
  - `recipe_digest` covers `train.py`'s bytes.
- `cascade/trainer/marketcast_leg.py`: `finetune` takes a `Hooks` object.
  Defaults = today's loss, AdamW, a single param group, `lr_at` and EMA, so
  no-hook recipes stay byte-identical (golden test).
- `cascade/trainer/mkt_child.py` (new): the sealed child entry point. It loads
  the base and the hooks and runs `finetune` on batches the parent writes to a
  bounded on-disk ring. It writes `out/model.safetensors` +
  `out/summary.json`.
- `cascade/trainer/worker.py` (`mkt_finetune`): with hooks present, spawn the
  child through the sandbox's GPU profile, wait on the leg wall, then harvest
  the child's output through the existing checks and export.
- `cascade/trainer/sandbox.py`:
  - a `train_child` profile: GPU, no network, private tmpfs workdir, bind
    mounts of only the base, the snapshot and the out dir;
  - with the user namespace unavailable, run the child as a dedicated
    unprivileged uid, or refuse under `sandbox_strict`.
- Miner CLI: `cascade lane-c check` runs the static check plus one CPU dry
  step of the hooks on a tiny synthetic batch, which is the miner's preflight.
- Docs: MINER.md (the hook interface and the `info` keys), AGENT_MINING.md (a
  worked custom-loss recipe), OPERATOR.md (the child profile).

No consensus term changes:

- the verdict, report (v3), receipt and snapshot formats are unchanged;
- `[marketcast]` gains only `max_train_hooks_bytes` (release-only);
- activation rides the existing `[marketcast] from_block`.

## Qualification (before testnet)

1. **Golden:** a recipe with no `train.py` gives bit-identical weights to
   today's leg (CPU, fixed seed).
2. **Equivalence:** a `train.py` that re-implements the default loss in user
   code gives the same weights as no `train.py` (≤ 1e-6).
3. **Red team on a real pod and on an operator lane.** The child must fail
   each of these, cleanly:
   - read the parent's environment or `/proc/<ppid>/*`;
   - read any credential file or the minted Hub robot;
   - open a socket (TCP, UDP, DNS, unix sockets outside its workdir);
   - write outside its out dir;
   - signal or ptrace the parent;
   - fill the disk past its quota;
   - outlive the leg wall;
   - leave a process after exit;
   - make the parent unpickle or `eval` its output.
4. **Hostile output:** wrong shapes, NaN, extra tensors, a meta field forged
   by the child (the parent writes the meta, never the child) → a
   deterministic `fault`, never an operator error.
5. **The Stage-1 offline replay** with a custom-loss recipe (train on
   `suite_crps`), scored end to end on real windows.
6. **One testnet leg** with hooks on a payer pod, harvested and judged.

## Must verify before arming

- Every MarketCast leg on a payer pod takes the orchestrator-harvest,
  no-credential branch of `runner_funded_pods` (MarketCast uses the ssh
  harvest, OPERATOR.md). Otherwise the minted Hub credential sits on a pod
  that runs miner code, outside the child but on the same host.
- Operator-billed MarketCast recovery runs hooks only in the sealed child.
  Refuse a hooked recovery leg on a host where the child profile is
  unavailable; never degrade to running it in-process.

## Rejected

- **A loss/masking menu of named blocks.** It is safe and bounded, but every
  idea a miner has becomes a feature request. The owner asked for unlimited.
- **A tensor expression language** (a JSON formula graph compiled to torch).
  It can express a lot, but it is still not unlimited, and miners would write
  losses in an awkward dialect instead of torch.
- **Uploading weights trained elsewhere.** This is the most flexible option,
  but it drops the recipe as the unit of value. The owner wants the recipe to
  be what is submitted and run.
- **Hooks in the parent process.** Miner code would sit next to the harvest
  and export code that decides what is uploaded.
