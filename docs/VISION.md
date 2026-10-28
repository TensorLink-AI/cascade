# Cascade: an open training stack for time-series foundation models

Time-series foundation models such as Chronos, TimesFM and Toto are built the
way large language models are: one lab runs a full training stack. It
assembles data, pre-trains a general model, refines it with targeted
mid-training and post-training, and fine-tunes it for the applications people
actually use. Each stage has its own data, its own recipe and its own tests.

Cascade's vision is to run that entire stack in the open, as a competition on
Bittensor. Every stage becomes a lane. Contributors propose what goes into
the stage, the network trains under a fixed contract, and only measured,
signed improvements are kept. Each accepted improvement becomes the starting
point for the next stage and the next era. Improvements build on
improvements: that is the cascade.

## Why compete on the inputs

Most decentralised training asks contributors for weights or gradients.
Cascade asks for something smaller and more inspectable: the inputs to
training. A synthetic data generator is a short Python program that encodes
a hypothesis about how real series behave: shifting seasonality, regime
changes, coupled channels, noise. A fine-tuning recipe is a declared schedule
and dataset. Because the model, budget, seeds and starting checkpoint are
fixed for each comparison, the only thing that differs between two entries is
the idea itself, and held-out real data decides whether it worked.

## The stack, lane by lane

**Data.** Every stage starts with data. Miners write deterministic,
sandboxed generators that emit 32-channel series, and the network generates
their training shards on CPU before any GPU is rented. Over time, curated
real data and selections from prior training corpora join the synthetic mix.

**Pre-training: Lane A.** Lane A trains the full model, Cascade MoE Base, a
1.23B-parameter mixture-of-experts forecaster with 112M parameters active per
token, a 21-quantile output head and attention across channels. Each
challenger's generator trains the model, which is then compared with the
reigning king on a private, rotating pool of real series and on the
GIFT-Eval, BOOM, TIME and FEV public benchmarks. A winner takes the throne,
and kings that keep passing scheduled checks are promoted to become the
shared starting checkpoint for the next era. Pre-training is not one long
run; it is a lineage that compounds across many independently funded runs.

**Mid-training: Lane B.** A mixture-of-experts model routes each input to
specialist subnetworks. Once the shared model's routing has stabilised,
Lane B lets miners target those specialists. A miner declares up to three
targets (domains such as energy, healthcare or transport, or short, medium
and long forecast horizons) and supplies a generator for them. The trainer
selects the experts the model actually uses for those targets and trains
only those, leaving the rest of the model frozen.

**Post-training: merging and distillation.** A Lane B update must improve its
targets without regressing any other domain, horizon or public benchmark. It
must then pass merge acceptance before it enters the next shared checkpoint,
so the general model gains specialist skill one verified update at a time.
Distillation extends this stage: a fourth lane brings in real data and
teacher forecasts from a panel of leading foundation models, compressing
their knowledge into Cascade's own model and into smaller versions of it.

**Fine-tuning: Lane C.** Lane C turns foundation models into applications.
Miners submit fine-tuning recipes for a pinned base model, and each
fine-tune is judged only on observations that did not exist when it was
trained, so it cannot have memorised the answers. The first competition,
MarketCast, forecasts the quantities that market-facing subnets depend on,
such as return distributions and realised volatility. A fine-tune must beat
the untouched base, stay well calibrated, and confirm its gain on fresh data
before it earns a slot.

## What every stage shares

- **Measured, not claimed.** A contribution advances only when a paired
  comparison's lower confidence bound clears a non-zero margin, with guards
  on everything it was not trying to improve.
- **Miner-funded compute.** Each contributor rents GPUs on their own provider
  key, choosing their hardware and price limits. Infrastructure faults are
  retried, not penalised.
- **Verifiable by anyone.** Training manifests, evaluation receipts and
  promotions are signed and hash-chained. Satellite validators replay them,
  and an audit tool re-checks every decision.
- **Rewards follow results.** Each lane has its own share of emissions, and
  anything not earned is burned rather than redistributed.
- **One stage at a time.** Each lane activates independently, so the stack
  can grow without disrupting the stages already running.

## Rollout

Lane A, pre-training on Cascade MoE Base, launches first, targeting
**30 October 2026, 12:00 UTC**. Lane C's application fine-tunes and Lane B's
specialist experts follow, each activated at its own era boundary.
Distillation, real data, fine-tunes built on Cascade's own model and a
family of smaller models complete the stack.

The goal is a forecasting model that keeps improving because anyone can
contribute the next good idea, and every idea has to prove itself on real
data before the model keeps it.

## Get involved

- Explore the research: [the competition introduction](../agent.md)
- Mine: [miner guide](MINER.md) and [agent walkthrough](AGENT_MINING.md)
- Validate: [validator guide](VALIDATOR.md)
- Audit the evidence: [audit guide](AUDIT.md) and the
  [public dashboard](https://s3.hippius.com/cascade-manifests/index.html)
