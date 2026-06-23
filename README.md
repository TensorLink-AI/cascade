# metronome — synthetic time-series data subnet

A Bittensor subnet where miners compete on the quality of **training data**, not
models. It is the dual of [`horizon`](../horizon): horizon scores trained TSFMs
that miners submit; metronome holds the *training process* fixed and scores the
**data generators** that feed it.

The fixed process is **a Toto2-4M backbone trained from random initialisation**
([Datadog/Toto-2.0-4m](https://huggingface.co/Datadog/Toto-2.0-4m), arXiv
2605.20119) — *not* a fine-tune of released weights. Training from scratch is the
point: the corpus is then the only source of learned signal, so the downstream
forecast skill measures the *data*, not what some pretrained checkpoint already
knew. Toto 2.0 itself is 57.5% synthetic data with zero public series in
pretraining and still tops GIFT-Eval — metronome turns that synthetic-prior
design into an open competition.

## How it works

```mermaid
flowchart TD
    subgraph miner["Miner (no GPU)"]
        gen["generator.py<br/>(DataGenerator)"]
        commit["commit on-chain pointer<br/>metro-v1:gen:hf:repo@sha"]
        gen --> commit
    end

    subgraph trainer["Trainer — owner-operated (the GPU boundary)"]
        resolve["resolve commitments →<br/>pick king + challenger"]
        seeds["derive one shared RoundSeeds<br/>from block hash<br/>(generation_seed + training_seed)"]
        corpusK["draw corpus<br/>(king's generator)"]
        corpusC["draw corpus<br/>(challenger's generator)"]
        trainK["train Toto2-4M<br/>from random init"]
        trainC["train Toto2-4M<br/>from random init"]
        manifest["publish signed TrainingManifest<br/>(2 ckpts + corpus/contract digests)"]
        resolve --> seeds
        seeds --> corpusK --> trainK --> manifest
        seeds --> corpusC --> trainC --> manifest
    end

    subgraph validator["Validator (eval GPU)"]
        gate["verify signature +<br/>matching contract / base-arch digests<br/>(controlled-experiment gate)"]
        eval["pull both ckpts → score on<br/>shared held-out windows<br/>(CRPS/MWSQL + MASE)"]
        koth["paired-bootstrap LCB of<br/>geomean(CRPS, MASE),<br/>challenger vs king → KOTH verdict"]
        weights["winner-take-all<br/>weights on king's UID"]
        gate --> eval --> koth --> weights
    end

    commit -->|on-chain| resolve
    manifest -->|manifest + HF ckpts| gate

    classDef invariant fill:#fff3cd,stroke:#d39e00,color:#5c4400;
    class seeds,gate invariant;
```

> The **highlighted boxes** are where the controlled experiment lives: the trainer
> reuses one `RoundSeeds` for both runs, and the validator's digest gate rejects any
> manifest where king and challenger didn't share that contract. Details below.

The **central invariant**: in a round, the king's generator and the
challenger's generator are trained into models under a *byte-identical* contract
— same Toto2 architecture and random initialisation, same compute budget,
optimiser, generation seed, and training seed. The only thing that differs is the
generator code. So the downstream eval is a controlled measurement of **data
quality**, not a confound of data + luck + hyperparameters. Because the run
starts from noise, the contract pins the *whole* recipe (see `chain.toml
[training]`). Each model trains for a fixed **wall-clock budget (~3h on the
owner's reference GPU)**, enforced as a fixed token count (`hours × reference
throughput`) so king and challenger get **identical compute** — a raw timer would
let a generator win by emitting cheap-to-step data rather than better data, and
wouldn't reproduce on a re-derived audit run.

A challenger only takes the throne after winning **`dethrone_cp` consecutive
rounds** by a confidence-bounded margin (paired bootstrap LCB clears the
tenure-adjusted win margin). Weights are pure winner-takes-all.

## Three roles

| role | package | needs GPU | needs chain |
|------|---------|-----------|-------------|
| **miner** | `metronome.miner` | no | to deploy |
| **trainer** (owner) | `metronome.trainer` | yes | to read king / sign manifest |
| **validator** | `metronome.validator` | yes (eval) | to set weights |

## Layout

```
metronome/
  interface/   miner-facing contract (DataGenerator ABC, output checks, static guard)
  eval/        scoring math: CRPS (MWSQL), MASE, paired bootstrap, KOTH decision
  trainer/     owner GPU service: corpus build, fixed contract, train+upload, manifest
  validator/   manifest gate, checkpoint evaluator, KOTH state machine, weights
  miner/       miner CLI: verify, deploy
  shared/      config loader, HF fetch/upload, chain client, manifest schema

docs/
  ARCHITECTURE.md   end-to-end flow, trust model, the controlled-experiment invariant
  INTERFACE.md      the DataGenerator submission contract for miners
scripts/
  example_generator/   a forkable reference generator (also a test fixture)
```

## Console scripts

After `uv sync` / `pip install -e .`:

* `metronome verify <repo_dir>` — run every check the trainer runs (layout,
  static guard, hash-locked deps, **and the determinism check**: your generator
  must produce a byte-identical corpus at a fixed seed).
* `metronome deploy <hf_repo> --revision <40-char-sha> --wallet-name ... --wallet-hotkey ...`
  — commit `metro-v1:gen:hf:<repo>@<sha>` on-chain.
* `metronome-trainer` — the owner training service (`--offline` for a config/seed smoke).
* `metronome-validator` — the validator loop (`--offline` for a state smoke).

Before launching, set `chain.toml [subnet] netuid`, `[training] base_arch_digest`
(sha256 of the frozen base architecture), `[manifest] trainer_hotkey`, and the
repo identifiers.

## Quick start

```bash
pip install -e .                 # core (numpy/scipy/huggingface-hub)
pip install -e '.[dev]'          # + pytest/ruff
python -m pytest tests/unit -q   # pure-numpy tests, no torch/HF/chain needed
```

The Toto2-4M from-scratch training is the **owner's** to implement behind the
`metronome.trainer.contract.BaseTrainer` protocol (the GPU boundary) — see
`docs/ARCHITECTURE.md`. Everything above that boundary is numpy/CPU and tested.

## License

MIT
