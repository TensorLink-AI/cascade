# Local iteration demo: small model, retired pool, bench preview

This walkthrough runs the whole offline loop on the 17.58M Cascade MoE Base
preset. You score a starter generator and a variant on a retired public
eval-pool snapshot, keep the checkpoints, and preview the public bench gate.
Nothing is spent or committed. Every number it prints is directional; the
private pool and the dethrone margin decide live duels
([how the verdict works](MINER.md#7-how-the-verdict-works)).

The example generator is the `cascade init` starter (the same template the
[miner guide](MINER.md#2-write-a-generator) forks), so the demo has no copy of
its own to drift.

Needs: a checkout with `pip install -e '.[train,hippius,chain]'` (torch for
training), ideally a CUDA GPU, network access the first time only, and about
50 MB of disk for the pool snapshot. The bench step also needs the sidecar
environment and staged benchmark data (step 5).

## 1. A small, complete configuration

```bash
.venv/bin/python scripts/pyramid_config.py --size 17.58m --out ./chain.local-17m.toml
```

This writes an offline configuration whose training contract is the 17.58M
preset (`batch_size = 512` sequences, so 16 series × 32 channels per step, and
`accepted_fields = ["groups"]`). It selects a cheap model for experiments, not
the live competition model.

## 2. A generator and a variant

```bash
cascade init ./demo-gen --lane a
cp -r ./demo-gen ./demo-gen-v2
# in ./demo-gen-v2/config.json set "packing": {"enabled": true, ...}
cascade verify ./demo-gen-v2 --quick --chain-toml ./chain.local-17m.toml
```

The starter emits exactly 32 channels per yield. The variant turns on its
optional `groups` packing, which this preset accepts. `cascade verify --quick`
prints the channel, length and batch geometry it checked. A configuration that
does not accept `groups` refuses the variant.

## 3. One score on a retired pool, keeping the checkpoint

```bash
cascade score ./demo-gen --chain-toml ./chain.local-17m.toml --pool-hf latest --train-hours 0.01 --device cuda --keep-ckpt ./demo-ckpt/base --json
cascade score ./demo-gen-v2 --chain-toml ./chain.local-17m.toml --pool-hf latest --train-hours 0.01 --device cuda --keep-ckpt ./demo-ckpt/v2 --json
```

`--pool-hf latest` lists `Tensor-Link/cascade-eval-pool`, picks the newest
`snapshots/<date>-block-<N>/` folder (never a legacy date-only rebuild),
downloads `block-<N>.tar`, refuses it unless its sha256 equals `POOL_SHA256`,
and extracts it to `.cascade-work/hf-pool/<sha256>/`. The tar is the scoring
source. The command prints a reminder that the pool is retired and is evaluation
data, never training data. A second run reuses the cache without the network.

The JSON document has this shape. Values vary by hardware, seed and snapshot;
none are shown here:

```json
{
  "schema": "cascade-cli/1", "command": "score", "ok": true, "exit_code": 0, "error": null,
  "data": {
    "geomean": "<float, lower is better>",
    "n_windows": "<int>", "n_series": "<int>", "corpus_digest": "<sha256>",
    "train_seconds": "<float>", "train_tokens": "<int>", "contract_digest": "<sha256>",
    "init": "random init",
    "pool": "hf:Tensor-Link/cascade-eval-pool/snapshots/<date>-block-<N> sha256=<12 hex> (retired)",
    "pool_snapshot": "<date>-block-<N>", "pool_sha256": "<sha256 of block-<N>.tar>",
    "checkpoint_dir": "./demo-ckpt/base",
    "lower_is_better": true
  }
}
```

## 4. Iterate: variant against the starter, same pool and seeds

```bash
cascade iterate ./demo-gen-v2 --baseline ./demo-gen --chain-toml ./chain.local-17m.toml --pool-hf latest --seeds 0,1 --train-hours 0.01 --device cuda --work-dir ./.cascade-17m --json
```

Both generators train and score under identical settings. Scores are cached
by generator content, settings and seed; the snapshot id and sha256 are part
of the settings, so a new `latest` snapshot retrains both sides instead of
reusing old baseline scores. `data` carries `candidate` (`mean`, `stdev`,
`per_seed`, `content_digest`), `baseline`, `comparison`
(`relative_improvement`, `seed_wins`, `seeds_compared`, `beats_baseline`),
`pool_hf` (`repo`, `snapshot`, `sha256`, `block`) and `history` (`path`, `runs`,
`best`). Each run appends one line to `./.cascade-17m/history.jsonl` with the
same `pool_hf` record, so only runs on the same snapshot are compared.
`iterate` caches the snapshot under its `--work-dir` (here
`./.cascade-17m/hf-pool/`); `score` uses `.cascade-work/`.

## 5. Would the public bench gate block it?

Install the sidecar's own locked environment once and stage benchmark data
([benchmark sidecar](../benchmarks/README.md)):

```bash
uv sync --project benchmarks --frozen --extra time --extra fev
uv run --project benchmarks cascade-benchmark-download --data-dir ./bench_data
```

Then add `--bench` (checkpoints are kept under `<work-dir>/checkpoints` unless
`--keep-ckpt` says otherwise):

```bash
cascade iterate ./demo-gen-v2 --baseline ./demo-gen --chain-toml ./chain.local-17m.toml --pool-hf latest --seeds 0,1 --train-hours 0.01 --device cuda --work-dir ./.cascade-17m --bench --bench-data-dir ./bench_data --json
```

Or bench two checkpoints you kept yourself:

```bash
cascade bench ./demo-ckpt/v2 --king ./demo-ckpt/base --chain-toml ./chain.local-17m.toml --bench-data-dir ./bench_data --device cuda --json
```

The sidecar scores the best candidate seed's checkpoint and the baseline's at
the same seed in one paired pass. The pair is judged by
`cascade.eval.bench_gate.evaluate_bench_gate`, the validator's function, with
the rule and the `[scoring]` thresholds of the configuration you passed.
`data.bench` (iterate) or `data.bench_gate` (bench) has this shape:

```json
{
  "verdict": "pass | block | uncomputable",
  "would_block": "<true | false | null when uncomputable>",
  "partial": "<true when a required suite was not fully scored on both sides>",
  "suites_not_ok": {"<suite>": "king <status>, candidate <status>"},
  "reason": "<the gate's reason, empty on a pass>",
  "gate": {"rule": "...", "avg_ratio": "...", "worst_lcb": "...", "win_rate": "...", "suites": {}},
  "staged_preview": {"<suite>": {"n": "<tasks>", "ratio": "<float>", "win_rate": "<float>"}},
  "settings": {"rule": "...", "suites": [], "tolerance": "...", "min_win_rate": "...",
               "min_items": "...", "alpha": "...", "B": "..."},
  "seed": "local-preview", "local_preview": true
}
```

A suite without staged data or without its extra (TIME needs `--extra time`,
FEV needs `--extra fev` and a pinned FEV manifest) comes back skipped. The gate
then reports `uncomputable` with `partial: true`, and `staged_preview` shows the
per-suite task ratio and win rate for the suites that did run. Those are
display numbers, not a verdict. If the sidecar environment is missing,
`--bench` stops before any training and prints the `uv sync` command.
