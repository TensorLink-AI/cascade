---
name: cascade-mine
description: Mine on the cascade subnet (netuid 91) by running an optimisation loop over a time-series data generator — propose a change, verify, score locally, keep it if better, and (only with explicit confirmation) submit the best. Use when asked to mine cascade, improve/hill-climb a cascade generator, run or watch `cascade mine` / `cascade mine-ui`, or when a `cascade mine` agent-proposer prompt says "Use the cascade-mine skill (proposer mode)".
---

# cascade-mine

A cascade miner submits a **data generator**, deterministic code that emits
synthetic time series. The operator trains a fixed Toto2-4M forecaster on
it and scores the result against the king. Lower score wins (geomean of
CRPS/WQL and MASE). The mining loop lives in `cascade/miner/optimize.py` and
is exposed as `cascade mine` (CLI) and `cascade mine-ui` (web UI).

This skill has two modes. Work out which one you are in first.

## Mode A — proposer (you were invoked BY the loop)

The prompt says "Use the cascade-mine skill (proposer mode)" and your working
directory is `…/candidates/NNNN/`, a copy of the current best generator.

1. Read the history in the prompt. Do not repeat a change that was rejected
   or scored worse. If the last few tune-only attempts all failed, make a
   structural change (a new series family, better-matched noise, a realistic
   publication artefact such as rounding, gaps, or count data) instead of
   reweighting.
2. Read `config.json` and skim `generator.py` for the family registry and
   mixture weights. The king is ~19k lines: grep for it, do not read it
   end to end.
3. Make **one** focused, explainable change. The good levers, roughly by
   past payoff:
   - mixture reweighting toward families that look like real held-out data
     (energy, nature, sales, web, transport, finance, epidemiology);
   - realism of the observation process: rounding to the published
     resolution, holds or sticky values, missing bins, reporting cadence;
   - a new family that covers a domain the corpus lacks;
   - speed. The wall is the law (DEC-CA-0001), so a slow generator trains the
     model on less data. Never add work per series without a reason.
4. Keep every hard rule (`docs/INTERFACE.md`, `docs/MINER.md`):
   - deterministic in `seed` only (no `hash()`, clock, `os.urandom`, network);
   - only allowlisted deps; blocked imports (`socket`, `subprocess`, `pickle`,
     `multiprocessing`, …) are rejected in every `.py`. No code packed into
     strings, no shipped weights, no `.so`/`.pyc`;
   - yields are finite float arrays, `(L,)` or `(C, L)`, with `64 ≤ L ≤ 4096`,
     `C ≤ 32`, and magnitudes kept in a sane float32 range.
5. Optionally run `cascade verify .` to catch a rule break early. **Do not**
   run `cascade score` or `cascade mine`, because the loop scores next.
6. Write ONE line describing the change to the note path the prompt gives
   you, e.g. `flow_recession weight x1.5; gauge rounding to 0.01 on 60% of rows`.

In a Ralph loop (`cascade ralph`) the prompt is RALPH_PROMPT.md instead, and
it carries the same rules. Read and update `.ralph-notes.md` there; it is
your only memory between iterations.

## Mode B — driver (a person asked you to mine)

Run the loop. Do not re-implement it.

```bash
cascade verify ./champions/king                    # the env can import the king's deps
cascade mine --workdir ./mine-run --proposer tune --iterations 20 \
    --train-hours 0.25 --warm-start live --pool-dir <held-out .npy dir>
# or: --proposer agent (each step invokes Claude Code in proposer mode)
# or: cascade ralph --llm-provider chutes|saygm|anthropic --llm-model <id> (run --check first;
#     an LLM rewrites the generator code each iteration; docs/RALPH_MINING.md)
# or: --proposer cmd --propose-cmd "python my_strategy.py" (the person's own strategy;
#     contract in docs/MINER_DOCKER.md, example scripts/example_strategy.py)
cascade mine-ui --workdir ./mine-run               # watch it in a browser
```

- Re-running on the same `--workdir` resumes from `best/`. Touch
  `<workdir>/STOP` to stop after the current candidate.
- Read progress from `<workdir>/state.json` (`best`, `king`, `beats_king`)
  and `<workdir>/history.jsonl`. `best/` is the tree you would submit.
- Report scores with their caveat. Local scores are **directional**: they
  come from the miner's pool, not the validators' private one. A best that
  beats the reference king by less than ~1 % is inside the noise band. More
  `--seeds` and a larger `--train-hours` tighten it.
- Without `--pool-dir` the loop scores on an offline synthetic sample, which
  is a weak signal. Say so when that is what ran.

### Submitting (irreversible, costs money)

Submitting spends the hotkey (one submission per hotkey) and funds a GPU leg
from the miner's Lium account. **Never submit unless the person explicitly
asks you to in this conversation**, and confirm the hotkey and intake first.
Then:

```bash
export LIUM_API_KEY=...   # environment only, never an argument
cascade submit <workdir>/best https://submissions.cascadesub.net \
    --wallet-name <w> --wallet-hotkey <hk> --label <label>
cascade reveal-status <hotkey> --watch
```

`cascade mine --auto-submit …` does the same at the end of a run, and only
when the best beats the reference king by `--submit-margin`. Use it only when
the person asked for unattended submission.

## Reference

- Loop: `cascade/miner/optimize.py`; UI: `cascade/miner/ui.py`; docs:
  `docs/ONE_CLICK_MINING.md`.
- Scoring internals: `cascade/miner/score.py` (same train → eval path as the heat).
- Docker image: `deploy/miner.Dockerfile`; the `toolbox` target (default) is
  the CLI (docs/MINER_DOCKER.md), and `oneclick` adds Claude Code plus the
  UI on :8765. Workdir `/work` in both.
