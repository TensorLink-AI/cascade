---
name: cascade-mine
description: Mine on the cascade subnet (netuid 91) by running an optimisation loop over a time-series data generator — propose a change, verify, score locally, keep it if better, and (only with explicit confirmation) submit the best. Use when asked to mine cascade, improve/hill-climb a cascade generator, run or watch `cascade mine` / `cascade mine-ui`, set up or operate the mining gauntlet (`cascade gauntlet`, deploy/harness), or when a `cascade mine` agent-proposer prompt says "Use the cascade-mine skill (proposer mode)".
---

# cascade-mine

A cascade miner submits a **data generator**, deterministic code that emits
synthetic time series. The operator trains a fixed Toto2-4M forecaster on
it and scores the result against the king. Lower score wins (geomean of
CRPS/WQL and MASE). The mining loop lives in `cascade/miner/optimize.py` and
is exposed as `cascade mine` (CLI) and `cascade mine-ui` (web UI).

This skill has three modes. Work out which one you are in first.

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
3. Make **one** focused, explainable change. What the eval data says wins
   (the dethrone analysis: winners and near misses had the same ~1.5% median
   gain; winners spread it over ≥6 domains, near misses packed it into one feed):
   - **breadth**: changes that help many domains and all three horizons
     (64/256/720), e.g. realism of the observation process (rounding to the
     published resolution, holds or sticky values, missing bins, reporting
     cadence, count data, saturation) or long multi-cycle structure for h720;
   - the domains the lineage is stuck on (nature, energy, finance, h64);
   - multichannel only with REAL cross-channel dependence (lagged links,
     cointegration, shared drivers), near the eval's ~12% share: a C>1 series
     is billed ~1.67× its length under points+mv20, and independent filler
     channels measured worse;
   - speed: the wall is the law (DEC-CA-0001), a slow generator trains on less.
   Narrow single-family/single-feed reweighting rarely survives a longer
   budget on new rounds; avoid it unless it is part of a broad change.
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
# or: cascade ralph --llm-provider chutes|saygm|engy|anthropic --llm-model <id> (run --check first;
#     an LLM rewrites the generator code each iteration; docs/RALPH_MINING.md)
# or: --proposer cmd --propose-cmd "python my_strategy.py" (the person's own strategy;
#     contract in docs/MINER_DOCKER.md, example scripts/example_strategy.py)
cascade mine-ui --workdir ./mine-run               # watch it in a browser
```

- Re-running on the same `--workdir` resumes from `best/`. Touch
  `<workdir>/STOP` to stop after the current candidate.
- Read progress from `<workdir>/state.json` (`best`, `king`, `beats_king`)
  and `<workdir>/history.jsonl`. `best/` is the tree you would submit.
- Report scores with their caveat. Local scores are **directional**. Scoring
  runs under the LIVE contract by default (the latest signed round's; the
  first printed line says which). `cascade score <dir> --replay-round <id>
  --snapshot-root <revealed pool>` scores on a real past round's exact windows
  against that round's king; `--pool-dir` scores single-horizon windows of the
  person's own data. A best that beats the reference king by less than ~1 % is
  inside the noise band. More `--seeds` and a larger `--train-hours` tighten it.
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

## Mode C — setting up or operating the gauntlet

The gauntlet (`cascade gauntlet`, docs/GAUNTLET.md) is the long-running harness:
LLM workers propose, a judge replays real past rounds on rented Lium GPUs, and
finalists wait for a person's approval. Follow docs/GAUNTLET_QUICKSTART.md, in
this order, and stop where it says:

1. **Free checks first.** `cascade gauntlet selftest` (must print `OK`),
   `cascade gauntlet check --config <harness.toml>` (no `FAIL`), and the worker
   model check (`cascade ralph --llm-provider <p> --llm-model <m> --check`).
2. **Show the person the config and get an explicit "go"** before `tick`, `run`
   or `docker compose up`: those rent GPUs. Confirm `[compute] daily_usd_cap`,
   `total_usd_cap` and `max_price_per_hour` with them; never raise a cap yourself.
3. **Keys only in `deploy/harness/.env`** (or the environment). Ask for each;
   never print, log or commit one. Check `git check-ignore deploy/harness/.env`.
4. If the host cannot reach huggingface.co, set `[rounds] sync_via = "executor"`.
5. After starting: `cascade gauntlet status` (funnel, spend, the `dethrone [...]`
   progress bar). The judge log has the same bar after G2, G3 and each cycle.

Hard rules:
- `[submit] mode` stays `approval` unless the person explicitly asks otherwise
  in this conversation. Never run `cascade gauntlet approve` for them, never
  mount or touch a wallet.
- To pause: `cascade gauntlet stop` (it finishes the current stage and tears its
  pods down). To stop everything: `docker compose down`. Then confirm with
  `lium ps` that no pod with the configured `pod_prefix` is left running.
- Report numbers honestly: G2 screens are noisy; only G3+ evidence matters, and
  a finalist still needs a person's approval.

## Reference

- Loop: `cascade/miner/optimize.py`; UI: `cascade/miner/ui.py`; docs:
  `docs/ONE_CLICK_MINING.md`.
- Scoring internals: `cascade/miner/score.py` (same train → eval path as the heat).
- Docker image: `deploy/miner.Dockerfile`; the `toolbox` target (default) is
  the CLI (docs/MINER_DOCKER.md), and `oneclick` adds Claude Code plus the
  UI on :8765. Workdir `/work` in both.
