# One-click mining: `cascade mine`, the UI, the image, the skill

`cascade mine` runs a local optimisation loop over your generator. It
proposes a change, runs `cascade verify`, runs `cascade score`, keeps the
change if it scored better, and repeats. You end up with a `best/` tree you
can submit. It ships in three wrappers:

| Wrapper | What it is |
|---|---|
| `cascade mine` | the loop, headless (`cascade/miner/optimize.py`) |
| `cascade mine-ui` | a local web UI: Start / Stop, live chart, candidate table, confirmed Submit (`cascade/miner/ui.py`) |
| `deploy/miner.Dockerfile --target oneclick` | the [miner toolbox image](MINER_DOCKER.md) plus Claude Code and the skill, starting the UI |
| `.claude/skills/cascade-mine` | the agent skill: drive the loop (driver mode) or be one proposal step inside it (proposer mode) |

This is miner-side tooling only. It touches no consensus, contract or
validator path.

## Quick start (Docker, the "one click")

The one-click image is the [toolbox image](MINER_DOCKER.md) with Claude Code,
the skill and the UI added on top. Everything in MINER_DOCKER.md, including
plugging in your own strategy, works here too.

```bash
docker run --gpus all -p 8765:8765 \
    -v "$PWD/mine:/work" \
    -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro \
    -e LIUM_API_KEY -e ANTHROPIC_API_KEY \
    ghcr.io/tensorlink-ai/cascade-miner:oneclick
# or build it: docker build -f deploy/miner.Dockerfile --target oneclick -t cascade-miner:oneclick .
```

Open the `http://localhost:8765/?token=…` URL it prints and press **Start
mining**. The run lives in `./mine/mine-run/` on the host, so it survives the
container. Every key is optional:

| Env / mount | Needed for |
|---|---|
| `--gpus all` | fast scoring (CPU works, much slower) |
| `ANTHROPIC_API_KEY` (or `CLAUDE_CODE_OAUTH_TOKEN`) | the `agent` proposer |
| `LIUM_API_KEY` | funding the leg when you **Submit** |
| wallet mount | **Submit** (signs the commit + intake request) |
| `-v …/pool:/pool` | scoring on your own held-out data (`/pool` in the form) |

Headless, same image: `docker run --gpus all -v "$PWD/mine:/work" ghcr.io/tensorlink-ai/cascade-miner:oneclick mine --iterations 30 --warm-start live`.

## Without Docker

```bash
pip install -e '.[train,hippius,chain]'
cascade mine-ui                       # → http://127.0.0.1:8765/
# or headless:
cascade mine --workdir ./mine-run --iterations 20 --warm-start live --pool-dir ./my-heldout
```

## What one iteration does

1. **Propose.** Copy `best/` to `candidates/NNNN/` and mutate it:
   - `tune` (default, no LLM) multiplies 1–3 knobs of `config.json` by
     `exp(N(0, σ))`. Knobs are mixture weights (renormalised, so a move is a
     pure reallocation), float scalars, and `min/max_length` (clamped to
     64–4096). Ints and `*seed*` keys stay fixed. The king's
     `family_weights` has ~70 families, so this is a sensible first lever.
   - `agent` runs `claude -p` in the candidate dir with the loop history
     and the `cascade-mine` skill in proposer mode. It makes one focused code
     change and writes a one-line note. Edits are confined to the
     candidate dir, and the only shell command allowed is
     `cascade verify`. Swap it with `--agent-cmd`.
   - `ralph` is a Ralph loop: Claude Code on Anthropic, Chutes, SayGM or any
     Anthropic-compatible model rewrites the generator's **code**, with one
     standing prompt and a persistent notebook. See
     [RALPH_MINING.md](RALPH_MINING.md).
   - `cmd` runs your own strategy command; the contract is in
     [MINER_DOCKER.md](MINER_DOCKER.md#plug-in-your-own-strategy). The UI
     offers it as "cmd: my own strategy".
2. **Verify.** The trainer's checks: layout, import guard, packed
   sources, hash-locked deps, determinism. A would-be-rejected candidate is
   recorded `rejected` and never scored.
3. **Score.** `score_generator` on one pool, one seed set and one
   init, fixed for the whole run. `--warm-start live` is resolved once, so
   a promotion mid-run cannot move the baseline. With several `--seeds` the
   score is their mean.
4. **Accept.** The candidate replaces `best/` iff
   `score < best × (1 − min_improvement)`.

Pass 0 scores the start generator (default `champions/king`) and the
reference king (default `champions/king`; `--king none` to skip). "Best vs
king" in the UI is therefore a paired number on your pool.

## Files

```
mine-run/
  state.json        status, phase, best, king, beats_king (UI reads this)
  history.jsonl     one line per candidate: score, per-seed, note, accepted, detail
  candidates/NNNN/  each candidate tree (+ NNNN.note.md, NNNN.agent.log)
  best/             the current best, which is what Submit sends
  loop.log          loop output (UI-started runs)
  STOP              touch to stop after the current candidate
```

Re-running on the same workdir **resumes** from `best/`. Start a fresh
workdir when you change pool, seeds, init or budget, because scores are only
comparable within one run.

## Reading the numbers honestly

- Scores are **directional** (MINER.md §3). Validators score on a private,
  rotating pool. Hill-climbing one local pool overfits it, so rotate
  `--pool-dir` between runs. Without a pool dir the loop scores on an
  offline synthetic sample, which is a weak signal.
- The default budget (`--train-hours 0.25`) is below the heat's. It is
  cheap enough to climb, and a short budget can rank generators
  differently from a full leg. Before you submit, re-score the best and
  the king with more seeds and a longer budget.
- A win under ~1 % over the king is inside the noise band. The dethrone bar on
  mainnet is a margin *plus* a bootstrap LCB > 0 (DEC-CA-0016/0040).

## Submitting

Submitting spends the hotkey (one submission per hotkey) and funds a GPU
leg from your Lium account. Nothing submits by default:

- UI: fill intake, wallet and hotkey, type `SUBMIT`, press **Submit best
  privately**. This runs `cascade submit <workdir>/best <intake> …`
  (DEC-CA-0036 direct path: the code stays private unless it takes the throne).
- CLI: `cascade mine … --auto-submit --wallet-name W --wallet-hotkey H`
  submits at the end only if the best beats the reference king by
  `--submit-margin` (default 1 %).

Then follow it as usual: `cascade reveal-status <hotkey> --watch`,
`cascade queue`, `cascade duel --hotkey <you>`.

## UI security

The UI can spend your compute, hotkey and Lium balance, so every API call
needs a per-process token. On `127.0.0.1` the page embeds it and only a
loopback `Host:` is served, which blocks DNS rebinding. On any other bind
(the image binds `0.0.0.0`) the page needs `?token=` once. Set your own
token with `--token` or `$CASCADE_UI_TOKEN`. Secrets never pass through the
UI: it shows only whether `LIUM_API_KEY`, Claude auth and a wallet are
present.
