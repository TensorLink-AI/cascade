# Mining in Docker: the `cascade-miner` image

One image gives every miner the same environment whatever their strategy:
the `cascade` CLI, the exact torch build, and the generator-runtime pins a
training pod runs. `cascade verify` and `cascade score` behave the same on
your laptop as they do in the round. Bring your own generator, and
optionally your own search strategy. The image does the checks, the
scoring, the loop and the submission.

| Tag | Target | What it is |
|---|---|---|
| `ghcr.io/tensorlink-ai/cascade-miner:latest` | `toolbox` | the miner CLI; no UI, no LLM |
| `ghcr.io/tensorlink-ai/cascade-miner:oneclick` | `oneclick` | toolbox + Claude Code + the web UI ([ONE_CLICK_MINING.md](ONE_CLICK_MINING.md)) |
| `cascade-miner:harness` (built locally by `deploy/harness/docker-compose.yml`) | `harness` | toolbox + the Lium CLI and SSH: the gauntlet judge ([GAUNTLET.md](GAUNTLET.md)) |

`.github/workflows/publish-miner.yml` keeps these tags fresh. `:latest` and
`:oneclick` are rebuilt whenever a change that affects the image lands on
main: code, the bundled king (`champions/king`, via king-sync), the
Dockerfile, deps or the skill. They are also rebuilt every Monday, for a
fresh CUDA base, security fixes and the latest Claude Code. Every build is
smoke-tested (`deploy/miner-smoke.sh`) before anything is pushed. A pull
request builds and smoke-tests without publishing.

| Pin | When |
|---|---|
| `:latest` / `:oneclick` | always the newest passing build of main |
| `:sha-<commit>` / `:sha-<commit>-oneclick` | immutable; pin this for reproducible runs |
| `:v<X>` / `:v<X>-oneclick` | releases, from a `miner-v<X>` git tag |

Check what you are running: `docker run --rm <image> python -c 'import os;
print(os.environ["CASCADE_MINER_BUILD_SHA"])'`, or the
`org.opencontainers.image.revision` label.

To build it yourself:

```bash
docker build -f deploy/miner.Dockerfile -t cascade-miner .                    # toolbox
docker build -f deploy/miner.Dockerfile --target oneclick -t cascade-miner:oneclick .
# CPU-only machine:
docker build -f deploy/miner.Dockerfile -t cascade-miner:cpu \
    --build-arg BASE_IMAGE=ubuntu:22.04 \
    --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
```

## Setup once

Everything runs in `/work`, which is your mounted directory. Keep an alias:

```bash
alias cm='docker run --rm -it --gpus all -v "$PWD:/work" \
  -v ~/.bittensor/wallets:/root/.bittensor/wallets:ro --env-file .env \
  ghcr.io/tensorlink-ai/cascade-miner:latest'
cm            # prints the command overview
```

Credentials go in `.env` next to your work, never in the image or on a
command line (argv is world-readable). You only need what you use:

```
LIUM_API_KEY=...            # funding your leg (submit / fund)
HIPPIUS_HUB_TOKEN=...       # public deploy path / fetching from the Hub
```

Drop `--gpus all` on a machine without an NVIDIA GPU. Scoring then runs on
CPU (slowly).

## The miner workflow, containerised

These are the same commands as [MINER.md](MINER.md), prefixed with `cm`:

```bash
cm fetch king --out ./king                      # study / fork the reigning generator
cp -r king my-gen                               # or: cm bash → cp -r /opt/cascade/scripts/example_generator …
cm verify ./my-gen                              # every admission check + determinism
cm score ./my-gen --warm-start live --device cuda --pool-dir ./my-heldout
cm score ./king   --warm-start live --device cuda --pool-dir ./my-heldout
cm submit ./my-gen https://submissions.cascadesub.net --wallet-name W --wallet-hotkey H
cm reveal-status <hotkey> --watch
cm duel --hotkey <hotkey>
```

## Plug in your own strategy

`cascade mine` is a propose → verify → score → keep-if-better loop. Scoring
is fixed on one pool, seed set and init per run, so candidates compare like
for like. Resume, the `best/` tree and the UI come with it. You supply only
the **propose** step, as any command in any language:

```bash
cm mine --start ./my-gen --king ./king --pool-dir ./my-heldout --warm-start live \
        --proposer cmd --propose-cmd "python /work/my_strategy.py" --iterations 40
```

Each iteration, your command runs with its **working directory set to a
fresh copy of the current best**. It edits the files in place and exits 0.
It gets:

| Channel | Content |
|---|---|
| stdin | JSON `{iteration, candidate_dir, workdir, history, best, king}`; `history` = every row of `history.jsonl` so far (score, per-seed scores, note, accepted, parent) |
| env | `CASCADE_CANDIDATE_DIR`, `CASCADE_ITERATION`, `CASCADE_WORKDIR`, `CASCADE_HISTORY`, `CASCADE_NOTE_FILE`, `CASCADE_BEST_SCORE`, `CASCADE_KING_SCORE` |
| argv | `{dir}` in `--propose-cmd` is replaced with the candidate dir |
| output | write one line describing the change to `$CASCADE_NOTE_FILE` (else your last stdout line is used). stdout and stderr land in `candidates/NNNN.cmd.log` |

A non-zero exit records the iteration as `error` and scores nothing. An
edit that changes nothing is `rejected` without scoring. A candidate that
fails `verify` is `rejected` with the reason in `detail`, so your strategy
can read that from `history` next time.

`scripts/example_strategy.py` is a complete, stdlib-only example (~70
lines). It runs a coordinate search over the mixture weights, skips moves
already tried from the same parent, and pushes a move that just won. Copy
it and replace `propose()`. Anything goes: a Bayesian optimiser over your
own knobs, a population kept in your own state file under
`$CASCADE_WORKDIR`, or code generation.

The built-in proposers use the same loop: `--proposer tune` (random
log-normal steps on `config.json`, no LLM), `--proposer agent` (Claude
Code, in the `oneclick` image), and `cascade ralph` (a Ralph loop on
Anthropic, Chutes, SayGM or any Anthropic-compatible model;
[RALPH_MINING.md](RALPH_MINING.md)).

### Reading results

```
mine-run/
  state.json        status, best, king, beats_king (relative; positive = best beats king)
  history.jsonl     one row per candidate
  candidates/NNNN/  every tree tried (+ NNNN.note.md, NNNN.cmd.log)
  best/             the current best, the tree you submit
  STOP              touch to stop after the current candidate
```

`cm mine-ui` is not needed for your own strategy, but it works on any
workdir: `docker run -p 127.0.0.1:8765:8765 -v "$PWD:/work" cascade-miner ui`.

## Numbers are directional

Local scores come from your pool, not the validators' private rotating pool
(MINER.md §3). A short `--train-hours` ranks differently from a full leg.
Before you spend a hotkey, re-score your best and the king with more
`--seeds` and a longer budget, and rotate `--pool-dir` between runs so your
strategy does not overfit one pool.
