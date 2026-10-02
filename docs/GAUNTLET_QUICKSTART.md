# Running the gauntlet: setup guide

This takes you from nothing to a running gauntlet that searches for a better
generator and asks you before it submits anything. For how it works, see
[GAUNTLET.md](GAUNTLET.md).

There are two ways to run it:

| | **A. Docker Compose (recommended)** | **B. One GPU machine, no Docker** |
|---|---|---|
| GPUs | rented on Lium, on demand | your own |
| Host | any always-on Linux box (a small VPS is enough) | the GPU machine |
| Isolation | judge, workers and operator in separate containers | one process, one user |
| Updates | automatic, between cycles | `git pull` + restart |

## What you need

| Item | Needed for | Notes |
|---|---|---|
| Lium account with credit, `LIUM_API_KEY` | A (renting GPUs), and any submission | submissions fund their own training leg |
| An LLM key for the **workers** | both | Chutes, SayGM, or Anthropic. Pick a strong coding model with reliable tool use |
| An OpenAI-compatible LLM key for the **operator** | optional | Hermes. Without it, steer by editing `DIRECTIVES.md` yourself |
| A Bittensor wallet + unused hotkey(s) | only when you submit | each submission spends one hotkey for good |
| Disk | both | ~150 MB per revealed snapshot; the window keeps about 8 |

**Cost guide (Lium, measured 2026-10-02):** the GPUs that worked were RTX 6000
Ada at $0.70–1.04/h (L40S was often sold out). The king is trained once per
round and budget (~15 min, cached), so each candidate costs ~0.4 GPU-hours to
screen, ~2 more if it reaches the confirm stage, and up to ~5 per round in the
full-budget replays. The default config caps spend at **$40/day** and refuses
pods above **$1.10/h**. Set your own numbers before you start.

## Path A: Docker Compose on a VPS

### 1. Get the stack

```bash
git clone https://github.com/TensorLink-AI/cascade && cd cascade/deploy/harness
mkdir -p work secrets
cp harness.example.toml work/harness.toml
cp .env.example .env
sed -i "s|^STACK_DIR=.*|STACK_DIR=$PWD|" .env
ssh-keygen -t ed25519 -N '' -f secrets/gauntlet_ssh     # the key the judge uses on its pods
```

### 2. Fill in `.env`

```
LIUM_API_KEY=...
CHUTES_API_KEY=...            # or SAYGM_API_KEY / ANTHROPIC_API_KEY: your workers' provider
HERMES_BASE_URL=https://llm.chutes.ai/v1      # optional operator
HERMES_MODEL=<model id>
HERMES_API_KEY=...
```

Keys live only in `.env`, never on a command line.

### 3. Edit the two configs

`work/harness.toml`, the judge. At minimum:

```toml
[compute]
daily_usd_cap      = 40.0      # your hard daily ceiling
max_price_per_hour = 1.50      # never rent above this
max_parallel       = 4         # pods at once

[workers]
llm_provider = "chutes"        # must match worker.toml
llm_model    = "<model id>"

[submit]
mode        = "approval"       # keep this until you trust the results
wallet_name = "<your wallet>"
```

`worker.toml`, the workers. Set the same `llm_provider` and `llm_model`.

The number of `worker` replicas in `docker-compose.yml` (4) should equal
`[search] proposals_per_cycle` (4).

### 4. Get the images

After this lands on `main`, CI publishes them:

```bash
docker compose pull
```

Until then, or to run your own changes, build them from the checkout. This
takes a while the first time (CUDA + torch):

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml build
```

If you build locally, run every `docker compose` command below with
`-f docker-compose.yml -f docker-compose.build.yml`, and stop the updater
(`docker compose stop updater`) so it doesn't swap your images back.

### 5. Preflight: free, and catches most problems

```bash
# the harness itself: one full cycle on synthetic data (no GPU, LLM or network)
docker compose run --rm judge gauntlet selftest

# keys, SSH key, Lium CLI, spend bounds, snapshots
docker compose run --rm judge gauntlet check --config /work/harness.toml

# the worker model answers through the exact path Claude Code will use
docker compose run --rm worker ralph --llm-provider chutes --llm-model <model id> --check
```

Fix every `FAIL` before going on. The worker check tells you what's wrong: a bad
key (401/403), an unknown model (400), a wrong URL (404) or no credit (429).

### 6. First tick: build the window and baseline the king (spends GPU)

```bash
docker compose run --rm judge gauntlet tick --config /work/harness.toml
```

This step:
1. downloads the newest signed receipts;
2. downloads the revealed snapshots they need, into `work/eval-pool/`;
3. rents pods and trains the king's short legs on every round in the window,
   which is about 6 GPU-hours.

It prints `baseline_ready: true` when done. If it says `waiting`, too few rounds
have been revealed yet. Reveals lag about 48h; wait and re-run.

### 7. Start it

```bash
docker compose up -d
docker compose logs -f judge          # Ctrl-C to stop following
```

To run without the operator: `docker compose up -d judge worker updater`.

## Day to day

```bash
alias g='docker compose exec judge cascade gauntlet'
g status --config /work/harness.toml          # funnel, population, spend, pending finalists
```

| You want to | Do |
|---|---|
| see what the operator decided | read `work/gauntlet/operator/reports/` |
| steer the workers yourself | edit `work/gauntlet/operator/DIRECTIVES.md`. A line `parent: c00042` focuses every proposal on one member |
| see why candidates die | `g status`, or `work/gauntlet/events.jsonl` |
| check spend | `g status` (today vs cap), `work/gauntlet/spend.json`, and your Lium console |
| pause | `g stop --config /work/harness.toml`: it finishes the current stage, tears its pods down and idles. Delete `work/gauntlet/STOP` to resume (if the operator stopped it, the file is `work/gauntlet/operator/STOP`, with the reason in its report) |
| block all submissions now | `touch work/gauntlet/submit/HOLD` |
| stop everything | `docker compose down` (the judge tears down idle pods; the next start reaps anything left over) |

### When a finalist appears

In approval mode you get it in `g status` as **PENDING APPROVAL**, and on your
`notify_url` webhook if you set one. Read the evidence first:

```bash
cat work/gauntlet/submit/pending/<id>.json      # G4 per-round wins, G4.5 result
```

To submit, uncomment the read-only wallet mount in `docker-compose.yml`, then:

```bash
docker compose up -d judge
g approve <id> --hotkey <unused hotkey> --confirm SUBMIT --config /work/harness.toml
docker compose run --rm judge reveal-status <hotkey> --watch
```

Or reject it: `g reject <id> --config /work/harness.toml`.

### Going autonomous (only once you trust it)

After a few approved finalists have actually won or lost on chain:

```toml
[submit]
mode        = "autonomous"
hotkeys     = ["5F...", "5G..."]   # UNUSED hotkeys, consumed in order
margin      = 0.01                 # improvement required at the G4.5 one-shot
max_per_day = 1
```

If any guardrail fails, that finalist goes to approval instead.
`touch work/gauntlet/submit/HOLD` stops autonomous submissions at once.

## Before you trust the numbers: the king self-replay

Replays compare your Lium training legs with the king's scores from the
operator's own hardware. Check that gap once, on a GPU, before relying on the
full-budget stage. Use a round id from `work/gauntlet/receipts/`:

```bash
docker compose run --rm judge score /opt/cascade/champions/king \
    --replay-round <round_id> --snapshot-root /work/eval-pool --device cuda
```

The printed `vs king` number should be well under 0.5% either way. If it isn't,
the full-budget stage can't be trusted on your hardware yet. The output names
the GPU the king's leg ran on, so rent the same type (`[compute] sku`).

## Path B: one GPU machine, no Docker

```bash
git clone https://github.com/TensorLink-AI/cascade && cd cascade
uv sync --all-extras                                  # or: pip install -e '.[train,hippius,chain]'
curl -fsSL https://claude.ai/install.sh | bash        # Claude Code, for the workers
export CHUTES_API_KEY=...                             # your workers' provider
uv run cascade gauntlet selftest
uv run cascade gauntlet init --dir mining
```

In `mining/harness.toml`, set:

```toml
start_dir = "../champions/king"
king_dir  = "../champions/king"

[compute]
executor     = "local"
max_parallel = 1          # number of GPUs

[workers]
mode         = "inline"
llm_provider = "chutes"
llm_model    = "<model id>"
```

Then:

```bash
uv run cascade gauntlet check --config mining/harness.toml --workers
uv run cascade gauntlet tick  --config mining/harness.toml
uv run cascade gauntlet run   --config mining/harness.toml     # in tmux/screen
```

With one GPU, a cycle is slow: the full-budget replays are hours each. Lower
`[stages] g4_rounds` to 2, or keep Path A's Lium executor while running
everything else locally.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `waiting: fewer than N replayable rounds` | reveals lag ~48h, or `sync_reveals` couldn't reach Hugging Face (set `[rounds] sync_via = "executor"` to download on a pod). Check `ls work/eval-pool/snapshots` |
| `king baseline incomplete` | pod or rental failures. See `work/gauntlet/jobs/*/job.log`. It retries next cycle |
| `waiting N min for GPU capacity` | the market has none of `sku_choices` under your price cap; it backs off up to ~1 h. Add a GPU type or raise `max_price_per_hour` |
| `daily spend cap reached` | working as intended: idle pods are torn down and it resumes at 00:00 UTC. Raise `daily_usd_cap` if you mean to |
| every candidate `dead@G0` | workers break a rule. Read `work/gauntlet/candidates/<id>.agent.log` and add the rule to `DIRECTIVES.md` |
| every candidate `worker: no worker finished it in time` | worker containers aren't running, or their model fails. Run `docker compose logs worker` and the `ralph --check` above |
| candidates `stalled` | infrastructure faults (SSH, OOM, network). They retry twice, then die as `dead@infra`. Check `jobs/*/job.log` |
| `rebuilt N windows do not reproduce the receipt's window_ids` | your image's `chain.toml` draw rules are older than the round's. Pull the newest image |
| operator log: `API call failed` | wrong `HERMES_BASE_URL`, model or key. Read `work/gauntlet/operator/reports/operator.log` |
| updater log: `judge did not park` | the judge is mid-cycle. A full-budget stage can take hours. It retries every 6h |
| a pod is still billing after `down` | start the judge again: it tears down every pod with its prefix that it doesn't own. Or remove it in the Lium console |
