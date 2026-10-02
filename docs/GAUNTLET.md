# The mining gauntlet

**Setting it up?** Start with [GAUNTLET_QUICKSTART.md](GAUNTLET_QUICKSTART.md).

`cascade gauntlet` is a long-running harness that searches for a better
generator and only reports one when it has survived the same tests a
validator would apply, on rounds it has never been tuned on. It wraps the
pieces that already exist (`cascade verify`, `cascade score --replay-round`,
the Ralph worker setup, the Lium provisioner) into one loop:

```
             ┌─ operator: Hermes Agent (OpenAI-compatible LLM) ──────────────┐
             │ reads operator/status.json, writes DIRECTIVES.md + reports    │
             └──────────────────────────────┬────────────────────────────────┘
                                            │ directives (text)
 population ─copy─▶ workers: Claude Code (Chutes / SayGM / Anthropic) ─▶ candidate
                                            │ one edit each, fresh context
             ┌─ judge (deterministic, holds the keys) ───────────────────────┐
             │ G0 verify + dedup → G1 throughput → G2 screen → G3 confirm    │
             │ → G4 full replays vs receipt kings → G4.5 one-shot → G5 submit │
             └──────────────────────────────┬────────────────────────────────┘
                                            │ jobs over SSH
                                 Lium pods (worker image) or this GPU
```

**LLMs decide what to try. Code decides what counts, what is spent and what is
submitted.** No model grades its own work, holds the Lium key, or submits.

## Hosts without Hugging Face

The revealed snapshots live on Hugging Face. If this host cannot reach it, set
`[rounds] sync_via = "executor"`: a pod downloads the folders the window needs
and they are copied back. Unrevealed blocks are retried at most every 6 h.

## Pools are replayed rounds

Every scored round leaves a signed receipt with its seeds, its verdict window
ids and the king's per-window scores. About 48h after its pool snapshot
retires, the exact bytes are revealed at
[Tensor-Link/cascade-eval-pool](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool).
With both, `cascade score --replay-round` rebuilds a round's exact verdict
windows and refuses to score if they differ from the receipt
([MINER.md §3](MINER.md#replay-a-past-round)).

The judge keeps a **window** of the newest replayable rounds:

| Pool | Rounds | Used by | The agents see |
|---|---|---|---|
| A | `n_a` (6) older rounds | G2 screen, a different round per candidate | the screen number |
| B | `n_b` (2) newest rounds | G3 confirm | pass / fail |
| C | the newest revealed snapshot, windows drawn with the day's seed (not any round's) | G4.5 one-shot | pass / fail |

The pool is published once a day, so the window slides once a day. Every slide,
king change, `chain.toml` change or image change starts a new **epoch**: the
king's short legs are re-run on the new rounds, noise is re-measured, and every
population member must pass G3 again or retire. Scores never mix across epochs.

## The stages

| Stage | What | Where | Passes when |
|---|---|---|---|
| G0 | static `cascade verify`; identical-tree dedup; no-op check | judge | clean, new, changed |
| G1 | runtime verify (determinism) + points/s, candidate and king in the SAME job | executor | ≥ king × (1 − `g1_throughput_tol`) |
| G2 | `g2_hours` replay of one pool-A round, paired with the king's same-budget leg on that round | executor | improvement ≥ `g2_margin_floor` |
| G3 | `g3_hours` replays of every pool-B round, pooled paired bootstrap | executor | mean improvement ≥ `g2_margin_floor` AND LCB > 0 |
| G4 | full-contract replays of the newest `g4_rounds`, judged against each receipt's king under that round's own rules | executor | ≥ `g4_min_wins` round wins and mean improvement > 0 |
| G4.5 | the finalist's G4 checkpoint vs the newest king's trained checkpoint, on fresh windows of the newest revealed snapshot | executor | LCB > 0 and improvement > 0 |
| G5 | submit | judge | see below |

**The reference** (`[stages] reference`):

- `cached` (default): the king is trained at the stage's own budget on the same
  round, seeds and contract, **once per (king, worker image, round, budget)**,
  in the same batch as the first candidate that needs it, and reused across
  days. A steady-state day trains one king leg (the newly revealed round).
- `receipt`: compare with the king's signed full-budget receipt scores, no king
  training at all. Measured: a 15-minute leg lands within ±0.5% of the receipt
  per round, which is larger than the 0.2% screen margin, so this mode can kill
  (or pass) a candidate on round bias alone.

Re-training the king under salted seeds measured the seed noise of a
15-minute leg at 0.013%, far below the 0.2% margin, so the margin is a fixed
floor (`g2_margin_floor`) rather than a per-epoch calibration.

**The contract:** every replay trains under the round's own signed contract
(the manifest's `contract_body`, digest-checked), not this checkout's
`chain.toml`. They differed in practice: live rounds since block ~9.17M bill
`points+mv20` on worker v0.13.0 while the repo read `series_points` / v0.9.0.

Statistical choices, and the failure each one prevents:

- **Fresh rounds at every stage.** G2 picks lucky winners, so G3 re-measures
  them on different rounds and never reuses G2's numbers.
- **Pass/fail only past G2.** Repeated queries leak a hidden pool's contents.
  Pools B and C feed back nothing numeric, and B rotates daily.
- **One-shot final check.** G4 compares several finalists, so the best of them
  clears by luck more often. G4.5 measures only the chosen one, once, on windows
  no stage scored: the newest revealed snapshot (the validators' own data, from
  the Hugging Face eval-pool dataset) drawn with the day's seed.
- **`g2_explore_frac`** (off by default). A fraction of G2 losers go to G3 anyway. Their G3
  outcomes (`explore` events) show how often the cheap screen kills a real
  improvement.
- **Infrastructure faults retry and never count against a candidate.** Only a
  candidate fault (rejected, crashed or stalled generator) kills it. A round
  this checkout cannot replay faithfully (its windows or contract do not
  rebuild) is checked on the judge and left out of the window before any pod
  is rented.
- **Paid work is never thrown away.** A batch the spend cap stops keeps every
  completed result; G4 keeps each full-budget leg on the candidate, so a rerun
  pays only for the missing ones. Finalists and submitted trees are not
  re-run or re-offered.

## Compute and spend

`[compute] executor = "local"` runs jobs as subprocesses on this machine's GPUs.
`executor = "lium"` rents pods:

- Pods use the worker image the newest round's signed contract names (its
  digest resolved to a tag through GHCR), or `[compute] image` if pinned, with
  this checkout's `cascade/` package streamed (tar over SSH) over the image's
  editable install. That is the same runtime a real leg uses.
- `sku_choices` lists the GPU types to rent, cheapest available first. All must
  be Ampere/Ada/Hopper: the pinned torch has no Blackwell kernels, and the pod
  health check runs a CUDA op to reject such a machine at boot.
- `max_price_per_hour` filters the rentals. The **write-ahead ledger**
  (`gauntlet/spend.json`) records each pod at the cap before renting, then at
  its listed price once rented; cap checks always reserve jobs at the cap.
  Accrued spend is therefore an upper bound.
- No job starts, on a new or a warm pod, when today's accrued spend plus one
  job at the cap would pass `daily_usd_cap`. At the cap the judge tears down
  idle pods and sleeps until 00:00 UTC. A job already running can finish, so
  the overshoot is at most one job.
- Pods idle for `idle_minutes` are torn down. On start, every pod with the
  gauntlet's prefix that the ledger does not own is torn down, so a crashed run
  never keeps billing.

## Submitting (G5)

| `[submit] mode` | What happens to a G4.5 finalist |
|---|---|
| `off` | recorded only |
| `approval` (default) | written to `gauntlet/submit/pending/<id>.json` and the `notify_url` webhook; a person runs `cascade gauntlet approve <id> --hotkey HK --confirm SUBMIT` |
| `autonomous` | submitted by the judge only when every guardrail holds; otherwise it falls back to `approval` with the reasons |

Autonomous guardrails: no `submit/HOLD` or `STOP` file; G4.5 passed with an
improvement of at least `margin` (≥ 0.005, the duel's floor); fewer than
`max_per_day` submissions in 24h; an unused hotkey left in `hotkeys`; the
exact tree never submitted before; `LIUM_API_KEY` set. Every attempt spends its
hotkey for good, even a failed one, so a hotkey is never retried.

## Running it

```bash
cascade gauntlet selftest                     # one synthetic cycle, no GPU/LLM/network
cascade gauntlet init --dir .                 # writes harness.toml (+ operator files)
cascade gauntlet check --config harness.toml  # keys, LLM endpoint, Lium, snapshots
cascade gauntlet tick  --config harness.toml  # optional: build the window + baseline now
cascade gauntlet run   --config harness.toml  # until `cascade gauntlet stop`
cascade gauntlet status --config harness.toml
```

Docker Compose runs four containers from CI-published images
(`deploy/harness/docker-compose.yml`; add `-f docker-compose.build.yml` to build
from your checkout):

| Service | Image | Mounts | Secrets |
|---|---|---|---|
| `judge` | `cascade-miner:harness` | `./work` (config, state, receipts, eval-pool) | `LIUM_API_KEY`, pod SSH key; wallet only if submitting |
| `worker` ×N | `cascade-miner:oneclick` | the proposal queue only | its LLM key |
| `operator` | Hermes Agent | `gauntlet/operator/` only | its LLM key |
| `updater` | `docker:27-cli` | the Docker socket + this directory | none |

The worker reaches nothing but the queue, so it cannot read pools, receipts or
keys. Its outbound network is not restricted by Compose. If you want it limited
to the LLM endpoint, put an egress proxy in front of it.

## Staying up to date

| What changes | How the gauntlet picks it up |
|---|---|
| The king (a dethrone) | `king_source = "live"`: every hourly refresh re-reads the anchor validator's latest SIGNED receipt (the trainer's own rule: `verdict.king_hotkey`, never a forfeited hotkey, code from the receipt's manifest) and fetches a new king once into `gauntlet/king/<digest>/`. The new tree changes the fingerprint, so the next refresh re-baselines. An unreadable or unverifiable receipt keeps the current king |
| The pool | the daily reveal slides the window ([above](#pools-are-replayed-rounds)) |
| Harness code, `chain.toml` rules, the bundled king | CI (`publish-miner.yml`) rebuilds, smoke-tests (`deploy/miner-smoke.sh`, which includes `cascade gauntlet selftest`) and publishes `:harness` / `:oneclick` on every relevant push to `main`, every new archived king and weekly; `publish-gauntlet-operator.yml` does the same for the operator image |
| A running stack | the `updater` service pulls every `UPDATE_INTERVAL_SECONDS` (6h). On a new image it writes `gauntlet/RESTART`; the judge finishes its current cycle (never mid-job), writes `RESTART.ack` and exits; the updater recreates every service on the new images and clears the handshake; the judge resumes from disk |

The updater holds the Docker socket (root on that host). Without it, update by
hand with the same handshake: `STACK_DIR=$PWD ./updater.sh --once`. Pin images
(`JUDGE_IMAGE=…:sha-<short>-harness` in `.env`) to stop following `main`.
Lium pods need no updating: the judge streams its own code onto every pod.

`cascade gauntlet selftest` runs one full cycle on synthetic rounds with fake
compute (no GPU, network or LLM). Run it after any change to the harness.

## What workers know

The prompt is short by design (~6k characters): the operator's directives, the
last 10 outcomes with their full failure reasons, the tail of the lessons
notebook, and one instruction: read the briefs and grep `attempts.jsonl` before
editing. The briefs and the full history live in the read-only `knowledge/`
folder (`--add-dir`; in Compose it travels with each queue item, next to the
tree, never inside it). Workers see screen numbers and G3+ pass/fail only.

## The operator (Hermes)

Hermes wakes every `OPERATOR_INTERVAL_SECONDS` (2h by default) with the
`cascade-operator` skill (`deploy/harness/hermes/`). Each wake it:

- reads `status.json`: funnel, population, spend, pending finalists. It never
  sees G3-or-later numbers;
- may rewrite `DIRECTIVES.md`, which every worker prompt includes. A line
  `parent: <id>` pins proposals to one member;
- writes a report to `reports/`;
- may create `STOP` for the anomalies the skill lists.

Hermes (pinned at 0.19.0) is configured with a `custom` OpenAI-compatible
provider (`HERMES_BASE_URL`, `HERMES_MODEL`, `HERMES_API_KEY`) and runs
headless through `hermes chat -Q -s cascade-operator -q`. The key goes into
`model.api_key` in the container's `~/.hermes/config.yaml` (mode 600), because
Hermes sends `OPENAI_API_KEY` only to openai.com. The gauntlet does not depend
on the operator: without it, `DIRECTIVES.md` is just a file you edit by hand.

## Files

```
gauntlet/
  state.json              epoch, window, margins, cycle, phase, progress
  progress.jsonl          the dethrone progress score after G2, G3 and each cycle
  king_cache/             the king's same-budget legs, per (king, image, round, budget)
  events.jsonl            epochs, baselines, deaths, members, explore, G4, finalists, G5
  candidates/<id>/        tree/, meta.json (every stage result), ckpt/ (G4)
  epochs/<n>/king/        the king's legs this epoch (scores per round)
  receipts/               cached signed receipts
  spend.json  jobs/       ledger; per-job spec/result/log
  queue/                  worker queue (compose)
  operator/               status.json, DIRECTIVES.md, NOTEBOOK.md, reports/, STOP,
                          and the briefs (LINEAGE.md, DETHRONES.md, RESEARCH.md)
  knowledge/              what workers read (regenerated each proposal): the briefs +
                          attempts.jsonl (every candidate: change, furthest stage,
                          screen number, exact failure reason)
  submit/                 pending/, frozen/, history.jsonl, HOLD
  king/<digest>/          live kings (king_source = "live")
  STOP                    stop after the current stage
  RESTART, RESTART.ack    the updater handshake (park between cycles)
```

## Limits

- Every number is directional. Replays train on your hardware against kings
  trained on the operator's. Replaying the king's own generator against its
  receipt measures that gap; do it before trusting G4.
- G4 judges each finalist as the round's only challenger. A real cohort round
  applies a family-wise correction.
- G4.5's snapshot is the newest REVEALED one, so it is ~2 days old: it checks
  generalisation to windows nobody scored, not to data newer than the reveal lag.
- Generator code runs in the judge container for the static checks. G1 and
  later run it in the executor: on a pod for Lium, or locally for `local`.
