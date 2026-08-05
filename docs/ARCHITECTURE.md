# cascade architecture

## The thesis

A time-series foundation model is only as good as the data it was trained on.
cascade makes **synthetic training data** the competitive resource: miners
write data generators, the subnet owner trains **a Toto2-4M backbone from random
initialisation** on each, and the generator whose data yields the best forecaster
wins. By holding the model architecture and the entire training process constant,
the subnet turns a noisy question ("is this model good?") into a controlled one
("is this *data* good?").

Training **from scratch** (not fine-tuning a released checkpoint) is what makes
that attribution clean: a fine-tune confounds data quality with what the
pretrained weights already encode, whereas from random init the corpus is the
*only* source of learned signal. This mirrors Toto 2.0 itself, whose pretraining
mix is 57.5% synthetic and 0% public time series yet still tops GIFT-Eval — the
synthetic prior is the lever, and cascade competes it.

## Roles and data flow

### 1. Miner — submits a generator

A miner writes `generator.py` exposing `Generator(DataGenerator)`, pushes the
repo to the **Hippius Hub registry** (OCI) with `cascade deploy`, and commits a
single on-chain pointer:

```
metro-v1:gen:hippius:<repo>@<digest>
```

The Hub `repo@digest` content-addresses the generator code, `config.json`, and
`requirements.txt` together — it both locates and pins the submission (the OCI
digest *is* the content hash, so there is no separate git SHA). A generator is
code-only (purely algorithmic): no shipped weights of any kind, so a miner cannot
distill a pretrained forecaster into the "generator". The distinction from horizon
is *what is scored*: cascade scores the **data** (via a fixed model trained on it),
horizon scores the submitted model directly. See `docs/INTERFACE.md`.

### 2. Trainer — owner-operated, the GPU boundary

A round is one ~12h epoch (`[round] epoch_blocks`, 24h before 2026-07-28); the
trainer runs one round per epoch (so the king is retrained twice a day). Each
round:

1. Resolves on-chain commitments to `(hotkey, uid, repo, revision)`, keeping only
   those revealed **before the epoch boundary** (`resolve_commitments(...,
   cutoff_block=epoch_start)`) — that boundary is the submission deadline.
2. Identifies the reigning **king** (highest-incentive UID on the metagraph) and
   the eligible **challenger field**.
3. Derives one `RoundSeeds` from the round's base seed (the block hash **at the
   epoch boundary**): a shared `generation_seed` and a shared `training_seed`,
   used by every training in the round — heat and final, all sizes — so the whole
   day shares one random init.
4. **Heat (screen).** Trains every eligible challenger cheaply
   (`[round] heat_train_hours`, ~30min, on the primary/smallest size), scores each
   on the held-out pool (geomean of CRPS/MASE), and keeps the top
   `[round] finalists` (default 1). A challenger that fails to train or score just
   doesn't qualify. Every entrant is screened on the *same* window slice, so a
   joint cluster bootstrap over the field is paired: the heat records `p_best`
   per entrant and a `leader_lcb` against the runner-up on the manifest's heat
   block, saying how decisive the screen was. Those are **diagnostics only** —
   the finalists are the observed ranking, never a bounded one (DEC-CA-0006).
   The standings are published the moment the heat settles — `status/heat.json`
   plus a per-round `heats/round-<id>.json` and `heats/index.json`
   (`cascade.shared.heat_status`) — so the dashboards and `cascade heat` show a
   miner where it placed while the duel is still training, instead of waiting
   for the round's receipt hours later. Presentational and unsigned, exactly
   like the manifest's copy.
5. **Final.** For the king and each surviving finalist, at **every configured
   size** (the `[training]` primary plus each `[[training.sizes]]`, e.g. 4M + 22M),
   **under that one shared seed pair**:
   - opens the round's corpus stream (`cascade.trainer.stream.open_round_stream`,
     selected by `[training] corpus_mode`): `stream_cpu` streams *fresh* `(C, L)`
     series from a sandboxed generator with no reuse (rolling byte-exact digest);
     `cache_reuse` draws a fixed corpus once (also sandboxed) and cycles it. Either
     way the trainer gets one budget-capped iterator (univariate `C = 1` today;
     the channel axis is carried so multivariate priors need no schema change),
   - trains a **fresh Toto2 model from random init** at that size via the owner's
     `BaseTrainer` (`cascade.trainer.contract`; reference:
     `cascade.trainer.toto2_trainer`) — it pulls series until the stream ends, for
     the per-size budget (~3h on the reference GPU, enforced as a fixed
     `train_tokens` count so king and challenger get identical compute), streaming
     per-step metrics (loss, lr, throughput) to **Hippius S3** (and, when
     `[wandb] enabled`, mirroring the *same* records into a live wandb run — one
     per round/competitor/size, tagged with the miner hotkey — so miners can watch
     their generator train as it occurs; observability only, never fed to scoring),
   - stamps one **host** record per run before the corpus stream opens
     (`cascade.trainer.host_probe`, `[telemetry]`): lane geometry, CPU/GPU
     capability, opaque pod + machine ids, and a *fixed* calibration bench
     (`host_bench_tokens_per_s`) that is identical on every pod and independent of
     the submission. It rides the same log channel and is what separates
     pod-attributable slowness from generator-attributable slowness — the wall
     prices generator speed deliberately (DEC-CA-0001), and `data_wait_frac` can
     only rule the *corpus* out. Never signed, never scored,
   - pushes the checkpoint to the **Hippius Hub registry** (OCI) and records its
     size-tagged ref.
6. Signs a `TrainingManifest` (trainer hotkey) listing every trained-model ref
   (one king + finalist pair per size, each tagged with its `size`) and the
   corpus/contract digests, and publishes it to the **Hippius S3** manifest
   bucket (`round-<id>.json` + `latest.json`).

`BaseTrainer` is a `Protocol` — the single GPU-dependent seam. Everything else
in the trainer is numpy/CPU and unit-tested. A reference implementation (a
Toto2-4M backbone trained from random init under the `chain.toml [training]`
recipe — `head_dim 64`, `patch_size 32`, a 9-quantile pinball head, u-μP, the
NorMuon+AdamW split) is the operator's to provide; it must be **stateless across
the king and challenger calls** so no information leaks between the two training
runs (shared `training_seed` ⇒ identical random init for both).

#### Two-device (remote) training

By default the king and challenger train sequentially on the trainer's own GPU.
For faster rounds the trainer can dispatch them **in parallel to separate
SSH-reachable GPU pods** (e.g. rented Lium/Targon boxes) via `--remote-hosts`
(`cascade.trainer.remote`). The remote unit is a **round-worker**
(`cascade.trainer.worker`), not a remote `BaseTrainer`: each pod pulls its
generator from the registry by ref, builds the corpus in its own sandbox, trains,
uploads the checkpoint, and returns a `TrainedEntry` receipt over SSH. The
orchestrator collects the receipts and signs + publishes the manifest, so **the
trainer hotkey never lands on a rented box**; pods need registry/S3 access, not
the wallet. The host list is a trainer-local file (`scripts/remote_hosts.example.toml`),
never `chain.toml`.

This preserves the controlled experiment: the budget is a fixed `train_tokens`
count, so king and challenger get **identical compute** regardless of which (or
how fast a) device runs them. King failure aborts the round; a challenger failure
just drops that challenger.

**Byte-exact audit (pinned GPU).** The reference trainer runs deterministically
(deterministic cuBLAS/cuDNN, the math attention kernel, all RNGs seeded from
`training_seed`), so on a **fixed GPU SKU** a re-derived run reproduces the exact
checkpoint. Each run records its `torch.cuda.get_device_name(...)` into the
manifest entry's `gpu_name`, and the validator's gate enforces matched hardware:
with `[training] expected_gpu` set, every entry must report that SKU; otherwise
king and challenger must at least match each other. So pin one SKU on both pods
(e.g. both an H100) and the round is byte-reproducible end-to-end; leave
`expected_gpu` empty and you only lose the cross-round SKU pin, not the
king-vs-challenger guarantee.

### 3. Validator — reads the manifest, decides the throne

The validator never trains. Each round it:

1. Reads the current manifest, verifies its signature and that king and
   challenger share the **contract digest** and **base-arch digest** (the
   controlled-experiment gate — `ValidatorRunner.check_manifest`). The contract
   digest covers every size at once (`[[training.sizes]]` is folded into it).
2. For each trained **size**, pulls the king's and finalist's checkpoints and
   scores them on the **same** held-out real-world eval windows
   (`cascade.validator.evaluator`), then **pools** the per-window scores across
   sizes (king-vs-finalist), preserving pairing because each size shares the
   window `abs_target`.
3. Runs ONE paired-bootstrap KOTH verdict on the pooled scores
   (`cascade.eval.koth.evaluate_round`) — a single throne decided on the combined
   4M+22M skill — and folds it into the champion state.
4. Sets weights: an equal share across the current king plus up to
   `[scoring] reward_prior_kings` registered prior kings (`reward_prior_kings = 0`
   ⇒ winner-take-all on the king; burns to `burn_uid` if none are registered).

### Cascade — king-reign promotion

On top of the daily KOTH sits **Cascade** (`cascade.validator.cascade`), a
wall-clock ratchet that periodically raises the floor the whole field trains up
from. A **reign clock** counts days since the current king last took the throne;
every dethrone re-crowns and resets it (Cascade reuses the KOTH dethrone signal —
it never re-implements dethroning). During a reign every checkpoint the king
produces is scored on the three public suites — **GIFT-Eval, BOOM, and TIME** —
`score = geomean(gifteval_crps, gifteval_mase, boom_crps, boom_mase, time_crps,
time_mase)`, lower better — and kept in a per-reign log. All three suites report
CRPS/MASE the same way — the shifted geometric mean, across tasks, of each metric
**normalized by the Seasonal-Naive baseline** (≈1.0 = baseline parity) — so the
six numbers are the same kind of quantity before they enter the geomean. When a king holds the
throne `[scoring] cascade_reign_days` (default 7) consecutive days undethroned —
counted in blocks (7200/day), anchored to the manifest's epoch-start block so
every validator fires on the same round — a
**Cascade** fires: the reign's lowest-score checkpoint (a lookup, not a re-eval)
is installed **as-is** as the warm-start init for all subsequent rounds; the king
**persists** on the throne with a fresh reign clock (DEC-CA-0004 — both roles
train from the shared init, so promotion confers no advantage worth vacating
over, and the throne only changes hands via a genuine dethrone). The reign clock
and checkpoint log persist next to the champion state, so Cascade survives
validator restarts.

Those six numbers are **authoritative from the trainer, not recomputed per
validator**. The trainer (owner-operated, already the manifest trust anchor)
benchmarks **both final-duel checkpoints — the king's and the challenger's —
strictly after the round's checkpoints are pushed and the manifest is
published**: manifest publication never waits on a benchmark, so validators
start scoring the duel the moment it lands, and a bench failure or timeout can
never fail, delay, or modify the round. The numbers ship as a **separate
trainer-signed artifact** — one JSON report per round at
`benchmarks/round-<id>.json` in the manifest bucket (same hotkey signing scheme
and same R2 dual-write as the manifest; `cascade.shared.bench_report`) — never
inside the manifest entry, so every validator records the identical values and
Cascade selection stays deterministic across validators rather than each
re-running a non-bit-reproducible GPU sweep. The validator reads the king's set
from the report into the reign log (the report lands ~30-60 min after the
manifest, so it re-probes for a few rounds), falls back to the in-entry
`bench_scores` older manifests carry, and tolerates a missing report — that
round simply contributes no bench numbers, and promotion selects over the reign
rounds that have them. The challenger's set is telemetry (and, when `[wandb]
enabled`, the king/challenger pair is logged per round). The eval is the
**full** GIFT-Eval + BOOM + TIME battery each round (`[eval]
cascade_bench_max_series = 0`; BOOM full ≈ 26 min on an RTX 5090, run with
`--bench-device cuda`), and TIME's Seasonal-Naive baseline —
checkpoint-independent — is cached so only the model forward is paid per round.
Because the sweep runs on the JIT-rented final pods *after* their normal
teardown signal (the manifest), the trainer brackets it with
`bench_pending.json` / `bench_complete.json` markers in the shared work root
and the provisioner holds the **final** pods — only that class — until the
report is uploaded or `[eval] bench_hold_max_hours` (default 2 h) expires, then
reaps regardless. The dethrone verdict itself stays entirely on the private
eval pool; these public-benchmark numbers drive only Cascade's warm-start
promotion. Cascade is opt-in — `[scoring] cascade_enabled` (off by default) —
and when off the trainer skips the eval, never writes the hold markers, and
validators run pure KOTH.

## The controlled-experiment invariant

For a round to be a fair measurement of data quality, at **each size** the king's
model and the challenger's model must differ in **exactly one** thing: the corpus.
cascade enforces this on three sides:

* **Trainer:** one `RoundSeeds` instance is reused for every run in the round —
  heat and final, king and challenger, all sizes — so weight initialisation
  (`training_seed`, the from-scratch init) and the generation seed are identical;
  only the per-size width/depth changes between sizes, never between king and
  challenger of the same size.
* **Manifest:** `contract_digest` (sha256 of the `TrainingContractConfig`,
  including every `[[training.sizes]]`) and `base_arch_digest` are recorded once;
  each size's frozen-arch digest is folded into the contract digest.
* **Validator:** rejects any manifest whose digests don't match its own
  `chain.toml`, so a tampered or mismatched training run can't score.

Auditability: because both seeds derive deterministically from the chain block
hash and every corpus carries a `corpus_digest`, a second honest trainer (or a
suspicious validator) can re-draw the corpus and re-train to confirm the run.

## Scoring

Per window, per channel, per model: MASE (Hyndman seasonal-naive denominator) and
the gluonts `MeanWeightedSumQuantileLoss` components `(qloss_per_q, abs_target)`
over the 9-level grid `0.1…0.9`. That grid is *exactly* Toto 2.0's training
objective — its quantile head predicts those nine levels under pinball loss — so
cascade's **score objective equals the model's train objective**, which
collapses the metric-layer gap between what's trained and what's measured.
Univariate windows produce one score each (`channel = 0`); a multivariate window
contributes one row per channel.

The KOTH decision is a **paired bootstrap LCB** on the relative improvement of
`geomean(WQL, MASE)`, challenger vs king, resampling window clusters once per
bag. Both halves are **geometric means over windows**, each window normalised by
its own scale first: WQL by that window's `sum|y|`, MASE by its own seasonal-
naive denominator. The challenger wins a round iff that LCB clears the win
margin on at least `min_windows` common windows.

The CRPS half was originally a *pooled* MWSQL — numerators and denominators
summed across all windows, divided once — which is scale-dominated on a
cross-domain pool: on a 2000-window round in July 2026 three high-magnitude
series (BTC difficulty, two US-debt series) were measured at 100% of the
denominator, giving half the decision statistic an effective sample size of 3.
Normalising per window first is what makes the aggregate scale-invariant.
Windows with `sum|y| == 0` have no scale to normalise by, so WQL is undefined
there and they are excluded from that half (they still count for MASE).
Receipts written before 2026-07-28 were judged under the pooled rule;
`cascade-audit` replays each receipt under both and reports which one
reproduces the recorded LCB. The windows
are a **rotating private slice** (`cascade.validator.windows`): seeded by the
round's block hash so every validator scores the identical set and the king/
challenger comparison is paired, but rotated each round so no fixed eval set can
be distribution-matched.

Dethroning is configurable. The shipped `chain.toml` sets `dethrone_cp = 1` with
a flat margin (`win_margin_start == win_margin_end`, `margin_warmup_rounds = 0`),
so a single round that clears the margin takes the throne and every king is
equally challengeable regardless of tenure. The sticky, tenure-weighted variant
is still available: set `dethrone_cp > 1` (a challenger must then win that many
**consecutive** rounds; a single loss or inconclusive round resets the streak)
and let `win_margin_end > win_margin_start` ramp over `margin_warmup_rounds` of
tenure so an entrenched king must be beaten more decisively.

**Stale-throne margin decay.** The shipped config also arms the opposite lever:
once a king survives `margin_decay_after_rounds` (3) rounds undethroned, the win
margin's excess above `margin_floor` is multiplied by `margin_decay_rate` (0.5)
each further round — `0.02 → 0.01 → 0.005 → …` — so a throne nobody takes gets
steadily cheaper to take. At the floor (`0.0`) a challenger still needs a
statistically significant improvement (LCB ≥ 0), just no extra margin. A
dethrone resets the king's tenure and with it the schedule for the new king.
The decayed margin is a pure function of `king_tenure_rounds`, which the
receipt already records, so audits replay it without any receipt change. It is
consensus-critical: all validators must run the same `margin_decay_*` values.

**Public-benchmark no-regression gate (optional, off by default).** With
`[scoring] gift_gate_mode = "enforce"`, a dethrone additionally requires that the
challenger has not *statistically meaningfully regressed* on broad public data
(GIFT-Eval). On a private-pool win, both models are scored via the isolated
`benchmarks/` sidecar and a **paired no-regression bootstrap**
(`cascade.eval.gift_gate`) checks `lcb >= -gift_gate_tolerance` on the shared
configs. The gate is **not winnable** — it can only block a dethrone the private
LCB already granted — and an uncomputable gate (sidecar down, too few configs,
or king/challenger on different pinned data revisions) makes the round
inconclusive rather than silently passing or failing. `gift_gate_mode = "shadow"`
computes and logs the verdict without enforcing it, to calibrate the tolerance
against real noise first.

## Trust model (v1) and the path to decentralisation

v1 centralises training in the owner's trainer and trust in `[manifest]
trainer_hotkey`. This is the pragmatic bootstrap: it makes the controlled
experiment trivially enforceable. The corpus/contract digests already make every
run *reproducible*, which is the hook for decentralising training later (have
validators or a trainer quorum re-derive and challenge a manifest).

## What's implemented vs. a boundary

Implemented and tested (numpy/CPU): the generator contract + output checks (with
the MV-ready `(C, L)` channel axis), the static guard, commit/pointer parsing
(Hippius Hub `repo@digest` scheme), config (the full from-scratch Toto2 contract, digest-pinned),
the manifest schema + digests + **signing/verification**, the full scoring + KOTH
math, the champion state machine, corpus building from a generator, the trainer's
pairing logic, the **Hippius storage layer** (Hub ref grammar + S3
manifest/log/pool-snapshot layout), the rotating private window selection *and* the **eval-pool
loader** (`cascade.validator.pool`), and the trainer-round assembly + both
**live service loops** (`trainer/main.py`, `validator/main.py`).

The **Toto2-4M from-scratch `BaseTrainer`** ships as a runnable reference
(`cascade.trainer.toto2_trainer`) behind the `[train]` extra — a causal patch
transformer with a 9-quantile pinball head, u-μP-style init, a Muon+AdamW
optimiser split, and a token-budget LR schedule. It is the one piece that needs a
**GPU to validate end-to-end** (no GPU in CI); run a real round on your reference
box, then pin `base_arch_digest` / `ref_throughput_tokens_per_s`. Other operator
inputs before launch: the Hippius `[storage]` credentials/endpoints and the
held-out eval-pool ref (`[eval] window_pool`). The corpus sandbox subprocess
caveats are unchanged.
