# Open questions — metronome scaffold

Substantive design calls the initial spec left ambiguous. Each is implemented
with a clear default; the listed location is where to change it if a different
intent was meant. Same convention as horizon's `OPEN_QUESTIONS.md`.

## 1. Manifest trust / training centralisation

**Question.** Validators need to know which trained checkpoint corresponds to
which miner's generator. Who produces that mapping, and why should a validator
trust it?

**Default.** A single owner-operated trainer publishes a signed
`TrainingManifest` to the owner-controlled Hippius S3 manifest bucket (`[storage]
manifest_bucket`, `round-<id>.json` + `latest.json`); validators trust manifests
signed by `[manifest] trainer_hotkey` only. Signing is **wired**:
`sign_manifest` signs `canonical_body()` with the trainer's bittensor hotkey and
`verify_signature` checks it against the configured ss58 address (the validator
gates every round on it; `ValidatorRunner.verify_signatures`). Training is
centralised in v1 because it makes the controlled-experiment invariant trivially
enforceable.

**Flip point.** The decentralisation path: every corpus carries a `corpus_digest`
and every run a `contract_digest`, and every checkpoint a content-addressed CID,
so a validator or a trainer quorum can re-derive the corpus from the pinned
generator + seed, re-train, and compare CIDs/digests to challenge a manifest.
Moving to a re-derivation challenge protocol is the milestone that removes the
single trusted trainer.

## 2. Generation sandbox

**Question.** Generators are miner-controlled code. How isolated must their
execution be?

**Default.** Two layers: a cheap AST static guard at submit time
(`interface/static_guard.py`) and a network-isolated, rlimited subprocess at run
time (`trainer/sandbox.py::run_in_sandbox`). The subprocess pre-flights layout +
size + static guard, runs the generator under POSIX rlimits (address space, CPU
seconds, core, output size) with a scrubbed env (no trainer secrets) and a
wall-clock timeout, wraps it in a `unshare --net` namespace when the host
supports unprivileged user namespaces (probed, with fallback) plus Python-level
socket blocking as defense-in-depth, and returns only `allow_pickle=False`
float64 arrays whose digest the parent re-derives. The trainer selects it via
`build_round_corpus(..., use_sandbox=True)` (the default; `TrainerRunner.use_sandbox`).

**Remaining.** RLIMIT_AS caps *virtual* memory, so torch generators need a
higher `max_repo_mb`/`max_memory_mb`; and `unshare` is unavailable on hardened
hosts (no unprivileged userns), where isolation falls back to the socket guard —
deploy the trainer in a no-egress container for hard network isolation there.

## 3. King identity across rounds

**Question.** The trainer must train the reigning king, but the dethrone
decision is the validators'. How does the trainer learn who the king is without
re-deciding it?

**Default.** King identity flows validators → chain weights → trainer. The
trainer reads the highest-incentive UID on the metagraph as the reigning king
(`plan_round(..., king_hotkey=<highest incentive>)`); validators are the sole
authority for dethroning and set weights accordingly. On a vacant throne
(genesis or king deregistered) the lowest-UID resolvable generator is promoted
to interim king so there is always something to defend.

**Flip point.** `metronome/trainer/loop.py::plan_round` (interim-king choice) and
the live loop's king lookup (TODO in `trainer/main.py`). An alternative is an
authoritative owner-maintained king pointer alongside the manifest; that
re-centralises the decision and is not the default.

## 4. Challengers per round

**Question.** How many challengers does the trainer train and the validator
judge per round?

**Default.** `TrainerRunner.run_round(..., max_challengers=1)` — one challenger
per round, the lowest-UID non-king resolvable generator. Simple and cheap (two
trainings per round). Rotating fairly through the field, or batching multiple
challengers into one manifest, is a straightforward extension.

**Flip point.** `metronome/trainer/loop.py::plan_round` /
`TrainerRunner.run_round`, and `validator/loop.py::process_round` (which today
reads the single `king`/`challenger` pair from the manifest).

## 5. Shared training + generation seed

**Question.** Should the king and challenger share the generation seed and the
training seed, or get independent ones?

**Default.** Both seeds are **shared** across king and challenger in a round
(`trainer/contract.py::RoundSeeds.derive`). Shared `training_seed` means
identical weight init and data-order RNG (the controlled experiment); shared
`generation_seed` means neither generator draws a "luckier" data seed. Both
derive deterministically from the chain block hash.

**Flip point.** `metronome/trainer/contract.py::RoundSeeds.derive`. If you want
per-miner generation seeds (so a generator can't tune to one fixed seed), give
each its own `generation_seed` while keeping `training_seed` shared — but note
that weakens reproducibility unless the per-miner seed is also chain-derived.

## 6. Eval-window source

**Question.** Where do the held-out real-world eval windows come from?

**Default.** A **private, rotating** pool. `chain.toml [eval] eval_source =
"private-rotating"` and `window_pool` names an owner-controlled held-out corpus.
`metronome/validator/windows.py` implements the selection: `RotatingWindowSource`
draws a slice seeded by the round's block hash, so every validator scores the
**same** windows for the king and challenger (paired, consensus-stable) while the
slice **rotates each round** so no fixed set can be distribution-matched
(TIME-benchmark philosophy). This was a public-`gift-eval` identifier in the
scaffold; it was moved to private+rotating to close the benchmark-matching
exploit (a named public benchmark is the easiest thing for a generator to overfit
without producing generally-good data).

**Flip point.** Both halves are now **wired**: the seeded selection/rotation
(`metronome.validator.windows`) and the **pool loader**
(`metronome.validator.pool::load_pool`), which fetches the `window_pool` **Hippius
registry CID**, loads its `.npy`/`.npz` series (+ optional `metadata.json`), and
slices them with `build_windows_from_series`. The live validator loop calls it on
startup. Operator inputs: upload the held-out pool to the registry with
`upload_dir_to_registry`, pin its CID in `[eval] window_pool`, keep it genuinely
held-out, and refresh it periodically so it stays contamination-resistant.

## 7. From-scratch budget and model size

**Question.** metronome trains a Toto2 backbone from random init, twice per round.
How big a model, and how much compute, so data-quality differences clear the
undertraining-noise floor without making rounds unaffordable?

**Default.** The smallest released size, **Toto2-4M**, trained for a fixed
**wall-clock budget — `target_train_hours` (3h) on the owner's reference GPU**.
The intent is operational ("each model gets ~3h of GPU"), but the *enforced*
budget is a fixed token count derived as `target_train_hours × 3600 ×
ref_throughput_tokens_per_s`. Going through a pinned token count rather than a raw
3h timer is deliberate and matters twice over:

* **Fairness / no throughput exploit.** A raw timer gives whichever corpus has
  higher train-throughput (e.g. shorter series ⇒ more steps/sec) *more* gradient
  updates in the same wall-clock — a generator could then win on cheap-to-step
  data rather than better data, a confound orthogonal to quality. A fixed token
  count gives king and challenger identical compute.
* **Reproducibility.** Step count from a timer is hardware/load-dependent, so a
  re-derived audit run wouldn't match; a pinned token count does.

Budgeting by compute (not epochs) also stops a tiny corpus winning by being
memorised in a few passes. `max_train_seconds` is the hard guard above the 3h
target. The hours, throughput, and `[generator]` corpus size are the signal/cost
knobs.

**Flip point.** `chain.toml [training]` (`target_train_hours`,
`ref_throughput_tokens_per_s`, architecture) and the owner's `BaseTrainer`.
Measure `ref_throughput_tokens_per_s` once on the reference GPU; tune the recipe
on a small u-μP proxy width and pin the result here (u-μP transfers it across
width). Raising the hours or corpus size tightens the signal at linear GPU cost;
calibrate `[scoring] win_margin_*` to the residual noise floor. If you genuinely
want "equal GPU-hours" semantics instead of equal compute, drop the derivation
and enforce `max_train_seconds` directly — but accept the throughput confound and
loss of re-derivation auditability.

## 8. Univariate now, multivariate-ready

**Question.** Toto2 is multivariate (variate-axis attention). Should generators
emit multivariate corpora?

**Default.** **Univariate now, MV-ready schema.** `max_channels = 1`, so
generators yield 1-D series, but every container carries a `(C, L)` channel axis
— `check_series`/`drain_generator`, `corpus_digest`, `EvalWindow`, and the
per-channel scorer all already handle `C > 1`. Turning on multivariate priors is
a config flip (`[generator] max_channels`) plus a multivariate `BaseTrainer` and
window pool — **no schema or digest-format change**, so univariate-era corpora
and digests stay valid.

**Flip point.** `chain.toml [generator] max_channels`; provide multivariate eval
windows in the pool and a `BaseTrainer` that exercises Toto2's variate-axis
attention. Until then the variate axis trains on `C = 1` (degenerate) and is
effectively dormant.

---

The questions below were raised while planning the **testnet launch** (the
two-VM setup in `docs/TESTNET.md`). They follow the same convention: each records
the current code behaviour as the **Default/Today** and the knob to change it.

## 9. Idle rounds — re-training the king with no challenger; daily-seed cadence

**Question.** When the FIFO backlog is empty (no challenger to compare against),
should the trainer keep spending a full from-scratch run re-training the king
every round? And does the round seed have to rotate every block?

**Today.** *It re-trains the king every round, idle or not.* `run_round` always
builds `jobs = [(king, "king"), *challengers]` and trains the king first
(`trainer/loop.py`), and `run_forever` starts a new round whenever the **block
hash changes** (`base_seed = client.block_seed(block)`), i.e. roughly every block.
With an empty queue that means a fresh ~45-min king run that produces a manifest
with only a `king` entry — which the validator reads as "no king/challenger pair;
king holds, no state change" (`validator/loop.py::process_round`). So idle rounds
burn GPU and move nothing.

**Intended cadence (the launch plan).** One seed init per **day**, so the king is
trained from scratch **once per day** and that checkpoint is *defended* against
challengers for the rest of the day instead of being re-rolled each block:

* Skip the round entirely when there is no eligible challenger (`select(...)`
  returns empty) — there is nothing to measure, so don't spend the king run.
* Key the round seed to a **daily** epoch (e.g. the block hash sampled once per
  UTC day, or `block // blocks_per_day`) rather than every block, so all rounds
  in a day share the king's init and the comparison stays controlled.
* Cache the day's king checkpoint (registry CID) and reuse it as the king entry
  for each challenger round that day instead of re-training it.

This is also what makes the layered schedule in the launch notes affordable: a
fast **45-min screen** (this default, `target_train_hours = 0.75`) to rank the
field daily, then a longer **~3h king+top-challenger** defence run on the daily
cadence, with headroom to screen the top-1/2 at the larger (20M) size.

**Flip point.** `trainer/loop.py`: short-circuit `run_round`/`_select_challengers`
when no challenger is eligible; derive `base_seed` from a daily epoch in
`run_forever`; add a "reuse cached king checkpoint for the reign/day" path so the
king entry can be an already-trained CID. None of this changes the manifest
schema or the contract digest.

## 10. Validator eval workload — every validator runs the full eval

**Question.** Does every validator independently score the whole eval set, or is
the eval work split across the validator set (one validator → some windows)?
(Raised as "does teutonic have every vali run all evals or just 1?")

**Today.** *Every validator runs the full eval.* `process_round` pulls **both**
checkpoints and scores each on **all** `[eval] n_windows` (2000) windows
(`validator/loop.py` → `evaluator.evaluate_checkpoint`), then runs its own
paired-bootstrap KOTH verdict. There is no work-sharding across validators and no
cross-validator aggregation beyond on-chain weight consensus — redundancy *is* the
mechanism: every validator re-derives the **same** seeded window slice
(`RotatingWindowSource`, OQ #6) and should reach the same verdict, so Yuma
consensus converges without trust in any single validator.

This is the right default for a controlled experiment (the comparison must be
reproducible per-validator), and it is cheap because validators only **score**
two small checkpoints — they never train. The cost knob is `n_windows` ×
`num_samples`, not the validator count.

**Flip point.** If the eval ever gets expensive enough to shard (bigger models,
many challengers per round), split `n_windows` by a validator-indexed seed in
`validator/windows.py` and aggregate off-chain — but that trades reproducibility
and adds an aggregation-trust assumption, so it is explicitly *not* the v1 design.
The teutonic-side question (do their validators each run the full eval or
distribute it?) should be confirmed against teutonic's validator source before we
cite it as precedent — metronome's choice stands on the controlled-experiment
argument regardless.

## 11. Registration slots vs. trainer-queue backpressure

**Question.** Testnet gives ~64 UID slots. Does registration adjust to that, and
how do we keep the trainer's backlog from exceeding what it can process (the goal:
**no more than ~40 in the queue to process**)?

**Today.** *Registration is a chain hyperparameter, and the trainer queue is
unbounded.* How many miners can register and how fast (max UIDs, registration cost
/ adjustment interval, immunity period) is set by **Bittensor subnet
hyperparameters on the netuid**, not by anything in this repo — metronome only
ever reads `client.n_uids()` to size the weight vector (`shared/chain.py`). The
trainer's `SubmissionQueue.pending` (`trainer/queue.py`) grows with the on-chain
field and has **no cap**: with `max_challengers = 1` and two ~45-min trainings per
round, the backlog can fill faster than it drains if the slots fill up.

**To add for launch.**

1. *Chain side* — set the testnet netuid's `max_allowed_uids` and
   registration/immunity hyperparameters so the registered field can't outrun the
   trainer (e.g. cap UIDs and slow the adjustment so the field stabilises around
   the throughput the trainer can actually screen in a day).
2. *Trainer side* — add a **queue cap with backpressure**: bound
   `SubmissionQueue.pending` (e.g. `[queue] max_pending = 40`) and refuse/defer
   intake past it, oldest-eligible-first, so the trainer never commits to a
   backlog longer than ~a day of rounds. The daily-screen cadence (#9) is what
   the "40" should be calibrated against: pick the cap = challengers screenable
   per seed-day.

**Flip point.** `trainer/queue.py::SubmissionQueue.enqueue` (add the
`max_pending` bound + a `SKIP_QUEUE_FULL` reason) and a `[queue] max_pending` key
in `chain.toml`; the registration limits live in the netuid's chain hyperparams,
documented in `docs/TESTNET.md`.

## 12. Deregistration of queued miners; phase-1 linear-projection model

**Question.** A miner can deregister (or re-deploy) while sitting in the trainer's
backlog. How is that handled, and how do teutonic/others handle it?

**Today — already handled.** Each round `_select_challengers` calls
`SubmissionQueue.prune_to_field(field_cids)` (`trainer/queue.py`), which **drops
any pending entry whose generator CID is no longer in the resolved on-chain
field** — exactly the dereg / re-deploy case (a deregistered miner's commitment
stops resolving; a re-deployed miner's old CID is superseded by
`latest-commit-wins`). `select(...)` also re-checks duplicate-of-king and
already-trained at the head of the queue, so a stale entry never reaches the GPU.
This is metronome's content-addressed analogue of teutonic's per-cycle
`evaluated_repos` / field-membership pruning (the queue docstring draws the
parallel). Confirming teutonic's exact dereg handling is a source-read TODO, but
the metronome behaviour above is the launch default.

**Phase-1 linear-projection TSFM.** Separate from dereg: the launch plan wants a
first **larger-model phase** built around a *linear-projection* time-series
foundation model (a linear-attention / linear-projection backbone as the first
rung above Toto2-4M). That is a `base_arch` / `arch_preset` addition under the
fixed-contract machinery (#7) and is sequenced in the scaling plan (#13), not a
queue change. Captured here because it was raised alongside the dereg question.

## 13. Scaling the fixed model — interleaving sizes and warm-starting

**Question.** The fixed model is Toto2-4M today (OQ #7). How do we move the
competition up the size ladder without losing the controlled-experiment property
or making rounds unaffordable?

**Direction (not yet implemented).**

* **Interleave sizes, don't flip.** Run rounds at different fixed sizes on a
  rotation (e.g. mostly 4M screening rounds, periodic 20M/larger rounds) so the
  leaderboard reflects data quality at the size we're scaling *toward*, while keeping
  the cheap size as the high-frequency signal. u-μP makes this honest: the recipe
  tuned once at the proxy width transfers across sizes (README "Why Toto2-4M"), so
  king and challenger stay under one contract at each size.
* **Warm-start from the best checkpoint, not always from noise.** The from-scratch
  invariant (the corpus is the only learned signal) is what makes the measurement
  clean, so warm-starting trades some of that purity for cheaper large-model rounds.
  If adopted, do it on a **cadence** (e.g. rebuild the base from the best checkpoint
  weekly, or once a size's signal goes *stale*), and fold the warm-start CID into
  the `contract_digest` so king and challenger still start from the *same* point —
  otherwise the comparison stops being controlled.
* **Each size is a fixed contract.** Adding a size is a new `arch_preset` +
  `base_arch_digest` + a budget (`target_train_hours`) for that size; the per-round
  invariant (OQ #5/#7) is unchanged within a size.

**Flip point.** `chain.toml [training]` (per-size presets + a round-size
schedule), `trainer/contract.py` (warm-start init folded into the contract
digest), and a size-aware round planner in `trainer/loop.py`. This is the biggest
open design item and should get its own DEC once a size schedule is chosen.

## 14. Generator-as-merged-repo (TempoPFN base) + LLM-as-judge screening

**Question.** Could the competition surface be a single shared synthetic-data /
PFN repo that miners improve via PRs, with an LLM judge screening submissions,
instead of (or alongside) independent generator repos?

**Context from the codebase.** A generator may already **be a trained model**
behind `generate()` — `[dependencies]` allows `torch`/`safetensors` and
`[generator] max_repo_mb` lets a submission ship weights (so a PFN-style generator
is in-contract today). What does *not* exist yet: a shared base repo, a
PR/merge intake, or any LLM-as-judge screening — intake today is purely the
mechanical gate (layout + static guard + hash-locked deps + determinism check,
`metronome verify`) plus the cheap queue dedup.

**To investigate.**

* Audit whether **TempoPFN's** synthetic loader has any *model-weight-based* prior
  design we'd inherit by basing on it (vs a pure analytic synthetic prior) — that
  decides whether "miners PR into one TempoPFN-based repo" is a data competition or
  a model competition, which changes the controlled-experiment story.
* **LLM-as-judge** as a *pre-screen* (cheap, before the GPU round) for submission
  quality/novelty/anti-gaming — useful as a soft filter, but it must stay
  advisory/non-consensus (validators can't depend on a non-reproducible judge for
  weights). Pair it with the existing deterministic gate, don't replace it.

**Flip point.** New intake path (out of scope for the current `interface/` +
`trainer/queue.py` mechanical gate); needs its own design note before any code.

## 15. Eval suite expansion — added benchmarks as a binary qualification gate

**Question.** We want our full eval battery — BOOM, GIFT-Eval, TIME, Tempus-bench,
DS-systems — in the loop. Which feed the KOTH score, and which are pass/fail gates?

**Today.** The dethrone decision runs on **one** scalar: the paired-bootstrap LCB
of `geomean(CRPS/MWSQL, MASE)` over the private rotating window pool
(`eval/scoring.py`, `eval/koth.py`, OQ #6). No public benchmark feeds weights —
deliberately, so a generator can't distribution-match a fixed public set.

**Direction.** Keep the private rotating pool as the **scoring** signal, and add
the named benchmarks as a **binary qualification gate** rather than letting them
move weights:

* *Binary gate* (pass/fail to qualify for a round / to be eligible as king): e.g.
  **BOOM**, **GIFT-Eval**, and **TIME** as "must not regress below threshold X"
  checks. A gate can use public benchmarks safely because passing a floor is not
  gameable into the throne the way a *scored* public set is.
* *Diagnostics* (logged, not gating): **Tempus-bench**, **DS-systems** dynamical
  benchmarks for tracking generalisation across rounds.

This keeps the controlled, contamination-resistant scoring intact while using the
broader battery to stop a generator that wins on the private slice but is broadly
broken.

**Flip point.** `validator/loop.py` (add a pre-scoring gate step) + `eval/`
(benchmark runners) + `chain.toml [eval]` (gate thresholds). Gates must be
reproducible across validators (fixed benchmark + fixed threshold) to stay
consensus-safe.
