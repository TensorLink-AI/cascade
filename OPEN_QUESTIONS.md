# Open questions — cascade scaffold

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
and every run a `contract_digest`, and every checkpoint a content-addressed Hub
`repo@digest`, so a validator or a trainer quorum can re-derive the corpus from
the pinned generator + seed, re-train, and compare refs/digests to challenge a
manifest.
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

**Flip point.** `cascade/trainer/loop.py::plan_round` (interim-king choice) and
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

**Flip point.** `cascade/trainer/loop.py::plan_round` /
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

**Flip point.** `cascade/trainer/contract.py::RoundSeeds.derive`. If you want
per-miner generation seeds (so a generator can't tune to one fixed seed), give
each its own `generation_seed` while keeping `training_seed` shared — but note
that weakens reproducibility unless the per-miner seed is also chain-derived.

## 6. Eval-window source

**Question.** Where do the held-out real-world eval windows come from?

**Default.** A **private, rotating** pool. `chain.toml [eval] eval_source =
"private-rotating"` and `window_pool` names an owner-controlled held-out corpus.
`cascade/validator/windows.py` implements the selection: `RotatingWindowSource`
draws a slice seeded by the round's block hash, so every validator scores the
**same** windows for the king and challenger (paired, consensus-stable) while the
slice **rotates each round** so no fixed set can be distribution-matched
(TIME-benchmark philosophy). This was a public-`gift-eval` identifier in the
scaffold; it was moved to private+rotating to close the benchmark-matching
exploit (a named public benchmark is the easiest thing for a generator to overfit
without producing generally-good data).

**Flip point.** Both halves are now **wired**: the seeded selection/rotation
(`cascade.validator.windows`) and the **pool loader**
(`cascade.validator.pool::load_pool`), which fetches the `window_pool` **Hippius
Hub `repo@digest`**, loads its `.npy`/`.npz` series (+ optional `metadata.json`),
and slices them with `build_windows_from_series`. The live validator loop calls it
on startup. The **producer** side is also wired: `cascade.pool` (the
`cascade-pool build` CLI) harvests recent real-world series from pluggable
sources (Open-Meteo weather, Wikimedia pageviews; extensible), cleans/validates
them, and writes exactly that loader layout — `--upload` pins the ref. Operator
inputs: run `cascade-pool build --out ./pool --upload --hub-repo cascade/eval-pool`,
set the printed ref in `[eval] window_pool`, keep it genuinely held-out, and re-build periodically (a
fresh `as_of`) so the pool rotates in time and stays contamination-resistant. See
`docs/EVAL_POOL.md`.

## 7. From-scratch budget and model size

**Question.** cascade trains a Toto2 backbone from random init, twice per round.
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

## 9. Warm-start base for entrenched kings

**Question.** cascade trains both models from random init every round (#7),
which is the controlled-experiment baseline. But once a king has defended the
throne for a long stretch, re-deriving it from scratch each round is wasteful,
and the contest keeps re-measuring "best from zero" instead of "whose data best
improves the field's current best model." Should a long-reigning king's training
warm-start from a persisted checkpoint, and if so, *which* checkpoint?

**Default.** Throne logic is **unchanged** — king identity and dethroning remain
the existing sticky paired KOTH on the private, rotating eval windows
(`eval/koth.py`, `validator/state.py`); nothing here touches who is king or how
rewards/weights are set. The change is narrow and lives in the *training
contract*: when the reigning king's tenure exceeds `warm_start_after_days`
(`0` = disabled; ships off), the round swaps random init for a persisted **base
checkpoint**, and **both** the king's and the challenger's generators train from
that *same* checkpoint. Shared init is preserved (now the checkpoint instead of
random weights), shared `training_seed`/`generation_seed` and the fixed token
budget are unchanged, so the controlled-experiment invariant — only the corpus
differs — still holds; the contest now measures whose data best *continues to
improve* the current base.

The persisted base is the king's **best checkpoint by industry-standard
benchmark score** (e.g. GIFT-Eval / Monash). Each round the king holds, the
trainer scores its freshly-trained checkpoint on the pinned public benchmark
suite and, if it beats the currently-persisted base's score, advances a single
pointer to it. Checkpoints are immutable content-addressed `repo@digest`
artifacts on the Hub, so "store the best, overwrite if beaten" is a *pointer*
move — the old digests persist for audit, nothing is destroyed. A **public**
benchmark is used here *on purpose*: unlike the throne eval (#6, private+rotating
to resist overfit), base selection only chooses which of the king's *own*
snapshots to warm-start from — it sets no reward — so an absolute, recognised,
comparable yardstick is exactly right, and the weak indirect overfit pressure is
acceptable because the throne is still decided by the rotating private pool. On
dethrone the base resets: a fresh king trains from scratch again until it
re-establishes `warm_start_after_days` of tenure, then its own best-by-benchmark
snapshot becomes the new base.

**Wiring.** `BaseTrainer.train` gains `init_from: Path | None` (`None` = today's
random init). `TrainingManifest` gains a signed `base_checkpoint: str | None` —
the trained pointer both entries were warmed from (`None` on cold rounds) —
folded into `canonical_body()` and asserted equal for the king and challenger
entries in the controlled-experiment gate alongside
`contract_digest`/`base_arch_digest`. The trainer (the GPU component) scores
checkpoints on the benchmark and maintains the persisted base pointer alongside
`latest.json` in the manifest bucket; validators re-derive the score from the
pinned checkpoint + public benchmark, so selection stays auditable under the
existing signed-manifest trust model (#1) — and even a mis-selected base cannot
steal the throne, it only yields a weaker warm start that the rotating-window
KOTH would expose. Tenure is read from the same bookkeeping that drives the
margin warmup (`koth.py::margin_for_tenure`, `validator/state.py`).

**Flip point.** `chain.toml [training] warm_start_after_days` (trigger; `0`
keeps pure from-scratch) and `[eval] checkpoint_benchmark` (which industry-std
suite ranks the king's snapshots). Days-vs-rounds for the trigger mirrors #7's
hours-vs-tokens trade: days is the operator-facing intent, enforced via a round
count for reproducibility. Choosing "advance to the king's latest checkpoint"
instead of "best by benchmark" would drop the benchmark dependency but lose the
non-regression guarantee on the base.
