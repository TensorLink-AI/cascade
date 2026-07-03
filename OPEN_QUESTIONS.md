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

**Default.** A round is one ~24h epoch and screens the whole eligible field down
to one finalist via a cheap heat: every eligible challenger is trained for
`[round] heat_train_hours` on the primary size, the owner scores them on the
held-out pool, and the top `[round] finalists` (default 1) advance to the full
final against the king. So the validator still judges a single king-vs-challenger
contest (now combined across sizes), but the challenger is the heat winner rather
than the lowest-UID generator. Raising `[round] finalists` promotes more than one
challenger to the final; `validator/loop.py::process_round` selects the
best-scoring challenger per size before the verdict.

This replaced the earlier FIFO submission queue with its **1-hotkey-1-eval
lifetime burn** (`cascade/trainer/queue.py`, now removed): under the daily
cadence a miner re-submits each round and competes whenever its latest pre-cutoff
commitment is the heat winner, so a permanent per-hotkey burn no longer fits.

**Flip point.** `cascade/trainer/loop.py::_run_heat` / `TrainerRunner.run_round`
(heat budget, finalist count), and `validator/loop.py::process_round` (per-size
pairing + combined verdict).

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

**Question.** cascade trains a Toto2 backbone from random init in each round's
final. How big a model, and how much compute, so data-quality differences clear
the undertraining-noise floor without making rounds unaffordable?

**Default.** The final trains at **more than one size** — the primary
**Toto2-4M** plus each `[[training.sizes]]` (e.g. **Toto2-22M**) — and the throne
is decided on the *combined* score across sizes (a scaling-aware KOTH; u-μP makes
the 4M-tuned recipe transfer up the ladder). The cheap heat screens on the
primary size only. Each final model trains for a fixed **wall-clock budget —
`target_train_hours` (3h) on the owner's reference GPU**, per size. The intent is
operational ("each model gets ~3h of GPU"), but the *enforced* budget is a fixed
token count derived per size as `target_train_hours × 3600 ×
ref_throughput_tokens_per_s` (each size measures its own throughput). Going
through a pinned token count rather than a raw 3h timer is deliberate and matters
twice over:

* **Fairness / no throughput exploit.** A raw timer gives whichever corpus has
  higher train-throughput (e.g. shorter series ⇒ more steps/sec) *more* gradient
  updates in the same wall-clock — a generator could then win on cheap-to-step
  data rather than better data, a confound orthogonal to quality. A fixed token
  count gives king and challenger identical compute.
* **Reproducibility.** Step count from a timer is hardware/load-dependent, so a
  re-derived audit run wouldn't match; a pinned token count does.

Budgeting by compute (not epochs) also stops a tiny corpus winning by being
memorised in a few passes. `max_train_seconds` is the hard guard above the 3h
target. The hours, throughput, per-size set, and `[generator]` corpus size are
the signal/cost knobs.

**Flip point.** `chain.toml [training]` + `[[training.sizes]]`
(`target_train_hours`, per-size `ref_throughput_tokens_per_s`, architecture),
`[round]` (cadence + heat budget), and the owner's `BaseTrainer`.
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
attention. The variate-axis layers now exist in `toto2_model.py` (one closes
each group of 4, matching the Toto-2.0-4m release) and `Toto2Model.forward`
accepts `(B, C, P, patch_size)` — but the trainer still feeds `C = 1`, so they
train degenerate/dormant until multivariate corpora flip on.
