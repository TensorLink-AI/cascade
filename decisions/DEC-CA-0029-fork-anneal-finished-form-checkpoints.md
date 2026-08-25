---
id: DEC-CA-0029
type: decision
title: "Fork-anneal: every training run ends with a cosine decay branch — scoring layers see finished-form checkpoints, the lineage continues mid-stable"
status: proposed
date: 2026-08-24
tags: [cascade, training-recipe, lr-schedule, wsd, anneal, contract-digest, benchmarks]
revisit_when: "A testnet cascade has run a full round cycle with anneal_fraction armed (gate for the mainnet cut); or measured anneal cost on the reference GPU materially exceeds the budgeted fraction (then revisit the wall stretch); or the 22M size seam arms (re-measure the anneal fraction at size — 15% was calibrated at 4M)"
relations: {depends_on: DEC-CA-0018, informs: DEC-CA-0017}
---
This is DEC-CA-0018's anticipated decay branch, built. WSD deliberately
deferred the "D": rounds hold base_lr flat so the lineage compounds, and no
in-round decay ever runs. The consequence became measurable once the flip was
live (r33+, 2026-08-22): every artifact cascade scores — duel checkpoints,
the benchmark sidecar's six public numbers, the promotion BenchScores — is a
MID-STABLE model, systematically below its own finished form. The public
benchmark series stepped worse at exactly the flip round, and offline
calibration (2026-08-23/24, OPSLOG) measured the mid-stable penalty at 6–9%
blended on the public suites, non-uniform across generators. Within a round
the comparison stays fair (both sides equally unfinished), but every layer
that reads an ABSOLUTE number — the no-downgrade guard (DEC-CA-0017)
comparing against cosine-era floors, the public benchmark stream, promotion
picks at small gaps — is reading a recipe artifact.

DECISION — fork-anneal, in the trainer, per training run
(`[training] anneal_fraction`, digest-bound, drop-when-default, 0.0 = off;
requires `lr_schedule = "wsd"`):

* **The fork.** When the stable token budget completes, the run snapshots
  mid-stable weights + full optimizer state (the lineage branch), then
  continues the same deterministic corpus stream for `anneal_fraction ×
  train_tokens` more tokens under a cosine decay base_lr → 0.
* **One artifact, two faces.** The ANNEALED weights ship as
  `weights.safetensors` — what the validator evaluator, benchmark sidecar,
  and every downstream consumer already load, so scoring layers see finished
  form with zero validator/schema change. The lineage branch rides beside it
  as `weights_stable.safetensors` + `optimizer.safetensors`; warm-starts
  resume from THAT (the loader prefers the stable file), never from a
  decayed endpoint. This revises 0018's "published beside — never instead
  of" clause for the *scored* artifact specifically: the undecayed line
  remains the carrier of the lineage, but the scored artifact becomes the
  finished fork — that was always the point of deferring the decay rather
  than deleting it.
* **Symmetry preserved.** King and challenger anneal under identical terms
  (same fraction, same schedule, same deterministic stream continuation), so
  the controlled experiment — only the data differs — is intact. Because the
  king's leg is retrained every round, the first armed round is already
  symmetric; no transition round ever compares annealed to raw.
* **Wall stretch.** The wall-clock guard extends by the anneal's share at
  the fork (total = max_train_seconds × (1 + fraction)). A wall that expires
  INSIDE the stable phase ships matching branches (the endpoint is
  mid-stable) with `deadline_hit` marking the under-budget run, and a
  partial decay is visible as `anneal_tokens_seen < anneal_tokens` — the
  DEC-CA-0001 self-penalizing semantics carry over unchanged.
* **Heats too.** The fraction applies to `for_hours` runs, so screen and
  duel stay on the same form — the tied-set selection (DEC-CA-0012) and the
  duel then measure the same object. Heat cost rises by the fraction of an
  already-small budget.
* **Arming is a contract cut.** The field is drop-when-default, so shipping
  the code moves no deployed digest (golden-vector enforced). Setting
  `anneal_fraction` recomputes `contract_digest` — testnet first, then the
  coordinated release-then-activate window (trainer + all external
  validators), the DEC-CA-0018 flip playbook. Proposed initial value: 0.15
  (the offline-calibrated fraction; decay completes well inside the stretch).

Rejected alternatives: annealing in the validator/eval path (puts training
inside the consensus loop, breaks "score exactly what the manifest pins");
annealing only in the benchmark sidecar (bench and duel would then score
different objects, and the guard/promotion would mix forms); a flat
correction offset on mid-stable numbers (the offline calibration measured
per-generator anneal deltas spanning ±1.5pp — the same order as the gaps the
guard adjudicates, so an offset is unsound where it matters most).
