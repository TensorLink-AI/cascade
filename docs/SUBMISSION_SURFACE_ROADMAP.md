# Miner submission surface — staged expansion roadmap

Status: **proposed** (design pass 2026-08-13; no implementation). Companion
decision records: DEC-CA-0016 (carrier) and DEC-CA-0017…0022 (one per gap).
This document sequences them, states each stage's contract-digest impact and
migration path, and settles the two cross-cutting questions (budget
denomination, the no-weights ceiling).

Binding constraints throughout (none are relaxed by any stage):

- byte-identical corpus at a fixed seed; `cascade verify` is the law;
- king and challenger share one `RoundSeeds` and a byte-identical contract;
- miner code stays untrusted (static guard + network-isolated sandbox);
- every deployed univariate bare-array generator stays valid, no
  resubmission, at every stage;
- the fixed model is Toto2-4M on a ~3h round budget; the one proposal that
  assumes a larger model (long context) is explicitly parked on the 22M seam.

## The ordering, and why the carrier goes first

The carrier (DEC-CA-0016) lands first and almost independently — confirmed,
not assumed: it touches the interface, the digests, the sandbox frame
protocol, and the eval-side shapes, all in ways that are behaviour-preserving
at today's payload (values-only corpora hash byte-identically; `C = 1`
forecasting is unchanged). No gap work can land before it without either
inventing a second migration or hashing a shape the canonical rule would
later re-define. Its only coupling to the gaps is forward: each gap's payload
becomes "accept one more named field".

Value-per-effort ranking of the gaps themselves (this disagrees with the
prompt's ordering; the argument for each is in its record):

| rank | gap | why |
|---|---|---|
| 1 | 6 — multivariate + roles (incl. 4, panels) | capacity already trained-but-dormant in every checkpoint; cost is eval hygiene needed anyway |
| 2 | 3 — missingness | model's CPM machinery is imputation-shaped already; long pole is pool gap support |
| 3 | 1+5 — time anchor / drift | carrier reservation is free; payload structurally blocked by the calendar-free arch pin |
| 4 | 2 — length | long-range priors are miner-expressible today via internal crops; long context belongs to the 22M seam |

## Stages

### Stage 0 — carrier, payload-free (DEC-CA-0016; land now)

Record-or-array yield with `values` only; canonical digest with golden-vector
tests frozen **before** the canonicaliser merges; reserved-name table with
hard rejection; `[generator] interface_version = 1`; bytes-denominated
carrier cap (`max_payload_bytes = 16e9`, numerically identical to today's
point cap); sandbox record frame; eval-side `(C, L)` `ForecastFn` +
per-window cluster fallback (`series_id`, not per-row singletons).

- **Contract digest: no movement.** Everything here is `[generator]` config,
  interface code, or eval internals. Verified: `TrainingContractConfig`
  carries `[training]` only.
- **Migration: none.** Bare arrays remain the documented common case.
- **Lockstep note:** digest-blind code changes to shared scoring paths
  (cluster fallback) follow the DEC-CA-0009 announced-restart discipline
  even though nothing detects drift mechanically.

### Stage 1 — channel correctness + eval groundwork (inert at C = 1)

Fix the trainer channel-drop (`iter_training_batches` consumes all channels,
buckets by `(P, C)`); new-checkpoint wrapper accepts `(C, L)` with a
per-channel adapter kept forever for archived 1-D wrappers; pool builder
stops dropping multi-channel harvested series and guarantees `source` on
every window; `min_clusters` raised per the existing plan; shadow logging of
per-series channel-correlation / effective-rank telemetry.

- **Contract digest: none.** All code + pool content.
- **Migration: none** (max_channels still 1; every path exercised at C = 1
  is behaviour-identical).
- Run the **MV-under-univariate-eval ablation** (DEC-CA-0022) on testnet
  here — its result gates Stage 2's shape.

### Stage 2 — arm multivariate targets (DEC-CA-0022, DEC-CA-0020)

Raise `max_channels` (4–8, per the cost measurement), with multivariate eval
windows live in the pool first. Panels ride as `(C, L)` groups
(DEC-CA-0020); `max_channel_corr = 0.999` arms only after its shadow logs
clear honest generators.

- **Contract digest: nominally none** (`max_channels` is `[generator]`) —
  but this stage should decide open question 4 of DEC-CA-0022: whether to
  mirror the cap into `[training]` so the digest gate can detect a
  cap-divergent trainer. If mirrored, one planned bump.
- **Migration: none** for univariate generators; multivariate is opt-in.
- **Testnet-first**, full cascade, per the DEC-CA-0005/0012 precedent.

### Stage 3 — missingness (DEC-CA-0019)

Pool masked-history slice first (real gaps preserved, `history_mask` in
window metadata, mask-aware wrapper contract); then accept the `mask` field
(pinned-0.0 filler, `max_missing_frac` gate) and consume it (input-mask OR +
loss exclusion).

- **Contract digest: one deliberate bump** — acceptance folds the accepted
  field set / `interface_version = 2` into `[training]` (DEC-CA-0016 G2
  layer 3), making corpus-semantics drift visible to the existing
  `contract_digest_mismatch` gate. Coordinated epoch-boundary restart, the
  routine re-pin protocol.
- **Migration: none**; maskless generators are the unchanged default.

### Stage 4 — roles and future-known covariates (DEC-CA-0022 back half)

Accept `roles` (per-channel, record field — positional convention rejected);
role-aware CPM mask construction; eval covariate windows **only after** the
EVAL_POOL covariate-curation rule (exogeneity by provenance + informativeness
screen) is written and its R² bar measured. `roles = 2` (future-known) stays
refused until then even if 0/1 arm earlier.

- **Contract digest: one bump** (field-set fold; co-schedule with Stage 3's
  if timelines allow, so the fleet restarts once).
- **Migration: none**; roles absent = all targets.

### Stage 5 — long context (DEC-CA-0018; parked on the 22M seam)

`context_length` (and `max_length` in lockstep) rise when the 22M size
activates — that seam already forces throughput re-measurement, arch-digest
recompute, and coordinated restart. Prior work: measured (not estimated)
per-token cost at 8192 on the reference L40S; a pool-composition answer for
sub-daily-only 8K-context windows.

- **Contract digest: bumps regardless** (the 22M activation moves it anyway).
- **Migration: none** — the length band widens, never narrows.

### Stage 6 — scaling ladder: 313M+ on size-conditional silicon (DEC-CA-0023)

Orthogonal to the submission-surface stages (it changes the model contract
and the fleet, not the carrier). Owner-directed shape: **22M screen, duel at
the large size on H100-class pods, small sizes keep their cheap ladders.**
From-scratch-per-round dies between 22M and ~100M on any plausible SKU, so
313M+ rounds are warm-start increments; the duel margin must be re-unitised
from %-of-level to %-of-increment (baseline-referenced statistic — score the
shared warm-start init as a third reference), or dethrones mechanically
stop as the lineage compounds. Increment length (~6h candidate), margin
form, and margin value are chosen jointly from the null-LCB noise-floor
measurement, not by feel.

- **Contract digest: bumps at arming** — new `[[training.sizes]]` block plus
  `SizeSpec.expected_gpu` / `SizeSpec.target_train_hours` overrides; the
  margin rework is a scoring-rule change (lockstep restart, receipt-recorded
  mode, audit replays both rules).
- **Migration: none for miners** — the submission surface is untouched; only
  what happens to their data downstream scales.

### Never scheduled

`start` / `freq` / `group_id` / `labels` / `quantiles` acceptance. Reserved
names with published semantics, refused data — each waits for a consumer,
and for `start`/`freq`/`group_id` the analysis says no consumer can exist
under the pinned calendar-free, batch-independent architecture
(DEC-CA-0017, DEC-CA-0020, DEC-CA-0021). They are carried in the reserved
table so no miner can squat them and no future migration is needed to accept
them.

## Cross-cutting: budget denomination (Part D)

The question assumed one flat point cap taxes long series and multivariate
groups linearly and identically, penalising the SOTA target. Findings:

1. **The binding budget in production is already compute-denominated.** In
   `stream_cpu` mode the stream stops at the training token budget
   (~40B point-passes = 3h × `ref_throughput`), and every point costs
   exactly what the model spends training on it. `max_total_points` binds
   only the materialised `cache_reuse` drain and the sandbox output rlimit.
   The premise that the point cap prices the competition is a `cache_reuse`
   fact, not a live one.
2. **The real mispricings found are bugs, not denominations**: multivariate
   points billed but not trained (the Stage 1 channel-drop fix), and — at
   long context — per-token attention cost drifting ~10% above the flat
   token price (acceptable skew at P ≤ 256; revisit at the 22M seam).
3. **Decision:** carrier cap in canonical-payload bytes (Stage 0), compute
   (token) budget stays the economics. Per-series caps + series-count and
   fully compute-normalised budgets are both rejected: the first
   re-introduces shape policy the token budget already prices better, the
   second buys ≤10% fidelity for a schedule-coupled formula every auditor
   would have to reproduce.

Note the corrected premise from verification: **no `[generator]` cap is in
`contract_digest`** — renegotiating budgets never breaks signatures or
manifest gates. Cross-round score comparability is likewise not a consensus
object (rounds are self-contained duels). Budget changes are operational,
not cryptographic.

## Cross-cutting: the no-weights ceiling (Part D)

**Stated as permanent roadmap policy:** generators are code-only. GAN,
diffusion, flow, VQ-VAE generators and anything *fit to real data* are
outside this competition's design space — not an unimplemented feature. The
ban is what keeps the competed object a prior rather than a distillation,
keeps submissions auditable (a 100KB program is reviewable; 100MB of
parameters is not), and keeps the anti-distillation thesis of
`docs/ARCHITECTURE.md` true.

Two honest riders:

- **The ceiling is enforced by extension globs, and its real wall is
  `max_repo_mb`.** `FORBIDDEN_*_GLOBS` (`interface/validation.py:36,56`)
  catch weight *files*; `config.json` is "any JSON object" up to the 128MB
  repo cap, so numeric parameters can ship as JSON undetected. The hole is
  size-bounded, not sealed — `validation.py`'s own comment says as much
  ("extensions are enumerable-by-hand; the size cap is the wall").
  Recommendation attached to this roadmap: lower `max_repo_mb` toward the
  observed field size (~100KB repos, per the dedup cost notes — even 8MB
  leaves 80× headroom) rather than pretend the glob list is the boundary.
- **No narrow exception is defensible.** Every candidate ("small aux
  sampler weights", "quantised seed tables") is a learned-parameter blob
  whose provenance cannot be audited, which is the exact distillation hole.
  The legitimate version already exists without exception: a generator may
  *fit parameters inside the sandbox at round time*, deterministically from
  the seed, from its own procedurally generated data — that is code
  expressing a prior, and it is legal today.

## What each stage must NOT do (standing tripwires)

- Accept any reserved field without a consumer in the same release
  (DEC-CA-0016's refuse-unconsumed rule).
- Move `corpus_digest` bytes for values-only corpora (golden vectors are the
  guard).
- Arm `roles = 2` before the EVAL_POOL exogeneity rule exists in writing.
- Raise `max_channels` before the Stage 1 channel-drop fix ships.
- Loosen the wall, the dup rules, or the determinism check to accommodate
  any of this (DEC-CA-0001, DEC-CA-0008 stand).

## Open questions (consolidated; each with its settling measurement)

1. **MV-vs-univariate ablation** under the current eval — paired testnet
   round, one `RoundSeeds` (gates Stage 2 sequencing). DEC-CA-0022.
2. **Context-8192 throughput** on the reference L40S — replaces the ~10%
   per-token estimate with a measurement (prices Stage 5). DEC-CA-0018.
3. **Calendar-feature ablation at 4M** on GIFT frequency slices — decides
   whether `freq` payload is ever worth an arch divergence. DEC-CA-0017.
4. **Pool gap-rate measurement** (pre-interpolation missing-fraction per
   source) — decides real-gaps vs injected-gaps for the masked eval slice.
   DEC-CA-0019.
5. **Exogenous-covariate informativeness distribution** — sets the R² bar
   for the covariate curation screen. DEC-CA-0022.
6. **Honest effective-rank distribution** from Stage 1 shadow telemetry —
   validates (or kills) the 0.999 channel-correlation gate. DEC-CA-0022.
7. **Digest-visible `[generator]` mirror**: whether `max_channels` (and the
   accepted field set) should be mirrored into `[training]` so validators
   mechanically detect a divergent trainer — decide at Stage 2 arming.
   DEC-CA-0022 / DEC-CA-0016.
