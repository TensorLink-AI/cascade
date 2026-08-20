# CLAUDE.md — cascade

Working notes for AI-assisted sessions on this repo. Keep entries short; each
records a DECISION and its revisit condition, not general documentation.
Decisions now live as graph nodes in `decisions/` — the node is canonical;
this file keeps a one-line pointer per decision so the summary stays
in-context.

## Design decisions

- **DEC-CA-0001** — Throughput policy: "wall is the law". `ref_throughput` (185k)
  is calibrated to a well-fed trainer, not the median miner pipeline (~80k);
  generator throughput is a compute multiplier and mass `deadline_hit`s are
  intentional — do NOT "fix" them by loosening the wall.
  (`decisions/DEC-CA-0001-throughput-wall-is-the-law.md`)
- **DEC-CA-0002** — Mainnet home is netuid 91 (decided 2026-07-14). `chain.toml`
  ships with the mainnet values baked in (netuid 91, L40S pin, worker-image
  digest, `pool_bucket`).
  (`decisions/DEC-CA-0002-mainnet-netuid-91.md`)
- **DEC-CA-0003** — Provisioner rules of escalation: an EMPTY stage walks the
  SKU ladder under a 30-min wall-clock deadline, a below-50% fleet gets one
  same-SKU top-up, failed stages retry on a 15-min cooldown while their
  window lasts, the final rents JIT at the heat_complete marker (mainnet),
  and the heat ladder's floor is 2× pods (no 1× singles).
  (`decisions/DEC-CA-0003-provisioner-rules-of-escalation.md`)
- **DEC-CA-0005** — Cascade warm-start: half-built (validator promotes, trainer
  never consumes). Revert testnet `cascade_enabled`; implement deterministic
  reign clock + synchronized handoff BEFORE trainer consumption + audit rework;
  mainnet stays unarmed until both survive a full testnet cascade.
  (`decisions/DEC-CA-0005-warm-start-sequencing.md`)
- **DEC-CA-0004** — Cascade promotion PERSISTS the king (re-crown, reset clock
  only); vacate removed, not configurable (consensus-critical). Vacate had no
  benefit (shared init ⇒ no incumbency advantage; old king earns through the
  vacancy anyway) and stalled all future promotions. Kills DEC-CA-0005's handoff-sync
  workstream. (`decisions/DEC-CA-0004-cascade-persist-throne.md`)
- **DEC-CA-0006** — Heat screen keeps ranking on the observed geomean; the
  bootstrap is a SHADOW diagnostic (`p_best`, `leader_lcb`), never the selection
  rule. "Rank by lowest UCB" was simulated and rejected (~91% of marginal
  variance is shared window difficulty; penalises dispersion the duel doesn't
  score; −20pp in the adverse case). Also aligns `global_geomean` to the
  bootstrap's geometric-mean MASE.
  (`decisions/DEC-CA-0006-heat-lcb-diagnostics-not-selection.md`)
- **DEC-CA-0008** — Pre-heat content dedup on EXACT identity only (tree /
  token / rename tiers, pairwise, never transitive); the 0.99 similarity
  threshold was REMOVED — a ratio bar is gameable by spacing and it
  false-dropped a finalist. Dropped copies still burn; `config_only` shadow-
  labels; behavioral probe enforces generator determinism and collapses
  identical-output processes (refuses to run without a kernel-enforced
  sandbox); copy contests resolve on earliest COMMIT (witnessed while
  sealed) — never on UID, which recycles; LLM judge is advisory-only, never
  in the enforcement path.
  (`decisions/DEC-CA-0008-content-dedup.md`)

- **DEC-CA-0009** — The CRPS half of the round metric is a per-window geomean of
  WQL, not a pooled MWSQL. Pooling a ratio weights each window by its magnitude;
  on a pool spanning ~15 orders of magnitude three windows reached 100% of the
  denominator (effective n=3) and held the throne for 8 rounds against
  challengers the diagnostics scored as better. Zero-`sum|y|` windows are masked
  from that half, not floored. Margin/alpha deliberately untouched; no receipt
  format change (it would break the signed audit trail) — `cascade-audit`
  replays under both rules. Trainer + validator must deploy together.
  (`decisions/DEC-CA-0009-scale-invariant-crps-aggregation.md`)

- **DEC-CA-0010** — Host variance is MEASURED, not normalized. Every run stamps a
  `host` record (lane geometry, CPU/GPU capability, opaque pod + machine ids, and
  a FIXED pre-stream calibration bench) into the public training log; nothing
  consumes it. `[telemetry]`, not `[training]` — no `contract_digest` change, no
  wire-format change. The bench cannot be resized (only disabled), or its numbers
  stop being poolable across the fleet.
  (`decisions/DEC-CA-0010-host-telemetry-measure-not-normalize.md`)
- **DEC-CA-0011** — Heat standings publish when the HEAT settles (trainer →
  `status/heat.json` + `heats/round-<id>.json` + `heats/index.json`), not when a
  validator's receipt lands hours later — and they publish for a round rejected
  at a gate, where the manifest copy never appears. Same field shape as the
  manifest block, still unsigned/presentational; a no-screen round publishes its
  reason so the live pointer can't serve the previous round's ranking.
  (`decisions/DEC-CA-0011-heat-standings-published-at-heat-completion.md`)
- **DEC-CA-0012** — `[round] finalists` responds to the screen's own statistic:
  a separated leader advances alone, a tied top is re-scored on a larger eval
  (CPU-only, incremental windows via the seeded-prefix property), and survivors
  advance capped. The whole cohort then duels — NOT sequentially (a full paired
  eval is 106s CPU; stopping early saves ~53s per challenger against 3h GPU
  lanes, decides ties by ordering noise, and breaks the alpha correction via
  optional stopping). Gate on LCB ≥ margin, crown the best POINT ESTIMATE among
  clearers. Per-challenger alpha tightens to `α/k` (quantile, NOT margin) or `k`
  challengers triple the king's false-dethrone risk; `k` derives from the signed
  manifest, so the adjusted alpha must NOT go in `VerdictRecord.params`; receipt
  publishes `cohort_k` + per-challenger LCBs (drop-when-default, signatures
  survive). Screen's drop bar stays 0 and uncorrected (field-size-dependent bars
  are padding-gameable). No heat-based dethrone-hopelessness gate, ever —
  cross-hardware bias, and u86 won with an 8.3% LCB from apparent hopelessness.
  ARMED 2026-08-20 (`max_finalists = 3`, owner; testnet multi-cohort duel
  validated). The tie RUN-OFF stays 0 under the jittered draw (DEC-CA-0019):
  its incremental windows assume the seeded-prefix property the jitter does
  not provide — re-arming needs a jitter-aware incremental draw first.
  (`decisions/DEC-CA-0012-tie-aware-finalists-cohort-duel.md`)
- **DEC-CA-0013** — Warm-start promotes a TOP-K member set (parallel lineages,
  zero extra GPU; rounds rotate across members) and flips to propose-and-verify:
  the TRAINER selects (signed `PromotionRecord`, king+challenger candidate pool,
  structural-diversity policy), validators verify an envelope (provenance +
  quality floor `cascade_quality_epsilon` + reign-clock ripeness + `cascade_top_k`
  cap) instead of re-deriving. Promotion pays the checkpoint's owner nothing.
  Supersedes DEC-CA-0005's "validator promotes" framing.
  (`decisions/DEC-CA-0013-warm-start-top-k-propose-and-verify.md`)
- **DEC-CA-0014** — The from-scratch signal survives warm-start, STAGED:
  (1) shadow scratch control — every M rounds the trainer also trains the
  king's generator from scratch and publishes the bench numbers, telemetry
  only; (2) reseed valve — scratch checkpoints enter the promotion pool via
  the existing quality floor (auto-admitted exactly when the lineage stops
  compounding; the one sanctioned crossing of DEC-CA-0013's generation band);
  (3) a second random-init THRONE only if shadows show the two regimes crown
  different generators. Stage 1 BUILT 2026-08-15 (`[telemetry]
  scratch_shadow_every_rounds`, signed `benchmarks/scratch/` reports,
  consensus-inert, rides the pinned worker CLI; testnet armed M=2) — one-cycle
  testnet validation (docs/SCRATCH_SHADOW.md) gates mainnet M=4.
  (`decisions/DEC-CA-0014-scratch-control-staged.md`)

- **DEC-CA-0015** — Promotion members selected by MEASURED error decorrelation
  (per-window residuals, log-centered per window) within the unchanged quality
  frontier; structural spacing is the vectorless fallback. `cascade_top_k`
  STAYS 3 — it is a consensus constant and netuid 91 has external validators;
  k=5 needs release-then-activate. Correlation source is the private-pool
  battery until raw GIFT rows are persisted beside bench reports.
  (`decisions/DEC-CA-0015-promotion-error-decorrelation-selection.md`)

- **DEC-CA-0016** — Dethrone margin DECAYS with king tenure (affine schedule in
  reverse: 2% → 0.5% over 8 rounds; floor must stay > 0 — it is the safety
  property, above the bootstrap noise band). Tenure survives warm-start
  promotion, so decay deepens across a hold. ACTIVE, ARMED AT RELEASE (owner
  2026-08-15): chain.toml ships the live schedule, so the release IS the
  activation — all 6 external validators upgrade in one coordinated window or
  verdicts fork (docs/MARGIN_DECAY_ROLLOUT.md); receipt replay
  (`scripts/replay_margin_decay.py`) runs pre-release for the announcement.
  (`decisions/DEC-CA-0016-tenure-decay-margin.md`)

- **DEC-CA-0017** — Promotion no-downgrade guard: a ripe reign whose best
  candidate benches WORSE than the live generation's best member HOLDS (clock
  stays ripe, candidates accumulate, fires on the first equal-or-better) —
  the shared init never ratchets downhill. Pure trainer policy (DEC-CA-0013),
  zero consensus impact. Global all-time top-k pool considered and REJECTED
  (winner's-curse freeze + Goodhart lock-in + envelope is reign-scoped);
  basin escape stays with DEC-CA-0014's staged path.
  (`decisions/DEC-CA-0017-promotion-no-downgrade-guard.md`)

- **DEC-CA-0018** — Warm-start recipe is WSD, not per-round cosine:
  `lr_schedule = "wsd"` warms up once at generation start (the from-scratch
  run), holds base_lr FLAT across warm-started rounds, and defers decay to a
  release cut (not built). wsd rounds checkpoint optimizer state
  (`optimizer.safetensors`, ~3×; Muon momentum + row-EMA + AdamW moments) and
  warm starts re-attach it; missing file ⇒ fresh state, shape mismatch ⇒
  abort. Flip changed `contract_digest` — trainer+validator deploy together.
  (`decisions/DEC-CA-0018-wsd-schedule-optimizer-continuity.md`)
- **DEC-CA-0019** — The per-round eval draw is jittered (TB:DEC-TB-0003 port):
  Dirichlet domain mix around uniform (alpha=4, block=8, capacity-capped,
  without-replacement always), salted-hash series bag, class rotation inert
  until pool snapshots carry `dgp_class`. Draw size 1200 (`mix_target_windows`)
  — at 2000 the caps crush the jitter. Block-gated activation, ARMED AT
  RELEASE (mainnet `mix_from_block = 8892000` = 2026-08-20 20:30 UTC,
  IMMEDIATE by owner decision — externals diverge until upgraded;
  testnet = 1): audit replays each round's own rule.
  Realised mix publishes as an unsigned `composition` manifest block.
  (`decisions/DEC-CA-0019-jittered-round-mix.md`)

## Proposed (design pass 2026-08-13 — miner submission surface; not yet owner-accepted;
## renumbered 2026-08-20: original 0016-0024 collided with the accepted decay/guard/wsd/jitter nodes)

- **DEC-CA-0020** (proposed) — Carrier: `generate()` yields array OR named-field
  record (`values` only accepted); canonical digest byte-identical for
  values-only; reserved names hard-rejected; bytes budget; interface_version.
  (`decisions/DEC-CA-0020-series-record-carrier.md`)
- **DEC-CA-0021** (proposed) — `(start, freq)` reserved, consumed by nothing:
  the pinned arch is calendar-free; payload waits on a measured ablation.
  (`decisions/DEC-CA-0021-time-anchor-reserved-not-consumed.md`)
- **DEC-CA-0022** (proposed) — Length cap: long-range priors are expressible
  today via internal crops; `context_length` growth parks on the 22M seam.
  (`decisions/DEC-CA-0022-length-cap-long-context.md`)
- **DEC-CA-0023** (proposed) — Missingness: parallel `mask` field, filler
  pinned 0.0, NaN rejected; eval-gated on a masked-history pool slice.
  (`decisions/DEC-CA-0023-missingness-mask-field.md`)
- **DEC-CA-0024** (proposed) — Panels ARE variate groups for this arch;
  `group_id` reserved with no possible consumer under the pin.
  (`decisions/DEC-CA-0024-panels-are-variates.md`)
- **DEC-CA-0025** (proposed) — Corpus drift rides the time anchor and is
  imperceptible to the fixed model; yield order is the real (existing) lever.
  (`decisions/DEC-CA-0025-drift-rides-the-time-anchor.md`)
- **DEC-CA-0026** (proposed) — Multivariate arms eval-first (variate-layer
  regime mismatch makes MV self-sabotage under the univariate scorer);
  per-series C bucketed; full-freight channel pricing; roles as record field;
  channel-drop trap fixed before any cap raise.
  (`decisions/DEC-CA-0026-multivariate-roles-covariates.md`)
- **DEC-CA-0028** (proposed) — Ceiling restated no-shipped-DATA (raw series =
  the weights hole); real data enters (if ever) as ONE owner-pinned shared
  corpus generators read (`[training] real_corpus_ref`, digest-bound
  drop-when-default; opt-in `real_corpus_dir` ctor kwarg; parent-side cached
  materialisation; container ro-mount). Machinery landed inert; arming waits
  on the pricing experiment + EVAL_POOL disjointness rule + the corpus.
  (`decisions/DEC-CA-0028-shared-real-corpus.md`)
- **DEC-CA-0027** (proposed) — Scaling to 313M+/1B: per-size GPU pins
  (`SizeSpec.expected_gpu` / `target_train_hours`), size-conditional
  provisioning (300M+ rents H100, owner-directed), 22M screen (mirror-lineage
  option), warm-start duels with the margin re-unitised to %-of-increment
  via a baseline-referenced statistic; decoupled flagship is the fallback if
  the noise floor kills the at-size duel.
  (`decisions/DEC-CA-0027-size-conditional-gpu-provisioning.md`)
- Staged rollout + budget denomination + no-weights ceiling:
  `docs/SUBMISSION_SURFACE_ROADMAP.md`. FULLY IMPLEMENTED to the
  config-only-arming bar (2026-08-14, this branch): Stages 0–1 + the Stage 2
  corr gate; mask/roles acceptance AND consumption behind `[training]
  accepted_fields` (contract-digest drop-when-default keeps today's digests
  untouched); Stage 6 plumbing + the increment margin (`[scoring]
  margin_mode`) + `warmup_flat`. Golden vectors pin corpus/stream/series-key
  bytes, the contract digest, and the signed receipt fixture. What remains is
  measurements, pool content, per-size pins, and deploys — never code.

New decisions get the next `DEC-CA-####` node in `decisions/` plus a one-line
pointer here (DEC-CA-0012 is claimed by PR-173's tie-aware cohort duel;
DEC-CA-0016..0022 are claimed by the 2026-08-13 submission-surface design
pass, DEC-CA-0024 by the 2026-08-14 shared-real-corpus design, status
proposed). Put the revisit condition in the node's `revisit_when:` key.

## Operational invariants (hard-learned)

Canonical node: `decisions/NOTE-ca-operational-invariants.md`.

- `[training]` edits change `contract_digest` → the VALIDATOR must restart too,
  or it rejects every manifest (`contract_digest_mismatch`).
- Pods are rsync'd trees, not git checkouts; `uv sync` needs `--all-extras`
  (torch lives behind the `train` extra).
- Never restart the provisioner inside its pre-boundary trigger window.
- A UID is not a seniority claim: Bittensor recycles deregistered UIDs to new
  registrants. Anything deciding "who was here first" needs a block number
  (commit or reveal), never a UID.
- The orchestrator holds the private eval pool and the trainer's wallet. Any
  path that runs generator code there (today: the dedup probe) needs
  `sandbox_mode = "container"` or `sandbox_strict = true` — the subprocess
  sandbox shares their uid and filesystem, so netns is the only real boundary.

## TensorLink graph

This repo is a spoke of the company strategy graph (`TensorLink-AI/strategy`).
Node ID prefix for this repo: **CA**. Decisions live in `decisions/` as
`DEC-CA-####` nodes (frontmatter per `strategy/knowledge/schema.md`).
Cross-repo edges use namespaced targets, e.g. `ME:EV-0021`, `CO:OQ-C1`.
