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
  Ships inert; validator/audit multi-challenger fix goes FIRST.
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

- **DEC-CA-0018** (PROPOSED, not implemented) — Miner feedback breaks down by
  facet: the heat score decomposed per entrant over `domain` / `freq` / computed
  `shape` / quantile band, with `gap_share` (a cell's signed contribution to the
  entrant's total deficit, sums to 1) as the prioritisation number and
  `rel_median` separating a miner's weakness from a cell the whole field finds
  hard. A pure reduction over scores `_run_heat` already holds (zero extra GPU);
  exact only because DEC-CA-0009 made both halves per-window geomeans; rides
  DEC-CA-0011's unsigned heat document. SHADOW ONLY — gates nothing, ever. No
  `source`/`series_id`/`dgp_class` breakout. Staged: local `cascade score
  --breakdown` first, publish second.
  (`decisions/DEC-CA-0018-generator-weakness-profile.md`, spec
  `docs/WEAKNESS_PROFILE.md`)

New decisions get the next `DEC-CA-####` node in `decisions/` plus a one-line
pointer here (DEC-CA-0012 is claimed by PR-173's tie-aware cohort duel). Put
the revisit condition in the node's `revisit_when:` key.

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
