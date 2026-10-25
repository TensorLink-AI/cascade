---
id: DEC-CA-0050
type: decision
title: "v2 incentive mechanism: Cascade-MOE-28M king lineage (lane A) + a concurrent domain-expert lane (lane B) that trains a frozen-router expert subset for one tsbench-forge domain, merges into the king, and pays on a 5-round decay"
status: proposed
date: 2026-10-08
tags: [v2, moe, scoring, emissions, consensus, eval, domains]
revisit_when: "the open questions below are answered by the owner; then after the first testnet era with lane B armed — record per-domain gains, other-domain/benchmark deltas on merged kings, how often concurrent selections collide, and the lane-B leg cost at the chosen budget"
relations: {depends_on: [DEC-CA-0001, DEC-CA-0013, DEC-CA-0036, DEC-CA-0039, DEC-CA-0041, DEC-CA-0043], relates_to: [DEC-CA-0014, DEC-CA-0027, DEC-CA-0032]}
---

## Infrastructure (2026-10-17)

The direct runtime runs on cascade's existing infrastructure, not AWS: see
DEC-CA-0052 (Lium/Vast/RunPod miner-funded pods, Hippius, emission rewards,
check-only validators).

## Latest owner direction — build direct ~220M competition

The proxy stage has been cancelled. Current requirements are in
[the direct-MoE lane document](../docs/V2_DIRECT_MOE.md): king-funded
subsequent blocks, cached incumbents, lifetime batch ramp, broad/expert phases,
combined and individual bootstrap guards for GIFT-Eval/FEV/TIME/BOOM, and
capacity/failure recovery with protected admitted-job rights. The old two-scale
proposal is superseded. Implement and test; this instruction does not activate
live contracts or authorise real infrastructure spending. The original design
below remains the record of the existing branch implementation.

## Decision (proposed — owner request 2026-10-08)

Owner: "build a v2 incentive mechanism … a new competition to train a MoE …
insert a 2nd competition type … 35–50% of emissions … a generator to train a
subset of experts to improve one specific domain of our eval … these get
merged into the king … cap the number of days a miner can get the reward:
after 5 rounds the emissions decay from 100% → 10%"; "tsbenchforge as main
eval"; "the 2nd competition runs concurrently to the first — we determine the
share of both dynamically"; "regression guards for BOOM, TIME, GIFT-Eval,
FEV, and potentially dsbench scoring in the future as part of eval weight".

### Architecture: Cascade-MOE-28M (`base_arch = "cascade_moe"`)

`cascade/trainer/moe_model.py` (landed, tested; 191.5M total / 27.95M active
at the pinned shape). A NEW file, not an edit of `toto2_model.py` — that file's
bytes pin the live toto2 king's `base_arch_digest`. `compute_base_arch_digest`
must dispatch the model source on `base_arch`, keeping toto2's digest
byte-identical. Where the owner spec is silent, toto2 conventions are kept
(weightless pre-norm, PerDimScale, xPos, u-μP residuals, robust scaler) — each
is a candidate ablation, not a commitment. The arch switch is a new
generation: random init (spec §7), contract-digest change, coordinated
trainer + validator release.

### Lane A — main KOTH (unchanged mechanism, new model + eval)

Same duel, margin, era-king, warm-start lineage as today, on the MoE preset.
The primary private eval becomes the **tsbench-forge** pool (already a pool
source; carries the 7 GIFT-Eval domain labels lane B needs).

### Lane B — domain experts (new, concurrent)

* **Opens at 200B tokens** (owner 2026-10-08) of king-lineage training,
  counted in CHANNEL-TIME points (what the model processes — the spec's
  64×8×4096 ≈ 2.1M per update), not series-points (DEC-CA-0042 bills a
  C-channel series as L, so the two units differ by up to 8× at C = 8).
  Co-gate (built, trainer policy): routing saturation (OLMoE, arXiv
  2409.02060) between the PREVIOUS era's king and the current one on each
  domain's affinity slice — on EVERY MoE layer of EVERY domain, token-level
  top-k overlap ≥ `routing_overlap` (0.8) and ESFT-selection Jaccard ≥
  `routing_jaccard` (0.7). One-way latch once open; per-layer numbers are
  kept as a saturation curve in the driver state. Lane B hangs off the
  king's routing, so it should not open while the router is still moving;
  200B is the floor, not the trigger.
* **Entry.** A miner submits a generator plus a declared target domain `d`
  (one of the pool's domain labels). Declared at commit — never inferred
  after scoring.
* **Expert selection is the trainer's, not the miner's.** Run the CURRENT
  king over a domain-`d` *affinity slice* (disjoint from the scored windows),
  count routing assignments per expert (`domain_expert_affinity`), and pick
  experts by the **ESFT-Token** rule (Wang et al., EMNLP 2024, arXiv
  2407.01906 — the closest published setup: tune only task-relevant experts,
  freeze router + shared experts + attention): per layer, rank by token-
  selection ratio and take experts until the cumulative ratio reaches
  `p = 0.2`, clamped to 8..38 of 256 (ESFT trains 5–15% of experts per layer
  at that p and retains general ability better than training all experts:
  61.5 vs 58.8). `select_domain_experts`, ties to the lower index. Experts
  held by another leg are excluded, so concurrent legs are disjoint by
  construction (2026-10-19: "held" = open, finished-but-unsettled, or a
  pending merge until the era init consumes it; and a layer never pads up
  to 8 with zero-affinity experts — see "Merge fixes" below).
  **Trust model (2026-10-19, corrected):** validators do NOT re-derive the
  rows. `judge_entry` accepts a `recompute_selection` hook, but the live
  validator passes none: the affinity slice's pool snapshot and block are
  not in the manifest, the rows are planned on the leg's BASE king rather
  than the settlement's king, and an exact match over float32 CPU routing
  on heterogeneous validator hardware would fork on a tie at the ESFT cut.
  The rows are trusted trainer output. What binds a leg: the merge
  byte-check (it changes nothing outside its declared rows), the verdict
  and every guard on those rows, and (from `merge_guard_from_block`) no
  overlapping or non-finite merge. The manifest records the real
  `exclude` (it was `{}`) for transparency, not re-derivation.
* **Legs.** One leg per submission; at most `legs_per_domain = 1` open per
  domain at a time (owner: "maybe 1"), so up to one lane-B leg per domain
  runs concurrently with lane A.
* **Leg.** Warm-start from the king checkpoint; everything frozen except the
  selected expert rows (`expert_grad_mask_hooks`): router weights AND the
  routing bias stay the king's (no bias updates) — a lane-B leg cannot
  re-route other domains' traffic. Seeds derive exactly as lane A
  (controlled experiment: the generator is the only free variable).
* **Judging** (`cascade/eval/expert_lane.py`, landed + tested). Candidate vs
  king, paired cluster bootstrap on domain-`d` windows: superiority gate
  `LCB_α ≥ gain_margin`. Plus the non-inferiority guards below, ALL must hold.
  **2026-10-19 hardening:** `[lane_b] gain_margin = 0.01` (it was 0.0). The
  loader refuses `gain_margin <= 0`: identical scores give LCB = 0, so a zero
  bar merged an unchanged expert. The verdict also needs a strictly positive
  point gain whatever the margin, plus at least `min_target_clusters = 5`
  distinct clusters (pool `source`) among the target windows; fewer is
  `inconclusive`, never a merge (100 windows of one feed are one
  observation). Shared with lane C through `ExpertLaneParams`. Testnet
  validators restart together for this change.
* **Merge — async, delta-transferred, re-verified.** Lane A keeps training
  while lane-B legs run, so a leg's base king is usually stale by merge time.
  1. The leg trains against a FROZEN king snapshot (its anchor).
  2. `merge_expert_slices` byte-checks the candidate against that snapshot:
     it may differ only on its declared rows.
  3. `rebase_expert_delta` carries the leg onto the CURRENT king as a diff
     vector on those rows, `now + (candidate − base)`. Row replacement would
     discard what lane A learned on those rows since.
  4. The verdict (gain + all guards) is RE-RUN on the rebased model vs the
     current king. The leg's own duel never pre-approves the merge.
  5. Merges land only at an era boundary, so every lane-A leg of an era starts
     from one init.
  6. Staleness cap `max_base_age_eras` (start 2): an older leg is refused,
     never rebased. Routing staleness (trainer policy): if the current king
     routes the leg's domain elsewhere — per-layer Jaccard of the re-planned
     ESFT selection vs the leg's rows < `rebase_jaccard` (0.5), or base-vs-
     current top-k overlap < `rebase_overlap` (0.7) — the leg is dropped as
     `routing_stale` the same way.

  Disjoint merges commute. The lane-A throne holder is unchanged by a merge;
  the merged checkpoint is the lineage's next init.

### Research basis — concurrent broad + specialist training

* **FlexOlmo** (AI2, NeurIPS 2025, arXiv 2507.07024): independently trained
  experts that compose. Each owner trains only its own FFN expert plus its
  router embedding against a FROZEN shared anchor (public FFN + attention).
  Removing that coordination drops the average from 47.8 to 38.8 — below
  BTM/BTX. Lane B's frozen-everything-but-selected-experts leg is this
  pattern. FlexOlmo also trains the expert's router row (domain-embedding
  init); we freeze the router entirely for now. Unfreezing the selected
  experts' router rows is the first ablation, with the guards catching any
  re-routing of other domains. The paper does NOT address the anchor being
  updated afterwards — that is our gap.
* **Fine-tuning transfer / Param∆** (Lin et al., EMNLP 2025; Cao et al., ICLR
  2025): a fine-tuning diff vector from an old base transfers to a newer base
  of the same architecture at near-zero cost. This is the basis of
  `rebase_expert_delta`.
* **BTX / BTM** (Meta, 2024): branch a seed, train specialists embarrassingly
  in parallel, mix FFNs into an MoE, then a short joint finetune for the
  router. It is the alternative merge (append new experts plus a router
  finetune rather than overwriting selected ones), parked because it changes
  the expert count, and with it the architecture fingerprint, on every merge.
* **Async DiLoCo / HeLoCo** (2025–26): stale pseudo-gradient correction (keep
  components aligned with the current update direction). A lane-B delta is a
  stale pseudo-gradient, so this is the refinement if plain delta transfer
  shows conflict with lane A.
* **Gap:** no published work continually pretrains the base WHILE
  specialists train against it and merge back repeatedly. The design combines
  (frozen anchor) + (delta transfer) + (re-verify) and must measure erosion.

### Regression guards (lane B) — statistical, not point estimates

Owner: "guards need statistical tests like an LCB bootstrap but might differ".
They do differ: the gain is a SUPERIORITY test, a guard is a NON-INFERIORITY
test — "worse by at most tau with confidence 1 − α" ⇔ `LCB_α(rel) ≥ −tau`.
Requiring every guard to pass is an intersection-union test, so each guard
runs at α with NO multiplicity correction (adding FEV doesn't loosen GIFT).

1. **Other pool domains** — same paired cluster bootstrap per domain,
   `LCB ≥ −tau_domain`. Domains under `min_guard_windows` are POOLED into one
   `_rest` guard (a per-domain bootstrap that small would either block every
   leg or be waved through).
2. **Public benchmarks** — GIFT-Eval, BOOM, TIME (already in the bench
   sidecar) and **FEV** (new sidecar target): per-item paired bootstrap of the
   geomean ratio (`paired_geomean_lcb`; GIFT's 97 configs are the items), CRPS
   and MASE each `LCB ≥ −tau_public`. Needs the sidecar to emit per-item rows
   (GIFT already can via `run_gift_rows`).
3. **dsbench** — later, as a WEIGHTED eval term (not a guard), kept separate
   from forecast scores per the model spec.

### Emissions

* **Split.** A hyperparameter (owner: "like 50:50"): `[lane_b] share`
  (consensus, active from the v2 block), 0.5. `split_emissions` (landed).
* **Within lane B.** `share` is cut into one slot per domain; a domain's
  live win is paid `slot × multiplier`. Multiplier (`reward_multiplier`),
  counted in ERAS (`era_settlements` settlements on the grid in force — 12 h
  on mainnet; amended 2026-10-10, was settlements): 100% for the first 5 eras
  after the merging settlement (now gain-tiered: see "Gain-tiered full-pay
  period 2026-10-20 (owner)" below), then exponential decay (half-life 1 era:
  50%, 25%, 12.5%) to a 10% floor, and the payout STOPS at `win_ttl_eras`
  (28 = 14 days of 12 h eras; the slot burns, the merged weights stay in the
  king forever). Per WIN, so a miner keeps earning only by landing new
  merged improvements; a newer win on the same domain replaces the old one's
  slot. BURN DECAY (owner 2026-10-08): the unpaid remainder (decay, expiry,
  empty slots) burns — otherwise the cap doesn't bite (a lone winner at 10%
  would be renormalised to the whole lane). `remainder = "lane_a"` exists
  only as an owner override. All four values are owner-tunable.
  `replay_settlements` defaults to `(win_ttl_eras + 1) × era_settlements`,
  so a rebuilt validator's lane-B state holds every still-paying win by
  construction (an explicit value below that is refused at load).
* **Lane A** keeps `king_decay` court weights within its share.

### Consensus surface (all block-gated, drop-when-default)

`WindowScoreRecord` gains `domain` (today it has none, so the audit cannot
re-derive per-domain verdicts); receipts gain a `domain_verdicts` block and
the lane-B reward vector; `cascade-audit` replays both. Public-benchmark guard
numbers are trainer-signed `BenchScores` (validators cannot afford a 6h
re-bench per leg) — same trust model as DEC-CA-0036's operator re-bench.

## Activation (go-live, owner 2026-10-10)

Owner: "no shadow mode … in the code"; "we just [want] a certain block [the]
new comp goes live". No readiness vote (unlike DEC-CA-0045/0049):

* **One typed key, `[v2] from_block`** (consensus). `0` = v2 not active — the
  mainnet placeholder until the owner types the go-live block. `N > 0` = v2
  for every era whose START block is `>= N`. `N` must be an era boundary on
  the grid in force at it (multiple of `epoch_blocks × era_settlements`) and
  at/after the DEC-CA-0043 rollover; the loader refuses anything else.
  Validators must run the v2 release before `N`. Testnet is armed at a typed
  block (`chain.testnet.toml`).
* **Lane A switches contract at the era start.** The post-v2 contract is a
  full `[v2.training]` table (the Cascade-MOE-28M preset), carried on
  `TrainingContractConfig` as a schedule that is never in `contract_digest`
  (`v2_from_block` / `v2_contract`, like DEC-CA-0047's
  `budget_denomination_after`). The effective contract is `at_block(era
  start)`: every pre-v2 era resolves to `[training]` byte-for-byte, a v2 era
  to the MoE contract; legs carry the era start as `--contract-block`; the
  validator gate, the receipt replay and `cascade-audit` resolve a
  settlement's contract from pure block arithmetic (the era containing
  `boundary − 1`). `train_seed_salt` must match across the two.
* **New lineage from random init.** A toto2 checkpoint cannot initialise the
  MoE. One rule, applied identically by the trainer's era ledger, every
  validator's `_check_era_init` and the audit (`v2_generation_usable`): in a
  v2 era only a generation whose record took effect at `>= v2_first_era + 2`
  (i.e. fired at or after `N` — the trainer stamps `min_effective_era` of its
  fire boundary) may initialise it; any older generation reads as
  generation 0 / random init. The promotion engine starts the lineage at the
  first boundary `>= N` (`start_lineage`: toto2 members stop carrying, the
  candidate log clears, the generation NUMBER continues); validators require
  every member of a v2-lineage record to be the v2 arch and measure its
  quality floor against v2 checkpoints only. The king persists (DEC-CA-0004)
  and its leg retrains under the v2 contract like every leg.
* **Lane B is active iff v2 is** — `[lane_b] from_block` is retired (an old
  key is ignored) and derived as `N + 1`, so the settlement AT `N` (the last
  toto2 era's) stays a pre-v2 receipt byte-for-byte.
* **Slow-adapting router** (the erosion mitigation from the risk list, arm
  (a)): `[v2.training] router_lr_scale = 0.1` (the router param group's LR)
  and `route_bias_rate_scale = 0.25` (the loss-free bias step), both
  digest-bound drop-when-default. They bite only once the lane-B routing gate
  has OPENED — every lane-A leg of an era whose SEED block the latch precedes
  (one rule for king and challengers) carries `--router-slow`; before that
  the router trains at full rate. Expert legs freeze router + bias anyway.
* **Receipts** stamp `v2_block` (drop-when-default) on every settlement that
  judges a v2 era; `cascade-audit`'s `v2` check replays each archived round
  under its own block (stamp, era contract, v2-arch entries, no warm start
  before `v2_first_era + 2`, no toto2 init).
* **Miners** commit an expert entry with `cascade deploy --expert-domain
  <domain>` (`metro-v1:expert:<domain>:hippius:<ref>`; the domain must be in
  `[lane_b] domains`). A hotkey holds one commitment.

## Build order

P0–P5, the generation surface and the go-live wiring above are built. Live
configuration and deployment procedure: `docs/OPERATOR.md` ("Before go-live /
before a testnet cycle").

## Open questions (owner)

1. **Unit of the 200B gate** — channel-time points (recommended) vs
   series-points; the routing co-gate is built — confirm its thresholds
   from the testnet saturation curve.
2. **10% floor duration** — held until a newer win on the same domain, or
   for a fixed further window? (Implemented 2026-10-10: until replaced or
   `win_ttl_eras` = 28 eras, whichever first.)
3. **Domain set** — the tsbench-forge labels as-is? (Configured: the seven
   catalog labels `econ_fin`, `energy`, `healthcare`, `nature`, `sales`,
   `transport`, `web_cloudops` from TSBench-Forge `src/sources/sources.yaml`.)
4. **Tolerances** — `gain_margin`, `tau_domain`, `tau_public`, window
   minimums; set from a testnet replay of the first lane-B legs, not a priori.
5. **Lane-A drift** — lane A keeps training ALL experts after a merge, so it
   can erode merged domain gains. Measure first (per-domain scores across
   eras after a merge). Fallbacks in order: a lower LR on recently merged
   rows, then freezing them for N eras.
6. **Router rows** — keep the router frozen (now), or train the selected
   experts' router rows as FlexOlmo does?

## Scored EMA per leg (owner 2026-10-19)

- `[v2.training] power_ema_scope = "leg"` (mainnet): a lane-A leg's power
  EMAs restart at its warm start, so the scored average covers only that
  leg's training. With "lineage" (the old behaviour, still the default and
  testnet's) `t` carried across warm starts and the window stayed σ_rel of
  the whole lineage: at a constant batch, 25B legs and σ_rel 0.05, the
  current leg's share of the scored weights falls from 96 % at 150B to 37 %
  at 1T (`1 − ((T−25)/T)^(γ+1)`), shrinking the difference between
  candidates while eval noise stays put. The raw lineage weights
  (`weights_stable`) and optimizer state are unchanged.
- Lane-B legs score a power EMA of their own training too (fresh at the
  king's scored weights, `power_ema_eval` of the contract); every frozen row
  is restored to the king's exact bytes before export, so the merge
  byte-check is unchanged. Legs ship the scored file only (no
  `weights_stable` / `power_ema` files). Trainer-only: validators judge the
  checkpoint as before.

## Merge fixes + miner-funded legs (2026-10-19, uncommitted on `v2-moe-expert-comp`)

### State at merge

Everything a lane-B merge touches, and how (`cascade/trainer/lane_b_driver.py`
`LaneBDriver.init_for`, helpers in `cascade/trainer/expert_lane.py`):

| state | how | consensus |
|---|---|---|
| era init `weights.safetensors` (the scored EMA under the power-EMA preset) | `apply_merges`: `init + Σ (rebased − king)` on each merge's rows, float32 add, cast back. Bytes unchanged by this change. | yes: validators rebuild it bit-exactly (`era_gate._verify_merged_init`) |
| era init `weights_stable.safetensors` (raw lineage weights a lane-A leg warm-starts from) | the same delta on the same rows (`merge_deltas` + `apply_merge_deltas`); it was copied unmerged / not written, so the next lane-A leg trained on pre-merge rows | trainer only |
| era init `optimizer.safetensors` (`muon.<param>.momentum` `(E, d_out, d_in)`, `muon.<param>.row_v` `(E, d_out, 1)`, `adamw.<param>.exp_avg[_sq]`) | merged expert rows zeroed (`reset_optimizer_rows`), i.e. a fresh optimizer's state for that row; step counters untouched. Was copied unchanged (stale moments push the merged row back) | trainer only |
| era init `power_ema.safetensors` (`ema<σ>.<param>` shadows + `ema<σ>.__t__`) | the delta on every shadow's merged rows (`merge_power_ema`). Under `power_ema_scope = "leg"` (mainnet) the next leg restarts its EMAs, so this is inert but consistent; under `"lineage"` (testnet) the average resumes around the merged weights | trainer only |
| `config.json`, `model.py`, `forecast_wrapper.py` | copied from the member | — |
| `pending_merges` / expert rows | a merge that overlaps an earlier one's rows waits for the next era init; a non-finite one is dropped; rows stay exclusive until consumed | trainer policy (validator refusal below) |

### Merge guards (consensus, `[lane_b] merge_guard_from_block`)

`"v2"` (the default, mainnet) = `[v2] from_block`; an integer is a typed
era boundary at or after the v2 block (load-checked); 0 = never. From it:

- the era-init gate refuses declared merges that overlap on any (layer, row)
  (`era_init_mismatch: lane-B merges overlap …`);
- `apply_merges(strict=True)` refuses a non-finite candidate row or merged
  row on the rebuild;
- `judge_entry(strict=True)` refuses a candidate with a non-finite value on
  its selected rows at stage `merge`;
- `cascade-audit`'s `lane_b` check replays the overlap rule on the embedded
  manifest's merges.

Testnet types `8280000` (a future era on the 600-block grid): lane B has
been live there since 8190000, and the old trainer only kept rows exclusive
while a leg was open, so settled history may hold overlapping merges that
`"v2"` would re-judge. The non-finite checks live in
`cascade/trainer/expert_lane.py` (`check_finite_rows`), never in
`moe_model.py`, whose bytes feed `base_arch_digest`.

### Selection padding (trainer policy)

`select_routed_experts` wraps `select_domain_experts` and bans zero-affinity
experts, so a layer takes fewer than `min_experts` rows rather than padding
with experts the slice never routed to; a leg with no free routed row
waits. `select_domain_experts` itself (digest-pinned, replayed by the direct
lane's `verify_selection`) is unchanged.

### Miner-funded legs (DEC-CA-0059 item 8)

`[lane_b] leg_funding = "miner"` (default, both networks): legs rent on the
miner's key through `/v1/lane-b/fund` and the funded challenger path, never
an operator lane. `"operator"` keeps the retired lane-pool dispatch for dev /
emergency only. See DEC-CA-0059 "Shipped — miner funding".

## Gain-tiered full-pay period 2026-10-20 (owner)

Owner: "great add this" — the lane-B analogue of DEC-CA-0055's margin-tiered
start share. CONSENSUS; replaces `[lane_b] full_rounds` in place (nothing on
this branch is deployed — no compatibility path; a config still typing
`full_rounds` is refused at load). The decay (`half_life_rounds`, `floor`),
`win_ttl_eras` and the slot takeover rule are unchanged; the decay starts after
the tiered full period.

* **Key.** `[lane_b] full_pay_tiers = [[lcb_threshold, full_eras], ...]`,
  default `[[0.05, 10], [0.02, 5], [0.01, 2]]` — UNCALIBRATED placeholders:
  lane-B domain LCBs are noisy (few windows / sources per domain); set the
  thresholds from real win LCBs before arming. Load checks
  (`cascade.shared.config.v2.parse_full_pay_tiers`): non-empty; thresholds
  strictly descending in (0, 1); `full_eras` integers `>= 0`, non-increasing as
  the threshold falls, `<= win_ttl_eras`. Overlay-settable (DEC-CA-0057 allows
  every `[lane_b]` key, as it did `full_rounds`).
* **Rule** (`cascade.eval.expert_lane.full_pay`). A win pays in full for the
  `full_eras` of the HIGHEST threshold its target gain LCB (`target_lcb`, the
  LCB that cleared `gain_margin`) meets; below the lowest threshold (cannot
  happen while the lowest is `<= gain_margin`) it gets the lowest tier and is
  flagged `below_floor`. A multi-target leg (DEC-CA-0059) tiers each target's
  win by that target's own LCB.
* **Fixed at the win.** `apply_judgements` records `tier` / `full_eras` /
  `below_floor` on the `DomainWin` (the gain LCB is its `target_lcb`) under the
  tiers in force at the merging settlement; it is never re-evaluated, so a
  later tier change touches only later wins. Payouts read only the recorded
  `full_eras` (`reward_multiplier(age, full_eras=…)`); a win without one is
  refused.
* **Receipts.** Each `lane_b_pay` win carries `gain_lcb`, `tier`, `full_eras`,
  `below_floor`; the DEC-CA-0058 `state_post` slot entry carries `[…, gain_lcb,
  tier, full_eras]`, so the receipt chain binds a recorded tier.
* **Audit.** `check_lane_b`: a win merged at this settlement has `gain_lcb ==`
  the winning row's `target_lcb` and its tier re-derived from it.
  `_lane_b_paid_wins` (payout replay, used by `check_weights` and
  `derive_weights`, so satellites follow): every paid win carries the fields,
  `full_eras` in `[0, win_ttl_eras]`, a win made at this settlement
  re-derived under this receipt's tiers, an older one equal to the receipt's
  `state_post` slot (the satellite continuity check binds that across
  receipts); the multiplier is replayed from the recorded `full_eras`.
* **Display.** `v2_status` (`payout.full_pay_tiers`; the pay record verbatim),
  the miner dashboard (`full N eras (tier t)`), the W&B summary
  (`lane_b_payouts.<domain>.tier/full_eras`).
