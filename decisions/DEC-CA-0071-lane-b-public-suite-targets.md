---
id: DEC-CA-0071
type: decision
title: "Lane B: improve one, regress none — a leg may target a public benchmark suite (GIFT-Eval, BOOM, TIME, FEV) as well as a pool domain or horizon; it wins only by significantly improving that suite while regressing nowhere, including the private cascade pool, a fixed anchor king, and the suite's own other metric"
status: accepted
date: 2026-11-21
tags: [consensus, lane-b, eval, benchmarks, v2]
revisit_when: "after the first testnet era with suite targets armed — record suite-target win rate, the per-item gains behind each win, private-pool deltas on suite-target merges, anchor-guard margins over successive merges, and any generator flagged for benchmark contamination"
relations: {depends_on: [DEC-CA-0050, DEC-CA-0059], relates_to: [DEC-CA-0061, DEC-CA-0063, DEC-CA-0070]}
---

## Decision (accepted — owner 2026-11-21)

Owner: "what if they only need to improve 1 benchmark and not regress on any
incl cascade eval" → "lets do for lane B".

Why: the subnet's goal is public state-of-the-art results. The current king
(5G6j, 2026-11-20 bench record) scores GIFT-Eval CRPS 0.511 / MASE 0.749. The
best published GIFT-Eval numbers (late 2026) are about 0.446 CRPS and about
0.646 MASE, so we trail by roughly 13% on both. That gap is not yet checked
against the official scorer. Lane B today only pays private-pool domain gains,
and the public suites act only as guards. This decision lets a leg aim
directly at a public suite, but only as a Pareto improvement: better on one
suite and no worse anywhere else.

Lane A is unchanged. The throne stays decided on the private pool (DEC-CA-0061
and DEC-CA-0063), where public-benchmark gaming pressure cannot crown a king.

### 1. Suite targets

- **Targets.** `[lane_b]` gains suite targets `suite:gift`, `suite:boom`,
  `suite:time` and `suite:fev`, next to the 7 pool domains and 3 horizons
  (DEC-CA-0059).
- **Arming.** They are armed by `[lane_b] suite_targets = true`, which
  defaults to false. While false the field drops from the digest, and every
  existing commitment, verdict, receipt and weight vector stays byte for byte
  the same.
- **Codes.** Target codes are `s-gift`, `s-boom`, `s-time` and `s-fev`. They
  fit the 32-character target field, and a domain named `s-*` is refused at
  load.
- **Declared before training.** The target is declared at commit, as in
  DEC-CA-0059 item 1, and nothing can be added after evaluation. This settles
  the "pick any one" multiplicity worry. The miner commits to one hypothesis
  in advance, so a leg is not judged on its best of five suites. A leg
  declaring k targets keeps the `alpha / k` rule.
- **Whole suite only.** A suite target is judged on the WHOLE suite, never a
  subset of items. Per-domain GIFT targets (for example `gift:energy`) are an
  Open item.

### 2. Gain test (superiority)

A suite target wins when, against the current king on that suite's items
(GIFT has 97 configs), all three of these hold:

- **CRPS improves significantly.** The per-item paired bootstrap of the
  geomean ratio (`paired_geomean_lcb`) gives `LCB_α ≥ gain_margin_public`,
  default 0.01.
- **MASE does not regress.** On the same suite, `LCB_α(MASE) ≥ −tau_public`.
  A leg cannot buy probabilistic accuracy with point accuracy.
- **There is enough evidence.** The point gain is strictly positive and the
  suite has at least `min_public_items` items. Otherwise the result is
  `inconclusive`, never a merge.

### 3. Guards (non-inferiority, all must hold, each at α)

These are the existing guards plus two new ones:

1. **Every other public suite** (already required: `require_public`). CRPS
   and MASE each need `LCB ≥ −tau_public`.
2. **Every private pool domain** (already required). Each needs
   `LCB ≥ −tau_domain`, with small domains pooled into `_rest`.
3. **Every scored horizon** (already required, DEC-CA-0059).
4. **NEW: the private pool overall must not get worse in point terms.** For a
   suite-target leg, the whole-pool paired gain point estimate must be `≥ 0`,
   on top of the per-domain non-inferiority guards. This is the
   contamination tripwire. A gain that exists only on a public test set,
   with nothing behind it on private data, is the 5GMz pattern (2026-11-16:
   overfit to our public eval reveal, then caught by BOOM 1.157). Here it is
   refused before merge.
5. **NEW: the anchor guard, against cumulative drift.** Each merge also has to
   pass guards 1–3 against a FIXED anchor king, not only the current king.
   The tolerances are `tau_anchor_public` and `tau_anchor_domain`, default
   0.0: no worse than the anchor at α.
   - **Why.** Non-inferiority at `tau = 0.01` lets through losses below the
     noise floor, and many merges can stack them into real decay. Anchoring
     bounds the total.
   - **Which anchor.** The anchor is the lineage init recorded at the era
     boundary where lane B opened. It moves forward only on a lane-A
     warm-start promotion (a validator-passed king check), so it never moves
     because of a lane-B merge.

Every guard is an intersection-union test, so there is no multiplicity
correction across guards (DEC-CA-0050).

### 4. Pay

- **Slots.** Suite targets are slots like domains and horizons, so there are
  7 + 3 + 4 = 14 slots when everything is armed. Decay, TTL, burn and
  merge-once are unchanged (DEC-CA-0050, DEC-CA-0059).
- **Slot weights are owner-tunable** (`[lane_b] target_weights`, defaulting
  to equal shares and normalised). The intent is to weight slots by the gap
  to the published leader, so emissions follow where the score moves most.
  The weights live in config and are consensus terms. They are never computed
  live from a leaderboard, because validators cannot agree on a scraped web
  page.

### 5. Capacity

No new bench capacity is needed. Every lane-B leg already needs paired
public numbers for its guards (`require_public` = all four suites), so a
suite-target leg costs the same sweep as a domain leg. The anchor guard reuses
the anchor's cached bench rows, swept once per anchor.

## Open

These were the open items of PR #482. Amendment rev4 below resolves the
miner-declared experts (item 1, D40), the static contamination check
(item 2, D38), the opening gate (item 3, D41) and the 256-expert clamp
(item 1, D39).

- **Miner-declared experts.** Today expert rows are picked by the trainer with
  ESFT on the king's routing, with the router and bias frozen. A suite target
  has no single affinity slice to select from. The candidates are:
  - (a) ESFT over the suite's own context inputs. These are public benchmark
    inputs, read for routing counts only, with no targets and no training on
    them.
  - (b) The miner declares the expert rows, and only those rows' router
    entries and bias unfreeze (the FlexOlmo pattern, DEC-CA-0050 "Research
    basis"). The merge byte-check proves nothing else moved.

  (b) also lets a leg revive experts that get no traffic, which (a) cannot
  do. Measure with PR #481's training-stream load telemetry first. Owner
  decision.
- **Contamination, static side.** Add a generator check against embedded
  benchmark test series: hashes of GIFT, BOOM, TIME and FEV test windows,
  matched against literals and data files in the submission. This adds to
  guard 4 and does not replace it.
- **Opening gate.** The routing-saturation co-gate (top-k overlap ≥ 0.8,
  Jaccard ≥ 0.7 on every layer of every domain) may never open, because the
  king re-learns its routing every era. Consider replacing it with
  `open_tokens` plus a king-check pass. That is a separate decision.
- **Initial `target_weights`.** These need the per-item gap of the king's
  latest signed bench report against the published leaders, from the
  official scorer rather than our sidecar.
- **Built for 256 experts.** The ESFT clamp `min_experts = 8` /
  `max_experts = 38` was set for 256 experts. The live 120m model has 512,
  so re-derive the clamp.

## Shipped

Nothing. This is design only, and no code or config has changed. The
implementation path is `judge_expert_targets`
(`cascade/eval/expert_lane.py`), which gains the suite branch and the anchor
guard. The audit check `check_lane_b` re-derives both. All of it goes through
testnet first and then release-then-activate on mainnet (six external
validators).

## Amendment rev4 (accepted, 2026-11-21)

Status of this amendment: **ACCEPTED** by the owner on 2026-11-21 (plan §6
owner answers and the §17.4 register). This file carries PR #482's text
above, now accepted, with this amendment on top. The full design is in
`docs/plans/LANE_D_PLAN.md` §12b; this section records the decisions, and
the §17.4 register is the authority if the two ever differ.

**Owner direction (2026-11-21).** Lane B stays **trainer-run**: miners
submit a code-only generator and our trainer trains. Miners may also
(1) **choose which expert rows** are trained, and (2) **target one public
suite**, improving it significantly while regressing nowhere.

1. **Open option (b) is adopted: miner-chosen experts.**
   - The commit declares `declared_rows = {layer: [indices]}`, validated
     against `experts_per_layer = (128, 256, 256, 512, 512, 512, 512)`
     (`cascade/trainer/pyramid_model.py:42`). Layer keys are MoE
     ordinals 0–6, not block indices (the MoE blocks are `(1, 2, 3, 5, 6, 7,
     8)`; ordinal i has `experts_per_layer[i]` experts). The clamp minimum
     applies only to layers with at least one declared row; zero rows on a
     layer is allowed, and at least one row in total is required. Typed
     refusals: `declared_rows_out_of_range`, `declared_rows_bad_layer`,
     `declared_rows_count`.
   - The per-layer clamp becomes fractional (D39, decided): `min_experts_frac =
     0.03125`, `max_experts_frac = 0.1484375`. That gives 128 → 4..19,
     256 → 8..38 and 512 → 16..76, replacing the 256-era `min_experts = 8` /
     `max_experts = 38`.
   - Claims follow the existing merge-acceptance deferral. The earliest
     reveal wins; a deferred claim shows `deferred: rows_held`.
   - ESFT stays the default when nothing is declared, and both modes
     coexist.
   - **Router rows (D40, decided): (a) frozen**, today's compact-rows
     backend (`cascade/trainer/expert_training.py:112-122`), for first
     arming. Option (b), the FlexOlmo unfreeze of the declared rows' router
     entries and bias with a `traffic_steal_bound` guard, is not adopted.
2. **The static contamination check is a precondition, not an open item.**
   - `suite_targets` arms only after the check ships **and passes on
     testnet**.
   - At intake, before compute, exact and normalised (scale-, offset- and
     rounding-robust) hashes of every GIFT-Eval, BOOM, TIME and FEV test
     window are matched against `.py` numeric literals, `config.json` and
     any non-code bytes. A match is refused as `benchmark_series_embedded`.
   - The hash set is versioned and digest-pinned
     (`[lane_b] contamination_hashset`), refreshed only when a suite
     identity changes.
   - Guard 4 (whole-pool point gain ≥ 0) and the anchor guard stay the
     backstop for procedurally recreated series.
   - **D38, decided: (a)** `suite_target_max_repo_mb = 8` for lane B
     suite-target entries; `[generator] max_repo_mb = 128` (`chain.toml:47`)
     is unchanged for everyone else. DEC-CA-0028 already plans to stop
     `config.json` carrying bulk data.
3. **The opening gate is replaced (D41, decided: (a)).**
   - The routing-saturation co-gate (`routing_overlap` 0.8, `routing_jaccard`
     0.7) **has never been measured on mainnet**. The driver state shows
     `gate_opened_block = 0`, an empty `routing_history` and
     `lineage_tokens = 0`, because lane B is off.
   - The only measurement, the 2026-11-20 probe, found king-vs-init top-8
     overlap of about 1.1/8.
   - Chosen: (a) `open_tokens` + a passing lane A king check. Not adopted:
     (b) `open_tokens` alone; (c) keep the co-gate with re-measured
     thresholds.
   - The new key is `[lane_b] open_rule`, overlay-settable after the
     capability release.
4. **Sentinel (D35, decided).**
   - With the cross-lane rotating sentinel (plan §16), a sentinel suite is
     not a lane B target or guard that era: a suite-target entry whose
     target is the current sentinel **defers** to the next era, typed
     `deferred: sentinel_target`, not charged, bounded by
     `sentinel_max_defer_eras`.
   - The alternative, exempting lane B's own target from the sentinel, is
     not adopted.
5. **Routing information for miners** (informational only, never consensus):
   per-king per-expert load, dead experts and per-suite context-only routing
   affinity, in a digest-pinned artefact.
6. **Expert-leg cost (D42, decided):** lane B has its own GPU list, so
   cheaper image-qualified GPUs are allowed, each measured first. The miner
   chooses the leg length, `train_hours`, from 0.5 to 6 h. Steps are fixed at
   admission and signed. The guard is capped by `[lane_b] max_leg_seconds`.
   `[training] max_train_seconds` is not changed. The rules (plan §12b.6):
   - **GPU list.** New `[lane_b] funded_pod_skus`, separate from `[round]`,
     so adding a cheap GPU for lane B never opens it to lane A legs. A GPU
     type enters the list only after the pinned image (torch 2.4.1+cu124) is
     qualified on it, its expert-only step rate and leg overhead are
     measured, and the leg fits memory without changing micro-batching. The
     type is recorded on the leg (`gpu_name`), and an audit re-derivation
     uses the same type.
   - **Leg length.** The miner sets `train_hours` at funding:
     `0.5 ≤ train_hours ≤ [lane_b] max_leg_hours`, with `max_leg_hours` = 6.0,
     in 0.5 h steps.
   - **D42b, decided: (a) real hours on the chosen GPU.** At admission,
     `steps = floor(train_hours × 3600 × measured_steps_per_s[gpu])`; the
     step count is fixed at admission and recorded in the signed leg record.
     Option (b), reference hours, is not adopted.
   - **Guard.** `train_hours × [lane_b] guard_factor` (1.25) plus the
     measured overhead, capped at `[lane_b] max_leg_seconds` (6 h × 1.25 =
     27,000 s), a new lane B cap. `[training] max_train_seconds` is not
     changed: it is a contract-digest field and is never edited for this.
     The lane B leg contract is derived with the lane B cap instead.
   - **Consensus.** `max_leg_hours`, `max_leg_seconds`, `guard_factor` and
     the `train_hours` range are consensus. The per-GPU step-rate table is
     trainer-side operational config: steps are fixed at admission and
     carried signed, so satellites never need the table.
   - Operator funding is not adopted.

New `[lane_b]` keys:

- `suite_targets`
- `contamination_hashset`
- `contamination_min_hits`
- `suite_target_max_repo_mb`
- `min_experts_frac`
- `max_experts_frac`
- `router_rows`
- `traffic_steal_bound`
- `open_rule`
- `sentinel_target_rule`
- `sentinel_max_defer_eras`

Each key: drop-when-default in the digest, overlay-settable via
`ALLOWED["lane_b"] = None` (`remainder` stays release-only, and so does
`contamination_hashset`, added to the release-only tuple), loader parse,
round-trip and mutation tests (plan WB10).

### Decision register

Copied from plan §17.4 (rows D35–D42 and D42b).

| D | Status | Option chosen | Class | Decided |
|---|---|---|---|---|
| D35 | decided | A sentinel suite is not a lane B target or guard that era | S | owner (default), 2026-11-21 |
| D36 | decided | Lane B arms trainer-run | S | rev4 |
| D37 | decided | `[lane_b] from_block` only once D41's gate is met; full share at that block (D45) | O | owner (default), 2026-11-21 |
| D38 | decided | (a) 8 MB `suite_target_max_repo_mb` | O | owner (default), 2026-11-21 |
| D39 | decided | Fractions 0.03125 / 0.1484375 | O | owner (default), 2026-11-21 |
| D40 | decided | (a) router rows frozen | S | owner (default), 2026-11-21 |
| D41 | decided | (a) `open_tokens` + a passing lane A king check | S | owner (default), 2026-11-21 |
| D42 | decided | Cheaper GPUs via `[lane_b] funded_pod_skus`; miner-chosen leg ≤ 6 h | S | owner, 2026-11-21 |
| D42b | decided | (a) real hours on the chosen GPU | S | owner, 2026-11-21 |
