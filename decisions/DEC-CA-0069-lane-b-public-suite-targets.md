---
id: DEC-CA-0069
type: decision
title: "Lane B: improve one, regress none — a leg may target a public benchmark suite (GIFT-Eval, BOOM, TIME, FEV) as well as a pool domain or horizon; it wins only by significantly improving that suite while regressing nowhere, including the private cascade pool, a fixed anchor king, and the suite's own other metric"
status: proposed
date: 2026-11-21
tags: [consensus, lane-b, eval, benchmarks, v2]
revisit_when: "after the first testnet era with suite targets armed — record suite-target win rate, the per-item gains behind each win, private-pool deltas on suite-target merges, anchor-guard margins over successive merges, and any generator flagged for benchmark contamination"
relations: {depends_on: [DEC-CA-0050, DEC-CA-0059], relates_to: [DEC-CA-0061, DEC-CA-0063]}
---

## Decision (proposed — owner 2026-11-21)

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
