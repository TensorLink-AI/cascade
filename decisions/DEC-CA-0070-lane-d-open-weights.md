---
id: DEC-CA-0070
type: decision
title: "Lane D: open weights on our architecture. Miners train anywhere and submit BF16 weights for the cascade_pyramid model; we only evaluate on forward-only private windows (data time AND first-seen after commit), with a confirmation duel, an unpaid house entry, a rotating public gate, a cross-lane sentinel and per-lane coldkey caps"
status: accepted
date: 2026-11-21
tags: [consensus, lane-d, lane-b, emissions, eval, v2]
revisit_when: "S1 measurements (B1 null FPR, B2 forward-eligible counts) move d_margin or the floors off their defaults; the owner revisits any of the five rows recorded as Option A under Proposed change 22 (D25-D27, D29, D31); or the testnet run (netuid 259) shows a lane D or lane B crowning the duel or the gate should have refused"
relations: {supersedes: [DEC-CA-0062], depends_on: [DEC-CA-0050, DEC-CA-0055, DEC-CA-0057, DEC-CA-0058], relates_to: [DEC-CA-0063, DEC-CA-0064, DEC-CA-0067, DEC-CA-0071]}
---

## Decision

Accepted by the owner on 2026-11-21 (plan §6 owner answers, "all other
recs"). The design is specified in `docs/plans/LANE_D_PLAN.md` (rev6); this
record fixes the decisions. Detail lives in the plan, and the §17.4 register
is the authority if the two ever differ.

- **Entries.** A dedicated hotkey commits a BF16 `cascade_pyramid`
  checkpoint (D14). The model code is unchanged and byte-identical, the
  config is the contract's 4096, and context is a scorer-side declaration:
  {4k, 8k, 16k} at launch, {32k, 64k, 128k} at S8 by overlay (D2). The
  checkpoint goes to a private vault, with ownership, dedup and per-lane
  coldkey caps (D3). Open licences only; kings are published (D13). Lane D
  generation = lane A generation (D15).
- **Judging.**
  - Forward-only: every target point's data time **and** first-seen time
    are after wall(max(T_entry, T_king)) + `forward_margin` (6 h, D5),
    within a per-feed `max_mirror_lag_hours`. Targets stay on the box (D18).
  - Frozen sets; a paired cluster bootstrap against KING_D; a mandatory
    confirmation duel on the next disjoint set, restarted against a new king
    if the king changes in between (D20).
  - `d_margin` 0.01, sized to a 5% per-era false-crown target (D24, D5).
  - Gates: one rotating public suite per crowning from the suites not held
    out (D2b), plus the cross-lane rotating sentinel of plan §16 (D33),
    ratcheted against the lane A king (D34) and never a lane B target or
    guard in its era (D35).
- **House, KING_D and pay.** Our lane A king is always entered as the house,
  scored as a BF16 cast and dated by its lineage (D46, D47). An entry must
  beat the house or, once the house has been beaten, the current lane D king
  (KING_D). The house is never paid and never in the court; its slot burns
  (D9, D21). Court: 2 prior miner kings, decay 0.5 (D7). Shares are fixed at
  A 25 / B 25 / C 25 / D 25 of the total, lane D taking from lane A's 25%
  (D1, D28, D48).
- **Trust.** The main validator scores and signs; satellites verify
  structure and re-run the bootstrap from the signed rows (D17, D23). The
  coldkey is recorded at `ready` and re-checked at reveal (D22).
- **Lane B (in scope).** Trainer-run with miner-chosen experts and suite
  targets (plan §12b; D32, D36). It arms at `[lane_b] from_block` only once
  D41's gate is met, at full share (D37); cheaper GPUs via
  `[lane_b] funded_pod_skus` with a miner-chosen leg of at most 6 h, billed
  on real hours (D42, D42b). Router rows frozen (D40).
- **Rollout.** Gated stages S0-S8 (no dates). No shadow modes: every feature
  goes live at one announced activation block, on every node, at full
  effect, by one signed activation overlay (D10, D45; DEC-CA-0057). Reveal
  is immediate (D43). Kill switch by overlay.

This supersedes DEC-CA-0062's recipe design (the trainer-run distillation /
real-data recipe lane). A lane D miner who wants distillation calls
Ephemeris, or anything else, on their own account.

Numbering: main has DEC-CA-0069 (MarketCast recipe code hooks), so this is
DEC-CA-0070 and the lane B suite-targets record is DEC-CA-0071.

## Decision register

Copied from plan §17.4. D44 is withdrawn (lanes A and B keep cooldown). The
five §15 rows D25-D27, D29 and D31 had no recommended option and are
recorded as Option A from the owner's "all other recs" answer;
Proposed change 22 asks the owner to confirm them.

| D | Status | Option chosen | Class | Decided |
|---|---|---|---|---|
| D1 | decided | `[lane_d] share = 0.25` from the total, `takes_from = "lane_a"`; A 25 / B 25 / C 25 / D 25 | O | owner, 2026-11-21 |
| D2 | decided | Launch {4k, 8k, 16k}; {32k, 64k, 128k} at S8 by overlay | O | owner (default), 2026-11-21 |
| D2b | decided | One gate suite per crowning from the suites not held out; the sentinel is held out | S | owner (default), 2026-11-21 |
| D3 | decided | 1 open per hotkey, 2 per coldkey, 8 judged per settlement | O | owner (default), 2026-11-21 |
| D4 | decided | No eval fee at launch | S | owner (default), 2026-11-21 |
| D5 | decided | `d_margin` 0.01 (re-checked by T14), `forward_margin` 6 h, 1000 windows / 20 sources | O | owner (default), 2026-11-21 |
| D6 | decided | Queue behind lane A; second pod if crownings slip > 1 settlement | O (ops) | owner (default), 2026-11-21 |
| D7 | decided | 2 prior miner kings, decay 0.5; house never in the court | O | owner (default), 2026-11-21 |
| D8 | decided | Losers private; vault TTL 4 eras; archive 14 days; delete | O | owner (default), 2026-11-21 |
| D9 | decided | House unpaid: the slot burns | S | owner (default), 2026-11-21 |
| D10 | decided | One activation overlay (D45) | S | owner, 2026-11-21 |
| D11 | decided | Launch eval host: valpriv-4090 or a pinned-torch pod (≥ 24 GB) | O (ops) | owner (default), 2026-11-21 |
| D12 | decided | `pool` untouched; `pool_d` joined to `pool`'s ids, history to 16384 + 64 | S | owner (default), 2026-11-21 |
| D13 | decided | Open licences only; kings published | S | owner (default), 2026-11-21 |
| D14 | decided | BF16 only | S | owner (default), 2026-11-21 |
| D15 | decided | Lane D generation = lane A generation | S | owner (default), 2026-11-21 |
| D16 | decided | Standing approval for automated TTL deletion granted | O (ops) | owner, 2026-11-21 |
| D17 | decided | Loser rows signed-but-not-reproducible accepted | S | owner (default), 2026-11-21 |
| D18 | decided | Targets stay on the box | S | owner (default), 2026-11-21 |
| D19 | decided | Owner-run test entries on testnet in S3 | O (ops) | owner, 2026-11-21 |
| D20 | decided | (a) restart the duel against the new king | S | owner (default), 2026-11-21 |
| D21 | decided | Option 1: burn | S | owner (default), 2026-11-21 |
| D22 | decided | (b) record at `ready`, re-check at reveal | S | owner (default), 2026-11-21 |
| D23 | decided | (ii) targets, ids, `rows_sha256`, bootstrap verifiable | S | owner (default), 2026-11-21 |
| D24 | decided | 5% per-era false-crown target | O | owner, 2026-11-21 |
| D25 | decided | Option A (keep v2) **(see Proposed change 22)** | S | owner ("all other recs"), 2026-11-21 |
| D26 | decided | Option A (keep v2); share source is D48 **(Proposed change 22)** | S | owner ("all other recs"), 2026-11-21 |
| D27 | decided | Option A (accept) **(Proposed change 22)** | S | owner ("all other recs"), 2026-11-21 |
| D28 | decided | A 25 / B 25 / C 25 / D 25 | O | owner, 2026-11-21 |
| D29 | decided | Option A (keep v2) plus `max_source_share` (W5b) **(Proposed change 22)** | O | owner ("all other recs"), 2026-11-21 |
| D30 | decided | Handled by the §16 sentinel (D33) | S | rev3 (D33) |
| D31 | decided | Option A (the A8 caps) **(Proposed change 22)** | O | owner ("all other recs"), 2026-11-21 |
| D32 | decided | Lane B trainer-run (§12b.7) | S | rev4 |
| D33 | decided | Adopt the cross-lane rotating sentinel | S | owner (default), 2026-11-21 |
| D34 | decided | Sentinel ratchet on | S | owner (default), 2026-11-21 |
| D35 | decided | A sentinel suite is not a lane B target or guard that era | S | owner (default), 2026-11-21 |
| D36 | decided | Lane B arms trainer-run | S | rev4 |
| D37 | decided | `[lane_b] from_block` only once D41's gate is met; full share at that block (D45) | O | owner (default), 2026-11-21 |
| D38 | decided | (a) 8 MB `suite_target_max_repo_mb` | O | owner (default), 2026-11-21 |
| D39 | decided | Fractions 0.03125 / 0.1484375 | O | owner (default), 2026-11-21 |
| D40 | decided | (a) router rows frozen | S | owner (default), 2026-11-21 |
| D41 | decided | (a) `open_tokens` + a passing lane A king check | S | owner (default), 2026-11-21 |
| D42 | decided | Cheaper GPUs via `[lane_b] funded_pod_skus`; miner-chosen leg ≤ 6 h | S | owner, 2026-11-21 |
| D42b | decided | (a) real hours on the chosen GPU | S | owner, 2026-11-21 |
| D43 | decided | Immediate reveal (`min_blocks`) | S | owner, 2026-11-21 |
| D44 | withdrawn | — (lanes A and B keep cooldown) | — | owner, 2026-11-21 |
| D45 | decided | No shadow modes; one activation block | S | owner, 2026-11-21 |
| D46 | decided | (a) house dated by its lineage | S | owner, 2026-11-21 |
| D47 | decided | (a) score the house as a BF16 cast; dedup against BF16 casts | S | owner (default), 2026-11-21 |
| D48 | decided | Fixed 25% shares of the total; lane D takes from lane A's 25% | S | owner, 2026-11-21 |

## Not decided here

- The S1 numbers that size `d_margin` and the floors are in the plan's
  Appendix B. Owner inputs OA1-OA8 (run inputs, not design decisions) are
  tracked in plan §17.4 and on the status board (`docs/plans/STATUS.md` on
  branch `laned/build-state`).
- D33-D35 also change lane A's gate and lane B's targets; they're recorded
  here because lane D depends on them.
