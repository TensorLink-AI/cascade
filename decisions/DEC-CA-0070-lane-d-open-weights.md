---
id: DEC-CA-0070
type: decision
title: "Lane D: open weights on our architecture. Miners train anywhere and submit BF16 weights for the cascade_pyramid model; we only evaluate on forward-only private windows (data time AND first-seen after commit), with a confirmation duel, an unpaid house entry, a rotating public gate and per-lane coldkey caps"
status: proposed
date: 2026-11-21
tags: [consensus, lane-d, emissions, eval, v2]
revisit_when: "S1 measurements (B1 null FPR, B2 forward-eligible counts) land; the owner settles the [DECIDE]s below; after the S6a shadow"
relations: {depends_on: [DEC-CA-0050, DEC-CA-0055, DEC-CA-0057, DEC-CA-0058], relates_to: [DEC-CA-0063, DEC-CA-0064, DEC-CA-0067], supersedes_on_acceptance: [DEC-CA-0062]}
---

## Decision (proposed)

Lane D is specified in `docs/plans/LANE_D_PLAN.md` (rev3). In summary:

- **Entries.** A dedicated hotkey commits a BF16 `cascade_pyramid`
  checkpoint. The model code is unchanged and byte-identical, the config is
  the contract's 4096, and context is a scorer-side declaration: 4k, 8k or
  16k at launch, with 32k–128k at stage S8 by overlay. The checkpoint goes
  to a private vault, with ownership, dedup and per-lane coldkey caps.
- **Judging.**
  - Forward-only: every target point's data time **and** first-seen time
    are after wall(max(T_entry, T_king)) + `forward_margin`, within a
    per-feed `max_mirror_lag_hours`.
  - Frozen sets; a paired cluster bootstrap against KING_D with the cohort
    max-T; a **mandatory confirmation duel** on the next disjoint set.
  - `d_margin` is calibrated to a per-era false-crown target (k = 32
    attempts per era at the caps).
  - Gates: a rotating public suite drawn from the boundary block hash, an
    unselected suite, and an always-run low-frequency private-pool check.
- **House and pay.** Our lane A king is always entered as the house. It's
  never paid and never in the court. Lane D's share, court and the
  destination of the unpaid slot follow D1, D7, D9 and D21 below.
- **Trust.**
  - The main validator scores and signs; satellites verify structure and
    re-run the bootstrap from the signed rows.
  - Each verdict carries `rows_sha256`; the coldkey is recorded at `ready`
    and carried in the report.
  - The lane D state is bound into receipts.
- **Lane B (in scope).** The §12 expert-delta variant is built with lane D (WB1–WB7) and armed at S6c by overlay after lane D pays.
- **Rollout.** Gated stages S0–S8 (no dates). Activation and payment go by
  signed overlay (DEC-CA-0057): the S6a shadow at share 0, then S6b. Kill
  switch by overlay.

When accepted, this supersedes DEC-CA-0062's recipe design (the trainer-run
distillation / real-data recipe lane). **DEC-CA-0062 is not marked superseded
until the owner accepts this record.**

Numbering (rev5): main has DEC-CA-0069 (MarketCast recipe code hooks), so
this is DEC-CA-0070 and the lane B suite-targets record (PR #482) becomes
DEC-CA-0071. PR #479 also claims 0069 and needs its own new number.

## Open decisions ([DECIDE], with defaults)

| # | Decision | Default / recommendation |
|---|---|---|
| D1 | Share and source | 20% from lane A, burn remainder; with B and C armed: A 30 / B 25 / C 25 / D 20 |
| D2 | Context variants | **(rev3)** Launch arms {4k, 8k, 16k}. {32k, 64k, 128k} arm at **S8** by overlay (`contexts`), each only after T7 and W1 `min_long_series` (owner goal 128k kept) |
| D2b | Gate | GIFT/BOOM/FEV drawn equally; TIME unselected; `tau_public` 0.01; anchors = KING_D and house |
| D3 | Rate limits | 1 open per hotkey; 2 open per coldkey; 8 judged per settlement |
| D4 | Eval fee | None at launch |
| D5 | Margins and floors | `d_margin` 0.01 (level units, re-checked by T14); `forward_margin` 6 h; 1000 windows / 20 sources |
| D6 | Bench capacity | Queue behind lane A; second pod if crownings slip > 1 settlement |
| D7 | Court | 2 prior miner kings, decay 0.5; house never in the court |
| D8 | Losers | Private; vault TTL 4 eras; owner-only archive 14 days; then delete |
| D9 | House paid? | No: the slot burns; prior miner tails keep paying |
| D10 | Activation | Two overlays: S6a `from_block` with `share = 0` (shadow ≥ 2 eras, observed `cfg-` stake ≥ 2/3), then S6b `share = 0.20`; kill switch by a staged revert overlay (latency up to about one era); **(rev3)** the S8 long-context arming is also an overlay (`contexts`) |
| D11 | Eval host | **(rev3)** Launch: valpriv-4090 or another pinned-torch 2.4.1+cu124 pod (≥ 24 GB; the 16k fit is checked in B2). **Not** Blackwell sm_120. valeval-laned (H100/A100 80 GB class, pinned torch) only at S8 |
| D12 | Pool packaging | `pool` untouched (keep length never raised); **(rev3)** launch `pool_d` = `pool`'s series ids joined, history to 16384 + 64, A1 timestamps, `pool_d_hist` (no targets) to the eval pod, `pool_d_tgt` on the box; 32k–128k history work at S8; dedicated builder |
| D13 | Weights license | Open licenses only (Apache-2.0 / OpenMDW-1.1); kings are published |
| D14 | dtype | **(rev3)** BF16 only (~2.5 GB), scored in lane A's eval precision (fp32 upcast); F32 refused (4.9 GB is near the 5 GiB single-PUT limit) |
| D15 | Arch generations | Lane D generation = lane A generation; a change opens a new board |
| D16 | Standing approval | Automated vault TTL deletion and archive expiry (otherwise Approve-tier per deletion) |
| D17 | Trust | Accept that lane D loser rows are signed-but-not-reproducible (satellites verify structure and bootstrap; kings become re-scorable after publication) |
| D18 | Data path | **Targets stay on the box** (histories-only to the pod, D12), keeping the private-offload rule. Alternative: let the pod hold targets, which is a new rule; **(rev3)** launch `pool_d_hist` carries ≤ 16384 history per series |
| D19 | Shadow test entry | At least one owner-run test entry on a dedicated hotkey during S6a (recommended), so the shadow exercises a real duel, not only plumbing |
| D20 | **(rev3)** King change between first clear and confirmation | (a) restart the duel against the new king (**recommended**); (b) confirm against the king at the first clear |
| D21 | **(rev3)** Unpaid lane D king slot while the house reigns | Option 1 burn (v2 default, kept) vs option 2 return to lane A. Tokenomics call, not adopted (§1.4 table) |
| D22 | **(rev3)** Coldkey resolution | (b) record at `/v1/lane-d/ready`, re-check at reveal, carry in the signed report (**recommended**); (a) archive-state resolution at T |
| D23 | **(rev3)** Loser re-scoring | (ii) targets, window ids, `rows_sha256` and bootstrap verifiable; histories private (**default, as v2**); (i) publish `pool_d_hist` after the dispute window |
| D24 | **(rev3)** Per-era false-crown target (sets `d_margin` via B1) | 5% recommended (OA3); `d_margin` stays 0.01 until B1 runs |
| **D25** | **Court pays rent indefinitely; salami-slicing.** The court is rank-decayed (57/29/14% of D's share at `king_decay` 0.5, 2 priors) with no TTL, and the bar is a fixed 0.01. A 3% gain released as three 1% crowns fills the whole court | Simple; matches lane A's court shape | Pay D wins on lanes B and C's `full_pay_tiers` + `half_life_rounds` + `win_ttl_eras` shape, keyed to the crowning LCB over max(KING_D, house). Or a tenure-decaying bar (e.g. 0.03 → 0.01). Or time-bound prior-king tails, like lane A's `court_bp` / `court_blocks` |
| **D26** | **Lane A is a free input to D, and D's 20% comes out of A.** The cheapest winning D entry is "fine-tune the published house". Generators and recipe are published (`[v2] king_publish = "recipe"`) | Lane A is paid by its own ladder; D pays only for beating the house | A **lineage royalty**: a fixed ρ ≈ 15–20% of each D win's pay goes to the lane A king hotkey that was the house at the winner's T_E. Calibrate ρ in shadow with a weight-space nearest-ancestor measurement. Never self-declared lineage |
| **D27** | **Withholding from A.** A strong generator may earn more used privately in D than published in A | Accept; monitor how many D kings are far from every published A checkpoint | The D26 royalty, plus keep A's share ≥ D's |
| **D28** | **Split.** v2 has A 30 / B 25 / C 25 / D 20 when all are armed | v2 split | Reviewer: **A 40 / D 25 / B 20 / C 15**, with D ramping 15 → 25% after ≥ 3 miner crowns pass the gate with the unselected suite not falling. Now (B, C off): A80/D20 is acceptable. Interacts with **D21**: the reviewer prefers returning the unpaid house slot to lane A over burning it |
| **D29** | **Frequency bias in frozen sets.** Sub-hourly data reaches the floors first, so miners specialise on sub-hourly | v2 freeze rule | Per-frequency-bucket and per-domain window quotas inside a frozen set, equal cluster weights. Freeze when every bucket meets its quota or a max wait passes |
| **D30** | **TIME isn't truly unselected.** It's in lane A's `require_public` gate and the king check, and DEC-CA-0071 (pending: PR #482) adds `suite:time` as a lane B target, so the house is optimised on it | Keep TIME as the unselected suite | Drop `s-time` from DEC-CA-0071, or choose a sentinel that no lane gates or pays on |
| **D31** | **Sybil queue griefing.** Cheap coldkeys can fill the 8 judged slots with junk | The A8 caps + registration cost | Round-robin by coldkey; priority by hotkey age or prior non-junk LCB; a cheap pre-screen (small window subset; drop LCB < −x before a full frozen set) |
| **D32** | **Lane B vs D.** Once D runs it dominates lane B as designed (one D king slot ~11–20% of emission vs ~1.8% per B slot) | Keep B as trainer-run legs | Re-scope B to the §12 expert-delta variant when D arms, with row-level dedup against published D kings' rows |

| D33 | **(rev3)** Cross-lane rotating sentinel (§16) | **Adopt**: one suite per era, drawn from the era-boundary block hash, held out of every gate and target in lanes A, B and D, swept and published for every new king. Alternative: no sentinel (v2: lane A gates all four; lane D holds TIME out) |
| D34 | **(rev3)** Sentinel ratchet vs the house | **Optional, recommended on**: every lane's candidates must not fall below the current lane A king on the sentinel by more than `sentinel_tau` (0.01), re-scored paired on one host. Off = the sentinel is publish-only, and a regression on it alone is never blocked |
| D35 | **(rev3)** Sentinel vs DEC-CA-0071 (pending: PR #482) lane B suite targets | While a suite is the sentinel it is **not** a lane B target or guard that era. Entries already declared on it are judged on the remaining suites, or deferred to the next era. Recommended: **defer** (the target stays honest) |
| D36 | **(rev3)** Lane B at S6c: which variant arms | Expert-delta variant (§12) arms at S6c. Whether trainer-run legs **also** stay on is **D32** (recommended: delta variant only once D runs) |
| D37 | **(rev3)** Lane B share at arm | The deployed `[lane_b] share = 0.25` (A/B 75/25 while C is off) vs a smaller share given D32's dominance finding; set in the S6c overlay |

D33–D35 add a cross-lane rotating held-out sentinel (plan §16) that also changes lane A's gate and lane B's targets; they're recorded here because lane D depends on them, but lane A/B adoption may warrant its own DEC.

Rows D25–D32 are tokenomics or owner calls from the rev3 incentive review
(plan §15). Both sides are recorded there, and none is adopted.

## Not decided here

- The loop-readiness build pass runs after these are settled.
- The S1 numbers that size `d_margin` and the floors are in the plan's
  Appendix B (`docs/plans/LANE_D_PLAN.md`). Runs blocked on owner inputs are
  listed in the plan's §17.4 and on the status board (`docs/plans/STATUS.md`
  on branch `laned/build-state`).
