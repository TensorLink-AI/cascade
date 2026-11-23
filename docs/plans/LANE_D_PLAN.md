# Plan: lane D "open weights" (anything goes on our architecture), with lane E and lane A flexibility — v3

Status: **PLAN v3, rev6.4 (proposed 2026-11-22; rev6.4 2026-11-23)**: v2's design, made buildable by an
autonomous loop. A work package is done **only when its Accept command exits
0**; **(rev6)** its level is only what `scripts/laned/status` computes (§17),
never a claim. Owner decisions are marked **[DECIDE]**. Nothing is armed. **No calendar
dates**: stages are gated (§5).

## rev6.4 (owner 2026-11-23; main @ 54e7f23)

Records two owner decisions, (a) and (b), and one owner finding, (c). Plan
text only; nothing else in the design changes. Every changed item is marked
**(rev6.4)**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **W8b accepted (owner 2026-11-23, DECIDED: "i accept w8b").** The rev6.2 (g) proposed WP W8b (validator production wiring of `settle_lane_d`) is accepted; its row is otherwise unchanged | Lane D core table, W8b row marker |
| (b) | **CI-first testing for loop agents (owner 2026-11-23, DECIDED).** An agent runs only its own WP's `tests/accept/<wp>/` dir locally; every other suite (unit shards, receipt-replay, CROSS) runs in CI, and all are required green before merge. Why: the production box (4 cores / 7 GB) also scores the validator, so local full suites stall scoring | §3.1 rule 10 |
| (c) | **Hippius has no Object Lock (owner finding 2026-11-23).** `GetObjectLockConfiguration` / `GetBucketVersioning` return 403 for a full-access bucket-scoped sub-token and for the main key; the console offers no lock permission. W3a re-runs its probe with `--no-object-lock` (lock probes recorded as `skipped`). Consequence: lane D kings are **not** locked by a COMPLIANCE copy at crowning; their immutability rests on the content-addressed path, the ingest re-hash on every read, the signed report's digest, and the R2/HF mirrors | §1.2 Storage (Object Lock bullet, "lock only kings" bullet); W3a row; rev5 B9 row |

## rev6.3 (owner 2026-11-22/23; main @ a244271)

Adds one proposed WP, (a), and one owner decision, (b). Plan text only;
nothing else in the design changes. Every changed item is marked **(rev6.3)**.

| Item | Decision | Landed in |
|---|---|---|
| (a) | **Proposed (owner 2026-11-22 approved adding it):** WB24 lane B miner CLI: `cascade lane-b fund` gains `--declared-rows <file>` (signed WB14 header + `rows=` tag, local refusal with `DECLARED_ROWS_CODES`), `--target suite:<s>` (WB18) and `--train-hours` (WB23, once built); `lane-b status` shows the deferral/refusal reasons; new `lane-b verify-repo <dir>` runs the WB13 scan locally (no upload, never executes the repo) | Lane B table, WB24 row |
| (b) | **Batch implementation PRs (owner 2026-11-23, DECIDED).** An implementation PR may batch **up to 3 WPs** when (1) their PATHS sets are disjoint, (2) none depends on another in the batch, (3) each WP's tests are already on main. Each WP in the batch is still checked on its own: scope (its PATHS only, files attributed per WP), its own accept, and CROSS; the ledger records one row per WP. A red WP is split out of the batch by the builder (the others proceed); WPs whose PATHS overlap stay serial. Tests-corrections found in one pass may likewise be batched in one PR (one dir per WP). Labels: one `laned:<WP>` per WP in the batch | §3.1 rule 7 (former rules 7–8 renumbered 8–9); §5 S2 Build row |

## rev6.2 (owner decisions 2026-11-22; main @ 6108ebd)

Records five decisions the owner approved on 2026-11-22, (c)–(f) ("do all ur
recommendations") and (h), plus one proposed WP, (g). Plan text only; nothing else in the design changes. Every
changed item is marked **(rev6.2)**. They are build-level rules inside
existing WPs, not new §6 D-numbers, so the §17.4 table is unchanged; the
dated entries are listed under it (§17.4, "Owner decisions 2026-11-22").

| Item | Decision | Landed in |
|---|---|---|
| (c) | LB1 normalised hashes use the WB11 **schema 2** z-grid rule (`cascade_contamination_hashset/2`, encoding `cascade-zgrid/2`), replacing "rounded to 4 significant figures"; low-information windows are excluded at build and ignored at lookup (RV6). PRs #617/#621, #632 | §12b.1; WB11 row |
| (d) | While a lane B funded entry is deferred (`sentinel_target` or `rows_held`), its payer key in `PayerKeyVault` gets its TTL refreshed on each deferred tick, the same rule as the entry (WB18 RV6, #629/#633). No WP owns the vault, so it's a follow-up note on WB18 | WB18 row |
| (e) | The W18 vault ledger `laned_vault_ledger/1` (current era, entries with statuses / `closed_era`, open disputes) is produced by W17 | W17 row |
| (f) | A W-S2 warm-start acceptance made at an era-init block is governed by the sentinel of the era that init block opens; a follow-up for the acceptance callers (trainer `accept_candidates`, validator `verify_acceptance`) | §16 Work and tests; W-S2 row |
| (g) | **Proposed (owner to accept):** new WP W8b wires `settle_lane_d` into the production validator loop (boundary hash, sentinel, bench `SweepSession`, `expect_prior_sha256`, lowfreq, house/king state; then sentinel sweep/board, signed report + receipt, pay into W8 emission only via the report); fail-closed, restart-safe, off or bounded against the weight loop | Lane D core table, W8b row |
| (h) | **Batch tests-only PRs (owner 2026-11-22).** Implementation, tests-correction, evidence and review PRs stay one WP per PR; a tests-only PR may carry several WPs' `tests/accept/<wp>/` directories, one `laned:<WP>` label per WP, each WP's expect-fail (or verification) accept run executed and ledger-recorded separately, the whole PR failing if any WP's check fails; dependents still only after the dependency's code is on main | §3.1 rule 6 (+ restatements: rev6 change 16, §3.0 Roles, §17 S2 row, risk table "PR hygiene") |

## rev6.1 (RV1 rubric fixes; main @ bdeed4d, 2026-11-21; RV3 fixes, main @ 45f8b61, 2026-11-22)

Plan text only, **no design change** (proposals 23–26 are not applied).
Every changed item is marked **(rev6.1)**. Sources: RV1 (`evidence/rv1.md`
on `laned/rv1`), non-blocking notes 1, 2, 10, 12 and the lane A consequence
of R3's lint (items 1–3); RV3 (`evidence/rv3.md` on main), BLOCKING 1 and
notes 2, 5, 10, 11, 21 (items 4–10).

| Item | Change | Landed in |
|---|---|---|
| 1 | W0 `Runs in` `ci + owner merge` → `ci`. The owner merge is the PR merge every WP has (§3.1 rule 6); as written it made W0 "external", so it needed an `evidence/w0.json` that nothing produces, blocking W0 at L4/L5 and so S2 (RV1 note 1) | §3.0 W0 |
| 2 | "Build WP" defined for §17.2: the §3.0 WPs whose `Runs in` is ci-only, or ci plus pod/testnet for their CI-checkable parts; owner execution WPs (R2x, R6x, R7t, R7m, R9, R13, R14, R15, R16, X*) are excluded and gate their own stages. Consistent with the §3.0 Review WPs count (RV1 note 2) | §17.2 S2 row |
| 3 | Proposed design changes 23 (code-enforced stake gate for `[lane_d] from_block`), 24 (`[lane_c] remainder` restatement refused) and 25 (lane A tuning after lane D activation). Not applied: owner decision (RV1 notes 10, 12) | Proposed design changes 23–25 |
| 4 | W6b must write `frontier` for every series (one record per earlier snapshot holding it) and a `feeds` entry (`mirror_lag_hours`, `forecast_product`) for every feed in `pool_d`; a `pool_d` missing either is invalid (RV3 BLOCKING 1) | §3.0 W6b, §1.3 (W6 metadata bullet), §1.7 Metadata |
| 5 | W9 refuses any snapshot/series lacking `frontier` or with an unlisted feed, and re-derives S0 / backfill / lag eligibility without importing `cascade/validator/lane_d_forward` (invariant 12) (RV3 BLOCKING 1) | §3.0 W9 |
| 6 | T8 clock labels corrected to §1.3 and `lane_d_forward.py:103-110`: data time passes, first-seen fails ⇒ `publish_ahead`; first-seen passes, data time fails ⇒ `late_mirror` (data old, mirror saw it late); missing `frontier` / unlisted feed added as refused cases (RV3 note 5) | §4 T8 |
| 7 | Pre-S5-restart snapshot extended to the trainer state under `/root/cascade/_train_work/` (`era_state.json`, `trainer_submissions.json`, `funded_queue.json`, `trainer_promotion.json`, `champion_publisher.json`, `lane_b_driver.json`, `coldkey_identities.json`, `published_settlements.json`, its `activation_*` / `netconfig` files), `provisioner_state.json`, `pod_monitor_state.json` and the deployed `chain.toml` (live overrides) with `hosts.toml` / `provision.toml` (RV3 note 21) | §10.8, §3.0 R5, §10.5 S5 |
| 8 | Wiring trap: `validator/weights.py` and the audit pass lane D's element to `emission_vector` only when a B or C pay record exists, so lane D's share would be lost; the WP that wires lane D pay must add the lane D record to that condition, and W9 decides invariant 12 for the shared lane D weight math (RV3 notes 10, 11) | §3.0 W7c, W9 |
| 9 | W3b5 defines the low-frequency result for a confirm candidate whose frozen set holds low-frequency windows: `pass`/`fail` with the true `n_windows` (replacing W7a's interim `not_candidate`); enum already holds both, extended only for a new value | §3.0 W3b5 |
| 10 | Proposed design change 26: arming the §16 sentinel (W-S1) gated on R4's observed `cfg-` stake ≥ 2/3 **and** all satellites upgraded. Not applied: owner decision (RV3 note 2) | Proposed design change 26, §16 (pointer) |

## rev6 (completion rubric, loop-readiness fixes, stale items; main @ 07117e1, 2026-11-21)

Plan text only. No code, chain.toml or prod-box changes, and **no design
change**. Every changed item is marked **(rev6)**. Anything else I'd change is
under "Proposed design changes" (change 22), not applied.

**Base.** The branch `claude/sleepy-cray-7uutm1` is on the remote and already
contains origin/main `07117e1` (main hasn't moved since rev5), so no rebase was
needed and no force-push was done. The repo facts below were re-checked on
`07117e1`.

**Changelog (brief item → where it landed):**

| Item | Change | Landed in |
|---|---|---|
| 1 | §17 "Completion rubric and status board": `scripts/laned/status` → `docs/plans/STATUS.md` + `status/laned_build.json` on `laned/build-state` (never merged); 17.1 levels L0–L7 + side states; 17.2 stage completion; 17.3 counters; 17.4 decision register (status, option, class S/O, who/when) + owner inputs OA1–OA8; 17.5 milestones; 17.6 agent rules | §17 (new); status line; §3 conventions |
| 2a | §3.1 rule 6: a tests-only PR then ONE implementation PR; consistent with rule 1. Rule 7 and the Review WPs paragraph count WPs at L3 | §3.1, §3.0 Review WPs |
| 2b | W0c adds (i) `scripts/laned/status` + `accept --gate`; (ii) `laned-accept.yml` (PRs labelled `laned:<WP>`, expect-fail mode for tests-only PRs, main after merge); (iii) the rule-1 CI check (and the `PATHS` check); (iv) self-tests (fixture repo ⇒ exact levels; planted rule-1 violation fails) | §3.0 W0c |
| 2c | Roles named (test author, builder, preparer, reviewer); a builder never edits `tests/accept/`, the plan, `scripts/laned/` or the status files | §3 conventions, §17.6 |
| 3a | S6 is one stage (D45): X6b removed; X6a Accept = observed `cfg-` stake ≥ 2/3 before `effective_block`, webhook test alert, first paid settlement's vectors match on ≥ 2 validators and satellite replay matches; no "≥ 2 eras of reports" | §3.0 X6a/X6b, Self-check stage map |
| 3b | S2b: "diffs only W2–W12" → "diffs only the lane D and lane B build WPs (§5 S2)" | §5 row 7, §14.5 |
| 3c | Appendix A → a pointer to §17 and STATUS.md; OA1–OA8 moved to §17.4 with status, what each unblocks and its evidence file; `LANE_D_PROGRESS.md` retired (pointer) | Appendix A, §17.4, Appendix B, `LANE_D_PROGRESS.md`, `decisions/DEC-CA-0070-…` |
| 3d | Leftover shadow / separate pay stage: §1.4 D21 table ("share-0 shadow (S6a)"), §1.4 "S6b signed overlay", §5 row 12b (struck), §13 row, §15 "before S6b", v3 changelog items 4, 6, 14, Proposed change 5; `LANE_D_S1_RESULTS.md` ("S6a shadow") is retired to a pointer (change 21) | those places |
| extra | W14 was still "blocked: needs owner (Proposed change 4)", but change 4 is accepted: W14 unblocked; T27 and the §5 kill-switch text now say "until W14 lands" | W14, T27, §5 kill switch, rev5 A2 rows |
| extra | §5 S0 gate reads the §17.4 register | §5 row 1 |
| extra | Proposed change 22: five §15 rows (D25–D27, D29, D31) have no recommended option; recorded as Option A, owner to confirm | Proposed design changes, §17.4 |

**Propagation (item × place):**

| Item | § | WP | T-row | Stage map | Docs |
|---|---|---|---|---|---|
| 1 §17 | done (§17, status line) | done (W0c builds the script) | n/a: no T-row tests build tooling | n/a: §17.2 is the stage rule itself | done (STATUS.md is generated, not committed) |
| 2a rule 6 | done (§3.1 rules 6, 7; Review WPs) | n/a: applies to all WPs | n/a | n/a | n/a |
| 2b W0c | done (§17.1 L1 expect-fail) | done (W0c) | n/a: W0c's own self-tests | n/a | n/a |
| 2c roles | done (§3 conventions, §17.6) | n/a | n/a | n/a | n/a |
| 3a S6 | done (§17.2 S6) | done (X6a, X6b removed) | n/a: T27 and T29 unaffected | done (S6 row) | n/a |
| 3b S2b | done (§5 row 7, §14.5) | n/a: R9's Accept unchanged | n/a | n/a | n/a |
| 3c Appendix A | done (Appendix A, §17.4, Appendix B) | n/a | n/a | n/a | done (`LANE_D_PROGRESS.md`, `LANE_D_S1_RESULTS.md`, DEC-CA-0070 pointer) |
| 3d shadow leftovers | done (§1.4, §5 row 12b, §13, §15, v3 changelog, change 5) | n/a | n/a | done (S6) | done (`LANE_D_S1_RESULTS.md`) |
| W14 unblocked | done (§5 kill switch) | done (W14) | done (T27) | n/a | n/a |

**Repo facts that turned out wrong or missing:**

- **W14 "blocked: needs owner (Proposed change 4)"** was stale: changes 1–17
  were accepted in rev5.
- **"The lane D release diffs only W2–W12"** (S2b) and Appendix A's "W2–W12 are
  not built": the build set is W1.5–W19, W-S1–W-S3, WB10–WB23 and W-EMA (§5 S2).
- **`LANE_D_S1_RESULTS.md`** still said the forward re-check "happens at the S6a
  shadow"; Appendix B had the rev5 fix.
- **Nothing runs `tests/accept/` in CI today.** `.github/workflows/ci.yml` runs
  `pytest tests/unit` only. L1 and L3 need W0c's workflow (ii).
- **There's no Makefile on main**, so W1.5 creates `make laned-e2e` from
  nothing (a new top-level `Makefile`).
- **No `laned:*` GitHub labels exist.** Creating them is an owner step before
  the first tests-only PR (W0c row).
- **`tests/accept/`, `scripts/laned/`, `evidence/` and branch
  `laned/build-state` don't exist yet**, as expected before W0c.


## rev5 (code alignment and review gaps; main @ 07117e1, 2026-11-21)

Plan text and decision drafts only. No code, chain.toml or prod-box changes.
Every changed item is marked **(rev5)**.

**Base.** The branch `claude/sleepy-cray-7uutm1` is rebased onto main `07117e1`
(2026-11-21) and pushed. Every fact below was re-checked on `07117e1`. The
MarketCast PRs #475 and #478 were merged into stacked base branches, but their
content is on main through #453's squash: the trees are identical. `3c67b64`,
cited in earlier revisions, was an older commit of this branch.

**Changelog (item → where it landed):**

| Item | Change | Landed in |
|---|---|---|
| A1 | Training rows 128 patches; 8k/16k/32k/128k = 2×/4×/8×/32×; CPM span ≤ 16 patches ⇒ horizons ≤ 512 points | §1.1 (architecture facts, 128k bullet, Horizon "longer rungs") |
| A2 | Missing lane D report = check can't run: old vector re-asserted for `stale_after_blocks` (3600), then a whole-vector burn; W14 blocked on Proposed change 4 (**(rev6)** change 4 accepted; W14 unblocked) | §5 kill switch, §10.8, W14, T27, Proposed change 4 |
| A3 | `[generation] king_publish` (`chain.toml:1555` on `07117e1`) | §15 D26 |
| A4 | King check = `king_check_gate` over four suites; W-S1 names it | §16, W-S1 |
| A5 | `netconfig.py:210`; `lane_d.py:78-79` | W2 |
| A6 | TIME → the era's sentinel; `unselected_suite` dropped (duplicates `[v2] sentinel_suites`) | §1.3b, §1.6, W2, W3b3, T7b, §10.4, §11.1, §11.2, §14.2 |
| A7 | Lane B record renumbered **DEC-CA-0071** (main has 0069 MarketCast hooks; 0070 = lane D); file renamed | everywhere (20 refs), Appendix C, `decisions/DEC-CA-0071-…`, DEC-CA-0070 numbering note |
| A8 | S0 and W0 cover D1–D43, D45–D48, D42b; D25–D32 rows in §6 | §5 row 1, W0, §6 |
| A9 | T8 two-clock rule; launch eval host (not valeval-laned) at S4; S2 = all build WPs; self-check T14 → B1 + S4 gate; §12 and §5 row 12b; §2 items 7/8; OA order | T8, §1.2, §1.5, §1.7, §5 S2/S4, X4, §10.2, §10.4, §10.5 S4, T7, §11.2, self-check, §12, §2, App. A |
| B1 | R8: AREA for `cascade/lane_d/`, every-path-classifies test, on-pod trainer modules in `WORKER_IMAGE`; consensus and worker-image paths listed | R8, §3 Services touched |
| B2 | `declared_rows` keys = MoE ordinals 0–6 | LB2, WB14, TB13, Appendix C |
| B3 | Clamp minimum only on layers with ≥ 1 row | LB2, WB14, TB15, TB15b, Appendix C |
| B4 | Shared image; dispatch `_dispatch_funded` → `_run_funded_leg`; declared rows need no image change (confirmed: `--selection-json`); D40(b) a shared repin; D40(b) and every repin ≥ rc16 depend on W-EMA | §3, D40, WB16, SB track |
| B5 | Release-only: `(lane_b, contamination_hashset)`, `(lane_c, remainder)`, `(lane_d, remainder)` | §14.2, WB10, Appendix C, §14.2 lane D row |
| B6 | W6a join on (source, pre-suffix source key, channel), same build | W6a, Proposed change 7 |
| B7 | Quantile head, 9-level grid, no copula sampling; W0b compares quantile-head outputs | §1.5, W0b, W5d, W7a |
| B8 | Deregistration: entry voided `deregistered`; a court slot burns | §1.5 state, W5h, W8, T18c |
| B9 | ~~Lock only kings (copy at crowning);~~ **(rev6.4)** no lock: Hippius has no Object Lock, so kings rest on content addressing + re-hash on every read + signed report digest + R2/HF mirrors; pending = content addressing + digest re-check; W3a probes `UploadPartCopy` | §1.2 storage, W3a |
| B10 | `[lane_d] ready_ttl_blocks` ⇒ `ready_expired` | §1.5 state, W2, W3, W5h, T18d, §14.2 |
| B11 | **W-EMA**: score the raw cooled endpoint on every leg; EMA files unscored; telemetry only | §8, W-EMA row (§3), §3 Services touched, D40, WB16 |
| B12 | Daily Merkle root of the first-seen log committed before use | W6fs, W7a, §9 conflict of interest |
| B13 | Licence attestation at `ready`; Hub/HF takedown path | W3, W13, §11.2 LANE_D.md |
| B14 | `[lane_d] max_source_share` | W5b, T18e, D29 (§6), W2, §14.2 |
| B15 | Rented eval host sees loser weights and `pool_d_hist` | §9 |
| C1 | **D46** house T (lineage commit block recommended) | §6; Proposed change 18 withdrawn (wrong premise) |
| C2 | **D47** house precision under BF16 | §6 |
| C3 | **D48** where D's share comes from inside A; `lane_pool_shares` signature | §6, W8 |
| D | Deployment facts (below) | this changelog |
| D42 | Owner: lane B GPU list + miner `train_hours` 0.5–6 h; D42b | §12b.0, §12b.6, D42, D42b, WB19, WB22, WB23, TB23–TB27, OA8, App. C |
| D43 | Owner: lane D reveals immediately | §1.2 step 4, W10, D43 |
| D44 | Drafted (EMA instead of cooldown), **withdrawn by the owner** | §6 (withdrawn row) |
| D45 | Owner: no shadow modes; one activation block | §5, D10, D19, D37, X6a, §14.5, §16, W-S1, SB5 |
| Harvest | Owner: long-history harvest to 128k now, scored from S8 | §1.7, R16, §5 S1 |

**Propagation table** ("n/a" says why):

| Item | § | WP | T-row | Key list | §14.2 | Report schema | State | Docs |
|---|---|---|---|---|---|---|---|---|
| A1 | done | n/a: fact only | n/a: T7 measures contexts | n/a | n/a | n/a | n/a | n/a: LANE_D.md cites §1.1 |
| A2 | done | done (W14) | done (T27) | n/a | n/a | n/a: change 4 undecided (**(rev6)** accepted) | n/a | n/a: OPERATOR row points at §5 |
| A3 | done | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| A4 | done | done (W-S1) | n/a: TS1–TS3 unchanged | n/a | n/a | n/a | n/a | n/a |
| A5 | done | done (W2) | n/a | n/a | n/a | n/a | n/a | n/a |
| A6 | done | done (W2, W3b3) | done (T7b) | done (dropped) | done | done (index row) | n/a | done (LANE_D.md) |
| A7 | done | done (WB21) | n/a | n/a | n/a | n/a | n/a | done (DEC files) |
| A8 | done | done (W0) | n/a | n/a | n/a | n/a | n/a | n/a |
| A9 | done | done (X4, S2 list) | done (T7, T8) | n/a | n/a | n/a | n/a | done (OPERATOR row) |
| B1 | done | done (R8) | n/a: R8 accept runs the classify test | n/a | n/a | n/a | n/a | n/a |
| B2 | done | done (WB14) | done (TB13) | n/a | n/a | n/a | n/a | done (App. C) |
| B3 | done | done (WB14) | done (TB15, TB15b) | n/a | n/a | n/a | n/a | done (App. C) |
| B4 | done | done (WB16, W-EMA) | n/a: (b) undecided | n/a | n/a | n/a | n/a | n/a |
| B5 | done | done (WB10, W2) | done (WB10/W2 accept: overlay refused) | done | done | n/a | n/a | done (App. C) |
| B6 | done | done (W6a) | n/a: W6a accept | n/a | n/a | n/a | n/a | n/a |
| B7 | done | done (W0b, W5d) | done (W5d CI test) | n/a | n/a | done (W7a `scoring_method`) | n/a | n/a |
| B8 | done | done (W5h, W8) | done (T18c) | n/a | n/a | done (typed reason) | done (`deregistered`) | n/a: LANE_D.md FAQ via WB20/W-docs |
| B9 | done | done (W3a) | n/a: W3a probe evidence | n/a | n/a | n/a | n/a | n/a |
| B10 | done | done (W2, W3, W5h) | done (T18d) | done | done | done (typed reason) | done (`ready_expired`) | n/a: LANE_D.md covers the ready flow |
| B11 | done (§8) | done (W-EMA) | done (W-EMA tests) | n/a: image constant, not a key | n/a | n/a: telemetry, not in the receipt | n/a | n/a |
| B12 | done (§9) | done (W6fs, W7a) | n/a: W6fs accept | n/a | n/a | done (W7a roots) | n/a | n/a |
| B13 | done | done (W3, W13) | n/a: W3/W13 accept | n/a | n/a | n/a | done (attestation at ready, W3) | done (LANE_D.md) |
| B14 | done | done (W5b, W2) | done (T18e) | done | done | n/a: windows already listed | n/a | n/a |
| B15 | done (§9) | n/a: accepted risk | n/a | n/a | n/a | n/a | n/a | n/a |
| C1–C3 | done (§6) | done (W8 for C3) | n/a: until decided | n/a | n/a | n/a | n/a | n/a |
| D42 | done | done | done (TB23–TB27) | done | done | done (signed leg record) | n/a | done (WB20) |
| D43 | done | done (W10) | done (W10 test) | n/a | n/a | n/a | n/a | n/a |
| D45 | done | done (X6a, W-S1) | n/a: removes shadow rows | n/a | n/a | n/a | n/a | n/a |
| Harvest | done | done (R16) | n/a: R16 accept | n/a | n/a | n/a | n/a | n/a |

**New keys** (each: loader parse, ALLOWED or release-only, round-trip and
mutation tests):

| Key | Parse + tests in | Overlay? | Tests named |
|---|---|---|---|
| `[lane_d] ready_ttl_blocks` | W2 | `ALLOWED["lane_d"]` | W2 accept: parse, round-trip, mutation, `unread_keys.py` |
| `[lane_d] max_source_share` | W2 | `ALLOWED["lane_d"]` | W2 accept: same |
| `[lane_d] remainder` (existing placeholder) | W2 | **release-only** (`netconfig.py:210` tuple) | W2 accept: an overlay setting it is refused |
| `[lane_c] remainder` (existing) | W2 (shared release) | **release-only** (newly) | W2 accept: overlay refused; unarmed digests byte-identical |
| `[lane_b] contamination_hashset` | WB10 | **release-only** | WB10 accept: overlay refused |
| `[lane_b] funded_pod_skus`, `max_leg_hours`, `max_leg_seconds`, `guard_factor` | WB10 | `ALLOWED["lane_b"] = None` | WB10 accept: parse, digest drop-when-default, round-trip, mutation |

**Repo facts in the brief that turned out wrong or imprecise:**

- **A1:** `max_patches` isn't "read by nothing". Lane B's horizon-affinity
  decode reads it (`expert_lane.py:365`); training doesn't.
- **A3:** on `07117e1`, `king_publish` is at `chain.toml:1555` (it was `:1554`
  on `5ce1f88`).
- **B6:** "source id + content hash" can't be the join key. `content_hash` is
  the sha256 of the values (`builder.py:295`), so it changes as tails move,
  and `pool` and `pool_d` histories hash differently. The plan uses (source,
  pre-suffix key, channel) within one build.
- **C3:** the king's ladder share runs from **10%** (after a fail) to 95%, not
  25–95% (`chain.toml:1223-1226`).
- **A7:** PR #479 also claimed DEC-CA-0069; **(rev6)** renumbered DEC-CA-0072 after merge.
- **B11:** rc15n (#477's minimal image, built on rc15 `5e3fade` + the pin)
  doesn't contain power EMA. W-EMA blocks #477's repin only if that image is
  built from main.
- **Earlier session errors corrected:**
  - "Stop publishing ⇒ immediate whole burn" was wrong; it's a one-era
    re-assert first (A2).
  - "The king retrains every era" was wrong; it trains once per seed epoch
    (Proposed change 18 withdrawn).
  - "`ALLOWED["lane_b"] = None` blocks overlays" was wrong; it allows every
    key.

**D: deployment-only facts (read-only checks on the box):**

| Fact | Result |
|---|---|
| rc37 validator / rc41 trainer | **verified:** service PYTHONPATHs are `/root/cascade-val-rc37` and `/root/cascade-rc41` |
| 21 LIVE OVERRIDEs | **verified:** 21 in the deployed chain.toml |
| keep-alive (#471) absent from rc37 | **verified:** `_weights_keepalive_tick` not in rc37 |
| valpriv-4090 spec (24 GB) | host listed in `valpriv-eval.toml`; **VRAM unverified (box)** |
| valeval-blackwell4 spec | **unverified (box):** not in the deployed config (only blackwell2's standing cost) |
| `[lane_d] takes_from = "lane_b"` | **verified** on the deployed file and on main |
| Lane B image = lane A image | **verified:** both pins are rc15 `5d41d94b…`; no image key under `[lane_b]` |
| OPSLOG line refs | **drifted** (newest-first log). `:2329` is replaced by its entry heading; `:164` is unverified |
| Any image ≥ rc16 in production? | **No evidence.** rc16 was built and qualified on 2026-11-01, but the switch was held as consensus-unsafe and the trainer stayed on rc15. rc15n (#477) was tried on 2026-11-20 and reverted. The deployed pin is rc15. Experiment pods aren't covered by this check |

## rev4 (lane B pass, 2026-11-21)

Lane B stays trainer-run, with miner-chosen expert rows (DEC-CA-0071 Open
option (b)) and one-suite targets ("improve one, regress none"); see §12b.
The miner-weights variant (§12) is parked. D32 is resolved, and D36–D42 are
new. Every changed item is marked **(rev4)**.

**rev4 mappings:**

| Item | Maps to |
|---|---|
| TB10–TB12 | WB11, WB13 |
| TB13, TB15 | WB14 |
| TB14 | WB16 |
| TB16, TB21 | WB15 |
| TB17, TB18 | WB17 |
| TB19, TB20 | WB18 |
| TB22 | WB10 |
| TS1–TS3, TS5 | W-S1 |
| TS4 | W-S2, WB18 |
| Sentinel board | W-S3 |
| Every new `[lane_b]` key | WB10 (plus the owning WP named in the §3 table) |
| LB6 measurement | WB19 |
| Docs | WB20 |
| DEC-CA-0071 amendment | WB21 |
| Lane B track | SB0–SB5 (§5) |

## rev3 (design pass, 2026-11-21)

This pass changes the design: items A1–A11 of the rev3 brief, plus the
incentive review in §15. Every changed item is marked **(rev3)**. New
decisions are D20–D24 and D25–D32 (§15). The S1 measurements are in
`LANE_D_S1_RESULTS.md` and the run status in `LANE_D_PROGRESS.md` **(rev6: both retired; status is
§17, results are Appendix B)**. The
v3 loop-readiness material below predates rev3. A new loop-readiness pass
runs only after the owner settles rev3's [DECIDE]s.

**rev3 additions to the v3 mappings:**

| Item | Maps to |
|---|---|
| T4b | W3b5 |
| T6b | W5h, W3, W9 |
| T9b | W5e |
| T18b | W5h |
| T14 (redefined) | B1 `scripts/null_fpr.py`, W5e, S4 gate |
| W6fs | S1 / S2 (launch-blocking) |
| W6-S8 and R15 | S8 and S3 |
| New stage S8 | W6-S8 |

## v3 changelog

The design is unchanged: lane D rules, judging, storage, pay, rollout order
and decisions D1–D18 stay as they were in v2. Changes:

**Contradictions fixed**

1. **§5 S2 gate.** Receipt replay now reads: unarmed ⇒ byte-identical; armed
   ⇒ lane A verdicts, court and lane A weight entries identical, and lane D
   fields present and schema-valid. It matches invariant 2.
2. **§10.5 S5** aligned with §5 S5. The order is eval host → validator →
   lane D intake → trainer, and all 8 snapshot files from §10.8 are listed.
3. **§14.4 / §14.5:** "S1b" renamed to S0a–S0c (§5).
4. **§13** "live overrides" row: "typed-block activation" → "signed-overlay
   activation (D10; S0c, S6a)" **(rev6: S6a is now S6, D45)**.
5. **§1.2:** sub tokens were said to suit the eval pod, which contradicted
   "presigned GETs only, no credentials on pods". It now says builder only;
   the pod gets presigned GETs.
6. **§1.4:** the `takes_from` change goes by the S6b signed overlay, not a
   chain.toml edit (consistent with §5 S6b and §14.2). **(rev6)** S6b is
   merged into S6: it's the one S6 activation overlay (D45).
7. **§1.3:** "judged once on that frozen set" → "judged once on its cohort's
   intersection set, ⊆ its frozen set" (consistent with §1.3 Cohorts).
8. **§1.3b:** "rows come from cached sweeps" → "re-swept in the same host
   session" (consistent with the paired-on-one-host rule).
9. **§1.2:** the cross-reference "§2.2" (which doesn't exist) → "§2 item 1".

**Citations re-verified against the repo**

10. `store.py:337-344` → `store.py:61, 340-367` (`DigestOwned` is defined at
    `:61` and raised at `:348`, `:365`). §1.2.
11. `builder.py:262-277` → `builder.py:279-290` (the metadata dict). §1.3.
12. `netconfig.py:495-505` → `netconfig.py:346-354` (`build_config` merges
    onto `effective_raw(base)`). §5 S0a.

    Checked and still correct: `checkpoint_guard.py:139-171, 151-153,
    175-196, 290-302`; `pyramid_model.py:89-93, 210-211, 259`;
    `pyramid_wrapper.py:35-38, 51`; `lane_c_manifest.py:199-203`;
    `chain.py:495-535`; `store.py:258-275`; `scripts/deploy_gate.py:55`;
    `builder.py:264-271`; `tsbench_forge.py:149, 204, 380-399, 446-460`;
    `windows.py:433-449, 556-611`; `bootstrap.py:225, 393-418`;
    `emission.py:330-420`; `eval_offload.py:7-10`; `validator/pool.py:71-77`;
    `publish-pool.yml:72-79` (timeout 180 at `:79`), `121-131` (10 GB at
    `:129`); `activation.py:1334-1357`; `lane_d.py:65-68`;
    `netconfig.py:83-118, 195-217`; `loop.py:1030, 1078, 1410`;
    `lane_b_driver.py:777`.

**Additions**

13. **§3.2 invariants 9–12:**
    - 9: `share = 0` gives an identical vector plus a valid report.
    - 10: paired unarmed and armed tests.
    - 11: typed reason enums.
    - 12: audit and replay code is independent of producer code.

    The invariants also now name their owning WPs.
14. **§5 S6a and §6: D19**, an owner-run shadow test entry on a dedicated
    hotkey (recommended). **(rev6) Superseded** by D45 and D19 (rev5): the test
    entries run on testnet in S3; there's no mainnet shadow and no S6a.
15. **§3 restructured for the loop:**
    - Accept and Runs-in columns.
    - The `scripts/laned/accept` convention.
    - External WPs are blocked, never mocked; owner steps are prepared but
      not run.
    - New WPs:
      - W0c loop tooling (stub scan, unread keys, follower patterns);
      - **W1.5 walking skeleton** (`make laned-e2e`);
      - W2b `[lane_d_ops]`;
      - W5 split into W5a–W5g; W6 into W6a–W6d plus W6s-*n* per source;
        W3b into W3b1–W3b4; W7 into W7a–W7d;
      - W3 and W10 are gated on W3a's storage decision record and build only
        the chosen path.
    - New owned work:
      - W13 king publication; W14 fast kill; W15 prefetch LRU; W16 reference
        fine-tune; W17 `status/lane_d.json`; W18 TTL sweeper;
      - W19 off-poll-thread + keep-alive;
      - W12 split into W12a docs and W12b status/discovery.
    - Rollout tooling R1–R14:
      - block arithmetic, the note-budget decoder plus its live run, overlay
        lint, the stake tracker;
      - previous-release worktrees and restart scripts;
      - the de-drift classifier plus its execution, overlay bring-up
        artefacts plus testnet and mainnet runs;
      - `deploy_gate` for the new services, the catch-up release;
      - the state-copy upgrade harness, the shipped-config startup-gate
        test;
      - the eval-host provisioning kit, builder provisioning, testnet stack
        provisioning.
    - Review WPs RVn and execution WPs X3–X7.
16. **§3.1 build rules:**
    - tests-first in read-only `tests/accept/`;
    - mock only external services;
    - no placeholders (stub scan in every Accept);
    - per-key parse, ALLOWED entry, round-trip and mutation tests, with an
      unread-key script;
    - follower patterns checked by a script;
    - one PR per WP, dependents only after the dependency is on main
      **(rev6.2:** implementation, tests-correction, evidence and review PRs
      stay one WP per PR; a tests-only PR may batch several WPs' tests, one
      `laned:<WP>` label and one separately recorded accept run per WP;
      dependents only after the dependency's code is on main**)**;
    - a fresh-context review after every 5 build WPs;
    - blocked beats faked.
17. **§3 "Services touched"** now lists `deploy_gate` and the trainer
    (capability release only).

### Self-check

**Every WP has an Accept and a Runs-in value: yes.** W1.5's Accept is
`make laned-e2e`, run in ci.

**T-rows → WPs** (all also executed in X3):

| Row | WPs |
|---|---|
| T1 | W5f, W8 |
| T2 | W3 |
| T3 | W16, W5e, W3b2, W13 |
| T4 | W5e, W3b2 |
| T5 | W4 |
| T6 | W3, W4 |
| T7 | W5d, W0b |
| T7b | W3b1–W3b4 |
| T7c | W3, W10, W15 |
| T8 | W5a, W6b |
| T9 | W5c |
| T10 | W5f |
| T11 | W7c |
| T12 | W7d, W9 |
| T13 | W1.5, W8 |
| T14 | **(rev5)** B1 (`scripts/null_fpr.py`, S1) + the **S4 gate**; W5e |
| T15 | W19 |
| T16 | W3, W5f |
| T17 | W3b4, W5g |
| T18 | W2, W5b, W14 |
| T19 | W4, R12 |
| T20 | W11 |
| T21 | R3, R7, W2 |
| T22 | R2 |
| T23 | W7d |
| T24 | R3, W2 |
| T25 | R10 |
| T26 | R11 |
| T27 | W14, R7 |
| T28 | W4, R8 |
| T29 | W12a (announcement), R14 |

**Invariants → WPs:**

| # | WPs |
|---|---|
| 1 | W1.5, W2, W8 |
| 2 | W1.5, W8, W9 |
| 3 | W4 |
| 4 | W5a, W9 |
| 5 | W7c |
| 6 | W19 |
| 7 | W7d |
| 8 | W6c |
| 9 | W1.5, W8 |
| 10 | W0c |
| 11 | W5g, W7a |
| 12 | W7d, W9 |

**Stage gates → WPs:**

| Stage | WPs |
|---|---|
| S0 | W0 |
| S0a | R6, R6x |
| S0b | R2, R2x |
| S0c | R7, R7t, R7m |
| S1 | W1, W0b, W3a |
| S2 | all W, plus RVn |
| S2b | R9 |
| S3 | R14, X3 |
| S4 | R12, X4 |
| S5 | R5, R8, X5 |
| S6 **(rev6)**: one stage (D45); S6a/S6b and X6b removed | R4, X6a |
| S7 | W11, X7 |

**§11.1 → WPs:**

| Item | WPs |
|---|---|
| vault | W3a, W3 |
| retention / archive | W18 |
| kings, index, cards, one-digest check | W13 |
| `pool_d` pins | W6d |
| eval cache | W15 |

**(rev4)** Lane B is no longer unchanged: its WPs are WB10–WB21 (§3) and its T-rows TB10–TB22 (§4). Lane A and the generator rows are unchanged, so no WP is needed for
them.

**§11.2 → WPs:**

| Item | WPs |
|---|---|
| LANE_D.md, INTERFACE, MINER, AGENT_MINING, AGENTS/agent.md, llms.txt (both), VALIDATOR, OPERATOR, ARCHITECTURE, website, README/CHANGES | W12a |
| `status/lane_d.json` | W17 |
| `cascade status` / `discovery-check` | W12b |
| Decisions (DEC-CA-0070, DEC-CA-0062 superseded) | W0 |

**Unmapped after the additions in item 15: none.** Before them, six items
had no WP: T15 / invariant 6, T25, T26, S4 eval-host provisioning, §10.2
builder provisioning, and §10.3 testnet provisioning. They're now owned by
W19, R10, R11, R12, R13 and R14.

**Nothing in §1–§2 design changed in v3.**

### Proposed design changes (NOT applied: owner decision)

**(rev5) Owner 2026-11-21: changes 1–17, 20 and 21 are ACCEPTED.** 18 is
withdrawn and 19 is adopted as D43. Each accepted change is built in the WPs
its "Affects" line names. The § text it affects is rewritten when that WP
lands, so until then the change text below is authoritative over the older
§ wording it contradicts.

These come from review round 2 (fresh context, read-only, checked against
the code). Each is verified where marked.

1. **Forward rule admits pre-T data.**
   - "Absent from the first snapshot after max(T)" is also true for values
     our mirror got late, and for series not selected in that snapshot
     (every point "absent").
   - Proposed rule: eligible iff (a) the series is present in S0 with a
     recorded `end_ts`, (b) `target_start_ts > max(T_E,T_K) + margin` in data
     time, and (c) every target ts > `end_ts(S0)`. Series absent from S0
     start their clock at first appearance.
   - T8's "backfills never eligible" needs this. Affects §1.3, W5a.
2. **Publish-ahead feeds aren't excluded automatically.** The forge reader
   drops rows with ts ≥ as_of + 1 day (`tsbench_forge.py:393-396`). Proposed:
   eligibility metadata from the mirror's landing time, or a per-source
   `publishes_ahead` deny-list. Affects §1.3, W6b.
3. **Copy of a fresh house can win.** `cohort_maxt_lcbs` compares challengers
   only against the king (`bootstrap.py:393-418`). Proposed:
   - a miner challenger must also clear `d_margin` against the current house
     on the same set;
   - the house ranks first on ties.

   Affects §1.3, T16.
4. **"Stop publishing ⇒ only lane D burns" is false.** **(rev5)** A missing
   report is a check that can't run: the satellite re-asserts the last
   verified vector for `stale_after_blocks` (3600), then burns the **whole**
   vector (`cascade/validator/satellite.py:30-45`). Proposed: a structured `lane_d: {status: "absent"}` receipt
   state with a lane-D-only burn, reusing MarketCast's `REPORT_ABSENT` /
   `INFRA_DEFERRED` (verified: `marketcast_report.py:32-34`). Affects §5 kill
   switch, §10.8, W14, T27.
5. **Refused overlays stay refused after upgrade.** The digest persists in
   `state.refused` (verified: `netconfig.py:822-826`). Proposed:
   - publish a restating seq N+1 after the stake gate if any node is known to
     have refused seq N;
   - or a code change that clears refusals on release upgrade.

   Affects §5 S6 **(rev6)**, §14.1, T23.
6. **Reuse MarketCast** (DEC-CA-0064: forward windows, absolute-grid window
   ids, signed per-window report with quantile sidecars, absent/deferred
   states) instead of reinventing W5/W7. Affects W5a–W7d.
7. **The window origin grid is undefined.** Proposed:
   - non-overlapping origins on the absolute grid `(ts/freq) % 64 == 0`;
   - every eligible origin included;
   - a stable source-based `series_id`. **(rev5)** Today's ids aren't
     stable: collision suffixes follow sort order (`builder.py:361-370,
     471-480`), and `content_hash` follows the values (`:295`). Proposed
     `series_id = <source>:<source-native key>[:<channel>]`, with no suffixes
     and in-source collisions refused. W6a uses that join rule from the
     start.

   Affects §1.3, W6a.
8. **MASE scale.** The box has only targets; per-model contexts would also
   change the scale. Proposed: a per-window MASE scale in `pool_d_tgt` from a
   fixed history slice. Affects §1.7, W6c.
9. **Target revisions and interpolation.** Proposed:
   - targets come from the first complete snapshot;
   - interpolated or missing target points are masked;
   - per-channel `end_ts`.

   Affects §1.3, W6b.
10. **Gate context.** The runners use the pinned 4096 wrapper. Proposed:
    either "gate always at 4096", or runners honour the declared context.
    Affects §1.3b, W3b2.
11. **Transfer host** for publishing kings, the vault mirror and the HF
    mirror (credentials can't be on the box or pods). Proposed: the builder
    VM with scoped write tokens. Affects §10.2, §10.6, W13.
12. **Loser audit trail.** Proposed:
    - per-model quantile sidecars in the private bucket;
    - a signed per-snapshot metadata digest published at build time.

    Affects §1.5, W7a.
13. **Testnet overlay is inert:** `owner_hotkey = ""`
    (`chain.testnet.toml:903`, verified). It's handled as a setup task in
    R14, with no design change, but noted here.
14. **(rev5) Moot under D45 (no shadow).** ~~Shadow semantics.~~ Proposed: S6a crowns are provisional and
    unpublished, and `lane_d_state` resets at the S6b `effective_block` (or
    say explicitly that they carry). Affects §5 S6a.
15. **Invariant 8 may not hold over time.** Later `pool_d_hist` snapshots
    contain earlier windows' targets as history. Under forward-only rules
    target secrecy after T protects nothing. Proposed: restate D18's
    rationale, or cut `pool_d_hist` so it never covers an open frozen window.
    Affects D18, §1.7.
16. **BF16 needs a lane-D-only guard path.** Today's guard requires the
    pinned F32 spec (verified: `checkpoint_guard.py:323-328`). This is
    implementation, recorded in W4; no design change.
17. **Minor design gaps:**
    - the house's declared context (4096);
    - what replaces "gate vs house" when the house is the challenger;
    - whether a house refresh replaces a pending house entry;
    - T for a payload revealed twice (first reveal, recorded in state);
    - `pool_d` snapshot retention;
    - bench-pod disk budget for 3 pinned checkpoints.
18. **(rev5) Withdrawn: wrong premise.** It claimed the lane A king's leg is
    retrained every era. It isn't: the king leg trains **once per seed epoch**
    (from one warm start to the next) and is carried across that epoch's eras
    (`trainer/rolling.py:125-128`, `shared/era.py:440-457`). The real issue,
    the house's T resetting on every KING_A change, is now owner decision
    **D46** (§6).
19. **(rev5) ADOPTED by the owner as D43 (§1.2 step 4, W10).** *Lane D
    reveal timing.* The standard timelock helper lands a
    reveal at `boundary − reveal_margin_blocks` (25) (`shared/chain.py:89-100`;
    copy-contest protection for lane A pointers). For lane D the weights stay
    private in the vault and exact-digest dedup applies, so revealing at once
    (`min_blocks`) moves T, and the forward clock, earlier by up to one
    settlement (3 h). Alternatively, take T as the commit-inclusion block,
    which binds the digest at commit, if poll_commitments can read it.
    Affects §1.2 step 4, §1.3.
20. **(rev5) Lanes A and B aren't forward-only.** Their pool snapshots "may
    include observations that existed before submission"
    (`docs/EVAL_POOL.md:22-24`). A generator can embed recent public series
    from the same feeds. Proposed: a **monitoring metric** (never gating, and not a rollout mode, so D45 isn't affected), once W6 timestamps
    exist, comparing each new lane A king's paired LCB on windows whose
    targets precede the generator's reveal against windows after it. A large
    gap is a memorisation flag, logged and never gating.
21. **(rev5) Retire `LANE_D_PROGRESS.md` and `LANE_D_S1_RESULTS.md`.**
    Appendices A and B duplicate them, and the progress file still has the
    old OA order and OA8 text. Keep one copy (the plan).
    **(rev6) Applied:** `LANE_D_PROGRESS.md` and Appendix A are retired in
    favour of the §17 status board; the owner inputs moved to §17.4;
    `LANE_D_S1_RESULTS.md` is now a pointer to Appendix B (its copy still
    said the forward re-check "happens at the S6a shadow").
22. **(rev6) §15 rows with no recommendation.** D25, D26, D27, D29 and D31
    give Option A (keep v2) and Option B, with no recommended option, so "all
    other recs" doesn't strictly settle them. §17.4 records them as **decided:
    Option A (keep v2, the design the WPs build)**, so they don't block W8 or
    W5b. Proposed: the owner confirms Option A, or names the Option B rows to
    adopt; any Option B is a structural change to W8 (court pay) and goes in
    as a new WP row.
23. **(rev6.1) Code-enforced stake gate for activation.** Today only the
    runbook enforces "observed `cfg-` stake ≥ 2/3 before `effective_block`"
    (§5 row 11, X6a, D10). `ALLOWED["lane_d"] = None` lets an overlay
    adopted at the 51% permit-stake lock-in set `[lane_d] from_block`, and
    old-release nodes refuse that overlay whole (§14.1), so arming early
    splits the fleet (RV1 note 10). Proposed: R3 (lint) or R7 (publish
    wrapper) refuses to publish an overlay that sets `[lane_d] from_block`
    unless R4's tracker output shows ≥ 2/3 observed `cfg-` stake.
    Affects R3, R4, R7, §5 S6.
24. **(rev6.1) `[lane_c] remainder` restatement.** It is now release-only
    (W2, `netconfig.py:213-215`). An overlay that merely restates
    `[lane_c] remainder = "burn"` would be refused whole by new nodes and
    accepted by old ones in a mixed fleet (RV1 note 12). R3 already refuses
    it; proposed: say so in §14.2 and in the operator runbook, so no
    overlay body ever carries it. Affects §14.2, R3, `docs/OPERATOR.md`.
25. **(rev6.1) Lane A tuning after lane D activation.** R3's lint, as its
    row is written, applies to any overlay containing a `[lane_d]` table,
    and overlays are full snapshots. After S6 every overlay carries
    `[lane_d]`, so any lane A key change is refused: lane A can no longer be
    tuned by overlay. Options: (a) accept it (lane A changes go through a
    release); (b) allow lane A keys to change in a seq that leaves every
    `[lane_d]` key byte-identical to seq N-1. Affects R3, §5, §14.2.
26. **(rev6.1) Stake-and-satellite gate for arming the §16 sentinel
    (W-S1).** `ALLOWED["v2"]` now admits the four `sentinel_*` keys
    (`netconfig.py:111-112`). A pre-W-S1 node refuses an overlay that
    carries them, and refuses it whole, so arming by overlay before the
    fleet upgrades splits the fleet. A pre-W-S1 satellite still verifies an
    armed receipt's signature (it uses the received body, `receipt.py:1220`),
    but its replay re-runs the gate on all four suites, the detail
    mismatches, and every armed settlement with a gate or king check fails
    replay there (RV3 note 2). Proposed: the R7 publish wrapper refuses an
    overlay that sets `[v2] sentinel_from_block` (non-zero) unless R4's
    tracker output shows observed `cfg-` stake ≥ 2/3 **and** every known
    satellite reports the W-S1 release; §5 states the same gate on the
    sentinel arming step. Affects R7 (publish wrapper), §5, §16.

## 0. Summary

- **Goal.** Find out how far our fixed architecture can go when training is
  unconstrained.
  - Miners train **anywhere, on anything, with any recipe or compute**: from
    scratch, from a public cascade checkpoint, by distillation.
  - They submit **only weights**. We **only evaluate**.
- **Winning.** A challenger beats the current lane D king on **forward-only
  private windows**: forecast targets that first appeared in our data after
  both models were committed. Nothing either side trained on can contain them.
- **The house.** Our own lane A king is always entered (the "house"). Lane D
  pays only for weights better than our pipeline produces.
- **Direction check.** A rotating public-benchmark gate runs, plus one
  never-gated benchmark, so we can see whether we're heading the right way.
- **Context.** Miners may run their weights at up to **128k context**, a
  scorer-side setting (§1.1).
- **Later phases.** Lane E (test-time optimisation) and the lane A recipe menu
  are phases 2 and 3 (§7, §8). **(rev4)** Lane B ships as part of this plan **trainer-run with miner-chosen experts and suite targets** (§12b: WB10–WB21, TB10–TB22, track SB0–SB5). The miner-weights variant (§12) is parked.

| Lane | Miner submits | Who trains | Judged on | Rollout |
|---|---|---|---|---|
| A (live) | data generator | us (fixed recipe) | private pool + bench gate | live |
| B (built, off) | **(rev4)** code-only generator + optional declared expert rows + optional one public-suite target | us (trainer-run) | per-target gain + guards + anchor + §16 sentinel; contamination-gated | **(rev4)** lane B track SB0–SB5 (§5), parallel to lane D S2–S3 |
| C (built, off) | fine-tune recipe | us | per-target eval | off |
| **D (this plan)** | **weights, our architecture** | **the miner, anywhere** | **forward-only private windows vs KING_D** | **gated stages S0–S7 (§5)** |
| E (phase 2) | test-time procedure | none | private pool vs KING_A plain | after D |
| A menu (phase 3) | generator + declared recipe | us | lane A duel | after E |

This supersedes DEC-CA-0062's recipe design, in which we trained a declared
recipe with an Ephemeris teacher. A lane D miner who wants distillation calls
Ephemeris, or anything else, on their own account. Recorded as **DEC-CA-0070**.

---

## 1. Lane D design

### 1.1 What an entry is, and how context works

**Checkpoint = a standard cascade_pyramid checkpoint, unchanged format.**

- `model.py` and `forecast_wrapper.py` must be byte-identical to the repo's
  `pyramid_model.py` and `pyramid_wrapper.py`. This is checked by
  `verify_checkpoint_code`, `cascade/eval/checkpoint_guard.py:139-171`.
- `config.json` must equal the contract's `from_contract().to_dict()`
  (`:290-302`). Its `context_length` is **4096**, because
  `CascadeMoEConfig.from_contract` requires it (`pyramid_model.py:89-93`).
- `weights.safetensors` must be **BF16 only** (D14, rev3): about 2.5 GB.
  - It's scored in lane A's eval precision: upcast to fp32 at load, the
    same for every entry.
  - F32's 4.9 GB sits close to the 5 GiB presigned single-PUT limit, which
    is W3a's fallback upload path.
  - Today's guard requires the pinned F32 spec
    (`checkpoint_guard.py:323-328`), so W4 adds a lane-D-only dtype path and
    leaves lane A's guard unchanged.
- **No other file is allowed.** Today's guard only refuses extra `.py` files
  (`checkpoint_guard.py:151-153`), so W4 adds an exact-filename allowlist that
  also refuses `.pt`, `.bin` and pickles.
- NaN and Inf weights are refused by `verify_weights_finite` (`:175-196`).
- Nothing here changes `pyramid_model.py`, `pyramid_wrapper.py`,
  `base_arch_digest` or any lane A contract term.

**Context is a scorer-side setting, not a checkpoint property.**

- At `ready` the miner declares a context from `[lane_d] contexts`.
  **Launch arms `{4096, 8192, 16384}` (rev3).** 32768, 65536 and 131072 are
  armed later by overlay at stage S8 (§5), through the same `contexts` key
  (D10). They're never a typed block.
- **Architecture facts (rev3, verified in `pyramid_model.py`):**
  - `patch_size = 32` (`:32`). **(rev5) Training rows are 128 patches**
    (4096 points): `pyramid_trainer.py:360-362` passes `max_ctx_patches=128`
    to the batch source. `max_patches = 136` (`pyramid_model.py:36`) is not
    the training length. Training never reads it; it's read only by lane B's
    horizon-affinity decode (`expert_lane.py:365`). So **8k = 2×, 16k = 4×,
    32k = 8×, 128k = 32× the trained length**.
  - **(rev5) Masked span limit.** CPM masks contiguous spans of ≤ 16 patches
    (`cpm_c_max = 16`; `pyramid_trainer.py:402`, the 120m preset and
    chain.toml). **Horizons > 512 points (16 × 32) are outside anything the
    model was trained to forecast.** That bounds every future lane D horizon
    (below).
  - `_xpos` uses `scale_base = 512.0` (`:154`) and a fixed, non-persistent
    `zeta = (idx + 0.4) / 1.4` buffer (`:211`). The q·k scale between
    patches d apart is `zeta^(d/512)`.
  - For the lowest-zeta dimensions (zeta ≈ 0.286) that is about **4.4e-5
    at d = 4096 patches**.
  - 128k context is 4096 patches, **(rev5) 32× the trained 128-patch rows**. The
    long-range signal in those dimensions is almost entirely suppressed, so
    quality beyond 16k is unproven and is left to S8's T7 measurements.
- A new lane D scorer module (`cascade/eval/lane_d_scorer.py`, outside the
  pinned wrapper) builds `CascadeMoEConfig` directly with that context and
  loads the same `state_dict`.
- **Why the weights fit any context:** the positional buffers (`inv_freq`,
  `zeta`) are non-persistent and sized by `head_dim`, and the route bias is
  sized by E (`pyramid_model.py:210-211, 259`). W0b pins this with a CI test:
  a 4096 checkpoint loads `strict=True` into every context variant.
- **No padding.** The pinned wrapper edge-pads short histories to the full
  context with `row[0]` (`pyramid_wrapper.py:35-38`). The lane D scorer
  instead **truncates each window's history to the largest patch multiple it
  really has** (≤ the declared context) and batches windows of equal length.
  - A long-context model gains only where long history exists.
  - Cost grows with real history.
  - A 4k model and a declared-128k copy of it score identically on windows
    with ≤ 4096 history (T7).
- **Precision:** scored in fp32, as the existing wrapper does
  (`pyramid_wrapper.py:51`). The context risk T7 measures is **extrapolation
  quality and memory**, not numeric overflow.
- **Decode cost (rev3, verified).**
  - The pyramid wrapper appends `ceil(h/32)` masked patches and runs **one
    forward pass** (`pyramid_wrapper.py:41-58`).
  - No KV cache exists anywhere in `cascade/`.
  - `STABLE_DECODE_STEPS = 768` is the **toto2** wrapper's constant
    (`toto2_trainer.py:1367`), not the pyramid's.
  - At h = 64 a window is therefore one pass over `context/32 + 2` patches.
    Cost is **roughly quadratic in context** for that single pass (time
    attention), and there's no per-horizon-step multiplier.
- **The comparison:** both king and challenger are scored by the lane D
  scorer, each at its declared context, on identical forecast targets. There
  is **one** lane D board across all contexts.

**Horizon:** lane D scores **h = 64 only** at launch (A3: already in v2; rev3 makes §1.7 and the floors consistent with it). Lane A's 64/256/720
ladder would push forward-only latency on hourly data to about 30 days at 720
(§1.3). Longer rungs are a later typed change. **(rev5) Hard limit on that
change:** any lane D horizon must be ≤ 512 points (CPM `cpm_c_max = 16`
patches × 32). Longer horizons are outside training, so 720 is refused
unless the training contract changes first.

### 1.2 Identity, submission, storage

**Hotkey and commitment**

- A hotkey holds one commitment at a time. A lane D commit would replace that
  hotkey's lane A pointer, the same rule as lanes B and C
  (`lane_c_manifest.py:199-203`). **Lane D uses a dedicated hotkey.** It never
  consumes a lane A lifetime submission.
- Lane D reads reveals with `poll_commitments(include_history=True)`
  (`chain.py:495-535`).
- The commit grammar is `metro-v1:weights:c<context>:hippius:<vault-ref>`.

**Ownership and dedup.** All of these run before any evaluation spend.

- **Uploader must be the committer** (the `DigestOwned` rule,
  `cascade/funding/store.py:61, 340-367`). Another hotkey committing your digest
  is dropped.
- **Exact-digest dedup.** Refuse any entry whose weights sha256 equals:
  - a published checkpoint (every lane A leg or king on the registry, every
    lane D king),
  - the current house,
  - any open entry.

  This closes the front-run in which a freshly crowned, published lane A king
  is resubmitted into lane D (§2 item 1).
- **Near-copies** are handled by the margin. A near-copy of a king scores
  about the king, LCB ≈ 0, below `d_margin`.

**Rate limits (sybil and griefing)**

**(rev3) Final rule set, counted per lane: lane D has its own keys.**
Precedents are `[marketcast] max_open_legs_per_coldkey`
(`cascade/shared/config/marketcast.py:315, 671`) and
`[round] ops_max_active_per_coldkey` (`cascade/shared/config/rounds.py:455`).

"Open" means ready and not yet judged; an open entry can span eras. "Per
settlement" is a separate, judging-time rule.

| Rule | Key | Default | Status |
|---|---|---|---|
| One open entry per hotkey | (structural) | 1 | kept from v2 |
| Open entries per coldkey | `max_open_per_coldkey` | 2 | kept; lane D's own key |
| Judged per coldkey per settlement | `max_judged_per_coldkey_per_settlement` | 2 | **new (rev3)** |
| Judged per settlement (all) | `max_entries_per_settlement` | 8 | kept |
| Per-era count | — | — | none existed in v2; none added (the per-settlement key replaces the idea) |

- A `pending` entry holds its open slot.
- Exact-dup refusal is free (no eval).
- k for §1.3 = 32 per era.
- **Coldkeys are free (§9):** these caps limit casual flooding, not a
  determined Sybil. Registration cost and an eval fee (D4) are the remaining
  levers.

**(rev3) Enforcement**

- **The intake** `/v1/lane-d/ready` refuses over-cap requests with a typed
  code (`coldkey_cap_open`). This is **advisory**.
- **The validator's queue selection is the authority.**
  - It walks ready entries FIFO by T.
  - An entry over `max_judged_per_coldkey_per_settlement` or the global cap
    is **skipped with `coldkey_cap_settlement`**, carried to the next
    settlement, and not charged.
  - The judged set and every skip reason are structured report fields.
    `check_lane_d` re-derives the judged set.
- **Visibility:** per-entry status in `status/lane_d.json`, and a LANE_D.md
  FAQ entry. Test row T6b.

**(rev3) Coldkey resolution: [DECIDE D22]**

- **What satellites can query (verified).** `chain.py` can read the
  metagraph at a past block (`_metagraph(block=…)` `:361-371`,
  `uid_for_hotkey_at_block` `:454`). But past state needs an archive node,
  and the activation design deliberately needs none
  (`cascade/shared/activation.py:27, 344`). Public endpoints prune: a
  past-block read was "unreadable at entry (pruned; skipped per #304)" about
  2 h after its boundary (**(rev5)** OPSLOG entry "2026-09-23 11:54–12:10 UTC — ROUND 7384490546013170946 SCORED"; the old line ref `:2329` drifted to `:2338`, because the log is newest-first). So satellites **cannot** be assumed
  to read the metagraph at T.
- **Option a:** resolve the coldkey at T via archive state. Every validator
  then needs archive access.
- **Option b, recommended:** record the coldkey early and carry it in the
  signed report.
  - The intake records it at `/v1/lane-d/ready` time. That's earlier than T,
    and always inside the unpruned window.
  - The validator re-checks it at reveal, still inside the window. A
    mismatch is a typed refusal, `coldkey_changed`.
  - Satellites and the audit replay the signed value, at the same trust
    level as the rows. An archive check is optional for the audit.
  - Reasons: it works on pruned nodes, satellites need no new capability,
    and it follows §14.3 (satellites verify and do arithmetic only).
- **An unresolvable coldkey is never a silent pending.**
  - The entry gets the typed reason `coldkey_unresolved` in state and
    report, plus a cascade-ops alert (`lane_d_coldkey_unresolved`).
  - After `max_coldkey_unresolved_settlements` (recommended 4) it closes
    unjudged (`closed_unjudged: coldkey_unresolved`), and that's logged.

**Storage: the vault for lane D is new.**

- Today's vault is a **local 0700 directory on the box**
  (`store.py:258-275`), not a bucket, and can't hold 2.5 GB entries (BF16, rev3). Lane D
  adds a **dedicated private bucket** `cascade-lane-d-vault`. It is never the
  shared manifest bucket: private code once leaked through a shared-bucket
  archive tar (§13).
- **W3a prerequisite: what Hippius S3 offers** (docs checked 2026-11-21:
  docs.hippius.com/storage/s3/compatibility and /use/console/s3; live probe
  still needed):
  - **No STS and no IAM.** Short-lived per-entry credentials don't exist.
  - **Sub tokens** are bucket-scoped (not prefix-scoped), read-only or
    read-write, with custom lifespan and immediate revocation. They're created
    in the console, so they aren't per-entry. They suit the
    `pool_d` builder. The eval pod gets **presigned GETs only**, never a
    credential.
  - **Presigned GET and PUT are supported** (max 7 days). Presigned
    **UploadPart is not documented**. Multipart itself is supported (≤ 512 MiB
    per part, ≤ 10,000 parts).
  - ~~**Object Lock COMPLIANCE is supported.** Lock headers must go on
    CreateMultipartUpload; they're not supported on Complete.~~
    **(rev6.4, owner finding 2026-11-23) Hippius has no Object Lock in
    practice:** `GetObjectLockConfiguration` and `GetBucketVersioning` return
    403 for a full-access bucket-scoped sub-token and for the main key, and
    the console offers no lock permission. W3a re-runs its probe with
    `--no-object-lock` (lock probes recorded as `skipped`).
  - **Lifecycle rules are accepted but not applied.** TTL deletion must be our
    own job (D16).
  - **Bucket policies** use AWS policy language, but with no IAM it's unclear
    whether a policy can scope by prefix. The probe tests it.
  - **Consequence.** The miner path is **presigned PUT per part** if the probe
    shows presigned UploadPart works. Otherwise the fallbacks are a single
    presigned PUT (BF16 weights at ~2.5 GB, if the single-PUT size limit
    allows) or a small upload relay off the box.
  - **(rev5) Immutability: lock only kings.** COMPLIANCE retention is fixed
    when the object is created (the lock headers go on
    CreateMultipartUpload) and **can't be shortened**, so locking every entry
    would block loser deletion (`loser_ttl_eras`, D16). **Pending entries**
    rely on content addressing (the path includes the manifest digest) plus
    the ingest digest re-check, so an overwrite fails `digest_mismatch`. At
    **crowning**, the king is server-side copied into a locked prefix. W3a
    also probes `UploadPartCopy` for ~2.5 GB objects, because a single
    CopyObject may cap below that.
    **(rev6.4) Superseded for kings: there is no Object Lock**, so lane D
    kings are **not** locked by a COMPLIANCE copy at crowning. A king's
    immutability rests on the content-addressed path (manifest digest in the
    object path), the ingest re-hash on every read (an overwrite fails
    `digest_mismatch`), the signed report's digest, and the R2/HF mirrors
    (§11.1). Pending entries are unchanged (content addressing + digest
    re-check).
  - If prefix-scoped, short-lived credentials exist, use them.
  - Otherwise use **presigned multipart PUT URLs** minted per part with short
    expiry. Presigned URLs can't be revoked before expiry, so integrity comes
    from content addressing instead: the object path includes the manifest
    digest, ingest re-hashes, and any later overwrite fails `digest_mismatch`
    (the miner's fault).
- **Credentials and URLs are never logged, printed or put on a command line,**
  and never placed on pods. Pods get presigned GETs only.

**Flow** (a new **`cascade/lane_d/` service**, `cascade-laned-intake`):

- It reuses the `/v1/submit` identity, signature, registration and quota
  *rules* by import of the shared signing helpers.
- It does **not** live under `cascade/funding/`. `deploy_gate.py:55` maps
  `cascade/funding/` to the **trainer** service, so touching it would force a
  trainer restart.

1. **`POST /v1/lane-d/submit`**
   - Inputs: signed headers, `X-Content-Digest` (manifest sha256) and
     `X-Context-Length`.
   - Returns upload grants for `lane-d/<hotkey>/<digest>/`.
2. The miner's CLI does a **direct multipart upload**. Nothing transits the
   box (130 MB Caddy cap, 7 GB RAM).
3. **`POST /v1/lane-d/ready`**, signed.
   - Checks the object list and sizes against the manifest. It never
     downloads the weights.
   - Records `ready_block` and returns `commit_payload`.
4. **The miner commits** through a timelocked reveal. The **reveal block is
   T**, the entry's commit time. Weights are immutable from here.
   - **(rev5, owner-adopted, D43) Immediate reveal.** The lane D CLI sets
     `blocks_until_reveal = min_blocks` (reveal now). It does **not** use the
     lane A timing, which lands the reveal at `next boundary −
     reveal_margin_blocks` (25) (`shared/chain.py:89-100`) to stop copiers of
     public generator pointers.
   - **Why that's safe for lane D:** the commitment reveals only the vault
     ref and digest; the weights stay private in the vault; exact-digest dedup
     refuses resubmissions; and `DigestOwned` drops a digest committed by a
     hotkey other than its uploader.
   - **Effect:** T, and with it the forward-data clock (§1.3), starts up to
     one settlement (≈ 3 h, 900 blocks) earlier.
   - The validator doesn't depend on the timing: it takes T from the observed
     reveal block either way. A miner who uses a delayed reveal only delays
     their own clock.
5. **Prefetch.** On reveal, **(rev5)** the launch eval host (D11: valpriv-4090-class pinned-torch pod) prefetches the entry into a
   sha-keyed, resumable cache, using presigned GETs. Hippius outages and slow
   transfers are the history (§13).

**Visibility and retention**

- **Losers** stay private:
  - removed from the vault `loser_ttl_eras` (recommended 4) after judgement;
  - then held in an **owner-only archive** for the dispute window
    (`retention_days`, 14);
  - then deleted.

  Automated deletion is an Approve-tier destructive operation, so it needs
  the owner's **standing approval (D16)**.
- **Kings** are published (§11.1).

### 1.3 Judging: forward-only windows

**The rule.** For an entry E with reveal time T_E, against king K with
commit time T_K:

> **(rev3)** A window is eligible only if, for **every** target point,
> **both** its **data timestamp** **and** its **first-seen time** (when our
> mirror first ingested it) are after **wall(max(T_E, T_K)) +
> `forward_margin`**. Its feed's measured mirror lag must also be within
> `[lane_d] max_mirror_lag_hours`; a feed above the limit is **never
> eligible**. A series without both timestamps is never eligible.

- **(rev3) First-seen doesn't exist today.** Verified: `grep -rnE
  "first_seen|ingested_at|ingest_time|landed_at|LastModified" cascade/pool/`
  finds nothing. Recording first-seen is the **launch-blocking WP W6fs**
  (§3).
- **(rev3) Both clocks, because each one alone admits pre-T data:**
  - data time alone admits values that were public before T but carry a
    future timestamp (publish-ahead feeds, day-ahead prices, schedules,
    forecast products such as `openmeteo`);
  - first-seen alone admits values that were public before T but reached our
    mirror late (outages, backfills, series absent from a snapshot).
- **(rev3) The v2 rule "absent from the first snapshot after max(T) + margin"
  is UNSAFE as a substitute.** It's a first-seen proxy only. It counts a
  series missing from that snapshot as "absent", so pre-T points pass. It
  must not be used for verdicts. It survives only as the **upper bound** B2
  reports ("snapshot-diff bound").

Why it's built this way:

- **Pool records carry no timestamps today.** Pool `metadata.json` holds
  `{freq, seasonal_period, domain, source, …}` (`builder.py:279-290`). The
  forge source parses timestamps and then discards them
  (`sources/tsbench_forge.py:380-399`). Some feeds have none.
- So **W6 stamps per-series `end_ts` into `pool_d` metadata**, and a series
  without timestamps is never eligible. **(rev6.1)** Likewise a series
  without `frontier`, or a feed without a `feeds` entry, is never eligible:
  W6b must write both, and W9 refuses a snapshot that lacks them. There is no build-time fallback: a
  series' tail can be up to `max_stale_days = 4` old, and
  `_freshest_segment` can fall back to an old segment (`:149, :204,
  :446-460`).
- **(rev3) Superseded by the two-clock rule above.** The v2 claim that
  publish-ahead feeds are "excluded automatically" was false: the forge
  reader drops rows with `ts ≥ as_of + 1 day` (`tsbench_forge.py:393-396`),
  so those values never appeared in the snapshot. `forward_margin`
  (recommended 6 h) is now applied to **both** clocks.
  `max_mirror_lag_hours` (recommended 6) bounds how far our mirror may trail
  the public source per feed. It's measured by W6fs from the source's
  publish stamp, where the feed provides one, against the first-seen time.
- **Both T values count.** max(T_E, T_K) matters: a king committed after a
  pending challenger could have trained on that challenger's targets.
- **The cost is about one extra day of latency.** W1 measures it.

**Window identity.** Each window has a stable id
`<series_id>:<target_start_ts>:h64`. Lane A's `h<h>-s<pool_index>` ids aren't
stable across snapshots (`windows.py:433-449`).

**Evidence accumulates, then freezes.**

- An entry's eligible set is the union over every daily `pool_d` snapshot
  since T.
- It **freezes** at the first settlement where it meets the floors:
  `min_windows` (1000) and `min_clusters` (20 sources).
- It is **judged once**, on its cohort's intersection set (§1.3 Cohorts), which is ⊆ its frozen set. A loser is never re-judged on new
  windows: rotation alone swung one pair's LCB by 0.62 (§13).
- Until then the entry is `pending`.

**Cohorts (several entries in one settlement).** `cohort_maxt_lcbs` assumes
every challenger shares one window draw and the king's scores
(`bootstrap.py:393-418`). So:

1. The cohort is every entry that freezes this settlement.
2. The window set is the intersection eligible for max(T over cohort ∪ king).
3. If the intersection misses the floors, the latest-T entries are deferred to
   the next settlement until it meets them.
4. The cohort is then judged with max-T on that shared set.

**Verdict.**

- The statistic is a paired cluster bootstrap of the CRPS/MASE geomean ratio
  (`paired_bootstrap_lcb_aggregated`, `bootstrap.py:225`), clusters = source.
- The challenger wins if `LCB_α ≥ d_margin` and the point gain is strictly
  positive.
- `d_margin` is **lane D's own level-unit bar** (recommended 0.01). It is not
  lane A's v2 increment bar, which runs 0.10 → 0.01 with tenure decay.
- **(rev3) Mandatory confirmation duel.** A candidate that clears must
  **clear again on the next disjoint eligible window set** (the next freeze
  of new forward windows, sharing no window with the first) before it's
  crowned.
  - It costs one more freeze period of latency: **about 1–4 days depending
    on data frequency** (5-min-heavy sets at the low end, hourly at the high
    end; B2 refines it).
  - While confirming, the entry's state is `confirm_pending`.
- **[DECIDE D20] (rev3): a king change between first clear and
  confirmation** (house refresh or KING_D change):
  - (a) **restart the duel against the new king** (first clear discarded;
    the entry re-freezes against the new king; **recommended**);
  - (b) confirm against the king that held at the first clear.

  The chosen rule is the `[lane_d] confirm_on_king_change` key (`"restart"`
  default | `"original"`). It's written into `lane_d_state`
  (`confirm.king_digest`, `confirm.rule`), the report schema (`confirmation`
  block) and tests (T9b).
- **(rev3) Repeated attempts.**
  - Cohort max-T corrects only **within one settlement**. Near-copies on
    separate hotkeys get independent window sets, so an adversary gets **k
    independent attempts per era**.
  - The per-era false-crown probability ≈ **1 − (1 − p_conf)^k**, where
    p_conf is the confirmation-duel null FPR. This assumes independent
    attempts; it's an approximation, conservative when attempts share
    windows.
  - **k at the A8 caps:** coldkeys are free, so the binding cap is global:
    `max_entries_per_settlement` (8) × `era_settlements` (4) = **k = 32
    judged attempts per era**. A single coldkey gets at most
    `max_judged_per_coldkey_per_settlement` (2) × 4 = 8 per era, and fewer in
    practice: `max_open_per_coldkey` and the freeze latency bound it.
- **(rev3) `d_margin` is calibrated to the per-era target, not a per-entry
  quantile.** Choose the margin so that 1 − (1 − p_conf)^k ≤ the owner's
  per-era target (OA3; recommended 5%).
  - A per-entry null p99 alone (p ≈ 1%) would give about 27% false crowns
    per era at k = 32, which is too loose.
  - T14 is redefined (§4): the null FPR per entry for each null-pair type,
    plus the per-era FPR at the caps. It's measured at S1 (B1,
    `scripts/null_fpr.py`) and is an **S4 gate** against the owner's per-era
    target.
  - B1 reports the margin needed for per-era targets of 1%, 5% and 10%. The
    `d_margin` default stays 0.01 until B1 runs.

### 1.3b Rotating external-benchmark gate (direction check)

- **Gate suites:** **(rev5)** the four suites (GIFT-Eval, BOOM, TIME, FEV)
  **minus this era's §16 sentinel**, run with the existing runners
  (`cascade/eval/benchmarks.py`, `public_guards.py`). TIME is no longer a
  fixed held-out suite.
- **The draw:** one suite per crowning, drawn from the **settlement boundary's
  block hash**. Every validator agrees on it, and no miner can know it at
  commit. Weights start equal.
- **Rule on the drawn suite.** It runs paired on one host in one sweep, never
  against stored absolute numbers, because driver and SKU shift absolute
  scores. CRPS and MASE must each satisfy `LCB_α ≥ −tau_public` (0.01) against
  both:
  1. the current KING_D;
  2. **the current house**: our own, uncontaminated pipeline.

  There's no "best-ever miner king" ratchet. A contaminated king would set an
  inflated best-ever and block honest successors. The house is the anchor.
- **(rev3) Held-out suite = the cross-lane rotating sentinel (§16).** The
  v2 fixed "TIME unselected" is replaced.
  - Each era one suite is the held-out **sentinel** for all lanes (A, B and
    D).
  - Lane D's drawn gate suite comes from the remaining suites.
  - The sentinel is swept for each new KING_D and published only.
  - If the sentinel falls for 3 kings in a row, that's an alert and an owner
    review.
- **Faults:** a candidate whose sweep errors, returns partial, NaN or a
  missing metric is a **`checkpoint_fault`, the miner's fault** (T17).
  - Suite identity is verified per drawn suite. Scope-skips are not
    "partial" (§13).
  - The house and KING_D are re-swept in the same host session as the candidate (cached rows are re-used only within that session).
- **(rev3) Always-run low-frequency check (A10).** Alongside the drawn
  suite, every crowning candidate also runs a **paired non-inferiority
  check on a low-frequency slice of the private pool**.
  - The slice: `[lane_d] lowfreq_freqs`, recommended daily and slower.
    Forward-only rarely reaches these series (§1.7).
  - The test: CRPS and MASE `LCB_α ≥ −tau_public` against both KING_D and
    the house, on the same window set, scored by the lane D scorer.
  - On failure: `lowfreq_regression` (typed), and the candidate isn't
    crowned.
  - **Limit (§9):** the private pool isn't forward-only, so a contaminated
    model passes this check more easily. It **catches honest regressions
    only**.
- **Pay:** public suites never pay and are never targets.
- **Progress board:** every king's suite numbers (gate and **(rev5)** the era's sentinel, suite named) go into the
  signed lane D index (§11.1) and `status/lane_d.json`.

### 1.4 Kings, the house, pay

**KING_D**

- There's one KING_D at a time.
- The court is KING_D plus `reward_prior_kings` (2) prior **miner** kings,
  decayed by `king_decay` (0.5) inside lane D's share.
- The house is never in the court.

**The house (our lane A king)**

- **Identity:** the house is the checkpoint digest lane A currently treats as
  king. Under v2 that is the epoch's king leg (DEC-CA-0055).
- **Refresh:** it's refreshed whenever that digest changes (a crowning, a
  seed-epoch rollover, a promotion). It enters at that settlement, and its
  T is the block at which that digest was published.
- **Forfeit or king-check fail:** the house is whatever lane A treats as king.
  If lane A has none, the previous house stays.
- **Pay:** if the house holds KING_D, the lane D king slot goes unpaid.
  Prior miner kings keep their decaying tails, because those were real
  crowns. Where the unpaid slot goes is **[DECIDE D21] (rev3)**, a
  tokenomics call, not adopted here:

  | | Option 1: **burn** (v2 default) | Option 2: **return to lane A** while the house reigns |
  |---|---|---|
  | For | Matches §14.2, where `remainder` is a release-only burn destination. Lane A's share is fixed, so invariant 2 and satellite weight derivation are unchanged (**(rev6)** the share-0 shadow no longer exists, D45) | The house is lane A miners' work, so lane A's contributors are paid for it |
  | Against | Emission withheld whenever our pipeline is the best | §14.2 must change (a new release-only `house_remainder` destination). Lane A's share varies per settlement, which touches invariant 2 and satellite derivation (W7d), and adds test rows (**(rev6)** no shadow stage, D45) |
  | Dethroned house | Never enters the court (the court is miner kings only). No tail | Same: no tail. The slot moves to the new miner king at once |

  Under **both** options the house is never a prior king, and prior miner
  kings' tails continue unchanged.

**Share [DECIDE D1]**

- **(rev5, owner)** `[lane_d] share = 0.25`, `takes_from = "lane_a"`, `remainder = "burn"`.
- The deployed file has `takes_from = "lane_b"`. It's changed **by the S6
  activation overlay** **(rev6;** S6b merged into S6, D45**)** (§5, §14.2), not by
  a chain.toml edit.
- When B and C also arm, the split is **(rev5, owner) A 25 / B 25 / C 25 / D 25**.
- Live emission is `lane_parts` + `lane_pool_shares` + `emission_vector`
  under the DEC-CA-0055 ladder (`cascade/shared/emission.py:330-420`). The
  legacy `split_emissions` helper is not used. W8 extends `emission_vector`
  with lane D.

### 1.5 Consensus, trust and state

**Who scores**

- **The main validator (ours)** orchestrates lane D:
  - **(rev5)** the launch eval host (D11: valpriv-4090-class pinned-torch pod) computes forecasts (valeval-laned only from S8);
  - **(rev5) Scoring method = lane A's quantile head.** The lane D scorer
    reads the model's quantile head directly at the 9-level scoring grid, with
    **no copula sampling and no Monte-Carlo noise**, as
    lane A does under `[v2] quantile_head_after = true` (DEC-CA-0067,
    `chain.toml:1247`, `ChainConfig.quantile_head_at`). It never uses the
    sample path. So lane D CRPS is computed the same way as the house's lane A
    CRPS, and a house-vs-entry comparison has no method gap;
  - **targets and scoring stay on the box** (§10.2);
  - the main validator signs the **lane D report**: frozen window ids, per-window
    CRPS/MASE rows for every scored model, the bootstrap seed, the cohort,
    the drawn suite, verdicts and `pool_d` sha256.
  - **(rev3) The report also carries:**
    - `rows_sha256` (A9);
    - per window, both clocks and the feed's mirror lag (A1);
    - the judged set and per-entry skip reasons, with `coldkey` as recorded
      at `ready` (A8);
    - the `confirmation` block: stage, first-clear set id, king at the first
      clear, rule (A2, D20);
    - the low-frequency check result (A10).
- **Satellites** hold no pool (DEC-CA-0058:17-33), and `pool_d` is private.
  They check the signature, the structural rules (declared timestamps
  against declared T values, floors, cohort construction), and they re-run
  the bootstrap from the signed rows, then set the same weights. They
  **cannot** re-derive targets.
- **The trust level is lower than lane A's bench numbers.** Lane A rows are
  reproducible from public checkpoints (`cascade-audit bench --reproduce`). A
  lane D loser's rows are not, because its weights stay private. Crowned
  kings are public, so their rows become re-scorable once a `pool_d` snapshot
  is revealed. **We accept this explicitly (D17).**
- **(rev3) Rows digest.** Every verdict publishes `rows_sha256`, the sha256
  of the exact scored per-window rows (canonical serialisation, both models,
  frozen set order). Anyone holding the rows can prove they're the ones the
  verdict used.
- **(rev3) Maximum reveal delay.** `[lane_d] max_reveal_delay_days`
  (recommended 14). An entry still pending at the limit closes unjudged
  (`closed_unjudged: reveal_delay_exceeded`), logged, and frees its slots.
- **(rev3) Loser re-scoring: [DECIDE D23].**
  - (i) Publish `pool_d_hist` after the dispute window, so losers can fully
    re-score their own weights against the published targets.
  - (ii) Soften the claim to "losers can verify targets, window ids,
    `rows_sha256` and the bootstrap; histories stay private". v2 keeps long
    history private on purpose (§1.7, lane A mimicry).

  The default is **(ii)**, matching v2. Choosing (i) also removes the
  private-history rationale in §1.7.
- **No free text in replay comparisons.** A transient fetch error is never
  read as "absent" (§13).

**State**

- A new `lane_d_state.json` holds KING_D, the court, the house digest and T,
  pending entries with their eligible-window progress, frozen sets, and a
  score cache keyed by (model digest, context, window id).
- **(rev3) It also holds, per entry:**
  - `coldkey_at_ready` and its re-check result (A8);
  - `ready_block`, `reveal_block` (T) and `reveal_deadline` (A9);
  - `confirm` (first-clear set id, `king_digest`, `rule` per D20) (A2);
  - `reason`, a typed enum: `pending_floor`, `confirm_pending`,
    `coldkey_cap_settlement`, `coldkey_unresolved`, `coldkey_changed`,
    `reveal_delay_exceeded`, `mirror_lag_exceeded`, `lowfreq_regression`,
    `checkpoint_fault`, `bench_hold`, **(rev5)** `deregistered` (B8), `ready_expired` (B10);
  - per feed, `measured_mirror_lag_hours` (A1).
- **(rev5) Deregistration (B8).** A pending, frozen or `confirm_pending` entry
  whose hotkey loses its UID is **voided**: typed `deregistered`, slot freed,
  never judged. A **court** member (KING_D or a prior king) that deregisters
  has its slot **burn** (precedent: lane B drops deregistered winners so the
  unpaid slot burns, `shared/lane_b.py:250-262`). It's checked at every
  settlement against the settlement's metagraph and recorded in the report.
- **(rev5) Ready → commit deadline (B10).** `[lane_d] ready_ttl_blocks`: an
  entry readied but not revealed within that many blocks of `ready_block`
  expires (typed `ready_expired`), its open slot is freed and its vault
  object becomes a loser for retention.
- Its digest is in the receipt's prior and posterior digests (DEC-CA-0058),
  and the report digest is bound into the receipt as the bench source is.
- `checks.derive_weights` and `check_lane_d` re-derive the lane D weights
  from the receipt chain.
- It's protected like `cascade_state.json`: snapshot before every restart,
  never hand-edited.

### 1.6 Capacity

| Resource | Load | Plan |
|---|---|---|
| **(rev3) Launch eval host:** valpriv-4090 or another pod running the pinned torch 2.4.1+cu124 venv | prefetch on reveal; forecasts for up to 8 entries + KING_D + house per settlement on frozen sets, at ≤ 16k context; one forward pass per window, cost roughly quadratic in context (§1.1) | **Not valeval-blackwell4**: sm_120 can't run the pinned torch (only the sidecar venv), so numerics would differ from testnet and the worker handshake fails. **S1 check (B2, OA4):** does a 16k-context batch fit in 24 GB on valpriv-4090? If not, name the smallest pinned-torch pod that fits. valeval-laned (96 GB) is rented **only at S8**. Measure in T7; arm only variants that fit ≤ 50% of a settlement |
| Box | target slicing, scoring and bootstrap for returned forecasts; report signing | Targets only: small, h=64. **No histories, weights or torch on the box.** Bound the report size |
| valeval-blackwell4 (43 GB, ~19 GB cache) | per crowning: **2.5 GB (BF16, rev3)** upload + 1 drawn suite; per new king: **(rev5)** the era's sentinel suite | Byte-based LRU (#467); lane A first; second bench pod if crownings slip > 1 settlement (D6) |
| Weight loop | must never wait on lane D | Lane D runs off the poll thread; keep-alive (#471) live before S6 (T15). **(rev3) Status:** #471 is merged on main (0c52d07; `_weights_keepalive_tick` at `cascade/validator/weights.py:317`, wired into `validator_bench.py:901`) but **not deployed**: the live rc37 validator has no `_weights_keepalive_tick`. It ships at the latest with S2b |

### 1.7 Increasing the context of the cascade eval pool

**How it works today:**

- `publish-pool.yml` (daily 03:00 UTC) truncates each series to its freshest
  `context_length + keep_horizon` points: 4096 + 720, from
  `[eval] context_length` and the longest scored horizon.
- The validator cuts **one window per series per horizon rung** (the last h
  steps as target, up to `[eval] context_length` before it as history;
  `windows.py:556-611`).
- The pool snapshot has about 4.6k series (snapshot 9509400: 4,633).

**Lane A must not change.**

- **`pool` is untouched.** Same builder inputs, same bytes, same receipts,
  same reveal.
- A longer keep can't simply be applied to `pool`. The builder truncates
  **before** its degenerate-series check and dedup (`builder.py:264-271`), so
  a longer keep changes which series pass, and therefore lane A's windows.

**`pool_d` (D12)**

- **(rev3) Launch `pool_d` is still a separate tar.** The main pool's keep
  length is **never** raised: `builder.py:264-271` truncates to
  `keep_length` before the `too_short` and degenerate checks (and dedup runs
  later still), so a longer keep would change which series pass and
  therefore lane A's windows. The lane A byte-identity test stays exactly as
  strict.
- **(rev3) Launch construction:** the **same selected series ids as `pool`**,
  joined (not re-selected), with history extended to **16384 + 64**. It
  carries the A1 timestamps (data time and first-seen per target point) and
  the hist/tgt split (D18).
- **(rev3) Moved to stage S8** (armed by overlay through `contexts`, D10):
  history to 131072 + 64, deep backfill, the new long sources (option 3
  below), and the 96 GB eval host. **(rev5) Only scoring and arming wait
  for S8. Harvesting long history starts now** (below, WP R16).

**(rev5) Harvest now, score later (up to 128k).**

- **Why now.** Sub-hourly history that isn't recorded is gone. The forge
  mirror prunes sub-hourly partitions (Getting the history, item 2), and many
  source APIs serve only recent data. 128k points at 5 min is about 1.2
  years. If recording starts at S8, the 32k–128k variants can't arm for
  months after S8, because `min_long_series` (300 from ≥ 20 sources) can't
  be met.
- **What starts now (before lane D launches, from stage S1/S2):**
  1. stop pruning sub-hourly forge-mirror partitions for every source that
     feeds `pool` (an owner/builder retention change, not a code path
     validators run);
  2. start recording the deep-history live sources (item 3 below);
  3. the `pool_d` builder (R13) appends each selected series' new points to a
     **long-history archive**, keeping up to **131072 + 64** points per series
     and channel. It uses the W6a join key, carries the per-point first-seen
     record (W6fs) once that exists, and lives in a private bucket,
     sha-pinned per day.
- **What does not change:** `pool` and lane A (byte-identity test as
  strict); nothing is scored, sent to an eval host or revealed from the
  archive before its context is armed at S8; nothing runs on the box.
- **Size bound (arithmetic, not measured):** 4,633 series × 131,136 points ×
  4 B (float32) is about **2.4 GB** for a full copy. Stored append-only, the
  daily delta is small. Most series never reach 128k (see the table below),
  so the real size is below this.
- **Readiness signal:** R16 reports, per context rung (32k/64k/128k), how many
  series and sources have reached it. S8 arms a rung only when
  `min_long_series` is met.
- **Metadata:** per-series (per-channel) `end_ts`, `first_target_ts` and the
  per-point first-seen record (W6fs). **(rev6.1)** Plus `frontier` for every
  series and a `feeds` entry (`mirror_lag_hours`, `forecast_product`) for
  every feed; a `pool_d` missing either is invalid (W6b).
- **Artifacts:** two tars from one build.
  - `pool_d_hist`: histories only, **no targets**. It goes to **(rev5)** the eval host (the launch eval host; valeval-laned from S8).
  - `pool_d_tgt`: targets plus timestamps, small. It stays on the box.

  This keeps the existing privacy rule: targets and scoring stay local
  (`eval_offload.py:7-10`).
- **"One longer pool" is ruled out on RAM.** The box validator loads every
  series as float64 (`validator/pool.py:71-77`). 1,000 series × 131k × 8 B is
  about **1 GB**, on a 7 GB box.

**Getting the history** (cheapest first):

1. **Extend what the forge mirror already has** for the selected series.
2. **Stop pruning sub-hourly mirror partitions,** and backfill full history
   from source APIs where allowed.
3. **Add deep-history live sources:** grid load and generation at 5–15 min,
   weather stations hourly, traffic at 5 min, market bars at 1–5 min. Each
   carries a domain and a source key.

**What 128k means in wall-clock history:**

| Frequency | 128k context | Usable? |
|---|---|---|
| 1 min | ~91 days | yes |
| 5 min | ~1.2 years | yes |
| 15 min | ~3.7 years | yes, with backfill |
| hourly | ~15 years | rare archives only |
| daily or slower | centuries | no: those series keep what they have |

A context variant is armed only if at least `min_long_series` (300, from
≥ 20 sources) can use it (W1).

**(rev3) Floor maths at h = 64.** These are consistent with A3. A window's
target is the last 64 steps, so for an entry committed at T the earliest
eligible window has its first target point after T + `forward_margin` by
both clocks. The confirmation duel (§1.3) needs a second, disjoint set, so
expect **two freeze periods** before crowning. B2 measures the counts.

**Forward latency at h = 64:**

- 5-min data: about 5.3 h of target span, plus up to a day for the next
  snapshot, plus 6 h margin ⇒ **about 1–1.5 days**.
- Hourly: about 2.7 days of span, plus a day ⇒ **about 4 days**.
- W1 measures the real distribution. More sub-hourly sources shorten it.
- **(rev3) With the mandatory confirmation duel**, add one more freeze
  period before crowning. That's **about 2–3 days (5-min-heavy) to about 8
  days (hourly-heavy)** from commit to crown. These latencies are the
  target-span arithmetic only, not measurements; B2 replaces them.
- **(rev3) The 6 h margin applies to both clocks (A1).** Daily-or-slower
  series almost never qualify, which is why A10's low-frequency check uses
  the private pool instead.

**Build host.**

- GitHub Actions `ubuntu-latest` has a 180-minute wall and a 10 GB cache
  quota (`publish-pool.yml:72-79, 121-131`). That's too small for multi-year
  backfill.
- **`pool_d` builds on a dedicated CPU builder**, never the box. It has its
  own disk, wall, retry and size budget, and the `TSFORGE_S3` read
  credentials live there only.
- The daily `pool_d_*` snapshots are pinned on Hippius with sha256 in every
  lane D report.

**Reveal.** `pool_d` adds **no reveal obligation**:

- weights are fixed at `ready`;
- the same series' recent tails are revealed in `pool` at about 48 h anyway;
- long history is withheld so lane A generators can't mimic it.

After the dispute window, `pool_d_tgt` and the window ids are published, so
anyone can check a crowned king's targets afterwards.

---

## 2. How checkpoints interact across lanes

```
     miner generators
            │
            ▼
   LANE A (we train) ──► KING_A (Hippius, public) ◄── LANE B merges (era boundary)
            │                     │
            │                     ├──► miners may start lane D runs from it
            │                     │
            │       house refresh whenever KING_A's digest changes
            ▼                     ▼
   LANE D (miners train anywhere): private vault ─► reveal T ─► prefetch ─►
        forward-only duel vs KING_D (house or miner) ─► gate ─► KING_D
            │
            └── no automatic flow back into lane A ("we just eval");
                an owner-gated promotion would be its own decision
```

1. **A → D:** the house enters whenever KING_A's digest changes (§1.4).
   Exact-digest dedup refuses resubmitted published checkpoints.
2. **The house can retake KING_D** when a better lane A king beats it on
   forward windows. The lane D king slot then burns.
3. **D → A: none.** KING_D never changes lane A's init, contract or receipts.
   Promoting a lane D king into lane A would be a separate decision, because
   lane D contamination would enter the lineage.
4. **B → D:** only via the house.
5. **C ↔ D:** none. **(rev3) Correction:** lane D does **not** cover lane C. Lane C is MarketCast on a pinned third-party Timestral-28M base (`[marketcast] base_ref`, DEC-CA-0064), not our architecture. The v2 text said lane C's fine-tune purpose is largely covered by lane D at
   zero training cost to us. Whether C stays on the roadmap is a later
   [DECIDE].
6. **E (phase 2)** targets KING_A. Whether it also applies to KING_D is a
   phase 2 decision.
7. **Contexts:** one board. Each model is scored at its declared context on
   the same targets. *(rev5: items 7 and 8 un-garbled.)*
8. **(rev4) Split of purpose.** **Lane B improves the lane A model on declared targets** (trainer-run, merged into KING_A, contamination-gated). **Lane D explores the architecture with unconstrained training** (miner-run, never merged, forward-only judged). That's why miner-trained weights are allowed in D and parked for B (§12).

---

## 3. Implementation (work packages)

The plan is built by an **autonomous loop**. A WP is done **only when its
Accept command exits 0**, never on a builder's word. Conventions:

- `scripts/laned/accept <WP> [--evidence <file>]` (built by W0c, then
  read-only to builders) runs, in order:
  1. `uv run --frozen pytest -q tests/accept/<wp>/`;
  2. the stub scan over the WP's declared paths (`tests/accept/<wp>/PATHS`);
  3. `make laned-e2e` (from W1.5 on);
  4. for external WPs, the evidence verifier.

  It exits non-zero on the first failure.
- **Runs in:**
  - `ci`: GitHub Actions; never the prod box.
  - `builder`: the `pool_d` builder VM.
  - `pod`: a rented GPU eval pod.
  - `testnet`: the netuid 259 stack (§10.3).
  - `owner`: the owner runs it.
- **External WPs** (live Hippius, a GPU, the builder or the chain) are
  **blocked** when the resource is unavailable. They are **never mocked**.
  Their Accept verifies evidence JSON written by the real run (endpoint,
  request ids, object digests, block numbers), and the evidence verifier
  cross-checks against the live resource where it can.
- **Owner steps** (remote writes, spend, merges, publishes, announcements):
  the loop prepares the script and artefacts and runs their dry-run checks;
  the owner executes. Accept checks the recorded result.
- **(rev6) Roles.** Each is a fresh-context agent; one agent never holds two
  roles on the same WP.
  - **Test author:** writes only `tests/accept/<wp>/` and its `PATHS`, from the
    WP row, its T-rows and the invariants it owns, copied from this plan. The
    tests must fail on main for the right reason (§17.1 L1). Opens a
    tests-only PR labelled `laned:<WP>` **(rev6.2:** or one tests-only PR
    batching several WPs, one `laned:<WP>` label per WP, per §3.1 rule 6**)**.
  - **Builder:** starts from origin/main after L1 and opens ONE implementation
    PR labelled `laned:<WP>`, staged with explicit paths. **A builder never
    edits `tests/accept/`, this plan, `scripts/laned/` or any status file
    (`status/laned_build.json`, `docs/plans/STATUS.md`).**
  - **Preparer** (external and owner WPs): writes the scripts and runs their
    dry-runs only; the owner executes and the evidence verifier checks the
    result.
  - **Reviewer** (RVn): read-only on main; writes `evidence/rv<n>.md`.
  - Completion is never self-reported: only `scripts/laned/status` moves a
    level (§17.6).

### 3.0 Work packages

**Build-rule tooling and decisions (first)**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| W0c | Loop tooling: `scripts/laned/accept`; `stub_scan.py` (refuses `TODO`, `FIXME`, `pass`-only bodies, `NotImplementedError`, `...` bodies, hardcoded sample values and mock imports outside `tests/`); `unread_keys.py` (every `[lane_d]`/`[lane_d_ops]` key is read by a parser); `follower_patterns.py` (every lane D log call has a cascade-ops follower pattern). **(rev6) Also:** (i) `scripts/laned/status` per §17: computes every WP's level, the stage and milestone state, the decision register and the §17.3 counters from `git log origin/main`, GitHub Actions results on main and `evidence/`, and writes `docs/plans/STATUS.md` + `status/laned_build.json` to branch `laned/build-state` (never merged to main); `accept --gate <stage>` verifies a stage's exit-gate evidence (§17.2); (ii) a GitHub Actions workflow `.github/workflows/laned-accept.yml` that runs `scripts/laned/accept <WP>` on every PR labelled `laned:<WP>` and, on push to main, for every WP at ≥ L2. On a tests-only PR it runs in **expect-fail** mode: it passes only if the new tests are collected and fail for the right reason (an assertion failure, or a missing module/attribute named in the WP's `PATHS`), never on a syntax, fixture or import error elsewhere. Today `ci.yml` runs only `pytest tests/unit`, so nothing runs `tests/accept/` yet; (iii) the CI check that fails any PR that edits `tests/accept/` together with a path outside `tests/accept/` (rule 1), and any implementation PR touching a path not in its WP's `PATHS`; (iv) self-tests for (i)–(iii), listed in Accept. Creating the `laned:<WP>` label set is an owner step (`gh label create`); none exists today | — | ci | `scripts/laned/accept W0c` (self-tests: each script fails on a planted violation fixture and passes on a clean one; **(rev6)** `scripts/laned/status` on a fixture repo (fixture git history, CI results and evidence) yields the exact expected level for every fixture WP, including side states and a WP that regresses from L3 to L2; a planted rule-1 violation (tests + implementation in one PR) fails the workflow's check; an expect-fail run passes on correctly failing tests and fails on a syntax-error test file) |
| W0 | DEC-CA-0070 (supersedes DEC-CA-0062's recipe design) recording **(rev5)** D1–D43, D45–D48 and D42b (D44 withdrawn) (D25–D32 from §15, D33–D35 §16, D36–D42 §12b). **(rev6.1)** Owner-merged like every WP (§3.1 rule 6); that merge doesn't make W0 external | — | ci **(rev6.1)** | `scripts/laned/accept W0` (front matter valid; `supersedes: DEC-CA-0062`; **(rev5)** every decision D1–D43, D45–D48 and D42b answered or marked resolved) |
| W0b | Context-equivalence test: the lane D scorer at 4096 equals the pinned `pyramid_wrapper` output (within tolerance) on windows with ≥ 4096 history; a 4096 checkpoint loads `strict=True` at every context variant  **(rev5)** It compares **quantile-head** outputs at the 9-level grid (the lane A scoring path under `quantile_head_after`), not samples. | W5d | ci | `scripts/laned/accept W0b` |

**Rollout tooling (prerequisites for S0a–S0c, S4–S6)**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| R1 | Block-arithmetic tool `cascade-laned blocks` (era starts, activation and effective blocks from the live grid; cross-checked from two computations) | — | ci | `scripts/laned/accept R1` (golden vectors including past off-by-one cases) |
| R2 | Note-budget decoder (S0b): decode live readiness notes, simulate adding `cfg-`, report trimmed segments | — | ci (fixtures) | `scripts/laned/accept R2` (the 2026-11-13 crash-loop note fixture reproduces the refusal) |
| R2x | S0b live run: decode mainnet notes | R2 | owner (chain read) | `scripts/laned/accept R2x --evidence evidence/s0b_notes.json` (no pending segment trimmed) |
| R3 | Overlay lint: refuse any `[lane_d]` overlay that touches non-`lane_d` keys (except a byte-identical restatement of seq N-1) or `remainder` | W2 | ci | `scripts/laned/accept R3` |
| R4 | Stake-weighted upgrade tracker: per-hotkey `cfg-<hex>` and release from chain notes, stake-weighted %, names the laggards | R2 | ci | `scripts/laned/accept R4` (fixture metagraph + notes ⇒ exact %) |
| R5 | Previous-release worktrees + restart scripts for validator, intake, trainer and eval host (§10.8), with `--dry-run`. **(rev6.1)** The pre-restart snapshot covers the full §10.8 list: the validator files, **plus** the trainer state under `/root/cascade/_train_work/` (`era_state.json`, `trainer_submissions.json`, `funded_queue.json`, `trainer_promotion.json`, `champion_publisher.json`, `lane_b_driver.json`, `coldkey_identities.json`, `published_settlements.json`, its `activation_*` and `netconfig` files), `provisioner_state.json`, `pod_monitor_state.json`, and the deployed `chain.toml` (live overrides) with `hosts.toml` / `provision.toml` (RV3 note 21); a file absent on the box is reported as absent | — | ci (scripts) + owner (execute) | `scripts/laned/accept R5` (shellcheck; dry-run prints the exact plan; refuses during a scoring or settlement window fixture) |
| R6 | De-drift classifier (S0a): classify each `LIVE OVERRIDE` in a deployed chain.toml as main / overlay / operator-only; emit the main PR diff and the overlay body | — | ci | `scripts/laned/accept R6` (fixture with the 21 markers ⇒ deterministic classification) |
| R6x | S0a execution: main PR merged; deployed file reconciled | R6 | owner | `cascade-audit decided --config /root/cascade/chain.toml --state-dir /root/cascade` exits 0 **and** `scripts/laned/accept R6x --evidence evidence/s0a_diff.json` (consensus-key diff empty) |
| R7 | Overlay bring-up artefacts (S0c): benign seq-1 body generator, a refused-overlay body, runbook wrapper around `cascade-validator --netconfig-overlay … --netconfig-publish` (Approve-tier, dry-run shown to the owner) | R1 | ci | `scripts/laned/accept R7` (generated bodies pass `validate_overlay` on a fixture base; the refused body is refused whole) |
| R7t | S0c testnet drill (needs testnet `[netconfig] owner_hotkey`, today `""` at `chain.testnet.toml:903`) | R7 | testnet + owner | `scripts/laned/accept R7t --evidence evidence/s0c_testnet.json` (seq-1 adopted ≥ 51%; refused body applied nothing) |
| R7m | S0c mainnet no-op overlay | R7t, R6x, R2x | owner | `scripts/laned/accept R7m --evidence evidence/s0c_mainnet.json` (no 404 on S3 or HF; ≥ 51% `cfg-`; receipts stamp `netconfig_seq`; trainer apply block logged) |
| R8 | `deploy_gate` support for the new services (`--service intake-laned|eval-host`), since today only trainer/validator are accepted (`scripts/deploy_gate.py:290,298`). **(rev5)** Plus: (i) a new AREA `lane_d_intake` for `^cascade/lane_d/` (its own service). Today `cascade/lane_d/` matches **no** AREA (`deploy_gate.py:41-58`), so it passes unclassified; (ii) a CI test that **every path in the repo classifies** into at least one AREA, or is on an explicit docs/tests allow-list; (iii) `WORKER_IMAGE` gains the on-pod trainer modules (`cascade/trainer/expert_training.py` and what `worker.py --role expert` imports). The Dockerfile bakes the whole repo (`COPY . /root/cascade`, `deploy/Dockerfile:36`), but the AREA lists only `worker.py`, `interface/`, the Dockerfile and the lock files | — | ci | `scripts/laned/accept R8` |
| R9 | Catch-up release (S2b): main-as-of-now without lane D to validator, trainer and satellites | R6x | owner | `scripts/laned/accept R9 --evidence evidence/s2b.json` (2 settlements, identical vectors, no satellite burn) |
| R10 | Upgrade-from-mainnet-state harness (T25): restores a copy of `validator_state.db`, `activation_*`, `netconfig_state.json` and `cascade_state.json` into a testnet node and starts the release | — | ci (fixture state) + testnet | `scripts/laned/accept R10` and, for T25, `--evidence evidence/t25.json` |
| R11 | Shipped-config startup-gate test (T26): run the shipped `chain.toml` and `chain.testnet.toml` through every validator and trainer startup gate (activation note, loader, preflight) at a pinned live head | W2 | ci | `scripts/laned/accept R11` |
| R12 | Eval-host provisioning kit: on-rent check (`nvidia-smi -L`, VRAM, driver, disk), pinned known_hosts, venv sync, handshake, owner-paste setup and recovery scripts (§10.4) | W4 | ci (scripts) + pod + owner | `scripts/laned/accept R12` (shellcheck + dry-run) and `--evidence evidence/r12_pod.json` for a live run |
| R13 | `pool_d` builder VM provisioning: disk, wall, retry, credentials placement (no secrets printed), systemd timer | — | builder + owner | `scripts/laned/accept R13 --evidence evidence/r13.json` |
| R14 | Testnet stack provisioning (§10.3): testnet-ctl, testnet-eval, testnet `[netconfig] owner_hotkey` set (release-level edit), hourly testnet `pool_d`, pod TTLs and verified teardown | R13 | testnet + owner | `scripts/laned/accept R14 --evidence evidence/r14.json` |
| R16 | **(rev5)** Long-history harvest (§1.7 "Harvest now, score later"): sub-hourly forge-mirror retention on; deep-history sources recording; the builder's append-only per-series archive up to 131072 + 64 (W6a join key, private bucket, daily sha pin); per-rung readiness counts. Never scored before S8; never on the box | R13, W6a | builder + owner | `scripts/laned/accept R16 --evidence evidence/r16.json` (archive grows daily; `pool` bytes unchanged) |
| R15 | **(rev3)** Testnet pool_mkt cadence for lane D: a scheduled publish for the testnet lane D path that doesn't depend on `[marketcast] from_block` (today scheduled runs skip while it is 0, `publish-mkt-pool.yml:84-95`) | R14 | ci + testnet | `scripts/laned/accept R15 --evidence evidence/r15.json` (≥ 4 consecutive 6 h snapshots published on testnet) |

**Lane D core**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| W1 | **(rev3)** Measurement (= S1 B2, `scripts/laned_w1_measure.py`): first-seen existence check; snapshot-diff **upper bound** on forward-eligible counts at both clocks; pool_mkt at 6 h; (§1.3, §1.7): forward-eligible counts and latency per frequency; long-history inventory per context variant; `pool_d` size | — | builder (needs `TSFORGE_S3`) | `scripts/laned/accept W1 --evidence evidence/w1.json` |
| W2 | `[lane_d]` config: the keys `from_block`, `until_block`, `share`, `takes_from`, `remainder`, `contexts`, `horizon`, `forward_margin_hours`, `min_windows`, `min_clusters`, `min_long_series`, `d_margin`, `alpha`, `bootstrap_B`, `gate_suites`, `gate_weights`, ~~`unselected_suite`~~ **(rev5) dropped:** the held-out suite is the §16 sentinel, drawn from `[v2] sentinel_suites` (W-S1); a lane D key would duplicate it, `tau_public`, `reward_prior_kings`, `king_decay`, `max_entries_per_settlement`, `max_open_per_coldkey`, `max_bench_holds`, `loser_ttl_eras`, `dtypes`, **(rev3)** `max_mirror_lag_hours` (A1), `confirm_on_king_change` (D20), `max_judged_per_coldkey_per_settlement` (A8), `max_coldkey_unresolved_settlements` (A8), `max_reveal_delay_days` (A9), `lowfreq_freqs` (A10); **(rev5)** `ready_ttl_blocks` (B10), `max_source_share` (B14); `contexts` default `[4096, 8192, 16384]` (A5); `dtypes` default `["BF16"]` (A6). **(rev3) Size M, not S:** `lane_d.py` hardcodes `active() = False` (`:65-68`), refuses `from_block > 0` (**(rev5)** `:78-79`) and carries DEC-CA-0062 placeholder keys that must keep parsing (`until_block` and `share` overlay-settable; activation by signed-overlay `from_block`, never a DEC-CA-0045 vote), plus legacy keys still parsed (the deployed file carries `targets`, `teacher_*`, `gain_margin`, …); `ALLOWED["lane_d"]` with `remainder` refused via the `(table, key)` special case (**(rev5)** `netconfig.py:210`); loader parse, round-trip and **mutation** test per key; the 131072 cap; `LaneDConfig.active()` real | W0c | ci | `scripts/laned/accept W2` (includes `unread_keys.py`) |
| W2b | `[lane_d_ops]` non-consensus table (vault bucket, `pool_d` location, eval host, queue policy), read only by the main validator and the intake; never in receipts | W2 | ci | `scripts/laned/accept W2b` (satellite code path imports nothing from it) |
| W1.5 | **Walking skeleton** `make laned-e2e` (CI): fixture `pool_d` (with `end_ts`) → tiny checkpoint → intake (filesystem adapter behind the real storage interface) → ingest guards → scorer → signed report → `lane_d_state.json` → emission → `check_lane_d` | W2 | ci | `make laned-e2e` asserts: non-zero lane D weight for the challenger; audit passes; one tampered row fails; unarmed byte-identical; `share = 0` leaves weights unchanged and still yields a valid report; `LaneDConfig.active()` is not a constant. **Every later WP keeps it green and adds ≥ 1 assertion** |
| W3a | Storage probe scripts (presigned UploadPart, single-PUT size limit, Object Lock on multipart, prefix bucket policy, sub-token scope) + the **storage decision record** `decisions/STORAGE-LANE-D.md` naming the chosen upload path  **(rev5)** Also probes `UploadPartCopy` for ~2.5 GB objects (~~locking a king by copy at crowning~~) and ~~confirms COMPLIANCE retention can't be shortened~~. **(rev6.4)** Hippius has no Object Lock (403 on `GetObjectLockConfiguration` / `GetBucketVersioning` for sub-token and main key): the probe re-runs with `--no-object-lock` and the lock probes are recorded as `skipped`; kings are not locked by copy (§1.2). | — | owner (live Hippius) | `scripts/laned/accept W3a --evidence evidence/w3a_probe.json` (every probe has a recorded live result; the decision record names exactly one path) |
| W3 | `cascade/lane_d/` intake service: submit/ready for **only the W3a-chosen upload path**; ownership; **(rev3)** records `coldkey_at_ready` and refuses over-cap with `coldkey_cap_open` (advisory, A8); dedup; rate limits; queue; systemd unit with the code-path guard; Caddy route `/v1/lane-d/*`; RAM budget  **(rev5)** Enforces `ready_ttl_blocks` (B10); records the licence attestation at `ready` (B13). | W3a, W2b, R8 | ci (fs adapter) + testnet | `scripts/laned/accept W3` |
| W4 | Ingest guards: presigned fetch; sha256; `verify_checkpoint`; exact-filename allowlist; **lane-D-only dtype path for BF16** (today's guard requires the pinned F32 spec, `checkpoint_guard.py:323-328`, which lane A keeps unchanged); `verify_weights_finite`; worker handshake (git head + package hash) | W1.5 | ci + pod (handshake live) | `scripts/laned/accept W4` (the T5 malformed set is refused with exact codes) |
| W5a | Forward eligibility (§1.3): **(rev3) both clocks** (data time and first-seen > wall(max T) + margin) and `max_mirror_lag_hours`; the v2 snapshot-only rule is refused | W1.5 | ci | `scripts/laned/accept W5a` |
| W5b | Union across snapshots + freeze at floors  **(rev5)** Cap any one source's share of a frozen set's windows at `[lane_d] max_source_share` (recommended 0.25), dropping the excess deterministically (T18e). | W5a | ci | `scripts/laned/accept W5b` |
| W5c | Cohort intersection + latest-T deferral | W5b | ci | `scripts/laned/accept W5c` |
| W5h | **(rev3)** Queue selection: FIFO by T; `max_judged_per_coldkey_per_settlement`, the global cap, coldkey re-check at reveal (`coldkey_changed`), `coldkey_unresolved` bound, `max_reveal_delay_days` close; typed skip reasons in the report (A8, A9)  **(rev5)** `deregistered` void (court slot burns, with W8) and `ready_expired` (`ready_ttl_blocks`) handling, both typed in state and report. | W5b, W3 | ci | `scripts/laned/accept W5h` (T6b, T18b) |
| W5d | No-pad scorer + context variants (`cascade/eval/lane_d_scorer.py`). **(rev5)** Quantile-head scoring at the 9-level grid, the same method as lane A under `[v2] quantile_head_after = true`; a CI test that the house scored by the lane D scorer at 4096 equals its lane A per-window CRPS on the same windows | W1.5 | ci (tiny model) + pod (real size, T7) | `scripts/laned/accept W5d` and, for T7, `--evidence evidence/t7.json` |
| W5e | Verdict (paired cluster bootstrap, `d_margin`, max-T; **(rev3)** mandatory confirmation duel on the next disjoint set, D20 king-change rule, `rows_sha256` (A2, A9)) | W5c, W5d | ci | `scripts/laned/accept W5e` |
| W5f | House entry and refresh | W5e | ci | `scripts/laned/accept W5f` |
| W5g | Holds, `checkpoint_fault`, typed reason enums | W5e | ci | `scripts/laned/accept W5g` (asserts the exact reason enum per case) |
| W6a | `pool_d` selection = `pool`'s selected series ids (joined, not re-selected). **(rev5) Join rule:** the emitted `series_id` isn't stable. Collision suffixes `-2`, `-3`… are assigned in sort order (`builder.py:361-370, 471-480`), and `content_hash` is the sha256 of the (truncated) values (`:295`), so it changes whenever the tail moves and differs between `pool` and `pool_d` histories. Join on **(source, the pre-suffix source series key, channel)**, both built in the **same builder run** from the same forge records, never by emitted id or content hash. A key collision inside one source refuses the series from `pool_d` (logged) | W1 | ci + builder | `scripts/laned/accept W6a` (`pool` bytes unchanged) |
| W6fs | **(rev3, launch-blocking)** First-seen recording: an append-only ingest log in the forge mirror path recording, per partition or point, the first time our mirror saw it, plus a per-feed mirror-lag measurement (source publish stamp where available). **Scope:** a recorder + reader used only by the `pool_d` builder. **Invariant:** `pool` bytes and lane A windows byte-identical (the test diffs a `pool` build with and without the recorder). **Tests:** first-seen never later than any snapshot containing the point; rewrites and backfills keep the *first* time; lag computed per feed. Not implemented in this PR  **(rev5) Attestation (B12):** a daily digest (Merkle root) of the ingest log is committed **before** any snapshot uses that day's records: in the signed receipt, or on chain via the main validator's commitment. A snapshot whose first-seen records postdate their committed root is refused. | W1 | builder | `scripts/laned/accept W6fs --evidence evidence/w6fs.json` |
| W6b | **(rev3)** Extended history **to 16384 + 64** + per-series (per-channel) `end_ts` / `first_target_ts`. **(rev6.1)** Also writes, for **every** series, `frontier` (a list, possibly empty: one record per earlier snapshot holding that series), and, for **every** feed in `pool_d`, a `feeds` entry (`mirror_lag_hours`, a number or `null`, and `forecast_product`). A `pool_d` in which any series lacks `frontier` (absent or `null`) or any feed lacks a `feeds` entry is **invalid**; the builder's tests assert that none does (RV3 BLOCKING 1: W5a's eligibility fails open on a missing key) | W6a | builder | `scripts/laned/accept W6b --evidence evidence/w6b.json` |
| W6c | `pool_d_hist` / `pool_d_tgt` split | W6b | ci + builder | `scripts/laned/accept W6c` |
| W6d | Pinning on Hippius; digests in reports; retention | W6c, W3a | builder | `scripts/laned/accept W6d --evidence evidence/w6d.json` |
| W6s-*n* | **(rev3: moved to S8)** One WP per deep-history source (option 3, §1.7), each passing cleaning, domain and source keys | W6b | builder | `scripts/laned/accept W6s-<name> --evidence evidence/w6s_<name>.json` |
| W6-S8 | **(rev3)** S8 long-context work: history to 131072 + 64, deep backfill, valeval-laned (96 GB pinned-torch) provisioning | W6d | builder + pod | `scripts/laned/accept W6-S8 --evidence evidence/w6s8.json` |
| W3b1 | Gate suite draw from the boundary block hash | W1.5 | ci | `scripts/laned/accept W3b1` |
| W3b2 | Paired sweep vs KING_D and house on one host | W3b1 | ci (recorded suite outputs) + pod | `scripts/laned/accept W3b2` |
| W3b3 | **(rev5)** Sentinel sweep + publish: the era's §16 sentinel suite (not fixed TIME), swept and published for every new king, never gating | W3b2 | ci | `scripts/laned/accept W3b3` |
| W3b4 | Suite identity + partial classification | W3b2 | ci | `scripts/laned/accept W3b4` |
| W3b5 | **(rev3)** Always-run low-frequency private-pool non-inferiority check (`lowfreq_freqs`, `tau_public`; `lowfreq_regression`) (A10). **(rev6.1)** Result for a confirm candidate (a verdict that cleared at stage `confirm`) whose frozen set **does** hold low-frequency windows: `pass` or `fail` (`fail` = `lowfreq_regression`, not crowned), with the true `n_windows`; it is never `not_candidate`. W3b5 replaces W7a's interim path, which emits `not_candidate` with the true count (`validator/lane_d.py:371-387`). `no_windows` stays for a candidate with none; `not_candidate` (with `n_windows = 0`) stays for every non-candidate verdict. The W7a enum (`schemas/lane_d_report_1.json`, `LOWFREQ_RESULTS`) already holds `pass`/`fail`; W3b5 adds a value to it only if it introduces a new one | W5d | ci | `scripts/laned/accept W3b5` (T4b) |
| W7a | Report schema `lane-d-report/1`, including §14.3 forward fields and **(rev3)** `rows_sha256`, both clocks per window, mirror lag, the judged set and skip reasons, `coldkey`, the `confirmation` block, the low-frequency result, **(rev5)** a `scoring_method: "quantile_head"` field (refused if absent or different)  **(rev5)** Plus the first-seen Merkle root(s) each frozen set relies on (B12). | W1.5 | ci | `scripts/laned/accept W7a` (JSON-schema; unknown version refused) |
| W7b | Report signing | W7a | ci | `scripts/laned/accept W7b` |
| W7c | `lane_d_state.json` + receipt prior/posterior digest binding. **(rev6.1)** Wiring trap (RV3 note 10): `validator/weights.py:517-524` passes the second element to `emission_vector` only when a lane B or C pay record exists (`pay is not None or pay_c is not None`); if lane D pays while B and C are off, the lane D share is dropped and `set_weights` normalisation hands it to lane A. The WP that wires lane D pay into weights (W7c, or the one that passes `lane_d_cfg` first) adds the lane D pay record to that condition, with a test for "lane D on, B and C off" | W7b | ci | `scripts/laned/accept W7c` |
| W7d | Satellite replay: an **independent** implementation, no imports from producer eligibility, verdict or weight code (invariant 12) | W7c | ci | `scripts/laned/accept W7d` (import-graph check included) |
| W8 | Emission: lane D in `lane_parts` / `lane_pool_shares` (`emission.py:330-383`, `lane_c.py`), court decay, house burn  **(rev5)** A deregistered court member's slot burns (B8). **C3 (D48)** decides where lane D's share comes from inside lane A; W8 also changes `lane_pool_shares` (`shared/lane_c.py:584`, returns a `(B, C)` tuple, callers `emission.py:361`, `lane_c.py:605`), a consensus signature at every caller. | W1.5 | ci | `scripts/laned/accept W8` |
| W8b | **(rev6.2) (rev6.4, accepted owner 2026-11-23)** Production wiring: today nothing outside tests calls `cascade.validator.lane_d.settle_lane_d`. The validator loop calls it at each lane D settlement boundary with: the boundary block hash (read-only chain read); the era's sentinel (W-S1 draw from the era-start hash); a bench `SweepSession` on the configured bench host (the W3b2/W5e gate inputs; an armed crown without them is a `bench_hold`); `expect_prior_sha256` from the last published receipt (W7c); the low-frequency check (W3b5); and the house/king state. After settlement: W3b3 `sweep_and_publish` for a new king; W-S3 `build_sentinel_board` with the era's kings list (from W7c state / W17); publish the signed report + receipt; lane D pay reaches weights (W8 emission) **only via the published report**. **Fail-closed:** any unreadable input ⇒ no settlement this tick (retry next tick), never a guessed input. **Restart-safe:** a re-call after restart is idempotent (W7c). **Weight loop:** lane D's bench work runs off the weight loop (W19) or is bounded so weight reasserts are never delayed past a stated bound (2026-11-22: weights went stale ≥ 2 h during a synchronous king-check sweep). **Unarmed:** byte-identical (invariant 1). Ships release-then-activate (external validators on netuid 91) | W5e, W3b5, W7c, W8, W3b3, W-S3, W17 | ci + testnet | `scripts/laned/accept W8b` |
| W9 | Audit `check_lane_d` + `derive_weights` (**rev3:** re-derives the judged set, coldkey caps, both-clock eligibility, the confirmation rule and `rows_sha256`), independent of producer code. **(rev6.1)** Refuses any judged snapshot in which a series lacks `frontier` (absent or `null`) or a feed is not listed in `feeds` (W6b). It re-derives S0 presence (`absent_from_s0`), the backfill rule and mirror-lag eligibility (`over_lag`, `forecast_product`) from the snapshot itself, **without importing `cascade/validator/lane_d_forward`** (invariant 12; RV3 BLOCKING 1); its reason labels follow §1.3 (`publish_ahead` = first-seen fails, `late_mirror` = data time fails). **(rev6.1)** Wiring trap (RV3 note 10): the audit's `emission_vector` call (`audit/checks.py:2834-2841`, `:3660`) passes the second element only when a lane B or C pay record exists, so a lane D share with B and C off would be dropped and handed to lane A by normalisation; W9 must add the lane D pay record to that condition. **(rev6.1)** W9 decides invariant 12 for the shared lane D weight math (`lane_pool_shares` / `lane_parts` / `lane_d_vector`, RV3 note 11) before its tests are written: either re-derive lane D's carve and court vector without these functions, or record them in the plan as shared consensus code exempt from invariant 12 | W7c, W8 | ci | `scripts/laned/accept W9` |
| W10 | Miner CLI `cascade lane-d export \| verify \| submit \| ready \| status`, **W3a-chosen upload path only**. **(rev5, D43)** The commit reveals immediately (`blocks_until_reveal = min_blocks`), never at boundary − margin; a test asserts the lane D commit path never uses the lane A boundary timing | W3a, W3, W4 | ci + testnet | `scripts/laned/accept W10` |
| W11 | cascade-ops checks (§10.7) + follower patterns | W5g, W7c | ci (cascade-ops repo) | `scripts/laned/accept W11` (includes `follower_patterns.py`) |
| W12a | Docs and agent surfaces (§11.2 text surfaces, website; **rev3:** LANE_D.md FAQ on coldkey caps, confirmation latency, both clocks, BF16; VALIDATOR.md on what satellites verify for the new report fields; OPERATOR.md on launch eval host, S8, first-seen recorder) | W10 | ci | `scripts/laned/accept W12a` (every §11.2 surface mentions lane D; links resolve; `llms.txt` lane list matches the status fixture) |
| W12b | `cascade status --json` and `discovery-check` lane D fields | W2, W17 | ci | `scripts/laned/accept W12b` |
| W13 | King publication (§11.1): Hippius Hub, R2, HF mirror, signed `champions/lane_d/index.json`, `laned-king-sync.yml`, cards, one-digest check  **(rev5)** A takedown path for the Hub and HF mirrors (licence or rights claim ⇒ unpublish the mirrors, keep the digest and verdict record). | W5e, W3a | ci + testnet | `scripts/laned/accept W13` |
| W14 | ~~**(rev5) blocked: needs owner (Proposed design change 4).**~~ **(rev6) Unblocked:** change 4 was accepted (owner 2026-11-21). Lane D-only fast kill: a `lane_d: {status: "absent"}` receipt state with a lane D-only burn. Today a missing lane D report makes satellites re-assert the old vector for one era (`stale_after_blocks = 3600`) and then burn the whole vector (`satellite.py:30-45`) | W7c | ci + testnet | `scripts/laned/accept W14` |
| W15 | Prefetch cache, LRU with KING_D and house pinned | W4 | ci + pod | `scripts/laned/accept W15` |
| W16 | Reference fine-tune script (miner kit) | W10 | ci (CPU tiny run) + pod | `scripts/laned/accept W16` |
| W17 | `status/lane_d.json` producer (queue, pending progress, king, board; **rev3:** per-entry reason enum including `coldkey_cap_settlement`, `confirm_pending` and reveal deadline). **(rev6.2, owner 2026-11-22)** Also produces the W18 vault ledger `laned_vault_ledger/1`: the current era, the vault entries with their statuses and `closed_era`, and the open disputes | W5b, W7c | ci | `scripts/laned/accept W17` |
| W18 | TTL sweeper: vault → owner-only archive → delete (live runs need D16) | W3a | ci (fs adapter) + owner | `scripts/laned/accept W18` |
| W19 | Lane D work runs **off the weight/poll thread**; the weight-loop keep-alive (#471) is in the release (invariant 6, T15) | W5e | ci + testnet | `scripts/laned/accept W19` (a fixture slow lane D step leaves weight-set timing unchanged) |

**(rev4) Lane B track (trainer-run, miner-chosen experts, suite targets; §12b)**

Every new `[lane_b]` key below ships with:

- a loader parse;
- the **drop-when-default** digest rule (unarmed digests don't move);
- `ALLOWED["lane_b"]` coverage, which is overlay-settable (`remainder` stays
  release-only);
- round-trip and mutation tests in the named WP.

| WP | What | Keys / fields | Depends | Runs in | Accept |
|---|---|---|---|---|---|
| WB10 | Lane B config keys (parse, digest drop-when-default, overlay, round-trip + mutation tests). **(rev5)** `contamination_hashset` is release-only (added to the `netconfig.py:210` tuple; test: an overlay setting it is refused); new D42 keys `funded_pod_skus`, `max_leg_hours`, `max_leg_seconds`, `guard_factor` | `suite_targets`, `contamination_hashset`, `contamination_min_hits`, `suite_target_max_repo_mb`, `min_experts_frac`, `max_experts_frac`, `router_rows`, `traffic_steal_bound`, `open_rule`, `sentinel_target_rule`, `sentinel_max_defer_eras` | W0c | ci | `scripts/laned/accept WB10` |
| WB11 | Contamination hash-set builder: exact + normalised sub-window hashes from the pinned suite data; versioned artefact + sha256. **(rev6.2)** Normalisation is schema 2 (`cascade_contamination_hashset/2`, `cascade-zgrid/2`) with RV6's low-information exclusion, per §12b.1 (#617/#621, #632) | `contamination_hashset` | WB10 | ci + owner (pin) | `scripts/laned/accept WB11` (hash set reproducible from the suite identities) |
| WB12 | Routing information producer (LB3): extend PR #481 telemetry to per-expert counts at phase level; per-suite context-only affinity on the bench host; digest-pinned `status/lane_b_routing-<king8>.json` | `status/lane_b.json` `routing_info` pointer + `stale_since_block` | PR #481 merged | ci + pod | `scripts/laned/accept WB12` |
| WB13 | Intake contamination check (LB1): AST literals, `config.json`, non-code bytes vs the hash set; typed `benchmark_series_embedded`; `suite_target_max_repo_mb` (D38) | receipt field `contamination: {hashset_version, result, suite, window_id}` | WB11 | ci | `scripts/laned/accept WB13` (TB10–TB12) |
| WB14 | Declared-rows parsing + per-layer clamps (LB2): **(rev5)** keys = MoE ordinals 0–6 (not block indices); clamp minimum only on layers with ≥ 1 row; validated vs `experts_per_layer`; codes `declared_rows_out_of_range`, `declared_rows_bad_layer`, `declared_rows_count` | manifest field `declared_rows` | WB10 | ci | `scripts/laned/accept WB14` (TB13, TB15, TB15b) |
| WB15 | Row claims + overlap deferral: earliest reveal; both modes coexist (declared first, then ESFT excluding held rows) | `status/lane_b.json` `deferred: rows_held` | WB14 | ci | `scripts/laned/accept WB15` (TB16) |
| WB16 | Router-row decision (D40): (a) the frozen path is today's backend, with a test that it stays frozen; if (b): unfreeze the declared router rows, router byte-check, `traffic_steal_bound` guard. **(rev5)** (b) is a worker-image change (shared with lane A): 5-pin repin + wall gate + DEC-CA-0067 qualification, after the scored EMA constant is fixed  **(rev5)** Depends on W-EMA. | receipt field `traffic_share_delta` (b only) | WB14 | ci + pod | `scripts/laned/accept WB16` (TB14) |
| WB17 | Opening rule (D41, LB4): `open_rule` implementation; driver state records the rule and the satisfying evidence | `lane_b_driver` state `gate_rule`, `gate_evidence` | WB10 | ci + testnet | `scripts/laned/accept WB17` (TB17–TB18) |
| WB18 | Sentinel interaction (D35/LB5): suite-target deferral, `sentinel_max_defer_eras`, no charge on deferral. **(rev6.2, owner 2026-11-22) Follow-up WP note:** while a funded entry is deferred (`sentinel_target` or `rows_held`), its payer key in `PayerKeyVault` (`cascade/funding/vault.py`) has its TTL refreshed on each deferred tick, the same rule as the entry (RV6, #629/#633), so a deferral never outlives the key the leg rents on. No WP owns the vault; the follow-up adds the vault hook and its own test (flagged, not pinned, in `tests/accept/wb18/README.md`) | `status/lane_b.json` `deferred: sentinel_target` | W-S1, WB10 | ci | `scripts/laned/accept WB18` (TB19–TB20, TS4) |
| WB19 | **(rev5)** Expert-leg measurement (LB6), **per candidate GPU type**: expert-only step rate, leg overhead split from training, peak memory at contract micro-batching, image qualification; plus one full lane A leg on 8×H100 (is 13,600 s training-only?) | `evidence/wb19.json` per-GPU table | — | pod (owner yes, OA8) | `scripts/laned/accept WB19 --evidence evidence/wb19.json` |
| WB22 | **(rev5)** Lane B GPU list (D42): `[lane_b] funded_pod_skus` separate from `[round]`; `GpuChoice` reads it for lane B legs only; trainer-side per-GPU step-rate table from WB19; audit re-derivation pinned to the recorded `gpu_name` | loader parse + round-trip + mutation test; lane A choice unchanged | WB10, WB19 | ci | `scripts/laned/accept WB22` (TB23, TB24) |
| WB23 | **(rev5)** Miner-chosen leg length (D42, D42b): `train_hours` on the funding request (0.5–`max_leg_hours`, 0.5 h steps); steps fixed at admission and signed in the leg record; guard = `train_hours × guard_factor` + overhead, capped at `[lane_b] max_leg_seconds` (lane B contract derivation; `[training] max_train_seconds` untouched); expected landing boundary returned to the miner; validator check of recorded steps and guard | keys `max_leg_hours`, `max_leg_seconds`, `guard_factor`; `status/lane_b.json` shows steps, GPU and landing boundary | WB10, WB22 | ci | `scripts/laned/accept WB23` (TB25–TB27) |
| WB24 | **(rev6.3, proposed)** Lane B miner CLI. Today `cascade lane-b fund` / `lane-b status` (`cascade/miner/lane_b.py`) do not send the declared-rows header, so the WB20 docs have miners hand-build that request. `cascade lane-b fund` gains `--declared-rows <file>` (builds and signs the WB14 `X-Lane-B-Declared-Rows` header and its signed `rows=` tag; refuses locally with the same `DECLARED_ROWS_CODES` the intake uses), `--target suite:<s>` (suite targets, WB18) and `--train-hours` (WB23, once built). `cascade lane-b status` shows the new deferral and refusal reasons (`rows_held`, `sentinel_target`, the `gate_closed_reason` values, the WB13 contamination codes incl. `contamination_budget_exceeded`) from `status/lane_b.json` and the intake replies. New `cascade lane-b verify-repo <dir>` runs the WB13 scan (`scan_repo`) locally against the pinned hash set (no upload; never executes the repo) so a miner can check before funding. Never prints keys; unarmed defaults unchanged | CLI flags `--declared-rows`, `--target suite:<s>`, `--train-hours`; subcommand `verify-repo` | WB13, WB14, WB15, WB17, WB18 (WB23 for `--train-hours`) | ci | `scripts/laned/accept WB24` |
| W-EMA | **(rev5)** Score the raw cooled endpoint on every leg (lanes A and B): `power_ema_active` → raw for warm-start legs; keep writing `weights_ema005`/`weights_ema010` unscored; main-validator EMA telemetry off the critical path (§8). Blocks every repin ≥ rc16 and D40(b) | worker image (`pyramid_trainer.py`) | — | ci + testnet | `scripts/laned/accept W-EMA` |
| WB20 | Lane B docs (MINER, INTERFACE, AGENT_MINING, VALIDATOR) | — | WB13–WB18 | ci | `scripts/laned/accept WB20` |
| WB21 | DEC-CA-0071 amendment (this PR drafts it; owner accepts) | — | — | ci + owner | `scripts/laned/accept WB21` |
| W-S1 | **(rev4: now in §3)** §16 sentinel draw + lane A gate rule behind `sentinel_from_block` + receipt field + `[v2]` overlay keys (**(rev5)** no shadow logging, D45). **(rev5) Changed consensus code:** the bench gate (`cascade/eval/bench_gate.py`) **and `king_check_gate` (`cascade/validator/king_check.py:93`)**, both of which must drop the sentinel suite from the rule; block-gated by `sentinel_from_block` | `[v2] sentinel_from_block`, `sentinel_suites`, `sentinel_ratchet`, `sentinel_tau` | W0c | ci | `scripts/laned/accept W-S1` (TS1–TS3, TS5) |
| W-S2 | **(rev4: now in §3)** lane B guard/target exclusion for the sentinel. **(rev6.2, owner 2026-11-22) Follow-up:** a warm-start acceptance made at an era-init block is governed by the sentinel of the era that init block opens; applies to the acceptance callers (trainer `accept_candidates`, validator `verify_acceptance`) (§16) | — | W-S1 | ci | `scripts/laned/accept W-S2` (TS4) |
| W-S3 | **(rev4: now in §3)** sentinel numbers on the status boards | — | W-S1 | ci | `scripts/laned/accept W-S3` |


**Review WPs.** After every 5 build WPs reach **L3** (**(rev6)**, §17.1; counting W/R build WPs, not
owner execution WPs), a **fresh-context review WP** runs. RV1, RV2, … are
read-only reviewers against main with the plan as the spec. Accept is
`scripts/laned/accept RVn --evidence evidence/rv<n>.md`: no unresolved
BLOCKING item, and each finding is either fixed or filed as a WP.

**Execution WPs (stage gates, §5)**

| WP | Stage | Runs in | Accept |
|---|---|---|---|
| X3 | S3 testnet matrix T1–T29 | testnet | `scripts/laned/accept X3 --evidence evidence/s3_matrix.json` (each row's recorded pass, 6 clean eras, ≥ 1 crowning, ≥ 1 house retake) |
| X4 | S4 release + **(rev5)** launch eval host bring-up (D11; valeval-laned only at S8) + dry-run + announcement + staged revert overlay | owner | `scripts/laned/accept X4 --evidence evidence/s4.json` |
| X5 | S5 capability-release upgrade (eval host → validator → intake → trainer; satellites) | owner | `scripts/laned/accept X5 --evidence evidence/s5.json` (lane A unchanged ≥ 2 settlements; `--netconfig-policy` lists `lane_d`) |
| X6a | **(rev5)** S6 readiness probe and activation evidence (observed `cfg-` stake before the activation block); D19 test entries moved to testnet **(rev6)** One stage: S6 (D45). | owner | `scripts/laned/accept X6a --evidence evidence/s6a.json` (**(rev6)** observed `cfg-` stake ≥ 2/3 before `effective_block` (R4); webhook test alert delivered; from the block, the first paid settlement's vectors match on ≥ 2 validators and satellite replay matches. No "≥ 2 eras of reports" before activation) |
| X7 | S7 operate | owner | `scripts/laned/accept X7 --evidence evidence/s7.json` |

### 3.1 Build rules

1. **Tests first.** Acceptance tests are written first in `tests/accept/<wp>/`
   and are **read-only to builders**. CI fails a PR that edits
   `tests/accept/` together with that WP's implementation.
2. **Mock only external services**, never the component under change.
3. **No placeholders.** No `TODO`, stubs, `pass`, `NotImplementedError`,
   hardcoded sample values or mocks outside `tests/`. The stub scan runs in
   every Accept.
4. **Every `[lane_d]` key** ships with a loader parse, an `ALLOWED` entry, a
   round-trip test and a mutation test. `unread_keys.py` fails on any key no
   parser reads.
5. **Every lane D log line** has a follower pattern, checked by
   `follower_patterns.py`.
6. **(rev6) Each WP has a tests-only PR (`tests/accept/<wp>/`) followed by
   ONE implementation PR.** Explicit paths, no stacks, owner-merged;
   dependents start only after the dependency's code is **on main** (verify `git log
   origin/main`, never a MERGED badge). This matches rule 1: the tests land
   first and alone, and the implementation PR never touches them.
   **(rev6.2, owner 2026-11-22)** Implementation, tests-correction, evidence
   and review PRs stay **one WP per PR**. A **tests-only PR** (new acceptance
   tests, no implementation yet) MAY carry several WPs' `tests/accept/<wp>/`
   directories, with one `laned:<WP>` label per WP; each WP's expect-fail (or
   verification) accept run is executed and ledger-recorded separately, and the
   whole PR fails if any WP's check fails.
7. **(rev6.3, owner 2026-11-23) Implementation PRs MAY batch up to 3 WPs**
   when (1) their PATHS sets are disjoint, (2) none depends on another in the
   batch, and (3) each WP's tests are already on main. Each WP in the batch is
   still checked on its own: scope (its PATHS only, files attributed per WP),
   its own accept, and CROSS; the ledger records one row per WP. A red WP is
   split out of the batch by the builder (the others proceed); WPs whose PATHS
   overlap stay serial. Tests-corrections found in one pass may likewise be
   batched in one PR (one dir per WP). Labels: one `laned:<WP>` per WP in the
   batch. This narrows rule 6's "one WP per PR" for implementation and
   tests-correction PRs; evidence and review PRs stay one WP per PR. (Former
   rules 7–8 are now 8–9.)
8. **A fresh-context review WP** after every 5 build WPs reach L3 **(rev6)**.
9. **Blocked is acceptable; faked completion is not.** If a WP can't be fully
   built, the builder stops and records the blocker in `evidence/blocked/<wp>.md`.
10. **(rev6.4, owner 2026-11-23) CI-first testing for loop agents.** An agent
    runs **only its own WP's accept dir** (`tests/accept/<wp>/`) locally.
    Every other suite (unit shards, receipt-replay, CROSS) runs in CI, and all
    of them are **required green before merge**. Why: the production box
    (4 cores / 7 GB) also scores the validator, so a local full suite stalls
    scoring.

### 3.2 Invariants

These are pinned by tests in the WPs named.

1. **Unarmed ⇒ byte-identical.** With `from_block = 0`, every receipt,
   manifest, verdict and weight vector matches today's. (W1.5, W2, W8)
2. **Armed ⇒ lane A verdicts and court unchanged.** Lane A's share-of-total
   changes by design, and the receipt gains lane D fields. (W1.5, W8, W9)
3. **No miner bytes execute.** Only repo code imports; the file allowlist is
   exact. (W4)
4. **Forward-only cannot be violated.** A scored window failing §1.3 for
   max(T_E, T_K) is an audit failure. (W5a, W9)
5. **Restarts are idempotent.** No re-score, no double crown. (W7c)
6. **Weights never wait for lane D** (off the poll thread; keep-alive). **(rev3)** #471 is merged (0c52d07) but not deployed (rc37 lacks it); it ships with S2b.
   (W19, X3 T15)
7. **Replay compares structured fields only.** (W7d)
8. **Targets never leave the box.** (W6c; see the proposed design changes in
   the changelog)
9. **Armed with `share = 0` ⇒ the weight vector is identical to unarmed, and
   a valid signed lane D report is still produced.** (W1.5, W8)
10. **Every "unchanged when unarmed" test has a paired "armed ⇒ non-empty
    lane D effect" test.** (every WP; enforced by `scripts/laned/accept`)
11. **`pending`, `hold` and `burn` always carry a typed reason enum,** and
    tests assert the exact reason. (W5g, W7a)
12. **Audit and satellite-replay code never import the producer's
    eligibility, verdict or weight functions.** (W7d, W9; import-graph check)

**Services touched:**

- the validator;
- the lane D intake service (new);
- the audit;
- the CLI;
- cascade-ops;
- the eval host;
- the `pool_d` builder;
- `deploy_gate`;
- **the trainer**, through the capability release only. The overlay resolver
  in `netconfig.py` gains `lane_d`. There's no trainer behaviour change, image
  or repin, and no `cascade/funding|trainer|…` path changes
  (`scripts/deploy_gate.py:55`). §14.5.

**(rev5) Blast radius by path** (`deploy_gate.py:41-58`):

- **CONSENSUS diffs** (every validator must compute them identically, so each
  is block-gated and ships in the capability release):
  - W2 (`cascade/shared/config/`);
  - W4 (`cascade/eval/checkpoint_guard.py`);
  - W5d (`cascade/eval/lane_d_scorer.py`);
  - W7–W9, including W8 (`cascade/shared/emission.py`) and the
    `cascade/validator/lane_d*.py` / `cascade/audit/` halves.
- **Worker-image changes:** any change to `uv.lock` or `pyproject.toml`
  (`WORKER_IMAGE`), and any on-pod trainer code (lane B D40(b), below). Each
  is a repin with 5 pins and the wall gate, never mid-seed-epoch.
- **Lane B (rev5, B4 and owner note).** Lane B shares lane A's worker image
  (`[round] funded_pod_image` and `[v2.training] train_image_digest`, both
  rc15 `5d41d94b…`; `[lane_b]` has no image key of its own).
  - Dispatch path: `lane_b_driver._dispatch_funded` (`:1297`) → `_run_funded_leg`
    (`:1329`), the lane A funded-leg path, so the same image pin.
  - **Declared rows need no image change.** The trainer passes the selection
    to the pod as data: `--selection-json` (`worker.py:119, 218-227`), keyed by
    MoE ordinal. The on-pod code is unchanged.
  - **D40(b) router unfreeze does need one.** It changes
    `expert_training.py`, which runs on the pod under `worker.py --role
    expert`. That's a **shared-image repin that also moves lane A**, with
    DEC-CA-0067 qualification.
  - Any image ≥ rc16 turns on power EMA for lane A at the next warm-start
    epoch (`POWER_EMA_SIGMA_RELS` is image-only). So **(rev5) D40(b) and every
    repin to an image ≥ rc16 depend on W-EMA** (§8): the next image scores the
    raw cooled endpoint on every leg.

---

## 4. Testnet plan (netuid 259)

**Setup**

- The testnet stack (§10.3) is **fully separate from the prod box**.
- `[v2] from_block = 8190000`; `epoch_blocks = 150`; `era_settlements = 4`.
  Lane D arms at a 600-block boundary at least one era out.
- **(rev3) Floors on testnet.** A daily pool can't meet the mainnet floors
  inside a testnet day, so the testnet forward path uses **`pool_mkt`**
  (`.github/workflows/publish-mkt-pool.yml`, cron `30 */6 * * *` since #480;
  `cascade/eval/marketcast.py`) and/or **testnet-only floors**.
  - **(rev3, verified)** pool_mkt's *scheduled* runs **skip while
    `[marketcast] from_block = 0`** (`publish-mkt-pool.yml:84-95`). That's
    the current value in both `chain.toml` and `chain.testnet.toml`, so
    today pool_mkt is published only by **manual `workflow_dispatch`**.
  - Using it for lane D testnet needs either a dispatch per run (an owner
    step), or a lane-D-specific schedule. The latter is a new WP, `R15`, and
    must not arm MarketCast. **The
  mainnet floor values (1000 windows / 20 sources) are therefore untested on
  testnet.** B2 measures them on mainnet snapshots instead.
- **The testnet `pool_d` builds hourly**, not daily, so the forward path runs
  inside a testnet day. Margin is 30 min.
- **The testnet house is a deliberately weakened checkpoint** (e.g. an early
  lane A leg), so an honest fine-tune can beat it deterministically.
- "House changes" are simulated by swapping the testnet house pointer through
  the testnet overlay. No real 8×H100 testnet lane A dethrone is needed.
- Test hotkeys are owner-registered; **never** the mainnet wallet.

**Matrix**

| # | Case | Expected |
|---|---|---|
| T1 | House only, 2 eras | KING_D = house; slot burns; lane A verdicts and court unchanged |
| T2 | Exact copy of the house | Refused at dedup (no eval spend) |
| T3 | Honest fine-tune of the weakened house on public data | Wins forward duel and gate; crowned; published; lane D share paid |
| T4 | Domain-overfit model | Loses, or blocked by the drawn suite, with the reason logged |
| T5 | Malformed: extra `.py`, edited `model.py`, wrong `config.json`, `.pt`/`.bin`/pickle, wrong dtype or shape, NaN, oversize | Each refused at ingest with a specific code; nothing executed |
| T6 | Second entry for a hotkey; coldkey cap; upload after `ready` (overwrite) | Refused; refused; `digest_mismatch` at ingest |
| T6b | **(rev3)** Three entries from one coldkey ready in one settlement; one coldkey changed between ready and reveal; an unresolvable coldkey | 2 judged, 1 skipped `coldkey_cap_settlement` and carried; `coldkey_changed` refusal; `coldkey_unresolved` alert, then closed after the bound |
| T7 | Each context variant: wall time and memory on **(rev5)** the launch eval host for 4k/8k/16k (valeval-laned for 32k–128k at S8); 4k model vs declared-128k copy on ≤ 4096-history windows | Identical scores on short windows; a variant is armed only if it fits the budget |
| T7b | Gate: two validators derive the same suite from the block hash; regression vs KING_D or vs house blocks; **(rev5)** the era's sentinel suite swept and published only, never gating | As stated |
| T7c | 2.5 GB (BF16, rev3) direct upload (multipart or single presigned PUT per W3a); prefetch; box RAM and disk flat | As stated |
| T8 | **(rev3)** Forward rule, both clocks: **(rev6.1, labels corrected to §1.3 and `lane_d_forward.py:103-110`)** data-time-only pass but first-seen fail (publish-ahead, `publish_ahead`: a future-stamped value our mirror saw before T); first-seen-only pass but data-time fail (late mirror, `late_mirror`: data old, mirror saw it late); both fail (`before_margin`); a series lacking `frontier` or a feed not listed in `feeds` (refused); a feed over `max_mirror_lag_hours`; entry and king T values in both orders; a forecast-product feed; a backfilled partition | **(rev5)** Eligible only if **both** clocks (data time and first-seen) are after wall(max(T_E, T_K)) + `forward_margin` for every target point, and the feed's mirror lag ≤ `max_mirror_lag_hours`; the late-mirror, publish-ahead, over-lag, forecast-product and backfill cases are all ineligible; the snapshot-absence rule is never used for verdicts |
| T9 | Three entries freezing in one settlement with different T | Intersection set; latest-T deferred if below floors; max-T crowns the best |
| T9b | **(rev3)** Clear, then a house refresh before confirmation | Per D20: restart vs new king (default) — first clear discarded, recorded in state and report |
| T10 | House pointer swapped to a stronger checkpoint | House retakes; slot burns; prior miner tails keep paying |
| T11 | Validator restart mid-judgement | No re-score, no double crown |
| T12 | Satellite replay + `cascade-audit`; one tampered row | Identical weights; `check_lane_d` passes; tamper fails |
| T13 | Lane A verdicts and court before and after arming | Identical |
| T14 | **(rev3)** Null FPR per entry for each null-pair type (prior-king pairs, seed replicas, perturbed king), confirmation-duel FPR, per-era FPR at k = 32 (S1: B1 / `scripts/null_fpr.py`) | **S4 gate:** per-era FPR ≤ the owner target (D24, OA3) at the chosen `d_margin` |
| T15 | 2 h lane D sweep across weight intervals | No weight gap > `weight_set_interval_blocks` |
| T16 | Front-run: published house resubmitted, plus a 1-ulp perturbation, in the crowning settlement | Exact refused; perturbation loses to the house |
| T17 | Checkpoint crafted to make a suite partial or NaN | `checkpoint_fault`; KING_D not held |
| T18 | Entry `ready` in an era's last settlement; `until_block` with entries pending (set by overlay) | Carries across eras; at `until_block` pending entries close unjudged and unpaid |
| T18b | **(rev3)** An entry pending at `max_reveal_delay_days` | `closed_unjudged: reveal_delay_exceeded`, logged, slots freed |
| T18c | **(rev5)** A pending entry's hotkey deregisters; a prior king deregisters | Entry voided `deregistered`, slot freed; the prior king's slot burns; satellite replay agrees |
| T18d | **(rev5)** Entry readied but not revealed within `ready_ttl_blocks` | `ready_expired`, open slot freed, nothing judged |
| T18e | **(rev5)** One source supplies more than `max_source_share` of a frozen set's windows | Excess windows from that source dropped deterministically (seeded by the settlement); the set re-checks its floors |
| T4b | **(rev3)** A model that regresses only on low-frequency series | Blocked by `lowfreq_regression` |
| TB10 | **(rev4)** Generator embedding an exact GIFT/BOOM/TIME/FEV test window as a literal | Refused at intake, `benchmark_series_embedded`, before any compute |
| TB11 | **(rev4)** Same window scaled ×3.7, offset +100 and rounded to 3 dp, in `config.json` | Refused (normalised match) |
| TB12 | **(rev4)** A clean generator; and a hash-set version change | Passes; the receipt records the hash-set version used |
| TB13 | **(rev4)** Declared rows out of range for a 128-expert layer (ordinal 0); **(rev5)** a key outside MoE ordinals 0–6 (e.g. 7, or block index 8) | `declared_rows_out_of_range`; `declared_rows_bad_layer` |
| TB14 | **(rev4, only if D40 (b))** Router-row delta pulling non-target routing toward its rows by > `traffic_steal_bound` | Refused, `traffic_stealing` |
| TB15 | **(rev4)** Declared counts at each layer size outside 4..19 / 8..38 / 16..76, **(rev5)** on layers with ≥ 1 declared row | `declared_rows_count` |
| TB15b | **(rev5)** Rows on 2 of 7 layers, each within its clamp, 0 rows on the other 5 | Accepted (ESFT-consistent) |
| TB16 | **(rev4)** Two entries on overlapping rows | The earliest reveal holds; the other shows `deferred: rows_held` with the holder and expected release |
| TB17 | **(rev4)** Opening rule (D41 default) on fixtures: tokens met + king-check pass | Gate opens; driver state records rule + evidence |
| TB18 | **(rev4)** Tokens met, king check failing (or tokens short) | Gate stays closed |
| TB19 | **(rev4)** Suite-target entry on the current sentinel suite | `deferred: sentinel_target`, not charged; judged in a later era |
| TB20 | **(rev4)** Same entry past `sentinel_max_defer_eras` | Judged on its target, logged |
| TB21 | **(rev4)** No declared rows | ESFT selection excluding held rows (both modes coexist) |
| TB22 | **(rev4)** Lane B unarmed with all new keys at default | Byte-identical digests, receipts and weights |
| TB23 | **(rev5)** Lane B leg on a cheap GPU in `[lane_b] funded_pod_skus` | Rents that type; `gpu_name` recorded; a lane A leg requesting the same type is still refused |
| TB24 | **(rev5)** GPU type in the lane B list with no measured step rate | Funding refused with a typed reason; no rental |
| TB25 | **(rev5)** `train_hours` = 6.0, and = 6.5 | 6.0 admitted with steps from the per-GPU rate and the guard ≤ `max_leg_seconds`; 6.5 refused at funding, nothing charged |
| TB26 | **(rev5)** 6 h leg funded late in an era | Held to the next era's pre-train window (cross-era admission); landing boundary shown to the miner |
| TB27 | **(rev5)** Leg record whose steps or guard exceed the consensus caps | Validator refuses the entry; satellite replay agrees |
| TS1 | **(rev4: now in §4)** Sentinel draw on two validators; recorded value replayed when the hash is unreadable | Identical |
| TS2 | **(rev4: now in §4)** Lane A with sentinel armed | Three-suite verdicts; sentinel-only regression blocked iff ratchet on |
| TS3 | **(rev4: now in §4)** Lane A sentinel unarmed | Byte-identical |
| TS4 | **(rev4: now in §4)** Lane B entry targeting the sentinel | Deferred (D35) |
| TS5 | **(rev4: now in §4)** Draw determinism and uniformity | As stated |
| T19 | Eval worker on stale code | Handshake mismatch ⇒ CRIT; nothing scored |
| T20 | Hippius primary write fails, R2 succeeds | Alert fires |
| T21 | Kill switch through the signed overlay | Share returns to lane A from the stated era; no release needed |
| T22 | Readiness note of the exact mainnet shape plus the `cfg-` segment | No pending segment trimmed |
| T23 | Old-code satellite with a new main validator | Matches until lane D arms; burns only after `from_block`; recovers after upgrade |
| T24 | Old-code trainer with a `[lane_d]` overlay published | Trainer refuses that overlay. Proves why the trainer is in S5, and that lane A stays safe while it waits |
| T25 | Upgrade from a **copy of mainnet state** (`validator_state.db`, `activation_*`, `netconfig_state`, `cascade_state.json`) | Clean start; identical next verdict |
| T26 | Shipped chain.toml through every validator and trainer startup gate at the live head | Starts (the #457/#460 class) |
| T27 | Revert overlay; stop-publishing path | **(rev5)** Overlay kill latency measured. Stop-publishing: the old vector (incl. the old lane D king) is re-asserted for `stale_after_blocks`, then the **whole** vector burns, logged (today's behaviour). **(rev6)** Change 4 is accepted: once W14 is on main, T27 adds the lane D-only burn row (stop-publishing then burns lane D only) |
| T28 | Eval host updated before and after the validator | Handshake fails closed when mismatched; works in the documented order |
| T29 | A docker-based satellite (code in `/app`) following the announcement verbatim | Upgrades and signals `cfg-` |

**Exit gate:** every row passes. **(rev3, A4) That includes T7, T7b, T7c and T14–T20 (already true in v2: "every row"), plus the new T4b, T6b, T9b and T18b.** At least one real crowning (T3) and one house
retake (T10) are observed end-to-end. No unexplained ERROR for 6 testnet eras.

---

## 5. Rollout stages (gated, no dates)

No calendar dates (owner 2026-11-21). Each stage starts only when the
previous gate is fully met; a failed gate goes back to the stage that owns
the fix.

Shaped by the rollout-mistake review (§13, §14). The worst past costs came
from four things:

- config drift between our box, main and the satellites;
- consensus changes landing before the fleet was ready;
- big-bang release bundles;
- announcements we couldn't verify.

So the order is: **de-drift → prove the overlay → catch-up release →
capability release → (rev5) one activation block, live everywhere.**

**(rev5) Go-live rule (owner 2026-11-21, D45): no shadow modes.**

- **Every feature goes live at one announced activation block, on every node,
  at once, at full effect.** That covers lane D, lane B, the sentinel, the D42
  leg options and D43. There are no `share = 0`
  eras, no log-only or "would-be verdict" periods, and no partial or
  per-validator rollout. Pay starts at the activation block.
- **What replaces shadow evidence:** CI, plus a **full testnet run
  (netuid 259)** that exercises real duels, real payouts and satellite replay
  (§4), plus the S1 measurements. Calibration that shadow eras used to provide
  (`d_margin`, gate tolerances, the sentinel) must come from testnet and S1
  **before** the block.
- **What stays (it isn't shadow):**
  - **Release-then-activate.** Six external validators run netuid 91. The
    activation overlay is published with an `effective_block` far enough
    ahead, and the **observed** `cfg-` stake must reach ≥ 2/3 (incl.
    5E2LP6) **before** that block. If it doesn't, publish seq N+1 with a later
    block. That's an upgrade check before the switch, not a trial after it.
    Without it the fleet forks.
  - **Activation at an era boundary.** Contract changes are never made
    mid-seed-epoch. Image repins follow DEC-CA-0067 (image-decoupled from
    `image_decoupled_from_block = 9493200`; legs are gated on the numerics
    digest), with 5 pins plus the wall gate.
  - **The kill switch** (revert overlay, §5).
- **Consequence:** a bug at activation pays out live until the revert overlay
  lands (up to about one era). Testnet exit gates (§4) are therefore stricter:
  every T-row must pass on testnet before the mainnet block is announced.

| # | Stage | Work | Exit gate (all must hold) |
|---|---|---|---|
| 1 | **S0 Decide** | **(rev5)** D1–D43, D45–D48 and D42b (incl. the §15 incentive [DECIDE]s D25–D32, the §16 sentinel D33–D35 and lane B D36–D42) (§6) in writing; W0 | All answered **(rev6)**: every D-number in the §17.4 register is decided, withdrawn or deferred-to-stage |
| 2 | **S0a De-drift** | Classify the 21 `LIVE OVERRIDE`s in the deployed chain.toml: main, overlay, or documented operator-only; type the decided blocks in main. Overlays merge onto **each node's own base** (`build_config(base, body)` merges onto `effective_raw(base)`, `netconfig.py:346-354`), so base drift survives any overlay | `cascade-audit decided --config /root/cascade/chain.toml` = 0 drift; the validator-read consensus-key diff between deployed and main is empty |
| 3 | **S0b Note budget** | Decode the live mainnet readiness notes and add the `cfg-` segment. It's trimmed **last** (`activation.py:1334-1357`), so older pending segments go first | No still-pending segment would be trimmed (type their decided blocks in main first if needed); T22 passes on testnet |
| 4 | **S0c Overlay bring-up** | Publish a **benign seq-1 overlay** (restates current values only) on testnet, then mainnet. Then a deliberately refused overlay on testnet | The 404-every-poll is gone. Fetched from S3 **and** the HF mirror by our validator, the trainer and satellites. `cfg-` lock ≥ 51% in `status/chain.json`. Receipts stamp `netconfig_seq`; the `cascade-audit` netconfig check passes; the trainer logs the same apply block. The refused overlay applies nothing |
| 5 | **S1 Measure** | W1, W0b, W3a (live Hippius probe); **(rev5)** start R16 long-history harvest (runs continuously from here, scored only from S8) | Forward floors met at the chosen margin (or re-decided); ≥ 1 context variant has `min_long_series`; storage mechanism chosen |
| 6 | **S2 Build** | **(rev5) All build WPs**, one PR each **(rev6.2: one implementation PR each; tests-only PRs may batch WPs per §3.1 rule 6; rev6.3: implementation PRs may batch up to 3 WPs per §3.1 rule 7)**: W1.5, W2 (with `ALLOWED["lane_d"]`) then W3–W19, W-S1–W-S3, the WB track (WB10–WB23) and W-EMA | CI green. Receipt replay on the last N mainnet receipts at the release head: **unarmed ⇒ byte-identical; armed ⇒ lane A verdicts, court and lane A weight entries identical, and lane D fields present and schema-valid** (W3b touches `bench_gate.py`/`public_guards.py`, which lane A replays). Startup-gate test of the shipped chain.toml at the live head for validator **and** trainer (T26). Docs gate |
| 7 | **S2b Catch-up release** | Ship main-as-of-now **without** lane D code to our validator, trainer and satellites, so the lane D release diffs only the lane D and lane B build WPs (§5 S2) **(rev6)** | 2 settlements with identical vectors; no satellite burn |
| 8 | **S3 Testnet** | §4 matrix including T22–T29 | §4 exit gate |
| 9 | **S4 Release** | Cut the **capability release** (lane D unarmed, `lane_d` allow-listed). **(rev5)** Bring up the launch eval host (D11; not valeval-laned, which is rented only at S8) (on-rent check, pinned known_hosts, handshake); dry-run the house. **Stage and review the revert overlay.** Send the announcement (§11.3) | Dry-run within the measured cross-host tolerance; announcement sent; revert overlay reviewed |
| 10 | **S5 Upgrade** | Snapshot all state (§10.8). Order, each through `deploy_gate check`/`verify` in quiet windows: **eval host → validator → lane D intake (systemd unit with the code-path guard) → trainer** (no legs in flight; memory-burst check; confirm preflight and worker-fingerprint lines unchanged). Satellites upgrade. Arm the broad watcher | Lane A unchanged on ≥ 2 settlements; `cascade-validator --netconfig-policy` lists `lane_d` on our nodes |
| 11 | **(rev5) S6 Activate (no shadow, D45)** | **D19 (rev5):** the owner-run test entries run on **testnet** (S3), not in a mainnet shadow. Overlay seq N, **one overlay**: `[lane_d] from_block` at a far era boundary **with `share = 0.25` (rev5, D1) and `takes_from = "lane_a"`**, all lane A keys **byte-identical** to seq N-1 (no lane A tuning in a lane D overlay, ever). ~~share = 0 shadow~~ | **Observed** `cfg-` stake ≥ 2/3 (incl. the largest holder, 5E2LP6 at about 49%) at a boundary before `effective_block`. That's the only stake-weighted proof of upgrade, because only nodes on the capability release can verify a `[lane_d]` overlay. Else publish seq N+1 with a later block. Webhook test alert delivered. Keep-alive live. **(rev5)** All §4 T-rows green on testnet before the block is announced. From the block: lane D live and paid everywhere; first paid settlement's vectors match on ≥ 2 validators, satellite replay matches, `check_lane_d` passes |
| 12 | ~~S6b Pay~~ **(rev5) merged into S6 (D45)** | — | — |
| 12b | **(rev5) NOT PLANNED.** ~~(rev3) S6c Lane B~~ (rev4: superseded by the SB track below; it was for the parked §12 variant) | ~~Overlay: `[lane_b] from_block` at an era boundary (shadow first: `[lane_b] share = 0` for ≥ 2 eras, then the D1 split)~~ **(rev6)** struck: no shadow (D45); lane B arms at SB5 | ~~TB1–TB8 passed in S3; ≥ 2 eras of valid lane B reports at share 0 with satellite replay matching; then first paid settlement's vectors match on ≥ 2 validators~~ |
| 12c | **(rev4) Lane B track SB0–SB5** (runs in parallel with lane D's S2–S3; depends on **S0a–S0c** de-drift, note budget and overlay bring-up, and on the shared S5 capability release) | **SB0** decide D35–D42, D42b · **SB1** build WB10–WB18, **(rev5)** WB22–WB23, W-S1–W-S2 · **SB2** WB19 measurement per GPU type (owner pod yes) · **SB3** testnet: TB10–TB27 (rev5), TS1–TS5, contamination check passing · **SB4** capability release (shared S5) · **SB5 (rev5, D45)** one overlay, once the D41 gate rule is satisfied and SB3 has passed (LB1): `[lane_b] from_block` with full share and `suite_targets` together, live everywhere at that block; no `share = 0` period. **(rev5)** If D40(b) is chosen: an image repin step (EMA constant fixed first; repin outside a seed epoch; moves lane A too) before SB3 | SB3 green including LB1's contamination check; **(rev5)** from the SB5 block, lane B live everywhere: first paid settlement's vectors match on ≥ 2 validators and satellite replay matches; gate-open evidence recorded |
| 13 | **S7 Operate** | §10.4 runbook; §10.7 checks; revert overlay stays staged | No pending past its latency expectation without progress; first crowning (reported immediately, with seated and judged counts) or a house hold observed end-to-end |
| 14 | **(rev3) S8 Long context** | W6-S8; rent valeval-laned (96 GB pinned torch) with owner yes; T7 at 32k/64k/128k; then an overlay adds the passing variants to `contexts` (D10) | Each added variant: T7 inside the settlement budget, W1 `min_long_series` met, overlay adopted ≥ 51% |

**Block arithmetic.** Every activation or effective block is computed with
code (`is_era_start`, the activation helpers) from the live chain and
cross-checked from two sources. There are past off-by-ones (§13).

**Kill switch and latency.**

- An overlay can't roll back: a fix is **forward only**, as seq+1 (`until_block`
  or `share = 0`).
- That needs a fresh ≥ 51% lock-in and an era start, so **kill latency is up
  to about one era (≈ 12 h) or more**.
- A reviewed revert overlay is always staged. T27 measures the latency on
  testnet.
- **(rev5) There is no lane D-only fast kill today.** A missing lane D
  report is **not** a failure for a satellite. It's a check that can't run,
  so the receipt is retried and the satellite **keeps re-asserting the last
  verified vector**, still paying the old lane D king. That lasts for
  `[satellite] stale_after_blocks = 3600` (one era) past that settlement's
  boundary. After that the satellite burns the **whole** vector, every lane,
  until a new receipt verifies (`cascade/validator/satellite.py:30-45`;
  `shared/config/satellite.py:13`). So stopping lane D reports first freezes
  lane D's pay (no kill), then burns everything.
  "Stop publishing ⇒ only lane D burns" was wrong.
  - **(rev6)** Proposed design change 4 (a `lane_d: {status: "absent"}`
    receipt state with a lane D-only burn) is accepted and built by W14.
    Until W14 ships, **the only kill is the revert overlay above**.
  - T27 measures the overlay path, and tests stop-publishing as "old vector
    re-asserted for one era, then a whole-vector burn" until W14 lands **(rev6)**.

## 6. Owner decisions (recommended defaults)

**(rev5) Owner answers, 2026-11-21.** Every recommended default in this table
is **accepted** unless a row says otherwise. The explicit answers:

- **D28/D1/D48:** an equal split, **A 25 / B 25 / C 25 / D 25**.
- **D24:** 5% per-era false-crown target (OA3 answered).
- **D46:** (a), date the house by its lineage.
- **D42b:** (a), real hours on the chosen GPU.
- **The duel reference is confirmed:** an entry must beat the lane A king
  (the house) or, once the house has been beaten, the current lane D king.
  That's KING_D as designed (§1.4).
- **D21:** stays at its default (burn), under "all other recs".
- **D16:** the standing deletion approval is granted.
- **Proposed design changes 1–17, 20 and 21:** accepted. 18 is withdrawn and
  19 is adopted as D43.
- **Not covered by these answers (needs a separate yes with cost caps):** the
  pod rentals in OA4, OA8 and §12b.6, and the data-host choices in OA5 and
  OA6. Renting pods needs an explicit yes plus a cap.

| # | Decision | Recommended |
|---|---|---|
| D1 | Share and source | **(rev5) DECIDED (owner): 25% from the total; equal split A 25 / B 25 / C 25 / D 25 when all lanes are armed.** Mechanically `[lane_d] share = 0.25`, `takes_from = "lane_a"`: lane A gets "the rest" (`lane_pool_shares`, `shared/lane_c.py:584-605`), and the king ladder works inside lane A's 25%. While a lane is off, its share stays with lane A (today's code); while lane B is armed but gated, its share burns (D37). Remainder burns |
| D2 | Context variants | **(rev3)** Launch arms {4k, 8k, 16k}. {32k, 64k, 128k} arm at **S8** by overlay (`contexts`), each only after T7 and W1 `min_long_series` (owner goal 128k kept) |
| D2b | Gate | **(rev3)** One gate suite drawn per crowning from the suites **not** held out this era; the held-out suite is the §16 cross-lane sentinel (replaces fixed "TIME unselected"); `tau_public` 0.01; anchors = KING_D and house |
| D3 | Rate limits | 1 open per hotkey; 2 open per coldkey; 8 judged per settlement |
| D4 | Eval fee | None at launch |
| D5 | Margins and floors | `d_margin` 0.01 (level units, re-checked by T14); `forward_margin` 6 h; 1000 windows / 20 sources |
| D6 | Bench capacity | Queue behind lane A; second pod if crownings slip > 1 settlement |
| D7 | Court | 2 prior miner kings, decay 0.5; house never in the court |
| D8 | Losers | Private; vault TTL 4 eras; owner-only archive 14 days; then delete |
| D9 | House paid? | No: the slot burns; prior miner tails keep paying |
| D10 | Activation | **(rev5, D45)** One overlay: `from_block` with **`share = 0.25`** (D1) and `takes_from = "lane_a"` together (no `share = 0` shadow), published with an `effective_block` far enough ahead that observed `cfg-` stake reaches ≥ 2/3 before it; kill switch by a staged revert overlay (latency up to about one era); **(rev3)** the S8 long-context arming is also an overlay (`contexts`), live at its block, no shadow |
| D11 | Eval host | **(rev3)** Launch: valpriv-4090 or another pinned-torch 2.4.1+cu124 pod (≥ 24 GB; the 16k fit is checked in B2). **Not** Blackwell sm_120. valeval-laned (H100/A100 80 GB class, pinned torch) only at S8 |
| D12 | Pool packaging | `pool` untouched (keep length never raised); **(rev3)** launch `pool_d` = `pool`'s series ids joined, history to 16384 + 64, A1 timestamps, `pool_d_hist` (no targets) to the eval pod, `pool_d_tgt` on the box; 32k–128k history work at S8; dedicated builder |
| D13 | Weights license | Open licenses only (Apache-2.0 / OpenMDW-1.1); kings are published |
| D14 | dtype | **(rev3)** BF16 only (~2.5 GB), scored in lane A's eval precision (fp32 upcast); F32 refused (4.9 GB is near the 5 GiB single-PUT limit) |
| D15 | Arch generations | Lane D generation = lane A generation; a change opens a new board |
| D16 | Standing approval | Automated vault TTL deletion and archive expiry (otherwise Approve-tier per deletion) |
| D17 | Trust | Accept that lane D loser rows are signed-but-not-reproducible (satellites verify structure and bootstrap; kings become re-scorable after publication) |
| D18 | Data path | **Targets stay on the box** (histories-only to the pod, D12), keeping the private-offload rule. Alternative: let the pod hold targets, which is a new rule; **(rev3)** launch `pool_d_hist` carries ≤ 16384 history per series |
| D19 | **(rev5)** Test entry | Owner-run test entries on dedicated **testnet** hotkeys during S3, so a real duel and payout are exercised before the mainnet block (no mainnet shadow, D45) |
| D20 | **(rev3)** King change between first clear and confirmation | (a) restart the duel against the new king (**recommended**); (b) confirm against the king at the first clear |
| D21 | **(rev3)** Unpaid lane D king slot while the house reigns | Option 1 burn (v2 default, kept) vs option 2 return to lane A. Tokenomics call, not adopted (§1.4 table) **(rev5) DECIDED: burn** (the default, accepted). |
| D22 | **(rev3)** Coldkey resolution | (b) record at `/v1/lane-d/ready`, re-check at reveal, carry in the signed report (**recommended**); (a) archive-state resolution at T |
| D23 | **(rev3)** Loser re-scoring | (ii) targets, window ids, `rows_sha256` and bootstrap verifiable; histories private (**default, as v2**); (i) publish `pool_d_hist` after the dispute window |
| D24 | **(rev3)** Per-era false-crown target (sets `d_margin` via B1) | 5% recommended (OA3); `d_margin` stays 0.01 until B1 runs **(rev5) DECIDED: 5%** (owner; OA3 answered). |
| D25 | **(rev5 row; text in §15)** Court rent and salami-slicing | See §15 D25 |
| D26 | **(rev5 row; text in §15)** Lane A as a free input to D; D's share out of A | See §15 D26 |
| D27 | **(rev5 row; text in §15)** Withholding generators from A | See §15 D27 |
| D28 | **(rev5 row; text in §15)** Lane split (A/B/C/D) | See §15 D28 **(rev5) DECIDED: A 25 / B 25 / C 25 / D 25** (owner). |
| D29 | **(rev5 row; text in §15)** Frequency bias in frozen sets | See §15 D29  **(rev5)** Partly addressed by `max_source_share` (B14, W5b). |
| D30 | **(rev5 row; text in §15)** TIME isn't truly unselected (now handled by the §16 sentinel, D33) | See §15 D30 |
| D31 | **(rev5 row; text in §15)** Sybil queue griefing | See §15 D31 |
| D32 | **(rev5 row; text in §15)** Lane B vs D | Resolved in rev4 (§12b.7); see §15 D32 |
| D33 | **(rev3)** Cross-lane rotating sentinel (§16) | **Adopt**: one suite per era, drawn from the era-boundary block hash, held out of every gate and target in lanes A, B and D, swept and published for every new king. Alternative: no sentinel (v2: lane A gates all four; lane D holds TIME out) |
| D34 | **(rev3)** Sentinel ratchet vs the house | **Optional, recommended on**: every lane's candidates must not fall below the current lane A king on the sentinel by more than `sentinel_tau` (0.01), re-scored paired on one host. Off = the sentinel is publish-only, and a regression on it alone is never blocked |
| D35 | **(rev3)** Sentinel vs DEC-CA-0071 (pending: PR #482; rev5 renumber) lane B suite targets | While a suite is the sentinel it is **not** a lane B target or guard that era. Entries already declared on it are judged on the remaining suites, or deferred to the next era. Recommended: **defer** (the target stays honest) |
| D36 | **(rev4) resolved:** lane B arms **trainer-run** (generator + optional declared rows + optional suite target). The miner-weights variant (§12) is not planned |
| D37 | **(rev4)** Lane B share and what it does before the gate opens | Deployed `[lane_b] share = 0.25` (A/B 75/25 while C off; DEC-CA-0050 burns empty slots). **Assumption stated:** while lane B is armed but its opening gate is closed, every lane B slot is empty, so **its whole share burns**. Recommended: set `[lane_b] from_block` only once the gate rule (D41) is satisfied, or arm at `share = 0` until the gate opens (SB5). The split itself is D28. **(rev5, D45)** No `share = 0` arming: `[lane_b] from_block` is set only once the D41 gate rule is satisfied, and lane B goes live at full share at that block |
| D38 | **(rev4)** `max_repo_mb` for lane B suite-target entries (LB1) | (a) 8 MB via `[lane_b] suite_target_max_repo_mb` (**recommended**); (b) keep 128 MB (`chain.toml:47`) |
| D39 | **(rev4)** Per-layer expert clamp (LB2) | Fractions `min_experts_frac = 0.03125`, `max_experts_frac = 0.1484375` ⇒ 128: 4..19, 256: 8..38, 512: 16..76 (**recommended**); or fixed per-layer lists |
| D40 | **(rev4)** Router rows (LB2) | (a) frozen (**recommended for first arming**; no image change); (b) FlexOlmo unfreeze of the declared rows' router entries and bias, with the `traffic_steal_bound` guard. **(rev5)** (b) changes on-pod `expert_training.py` ⇒ a **shared worker-image repin that also moves lane A** + DEC-CA-0067 qualification, and since any image ≥ rc16 arms power EMA for lane A, (b) is **blocked on fixing the scored EMA constant first** (§3 Services touched)  **(rev5)** Depends on W-EMA (§8). |
| D41 | **(rev4)** Lane B opening rule (LB4) | (a) `open_tokens` + a passing lane A king check (**recommended**); (b) `open_tokens` alone; (c) co-gate with re-measured thresholds |
| D42 | **(rev5) DECIDED (owner 2026-11-21):** expert-leg cost (LB6) | **Cheaper GPUs allowed** via a lane B-only `[lane_b] funded_pod_skus` (each type image-qualified and measured first), and **miner-chosen leg length** `train_hours` 0.5–6.0 h (`[lane_b] max_leg_hours = 6.0`), guard capped by a new `[lane_b] max_leg_seconds`, never by editing `[training] max_train_seconds`. Operator funding not adopted. §12b.6 |
| D42b | **(rev5)** What `train_hours` means (§12b.6) | (a) wall hours on the chosen GPU, steps fixed at admission from the measured per-GPU rate (**recommended**); (b) reference hours, steps = 23,841 × h / 3 on any GPU **(rev5) DECIDED: (a) real hours on the chosen GPU** (owner). |
| D43 | **(rev5) DECIDED (owner 2026-11-21):** lane D reveal timing | **Immediate reveal** (`min_blocks`), not lane A's boundary − 25 blocks; T = the observed reveal block. Starts the forward clock up to ≈ 3 h earlier. §1.2 step 4, W10. (The commit-inclusion-block alternative is not adopted) |
| D44 | **(rev5) WITHDRAWN (owner 2026-11-21)** | Power EMA instead of cooldown was drafted and then dropped by the owner. Lanes A and B keep today's cooldown. Not planned |
| D45 | **(rev5) DECIDED (owner 2026-11-21):** go-live rule | No shadow modes: every feature goes live at one activation block on every node at full effect. Testnet + S1 replace shadow evidence; release-then-activate (observed stake ≥ 2/3 **before** the block) and the kill switch stay. §5 |
| D46 | **(rev5, C1)** House T, which fixes starvation (§1.3, §1.4; interacts with D20) | Today the house's T resets on every KING_A change (a dethrone or a promotion; the king leg trains once per seed epoch, `shared/era.py:440-457`), so pending entries restart. If lane A changes king faster than lane D's commit-to-crown latency (about 2–8 days with confirmation), no challenger can crown. (a) **T_house = the latest commit (reveal) block of any generator in its lineage (recommended)**; (b) an entry keeps the T_K it froze against; (c) unchanged. B2 measures house refresh frequency **(rev5) DECIDED: (a)** (owner). |
| D47 | **(rev5, C2)** House precision under BF16-only | (a) **score the house as a BF16 cast of KING_A, and dedup entries against BF16 casts of every published checkpoint (recommended)**, so the house plays by the entry rules and a cast copy can't dodge dedup; (b) the house stays F32 |
| D48 | **(rev5, C3)** Source of lane D's share inside lane A (W8) | (a) before the ladder (cuts the lane A king); (b) from the ladder's burned remainder: free to lane A miners, but D's budget varies with the king's ladder share (10% after a fail, up to 95%, `chain.toml:1223-1226`) and can be ~5%; (c) from the total. W8 changes `lane_pool_shares` (`shared/lane_c.py:584`), a consensus signature at every caller **(rev5) DECIDED by the equal split:** fixed 25% shares of the total. Lane D takes from lane A, whose 25% is the pot the ladder works within, so lane D's budget never varies with the ladder. |

---

## 7. Phase 2: lane E (test-time optimisation), sketch

- **Entry:** a test-time procedure on KING_A with fixed weights (view
  ensembling, context adaptation, recalibration).
- **Judging:** on the private pool vs KING_A plain, with "improve one, regress
  none" guards (DEC-CA-0071 (pending: PR #482; rev5 renumber) pattern), at ≤ 4× base inference compute.
- **Blocker:** lane E runs miner code, so it needs a sandbox (no network,
  read-only weights, time and memory caps).
- **Phase 2a:** a fixed menu of procedures we implement, with miners choosing
  parameters only. No miner code; it proves the eval-cost model first.

## 8. Phase 3: lane A recipe menu, sketch

- **Declared options:** window and context sampling, domain mix weights,
  cooldown fraction plus an optional higher-LR continuation, and power-EMA
  averaging. **(rev3) Correction:** power EMA isn't a config knob.
  It's **image-only** (`POWER_EMA_SIGMA_RELS = (0.05, 0.10)`, hard-coded in
  `pyramid_trainer.py`, owner decision 2026-11-01). The contract's
  `power_ema_sigma_rels = []` only pins the contract value. The live rc15
  image has no EMA code, but **the first repin to rc16 or later turns EMA on
  automatically** at the next warm-start seed epoch, scoring the σ_rel 0.10
  average. That would make warm-start legs look worse. Fix the scored
  constant (0.05, raw, or cooldown-only averaging) **before** that repin.
- **(rev5) WP W-EMA (blocks every repin to an image ≥ rc16, including any
  image #477's NCCL pin ships in if it's built from main):**
  - The next image **scores the raw cooled endpoint on every leg**, in lanes
    A and B. `power_ema_active` is changed so warm-start legs behave like
    random-init legs do today: `weights.safetensors` is the raw endpoint, and
    `weights_ema005` / `weights_ema010` are written alongside, **unscored**.
  - The main validator scores the EMA files **off the critical path** and
    publishes them as telemetry only. They never gate, never pay and never
    enter a receipt's verdict (so D45 isn't affected: it's telemetry, not a
    rollout mode).
  - **Never "best of both" per model.** The owner decides later whether to
    switch the scored file, as one activation for every leg (D45).
  - Tests: a warm-start leg's scored file is byte-identical to the raw
    endpoint; both EMA files are present; a lane B leg from an EMA-marked king
    still scores raw.
- **Reproducibility:** options are folded into each leg's effective contract
  and replayed by the trainer, validators and audit.
- **Cost:** it's a lane A contract change (image, repin and seed-epoch rules;
  release-then-activate), so it has the highest blast radius of the three.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Too few forward-eligible windows, or long latency | W1 first; floors make it `pending`, never a guess; add sub-hourly sources |
| First-seen rule weaker than assumed (mirror lag > margin) | Margin ≥ the measured scrape lag; T8 forecast-feed and backfill cases |
| Main-validator centralisation of lane D scoring | Signed rows; satellite bootstrap replay; structural checks; kings re-scorable; D17 |
| **(rev3)** Coldkeys are free | The A8 caps limit casual flooding only, not a determined Sybil (one IP drove ~70 hotkeys across coldkeys; **(rev5) unverified:** the `OPSLOG:164` line ref has drifted and a grep for the fact didn't re-find it). Registration cost, the eval fee (D4), and the §15 pre-screen [DECIDE] are the levers. k = 32 per era is sized for this in §1.3 |
| **(rev3)** Owner conflict of interest | The owner runs the judging (main) validator **and** enters the house. Mitigations: the house is never paid (D9/D21); verdicts carry `rows_sha256` and signed rows; satellites re-run the bootstrap; `check_lane_d` re-derives the judged set and eligibility; kings are public and re-scorable; D23 (i) would let losers fully re-score; the builder's per-snapshot metadata digest is published (§15 / review round 2 item 12). **(rev5)** The first-seen log's daily Merkle root is committed before any snapshot uses it (W6fs), so we can't backdate first-seen to admit or exclude windows after seeing entries |
| **(rev5)** Rented eval host sees private data | Loser weights and `pool_d_hist` (histories, no targets) are visible to the rented (Lium) eval host. That's the same accepted risk as lane A's private offload: targets and scoring stay on the box, the host gets presigned GETs only, and losers are deleted on TTL |
| **(rev3)** The low-frequency check is weak against contamination | The private pool isn't forward-only, so a contaminated model passes A10's check more easily. It catches honest regressions only |
| **(rev4)** Procedurally recreated benchmark series in lane B | The static contamination check (LB1) can't see a series regenerated from fitted parameters. **Guard 4** (whole-pool point gain ≥ 0) and the **anchor guard** are the backstop: a gain that exists only on a public test set is refused before merge |
| Bench-pod contention and disk | Lane A first; byte LRU; second pod on demand |
| Eval GPU can't hold 64k/128k | Arm only measured variants; larger pinned-torch GPU later |
| Satellites lag the upgrade | Stake-weighted gate; later boundary |
| Scope pressure | No dates; arm a subset of contexts rather than drop a test row |

---

## 10. Operations: everything that has to run, end to end

### 10.1 What runs today (inventory 2026-11-21)

| Component | Where | Role |
|---|---|---|
| Trainer (rc41), validator (rc37), intake (`cascade.funding.main` :8790 behind Caddy, 130 MB cap), cascade-ops follower and health timer, three guards | prod box: 4 cores / 7 GB RAM / 154 GB disk (50%) | lane A, monitoring |
| valeval-blackwell4 | Lium RTX PRO 6000 96 GB, $1.48/h, driver 595; **benchmarks sidecar venv** (torch 2.12.1+cu130) | public bench sweeps |
| valpriv-4090 | Lium RTX 4090 24 GB, $0.60/h; pinned venv | lane A private eval offload |
| Lane A training pods | miner keys | lane A legs |
| Pool | GitHub Actions daily; Hippius pin; reveal at ~48 h | private windows |
| Storage | Hippius Hub/S3, R2, HF fallback; vault = local dir on the box | artifacts |
| Validators | ours (main, full) + satellites | weights |

Three `lium up` CLI processes on the box have been running for 3.6–8.3 days.
They look like hung rental calls. Check on-pod and confirm ownership before
anything else touches them.

### 10.2 What lane D adds

- **Training images:** none. The lane A image is unchanged, and lane D never
  triggers a repin.
- **Miner kit** (public): `cascade lane-d export | verify | submit | ready |
  status`; the latest public KING_A as a start point; a reference fine-tune
  script (runs on the miner's GPU).
- **(rev3) Launch eval host:** valpriv-4090 or another pinned torch 2.4.1+cu124 pod (≥ 24 GB; the 16k fit is checked in B2, OA4). **valeval-laned is rented only at S8:**
  - Hardware: a **pinned-venv-capable GPU** (H100 or A100 80 GB; not
    Blackwell sm_120); ≥ 300 GB disk for the prefetch cache and
    `pool_d_hist`.
  - Software: the release's `uv sync --frozen` venv (torch 2.4.1+cu124),
    re-synced with every validator deploy, **with the worker handshake**.
  - Access: pinned known_hosts per pod, and an on-rent check (`nvidia-smi -L`,
    VRAM, driver, disk) before wiring.
  - Cost: about $1.3–2.5/h (Lium pricing varies; measured at rent).
- **Data paths** (nothing large through the box):
  1. Miner → vault bucket: direct multipart (§1.2).
  2. Vault → **(rev5)** the launch eval host (valeval-laned from S8): presigned GET prefetch at reveal; sha-keyed LRU
     with KING_D and the house pinned.
  3. Builder → Hippius: `pool_d_hist` (to the pod) and `pool_d_tgt` (to the
     box).
  4. Pod → box: forecasts only. The box scores against `pool_d_tgt`, then
     signs the report to S3 and R2.
- **Box load:** the lane D intake service (small), forecast scoring at h = 64
  (small), the bootstrap. **No weights, histories or torch.** `box_ram` is
  watched.
- **pool_d builder:** a dedicated small CPU VM with its own disk. It holds
  `TSFORGE_S3` and Hippius pin credentials only; never the prod box, never
  rented GPU pods.
- **Satellites:** no new hardware; the release before the activation block.

### 10.3 Testnet stack (separate from production)

- **testnet-ctl** (cheap CPU VM): testnet validator, lane D intake,
  cascade-ops; RC commit; testnet wallets only.
- **testnet-eval:** a **24 GB pinned-torch pod (rev3: launch contexts ≤ 16k)**, rented just-in-time for the T7
  rows and terminated after an on-pod check. A cheaper pinned-torch GPU
  handles the other rows.
- **Testnet `pool_d`:** hourly builds on the builder, testnet prefix.
- **Testnet bench:** valeval-blackwell4 **only outside lane A's king-check and
  dethrone windows** (check `v2_check_due`), or reduced-series suites on
  testnet-eval.
- **Test miners:** us. T3 needs one rented GPU for a few hours with **public
  data only**. T5 artifacts are made by script.
- Every testnet pod gets a TTL and verified teardown.

### 10.4 Per-settlement runbook

```
boundary B
 ├─ refresh house if KING_A digest changed (T = its publish block)
 ├─ newest pool_d_hist/_tgt pinned? (stale > 30 h ⇒ CRIT; entries stay pending)
 ├─ for each open entry: update eligible union (§1.3, max(T_E, T_K) + margin)
 │     → frozen this settlement? → cohort (intersection; defer latest-T if short)
 ├─ eval host (rev5: launch eval host; valeval-laned from S8): forecasts for cohort + KING_D + house on the frozen set
 │     (handshake first; cached rows re-used by (digest, context, window id))
 ├─ box: score vs pool_d_tgt → bootstrap → max-T → candidate or none
 ├─ candidate: drawn suite (block hash) on valeval-blackwell4, behind lane A,
 │     paired vs KING_D + house; sentinel after crowning (publish only, rev5)
 ├─ signed lane D report + lane_d_state.json → receipt digests → weights
 └─ crowned: publish king (async mirror; never gates the verdict); index; board
```

| Fault | Outcome |
|---|---|
| Eval pod down; vault or Hippius unreadable | Entries stay queued or pending, no cost; CRIT after 1 settlement |
| Bench pod down before the candidate's sweep | Crowning held; CRIT after 1; after `max_bench_holds` (4) the candidate re-queues unjudged. Deliberately not lane A's never-timeout hold |
| Candidate sweep error, partial, NaN, missing metric | `checkpoint_fault`: entry refused; evidence kept |
| Handshake mismatch (stale eval code) | Nothing scored; CRIT; stderr captured |
| `pool_d` stale or missing | Entries pending; CRIT |
| Validator restart | Idempotent from state and reports |
| Hippius primary failure with R2 success | Alert (fallback must not hide it) |

Pre-staged, reviewed **owner-paste scripts** cover **(rev5)** launch-eval-host (and, from S8, valeval-laned) setup and
recovery, because remote writes are often classifier-blocked.

### 10.5 Operations checklist by stage

| Stage | Action | Needs |
|---|---|---|
| S1 | W1 on GH Actions or the builder; W3a storage answer | — |
| S3 | Rent testnet-ctl, testnet-eval and the builder; deploy the RC; hourly testnet `pool_d`; arm | Owner yes on rentals |
| S3 | T3 fine-tune pod and T7 pod just-in-time; terminate after on-pod checks | Owner yes |
| S4 | **(rev5)** Bring up the launch eval host (D11; valeval-laned is rented only at S8) (on-rent check, known_hosts, handshake); dry-run; add `laned-eval.toml` to **`operator_inventory_host_files`** and **`operator_standing_pods_usd_per_hour`** (that list still names valeval-blackwell2; an Approve-tier chain.toml edit through the Edit tool, typed in main the same day, never left as a live-only override) | Owner yes |
| S4 | Cut the release; the owner sends the validator announcement (release, activation block, "no new hardware", how to verify) and the miner announcement | Owner |
| S5 | Snapshot **all 8 state files** first (§10.8: `validator_state.db`, `cascade_state.json`, `warm_start_init.json`, `lane_d_state.json`, `netconfig_state.json`, `activation_*_state.json`, the satellite state file, `decided_votes.toml`; **(rev6.1)** plus the trainer state under `/root/cascade/_train_work/` (`era_state.json`, `trainer_submissions.json`, `funded_queue.json` and the rest of the §10.8 list), `provisioner_state.json`, `pod_monitor_state.json` and the deployed `chain.toml` with its live overrides, `hosts.toml`, `provision.toml`). Then restart in order **eval host → validator → lane D intake → trainer** (no legs in flight), each with `python -P`, the code-path guard and `deploy_gate check`/`verify` (R8). **Outside settlement, harvest and scoring windows; never during a bench sweep; never Wed 22:00 – Thu 08:00 UTC.** Satellites upgrade per the announcement | Notify (plan and timing to the owner first) |
| S5 → S6 | Go/no-go on the stake gate; webhook test alert delivered | Owner |
| S6–S7 | First settlements watched end-to-end; first crowning reported immediately; daily load, cost and vault size | — |

### 10.6 Cost (operator-paid; training stays miner-paid)

| Item | Recurring | One-off |
|---|---|---|
| **(rev3)** Launch: valpriv-4090 (shared, outside lane A duel windows) **or** a dedicated 24 GB pinned-torch pod | ~$0 extra (shared) or ~$14–20/day (dedicated 4090-class, measured at rent) | — |
| **(rev3) S8 only:** valeval-laned (80–96 GB pinned-torch GPU) | ~$31–60/day (measured at rent) | — |
| pool_d builder (CPU VM + disk) | ~$2–5/day | — |
| Second bench pod (only if D6 triggers) | ~$36–46/day | — |
| Testnet ctl + eval + T3/T7 pods | — | per testnet day ~$10–30 plus a few hours of 80 GB GPU |
| Vault + `pool_d` storage | small | — |

### 10.7 Monitoring (cascade-ops, W11)

- `lane_d_report`: a signed report each settlement.
- `lane_d_pending`: a pending entry past its latency expectation with no
  progress.
- `lane_d_eval_pod`: reachable, handshake OK, disk ≥ cache budget, GPU
  visible.
- `lane_d_bench_hold`: a held crowning.
- `pod_disk` for valeval-blackwell4 under the added lane D load.
- `pool_d_stale`: the newest snapshot is older than about 30 h.
- `runway`: the balance covers ≥ 3 days of **all** standing pods.
- `primary_store_failed`.
- `vault_lane_d_bytes`.
- `box_ram`.
- **A follower pattern for every lane D log line.**

**Alert delivery.** `/etc/cascade-ops.env` is empty, so alerts reach only
`alerts.jsonl`. The owner fills a webhook (I never write that file), and a
delivered test alert is part of the S6 gate.

### 10.8 Rollback

- **Config:** forward-only overlays (§5 kill switch). The revert overlay is
  staged and reviewed before S4.
- **Code:** keep the previous release worktree and restart script for every
  service (validator, intake, trainer, eval host). A rollback goes through the
  same quiet-window rules and the same `deploy_gate` verify.
- **Snapshots before every S5 restart:** `validator_state.db`,
  `cascade_state.json`, `warm_start_init.json`, `lane_d_state.json`,
  `netconfig_state.json`, `activation_*_state.json`, the satellite state
  file, `decided_votes.toml`. **(rev6.1)** Plus (RV3 note 21) the trainer
  state under `/root/cascade/_train_work/`: `era_state.json`,
  `trainer_submissions.json`, `funded_queue.json`,
  `trainer_promotion.json`, `champion_publisher.json`,
  `lane_b_driver.json`, `coldkey_identities.json`,
  `published_settlements.json` and its own `activation_*` and `netconfig`
  files; `provisioner_state.json` and `pod_monitor_state.json`; and the
  deployed `chain.toml` (with its live overrides) together with
  `hosts.toml` / `provision.toml`. Copy them under `_state_backups/` with a
  timestamp.
- **Never restore validator state across a settlement.** A restore is only
  valid back to the last settlement it already covers. Otherwise forward-fix.
- **Lane D only:** revert overlay, forward-fix, republish. Lane A never
  depends on lane D state. **(rev5)** Stopping lane D reports doesn't stop
  lane D's pay. Satellites re-assert the last verified vector for
  `stale_after_blocks` (3600, one era), then burn the **whole** vector
  (`satellite.py:30-45`). It's not a lane D kill until Proposed design change
  4 ships.

---

## 11. Artifacts, publication and documentation

### 11.1 Where checkpoints live

| Artifact | Location |
|---|---|
| Lane A legs and kings | Hippius Hub `{ns}/ckpt-r{base_seed}-{role}-{size}{suffix}@sha256` (`loop.py:1030,1078,1410`); R2; HF fallback; non-kings expire after 14 days. Unchanged; the house reads KING_A by its existing ref |
| Lane B merges | `ckpt-r{rid}-expert-{label}` (`lane_b_driver.py:777`). Unchanged |
| Lane A king generator | `champions/<digest>.zip` (`champion_publish = "crown"`) and repo `champions/king/` via `king-sync.yml`. Unchanged |
| Lane D entries **(rev3: BF16, ~2.5 GB each)** | private bucket `cascade-lane-d-vault/lane-d/<hotkey>/<digest>/`; private R2 mirror; TTL then owner-only archive (§1.2) |
| Lane D kings | public Hippius Hub `{ns}/laned-king-c{context}-{hotkey8}@sha256` (the same digest as the vault), R2 public mirror, HF mirror `tensorlink-dev/cascade-lane-d-kings`; kept permanently |
| Lane D index | signed `champions/lane_d/index.json`: every king, context, digest, crown block, LCB, gate and **(rev5)** sentinel numbers (suite named), license |
| Repo cards | `champions/lane_d/<digest>/{PROVENANCE.json, MODEL_CARD.md}` via `laned-king-sync.yml`; no weights in git |
| `pool_d_hist` / `pool_d_tgt` | Hippius pins; sha256 in every lane D report; `pool_d_tgt` and window ids published after the dispute window |

One digest, one identity: vault, public repo, index and report must agree,
and the audit checks it.

### 11.2 Documentation and agent surfaces (ship with the release; S2 docs gate)

| Surface | Change |
|---|---|
| `docs/LANE_D.md` (new) | what lane D is; checkpoint format; context as a declared scorer setting; export, verify, submit, ready, commit; dedicated hotkey; **(rev5)** the licence attestation at `ready` (weights and training data licensed for Apache-2.0/OpenMDW publication; no restricted-licence teacher outputs, e.g. Timestral under MarketCast's `[marketcast]` licence terms) and the takedown path; forward-only rule and **expected latency by frequency**; house; gate and **(rev5)** the rotating sentinel; pay and court; license; FAQ; **(rev3)** coldkey caps and the typed skip reasons, confirmation-duel latency, both clocks and mirror lag, BF16 only, launch contexts 4k–16k (S8 later), max reveal delay |
| `docs/INTERFACE.md` | a lane D checkpoint interface section |
| `docs/MINER.md` | lane table, link, "no compute keys needed for lane D" |
| `docs/AGENT_MINING.md` | lane D agent workflow; **local scores on the revealed pool overstate forward gains** |
| `AGENTS.md`, `agent.md` | lane D role; human authorization still needed for spend and publish |
| `llms.txt` (repo + website) | lane D links; fix the stale "Planned order A → C → B" and activation text |
| `docs/VALIDATOR.md` | lane D replay: what satellites verify and what they can't; **(rev3)** the new report fields (`rows_sha256`, `coldkey`, judged set, `confirmation`, low-frequency result, both clocks) |
| `docs/OPERATOR.md` | **(rev5)** the launch eval host (valeval-laned at S8), builder, vault, retention, runbook, kill switch; **(rev3)** launch eval host (pinned torch, not Blackwell), S8 long context, the first-seen recorder (W6fs), the coldkey alert |
| **(rev4) Lane B docs** | `docs/MINER.md` lane B: declaring rows, suite targets, the contamination refusal, deferrals. `docs/INTERFACE.md`: the ban on embedded benchmark series and the `declared_rows` format. `docs/AGENT_MINING.md`: using the routing artefact (LB3). `docs/VALIDATOR.md`: what validators check (hash-set version, declared rows, the opening rule) |
| `docs/ARCHITECTURE.md` | lane D flow |
| `status/lane_d.json`, website | queue, pending progress, king, board; **(rev3)** per-entry reason enum, confirm stage, reveal deadline |
| `cascade status --json`, `discovery-check` | lane D activation, contexts, king, endpoints |
| `README.md`, `CHANGES.md` | summary, release notes |
| Decisions | DEC-CA-0070 new; DEC-CA-0062 superseded |

Docs gate: all of the above in the release PR set, links resolve, and
`llms.txt` matches `cascade status --json` on testnet.

### 11.3 Other items

- **Arch generations:** D15.
- **Announcements:** drafted at S4; the owner sends them. Validator
  announcement content:
  - the release commit and docker tag;
  - "pull main's chain.toml, no local edits, keep every state file";
  - expected confirmation lines: `cascade-validator --netconfig-policy` lists
    `lane_d`, `activation: signalled … cfg-<hex>`, `git log -1`, satellite
    status `mode` / `last_verified_round`;
  - the overlay seq and digest and the expected lock boundary;
  - what lagging costs (burn after `from_block`);
  - the deadline: one era before `effective_block`.

  We track a per-hotkey, stake-weighted checklist from the chain notes. A
  feature name is **never renamed after it's announced**.
- **Support:** `status/lane_d.json` shows each pending entry's eligible-window
  progress against the floors.
- **Leaks:** a validator operator leaking private weights gains nothing in
  lane D (copies can't beat the margin), but losers stay private and
  short-lived regardless.

---

## 12. Lane B variant: miners submit expert weights — **not planned (rev4: parked)**

**(rev4)** Parked by owner direction (2026-11-21): miner-trained rows can train directly on public test sets, which breaks suite targets and lets contamination into the lineage. Lane B stays trainer-run with miner-chosen experts (§12b). The text below is kept for the record; WB1–WB7 and TB1–TB8 in this section are **not planned**.


~~(rev3) In scope: built, tested and armed as part of this plan, armed at S6c.~~ **(rev5) Not planned: parked in rev4 (see the heading). The S6c stage is gone; lane B arms on the SB track (§5 row 12c).** Historical rev3 text follows. **(rev3)** Caveat: most low-frequency domains will rarely reach the forward-only floors, so forward-only judging of per-domain expert deltas works mainly for sub-hourly domains. It reuses lane D's vault, forward-only
judging and GPU ingest.

- **Entry:**
  - the base-king digest (≤ `max_base_age_eras` old);
  - the declared (layer, expert) rows;
  - optionally those rows' router and route-bias entries (FlexOlmo pattern;
    revives dead experts);
  - a safetensors delta of only those rows;
  - the declared target (DEC-CA-0059 / 0069).
- **Verification** uses existing code: `merge_expert_slices` (byte-check),
  `rebase_expert_delta` onto the current king, then re-run the verdict and
  guards. No training on our side.
- **Changes vs today's lane B:**
  - Rows are miner-claimed, held from `ready` to verdict or merge, earliest
    reveal wins.
  - Judging moves to forward-only windows (miners can train on anything).
  - No compute funding.
  - Router-row deltas need a deliberate traffic-stealing test case blocked by
    the guards.
- **Interaction:** a won delta merges into KING_A at the era boundary. It's
  the only path by which miner-trained weights reach lane A, bounded by the
  byte-check, the rebase and re-verification.
- **New design:** this is *not* DEC-CA-0071 (pending: PR #482; rev5 renumber)'s Open
  option (b). That option is miner-declared rows with trainer-run training.
  **(rev3) WB7 writes the DEC-CA-0071 amendment** that records this variant,
  the §16 sentinel rule (D35) and forward-only judging.

**(rev3) Execution in this plan**

- **Deployed state:** `[lane_b] from_block = 0` (off) and `leg_funding =
  "miner"`. The A/B split is 75/25 while C is off.
- **Arming:** `[lane_b]` is fully overlay-settable (`ALLOWED["lane_b"] =
  None`, `netconfig.py:83-87`; `remainder` stays release-only), so (rev5: historical, parked variant) S6c arms
  lane B **by signed overlay**, with no validator action. The capability
  release (S5) carries the lane B code.
- **Existing code reused:**
  - `merge_expert_slices` and `rebase_expert_delta` (`pyramid_model.py:706,
    738`) for the byte-check and rebase;
  - `check_lane_b` (`cascade/audit/checks.py:2887`) for the audit;
  - lane D's W3 intake, W3a vault, W4 ingest, W5a/W5b forward eligibility,
    W7 report pattern and W15 prefetch.

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| WB1 | Expert-delta submission format (base-king digest, declared (layer, expert) rows, optional router and route-bias rows, BF16 safetensors delta of only those tensors, declared target) + intake via the lane D service (`/v1/lane-b/submit`, `/ready`) | W3, W3a | ci + testnet | `scripts/laned/accept WB1` |
| WB2 | Verification: `merge_expert_slices` byte-check (only declared rows differ from the base king); `rebase_expert_delta` onto the current king; `max_base_age_eras`; re-run verdict and guards on the rebased model | WB1, W4 | ci + pod | `scripts/laned/accept WB2` |
| WB3 | Row claims: rows held from `ready` to verdict or merge, earliest reveal wins; **row-level dedup against published KING_D rows** (D32; cosine/L2 within ε ⇒ refuse) | WB1, W13 | ci | `scripts/laned/accept WB3` |
| WB4 | Forward-only per-target judging for lane B (lane D's W5a/W5b eligibility on the target's windows; floors; the confirmation duel per D20) + guards on every other domain and suite + §16 sentinel exclusion (W-S2) | WB2, W5a, W5b, W-S2 | ci | `scripts/laned/accept WB4` |
| WB5 | Lane B report fields (rows digest, claims, rebase base digest, merge identity), satellite replay and the `check_lane_b` extension | WB4, W7a, W9 | ci | `scripts/laned/accept WB5` |
| WB6 | Lane B docs: `docs/MINER.md` and `docs/AGENT_MINING.md` lane B sections for expert deltas, the LANE_D.md cross-reference, VALIDATOR.md/OPERATOR.md lane B notes | WB5 | ci | `scripts/laned/accept WB6` |
| WB7 | DEC-CA-0071 (pending: PR #482; rev5 renumber) amendment: expert-delta variant, forward-only judging, the D35 sentinel rule | — | ci + owner merge | `scripts/laned/accept WB7` |

| Test | Case | Expected |
|---|---|---|
| TB1 | Delta touching an undeclared row | Refused at the byte-check |
| TB2 | Delta trained on a base king 1 era old | Rebased onto the current king and re-verified; a base older than `max_base_age_eras` is refused |
| TB3 | Router-row delta that pulls other domains' traffic | Blocked by the other-domain guards |
| TB4 | Two entries claiming overlapping rows | The earliest reveal holds the claim; the other is refused with a typed reason |
| TB5 | Delta copied from a published KING_D's rows | Refused by the row-level dedup |
| TB6 | Lane B unarmed | Byte-identical receipts and weights (invariant 1) |
| TB7 | A lane B entry targeting the current sentinel suite | Deferred (D35) |
| TB8 | A won delta merges at the era boundary | KING_A updated; the house refreshes in lane D (§1.4) |

---

## 12b. Lane B (rev4): improve one, regress none, miner-chosen experts

**Direction (owner, 2026-11-21).** Lane B stays trainer-run:

- the miner submits a **code-only generator**, and our trainer trains the leg;
- the miner may also **(1) choose which expert rows are trained**
  (DEC-CA-0071 Open option (b));
- and **(2) target one public suite** (DEC-CA-0071 suite targets): improve it
  significantly while regressing nowhere — other suites, every private-pool
  domain, every horizon, the whole-pool point gain, and the fixed anchor king.

The rules below are recorded in the DEC-CA-0071 amendment
(`decisions/DEC-CA-0071-lane-b-public-suite-targets.md`, §"Amendment rev4").

### 12b.0 What carries over unchanged from DEC-CA-0071

- **Gain test:** CRPS `LCB_α ≥ gain_margin_public` (0.01) on the whole suite;
  MASE non-inferior; at least `min_public_items` items.
- **Guards:** every other suite, every pool domain, every horizon; the
  whole-pool point gain must be ≥ 0 (guard 4); the fixed anchor king (guard
  5).
- **Pay:** 7 domains + 3 horizons + 4 suites = 14 slots, weighted by
  `target_weights`, with decay, TTL, burn and merge-once.
- **Merge:** `merge_expert_slices` byte-check and `rebase_expert_delta`
  (`pyramid_model.py:706, 738`), re-verified on the current king.
- **Judging:** `judge_expert_targets` (`cascade/eval/expert_lane.py:249`).
- **Training backend:** the compact-rows backend. Every parameter is set
  `requires_grad_(False)` and only the selected rows get trainable leaves
  (`expert_training.py:3-6, 112-122`); the router and shared weights are
  reused as-is (`:49-51`).
- **Legs:** `[lane_b] leg_hours = 3.0`, `leg_funding = "miner"` (deployed).
  **(rev5)** Replaced by miner-chosen `train_hours` (0.5–6 h) on a lane B GPU
  list (D42, §12b.6); `leg_hours` stays the default when a miner sets none.

### 12b.1 LB1: contamination check — a precondition for arming

- **Arming rule.** `[lane_b] suite_targets` may arm **only after** the static
  contamination check ships **and passes on testnet** (stage SB3). It's a
  gate, not an open item.
- **What it checks.** Hashes of every GIFT-Eval, BOOM, TIME and FEV **test
  window**, in two forms:
  - **exact:** sha256 of the float64 values;
  - **normalised (rev6.2, owner 2026-11-22; WB11 schema 2,
    `cascade_contamination_hashset/2`, encoding `cascade-zgrid/2`; PRs
    #617/#621, #632):** sha256 over sliding sub-windows of 32 points, each
    window z-scored with its own mean and **population** std, every value
    binned as `floor(z / 0.5)`. At lookup, a value within 0.03 of a bin edge
    also probes the neighbouring bin (nearest edge first, at most 12 such
    values per window, at most 4096 probes per window); both the window and
    its negation are looked up. This is robust to scale, offset and
    rounding. It replaces "rounded to 4 significant figures", which TB11's
    evasions defeated at some scales.
  - **low-information windows (RV6):** a window with fewer than 4 distinct
    values, one value making up ≥ 3/4 of the window, std ≤ 2^-46 · |mean|,
    or a near-perfect linear ramp is excluded from both hash sets at build
    and ignored at lookup.
- **Matched against** everything in the submission:
  - numeric literals in `.py` files (AST walk, all number constants and list
    or array literals);
  - every value in `config.json`;
  - any non-code bytes (decoded as float32/float64/int arrays).
  - A match on ≥ `contamination_min_hits` sub-windows (recommended 1 exact or
    3 normalised) is a refusal.
- **When.** At intake, **before any compute**. The typed refusal is
  `benchmark_series_embedded`, with the suite and window id in the receipt
  field.
- **The hash set.**
  - Built from the pinned benchmark data the bench sidecar uses: the suite
    identities recorded on the bench pod (gift 30841734 / boom 69325b54 /
    time 83e3d0b3 / fev 6de245af, OPSLOG).
  - Published as a versioned, digest-pinned artefact
    (`benchmarks/contamination/hashset-v<n>.json.zst`, sha256 in
    `[lane_b] contamination_hashset`).
  - Refreshed **only** when a suite identity changes (a new benchmark
    release), as a new version under a new digest.
- **Backstop (stated in §9).** A miner can recreate a benchmark series
  procedurally (fit it, then generate it from parameters), and a static hash
  check can't see that. **Guard 4** (whole-pool point gain ≥ 0) and the
  **anchor guard** remain the backstop: a gain that exists only on a public
  test set is refused before merge.
- **[DECIDE D38] `max_repo_mb` for suite-target submissions.**
  - Today `[generator] max_repo_mb = 128` (`chain.toml:47`). DEC-CA-0028 notes
    `config.json` can carry bulk data and plans a reduction ("no longer any
    excuse for numbers in `config.json`", DEC-CA-0028:56).
  - (a) **Yes: 8 MB for lane B suite-target entries (recommended).** Less room
    to hide benchmark data, and a faster check.
  - (b) No: keep 128 MB for everyone.
  - Either way it's a new key, `[lane_b] suite_target_max_repo_mb`.

### 12b.2 LB2: rules for miner-chosen experts

- **Declaration.** At commit, `declared_rows = {layer: [expert indices]}`,
  validated against `experts_per_layer = (128, 256, 256, 512, 512, 512, 512)`
  (`pyramid_model.py:42`; counts are **per layer**).
  - **(rev5) Keys are MoE ordinals 0–6**, the existing selection format
    (`trainer/expert_lane.py:108-109`, one entry per `moe_layers()`; the
    worker parses `{int: [int]}`, `worker.py:226-227`). They are **not** block
    indices: the MoE blocks are `(1, 2, 3, 5, 6, 7, 8)`
    (`pyramid_model.py:39`). Ordinal i has `experts_per_layer[i]` experts.
  - `declared_rows_out_of_range`: an index ≥ that layer's expert count.
  - `declared_rows_bad_layer`: **(rev5)** a key outside 0–6.
  - `declared_rows_count`: a count outside the per-layer clamp.
  - **(rev5) The clamp minimum applies only to layers with ≥ 1 declared
    row.** Zero rows on a layer is allowed, matching ESFT, where a layer may
    take fewer than `min_n` rows, possibly none (`expert_lane.py:95-99`). At
    least one row in total is required (TB21 covers "no declared rows").
- **Per-layer clamp.** `min_experts = 8` / `max_experts = 38` were set for 256
  experts, i.e. 3.125% / 14.84% of a layer. Replaced by fractions:
  - `min_experts_frac = 0.03125` (8/256) and `max_experts_frac = 0.1484375`
    (38/256), the exact ratios of the current values;
  - per layer, `min = ceil(frac_min · E_l)`, `max = floor(frac_max · E_l)`;
  - ⇒ **128 → 4..19, 256 → 8..38 (unchanged), 512 → 16..76.**
  - This keeps ESFT's "5–15% of experts per layer" regime (DEC-CA-0050) at
    every layer size.
  - **[DECIDE D39]:** the fractions (recommended above), or fixed per-layer
    lists.
- **Router rows [DECIDE D40].**
  - (a) **Router and bias stay frozen (recommended for first arming).** This
    is today's backend (`expert_training.py:112-122`). Dead experts can't be
    revived, because a frozen router sends them no tokens.
  - (b) **FlexOlmo pattern:** only the declared rows' router entries and bias
    unfreeze, which can revive dead experts. If (b):
    - the merge byte-check also proves no other router row moved;
    - a **traffic-stealing guard** applies: on each **non-target** domain's
      windows, the routing share captured by the declared rows may rise by at
      most `traffic_steal_bound` (recommended 0.02 absolute per layer),
      measured on the rebased model against the current king, else refused
      with `traffic_stealing`;
    - T-row TB14.
- **Overlaps.** Claims on overlapping rows follow the existing merge-acceptance
  deferral: rows are exclusive against open legs, finished-but-unsettled legs
  and pending merges (DEC-CA-0059).
  - **Claim order:** earliest reveal.
  - A deferred claim shows `deferred: rows_held` in `status/lane_b.json`,
    with the holder's entry id and the expected release (the holder's
    verdict or merge era).
- **Trainer-picked rows (ESFT) remain the default** when a miner declares none.
  **Both modes coexist:** a declared entry claims its rows first, then
  ESFT-selected entries are planned excluding all held rows.

### 12b.3 LB3: routing information for miners

- **What's published, per king:**
  - per-layer **per-expert** load on the training stream;
  - dead and near-dead experts (load < 0.1 × mean);
  - per-suite routing affinity: routing counts over each public suite's
    **context inputs only**, with no targets and no training on them.
- **Repo fact (rev4).** PR #481's telemetry is **open, not merged, not
  deployed**. It records per-layer **summary** stats only (`eff`, `dead`,
  `max_mean` per 50-step window and per trunk/cooldown phase), **not**
  per-expert counts. The producer therefore extends it to emit the per-expert
  count vector at the phase level (WB12).
- **Where.** A digest-pinned artefact `status/lane_b_routing-<king8>.json`
  (sha256 recorded in `status/lane_b.json`), refreshed when the king
  changes. The producer is **WB12**; the suite affinity runs on the bench
  host's existing suite inputs.
- **It's informational:** never scoring, never consensus. Stale info
  (king changed, no new artefact yet) shows `stale_since_block`. Miners
  declare rows at their own risk, and a stale artefact never affects a
  verdict.

### 12b.4 LB4: replacing the opening gate

- **Today:** `open_tokens = 200B`, `routing_jaccard = 0.7`, `routing_overlap =
  0.8`, `rebase_jaccard = 0.5`, `rebase_overlap = 0.7` (deployed `[lane_b]`).
  The gate check is `gate_open(lineage_tokens, routing, lb)`
  (`lane_b_driver.py:503-510`).
- **Repo fact (rev4): the co-gate has never been measured on mainnet.**
  - The mainnet driver state (`/root/cascade/_train_work/lane_b_driver.json`,
    last written 11-15) has `gate_opened_block = 0`, `routing = {}`, an empty
    `routing_history` and `lineage_tokens = 0`.
  - Lane B is off (`[lane_b] from_block = 0`), so the tick that measures it
    never runs.
  - The only measurement is the 2026-11-20 manual probe: king vs init top-8
    overlap about 1.1/8, far below 0.8.
- **[DECIDE D41] the opening rule:**
  - (a) **`open_tokens` + a passing lane A king check (recommended).** It
    proves the lineage is maturing without needing routing to stop moving.
  - (b) `open_tokens` alone.
  - (c) keep the co-gate with re-measured thresholds.

  The new key is `[lane_b] open_rule` (`"tokens_king_check"` | `"tokens"` |
  `"cogate"`; default `"cogate"`, so unarmed digests don't move).
- **Consensus classification (§14.2).**
  - `open_rule` and the `[lane_b]` thresholds are overlay-settable
    (`ALLOWED["lane_b"] = None`).
  - The code for (a) is new, so it needs the **capability release**, then an
    overlay switches it.
  - `remainder` stays release-only.

### 12b.5 LB5: sentinel interaction (D35)

- With 4 suite slots and one suite held out each era (§16), **about a quarter
  of suite targets are deferred at any time.**
- **[DECIDE D35, rev4]:**
  - (a) **Accept and document (recommended).** A suite-target entry whose
    target is the current sentinel defers to the next era.
  - (b) Exempt lane B's own declared target from the sentinel, so it's held
    out of guards only.
- **What the miner sees.** The next era's sentinel **can't be known at
  commit**, by design (drawn from the era-boundary block hash).
  - `cascade status` shows the current era's sentinel.
  - A deferred entry shows `deferred: sentinel_target`.
  - **A deferral doesn't consume the miner's per-round submission:** the
    entry stays queued and uncharged until its target isn't the sentinel.
  - There's a bound, `sentinel_max_defer_eras` (recommended 4), after which
    it's judged on its target regardless, logged.

### 12b.6 LB6: miner cost of expert-only legs

**How a leg is budgeted today (verified).**

- `contract.for_hours(leg_hours)` (`cascade/shared/config/training.py:380-413`)
  sets `training_steps = 23,841 × leg_hours / target_train_hours`. Deployed
  `[v2.training] target_train_hours = 3.0` and `[lane_b] leg_hours = 3.0`, so
  a lane B leg runs **the same 23,841 steps as a full lane A leg**.
- The leg stops at the step count or at the wall guard,
  `max(heat_guard_factor × hours, heat_guard_floor_seconds)` = 1.0 × 3 h =
  10,800 s, **capped at the contract's `max_train_seconds` (18,000 s,
  `[v2.training]`)**. A time stop is flagged `deadline_hit`.
- **The budget is steps, not hours.** The miner pays until the leg ends, so a
  faster expert-only step does lower the bill. (An earlier session answer said
  "rent is SKU × hours"; that was wrong.)
- **Guard-vs-wall risk.** The measured full-leg wall on 8×H100 is 13,600 s
  (`[round] funded_sku_wall_seconds`), above the 10,800 s guard. Whether that
  figure is training-only or the whole leg (boot, corpus, harvest) is
  **unverified**. If it's training-only, an expert-only step must be ≥ about
  1.26× faster or every 8×H100 lane B leg truncates. LB6 splits the two.
- GPU choice today: lane B uses `GpuChoice` from the **shared** `[round]
  funded_pod_skus` (deployed `["8xH100", "8xH200"]`, `funded_sku_per_leg =
  true`, $20/h cap for no-choice legs). `lane_b_driver.py:1305`.

**D42 decided (owner, 2026-11-21): cheaper GPUs allowed, and the miner chooses
the leg length, up to 6 h.**

1. **Lane B's own GPU list.** New `[lane_b] funded_pod_skus`, separate from
   `[round]`, so adding a cheap GPU for lane B never opens it to lane A legs.
   - A GPU type enters the list only after (i) the pinned image (torch
     2.4.1+cu124) is qualified on it, (ii) its expert-only step rate and leg
     overhead are measured (LB6), and (iii) the leg fits memory **without
     changing micro-batching** (a different accumulation changes numerics).
   - The type is recorded on the leg (`gpu_name`, already passed to `done()`),
     and an audit re-derivation must use the same type, because a different GPU
     count changes reduction order.
2. **Miner-chosen leg length.**
   - The miner sets `train_hours` at funding: `0.5 ≤ train_hours ≤
     [lane_b] max_leg_hours` (**6.0**), in 0.5 h steps.
   - **[DECIDE D42b] What "hours" means:**
     - (a) **wall hours on the chosen GPU (recommended).** At admission,
       `steps = floor(train_hours × 3600 × measured_steps_per_s[gpu])`,
       fixed then and recorded in the signed leg record. Reproducible, and it
       matches what the miner pays for. A cheap GPU buys fewer steps per hour.
     - (b) **reference hours.** `steps = 23,841 × train_hours / 3.0` on any
       GPU. Equal training for an equal setting, but wall time on a slow GPU
       can be several times longer, which collides with the 5 h contract cap
       and the era timing below.
   - **Guard:** `train_hours × [lane_b] guard_factor` (recommended 1.25) plus the
     measured overhead, capped at **`[lane_b] max_leg_seconds`** (recommended
     6 h × 1.25 = 27,000 s). This is a new lane B cap. **`[training]
     max_train_seconds` is a contract-digest field and is never edited for
     this.** The lane B leg contract is derived with the lane B cap instead
     (a code change in `for_hours`'s caller, release-gated).
   - **Era timing.** The existing cross-era admission (`rolling.py:401-425`)
     already holds a leg that can't land inside the current era until the next
     era's pre-train window. A 6 h leg in a 12 h era therefore starts only in
     the first ~half of an era (or waits). The miner sees the expected landing
     boundary at funding.
3. **Pay and gains.** Longer legs are a miner trade of cost for gain; pay
   tiers, `gain_margin` and the regress-none guards are unchanged. More drift
   per merge is what the guards are for.
4. **Spam.** Cheaper legs mean more entries. The existing per-coldkey caps
   apply unchanged; revisit if the queue grows.
5. **Consensus.** `max_leg_hours`, `max_leg_seconds`, `guard_factor` and the
   `train_hours` range are consensus (validators check the leg's recorded steps
   and guard against them). The per-GPU step-rate table is **trainer-side
   operational config**: steps are fixed at admission and carried signed, so
   satellites never need the table. **(rev5 correction)** `ALLOWED["lane_b"] =
   None` means every `[lane_b]` key is overlay-settable (`netconfig.py:83-87`,
   `:213`; only `remainder` is release-only). The **new keys** still need the
   capability release, because nodes must parse and enforce them. After that,
   their values (`max_leg_hours`, `funded_pod_skus`, …) can change by signed
   overlay.
6. **Miner guide.** A per-GPU table of measured step rate, $/h cap and cost for
   1, 3 and 6 h, generated from the LB6 evidence (WB20).

**LB6 now measures, per candidate GPU type (one pinned-torch testnet pod per
type, OA8):**

- expert-only step rate and leg overhead (boot, corpus, harvest), measured
  separately;
- a full lane A leg on 8×H100 for comparison, which settles whether 13,600 s is
  training-only;
- peak memory at the contract micro-batching;
- image qualification (driver and pinned torch runs).

**Status: not run: blocked: needs owner (pod approval, OA8).**

### 12b.7 Split of purpose (rev4; also stated in §2)

- **Lane B improves the lane A model on declared targets:** trainer-run,
  merged into KING_A, contamination-gated.
- **Lane D explores the architecture with unconstrained training:**
  miner-run, never merged, forward-only judged.

---

## 13. Lessons from past incidents (OPSLOG review 2026-11-21)

| Past failure class | Example | Defence here |
|---|---|---|
| Bench hold that never ends; partial counted as infra | 11-15 / 11-16 "sidecar produced no report", `gift-eval:partial` | §1.3b `checkpoint_fault`; §10.4 hold cap; evidence kept; T17 |
| Bench pod disk full | 11-15 at 98%; leftovers deferred king checks | byte LRU; `pod_disk` under lane D load; no ad-hoc work on eval pods |
| Sweep blocks the weight loop | weight gaps during sweeps | invariant 6; keep-alive before S6; T15 |
| Eval-host code drift | valpriv `TypeError` for 8 h; bench image lacking `fev` | worker handshake; stderr capture; eval host synced at S5; T19 |
| Data time ≠ first-known time | partitions landing late, backfills | §1.3 first-seen rule; T8 |
| Front-running / copying | incentive lag after dethrones; scraping public repos | house refresh on every KING_A change; exact dedup; T16 |
| Private artifact leaks | private code via a shared-bucket tar; credentials in `ps` | own bucket; no credentials on command lines; presigned only |
| Hippius credential scope and rotation | rotated key lacking bucket access | W3a; rotation runbook; T20 |
| Hippius outages, slow transfers | 500s; upload timeouts; 1 MB/s scp | prefetch at reveal; resumable cache; async mirror |
| Box OOM / restart in scoring windows | 10-31 OOM killed all services | nothing big on the box; S5 rules; report size bound; `box_ram` |
| Live overrides vs repo; startup gates at externals | 21 unPR'd overrides; a 128-byte note crash-loop | keys typed in main the same day; CI startup gates; signed-overlay activation (D10; S0c, S6 **(rev6)**) |
| Stale service code | intake ran stale code for a day | `python -P` + code-path guard; deploy-gate verify |
| Settlement-timing traps | S+2 rule; legs after the last settlement | pending carries across eras; T18 |
| Satellite replay divergence | reason-string mismatch ⇒ burn | invariant 7; replay CI; T12 |
| Window-rotation variance | 0.62 LCB swing | judged once on a frozen set; T14 |
| Cross-host numerics | driver 595 vs 580 | paired-on-one-host gate; S4 tolerance |
| Lium lemons / wrong listings | "8×H100" booting as 4× PCIe | on-rent check; replacement runbook |
| Stale known_hosts | king leg hung on a host key | per-pod pinned known_hosts |
| Balance zero / orphan pods | Lium terminated all pods at zero | `runway`; testnet TTLs |
| Alerts not delivered | `bench_infra` into an empty webhook | delivered test alert in the S6 gate; follower patterns |
| Contract-digest drift | a runtime field blocked the king leg | lane D touches no contract term; a pending entry is pinned to its `ready` contract |
| PR hygiene | stacked merges; `git add -u` swept an override | one PR per WP (rev6.2: tests-only PRs may batch WPs, §3.1 rule 6); explicit paths; verify main |
| Identity / disputes | recycled uids | hotkey + digest identity; owner-only archive 14 days |
| Pool publish fragility | the job died for 6 days | dedicated builder; `pool_d_stale` |
| Suite identity / partial | FEV scope-skips counted as partial | W3b identity checks |
| Remote writes need the owner | classifier blocks | owner-paste scripts |
| Blackwell vs pinned torch | sm_120 unsupported by torch 2.4.1 | D11 eval-host GPU class |
| Private offload rule | targets stay local | D18 / invariant 8 |

---

## 14. chain.toml, the signed overlay, and keeping validator updates rare

**Owner goal (2026-11-21):** codify as much as possible so that validators,
external satellites especially, do **not** have to keep updating. In the past
we drifted into relying on hand-edited files and ad-hoc satellite updates.

### 14.1 Where we are today (verified)

- **The signed overlay exists in code but has never run on mainnet.**
  - DEC-CA-0057's owner-signed network config (`config/network/seq-<n>.json`
    and `latest.json`, adopted at 51% stake, applied at an era start) is built
    (`cascade/shared/netconfig.py`).
  - It has **never been published**. The live validator still logs
    `S3 get failed for config/network/latest.json … HF fallback … 404` on
    every poll (`/root/cascade/validator.log`, 2026-11-21 11:27).
- **Config still changes by hand.**
  - The deployed `/root/cascade/chain.toml` carries **21 `LIVE OVERRIDE`
    markers** and differs from `origin/main` by about 119/116 lines.
  - Decided-vote blocks typed only into our deployed file caused the
    external-validator crash-loop and PR #457 (OPSLOG 2026-11-13).
  - The drift guard (`decided_votes.toml` + `cascade-audit decided`, PR #458)
    exists to catch this.
- **An old node can't partly accept an overlay.** An overlay with any table
  the node doesn't know is **refused whole**: `"[{table}] (not on the
  DEC-CA-0057 allow-list)"` (`netconfig.py:195-217`). `ALLOWED` has no
  `lane_d` today (`netconfig.py:83-118`).
  - An overlay that touches `[lane_d]` before a node runs the capability
    release would be refused **along with every other key in that overlay**.
  - The **trainer runs the same resolver** (DEC-CA-0057 §6), so it must take
    the capability release too.
- **The readiness note is full** (128 bytes). Overlay adoption uses the
  `cfg-` segment, which outranks older feature segments, so lane D must not
  add a note segment of its own.

### 14.2 The rule: one capability release, then overlays only

| Change type | How it ships | Validator action |
|---|---|---|
| New code (lane D scorer, report schema, replay rules, intake, audit) | **One capability release** | Update **once** |
| Lane D tuning: `from_block`, `until_block`, `share`, `takes_from`, `contexts`, `horizon`, `forward_margin_hours`, floors, `d_margin`, `alpha`, `bootstrap_B`, `gate_suites`, `gate_weights`, ~~`unselected_suite`~~ **(rev5) dropped:** the held-out suite is the §16 sentinel, drawn from `[v2] sentinel_suites` (W-S1); a lane D key would duplicate it, `tau_public`, court, rate limits, TTLs, `max_bench_holds`, **(rev3)** `max_mirror_lag_hours`, `confirm_on_king_change`, `max_judged_per_coldkey_per_settlement`, `max_coldkey_unresolved_settlements`, `max_reveal_delay_days`, `lowfreq_freqs`, **(rev5)** `ready_ttl_blocks` (B10), `max_source_share` (B14); `remainder` is release-only (B5) | **Signed overlay** (add `ALLOWED["lane_d"] = None` minus `remainder`, which stays release-only as a burn destination, the same rule as `lane_b.remainder`) | **None**: nodes poll every 300 s, adopt by stake, apply at `effective_block` |
| **(rev3)** Cross-lane sentinel (§16): `[v2] sentinel_from_block`, `sentinel_suites`, `sentinel_ratchet`, `sentinel_tau` | **Signed overlay** once the capability release adds them to `ALLOWED["v2"]` (`netconfig.py:99-108`) | None after that release |
| **(rev4)** Lane B: `suite_targets`, `contamination_hashset`, `contamination_min_hits`, `suite_target_max_repo_mb`, `min_experts_frac`, `max_experts_frac`, `router_rows`, `traffic_steal_bound`, `open_rule`, `sentinel_target_rule`, `sentinel_max_defer_eras` | **Signed overlay** (`ALLOWED["lane_b"] = None`, `netconfig.py:83-87`) once the capability release ships the code; `remainder` stays release-only, **(rev5) and so do `contamination_hashset`, `[lane_c] remainder` and `[lane_d] remainder`**: the release adds `("lane_b", "contamination_hashset")`, `("lane_c", "remainder")` and `("lane_d", "remainder")` to the release-only tuple (`netconfig.py:210`). Lane C's remainder is overlay-settable today (`ALLOWED["lane_c"] = None`, `:87`, not in the tuple), an existing gap closed in the same release, because overlays can't replace release-only data pins (`docs/ARCHITECTURE.md:471`). The other new keys (incl. rev5 `funded_pod_skus`, `max_leg_hours`, `max_leg_seconds`, `guard_factor`) stay overlay-settable. Each key drops from the digest at its default, so unarmed digests don't move | None after that release |
| Activation | Overlay types `[lane_d] from_block` (at an era boundary, as lanes B and C allow today) | None |
| Kill switch | Overlay sets `until_block` or `share = 0` | None |
| Main-validator-only operations (vault bucket, `pool_d` location, eval host, sub tokens, bench queue policy) | A **non-consensus** `[lane_d_ops]` table read only by our main validator and the lane D intake; satellites never read it | Our main validator only |
| New benchmark suite for the gate | Runner code is **main-validator-only** (satellites replay rows; suite names are data in the signed report) | Our main validator only |
| Locked training terms, security roots, validator-read storage keys, SDK pin | Release (unchanged DEC-CA-0057 rule) | Update |

**(rev3) If D21 option 2 is chosen** (return the unpaid king slot to lane
A), `[lane_d]` gains a **release-only** destination key, `house_remainder`,
refused by the overlay lint and `check_overrides` like `remainder`. Under
option 1 nothing changes here.

### 14.3 Design rules so satellites rarely need code

1. **Satellites only verify and do arithmetic.** Anything that needs private
   data, GPUs or benchmarks (forecasting, scoring, sweeps, eligibility against
   `pool_d`) runs **only on our main validator**. Satellites verify the
   signature, check that the parameters stated in the report equal
   `config_at(boundary)` from the adopted overlay, check structural rules,
   re-run the bootstrap from signed rows, and derive weights.
2. **The report carries its parameters.** Every value a satellite needs (cohort, frozen window
   ids, margins, alpha, B, seed, suite name, court, share) is **in the signed
   report** and cross-checked against the overlay. New knobs then need no
   satellite logic, as long as the arithmetic is generic.
3. **Anticipate later changes in schema v1** (`lane-d-report/1`): a horizon
   *list*, a suite *list*, per-entry context, room for multiple court
   members, an optional field map. Satellites refuse an unknown schema version
   cleanly, so a schema bump is a known, rare release, never a surprise fork.
4. **Structured comparisons only.** No free text in anything replayed (the
   reason-string burn incident).
5. **Code and config keys are added together.** Every new key ships with its
   loader parse, its `ALLOWED` entry and a round-trip test in the same PR
   (the config-knob memory rule).

### 14.4 Prerequisites before lane D (fix the overlay pipeline first)

These come before S2's lane D work, as stages **S0a (de-drift), S0b (note
budget) and S0c (overlay bring-up)** in §5:

1. **Reconcile config.**
   - Diff the deployed chain.toml against main and classify each of the 21
     overrides: belongs in main, belongs in an overlay, or is an operator-only
     knob.
   - Then PR main and run `cascade-audit decided` clean.
   - **No new hand edits on any validator** from here on.
2. **Testnet overlay drill.** Publish a **no-op overlay (seq 1)**, then a
   real one (one harmless allow-listed key). Show fetch, verify, `cfg-`
   signalling, the stake lock, application at `effective_block`, and
   `config_at` replay by the audit. Then show a deliberately refused overlay
   (an unknown table), refused cleanly with **no** partial application.
3. **Mainnet overlay drill.** A no-op seq-1 overlay on mainnet, to show
   satellites actually fetch and signal it, before any overlay carries a real
   change.
4. **Runbook.** Every future config change is an overlay plus a PR to main
   the same day, never a validator edit. Rollover-decided blocks are typed
   by overlay (DEC-CA-0057 §7). Run `cascade-audit decided` after every
   settlement.

### 14.5 Consequences for the rollout stages (§5)

All folded into §5:

- S0a de-drift, S0b note budget and S0c overlay bring-up before any lane D
  code ships.
- S2b catch-up release, so the lane D release diffs only the lane D and lane B
  build WPs (§5 S2) **(rev6)**.
- S5 capability release to every node **including the trainer** (shared
  overlay resolver).
- **(rev5)** S6: activation and payment are **one overlay** (D45, no
  `share = 0` shadow). The observed `cfg-` stake before its `effective_block`
  is the upgrade proof.
- Every `[lane_d]` overlay restates lane A keys byte-identically, because an
  old node refuses the whole overlay.

---

## 15. Incentive review (rev3): game theory across lanes

This comes from a read-only mechanism-design review against the deployed
economics (`/root/cascade/chain.toml`), the lane decisions and OPSLOG miner
behaviour.

**Everything here is a tokenomics or owner call: both sides are given and
nothing is adopted.** The judging core (forward-only windows, weights frozen
at `ready`) was found sound against target leakage. The weak part is pay.

| # | Issue | Option A (keep v2) | Option B (proposed) |
|---|---|---|---|
| **D25** | **Court pays rent indefinitely; salami-slicing.** The court is rank-decayed (57/29/14% of D's share at `king_decay` 0.5, 2 priors) with no TTL, and the bar is a fixed 0.01. A 3% gain released as three 1% crowns fills the whole court | Simple; matches lane A's court shape | Pay D wins on lanes B and C's `full_pay_tiers` + `half_life_rounds` + `win_ttl_eras` shape, keyed to the crowning LCB over max(KING_D, house). Or a tenure-decaying bar (e.g. 0.03 → 0.01). Or time-bound prior-king tails, like lane A's `court_bp` / `court_blocks` |
| **D26** | **Lane A is a free input to D, and D's 20% comes out of A.** **(rev6:** D's share is now 25% of the total, D1/D48**)** The cheapest winning D entry is "fine-tune the published house". Generators and recipe are published (**(rev5)** `[generation] king_publish = "recipe"`, `chain.toml:1555` on main `07117e1`; not `[v2]`) | Lane A is paid by its own ladder; D pays only for beating the house | A **lineage royalty**: a fixed ρ ≈ 15–20% of each D win's pay goes to the lane A king hotkey that was the house at the winner's T_E. Calibrate ρ in shadow with a weight-space nearest-ancestor measurement. Never self-declared lineage |
| **D27** | **Withholding from A.** A strong generator may earn more used privately in D than published in A | Accept; monitor how many D kings are far from every published A checkpoint | The D26 royalty, plus keep A's share ≥ D's |
| **D28** | **Split.** v2 has A 30 / B 25 / C 25 / D 20 when all are armed | v2 split | Reviewer: **A 40 / D 25 / B 20 / C 15**, with D ramping 15 → 25% after ≥ 3 miner crowns pass the gate with the unselected suite not falling. Now (B, C off): A80/D20 is acceptable. Interacts with **D21**: the reviewer prefers returning the unpaid house slot to lane A over burning it **(rev5) DECIDED (owner): A 25 / B 25 / C 25 / D 25** (§6 D1, D28). |
|  | *(rev4) Share assumption: B launches trainer-run, so its share pays generator-funded legs merged into KING_A. Until B's gate opens its share burns (D37). The A40/D25/B20/C15 proposal stands as an option; no default change.* | | |
| **D29** | **Frequency bias in frozen sets.** Sub-hourly data reaches the floors first, so miners specialise on sub-hourly | v2 freeze rule | Per-frequency-bucket and per-domain window quotas inside a frozen set, equal cluster weights. Freeze when every bucket meets its quota or a max wait passes |
| **D30** | **TIME isn't truly unselected.** It's in lane A's `require_public` gate and the king check, and DEC-CA-0071 (pending: PR #482; rev5 renumber) adds `suite:time` as a lane B target, so the house is optimised on it | Keep TIME as the unselected suite | Drop `s-time` from DEC-CA-0071, or choose a sentinel that no lane gates or pays on |
|  | *(rev3) D30 is addressed by D33: with a rotating sentinel no suite is permanently gated or targeted, so the held-out reading stays unbiased in expectation.* | | |
| **D31** | **Sybil queue griefing.** Cheap coldkeys can fill the 8 judged slots with junk | The A8 caps + registration cost | Round-robin by coldkey; priority by hotkey age or prior non-junk LCB; a cheap pre-screen (small window subset; drop LCB < −x before a full frozen set) |
| **D32** | **Lane B vs D.** Once D runs it dominates lane B as designed (one D king slot ~11–20% of emission vs ~1.8% per B slot) | Keep B as trainer-run legs | Re-scope B to the §12 expert-delta variant when D arms, with row-level dedup against published D kings' rows |
|  | *(rev4) **D32 resolved:** keep lane B trainer-run with miner-chosen experts (§12b). Lane B and D have distinct purposes (§2 item 8), so D's higher king pay doesn't make B redundant: B is the only lane whose wins merge into KING_A.* | | |

Synergies from the same review:

- **Keep:** the house as anchor and floor (A → D).
- **Reject for now:** D → A warm start, and D as a distillation teacher for A
  (contamination and centralisation risk).
- **Later:** lane E on any king, judged only against a plain king so E pay
  never stacks on D pay for the same gain.

The reviewer's view on the v3 "Proposed design changes": adopt 1–4, 7–10, 12
and 14 before S6 (**(rev6)** S6b is merged into S6, D45) (with 10 = gate at 4096 for everyone). Treat 5, 6, 11, 13,
15 and 16 as incentive-neutral ops. Those remain owner decisions and are
**not** applied here.

---

## 16. Cross-lane rotating held-out sentinel (rev3)

**Why.** Every public suite we gate or pay on gets optimised by miners and by
our own pipeline. Today:

- lane A gates on all four suites (`[v2] bench_gate_mode = "enforce"`, the
  `avg_noninf_winrate` rule; `require_public` for lane B);
- **(rev5)** the lane A v2 king check is the bench gate's rule over the same
  **four** suites (GIFT-Eval, BOOM, TIME, FEV; `BENCH_FIELDS_V2`,
  `validator/king_check.py:77-82`): `king_check_gate` (`:93`) calls
  `evaluate_bench_gate` with the king as challenger and the init as reference
  (average ratio, per-suite non-regression at `alpha / 4`, win rate). It does
  not "average six numbers";
- DEC-CA-0071 (pending: PR #482; rev5 renumber) would add suites as lane B targets.

So no suite stays a clean "are we really improving" reading. That's the D30
problem. A fixed held-out suite doesn't fix it either: a suite held out
forever drops out of regression coverage, and miners can still optimise for
it.

**Rule (D33).**

1. **The draw.** At each era boundary, one suite from `sentinel_suites`
   (default `["gift", "boom", "time", "fev"]`) is drawn from the
   **era-boundary block hash**, the same for every validator and
   unpredictable at commit time.
   - The main validator records the drawn suite in the settlement receipt
     and the lane D report as a structured field. Satellites verify it from
     the block hash; block hashes stay readable on pruned nodes, unlike
     past-block state.
   - A satellite that can't read the hash replays the recorded value.
2. **Held out that era, in every lane:**
   - **Lane A:** the gate and the king check use the remaining three suites.
     The sentinel is swept for each candidate that reaches the gate.
   - **Lane B:** it's neither a guard nor a target that era (D35).
   - **Lane D:** it isn't the drawn gate suite (§1.3b); it's swept for each
     new KING_D.
3. **The other suites remain full regression guards** in every lane. That's
   stronger than a rotating gate, which would check only one suite per
   crowning.
4. **Published every era:** each lane's new king's sentinel numbers, plus the
   house's, on a per-suite board in `status/lane_d.json` and the lane A
   status.
5. **Optional ratchet (D34).** A candidate in any lane must satisfy
   `LCB_α(sentinel) ≥ −sentinel_tau` against the **current lane A king**,
   re-scored paired on one host, never against stored numbers. With the
   ratchet on, a regression on the held-out suite is still blocked; it's
   just never used to *select* or *pay*.

**Cost.** Lane A's bench sweep stays at four suites: three gate and one
sentinel. With the ratchet off, the sentinel sweep can run after the verdict
and stays off the critical path. Lane D: one drawn suite plus the sentinel
per new king, roughly as v2.

**Consensus and rollout.**

- **Lane A:**
  - This is a change to lane A's gate rule, so it's a capability release,
    then a signed-overlay switch-on.
  - New `[v2]` keys: `sentinel_from_block` (0 = off), `sentinel_suites`,
    `sentinel_ratchet` (bool), `sentinel_tau`. Each needs a loader parse, an
    `ALLOWED["v2"]` entry (`netconfig.py:99-108` lists the `[v2]` allow-set
    explicitly) and round-trip and mutation tests.
  - It ships in the **S2b catch-up** or the S5 capability release.
  - **(rev5, D45)** It arms by overlay at an era boundary and affects the
    gate from that block, with no shadow eras. Its behaviour is proven on
    testnet first (TS1–TS5).
  - **(rev6.1) Mixed-fleet hazard (RV3 note 2; not applied, see Proposed
    design change 26).** A pre-W-S1 node refuses a whole overlay carrying
    `sentinel_*` keys, and a pre-W-S1 satellite fails replay on every armed
    receipt with a gate or king check. Change 26 proposes gating the arming
    overlay on R4's observed `cfg-` stake ≥ 2/3 **and** every known
    satellite on the W-S1 release.
- **Lane B:** it's off, so it adopts the sentinel from its first armed era.
  DEC-CA-0071 (pending: PR #482; rev5 renumber) needs a one-paragraph amendment for D35.
- **Lane D:** built in from launch (§1.3b).
- **Satellites:** they verify the recorded draw against the block hash and
  re-run the gate arithmetic from signed rows (§14.3). They need no bench
  hardware.

**Work and tests.**

- **WP W-S1:** the sentinel draw, the lane A gate rule change behind
  `sentinel_from_block`, the receipt field and the overlay keys (**(rev5)**
  no shadow logging, D45).
- **WP W-S2:** the lane B guard/target exclusion (with the DEC-CA-0071
  amendment).
  - **(rev6.2, owner 2026-11-22) Era-init acceptances.** A warm-start
    acceptance made at an era-init block is governed by the sentinel of the
    **era that init block opens**, not the era it closes. A follow-up for
    the acceptance callers: the trainer's `accept_candidates`
    (`cascade/trainer/merge_acceptance.py`) and the validator's
    `verify_acceptance` (`cascade/validator/merge_acceptance.py`) both pass
    that era's sentinel.
- **WP W-S3:** publication on the status boards.
- **Test rows:**
  - **TS1:** two validators derive the same sentinel from the block hash, and
    a satellite replays the recorded value when the hash is unreadable.
  - **TS2:** lane A with the sentinel armed. Verdicts use three suites, and
    a candidate regressing only on the sentinel is blocked iff the ratchet is
    on.
  - **TS3:** lane A **unarmed ⇒ byte-identical** receipts and verdicts
    (invariant 1 applies to lane A here).
  - **TS4:** a lane B entry targeting the sentinel suite is deferred (D35).
  - **TS5:** draws are independent per era (a repeat has probability
    1/4). The test checks the draw is deterministic from the hash and
    uniform over `sentinel_suites`.

---

## 17. Completion rubric and status board (rev6)

**(rev6)** This section is the one source of truth for completion.

- `scripts/laned/status` (built by W0c) computes every level below from git
  (`git log origin/main`), GitHub Actions results on main and the files under
  `evidence/`. It writes two outputs to branch **`laned/build-state`**, which
  is never merged to main:
  - `docs/plans/STATUS.md`, for people;
  - `status/laned_build.json`, for agents.
- **Agents never hand-edit status.** "Done" claimed in a PR, in chat or in any
  other file doesn't count. The rest of this plan's "a WP is done only when
  its Accept exits 0" is the L3 rule below.
- **Bootstrap.** Until W0c is at L3, the orchestrator maintains
  `status/laned_build.json` on `laned/build-state` by hand, using these levels
  and the same evidence. W0c is the first WP. From W0c at L3 on, status is
  only what the script computes.

### 17.1 WP levels

Each level needs all the levels below it.

| Level | Name | Condition |
|---|---|---|
| L0 | todo | Listed in §3 |
| L1 | specified | `tests/accept/<wp>/` and `PATHS` merged on main, and those tests fail on main before L2 (checked by the W0c workflow's expect-fail run) |
| L2 | built | Implementation PR merged on main (checked with `git log origin/main`), explicit paths only (every changed path is in the WP's `PATHS`) |
| L3 | accepted | `scripts/laned/accept <WP>` green in CI on main after the merge. A later red run on main drops the WP back to L2 |
| L4 | evidenced | External and owner WPs only (Runs in includes `builder`, `pod`, `testnet` or `owner`): evidence JSON present, and the verifier passes against the live resource. n/a for ci-only WPs, which pass L4 automatically |
| L5 | reviewed | Covered by an RVn whose `evidence/rv<n>.md` has no open BLOCKING item for this WP |
| L6 | released | Its commit is an ancestor of a tagged capability release |
| L7 | live | Its activation block has passed on mainnet and its §10.7 monitor is green for ≥ 1 era. **Owner-confirmed:** the owner merges `evidence/l7/<wp>.json` (activation block, monitor, era range) |

**Side states**, always with a reason:

- `blocked:<decision Dnn | external | owner | dependency>`;
- `parked` (e.g. the §12 variant).

### 17.2 Stage completion

A stage is complete when every WP it lists reaches its required level **and**
its §5 exit gate has an evidence file the verifier accepts. That's the stage's
execution-WP evidence where one exists (e.g. X3's `evidence/s3_matrix.json`),
otherwise `evidence/gates/<stage>.json`, checked by
`scripts/laned/accept --gate <stage>`.

| Stage | Required |
|---|---|
| S0 | Every §6 decision is decided, withdrawn or deferred-to-stage in the §17.4 register; W0 at L3 |
| S0a | R6x at L4 |
| S0b | R2x at L4 |
| S0c | R7t and R7m at L4 |
| S1 | W1, W0b and W3a at L4; R16 archive growing (R16 evidence) |
| S2 | Every build WP at L5; the S2 gate evidenced. **(rev6.1)** A **build WP** is a §3.0 WP whose `Runs in` is ci-only, or ci plus pod/testnet for its CI-checkable parts. Owner execution WPs (R2x, R6x, R7t, R7m, R9, R13, R14, R15, R16 and every X*) are **not** build WPs; they belong to their own stages (S0a–S0c, S1, S2b, …). This is the set the §3.0 Review WPs paragraph counts ("W/R build WPs, not owner execution WPs") and the set "all build WPs" means in S4 and S5 |
| S2b | R9 at L4 |
| S3 | X3 at L4, with every §4 T-row passed |
| S4 | X4 at L4; all build WPs at L6 |
| S5 | X5 at L4; all build WPs at L6 |
| S6 | X6a at L4; the lane D WPs (§3.0 "Lane D core") at L7 |
| SB0–SB5 | The same levels per §5 row 12c: SB0 as S0 for D35–D42 and D42b; SB1 as S2 for its WPs; SB2 WB19 at L4; SB3 as S3 for TB10–TB27 and TS1–TS5; SB4 = S5 (shared release); SB5 the lane B WPs at L7 |
| S7 | X7 at L4 |
| S8 | W6-S8 at L4; each added context rung at L7 |

### 17.3 Coverage counters

Published on every run, in both outputs:

- **T-rows** passed on testnet / total (S3 needs 100%);
- **invariants 1–12**, each with ≥ 1 test at L3 (S2 needs 100%);
- **orphan check:** every WP row, T-row and decision maps to an owning WP;
- **blocked count** by reason, oldest first.

### 17.4 Decision register and owner inputs

Each D-number has a status (`open` | `decided` | `withdrawn` |
`deferred-to-stage`), the option chosen, the date and who decided. The status
script parses this table on main; it changes only by an owner-merged plan PR.

- An **open STRUCTURAL (S)** decision, one that changes code structure or a
  release-only rule (e.g. D21, D40, D46, D48), **blocks its WPs**.
- A decision that only sets an **overlay-tunable (O)** value, or an ops-only
  choice, **doesn't block**. The WP builds with the recommended default and
  lists it under "defaults assumed" in its PR.

"Default" below means the §6 recommended default, accepted by the owner's
2026-11-21 answer ("all other recs"; §6 header).

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

**No decision is open, so no WP is blocked on a decision.** Proposed change 22
asks the owner to confirm the five §15 rows recorded from "all other recs".

**(rev6.2) Owner decisions 2026-11-22** ("do all ur recommendations"). Build
rules inside existing WPs, not §6 D-numbers, so they aren't table rows:

| Item | Decision | WP | Decided |
|---|---|---|---|
| (c) | LB1 normalisation = WB11 schema 2 z-grid with RV6 low-information exclusion (§12b.1; #617/#621, #632) | WB11 | owner, 2026-11-22 |
| (d) | A deferred lane B funded entry's `PayerKeyVault` key has its TTL refreshed on each deferred tick (#629/#633) | WB18 follow-up | owner, 2026-11-22 |
| (e) | `laned_vault_ledger/1` is produced by W17 | W17 | owner, 2026-11-22 |
| (f) | An era-init warm-start acceptance is governed by the sentinel of the era its init block opens | W-S2 follow-up | owner, 2026-11-22 |
| (h) | Tests-only PRs may batch several WPs (one `laned:<WP>` label and one separately recorded accept run per WP; whole PR fails if any WP fails); implementation, tests-correction, evidence and review PRs stay one WP per PR (§3.1 rule 6) | all build WPs | owner, 2026-11-22 |

**Owner inputs** (moved from Appendix A, rev6). These are inputs to runs, not
design decisions. They're never guessed: the dependent WP stays
`blocked:owner` until the input's evidence file exists. "Here" in the list
below now means that evidence file.

1. **OA1, prior-king pairs (B1 null type 2).** List the checkpoint refs
   (`repo@sha256:…`) of ≥ 3 consecutive lane A v2 kings that are light
   fine-tunes of each other (same lineage, adjacent reigns). Write them here
   as `king_t, king_t+1` pairs.
2. **OA2, seed replicas (B1 null type 2b).** Do any same-corpus,
   different-seed replicas exist (DEC-CA-0061, docs/OPERATOR.md
   seed-noise)? If yes, give their refs. If none, write "none".
3. **OA3, per-era false-crown target. (rev5) ANSWERED: 5%** (owner 2026-11-21). The recommended value is 5%. Confirm
   or set a value; it gates S4 (T14) and sizes `d_margin`.
4. **OA4, pod approval for B1 scoring and the B2 16k memory fit.** Either:
   - (a) use valpriv-4090 **outside lane A duel windows** (no rental; it
     touches an eval pod), or
   - (b) approve renting one pinned-torch pod (torch 2.4.1+cu124 capable,
     ≥ 24 GB), with a cost cap.

   State which, plus the time window.
5. **OA5, forge-mirror read access for B2.** The B2 history-length counts
   (≥ 16k, and the S8 lengths) and first-seen checks can't come from pool
   snapshots: they truncate history to 4096+720 and carry no timestamps.
   Approve running `scripts/laned_w1_measure.py` (or a mirror variant) on
   GitHub Actions or the builder with the `TSFORGE_S3` read credentials. Say
   where.
6. **OA6, pinned pool snapshots for B2's snapshot-diff bound.** Approve
   fetching ≥ 2 consecutive pinned daily `pool` snapshots and one `pool_mkt`
   run of snapshots to a non-box host (GitHub Actions or the builder), and say
   where. They're never fetched to the prod box.
7. **OA7, input plan.** The brief's PLAN START/END block contained only the
   template placeholder. I used the latest local plan (v2 design + the v3
   loop-readiness §3) as commit 1. If you meant a different v2 file, say so
   and the PR is rebuilt on it.
8. **OA8, pod approval for LB6 (WB19).** (rev5) Approve short pinned-torch testnet pods: one 8×H100 (expert-only leg plus one full lane A leg), and one of each cheaper GPU type you want offered to lane B miners (name them). Give a total cost cap.

| OA | Status | Unblocks | Evidence file |
|---|---|---|---|
| OA1 | open | B1 null type 1 (T14, S4 gate) | `evidence/owner/oa1.json` (king pairs) |
| OA2 | open | B1 null type 2b | `evidence/owner/oa2.json` (refs, or `"none"`) |
| OA3 | answered: 5% (D24) | — | — |
| OA4 | open (pod yes + cost cap, or the valpriv-4090 window) | B1 scoring; W1's 16k memory fit | `evidence/owner/oa4.json` |
| OA5 | open (host for `TSFORGE_S3` reads) | W1 history counts, `pool_d` size | `evidence/owner/oa5.json`, then `evidence/w1.json` |
| OA6 | open (non-box host for pinned snapshots) | W1 snapshot-diff bound | `evidence/owner/oa6.json`, then `evidence/w1.json` |
| OA7 | moot: the plan on the branch has been the input since rev3 | — | — |
| OA8 | open (pod yes, GPU types, total cost cap) | WB19 (SB2) | `evidence/owner/oa8.json`, then `evidence/wb19.json` |

### 17.5 Milestones

| Milestone | Condition |
|---|---|
| Build-complete | S2 complete |
| Launch-ready | S0–S5 complete; S3 at 100%; the revert overlay staged |
| Live | S6 complete (lane D) and SB5 complete (lane B) |
| Done | S7 complete for D and B |

S8 is tracked separately.

### 17.6 Agent rules

- Read `status/laned_build.json` (branch `laned/build-state`) before any work,
  and run `scripts/laned/status` after it.
- Only the status script (or the owner, for L7) moves a WP up a level. An agent
  may only set `blocked`, with a reason.
- Roles and the builder's no-edit list are in §3 conventions.
- Appendix A and `LANE_D_PROGRESS.md` are retired in favour of this board
  (Proposed change 21 adopted).

---

## Appendix A: run status and owner actions (rev6: retired)

**(rev6)** Retired in favour of §17. Run status is the board
(`docs/plans/STATUS.md` and `status/laned_build.json` on branch
`laned/build-state`). The owner inputs OA1–OA8 are in §17.4.

---

## Appendix B: S1 results

**(rev6)** `docs/plans/LANE_D_S1_RESULTS.md` is now a pointer to this appendix
(Proposed change 21). Raw outputs are in
`docs/plans/lane_d_s1/`.

**Evidence rule.** Every number below cites its command, the pool snapshot
sha256, the host and a raw output file under `docs/plans/lane_d_s1/`. There
are no estimates, extrapolations or placeholders. Anything not run says
`not run: <blocker>`; anything below its floor says `below floor` plus the
counts.

**(rev6)** Owner inputs are listed in §17.4 (OA1–OA8).

### B2: W1 measurements

| Measurement | Result | Command | Snapshot sha256 | Host | Raw output |
|---|---|---|---|---|---|
| Do first-seen or ingest timestamps exist in `cascade/pool/`? | **No.** No match for `first_seen`, `ingested_at`, `ingest_time`, `landed_at` or `LastModified` (grep exit 1) | `grep -rnE "first_seen\|ingested_at\|ingest_time\|landed_at\|LastModified" cascade/pool/` at repo `3c67b64` | n/a (code check) | droplet (prod box, read-only grep) | `lane_d_s1/b2_first_seen_grep.txt` |
| pool_mkt cadence | **Scheduled every 6 h (`30 */6 * * *`), but scheduled runs skip while `[marketcast] from_block = 0`, which is the current value on mainnet and testnet.** Today pool_mkt publishes only on manual dispatch | `grep -nE "cron\|schedule" .github/workflows/publish-mkt-pool.yml` plus `publish-mkt-pool.yml:84-95` | n/a (code check) | droplet (read-only) | `lane_d_s1/b2_pool_mkt_cadence.txt` |
| Forward-eligible windows and distinct sources per daily snapshot (6 h margin, h = 64) | **not run: blocked: needs owner (OA6)**, a non-box host to fetch ≥ 2 consecutive pinned `pool` snapshots. Pool records carry no timestamps, so this run can only produce a **snapshot-diff bound (upper bound on eligible counts)**, never measured first-seen | `python -I scripts/laned_w1_measure.py diff …` (script in this PR) | — | — | — |
| Floors met under the snapshot-diff upper bound? | **not run** (depends on the row above) | — | — | — | — |
| Same for pool_mkt at 6 h | **not run: blocked: needs owner (OA6)**, plus pool_mkt isn't published on schedule while MarketCast is off (row 2) | — | — | — | — |
| Series and sources with history ≥ 4k / 8k / 16k (+64), the launch `pool_d` | **not run: blocked: needs owner (OA5).** Not answerable from pool snapshots, which truncate every series to 4096 + 720 (`builder.py:264-266`). Needs the forge mirror (`TSFORGE_S3`) | — | — | — | — |
| Series and sources with history ≥ 32k / 64k / 128k (+64), for S8 | **not run: blocked: needs owner (OA5)** (same reason) | — | — | — | — |
| Launch `pool_d` size | **not run: blocked: needs owner (OA5)** | — | — | — | — |
| 16k-context batch memory fit on valpriv-4090 (24 GB) | **not run: blocked: needs owner (OA4)**, pod approval (valpriv-4090 outside lane A duel windows, or a rented pinned-torch pod) | — | — | — | — |

### B1: null FPR

| Measurement | Result |
|---|---|
| Prior-king pairs (null type 1, primary) | **not run: blocked: needs owner (OA1** pairs, **OA4** pod) |
| Seed replicas (null type 2) | **not run: blocked: needs owner (OA2)** |
| Perturbed king (null type 3, the easy case; no recommendation from it alone) | **not run: blocked: needs owner (OA4)** pod approval |
| Per-entry FPR, null LCB quantiles 50/90/95/99, disjoint/resampled spread ratio, confirmation-duel FPR, per-era FPR at k = 1 / 8 / 32, `d_margin` for per-era targets 1/5/10%, at min_clusters 20/30/40 | **not run** (depends on the rows above). The per-era target itself awaits **OA3** |

Other notes from the build:

- `score` doesn't pass a training contract to `evaluate_checkpoint`, so the
  config.json and weights-header pin isn't armed. The code-identity
  byte-check still runs.
- The snapshot sha256 the scripts record is a digest of the unpacked
  directory; the published tar sha256 is recorded as given.
- W1 holds two snapshots in float64, so it must run off the prod box (OA6).

The script (`scripts/null_fpr.py`) and its tests
(`tests/unit/test_null_fpr.py`) are in this PR. CI runs the tests; nothing
was executed on the prod box.

**Repo fact affecting B1 (verified by the script build):**

- `calib_draw` (`cascade/validator/calibration.py:52`, wrapping
  `windows.horizon_draw` at `windows.py:414`) cuts **one window per series**:
  the last 64 points.
- A pinned pool of about 4.6k series (snapshot 9509400: 4,633) therefore
  gives about **4.6k windows at h = 64, not ~12k**. That allows about **4
  disjoint 1000-window sets, not 12**, for the spread check. Fewer sets make
  the disjoint SD noisier; the script reports the set count with the ratio.
- Reaching ~12 disjoint sets would need several snapshots' windows, which
  conflicts with "one window list", or a smaller `--n`.

**Known limitation (B1):** the scoring snapshot isn't forward-only, and fresh
forward windows shift in distribution. **(rev5)** The forward-window re-check
happens on testnet (S3); there's no mainnet shadow (D45).

---

## Appendix C: DEC-CA-0071 amendment rev4 (text)

This is the amendment as drafted in
`decisions/DEC-CA-0071-lane-b-public-suite-targets.md`, on top of PR #482.

### Amendment rev4 (proposed, 2026-11-21)

Status of this amendment: **PROPOSED**. This file carries PR #482's
unmerged text above unchanged, with this amendment on top. The full design
is in `docs/plans/LANE_D_PLAN.md` §12b; this section records the decisions.

**Owner direction (2026-11-21).** Lane B stays **trainer-run**: miners
submit a code-only generator and our trainer trains. Miners may also
(1) **choose which expert rows** are trained, and (2) **target one public
suite**, improving it significantly while regressing nowhere.

1. **Open option (b) is adopted: miner-chosen experts.**
   - The commit declares `declared_rows = {layer: [indices]}`, validated
     against `experts_per_layer = (128, 256, 256, 512, 512, 512, 512)`
     (`cascade/trainer/pyramid_model.py:42`). **(rev5)** Layer keys are MoE
     ordinals 0–6, not block indices. The clamp minimum applies only to
     layers with at least one declared row. Typed refusals:
     `declared_rows_out_of_range`, `declared_rows_bad_layer`,
     `declared_rows_count`.
   - The per-layer clamp becomes fractional (D39): `min_experts_frac =
     0.03125`, `max_experts_frac = 0.1484375`. That gives 128 → 4..19,
     256 → 8..38 and 512 → 16..76, replacing the 256-era `min_experts = 8` /
     `max_experts = 38`.
   - Claims follow the existing merge-acceptance deferral. The earliest
     reveal wins; a deferred claim shows `deferred: rows_held`.
   - ESFT stays the default when nothing is declared, and both modes
     coexist.
   - **Router rows, [DECIDE] D40:** (a) frozen, today's compact-rows
     backend (`cascade/trainer/expert_training.py:112-122`), recommended for
     first arming; or (b) the FlexOlmo unfreeze of the declared rows' router
     entries and bias, with a `traffic_steal_bound` guard.
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
   - **[DECIDE] D38:** `suite_target_max_repo_mb = 8` (recommended) vs keep
     `[generator] max_repo_mb = 128` (`chain.toml:47`). DEC-CA-0028 already
     plans to stop `config.json` carrying bulk data.
3. **The opening gate is replaced ([DECIDE] D41).**
   - The routing-saturation co-gate (`routing_overlap` 0.8, `routing_jaccard`
     0.7) **has never been measured on mainnet**. The driver state shows
     `gate_opened_block = 0`, an empty `routing_history` and
     `lineage_tokens = 0`, because lane B is off.
   - The only measurement, the 2026-11-20 probe, found king-vs-init top-8
     overlap of about 1.1/8.
   - Options: (a) `open_tokens` + a passing lane A king check
     (**recommended**); (b) `open_tokens` alone; (c) keep the co-gate with
     re-measured thresholds.
   - The new key is `[lane_b] open_rule`, overlay-settable after the
     capability release.
4. **Sentinel ([DECIDE] D35).**
   - With the cross-lane rotating sentinel (plan §16), a suite-target entry
     whose target is the current sentinel **defers** to the next era: typed
     `deferred: sentinel_target`, not charged, bounded by
     `sentinel_max_defer_eras` (recommended).
   - The alternative is to exempt lane B's own target from the sentinel.
5. **Routing information for miners** (informational only, never consensus):
   per-king per-expert load, dead experts and per-suite context-only routing
   affinity, in a digest-pinned artefact.
6. **Expert-leg cost (D42, decided):** lane B has its own GPU list, so
   cheaper image-qualified GPUs are allowed, each measured first. The miner
   chooses the leg length, `train_hours`, from 0.5 to 6 h. Steps are fixed at
   admission and signed. The guard is capped by `[lane_b] max_leg_seconds`.
   `[training] max_train_seconds` is not changed.

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
`ALLOWED["lane_b"] = None` (`remainder` stays release-only; **(rev5)** so does
`contamination_hashset`, added to the release-only tuple), loader parse,
round-trip and mutation tests (plan WB10).

---

## Appendix D: related records

- `decisions/DEC-CA-0070-lane-d-open-weights.md`: lane D decision record
  (proposed). Its [DECIDE] table is the same as §6.
- `scripts/null_fpr.py`, `scripts/laned_w1_measure.py` (+ tests): S1
  measurement tools (B1, B2).

