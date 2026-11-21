# Plan: lane D "open weights" (anything goes on our architecture), with lane E and lane A flexibility — v3

Status: **PLAN v3 (proposed 2026-11-21)**: v2's design, made buildable by an
autonomous loop. A work package is done **only when its Accept command exits
0**. Owner decisions are marked **[DECIDE]**. Nothing is armed. **No calendar
dates**: stages are gated (§5).

## rev3 (design pass, 2026-11-21)

This pass changes the design: items A1–A11 of the rev3 brief, plus the
incentive review in §15. Every changed item is marked **(rev3)**. New
decisions are D20–D24 and D25–D32 (§15). The S1 measurements are in
`LANE_D_S1_RESULTS.md` and the run status in `LANE_D_PROGRESS.md`. The
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
   activation (D10; S0c, S6a)".
5. **§1.2:** sub tokens were said to suit the eval pod, which contradicted
   "presigned GETs only, no credentials on pods". It now says builder only;
   the pod gets presigned GETs.
6. **§1.4:** the `takes_from` change goes by the S6b signed overlay, not a
   chain.toml edit (consistent with §5 S6b and §14.2).
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
    hotkey (recommended).
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
    - one PR per WP, dependents only after the dependency is on main;
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
| T14 | W5e |
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
| S6a | R4, X6a |
| S6b | X6b |
| S7 | W11, X7 |

**§11.1 → WPs:**

| Item | WPs |
|---|---|
| vault | W3a, W3 |
| retention / archive | W18 |
| kings, index, cards, one-digest check | W13 |
| `pool_d` pins | W6d |
| eval cache | W15 |

Lane A, lane B and the generator rows are unchanged, so no WP is needed for
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
4. **"Stop publishing ⇒ only lane D burns" is false.** A satellite that can't
   replay burns the **whole** vector (verified: `cascade/validator/satellite.py`
   burn path). Proposed: a structured `lane_d: {status: "absent"}` receipt
   state with a lane-D-only burn, reusing MarketCast's `REPORT_ABSENT` /
   `INFRA_DEFERRED` (verified: `marketcast_report.py:32-34`). Affects §5 kill
   switch, §10.8, W14, T27.
5. **Refused overlays stay refused after upgrade.** The digest persists in
   `state.refused` (verified: `netconfig.py:822-826`). Proposed:
   - publish a restating seq N+1 after the stake gate if any node is known to
     have refused seq N;
   - or a code change that clears refusals on release upgrade.

   Affects §5 S6a, §14.1, T23.
6. **Reuse MarketCast** (DEC-CA-0064: forward windows, absolute-grid window
   ids, signed per-window report with quantile sidecars, absent/deferred
   states) instead of reinventing W5/W7. Affects W5a–W7d.
7. **The window origin grid is undefined.** Proposed:
   - non-overlapping origins on the absolute grid `(ts/freq) % 64 == 0`;
   - every eligible origin included;
   - a stable source-based `series_id`.

   Affects §1.3.
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
14. **Shadow semantics.** Proposed: S6a crowns are provisional and
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
  are phases 2 and 3 (§7, §8). The lane B expert-weights variant is §12, and **(rev3) it ships as part of this plan** (WB1–WB7, TB1–TB8, stage S6c).

| Lane | Miner submits | Who trains | Judged on | Rollout |
|---|---|---|---|---|
| A (live) | data generator | us (fixed recipe) | private pool + bench gate | live |
| B (built, off) | **(rev3)** expert-weight deltas (§12), plus generator + target legs until D32 decides | **the miner** (delta variant) / us (legs) | forward-only per-target gain + guards + §16 sentinel | **in this plan: stage S6c (§5), after lane D pays** |
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
  - `patch_size = 32` (`:32`) and `max_patches = 136` (`:36`). The
    training window is 136 patches (4096 + horizon).
  - `_xpos` uses `scale_base = 512.0` (`:154`) and a fixed, non-persistent
    `zeta = (idx + 0.4) / 1.4` buffer (`:211`). The q·k scale between
    patches d apart is `zeta^(d/512)`.
  - For the lowest-zeta dimensions (zeta ≈ 0.286) that is about **4.4e-5
    at d = 4096 patches**.
  - 128k context is 4096 patches, **about 30× the training window**. The
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
(§1.3). Longer rungs are a later typed change.

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
  2 h after its boundary (OPSLOG:2329). So satellites **cannot** be assumed
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
  - **Object Lock COMPLIANCE is supported.** Lock headers must go on
    CreateMultipartUpload; they're not supported on Complete.
  - **Lifecycle rules are accepted but not applied.** TTL deletion must be our
    own job (D16).
  - **Bucket policies** use AWS policy language, but with no IAM it's unclear
    whether a policy can scope by prefix. The probe tests it.
  - **Consequence.** The miner path is **presigned PUT per part** if the probe
    shows presigned UploadPart works. Otherwise the fallbacks are a single
    presigned PUT (BF16 weights at ~2.5 GB, if the single-PUT size limit
    allows) or a small upload relay off the box.
  - **Immutability after `ready`:** server-side copy into an Object-Lock
    (COMPLIANCE) prefix in the same bucket, plus the digest re-check at
    ingest.
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
4. **The miner commits** through the normal timelocked reveal. The **reveal
   block is T**, the entry's commit time. Weights are immutable from here.
5. **Prefetch.** On reveal, valeval-laned prefetches the entry into a
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
  without timestamps is never eligible. There is no build-time fallback: a
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

- **Gate suites:** GIFT-Eval, BOOM and FEV, run with the existing runners
  (`cascade/eval/benchmarks.py`, `public_guards.py`).
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
- **Progress board:** every king's suite numbers (gate and TIME) go into the
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
  | For | Matches §14.2, where `remainder` is a release-only burn destination. Lane A's share is fixed, so invariant 2, the share-0 shadow and satellite weight derivation are unchanged | The house is lane A miners' work, so lane A's contributors are paid for it |
  | Against | Emission withheld whenever our pipeline is the best | §14.2 must change (a new release-only `house_remainder` destination). Lane A's share varies per settlement, which touches invariant 2, the share-0 shadow (S6a) and satellite derivation (W7d), and adds test rows |
  | Dethroned house | Never enters the court (the court is miner kings only). No tail | Same: no tail. The slot moves to the new miner king at once |

  Under **both** options the house is never a prior king, and prior miner
  kings' tails continue unchanged.

**Share [DECIDE D1]**

- `[lane_d] share = 0.20`, `takes_from = "lane_a"`, `remainder = "burn"`.
- The deployed file has `takes_from = "lane_b"`. It's changed **by the S6b
  signed overlay** (§5, §14.2), not by a chain.toml edit.
- When B and C also arm, the split is **A 30 / B 25 / C 25 / D 20**.
- Live emission is `lane_parts` + `lane_pool_shares` + `emission_vector`
  under the DEC-CA-0055 ladder (`cascade/shared/emission.py:330-420`). The
  legacy `split_emissions` helper is not used. W8 extends `emission_vector`
  with lane D.

### 1.5 Consensus, trust and state

**Who scores**

- **The main validator (ours)** orchestrates lane D:
  - valeval-laned computes forecasts;
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
    `checkpoint_fault`, `bench_hold`;
  - per feed, `measured_mirror_lag_hours` (A1).
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
| valeval-blackwell4 (43 GB, ~19 GB cache) | per crowning: **2.5 GB (BF16, rev3)** upload + 1 drawn suite; per new king: TIME | Byte-based LRU (#467); lane A first; second bench pod if crownings slip > 1 settlement (D6) |
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
  below), and the 96 GB eval host.
- **Metadata:** per-series (per-channel) `end_ts`, `first_target_ts` and the
  per-point first-seen record (W6fs).
- **Artifacts:** two tars from one build.
  - `pool_d_hist`: histories only, **no targets**. It goes to valeval-laned.
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
   the same targets.

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

### 3.0 Work packages

**Build-rule tooling and decisions (first)**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| W0c | Loop tooling: `scripts/laned/accept`; `stub_scan.py` (refuses `TODO`, `FIXME`, `pass`-only bodies, `NotImplementedError`, `...` bodies, hardcoded sample values and mock imports outside `tests/`); `unread_keys.py` (every `[lane_d]`/`[lane_d_ops]` key is read by a parser); `follower_patterns.py` (every lane D log call has a cascade-ops follower pattern) | — | ci | `scripts/laned/accept W0c` (self-tests: each script fails on a planted violation fixture and passes on a clean one) |
| W0 | DEC-CA-0070 (supersedes DEC-CA-0062's recipe design) recording D1–D19 | — | ci + owner merge | `scripts/laned/accept W0` (front matter valid; `supersedes: DEC-CA-0062`; D1–D19 each answered) |
| W0b | Context-equivalence test: the lane D scorer at 4096 equals the pinned `pyramid_wrapper` output (within tolerance) on windows with ≥ 4096 history; a 4096 checkpoint loads `strict=True` at every context variant | W5d | ci | `scripts/laned/accept W0b` |

**Rollout tooling (prerequisites for S0a–S0c, S4–S6)**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| R1 | Block-arithmetic tool `cascade-laned blocks` (era starts, activation and effective blocks from the live grid; cross-checked from two computations) | — | ci | `scripts/laned/accept R1` (golden vectors including past off-by-one cases) |
| R2 | Note-budget decoder (S0b): decode live readiness notes, simulate adding `cfg-`, report trimmed segments | — | ci (fixtures) | `scripts/laned/accept R2` (the 2026-11-13 crash-loop note fixture reproduces the refusal) |
| R2x | S0b live run: decode mainnet notes | R2 | owner (chain read) | `scripts/laned/accept R2x --evidence evidence/s0b_notes.json` (no pending segment trimmed) |
| R3 | Overlay lint: refuse any `[lane_d]` overlay that touches non-`lane_d` keys (except a byte-identical restatement of seq N-1) or `remainder` | W2 | ci | `scripts/laned/accept R3` |
| R4 | Stake-weighted upgrade tracker: per-hotkey `cfg-<hex>` and release from chain notes, stake-weighted %, names the laggards | R2 | ci | `scripts/laned/accept R4` (fixture metagraph + notes ⇒ exact %) |
| R5 | Previous-release worktrees + restart scripts for validator, intake, trainer and eval host (§10.8), with `--dry-run` | — | ci (scripts) + owner (execute) | `scripts/laned/accept R5` (shellcheck; dry-run prints the exact plan; refuses during a scoring or settlement window fixture) |
| R6 | De-drift classifier (S0a): classify each `LIVE OVERRIDE` in a deployed chain.toml as main / overlay / operator-only; emit the main PR diff and the overlay body | — | ci | `scripts/laned/accept R6` (fixture with the 21 markers ⇒ deterministic classification) |
| R6x | S0a execution: main PR merged; deployed file reconciled | R6 | owner | `cascade-audit decided --config /root/cascade/chain.toml --state-dir /root/cascade` exits 0 **and** `scripts/laned/accept R6x --evidence evidence/s0a_diff.json` (consensus-key diff empty) |
| R7 | Overlay bring-up artefacts (S0c): benign seq-1 body generator, a refused-overlay body, runbook wrapper around `cascade-validator --netconfig-overlay … --netconfig-publish` (Approve-tier, dry-run shown to the owner) | R1 | ci | `scripts/laned/accept R7` (generated bodies pass `validate_overlay` on a fixture base; the refused body is refused whole) |
| R7t | S0c testnet drill (needs testnet `[netconfig] owner_hotkey`, today `""` at `chain.testnet.toml:903`) | R7 | testnet + owner | `scripts/laned/accept R7t --evidence evidence/s0c_testnet.json` (seq-1 adopted ≥ 51%; refused body applied nothing) |
| R7m | S0c mainnet no-op overlay | R7t, R6x, R2x | owner | `scripts/laned/accept R7m --evidence evidence/s0c_mainnet.json` (no 404 on S3 or HF; ≥ 51% `cfg-`; receipts stamp `netconfig_seq`; trainer apply block logged) |
| R8 | `deploy_gate` support for the new services (`--service intake-laned|eval-host`), since today only trainer/validator are accepted (`scripts/deploy_gate.py:290,298`) | — | ci | `scripts/laned/accept R8` |
| R9 | Catch-up release (S2b): main-as-of-now without lane D to validator, trainer and satellites | R6x | owner | `scripts/laned/accept R9 --evidence evidence/s2b.json` (2 settlements, identical vectors, no satellite burn) |
| R10 | Upgrade-from-mainnet-state harness (T25): restores a copy of `validator_state.db`, `activation_*`, `netconfig_state.json` and `cascade_state.json` into a testnet node and starts the release | — | ci (fixture state) + testnet | `scripts/laned/accept R10` and, for T25, `--evidence evidence/t25.json` |
| R11 | Shipped-config startup-gate test (T26): run the shipped `chain.toml` and `chain.testnet.toml` through every validator and trainer startup gate (activation note, loader, preflight) at a pinned live head | W2 | ci | `scripts/laned/accept R11` |
| R12 | Eval-host provisioning kit: on-rent check (`nvidia-smi -L`, VRAM, driver, disk), pinned known_hosts, venv sync, handshake, owner-paste setup and recovery scripts (§10.4) | W4 | ci (scripts) + pod + owner | `scripts/laned/accept R12` (shellcheck + dry-run) and `--evidence evidence/r12_pod.json` for a live run |
| R13 | `pool_d` builder VM provisioning: disk, wall, retry, credentials placement (no secrets printed), systemd timer | — | builder + owner | `scripts/laned/accept R13 --evidence evidence/r13.json` |
| R14 | Testnet stack provisioning (§10.3): testnet-ctl, testnet-eval, testnet `[netconfig] owner_hotkey` set (release-level edit), hourly testnet `pool_d`, pod TTLs and verified teardown | R13 | testnet + owner | `scripts/laned/accept R14 --evidence evidence/r14.json` |
| R15 | **(rev3)** Testnet pool_mkt cadence for lane D: a scheduled publish for the testnet lane D path that doesn't depend on `[marketcast] from_block` (today scheduled runs skip while it is 0, `publish-mkt-pool.yml:84-95`) | R14 | ci + testnet | `scripts/laned/accept R15 --evidence evidence/r15.json` (≥ 4 consecutive 6 h snapshots published on testnet) |

**Lane D core**

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| W1 | **(rev3)** Measurement (= S1 B2, `scripts/laned_w1_measure.py`): first-seen existence check; snapshot-diff **upper bound** on forward-eligible counts at both clocks; pool_mkt at 6 h; (§1.3, §1.7): forward-eligible counts and latency per frequency; long-history inventory per context variant; `pool_d` size | — | builder (needs `TSFORGE_S3`) | `scripts/laned/accept W1 --evidence evidence/w1.json` |
| W2 | `[lane_d]` config: the keys `from_block`, `until_block`, `share`, `takes_from`, `remainder`, `contexts`, `horizon`, `forward_margin_hours`, `min_windows`, `min_clusters`, `min_long_series`, `d_margin`, `alpha`, `bootstrap_B`, `gate_suites`, `gate_weights`, `unselected_suite`, `tau_public`, `reward_prior_kings`, `king_decay`, `max_entries_per_settlement`, `max_open_per_coldkey`, `max_bench_holds`, `loser_ttl_eras`, `dtypes`, **(rev3)** `max_mirror_lag_hours` (A1), `confirm_on_king_change` (D20), `max_judged_per_coldkey_per_settlement` (A8), `max_coldkey_unresolved_settlements` (A8), `max_reveal_delay_days` (A9), `lowfreq_freqs` (A10); `contexts` default `[4096, 8192, 16384]` (A5); `dtypes` default `["BF16"]` (A6). **(rev3) Size M, not S:** `lane_d.py` hardcodes `active() = False` (`:65-68`), refuses `from_block > 0` (`:77-79`) and carries DEC-CA-0062 placeholder keys that must keep parsing (`until_block` and `share` overlay-settable; activation by signed-overlay `from_block`, never a DEC-CA-0045 vote), plus legacy keys still parsed (the deployed file carries `targets`, `teacher_*`, `gain_margin`, …); `ALLOWED["lane_d"]` with `remainder` refused via the `(table, key)` special case (`netconfig.py:207`); loader parse, round-trip and **mutation** test per key; the 131072 cap; `LaneDConfig.active()` real | W0c | ci | `scripts/laned/accept W2` (includes `unread_keys.py`) |
| W2b | `[lane_d_ops]` non-consensus table (vault bucket, `pool_d` location, eval host, queue policy), read only by the main validator and the intake; never in receipts | W2 | ci | `scripts/laned/accept W2b` (satellite code path imports nothing from it) |
| W1.5 | **Walking skeleton** `make laned-e2e` (CI): fixture `pool_d` (with `end_ts`) → tiny checkpoint → intake (filesystem adapter behind the real storage interface) → ingest guards → scorer → signed report → `lane_d_state.json` → emission → `check_lane_d` | W2 | ci | `make laned-e2e` asserts: non-zero lane D weight for the challenger; audit passes; one tampered row fails; unarmed byte-identical; `share = 0` leaves weights unchanged and still yields a valid report; `LaneDConfig.active()` is not a constant. **Every later WP keeps it green and adds ≥ 1 assertion** |
| W3a | Storage probe scripts (presigned UploadPart, single-PUT size limit, Object Lock on multipart, prefix bucket policy, sub-token scope) + the **storage decision record** `decisions/STORAGE-LANE-D.md` naming the chosen upload path | — | owner (live Hippius) | `scripts/laned/accept W3a --evidence evidence/w3a_probe.json` (every probe has a recorded live result; the decision record names exactly one path) |
| W3 | `cascade/lane_d/` intake service: submit/ready for **only the W3a-chosen upload path**; ownership; **(rev3)** records `coldkey_at_ready` and refuses over-cap with `coldkey_cap_open` (advisory, A8); dedup; rate limits; queue; systemd unit with the code-path guard; Caddy route `/v1/lane-d/*`; RAM budget | W3a, W2b, R8 | ci (fs adapter) + testnet | `scripts/laned/accept W3` |
| W4 | Ingest guards: presigned fetch; sha256; `verify_checkpoint`; exact-filename allowlist; **lane-D-only dtype path for BF16** (today's guard requires the pinned F32 spec, `checkpoint_guard.py:323-328`, which lane A keeps unchanged); `verify_weights_finite`; worker handshake (git head + package hash) | W1.5 | ci + pod (handshake live) | `scripts/laned/accept W4` (the T5 malformed set is refused with exact codes) |
| W5a | Forward eligibility (§1.3): **(rev3) both clocks** (data time and first-seen > wall(max T) + margin) and `max_mirror_lag_hours`; the v2 snapshot-only rule is refused | W1.5 | ci | `scripts/laned/accept W5a` |
| W5b | Union across snapshots + freeze at floors | W5a | ci | `scripts/laned/accept W5b` |
| W5c | Cohort intersection + latest-T deferral | W5b | ci | `scripts/laned/accept W5c` |
| W5h | **(rev3)** Queue selection: FIFO by T; `max_judged_per_coldkey_per_settlement`, the global cap, coldkey re-check at reveal (`coldkey_changed`), `coldkey_unresolved` bound, `max_reveal_delay_days` close; typed skip reasons in the report (A8, A9) | W5b, W3 | ci | `scripts/laned/accept W5h` (T6b, T18b) |
| W5d | No-pad scorer + context variants (`cascade/eval/lane_d_scorer.py`) | W1.5 | ci (tiny model) + pod (real size, T7) | `scripts/laned/accept W5d` and, for T7, `--evidence evidence/t7.json` |
| W5e | Verdict (paired cluster bootstrap, `d_margin`, max-T; **(rev3)** mandatory confirmation duel on the next disjoint set, D20 king-change rule, `rows_sha256` (A2, A9)) | W5c, W5d | ci | `scripts/laned/accept W5e` |
| W5f | House entry and refresh | W5e | ci | `scripts/laned/accept W5f` |
| W5g | Holds, `checkpoint_fault`, typed reason enums | W5e | ci | `scripts/laned/accept W5g` (asserts the exact reason enum per case) |
| W6a | `pool_d` selection = `pool`'s selected series ids (joined, not re-selected) | W1 | ci + builder | `scripts/laned/accept W6a` (`pool` bytes unchanged) |
| W6fs | **(rev3, launch-blocking)** First-seen recording: an append-only ingest log in the forge mirror path recording, per partition or point, the first time our mirror saw it, plus a per-feed mirror-lag measurement (source publish stamp where available). **Scope:** a recorder + reader used only by the `pool_d` builder. **Invariant:** `pool` bytes and lane A windows byte-identical (the test diffs a `pool` build with and without the recorder). **Tests:** first-seen never later than any snapshot containing the point; rewrites and backfills keep the *first* time; lag computed per feed. Not implemented in this PR | W1 | builder | `scripts/laned/accept W6fs --evidence evidence/w6fs.json` |
| W6b | **(rev3)** Extended history **to 16384 + 64** + per-series (per-channel) `end_ts` / `first_target_ts` | W6a | builder | `scripts/laned/accept W6b --evidence evidence/w6b.json` |
| W6c | `pool_d_hist` / `pool_d_tgt` split | W6b | ci + builder | `scripts/laned/accept W6c` |
| W6d | Pinning on Hippius; digests in reports; retention | W6c, W3a | builder | `scripts/laned/accept W6d --evidence evidence/w6d.json` |
| W6s-*n* | **(rev3: moved to S8)** One WP per deep-history source (option 3, §1.7), each passing cleaning, domain and source keys | W6b | builder | `scripts/laned/accept W6s-<name> --evidence evidence/w6s_<name>.json` |
| W6-S8 | **(rev3)** S8 long-context work: history to 131072 + 64, deep backfill, valeval-laned (96 GB pinned-torch) provisioning | W6d | builder + pod | `scripts/laned/accept W6-S8 --evidence evidence/w6s8.json` |
| W3b1 | Gate suite draw from the boundary block hash | W1.5 | ci | `scripts/laned/accept W3b1` |
| W3b2 | Paired sweep vs KING_D and house on one host | W3b1 | ci (recorded suite outputs) + pod | `scripts/laned/accept W3b2` |
| W3b3 | TIME unselected sweep + publish | W3b2 | ci | `scripts/laned/accept W3b3` |
| W3b4 | Suite identity + partial classification | W3b2 | ci | `scripts/laned/accept W3b4` |
| W3b5 | **(rev3)** Always-run low-frequency private-pool non-inferiority check (`lowfreq_freqs`, `tau_public`; `lowfreq_regression`) (A10) | W5d | ci | `scripts/laned/accept W3b5` (T4b) |
| W7a | Report schema `lane-d-report/1`, including §14.3 forward fields and **(rev3)** `rows_sha256`, both clocks per window, mirror lag, the judged set and skip reasons, `coldkey`, the `confirmation` block, the low-frequency result | W1.5 | ci | `scripts/laned/accept W7a` (JSON-schema; unknown version refused) |
| W7b | Report signing | W7a | ci | `scripts/laned/accept W7b` |
| W7c | `lane_d_state.json` + receipt prior/posterior digest binding | W7b | ci | `scripts/laned/accept W7c` |
| W7d | Satellite replay: an **independent** implementation, no imports from producer eligibility, verdict or weight code (invariant 12) | W7c | ci | `scripts/laned/accept W7d` (import-graph check included) |
| W8 | Emission: lane D in `lane_parts` / `lane_pool_shares` (`emission.py:330-383`, `lane_c.py`), court decay, house burn | W1.5 | ci | `scripts/laned/accept W8` |
| W9 | Audit `check_lane_d` + `derive_weights` (**rev3:** re-derives the judged set, coldkey caps, both-clock eligibility, the confirmation rule and `rows_sha256`), independent of producer code | W7c, W8 | ci | `scripts/laned/accept W9` |
| W10 | Miner CLI `cascade lane-d export \| verify \| submit \| ready \| status`, **W3a-chosen upload path only** | W3a, W3, W4 | ci + testnet | `scripts/laned/accept W10` |
| W11 | cascade-ops checks (§10.7) + follower patterns | W5g, W7c | ci (cascade-ops repo) | `scripts/laned/accept W11` (includes `follower_patterns.py`) |
| W12a | Docs and agent surfaces (§11.2 text surfaces, website; **rev3:** LANE_D.md FAQ on coldkey caps, confirmation latency, both clocks, BF16; VALIDATOR.md on what satellites verify for the new report fields; OPERATOR.md on launch eval host, S8, first-seen recorder) | W10 | ci | `scripts/laned/accept W12a` (every §11.2 surface mentions lane D; links resolve; `llms.txt` lane list matches the status fixture) |
| W12b | `cascade status --json` and `discovery-check` lane D fields | W2, W17 | ci | `scripts/laned/accept W12b` |
| W13 | King publication (§11.1): Hippius Hub, R2, HF mirror, signed `champions/lane_d/index.json`, `laned-king-sync.yml`, cards, one-digest check | W5e, W3a | ci + testnet | `scripts/laned/accept W13` |
| W14 | Fast kill: stop publishing lane D reports (T27) | W7c | ci + testnet | `scripts/laned/accept W14` |
| W15 | Prefetch cache, LRU with KING_D and house pinned | W4 | ci + pod | `scripts/laned/accept W15` |
| W16 | Reference fine-tune script (miner kit) | W10 | ci (CPU tiny run) + pod | `scripts/laned/accept W16` |
| W17 | `status/lane_d.json` producer (queue, pending progress, king, board; **rev3:** per-entry reason enum including `coldkey_cap_settlement`, `confirm_pending` and reveal deadline) | W5b, W7c | ci | `scripts/laned/accept W17` |
| W18 | TTL sweeper: vault → owner-only archive → delete (live runs need D16) | W3a | ci (fs adapter) + owner | `scripts/laned/accept W18` |
| W19 | Lane D work runs **off the weight/poll thread**; the weight-loop keep-alive (#471) is in the release (invariant 6, T15) | W5e | ci + testnet | `scripts/laned/accept W19` (a fixture slow lane D step leaves weight-set timing unchanged) |

**Review WPs.** After every 5 build WPs merge (counting W/R build WPs, not
owner execution WPs), a **fresh-context review WP** runs. RV1, RV2, … are
read-only reviewers against main with the plan as the spec. Accept is
`scripts/laned/accept RVn --evidence evidence/rv<n>.md`: no unresolved
BLOCKING item, and each finding is either fixed or filed as a WP.

**Execution WPs (stage gates, §5)**

| WP | Stage | Runs in | Accept |
|---|---|---|---|
| X3 | S3 testnet matrix T1–T29 | testnet | `scripts/laned/accept X3 --evidence evidence/s3_matrix.json` (each row's recorded pass, 6 clean eras, ≥ 1 crowning, ≥ 1 house retake) |
| X4 | S4 release + valeval-laned provisioning + dry-run + announcement + staged revert overlay | owner | `scripts/laned/accept X4 --evidence evidence/s4.json` |
| X5 | S5 capability-release upgrade (eval host → validator → intake → trainer; satellites) | owner | `scripts/laned/accept X5 --evidence evidence/s5.json` (lane A unchanged ≥ 2 settlements; `--netconfig-policy` lists `lane_d`) |
| X6a | S6a readiness probe + shadow (+ D19 test entry) | owner | `scripts/laned/accept X6a --evidence evidence/s6a.json` (R4 shows ≥ 2/3 `cfg-` stake before `effective_block`; ≥ 2 eras of valid reports; satellite replay matches; webhook test alert delivered) |
| X6b | S6b pay | owner | `scripts/laned/accept X6b --evidence evidence/s6b.json` |
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
6. **One PR per WP,** explicit paths, no stacks, owner-merged. Dependents
   start only after the dependency is **on main** (verify `git log
   origin/main`, never a MERGED badge).
7. **A fresh-context review WP** after every 5 build WPs.
8. **Blocked is acceptable; faked completion is not.** If a WP can't be fully
   built, the builder stops and records the blocker in `evidence/blocked/<wp>.md`.

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
| T7 | Each context variant: wall time and memory on valeval-laned; 4k model vs declared-128k copy on ≤ 4096-history windows | Identical scores on short windows; a variant is armed only if it fits the budget |
| T7b | Gate: two validators derive the same suite from the block hash; regression vs KING_D or vs house blocks; TIME swept and published only | As stated |
| T7c | 2.5 GB (BF16, rev3) direct upload (multipart or single presigned PUT per W3a); prefetch; box RAM and disk flat | As stated |
| T8 | **(rev3)** Forward rule, both clocks: data-time-only pass but first-seen fail (late mirror); first-seen-only pass but data-time fail (publish-ahead); a feed over `max_mirror_lag_hours`; entry and king T values in both orders; a forecast-product feed; a backfilled partition | Only windows absent from the snapshot after max(T) + margin; the forecast feed and the backfill are never eligible |
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
| T4b | **(rev3)** A model that regresses only on low-frequency series | Blocked by `lowfreq_regression` |
| T19 | Eval worker on stale code | Handshake mismatch ⇒ CRIT; nothing scored |
| T20 | Hippius primary write fails, R2 succeeds | Alert fires |
| T21 | Kill switch through the signed overlay | Share returns to lane A from the stated era; no release needed |
| T22 | Readiness note of the exact mainnet shape plus the `cfg-` segment | No pending segment trimmed |
| T23 | Old-code satellite with a new main validator | Matches until lane D arms; burns only after `from_block`; recovers after upgrade |
| T24 | Old-code trainer with a `[lane_d]` overlay published | Trainer refuses that overlay. Proves why the trainer is in S5, and that lane A stays safe while it waits |
| T25 | Upgrade from a **copy of mainnet state** (`validator_state.db`, `activation_*`, `netconfig_state`, `cascade_state.json`) | Clean start; identical next verdict |
| T26 | Shipped chain.toml through every validator and trainer startup gate at the live head | Starts (the #457/#460 class) |
| T27 | Revert overlay; stop-publishing path | Kill latency measured; share burns cleanly meanwhile |
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
capability release → stake-proven shadow → pay.**

| # | Stage | Work | Exit gate (all must hold) |
|---|---|---|---|
| 1 | **S0 Decide** | D1–D37 (incl. the §15 incentive [DECIDE]s and the §16 sentinel) (§6) in writing; W0 | All answered |
| 2 | **S0a De-drift** | Classify the 21 `LIVE OVERRIDE`s in the deployed chain.toml: main, overlay, or documented operator-only; type the decided blocks in main. Overlays merge onto **each node's own base** (`build_config(base, body)` merges onto `effective_raw(base)`, `netconfig.py:346-354`), so base drift survives any overlay | `cascade-audit decided --config /root/cascade/chain.toml` = 0 drift; the validator-read consensus-key diff between deployed and main is empty |
| 3 | **S0b Note budget** | Decode the live mainnet readiness notes and add the `cfg-` segment. It's trimmed **last** (`activation.py:1334-1357`), so older pending segments go first | No still-pending segment would be trimmed (type their decided blocks in main first if needed); T22 passes on testnet |
| 4 | **S0c Overlay bring-up** | Publish a **benign seq-1 overlay** (restates current values only) on testnet, then mainnet. Then a deliberately refused overlay on testnet | The 404-every-poll is gone. Fetched from S3 **and** the HF mirror by our validator, the trainer and satellites. `cfg-` lock ≥ 51% in `status/chain.json`. Receipts stamp `netconfig_seq`; the `cascade-audit` netconfig check passes; the trainer logs the same apply block. The refused overlay applies nothing |
| 5 | **S1 Measure** | W1, W0b, W3a (live Hippius probe) | Forward floors met at the chosen margin (or re-decided); ≥ 1 context variant has `min_long_series`; storage mechanism chosen |
| 6 | **S2 Build** | W2 (with `ALLOWED["lane_d"]`) then W3–W12, one PR each | CI green. Receipt replay on the last N mainnet receipts at the release head: **unarmed ⇒ byte-identical; armed ⇒ lane A verdicts, court and lane A weight entries identical, and lane D fields present and schema-valid** (W3b touches `bench_gate.py`/`public_guards.py`, which lane A replays). Startup-gate test of the shipped chain.toml at the live head for validator **and** trainer (T26). Docs gate |
| 7 | **S2b Catch-up release** | Ship main-as-of-now **without** lane D code to our validator, trainer and satellites, so the lane D release diffs only W2–W12 | 2 settlements with identical vectors; no satellite burn |
| 8 | **S3 Testnet** | §4 matrix including T22–T29 | §4 exit gate |
| 9 | **S4 Release** | Cut the **capability release** (lane D unarmed, `lane_d` allow-listed). Provision valeval-laned (on-rent check, pinned known_hosts, handshake); dry-run the house. **Stage and review the revert overlay.** Send the announcement (§11.3) | Dry-run within the measured cross-host tolerance; announcement sent; revert overlay reviewed |
| 10 | **S5 Upgrade** | Snapshot all state (§10.8). Order, each through `deploy_gate check`/`verify` in quiet windows: **eval host → validator → lane D intake (systemd unit with the code-path guard) → trainer** (no legs in flight; memory-burst check; confirm preflight and worker-fingerprint lines unchanged). Satellites upgrade. Arm the broad watcher | Lane A unchanged on ≥ 2 settlements; `cascade-validator --netconfig-policy` lists `lane_d` on our nodes |
| 11 | **S6a Readiness probe + shadow** | **[DECIDE D19]** at least one owner-run test entry on a dedicated hotkey during the shadow (recommended); without it S6a proves only plumbing. Overlay seq N: `[lane_d] from_block` at a far era boundary, **`share = 0`**, all lane A keys **byte-identical** to seq N-1 (no lane A tuning in a lane D overlay, ever) | **Observed** `cfg-` stake ≥ 2/3 (incl. the largest holder, 5E2LP6 at about 49%) at a boundary before `effective_block`. That's the only stake-weighted proof of upgrade, because only nodes on the capability release can verify a `[lane_d]` overlay. Else publish seq N+1 with a later block. Webhook test alert delivered. Keep-alive live. Then ≥ 2 eras of signed lane D reports at share 0: satellite replay matches, `check_lane_d` passes, no weight gap |
| 12 | **S6b Pay** | Overlay seq N+1: `share = 0.20`, `takes_from = "lane_a"`, lane A keys byte-identical | First paid settlement's vectors match on ≥ 2 validators |
| 12b | **(rev3) S6c Lane B** | Overlay: `[lane_b] from_block` at an era boundary (shadow first: `[lane_b] share = 0` for ≥ 2 eras, then the D1 split) | TB1–TB8 passed in S3; ≥ 2 eras of valid lane B reports at share 0 with satellite replay matching; then first paid settlement's vectors match on ≥ 2 validators |
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
- For anything needing faster action, our main validator can stop
  *publishing* lane D reports. The lane D share then burns (no valid report ⇒
  no lane D payout) until the overlay lands. T27 covers that path too.

## 6. Owner decisions (recommended defaults)

| # | Decision | Recommended |
|---|---|---|
| D1 | Share and source | 20% from lane A, burn remainder; with B and C armed: A 30 / B 25 / C 25 / D 20 |
| D2 | Context variants | **(rev3)** Launch arms {4k, 8k, 16k}. {32k, 64k, 128k} arm at **S8** by overlay (`contexts`), each only after T7 and W1 `min_long_series` (owner goal 128k kept) |
| D2b | Gate | **(rev3)** One gate suite drawn per crowning from the suites **not** held out this era; the held-out suite is the §16 cross-lane sentinel (replaces fixed "TIME unselected"); `tau_public` 0.01; anchors = KING_D and house |
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
| D33 | **(rev3)** Cross-lane rotating sentinel (§16) | **Adopt**: one suite per era, drawn from the era-boundary block hash, held out of every gate and target in lanes A, B and D, swept and published for every new king. Alternative: no sentinel (v2: lane A gates all four; lane D holds TIME out) |
| D34 | **(rev3)** Sentinel ratchet vs the house | **Optional, recommended on**: every lane's candidates must not fall below the current lane A king on the sentinel by more than `sentinel_tau` (0.01), re-scored paired on one host. Off = the sentinel is publish-only, and a regression on it alone is never blocked |
| D35 | **(rev3)** Sentinel vs DEC-CA-0069 (pending: PR #482) lane B suite targets | While a suite is the sentinel it is **not** a lane B target or guard that era. Entries already declared on it are judged on the remaining suites, or deferred to the next era. Recommended: **defer** (the target stays honest) |
| D36 | **(rev3)** Lane B at S6c: which variant arms | Expert-delta variant (§12) arms at S6c. Whether trainer-run legs **also** stay on is **D32** (recommended: delta variant only once D runs) |
| D37 | **(rev3)** Lane B share at arm | The deployed `[lane_b] share = 0.25` (A/B 75/25 while C is off) vs a smaller share given D32's dominance finding; set in the S6c overlay |

---

## 7. Phase 2: lane E (test-time optimisation), sketch

- **Entry:** a test-time procedure on KING_A with fixed weights (view
  ensembling, context adaptation, recalibration).
- **Judging:** on the private pool vs KING_A plain, with "improve one, regress
  none" guards (DEC-CA-0069 (pending: PR #482) pattern), at ≤ 4× base inference compute.
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
| **(rev3)** Coldkeys are free | The A8 caps limit casual flooding only, not a determined Sybil (one IP drove ~70 hotkeys across coldkeys, OPSLOG:164). Registration cost, the eval fee (D4), and the §15 pre-screen [DECIDE] are the levers. k = 32 per era is sized for this in §1.3 |
| **(rev3)** Owner conflict of interest | The owner runs the judging (main) validator **and** enters the house. Mitigations: the house is never paid (D9/D21); verdicts carry `rows_sha256` and signed rows; satellites re-run the bootstrap; `check_lane_d` re-derives the judged set and eligibility; kings are public and re-scorable; D23 (i) would let losers fully re-score; the builder's per-snapshot metadata digest is published (§15 / review round 2 item 12) |
| **(rev3)** The low-frequency check is weak against contamination | The private pool isn't forward-only, so a contaminated model passes A10's check more easily. It catches honest regressions only |
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
  2. Vault → valeval-laned: presigned GET prefetch at reveal; sha-keyed LRU
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
 ├─ valeval-laned: forecasts for cohort + KING_D + house on the frozen set
 │     (handshake first; cached rows re-used by (digest, context, window id))
 ├─ box: score vs pool_d_tgt → bootstrap → max-T → candidate or none
 ├─ candidate: drawn suite (block hash) on valeval-blackwell4, behind lane A,
 │     paired vs KING_D + house; TIME after crowning (publish only)
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

Pre-staged, reviewed **owner-paste scripts** cover valeval-laned setup and
recovery, because remote writes are often classifier-blocked.

### 10.5 Operations checklist by stage

| Stage | Action | Needs |
|---|---|---|
| S1 | W1 on GH Actions or the builder; W3a storage answer | — |
| S3 | Rent testnet-ctl, testnet-eval and the builder; deploy the RC; hourly testnet `pool_d`; arm | Owner yes on rentals |
| S3 | T3 fine-tune pod and T7 pod just-in-time; terminate after on-pod checks | Owner yes |
| S4 | Rent valeval-laned (on-rent check, known_hosts, handshake); dry-run; add `laned-eval.toml` to **`operator_inventory_host_files`** and **`operator_standing_pods_usd_per_hour`** (that list still names valeval-blackwell2; an Approve-tier chain.toml edit through the Edit tool, typed in main the same day, never left as a live-only override) | Owner yes |
| S4 | Cut the release; the owner sends the validator announcement (release, activation block, "no new hardware", how to verify) and the miner announcement | Owner |
| S5 | Snapshot **all 8 state files** first (§10.8: `validator_state.db`, `cascade_state.json`, `warm_start_init.json`, `lane_d_state.json`, `netconfig_state.json`, `activation_*_state.json`, the satellite state file, `decided_votes.toml`). Then restart in order **eval host → validator → lane D intake → trainer** (no legs in flight), each with `python -P`, the code-path guard and `deploy_gate check`/`verify` (R8). **Outside settlement, harvest and scoring windows; never during a bench sweep; never Wed 22:00 – Thu 08:00 UTC.** Satellites upgrade per the announcement | Notify (plan and timing to the owner first) |
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
  file, `decided_votes.toml`. Copy them under `_state_backups/` with a
  timestamp.
- **Never restore validator state across a settlement.** A restore is only
  valid back to the last settlement it already covers. Otherwise forward-fix.
- **Lane D only:** stop publishing lane D reports (share burns), forward-fix,
  republish. Lane A never depends on lane D state.

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
| Lane D index | signed `champions/lane_d/index.json`: every king, context, digest, crown block, LCB, gate and TIME numbers, license |
| Repo cards | `champions/lane_d/<digest>/{PROVENANCE.json, MODEL_CARD.md}` via `laned-king-sync.yml`; no weights in git |
| `pool_d_hist` / `pool_d_tgt` | Hippius pins; sha256 in every lane D report; `pool_d_tgt` and window ids published after the dispute window |

One digest, one identity: vault, public repo, index and report must agree,
and the audit checks it.

### 11.2 Documentation and agent surfaces (ship with the release; S2 docs gate)

| Surface | Change |
|---|---|
| `docs/LANE_D.md` (new) | what lane D is; checkpoint format; context as a declared scorer setting; export, verify, submit, ready, commit; dedicated hotkey; forward-only rule and **expected latency by frequency**; house; gate and TIME; pay and court; license; FAQ; **(rev3)** coldkey caps and the typed skip reasons, confirmation-duel latency, both clocks and mirror lag, BF16 only, launch contexts 4k–16k (S8 later), max reveal delay |
| `docs/INTERFACE.md` | a lane D checkpoint interface section |
| `docs/MINER.md` | lane table, link, "no compute keys needed for lane D" |
| `docs/AGENT_MINING.md` | lane D agent workflow; **local scores on the revealed pool overstate forward gains** |
| `AGENTS.md`, `agent.md` | lane D role; human authorization still needed for spend and publish |
| `llms.txt` (repo + website) | lane D links; fix the stale "Planned order A → C → B" and activation text |
| `docs/VALIDATOR.md` | lane D replay: what satellites verify and what they can't; **(rev3)** the new report fields (`rows_sha256`, `coldkey`, judged set, `confirmation`, low-frequency result, both clocks) |
| `docs/OPERATOR.md` | valeval-laned, builder, vault, retention, runbook, kill switch; **(rev3)** launch eval host (pinned torch, not Blackwell), S8 long context, the first-seen recorder (W6fs), the coldkey alert |
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

## 12. Lane B variant: miners submit expert weights (rev3: in scope of this plan)

**(rev3) In scope: built, tested and armed as part of this plan** (owner 2026-11-21), in S2 alongside lane D, tested in S3, armed at **S6c** after lane D pays (S6b). **(rev3)** Caveat: most low-frequency domains will rarely reach the forward-only floors, so forward-only judging of per-domain expert deltas works mainly for sub-hourly domains. It reuses lane D's vault, forward-only
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
- **New design:** this is *not* DEC-CA-0069 (pending: PR #482)'s Open
  option (b). That option is miner-declared rows with trainer-run training.
  **(rev3) WB7 writes the DEC-CA-0069 amendment** that records this variant,
  the §16 sentinel rule (D35) and forward-only judging.

**(rev3) Execution in this plan**

- **Deployed state:** `[lane_b] from_block = 0` (off) and `leg_funding =
  "miner"`. The A/B split is 75/25 while C is off.
- **Arming:** `[lane_b]` is fully overlay-settable (`ALLOWED["lane_b"] =
  None`, `netconfig.py:83-87`; `remainder` stays release-only), so S6c arms
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
| WB7 | DEC-CA-0069 (pending: PR #482) amendment: expert-delta variant, forward-only judging, the D35 sentinel rule | — | ci + owner merge | `scripts/laned/accept WB7` |

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
| Live overrides vs repo; startup gates at externals | 21 unPR'd overrides; a 128-byte note crash-loop | keys typed in main the same day; CI startup gates; signed-overlay activation (D10; S0c, S6a) |
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
| PR hygiene | stacked merges; `git add -u` swept an override | one PR per WP; explicit paths; verify main |
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
| Lane D tuning: `from_block`, `until_block`, `share`, `takes_from`, `contexts`, `horizon`, `forward_margin_hours`, floors, `d_margin`, `alpha`, `bootstrap_B`, `gate_suites`, `gate_weights`, `unselected_suite`, `tau_public`, court, rate limits, TTLs, `max_bench_holds`, **(rev3)** `max_mirror_lag_hours`, `confirm_on_king_change`, `max_judged_per_coldkey_per_settlement`, `max_coldkey_unresolved_settlements`, `max_reveal_delay_days`, `lowfreq_freqs` | **Signed overlay** (add `ALLOWED["lane_d"] = None` minus `remainder`, which stays release-only as a burn destination, the same rule as `lane_b.remainder`) | **None**: nodes poll every 300 s, adopt by stake, apply at `effective_block` |
| **(rev3)** Cross-lane sentinel (§16): `[v2] sentinel_from_block`, `sentinel_suites`, `sentinel_ratchet`, `sentinel_tau` | **Signed overlay** once the capability release adds them to `ALLOWED["v2"]` (`netconfig.py:99-108`) | None after that release |
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
- S2b catch-up release, so the lane D release diffs only lane D.
- S5 capability release to every node **including the trainer** (shared
  overlay resolver).
- S6a/S6b: activation and payment are **overlays** (shadow at `share = 0`
  first). The observed `cfg-` stake on the first `[lane_d]` overlay is the
  upgrade proof.
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
| **D26** | **Lane A is a free input to D, and D's 20% comes out of A.** The cheapest winning D entry is "fine-tune the published house". Generators and recipe are published (`[v2] king_publish = "recipe"`) | Lane A is paid by its own ladder; D pays only for beating the house | A **lineage royalty**: a fixed ρ ≈ 15–20% of each D win's pay goes to the lane A king hotkey that was the house at the winner's T_E. Calibrate ρ in shadow with a weight-space nearest-ancestor measurement. Never self-declared lineage |
| **D27** | **Withholding from A.** A strong generator may earn more used privately in D than published in A | Accept; monitor how many D kings are far from every published A checkpoint | The D26 royalty, plus keep A's share ≥ D's |
| **D28** | **Split.** v2 has A 30 / B 25 / C 25 / D 20 when all are armed | v2 split | Reviewer: **A 40 / D 25 / B 20 / C 15**, with D ramping 15 → 25% after ≥ 3 miner crowns pass the gate with the unselected suite not falling. Now (B, C off): A80/D20 is acceptable. Interacts with **D21**: the reviewer prefers returning the unpaid house slot to lane A over burning it |
| **D29** | **Frequency bias in frozen sets.** Sub-hourly data reaches the floors first, so miners specialise on sub-hourly | v2 freeze rule | Per-frequency-bucket and per-domain window quotas inside a frozen set, equal cluster weights. Freeze when every bucket meets its quota or a max wait passes |
| **D30** | **TIME isn't truly unselected.** It's in lane A's `require_public` gate and the king check, and DEC-CA-0069 (pending: PR #482) adds `suite:time` as a lane B target, so the house is optimised on it | Keep TIME as the unselected suite | Drop `s-time` from DEC-CA-0069, or choose a sentinel that no lane gates or pays on |
|  | *(rev3) D30 is addressed by D33: with a rotating sentinel no suite is permanently gated or targeted, so the held-out reading stays unbiased in expectation.* | | |
| **D31** | **Sybil queue griefing.** Cheap coldkeys can fill the 8 judged slots with junk | The A8 caps + registration cost | Round-robin by coldkey; priority by hotkey age or prior non-junk LCB; a cheap pre-screen (small window subset; drop LCB < −x before a full frozen set) |
| **D32** | **Lane B vs D.** Once D runs it dominates lane B as designed (one D king slot ~11–20% of emission vs ~1.8% per B slot) | Keep B as trainer-run legs | Re-scope B to the §12 expert-delta variant when D arms, with row-level dedup against published D kings' rows |

Synergies from the same review:

- **Keep:** the house as anchor and floor (A → D).
- **Reject for now:** D → A warm start, and D as a distillation teacher for A
  (contamination and centralisation risk).
- **Later:** lane E on any king, judged only against a plain king so E pay
  never stacks on D pay for the same gain.

The reviewer's view on the v3 "Proposed design changes": adopt 1–4, 7–10, 12
and 14 before S6b (with 10 = gate at 4096 for everyone). Treat 5, 6, 11, 13,
15 and 16 as incentive-neutral ops. Those remain owner decisions and are
**not** applied here.

---

## 16. Cross-lane rotating held-out sentinel (rev3)

**Why.** Every public suite we gate or pay on gets optimised by miners and by
our own pipeline. Today:

- lane A gates on all four suites (`[v2] bench_gate_mode = "enforce"`, the
  `avg_noninf_winrate` rule; `require_public` for lane B);
- the lane A king check averages all six numbers;
- DEC-CA-0069 (pending: PR #482) would add suites as lane B targets.

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
  - It arms by overlay at an era boundary, with shadow first: log the
    sentinel and the would-be verdicts for ≥ 2 eras before it affects the
    gate.
- **Lane B:** it's off, so it adopts the sentinel from its first armed era.
  DEC-CA-0069 (pending: PR #482) needs a one-paragraph amendment for D35.
- **Lane D:** built in from launch (§1.3b).
- **Satellites:** they verify the recorded draw against the block hash and
  re-run the gate arithmetic from signed rows (§14.3). They need no bench
  hardware.

**Work and tests.**

- **WP W-S1:** the sentinel draw, the lane A gate rule change behind
  `sentinel_from_block`, the receipt field, the overlay keys, and the
  shadow logging.
- **WP W-S2:** the lane B guard/target exclusion (with the DEC-CA-0069
  amendment).
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

