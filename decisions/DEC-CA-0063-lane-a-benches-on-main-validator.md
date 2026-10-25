---
id: DEC-CA-0063
type: decision
title: "Lane-A public benchmarks run on the main validator's GPU, and only for a challenger that has already won the private pool (plus the king and the init it is compared with). Other validators and satellites read the main validator's signed report."
status: proposed
date: 2026-10-19
tags: [consensus, scoring, benchmarks, validators, v2]
revisit_when: "a winning settlement's sweep regularly pushes judging past the next 3 h boundary (cap cascade_bench_max_series or move the sweep to an offload GPU); or lane B / C guards should move off the trainer too; or top-k (k > 1) promotion returns and needs every challenger's numbers; or the main validator's report is late often enough that the fleet's hold shows up as missed weight sets; or budget-deferred clearers often exhaust their two carries or land in an era's last settlement (raise the cap, or carry across eras); or checkpoint_fault verdicts are disputed (tighten the classifier toward infra); or settlements are sparse enough that the one-settlement record wait delays crowns materially"
relations: {depends_on: [DEC-CA-0054, DEC-CA-0055, DEC-CA-0058, DEC-CA-0061], relates_to: [DEC-CA-0057, DEC-CA-0036]}
---

## Decision

Owner (2026-10-19): "Can we have so that on main branch it scores on main
eval first then only evals the benches if the competitor wins and we will
actually have our dedicated GPU for validation."

"Main branch" means lane A, the king lineage. From `[v2]
bench_source_from_block` with `[v2] bench_source = "validator"`:

1. **Private pool first, unchanged.** Every validator scores the pool and
   judges the cohort exactly as before.
2. **Benches only for a winner.** The bench gate (DEC-CA-0054 / 0061) still
   runs only on the existing winning-result path (`_run_bench_gate`). Only
   there does the MAIN validator (`[netconfig] owner_hotkey`, the same key
   as DEC-CA-0057 / 0058) sweep the challenger and the era king on its own
   GPU, either `--device` or the `--eval-hosts` offload pod. The suites are
   GIFT-Eval, BOOM and TIME, plus FEV when the DEC-CA-0061 rule is in force.
   King rows are cached per checkpoint pointer, so a king is swept once.
   If no challenger clears the pool, nothing is swept.
3. **King check.** When the DEC-CA-0055 check is due, the main validator
   sweeps the king and the current init (cached the same way) at that
   settlement. The init is the era's `warm_start_ckpt`. In a random-init
   seed epoch it is the init pointer in the trainer's report, which only the
   main validator reads (a READ failure holds the settlement); every reader
   takes the init pointer from the main validator's report (`role = "init"`
   entry), so no validator defers on a local read failure while the main
   validator records a verdict. Under the DEC-CA-0061 rule the bench half
   compares all eight suite means (FEV included).
4. **One signed report per settlement.** It uses the same `BenchReport` and
   rows format as the trainer's report. The main validator signs it with its
   own hotkey and publishes it to `bench_v/round-<id>.json`, with rows under
   `bench_v/rows/<id>/` (per round, so a later re-sweep never changes bytes
   an earlier report pins). It is published before the settlement is marked
   handled, and it is published even when nothing was swept. If the publish
   fails, the judgement is rolled back and the settlement is held.
5. **Everyone else reads — only when they need to.** Other validators never
   sweep. A validator needs the report only when its own private-pool
   verdict sends a clearer to the bench gate, or the king check is due; both
   are computed identically on every validator, so a settlement with no
   clearer is judged without the report (a main validator that is down does
   not stall it). When it does need it, it reads it through THE ARBITER
   (item 7), verifies it against `[netconfig] owner_hotkey`, and judges from
   it. The receipt binds `bench_report = {key, sha256, signer}` (or `{key,
   signer, record: "absent"}`); a settlement that never needed the report
   binds nothing. Satellites and `cascade-audit` replay from exactly those
   bytes (the `bench-source`, `bench-coverage`, `bench-gate` and `king-check`
   rows). A satellite never sweeps.
6. **Failure semantics (2026-10-19 review; revised 2026-10-20, items 9-10).**
   A sweep that fails is retried (2 attempts) inside that checkpoint's own
   allowance (item 9); one that still fails is classified (item 10) and
   published in the signed report's `infra_failed` or `checkpoint_fault`
   list (pointer, role, reason) — distinct from an entry the report merely
   omits. INFRA is never the miner's loss:
   an infra clearer (an `infra_failed` side, or a report the trainer
   recorded `"absent"`) is SKIPPED — its streak is untouched, the gate falls
   through to the next clearer, and with none left the king holds
   (inconclusive). The verdict note records `bench_infra <hotkey>: <why>`.
   An omitted entry, rows failing their sha256, a report failing its
   signature, or published bytes differing from the trainer's record stay
   *uncomputable* (under `enforce`: inconclusive, the king holds). The king
   check defers on either. An infra clearer is not re-judged (its miner
   re-submits); a budget-deferred one is (item 9).
7. **The arbiter.** Whether a settlement's report counts is decided by the
   trainer, never by when a validator looked. The signed manifest of a later
   settlement records `validator_bench_reports[<round>]` = the sha256 of the
   exact text the trainer first read at `bench_v/round-<round>.json`, or
   `"absent"` once `[v2] bench_report_timeout_settlements` (default 8, ≥ 1)
   later settlements were built without it. Each round is recorded once; the
   EARLIEST later manifest that records it is authoritative. A validator that
   needs the report HOLDS until that record is in a manifest its chain walk
   holds, then reads exactly the recorded bytes (or applies "absent"). The
   MAIN validator does the same: on its first pass it sweeps, publishes, then
   rolls the judgement back and holds; after the record it re-judges as a
   reader. So nobody judges from a different view of the store, at the cost
   of one settlement's delay on every settlement that needed a sweep. The
   report is write-once: the first published report of a round is final
   (a restarted main validator reads it rather than re-sweeping or
   re-signing), and rows are content-addressed
   (`bench_v/rows/<round>/<pointer-hash>/<rows-sha256>.json`), so no later
   sweep can overwrite bytes a report pins. A settlement only resolves once a
   LATER settlement is published: the trainer publishes only when legs
   finish, so in a quiet network the hold lasts until the next settlement
   (uniformly — no split). The field is drop-when-empty, so every manifest
   without a record keeps its exact bytes.
8. **Trainer.** For those eras the trainer runs no lane-A challenger sweep
   (no payer pod time), no king sweep, no payer-pod verification queue, no
   bench hold and no `bench_first`. Only the once-per-seed-epoch init bench
   on the operator pod stays. It is the only carrier of a random-init
   epoch's init checkpoint pointer, and the validator re-sweeps that init
   itself rather than using the trainer's numbers. Promotion under `[v2]
   king_ladder` is k = 1, backed by a passing king check. Its member check
   (`_verify_members`) and the reign log (`_report_bench_scores`) fall back
   to the validator report. The trainer's member size/score metadata and the
   website pages read both reports.
9. **Per-clearer allowance, settlement cap, deferral (2026-10-20 red-team).**
   The single per-settlement budget let higher-ranked clearers that failed
   the gate or failed to sweep eat the time of later ones, and cut-off
   clearers were merely skipped. Now every checkpoint sweep has its OWN
   allowance, `[v2] bench_sweep_hours` (retries included; each attempt's
   timeout is what is left of it, under the `[eval]` ceiling). The king's
   sweep inside a clearer's gate has its own allowance and is never cut by
   the cap. `[v2] bench_budget_hours` is kept, as the settlement CAP on the
   sum of all sweeps: a clearer starts only while `cap − spent ≥ allowance`
   (`2 × allowance` when the king is not swept / cached yet — its slot is
   reserved), so no clearer ever starts on a truncated timeout. A clearer
   that does not fit is published `budget_deferred` (signed, distinct from
   `infra_failed`), never swept, SKIPPED on every validator (streak kept;
   note `bench_deferred <hotkey>: <why>`), and the gate falls through.
   **Carry (verified in code before building):** the era gate
   (`era_gate.py` `era_entry_out_of_window`) only requires `train_block`
   inside the era window and the ref bound as of `train_block`, so the SAME
   entry is valid in any later settlement of its era. The trainer's "burn"
   is its own one-submission ledger (`_burn_hotkeys`), not a validator rule,
   and settled legs stay in `RollingState.published` for the era. So the
   trainer, when it records a report (item 7) whose `budget_deferred` names
   one of its own published legs of the current era, puts that leg into the
   SAME manifest that records the report (`_vbench_carry`), at most
   `VBENCH_MAX_CARRIES` = 2 times per leg (`RollingState.vbench_carries`),
   never alongside a fresh leg of the same hotkey. Every validator then
   judges it again exactly like any other leg (private pool on that
   settlement's draw, bench gate) — deterministically, because the carry is
   in the signed manifest. Bounds: a deferral at an era's last settlement,
   past two carries, or in an era where no later manifest is published
   (the trainer publishes only when fresh legs finish) is not re-judged; the
   clearer stays skipped (never a loss) and its miner re-submits. The
   alternative (re-applying the ORIGINAL settlement's verdict later) was
   rejected: settlements are handled in order and their state transitions
   already applied, so a late verdict would re-order the throne history.
10. **Checkpoint faults (2026-10-20 red-team).** Not every sweep failure is
   infra. `classify_sweep_failure` (`cascade/eval/benchmarks.py`) calls a
   failure the CHECKPOINT's only on positive evidence with no environmental
   marker: a required suite `ok` but with a non-finite crps/mase, or
   `error` with every requested config failing at the sidecar's `score`
   stage (the model's own load / shape / forward). Anything with an
   OOM / CUDA / timeout / network / storage / dataset / download marker, a
   `load`-stage (data) failure, a skipped / absent suite, a sidecar crash or
   timeout (no report), a raised exception (checkpoint fetch, offload) is
   INFRA. A fault must be REPRODUCED: both attempts must classify as
   checkpoint, else the sweep is infra (an attempt cut by the allowance
   counts as not reproduced). The report's `checkpoint_fault` list is
   signed; a CHALLENGER in it fails the bench gate on every validator
   (`checkpoint_fault_bench_gate`: computed, `passed = false`, reason
   `checkpoint fault: <why>`), its streak resets like any gate block, and
   the note records `bench_fault <hotkey>: <why>`. A king's or init's fault
   is infra (never the clearer's loss; the king check defers). The suites
   already passed the checkpoint through the private pool's ingest guard,
   so a scoring-stage failure there is the checkpoint's, not the code's.

## Config

- `[v2] bench_source = "trainer" | "validator"`. The default is "trainer",
  which keeps today's behaviour bit-identical.
- `[v2] bench_source_from_block`: an integer era boundary, or `"v2"`
  (= `[v2] from_block`). 0 means never. A "trainer" source always resolves
  to 0.
- Mainnet is `"validator"` from `"v2"`, so it is inert until the go-live is
  typed. Testnet is unset (trainer).
- Load checks apply once the resolved block is > 0. The block must be an
  era boundary at or after `[v2] from_block`. `[v2] king_ladder = true` is
  required, because top-k promotion selects over every challenger's numbers
  and the validator benches only winners. `[netconfig] owner_hotkey` must
  be set, because without a main validator every validator would hold
  forever. So typing the v2 go-live on mainnet now also requires naming the
  main validator.
- Sweep settings reuse the existing `[eval]` keys: `benchmark_project_dir`,
  `benchmark_num_samples` (0 = `num_samples`), `cascade_bench_max_series`,
  `bench_hold_max_hours` (the per-sweep timeout ceiling) and
  `gift_gate_data_dir` (the staged data).
- `[v2] bench_report_timeout_settlements` (int ≥ 1, default 8): later
  settlements after which the trainer records a missing report `"absent"`.
- `[v2] bench_sweep_hours` (> 0 and ≤ `bench_budget_hours`, default 2.0;
  mainnet 2.0): ONE checkpoint's sweep allowance, retries included (item 9).
- `[v2] bench_budget_hours` (> 0, default 4.0; mainnet 6.0 = a first
  clearer + the king, then one more clearer against the cached king): the
  settlement cap over every sweep (wall clock on its own box). A clearer it
  cannot hold a whole allowance for is published `budget_deferred`; a king
  check sweep it cannot hold is `infra_failed: "bench budget exhausted"`.
  The outcome is published, so readers never see the main validator's clock.
- All three are overlay-settable after the 2026-10-21 flexibility update
  (DEC-CA-0057); the fleet must first install that capability release.
- `[manifest] validator_hotkey` (the trainer's / audit's receipt anchor) and
  `[netconfig] owner_hotkey` (the report signer) both name the main
  validator. Neither is derived from the other (a silent derivation would
  move a trust root); once the validator source has a block, the loader
  refuses the two when both are set and differ. Mainnet's `validator_hotkey`
  is a local override, so typing the go-live means setting `owner_hotkey` to
  the same key.
- A main validator (wallet hotkey == `owner_hotkey`) whose wallet cannot sign
  refuses to start (`check_main_bench_signer`), and the publish path never
  publishes an unsigned report.

## Runtime policy by overlay (DEC-CA-0057)

Updated 2026-10-21: once the fleet has the benchmark-source implementation
and expanded overlay allow-list, `bench_source`, `bench_source_from_block`,
`bench_gate_v2_from_block`, report timeouts and sweep budgets are
overlay-settable. A source selects the already pinned trainer or main
validator; their identity keys remain release-only. Normal loader checks,
future activation boundaries and stake lock-in still apply. Old nodes
need this capability release once; an overlay cannot install benchmark
logic on a node that does not have it. `"v2"` continues to resolve from
`[v2] from_block`.

## Out of scope

The lane-B / lane-C public guards (DEC-CA-0050 / 0056 / 0059) keep their
trainer-produced rows.

## Budget against the 3 h cadence

A winning settlement now includes one sweep (king cached) or two (new king
pointer, or a king cached on another data revision). The king-check
settlement includes up to two (king and init). The trainer's figure is about
1 h per full-battery checkpoint (not re-measured here), and mainnet's
`bench_hold_max_hours` ceiling is 9 h per sweep. Each sweep is bounded by
its allowance (`bench_sweep_hours`, 2 h); `[v2] bench_budget_hours` (6 h)
caps the whole settlement, and since a clearer starts only when its whole
allowance (+ the king's) fits, the worst case is the cap plus one
revision-mismatch king re-sweep. Clearers past it are budget-deferred and
carried (item 9). Weights are re-asserted meanwhile.
`cascade_bench_max_series` or a faster offload GPU are the levers. The
trainer no longer pays roughly one sweep per leg plus one operator re-bench.

**Owner measurement item (before arming):** time one full GIFT-Eval + BOOM +
TIME (+ FEV) sweep of a v2 checkpoint on the main validator's GPU / offload
pod, and set `bench_sweep_hours` to cover one sweep with its retry and
`bench_budget_hours` to the clearers a settlement should judge (king
cached: one allowance each; new king: one more). No sweep timing exists in
this repo.

**2026-10-20 note — concurrent and paired sweeps (operator-local).** The
sidecar now splits every suite into work units (one GIFT config / BOOM
dataset / TIME task / FEV task) run over N spawned workers
(`--workers auto|N`; `auto` = min(free VRAM ÷ measured per-worker peak,
CPU − 1, cap)), and can sweep several checkpoints in ONE pass
(`--extra-checkpoint`: each dataset built once). Each report is
byte-identical to a sequential solo sweep, so neither knob is consensus: only
the main validator sweeps, everyone else replays its signed rows. The main
validator uses them via `[eval] bench_workers = "auto"` and `bench_paired =
true`: challenger + unswept king (and king + init at the king check) are one
pass. Charging: the pass may run for its members' combined
`bench_sweep_hours`; its wall time counts once, in full, toward
`bench_budget_hours` and in equal shares toward each member's allowance, and
it is each member's first attempt (a member that failed retries alone within
what its allowance has left). A pass starts only while the cap holds both
allowances; otherwise the members are swept one at a time as before. The
budgets above should be re-set from `scripts/bench_timing.py` measurements
(docs/OPERATOR.md "Sizing the sweep").

## Audit (report-only)

- `bench-coverage` (tier 0): the decided challenger of a verdict that
  cleared the pool reached the gate, so its pointer and the king's must be in
  the report as an entry or as `infra_failed`; an OMITTED one FAILs. Every
  `bench_infra` hotkey in the note must be infra-failed in the report (or the
  report recorded absent). Every `bench_deferred` hotkey must be
  `budget_deferred` in the report and every `bench_fault` hotkey a
  `checkpoint_fault`; conversely a challenger the report lists deferred
  (under `enforce`) or faulted must appear in the note — a validator that
  ignored the signed outcome FAILs. Clearers before the decided one in a
  fall-through are only checked through the note.
- `bench-gate` (tier 0): a decided challenger listed `checkpoint_fault`
  replays as a failed gate (`passed = false`, same reason).
- `bench-source`: a note with `bench_deferred` counts as the gate having run.
- Not audited per receipt: that the trainer carried a deferred leg (the
  carrying manifest is a later settlement's; its entries are judged like any
  other).
- `cascade-audit bench <round> [--reproduce]`: the report's signature and
  every rows sha256; with `--reproduce`, a re-sweep of each entry's public
  checkpoint through the same sidecar, comparing a deterministic sample of
  items (`--sample`, default 20 per suite) at a relative `--tolerance`
  (default 5 %: GPU sweeps are not bit-reproducible and forecasts are
  sampled). Never consensus; WARNs when torch / the sidecar is unavailable.
