# Auditing cascade rounds — `cascade-audit`

Training is owner-operated. `cascade-audit` lets readers check published
signatures, decisions and reward accounting, with deeper reproduction where
the required data and hardware are available.
Each settlement (a "round" in receipt and CLI vocabulary) binds chain context
and published signed evidence. Replay follows the effective historical rules:

* the **participant set** is the on-chain generator commitments revealed
  strictly before the settlement boundary, each entry bound to its hotkey's
  commitment as of the entry's `train_block`;
* the **training seeds** (`RoundSeeds`) derive from the recorded era/seed-epoch
  block hash (`era_base_seed`) under the effective rule; the settlement's own
  boundary block hash (`base_seed`) keys the round id and the eval draw;
* the **contract** is the effective era contract, including the manifest's
  digest-bound declaration and the applicable signed configuration history;
* the **verdict** is a deterministic paired bootstrap over recorded per-window
  scores;
* the **weights** use historical court decay or the v2 lane allocations,
  king ladder and eligible B/C holders. Replay must use that settlement's rule.

After each round the validator publishes a signed
[`RoundReceipt`](../cascade/shared/receipt.py) to the manifest bucket under its
own prefix (`receipts/<hotkey>/round-<id>.json` + `receipts/<hotkey>/latest.json`,
plus the shared `receipts/latest.json` pointer — per-validator prefixes mean
concurrent validators never overwrite each other's audit trail) recording all of the
above — including the trainer's manifest verbatim and every per-window score
that fed the KOTH bootstrap. A round the validator *rejected* still gets a
receipt (`"status": "rejected"`) carrying the gate's reason.

`cascade-audit` re-derives a receipt at three tiers:

| tier | cost | needs | verifies |
|------|------|-------|----------|
| 0 (default) | seconds, CPU | repo + `chain.toml` (chain optional) | signatures, seed derivations, contract/arch digests and the declared contract body, the era envelope, the on-chain activation blocks, commitment cutoff, the KOTH verdict (and every cohort challenger's bound) recomputed from the receipt's own scores, transition consistency, the weight vector, funded-roster seniority (WARN-only) |
| 1 | generation-dependent, CPU | + generator access or local shard corpus | Recreate supported generator streams or verify supplied shards, then compare the consumed corpus digest; unavailable inputs WARN |
| 2 | hours, GPU (`[train]` extra) — **experimental** | + a training GPU | + re-train from the contract under the recorded seeds/budget; byte-exact checkpoint comparison on matched `gpu_name` + `train_image_digest`, else eval-score comparison within a documented tolerance |

No credentials are required where the storage allows it: receipts are fetched
with an unsigned S3 request first (falling back to `HIPPIUS_S3_*` if set), or
read from a local file with `--receipt`.

**Choose a check:** [quick start](#quick-start) · [interpret results](#reading-the-output)
· [what replay proves](#what-each-tier-proves) · [MarketCast](#marketcast-rows-dec-ca-0064)
· [B/C reward tiers](#gain-tiered-full-pay-period-lane-b--lane-c-owner-2026-10-20)
· [trusted signers](#trust-anchors).

## Quick start

```bash
pip install -e '.[hippius]'        # boto3 for the receipt fetch
cascade-audit latest               # tier 0 on the newest round
cascade-audit round 2269901645662351552 --tier 1
cascade-audit latest --json        # machine-readable; exit 1 on any FAIL
cascade-audit round <id> --validator 5F…   # one validator's receipt specifically
```

Without `--validator`, `latest` reads the shared pointer (the most recently
published receipt, whichever validator wrote it) and `round <id>` discovers the
receipt through the public `receipts/index.json` (legacy un-namespaced rounds
are tried first).

A completed audit exits 1 if a check FAILs. WARNs alone can still exit 0;
inspect the rows before calling an audit complete. Fetch/usage errors can also
stop the command without a completed audit. A CI invocation is:

```yaml
- run: cascade-audit latest --tier 1 --json
```

## Reading the output

Each check reports one of:

* **PASS** — the receipt matches an independent re-derivation.
* **FAIL** — the receipt *contradicts* a re-derivation. This is the signal;
  the exit code turns nonzero.
* **WARN** — could not fully verify (no chain connection, lite node without
  the historical block, pruned commitment history, unpinned signer, a
  stream-mode corpus without `--full-stream`). Deliberately explicit — a check
  never silently passes because its inputs were unavailable.
* **SKIP** — not applicable (e.g. verdict checks on a rejected round).

On a `rejected` receipt, a check that re-detects the recorded rejection reason
(e.g. `manifest-signature` on a `signature_invalid` rejection) reports PASS
with a note that it *confirms* the validator's gate.

## A worked transcript

Historical-format example, not a newly executed audit: Tier 0 against a signed
receipt, offline (`--no-chain`), with both signers pinned in
`chain.toml [manifest]`. Current v2 receipts can carry additional checks.

```text
$ cascade-audit round 2269901645662351552 --receipt round-2269901645662351552.json --no-chain
round 2269901645662351552  status=scored  tier=0

  [PASS] status              scored round
  [PASS] receipt-signature   signed by 5CzRdZLnRxWkyWoGZZbfvSCY4mbw1cmsow5q8cgZyGg1aiDq
  [PASS] manifest-signature  signed by 5DpzeqS6r8xy5ajWGJfsPxKEsWEj7bPTN6Bz6knnvez5PPiU
  [PASS] base-seed           base_seed 2269901645662351552 derives from the recorded block hash
  [PASS] round-seeds         generation + training seeds derive from base_seed
  [PASS] epoch-alignment     boundary 21600 on the epoch grid
  [WARN] block-hash-onchain  no chain connection; recorded block hash not verified on-chain
  [PASS] contract-digest     contract_digest d6e93bf5fb306408… matches chain.toml
  [PASS] base-arch-digest    manifest matches the pin; pin recomputes from local source
  [WARN] commit-cutoff       all 2 participant(s) pre-cutoff and entries match the recorded set; chain payloads not cross-checked (no connection)
  [PASS] koth-params         recorded params match chain.toml [scoring]
  [PASS] verdict             lcb=0.40000 margin=0.02000 win=True reproduced over 1 size(s)
  [PASS] transition          transition 'dethroned after 1 consecutive win(s)' consistent with the verdict
  [WARN] weights             vector recomputes; on-chain weights not compared (no chain connection)

  WARN=3  PASS=11
$ echo $?
0
```

Drop `--no-chain` and the three WARNs resolve on-chain: the recorded block
hash is compared against the node, participant payloads against the visible
reveals, and the weight *support* against the validator's current row.

Now tamper with one byte — say the manifest's `contract_digest`:

```text
$ cascade-audit round 2269901645662351552 --receipt tampered.json --no-chain
  ...
  [FAIL] receipt-signature   signature does not verify against 5CzRdZLnRxWkyWoGZZbfvSCY4mbw1cmsow5q8cgZyGg1aiDq
  ...
  [FAIL] contract-digest     manifest contract_digest 0000000000000000… != local chain.toml d6e93bf5fb306408… (different training contract)
  ...
$ echo $?
1
```

Every field of the receipt is inside the signed canonical body, so any
mutation kills the signature *and* trips its specific semantic check (the test
suite asserts one tamper test per check — `tests/unit/test_audit.py`).

## What each tier proves

**Tier 0** checks internal consistency against the public contract: whether
the verdict follows from the recorded scores under the
recorded seed and the published `[scoring]` params, the seeds really derive
from the recorded block hash, the participant set respected the deadline, and
the published weight vector is exactly the decayed-share function of the
recorded reward state under the applicable court or v2 lane rules. What it
takes on trust: that the recorded *scores*
describe the published checkpoints. Experimental Tier 2 and the separate
benchmark/MarketCast reproduction commands provide deeper checks. Offline, it also trusts that
the recorded block hash is the real chain's (drop `--no-chain`, or check any
block explorer).

On a cohort round (more than one challenger in the settlement) the receipt
also publishes, per challenger, its bootstrap bound under the family-wise
correction in force (`verdict.cohort_lcbs`; max-T, DEC-CA-0038) and its shadow diagnostics — observed geomean, window
win rate, Wilcoxon p, bootstrap p50/p95, per-domain win rates
(`verdict.cohort_stats`). Both are signed and both are replayed from the
recorded scores by the `duel-cohort` check: a published number that does not
reproduce fails the round. The diagnostics never gate; they are there so a
dashboard or a miner can read every challenger's outcome, not only the
crowned one's, without re-running the bootstrap.

When the lane-A bench gate ran (DEC-CA-0054 / DEC-CA-0061 rule v2), the `bench-gate`
check fetches the round's trainer-signed bench report
(`benchmarks/round-<id>.json`, signature checked against `[manifest]
trainer_hotkey`) and the per-item rows each entry pins by sha256, then
replays the gate. The recorded `params.bench_gate_rule` must be the rule in
force at the receipt's block (`[v2] bench_gate_v2_from_block`), the outcome
must reproduce, and under DEC-CA-0061 (`"avg_noninf_winrate"`) the recorded
`verdict.bench_gate_detail` must match the replay too: the three numbers
(`avg_ratio`, `worst_lcb`, `win_rate`), the thresholds (`tolerance`,
`min_win_rate`, from `params.bench_gate_tolerance` / `bench_gate_min_win_rate`,
dropped at their 0.01 / 0.5 defaults) and per suite `n`, `masked`, `ratio`,
`ratio_lcb` and `win_rate`. The unit is one task (one signed item row); the
bootstrap seed is the verdict's (`{seed}|bench|{suite}`), so the replay is
bit-identical. If the report cannot be fetched the check WARNs, after first
testing that the recorded per-suite numbers re-derive the recorded three and
imply the recorded outcome.

From `[v2] bench_source_from_block` (DEC-CA-0063) the rows come from the
main validator's report, not the trainer's. The receipt binds that report
as `bench_report = {key, sha256, signer}` (or `{key, signer, record:
"absent"}` when the trainer recorded the report absent). The `bench-source`
row checks the following. The binding is present on every settlement of such
an era whose bench gate ran (or whose king check read the report), and
absent on trainer-sourced settlements; a validator-sourced settlement with no
clearer is judged without the report and binds nothing. The key is
`bench_v/round-<id>.json`. The signer
is `[netconfig] owner_hotkey`. The published bytes hash to the bound sha256.
A king check on that settlement must say `bench_source = "validator"`.
`bench-gate` and `king-check` then replay from exactly that report. A report
that fails the owner signature replays as uncomputable, which is what every
validator saw. In this mode the king check's `bench_round` is the settlement
itself, not the previous one.

The `bench-coverage` row (DEC-CA-0063) checks that the report covers what the
verdict sent to the gate: the decided challenger of a verdict that cleared
the pool, and the king, each have an entry or an `infra_failed` record — an
omitted one FAILs — and every `bench_infra` hotkey in the verdict note is
infra-failed in the report. The same row covers DEC-CA-0063 items 9-10.
Every `bench_deferred` hotkey must be `budget_deferred` in the report, and
every `bench_fault` hotkey must be a `checkpoint_fault`. The reverse also
holds: a challenger the signed report lists as deferred (under `enforce`) or
faulted must appear in the note, so a validator that ignored the signed
outcome FAILs. `bench-gate` replays a decided challenger's checkpoint fault
as a failed gate with the same reason. That a deferred leg was carried into
a later settlement is not checked per receipt. Re-sweeping the numbers is a
separate, report-only command:

```bash
cascade-audit bench ROUND_ID                         # signature + rows sha256
cascade-audit bench ROUND_ID --reproduce --device cuda --sample 20 --tolerance 0.05
```

`--reproduce` re-runs the benchmark sidecar on each entry's public checkpoint
and compares a deterministic sample of per-item CRPS / MASE at a relative
tolerance (default 5 %). Quantile-capable checkpoints use batched quantile
forecasts; older sampling wrappers use samples. Hardware/library differences
can prevent bit-identical GPU sweeps. Missing torch or sidecar support WARNs.

On a v2 settlement under `[v2] king_ladder` the `king-check` row
re-derives the DEC-CA-0055 king check (which sets the lane-A ladder share and
is the only evidence a k = 1 promotion is accepted on) instead of trusting
the recorded `king_check.result`:

* **Due-ness.** The ladder record (`emission.ladder.reign_start_block`)
  carries the reign clock the validator read before the check restarted it.
  A check recorded where `v2_check_due` is false under the receipt's config
  FAILs. A missing check where the clock says due FAILs. A missing check on
  a receipt with no recorded clock WARNs: the validator skips the check
  while its cascade view has not adopted the champion yet, and the receipt
  cannot show that.
* **Pool half.** `evaluate_round(init, king, cfg.king_check_params(boundary),
  seed=base_seed, king_tenure_rounds=…, baseline_scores=init)` is replayed on
  the recorded rows. The king's rows come from `entry_scores` (role `king`)
  and the init's from `entry_scores` (role `baseline`) when the verdict scored
  them. Otherwise the check's own rows are in the receipt's
  `king_check_scores` (roles `king` / `init`). The result, `lcb`, `margin`
  and reasons must reproduce. Tenure is the verdict's on a scored receipt.
* **Bench half.** Under DEC-CA-0054, `bench_failures` is replayed on the
  recorded six suite means (`king_check.bench`). Those means are then
  compared with the PREVIOUS settlement's trainer-signed bench report
  (`benchmarks/round-<bench_round>.json`). Under DEC-CA-0061 the bench half
  is the gate's rule with the king as challenger and the init as reference
  (`king_check_gate`, seed `{base_seed}|king-check`, thresholds from
  `cfg.koth_params(block)`): the recorded `king_check.bench.gate` detail
  must imply the result, and with the king's and the init's sha256-pinned
  task rows (the report's entries' `rows_key`) the gate is replayed and must
  match the detail. Without the report (or the rows) the row is
  `king-check-bench` WARN after the pool half and the internal consistency
  pass. A satellite treats that WARN as "could not verify" and retries.
* **Deferrals.** The stated reason is checked where it is derivable: no king
  entry, no report or entry in the signed report, and too few windows (the
  replay must be inconclusive too). Transient reasons (store read, an
  unscorable eval) WARN.

The receipt also carries pure DISPLAY breakdowns the audit does not replay:
`verdict.per_horizon` / `cohort_per_horizon` (the verdict geomean per ladder
rung) and `verdict.per_domain` / `cohort_per_domain` (the same per pool
domain — which domains each challenger beat the king in, and by how much;
what `cascade duel --hotkey` and the website's cohort panel render). They
are signed like everything else, so they cannot be altered after the fact,
but no check re-derives the per-horizon split. Per-domain: receipts written
by DEC-CA-0050 P2 validators record each window's `domain` on its
`entry_scores` row (drop-when-default — receipts of unlabeled pools, and every
archived receipt, keep their bytes), so the `duel-cohort` check's
`per_domain_win_rate` comparison activates on those receipts automatically,
and the signed per-domain scores are enough for anyone to re-derive a lane-B
domain verdict (target-domain gain + per-domain non-inferiority guards) with
`cascade.eval.expert_lane.judge_expert_leg`. Older receipts keep the Tier-0 limit (no domain
labels; the comparison is skipped). All of them are drop-when-default, so
older receipts verify unchanged.

A receipt judged after the validators decided the DEC-CA-0043 rollover on
chain (DEC-CA-0045, stake-weighted activation) records that block as
`activation_block` (absent when the rollover is typed into `chain.toml`). The audit REPLAYS the round
under that block — every DEC-CA-0043 rule (era envelope, tenure in blocks,
the grid switch, the increment-unit max-T) is evaluated as if the block had
been typed in — so a `chain.toml` with the rollover keys at 0 still verifies
a post-rollover receipt. The `activation` check then verifies the block
itself: it must be a boundary of the grid before it, and, with a chain, the
validators' current on-chain notes (`cascade-ready:1:<feature>:<lock>:<block>`)
must name the SAME block by the configured stake threshold — a validator
that stamped a block the fleet did not agree on FAILS; without a chain the
check WARNs, like every chain-dependent half. The `margin-v2` check does the
same for the DEC-CA-0049 dethrone-bar block (`margin_v2_block`), and `era`
replays the DEC-CA-0043 envelope: era context is required after
`era_king_from_block` and forbidden before it, every entry's `train_block`
sits inside the era's window, and — with a chain — `era_base_seed` is the
seed of the era's seed block.

Every block-gated rule stays in the code after it activates, so each receipt
replays under the rule in force at its own block.

**Tier 1** checks the consumed corpus digest, with different scope by feed:

* `cache_reuse` corpora byte-compare in one materialisation.
* `stream_cpu` digests cover the *full consumed training budget* (that's what
  makes them binding), so re-deriving one costs about the round's own
  generation time — run with `--full-stream` to spend it; otherwise the check
  WARNs rather than pretending.
* `stream_gpu` is tolerance/same-hardware by design and never byte-compares on
  CPU.
* `shards` verifies shard hashes and replays the same budget-limited consumed
  stream; it does **not** regenerate the shards from miner code. The library
  entry point `cascade.audit.rederive.run_tier1(..., corpus_root=PATH)` expects
  `PATH/<hotkey>/corpus.json` and its shards. The current `cascade-audit` CLI
  does not expose that `corpus_root` argument, so shard entries WARN there.
  Public generator/recipe access alone does not provide the complete corpus.

**Tier 2 (experimental)** closes the loop: re-train from the contract under
the recorded seeds and token budget. On a runtime matching both the entry's
recorded `gpu_name` and the contract's `train_image_digest` pin, the
checkpoint must reproduce **byte-identically** (the reference trainer is
deterministic). On other hardware, the retrained checkpoint is scored against
the published one on the receipt's eval slice and must agree within
`TIER2_SCORE_RTOL` (5% relative on geomean(CRPS, MASE) — generous against
kernel-reduction drift, tight against a swapped corpus or checkpoint).

## MarketCast rows (DEC-CA-0064)

At a settlement where `[marketcast]` is active, Tier 0 gains a `marketcast`
check, placed right after `lane_c`. Before that settlement the check is
absent, so audit output for earlier receipts is unchanged.

A MarketCast round's rows land in the receipt of the settlement that APPLIES
it: the first one whose trainer-signed lane-C manifest records the round's
report in `mkt_reports` (its sha256, or `"absent"`). Each `mkt:` row names its
round (`mkt.round`), the round whose lane-C manifest carried the entry
(`mkt.entry_round`) and the trainer's dispatch time (`mkt.train_time`).
Rows stamped `mkt.at = "register"` are pre-check, duplicate or eviction rows
written when the entry was first seen.

`cascade-audit` fetches the bound `marketcast/round-<id>.json` reports, their
per-window quantile sidecars (`marketcast/round-<id>/q-*.npy`, public-read)
and the lane-C manifests the rows name. It checks:

- the report text's sha256 is the bound one, it verifies against
  `[netconfig] owner_hotkey`, and it names the row's round;
- the entry's `after` is the TRAINER's `train_time` (read from the
  trainer-signed lane-C manifest that carried the entry) plus
  `start_margin_seconds`, and on a confirm ≥ the recorded `fresh_after`.
  The main validator's snapshot anchor never sets it; a train block more
  than an hour off the anchor is noted;
- with the published snapshot (`--mkt-snapshots DIR` on `cascade-audit
  marketcast`): the report's window set EQUALS the forward-only set
  `snapshot_windows` re-cuts — nothing omitted, nothing added. Without it,
  every window must start after `after`, end by the snapshot's end, and no
  (asset, horizon) cell hold more than `windows_per_cell`;
- with the snapshot and the sidecar (sha256 bound in the signed report):
  every per-window loss and coverage tally is recomputed on CPU from the
  snapshot's targets and the published float32 quantiles
  (`score_windows`) and must match exactly;
- the verdict re-derived from the rows (paired geomean LCB at the recorded
  `k`, the coverage gate on `mkt:returns`, minimum windows and assets)
  reproduces the row's pass, gain LCB, windows and clusters exactly;
- an `infra` row is consistent only if no usable rows existed (report absent
  or unverifiable, the entry missing or `infra_deferred`, or rows cut after
  another instant).

Red-team fixes (2026-10-20). The per-window `tail` tallies are recomputed
with the losses. The verdict uses the (class × horizon) mean-CRPS ratio and
the class-weighted calibration gate (central + tail rows, on `mkt:returns`
and `mkt:vol`). The row's `mkt.masked_cells` must equal the re-derived count.
Each judge-phase row's `k` must be `C × m` over the round's owners: the
trainer-stamped coldkey in `mkt.owner`, checked against the trainer-signed
lane-C manifest when fetched; `check_lane_c` applies the same rule. A signed
`near_dup` record must be copied onto the row, and the stage must follow
`near_dup_tol`: below it the row is a `duplicate`, never a verdict. With the
reference sidecar (`marketcast/round-<id>/r-*.npy`, sha256-bound) the
distance is recomputed exactly and its base plane must equal the entry's.

It also FAILs a report row with a non-finite fine-tune loss (that entry is a
fault). Since review II (2026-10-19) the re-cut window set includes the
eligibility rules (spread selection, the liquidity floor, the stale-context
rule — the stale-TARGET floor was removed by the red-team fixes) and the verdict uses asset class × 6 h time-block clusters, equal
horizon and class weights, `min_time_blocks`, and the per-horizon two-sided
coverage gate — all re-derived from the signed rows and the snapshot. The
`gain_vs_base_cov` diagnostic (base given the fine-tune's covariates) is
reported, not gated and not recomputed.

It PASSes only when every input was available, and WARNs naming what was not
fetched. `check_lane_c` checks `mkt:` rows against the `[marketcast]` terms,
groups the Bonferroni family and the best passer per MarketCast round,
accepts the `infra` and `duplicate` stages on `mkt:` rows only, and requires
an infra drop past `max_infra_deferrals` to be `dropped`. `check_lane_c_chain`
follows MarketCast carries across receipts (a round may apply several
settlements later).

`cascade-audit marketcast <id> --mkt-snapshots DIR --reproduce` goes further
and checks the inputs against the world:

- it re-fetches a deterministic sample of the snapshot's bars from
  Hyperliquid's public `candleSnapshot` and compares closes within
  `--bar-tolerance-bp` (1 bp). Hyperliquid serves only its most recent ~5000
  candles per interval (measured 2026-10-19: 1m ≈ 3.6 days, 1h ≈ 208 days),
  so it samples the hourly bank and the still-served minute bars; bars the
  feed no longer serves are counted, not failed;
- with torch and the `marketcast` extra installed, it re-forecasts a sample
  of windows (`--sample`) from the public checkpoints (the pinned base and
  the fine-tune) and compares them with the sidecar within
  `|Δ| ≤ 1e-3 + rtol·|q|` (`--tolerance`, default `rtol` 2e-3: float32
  rounding, cross-GPU noise, one-window NaN padding).

It reports and never changes consensus.

## Lane C manifest and payout checks (DEC-CA-0056)

Lane C currently accepts MarketCast only. Every lane-C row is an `mkt:`
row (above). A signed lane-C manifest with any other target cannot be loaded
and is recorded as a `lane_c_manifest_malformed` error row. No lane-C pool
re-score exists: the `pool_c` verdict is retired. With MarketCast off, the
payout record has `slots = 0`. If C has launched, its reserved share burns;
before C launches it has no pool.
The vector rebuild expects exactly that.

`check_lane_c` accepts the `duplicate` stage (a later commit of weights an
earlier commit holds) and skips the `lane_c_duplicate_evicted` error rows
(a copy losing its slot). `_lane_c_paid_wins` checks the payout record's
`slots` against the slot-bearing targets at the settlement — an `mkt:`
target counts only once `[marketcast]` is live — and the vector is rebuilt
from the per-target `slot_frac` (concurrent competitions, below). A lane-C payout record without a lane-B one is reported, never
rebuilt: lane C without lane B is refused.

## Concurrent lane-C competitions (DEC-CA-0056, owner 2026-10-20)

A lane-C payout record carries `competitions` (per configured competition:
`kind`, `weight`, `from_block` — `[marketcast] from_block` for `mkt` —
`until_block`, `wind_down`, `wind_down_max_eras`, `frac` = weight over the
total of every configured competition, `state`, `targets`, `slot_frac`),
`slot_frac` (per slot-bearing target) and `slots` (their count).
`_lane_c_paid_wins` re-derives all of it from the config in force
(`cascade.shared.lane_c.lane_c_split`) and fails on any difference, so a
forged per-competition share or state is rejected. A competition is
`pending` before its start, `active` through its `until_block`,
`winding_down` for fewer than `wind_down_max_eras` eras after it (only under
`wind_down = "finish_full_pay"`), then `ended`. A paid win of a
winding-down competition must carry `"wind_down": true`, multiplier 1.0 and
be inside its own full-pay period; any other win must not carry the flag.
The vector is rebuilt with `share × slot_frac[target] × multiplier` per paid
win, the rest of lane C's share on the burn uid. `check_weights` and
`derive_weights` run this, so satellites do too. `check_lane_c` fails any
row of a competition past its end other than an `ended` row (not passed,
outcome `""` or `dropped`), and any `ended` row of a competition that has
not ended.

## Gain-tiered full-pay period (lane B / lane C, owner 2026-10-20)

A lane-B / lane-C win pays in full for the `full_eras` of the highest
`full_pay_tiers` threshold its gain LCB meets (`[lane_b]` / `[lane_c]`;
`[marketcast]` for an `mkt:` target), fixed at the win. The audit never
trusts the recorded tier:

- `check_lane_b` — a slot won at this settlement: the payout record's
  `gain_lcb` equals the winning row's `target_lcb` (per target of a
  multi-target leg), and `tier` / `full_eras` / `below_floor` re-derive from
  it (`cascade.eval.expert_lane.full_pay_mismatch`).
- `check_lane_c` — a passing confirm row's `full_pay` re-derives from its
  `gain_lcb` under the target's tiers; the slot paid at this settlement
  carries the confirmed row's `gain_lcb` and `full_pay`.
- `_lane_b_paid_wins` / `_lane_c_paid_wins` (the payout replay behind
  `check_weights` and `derive_weights`, so satellites run it too) — every
  paid win carries the four fields; `full_eras` is an integer in `[0,
  win_ttl_eras]`; a win made at this settlement re-derives under this
  receipt's tiers; an OLDER win is not re-derived under today's tiers (a
  signed overlay may have changed them since) but must be self-consistent
  (`below_floor` ⟺ `gain_lcb < tier`) and equal its slot in the receipt's
  `state_post` (`[…, gain_lcb, tier, full_eras]`), which the satellite
  continuity check binds to the previous receipt; the multiplier is replayed
  from the recorded `full_eras`.

What a single-receipt audit cannot see: a forged tier on an older win in a
receipt without `state_post` — walk the receipts from the win's settlement
(or run a satellite) to bind it.

## Direct-MoE journals (DEC-CA-0052)

`cascade-audit direct-moe` audits the v2 direct lane, whose validators never
evaluate. Without flags it verifies the trainer's signature (`--trainer-hotkey`)
and replays every event and decision from the journal's own scores — what each
validator does. `--reproduce` adds the part validators skip: it fetches each
decision's bound checkpoints from the public bucket (credential-free), re-scores
them on the revealed private panel and on the public suites (`--sidecar`, pinned
to the decision's dataset revisions), and compares every per-case score
(`--tolerance`, default 1e-3 relative) and the re-derived verdict with the
recorded ones.

```bash
cascade-audit direct-moe --journal-bucket BUCKET --trainer-hotkey TRAINER_SS58 --reproduce --device cuda
```

A private panel is only revealed after the trainer rotates it out, so a recent
decision's private half WARNs until then; the public half reproduces at once.

## Trust anchors

Two ss58 addresses in `chain.toml [manifest]` anchor the audit:

* `trainer_hotkey` — must sign the embedded `TrainingManifest`.
* `validator_hotkey` — must sign the `RoundReceipt`. If left empty, the audit
  verifies the signature against the receipt's self-declared signer and WARNs
  that the signer is unpinned (internal consistency only).

Everything else is *derived*, not trusted: digests recompute from the repo,
seeds from the block hash, the verdict from the scores.
