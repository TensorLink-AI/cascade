# W9 acceptance tests: the audit re-derives lane D

W9 (plan rev6 §3.0 W9 row, depends on W7c and W8): "Audit `check_lane_d` +
`derive_weights` (**rev3:** re-derives the judged set, coldkey caps,
both-clock eligibility, the confirmation rule and `rows_sha256`),
independent of producer code", plus its **rev6.1** additions: the audit
refuses a judged snapshot in which a series lacks `frontier` or a feed isn't
listed in `feeds`, and re-derives S0 presence, the backfill rule and
mirror-lag eligibility from the snapshot itself with §1.3's labels, without
importing `cascade/validator/lane_d_forward`; the audit's `emission_vector`
call must carry lane D's part with lanes B and C off (RV3 note 10); and W9
decides invariant 12 for the shared lane D weight math (RV3 note 11). Also
plan §1.2 "(rev3) Enforcement" ("The
judged set and every skip reason are structured report fields.
`check_lane_d` re-derives the judged set"), §1.3 (both clocks, mirror lag,
the confirmation duel, D20), §1.5 (`rows_sha256`; "`checks.derive_weights`
and `check_lane_d` re-derive the lane D weights from the receipt chain"),
§14.3 rule 1, invariants 2, 4 and 12, and RV3 notes 9-11
(`evidence/rv3.md`).

These tests are the spec. The builder may touch only the paths in `PATHS`:
`cascade/audit/lane_d.py` and `cascade/audit/checks.py`. The W7a, W7c, W8,
W5e, W5c, W1.5 suites and `tests/unit/test_satellite_replay.py` must stay
green.

## What W7a already pins, and what W9 adds

W7a (`tests/accept/w7a/test_w7a_audit_gate.py`, and the RV7 B2 contract in
`test_w7a_audit_provenance.py`, PR #658) pins the schema, the signature, the
stated params, `rows_sha256` against the listed rows, both clocks against
the stated `t_max`, the floors, the max-T verdicts, the gate, the crown's
provenance (`held_from`, the confirmation's first clear, the seed,
`king_before` of a stage-`first` cohort) and the chain-mode keyword
`check_lane_d(..., prior_reports=[...])`. W9 does not repeat those. Three
minimal W9 cases overlap them and pass on main (below).

W9 adds:

- the **judged set** and every selection skip, re-derived from the signed
  entry records and the caps;
- the **registration rules** a signed entry record shows (the block clock
  against `ready_ttl_blocks`, the armed contexts, a resolved coldkey);
- the **mirror lag** and the **king's T**, which main takes on trust;
- the canonical **row order** under `rows_sha256`;
- **chain-mode continuity**: what only the previous signed reports refute;
- **(rev6.1) snapshot eligibility**: `window_refusal` over a `pool_d/1`
  series record, and `check_lane_d(..., pool_dirs=...)` against the judged
  snapshots;
- **`derive_weights` with lane D**, from the signed report and its W7c
  receipt record only, independent of W8's producer functions.

## Fixtures

Every report comes from real `settle_lane_d` runs on W7a's synthetic chain
(`tests/accept/w7a/helpers.py`: numpy only, no torch). The bench is the one
mocked external service (W7a's `HealthyBench`). The signer is W7a's
deterministic test keypair `//w7a-signer`. A doctored report is re-signed
by that key with `rows_sha256` recomputed (`helpers.resign`), so the
defect a test introduces is the only one. Every doctored case first checks
that its honest report PASSes.

| fixture | what it is |
|---|---|
| `crown_chain` | **(rev6.11)** W9's own copy of W7a's single-challenger chain: b1 = `ARM + 900`, X's first verdict against the house H0: clear, gate `pass`, crowned (uid 6); b2 = `ARM + 1800` judges nothing. Then b3 = `ARM + 2700` is settled twice from copies of the b2 state: `s3` judges nothing (X keeps KING_D); `s3d` is the same with X deregistered (its slot burns, `deregistered`) |
| `snapshots` | b1 settled on W7a's snapshot 1 (`honest`) and on three defective copies of it: series `src5-s7` with `frontier` deleted (`frontier_absent`) or `null` (`frontier_null`), and feed `src5` removed from `feeds` (`unlisted_feed`). The producer fails those windows closed and judges the rest, so each report is genuine and PASSes the report-only audit |
| `caps` | one b1 settlement, three entries A (T0), B (T0+60), C (T0+120). `cold`: `max_judged_per_coldkey_per_settlement = 1`, A and B on coldkey K1, C on K2, so A and C are judged and B is skipped `coldkey_cap_settlement`. `glob`: `max_entries_per_settlement = 2`, three coldkeys, so A and B are judged and C is skipped |

`derive_weights` runs on the repo's `chain.toml` parse with lanes B and C
switched off (`from_block = 0`), `[lane_d]` = W7a's armed config and `[manifest]
validator_hotkey` = the fixture signer (W7b's `pinned_report_signer`).
The lane A receipt is `tests/unit/receipt_fixture.py`'s, moved to the
settlement's boundary with 8 uids, reward uid 1 and no DEC-CA-0055, lane B
or lane C record.

## Contract: `check_lane_d` (`cascade/audit/lane_d.py`)

`check_lane_d(report, cfg, *, signer_hotkey, prior_reports=None, ...)`.
Every rule below FAILs the report on a breach. They are single-report rules
except where marked "chain mode".

### Judged set

The report's `entries[]` carries, per entry, the signed inputs `coldkey` (as
recorded at `ready`), `context`, `ready_block`, `reveal_block` and
`reveal_ts`, plus the result `status` and `reason`.

1. `judged` equals the verdicts' digests in order, and the verdicts cover
   exactly the cohort members.
2. Each `skipped[]` digest is a listed entry and appears once. Its `reason`
   equals that entry's `entries[].reason`.
3. **No silent omission.** Every entry whose `status` is `pending` ~~or
   `confirm_pending`~~ (rev6.11: no confirmation stage) after the settlement is in `judged`, in `skipped`, or
   has a gate result.
4. **The caps, FIFO by T.** The *selection candidates* are the judged
   entries plus the entries skipped with `coldkey_cap_settlement`,
   `pending_floor`, `king_unscored` or `checkpoint_fault`. A held gate
   candidate (a gate result with no verdict) is excluded: it is never
   selected and never counts against a cap. The audit walks the candidates
   ordered by `(reveal_ts, digest)` and counts the selected ones, overall
   and per `coldkey`. A candidate is *over* when the overall count has
   reached `max_entries_per_settlement` or its coldkey's count has reached
   `max_judged_per_coldkey_per_settlement`. An over candidate must be
   skipped `coldkey_cap_settlement`. Any other candidate must not be, and it
   counts. (A `pending_floor`, `king_unscored` or malformed-forecast
   `checkpoint_fault` entry was selected and then dropped, so it counts
   against the caps, as in the producer's `_select`.)
5. **Registration rules** for every selection candidate: `coldkey` is not
   null (a null coldkey is `coldkey_unresolved`, closed at registration),
   `context` is in `cfg.contexts` (`context_not_armed`), and
   `reveal_block - ready_block <= ready_ttl_blocks` (`ready_expired`, B10).
   Only the rules the producer's `_register` applies are pinned, so the
   audit never refutes an honest settlement. `ready_block <= reveal_block`
   and `reveal_block <= boundary` are not pinned, because the producer
   doesn't check them.

### Clocks (§1.3, invariant 4)

6. A judged window's `mirror_lag_hours`, when not null, is at most
   `cfg.max_mirror_lag_hours` (equal passes; null is an unmeasured feed,
   which W5a's feed rules govern).
7. **The king's T.** When a cohort's `king_digest` is `king_before.digest`
   or `house.digest`, its `king_reveal_ts` equals that record's
   `reveal_ts`. The forward bound uses max(T_E, T_K), so a lowered T_K
   would admit windows published before the king if the king were the later
   of the two. Under D20 `"original"`, a first-clear king that's neither of
   them comes from the chain (not pinned here).

Both clocks against the stated `t_max` are main's existing check.

### Rows (§1.5)

8. `rows` is in the canonical order: cohorts by `index`, each cohort's
   `window_ids` in order, and per window the models `[king_digest,
   house_digest (when present), *members]`. That's the producer's order,
   checked by `test_honest_rows_are_in_the_pinned_order`. `rows_sha256` is
   over that list, so two auditors holding the same rows publish one
   digest. A permuted list with a recomputed digest FAILs.

### Chain mode (`prior_reports`)

The keyword and the evidence rules come from W7a RV7 B2. A prior report
counts only if its signature verifies under `signer_hotkey` and its
`boundary` is below the report's. W9 adds three rules:

9. **King continuity.** `king_before` equals the `king_after` of the
   latest verified prior report. The one exception is a house refresh: when
   that `king_after` is a house (`house: true`), `king_before` may instead
   equal this report's `house` record (the producer's `refresh_house`).
10. **The house carries over.** A prior report whose `house.digest` equals
    this report's has an identical `house` record (same T).
11. **Recorded inputs never change.** For every digest listed in both this
    report and a prior one, `hotkey`, `coldkey`, `context`, `ready_block`,
    `reveal_block` and `reveal_ts` are identical.

Each chain test's forgery PASSes the single-report audit (asserted, except
the undo-crown case), so only the chain refutes it. ~~The D20 confirmation
rule end to end~~ **(rev6.11 (b): no confirmation duel, no D20.)** Every
verdict is a first verdict against `king_before`, and `king_before` is where
the chain left KING_D (rule 9), so a crown can't rest on a stale king.

## rev6.11 / rev6.12 (`test_w9_rev611.py`, `test_w9_embargo.py`, `test_w9_windows.py`)

Plan rev6.9 (W5i), rev6.10 (a)/(b), rev6.11 (a)/(b), rev6.12 (a)/(b) and
the W9 row. Field names are the ones the W5j / W5i / W7a tests pin.

- **Fixture.** `crown_chain` is now W9's own copy of W7a's single-challenger
  chain (W7a's building blocks, not `two_settlements`, so a W7a fixture
  change never reshapes W9): b1 is X's first verdict, crowned there; b2
  judges nothing. Every W7a snapshot W9 settles on pins `embargo_hours = 48`
  in each listed feed (`helpers.pin_embargo`, W6b). `test_w9_snapshot.py`'s
  `FEEDS` pin `embargo_hours = 0`, so its cases stay the rev3 rule.
- **No confirmation stage.** `check_lane_d` FAILs a report whose cohort or
  verdict `stage` isn't `first`, a verdict with a `confirmation` block, or
  an entry `confirm_pending`, with `confirmation_stage_present` in `detail`.
  Rule 3 ("No silent omission") reads `pending` only.
- **Crowning** is re-derived as first clear ∧ A10 not `fail` ∧ gate `pass`:
  a crown with an A10 `fail`, no gate, or `cleared = false` FAILs. The honest
  crown at a first verdict PASSes.
- **`window_refusal(series, feeds, bound, cfg)`** takes `e = min(feeds[feed]
  .embargo_hours, cfg.embargo_hours)` for every cadence (no `freq` term,
  rev6.12 (a)) and applies the rev6.10 rule at the feed bound `bound − 3600
  e`: first-seen `+ e > bound`, data time `> bound − e`, S0 and backfill at
  `bound − e`, the §1.3 labels unchanged. A listed feed whose
  `embargo_hours` is absent, not a number (bool and NaN included) or
  negative is `unlisted_feed`. `e = 0` is the rev3 rule.
- **`pool_dirs`:** a snapshot whose feed `e` is above `[lane_d]
  embargo_hours`, or that carries a non-zero `e` on a publish-ahead feed (a
  series point whose data time is after its first-seen time), FAILs; the
  honest snapshot with `e = 48` PASSes.
- **Report-only re-derivations.** `params` echoes `embargo_hours`,
  `min_fresh_window_share` and `max_windows_per_settlement` (= `cfg`), and no
  `embargo_window_share_cap`. Each judged window carries `embargo_hours`
  (≤ the config), `embargo_window`, `freq`, `kind`, `timeframe` (= `freq`),
  `bin`, `h`, `origin` (= `min_target_ts`); its clocks satisfy the rule at
  its `e` against `b = t_max + margin`; `embargo_window` is true iff the
  window fails the e = 0 rule. F / (F + E) ≥ `min_fresh_window_share` on
  every judged set, else FAIL with `fresh_share_short` in `detail`. The draw
  record (top-level `draw` or `cohorts[i].draw`) has `boundary_block_hash`
  (the boundary's), `seed = f"{boundary_block_hash}:lane_d_windows"`,
  `max_windows_per_settlement`, `quota` / `capacity` (`{"<timeframe>|<bin>":
  int}`) and `embargo_windows`; a changed seed, block hash or quota FAILs,
  and so does a drawn window dropped from the set. Under
  `max_windows_per_settlement = 32` the b1 draw holds 24…32 windows, no
  source above `max_source_share`, the same picks in two independent runs,
  and PASSes.
- **The clamp.** `cascade.audit.lane_d.combined_log_ratio(r_fwd, r_bt, *,
  n_fwd, n_bt, cap, floor)` re-implements rev6.9's rule: `c = min(cap, n_bt
  / (n_bt + n_fwd))` (0 without backtest windows), `log R = (1 − c) · log
  r_fwd + c · log clamp(r_bt, floor, 1 / floor)`; exactly `log r_fwd` when
  off; memorisation gains at most `1 − floor^cap`.
- **Not pinned:** e = 0 on sub-hourly series (reversed by rev6.12 (a)); how
  the audit treats a listed feed without `embargo_hours` in a snapshot W7a's
  own fixture writes (W7a / W5a own that fixture).

On main (`315d240`) 46 of the 137 tests fail, each on an `AssertionError` or
on `ImportError: cannot import name 'combined_log_ratio' from
'cascade.audit.lane_d'` (in PATHS); the other 91 pass.

### (rev6.1) Snapshots

```python
window_refusal(series: dict, feeds: dict, bound: float, cfg) -> str | None   # cascade.audit.lane_d
```

`series` is one `pool_d/1` metadata series record (`series_id`, `source`,
optional `feed`, `target_ts`, `first_seen_ts`, `end_ts`, `frontier`).
`feeds` is the snapshot's top-level `feeds` block, and `bound` is
`t_max + 3600 * forward_margin_hours`. It returns `None` when the window is
eligible. Otherwise it returns the first failing rule's label, from W7a's
`WINDOW_REFUSAL_REASONS`, in this order:

1. `missing_clock`: `target_ts` and `first_seen_ts` don't both hold
   `cfg.horizon` points, or `end_ts` is missing or isn't `target_ts[-1]`.
2. `unlisted_feed`: the feed (the record's `feed`, else its `source`) has no
   entry in `feeds`. An empty or absent block counts.
3. `forecast_product`.
4. `over_lag`: the feed's `mirror_lag_hours` is not null and is greater than
   `cfg.max_mirror_lag_hours`. A null (unmeasured) lag, or one equal to the
   limit, is not a refusal.
5. The clocks, each strictly after the bound for every point:
   `before_margin` when both fail; `publish_ahead` when only first-seen
   fails; `late_mirror` when only data time fails. These are the §1.3 labels
   (rev6.1).
6. `absent_from_s0`: no `frontier` record has `as_of_ts > bound`. An absent,
   `null` or empty `frontier` counts (fail closed).
7. `backfill`: `min(target_ts) <= end_ts` of S0, the EARLIEST frontier record
   with `as_of_ts > bound`.

It is a re-implementation from this spec (W5a's README and §1.3).
Invariant 12 bans importing `lane_d_forward`.

`check_lane_d(..., pool_dirs=[dir, ...])` adds these rules. FAIL on any
breach:

12. A snapshot's digest is the sha256 of the concatenated sha256 hex of its
    `metadata.json`, `hist.npz` and `tgt.npz`. Every `snapshot_sha256` that
    a judged window names must match a provided directory. Extra
    directories are ignored.
13. In every judged snapshot (one a judged window was scored from), every
    series has a `frontier` list (absent or `null` FAILs), and every series'
    feed is listed in `feeds`.
14. Every judged window's series record exists in its snapshot (window id
    `<series_id>:<target_ts[0]>:h<horizon>`). Its report record equals
    what the snapshot gives: `source`, `feed` (`feed`, else `source`),
    `min_target_ts`, `min_first_seen_ts` and `mirror_lag_hours` (the feed
    entry's). Its `window_refusal` at its cohort's bound is `None`.

Satellites hold no pool, so `pool_dirs` is for an auditor holding the
snapshots: the main validator's own audit, or anyone once a snapshot is
revealed (D23). Without `pool_dirs`, rules 12-14 don't run.

## Contract: `derive_weights` (`cascade/audit/checks.py`)

```python
LANE_D_WEIGHT_ERRORS: tuple[str, ...]   # module level; at least the five codes below

derive_weights(receipt, cfg, *, lane_d=None) -> tuple[list[float] | None, str]
```

`lane_d` is `None`, or a mapping:

- `"report"`: the settlement's signed `lane-d-report/1`;
- `"receipt"`: its W7c receipt record (`LaneDSettlement.receipt`);
- optionally `"prior_reports"`: the earlier signed reports, for chain mode.

`cfg` is the `ChainConfig` the receipt was judged under. Lane D *pays* at the
settlement iff `cfg.lane_d.active(receipt.epoch_start_block)`.

### Refusals

A refusal returns `(None, why)`, where `why.split(":", 1)[0].strip()` is the
code. The checks run in this order:

| code | when |
|---|---|
| `lane_d_not_armed` | `lane_d` is given but lane D doesn't pay at the boundary |
| `lane_d_record_missing` | lane D pays but `lane_d` is `None`. Never a silent lane A-only vector (RV3 note 10) |
| `lane_d_boundary_mismatch` | the report's or the record's `boundary` isn't `receipt.epoch_start_block` |
| `lane_d_report_mismatch` | the record's `report_sha256` isn't the sha256 of the report's canonical body (every field but `signature`, W7c) |
| `lane_d_report_unverified` | `check_lane_d(report, cfg.lane_d, signer_hotkey=cfg.manifest.validator_hotkey, prior_reports=...)` isn't `PASS` (a bad signature, a foreign signer, a FAIL on any rule, a FAIL in chain mode) |

### Behaviour

- **Unarmed with no `lane_d`.** The result is exactly today's (invariant 1).
- **Paying.** Lane A's vector is rebuilt as today, then scaled to lane A's
  pool `1 - b - c - d`, where `d = min(share, max(0, 1 - b - c))` for
  `takes_from = "lane_a"`. Lane D's part is added from the signed
  `pay.court` alone:
  - slot `i` is worth `cfg.lane_d.court_weights[i]` (rev6.23: 0.80 /
    0.15 / 0.05), normalised over the slots, times `pay.share`;
  - a slot whose `uid` is null, out of range or repeated burns to
    `cfg.scoring.burn_uid`;
  - an empty court burns the whole share.
- **A DEC-CA-0055 policy.** When the receipt carries an `emission` record,
  its policy re-shapes lane A's part only. Lane D's part goes to
  `emission_vector`'s second argument even with lanes B and C off. That's
  the RV3 note 10 trap at `_lane_vector_want` (`audit/checks.py:2834-2841`),
  pinned by `test_lane_d_survives_a_dec55_policy_with_b_and_c_off`.
- **Agreement.** On the honest chain (`s1` house slot burns, `s2` X paid,
  `s3` X keeps the slot, `s3d` X deregistered so its slot burns, and `s3`
  in chain mode) the vector equals W8's emission
  (`lane_parts(..., lane_d_cfg=cfg.lane_d, d_court_uids=pay["court_uids"])`
  then `emission_vector`), uid for uid at `rel = 1e-9`.
- **The wiring trap.** A receipt whose recorded vector is the lane A-only
  one still gets the true vector back, so a satellite comparing the two
  disagrees.
- **What isn't checked here.** `derive_weights` doesn't compare against
  `receipt.weights`; `check_weights` and the satellite do that.

## Invariant 12

Plan §3.2: "Audit and satellite-replay code never import the producer's
eligibility, verdict or weight functions." RV3 note 11 asked for a decision
before these tests. The tests pin this one:

- the audit already shares the lane A / B / C builders (`lane_parts`,
  `emission_vector`, `decayed_share_vector`) and keeps sharing them;
- **lane D's carve and court split are re-derived in the audit**, without
  W8's `lane_d_vector`, without `lane_parts(..., lane_d_cfg=...)` and
  without `lane_pool_shares(...)` asked for lane D.

Views:

| view | rule | on main |
|---|---|---|
| direct imports | `cascade/audit/lane_d.py` imports only the stdlib, numpy, `cascade.eval.bootstrap`, `cascade.shared.sr25519`, `cascade.shared.config.lane_d` and `cascade.audit.checks` | passes |
| static closure | the closure of `cascade.audit.lane_d` + `cascade.audit.checks` (function-level imports included) reaches no `cascade.validator.lane_d*` (`lane_d_forward` included, rev6.1), `cascade.eval.lane_d_*`, `cascade.lane_d*`, `cascade.laned` or `cascade.satellite` module. `cascade.audit.checks` already reaches `cascade.trainer` and parts of `cascade.validator` for lanes A and C, so those aren't banned | passes |
| AST | no `lane_d_vector` reference, no call with a `lane_d_cfg` / `d_court_uids` keyword, no `lane_pool_shares` call with 3+ positional arguments in either file | passes |
| runtime patch | with `lane_d_vector` raising, and `lane_parts` / `lane_pool_shares` raising when asked for lane D, the derivation is unchanged | fails (`lane_d`) |
| runtime modules | a fresh interpreter that audits b1-b3 in chain mode and derives their weights loads none of the modules above and no torch | fails (exit 1) |

The satellite (`cascade.satellite`, W7d) is banned too. The two independent
re-derivations cross-check each other only if neither leans on the other.

**Determinism.** Two fresh interpreters (`PYTHONHASHSEED` 0 and 4242) give
identical audit results (status and detail) and identical vectors.

## Not pinned

- How a lane A `RoundReceipt` carries the lane D record, and how the live
  satellite gets it. W14 and the live wiring own that, and `RoundReceipt`
  isn't in PATHS. W9 pins the function both of them call.
- `check_weights` with lane D (it needs the same record, so it follows the
  same wiring).
- Lane D with lane B or C live, a DEC-CA-0055 policy, or `takes_from`
  other than `lane_a`. W8's vector tests pin that arithmetic. RV3 note 9
  applies: compute lane A's pool from the shares as W8 does.
- Court slots beyond KING_D: the honest chains have at most one paid slot.
  The decay formula above is the contract.
- `max_open_per_coldkey` (advisory, at intake), `max_reveal_delay_days`
  (needs the boundary's wall time, which the report doesn't carry),
  `duplicate_digest` and `hotkey_has_open_entry` (registration-time facts a
  later report can't show).
- The wording of any `detail` or `why` after the code.

## Why each test fails on main

| test | on main |
|---|---|
| every doctored case in `test_w9_judged_set.py` | assert: main never reads `judged`, `skipped` or `coldkey`, so the forgery PASSes |
| `test_a_window_over_the_mirror_lag_limit_fails`, `test_the_kings_t_is_the_named_kings_reveal` | assert: PASS (mirror lag and T_K taken on trust) |
| `test_permuted_rows_with_a_recomputed_digest_fail` | assert: PASS (rows looked up by key) |
| every test in `test_w9_chain.py` | assert: `check_lane_d` takes no `prior_reports` (after W7a RV7 lands: the forgery PASSes) |
| `test_weight_error_codes_are_exported` | `ImportError`: cannot import name `LANE_D_WEIGHT_ERRORS` from `cascade.audit.checks` (in PATHS) |
| every other `test_w9_derive_weights.py` test, the runtime patch test | assert: `derive_weights` takes no `lane_d` |
| the runtime-modules and determinism tests | assert: the subprocess exits 1 (`TypeError` on `prior_reports`) |
| `window_refusal` tests in `test_w9_snapshot.py` | `ImportError`: cannot import name `window_refusal` from `cascade.audit.lane_d` (in PATHS) |
| the `pool_dirs` tests in `test_w9_snapshot.py` | assert: `check_lane_d` takes no `pool_dirs` |

These pass on main and pin behaviour W9 must keep:

- the honest reports;
- the two both-clock `*_at_the_bound_fails` cases;
- the mirror lag exactly at the limit;
- the honest row order and the tampered `rows_sha256`;
- `test_inv1_derive_weights_unchanged_when_unarmed`, paired with
  `test_inv1_derive_weights_armed_nonempty_lane_d_effect` (invariant 10);
- the three static invariant-12 views.

On main (1e3d181): 77 failed, 12 passed. A scratch prototype of this
contract (not committed) passed all 89 tests, plus the w7a, w5e, w5c, w7c,
w8, w5a, w5b, w1_5 suites and `tests/unit/test_satellite_replay.py`. w3b5
fails the same 17 tests with and without the prototype (#651 isn't on
main). That shows the contract can be built and that every honest fixture
report PASSes.

Run locally with `-m "not torch"`. Nothing here imports torch.

## Multichannel windows (rev6.18 F2): `test_w9_multivariate.py`

- `check_lane_d(..., max_channels=)`: `[generator] max_channels` at the
  boundary (`derive_weights` passes it, `checks._lane_d_max_channels`).
  Every judged window carrying the channel fields must carry all of them and
  its `channels` must equal `channel_pick(draw.boundary_block_hash,
  series_id, range(n_channels_raw) minus empty_channels, max_channels)` (the
  rule re-stated in `cascade.audit.lane_d.channel_pick`, equal to the
  producer's `cascade.eval.lane_d_channels.pick_channels` and the spec),
  with `n_channels` and `channels_sha256` to match; an empty pick fails;
  `max_channels=None` fails any multichannel window.
- With `pool_dirs`: the scored snapshot's `hist.npz` series has
  `n_channels_raw` channels and the window's `series_id`, every scored
  channel holds a finite value, and a C >= 2 series is never recorded
  univariate.

On main these tests fail on the `settle_mv` / `max_channels` assertions or
`cannot import name` (`channel_pick` from `cascade.audit.lane_d`,
`_lane_d_max_channels` from `cascade.audit.checks`, both in PATHS).

## rev6.22: lane B weights and slot rows (`test_w9_lane_b_slots.py`)

One pay rule: every paid holder has multiplier 1.0 at any age (no TTL), the
re-derived `age` and `since_era`, a finite `gain_lcb`, and equals its holder
in `state_post`; tampering with any of them FAILs. A judged leg has one row
per slot; `check_lane_b` re-derives each row for its family (`declared` at
`gain_margin`, `undeclared` at `undeclared_gain_margin`) and binds one
critical value per family. A wrong bar, family, declared flag, verdict, a
second critical value in one family, or a missing slot FAILs. Rows come from
the validator's `judgements_record` over a real `judge_expert_slots` verdict.

## rev6.14 (a): judging start vs eligibility boundary (owner 2026-11-23)

The audit re-derives the judging start (`frozen_before_eval_delay` FAILs, a judged cohort without `boundary_ts` FAILs) and judges eligibility at `b = T + forward_margin_hours` (default 0). `test_w9_rev614.py`.
