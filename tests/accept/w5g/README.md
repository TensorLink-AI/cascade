# W5g acceptance tests: bounded bench holds, `checkpoint_fault`, exact typed reasons

W5g (plan rev6 §3.0 W5g row: "Holds, `checkpoint_fault`, typed reason enums
| W5e | ci | `scripts/laned/accept W5g` (asserts the exact reason enum per
case)") pins three things in `settle_lane_d`: a bench hold is bounded by
`[lane_d] max_bench_holds`, a `checkpoint_fault` closes the entry for good,
and every reason the settlement emits is a typed enum value that a case here
asserts exactly. These tests are the spec. The builder may touch only the
paths in `PATHS`. The W5e, W7a, W7c and W5b/W5c tests must stay green.

Every case is a real `settle_lane_d` call over real `pool_d/1` snapshots
(fixtures copied from W5e). The bench pod is the one external service:
`helpers.BenchHost` returns recorded per-item rows behind a real
`lane_d_bench.SweepSession`, and `helpers.DownHost` raises on every sweep.

## The plan text this pins

- §10.4: "Bench pod down before the candidate's sweep | Crowning held; CRIT
  after 1; after `max_bench_holds` (4) the candidate re-queues unjudged.
  Deliberately not lane A's never-timeout hold".
- §10.4: "Candidate sweep error, partial, NaN, missing metric |
  `checkpoint_fault`: entry refused; evidence kept". §1.3b: that is "a
  **`checkpoint_fault`, the miner's fault** (T17)". T17: "`checkpoint_fault`;
  KING_D not held".
- §3.2 invariant 11: "`pending`, `hold` and `burn` always carry a typed
  reason enum, and tests assert the exact reason. (W5g, W7a)".
- RV7 (`evidence/rv7.md` on `laned/rv7`): "bench holds never expire and
  aren't capped". W5e README: "Counting holds against `max_bench_holds` is
  W5g's".

## Contract

### The config key

`[lane_d] max_bench_holds` (W2 already parses it into
`LaneDConfig.max_bench_holds`, integer, `>= 0`, default **4**). No new key.

### Bounded holds

A **hold** is a `bench_hold` gate outcome for a crowning candidate: a
confirmation that cleared (stage `confirm`) or a held clear re-gated at a
later settlement. Every kind counts the same: no `bench` session, no
`boundary_block_hash` (W5e fail closed), a sweep that raises, a reference
(KING_D or house) fault.

With `max_bench_holds = N`:

- **Holds 1..N** keep the clear, exactly as W5e: the entry is `bench_hold`
  with status `hold` or `confirm_pending`, nothing is crowned, the king is
  unchanged. A held clear is re-gated at the next settlement without a new
  duel. If the bench is healthy there and the gate passes, it crowns (no rows
  for it in that report): **a hold within the bound never consumes the
  entry's chance.**
- **Hold N + 1 re-queues the candidate unjudged** (§10.4). In that
  settlement:
  - state and report entry: status **`pending`**, reason **`bench_hold`**
    (the typed reason it is pending again; no new enum value, so the W7a
    enum pins stay exact);
  - the held clear and its first clear are dropped: state `confirm` is
    `null` and there is no (or an empty) `gate_hold`;
  - not charged: not `lost`, not `closed`, no court slot, `crowned` is
    `null`, `king_after == king_before`, and if a `gate` record is written
    none of its results is `pass`;
  - on a re-gate (N >= 1) the entry is considered and not judged: exactly one
    `skipped` item `{digest, "bench_hold"}`, no verdict, no rows. With
    `N = 0` the first hold is already the exhausting one; that candidate was
    judged in the settlement, so it is not a skip.
- **The next settlement with fresh windows judges the re-queued entry
  afresh:** it is selected again and duelled at stage `first` (rows and a
  verdict), going `confirm_pending` / `confirm_pending` when it clears. It is
  never re-gated on the dropped clear.
- **The count belongs to one held clear and is consecutive.** A re-queued
  entry that clears and confirms again starts a new count (its next hold is
  hold 1 and keeps the clear; a later pass crowns).
- **The count is state.** It survives a restart: a copy of the state
  directory after hold 1 re-queues at the same boundary with an identical
  report body (`lane_d_state.report_digest`) and identical state entries.
- **Deterministic:** the exhausting settlement's report body, `entries`,
  `skipped`, `judged`, `verdicts`, `rows_sha256` and state entries do not
  depend on the order of the `entries` argument.

### `checkpoint_fault`

In every case the entry is `closed` / `checkpoint_fault` in state and
report, keeps no `gate_hold`, is never crowned, and KING_D is unchanged:

| way in | when | in `skipped`? |
|---|---|---|
| the candidate's rows are NaN on the drawn suite at its confirmation (T17) | the confirming settlement | no: it was judged |
| the same on the re-gate of a held clear | the re-gate | yes, `checkpoint_fault` |
| malformed eval forecasts at selection | the selecting settlement | yes, `checkpoint_fault` |

**No further gating.** At every later settlement (fresh windows, a healthy
bench) the entry is not selected, judged, skipped or swept on the bench, and
its state record is byte-for-byte the same.

### The exact reason per case

`test_the_exact_reason_per_case` has one case per row. Each asserts the
state's `(status, reason)` by exact value, the report entry equal to it,
`skipped` holding exactly `[reason]` for a skip (and `reason ∈
SKIP_REASONS`, the digest not in `judged`) or nothing for a judged entry
(digest in `judged`), and `validate_report` passing.

| case | reason | status | skip |
|---|---|---|---|
| revealed after every window | `pending_floor` | `pending` | yes |
| over `max_judged_per_coldkey_per_settlement` | `coldkey_cap_settlement` | `pending` | yes |
| empty `coldkey_at_ready` | `coldkey_unresolved` | `closed` | yes |
| no house forecasts | `king_unscored` | `pending` | yes |
| hotkey has no UID | `deregistered` | `closed` | yes |
| revealed past `ready_ttl_blocks` | `ready_expired` | `closed` | yes |
| digest equals the house's | `duplicate_digest` | `closed` | yes |
| second entry of an open hotkey | `hotkey_has_open_entry` | `closed` | yes |
| context not in `contexts` | `context_not_armed` | `closed` | yes |
| **(plan rev6.15)** armed record given, no armed variant ≤ the declared context (declared 4096, armed 8192/16384) | `context_below_armed` | `pending` | yes |
| NaN forecasts at selection | `checkpoint_fault` | `closed` | yes |
| NaN on the drawn suite at confirmation | `checkpoint_fault` | `closed` | no |
| NaN on the drawn suite at a re-gate | `checkpoint_fault` | `closed` | yes |
| settlement at `until_block` | `lane_closed` | `closed` | yes |
| first clear | `confirm_pending` | `confirm_pending` | no |
| loses the first duel | `lost_duel` | `lost` | no |
| loses the confirmation | `lost_confirmation` | `lost` | no |
| bench down at confirmation | `bench_hold` | `hold` or `confirm_pending` | no |
| reference fault at a re-gate | `bench_hold` | `hold` or `confirm_pending` | yes |
| hold N + 1 (re-queued) | `bench_hold` | `pending` | yes |
| gate regression | `gate_regression` | `lost` | no |
| gate pass | `crowned` | `crowned` | no |
| clears the confirmation duel, fails the A10 low-frequency check (W3b5's T4b fixture) | `lowfreq_regression` | `lost` | no |

Also, for every case: every entry in the state (not only the one under test)
has a reason in `lane_d_state.ENTRY_REASONS` and a status in
`ENTRY_STATUSES`, the report's entry equals it, every `skipped` reason is in
`SKIP_REASONS` and equals the state's reason for that digest, and every court
`burn_reason` is `null` or in `BURN_REASONS`. No free text anywhere.

**Coverage is enforced.** `test_every_reason_the_settlement_can_emit_has_a_case`
collects the `ENTRY_REASONS` values that appear as string literals (not
docstrings) in `cascade/validator/lane_d.py`, plus the gate outcomes that are
entry reasons, and fails on any with no case. A new emitted reason needs a
case here. `test_every_enum_reason_is_cased_or_owned_elsewhere` accounts for
every enum value. The three nothing emits yet belong to other WPs:

| reason | owner |
|---|---|
| `coldkey_changed` | W5h (A8, T6b) |
| `reveal_delay_exceeded` | W5h (A9, T18b) |
| `mirror_lag_exceeded` | W5a (A1) |

`lowfreq_regression` is emitted by W3b5 (A10, T4b; `cascade/validator/lane_d.py`)
and pinned here by the case above; W3b5's own tests (tests/accept/w3b5) own
the check's semantics.

## Not pinned

- Where the count lives in the state record (`gate_hold` is main's field;
  any field name works if it is persisted and restart-safe), and whether the
  exhausting settlement writes a `gate` record.
- Whether a re-queued entry's `used_windows` are kept (the fixture's fresh
  snapshot never overlaps them).
- The `lane_d_bench_hold` CRIT alert and its follower pattern (W11).
- `status/lane_d.json` (W17 is not on main, so the reasons are pinned in
  state and report only).

## On main

- 9 tests fail on an AssertionError, because a hold never ends (the entry
  stays `confirm_pending` / `bench_hold`): the 8 tests in `test_w5g_holds.py`
  other than `test_a_hold_within_the_bound_keeps_the_chance` (the bound test
  counts once per id: `max0`, `max2`, `default4`) and the
  `bench_hold_requeued` exact-reason case.
- Everything else passes: main already emits the other reasons exactly, and
  `checkpoint_fault` already closes for good. W5g owns them so they stay
  pinned.

## Correction (plan rev6.10 (a) / rev6.12 (a)): fixture feeds pin `embargo_hours`

W5a (tests in #758) makes forward eligibility embargo-aware and fails a
listed feed whose `embargo_hours` is missing closed as `unlisted_feed`. This
WP's pool fixture wrote feeds without the key, so once the W5a
implementation lands every fixture window would be refused. Every fixture
feed now pins `embargo_hours: 0.0` (e = 0, the rev3 rule byte for byte, so
every verdict is unchanged); on main the key is ignored. No assertion
changed; every test passes on main and with the W5a implementation.

## Correction (plan rev6.11 (b)): crown at the first verdict

The confirmation duel is dropped (owner 2026-11-23): a crown is decided at
the first verdict, first clear (max-T LCB vs KING_D) ∧ A10 ∧ the §1.3b gate.
There is no `confirm` stage, no `confirm_pending` status or reason and no
`lost_confirmation`; a held first clear is `hold` / `bench_hold`. This WP's
crowning chains move to that rule with nothing weakened. The Flow lists X from boundary 2, where its first clear is gated with `mode2`. A held clear is `hold` / `bench_hold` (the only held status). A re-queued entry's fresh first verdict is decided at once (crowned on a passing gate, else held with a new count). The `confirm_pending` and `lost_confirmation` cases are gone with their reasons, and `test_no_confirm_stage_reason_or_status_exists` pins that. On main the
old code still runs the confirmation stage, so the changed tests fail on an
assertion until the W5e implementation lands.
