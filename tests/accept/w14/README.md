# W14 acceptance tests: the lane D-only fast kill

W14 (plan rev6 §3.0 W14 row, Proposed design change 4, §5 "Kill switch and
latency", §10.8 "Lane D only") depends on W7c. These tests are the spec. The
builder may touch only the paths in `PATHS`. The W7c, W8 and W1.5 tests must
stay green.

## What the plan says

The W14 row, verbatim:

> ~~**(rev5) blocked: needs owner (Proposed design change 4).**~~ **(rev6)
> Unblocked:** change 4 was accepted (owner 2026-11-21). Lane D-only fast
> kill: a `lane_d: {status: "absent"}` receipt state with a lane D-only burn.
> Today a missing lane D report makes satellites re-assert the old vector for
> one era (`stale_after_blocks = 3600`) and then burn the whole vector
> (`satellite.py:30-45`)

Proposed design change 4, verbatim:

> **"Stop publishing ⇒ only lane D burns" is false.** **(rev5)** A missing
> report is a check that can't run: the satellite re-asserts the last
> verified vector for `stale_after_blocks` (3600), then burns the **whole**
> vector (`cascade/validator/satellite.py:30-45`). Proposed: a structured
> `lane_d: {status: "absent"}` receipt state with a lane-D-only burn, reusing
> MarketCast's `REPORT_ABSENT` / `INFRA_DEFERRED` (verified:
> `marketcast_report.py:32-34`). Affects §5 kill switch, §10.8, W14, T27.

So the trigger is **a settlement with no lane D report**: the operator stops
lane D reports (the kill), or the settlement can't run (an infra deferral).
Instead of publishing nothing (which satellites read as "check can't run":
hold the old vector, then burn everything), the main validator publishes its
normal signed lane A receipt carrying a typed `lane_d` record with
`status: "absent"`, and lane D's share, only lane D's, burns at that
settlement.

Every chain is W7c's real `settle_lane_d` fixture (`tests/accept/w7c/helpers.py`,
numpy only, no torch): boundary(1) is X's first clear, boundary(2) confirms
and crowns X, boundary(3) judges nothing. `helpers.settle_b4` adds a fourth
boundary on snapshot 3. The bench pod is the one mocked external service.

## Contract: `cascade/validator/lane_d_state.py`, the absent record

Exports:

- `ABSENT_STATUS == cascade.shared.marketcast_report.REPORT_ABSENT == "absent"`.
- `ABSENT_REASONS`: a tuple of distinct lower-case strings containing at
  least `"report_absent"` (no lane D report was produced or published: the
  stop-publishing kill) and `"infra_deferred"` (`== marketcast_report.
  INFRA_DEFERRED`: the settlement couldn't run). The burn carries a typed
  reason (invariant 11).
- `make_absent_receipt(*, boundary, state_sha256, reason) -> dict`.
- `RECEIPT_ERROR_CODES` gains `absent_state_change` and `absent_with_report`
  (W7c's codes are kept).

The record is exactly:

```json
{"schema": "<RECEIPT_SCHEMA>", "status": "absent", "boundary": <int>,
 "reason": "<one of ABSENT_REASONS>",
 "prior_state_sha256": "<64 hex>", "posterior_state_sha256": "<64 hex>"}
```

- `schema` is W7c's `RECEIPT_SCHEMA`. There's no `report_sha256`.
- `prior_state_sha256 == posterior_state_sha256 == state_sha256`: an absent
  settlement moves no lane D state.
- W7c's present record is unchanged (exactly its five keys, no `status`).
  A record is **absent-kind iff it has a `status` key**. So a present record
  missing `report_sha256` is still W7c's `missing_field`.

Structure (by `make_absent_receipt`, `verify_receipt` and
`verify_receipt_chain`), all `LaneDReceiptError` with `.code`:

| code | when |
|---|---|
| `missing_field` | a key of the absent record is absent (`status` included: the rest is then read as a present record) |
| `unknown_field` | a key outside the record (`report_sha256` included) |
| `bad_value` | `status` isn't `"absent"`; `reason` isn't in `ABSENT_REASONS`; `boundary` isn't a non-negative `int` (`bool`, `float`, `str` refused); a digest isn't 64 lower-case hex |
| `absent_state_change` | `prior_state_sha256 != posterior_state_sha256` |

`verify_receipt(absent, *, report, state=None)`: structure first; then
`report` must be `None`, else `absent_with_report` (any report for the
boundary contradicts the record); then, when `state` is given, its
`state_digest` must equal the posterior, else `state_mismatch`.

`verify_receipt_chain` accepts present and absent records mixed, with W7c's
rules unchanged: strictly increasing boundaries (`boundary_order`) and every
prior equal to the previous posterior (`chain_break`).

## Contract: `cascade/validator/lane_d.py`, the trigger

```python
absent_settlement(*, cfg, boundary, state_path, house, reason,
                  expect_prior_sha256=None) -> LaneDSettlement
```

- **Unarmed** (`not cfg.active(boundary)`): `LaneDSettlement(None, None,
  None)` and no file is written (invariant 1).
- **Armed:** `report is None`; `receipt` is the absent record; `pay` has
  `share == cfg.share` and `court_uids == []` (other `pay` keys aren't
  pinned). The caller pays it with `lane_parts(..., d_absent=True)`.
- The prior state is loaded with W7c's fail-closed rules
  (`load_prior_state`): a lost state after a persisted settlement is
  `state_missing`, never genesis; `expect_prior_sha256` that doesn't match is
  `state_digest_mismatch`. From a fresh path with no reports the prior is
  genesis (`state_digest(genesis(house))`).
- A `reason` outside `ABSENT_REASONS` raises `ValueError`
  (`LaneDStateError` or `LaneDReceiptError` both qualify).
- **It never replaces a judgement.** A boundary the state already settled
  (state boundary >= boundary) is refused, `stale_boundary` or
  `report_conflict`. A boundary whose signed report is persisted
  (`report_path(state_path, boundary)` exists: a crash after the report
  landed, which W7c recovery must adopt) is `report_conflict`.
- **It writes nothing**, on success and on every refusal: the state
  directory stays byte-identical, no report file appears, the state's
  `boundary` doesn't move. A re-call at the same boundary returns the same
  settlement.

## Contract: `cascade/shared/emission.py`, the lane D-only burn

`lane_parts(..., lane_d_cfg=None, d_court_uids=(), d_absent=False)`:

- `d_absent=True` with a `lane_d_cfg`: lane D's share is carved exactly as
  when it pays, and the **whole share goes to `burn_uid`**, whatever
  `d_court_uids` holds (a full court, a `None` slot, empty). Lane A's element
  is byte-identical (float64) to the paying call; every non-burn uid's value
  in the second element other than lane D's court is byte-identical; the
  court gets `0.0`; `burn_uid` gains `cfg.share` (`rel = 1e-12`).
- Through `emission_vector` and any `EmissionPolicy` (ladder, cap, slash),
  lane A, B and C uids are byte-identical between paying and absent.
- `lane_d_cfg=None` (unarmed): `d_absent` changes nothing. `d_absent=False`
  is today's result.

## Contract: the round receipt

- `cascade/shared/receipt.py`: `RoundReceipt.lane_d: dict | None = None`,
  drop-when-default under key `"lane_d"` in `canonical_body()`;
  `build_receipt(..., lane_d=None)`; `load_receipt(dump_receipt(r)).lane_d`
  round-trips a present or absent record, the bodies are equal and
  `receipt_has_unknown_fields` is `False`. The record is inside the signed
  body, so an edit after signing fails `verify_receipt_signature`.
- Unarmed receipts keep today's bytes: the `canonical_body` sha256 of
  `tests/unit/receipt_fixture.py`'s byte-stable scored and rejected receipts
  is pinned as recorded on main.
- `cascade/validator/receipts.py`: `build_round_receipt` takes `lane_d`
  (passed to `build_receipt`).

## Contract: `cascade/audit/lane_d.py`, audit and satellites

```python
check_lane_d_absent(record, cfg, *, boundary, previous_posterior_sha256,
                    report=None) -> CheckResult
```

`PASS` exactly when the record is a well-formed absent record (the structure
table above), `cfg.active(boundary)`, `record["boundary"] == boundary`, its
prior equals `previous_posterior_sha256` (the previous lane D receipt's
posterior, or the genesis digest before the first settlement), and no
`report` is given for that boundary. Otherwise `FAIL`. It is pure and
deterministic. Satellites run it through the audit (DEC-CA-0058 step d) and
burn lane D's share on a `PASS`.

Invariant 12: `cascade/audit/lane_d.py` imports no `cascade.validator`
module (checked statically); the absent rules are re-implemented there.

## Why an absent record can't suppress a real crown

The threat: an `absent` at a boundary where X would crown, or after X
crowned, so that X is never paid. The plan's mechanisms, each pinned:

1. **Absent moves no state.** The record has one state digest
   (`absent_state_change` otherwise) and `absent_settlement` writes nothing.
   A pending crown stays pending and is judged at the next real settlement
   (`test_absent_defers_a_pending_crown_never_erases_it`: absent at
   boundary(2), X crowns at boundary(3)). A crowned KING_D stays in the state
   and is paid at the next real settlement
   (`test_absent_after_a_crown_keeps_the_king`). Absent defers pay; it never
   rewrites the court.
2. **The chain binds it.** An absent record binding a state from before the
   crown breaks `verify_receipt_chain` against the crowning receipt before it
   and against the next real receipt after it (`chain_break`), and the audit
   fails it against the previous posterior.
3. **It can't relabel a judgement.** A settled boundary or a persisted
   report can't be marked absent, and a published report for the boundary
   makes the audit fail the record.
4. **It can't restart lane D.** Lost or rolled-back state fails closed
   (W7c), so an absent can't reset KING_D to the house.
5. **It pays nobody.** The share burns, never moves to lane A, B or C, so no
   one gains by faking it. Only the main validator's signed receipt carries
   it (DEC-CA-0058 `main-signature`).

## Invariants and determinism

| | test |
|---|---|
| inv. 1 | `test_inv1_absent_settlement_unchanged_when_unarmed`, `test_inv1_weights_unchanged_when_unarmed`, `test_inv1_receipt_unchanged_when_unarmed`, each paired with its `..._armed_nonempty_lane_d_effect` |
| inv. 11 | the typed `reason` (`ABSENT_REASONS`) |
| inv. 12 | `test_inv12_the_audit_never_imports_producer_code` |
| determinism | two runs on the same pool bytes give equal records; a child process under `PYTHONHASHSEED` 0 and 4242 gives the same record |

## Not pinned

- Live validator-loop wiring: when the loop calls `absent_settlement`
  (lane D isn't driven by the loop on main yet), the operator switch that
  stops reports, and the satellite's own replay (W7d, open as #661/#662) and
  `derive_weights` (W9, open as #666) handling the record. They build on
  this contract.
- The RV3 note 10 wiring trap in `validator/weights.py:517-524` (W7c/W9).
- T27 (testnet): measuring the kill latency; the row's `testnet` half.
- The `CheckResult.name` and detail text; other `pay` keys of an absent
  settlement; what a present settlement at a boundary already recorded
  absent does.

## Why each test fails on main

| test | failure on main |
|---|---|
| `test_w14_record.py` (all but one) | `ImportError`/assert: `lane_d_state.py` (in PATHS) has no `make_absent_receipt`, `ABSENT_STATUS`, `ABSENT_REASONS`, and `RECEIPT_ERROR_CODES` lacks the new codes |
| `test_w14_settlement.py` | `ImportError`: no `absent_settlement` in `lane_d.py` (in PATHS) |
| `test_w14_emission.py` armed tests | assert: `lane_parts` takes no `d_absent` |
| `test_w14_receipt.py` armed tests | assert: `RoundReceipt` has no `lane_d` field, `build_receipt` / `build_round_receipt` take no `lane_d` |
| `test_w14_audit.py` | `ImportError`: no `check_lane_d_absent` in `cascade/audit/lane_d.py` (in PATHS) |

Four tests pass on main and must keep passing: the three unarmed invariant-1
tests and `test_present_records_keep_the_w7c_shape`. `accept W14
--expect-fail` on main: 82 collected, 4 passed, 78 right-reason, 0
wrong-reason.

## Correction (plan rev6.11 (b)): crown at the first verdict

The confirmation duel is dropped (owner 2026-11-23): a crown is decided at
the first verdict, first clear (max-T LCB vs KING_D) ∧ A10 ∧ the §1.3b gate.
There is no `confirm` stage, no `confirm_pending` status or reason and no
`lost_confirmation`; a held first clear is `hold` / `bench_hold`. This WP's
crowning chains move to that rule with nothing weakened. The helpers use W7c's own 48-window `W7cChain`: boundary(1) waits on the floors, boundary(2) is X's first, crowning verdict, boundary(3) judges nothing. The absent-settlement test defers a pending (floor-waiting) entry, which then crowns at its first verdict at boundary(3). On main the
old code still runs the confirmation stage, so the changed tests fail on an
assertion until the W5e implementation lands.
