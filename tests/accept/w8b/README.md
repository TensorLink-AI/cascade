# W8b acceptance tests: validator production wiring for lane D

W8b (plan rev6.2 item (g), W8b row; owner accepted 2026-11-23) depends on
W5e, W3b5, W7c, W8, W3b3, W-S3 and W17, and plugs into W19's off-thread seam.
These tests are the spec. The builder may touch only the paths in `PATHS`.
W19, W7c, W8, W9, W3b3, W-S3 and W17's suites must stay green.

## What the plan says

The W8b row, verbatim:

> | W8b | **(rev6.2, proposed)** Production wiring: today nothing outside tests calls `cascade.validator.lane_d.settle_lane_d`. The validator loop calls it at each lane D settlement boundary with: the boundary block hash (read-only chain read); the era's sentinel (W-S1 draw from the era-start hash); a bench `SweepSession` on the configured bench host (the W3b2/W5e gate inputs; an armed crown without them is a `bench_hold`); `expect_prior_sha256` from the last published receipt (W7c); the low-frequency check (W3b5); and the house/king state. After settlement: W3b3 `sweep_and_publish` for a new king; W-S3 `build_sentinel_board` with the era's kings list (from W7c state / W17); publish the signed report + receipt; lane D pay reaches weights (W8 emission) **only via the published report**. **Fail-closed:** any unreadable input ⇒ no settlement this tick (retry next tick), never a guessed input. **Restart-safe:** a re-call after restart is idempotent (W7c). **Weight loop:** lane D's bench work runs off the weight loop (W19) or is bounded so weight reasserts are never delayed past a stated bound (2026-11-22: weights went stale ≥ 2 h during a synchronous king-check sweep). **Unarmed:** byte-identical (invariant 1). Ships release-then-activate (external validators on netuid 91) | W5e, W3b5, W7c, W8, W3b3, W-S3, W17 | ci + testnet | `scripts/laned/accept W8b` |

Also §1.5 "State" ("Its digest is in the receipt's prior and posterior
digests ... `checks.derive_weights` and `check_lane_d` re-derive the lane D
weights"), invariants 1 and 6, T15, and the W7c rev6.1 wiring trap (RV3 note
10: lane D's part must reach `emission_vector`'s second argument even with
lanes B and C off).

## Today's code (main @ 54e7f23)

- Nothing outside tests calls `settle_lane_d` (`cascade/validator/lane_d.py:702`).
- `weights.py` `_apply_weights_inner` / `_set_policy_vector` never pass
  `lane_d_cfg` to `lane_parts`, so an armed lane D's 0.25 never reaches the
  vector (`{0: 0.95, 7: 0.05}` armed or not at the fixture block).
- `lane_d_status.py` says "Wiring the publish into the validator loop is
  W8b's"; `lane_d_sentinel.sweep_and_publish` and
  `shared/sentinel_board.build_sentinel_board` have no production caller.
- W19's implementation is not on main when these tests land. W8b pins
  against W19's README contract (as corrected by #691):
  `cascade/validator/offthread_worker.py` (`LaneDWorker`: `submit` /
  `running` / `join`), `ValidatorRunner.lane_d_step`, the per-poll
  `_lane_d_tick(client)` and `self._lane_d_worker`. The W8b build needs W19
  on main first; the off-thread test asserts W19's seam before using it.
- **W9 invariant 12** (`tests/accept/w9/test_w9_independence.py`): the
  audit's static import closure (every import at any depth) reaches
  `cascade/validator/loop.py` and `weights.py` and must load no
  `cascade.validator.lane_d*` module. So neither may import the wiring or
  `settle_lane_d`, even lazily; `cascade/validator/main.py` is outside the
  closure and is the call site.

## Contract

### `cascade/validator/lane_d_wiring.py` (new)

```
LaneDInputs(pool_dir, entries, forecasts, uid_of, house, prior_pool_dirs=())   # keyword-constructible
LaneDWiringError(RuntimeError)   .code in WIRING_ERROR_CODES
WIRING_ERROR_CODES ⊇ {"block_hash_unreadable", "era_hash_unreadable", "inputs_unavailable",
                      "receipt_unreadable", "settle_refused", "publish_failed"}
report_key(boundary) -> str      # "lane-d/…/lane_d_report_<boundary>.json"
receipt_key(boundary) -> str     # "lane-d/…", distinct from report_key
RECEIPT_LATEST_KEY: str          # the last published receipt
build_lane_d_step(cfg, *, state_dir, signer, chain_factory, store,
                  inputs=None, bench=None) -> step | None
```

- `cfg` is the full `ChainConfig`. **Unarmed** (`[lane_d] from_block = 0`):
  returns `None` and calls nothing (no factory, no store, no file).
- `state_dir`: the lane D state is `state_dir / "lane_d_state.json"` (W7c's
  reports sit beside it, `report_path`).
- `signer`: the validator's hotkey (keypair or wallet, what
  `settle_lane_d` takes).
- `chain_factory()`: builds the step's **own** read-only chain reader
  (`block_hash(block) -> str`). It is called on the step's thread (never on
  the constructing thread; lazily, at the first armed call) and is never the
  poll thread's client (W19 README "Not pinned here": a websocket must not be
  shared across threads).
- `store`: the bucket (`get_text(key)` raising `ObjectNotFound` for a
  missing key and `StorageError` otherwise; `put_text(key, text,
  content_type=, acl=)`).
- `inputs(boundary) -> LaneDInputs`: the settlement's pool snapshot(s), the
  revealed entries, the eval host's forecasts, the metagraph `{hotkey: uid}`
  and the house. `None` ⇒ every armed call is `inputs_unavailable`.
- `bench(boundary) -> SweepSession | None`: the boundary's bench session on
  the configured bench host. `None` (or a factory returning `None`) ⇒
  `settle_lane_d(bench=None)`, so a crowning candidate is `bench_hold`.

The step is `step(boundary)` (what W19's worker runs) plus
`step.court_for(boundary)`.

**`step(boundary)`**, in order:

1. Not `cfg.lane_d.active(boundary)` (e.g. `boundary == from_block`, which
   belongs to the era before): returns `None`, touches nothing.
2. Reads the boundary block hash through its reader; an exception or a value
   that isn't `0x` + 64 lower-case hex is `block_hash_unreadable` (also when
   `chain_factory` raises).
3. With W-S1 armed for the boundary's era
   (`sentinel_armed(cfg, settlement_era_start(cfg, boundary))`), reads the
   era-start block hash (`era_hash_unreadable`) and draws the sentinel
   `sentinel_for_era(era_hash, cfg.v2.sentinel_suites)`; unarmed it is `None`.
4. Reads the last published receipt at `RECEIPT_LATEST_KEY` (absent ⇒ no
   prior; `StorageError` or a malformed record ⇒ `receipt_unreadable`) and
   settles with `expect_prior_sha256` bound to it, so a local state that is
   not the last published posterior is refused (`settle_refused`, the W7c
   `LaneDStateError` chained as `__cause__`). Crash recovery must still work
   (below), so how the bound is applied to a re-call is the builder's.
5. Reads `inputs(boundary)` (`inputs_unavailable` on any exception).
6. Calls `settle_lane_d(cfg=cfg.lane_d, boundary, pool_dir, entries, house,
   forecasts, uid_of, state_path, signer, prior_pool_dirs,
   expect_prior_sha256, boundary_block_hash, sentinel, bench)`. The W3b5
   low-frequency check, the W5e gate and the W7c state/receipt are inside it.
   Any `LaneDStateError` is `settle_refused`.
7. For a report that crowned a new king with W-S1 armed: W3b3
   `sweep_and_publish(report, era_block=<era start>, era_block_hash=<era
   hash>, sentinel_suites=cfg.v2.sentinel_suites,
   gate_suite=report["gate"]["suite"], bench=<the same session>, store)`. It
   never gates: an errored sentinel suite is recorded, the crown and pay
   stand. Unarmed W-S1: no sentinel key and no extra bench call.
8. Publishes the persisted signed report **byte for byte** at
   `report_key(boundary)`, then the receipt record (canonical JSON; it loads
   equal to `LaneDSettlement.receipt`) at `receipt_key(boundary)`, then moves
   `RECEIPT_LATEST_KEY` to that receipt's text. A failed put is
   `publish_failed` (retried next tick).
9. Publishes `status/lane_d.json`: exactly `publish_lane_d_status(store,
   build_lane_d_status(cfg, state=<posterior state>, report=<report>,
   block=boundary, era_block=<era start>, era_block_hash=<era hash>,
   store=store))`. W17's builder embeds W-S3's `build_sentinel_board` with the
   era's kings list when W-S1 is armed (that is the row's W-S3 call site). A
   status failure never withholds pay.

Every refusal in 2-6 writes no file, publishes nothing and calls no bench;
the court stays what it was; the next call with the input readable settles
normally (W19: a raising step is retried at a later submit).

**`step.court_for(boundary) -> list[int | None] | None`**: the court uids
(`[s["uid"] for s in report["pay"]["court"]]`) of the latest settlement this
step has **published** (report and receipt, step 8) at a boundary
`<= boundary`, or `None`. No I/O (memory only), never raises. A settlement
that is saved but not published never reaches it.

### Restart (W7c)

- A re-call at a published boundary, in the same or a fresh step: no bench
  call, no changed published object, no changed file, and `court_for`
  returns the published court.
- A crash between the report file and the state file: the fresh step adopts
  the persisted report (no bench call), publishes those exact bytes, the
  receipt verifies against it and `verify_receipt_chain` links.
- State saved but the publish failed: the next tick (fresh step) publishes
  the receipt a clean run gives, without a bench call; the chain links on
  through B3.
- An older boundary after a newer one: `settle_refused` or a no-op, nothing
  changes.

### Weights (`cascade/validator/weights.py`)

When `cfg.lane_d.active(boundary)` for the vector's boundary
(`_reassert_boundary(cur)` on a re-assert) and not `force_burn`:

- the court is read **only** through `self.lane_d_step.court_for(boundary)`
  on the poll thread (`weights.py` imports no lane D producer module);
- the vector is W8's: `lane_parts(..., lane_d_cfg=cfg.lane_d,
  d_court_uids=court)` with lane D's part passed in `emission_vector`'s
  second argument even with lanes B and C off (RV3 note 10), under the
  runner's DEC-CA-0055 policy;
- no step, a step without `court_for`, `court_for` returning `None` or
  raising ⇒ the empty court: lane D's whole share burns to `burn_uid`, and
  every other uid's weight is **byte-identical** to the paid vector (lanes
  A/B/C never move with lane D);
- the cross-check: `cascade.audit.checks.derive_weights(receipt, cfg,
  lane_d={"report", "receipt"})` over the published B2 report and receipt
  re-derives the pushed vector.

Not active at the boundary (unarmed, before or at `from_block`) or
`force_burn`: today's vector, byte for byte, and `court_for` is not called.

### The entry point (`cascade/validator/main.py`)

`main()` calls `build_lane_d_step(...)` and assigns `runner.lane_d_step`
before `runner.run_forever(...)`, with a `chain_factory` that is not the poll
`client` (AST check; e.g. a second `ChainClient.from_config` for the step's
thread). W19's `_lane_d_tick` then submits each armed boundary off the poll
thread. `loop.py` is not touched by W8b. With the wiring in place the
audit's import closure still reaches no `cascade.validator.lane_d*` module
(`test_inv12_audit_import_closure_still_reaches_no_lane_d_module`).

### Weight timing (invariant 6, T15)

The real step with a bench host whose B2 gate sweep hangs, driven by W19's
tick and the head re-assert once per poll (10 blocks) from B1 − 300 to
B2 + 1500: the push blocks equal the no-lane-D baseline, every poll is under
1 s, no gap exceeds the interval + one poll − 1, pushes happen only on the
poll thread, the bench and the reader run off it, the poll client's
`block_hash` is never used by the step, X is unpaid while its sweep hangs and
paid 0.25 once B2 is published.

## Unarmed goldens (invariant 1)

Recorded on main @ 54e7f23 (`helpers.py`): the shipped `chain.toml` (loader
parse), `genesis("w8bKingHotkey", 7)`, 256 uids, a fresh runner's head
re-assert at `9,720,905` and `9,721,805`:

- the vector (little-endian float64) sha256 `951fc9ae…eea35c` at both;
- the round receipt `build_round_receipt` assembles for that settlement
  (a no-contest round from `tests/unit/receipt_fixture.make_manifest`,
  base seed 1234567, carrying the vector and `_last_emission`):
  `c1b7b524…a47f22` and `e34e5474…729e6bf9`.

`test_inv1_weights_and_receipt_unchanged_when_unarmed` (with and without a
step injected: it is never consulted) pairs with
`..._armed_nonempty_lane_d_effect`; `test_inv1_no_lane_d_step_*` pins the
unarmed `build_lane_d_step` (returns `None`, calls nothing) and its armed pair.

## Fixtures and services

Every settlement is a real `settle_lane_d` run on W7a's synthetic chain
(numpy only): X first-clears at B1 (`ARM + 900`), is confirmed, gated and
crowned at B2 (`ARM + 1800`, uid 6), B3 judges nothing. `[lane_d]` is W7a's
armed table on the shipped `chain.toml`; the signer is W7a's test key,
pinned as `[manifest] validator_hotkey`. The oracle (`helpers.Oracle`) makes
the same settlements by hand with the row's inputs. Fakes: the step's reader,
the poll chain client, the bench pod, the bucket. No chain, storage, pod,
torch or real key; everything under `tmp_path`.

## Readings chosen (the row is silent; flagged)

- **Where the inputs come from.** The row names the gate, sentinel, prior and
  state inputs; it does not name the source of entries, forecasts and the
  pool (W3 intake, the eval host, W6). W8b takes them through `inputs`; with
  none configured production fails closed (`inputs_unavailable`, lane D
  burns). Wiring a real source is a later WP's.
- **Pay between settlements.** The court of the latest *published*
  settlement stands until the next one is published (the lane B precedent:
  `weights.py` pays from the persisted last settled boundary). Before any
  publish, or while a settlement is unpublished, the share burns. No
  staleness bound is pinned. `derive_weights` requires the lane D record of
  the receipt's own boundary, so a lane A receipt published while the lane D
  step for that boundary is still running pays the previous court and won't
  re-derive with that boundary's record: flagged for W9 / the owner.
- **Bench host unreachable** (the factory raises): either no settlement this
  tick or a `bench_hold` settlement; never a crown.
- **The round receipt.** "Folds the lane D pay into the receipt" is read as:
  the vector the round receipt records carries lane D's part, and the lane D
  report + its W7c receipt record are published beside it. No new
  `RoundReceipt` field is pinned (unarmed receipt bytes are).
- **After a restart** a fresh step's `court_for` before its first call isn't
  pinned (None, or the published court).
- The testnet half of `Runs in` is not here.

## Failure on main

Every wiring test fails with `ModuleNotFoundError: cascade.validator.lane_d_wiring`
(in `PATHS`); the weight fold tests fail on assertions (lane D unpaid, the
court never read); the off-thread
test on W19's missing seam (assertion); the entry-point check on an
assertion; the invariant 12 check on the missing wiring module. The unarmed goldens, the burn cases
(under the shipped policy a burnt lane D equals today's vector), the
inactive-boundary and `force_burn` checks pass on main. Run locally with
`-m "not torch"`.
