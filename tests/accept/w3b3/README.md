# W3b3 acceptance tests: the era's sentinel, swept and published for every new king

W3b3 (plan §3.0 W3b3 row, §1.3b "Held-out suite = the cross-lane rotating
sentinel", §16 rules 1, 2 and 4, §4 T7b) adds the lane D sentinel step. For
every NEW KING_D, the era's §16 sentinel suite is swept for the new king, the
previous KING_D and the house in one bench host session, and a sentinel
record is published. The sentinel is never a gate: it can't change the crown,
a verdict, the pay (the weight input) or a later settlement.

These tests are the spec. The builder may touch only the paths in `PATHS`,
which is one new module, `cascade/validator/lane_d_sentinel.py`. The W3b1,
W3b2 and W5a–W5d tests must stay green.

Some rules for the tests themselves:

- **The crowning is real.** `helpers.build_chain` runs four `settle_lane_d`
  settlements on main, with no hand-written state:
  - BOUNDARY_1: K1 first-clears against the genesis house H0;
  - BOUNDARY_2: K1 confirms and is crowned, so the previous KING_D is the
    house;
  - BOUNDARY_3: Y first-clears against K1;
  - BOUNDARY_4: Y confirms and is crowned, so the previous KING_D is the
    miner K1.
- **Only the two external services are stand-ins.** `helpers.RecordingHost`
  is the bench pod. It returns recorded per-item rows behind a real
  `lane_d_bench.SweepSession`. `helpers.MemoryStore` is the public bucket,
  with the `put_text` / `get_text` duck type of
  `cascade.shared.hippius.S3Store`. An absent key raises `ObjectNotFound`.
  There's no network, GPU, torch or checkpoint.
- **The module is never imported at module level, in a fixture or at
  helpers import time.** It's imported only inside functions that test bodies
  call (`helpers.run`). The `never_gates` chains are built lazily from the
  test body for the same reason.

On main 41 tests fail with a `ModuleNotFoundError` for
`cascade.validator.lane_d_sentinel`, which is in `PATHS`. The hash-seed test
fails on the AssertionError that carries the subprocess's
`ModuleNotFoundError`. One test passes: the fixture check that the chain
crowns K1 and then Y.

## `sweep_and_publish(report, *, era_block, era_block_hash, sentinel_suites, gate_suite, bench, store) -> dict | None`

| argument | meaning |
|---|---|
| `report` | the `lane-d-report/1` dict `settle_lane_d` returned, or `None` (unarmed) |
| `era_block`, `era_block_hash` | the era-start block of the settlement and its hash (`0x` + 64 hex) |
| `sentinel_suites` | `[v2] sentinel_suites` (W-S1) |
| `gate_suite` | the suite W3b1 drew for this settlement's gate |
| `bench` | the settlement's `lane_d_bench.SweepSession` |
| `store` | the public object store (`put_text(key, text, *, content_type, acl)`, `get_text(key)`) |

### Who is swept

- **A new king only.** When `report` is `None` or `report["crowned"]` is
  `None`, the call returns `None`. It makes no host call and no store write.
- **The roles** are `ROLES = ("king", "previous_king", "house")`:
  - `king` is `report["crowned"]`;
  - `previous_king` is `report["king_before"]["digest"]`;
  - `house` is `report["house"]["digest"]`.
- **One session.** The three are swept through `bench.paired(king,
  previous_king, house, suites=…)`. A fresh session makes **one** host call
  holding every distinct digest. A house that was the previous KING_D is
  swept once and recorded under both roles.
- **Rows the session already holds are re-used.** If the settlement's gate
  sweep already ran the sentinel for those digests in this session, there's
  no new host call.

### Which suite

- The suite is `sentinel_for_era(era_block_hash, sentinel_suites)`. It's the
  W-S1 draw, never a fixed TIME. The module must import
  `cascade.shared.sentinel.sentinel_for_era`, and the tests check
  `lane_d_sentinel.sentinel_for_era is cascade.shared.sentinel.sentinel_for_era`.
  - Each of the four suites is swept when the hash draws it.
  - A `sentinel_suites` without TIME never sends TIME to the host.
- **Refusals.** These raise `ValueError` / `TypeError` before any host call
  or store write:
  - the drawn sentinel equals `gate_suite`;
  - a malformed `era_block_hash`;
  - an empty `sentinel_suites`.

### The record

`SCHEMA = "lane-d-sentinel/1"`. The record has these keys:

| key | value |
|---|---|
| `schema` | `"lane-d-sentinel/1"` |
| `boundary` | `report["boundary"]` |
| `era` | `{"block": era_block, "block_hash": era_block_hash}` |
| `suite` | the drawn sentinel |
| `sentinel_suites` | the list it was drawn from |
| `gate_suite` | `gate_suite` (never equal to `suite`) |
| `king_digest` | the new king |
| `session` | `bench.session_id` |
| `status` | `"ok"`, `"fault"` or `"error"` |
| `reason` | `None` when `ok`, otherwise a non-empty str |
| `gating` | `False` |
| `models` | `{role: {"digest", "status", "items", "crps", "mase"}}` for each of the three roles |

- **Per model, when the rows are clean:** `status` is `"ok"`, `items` is
  the row count, and `crps` / `mase` are the geometric means of that model's
  per-item values on the sentinel suite (to a relative 1e-9).
- **Faults.** A model's rows are a fault when any of these holds:
  - the suite is absent;
  - its status isn't `ok`;
  - it's `partial: true`;
  - it has no rows;
  - a metric is NaN, inf or `None`;
  - a metric key is missing.

  A faulty model has `status: "fault"` and `crps` / `mase` `None`. The
  record's `status` is then `"fault"`, and the other models keep their
  numbers.
- **A host error.** When `bench.paired` raises (the pod is unreachable or
  resets), the record has `status: "error"` and a reason, every model has
  `status: "error"` with `None` metrics, and **it is still published**. The
  call doesn't raise.
- **Strict canonical JSON.** The text is `json.dumps(record, sort_keys=True,
  separators=(",", ":"), allow_nan=False)`, and `sentinel_bytes(record)` is
  that text, encoded. It parses back with NaN/Infinity refused.
- **It's deterministic.** The same inputs give the same text, even in two
  subprocesses with different `PYTHONHASHSEED`s. So there are no wall-clock
  fields.

### Publishing

- **The key** is `sentinel_key(era_block, king_digest) ==
  f"lane-d/sentinel/era-{era_block}/{king_digest}.json"`.
- **The return value** is `{"key", "record", "text", "published"}`.
- **Idempotent.** Re-publishing identical text writes nothing, and the call
  still returns `published: True`. A published `ok` record is never
  replaced, even by a later re-sweep with different numbers.
- **The typed retry.** An `error` record at the key is replaced by a later
  sweep's `ok` record, in a new session.
- **A store failure** (`put_text` raises `StorageError`) never raises. The
  call returns `published: False` with the record and text, so the caller can
  retry the same bytes.

### Never gating

- **The inputs are untouched.** `report` and `pay` are unchanged by the call,
  and so are the state file and every report field.
- **The whole chain is identical whatever the sentinel numbers are.** The
  test runs it with the sentinel step after every settlement under four
  scenarios:
  1. good;
  2. a regression vs the previous king and the house;
  3. a NaN fault;
  4. a host error.

  Every report (minus its randomised sr25519 signature), every pay record and
  the final state (minus `last_report.signature`) are identical across them.
  Y is crowned in all four, even though the four published records differ.

## Not pinned

- **Where the step runs.** These tests give the validator nothing to call,
  and `lane_d.py` isn't in `PATHS`, because the step runs after
  `settle_lane_d` on its returned report (§16 "the sentinel sweep can run
  after the verdict and stays off the critical path"). The call site needs
  these from the validator's post-settlement path: the era-start block and
  hash (`cascade.shared.sentinel.settlement_era_start`), the gate record W5e
  writes, the bench session and the manifest store. The orchestrator must
  assign that wiring explicitly.
- **The D34 ratchet** (a sentinel regression vs the lane A king blocks). The
  W3b3 row is publish-only and never gating. A ratchet would be a separate
  consensus change.
- **The "falls for 3 kings in a row" alert** (§1.3b), the status boards
  (W-S3) and the signed lane D index (§11.1).
- **Behaviour when W-S1 is unarmed.** That means an era with no drawn
  sentinel, the case where W3b1's `gate_suite_for(..., sentinel=None)` holds
  nothing out.
- Paired LCBs between the roles, extra record keys, the reason text, the
  content type and ACL of the put, and what a `get_text` outage does when
  `put_text` works.

## Tests

| Rule | Test |
|---|---|
| the chain crowns K1, then Y | `test_fixture_chain_crowns_y_over_k1_and_k1_over_the_house` |
| W-S1 draw, schema, roles | `test_the_module_uses_the_w_s1_era_draw` |
| swept suite = era draw (each suite) | `test_the_swept_suite_is_the_era_draw_not_a_fixed_suite` |
| configured `sentinel_suites` | `test_the_draw_follows_the_configured_sentinel_suites` |
| self-describing record | `test_the_record_is_self_describing` |
| strict canonical JSON | `test_the_record_is_strict_canonical_json` |
| deterministic key | `test_the_key_is_deterministic` |
| one session call | `test_king_previous_king_and_house_are_swept_in_one_session_call`, `test_a_house_previous_king_is_swept_once_and_recorded_twice`, `test_rows_already_swept_in_the_session_are_reused` |
| determinism | `test_the_record_is_deterministic`, `test_the_record_is_independent_of_the_hash_seed` |
| publish once, idempotent | `test_the_record_is_published_once_under_its_key`, `test_republishing_the_same_record_writes_nothing`, `test_a_published_ok_record_is_never_replaced` |
| no new king, no sweep | `test_no_new_king_means_no_sweep_and_no_publish`, `test_an_unarmed_settlement_report_publishes_nothing` |
| host error → error record | `test_a_host_error_publishes_an_error_record_and_never_raises` |
| typed retry | `test_an_error_record_is_replaced_by_a_later_good_sweep` |
| model faults | `test_a_faulty_model_is_recorded_as_a_fault` |
| store failure | `test_a_store_failure_never_raises` |
| refusals | `test_a_sentinel_equal_to_the_gate_suite_is_refused`, `test_a_malformed_era_hash_is_refused`, `test_empty_sentinel_suites_are_refused` |
| never gating | `test_the_sentinel_ran_for_every_crowning_and_only_those`, `test_the_sentinel_saw_the_numbers`, `test_settlements_are_identical_whatever_the_sentinel_numbers`, `test_the_record_declares_it_never_gates`, `test_the_sentinel_step_writes_nothing_but_its_record` |

## RV5 correction: the crowning chain passes the gate inputs (W5e fail-closed)

Once W5e lands (`tests/accept/w5e/README.md` "Fail closed"), an armed
settlement that would crown without a `bench` session and a
`boundary_block_hash` is `bench_hold`, not a crown. The fixture chain crowns
at BOUNDARY_2 (K1 over the house) and BOUNDARY_4 (Y over K1) through
`helpers._settle`, so `_settle` now passes `boundary_block_hash`
(`helpers.gate_block_hash`), `sentinel=None` and a healthy `bench`
(`lane_d_bench.SweepSession` over `helpers.HealthyBench`) whenever
`settle_lane_d` accepts those keywords (`inspect.signature`), and makes the
legacy call otherwise. The tests pass both on today's main and after W5e.

- The settlement's gate bench is a session of its own. It is not the
  sentinel `bench` the tests hand to `sweep_and_publish` (`RecordingHost`),
  so every host-call assertion on the sentinel sweep is unchanged.
- Every digest but the house returns the same recorded rows (a tie with
  the miner KING_D K1 never regresses) and the house is 10% worse, so both
  crowning candidates pass on any drawn suite and the duel alone decides.
- No assertion changed. No W3b3 test validates a report, so the gate record
  W5e writes on a crowned report isn't checked here.

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
crowning chains move to that rule with nothing weakened. The chain lists each challenger from the settlement that crowns it, so K1 is still crowned at BOUNDARY_2 and Y at BOUNDARY_4 (BOUNDARY_1 and BOUNDARY_3 judge nothing). On main the
old code still runs the confirmation stage, so the changed tests fail on an
assertion until the W5e implementation lands.
