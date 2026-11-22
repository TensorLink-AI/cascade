# W3b4 acceptance tests: suite identity and partial classification

W3b4 (plan §3.0 W3b4 row, §1.3b "Faults" and "Suite identity is verified per
drawn suite", §13 "Suite identity / partial", §4 T7b and T17) closes the two
things W3b2's README left open: how a swept suite is **identified**, and how
an `error` status, an absent suite, a timeout or a partial / NaN row set is
**classified** (the miner's `checkpoint_fault` or an infra `bench_hold`, with
a typed sub-reason).

These tests are the spec. The builder may touch only the paths in `PATHS`:

- `cascade/eval/suite_identity.py`, a new module. Identity sits next to the
  runners (`benchmarks.py`, `public_guards.py`) because it is a property of
  the suite data, not of lane D. Lane A and lane B (WB11's hash set, LB1's
  "refreshed only when a suite identity changes") can re-use it.
- `cascade/validator/lane_d_bench.py`, W3b2's module, extended.

Rules for the tests themselves are W3b2's: nothing is mocked except the
bench host's sweep (an external service; `test_w3b4_gate._Host`); no PATHS
module is imported at module level, in a fixture or in `helpers.py`; no test
needs torch or a GPU; every test runs with `--noconftest`.

On main every test fails with a `ModuleNotFoundError` for
`cascade.eval.suite_identity` or an `ImportError` ("cannot import name") for
`classify_suite` / `FAULT_CODES` / `HOLD_CODES` from
`cascade.validator.lane_d_bench`. Both are in `PATHS`. The W3b2 tests must
stay green: every W3b2 call (no `pinned_identity`) keeps its outcome.

## (a) Suite identity: `cascade/eval/suite_identity.py`

The repo's benchmark metadata for a swept suite is its key (`gift`, `boom`,
`time`, `fev`: `cascade.eval.benchmarks.ITEM_SUITE_KEY` values), its pinned
data revision (`data_revision`, which `item_rows_from_report` copies from the
report's `data_revisions`) and its item rows. The identity is built from
those three.

### `item_set_digest(item_ids) -> str`

`sha256` (lower-case hex) of
`json.dumps(sorted(set(ids)), separators=(",", ":"), ensure_ascii=True)`
encoded as ASCII. Order and repeats don't matter. A non-str or empty id
raises `TypeError`/`ValueError`.

### `suite_identity(suite, entry) -> dict`

`entry` is one suite entry of a sweep record (`{"status": "ok", "rows": […],
"data_revision"?: str, …}`). It returns:

```json
{"suite": "boom", "version": "boom@3f1c2a9", "items_sha256": "<64 hex>",
 "n_items": 40, "digest": "<64 hex>"}
```

- `version` is the entry's `data_revision` as a str, or `None` when it has
  none.
- `items_sha256` is `item_set_digest` over the rows' `item` ids, every row
  counted whatever its metric values (`None`, NaN). `n_items` is the number
  of distinct ids.
- `digest` is the `sha256` hex of `json.dumps({"items_sha256", "suite",
  "version"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`
  encoded as ASCII. It's one value a config can pin.
- The result is strict JSON. It's a pure function of the item set and the
  version: row order, metric values and the process hash seed don't change
  it. Two validators (two subprocesses with different `PYTHONHASHSEED`s, rows
  in different orders) get the same identity.
- **Refusals:** an unknown suite key, or an entry whose `status` isn't `"ok"`,
  raises `ValueError`.

### `identity_mismatch(observed, pinned) -> list[str]`

The fields among `("suite", "version", "items_sha256")`, in that order,
whose values differ. `[]` means the same identity.

### `identities_from_rows(item_rows) -> dict`

`{suite: suite_identity(suite, entry)}` for every `"ok"` suite in an
`item_rows_from_report` result. Non-ok suites are left out.

`lane_d_bench` must import `suite_identity` from this module
(`lane_d_bench.suite_identity is cascade.eval.suite_identity.suite_identity`).

## (b) Classification: `classify_suite` in `lane_d_bench`

```
classify_suite(candidate_rows, king_rows, house_rows, *, suite,
               pinned_identity=None) -> dict
```

It takes W3b2's three sweep records from one host session and the drawn
suite. It returns a strict-JSON dict:

| key | value |
|---|---|
| `outcome` | `"ok"`, `"checkpoint_fault"` or `"bench_hold"` |
| `code` | the typed sub-reason (below), `None` when `ok` |
| `codes` | `{role: code}` for **every** defective role (`candidate`, `king_d`, `house`), `{}` when `ok` |
| `fault_on` | `["candidate"]` for `checkpoint_fault`; the sorted defective **references** for `bench_hold` (never `"candidate"`, so it may be `[]`); `[]` when `ok` |
| `identity` | `{"pinned": pinned_identity or None, "candidate": …, "king_d": …, "house": …}`, each role's `suite_identity` of its drawn-suite entry, or `None` when that entry isn't `"ok"` |

The same session and suite checks as `paired_gate` apply: mixed sessions
raise `MixedSessionError` (`code == "mixed_session"`), and an unknown suite
raises `ValueError`.

### Per-role defect codes

Each role's drawn-suite entry gets at most one code:

| code | when |
|---|---|
| `absent` | the record has no entry for the suite |
| `timeout` | `status == "timeout"`, or `status == "error"` whose `detail` contains `timeout` / `timed out` (case-insensitive) |
| `error` | any other `status == "error"` |
| `partial` | the entry falls short of its coverage. When it carries any of `skipped`, `n_expected`, `max_series`, that's `cascade.eval.benchmarks.suite_incomplete` on it (with the sidecar suite name, so FEV `stage == "scope"` skips are **not** partial and its own `partial` flag is ignored). Otherwise it's `partial: true`. |
| `suite_mismatch` | the entry's identity differs from the pin (see below) |
| `missing_items` | items are missing: the rows are a strict subset of another record's items (with a pin: of a record that matches the pin; without one: W3b2's rule, items every other checkpoint has). An `"ok"` entry with no rows is `missing_items`. |
| `nonfinite_metric` | a `crps`/`mase` is NaN, `inf` or `-inf` |
| `null_metric` | a `crps`/`mase` is `None` |
| `missing_metric` | a row lacks the `crps` or `mase` key |

A non-ok `status` decides before the rows are looked at. The order among
row-level codes when one entry has several defects is not pinned.

### Suite identity against the pin

`pinned_identity` is one identity for the drawn suite: a mapping with
`suite` (equal to the gate suite), `version` (str or `None`) and
`items_sha256` (64 lower-case hex). `n_items`/`digest` may ride along. A
malformed pin, a non-mapping or a pin for another suite raises
`TypeError`/`ValueError`.

- **With a pin,** a role whose `version` differs from the pinned one is
  `suite_mismatch`. A role whose item set differs is `missing_items` when its
  items are a strict subset of a pin-matching record's items. Otherwise
  (extra items, or no record matches the pin, for example the same item
  missing from every record) it's `suite_mismatch`.
- **Without a pin,** records whose `version`s disagree are a
  `suite_mismatch` hold. Item completeness is relative to the other records
  (W3b2), so an item missing from every record can't be detected.
- The sentinel suite isn't checked against the pin.

### The outcome

1. If any role carries `suite_mismatch` or `timeout`, the outcome is
   `bench_hold` and `code` is the first of `("suite_mismatch", "timeout")`
   carried by any role. That includes the candidate alone: a different suite
   version or a timeout is never the miner's fault.
2. Otherwise, if a reference is defective, the outcome is `bench_hold` and
   `code` is the code of the first defective reference in sorted order
   (`house` before `king_d`). This is the case whatever the candidate's rows
   are, and it covers the same pattern on every row set.
3. Otherwise, if the candidate is defective (and both references are
   complete), the outcome is `checkpoint_fault`, the miner's fault (T17), and
   `code` is the candidate's code. That includes `error` and `absent`: the
   candidate alone errors, or the host returns nothing for it, while the
   references in the same session are fine.
4. Otherwise the outcome is `ok`.

`FAULT_CODES` (tuple) holds at least `absent`, `error`, `partial`,
`missing_items`, `missing_metric`, `null_metric` and `nonfinite_metric`.
`HOLD_CODES` (tuple) is a superset of it plus at least `suite_mismatch` and
`timeout`. Neither of those two is in `FAULT_CODES`. Every `bench_hold`
code, including one W3b2's paths raise (for example a reference sharing no
item with the candidate), is in `HOLD_CODES`.

## `paired_gate` uses it

`paired_gate` gains `pinned_identity=None` (same meaning) and calls the
same classification:

- **`classify_suite` not `ok`:** `paired_gate`'s `outcome`, `code`, `codes`
  and `fault_on` equal `classify_suite`'s. W3b2's `passed`/`hold`/`reason`
  rules still hold.
- **`ok`:** the outcome is `pass` or `gate_regression` (W3b2's LCB rule).
  `code` is `None` and `codes` is `{}`.
- **Every outcome** carries `identity`, equal to `classify_suite`'s.
- A matching pin changes nothing: same outcome, same `lcb` as without one.
- The result stays strict JSON.

## Not pinned

- Where the pin comes from (a `[lane_d]` key, the signed overlay, a house
  sweep). That's config wiring for the WP that reads it, with W3b2's
  `settle_lane_d` wiring.
- Statuses other than `ok`/`error`/`timeout`, such as `skipped`. They are
  defects whose code is in `HOLD_CODES`. Who they blame isn't pinned.
- Duplicate item ids. They're a defect (W3b2), with a code of the builder's
  choice.
- The order among row-level codes, the reason text, and extra result keys.
- Identity of the sentinel suite.
