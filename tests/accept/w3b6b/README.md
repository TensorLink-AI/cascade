# W3b6b acceptance tests: lane D bench rows published, bound and re-derived

W3b6b (plan rev6.13, "W3b6 spec" and the W3b6b row; owner 2026-11-23 "build
this") depends on W3b2, W7a, W7d and W8b. These tests are the spec. The
builder may touch only the paths in `PATHS`. The W3b2, W5e, W7a, W7d, W8b, W8c
and W9 suites must stay green.

## The gap

The §1.3b gate decides on three sweep records (candidate, KING_D, house)
from one bench session, but only their four LCBs reach the signed report. A
satellite can re-check the outcome rule on those numbers, but not the
numbers themselves. GPU sweeps aren't bit-reproducible, so a satellite can't
re-sweep either. Lane A solves this by publishing its rows content-addressed
under `bench_v/` and pinning them by sha256 in a signed report. W3b6b does
the same for lane D.

## Contract

### Bind (`cascade/validator/lane_d_bench.py`, `cascade/validator/lane_d.py`)

```
lane_d_bench.sweep_record_text(record) -> str
```

- It produces canonical JSON: `sort_keys`, `(",", ":")` separators,
  `ensure_ascii=True`, and every non-finite float written as `null`. The
  result is deterministic, and a parsed text re-serialises to the same
  bytes.
- When the bench session returns the three records, `settle_lane_d`'s gate
  adds `rows = {"candidate": sha256(text), "king_d": …, "house": …}` (ASCII
  text bytes) to that result. That covers every outcome the classification
  reaches: pass, gate_regression, checkpoint_fault and bench_hold. A sweep
  that raises gets a `bench_hold` with **no** `rows`. The digests sit inside
  the signed body.
- Before the report is persisted, each bound text is written as its exact
  bytes to `<state dir>/<state stem>_bench_rows/<sha256>.json`, beside
  `<stem>_reports`, so it is snapshotted with the state. A settlement with
  no gate writes nothing, and its report is unchanged.
- The re-adopted gate of a restarted settlement keeps its persisted `rows`
  and files (RV7 B1).
- Schema: `$defs.gate_result.properties.rows` is optional, with exactly the
  three role keys, each `^[0-9a-f]{64}$`, and `additionalProperties: false`.
  The report stays `lane-d-report/1`. The signed LCBs equal `paired_gate`
  re-run on the bound records with the settlement's seed
  (`lane-d/1:gate:<boundary>:<block_hash>`).

### Publish (`cascade/validator/lane_d_wiring.py`)

- Before the report, W8b's step PUTs every bound digest's persisted text at
  `bench_v/lane_d/<boundary>/<sha256>.json`. It uses its existing store,
  content type and ACL: lane A's `bench_v/` prefix, with no new bucket. The
  bytes are the persisted ones, and the key is content-addressed (a re-sweep
  lands at a new key).
- A failed PUT, or a missing or mismatched persisted file, is
  `publish_failed`. The report isn't published, the court doesn't move, and
  the next tick (or W8b's catch-up path) publishes them.
- Nothing is written under lane A's `bench_v/round-*` or `bench_v/rows/*`.

### Re-derive (`cascade/satellite/lane_d_replay.py`)

```
check_gate_rows(report, bench_rows, cfg) -> [mismatch path]     # pure; [] when honest
replay_chain(reports, receipts, cfg, *, signer_ss58, uid_of=None, states=None,
             bench_rows=None)      # calls check_gate_rows inside the gate check
```

`cfg` is the parsed `[lane_d]` config (`alpha`, `bootstrap_B`). The paths
are `gate.results[i].rows` and `gate.results[i].lcb`, and `replay_chain`
adds them to the settlement's mismatches.

- `bench_rows` is a directory of `<sha256>.json` files or a `{sha256:
  bytes}` mapping. When `None` and `reports` is a `<stem>_reports`
  directory, it defaults to the sibling `<stem>_bench_rows`. When `None`
  with a mapping of reports, the rows checks are skipped: the legacy call,
  W7d's existing tests.
- For every gate result with `rows`:
  - each digest must resolve to bytes hashing to it;
  - the parsed record's `session` must equal the result's;
  - its `digest` must equal its role's: the result's digest, the report's
    `king_before.digest`, and the house the gate swept;
  - otherwise the mismatch is `gate.results[i].rows`.
- For a `pass` / `gate_regression` result, the four LCBs are recomputed from
  the drawn suite's rows: items paired by id; per reference and metric, the
  paired LCB of `1 − geomean(cand)/geomean(ref)` with seed
  `cascade|lane_d|paired_gate|v1|lane-d/1:gate:<b>:<hash>|<suite>|<metric>`,
  `alpha`, `bootstrap_B`. They are compared within 1e-12, and any difference
  is `gate.results[i].lcb`.
- A pass result for the crowned digest without `rows` (when rows are
  checked) is `gate.results[i].rows`.
- **Independence (invariant 12; W7d's import rule).** The satellite may
  import only stdlib, numpy and the allowed primitives
  (`cascade.eval.bootstrap`, …), never `cascade.eval.expert_lane`. So it
  re-implements the statistic over `bootstrap._seed_to_int`, and the tests
  pin it equal to `expert_lane.paired_geomean_lcb`. This deviates from the
  plan row's "with `expert_lane.paired_geomean_lcb`" because W7d forbids
  that import; it is flagged.
- Pure: no input is mutated.

### Audit (`cascade/audit/lane_d.py`)

W9's `_result_problem` accepts the optional `rows` object (three sha256)
and still refuses any other extra key. The plan row's PATHS list omitted
this file; the audit checks result keys exactly, so it is added here.

## Consensus

- Deterministic: the same records give the same text, sha256 and LCBs on
  every machine.
- Signed: the digests sit in the signed body.
- Content-addressed: one key per sha256.
- Satellites re-derive the LCBs from published bytes and never re-sweep.
- Lane A is byte-identical. Unarmed is byte-identical. There are no chain
  reads or writes and no new bucket.

## Fixtures

The fixture is W8b's production step over W8b's fakes: W7a's synthetic chain,
a recorded-rows bench pod and the bucket. X is gated and crowned at B2.
The re-derivation tests call `check_gate_rows` directly on that report, plus
doctored copies of it. The full `replay_chain` on W7d's chains waits on the
rev6.11 W5e/W7d rebuild (W7d's own suite is red on main), so only its
`bench_rows` keyword is pinned here. No chain, storage, pod, torch or GPU.

## Failure on main

The bind, publish and replay tests fail on assertions: `rows` absent,
`sweep_record_text` missing, `replay_chain` without `bench_rows`, no
`bench_v/lane_d/` keys. The schema test fails on the missing property, and
the audit test on its key-set refusal. A few no-gate and refusal checks
already hold on main and pass.
