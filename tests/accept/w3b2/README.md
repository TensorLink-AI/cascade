# W3b2 acceptance tests: the paired gate sweep vs KING_D and the house on one host

W3b2 (plan §3.0 W3b2 row, §1.3b "Rule on the drawn suite" and "Faults", §4
T7b, T17) adds the lane D rotating gate's **paired** judgement on the suite
W3b1 draws. The rule: the candidate, KING_D and the house run on one host in
one sweep, never against stored absolute numbers. CRPS **and** MASE must each
satisfy `LCB_α ≥ −tau_public` against KING_D **and** against the current
house. The era's §16 sentinel suite is swept and recorded but never gates.

These tests are the spec. The builder may touch only the paths in `PATHS`,
which is one new module, `cascade/validator/lane_d_bench.py`.

Some rules for the tests themselves:

- Nothing is mocked. The only stand-in is the bench host's sweep, which is an
  external service. `helpers.RecordingHost` returns recorded suite outputs and
  logs each call.
- No PATHS module is imported at module level, in a fixture or in
  `helpers.py`. Every test imports it in its body.
- No test needs torch or a GPU, and every test runs locally with
  `--noconftest`.

On main every non-skipped test fails with a `ModuleNotFoundError` for
`cascade.validator.lane_d_bench`, which is in `PATHS`. `accept W3b2
--expect-fail` reports 86 collected: 84 fail for a right reason, 0 for a
wrong one, and 2 skip (the `test_evidence_*` pair, because `LANED_EVIDENCE`
is unset).

## Sweep records (the recorded suite outputs)

A sweep record is one checkpoint's output from one host session:

```json
{"session": "<session id>", "digest": "<64 hex>",
 "suites": {"boom": {"status": "ok", "rows": [{"item": "boom/…", "crps": 0.41, "mase": 1.2}]},
            "gift": {"status": "ok", "rows": […]}}}
```

`suites` has the same shape `cascade.eval.benchmarks.item_rows_from_report`
returns (including its optional `partial` flag). Suite keys are `gift`,
`boom`, `time` and `fev`. Extra suites and extra keys are ignored.

## `paired_gate(candidate_rows, king_rows, house_rows, *, suite, alpha, tau_public, B, seed, sentinel=None) -> dict`

### The paired LCB

The module must import `cascade.eval.expert_lane.paired_geomean_lcb`, the
bench gate's paired LCB, and use it. It must not re-implement it. The test
checks that `lane_d_bench.paired_geomean_lcb is expert_lane.paired_geomean_lcb`.

For each reference `ref` in `king_d` and `house`, and each metric `m` in
`crps` and `mase`, on the drawn `suite`:

- the items are paired by item id, so row order doesn't matter;
- `lcb[ref][m]` is the uncorrected one-sided `LCB_α` of
  `1 − geomean(candidate)/geomean(ref)`. That is
  `paired_geomean_lcb(ref_values, candidate_values, alpha=alpha, B=B, seed=…)`.
  The seed string is the builder's choice, but it must be a pure function of
  the arguments.

The test checks this against an independent `B = 40000` computation, to
within 0.002. A Bonferroni-corrected quantile falls outside that tolerance.

A host-wide multiplicative shift of every number (driver or SKU) leaves every
LCB unchanged to within 1e-9, and leaves the verdict unchanged.

### Outcomes

| `outcome` | when | `passed` | `hold` | other keys |
|---|---|---|---|---|
| `pass` | all four LCBs `≥ −tau_public` (inclusive) | `True` | `False` | `lcb`, `blocked_by == []` |
| `gate_regression` | any LCB `< −tau_public` | `False` | `False` | `lcb`, `blocked_by` = sorted refs with a failing LCB (`["house"]`, `["king_d"]` or `["house", "king_d"]`) |
| `checkpoint_fault` | the candidate's drawn-suite rows are faulty and both references' rows are clean | `False` | `False` | `fault_on` contains `"candidate"`, `reason` is a non-empty str |
| `bench_hold` | a reference's drawn-suite rows are faulty, whatever the candidate's rows are | `None` | `True` | `fault_on` contains every faulty ref (`"king_d"`, `"house"`) and never `"candidate"`, `reason` is a non-empty str |

- **Regression vs the house blocks even when KING_D passes,** and the other
  way round. A weak KING_D never lowers the bar.
- **`checkpoint_fault` is the miner's fault** (a loss). **`bench_hold` is
  infra,** and the candidate isn't judged. Both strings are members of
  `lane_d_report.SKIP_REASONS`.
- **A fault** on the drawn suite is any of these:
  - a NaN, inf or `None` `crps`/`mase`;
  - a row missing a metric key;
  - `partial: true`;
  - an item missing from that record that the other two records have.
- **The keys every outcome carries:** `suite`, `session`, `alpha`,
  `tau_public`, `digests` (`{"candidate", "king_d", "house"}` → digest) and
  `sentinel`.
- **The result is strict JSON.** `json.dumps(res, allow_nan=False)`
  round-trips it.
- **It's deterministic.** Two subprocesses with different `PYTHONHASHSEED`s
  give the same JSON.

### A house identical to KING_D

Passing the same record (or a deep copy) for both references works. The two
`lcb` entries are equal, and a regression lists both refs in `blocked_by`. A
faulty identical record holds on both.

### One session

All three records must carry the same non-empty `session` str.

- **Different sessions** raise a `ValueError` subclass whose `code ==
  "mixed_session"`, with `mixed_session` in its message. This applies even
  when the rows are byte-identical, for example stored KING_D rows from an
  earlier session.
- **A missing, empty or non-str session** raises `ValueError`/`TypeError`.

### The sentinel

`sentinel` (the era's §16 suite, or `None`) is swept and recorded, and never
gates.

- **No sentinel.** `res["sentinel"]` is `None`.
- **A sentinel is given.** `res["sentinel"]["suite"] == sentinel`. When its
  rows are clean, `res["sentinel"]["lcb"]` has the same `{ref: {metric}}`
  shape and the same paired-LCB definition.
- **It never changes the outcome.** A sentinel-only regression passes, and so
  does a faulty sentinel row set (in the candidate's or the house's records).
  A sentinel improvement can't rescue a gate regression.
- **Refusals.** `sentinel == suite` raises `ValueError`.

### Other refusals

These raise `ValueError`/`TypeError`:

- an unknown `suite`;
- `alpha` outside `(0, 0.5)`;
- a negative or NaN `tau_public`.

## `SweepSession(session_id, sweep, cache=None)`

This is one host session. `sweep(digests, suites)` is the bench host
(external). It returns `{digest: {suite: {"status", "rows", …}}}`.

- **`session_id`** is a non-empty str, otherwise `ValueError`/`TypeError`. It
  is exposed as `.session_id`.
- **`.paired(candidate, king_d, house, *, suites)`** returns the three sweep
  records `(candidate, king_d, house)`. Each is tagged with this session's id
  and holds at least the requested suites, with the host's rows unchanged.
  - Every digest that this session hasn't swept yet for all the requested
    suites is swept in **one** host call, deduplicated: a house equal to
    KING_D is swept once.
  - A fully cached call makes no host call and returns equal records.
- **Cached rows are re-used only within the session.** A later candidate in
  the same session re-uses KING_D's and the house's rows, and the host call
  holds only the new candidate. A new session id re-sweeps KING_D and the
  house, even when it shares the `cache` mapping with an earlier session.
- **A suite not yet swept** re-sweeps the cached digests for it.
- **Two sessions' records never pair.** `paired_gate` refuses them with
  `mixed_session`.

## Pod evidence (`test_w3b2_evidence.py`)

W3b2 runs in `ci (recorded suite outputs) + pod`. The pod part is one live
paired sweep of a candidate, KING_D and the house on one bench host, in one
session, on a suite drawn by W3b1. Its evidence is `evidence/w3b2.json`
(the default L4 evidence path), verified by `scripts/laned/accept W3b2
--evidence evidence/w3b2.json`.

```json
{"schema": "laned_w3b2/1", "wp": "W3b2", "source": "pod", "host": "<bench host>",
 "tool": {"git_sha": "<40 hex>", "git_dirty": false}, "session": "<session id>",
 "draw": {"block": 9612000, "block_hash": "0x<64 hex>"}, "suite": "boom", "sentinel": "gift",
 "params": {"alpha": 0.05, "tau_public": 0.01, "B": 10000, "seed": "<seed>"},
 "rows": {"candidate": <sweep record>, "king_d": <sweep record>, "house": <sweep record>},
 "gate": <paired_gate result>, "passed": true}
```

The verifier checks the following:

- the schema, the WP and `source == "pod"` (fixture evidence is never
  accepted), plus the host and a clean tool sha;
- every record carries the evidence's `session`, a 64-hex digest, and the
  drawn suite `ok` with at least 10 items;
- the sentinel was swept, and the candidate's digest differs from both
  references';
- `suite` is W3b1's `gate_suite_for(draw.block_hash, …)` under the shipped
  `chain.toml` with that sentinel;
- `gate` equals `paired_gate` recomputed from the recorded rows and params,
  so a hand-edited row or number fails;
- the outcome is a computed verdict (`pass` or `gate_regression`), and the
  sentinel has numbers.

The `test_evidence_*` tests skip when `LANED_EVIDENCE` is unset, and fail
hard when it's set and the file is bad. The `test_verifier_*` tests prove
offline that tampering is rejected: the gate, a row, the session, the
source, the suite, a fault or the WP.

## Not pinned

- **Wiring into `settle_lane_d`.** That covers which candidates are swept,
  how the boundary hash and the era's sentinel reach the sweep, and mapping
  `gate_regression`/`checkpoint_fault`/`bench_hold` into verdicts, report
  enums and `max_bench_holds`. W3b1's README assigns that wiring to W3b2, but
  it needs report schema changes (W7a's file), so these tests leave it out
  and `lane_d.py` isn't in `PATHS`. The orchestrator must assign it to a WP
  explicitly.
- **How an `error` status or an absent suite is classified.** W3b4 owns suite
  identity and partial classification, including FEV scope skips and
  data-revision checks.
- The exact seed string, the reason text, and extra result keys.
