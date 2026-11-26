# W3b2 acceptance tests: the paired gate sweep vs KING_D and the house on one host

W3b2 (plan §3.0 W3b2 row, §1.3b "Rule on the drawn suite" and "Faults", §4
T7b, T17) adds the lane D rotating gate's **paired** judgement on the suite
W3b1 draws. The rule: the candidate, KING_D and the house run on one host in
one sweep, never against stored absolute numbers. CRPS **and** MASE must each
satisfy `LCB_α ≥ −tau_public` against KING_D **and** against the current
house. The era's §16 sentinel suite is swept and recorded but never gates.

These tests are the spec. The builder may touch only the paths in `PATHS`:
`cascade/validator/lane_d_bench.py` (built, #574) and, since RV5, the new
owner verify module `cascade/laned/bench_evidence.py` (see "The live verify").

Some rules for the tests themselves:

- Nothing is mocked. The only stand-ins are external services: the bench
  host's sweep (`helpers.RecordingHost` returns recorded suite outputs and
  logs each call), and the chain, the checkpoint registry and the
  KING_D / house pointers for the live verify (`evidence_lib.FakeChain`,
  `FakeRegistry`, `FakePointers`, read-only, logging every call).
- No PATHS module is imported at module level, in a fixture or in
  `helpers.py` / `evidence_lib.py`. Every test imports it in its body.
- No test needs torch or a GPU, and every test runs locally with
  `--noconftest`.

History: before #574 every non-skipped test failed with a
`ModuleNotFoundError` for `cascade.validator.lane_d_bench`.

**RV5 tests-correction (BLOCKING item 6).** On main (lane_d_bench built,
`bench_evidence` absent) the unchanged files pass, and in the changed files:

- `test_w3b2_live_verify.py` fails, every test but
  `test_fakes_are_chain_client_shaped`, for a right reason: a
  `ModuleNotFoundError` for `cascade.laned.bench_evidence` in-process, and an
  `AssertionError` (exit 1, not the pinned refusal exit 2) for the CLI runs;
- `test_w3b2_evidence.py` passes on main except the 2 `test_evidence_*`
  skips. Its verifier lives in the tests (`evidence_lib.py`), and the old
  forgery-accepting check (`test_verifier_accepts_consistent_evidence`) is
  inverted into `test_verifier_refuses_self_consistent_synthetic_pod_evidence`
  and `test_verifier_refuses_rv5_helper_rows_relabelled_as_pod`;
- `test_w3b2_session.py` gains the host-omission tests (RV5 note 1), which
  pass on main.

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
- **A digest the host omits** (RV5 note 1) gets a record with no entry for
  the suite, and the gate classifies it as W3b4's `classify_suite` does: the
  candidate alone absent, with both references complete, is the miner's
  `checkpoint_fault` (code `absent`); an absent reference is a `bench_hold`
  whatever the candidate's rows are. So a host adapter must report a
  per-digest infra failure as a status (`timeout`, or `error` with detail),
  never by leaving the digest out.

## Pod evidence (`test_w3b2_evidence.py`, `evidence_lib.py`)

W3b2 runs in `ci (recorded suite outputs) + pod`. The pod part is one live
paired sweep of a candidate, KING_D and the house on one bench host, in one
session, on a suite drawn by W3b1. Its evidence is `evidence/w3b2.json`
(the default L4 evidence path), verified by `scripts/laned/accept W3b2
--evidence evidence/w3b2.json`.

**RV5 (BLOCKING item 6).** Under schema `laned_w3b2/1` the verifier checked
only internal consistency: synthetic rows labelled `source: "pod"`, any block
hash and `gate = paired_gate(...)` reached L4 with no pod run. Schema
`laned_w3b2/2` binds the evidence to facts an offline writer can't choose,
and the old fixture that showed a forgery passing is inverted: a
self-consistent synthetic body (`evidence_lib.synthetic_body`, real pinned
item ids and all) is **refused**.

```json
{"schema": "laned_w3b2/2", "wp": "W3b2", "source": "pod", "host": "<bench host>",
 "tool": {"git_sha": "<40 hex>", "git_dirty": false}, "session": "<session id>",
 "draw": {"network": "finney", "netuid": 91, "block": 9612000, "block_hash": "0x<64 hex>"},
 "suite": "gift", "sentinel": "time",
 "pin": {"suite": "gift", "version": "<pinned revision>", "items_sha256": "<64 hex>",
         "n_items": 97, "digest": "<64 hex>"},
 "params": {"alpha": 0.05, "tau_public": 0.01, "B": 10000, "seed": "<seed>"},
 "refs": {"candidate": "<repo>@sha256:<64 hex>", "king_d": "…", "house": "…"},
 "rows": {"candidate": <sweep record>, "king_d": <sweep record>, "house": <sweep record>},
 "gate": <paired_gate result, with pinned_identity = pin>, "passed": true,
 "verified_live": <the owner's verify record, below>}
```

### (a) The suite identity is the repo pin

The repo pins a suite by its data revision,
`benchmarks/cascade_benchmark/datasets.py` `DATASETS[<sidecar>].revision`
(read by parsing the `DatasetSpec(...)` calls, not by importing it), and its
item set, the keys of the suite's Seasonal-Naive baseline table
`benchmarks/cascade_benchmark/data/gifteval_seasonal_naive.json` (`gift`, 97
items) or `boom_seasonal_naive.json` (`boom`). `evidence_lib.repo_pin(suite)`
builds W3b4's identity from those two (`test_repo_pin_is_w3b4_suite_identity_of_the_pinned_items`
checks it equals `cascade.eval.suite_identity.suite_identity`).

- `pin` must equal `repo_pin(suite)`.
- Every role's drawn-suite entry carries `data_revision` equal to the pinned
  revision, and its item ids are exactly the pinned set, no duplicates.
- `gate` is recomputed with `pinned_identity = pin`. Every role's
  `gate.identity` digest is the pin's digest.
- **TIME and FEV have no repo item set** (and FEV no repo revision), so
  evidence whose drawn suite is `time` or `fev` is refused ("no repo pin").
  The owner runs on a block whose draw is `gift` or `boom`, or a pin lands
  first. The sentinel suite isn't checked against a pin (W3b4).

### (b) The checkpoints are published refs

`refs.<role>` is `<repo>@sha256:<64 hex>` and its hex is
`rows.<role>.digest`. The candidate's digest differs from both references'
(KING_D may equal the house: while lane D is unarmed the house holds KING_D).
Offline that's the format. The owner's verify resolves every ref in the
checkpoint registry and checks `refs.king_d` / `refs.house` against the
KING_D and house pointers in force at `draw.block`.

### (c) The block, the draw and the params

Offline: `draw.network == "finney"`, `draw.netuid` is chain.toml's
`[subnet] netuid`, `draw.block` a positive int, `draw.block_hash` `0x` + 64
hex, and `suite` is `gate_suite_for(draw.block_hash, …)` under the shipped
`chain.toml` with the recorded sentinel. `params.alpha`, `tau_public` and `B`
equal `[lane_d] alpha`, `tau_public` and `bootstrap_B`. Whether the hash is
the chain's is only checkable live; that's the `verified_live` record.

The rest of the `/1` checks stay: schema, WP, `source == "pod"`, host, a
clean tool (`git_sha` a commit here containing
`cascade/validator/lane_d_bench.py`, `git_dirty: false`), one session on
every record, the drawn suite `ok` with ≥ 10 items, the sentinel swept to
numbers, `gate` equal to the recomputation, and a computed verdict (`pass`
or `gate_regression`).

### (d) What is refused offline

`test_verifier_rejects_tampering` covers, each refused by the body checks
alone and again after a record is re-bound to the edited body: the gate, a
row, the session, the source, the suite, a fault, the WP, made-up item ids,
the same items dropped from every record, another revision, an edited pin,
no pin, a bare digest as ref, a ref to another digest, no refs, a malformed
hash, testnet, another netuid, and `alpha` / `tau_public` / `B` other than
chain.toml's. `test_verifier_refuses_self_consistent_synthetic_pod_evidence`
refuses a fully consistent body with no record, and
`test_verifier_refuses_rv5_helper_rows_relabelled_as_pod` refuses RV5's
literal forgery even with a record forged for it.

## The live verify (`test_w3b2_live_verify.py`; module `cascade/laned/bench_evidence.py`)

**Owner command** (from a clean checkout of main, after the pod run wrote
`evidence/w3b2.json`; read-only chain access, no wallet, no keys):

```
python -P -m cascade.laned.bench_evidence verify --evidence evidence/w3b2.json
```

It re-reads, writes `verified_live` into the evidence (replacing an older
one) and exits 0. On a refusal it exits 2 with the code on stderr and leaves
the file byte-identical. A missing or non-JSON file exits 2 naming the file /
saying "evidence". `not_pod` and `wrong_network` refusals come before any
network use (the CLI tests run with every socket connect refused and
bittensor/boto3 unimportable). The CLI builds the three adapters below from
the deployed config, read-only (`ChainClient` reads only: no `set_weights`,
no commitment, no wallet).

**In-process:** `bench_evidence.verify_live(evidence, *, chain, registry,
pointers, now=None) -> dict` returns the record, never mutates `evidence`,
and raises `bench_evidence.VerifyRefused` (`.code`). Importing the module
must not import bittensor or boto3.

| adapter | the only reads it may use |
|---|---|
| `chain` | `ChainClient`-shaped: `finalized_block() -> int`, `block_hash(block) -> "0x…"` (raises for an unreadable block) |
| `registry` | `resolve(ref) -> "sha256:<hex>"` of the manifest the ref pins, or `None` when the registry doesn't hold it |
| `pointers` | `at_block(block) -> {"house": ref, "king_d": ref}`: lane A's king checkpoint (the house, §1.4) and KING_D in force at `block` (the house while lane D is unarmed) |

Any other attribute use on an adapter is a failure (`forbidden` in the fakes).

| code | when |
|---|---|
| `not_pod` | `source != "pod"`; before any adapter call |
| `wrong_network` | `draw.network != "finney"` or `draw.netuid` isn't chain.toml's; before any call |
| `malformed` | `draw.block` not an int, `draw.block_hash` not `0x` + 64 hex, `refs` missing, or a ref that isn't `<repo>@sha256:<hex>` of its record's digest; before any call |
| `block_not_finalized` | `draw.block > chain.finalized_block()` |
| `block_hash_mismatch` | the chain's hash at `draw.block` differs from `draw.block_hash`, or can't be read |
| `draw_mismatch` | `gate_suite_for(<the chain's hash>, …)` under the shipped chain.toml, with the evidence's sentinel, isn't `suite` |
| `pointer_mismatch` | `pointers.at_block(draw.block)` differs from `refs.king_d` or `refs.house` |
| `ref_unpublished` | `registry.resolve(refs.<role>)` isn't `sha256:<rows.<role>.digest>`, for any role |

The record (schema `laned_w3b2_verify/1`), checked offline by
`evidence_lib.verified_live_problems`:

```json
{"schema": "laned_w3b2_verify/1", "verified": true, "body_sha256": "<64 hex>",
 "network": "finney", "netuid": 91, "block": 9612000, "block_hash": "0x…",
 "finalized_block": 9612005, "suite": "gift", "sentinel": "time",
 "refs": {"candidate": "…", "king_d": "…", "house": "…"},
 "pointers": {"house": "…", "king_d": "…"},
 "resolved": {"candidate": "sha256:…", "king_d": "sha256:…", "house": "sha256:…"},
 "verified_at": "<ISO 8601 with a timezone>",
 "tool": {"module": "cascade.laned.bench_evidence", "git_sha": "<40 hex>", "git_dirty": false}}
```

- `body_sha256` is W3a's canonical form: the evidence without
  `verified_live`, `json.dumps(sort_keys=True, separators=(",", ":"),
  ensure_ascii=True)`, UTF-8. Any edit after verification fails.
- `network`, `netuid`, `block`, `block_hash` equal `draw`'s;
  `finalized_block ≥ block`; `suite` / `sentinel` equal the evidence's and
  `suite` is the draw of the record's hash; `refs` equal the evidence's;
  `pointers` equal `refs.king_d` / `refs.house`; `resolved.<role>` is
  `sha256:<rows.<role>.digest>`; `verified_at` is tz-aware and not in the
  future; `tool.module` is this module, and at L4 `git_sha` is a commit here
  containing `cascade/laned/bench_evidence.py` with `git_dirty: false`.

**Residual (as W3a, RV5 note 9).** The record is hash-bound, not signed.
Everything it attests (the hash at a block, the pointers at that block, the
registry digests) is public, so anyone can re-run `verify` read-only and
detect a hand-written record; that's stronger than W3a, whose locks expire.
What stays owner-attested is that the numbers came from a real sweep: a
writer with the real pinned item ids, real public refs and a real block can
make up rows and still pass `verify`. Closing that needs a signed bench-host
attestation or a reproduction by a satellite, which no WP pins yet.

The `test_evidence_*` tests skip when `LANED_EVIDENCE` is unset, and fail
hard when it's set and the file is bad (`evidence_problems(ev)` with git
checks and the record required).

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
- **The pod run that writes `evidence/w3b2.json`** (how the sweep is driven
  on the host). Only its output is pinned.
- **R12 on-rent evidence for `host`** (RV5 asked for it). R12's evidence
  schema isn't on main, so there's nothing to bind to yet. When R12 reaches
  L4, a tests-correction PR can require `host` to name an R12 evidence file.
- **How the CLI builds its adapters** (which config keys name the registry,
  and how lane A's king pointer and KING_D at a block are read). Only the
  adapter reads and the refusal codes are pinned.
- **The era's sentinel.** It's recorded, not re-derived from the era
  boundary.
