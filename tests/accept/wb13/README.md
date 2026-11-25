# WB13 acceptance tests: the intake contamination check (LB1, D38)

WB13 is in the plan §3.0 lane B track. Its rules come from §12b.1 LB1, D38
(decided: "(a) 8 MB `suite_target_max_repo_mb`") and the T-rows TB10–TB12.
It matches a submitted lane B generator against the WB11 hash set at intake,
before any compute. These tests are the spec. The builder may touch only the
paths in `PATHS`.

The WB13 row, verbatim:

> | WB13 | Intake contamination check (LB1): AST literals, `config.json`, non-code bytes vs the hash set; typed `benchmark_series_embedded`; `suite_target_max_repo_mb` (D38) | receipt field `contamination: {hashset_version, result, suite, window_id}` | WB11 | ci | `scripts/laned/accept WB13` (TB10–TB12) |

TB10–TB12 (§4), verbatim:

> | TB10 | **(rev4)** Generator embedding an exact GIFT/BOOM/TIME/FEV test window as a literal | Refused at intake, `benchmark_series_embedded`, before any compute |
> | TB11 | **(rev4)** Same window scaled ×3.7, offset +100 and rounded to 3 dp, in `config.json` | Refused (normalised match) |
> | TB12 | **(rev4)** A clean generator; and a hash-set version change | Passes; the receipt records the hash-set version used |

## Tests correction (RV6 B6, B7)

RV6 found two gaps in WB13 as built: one crafted file could stall the live
intake for hours (B6), and trivial packaging of an exact benchmark copy
evaded the scan (B7). `test_wb13_rv6.py` is new, and section "RV6" below
is its contract. Its tests fail on main with an `AssertionError`, pending an
implementation PR. A killed child process counts as a failure, never a hung
test. Three cases already pass on main and guard behaviour that must stay:
`test_npz_inflation_is_capped`, `test_float64_at_any_byte_offset[1]` (main
catches it by accident, through the int64 decoding) and
`test_benign_text_and_binary_stay_clean`.

The flat `[3] * 40` series is low-information under WB11's RV6 B5 rule
(tests/accept/wb11/README.md "Low-information windows"), so it stops being
evidence. `test_int_bytes`, `test_scan_is_deterministic` and
`test_fixture_windows_are_in_the_hash_set` now pin the int decodings on a
new informative integer series, `time/counts/5`, with the same assertions.
They pass on main and after WB11's B5 implementation. Every other existing
assertion is unchanged.

## Tests correction (RV7 B3, note 9)

RV7 B3 (RV6 B8 still unwired) found that the intake loads the pinned hash
set as the full dict, and parses every `hashset-v<n>` file whole until one
matches. WB11's bounded `load_lookup` had no production caller. RV7 note 9
found that a load-driven `budget_exceeded` stuck in the result cache.
`test_wb13_rv7.py` is new, and section "RV7" below is its contract.

- **Fail on main** with an `AssertionError`:
  - `test_pinned_hashset_is_the_bounded_lookup`;
  - `test_only_the_pinned_file_is_loaded_and_only_through_load_lookup`;
  - `test_loader_and_main_use_load_lookup_never_load_artefact`;
  - `test_large_pinned_hashset_loads_within_the_lookup_rss_bound`;
  - `test_deadline_budget_exceeded_is_not_cached`.
- **Pass on main** and guard behaviour that must stay: the tampered or
  missing pin, the intake on the loaded hash set (clean, embedded, another
  pin), transient fetch and scan errors, and the cached window-budget
  overrun.

One existing assertion changed. `test_load_pinned_hashset` compared the
loaded hash set to the dict with `==`. It now compares the header keys and
every hash. Every other assertion in it is unchanged.

## How the tests fail on main

Every failure on main is one of these, inside a test body:

- a `ModuleNotFoundError` for `cascade.interface.contamination_scan`;
- an `AssertionError`: `FundingIntake takes lane_b_hashset` (the intake
  helper checks the signature first), or the `main.py` wiring check.

The four tests in `test_wb13_plan.py` pass on main. They pin the plan text
and check that the WB11 hash set (on main) sees every fixture embedding, so a
WB13 failure is only about WB13.

## Fixtures

- **Nothing is mocked except the outside world.** The intake is the real
  `FundingIntake` with a real `LaneBQueue` file and a real `PayerKeyVault`.
  The config comes from the real `parse_lane_b` (WB10). The hash set is
  built by the real WB11 `build_hashset` from four tiny seeded suites
  (`helpers.fixture_suites`). The stand-ins are the reveal oracle, the
  signature verifier and the repo fetch (`helpers.Fetcher`, which returns a
  `tmp_path` tree and records its calls).
- **Miner code is never run.** The fixture repos are plain files. The tests
  never import or execute them, and neither may the scanner.
  `test_submitted_code_is_never_executed` plants top-level code that writes a
  marker file, and checks that the marker never appears and that no fixture
  module enters `sys.modules`.
- **No torch.** An autouse guard makes any torch import raise.
- **The embedded window** is `gift/electricity/7[20:60]`: 40 points, which
  is 9 sub-windows of 32. Its first window id is `gift/electricity/7@20`.

## The scanner: `cascade/interface/contamination_scan.py`

```python
def scan_repo(repo_dir, artefact, *, min_hits=None) -> dict
def load_pinned_hashset(pin: str, root) -> dict | None
```

### `scan_repo`

It reads the tree **without executing it** and returns exactly the receipt
field:

```json
{"hashset_version": 1, "result": "clean" | "embedded", "suite": "gift" | null, "window_id": "gift/electricity/7@20" | null}
```

- `hashset_version` is the artefact's `version` (the `n` of
  `hashset-v<n>`). TB12: a v2 artefact gives 2.
- `result` is `"embedded"` when any sequence extracted from the tree meets
  the WB10 rule: `exact_hits ≥ min_hits.exact` **or** `normalised_hits ≥
  min_hits.normalised`. That's WB11's `match_series` / `is_contaminated`.
  `min_hits` is anything `dict()` accepts. `LaneBConfig.contamination_min_hits`
  (pairs) is the one the intake passes. `None` means the WB10 default
  `{exact: 1, normalised: 3}`.
- `suite` and `window_id` are the WB11 lookup's `first` hit of the sequence
  that matched. On a clean tree both are `null`.
- The artefact is not mutated.

**What it extracts.** These are LB1's three sources. "Matched against
everything in the submission":

| source | pinned by |
|---|---|
| `.py` files, everywhere in the tree (subpackages too), by AST: every list / tuple literal, and **all number constants of a file in source order** (LB1 "all number constants and list or array literals"). So a window spread over 40 scalar assignments, or split across nested lists, is caught. A unary minus is part of the number. | `test_tb10_*` (list, tuple, `np.asarray([...])`), `test_literal_in_a_sibling_module_is_found`, `test_all_number_constants_in_source_order`, `test_shifted_negative_literals_hit_normalised` |
| `config.json`: numeric arrays anywhere in the document, nested inside objects and lists ("every value in `config.json`") | `test_exact_window_nested_deep_in_config_json`, `test_tb11_*` |
| non-code bytes, any other file, decoded at offset 0 as little-endian float64, float32, int64 and int32 arrays (LB1 "decoded as float32/float64/int arrays"), plus the numbers in a one-column text table | `test_float64_bytes`, `test_float32_bytes`, `test_int_bytes[<40q / <40i]`, `test_numbers_in_a_text_data_file` |

The text-table row is a test-author reading of "any non-code bytes". A CSV of
decimal reprs is the plainest way to ship a series, and the binary decodings
can't see it. Fewer than 32 points never hit.

**TB10:** the exact window as a `.py` literal is `embedded`, suite `gift`,
window `gift/electricity/7@20`. **TB11:** `round(3.7·x + 100, 3)` of the
same window in `config.json` has no exact hit and is `embedded` through the
normalised set.

**Determinism.** The same tree gives the same receipt whatever order its
files were written in and under `PYTHONHASHSEED` 0, 1 and 4242. The hash
seed check runs `python -P -m tests.accept.wb13.helpers <tree>` in fresh
interpreters. The fixture has two different embeddings (a gift literal and
the flat TIME series as int64 bytes). Which one the receipt names is the
builder's choice, but the choice must be deterministic.

### `load_pinned_hashset(pin, root)`

This is the production path from the `[lane_b] contamination_hashset` pin
to an artefact.

- `""` (unpinned) gives `None`.
- Otherwise it returns the `benchmarks/contamination/hashset-v<n>.json`
  under `root` whose WB11 `artefact_sha256` equals `pin`.
- No match raises `ValueError`. That includes a file edited after pinning
  and a `root` with no artefacts. It never falls back to another version.
- Since RV7 it returns WB11's bounded lookup, not the dict. See "RV7" below.

## The wire point: `POST /v1/lane-b/fund` (`FundingIntake.lane_b_fund`)

`FundingIntake` gains two keywords:

- **`lane_b_hashset`**, the loaded artefact (or `None`);
- **`fetch_lane_b_repo(ref)`**, which returns the local tree of the committed
  `X-Commit-Ref` or `None`. It may raise.

`cascade/funding/main.py` passes both to `FundingIntake`, and it loads the
hash set through `load_pinned_hashset`. Both are checked by AST. How the
fetch materialises a vault or Hub ref (for example `SubmissionStore.extract`)
is the builder's choice, but it must never execute the tree.

**Armed** means `lane_b_cfg.contamination_hashset` is a pin. The check then
runs on **every** lane B fund, both domain and `suite:<s>` targets (TB10
names no target), in this order:

1. **The identity gate comes first.** That's targets, signature, timestamp,
   registration, key and the revealed commitment. A bad signature (`400
   bad_signature`) or an unrevealed commitment (`403 not_revealed`) never
   calls the fetch.
2. **Fail closed while pinned.** If `lane_b_hashset` is `None`, or its
   `artefact_sha256` isn't the pin (an artefact of another version), the
   reply is `503`. It is also `503` when the fetch is missing, returns
   `None` or raises. Nothing is queued or vaulted. The code isn't pinned,
   but it is never `benchmark_series_embedded`.
3. **D38.** A suite-target entry (any `suite:*` target) whose fetched tree
   totals more than `suite_target_max_repo_mb` MiB (1 MiB = 1048576 bytes,
   as `check_repo_size`) is refused with code `repo_too_large`, status 400 or
   413. Exactly the cap passes. The cap is read from the config
   (`dataclasses.replace` to 2 MiB admits the same tree), and it doesn't
   apply to domain entries ("for lane B suite-target entries"; they keep
   `[generator] max_repo_mb`). The plan names no code for D38, so
   `repo_too_large`, the existing `check_repo_size` reason, is the test
   author's choice.
4. **The scan.** `scan_repo(tree, lane_b_hashset, min_hits=lane_b_cfg.contamination_min_hits)`.
   - **`embedded`** gives `400 {"code": "benchmark_series_embedded",
     "message": …, "contamination": <receipt>}`. Nothing is queued, no key
     is vaulted and nothing is charged: "before any compute".
   - **`clean`** gives `202`, and the reply carries `"contamination":
     <receipt>`. The queue entry carries it as
     `LaneBQueue.get(hotkey).contamination`, which survives a reload of the
     queue file.
   - The hit rule comes from the config. With `normalised = 1000`, TB11 is
     clean.
5. **The signed message doesn't change.** The check adds no request field.

**Unarmed** is the shipped `contamination_hashset = ""`. The fund is today's
byte for byte (TB22: "Byte-identical digests, receipts and weights"), even
for a tree that embeds a test window: the fetch is never called, and the
reply and the queue file never contain `contamination`.

### What "no hash set pinned" means (the rule, quoted)

- §12b.1: "`[lane_b] suite_targets` may arm **only after** the static
  contamination check ships **and passes on testnet** (stage SB3). It's a
  gate, not an open item."
- WB10's loader already enforces it: `parse_lane_b` refuses
  `suite_targets = true` without a pin ("needs contamination_hashset pinned
  (the contamination check gates every suite-target entry)").

So no suite-target entry can exist unpinned. A domain entry is unchecked
(the unarmed invariant), and nothing is recorded. **`result` is only ever
`clean`, `embedded` or (RV6 B6) `budget_exceeded`.** There is no "skipped" value. Pinned means checked or
refused (fail closed, above). Unpinned records nothing, so that TB22's
byte-identity holds.

## Invariants (1, paired per invariant 10)

| test | pins |
|---|---|
| `test_inv1_lane_b_fund_unchanged_when_unarmed` | at the default `""`, even an embedding tree gets `202 {"status": "queued", "lane": "b", "targets": ["energy"]}` exactly. The signed message is today's `canonical_fund_message`, there is no fetch, the entry's `contamination` is `None` and the queue file has no `contamination` |
| `test_inv1_lane_b_fund_armed_nonempty_lane_d_effect` | pinned: one fetch of `X-Commit-Ref`, `202` with the clean receipt on the reply, the entry and the reloaded file, and an unchanged signed message |

## TB11 and series scale: a WB11 finding

The WB11 schema-1 normalised hash z-scores a window and keeps **4
significant figures** of each z (`format(z, ".3e")`). The precision is
relative, so a z near 0 keeps very fine digits. TB11's rounding to 3 dp is
an **absolute** error of up to 5e-4, which is about `5e-4 / (3.7·σ)` in z.
On a series with a small σ, that flips the digits of some near-zero z in
almost every window. On the WB11-style fixture series, the share of TB11
windows that still hit is:

| series (level, step) | TB11 windows that hit |
|---|---|
| 5000, 40 (`gift/electricity/7`) | 24 / 41 |
| 1e6, 5e3 | 41 / 41 |
| 300, 9 | 2–20 / 25–41 |
| 100, 3; 0.4, 0.02; 7, 0.3 | **0** |

float32 bytes behave the other way round. They keep about 7 *relative*
digits, so they hit at level 100 (14 / 17) but rarely at level 5000 (1 / 9).

So WB13 pins TB11 on `gift/electricity/7`, where the WB11 hash can catch it,
and the float32 case on `gift/m4_hourly/0`. `test_fixture_tb11_margin`
records the margin. **A real GIFT/BOOM/TIME/FEV window with a small σ,
scaled ×3.7, offset +100 and rounded to 3 dp, would pass the check as
shipped.** Closing that needs a new WB11 hash schema, for example
fixed-point z rounding or a tolerant match. That is out of WB13's PATHS and is
reported for the plan owner. Guard 4 and the anchor guard remain the
backstop (§12b.1, §9).

## RV6: scan budget (B6) and packaging evasions (B7)

### Scan budget (`test_wb13_rv6.py`, B6)

**Cost, measured on main.** WB11's tolerant lookup tries up to 2 × 2¹²
probes for a window whose z-scores all sit within `TOLERANCE` of a bin edge.

- **The crafted window.** `helpers.EDGE_BLOCK` has 32 values with mean 0 and
  std 1, and every z is a bin edge. Repeating it with period 32 makes every
  sliding window crafted.
- **Per window:** ~21 ms crafted against ~0.19 ms random, measured on
  small inputs (64 and 2 000 windows).
- **Extrapolated:** 1 MiB of crafted f64 is ~131k windows, ~2 700 s of CPU.
  The 8 MiB D38 cap is ~6 h. All of it ran in the `ThreadingHTTPServer`
  handler thread.

The tests never build large files. They take 1 024 crafted windows (~21 s
on main), run them in a niced child process and kill it on overrun.

**Pinned (scanner):**

- `scan_repo(repo_dir, artefact, *, min_hits=None, max_windows=None, deadline_s=None)`.
- `RESULT_BUDGET_EXCEEDED = "budget_exceeded"`. Its receipt is
  `{hashset_version, "result": "budget_exceeded", "suite": null, "window_id": null}`.
- **`max_windows`** is the most sub-windows matched in one scan, counted over
  every sequence and decoding. Going past it gives `budget_exceeded`. `None`
  means the module constant `MAX_SCAN_WINDOWS` (an int ≥ 1), **read when the
  scan runs**, so the tests can monkeypatch it.
- **`deadline_s`** is the seconds a scan may run. Going past it gives
  `budget_exceeded`. `None` means `SCAN_DEADLINE_S`, which is > 0 and ≤ 30,
  also read at scan time. On the crafted file with `deadline_s = 0.5`, the
  scan returns within 3.5 s (wall clock, in the child). That needs a check
  at least every few windows, not every 65 536-value chunk.
- If the scan finishes in time (for example a cache of repeated windows),
  `clean` is the correct answer and also passes. The tests pin the time
  bound and the result's correctness, not a slow path.

**Pinned (intake):**

- **`budget_exceeded` fails closed** as `413 {"code":
  "contamination_budget_exceeded", "message": …, "contamination": <receipt>}`.
- **Uncharged:** nothing is queued, no key is vaulted and no lane A
  submission is consumed, as with every refusal before the queue.
- **Why 413, not 503.** The window budget is a property of the tree, so a
  retry can't succeed. The miner has to ship less bulk data, and DEC-CA-0028
  says there is "no longer any excuse for numbers in `config.json`". A deadline
  overrun gets the same code, so a miner can't probe load.
- **Bounded handler time.** With `SCAN_DEADLINE_S = 1`, `lane_b_fund` on the
  crafted tree returns within 5 s: either that 413, or a correct 202 clean.
  The scan may run off the request thread or under a deadline. Only the
  return time is pinned.
- **Result cache.** Funding the same ref twice under the same pin calls
  `contamination_scan.scan_repo` once, and both replies carry the same receipt
  (clean: 202, then 200 `already-queued`; embedded: 400 twice). The cache key
  must be content-addressed: the ref (vault and Hub refs are content-pinned)
  plus the hash-set pin, per `FundingIntake`, or the tree's content hash. A
  cache shared across intakes by ref alone would leak between tests and
  hosts. Since RV7 note 9, a window-budget `budget_exceeded` is cached and a
  deadline-driven one is not (see "RV7").

### Packaging evasions (`test_wb13_rv6.py`, B7)

Each of these exact or float32 copies is a one-line change for the miner, and
each must now be `embedded` with the TB10 receipt (gift,
`gift/electricity/7@20`, or the WB11 first hit for float32):

| form | how the scan reads it |
|---|---|
| a `.py` **str** constant of numbers, comma- or space-separated | numeric tokens of the string |
| a `.py` **bytes** constant: number text, or raw f64 bytes (`np.frombuffer`) | as text, and as binary like a data file |
| an **`.npz`**, `np.savez` (stored) or `np.savez_compressed` (deflated), `<f8` or `>f8` members | zip members inflated, `.npy` headers parsed (descr, shape), data decoded |
| raw f64 at **byte offset 1–7** of a blob | every alignment |
| **big-endian** f64, and big-endian f32 of `gift/m4_hourly/0` | both endiannesses |
| a **one-line** CSV or whitespace row (`,` ` ` tab `;` `, `), after a text label | split on any delimiter; a non-numeric token breaks the run |

**Safety, pinned.**

- **No unpickling.** An object-dtype (`|O`) member whose pickle would write a
  marker file is never unpickled. The scan still finds the f64 member in the
  same `.npz`. Parsing never executes.
- **Inflation cap.** An `.npz` member that inflates to 64 MiB of zeros
  (< 1 MiB on disk) returns `clean` or `budget_exceeded`, with the scanning
  process at ≤ 512 MiB peak RSS, within the default deadline. Size and ratio
  caps on inflation are the builder's choice.
- **No new false positives.** The new decodings add none on a README with
  numbers, a small CSV of unrelated values, or 1 KiB of random bytes.

**Residual limits.** Record these in the decision register with WB11's.
Guard 4 is the backstop for them, as for procedural regeneration:

- delta encoding;
- interleaving with zeros or another series;
- base64 or hex text;
- an outlier every 31 points;
- noise above ~0.03 σ;
- container formats other than zip/`.npy`.

## RV7: the bounded hash-set load (B3) and transient refusals (note 9)

### The pinned hash set is loaded bounded (`test_wb13_rv7.py`, B3)

**Pinned (`load_pinned_hashset(pin, root)`):**

- **The return is bounded.** It's a read-only `Mapping`: WB11
  `ContaminationLookup`, or a wrapper around one (to carry the pin, say). It
  is never a `dict`, and its `exact` and `normalised` tables aren't `dict`s.
- **It answers like the artefact.** The header keys (`schema`, `version`,
  `window`, `stride`, `suites`) are equal. Every hash `.get`s the same
  `[suite, window_id]`. WB11 `match_series` gives the same result on every
  fixture embedding.
- **It loads one file, and only through `load_lookup`.** WB11
  `load_lookup(<the pinned file>, expect_sha256=pin)` is called exactly
  once, and WB11 `load_artefact` is never called. The fixture publishes v1,
  v2 (pinned, `.json.gz`), v3 and a junk `hashset-v4.json`.
- **How it finds the file.** The pin is the sha256 of the canonical JSON,
  and `write_artefact` writes exactly those bytes (gzipped for `.gz`). So a
  streaming sha256 of each candidate's raw (decompressed) bytes finds the
  file with O(1) memory and no parse. The method isn't pinned. Only the
  parser calls are pinned.
- **AST check.** The body of `load_pinned_hashset` names `load_lookup` and
  not `load_artefact`. `cascade/funding/main.py` names `load_pinned_hashset`
  and never `load_artefact`. So the claim in
  `cascade/eval/contamination.py` is true: "the intake check loads the
  pinned file with load_lookup".
- **Fail closed, unchanged.** A file edited after pinning, or a `root`
  without artefacts, raises `ValueError`.

**Memory (measured in a fresh child).**

- **The fixture.** WB11's 50 000-window `synth_suites`, published as
  `hashset-v1.json.gz` (a decoy) and `hashset-v2.json.gz` (pinned). Each is
  about 4 MB on disk, and the test refuses anything over 8 MiB.
- **The bound.** Growth is the child's `VmHWM` after the load minus its
  `VmRSS` just before. The bound is WB11's:
  2 × `LOOKUP_PEAK_BYTES_PER_WINDOW` × n + 16 MiB, about 22 MiB here.
- **Measured on main's WB11 functions:**
  - `load_lookup` grows about 3.5 MB;
  - `load_artefact` about 66 MB.
- **The control.** The test checks that the full load breaks the bound, so
  the measurement can tell the two loads apart.

**Pinned (intake).**

- `FundingIntake` takes what `load_pinned_hashset` returns. A clean tree
  gets 202 with the clean receipt. TB10 gets 400 `benchmark_series_embedded`
  with its receipt, and nothing is queued.
- A lookup of another pin answers 503 and queues nothing. So the intake has
  to know the loaded lookup's pin. `ContaminationLookup` carries no sha, so
  the loader has to attach one, or the intake has to get the pin some other
  way, never by serialising the lookup.

### A refusal caused only by load is not cached (note 9)

**Not cached:**

- a fetch that raises or returns `None` (503);
- a `scan_repo` that raises (503);
- a `budget_exceeded` from the scan deadline (`SCAN_DEADLINE_S`
  monkeypatched to 1 ns, 413).

In each case, a re-fund after the cause clears is scanned again and gets 202
with the clean receipt. The deadline test sees exactly two `scan_repo` calls.

**Cached:** a content-determined result is cached as before. A window-budget
`budget_exceeded` (`MAX_SCAN_WINDOWS` monkeypatched to 8, on a tree of 33
windows) gives 413 twice, from one `scan_repo` call.

**Not pinned:**

- How the intake tells a deadline overrun from a window-budget overrun (a
  flag, an exception, an extra return value). The receipt stays
  `{hashset_version, "result": "budget_exceeded", "suite": null,
  "window_id": null}` either way, and so does the 413.
- Whether `clean` and `embedded` share the cache with the window-budget
  result.

## Not pinned

- The order of steps 2 to 4 among themselves.
- The 503 codes and every message text.
- Which hit the receipt names when a tree has several embeddings (it must be
  deterministic).
- Whether hits add up across files or sequences. Every test has one
  embedding with a clear margin.
- Scalar values spread over `config.json` keys and other integer widths.
  (Numbers inside str / bytes constants and big-endian and unaligned
  decodings are pinned since RV6; see "RV6" below.)
- Exclusions such as `__pycache__/` and `.git/`, and symlink handling. The
  scan must not read outside the tree it is given.
- Propagating the field from the queue entry into the trainer's leg record
  and manifest. That is downstream of intake. WB14's `declared_rows` set the
  same scope.
- The miner CLI (`cascade verify`) running the same scan.

## Files

| file | pins |
|---|---|
| `PATHS` | the modules WB13 may change |
| `helpers.py` | the suite fixture, the WB11 artefact, the config through `parse_lane_b`, fixture repos as plain files, the real intake builder, the recording fetch, the torch guard, the hash-seed entry point; since RV7: `publish`, `Spy`, `FlakyFetcher` and the 50 000-window build / load children |
| `test_wb13_plan.py` | the WB13 / TB10–TB12 / LB1 / D38 plan text verbatim, the shipped 8 MB cap, fixture sanity against WB11 (passes on main) |
| `test_wb13_scan.py` | `scan_repo`: the three sources, TB10, TB11, TB12 versions, the hit rule, never executing, determinism; `load_pinned_hashset` |
| `test_wb13_rv6.py` | RV6 B6 scan budget (`max_windows`, `deadline_s`, `budget_exceeded`, 413, bounded handler time, result cache) and B7 packaging evasions (str/bytes literals, `.npz`, offsets, big-endian, one-line rows; never unpickled; inflation cap) |
| `test_wb13_rv7.py` | RV7 B3: `load_pinned_hashset` returns the bounded lookup, built by one `load_lookup(path, expect_sha256=pin)` and never `load_artefact`, within WB11's RSS bound in a child; the intake on it; note 9: transient refusals are not cached, window-budget overruns are |
| `test_wb13_intake.py` | `main.py` wiring, the invariant pair, TB10 and TB11 refused at intake, TB12 version change, fail closed, the identity gate before the fetch, D38 |
| `test_wb13_overlay_lanes.py` | launch audit 2026-11-24: `--lane-b/--lane-c auto` follow the adopted DEC-CA-0057 overlay (`main.EffectiveLanes`, read-only `NetConfigResolver`): an overlay arming lane B (or C) at block N answers 503 before N and opens at/after N with no restart; per-poll rate limit; a chain read failure keeps the lanes; no overlay / no owner key = the static wiring; `on` / `off` unchanged |
| `test_wb13_lane_ended.py` | #931 kill switches at the intake (`cascade.funding.lane_ends`): lane B / lane C funds at or after the lane-wide `until_block` in force answer `409 lane_ended` with the end block before the key is vaulted; a lane-C competition (table or DEC-CA-0075 spec instance) past its end answers `409 competition_ended`; queued, not yet started entries are released (`failed` / `lane_ended` or `competition_ended`, nothing charged; the key forgotten unless another live entry, pod or direct funding needs it; in-flight legs untouched); share / weight 0 is accepted with `pay_warning`; nothing ended or no head wired = the old replies; `EffectiveLanes` points `lane_c_cfg` at the view; `CachedHead` |
