# WB11 acceptance tests: the contamination hash-set builder

WB11 (plan §3.0 WB11 row, §12b.1 LB1, Appendix C item 2) builds the hash set
that LB1's intake check (WB13) matches submissions against. It holds exact and
normalised sub-window hashes of every suite test window. The artefact is
versioned and pinned by its sha256 in `[lane_b] contamination_hashset` (WB10:
release-only, `""` until pinned). Mode is `ci + owner (pin)`: CI checks the
builder, and the owner builds the real artefact on the bench pod and pins it.

These tests are the spec. The builder may touch only `PATHS`, which is the
new module `cascade/eval/contamination.py`. The module sits next to the suite
runners (`benchmarks.py`) because it describes the suite data. WB13's intake
imports its lookup.

Rules for the tests:

- No PATHS module is imported at module level, in a fixture or at the top of
  `helpers.py`.
- On main every test that isn't skipped fails. The failure is a
  `ModuleNotFoundError` for `cascade.eval.contamination`, raised in the test
  body or shown in a CLI subprocess's stderr inside an `AssertionError`.
- **Tests correction (schema 2, TB11 at every scale).** WB11 is built on
  main with schema 1. The changed tests fail on main with an
  `AssertionError`, pending a follow-up implementation PR:
  - `test_wb11_scales.py` (all of it);
  - `test_artefact_schema_and_recorded_parameters` and
    `test_a_schema_1_artefact_is_refused`;
  - `test_float32_rounded_copy_hits_normalised_not_exact` (floor 100 %);
  - the three `test_verifier_*` good-evidence cases (artefact schema 2).

  Every other test passes on main.
- **Tests correction (RV6 B5 + B8).** The review found benign repos refused
  (B5) and the lookup's memory unbounded (B8). These tests fail on main with
  an `AssertionError`, pending an implementation PR:
  - `test_wb11_low_info.py` (low-information windows; some cases already
    pass on main);
  - `test_wb11_memory.py` (`load_lookup` and its memory budget; the file-size
    case passes on main);
  - `test_exact_set_is_sha256_of_float64_sub_windows`,
    `test_exact_set_skips_nan_..._smallest_window`,
    `test_every_suite_window_is_in_both_sets`,
    `test_stride_is_recorded_and_the_scan_is_still_every_offset` and
    `test_offset_copy_and_int_inputs` (a constant window is no longer in the
    exact set, and `[3]*40` no longer hits);
  - every `test_verifier_*` case: `good_evidence` now records the
    `load_lookup` peak.
- Nothing is mocked and no data is downloaded. The fixture suites
  (`helpers.fixture_suites`) are four tiny seeded in-memory suites, and no
  test needs torch.

## Input: the suites

```python
{"gift": {"version": "30841734ac5c…", "series": {"<item id>": [float | None, ...]}},
 "boom": {...}, "time": {...}, "fev": {...}}
```

- **Suite keys** are the item-suite keys `gift`, `boom`, `time` and `fev`
  (`cascade.eval.benchmarks.ITEM_SUITE_KEY` values). Any other key raises
  `ValueError`. So does an empty mapping.
- **`version`** is the suite's pinned data revision, a non-empty `str`. For
  FEV it is the manifest digest. A missing, empty or non-str version raises
  `ValueError`/`TypeError`, because the hash set must come from pinned data.
- **Each item** is one test window's values. `None` and NaN both mean a
  missing value, and they are equivalent: the same data gives the same
  artefact. Ints are taken as floats. The input is not mutated.

How the bench pod turns its `/data/bench` suite directories into this mapping
is not pinned. The builder may add a `--data-dir` reader, as long as the
`--input` path pinned below keeps working.

## `build_hashset(suites, *, version, window=32, stride=1) -> dict`

`version` is the artefact version `n` (`hashset-v<n>`), an int ≥ 1. A bool,
a str or a value < 1 raises. So does `window` < 2 or `stride` < 1. Module
constants: `SCHEMA`, `WINDOW = 32`, `STRIDE = 1`. (v1's `SIG_FIGS = 4` is
no longer pinned: see "Normalised matching at every scale".)

The artefact:

| key | value |
|---|---|
| `schema` | `"cascade_contamination_hashset/2"` |
| `version` | the int `n` |
| `suites` | `{suite: identity}`. The identity is W3b4's `suite_identity` shape: `suite`, `version`, `items_sha256`, `n_items` and `digest`, built from the item ids and the version. The formula is in `tests/accept/w3b4/README.md`, and `helpers.expected_identity` recomputes it independently. When `cascade.eval.suite_identity` (W3b4) is on main, the artefact must use it, not a copy. That test skips until then. |
| `window`, `stride` | as built: 32 and 1 by default (§12b.1 "sliding sub-windows of 32 points") |
| `hash` | `"sha256"` |
| `normalisation` | at least `{"method": "zscore", "scope": "subwindow"}`, plus the scheme's parameters (recommended record below); no `sig_figs` key, since schema 2 doesn't round relatively |
| `exact` | `{hash: [suite, window_id]}` |
| `normalised` | `{hash: [suite, window_id]}` |

The sub-windows are taken at offsets `0, stride, 2·stride, …` of each item.
Only full windows count. A window that contains a missing value is skipped
in both sets. So is a **low-information** window (RV6 B5, section
"Low-information windows" below), constant windows included.

- **exact (pinned to the byte):** each hash is the lower-case hex `sha256`
  of the window's values packed as little-endian IEEE 754 float64
  (`struct.pack("<32d", …)`), which is §12b.1's "sha256 of the float64
  values". `helpers.expected_exact` recomputes it.
- **normalised:** the window is z-scored on its own values (per sub-window,
  so a fragment of a series matches), the z-scores are quantised and the
  result is hashed with sha256. The quantisation must meet the detection
  promise in "Normalised matching at every scale" below; the recommended
  scheme is there. The exact byte encoding is the builder's choice, fixed by
  `SCHEMA` version 2, and a later change needs a new schema. A **constant** window (std 0) has
  no z-score, so it has no normalised hash. Since RV6 it has no exact hash
  either: it is low-information.
  Normalised hashes never equal exact hashes.
- **`window_id`** is `"<item id>@<offset>"`. When several windows share one
  hash, the value is the smallest `(suite, item, offset)` tuple. That is
  deterministic and gives the receipt's `suite` and `window_id` (WB13).

## Bytes, digest, file

- `artefact_bytes(a)` is
  `json.dumps(a, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")`.
- `artefact_sha256(a)` is `"sha256:" + sha256(artefact_bytes(a)).hexdigest()`.
  That's the value `[lane_b] contamination_hashset` carries, and the WB10
  loader accepts it (test).
- `write_artefact(a, path)` writes exactly `artefact_bytes(a)` and returns
  the sha256.
- `load_artefact(path)` returns an equal dict. It refuses a wrong `schema`
  with `ValueError`, **including a `cascade_contamination_hashset/1`
  artefact** (test). A v1 artefact's normalised hashes use the old encoding,
  so loading it would silently miss every normalised match. No v1 artefact was
  ever pinned (`[lane_b] contamination_hashset = ""` in both chain files), so
  refusing it breaks nothing. A `.json.gz` is allowed: the gzip is decompressed and the
  digest is over the canonical JSON. The plan names `.json.zst`, but
  `zstandard` isn't a dependency and adding one is an Approve-tier change, so
  `.zst` isn't pinned.
- **Determinism:** the bytes depend only on the suite data, the versions,
  `window`/`stride` and `n`. Suite and item order, the process hash seed, the
  time, the host and the paths don't change them. Two CLI runs with different
  `PYTHONHASHSEED`s and shuffled inputs write identical files. That's the
  Accept note, "hash set reproducible from the suite identities".
- **What moves the digest:**
  - a different suite version, which changes the identity but leaves the
    hashes the same;
  - a changed value;
  - an added item;
  - a different `n`.

  A data edit doesn't change the identity. That's why the identity carries
  the data revision, and why §12b.1 says the hash set is refreshed only when
  a suite identity changes.

## Lookup

`match_series(artefact, values) -> dict` scans the submission at **every**
offset (stride 1, whatever the artefact's stride) with the artefact's window:

```json
{"exact_hits": 3, "normalised_hits": 3,
 "first": {"kind": "exact", "suite": "gift", "window_id": "gift/m4_hourly/0@20", "at": 50}}
```

- **Hits:** the number of submission sub-window positions whose exact or
  normalised hash is in the set. A window with a missing value is skipped,
  and so is a low-information window (constant included), for both sets.
- **`first`:** the hit at the lowest submission offset `at`, with exact
  checked before normalised at the same offset. It is `None` when nothing
  hits.
- It works the same on a built and on a loaded artefact.

`is_contaminated(artefact, values, min_hits=None) -> bool` is true when
`exact_hits ≥ min_hits["exact"]` **or** `normalised_hits ≥
min_hits["normalised"]`. `min_hits` is anything `dict()` accepts, including
`LaneBConfig.contamination_min_hits` (pairs). `None` means the WB10 default,
`{exact: 1, normalised: 3}`.

What the tests pin:

- **A clean series:** 0 hits and not contaminated.
- **An embedded 34-point segment:** 3 exact hits, the right `suite`,
  `window_id` and `at`, and contaminated.
- **31 points:** 0 hits.
- **A scaled and shifted copy** (`7.5·x + 1234.5`, `x − 1e6`, `2·x − 9`):
  0 exact hits, and at least 90 % of its windows hit normalised.
- **A float32-rounded copy:** 0 exact hits, and **every** window hits
  normalised (v1 asked for 50 %: its 4-sig-fig rounding lost about 1 window
  in 7), which makes it contaminated.
- **TB11 evasions at every scale:** see the next section.
- **Constant data:** the ints `[3]*40` equal the flat suite series but hit
  nothing and are not contaminated (RV6 B5; before RV6 they gave 9 exact
  hits).
- **A NaN in the submission** removes only the windows that contain it.

## Normalised matching at every scale (`test_wb11_scales.py`, schema 2)

**Why schema 1 fell short.** v1 rounded each z-score to 4 significant
figures. That is relative rounding: a z-score of 0.0123 is cut at 1e-6, one
of 1.23 at 1e-3. Rounding a copy to 3 decimals moves a z-score by about
`0.6 · 0.001 / (|a| · window std)`. That's ~6e-3 for TB11's `×3.7` at series
level 0.4 (window std ~0.03), ~4e-4 at level 7 and ~4e-5 at level 100, so the
4th significant figure flips in most windows. Measured on main (v1): TB11's
`round(3.7·x + 100, 3)` hits 0/17 windows at levels 0.4 and 7, 0/17 at 100
(2/49 on `gift/m4_hourly/0`) and 11/17 at 5000. The float32 casts hit
between 1 and 16 of 17. WB13 (PR #615) found the same gap. A miner could then dodge LB1 by
rounding, which is the evasion TB11 names.

**The promise (pinned behaviour).** Take every sub-window of a suite series
at levels **0.4, 7, 100 and 5000** (`helpers.scale_suites`: one 48-point
random walk per level, in boom / fev / gift / time, with WB13's walk
parameters; plus TB11 on the base fixture's `boom/svc-a/cpu`,
`gift/m4_hourly/0` and `gift/electricity/7`). Then:

| copy | minimum normalised-hit share | why that floor |
|---|---|---|
| `round(a·x + b, 3)` for `(a, b)` = TB11's `(3.7, 100)`, `(−2.5, 7)`, `(12, −1e4)`, `(1, 0)` | **90 %** of the copy's windows | The needed tolerance is bounded (below), so a correct scheme hits every window. 10 % slack is room for a probe cap. A 40-point fragment (9 windows) still gives ≥ 8 hits, well above `min_hits.normalised = 3`. |
| `float32(x)` and `float32(3.7·x + 100)` | **100 %** | float32 moves a z-score by ≤ ~1e-5 at these levels, far inside any tolerance that meets the rounded rows. |

Each copy also needs 0 exact hits, a `first` hit of kind `normalised` naming
the right suite and item, and `is_contaminated` (WB10 default rule) true.

- **`a < 0` is covered** (a negated copy flips every z-score's sign), so
  "whether a negated copy matches" is no longer an open point.
- **The bound on `a`.** A rounded case is promised only while the rounding
  step is small against the copy's spread: `0.001 ≤ |a| · std(window) / 25`
  for every window (rounding step ≤ 4 % of the window std). Each test checks
  this on its fixture as a precondition. Below it, rounding destroys the
  shape itself (with `a = 1e-6`, every value rounds to one number), and no
  normalisation can recover it. The plan's TB11 (`×3.7` at 3 dp) is inside
  the bound at all four levels, and so is plain 3-dp rounding of the series
  itself.
- **No false positive.** At each level, 30 unrelated series (15 random walks
  with the suite's own walk parameters and other seeds, 15 i.i.d. Gaussian
  around the level) give 0 exact and 0 normalised hits. So do their TB11
  copies, and none is contaminated. That's 2040 clean sub-windows with 0
  hits, which bounds the per-window false-positive rate below
  3/2040 ≈ 1.5e-3 (95 %, rule of three). A series needs 3 hits to be refused.
  The recommended scheme measured 0 hits in 19 800 windows (about 72 probes
  each) while this was written.

**Recommended scheme (`cascade-zgrid/2`).** This is a suggestion that meets
the promise; only the behaviour above is pinned. Determinism is already pinned
by the byte-identical rebuild tests, so the encoding needn't be.

1. z-score each window with the population std. A constant window gets no
   hash.
2. **Fixed-point grid:** the bin of z is `floor(z / GRID)`, with
   `GRID = 0.5`. These are absolute bins, the same width at every |z|, so
   near-zero z values aren't special and the sign is just part of the bin.
3. **Build:** store one hash per window, sha256 of a domain tag
   (`cascade-contamination-zgrid/2\n`) plus the comma-joined bin ints.
4. **Lookup (tolerant multi-probe):** for each submission window, every value
   whose `z / GRID` lies within `TOLERANCE / GRID` of a bin edge
   (`TOLERANCE = 0.03`) also probes its neighbour bin. Try every combination
   of those alternatives, nearest-to-edge first, capped at the 12 nearest
   (≤ 4096 probes). Do it for the window **and its negation** (−z, so
   `a < 0` matches). It's a hit if any probe is in the set. The first probe
   is the plain floor-bin hash, so a suite series always hits itself. On
   average, ~72 probes per window across both orientations.
5. Correctness: if every |z_copy − z_suite| ≤ TOLERANCE, the copy's probes
   include the suite window's bins. Measured worst-case |Δz| inside the
   bound above is ≈ 0.02 (level 0.4, plain 3-dp rounding), so
   `TOLERANCE = 0.03` has margin.
6. Record:
   `{"method": "zscore", "scope": "subwindow", "std": "population",
   "grid": 0.5, "tolerance": 0.03, "orientation": "both",
   "encoding": "cascade-zgrid/2"}`.

Rank or ordinal encodings were considered and rejected. At level 0.4, 3-dp
rounding creates or breaks near-ties in most 32-point windows, so the rank
vector changes.

**Plan wording.** §12b.1 says "z-score, rounded to 4 significant figures".
Schema 2 replaces that rounding rule with the scheme above to meet §12b.1's
own "robust to scale, offset and rounding" and TB11. The plan text should
follow, as an owner edit outside this PR.

## Low-information windows (`test_wb11_low_info.py`, RV6 B5)

**The problem.** Real GIFT/BOOM data has zero runs (solar at night), sparse
0/1 counters, exact index counters and values at a level plus float noise.
Schema 2 as first built hashed every one of these windows. So a benign repo
holding 1 KB of zero bytes, 32 `x = 0` lines, one-hot list literals or a
`linspace` ramp was refused (measured on main against `helpers.hazard_suites`:
the zeros give 97 exact hits, three one-hot lists 65 exact and 48 normalised,
`range(48)` 17 + 17, a linspace ramp 1 normalised).

**The rule (pinned; `helpers.low_information`).** A 32-point window is
low-information when **any** of these holds:

| | condition | catches |
|---|---|---|
| (a) | fewer than 4 distinct values | constant, one-hot, 0/1 and 3-level steps |
| (b) | one value fills ≥ 3/4 of the window (24 of 32) | zero runs with a few spikes |
| (c) | population std ≤ 2⁻⁴⁶ · \|mean\| (std 0 included) | a level plus float noise: the window spans < 64 float64 ulps of its level |
| (d) | a least-squares line in the index leaves ≤ 1e-6 of the variance | `range`, `arange`, `linspace`, exact counters |

Every rule is invariant under `a·x + b` with `a ≠ 0` (for (c): while the
shift stays below 2⁴⁶ window stds), so a TB11 copy is low-information exactly
when its suite window is.

**Decision: both sides apply it.**

- **Build:** a low-information window is in **neither** set. Its exact hash is
  not evidence: an all-zero window in a suite and in a benign file are the same
  bytes for unrelated reasons.
- **Lookup:** a low-information **submission** window is never counted, exact
  or normalised, whatever the artefact holds. The build rule alone isn't
  enough. A one-hot list's z-pattern equals that of a *noisy* spike train,
  and a linspace's equals a noisy trend's. Both are informative suite windows
  that stay in the normalised set (`boom/spikes/0` and `time/trend/0` in the
  fixture). The lookup rule also covers an artefact built before the rule.

**What the tests pin:**

- the exact set equals the low-information-filtered reference, and no window
  of a low-information item is in either set;
- six benign streams give 0 hits and aren't contaminated: 128 zero f64s, 32
  int zeros, three one-hot lists, a linspace ramp, `range(48)`, and 1000 plus
  1e-12 noise;
- an artefact with hand-injected low-information hashes gives no hit;
- real suite windows are still detected: every informative window of the
  zero-run, spike-train and trend items hits both sets, the series is
  contaminated, and a scaled copy of the spike train still hits ≥ 90 %.

**Residual false negatives** (record them in the decision register with
WB13's): a benchmark window that is itself low-information can't be caught;
neither can a copy shifted by ≥ 2⁴⁶ of its window std (rule c). At that
shift float64 keeps under 6 bits of shape per value, and Guard 4 is the
backstop.

## Memory budget (`test_wb11_memory.py`, RV6 B8)

**The problem.** `load_artefact` returns the whole artefact as dicts. Measured
on main at 50 000 windows: a heap peak of 57.6 MB (~1 150 B per window) for
`load_artefact` plus `artefact_sha256`, and 9.2 MB `.json` / 4.2 MB `.json.gz`
(183 / 84 B per window) on disk. At the review's "tens of millions" of
windows (2·10⁷), that's a ~23 GB load. The intake runs on the 7 GB production
box next to the validator, so this would OOM every service.

**The contract.** `load_artefact` keeps returning the full dict: the CLI,
the evidence check and `loaded == built` use it. The intake gets a new
loader:

`load_lookup(path, *, expect_sha256=None)` returns a read-only lookup
structure.

- `match_series` and `is_contaminated` accept it, and give the same answers
  as on the dict artefact.
- It exposes `["schema"]`, `["version"]`, `["window"]`, `["stride"]` and
  `["suites"]`. Those are the header keys WB13's `scan_repo` reads.
- It raises `ValueError` in three cases: the file's canonical-JSON sha256 isn't
  `expect_sha256`, the file is tampered, or the schema isn't 2 (schema 1
  included).
- How it is built isn't pinned. It doesn't have to load the JSON into
  dicts. For example: stream-parse the canonical JSON, whose keys are
  already sorted, and hash it incrementally. Then keep sorted `array("Q")`
  64-bit hash prefixes plus an item index and an offset per hash. A
  throwaway prototype of that measured 50.7 B/window peak, including the
  fixed buffers, and 60 B/window of RSS growth.

**The bounds (pinned; `helpers`).** They're measured on a synthetic
50 000-window artefact (4 seeded random walks of 12 531 points). It is built in
one fresh process and loaded in another, about 3 s of CPU each:

| quantity | bound |
|---|---|
| `load_lookup` heap peak (tracemalloc) | ≤ 64 B/window + 1 MiB |
| process RSS growth over the load (ru_maxrss) | ≤ 128 B/window + 16 MiB |
| `hashset-v<n>.json.gz` as `write_artefact` writes it | ≤ 96 B/window + 64 KiB (main: 84) |
| any pinned artefact's lookup peak | ≤ 1.5 GiB, whatever N is |

**What that means at real-suite scale.** 2·10⁷ windows load in ≤ 1.28 GB of
heap and at most 2.6 GB of RSS growth. The committed `.json.gz` is ≤ 1.9 GB,
which is large for a git repo: an owner decision at pin time. The 1.5 GiB cap
admits at most ~2.5·10⁷ suite windows. Above that the pin is refused, and the
owner decides first: a shorter suite list, a stride > 1, or a host with more
memory.

**The pin records it.** `verify` also prints `n_exact`, `n_normalised`,
`file_bytes` and `lookup_peak_bytes`, which is the traced peak of
`load_lookup` on that file. The evidence carries the same four fields, and
`helpers.evidence_problems` checks them:

- the counts equal the artefact's set sizes;
- `file_bytes` is the file's size, within the file bound;
- `lookup_peak_bytes` is a measured int ≥ 1 within the per-window budget and
  the 1.5 GiB cap.

## CLI (the owner runs it on the bench pod)

```
python -P -m cascade.eval.contamination build --input SUITES.json --version N --out PATH
python -P -m cascade.eval.contamination verify --artefact PATH --sha256 sha256:<hex> [--input SUITES.json]
```

- **`SUITES.json`** is `{"suites": <the input mapping>}`, with a missing
  value written as `null`.
- **`build`** writes `artefact_bytes` to `--out`. It prints one JSON object
  with `sha256`, `version`, `suites` (identities), `n_exact`, `n_normalised`
  and `out`, and exits 0. Bad input exits 2 and writes nothing.
- **`verify`** prints one JSON object with `ok` (bool). It exits 0 only when
  three things hold:
  - the file loads;
  - its sha256 equals `--sha256`;
  - with `--input`, a rebuild from that data (same `n`, `window` and
    `stride`) is byte-identical.

  Otherwise it exits 1. So a tampered file, a wrong pin, or other suite data
  (for example another `time` version) fails.

## The owner "pin" step and its evidence (`test_wb11_evidence.py`)

The owner runs these steps:

1. Build `benchmarks/contamination/hashset-v<n>.json` (or `.json.gz`) on the
   bench pod from the pinned suites. Build it twice, with different hash
   seeds.
2. Commit the artefact.
3. Pin its sha256 in `chain.toml [lane_b] contamination_hashset` with a
   release. The key is release-only (WB10), never an overlay.
4. Add `evidence/wb11.json`:

```json
{"wp": "WB11", "passed": true, "schema": "laned_wb11_pin/1", "source": "bench_pod",
 "artefact": "benchmarks/contamination/hashset-v1.json", "version": 1,
 "sha256": "sha256:<64 hex>", "rebuild_sha256": "sha256:<64 hex>",
 "suites": {"gift": {…identity…}, "boom": {…}, "time": {…}, "fev": {…}},
 "n_exact": 123, "n_normalised": 120, "file_bytes": 4567, "lookup_peak_bytes": 8910}
```

(The last four fields are from RV6 B8, as `verify` prints them; see "Memory budget".)

Then `scripts/laned/accept WB11 --evidence evidence/wb11.json` exports it as
`$LANED_EVIDENCE`. The `test_evidence_*` tests **skip** while it is unset,
following the r2x pattern. When it is set, `helpers.evidence_problems` must
return `[]`. It checks these:

- **Fields:** `wp`, `passed` and `schema` are as above. `source` is a
  non-empty string and never `"fixture"`.
- **Digests:** `sha256` has the form `sha256:<64 hex>`, and
  `rebuild_sha256 == sha256`, which shows the second build reproduced it.
- **Suites:** they cover exactly gift / boom / time / fev. Each `version`
  contains the plan's pinned identity (`30841734`, `69325b54`, `83e3d0b3`,
  `6de245af`; §12b.1, case-folded). A suite refresh changes these and the
  plan together.
- **Pins:** `chain.toml [lane_b] contamination_hashset == sha256`.
  `chain.testnet.toml` holds either the same value or `""`.
- **The artefact file:** its path is `benchmarks/contamination/hashset-v<n>.json[.gz]`,
  `n` equals `version`, and the file is in the repo. It loads through
  `load_artefact`. Its `artefact_sha256` equals the pin and is sha256 over the
  canonical JSON. Its schema, `version`, `suites` (equal to the evidence's),
  `window` 32, `stride` 1, `hash` sha256 and normalisation (`zscore`,
  `subwindow`) match. The schema is `cascade_contamination_hashset/2`. Both sets are non-empty.

The `test_verifier_*` tests build good evidence offline from the fixture
suites in `tmp_path` and check two things. Good evidence passes. Each of the
27 `TAMPER` cases fails, for example: a fixture source, a rebuild mismatch, a
missing suite, a wrong identity, an unpinned or mismatched mainnet pin, a
mismatched testnet pin, an edited or absent artefact, a `.zst` name, a
version mismatch, or (RV6 B8) a missing or wrong `n_exact` /
`n_normalised` / `file_bytes`, or a `lookup_peak_bytes` that is missing, 0,
over the per-window budget or over the 1.5 GiB cap. So do an artefact of other suite versions and a v1 name
on version-2 evidence.

A from-data rebuild in CI isn't possible, because the suite data lives on the
bench pod and nothing is downloaded. The evidence records the pod's second
build as `rebuild_sha256`. CI re-derives the digest from the committed file.

## Not pinned

- The normalised byte encoding and the grid/tolerance values. The scheme in
  "Normalised matching at every scale" is recommended, and only its
  detection and false-positive behaviour is pinned.
- How `load_lookup` stores hashes in memory, within the "Memory budget"
  bounds, and the probe order or cap, within the 90 % floor.
- The real-data reader.
- Extra artefact keys, as long as they're deterministic: no timestamps,
  hosts or paths.
- `.gz` support in `write_artefact`.
- The CLI's error text.

(A negated copy, `a < 0`, **must** match under schema 2. It is no longer
unpinned.)
