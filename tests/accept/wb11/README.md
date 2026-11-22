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
constants: `SCHEMA`, `WINDOW = 32`, `STRIDE = 1`, `SIG_FIGS = 4`.

The artefact:

| key | value |
|---|---|
| `schema` | `"cascade_contamination_hashset/1"` |
| `version` | the int `n` |
| `suites` | `{suite: identity}`. The identity is W3b4's `suite_identity` shape: `suite`, `version`, `items_sha256`, `n_items` and `digest`, built from the item ids and the version. The formula is in `tests/accept/w3b4/README.md`, and `helpers.expected_identity` recomputes it independently. When `cascade.eval.suite_identity` (W3b4) is on main, the artefact must use it, not a copy. That test skips until then. |
| `window`, `stride` | as built: 32 and 1 by default (§12b.1 "sliding sub-windows of 32 points") |
| `hash` | `"sha256"` |
| `normalisation` | at least `{"method": "zscore", "scope": "subwindow", "sig_figs": 4}` |
| `exact` | `{hash: [suite, window_id]}` |
| `normalised` | `{hash: [suite, window_id]}` |

The sub-windows are taken at offsets `0, stride, 2·stride, …` of each item.
Only full windows count. A window that contains a missing value is skipped
in both sets.

- **exact (pinned to the byte):** each hash is the lower-case hex `sha256`
  of the window's values packed as little-endian IEEE 754 float64
  (`struct.pack("<32d", …)`), which is §12b.1's "sha256 of the float64
  values". `helpers.expected_exact` recomputes it.
- **normalised:** the window is z-scored on its own values (per sub-window,
  so a fragment of a series matches), then each value is rounded to 4
  significant figures and the result is hashed with sha256. The exact byte
  encoding, the std convention and the handling of signed or near-zero
  values are the builder's choice. They are fixed by `SCHEMA` version 1,
  and a later change needs a new schema. A **constant** window (std 0) has
  no z-score, so it has no normalised hash. It still has an exact hash.
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
  with `ValueError`. A `.json.gz` is allowed: the gzip is decompressed and the
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
  and so is a constant window for the normalised set.
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
- **A float32-rounded copy:** 0 exact hits, and at least 50 % of its windows
  hit normalised, which makes it contaminated. A z-score near a 4-sig-fig
  boundary flips under any rounding rule, about 1 window in 7 on the fixture.
- **Constant data:** the ints `[3]*40` hit the flat suite series exactly and
  never normalised.
- **A NaN in the submission** removes only the windows that contain it.

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
 "suites": {"gift": {…identity…}, "boom": {…}, "time": {…}, "fev": {…}}}
```

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
  `window` 32, `stride` 1, `hash` sha256 and normalisation (`zscore`, 4)
  match. Both sets are non-empty.

The `test_verifier_*` tests build good evidence offline from the fixture
suites in `tmp_path` and check two things. Good evidence passes. Each of the
19 `TAMPER` cases fails, for example: a fixture source, a rebuild mismatch, a
missing suite, a wrong identity, an unpinned or mismatched mainnet pin, a
mismatched testnet pin, an edited or absent artefact, a `.zst` name, or a
version mismatch. So do an artefact of other suite versions and a v1 name
on version-2 evidence.

A from-data rebuild in CI isn't possible, because the suite data lives on the
bench pod and nothing is downloaded. The evidence records the pod's second
build as `rebuild_sha256`. CI re-derives the digest from the committed file.

## Not pinned

- The normalised byte encoding (above).
- How hashes are stored in memory for fast lookup.
- The real-data reader.
- Extra artefact keys, as long as they're deterministic: no timestamps,
  hosts or paths.
- `.gz` support in `write_artefact`.
- The CLI's error text.
- Whether a negated copy (`−x`) matches.
