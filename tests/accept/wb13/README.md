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
`clean` or `embedded`.** There is no "skipped" value. Pinned means checked or
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

## Not pinned

- The order of steps 2 to 4 among themselves.
- The 503 codes and every message text.
- Which hit the receipt names when a tree has several embeddings (it must be
  deterministic).
- Whether hits add up across files or sequences. Every test has one
  embedding with a clear margin.
- Scalar values spread over `config.json` keys, numbers inside string
  constants, big-endian and unaligned decodings, and other integer widths.
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
| `helpers.py` | the suite fixture, the WB11 artefact, the config through `parse_lane_b`, fixture repos as plain files, the real intake builder, the recording fetch, the torch guard, the hash-seed entry point |
| `test_wb13_plan.py` | the WB13 / TB10–TB12 / LB1 / D38 plan text verbatim, the shipped 8 MB cap, fixture sanity against WB11 (passes on main) |
| `test_wb13_scan.py` | `scan_repo`: the three sources, TB10, TB11, TB12 versions, the hit rule, never executing, determinism; `load_pinned_hashset` |
| `test_wb13_intake.py` | `main.py` wiring, the invariant pair, TB10 and TB11 refused at intake, TB12 version change, fail closed, the identity gate before the fetch, D38 |
