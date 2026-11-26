# R4 acceptance tests: stake-weighted upgrade tracker (S6)

R4 builds the tracker the S6 activation proof depends on (plan §3.0 R4 row,
§5 row 11 S6, X6a, D10, §14). At S6 the **observed** `cfg-` stake must reach
≥ 2/3 of eligible validator stake, including the largest holder (5E2LP6 at
about 49%), before the overlay's `effective_block`. That is the only
stake-weighted proof of upgrade, because only nodes on the capability
release can verify a `[lane_d]` overlay. These tests are the spec. The
builder may touch only the paths in `PATHS`: `cascade/laned/upgrade_tracker.py`,
and `cascade/laned/__init__.py`, which already exists and is shared with R1,
R2, R6 and R10. Keep it and every other WP's file intact. R4 depends on R2
(the note grammar); it owns no invariants and no T-rows.

The tool runs as a subprocess with cwd = repo root:

```
python -P -m cascade.laned.upgrade_tracker --metagraph F --notes G --cfg X [--threshold T]
```

It reads no chain and no network. The operator feeds it a metagraph and the
notes read elsewhere (fixtures here; X6a feeds live reads).

## Ground truth (reuse, don't re-implement)

- **Note grammar:** `cascade.shared.activation.parse_signals`. A note is
  well-formed iff that returns segments. Anything else is `malformed`: bad
  prefix or version, a short segment, `activation <= lock` when locked,
  activation without a lock, a duplicate feature, any other text.
- **Adoption math:** `cascade.shared.activation.tally` (with
  `eligible_validators` and `signal_for`), the same stake-weighted count
  every node runs for the `cfg-` lock-in. Eligible means `validator_permit`
  and `stake > 0`. A hotkey is signed for the target iff its note is
  well-formed and has a segment whose feature equals the target, at any
  lock/activation blocks. `test_r4_crosscheck.py` calls the real `tally` on
  the same inputs and requires the same eligible set, signed set, stakes,
  ratio and verdict.
- **Target name:** `cascade.shared.netconfig.cfg_feature_name`, which is
  `cfg-` plus the first 8 hex chars of the overlay body's sha256
  (`body_digest`).

## Inputs

- `--metagraph F`: a JSON object `{"<hotkey>": {"stake": <number>,
  "validator_permit": <bool>}, ...}`. `stake` is a JSON number ≥ 0 (not a
  bool or a string), taken **exactly as written in decimal** (`0.1` is 1/10,
  e.g. `json.load(..., parse_float=Fraction)`). `validator_permit` must be a
  JSON bool. Other keys in a row are ignored.
- `--notes G`: a JSON object `{"<hotkey>": "<note text>" | null}`. A hotkey
  that is absent or `null` has a `missing` note. Any other value type is bad
  input. Notes for hotkeys outside the metagraph are ignored.
- `--cfg X`: `cfg-<8 lowercase hex>`, bare `<8 lowercase hex>` (prefixed
  with `cfg-`), or a full 64-lowercase-hex body digest (mapped through
  `cfg_feature_name`). Anything else (upper case, wrong length, another
  feature) exits 2.
- `--threshold T`: a decimal in `(0, 1]`, default `0.6667`, parsed exactly
  (`Fraction(T)`). `0`, negatives, `> 1`, `nan` and non-numbers exit 2.

## Output (stdout, one JSON object, on exit 0 and 1)

```json
{"target": "cfg-1a2b3c4d", "threshold": 0.6667,
 "hotkeys": [{"hotkey": "...", "stake": 490000, "validator_permit": true,
              "counted": true, "note_state": "ok|malformed|missing",
              "release": "rolling-era-king" | null, "cfg": "cfg-…" | null,
              "on_target": false}, ...],
 "signed_stake": 510000, "total_stake": 1000000,
 "ratio": "51/100", "pct": 51.0, "meets_threshold": false,
 "signed": ["..."], "eligible": ["..."],
 "laggards": [{"hotkey": "...", "stake": 490000, "note_state": "ok",
               "release": "rolling-era-king", "cfg": null}]}
```

- `hotkeys`: every metagraph hotkey, sorted by stake descending, then hotkey
  ascending. `counted` is `validator_permit and stake > 0`.
- `release`: the **primary** (first) segment's feature of a well-formed
  note. Each release signals its own primary feature
  (`activation.primary_feature`). It is `null` for malformed or missing notes.
- `cfg`: the target if the note carries it, otherwise the first `cfg-`
  segment in note order, otherwise `null`. It is `null` for malformed and
  missing notes. A laggard on an older overlay shows that overlay's `cfg-`.
- `on_target`: well-formed and some segment's feature equals the target.
  It is computed for uncounted hotkeys too, but they never affect totals.
- `signed_stake` / `total_stake`: the exact sums, emitted as JSON numbers
  (float of the exact value), over counted hotkeys only.
- `ratio`: the exact rational `signed_stake / total_stake` as
  `str(fractions.Fraction)`, in lowest terms (`"1"`, `"51/100"`, `"2/3"`).
- `pct` (**the fixed rounding**): `float(round(100 * ratio, 4))` on the
  exact `Fraction`, i.e. 4 decimal places, **round half to even**. So 2/3
  gives `66.6667`, 0.00005% gives `0.0`, 0.00015% gives `0.0002` and
  99.99995% gives `100.0`. `ratio` stays exact, and the verdict never uses
  `pct`.
- `meets_threshold`: `ratio >= Fraction(T)`, exactly. So 2/3 fails the
  default 0.6667, and 51/100 meets 0.51.
- `signed` / `eligible`: hotkeys sorted ascending, equal to `tally`'s
  `signed` / `eligible`.
- `laggards`: counted hotkeys not on target (malformed and missing notes
  included, so they are named), sorted by stake descending, then hotkey
  ascending.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | `meets_threshold` |
| 1 | below threshold |
| 2 | bad input: unreadable file, invalid JSON, wrong shapes or types, a bad `--cfg` or `--threshold`, or no eligible stake (`total_stake == 0`). A message goes to stderr. |

## The S6 fixture shape (helpers.py)

Six permit holders, total 1,000,000: `5E2LP6…` 490,000 (49%), then 180,000,
120,000, 100,000, 70,000 and 40,000. There is also a no-permit hotkey with
300,000 and a permit holder with 0 stake, neither counted.

| Case | ratio | pct | exit |
|---|---|---|---|
| all on target | `1` | 100.0 | 0 |
| largest off, the rest on | `51/100` | 51.0 | 1 |
| largest + second only | `67/100` | 67.0 | 0 |
| largest + third + sixth | `13/20` | 65.0 | 1 |
| two smallest off | `89/100` | 89.0 | 0 |

## What the tests pin

| Test | Pins |
|---|---|
| `test_upgrade_tracker_module_exists` | `cascade.laned.upgrade_tracker` exists |
| `test_s6_*` | exact golden %, verdicts, laggards and per-hotkey rows on the S6 shape |
| `test_uncounted_*`, `test_notes_for_hotkeys_outside_*` | no-permit, zero-stake and unknown hotkeys never move totals |
| `test_exact_*`, `test_threshold_*`, `test_decimal_*`, `test_pct_rounds_*` | exact rational, the half-even 4-dp rounding, the exact threshold compare |
| `test_release_*`, `test_locked_cfg_*`, `test_malformed_*`, `test_missing_*`, `test_laggard_ties_*` | release/cfg reporting; malformed and missing notes are not on target and are named |
| `test_cfg_*`, `test_bad_cfg_*` | `--cfg` forms, including the netconfig body digest |
| `test_bad_*`, `test_no_eligible_*`, `test_missing_metagraph_*` | exit 2 |
| `test_cross_check_*`, `test_malformed_fixtures_*`, `test_release_matches_*` | agreement with the real `activation.tally` / `parse_signals` |

Run (prod box, tests only):
`nice -n 19 env PYTHONPATH=<wt> /root/cascade/.venv/bin/python -P -m pytest --rootdir <wt> --noconftest -p no:cacheprovider -q <wt>/tests/accept/r4/`
