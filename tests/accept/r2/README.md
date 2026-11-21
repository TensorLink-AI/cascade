# R2 acceptance tests: note-budget decoder (S0b)

R2 builds the S0b note-budget decoder (plan §3.0 R2 row, §5 row 3 S0b,
§13 "a 128-byte note crash-loop", §14.1, T22). These tests are the spec. The
builder may touch only the paths in `PATHS`: `cascade/laned/notes.py`, and
`cascade/laned/__init__.py`, which is shared with R1 (`cli.py`, `blocks.py`)
and R6 (`dedrift.py`). Create it only if absent and leave other WPs' files
alone. R2 owns no invariants. It owns T22's CI-checkable part; the testnet
run of T22 belongs to S3 (X3).

The tool runs as a subprocess, `python -m cascade.laned.notes <command> ...`,
with cwd = repo root. It reads no chain and no network. R2x feeds it the live
mainnet notes (evidence `evidence/s0b_notes.json`).

## Ground truth

The trim rule is in `cascade/shared/activation.py` (`own_signal_payload`,
`_own_note_parts`, `note_trimmed_segments`, `v2b_note_problem`). The plan
cites it as `activation.py:1334-1357`; the module lives in `cascade/shared/`.

- A note is `cascade-ready:1:<feature>:<lock>:<activation>[:<f>:<l>:<a>...]`.
  It lives in one `Raw0`-`Raw128` commitment field, so `NOTE_MAX_BYTES = 128`
  (UTF-8 bytes).
- The first segment is the **primary** segment. It is never trimmed.
- A `cfg-<8 hex>` segment (DEC-CA-0057 overlay vote, `netconfig.cfg_feature_name`)
  is the **tail**. Every other segment is an **extra**, in note order.
- While the note is over 128 bytes and something is left to drop, the
  builder drops the **last extra** first (newest). It drops the `cfg-` tail
  only once no extra is left.

The cross-check tests call the real builder on the same inputs and require
the decoder to give the same final note and the same trimmed list, in the
same order.

## Input (both commands)

Give exactly one of these (mutually exclusive):

- `--hex HEX`: the commitment bytes as hex. An optional `0x` prefix is
  allowed, and either case works.
- `--text NOTE`: the note string itself.

`--head-block N` is required.

A note must decode under the real v1 grammar (`activation.parse_signals`):
the right prefix and version, three fields per segment, integer blocks
≥ 0, `activation > lock` when locked, no activation without a lock, and
distinct features. Anything else, including bad hex, exits **2** with a
message on stderr.

**State of a segment:** `decided` iff `lock_block > 0` and
`activation_block <= head_block`. Otherwise it is `pending`, which covers an
unlocked vote or a lock whose activation is still ahead.

**Role:** `primary` (the first segment), `cfg` (a later segment whose name
starts with `cfg-`) or `extra`.

## `decode`

`decode (--hex H | --text T) --head-block N` exits 0 and prints one JSON
object on stdout:

```json
{"note": "<decoded text>", "bytes": 97, "max_bytes": 128, "head_block": N,
 "segments": [{"feature": "...", "lock_block": 0, "activation_block": 0,
               "role": "primary|extra|cfg", "state": "pending|decided"}, ...]}
```

## `simulate`

`simulate (--hex H | --text T) --head-block N --cfg cfg-<8 lowercase hex>
[--cfg-lock L] [--cfg-activation A] [--worst-case]`

- `L` and `A` default to 0. They must be integers ≥ 0, but nothing else is
  required of them: `L == A` is allowed, because the real gate builds its
  budget worst case as `big:big`.
- `--cfg` must match `cfg-[0-9a-f]{8}`. A bad `--cfg`, a negative block or
  an unusable input exits 2, with a message on stderr.
- Any `cfg-` segment already in the note is **replaced** by the new one. The
  builder carries exactly one `netconfig_segment`, and the new one goes last.
- **`--worst-case`** mirrors `v2b_note_problem`. Let `D = max(7,
  len(str(head_block + 216000)))` (`NOTE_BLOCK_HORIZON`) and
  `big = 10**D - 1`. Every **pending extra** segment, and the new `cfg-`
  segment if it is pending, is judged at `big:big`. The primary segment and
  decided segments keep their blocks.
- The 128-byte trim rule above is then applied.

Output is one JSON object on stdout:

```json
{"input": <the decode object for the input note>,
 "cfg": {"feature": "cfg-…", "lock_block": L', "activation_block": A'},
 "worst_case": false,
 "note": "<the note the builder would write>", "bytes": 109, "fits": true,
 "trimmed": [{"feature": "...", "role": "extra|cfg", "state": "pending|decided"}],
 "pending_trimmed": ["..."], "refused": true}
```

- `cfg` holds the blocks as simulated, so after `--worst-case` inflation.
- `trimmed` is in trim order. `state` is each segment's state from the
  **input** blocks; for the `cfg-` segment, it's from `--cfg-lock` and
  `--cfg-activation`.
- `pending_trimmed` lists the trimmed features whose state is pending.
- `refused` is `pending_trimmed != []`. That is the S0b exit gate: no
  still-pending segment may be trimmed.
- `fits` is `bytes <= 128`. It's false only when the primary segment alone
  is over budget.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | not refused |
| 1 | refused |
| 2 | usage or input error |

## The 2026-11-13 crash-loop fixture (`fixtures/crashloop_20261113.json`)

Source: OPSLOG 2026-11-13 00:13–02:25 UTC (PR #457). An external validator
refused to start every ~5 min with this error:

`ActivationStartupRefused: the activation note would exceed 128 bytes and
TRIM the v2b-69630f92 segment (note without it:
'cascade-ready:1:rolling-era-king:9374400:9378000:prelaunch-95-r2:9486000:9486900:cfg-ffffffff:9999999:9999999')`

The input note is that node's note with the v2b vote still open:
`…:prelaunch-95-r2:9486000:9486900:v2b-69630f92:0:0` (97 bytes). The head
block was about 9,493,750; any head with `head + 216000 < 10**7` gives the
same 7-digit worst case. Running `simulate --worst-case --cfg cfg-ffffffff`
on it must:

- exit 1;
- trim exactly `v2b-69630f92` (pending);
- produce the logged note byte for byte.

The real `v2b_note_problem` on the same config and records (the pre-decision
release with v2b untyped, built from `tests/unit/predecision_main.py`) must
also produce the logged text. The PR #457 shape, with v2b typed and its
segment gone, fits with nothing trimmed.

## What the tests pin

| Test | Pins |
|---|---|
| `test_notes_module_exists` | `cascade.laned.notes` exists |
| `test_decode_*` | segments, roles, states, hex/text equivalence, exit 2 on malformed input |
| `test_crashloop_*`, `test_pr457_*`, `test_without_worst_case_*` | the refusal reproduced, matched to the live gate's text; the #457 fix fits; `--worst-case` is what triggers it |
| `test_worst_case_*` | the digit rule (7 digits minimum, 8 from head 9,784,000); primary and decided segments are not inflated |
| `test_only_decided_*`, `test_newest_extra_first_*`, `test_primary_alone_*`, `test_cfg_outranks_*`, `test_simulate_replaces_*` | trim order, the refusal rule, replacing an existing `cfg-` |
| `test_cross_check_*` | the decoder equals the real `own_signal_payload` / `note_trimmed_segments` on the crash-loop config, an all-votes-open config (a real multi-segment trim) and the worst case |
| `test_t22_*` | T22 (CI part): main's shipped `chain.toml` signals only the primary segment, plus `cfg-`, with nothing trimmed even at an 8-digit worst case; the live external shape refuses unless v2b is typed |

Run (prod box, tests only):
`nice -n 19 env PYTHONPATH= /root/cascade/.venv/bin/python -m pytest --noconftest -p no:cacheprovider -q tests/accept/r2/`
