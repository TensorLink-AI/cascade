# W-S3 acceptance tests: the era's sentinel numbers on the status boards

W-S3 (plan §3.0 W-S3 row, §16 rule 4, §1.3b "Progress board", §3.2
invariants 1 and 10) publishes the §16 cross-lane sentinel on the status
boards. §16 rule 4 says: "each lane's new king's sentinel numbers, plus the
house's, on a per-suite board in `status/lane_d.json` and the lane A status."

W-S1 draws the era's suite (`cascade.shared.sentinel`). W3b3
(`cascade.validator.lane_d_sentinel`) sweeps it for each new KING_D, the
previous KING_D and the house, and publishes a `lane-d-sentinel/1` record
under `lane-d/sentinel/era-<n>/<king>.json`. W-S3 reads those records and
shows them. It computes no new numbers and never gates.

These tests are the spec. The builder may touch only the paths in `PATHS`:

- `cascade/shared/sentinel_board.py`, a new module;
- `cascade/shared/v2_status.py`, which builds `status/v2.json`, the lane A
  status board;
- `cascade/website/index.html`, to render the board on the dashboard. This
  is presentational and not pinned.

Rules for the tests themselves:

- **Only the two external services are stand-ins.** `helpers.RecordingHost`
  is the bench pod, behind the real `lane_d_bench.SweepSession`.
  `helpers.MemoryStore` is the public bucket. It has the `put_text` /
  `get_text` duck type of `S3Store`, an absent key raises `ObjectNotFound`,
  and `fail_get` makes a read raise `StorageError`.
- **The records come from the real producer.** The store's records are
  written by W3b3's `sweep_and_publish` (`helpers.publish_record`). The
  invalid-record tests then overwrite one with a tampered copy.
- **The model card is replaced.** `v2_status.moe_model_card` builds a
  meta-device torch model, and torch is never loaded on the test host. So
  the tests replace it with the empty card it returns when torch is
  unavailable (`helpers.no_torch_model_card`). No W-S3 rule concerns the
  card.
- **No PATHS module is imported at module level, in a fixture or at helpers
  import time.** Every test runs locally with `--noconftest`, without torch.

On main, 54 tests fail with an `ImportError` for `cascade.shared.sentinel_board`,
which is in `PATHS`. Two fail on an `AssertionError`:

- `build_v2_status` refuses the `sentinel` keyword;
- `lane_a_view` drops the receipt's sentinel draw.

One test passes: `test_lane_a_view_unchanged_when_unarmed`, the unarmed
control. `accept W-S3 --expect-fail` reports 57 collected, 1 passed and
56 failing for a right reason.

## `build_sentinel_board(cfg, *, era_block, era_block_hash, kings, store) -> dict | None`

| argument | meaning |
|---|---|
| `cfg` | the chain config (`[v2] sentinel_from_block`, `sentinel_suites`, W-S1) |
| `era_block`, `era_block_hash` | the era-start block and its hash (`0x` + 64 hex), the ones the W3b3 records were drawn from |
| `kings` | the era's new KING_D digests, in the order the board lists them |
| `store` | the public object store (`get_text(key)`) |

### When there is no board

- **Unarmed: `None`.** When `sentinel_armed(cfg, era_block)` is false
  (`sentinel_from_block = 0`, or an era before it), the call returns `None`
  without reading the store. It is armed exactly from `sentinel_from_block`.
- **Refusals.** These raise before any store read:
  - duplicate kings, an empty digest or a non-string digest (`ValueError`,
    or `TypeError` for `None`);
  - a bare string as `kings` (`TypeError`);
  - a malformed `era_block_hash` (`""`, `"0x"`, non-hex, no `0x` prefix,
    `None`), which raises `ValueError` / `TypeError`, as W-S1's draw does.

### The board

| key | value |
|---|---|
| `schema` | `"sentinel-board/1"` |
| `gating` | `False` |
| `label` | a str containing "non-gating" (case-insensitive) |
| `era` | `{"block": era_block, "block_hash": era_block_hash}` |
| `suite` | `sentinel_for_era(era_block_hash, cfg.v2.sentinel_suites)`, the era's draw |
| `sentinel_suites` | the list it was drawn from |
| `kings` | one row per entry of `kings`, in that order (`[]` for none) |

The era's suite is shown even when `kings` is empty. Extra keys are allowed.

### A row

| key | value |
|---|---|
| `king_digest` | the king |
| `key` | `lane_d_sentinel.sentinel_key(era_block, king)` |
| `state` | `"ok"`, `"fault"`, `"error"`, `"missing"`, `"unavailable"` or `"invalid"` |
| `reason` | the record's `reason` for `ok` / `fault` / `error`; a non-empty str otherwise |
| `boundary`, `suite`, `gate_suite`, `session` | the record's values |
| `models` | the record's `models`, verbatim: `{role: {"digest", "status", "items", "crps", "mase"}}` for `king`, `previous_king` and `house` |

- **A valid record** is shown with the record's `status` as its `state`.
  Its numbers are the published floats, exactly. They aren't re-derived,
  rounded or re-scored. The house's and the previous KING_D's numbers are
  on the row too.
  - A `fault` row keeps the record's `None` metrics for the faulty model
    and the other models' numbers.
  - An `error` row has no numbers.
- **A regressed king** (worse than the previous KING_D and the house) is
  shown with its numbers and `state: "ok"`. The board never hides, flags or
  re-ranks a king.
- **No record is never a number.** Each of these rows has `models: None`,
  no float anywhere and a non-empty `reason`:
  - **`missing`:** the key is absent (`ObjectNotFound`). That includes a
    record published for another era.
  - **`unavailable`:** the read raised `StorageError`. The call never
    raises. A store outage makes every row unavailable, and the era's
    suite is still shown.
  - **`invalid`:** the object at the key isn't a valid record of this
    king and era. The tampers are:
    - text that isn't JSON, or a JSON array;
    - a NaN or Infinity number;
    - a schema other than `lane-d-sentinel/1`;
    - another king's `king_digest`;
    - another era's block or block hash;
    - a `suite` other than the era's draw;
    - `gating` not `False`;
    - an unknown `status`;
    - an `ok` model whose `crps` / `mase` isn't a finite float (`None` or a
      string);
    - no `models`.
- **Reads and writes.** The board reads only the listed kings' keys and
  writes nothing.

### Deterministic rendering (`test_w_s3_determinism.py`)

- **Strict JSON:** the board serialises with `allow_nan=False` and parses
  back equal, even with fault, error, invalid and missing rows.
- **Same inputs, same board:** equal across calls, independent of the order
  the records were written, and identical in two subprocesses with
  different `PYTHONHASHSEED`s.
- **No wall-clock field:** no `as_of`, `generated_at`, `timestamp` or
  `updated_at`.

## `status/v2.json`: the lane A status board (`test_w_s3_v2_status.py`)

- **`build_v2_status(..., sentinel=board)`** puts the board at
  `doc["sentinel"]`, equal to the board and copied, not aliased. Every
  other key is unchanged.
  - `publish_v2_status` writes the same bytes for the same document, and
    those bytes parse strictly.
  - Missing and fault rows reach the document as such.
- **`lane_a_view(receipt)`** adds a `sentinel` entry when the receipt
  carries W-S1's `sentinel` draw record. The entry holds at least that
  record's `suite`, `block` and `block_hash`, and every other key of the
  view is unchanged.
  - Built from the same era, `doc["lane_a"]["sentinel"]` and
    `doc["sentinel"]` name the same suite and era block.

## Invariants 1 and 10

Each `*_unchanged_when_unarmed` test has its `*_armed_nonempty_lane_d_effect`
pair:

| pair | unarmed | armed |
|---|---|---|
| board | `None` and no store read, for `sentinel_from_block = 0` and for an era before it | three `ok` rows with numbers, armed exactly at the arming block |
| `status/v2.json` | with no keyword and with `sentinel=None`, the document equals `helpers.GOLDEN_V2_DOC` and has no `sentinel` key | `doc["sentinel"]` is the board, the rest is the golden |
| `lane_a_view` | a receipt without `sentinel` gives `helpers.GOLDEN_LANE_A_VIEW` | the view adds the draw, the rest is the golden |

The goldens were computed on main f9e9146, over `helpers.minimal_v2_cfg()`
and `helpers.LANE_A_RECEIPT`.

## Not pinned

- **Where the board is built and published.** Neither the trainer's
  `publish_trainer_v2_status` nor the validator holds the era's list of new
  KING_D digests today. The lane D state and reports (W7c / W17) do, and
  the call site needs that list, the era-start block and hash
  (`cascade.shared.sentinel.settlement_era_start`) and the lane D public
  store. The orchestrator must assign that wiring explicitly.
- **`status/lane_d.json`.** That file is W17's (`status/lane_d.json`
  producer: queue, pending progress, king, board), which isn't built yet.
  W17 should embed `build_sentinel_board`'s output there. These tests don't
  pin a W17 key.
- **Lane A sentinel numbers.** W-S1 records the draw on the receipt and,
  under the D34 ratchet, the sentinel's ratio and LCB in the bench gate
  detail. Lane A publishes no per-king absolute sentinel numbers today, so
  the lane A status pins only the draw. Lane A's sentinel can block under
  the ratchet, so the lane A entry isn't labelled non-gating here.
- **The "falls for 3 kings in a row" alert** (§1.3b), the signed lane D
  index (§11.1), the dashboard's HTML, the label's exact text, extra keys,
  and the reason text of the non-record rows.
