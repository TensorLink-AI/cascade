# W17 acceptance tests: `status/lane_d.json` + the W18 vault ledger

W17 (plan rev6 §3.0 W17 row, depends on W5b and W7c) is the producer of
`status/lane_d.json`: "queue, pending progress, king, board; **rev3:**
per-entry reason enum including `coldkey_cap_settlement`, `confirm_pending`
and reveal deadline" (§11.2, §11.3 "Support": "each pending entry's
eligible-window progress against the floors"; §1.3b "Progress board"; §16
rule 4 and the W-S3 README: "W17 should embed `build_sentinel_board`'s output
there"). Plan rev6.2 item (e) (owner 2026-11-22, PR #636) also gives W17 the
W18 vault ledger `laned_vault_ledger/1`: "the current era, the vault entries
with their statuses and `closed_era`, and the open disputes".

These tests are the spec. The builder may touch only the paths in `PATHS`:
one new module, `cascade/validator/lane_d_status.py`. Wiring the publish call
into the validator loop is W8b's (rev6.2, proposed), not W17's.

Rules for the tests themselves:

- **Real states and reports.** Every `lane_d_state.json` and signed
  `lane-d-report/1` comes from real `settle_lane_d` runs on the W7c synthetic
  chain (`tests/accept/w7c/helpers.py`: its pools, forecasts and test
  keypairs; numpy only, no torch), with a bench `SweepSession` per settlement.
  The fixtures are in `helpers.py` (below).
- **Only external services are stand-ins:** the bench pod
  (`w7c.BenchHost`, `helpers.DownHost`) and the public bucket
  (`helpers.MemoryStore`, the `put_text` / `get_text` duck type of
  `cascade.shared.hippius.S3Store`; an absent key raises `ObjectNotFound`).
  The sentinel record is written by W3b3's real `sweep_and_publish`
  (`tests/accept/w_s3/helpers.publish_record`). The W18 round trip uses W18's
  real `parse_ledger`, `FsStore` and `sweep` (dry-run), all on main.
- **The config is the shipped `chain.toml`** through the real loader
  (`parse_chain_config`), `[lane_d]` armed at `ARM_BLOCK = 9_720_000` with
  the W7c floors (24 windows / 4 sources), `[v2] sentinel_from_block` set only
  where a test arms W-S1.
- **No PATHS module is imported at module level or at helpers import time**;
  `helpers.mod()` imports it inside test bodies.

## The fixture chains (`helpers.py`)

| scenario | settlements | entries afterwards |
|---|---|---|
| `rich_b1` (`max_judged_per_coldkey_per_settlement = 1`) | B1 = ARM + 900 | X `confirm_pending`/`confirm_pending`; Y (X's coldkey) `pending`/`coldkey_cap_settlement`; Z `pending`/`pending_floor` (no eligible window); L `lost`/`lost_duel`; E `closed`/`ready_expired`; D `closed`/`deregistered` |
| `crowned` | B1, then B2_NEXT_ERA = ARM + 3600 + 900 (the next era) | X `crowned` (KING_D, `since_boundary` = B2_NEXT_ERA); L `lost` at B1 |
| `held` | B1, then B2 = ARM + 1800 with the bench pod down | X `confirm_pending`/`bench_hold`, its confirmation set frozen (24 windows, 4 sources) |
| `progress_b1` (`min_windows = 30`) | B1 | X `pending`/`pending_floor` with 24 windows over 4 sources; Z with none |

## Module surface (`cascade.validator.lane_d_status`)

| name | contract |
|---|---|
| `LANE_D_STATUS_KEY` | `"status/lane_d.json"` |
| `STATUS_SCHEMA` | a non-empty str, the doc's `schema` (value not pinned) |
| `LEDGER_SCHEMA` | `"laned_vault_ledger/1"` |
| `REASON_ENUM` | `tuple(lane_d_state.ENTRY_REASONS)`: the code's per-entry reason enum, never a copy that can drift |
| `build_lane_d_status(cfg, *, state, report, block, era_block=None, era_block_hash=None, store=None) -> dict \| None` | the document (below) |
| `publish_lane_d_status(store, doc) -> str \| None` | the publication (below) |
| `build_vault_ledger(cfg, *, state, reports, block, disputes=()) -> dict` | the ledger (below) |

`cfg` is a loaded `ChainConfig` (`cfg.lane_d`, `cfg.round`, `cfg.v2`).
`state` is the parsed `lane_d_state.json`, `report` the latest signed report
(the one the state's `last_report` binds), `block` the current chain block.

## `build_lane_d_status`

**Pure.** It never mutates `state` or `report`, never writes a file and makes
no store call except the sentinel board's reads (below). Equal inputs give
equal documents, whatever the key order of `state`; no wall-clock field
(`as_of`, `generated_at`, `timestamp`, `updated_at`, `published_at`) appears at
any depth. The document is strict JSON (`allow_nan=False`) and round-trips.

**Unarmed (invariant 1): `None`.** When `cfg.lane_d.from_block == 0`, or the
block is before `from_block`, it returns `None` and calls nothing; there is no
status file. `publish_lane_d_status(store, None)` returns `None` with no store
call. Armed, the document is published (`*_armed_nonempty_lane_d_effect`).

**The report must be the one the state binds.** If `report["boundary"] !=
state["boundary"]` or the sha256 of the report's canonical body
(`lane_d_state.report_digest`) isn't `state["last_report"]["report_sha256"]`,
it raises `ValueError` (an older report, a tampered one). The signature check
is W7b's `verify_report` and isn't pinned here.

### Top-level keys (at least these)

| key | value |
|---|---|
| `schema` | `STATUS_SCHEMA` |
| `lane` | `"closed"` when `until_block > 0` and `block > until_block`, else `"open"` (the closed doc keeps `schema`; its other keys aren't pinned) |
| `block` | `block` |
| `era` | `cascade.shared.era.era_index_of(cfg.round, block)` |
| `boundary` | `state["boundary"]` |
| `reason_enum` | `list(REASON_ENUM)` |
| `floors` | `{"min_windows", "min_clusters"}` from `cfg.lane_d` |
| `king` | below |
| `house` | `{"digest", "reveal_ts"}` of `state["house"]` |
| `prior_kings` | one row per `state["prior_kings"]` entry (`[]` while only the house has reigned: the house is never a prior king) |
| `queue` | below |
| `board` | below |
| `sentinel` | below |

### `queue`

Every entry whose state `status` is open (`pending`, `confirm_pending`, or
`hold`), ordered FIFO by `(reveal_ts, digest)` as queue selection walks them.
Closed, lost and crowned entries are not in it. Each row has at least:

| key | value |
|---|---|
| `digest`, `hotkey`, `context`, `status`, `reason`, `ready_block`, `reveal_block`, `reveal_ts` | the state entry's values |
| `coldkey` | the state's `coldkey_at_ready` |
| `reveal_deadline_block` | `reveal_block + cfg.lane_d.max_reveal_delay_days * 7200` (A9 "maximum reveal delay"; 7200 = 86400 s / 12 s blocks), from the config, never a constant |
| `confirm_stage` | `"confirm"` when the entry holds a first clear (state `confirm` record), else `"first"` |
| `progress` | `{"windows", "clusters", "min_windows", "min_clusters", "frozen"}`: `windows` = `len(frozen_set["window_ids"])`, `clusters` = the number of distinct sources in `frozen_set["sources"]`, `frozen` = `frozen_set["frozen"]`, the floors from the config; `0, 0, False` when `frozen_set` is `null` (after a first clear the confirmation set starts afresh) |

`reason` is always a member of `reason_enum`. The tests reach
`coldkey_cap_settlement`, `confirm_pending`, `pending_floor` and `bench_hold`
through real settlements.

### `king`

`{"digest", "hotkey", "house", "since_boundary", "tenure_blocks"}` (at least):
while the house holds KING_D, `hotkey`, `since_boundary` and `tenure_blocks`
are `null` and `house` is `true`; for a miner king, `since_boundary` is the
state's and `tenure_blocks = block - since_boundary`.

### `board` (the latest signed report)

`{"boundary", "crowned", "judged", "skipped", "verdicts", "gate"}` (at
least): `boundary`, `crowned`, `judged` and `skipped` verbatim from the
report; `verdicts` one row per report verdict, in order, exactly
`{"digest", "king_digest", "stage", "outcome", "lcb", "gain"}`; `gate` the
report's `gate` record verbatim (`null` when absent). Numbers are the report's
floats exactly, never re-derived or rounded.

### `sentinel` (W-S3)

When `era_block` is given and `sentinel.sentinel_armed(cfg, era_block)`,
`sentinel` equals `build_sentinel_board(cfg, era_block=era_block,
era_block_hash=era_block_hash, kings=kings, store=store)`, where `kings` is
`[KING_D digest]` if a miner holds KING_D and
`sentinel.settlement_era_start(cfg, king["since_boundary"]) == era_block`, else
`[]` (the house never appears; a king crowned in another era doesn't). The
only store reads are the board's. Otherwise (`[v2] sentinel_from_block = 0`,
or no `era_block`) `sentinel` is `null` and the store isn't touched.

## `publish_lane_d_status(store, doc)`

The same store interface as the other status files: exactly one
`store.put_text(LANE_D_STATUS_KEY, text, content_type="application/json",
...)` (an `acl` keyword, as `publish_public_json` passes, is allowed), no read,
and it returns the key. `text` parses strictly (no NaN/Infinity, no duplicate
key) to `doc`. The same doc gives the same bytes, and the bytes are identical
in two subprocesses with different `PYTHONHASHSEED`s.

**Never secrets or URLs.** Neither the published text nor the ledger contains
the values of `HIPPIUS_S3_ACCESS_KEY`, `HIPPIUS_S3_SECRET_KEY` or `HF_TOKEN`,
any `://`, `X-Amz-`, `Signature=`, `Credential=` or `presigned`, or the
report's `signature` (§1.2: credentials and URLs are never published).

## `build_vault_ledger` (`laned_vault_ledger/1`, read by W18)

```json
{"schema": "laned_vault_ledger/1", "era": 2704,
 "entries": [{"hotkey": "5F…", "digest": "<64 hex>", "status": "lost", "closed_era": 2700}],
 "disputes": ["<64 hex>"]}
```

- Exactly those four keys; each entry exactly `hotkey`, `digest`, `status`,
  `closed_era`; one entry per state entry (never the house), sorted by
  `digest`. Strict JSON.
- `era` = `era_index_of(cfg.round, block)`.
- **Ledger statuses** come from the state entry's `(status, reason)`:

  | state | ledger `status` | `closed_era` |
  |---|---|---|
  | `pending`, `confirm_pending`, `hold` | `pending` | `null` |
  | `crowned` (KING_D or a prior king: kings are kept permanently, §11.1) | `king` | `null` |
  | `lost` (any reason) | `lost` | era of closure |
  | `closed` / `deregistered` (B8 "voided") | `voided` | era of closure |
  | `closed` / `ready_expired` (B10) | `ready_expired` | era of closure |
  | `closed` / any other reason (`checkpoint_fault`, `lane_closed`, `coldkey_unresolved`, `coldkey_changed`, `reveal_delay_exceeded`, `duplicate_digest`, `hotkey_has_open_entry`, `context_not_armed`) | `closed_unjudged` | era of closure |

- **Era of closure.** `reports` is the persisted report chain
  (`lane_d_report_<b>.json` next to the state). The closing settlement is the
  lowest-boundary report whose `entries` list the digest with status `lost` or
  `closed`; `closed_era = cascade.shared.era.settlement_era(cfg.round,
  boundary).index`. The reports supply only that boundary; the status and
  reason always come from the state. A closed entry no report records is a
  `ValueError` (never a guessed era).
- `disputes`: the given digests, deduplicated and sorted; anything but 64
  lowercase hex is a `ValueError`.
- **Round trip.** The ledger passes W18's `parse_ledger` with the same era,
  entries, statuses and `closed_era`s, and drives W18's dry-run sweep: open
  entries are kept `open_entry`, the king `king`, losers inside
  `loser_ttl_eras` `ttl_not_expired`, a disputed loser `dispute_open`, and due
  losers (`lost`, `ready_expired`, `voided`) are planned for archive and
  delete.

## Invariants 1 and 10

`test_inv1_status_unchanged_when_unarmed` (no doc, no store call, for
`from_block = 0` and for a block before `from_block`) pairs with
`test_inv1_status_armed_nonempty_lane_d_effect` (a non-empty queue, published
once at `status/lane_d.json`).

## Not pinned

- The value of `STATUS_SCHEMA`; extra keys in the doc, rows or board.
- The `prior_kings` row shape; the closed doc's keys beyond `schema` and
  `lane`; the board while `report` is `null`.
- The sentinel `kings` list when two miners were crowned in one era (the state
  keeps only the current KING_D's `since_boundary`).
- Where the publish runs and how often (W8b); the signed lane D index (§11.1).
- Where disputes are registered; the ledger file's location on disk.

## Why each test fails on main

`cascade/validator/lane_d_status.py` doesn't exist. Every test that calls
`helpers.mod()` fails with `ModuleNotFoundError` for
`cascade.validator.lane_d_status`, which is in `PATHS`, and
`test_bytes_identical_across_hash_seeds` fails on the `AssertionError` that
carries its subprocess's `ModuleNotFoundError`. The fixtures (real
`settle_lane_d` runs) and the W18 / W-S3 modules they call are on main, so
nothing fails for another reason.

## Correction (plan rev6.11 (b)): crown at the first verdict

The confirmation duel is dropped (owner 2026-11-23): a crown is decided at
the first verdict, first clear (max-T LCB vs KING_D) ∧ A10 ∧ the §1.3b gate.
There is no `confirm` stage, no `confirm_pending` status or reason and no
`lost_confirmation`; a held first clear is `hold` / `bench_hold`. This WP's
crowning chains move to that rule with nothing weakened. The rich scenario's bench pod is down at B1, so X's first clear is held open (`hold` / `bench_hold`, confirm stage `first`, verdict `cleared`). The king scenario lists X from B2_NEXT_ERA, where it is crowned at its first verdict. The held scenario lists X from B2. The confirm-stage queue and progress tests now pin the held first clear and its judged set, and the reason enum is pinned free of `confirm_pending`. On main the
old code still runs the confirmation stage, so the changed tests fail on an
assertion until the W5e implementation lands.

## Correction (plan rev6.10 (a) / rev6.12 (a)): embargo windows count toward progress

The fixture feeds (W7c's chain) pin `embargo_hours = 48`. Under the W5a
embargo-aware eligibility a target point is eligible when its data time is in
`(b - e, b]` and it is released to miners (`first_seen + e`) after `b`, so Z,
revealed 5 h after X, counts the same 24 windows over 4 sources of snapshot 1
as X. `test_pending_progress_counts_windows_and_clusters_against_the_floors`
now expects `(24, 4, False)` for Z (it expected the rev3 `(0, 0, False)`). X's
row and the floors are unchanged. On main (rev3 rule) the changed test fails
on that assertion until the W5a implementation (#800) lands.
