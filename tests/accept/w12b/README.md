# W12b acceptance tests: `cascade status --json` and `discovery-check` lane D fields

W12b (plan rev6 §3.0 W12b row) depends on W2 (`[lane_d]` config) and W17
(the `status/lane_d.json` producer). These tests are the spec. The builder
may touch only the paths in `PATHS`. The W17 tests must stay green.

## What the plan says

The W12b row, verbatim:

> | W12b | `cascade status --json` and `discovery-check` lane D fields | W2, W17 | ci | `scripts/laned/accept W12b` |

The W17 row (the producer these fields read), verbatim:

> `status/lane_d.json` producer (queue, pending progress, king, board;
> **rev3:** per-entry reason enum including `coldkey_cap_settlement`,
> `confirm_pending` and reveal deadline). **(rev6.2, owner 2026-11-22)** Also
> produces the W18 vault ledger `laned_vault_ledger/1`: the current era, the
> vault entries with their statuses and `closed_era`, and the open disputes

§11.3 Support: "`status/lane_d.json` shows each pending entry's
eligible-window progress against the floors." §3.2 invariant 1: "Unarmed ⇒
byte-identical." Invariant 10: every "unchanged when unarmed" test has a
paired "armed ⇒ non-empty lane D effect" test.

## Today's code (main)

- `cascade/miner/live_status.py`: `SOURCES` names five public objects
  (`status/chain.json`, `status/v2.json`, `funded/latest.json`,
  `status/lane_b.json`, `status/lane_c.json`); `status_snapshot` returns
  `lanes` and `queues` for lanes a/b/c only and one wall-clock `freshness`
  entry (from the doc's `as_of`) per source; `fetch_snapshot` adds a
  `sources` URL per source.
- `cascade/miner/discovery_check.py`: `check_snapshot` adds a
  `lane_<x>_queue` freshness check for lane a and every active lane b/c.
- `cascade/validator/lane_d_status.py` (W17): `LANE_D_STATUS_KEY =
  "status/lane_d.json"`, schema `lane-d-status/1`, keys `schema, lane
  ("open"|"closed"), block, era, boundary, reason_enum, floors, king, house,
  prior_kings, queue, board, sentinel`. **No wall-clock field** (W17 pins
  that: the producer reads no clock). Unarmed, W17 builds `None` and publishes
  nothing.

Nothing in the CLI reads `status/lane_d.json` yet.

## Contract

### `cascade status --json` (`live_status.py`)

1. `SOURCES["lane_d_queue"] == "status/lane_d.json"` (equal to W17's
   `LANE_D_STATUS_KEY`); the five existing entries are unchanged and there is
   no other new source. `fetch_snapshot` therefore reads it (one more public
   GET) and `sources["lane_d_queue"]` is `<endpoint>/<bucket>/status/lane_d.json`.
2. `queues["d"]` is the fetched `status/lane_d.json` document **verbatim**
   (queue rows with their reason enum, progress and reveal deadline, king,
   board, sentinel, as W17 wrote them), or `None` when absent.
3. `lanes["d"]` is `None` when the document is absent; otherwise a dict with
   at least `active` (`True` iff the document's `lane == "open"`) and the
   document's own `lane`, `schema`, `block`, `era`, `boundary`, `king`,
   `house` and `floors`, unchanged.
4. `freshness["lane_d_queue"]` is on the **block clock**, because W17's
   document has no wall-clock stamp: a dict with at least `available`
   (document present), `block` (its `block`), `age_blocks`
   (`status/chain.json` `current_block` − `block`, or `None` when either is
   not an int) and `fresh`, which is `True` iff the document is present,
   `age_blocks` is an int and `0 ≤ age_blocks ≤ status/chain.json
   era_blocks` (one era; a future-dated document is never fresh). Lane D is
   published per settlement, so a document older than one era is stale.
5. Lane D never changes lane A's view: the top-level `fresh` and
   `ready_for_training` still depend only on chain + competition, a stale or
   absent lane D document never makes the snapshot stale, and `warm_start`,
   `chain`, `king` and lanes a/b/c are what they would be without it.
6. `status_snapshot` is deterministic and does not mutate its inputs.
7. The CLI (`cascade status --json`, no wallet) carries all of the above in
   `data`, and still exits 0 when `status/lane_d.json` is absent.

### `cascade discovery-check` (`discovery_check.py`)

8. When `lanes["d"]` is a dict with `active` true, `check_snapshot` adds
   `checks["lane_d_queue"] = freshness["lane_d_queue"]["fresh"]`; so a stale
   or future-dated lane D queue lands in `remaining` and the release check
   fails, with `writes_performed: false`.
9. A closed lane (`lane == "closed"`) or an absent document adds no
   `lane_d_queue` check.

### Unarmed (invariant 1) and its pair (invariant 10)

- `test_status_snapshot_unchanged_when_unarmed` /
  `test_status_snapshot_armed_nonempty_lane_d_effect`: with no lane D
  document every field today's snapshot has keeps today's value (`fresh`,
  `ready_for_training`, the five freshness entries, lanes/queues a–c), lane D
  reads as absent; armed, the queue, the summary and a fresh block clock are
  present and lane A's view is identical.
- `test_discovery_checks_unchanged_when_unarmed` /
  `test_discovery_checks_armed_nonempty_lane_d_effect`: unarmed, the check
  set is exactly today's (`fresh_feeds, current_init, effective_contract,
  agent_protocol, submission_tracking, lane_a_queue`); armed it gains only
  `lane_d_queue`, every other check unchanged.

## Fixtures

Every lane D document is real W17 output (`build_lane_d_status`) over the
W17 "rich" chain (a real `settle_lane_d` at B1 on the W7c synthetic pools,
numpy only), reused through `tests/accept/w17/helpers.py`. The closed-lane
document is the same state under `until_block` = the next era boundary, at a
block past it. The public bucket is the only external service:
`cascade.miner.dashboard.fetch_public_json` is replaced by a dict of
published objects for the CLI tests.

On main every armed test fails with `AssertionError` (the accessors in
`helpers.py` assert the W12b fields exist); the unarmed tests pass.

## PATHS

- `cascade/miner/live_status.py`: `SOURCES`, `status_snapshot`,
  `fetch_snapshot` (`:17-139`).
- `cascade/miner/discovery_check.py`: `check_snapshot` (`:33-51`).
- `tests/unit/test_live_status.py`: pins today's five reads and the a/b/c
  queue set exactly (`:30`, `:82`, `:93`, `:113`); the new source changes
  them.

Not in scope: `cascade lane-d status` (submission tracking, W10) and the
`llms.txt` lane list (W12a).
