# W5j acceptance tests: the seeded per-bin window draw

W5j (plan §3.0 W5j row; rev6.10 (b); rev6.11 (a), (b); rev6.12 (a), (b)) makes each
settlement's lane D windows a seeded random draw, equal per (timeframe, bin)
cell, from the boundary block hash. King and cohort get the identical draw.
These tests are the spec; `helpers.reference_draw` is the algorithm as code.
The builder may touch only the paths in `PATHS`. W5a / W5b / W2e tests must
stay green.

## Module contract: `cascade/validator/lane_d_draw.py` (new)

`draw_windows(cells, *, boundary_block_hash: str, cfg, budget: int | None = None) -> dict`

### Inputs

`cells` is a list of cell dicts:

- `timeframe`: the series' pool `freq` string, verbatim;
- `bin`: `"forward"` for the one sub-daily h = 64 cell of a timeframe
  (rev6.9's slow bins `"short"`, `"medium"`, `"long"` come with W5i);
- `kind`: `"forward"` (W5i adds `"backtest"`);
- `h_lo`, `h_hi`: the bin's horizon range in points (`64, 64` for a fast cell);
- `series`: a list of `{"series_id", "source", "ts", "eligible", "e0"}`.
  `ts` are the series' point timestamps in order. `eligible[i]` says point i
  passes the window's rule (W5a, the effective `e`), and `e0[i]` says it
  passes at `e = 0`. A forward window at origin index `i` and horizon `h` is
  a candidate iff `eligible[i:i+h]` are all true. It is an **embargo
  window** iff it is a candidate and some `e0[i:i+h]` is false.

`cfg` is the parsed `LaneDConfig`. The draw reads `max_windows_per_settlement`,
`min_fresh_window_share`, `max_source_share`, `min_windows` and
`min_clusters`. `budget` (default: `cfg.max_windows_per_settlement`) is the
part of the settlement's window budget left for this cohort.

The input is never mutated, and its order doesn't matter: cells are taken
sorted by key `"<timeframe>|<bin>"`, series by `series_id`, origins ascending.

### Algorithm (all of it is pinned)

1. **Seed.** `seed = f"{boundary_block_hash}:lane_d_windows"`; one stream
   `np.random.default_rng(_seed_to_int(seed))` (`cascade/eval/bootstrap.py`),
   consumed only in the order below.
2. **Budget.** If `budget < cfg.min_windows`, nothing is drawn: the result has
   `deferred = "window_budget"`, no windows, and every quota 0.
3. **Capacity** of a cell: the number of non-overlapping candidates at `h_lo`,
   i.e. `Σ_series Σ_runs floor(run / h_lo)` over the maximal runs of eligible
   points.
4. **Quota.** `budget` is split by lane A's even pick (`_pick_series`,
   `windows.py`): while cells remain and budget remains, `share = max(1,
   want // len(remaining))`, take the remaining cell with the smallest
   capacity (ties by key), give it `min(capacity, share)` (the last cell
   `min(capacity, want)`), and subtract. Scarce cells are taken whole; the
   surplus is re-split. Cells never reached get 0.
5. **Picks**, cells in key order, up to `quota` each: `h = h_lo` when
   `h_lo == h_hi`, else `int(rng.integers(h_lo, h_hi + 1))`; the valid
   candidates are every `(series, origin)` at that `h` whose target points
   don't overlap a window already drawn from that series (any cell), ordered
   by `(series_id, origin index)`. **Fresh first** (rev6.12 (b)): if any
   valid candidate is fresh (all its `e0` points true) the pick is
   `fresh[int(rng.integers(len(fresh)))]` over the fresh ones only, else it
   is drawn the same way from the embargo ones; none left ends the cell (no
   refill).
6. **Source cap.** `cap_source_share(picks, max_share=cfg.max_source_share,
   seed=seed)` (W5b); the dropped windows are not refilled.
7. **Minimum fresh share** (rev6.12 (b) step 3, `s =
   cfg.min_fresh_window_share`). With `F` fresh and `E` embargo windows left,
   keep the largest `K ≤ E` with `F ≥ s · (F + K)`: the first `K` embargo
   windows in draw order (the last-drawn are dropped first). No refill. (So
   `s = 0.5` keeps `K = min(E, F)`, `s = 1` keeps none, `s = 0` keeps all.
   The plan text writes the bound as `F × s / (1 − s)`; that equals this at
   the default 0.5, and the tests pin the share rule the plan states, "at
   least `s` of the set is fresh".)
8. **Floors after.** `meets_floors` = at least `min_windows` windows from at
   least `min_clusters` distinct sources, on the final set. If they're missed
   but the set before step 7 met them, `short_reason = "fresh_share_short"`
   (the entry stays pending; the share is never relaxed).

### Result (plain JSON)

- `seed`, `boundary_block_hash`, `max_windows_per_settlement` (from `cfg`);
- `capacity`, `quota`: `{"<timeframe>|<bin>": int}` for every input cell;
- `windows`: sorted by `window_id`, each with `window_id`
  (`"<series_id>:<origin>:h<h>"`, the §1.3 identity), `series_id`, `source`,
  `timeframe`, `bin`, `kind`, `h`, `origin` (the target start timestamp) and
  `embargo_window`;
- `fresh_windows`, `embargo_windows`: the kept fresh / embargo counts;
- `meets_floors`: bool; `short_reason`: `None` or `"fresh_share_short"`;
  `deferred`: `None` or `"window_budget"`.

There is no confirmation draw (rev6.11 (b)).

## Settlement wiring (`cascade/validator/lane_d.py`)

One draw per cohort over its intersection eligible set (never per entry),
with the settlement's `boundary_block_hash`. The judged windows are the draw,
so KING_D, the house and every member are scored on the identical window ids.
The signed report carries a top-level `draw` record per settlement
(`boundary_block_hash`, `seed`, `max_windows_per_settlement`, and per cohort
the quota and capacity), and every report window carries `timeframe`, `bin`,
`h`, `origin`, `kind` and `embargo_window`. The total judged windows never
exceed `max_windows_per_settlement`.

## Why the tests fail on main

`cascade.validator.lane_d_draw` doesn't exist (a `ModuleNotFoundError` for a
path in `PATHS`); the settlement tests fail on an `assert` (no `draw` record,
no per-window draw fields, all eligible windows judged).

## Correction: feeds pin `embargo_hours` (W5a fail-closed)

W5a (rev6.10 (a)) refuses every window of a listed feed whose
`embargo_hours` is missing (`unlisted_feed`). The settlement fixture
(`settle_helpers.write_pool`) now pins `embargo_hours: 0.0` on every feed, so
every window stays eligible and fresh once W5a lands; on main the key is
ignored. No assertion changed.
