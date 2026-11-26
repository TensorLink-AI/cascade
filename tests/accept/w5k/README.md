# W5k acceptance tests: the public-source draw (rev6.18 A1, A2, A4; F3)

W5k is new in plan rev6.18 (F3 row). Each lane D settlement draws, per
public source, a window set from that source's W6p index and the settlement
block hash; the same windows serve KING_D, the house and every challenger.
Module: `cascade/validator/lane_d_public.py`; wiring:
`settle_lane_d(..., public_indexes={source: index dir})` records the report's
`public` block (W7a). Scoring the windows is exposed here; the 50/50
combination with the live half is F4.

Accept: `scripts/laned/accept W5k` (CI). Fixtures: `f3_fixture.py` reaches
`tests/accept/w6p/fixtures.py` lazily (an `AssertionError` on a main without
F3). Nothing here imports torch; pyarrow writes the tiny fixtures.

## The draw (pinned; `draw_public`)

Inputs: the index (W6p), `source`, the boundary block hash, `horizon` =
`[lane_d] horizon` (64), `n_windows` = 5000 (A1), `m` = 128 history points.

1. A series of `L` points has `k = max(0, (L - m) // h)` target slots;
   `k = 0` series are never drawn.
2. Cells = `"<family>|<freq>"`; families merge year-split folders (W6p
   `family_of`: `cmip6_*`, `era5_*`, `largest_*`). Capacity = the series with
   `k >= 1` (GiftEvalPretrain, one window per series) or `Σ k` (RMISC,
   `multi_window`).
3. Quota: `cascade.validator.lane_d_draw.split_quota` (the live W5j rule:
   smallest capacity first, scarce cells whole, surplus re-split).
4. Cells in key order, rounds `r = 1, 2, …` (one round without
   `multi_window`): the pool is the series holding `r - 1` windows with
   `k >= r`, ranked by `H("pick", ordinal, r)` (ties by ordinal); the first
   `remaining` get one cut each:
   - one window per series: `c = m + H("cut", ordinal, 1) mod (L - h - m + 1)`;
   - several: slots `m + o + j·h` (`j < k`), `o = H("offset", ordinal, 0) mod
     (L - m - k·h + 1)`; round `r` takes the unused slot at
     `H("slot", ordinal, r) mod #unused` (ascending). Targets never overlap.
5. `H(kind, o, r) = int(sha256(f"{seed}:{kind}:{o}:{r}"))`, `seed =
   f"{block_hash}:lane_d_public:{source}"` (hash ranking: numpy-independent,
   re-derivable by any auditor from the index).

Window id `"<source>:<series_id>:<cut>:h<h>"`; the history is everything
before the cut (≤ 131072 points), the target the next `h` points.

## Reading and scoring

- `read_public_windows(draw, index_dir=, data_root=, block_hash=,
  max_channels=)` reads ONLY the drawn series (W6p `read_series`) and builds
  each window. A `(C, L)` series is one multichannel window under F2's rule
  (`cascade.eval.lane_d_channels`: empty channels dropped, at most
  `max_channels` by the block hash). Drops, identical for every model:
  `public_nonfinite_target`, `public_short_history` (finite history suffix
  < `m`), `public_no_channel`, `public_no_scale`.
- `score_public_windows(windows, {digest: {window_id: q}})` →
  `{digest: {source: [{window_id, family, crps, mase}]}}` through F2's
  `score_window`; a missing forecast is a `ValueError`.

| Item | Test |
|---|---|
| the spec rule, bound to the block hash, deterministic | `test_the_draw_is_the_spec_rule_and_bound_to_the_block_hash` |
| 5000 per source (A1) | `test_budget_is_5000_windows_per_source` |
| equal share per (family, freq), live rule, family merge | `test_equal_share_per_family_and_frequency_reuses_the_live_rule` |
| one window per series, cut inside the series | `test_one_window_per_series_with_the_cut_inside_the_series` |
| RMISC: several non-overlapping windows per series | `test_small_rmisc_folders_give_several_non_overlapping_windows_per_series` |
| reads only the drawn series | `test_reads_only_the_drawn_series` |
| (C, L) windows under F2's channel rule | `test_multichannel_windows_follow_the_f2_channel_rule` |
| drops | `test_unscorable_windows_are_dropped_for_every_model` |
| per-source per-window scores via F2's API | `test_public_windows_score_through_the_f2_per_window_api` |
| one draw for every model | `test_one_draw_serves_the_king_the_house_and_every_challenger` |

## Hippius mirror reader (owner 2026-11-24, issue #894)

`read_public_windows(draw, index_dir=, reader=, ...)`: with `reader(row) ->
(C, L)` (the eval host's `PublicMirror.reader`, W6p) the series come from
it instead of `data_root`; it is called only for the drawn rows and the
windows / drops equal the local read's. Neither is a `ValueError`. The draw
itself is unchanged. `test_w5k_reader.py`.
