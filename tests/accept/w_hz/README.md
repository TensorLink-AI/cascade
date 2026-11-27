# W-HZ acceptance tests: per-window horizons (GIFT-Eval bands)

Owner decision 2026-11-27: lane D scores each window at its OWN horizon,
drawn per window from the GIFT-Eval terms of its series' frequency, on both
halves (the live `pool_d` forward windows and the public GiftEvalPretrain /
RMISC windows). These tests are the contract; the builder may touch only the
paths in `PATHS`.

## The bands (`cascade/shared/lane_d_horizons.py`, new)

Source: `SalesforceAIResearch/gift-eval` `src/gift_eval/data.py` at
`1b520f911c24e3f45d990a6dc966fadab2ba475b`:

- base horizon `PRED_LENGTH_MAP = {M: 12, W: 8, D: 30, H: 48, T: 48, S: 60}`;
  GIFT-Eval's quarterly and yearly series are M4's, so `Q: 8`, `A: 6` from
  `M4_PRED_LENGTH_MAP`;
- `Term.multiplier`: short 1, medium 10, long 15
  (`prediction_length = multiplier * base`);
- lane D: `h = min(max_horizon, max(32, ceil(multiplier * base / 32) * 32))`,
  `max_horizon = 2048`.

| base | short | medium | long |
|---|---|---|---|
| S | 64 | 608 | 928 |
| T, H | 64 | 480 | 736 |
| D | 32 | 320 | 480 |
| W | 32 | 96 | 128 |
| M | 32 | 128 | 192 |
| Q | 32 | 96 | 128 |
| A | 32 | 64 | 96 |

`gift_base_freq(freq)`: GIFT-Eval's `norm_freq_str(to_offset(freq).name)`
(multiple and anchor dropped), after a tick frequency is written in its
largest exact unit (`21600S` -> H, `240S` -> T, `150S` -> S). Sub-second and
unknown frequencies have no band. The equal-weight cell of a window is
`"<frequency class>|<band>"`, class subdaily (S/T/H), daily, weekly, monthly,
quarterly, yearly.

## Config (`[lane_d]`, release-only keys)

`horizon_bands = ["short", "medium", "long"]` (default; `[]` = the fixed
`horizon` rule, byte for byte the pre-band behaviour), `max_horizon = 2048`
(a multiple of 32, <= 2048), `band_slots = 1`, `min_cell_windows = 50`.
`horizon` stays the fixed rule's 64 (<= 512). All four are in
`RELEASE_ONLY_KEYS` (never an overlay).

## Live half

- `pool_d` band snapshot: per series the newest `target_points` points form
  the target region (the tail of `hist.npz` then `tgt.npz`), with both clocks
  per point in `clk.npz` (`(2, target_points)` int64), bound by
  `metadata.clocks.sha256`. The snapshot digest stays the three-file digest;
  a pre-band reader loads the snapshot unchanged (its 64-point window is the
  region's newest 64 points). `region_points`: the longest band of the
  frequency within the raw finite tail, leaving `min_context` history.
- `load_pool(..., bands)`: per band whose `h` fits the region, the newest
  `band_slots` non-overlapping targets; window id `<series>:<target start>:h<h>`,
  history = every point before the cut. A pre-band snapshot's region is its
  stored target.
- Eligibility is the §1.3 / W2e rule on every one of the window's `h` target
  points: each point embargo-held at the entry (first seen within the feed's
  `embargo_hours` before the bound) or arrived after it; a target may mix
  both. A feed with `embargo_hours = 0` (public at its source) has no credit.
- Draw (`draw_band_windows`): one stream from the boundary block hash; cells
  `"<class>|<band>"`, quota = the live draw's even split, one permutation of
  each cell's fresh then embargo candidates, picks round-robin over the cells,
  never a target overlapping (in data time) a window already drawn from that
  series; then the source cap, then `balance_cells` (cells below
  `min_cell_windows` left out, the rest to equal counts) and the fresh share,
  repeated until stable.
- Forecasts: one eval host call per (digest, h); the eval host's horizon cap
  is 2048 (`MAX_HORIZON`, 64 patches).

## Public half

`draw_public(..., bands=)`: each series draws its band, uniform over the bands
its length fits, by `H(":band:", ordinal, 0)`; slots and cuts use that `h`
(the full range of historical data is allowed). The box balances the kept
public windows to equal counts per cell (`balance_public`); the windows left
out are drops with reason `public_cell_balance`.

## Report (backward compatible)

`params` is unchanged (the deployed checker compares it to its own config);
the band rule is the optional top-level `horizon_bands`; windows carry
`band`, cohorts `cells` / `cells_dropped`, verdicts `cells` (point gain per
`"<half>|<class>|<band>"`). Equal counts per cell make the unweighted mean
every checker re-derives from the rows the equal-weight-per-cell statistic.
`test_the_old_checker_pays_a_band_report` runs the deployed checker
(`git archive 09f19baa`; the checkers of 09f19baa / a4600dfe / 00f3139c are
one blob) on a new band report: no burn-class finding (its only finding is
the alert-class `ld_embargo`, which expects h = 64).
