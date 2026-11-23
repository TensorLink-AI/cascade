# W5i acceptance tests: slow-cadence backtest windows, owner bins, clamped weight

W5i (plan §3.0 W5i row; rev6.9 (a); its draw is rev6.10 (b) / W5j) adds the
owner's horizon bins for daily-and-slower series, backtest hold-out windows
for every slow bin except daily short, and a combined statistic whose
backtest part is capped in weight and clamped in ratio. These tests are the
spec; `helpers.py` holds the bins table and the reference draw. The builder
may touch only the paths in `PATHS`. W5j's and W2e's tests must stay green.

## `cascade/validator/lane_d_backtest.py` (new)

- `HORIZON_BINS: dict[str, dict[str, tuple[int, int]]]`: the rev6.9 table,
  pool `freq` (exact match, today's `LANE_D_LOWFREQ_FREQS`) to
  `{"short": (lo, hi), "medium": ..., "long": ...}`, inclusive, every bound
  `<= 512`.
- `freq_class(freq) -> str`: `"daily"` (`D`), `"weekly"` (`W`), `"monthly"`
  (`M`, `ME`, `MS`), `"quarterly"` (`Q`, `QE`, `QS`), `"yearly"` (`A`, `Y`,
  `YE`), else `"subdaily"`.
- `cells_for_freq(freq, cfg) -> list[dict]`: the cell headers
  (`timeframe`, `bin`, `kind`, `h_lo`, `h_hi`) a series of that `freq` feeds.
  - A sub-daily `freq`, **or any `freq` while `cfg.backtest_weight_cap == 0`**
    (an explicit off; the shipped default is 0.05, ON: nothing built or scored, daily stays h = 64 forward):
    one `{"bin": "forward", "kind": "forward", "h_lo": cfg.horizon,
    "h_hi": cfg.horizon}`.
  - A slow `freq` with the cap > 0: its three bins in the order short,
    medium, long. Daily short is `kind "forward"` (through W5a's rule at the
    drawn h); every other slow bin is `kind "backtest"`.
- `backtest_weight(n_fwd, n_bt, cap) -> float`:
  `c = min(cap, n_bt / (n_bt + n_fwd))`, 0 when `n_bt == 0`.
- `combined_log_ratio(r_fwd, r_bt, *, c, floor) -> float`:
  `(1 - c)·log r_fwd + c·log clamp(r_bt, floor, 1/floor)`.

## Backtest cells in `draw_windows` (`cascade/validator/lane_d_draw.py`)

A cell with `kind "backtest"` carries series `{"series_id", "source", "ts"}`
(every point of the series in the boundary's pinned `pool_d` snapshot). Its
window at horizon h is the hold-out of the last h points: origin index
`n - h`, `origin = ts[n - h]`, needing `n - h >= 32` (one patch of history).

- Capacity: the series with `n - h_lo >= 32`.
- Picks (in the W5j stream, cells in key order; backtest windows are neither fresh nor embargo): `h = int(rng.integers(h_lo,
  h_hi + 1))`, valid = the cell's series (sorted by id) not yet picked in this
  cell with `n - h >= 32`; none ends the cell; else
  `valid[int(rng.integers(len(valid)))]`. The forward non-overlap rule does
  not apply to backtest windows.
- `max_source_share`, the minimum fresh share and the floors count **forward windows
  only**; backtest windows are kept as drawn and are never embargo windows.
- `skipped`: `{"series_id", "timeframe", "bin", "reason":
  "backtest_short_history"}` for each backtest-cell series with
  `n - h_lo < 32`, sorted by (cell key, series id).
- Every window also carries `freq_class`.

## The verdict (`cascade/validator/lane_d.py`, `cascade/eval/bootstrap.py`)

`judge_cohort(king, members, *, clusters, cfg, seed, kinds=None)`: `kinds`
is a per-window list of `"forward"` / `"backtest"` (`None` = all forward).
Per sub-score `R_x = geo_challenger / geo_bar` (`_geo` over that sub-score's
windows), `c = backtest_weight(n_fwd, n_bt, cfg.backtest_weight_cap)`,
`log R = combined_log_ratio(R_fwd, R_bt, c=c, floor=cfg.backtest_ratio_floor)`,
`gain = 1 - exp(log R)`. The clamp is applied inside **every** bootstrap
replicate of the max-T cohort LCB, so the bound holds for the LCB too.

- `c == 0` and all windows forward: the output equals today's exactly
  (`gain = (kg - cg) / kg`, the same LCB).
- Perfect memorisation of the backtest windows with forward equal gains at
  most `1 - floor ** c` and doesn't clear at the defaults
  (`1 - 0.85 ** 0.05 ≈ 0.0081 < d_margin 0.01`).
- Each member's result adds `forward` and `backtest` sub-scores
  (`geomean`, `n`).

## Config

`backtest_weight_cap` (default **0.05**, ON at launch: owner 2026-11-23, W2e #781, plan rev6.15; 0 switches the backtest off) and `backtest_ratio_floor` (0.85): the
loader refuses a cap > 0 unless `1 - floor ** cap < d_margin` (W2e pins the
full matrix; one case is repeated here).

## Not covered here

The settlement-level report fields for backtest windows and the audit /
replay re-derivations are W7a / W9 / W7d test work; the W5i row also runs a
testnet check.

## Why the tests fail on main

`cascade.validator.lane_d_backtest` and `cascade.validator.lane_d_draw`
don't exist (`ModuleNotFoundError` for paths in `PATHS`); `judge_cohort` has
no `kinds` parameter (an `assert` on its signature).

## Correction (owner 2026-11-23): the cap is ON by default

`backtest_weight_cap` defaults to 0.05 (W2e #781, plan rev6.15), not 0. The
text above no longer calls the backtest "off by default"; 0 is an explicit
off. `test_w5i_the_default_cap_builds_the_slow_bins` pins the parsed default
(0.05) and that a slow `freq` then feeds its three bins. Every other test
passes its cap explicitly and is unchanged. On main `lane_d_backtest` does not
exist, so the new test fails with `ModuleNotFoundError` (in PATHS), like the
rest of this dir.
