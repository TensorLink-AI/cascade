# W-EVALPUB: publish each settlement's judged eval set once it is retired

Owner 2026-11-28: "publish the eval dataset for lane D after it occurs, and
lane C: not 30 days". Hard rule: publish only windows / bars that can NEVER
be judged again in any later settlement.

## How each lane draws, and whether it re-uses windows

**Lane D (live half).** Every settlement re-draws (`lane_d_draw.draw_band_windows`,
seed `"<boundary block hash>:lane_d_windows"`) from the pinned `pool_d`
snapshots' candidates eligible for the cohort (`lane_d_forward`): bound
`b = max(T_E, T_K) + forward_margin` (0 h), a window is eligible when every
target point has `data_ts > b - e` and `first_seen > b - e` (`e` = the feed's
embargo, at most `[lane_d] embargo_hours` = 48 h), plus S0 / backfill. A member
is never judged twice on a window (`used_windows`), but OTHER entries can be:
the window stays eligible for every still-open entry and, as an embargo
window, for every entry revealed up to `e` after its target was first seen.
So **the current design re-uses windows; immediate publication would leak**
(`test_laned_judged_window_is_redrawable_so_immediate_publication_would_leak`).

*Retired* (`window_retired`): with `T = min(now, state mtime, reveal_ts of
every open entry) - 6 h` (#997's `protect_ts`) and `cut = T - 3600 E`
(`E` = max(deployed, template) `embargo_hours`), every target point's data
time and first-seen time is `<= cut`. Every published value (history precedes
the target) then has data time `<= cut <= b - e` for every bound `b >= T`, so
`data_ok` fails for any window holding it, in any snapshot or band
(`test_laned_retired_values_never_enter_an_eligible_window`: every window of
two snapshots holding a published value is ineligible for every reveal in
`[T, T + 400 h]` at e = 0, 12, 48 h). **Delay:** the last judged target point's
first-seen + 54 h: at most ~54 h after the settlement (its targets were seen by
it); longer only while an older entry is still open.

The public half (GIFT-Eval pretrain, RMISC) is a pure function of the block
hash and a public index: anyone re-derives it; nothing to publish.

**Lane C (MarketCast).** Each entry's draw (`pool.marketcast.embargo_draw`):
backtests wholly inside `(dispatch - embargo_eval_days, dispatch]` (28 d), in
the judge / confirm half; forwards after `dispatch + start_margin`. A bar judged
for a leg dispatched at `d` lies in the backtest span of every leg dispatched
in the next ~28 days: **re-use by design**
(`test_mkt_draw_reuses_bars_for_the_embargo_and_the_cut_retires_them`).

*Retired*: the round's `snapshot.end <= cut_c = T_c - 86400 E_c - 1`, `T_c =
min(now, train_time of every MarketCast entry still in the validator's lane C
state) - 72 h` (legs dispatched, not registered yet), `E_c` = the largest
`embargo_days` / `embargo_eval_days` (deployed, template, recorded on open
entries). Every later draw starts after `t - 86400 eval_days > cut_c`. Fail
closed on a missing / malformed `train_time`. **Delay: ~33 days.**

## Minimal design change for immediate publication (NOT implemented: consensus)

Both lanes: a **retired-span registry** derived from the signed reports.
Every validator adds each judged window's `(series / asset, data-time span of
its published values)` at the settlement that judged it, and every later
draw drops any candidate whose target overlaps a registered span (lane D:
`_cohort_set` / `draw_band_windows` candidates, the frozen sets; lane C:
`cut_windows(keep=)`). Published history must be inside the span too (or
capped to points already past `cut`). Then a window is retired at its
settlement and can be published right after it. Cost: fewer fresh candidates
per settlement (lane D's recent 5-min feeds; lane C's 28-day span shrinks by
what was judged). Rollout: release-then-activate (`*_from_block`) on lane C;
lane D per the lanes B/C/D defaults rule.

## Files

- `lane-d/evals/<boundary>/{windows.npz, README.md, manifest.json}` +
  `lane-d/evals/index.json` (items + the boundaries still waiting).
  `windows.npz` is built on the lane D eval host
  (`cascade.eval.settlement_eval_build`, niced, its pinned copies re-hashed
  against the report's `snapshot_sha256`), streamed to the box's stage and
  re-verified there; the box loads no pool.
- `marketcast/evals/<round>/{windows.json, README.md, manifest.json}` +
  `marketcast/evals/index.json`; bars = the public post-embargo bars.
- `public-read` on `lane_b/round-*`, `lane_c/round-*` (also at write, trainer)
  and `lane-d/reports/*` (backfill; `acl_done.txt`).

Uploads are `public-read`, `manifest.json` last; a set whose manifest exists is
never rebuilt; indexes are rewritten only when they change.
