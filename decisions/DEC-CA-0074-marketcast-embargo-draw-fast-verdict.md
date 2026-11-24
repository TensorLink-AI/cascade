---
id: DEC-CA-0074
type: decision
title: "MarketCast: a verdict within 13 h, every horizon judged on windows drawn from the embargo by the round block hash"
status: accepted
date: 2026-11-24
tags: [marketcast, lane-c, validator, trainer, audit, consensus, contamination]
revisit_when: "the first MarketCast verdicts land: check every entry's judge row lands at the first settlement after its leg, the forward / backtest sub-scores agree in sign, and no generator is refused embargo_series_embedded without cause"
relations: {depends_on: [DEC-CA-0064, DEC-CA-0056], amends: [DEC-CA-0064], relates_to: [DEC-CA-0057, DEC-CA-0063]}
---

## Problem

DEC-CA-0064 judges a MarketCast fine-tune on FORWARD-only windows: 48 per
(asset, timeframe) spread over the 6 days after the leg's dispatch
(`verdict_wait_blocks = 43200`), plus the embargo gap for the two week-long
paths (`gap_horizons = ["4hx42", "1dx7"]`). A miner waits up to six days for a
verdict, and the forward requirement is the only reason.

Owner decisions 2026-11-24:

1. "A verdict within 13 hours of submission."
2. "For the long timeframes, evaluate on windows sampled at random from prior
   history." Refined the same day: "for lane C WE train, so it doesn't matter
   if we don't use forward-only data" — backtest windows are acceptable for
   EVERY timeframe; and "just use the embargo data we already collect, so no
   wait once the fine-tune ends" — the embargo IS the hold-out.
3. Keep the two guards: the trainer's real-data mix never reads the bars the
   windows are drawn from, and the generator is checked for embedded
   embargoed prices.
4. MarketCast is not rolled out, so there is no activation block: this is
   simply the rule.

## Decision

### The embargo is the hold-out

A recipe's real data ends `embargo_days` (30) before its commit, and the
trainer enforces it before anything is rented (review M12): the recipe check
runs against the commitment time, and the bar relay only serves
`[data.start, data.end + 1 day)`, clipped at the dispatch. So the days
`(day(dispatch) − 29 d, dispatch]` are bars the recorder holds but no
training leg ever reads (`embargo_span`). That is the hold-out; nothing new
is excluded from training, and guard (a) is the existing embargo.

### Windows are drawn from it

For every enabled (suite, horizon) — 1m×60 to 1d×7 alike — and every asset
and series, the validator cuts the candidate windows lying wholly in
`(dispatch − embargo_eval_days, dispatch]` (clamped to the embargo), keeps
the phase's half of a fixed public partition of window ids (sha256 tag, bit
0: judge = 0, confirm = 1, so a confirm never re-scores its judge's
windows), ranks them by `sha256("embargo-v1|<round block hash>|<wid>")` and
takes the first `backtest_windows_per_cell` that pass the existing
eligibility rules (liquidity floor, stale context, finite scale — none reads
the realised target). The round block is the settlement that registers the
entry, after the leg ended, so its hash is unknown while the leg trains or
the generator is written. Every entry of a round is drawn with the same hash;
the base, the fine-tune and (on a confirm) the slot holder are scored on the
same windows.

`forward_horizons` (1m×60, 5m×36) also take up to `windows_per_cell` forward
windows that are ALREADY closed in the round's snapshot (start > dispatch +
`start_margin_seconds`, end ≤ dispatch + the wait). They fit in 13 h and
never delay the verdict; a leg that finishes quickly simply has few of them.

### Timing

Everything the judge needs exists at the first settlement after the leg
finishes, so the verdict comes then. `verdict_wait_blocks` now only bounds
how long an inconclusive entry (too few windows / clusters / time blocks)
may wait: default and code cap 3900 blocks (13 h); a larger value from an
older file or an overlay is capped in code. `gap_horizons` is retired (an
older file naming it still loads; the key does nothing).

### Combined score, separate sub-scores

Forward and backtest windows form one row set; the composite, the suite
weights, the per-timeframe weights, the gain test and the floors
(`min_windows` 200, `min_clusters` 8, `min_time_blocks` 8) apply to the
combined set unchanged. The judgement record carries the draw summary and,
reported only, the verdict's numbers on the backtest and the forward windows
separately (`mkt.kinds`).

### Guard (b): the generator contamination check

The trainer scans the recipe tree, when it ships a generator, BEFORE anything
is rented: parsed, never executed (`cascade.interface.contamination_scan.
scan_repo`, lane B's LB1 scanner), against a reference set built from the
signed `pool_mkt` snapshot over the eval span — per asset the close and its
log return (bp) of the hourly and the ≥ 15-minute native banks, 32-step
sub-windows, exact (float64 bytes) and z-normalised (WB11 tolerant z-grid)
digests, lane B's hit thresholds. `embedded` or `budget_exceeded` refuses the
leg as a miner fault, `embargo_series_embedded`; no scanner / no snapshot is
the operator's side (requeued uncharged). The receipt (`embargo_scan`:
result, snapshot digest, span) rides the trainer-signed `marketcast` record;
validators judge an entry only with a `clean` / `no_generator` receipt
(`embedded` = `fault`, absent = `stale`).

### Consensus and audit

The rule is in force wherever MarketCast judges (no activation key: not yet
rolled out). The main validator reads the round block's hash, records the
whole draw in the signed rows (`draw`: rule, block, hash, phase, half, span,
`n`, the forward span and horizons); every validator re-derives the draw from
the trainer's `train_time`, the round block, the phase and `fresh_after` and
treats a mismatch as an infra deferral (as for `after`). `cascade-audit
marketcast` re-derives it too, checks the hash against the chain, bounds
every window by its kind, and with the snapshot re-cuts the exact window set.

### Defaults

| Key | Default | Bounds (code) | Overlay | Why |
|---|---|---|---|---|
| `embargo_eval_days` | 28 | 1 … `embargo_days` − 1 | yes | the widest span inside the 30-day embargo (its first day is the commit day − 29); the 7-day paths need ≥ 2 × 7 days for an aligned window and get ~4 per cell (≈ 2 per half) |
| `backtest_windows_per_cell` | 48 | 1 … 256 | yes | the old `windows_per_cell`: the main validator's per-entry GPU budget is unchanged, and ~4 aligned candidates per long-path cell mean the long paths take all they have |
| `verdict_wait_blocks` | 3900 | ≤ 3900 (13 h) | yes | the owner's 13 h |
| `forward_horizons` | 1m×60, 5m×36 | enabled horizons that fit the 13 h | release | the short paths fit without a wait |

`embargo_days` stays release-only: raising it without a release would let a
leg train on days the draw then scores.

## Leakage risk not closed

- **Regime knowledge.** The embargoed bars are public market data; a miner
  sees them before committing and can tune scalar choices (learning rate,
  augmentation scale, a generator's volatility) to the recent regime. The
  scan catches embedded series, not a few tuned numbers. The forward windows
  and the disjoint confirm half limit, not remove, this.
- **Encodings the scan does not decode**: compressed or encoded byte strings
  (e.g. base64 / zlib of float arrays outside a zip), series derived in code,
  and 1m / 5m-resolution copies (only ≥ 15-minute banks are scanned).
- **The scan's snapshot lags**: `pool_mkt` publishes every 6 h, so up to a
  few hours of embargo before the dispatch are not in the reference set.
- **Judging validators take the block hash from the signed rows**; only the
  audit checks it against the chain.

Closed: the training relay (embargo), warm-start `init` (only this trainer's
checkpoints, dispatched before the submission, whose own data ended before
it), the pinned base (its revision predates any current embargo; a re-pin
must too), the sandboxed generator (no network).
