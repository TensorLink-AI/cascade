# Fixture lane D plan (W0c status self-test)

This file is the plan the status script parses in the fixture repo. Only the
table shapes matter; they copy the real plan's table headers.

## 3. Implementation (work packages)

### 3.0 Work packages

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| W0c | Loop tooling | — | ci | `scripts/laned/accept W0c` |
| W0 | Decision record | — | ci + owner merge | `scripts/laned/accept W0` |
| W1 | Measurement | W0c | ci | `scripts/laned/accept W1` |
| W2 | Config table | — | ci | `scripts/laned/accept W2` |
| W3 | Guard | — | ci | `scripts/laned/accept W3` |
| W4 | Scorer | — | ci | `scripts/laned/accept W4` |
| W5 | Intake | W2 | ci | `scripts/laned/accept W5` |
| W6 | Release-only rule | D99 | ci | `scripts/laned/accept W6` |
| W7 | Eval host | — | ci | `scripts/laned/accept W7` |
| R6x | S0a execution | — | owner | `scripts/laned/accept R6x --evidence evidence/s0a_diff.json` |
| R2x | S0b live run | — | owner (chain read) | `scripts/laned/accept R2x` |

## 4. Testnet plan

**Matrix**

| # | Case | Expected |
|---|---|---|
| T1 | House only | slot burns |
| T2 | Exact copy | refused at dedup |
| T3 | Late reveal | refused |

**T-rows to WPs**

| Row | WPs |
|---|---|
| T1 | W1 |
| T2 | W3 |
| T3 | W99 |

## 12. Lane B variant: miners submit expert weights (parked)

| WP | What | Depends | Runs in | Accept |
|---|---|---|---|---|
| WB1 | Expert-delta format | — | ci | `scripts/laned/accept WB1` |

## 17. Completion rubric and status board

### 17.4 Decision register and owner inputs

| D | Status | Option chosen | Class | Decided |
|---|---|---|---|---|
| D1 | decided | share 0.25 | O | owner, 2026-11-21 |
| D99 | open | — | S | — |

| OA | Status | Unblocks | Evidence file |
|---|---|---|---|
| OA8 | open | W0 notes | `evidence/owner/oa8.json` |
| OA9 | open | W2 history counts | `evidence/owner/oa9.json` |
