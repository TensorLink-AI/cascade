# The weakness profile — where a generator's data is failing

**Status: design (not implemented).** Decision node:
`decisions/DEC-CA-0018-generator-weakness-profile.md`. This doc is the
implementation spec: facet taxonomy, statistic, published shape, surfaces,
staging and test plan.

## The gap this closes

A miner's feedback today is a **scalar**. The heat standings
(`status/heat.json`, `cascade heat` — DEC-CA-0011) give every entrant its
`rank`, `rel_score` (`heat_score / best`), `p_best`, and the two raw error
components `crps` / `mase` on the round's slice. That says *how far off* a
generator was. It says nothing about *what kind of data it was off on* — and
that is the only question a generator author can act on. "You are 4.8% behind
the leader" is not a work item. "Three quarters of your deficit is on
intermittent daily series, and you are ahead of the leader on strong-seasonal
hourly" is.

A finalist gets marginally more: the duel receipt carries
`per_domain_win_rate` (`{domain: (win_rate, n)}`, `eval/koth.py`), which is a
win/loss count per domain, unweighted by how much was lost, computed only for
the one challenger that reached the duel, published hours later. Non-finalists
— most of the field, every round — get nothing.

**The data to fix this already exists in memory.** The heat screener returns
`list[WindowScore]` per entrant (`trainer/loop.py::_run_heat` → `components`),
every entrant scored on the *same* window slice in the same order, and each
`WindowScore` already carries `domain`, `source`, `mase`, and the per-quantile
`qloss_per_q` / `abs_target`. The breakdown is a **pure reduction over data the
round has already paid for**: no extra GPU, no extra eval pass, no second
scoring path.

## 1. Facets

A *facet family* partitions the round's scored rows; a *cell* is one bucket of
one family. Family definitions are pure, deterministic, versioned
(`facet_schema`), and live in a new `cascade/eval/facets.py` so a miner runs the
identical code on their own pool.

| Family | Cells | Source | Cost |
|---|---|---|---|
| `domain` | `weather`, `web_traffic`, `energy`, … | `WindowScore.domain` (already there) | free |
| `freq` | `sub_hourly`, `hourly`, `daily` | window metadata `freq` / `seasonal_period` | one field added to `WindowScore` |
| `shape` | `seasonal_strong` / `seasonal_weak`, `trending` / `flat`, `intermittent` / `dense`, `volatile` / `smooth`, `short_context` / `long_context` | computed from the window's **own history** | one pass of numpy per window |
| `qband` | `lower` (q .1–.3), `median` (.4–.6), `upper` (.7–.9) | slices `qloss_per_q`, already per-quantile | free |

`shape` is the family that maps onto generator knobs directly — a DGP author
reads "intermittent: 1.9× the leader" as *my prior emits no zero-inflated or
count-like processes*, and "upper: 1.4×" as *my noise model is too clean, the
model never learned wide upper intervals*. `domain` and `freq` are coarser and
mostly useful for spotting that a whole family of real-world data is missing
from the corpus.

`shape` buckets are cut at **fixed, published thresholds** (seasonal strength
from the ACF at `seasonal_period`; trend from a robust slope over residual
scale; intermittency from the zero/flat-run fraction; volatility from the CV of
first differences; context length in units of `seasonal_period`). They are
scale-free by construction, so they mean the same thing on a temperature series
and on a pageview series.

`qband` is not a window partition — it decomposes the CRPS half *within* every
window. It is kept in the same output shape because it answers the same
question in the same units.

**Deliberately not facets:** `source` (the upstream feed id), `series_id`, and
the tsbench-forge catalog's `dgp_class`. Those name the private pool's contents
or its internal taxonomy; the computed `shape` family carries the same
actionable signal without naming anything private. Cluster **counts** are
published, cluster **identities** never are. Horizon-position facets (early vs.
late in the horizon) are excluded from stage 1 for a different reason: cost —
`qloss_per_q` is summed over the horizon in `mwsql_components`, so per-segment
loss means widening the component record. See stage 3.

## 2. The statistic per cell

Each cell reports the **same functional form as the ranking**,
`global_geomean` restricted to the cell's rows:

```
log g_c = ½ · [ mean_{i∈c, wql defined} log wql_i  +  mean_{i∈c} log mase_i ]
```

This choice is what makes the breakdown *arithmetic* rather than merely
suggestive. Because DEC-CA-0009 made **both** halves per-window geometric
means, the round's headline score decomposes **exactly** over any partition:

```
log g_total = ½ · Σ_c [ (m_c/M)·L_wql,c + (n_c/N)·L_mase,c ]
```

(`m_c/M` = the cell's share of windows with a defined WQL — DEC-CA-0009 masks
zero-`sum|y|` windows from that half — and `n_c/N` its share of all windows.)
Under the pre-DEC-CA-0009 pooled MWSQL no such decomposition existed. Note that
the weights are identical for every entrant: they are all scored on the same
windows, and `joint_bag_geomeans` already asserts `abs_target` equality across
competitors, so the valid masks agree by construction.

Fields per cell, per entrant:

- `n`, `n_clusters` — the evidence behind the cell. `n_clusters` is the honest
  effective sample size (windows from one feed are correlated).
- `share` — `n_c / N`, the cell's weight in the headline score.
- `rel_leader` — `g_you,c / g_leader,c`. The leader is the field's overall best
  entrant, the same reference in every cell, so cells are comparable.
- `rel_median` — `g_you,c / median_e g_e,c`. **This is the confound control.** A
  cell where the *whole field* is 3× worse is a hard cell of the pool, not your
  weakness; `rel_leader` alone would send every miner chasing the pool's
  hardest data.
- `gap_share` — the cell's signed contribution to your total deficit,
  `Δ_c / Δ` where `Δ = log g_you − log g_leader` and `Δ_c` is that cell's term
  in the decomposition above. Sums to 1 over a family. **This is the
  prioritisation number**: a cell where you are 8% worse over 100 windows is
  worth more work than one where you are 2× worse over 4. Negative values are
  strengths and are kept (they have to be, for the sum to hold).
- `lcb` — paired cluster-bootstrap lower bound of leader-over-you *restricted to
  the cell*: is this gap real, or is it four windows of noise? `≤ 0` renders as
  "not separated".
- `crps`, `mase` — the cell's raw components, gated by
  `[telemetry] heat_breakdown_absolute`.

**Underpowered cells.** A cell with `n < heat_breakdown_min_cell` (default 8) or
`n_clusters < 2` publishes its counts with the numbers suppressed and
`underpowered: true`. It is never silently dropped — absence must not read as
competence — and never silently merged into a neighbouring cell, which would
move a miner's history under them.

**Bootstrap cost.** The per-cell LCB reuses `joint_bag_geomeans` with the row
stack masked to the cell and clusters intersected. Summed over one family's
cells the resample touches exactly `N` rows, so **one family at `B` bags costs
the same as one joint bootstrap at `B`** — and the facet layer runs at
`heat_breakdown_B` (default 2000), a fifth of the screen's `bootstrap_B`. It is
a shadow of a shadow; it does not need 10k bags.

## 3. Where it is computed and published

Computed in `trainer/loop.py::_run_heat`, immediately after
`_screen_diagnostics`, off the `components` dict that is already assembled and
already paired. Wrapped exactly like the existing diagnostics: **a breakdown
must never cost a round its heat** — any exception logs and the standings
publish without it.

Published on the DEC-CA-0011 channel, unsigned and presentational:

- round level, once: `breakdown_ref` — per cell, `n`, `n_clusters`, `share`,
  the leader's and the median entrant's `g`. The field reference lives here so
  it is not repeated 25 times.
- per entrant: `breakdown` — per cell, only that entrant's own numbers.

Both ride `manifest.heat_to_json`, so the manifest's heat block, `status/
heat.json` and `heats/round-<id>.json` carry the identical shape and the
dashboards keep one render path (DEC-CA-0011's constraint). Consumers read by
key, so an archived document without the new keys loads unchanged, and an older
consumer ignores them. Payload at 25 entrants × ~16 cells ≈ tens of KB.

Nothing about this touches `contract_digest`, the signed manifest body, the
receipt, weights, or the throne.

## 4. Miner surfaces

- **`cascade heat --hotkey <you> --breakdown`** — your profile, sorted by
  `gap_share` descending: the work list, biggest payoff first. Columns: cell,
  `n`, `rel_leader`, `rel_median`, `gap_share`, and a flag for
  underpowered / not-separated cells.
- **`cascade score --breakdown`** (local, offline, minutes) — the same
  `facets.py` on the miner's own pool, with the fetched king
  (`cascade fetch king`) standing in for the field leader. This is the loop that
  actually matters: the published profile arrives once per round, the local one
  arrives every time you change your generator. Miners can already build a
  facet-labelled pool from public sources with `cascade-pool build` — no new
  data infrastructure is needed for this half.
- `cascade round` stays compact; it gains one hint line pointing at
  `cascade heat --breakdown`.
- Dashboard: the heat panel gains an expandable per-entrant profile.

**Honest caveat to print next to the numbers:** the heat trains a small model
at `heat_train_hours` (0.5h) on the primary size. A weakness at the heat budget
is directional evidence about the final's budget and sizes, not proof.

## 5. Anti-gaming and what this leaks

1. **Never a selection rule.** Facet numbers gate nothing, weight nothing,
   reorder nothing — not the heat ranking, not the finalists, not the duel, not
   weights. Same posture as DEC-CA-0006's `p_best`. A per-facet score that
   *decided* anything would immediately become a target to pad, and the field
   would optimise the taxonomy instead of the data.
2. **No private identifiers.** Coarse `domain` (already public via
   `per_domain_win_rate` on receipts) and computed shape buckets only. Never
   `source`, `series_id`, `dgp_class`, or per-window rows. A regression test
   asserts the serialised document contains none of them.
3. **Min cell size** — under 8 windows or 2 clusters, counts only. This serves
   both statistical honesty and privacy: a 2-window cell is close to naming a
   series.
4. **The composition leak is real, bounded, and mostly aligned.** The breakdown
   does reveal the round's slice composition at facet resolution, which
   aggregate `crps`/`mase` does not. Three things bound it: the pool rotates
   daily and the slice rotates per round, so shares are a moving target; at
   coarse-bucket resolution the shares restate what `docs/EVAL_POOL.md` already
   says qualitatively; and knowing shares only helps a miner **cover** the
   distribution better, which is the behaviour the subnet wants ("forecast
   generally"). The contamination risk the private pool actually guards is
   *memorising specific series* — facet shares are no help there at all. Escape
   hatch if the owner later disagrees: `heat_breakdown_shares = false` publishes
   `gap_share` and the rels without the raw counts.
5. **Presentational, best-effort, single-writer.** Unsigned, like everything
   else on this channel. It must never become an input to anything
   consensus-bearing.
6. **The facet schema is frozen and versioned.** Re-cutting a bucket edge
   rewrites every miner's read of their own history. Bump `facet_schema` and
   publish both families for a transition round rather than redefining in place.

## 6. Staging

**Stage 1 — the reduction and the local loop.** `cascade/eval/facets.py` (facet
assignment, per-cell reduction, masked-bootstrap LCB), `freq` on `WindowScore`,
unit tests, and `cascade score --breakdown`. Nothing published, nothing on the
wire, no trainer change. Miners get the fast iteration loop against their own
pool on day one.

**Stage 2 — publish.** Trainer computes it in `_run_heat`; `breakdown_ref` +
per-entrant `breakdown` on the heat document and the manifest heat block;
`cascade heat --breakdown`; dashboard panel; `docs/MINER.md` section.

**Stage 3 — only if stage 2 gets used.** Duel-side facet breakdown for
finalists on the receipt (drop-when-default in `_verdict_body`, the DEC-CA-0012
convention — the receipt body must stay byte-identical for archived rounds),
generalising `per_domain_win_rate` from a win count to a weighted gap; and
horizon-segment facets, which need per-step quantile loss retained in the
component record.

Config, all under `[telemetry]` (DEC-CA-0010's precedent: never
`contract_digest`):

```toml
[telemetry]
heat_breakdown = true          # stage 2 publish switch
heat_breakdown_B = 2000        # bootstrap bags for the per-cell LCB
heat_breakdown_min_cell = 8    # windows below this → counts only
heat_breakdown_absolute = true # per-cell crps/mase (aggregates are already public)
heat_breakdown_shares = true   # publish raw per-cell window counts/shares
```

## 7. Test plan

- **Facet assignment is pure and deterministic** — golden tests on synthetic
  windows with known structure (a pure sine lands in `seasonal_strong`, a
  zero-inflated count series in `intermittent`, …).
- **The decomposition identity holds** — `Σ_c weighted log g_c == log g_total`
  to floating-point tolerance, and `Σ_c gap_share == 1`. This is the property
  that makes `gap_share` mean what the CLI says it means; if it ever fails, the
  prioritisation is a lie.
- **Underpowered suppression** — a 3-window cell publishes counts, no numbers.
- **Failure isolation** — a facet reduction that raises leaves the `HeatResult`
  published and complete, exactly as `_screen_diagnostics` does today.
- **Schema tolerance** — heat JSON round-trips with and without the new keys; a
  fixture document predating them loads byte-identically.
- **Privacy regression** — the serialised heat document contains no `source`,
  no `series_id`, no raw window values.
