---
id: DEC-CA-0018
type: decision
title: "Miner feedback breaks down by facet: a per-cell weakness profile off the heat's existing paired scores, shadow-only"
status: proposed
date: 2026-08-19
tags: [heat, miner-feedback, diagnostics, eval, presentational, dashboard]
revisit_when: "the breakdown is ever proposed as an input to the ranking, the finalist pick, the duel or weights (it is a view — a facet score that decides anything becomes a facet to pad); or the published per-cell shares are observed to move the field's corpora toward the pool's composition rather than toward broader coverage (then drop to heat_breakdown_shares = false and keep only gap_share); or miners stop reading it (then stage 3 never happens and stage 2 can be retired); or the shape bucket edges need re-cutting (bump facet_schema and publish both for a transition round — never redefine in place)"
relations: {depends_on: [DEC-CA-0009, DEC-CA-0011], relates_to: [DEC-CA-0006, DEC-CA-0010]}
---
A miner's feedback is a scalar. The heat standings give `rank`, `rel_score`,
`p_best` and the raw `crps`/`mase` components — *how far off* a generator was,
never *on what kind of data*. That is the only question a DGP author can act
on, and the answer is the difference between "you are 4.8% behind" and "three
quarters of your deficit is intermittent daily series". Finalists get slightly
more (`per_domain_win_rate` on the duel receipt: an unweighted win count, one
challenger, hours later); the rest of the field gets nothing.

DECISION: publish a **weakness profile** — the heat score decomposed over
facets of the eval slice, per entrant. Four families: `domain` (already on
`WindowScore`), `freq`, a computed `shape` family (seasonal strength / trend /
intermittency / volatility / context length, cut at fixed scale-free
thresholds from each window's own history), and `qband` (a decomposition of the
CRPS half over quantile bands, free from the already-per-quantile
`qloss_per_q`). Each cell reports `n`, `n_clusters`, `share`, `rel_leader`,
`rel_median`, a cell-restricted paired bootstrap `lcb`, and `gap_share` — the
cell's signed contribution to the entrant's total deficit, which sums to 1 over
a family and is therefore the prioritisation number (a small gap over many
windows outranks a large gap over four).

Three properties make this cheap and safe rather than a new eval:

**It is a pure reduction.** `_run_heat` already holds `list[WindowScore]` per
entrant, every entrant scored on the same slice in the same order (that pairing
is what `_screen_diagnostics` and `joint_bag_geomeans` already assert). Zero
extra GPU, no second scoring path; one family's per-cell bootstrap touches
exactly N rows, so at `heat_breakdown_B = 2000` the whole layer costs a
fraction of the screen's existing joint bootstrap.

**The decomposition is exact, and only because of [[DEC-CA-0009]].** With both
halves of the round metric now per-window geometric means, `log g_total` is a
share-weighted sum of per-cell `log g_c` over any partition; under the old
pooled MWSQL no such identity existed and `gap_share` would have been a
fiction. The identity is a unit test, not a claim.

**It rides an existing presentational channel.** [[DEC-CA-0011]]'s heat
document (`status/heat.json`, `heats/round-<id>.json`, the manifest's unsigned
heat block) with the round-level field reference published once and each
entrant's own numbers on its row. No `contract_digest` change, no signed-body
change, no receipt format change, no consensus surface. Config lives under
`[telemetry]`, on [[DEC-CA-0010]]'s precedent.

SHADOW ONLY, permanently: the profile gates nothing, weights nothing, reorders
nothing — the same posture as [[DEC-CA-0006]]'s `p_best`. A per-facet number
that decided anything would be padded rather than fixed.

Two confounds are handled in the statistic itself, not in the caption.
`rel_median` (you against the field's median in the same cell) separates *your*
weakness from a cell the whole field finds hard — without it every miner would
chase the pool's hardest data. And cells under 8 windows or 2 clusters publish
counts with the numbers suppressed and `underpowered: true`: never silently
dropped (absence must not read as competence), never merged into a neighbour
(that would move a miner's history under them).

What is deliberately NOT broken out: `source`, `series_id` and the forge
catalog's `dgp_class` — those name the private pool's contents or its internal
taxonomy. The computed `shape` family carries the same actionable signal
without naming anything private, and it is reproducible by miners offline,
which the catalog labels are not. Cluster counts are published; cluster
identities never are.

The composition leak is acknowledged and accepted as bounded: the profile does
reveal the round's slice composition at coarse facet resolution, which the
aggregate does not. The pool rotates daily and the slice per round; the shares
at bucket resolution restate what `docs/EVAL_POOL.md` already says
qualitatively; and knowing shares helps a miner *cover* the distribution, which
is the intended behaviour, while the risk the private pool actually guards —
memorising specific series — is untouched by them. `heat_breakdown_shares =
false` is the retreat if that reading proves wrong.

STAGED. Stage 1 is `cascade/eval/facets.py` plus `cascade score --breakdown`:
the reduction, its tests, and the *local* loop against a miner's own pool
(buildable today from public sources with `cascade-pool build`) — nothing
published, nothing on the wire. The fast loop is the larger share of the value:
the published profile arrives once per round, the local one every time a
generator changes. Stage 2 publishes on the heat document and adds `cascade
heat --breakdown`. Stage 3 — a duel-side per-facet breakdown on the receipt
(drop-when-default per [[DEC-CA-0012]]'s convention, since archived receipt
bodies must stay byte-identical) and horizon-segment facets (which need
per-step quantile loss retained) — happens only if stage 2 is used.

Spec: `docs/WEAKNESS_PROFILE.md`.
