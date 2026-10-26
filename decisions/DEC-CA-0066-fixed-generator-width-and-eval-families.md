---
id: DEC-CA-0066
type: decision
title: "Fixed 32-channel v2 generators, broader private pools and source-family evidence"
status: accepted
date: 2026-10-27
tags: [generator, eval, pool, bootstrap]
relations: {amends: [DEC-CA-0026, DEC-CA-0041, DEC-CA-0065]}
---

## Generator policy

The mainnet/testnet v2 generator policies now require exactly 32 channels per
yield. Optional `groups` may divide those channels into any canonical groups,
including singletons. The record validator retains its general 8–32 structural
range for earlier accepting configurations; the effective generator policy
adds the exact 32-channel floor. Cached corpora containing narrower synthetic
yields cannot be reused for a new leg. The starter emits 32 channels and refuses
other configured widths. Historical pre-v2 admission retains its prior floor;
archived generator/model source is not rewritten. A new compliant launch
baseline must pass preflight. Champions remain untouched.

Private eval accepts native widths up to 32. Narrow real series remain valid;
channels are never duplicated to fill a width. A multivariate window still
counts once, not once per channel. Public benchmark data and metrics keep their
upstream definitions; this decision does not force them into 32-channel arrays.

## Pool and verdict budgets

New CLI builds default to 800 series per domain and 400 per domain/frequency
cell. Existing explicit operator flags still override defaults, so deployed
cron commands naming 600/300 must be updated separately. Both increases are
needed when cell caps would otherwise prevent a domain from filling.

`[v2] eval_n_windows = 4000` sets the mainnet private verdict budget for
settlements judging v2 eras. The settlement at the v2 boundary still judges
the preceding era and keeps its earlier budget. An absent/zero override is
inert; testnet retains its small budget. The same resolver drives Lane A,
Lane B and audit replay. The horizon ladder divides the larger budget equally
among its rungs, subject to eligible series capacity.

In the single-horizon v2 path the explicit budget supersedes the legacy
`mix_target_windows` and series-bag cap. Domain/class selection still rotates
and sampling remains without replacement. A smaller pool produces fewer real
windows, never invented evidence. Historical draws are unchanged. Increasing
pool size alone does not increase the number judged per settlement.

## Source-family bootstrap

New pool builds default to `--cluster-by source_family`. The producer records
the family in metadata `source` (the existing bootstrap key), retains the
original feed as `catalog_source`, and also records `source_family`. This works
with existing window/score transport: the same family moves together across
venues, panel rows, channels and forecast horizons. Validators never infer
families from names or network calls. Historical immutable snapshots keep
their recorded source keys; new snapshots must be future-effective.

The forge catalog may set `source_family` explicitly. Without a curated label,
known crypto-market feeds are conservatively one market family, FX feeds one
FX family, known sensor networks one network, and other feeds one endpoint
origin. Declared clean/noisy pairs are joined transitively. This deliberately
sacrifices apparent evidence rather than treating provider siblings as
independent. Metadata without an origin or curated family is marked unresolved
and refused by the family build mode. `--cluster-by source` reproduces earlier
producer semantics; it is not the new primary-pool recommendation.

These keys are conservative dependence controls, not proof of independence
between all remaining families. Review catalog provenance and the resulting
cluster count before publishing; existing `min_clusters` still applies. More
correlated series do not manufacture independent evidence, and the corrected
LCB may be wider even when more windows are evaluated.

## Release checks

No model tensor geometry/source pin changes beyond DEC-CA-0065. Generator
bounds, pool defaults and v2 eval policy change; all validators must use the
same future-era policy and snapshot. No live pool, cron, chain configuration,
provider account or checkpoint is modified by these repository changes.
The actual pool size and real 32-channel GPU performance require a local
mirror rebuild and subsequent authorized integration run.
