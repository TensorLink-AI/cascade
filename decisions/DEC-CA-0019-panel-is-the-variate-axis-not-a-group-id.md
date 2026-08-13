---
id: DEC-CA-0019
type: decision
title: "Panel structure IS the variate axis for this model — no group_id payload; the urgent related fix is that the eval's cluster label is missing on the source that dominates the pool"
status: proposed
date: 2026-08-13
tags: [interface, generator, model-arch, panel, bootstrap, clustering, eval-pool]
revisit_when: "the fixed model gains a cross-sample pathway (a group larger than the variate axis can hold, or members trained in different batches) — only then does a group id carry information the (C, L) axis cannot; also revisit if max_channels is ever capped below plausible panel widths, which would make group_id the only expression of a wide panel"
relations: {depends_on: DEC-CA-0015, related: DEC-CA-0021}
---
The corpus is a flat list with no way to say "these 500 series are the same
metric across 500 hosts". Chronos-2's group attention and cross-learning exist
to exploit exactly that, and it is the dominant shape of real observability
data. The distinction drawn against multivariate — a panel is many related
series, not many variates of one system — is right semantically. **For this
model it is the same tensor axis.**

**Why.** Chronos-2's group attention is attention over an arbitrary member set.
`Toto2Model` already routes every 4th layer over the variate axis
(`toto2_model.py:111-115,373-381`), which is attention over an arbitrary,
unordered, position-free member set — the same operation. The trainer's *only*
mechanism for letting series see each other is putting them in one sample's `C`
axis. A panel of K related series is a `(K, L)` sample; there is no second
pathway for a group id to feed.

**DECISION: `group_id` is a reserved name in the DEC-CA-0015 namespace and is
not accepted as payload.** The capability is delivered entirely by DEC-CA-0021's
variate axis. A generator that wants panel structure emits `(K, L)`. Adding a
group id as well would be a second name for the same thing, costing budget and
carrying nothing — the unconsumed-field failure this roadmap refuses.

What a group id *would* additionally buy is cross-**batch** grouping: a group
too wide for the variate axis, or members trained in separate batches. Neither
has a consumer in a model with no cross-sample pathway, and inventing one is a
model-design change, not a submission-surface one.

## The real, live, and currently-broken panel problem is on the eval side

`koth._window_clusters` keys the paired bootstrap on pool metadata `source`,
falling back to per-row singletons (`eval/koth.py:150-160`). `prepare_series`
copies `hs.source` only when the harvester supplies it
(`pool/builder.py:189-190`) — and **only `tsbench_forge` supplies it**
(`sources/tsbench_forge.py:234`). Open-Meteo, which per `docs/EVAL_POOL.md`
dominates the pool at roughly 3000 of ~3000 raw series, emits none. So today
essentially every window is its own cluster, `[scoring] min_clusters = 0`, and
the cluster bootstrap degrades to the classic per-window bootstrap.

That matters because Open-Meteo's global grid **is a panel**: one variable
across ~252 locations, plus 12 correlated variables per location
(`temperature_2m`, `dew_point_2m`, `apparent_temperature`, …). Those windows are
correlated and are currently resampled as if independent, which inflates the
effective sample size behind every LCB the subnet has ever published. This is
the same defect Part C needs fixed before multivariate windows can be scored —
and it is wrong **now**, with no surface change in sight.

**Fix, ahead of everything else on this roadmap:** stamp `source` on Open-Meteo
(by variable, or variable × region — the choice is what "correlated" should mean
here) and Wikimedia, then raise `[scoring] min_clusters` off 0 as
`docs/EVAL_POOL.md` already anticipates. Consensus-relevant: `min_clusters`
gates the verdict, so it lands in lockstep like any scoring change.

**Open question — how much is currently over-claimed.** Re-run an archived
round's paired bootstrap with Open-Meteo windows clustered by variable, and
again by variable × coarse region, against the recorded per-row-singleton LCB.
If the LCB moves by an appreciable fraction of the 0.02 margin, some past
verdicts were decided on evidence the bootstrap over-counted, and the labelling
fix is urgent independently of the submission surface. `cascade-audit` already
replays receipts under alternative rules (DEC-CA-0009's `wql_mode`), so this is
a replay, not a re-eval.
