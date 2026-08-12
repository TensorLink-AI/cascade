---
id: DEC-CA-0015
type: decision
title: "Promotion members are selected by measured error decorrelation within the quality frontier; cascade_top_k=5 deferred to a coordinated release"
status: active
date: 2026-08-12
tags: [cascade, warm-start, promotion, diversity, selection]
revisit_when: "Two promotion generations have fired under this policy: compare the member sets' realized trajectory divergence (post-promotion round scores per member lineage) against what v1 structural selection would have picked; also revisit when raw GIFT per-config rows are persisted alongside bench reports — the correlation source should then move from the private-pool battery to signed public GIFT vectors"
relations: {depends_on: DEC-CA-0013}
---
DEC-CA-0013's v1 selection arbitrated diversity structurally (prefer a new
generator hotkey, then round spacing) — a proxy for what the owner actually
wants from a top-k member set: **trajectory diversity**, lineages that fail in
different places so rotation explores genuinely different basins.

This decision replaces the proxy with the measurement. Within the unchanged
envelope (quality floor `cascade_quality_epsilon`, ripeness, `cascade_top_k`
cap, provenance — validators verify exactly what they verified before):

- Each candidate carries a per-window error vector (`sqrt(wql_w * mase_w)` on
  the private-pool battery — the same per-window quantity the round geomean
  aggregates; zero-|y| windows masked as the round masks them).
- Vectors are log-transformed and centered per window across the pool before
  correlating: raw errors correlate near 1.0 for any two competent models
  because shared window difficulty dominates (the DEC-CA-0006 lesson), so what
  is correlated is each checkpoint's relative strengths and weaknesses.
- The geomean-best anchors; each slot greedily takes the spaced candidate with
  the lowest maximum residual correlation to the already-chosen set.
  Candidates without vectors fall back to v1 structural ranking and never
  outrank a measured candidate.

Why the private pool and not GIFT (the owner's first instinct): the signed
bench reports carry only the six aggregates — the per-config GIFT rows live in
the raw sidecar on the bench pod, which is torn down after the round, so GIFT
vectors for the backfilled r10–r12 candidates no longer exist and cannot be
regenerated before a firing (~2.5 GPU-hours per checkpoint). The private-pool
battery measures the same divergence signal at finer grain (1978 windows vs 74
configs), costs ~1 CPU-minute per candidate, and works for every candidate
retroactively. When raw GIFT rows are persisted beside future bench reports
(planned), the correlation source can switch to signed public vectors without
touching the selection code — `select_members` is vector-source-agnostic.

`cascade_top_k` stays **3** for now. The owner wants 5, but the k cap is a
consensus constant every validator verifies — netuid 91 has multiple active
third-party validators (uids 0/1/139/9/2/213 as of 2026-08-12), all on the
released chain.toml with k=3, and a 5-member record would make OUR validator
the fork minority. Raising k requires a release-then-activate protocol:
publish chain.toml k=5 + release notice, give validator operators a window of
rounds to update, THEN flip the trainer. (A same-day local flip was deployed
for 25 minutes on 2026-08-12 and reverted before any firing — process lesson
recorded in memory: consensus-constant blast radius goes in the approval ask.)

Vectors are precomputed to `<work_root>/promotion_error_vectors.json` by
`scripts/compute_promotion_error_vectors.py`; the engine reads the cache
best-effort at fire time — a missing cache degrades to v1 selection, never
blocks a ripe firing.
