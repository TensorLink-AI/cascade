---
id: DEC-CA-0016
type: decision
title: "Eval-pool composition is published — aggregate domain x granularity counts only"
status: active
date: 2026-08-14
tags: [eval-pool, dashboard, miner-feedback, privacy, presentational]
revisit_when: "there is evidence miners distribution-match the eval using the published mix (e.g. a generator whose synthetic curriculum tracks the breakdown's proportions round-over-round and gains from it) — the remedy is coarsening or delaying the doc, never publishing more; or the doc is ever proposed as an input to anything consensus-bearing — it is unsigned and must stay a view"
relations: {depends_on: DEC-CA-0011}
---
Each `cascade-pool publish` also mirrors the snapshot's **aggregate shape** to
the public manifest bucket (`status/pool.json`, `cascade.shared.pool_status`):
a rolling window of per-snapshot summaries — `effective_block`, data cutoff,
tar sha256, and series counts per **domain x granularity** (the builder's
`per_domain_freq`). The web dashboard's **Eval pool** tab and `cascade pool`
render the snapshot governing the current round plus the history of prior
pools; both map round → snapshot by the SAME greatest-`effective_block` rule
validators select by, so the tab names the pool a validator would actually
score on.

**Why publish anything about a private pool.** The pool's privacy lever is
about series *content* — fresh, unpublished data that cannot be memorised or
distribution-matched. The domain/granularity *mix* is already public in
substance: EVAL_POOL.md documents the sources and their cadences, the forge
catalog is a public repo, and RETIRED snapshots are fully revealed to HF weeks
later. Meanwhile miners were tuning generators blind against "7 GIFT domains,
30S–D" prose that drifts from the actual build. Publishing the counts closes
that asymmetry (the owner and anyone who diffs revealed snapshots already had
them) without handing over a matching target.

**The line: aggregate counts, nothing else.** No series identities, no
per-series source labels, no values, no per-feed anything. Those would give a
generator the distribution-matching target the privacy lever exists to deny.
The doc also stays **presentational and unsigned**: the signed manifest's
`eval_pool_key`/`eval_pool_sha256` pin remains the audit record (the doc
carries the same sha256 precisely so readers can cross-check it against the
pin), and the write is best-effort — a manifest-bucket failure warns and never
fails the snapshot publish validators depend on.
