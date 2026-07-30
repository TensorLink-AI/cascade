---
id: DEC-CA-0010
type: decision
title: "Heat standings publish when the heat settles, mirrored publicly per round"
status: active
date: 2026-07-30
tags: [heat, dashboard, miner-feedback, storage, presentational]
revisit_when: "the heat mirror is ever proposed as an input to anything consensus-bearing (weights, the throne, an audit gate) — it is unsigned, single-writer, best-effort and must stay a view; or heats/index.json outgrows one object (>400 rounds is the current cap, ~a year of daily rounds), at which point it needs sharding rather than a bigger window"
relations: {depends_on: DEC-CA-0006}
---
The heat's standings become public **when the heat settles** — screened, burned,
finalists chosen, before a single duel pod is dispatched — as
`status/heat.json` (live pointer), `heats/round-<id>.json` (immutable per-round
mirror) and `heats/index.json` (a rolling discovery window). Previously the
standings existed only as the presentational `heat` block of the manifest,
which reaches the public solely through a validator's signed receipt.

**Why the receipt was too late.** The heat settles early in the round; the duel
then trains at the full `target_train_hours` at every size, and only then does a
validator score it and publish. A miner's heat placement — the ONLY feedback a
non-finalist ever gets, and the whole point of DEC-CA-0006's per-entrant
diagnostics — therefore landed hours after the fact, routinely after the next
round's submission deadline had already passed. Worse, a round rejected at a
validator gate (`contract_digest_mismatch`, `king_resyncing`, a pool-pin miss)
produces a rejected receipt with no manifest body at all: the field was
screened, the compute was spent, the entrants were burned, and their standings
were never published anywhere. The mirror closes both gaps; the dashboards
(web panel + `cascade round`) and the new `cascade heat` read it.

**Why a separate document rather than an earlier manifest publish.** The
manifest is the signed claim, gated on `contract_digest` and asserted entry by
entry; publishing it before the duel exists would mean signing a manifest whose
entries do not. The heat block was already excluded from
`canonical_body` — an auditor cannot cheaply reproduce a discarded heat
checkpoint — so nothing about it belongs on the signing path. The mirror carries
the identical field shape (`manifest.heat_to_json`), so the dashboards render a
live heat and a settled round's heat through one code path, and the manifest
copy stays exactly as it was: no receipt format change, no audit-trail change.

**Status: presentational, and staying that way.** Unsigned, single-writer (the
trainer), best-effort — every publish is wrapped so a storage failure cannot
disturb the round it describes, mirroring the `status/round.json` stage doc.
Consumers must survive it being absent, stale, or another round's: the join key
is `epoch_start_block` and `live_heat` rejects anything that does not match a
fresh THIS-round doc, falling back to the receipt's block.

**A no-screen round publishes too**, carrying its reason. This is not cosmetic:
without it the live pointer keeps serving the PREVIOUS round's standings, and a
dashboard that joins on nothing would present last round's ranking as this
round's.

**What this deliberately does not do.** It does not change what the heat ranks
on, what advances, or what the duel scores (DEC-CA-0006 still holds: the screen
ranks on the observed geomean, the bootstrap stays a shadow diagnostic). It adds
no new information beyond what the manifest block already carried — including
the raw `crps`/`mase` per entrant, published by owner decision on 2026-07-26 —
it only makes the same numbers arrive in time to be useful.
