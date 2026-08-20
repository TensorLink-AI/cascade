---
id: DEC-CA-0024
type: decision
title: "For the pinned architecture, a panel IS a variate group — group_id stays reserved; panel capability rides the multivariate arm"
status: proposed
date: 2026-08-13
tags: [interface, generator, trainer, eval, pool]
revisit_when: "panels materially larger than a practical max_channels are the demonstrated competitive shape (then batching-by-group becomes a real trainer feature and group_id gets a consumer); or the arch pin moves to a model with cross-example group attention distinct from its variate axis (Chronos-2-style), which would give group_id semantics the variate axis cannot express"
relations: {depends_on: [DEC-CA-0020, DEC-CA-0026]}
---

Gap 4: the corpus is a flat list with no way to say "these 500 series are the
same metric across 500 hosts"; Chronos-2's group attention and cross-learning
exploit exactly that; and it is correctly noted this is not multivariate — a
panel is many related series, not many variates of one system.

**The distinction is real in general and vacuous for this architecture.**
That collapse is the decision.

## Does the fixed model's grouped attention have somewhere to use a group id?

Yes — and it is the variate axis itself. Toto2's variate-attention layers
(`toto2_model.py:111` `layer_axis`; `_Block` with `axis="variate"`) attend
**fully and without positions** over the channel axis — "variates are
unordered" (`toto2_model.py:246`). Unordered, exchangeable, full attention
over a set of related series is precisely group attention over panel members.
The architecture does not distinguish "500 hosts' CPU metric" from "500
variates of one system" because *nothing in it orders or types the channel
axis*. Batch elements, meanwhile, are strictly independent — there is no
cross-example mechanism a separate `group_id` could feed.

So for the pinned model, the way a miner expresses a panel is: **yield the
panel as one `(C, L)` series** once `max_channels` rises. The corpus schema
already carries the axis (`drain_generator` canonicalises to `(C, L)`); the
"panel gap" and the "multivariate gap" are one gap with two names, and every
hard question — per-series C, budget pricing, rank-collapse, eval — is
answered in DEC-CA-0026.

What is genuinely lost by this collapse, stated honestly:

- **Panels wider than `max_channels`.** A 500-host panel at a cap of 8 must
  be chunked into 8-member groups; cross-chunk structure is invisible to the
  model. But it would be invisible anyway — no mechanism spans batch
  elements — so the cap, not the missing field, is the binding constraint,
  and raising the cap for wide panels is a compute question (variate
  attention is O(C²) per step) that belongs to the same measurement as the
  rest of DEC-CA-0026.
- **Alignment metadata.** Panel members sharing a clock is expressed by
  sharing one series' time axis — which `(C, L)` enforces for free (uniform
  L). Cross-*series* alignment would need the time anchor (DEC-CA-0021),
  another reason `start` is reserved.

## Decision

- `group_id` stays a **reserved, refused** field (DEC-CA-0020 table). It
  gets accepted the day a consumer exists, and per the analysis above no
  consumer can exist under the pinned arch — the revisit conditions are an
  arch move or a demonstrated need for trainer-side group batching.
- No corpus-schema change beyond DEC-CA-0020. No trainer change beyond
  DEC-CA-0026's. No `chain.toml` change.
- Eval side: real panel data enters the pool as multivariate windows
  (DEC-CA-0026 Stage E1) and as `source`-labelled clusters — the KOTH
  bootstrap's cluster key (`koth.py:150`) is already the right instrument
  for "these windows are correlated because they share an upstream feed",
  which is the eval-side shadow of panel structure. Guaranteeing `source`
  on pool windows (already planned — `[scoring] min_clusters` comment,
  `chain.toml:322`) is the one concrete action this gap adds.

Migration: none. Digest: none. Gaming surface: none of its own — a panel
submitted as `(C, L)` inherits DEC-CA-0026's channel economics and gates.

## Rank

#4 as proposed is fine, but it is not independently rankable: its entire
substance ships inside gap 6. Treat it as a corollary, not a stage.
