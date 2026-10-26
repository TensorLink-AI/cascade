---
id: DEC-CA-0065
type: decision
title: "Optional per-channel groups for the Cascade MoE Base generator contract"
status: accepted
date: 2026-10-27
tags: [interface, generator, trainer, architecture, shards]
relations: {supersedes: [DEC-CA-0024], depends_on: [DEC-CA-0020, DEC-CA-0026, DEC-CA-0053]}
---

## Decision

An accepting Cascade pyramid contract allows a generator to yield
`{"values": float_array(C,L), "groups": uint8_array(C)}`. The yield still has
8–32 channels; each group may contain 1–C of them. Labels must begin at zero
and introduce each new label in order of first appearance, without gaps.
`[0,1,0,2]` is canonical; `[0,2,0,1]` and `[1,1,2,2]` are not. Validation
checks integer dtype and labels before conversion to uint8, preventing overflow
or fractional truncation. Grouping is part of the seeded generator output,
not an operator-controlled draw. Group-size distributions live in the miner's
`config.json`; there is no protocol-wide distribution or group-count reward.

This supersedes only DEC-CA-0024's deferral of trainer-side group handling and
its statement that no corpus/trainer change is needed. Its separate per-series
`group_id` stays reserved and refused. No grouping spans yield or batch rows.

## Consumption and compatibility

Batches remain bucketed by patch count and channel count. There is no channel
padding. Time crops preserve the full groups vector; bare rows in a mixed
batch receive all-zero labels. Cross-variate attention uses a boolean equality
mask per row and patch. Time attention, normalization, CPM, routing, budgets
and the marginal/energy objective definitions do not change. Routing auxiliary
statistics still operate across the full optimizer batch; group isolation is
an attention/output property, not separate optimization of each group's loss.

Copula samples draw an independent shared latent for each group, sharing it
only among that group's channels. The energy score remains per channel and
32-point patch. Real benchmark inference supplies no groups and retains its
existing joint-channel semantics. Shadow partner telemetry compares channels
only within their declared group; singleton groups count as unpartnered.

No field, or `groups=None`, collapses to the original values-only path.
All-zero groups use unmasked attention and the original copula RNG sequence.
An explicit groups vector participates in corpus/stream/dedup digests even
when it contains only zeroes. It is never silently ignored.

Shard schema 2 adds nullable uint8 label bytes per row. Shards with no labels
retain schema 1, its exact writer bytes and index shape. Mixed corpora may
contain both shard schemas; absent labels read as bare arrays. Contract field
acceptance is checked again by the shard stream before training.

## Release and audit

Interface version 2 announces this field. Effective accepting contracts expose
version 2; historical contracts retain their interface policy. Only the pyramid
backend may arm `groups`. Mainnet/testnet v2 and the four pyramid presets arm
it explicitly. Empty `accepted_fields` still drops from the contract digest;
the digest omission rules and all existing values-only goldens stay unchanged.

The model-source bytes change, so all four pyramid architecture pins change.
The accepting training contract digests change as well. This is a new release
and checkpoint lineage: rebuild and qualify CPU/worker images and publish a
matching baseline. The checkpoint guard remains strict; do not relabel old
checkpoint code or bypass verification. Historical Toto2/MoE source is untouched.

Validators consume signed artifacts under the era contract; ordinary settlement
does not execute a miner's generator again. Tier-1 shard audit verifies stored
shard hashes and re-derives the consumed stream digest, including groups. Full
regeneration from generator, child seeds and recorded prefix counts remains a
separate operator audit. Local verification draws twice and checks determinism
of both values and group labels; it cannot prove all future seeds deterministic.
