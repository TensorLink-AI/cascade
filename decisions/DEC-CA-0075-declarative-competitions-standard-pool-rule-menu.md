---
id: DEC-CA-0075
type: decision
title: "Lane C competitions are declarative specs over a standard pool; tests, pay curves, embargo and the contamination hash set are overlay choices"
status: proposed
date: 2026-11-24
tags: [lane-c, lane-b, lane-d, marketcast, netconfig, consensus, validator, trainer, audit, pool]
revisit_when: "the first overlay that changes a spec, a menu choice or the embargo is adopted: check every satellite pushes the main validator's vector and the audit re-derives each recorded curve / embargo; and when the first spec instance (a new competition) is adopted"
relations: {depends_on: [DEC-CA-0057, DEC-CA-0064, DEC-CA-0073, DEC-CA-0074], amends: [DEC-CA-0057, DEC-CA-0064, DEC-CA-0071], relates_to: [DEC-CA-0056]}
---

## Problem

Owner 2026-11-24 (firm): "new test or formula and embargo days etc. should be
up to our choice, and we should bring new lane C competitions without
validators updating." Clarified the same day: lane C is the umbrella and
MarketCast is one competition in it; a new domain must not need a validator
release — not even for its data source.

Validators must never execute owner-supplied code (six external validators;
an overlay is data, not code). So "our choice" has to be a choice among
blocks every validator already runs.

## Decision

### 1. A competition is a spec composed of registry blocks

`[lane_c.specs.<id>]` (overlay-settable, DEC-CA-0057) names a runner, a pool
(prefix + signer + schema version), a series selection (standard-pool tags),
the universe / timeframes / horizons, metrics from the registry, the score and
guards, metric weights, `gain_margin`, `alpha`, a test family and a pay curve
(item 4), the competition's weight inside lane C, `until_block` and its
embargo. Absent keys keep the runner's own table. The registry
(`cascade/shared/competition_spec.py`, published by `overlay_policy()`):

| Block | Entries in this release |
|---|---|
| runner | `marketcast` (multi-instance: `mkt` + any spec id, W-CMP2) |
| series metric | `returns`, `vol`, `range`, `high`, `low`, `liquidity`, `ohlcv`, `funding`, `alpha` (each names the pool channels it reads) |
| score | `crps` |
| guard | `mase`, `qlike`, `tail` |
| pool schema | `cascade-std-pool/1` (+ "" = legacy snapshot or standard pool) |
| test family | `plain_lcb`, `maxt`, `bonferroni` |
| pay curve | `tiered_decay`, `fixed_term`, `indefinite_until_beaten` |

Code bounds (`SPEC_BOUNDS`, W-OV style) cap every numeric key. A spec is
compiled onto its runner's tables and parsed by the SAME parser and checks as a
hand-written table, so a spec can never reach a value the tables could not.
MarketCast's spec restates today's config byte-identically (tests pin it).

**Refusal, never a crash or a fork.** An unknown block, unknown key, missing
runner or unreadable schema is a typed `SpecRefusal`: that competition judges
nothing, its weight stays in lane C's split so its part burns, and the refusal
is logged and recorded. Out-of-bounds values of known keys are refused on the
overlay path before any node signals the overlay; unknown NAMES are not, so a
node one release behind burns that one competition's share instead of
rejecting the whole overlay (its weights differ from an up-to-date node's by
that share only).

### 2. One data format: the standard pool

Every lane-C competition reads a content-addressed, signed STANDARD POOL
(`cascade-std-pool/1`, `cascade/shared/std_pool.py`): per series id a regular
grid or an int64 timestamp array, float64 channels (C, L), freq, string tags
(`domain`, `source`, ...), `first_seen` / `embargo_until`; a `files` manifest
of sha256s; an optional block clock; signed (sr25519) by the pool signer.
Validators, the trainer and the audit read it through ONE reader
(`load_std_pool`). Source-specific code (market APIs, relays, energy feeds,
scrapers) lives only in pool BUILDERS (owner infra). A new domain is a new
builder job publishing a standard pool plus an overlay spec naming it.

The MarketCast snapshot is a special case: series `<bank>/<class>/<asset>`,
the OHLCV(+funding) fields as channels at the same paths. The builder re-emits
it as a standard pool (`write_std_pool_from_mkt`); `load_snapshot` reads either
form and projects a standard pool onto the snapshot API with byte-identical
arrays and windows (tests pin it). The pool_d / lane A pool formats map the
same way (`pool_d` series → series ids with `domain` / `source` tags and
`first_seen`); their projections are not built, because no lane-C
competition reads them today.

### 3. The embargo is recorded per leg

The trainer stamps `embargo_days` / `embargo_eval_days` in force at a leg's
DISPATCH into the trainer-signed `marketcast` record; the draw, the scan and
the verdict use the recorded values (`entry_embargo_mc`), the audit re-derives
from the same record. A legacy entry without the record is drawn on the
config, exactly as before. `[marketcast] embargo_days` is then overlay-settable
within [2, 120] days. (Amends the W-OV note that kept it release-only: the
reason — "a raise would score gap windows on training data" — no longer holds.)

### 4. The rule menu

`cascade/shared/rules_menu.py`. Each lane (and each spec) names its test family
and pay curve; the defaults are today's rules and change nothing:

| Lane | Default family / curve | Runs in this release |
|---|---|---|
| B | `bonferroni` (a leg's own targets) / `tiered_decay` | families `plain_lcb`, `bonferroni`; all three curves |
| C | `bonferroni` (weighted owner family, red-team F6) / `fixed_term` (DEC-CA-0073 5–20 d) | families `plain_lcb`, `bonferroni`; all three curves |
| D | `maxt` (crown cohort) / `indefinite_until_beaten` (the court) | `maxt` / `indefinite_until_beaten` only |

A curve other than the lane's legacy one is RECORDED on the win (`curve`), so
a later overlay never changes a promise already made; legacy wins carry no
field and their bytes are unchanged. Choosing something a lane does not run is
refused at load (no silent no-op). `maxt` for lanes B and C needs the joint
bootstrap over a family's entries on the same windows — not in this release.

### 5. The contamination hash set by content hash

`[lane_b] contamination_hashset` (a `sha256:` pin) plus
`contamination_hashset_key` (its storage key) are overlay-settable. Readers
load a release artefact at the pin, else fetch the key, verify the bytes'
sha256 against the pin BEFORE parsing (`HashsetMismatch`, fail closed), and
cache by pin. The intake reloads when an adopted overlay names a new pin.

## Consensus

This release must reach every validator, satellite and the trainer once
(release-then-activate: nothing changes until an overlay selects something).
After it, specs, menu choices, the embargo and the hash-set pin are overlay
data. A new BLOCK (metric, family, curve, schema version, runner) is still a
release; a new data source is a pool builder job, never a release.

## The multi-instance runner (W-CMP2, owner 2026-11-24: this release)

The `marketcast` runner hosts N specs at once. `mkt` is its legacy
instance (`[marketcast]`, `mkt_reports`, `mkt_backlog` / `mkt_seen`,
`marketcast/round-<R>.json`), byte-identical. Every other accepted spec is
an INSTANCE: its id is a lane-C kind (targets `<id>:<name>`), with its own
legs, main-validator rows and report (`marketcast/<id>/round-<R>.json`,
arbitrated by `reports[<id>]`), verdicts, state (`comp[<id>]`), slots,
payouts (its competition weight) and audit replay. An instance spec must name
its pool, metrics, universe, timeframes, horizons and `from_block` (era
boundary, inside lane C, never before the overlay applies); it inherits the
deployed `[marketcast]` model pins, recipe bounds and verdict statistics.
Its classes are its pool's `class` tags. Any builder's standard pool is
projected generically (grid step to bank, `class` / `asset` tags,
channels renamed by the spec's `channels` map). The trainer relays the
instance's pool snapshot to the pod (presigned link + sha256); the pod clips
it to the recipe range before the dispatch. Instance legs train only on a
worker image built from this release (rolled at a seed-epoch boundary).

Still not done: joint `maxt` for lanes B and C (lane D keeps its own), and
the pool_d / lane A projections (no lane-C competition reads them).
