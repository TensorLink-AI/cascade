---
id: DEC-CA-0030
type: decision
title: "Bench-anneal: the benchmark sidecar scores an annealed copy of each duel checkpoint — finished-form BenchScores with zero validator involvement"
status: proposed
date: 2026-08-24
tags: [cascade, benchmarks, telemetry, wsd, anneal, promotion, guard]
revisit_when: "DEC-CA-0029 (fork-anneal) OR DEC-CA-0033 (ema_decay) arms on mainnet — bench-anneal then benches an already-finished artifact and must be DISARMED in the same window (double-annealing / EMA-of-anneal is a different recipe; the worker's anneal_recipe refuses an EMA-armed contract); or a testnet cycle shows the anneal leg regularly missing its wall (then raise the leg's guard, not the fraction); or bench_hold_max_hours proves too small for duel bench + anneal legs"
relations: {depends_on: DEC-CA-0018, informs: DEC-CA-0017, superseded_by_when_armed: DEC-CA-0029, superseded_by_when_armed_alt: DEC-CA-0033}
---
Same problem as DEC-CA-0029, different deployment constraint. Under wsd
(DEC-CA-0018) every scored artifact is mid-stable — the offline calibration
(2026-08-23/24, OPSLOG) measured the finished-form gap at 6–9% blended on the
public suites, non-uniform across generators — so every absolute-number
consumer of the signed BenchScores (the public benchmark stream, promotion
picks, the DEC-CA-0017 no-downgrade guard comparing against cosine-era
floors) reads a recipe artifact. DEC-CA-0029 fixes this at the source but is
a contract cut requiring a coordinated window with all external validators.
The owner needs the guard/promotion layers fixed NOW, without that
coordination.

DECISION — `[telemetry] bench_anneal_fraction` (trainer-local, never
digest-relevant): when armed, the post-round bench stage runs one extra
worker leg per duel checkpoint, on the pod that trained it, BEFORE its sweep:

* **The leg** resumes the canonical checkpoint (`--warm-start-ref` = its
  trained_pointer: weights + optimizer state) on a FRESH salted corpus
  (`base_seed ^ BENCH_ANNEAL_SALT`) for `fraction × target_train_hours`
  under the pure-decay recipe (`--anneal` = `warmup_cosine` +
  `warmup_fraction 0`: cosine base_lr → 0 across the leg's budget —
  DEC-CA-0018's decay shape applied post-hoc). The salt keys the leg's work
  dir and checkpoint repo (`…-anneal-u<uid>`) away from every canonical
  name.
* **The sweep** then benches the annealed copy's dir instead of the raw
  checkpoint's; the six numbers flow into the signed BenchScores unchanged
  in shape. Any leg failure falls back to benching the raw checkpoint — a
  mid-stable number beats a missing one, and the miss is logged.
* **What does NOT change**: the canonical duel checkpoint and its digest,
  the manifest, the duel verdict (validators re-derive that from the pinned
  raw checkpoint — untouched), the bench-report WIRE FORMAT (deliberately:
  adding a signed field would break signature verification on un-upgraded
  external validators and silently stall their reign logs), and therefore
  every validator-verified byte. Deploying this is a trainer restart only.
* **Semantics, declared**: BenchScores now mean "this checkpoint's finished
  form" — declared here and in the config comment rather than on the wire.
  The annealed artifact itself is pushed to the registry under the salted
  `-anneal-u<uid>` repo, so every published number remains independently
  reproducible and auditable against a public artifact. Reign logs mix
  raw-form rows (r33 → arming) with anneal-form rows going forward; the
  raw-form wsd rows read ~7% pessimistic and will simply lose promotion
  picks to anneal-form rows — acceptable, and the guard's cosine-era floor
  comparison becomes finished-vs-finished, which is the point.
* **Cost**: ~fraction × one training leg per benched checkpoint, serial
  with that pod's sweep. Size `[eval] bench_hold_max_hours` for duel bench
  + anneal legs; the scratch shadow (DEC-CA-0014) stays raw-form for now —
  its scratch-vs-lineage comparison is internally consistent either way.

Relationship to DEC-CA-0029: this is the interim, not the destination. When
the contract cut arms, checkpoints arrive already-finished; benching an
anneal of an anneal would be a new recipe nobody calibrated — the two knobs
must never be armed together (revisit_when).
