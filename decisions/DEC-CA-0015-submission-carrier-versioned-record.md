---
id: DEC-CA-0015
type: decision
title: "The submission carrier churns ONCE: a versioned, closed-namespace series record with a field-set-tagged digest and a byte-denominated budget"
status: proposed
date: 2026-08-13
tags: [interface, generator, carrier, determinism, audit, versioning, budget]
revisit_when: "a payload field is wanted that is not per-series and rectangular — a corpus-level covariate, a cross-series adjacency, or a variate sampled at a different rate than its siblings — at which point the per-series record is the wrong container and a second carrier has to be argued on its own merits, not bolted on as another reserved key"
relations: {enables: [DEC-CA-0016, DEC-CA-0017, DEC-CA-0018, DEC-CA-0019, DEC-CA-0021]}
---
Gaps 1, 3, 4 and 6 all want `generate()` to carry something beyond a bare float
array. Solving them one at a time churns the miner-facing contract four times
and strands four cohorts of deployed generators. DECISION: settle the carrier
now, add **no payload**, and make every later field an additive key.

**The shape.** `generate()` may yield either a bare `np.ndarray` (today,
unchanged, valid forever) **or** a record with `values` required. Reserved key
names are published now — `mask`, `group_id`, `start`, `freq`, `roles`,
`labels`, `quantiles` — and **none is accepted yet**. Anything unrecognised is
a hard reject, never a silent drop: a dropped field means the corpus the
trainer built differs from the one the miner verified locally, which turns the
determinism guarantee into a lie while `cascade verify` still prints OK. The
namespace is closed on purpose (see *forecloses*).

**The digest.** `corpus_digest` and `_series_key` hash a `(C, L)` float64 array
today. They become a canonical serialisation carrying a **field-set tag**, such
that a values-only series hashes byte-identically to today and any series with
an extra field hashes differently.

The byte-identity claim **cannot be verified against the current tests** —
there is no golden digest anywhere in the suite. `test_manifest.py::
test_corpus_digest_is_order_and_value_sensitive` only asserts relational
properties (order-sensitive, value-sensitive), and `test_audit.py:388` sets the
expected digest from a live re-derivation. Both pass under any reasonable
rewrite of the hash. So G1 ships **with** a golden-vector test pinning today's
bytes. Measured against the current code:

| input | `corpus_digest` |
|---|---|
| `[zeros(10), ones(20)]` | `1adeb36d64f28a0a570968094120406ff2a1f7a3ad00e2e8525995725be3fb0a` |
| `[arange(4.0)]` | `1c930e1ffc3f4fb49f28d328aaa4e2de70a9fb6713262342dfc7576d56b061f9` |
| `[arange(4.0)[None,:]]` | *(identical to the row above — 1-D promotes to `(1, L)`)* |

and `_series_key(atleast_2d(arange(4.0)))` = `f606e50e818df266ff39f1128950d0cf`.

**The version.** `interface_version` in `chain.toml [generator]` plus a declared
version per generator (in `config.json`, which the OCI digest already pins).
Declaration is **required only when a generator uses a non-default field**;
absent = 1 = bare array. A newer trainer then runs an older generator under the
semantics it was verified against instead of reinterpreting it, and rejection of
an unknown key is fail-closed rather than version-blind.

**The budget.** `max_total_points` counting `C * L` is unambiguous only while
values are the sole payload. It becomes **bytes of canonical payload**, defined
while there is exactly one field to count. See `docs/SUBMISSION_SURFACE_ROADMAP.md`
§budget for why the *binding* budget is `train_tokens`, not this cap.

**The eval-side shape.** `ForecastFn` generalises from `(L,) → (1, ns, H)` to
`(C, L) → (C, ns, H)` now, while `C` is always 1. `WindowScore.channel` already
models this correctly.

## Are G1–G3 actually irreversible? Three challenges

**G1 — overstated, but do it now anyway.** Changing the digest definition later
does not make archived corpora unreplayable: the digest is a function of the
archived *inputs* (`gen_ref` + `generation_seed` + config), and `cascade-audit`
re-derives by re-running the generator (`audit/rederive.py:147`), not by reading
a stored corpus. What breaks is comparing an old receipt's recorded digest
against a new code's re-derivation — and DEC-CA-0009 already set the precedent
for exactly that: a replay mode (`wql_mode`) that lets the audit reproduce a
receipt under the rule that decided it. So G1 is recoverable, at the price of
one permanent replay mode per deferral, compounding. Cheap now, annoying later,
never fatal. Do it now on cost, not on irreversibility.

**G2 — the deadline is G4's ship date, not today.** Submissions are
content-addressed and commitments carry a reveal block, so "what semantics did
this archived submission assume?" is answerable retroactively from a
block-height table. Today that table has one row, because exactly one semantics
has ever existed. Version drift only becomes unrecoverable once a *second*
shape can be submitted. Therefore: land the `chain.toml` key now (free), and
make the per-generator declaration mandatory in the same release that first
accepts a record. Not before, and not after.

**G3 — wrong on the mechanism.** `[generator]` is **not** in `contract_digest`;
that digest is a sha256 over `TrainingContractConfig`, i.e. `[training]` only
(`shared/manifest.py:83`, `shared/config.py:163`). Renegotiating
`max_total_points` bumps no digest and invalidates no manifest. Round-over-round
comparability is real but the subnet already breaks it deliberately and
routinely (`heat_n_windows` 256→2000, `epoch_blocks` 7200→3600, the 20× token
budget). Worse for the premise: **the cap does not bind today.** The live path
is `corpus_mode = "stream_cpu"`, and `_FreshSeriesStream` never enforces
`max_total_points` — it stops at `token_budget`, and the cap survives only as an
input to an rlimit fsize (`sandbox.py:448,698`). On the `cache_reuse` path where
it *is* enforced, the reachable maximum is `16384 × 4096` = 67M against a cap of
2e9, i.e. 30× slack. G3 is deferrable. It ships in the same edit only because
redefining a dead knob while it has one dimension is a two-line change.

**G4 — the load-bearing one.** Not irreversible either (a tuple could be
extended later, breaking everyone once). But the deployed-generator population
only grows, and this is the one decision that makes mask, group id, start/freq,
roles and labels all become *added keys with no resubmission*. Ship it.

**G5 — free, correct, and only coherent with G2.** Hard-reject must be
fail-closed: an older trainer meeting a newer key rejects the submission rather
than dropping the key. That is the right failure and it is the reason the
version has to exist.

**G6 — the least deferrable of the "deferrable" set,** for a reason not in the
original framing: `forecast_wrapper.py` is **copied into every checkpoint**
(`toto2_trainer.py:462`) and the validator imports *the checkpoint's own*
wrapper (`validator/evaluator.py:load_forecaster`). Warm-start (DEC-CA-0013)
keeps old checkpoints alive across promotion generations. So the scorer can
never *replace* the old signature — it must **promote** a `(1, ns, H)` return to
one channel row and accept `(C, ns, H)` when offered. Shaped that way it is a
zero-behaviour-change edit today and a blocking one later.

## What this carrier forecloses — explicitly

- **Miner-defined fields.** A closed namespace means miners can never carry
  information the owner has not named. That is the point (an unread field is a
  free steganography channel), but it also means there is no miner-side
  experimentation path: every new capability needs an owner release.
- **Anything that is not per-series.** A corpus-level object — a shared
  calendar, a global covariate, a graph over series — has no home in a
  per-series record. `group_id` is a label, not a structure; a genuine
  cross-series adjacency needs a second carrier.
- **Ragged variates.** `values` is a rectangular `(C, L)`. A covariate sampled
  at a different rate than its target cannot be expressed without per-channel
  `freq` plus resampling semantics the trainer does not have.
- **"Labels are free."** Byte-denominated budgeting means every field a miner
  adds costs them budget. A field the fixed model consumes weakly is a net loss
  for an honest miner, which is a feature for gating payload and a tax on
  experimentation.
