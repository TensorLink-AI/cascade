---
id: DEC-CA-0020
type: decision
title: "The submission carrier becomes an extensible series record — settled once, payload-free, before any gap opens"
status: proposed
date: 2026-08-13
tags: [interface, generator, corpus, digest, dedup, budget, eval, audit]
revisit_when: "the first payload field arms (mask or roles) and the interface_version fold into [training] is due; or a proposed capability cannot be expressed as an added named field of per-series parallel-array shape (ragged channels, event streams, within-series streaming) — the record's one structural commitment; or the golden-vector digest test ever has to change, which means the byte-identity guarantee this node rests on was broken"
relations: {enables: [DEC-CA-0021, DEC-CA-0023, DEC-CA-0024, DEC-CA-0026], refines: DEC-CA-0008}
---

`generate()` today yields bare float arrays; four of the six capability gaps
(time anchor, mask, group id, roles) each need one more thing to ride from the
generator to the trainer. This node settles the carrier ONCE — the yield shape,
its digest, its version, its budget, and its unknown-field policy — so the
contract churns one time instead of four, and adds **no payload**: every new
field is reserved and refused until a consumer exists.

The carrier/payload split is the discipline: a carrier change is cheap and
lands now; a payload field is accepted only when something in the fixed
pipeline actually consumes it. Carrying a field the model ignores is untested
surface, budget spend, and a semantics squat waiting to happen.

## G4 — the yield shape: a record with named optional fields

`generate()` may yield either a bare `np.ndarray` — today's contract, unchanged
byte-for-byte — or a mapping (`dict` or a provided `SeriesRecord` helper) with
`values` **required** and, at interface v1, **nothing else accepted**. A bare
array is defined as exactly equivalent to `{"values": arr}`.

Named optional fields are additive indefinitely; a positional tuple is a
breaking change on every extension and is rejected. The record is also
**self-describing**: a bare array or values-only record IS v1 semantics, which
does most of G2's work for free (below).

Touch points, all mechanical:

- `cascade/interface/generator.py` — a record-validation layer in front of
  `check_series` (which keeps validating `values` exactly as today);
  `drain_generator` canonicalises the record.
- `cascade/trainer/sandbox.py::_child_stream` / `_write_frame` — the frame
  protocol gains a record frame type; a values-only record emits the legacy
  frame so the parent-side reader is unchanged for every deployed generator.
- `cascade/trainer/stream.py` / `corpus.py` — carry the canonical record
  instead of the bare array; the `BaseTrainer` keeps receiving `(C, L)` values
  (it reads no other field until a payload decision says so).
- `docs/INTERFACE.md` — the record contract and the reserved-name table.

**What this forecloses, stated plainly:** the canonical record fixes a series
as *uniform-length parallel arrays keyed by name* — every per-point field has
shape `(C, L)` (or `(C,)` per-channel, or scalar per-series). Ragged
per-channel lengths, event-stream payloads, and within-series streaming do not
fit and would need an interface v-next, not a new field. Reserved names bind
their one-line semantics forever — renaming or repurposing one is a breaking
change by construction. That is the price of never migrating again, and it is
accepted.

## G1 — the digest is defined over the canonical record, now

`corpus_digest` (`cascade/shared/manifest.py:62`), `_StreamDigest`
(`cascade/trainer/stream.py:37`), and the dedup `_series_key`
(`cascade/interface/generator.py:173`) all hash the raw `(C, L)` float64 array
with a shape prefix. The canonical serialisation becomes:

- **values-only series** — byte-identical to today: the 8-byte channel count,
  8-byte length, raw float64 bytes (4-byte channel prefix for `_series_key`).
  Nothing moves; every archived corpus digest, stream digest, and dedup key
  reproduces exactly.
- **extended record** — a `0xFF` sentinel byte, then a field-set tag (sorted
  field names), then each field's dtype/shape/bytes in name order.

The sentinel is collision-free by construction: a legacy series' first byte is
the high byte of its channel count, and `C <= max_channels` keeps that `0x00`
for any conceivable cap, so `0xFF` can never open a legacy hash. A values-only
corpus therefore hashes identically under old and new code, and a corpus with
any extra field hashes differently — the property that keeps every archived
round replayable across the migration with **one** digest rule, not a
per-migration fork of the DEC-CA-0009 `wql_mode` kind.

**Verified, not asserted:** the existing tests do NOT pin today's digest bytes
— `test_corpus_digest_is_order_and_value_sensitive`
(`tests/unit/test_manifest.py:35`) and `tests/unit/test_corpus_and_verify.py`
check only self-consistency, order-sensitivity, and hex length. So byte
identity cannot be *confirmed against the current suite*; it must be *frozen
by it*. G1 ships with golden-vector tests: fixed input series → literal
expected hex digests for `corpus_digest`, `_StreamDigest`, and `_series_key`,
committed before the canonicaliser lands. Any later change that moves those
bytes fails loudly.

The behavioral probe (DEC-CA-0008) needs no separate work: `_probe_digest`
rides `build_corpus`/`corpus_digest`, so probe-byte comparisons inherit the
canonical rule automatically.

**Deferability verdict (challenged as instructed): deferrable, at real cost —
do it now anyway.** "Every archived corpus becomes unreplayable" overstates:
an auditor can always replay under the code at that git revision, and
`cascade-audit` already demonstrates the replay-under-both-rules pattern. But
that pattern is exactly the scar tissue DEC-CA-0009 left — every deferred
digest change becomes a permanent rule-fork the audit must carry. One
canonical rule now, while values-only hashes are unchanged, costs a sentinel
byte and a test file.

## G2 — explicit interface versioning

Three layers, cheapest first:

1. **Self-describing carrier** (free, above): the record's field set states
   what the submission uses.
2. **`[generator] interface_version = 1`** in `chain.toml`, plus a reserved
   top-level `interface_version` key in the submission's `config.json`
   (absent = 1). A trainer runs an older generator under the semantics it
   declared; `cascade verify` embeds the version in its output.
3. **At first payload arming** (not now): fold the accepted field set into
   `TrainingContractConfig` — e.g. `interface_version` or `accepted_fields`
   as a `[training]` key — so the existing validator digest gate
   (`contract_digest_mismatch`) detects corpus-semantics drift the way it
   detects recipe drift. This is a deliberate one-time digest bump at an
   epoch boundary, the routine re-pin protocol.

**Deferability verdict: partially retrofittable, so "irreversible" is
overstated — but the cheap layer is still right to land now.** Every archived
submission carries an on-chain commit block, and carrier changes activate at
announced boundaries, so "the semantics at commit time" is derivable
after the fact from block numbers plus the self-describing record. What is
genuinely unretrofittable is a change that alters the meaning of an
*existing* shape with no observable marker — and layer 3 exists precisely so
we never make one. Layers 1–2 cost a config key and a doc paragraph.

## G3 — the budget goes dimension-agnostic while there is one field to count

`max_total_points` counts `C * L` of values (`drain_generator`,
`generator.py:242`). Redefine the carrier cap as **bytes of canonical
payload**: `max_payload_bytes = 16_000_000_000` — exactly
`max_total_points × 8`, so a values-only corpus has an identical cap and the
sandbox file rlimit (`sandbox.py:448` already sizes in bytes) needs no change.
A future `mask` prices at 1 byte/point, `roles` at `C` bytes/series, with no
renegotiation of what "a point" means.

**Premise correction (challenged as instructed):** `[generator]` fields are
NOT in `contract_digest` — `TrainingContractConfig` (`shared/config.py:163`)
carries `[training]` only, so re-denominating the cap later would bump
nothing and break no signature. G3 is therefore *deferrable at low cost*, not
irreversible. It ships now anyway because it is one config key while `values`
is the only field, and because the budget that actually binds is elsewhere:
in the live `stream_cpu` mode the stream stops at the **training token
budget** (`_FreshSeriesStream.series()`, `stream.py:186` — ~40B point-passes
at 3h × 3.7M tok/s), and `max_total_points` binds only the materialised
`cache_reuse` drain and the sandbox rlimit. The real economics of a point is
the compute spent training on it, which is already dimension-agnostic and
compute-normalised. See the roadmap's budget-denomination section for the
Part-D consequences.

## G5 — reserve the namespace, hard-reject unknown keys

Reserved field names, published in `docs/INTERFACE.md` with one-line intended
semantics so nothing can be squatted with incompatible meaning:

| field | reserved semantics (not yet accepted) |
|---|---|
| `mask` | `(C, L)` uint8, 1 = value unobserved; masked values pinned 0.0 |
| `group_id` | scalar int/str: panel membership across series |
| `start` | scalar int64: UTC epoch-seconds of the first step |
| `freq` | pandas-style frequency string, vocabulary of `eval/seasonality.py` |
| `roles` | `(C,)` uint8 per-channel: 0 target, 1 past-cov, 2 future-known |
| `labels` | per-series or per-channel tags (semantics TBD before acceptance) |
| `quantiles` | per-point distributional payload (semantics TBD before acceptance) |

At interface v1 the record validator accepts `values` and **rejects
everything else** — reserved names included — with a `ValueError` that fails
the generator's round, and `cascade verify` fails the same way miner-side
first. Silent-drop is the disqualifying alternative: a field a trainer
doesn't understand, dropped, means the trained corpus differs from what the
miner verified locally and the determinism guarantee is quietly false.

On steganography, honestly: for a deterministic, network-isolated generator,
an unconsumed field that nothing downstream reads is a channel to nowhere —
the practical exfiltration threat is narrow. The load-bearing reasons to
refuse data are silent-drop divergence, untested surface, budget semantics,
and squatting. "Reserve the name, refuse the data" stands on those.

## G6 — generalise the eval-side shapes while C ≡ 1 makes it free

`ForecastFn` goes `(L,) → (1, ns, H)` to `(C, L) → (C, ns, H)`
(`eval/scoring.py:48`); `score_forecaster_on_windows` calls it once per
window instead of once per channel; `WindowScore.channel` already exists and
keeps meaning row-per-channel. At `C = 1` everywhere this is a
zero-behaviour-change refactor. Two companions that must ride along, found in
verification:

- **Cluster fallback must become per-window, not per-row.**
  `_window_clusters` (`eval/koth.py:159`) makes each sourceless *row* its own
  singleton — under multivariate windows, C perfectly-correlated channels of
  one window would resample as C independent clusters, inflating the
  bootstrap's effective sample size. The fallback key becomes `series_id`
  (identical behaviour at C = 1).
- **Archived checkpoints keep their 1-D `forecast_wrapper.py` forever** — the
  evaluator keeps a per-channel adapter path for wrappers that predate the
  `(C, L)` contract, capability-detected, so old checkpoints stay scoreable.

**Deferability verdict: genuinely deferrable** — all internal, nothing signed
records these shapes — but it is the cheapest item here and it converts gap
6's future "eval must change first" blocker into "eval already speaks the
shape". Ships in Stage 0.

## What does NOT change

No deployed generator resubmits; a bare `(L,)` yield stays valid under every
stage of this node. `contract_digest` does not move at Stage 0 (the version
fold is explicitly deferred to first payload arming). `RoundSeeds`, the
manifest schema, the receipt schema, and the scoring rule are untouched.
`cascade verify` continues to reject everything it rejects today, plus
records with unaccepted fields.
