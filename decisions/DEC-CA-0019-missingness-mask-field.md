---
id: DEC-CA-0019
type: decision
title: "Missingness is a parallel mask field with pinned filler — never NaN — and it is eval-gated: the pool must carry real gaps before the field is accepted"
status: proposed
date: 2026-08-13
tags: [interface, generator, trainer, eval, pool, digest, dedup]
revisit_when: "the pool gap-rate measurement (below) shows real missingness is too rare in our sources to build a credible masked-eval slice — then the eval gate cannot be satisfied and the field stays reserved; or the masked-history eval ships and a mask-trained model shows no gain on it — then the capability itself is dead at 4M and acceptance waits for a bigger size"
relations: {depends_on: DEC-CA-0016, informs: DEC-CA-0022}
---

Gap 3: `check_series`'s finiteness gate (`interface/generator.py:159`) means
no gaps; TiRex-2 injects contiguous NaN blocks deliberately; Toto's CPM is
imputation-shaped and already carries a per-entry mask channel; the trainer
synthesises masks (`sample_cpm_masks`, `toto2_trainer.py:84`) and miners
cannot compete on realistic missingness. All confirmed. Two decisions
follow: the representation, and the sequencing — and the sequencing finding
is that this gap is **eval-gated**, against the prompt's expectation.

## Representation: `mask` field, filler pinned to 0.0 — NaN is rejected

**NaN sentinel, rejected.** It looks minimal (no new field) and touches
everything:

- *Digest determinism, invisibly.* IEEE-754 has many NaN bit patterns;
  `tobytes()` digests (`corpus_digest`, `_StreamDigest`, `_series_key`)
  distinguish them. A generator whose NaNs come from `np.nan` constants
  verifies; one whose NaNs fall out of arithmetic (`0/0`, overflow paths)
  can carry payload-varying NaNs that break byte-reproducibility in ways
  `cascade verify` would catch only by luck. A correctness class we cannot
  explain to miners is a correctness class we will burn hotkeys on.
- *Every gate needs a NaN-aware fork*: the finiteness gate itself,
  `max_abs` (`np.abs().max()` → NaN), `reject_constant` (`np.ptp` → NaN),
  the dup key, `causal_standardize`'s float64 moments, and
  `iter_training_batches`. Each fork is a place the univariate path can
  regress.

**Parallel `mask` field, adopted** (the DEC-CA-0016 reservation, made
concrete):

- shape `(C, L)` uint8 (or bool), 1 = unobserved; shape/dtype-validated
  beside `values` in `check_series`'s record layer.
- **masked positions of `values` MUST be exactly 0.0.** Not a style rule —
  three invariants hang on it: the digest stays a function of the *visible*
  data (two semantically identical corpora can't hash apart on hidden
  filler); the filler can't be a covert byte-channel; and the byte-dup key
  can't be evaded by jittering invisible entries. `check_series` enforces
  it; the finiteness gate is untouched (0.0 is finite).
- per-series cap `max_missing_frac` (new `[generator]` key, suggest 0.5,
  shadow-logged first per the DEC-CA-0010 pattern) so "99% masked" series
  can't be minted as near-free points.

Budget: under G3's bytes denomination the mask prices at 1 byte/point — a
masked series costs 9/8 of an unmasked one, no renegotiation.

## Trainer consumption (the cheap half — the machinery exists)

The model already takes a per-entry mask and zeroes masked inputs
(`toto2_model.py:352-371`); `causal_standardize` already excludes masked
entries from the causal stats (`toto2_model.py:151`); the trainer already
expands patch masks to entries (`toto2_trainer.py:318`). Consumption is:

1. OR the miner mask into the input observation mask (miner-missing entries
   look exactly like CPM-masked entries on the way in), and
2. **exclude miner-masked entries from the loss** — CPM targets stay
   unmasked because ground truth exists under the synthetic mask the trainer
   drew; under a *miner* mask there is no observed value, only pinned 0.0
   filler, and training the head to predict 0.0 at gaps would be a
   corpus-poisoning bug, not a capability.

Contract churn, stated precisely: this consumption rule is trainer *code*,
which `contract_digest` cannot see (the digest hashes config fields,
`shared/manifest.py:84`). Arming therefore folds the accepted-field set into
`[training]` (DEC-CA-0016 G2 layer 3) so the change is digest-visible and
king/challenger/validator move in lockstep — one deliberate bump.

## The eval gate — this gap does NOT escape Part C

The prompt expected the eval-precedence claim to bind on gap 6 and "not at
all" on gaps 1–2, leaving gap 3 open. Traced through the code, it binds:

- Eval histories are **complete by construction**: the pool builder
  interpolates gaps and drops series above `--max-missing-frac`
  (`docs/EVAL_POOL.md`, "Cleaning"), and `EvalWindow` carries no history
  mask. The model's mask channel is exercised at eval only by the horizon
  filler patches CPM decoding appends.
- So a model trained on realistic miner missingness has learned to condition
  on gapped histories — an input state the eval **never presents**. Its only
  measurable channels are second-order (regularisation side-effects). A
  rational miner ignores the field; the capability ships dead on arrival.

Sequence, therefore: the pool grows a **masked-history slice first** — stop
interpolating a held-out subset of genuinely-gappy harvested series, carry
`history_mask` in window metadata (an `EvalWindow.metadata` entry, no
container change), extend the wrapper contract so mask-aware checkpoints
receive it (capability-detected; old checkpoints score with mask ignored,
paired and fair since both sides share windows). Only when that slice is live
does `mask` move from reserved to accepted.

## Gaming surface

- **Mask-spam** (mostly-masked cheap series): priced by bytes, capped by
  `max_missing_frac`, and self-harming — masked points carry no loss signal,
  so the miner pays budget for silence.
- **Filler abuse**: foreclosed by the pinned-0.0 rule.
- **Mask-pattern steganography toward the eval**: nothing eval-side reads
  training masks; no channel exists.

## Open questions, with the measurements that settle them

1. **Real gap rate**: instrument one `cascade-pool build` to report the
   pre-interpolation missing-fraction distribution per source. Decides
   whether a masked-eval slice can be built from real gaps (credible) or
   would need synthetic gap injection on real series (weaker — the TiRex-2
   move, acceptable but second-best).
2. **Capability check at 4M**: after the masked slice exists, one paired
   testnet round with a mask-emitting generator vs its unmasked twin — does
   masked training measurably move the masked-slice score at this scale?

## Rank

#3 of six as proposed — agreed, with the caveat that its *effort* estimate
must include the pool work, which is the long pole; the interface and
trainer halves are small because Toto2's CPM machinery was built for exactly
this shape.
