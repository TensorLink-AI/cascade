---
id: DEC-CA-0018
type: decision
title: "Missing data rides an explicit parallel mask, NEVER a NaN sentinel — and the payload waits for a gap-preserving eval pool"
status: proposed
date: 2026-08-13
tags: [interface, generator, determinism, cpm, eval, pool, missingness]
revisit_when: "the pool build's raw-missingness histogram (below) shows enough real gap structure to build a gap-preserving slice at n_windows scale AND the scorer masks unobserved target steps — until both exist, a corpus that models gaps is trained for a capability no round can see"
relations: {depends_on: DEC-CA-0015}
---
`check_series` rejects any non-finite value (`interface/generator.py:159`), so a
generator cannot emit a gap. Toto 2.0's CPM is imputation-shaped and the model
already carries a per-entry mask channel; real telemetry is full of holes. The
gap is genuine. Two things about it are not obvious.

**A NaN sentinel is disqualified on determinism, not on taste.** Measured:
`np.nan` serialises as `000000000000f87f`, while `0.0/0.0` and `inf − inf` both
serialise as `000000000000f8ff` — the sign bit differs. Two generators that both
"emit a gap" would produce **different `corpus_digest`s** depending on the
arithmetic that produced the NaN, and one generator could differ from itself
across a numpy/BLAS version bump. That breaks Part F's first constraint
(byte-identical corpus at a fixed seed) in a way no amount of care in the miner's
code prevents. A sentinel also costs the finiteness gate, which is load-bearing
for the trainer's numerics downstream (`causal_standardize`'s eps-floored ratio,
`CAST_SAFE_MAX_FLOAT32`).

**DECISION: an explicit `mask` key** in the DEC-CA-0015 record — same shape as
`values`, 1 = unobserved — with `values` staying finite everywhere and
`check_series` unchanged. Masked positions carry a real (ignored) float, so the
digest stays a digest of well-defined bytes.

**The payload is unusually cheap to consume — and still must wait.**
`causal_standardize(x, mask=…)` already excludes masked entries and carries the
stats forward across masked spans (`toto2_model.py:132-175`).
`Toto2Model.forward` already accepts a per-entry mask. `sample_cpm_masks`
already produces exactly the tensor a corpus mask would OR into
(`toto2_trainer.py:311-318`). So consuming `mask` is a trainer change with **no
arch change and no `base_arch_digest` bump** — the cheapest payload on the whole
roadmap. Two rules come with it: the corpus mask ORs into the CPM mask, and
masked positions are **excluded from the loss target** (a step declared
unobserved is not a label).

**What blocks it is the eval, not the trainer.** `pool/builder.py::
prepare_series` linear-interpolates every gap and drops any series above
`max_missing_frac = 0.2` (`builder.py:127-133`); `Wrapper._prep` left-pads short
histories with the first value. There is no missingness anywhere in the scored
set. A model trained on realistic gaps is therefore scored entirely on
gap-filled windows, and its only benefit is the indirect regularisation CPM
already supplies for free. Under this roadmap's own rule — a capability the eval
cannot see is one miners are rationally required to ignore — the `mask` payload
does not ship until the pool can present gaps and the scorer can skip masked
target steps.

**Budget.** Under DEC-CA-0015's byte denomination a mask costs bytes, correctly:
hiding 40% of a series costs the full series *plus* the mask. There is no
discount for declaring data absent.

**Contract churn.** Carrier-side: none beyond DEC-CA-0015 (a reserved name).
Payload-side: `contract_digest` bumps (new `[training]` keys for mask
composition and masked-target loss), `base_arch_digest` does not. Miner
migration: zero — a values-only generator stays valid and hashes identically.

**Open question — the measurement that decides whether the payload is even
reachable.** `prepare_series` already computes `missing_frac` per channel and
discards it. Record it (and the pre-fill gap-run-length distribution) into
`provenance.json` for the next few daily builds. That histogram answers whether
a gap-preserving pool slice can reach `[eval] n_windows = 2000` at all, or
whether real feeds are so well-filled upstream that the honest answer is "the
private pool contains no missingness to score".
