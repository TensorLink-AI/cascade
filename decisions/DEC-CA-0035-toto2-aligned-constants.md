---
id: DEC-CA-0035
type: decision
title: "Toto2-aligned optimizer constants + warm-start LR scale — the measured recipe that beats the converged init, shipped inert behind drop-when-default"
status: proposed
date: 2026-08-27
tags: [cascade, training-recipe, optimizer, normuon, lr-schedule, umup, contract-digest, warm-start]
revisit_when: "Wave 3 measures cross-generator separation at the aligned recipe (the arming gate: a better generator must still WIN under it, not just score near init); or a testnet cascade runs a full round cycle with the bundle armed (the mainnet-cut gate); or the 22M size seam arms (µP transfers the constants by design — verify warm_lr_scale at size); or Datadog publishes a revised recipe"
relations: {depends_on: DEC-CA-0018, informs: DEC-CA-0033, supersedes_measurement_of: DEC-CA-0033}
---
Owner directive 2026-08-26: align the training recipe with Datadog's
published Toto 2.0 hyperparameters (arXiv 2605.20119) — their HP sweep was
paid once at 4M under u-µP and transfers across sizes; cascade should
inherit it rather than re-derive. Measured 2026-08-26/27 on a rented
4×L40S pod (u88 generator, production r41 seeds + warm-start init, 1h heat
fence, EMA variants alongside; full grid in
docs/notes/2026-08-27-toto2-alignment.md):

- **The full constants bundle BEATS the converged warm-start init** — the
  first checkpoints all campaign to do so. Bundle @ base_lr×0.125 (5e-4):
  EMA-999 0.20483 vs init 0.20864 (−0.0038, campaign best). Bundle @ ¼:
  0.20604. Bundle @ full 4e-3: 0.20794. Current constants at the same LRs:
  +0.0020…+0.0101 ABOVE init.
- **The bundle is much more than its parts**: the 54:1 split alone and
  wd→2e-8 alone each help but neither crosses init; the row-EMA β₂
  0.95→0.999 gap is the prime suspect for the remainder (untested in
  isolation).
- **Low LR alone also crosses init** (current constants @ 1e-4: −0.0031)
  but with endpoint ≈ init — the degenerate barely-train route. The
  bundle's value is tolerating real LR: it learns AND beats init, which is
  what should preserve the heat's generator discrimination.
- LR *values* do not transfer across parametrization conventions (their
  NorMuon η=0.65 / AdamW 0.012 are u-µP units; ours are Muon-convention) —
  only the dimensionless constants and the matrix:AdamW ratio carry.

DECISION — seven `[training]` knobs, defaults = the previously hardcoded
deployed behavior, all digest-bound drop-when-default (deployed digests
untouched until an operator arms them; arming is one deliberate contract
cut — release-then-activate, trainer + all validators together, testnet
first):

| knob | default (deployed) | measured target |
|---|---|---|
| `muon_momentum` | 0.95 | 0.96 |
| `muon_row_beta2` | 0.95 | 0.999 |
| `grad_clip` | 1.0 | 7.0 |
| `adamw_beta1` / `adamw_beta2` | 0.9 / 0.999 | 0.91 / 0.972 |
| `adamw_lr_scale` | 1.0 (shared LR) | 1/54 |
| `warm_lr_scale` | 1.0 | 0.125 (→ 5e-4) |

plus `weight_decay` → 2e-8 at arming (already a config field; its VALUE
change is part of the same cut). `warm_lr_scale` keys off the same
`warm_started` signal as wsd's warmup-once: the whole warm-started run
(fork-anneal branch included) trains at `base_lr × warm_lr_scale`;
from-scratch generation starts keep full `base_lr` — the constants were
tuned for from-scratch, so alignment is safest exactly there, and a
converged init measurably tolerates far less LR than a random one
(DEC-CA-0033's +0.11 step-1 kick; re-warmup does NOT substitute — measured
dead, that knob stays unarmed).

Also fixed here: the DEC-CA-0033 fields (`ema_decay`, `gen_seed_mix`,
`rewarmup_fraction`) shipped with dataclass defaults but NO loader
parsing — arming them via chain.toml silently no-oped. Loader round-trip
now pinned by test for every armable [training] knob.

Deployment coupling: arming changes training numerics for every role, so
king and challenger stay paired (both retrain under the new contract from
the same init) and score reuse across the activation boundary is invalid
for one round. The zero-train guard (DEC-CA-0034's open item) gains
urgency: near-init scores at low warm LR narrow the gap the init-floor
must police.
