---
id: DEC-CA-0040
type: decision
title: "Make the king beatable more often by flattening the fresh-king margin ramp (win_margin_start → the floor), NOT by touching the LCB noise gate: the margin below ~0.3% is inert because the paired-bootstrap LCB>0 requirement binds first"
status: proposed
date: 2026-09-07
tags: [cascade, koth, margin, tenure, beatability, consensus, scoring]
revisit_when: "before arming, the first-order replay (scripts/replay_margin_decay.py --start 0.005 --ends 0.005 --warmups 8) must be re-run on the current receipt trail and its flip / near-miss counts published in the upgrade announcement (the DEC-CA-0016 convention); a testnet cycle must show no crown ping-pong between near-equal strong lineages (the one behaviour the fresh-king ramp currently damps) — if it appears, fall back to a LIGHT ramp (start=0.007, end=0.005) rather than the full flat; revisit the whole premise if a future change makes the paired bootstrap see seed noise it currently cannot (today the LCB gate binds at ~1.3% point improvement, which is why sub-0.3% margins are inert)"
relations: {revises: DEC-CA-0016, depends_on: DEC-CA-0012, relates_to: [DEC-CA-0033, DEC-CA-0039]}
---

GOAL (owner, 2026-09-07): make the king dethroneable MORE OFTEN without letting
the crown churn on noise ("beatable more often while not going crazy"). This
node records where that lever actually is — established by replaying the real
mainnet receipt trail, not by modelling — and rejects the levers that look
equivalent but buy beatability by loosening the noise floor.

**FINDING — the margin threshold VALUE is not what holds the king; the
fresh-king RAMP is.** Over 86 recent mainnet rounds (18 dethrones, 68 held), a
flat-bar sweep of the recorded LCBs shows every extra dethrone unlocked by
lowering the bar is bootstrap-confirmed real (LCB>0) and clusters at ≥1.36%
point improvement — and lowering the bar below ~0.3% unlocks NOTHING more:

    flat bar X   held→dethrone flips   of which LCB>0   noise (LCB≤0)   min point-improvement
      1.0%              4                    4               0              2.49%
      0.5%              9                    9               0              1.83%
      0.3%             11                   11               0              1.36%
      0.2%             14                   14               0              1.36%
      0.0%             14                   14               0              1.36%

No held round with under ~1% point improvement ever produces LCB>0: the real
noise guard is the **paired-bootstrap `LCB > 0` requirement at
`bootstrap_alpha`**, and empirically it binds at ~1.3% of score — well above any
margin we would set. So a "0.1% margin" is harmless but inert; the bar only does
work in the 0.3%–1% band, and there it is the FRESH-KING ramp
(`win_margin_start = 0.01`, decaying to `win_margin_end = 0.005` over
`margin_warmup_rounds = 8`, DEC-CA-0016) that rejects the real, LCB>0 challengers
— they cluster on fresh-king rounds because that is where the closest real
competition (the runners-up who just lost) sits.

**DECISION (proposed): flatten the ramp to the floor — do not touch the noise
gate.**

    win_margin_start   0.01  → 0.005      # collapse the fresh-king ramp to the floor
    win_margin_end     0.005 (unchanged)  # the single flat bar; still the hysteresis floor
    bootstrap_alpha    0.05  (unchanged)  # the noise gate stays flat for everyone
    margin_mode        "level" (unchanged)

A single flat **0.5% margin at every tenure**. First-order replay: unlocks the
**+9** bootstrap-confirmed real dethrones the ramp was over-protecting against
(dethrone rate ~21% → ~31%), with **zero** noise flips. The DEC-CA-0016 HARD
GUARDRAIL holds — `win_margin_end` stays 0.005 > 0, so the floor + `LCB>0` still
require a challenger to be 0.5%-better with bootstrap confidence to unseat
anyone, fresh or veteran. What is removed is only the EXTRA fresh-king
protection, not the noise/hysteresis floor. 0.3% (start=end=0.003, +11) is the
more-open option; below 0.3% is pure hysteresis erosion for no extra real
dethrones (the LCB gate binds).

**Levers considered and REJECTED — each buys the same beatability by loosening
the noise floor or by inverting the incentive:**

- **`LCB>0` only (margin → 0).** Removes the hysteresis band entirely; the
  per-round false-dethrone leak at `alpha=0.05` (higher under the cohort
  coverage limit, DEC-CA-0038) is then unbacked, and near-equal lineages
  ping-pong the crown every round. Violates the DEC-CA-0016 floor>0 guardrail.
- **Lower the FLOOR below 0.5% (`win_margin_end` → 0.003/0.002/0).** Buys almost
  nothing (LCB gate binds below ~0.3%) while eroding the hysteresis that damps
  ping-pong. A 0.001 floor was already rejected by DEC-CA-0016 as a "slow
  term-limit lottery."
- **FLIP the decay direction (fresh easy, established hard).** Entrenches
  veterans (higher bar the longer a hold — the OPPOSITE of "beatable more
  often") and thrashes freshly-crowned marginal kings. Wrong shape on both ends.
- **Make `bootstrap_alpha` tenure-dependent / decay it.** Conflates two
  orthogonal axes: `alpha` is the noise-tolerance / certainty knob, `margin` the
  effect-size / "worth-the-switch" knob. Varying alpha by tenure varies the
  false-dethrone RATE by tenure — no principled basis, and it is the "crazy"
  direction. The margin delivers the identical unlock without touching the noise
  floor (the +9 flips are all LCB>0), so there is no upside to reaching for alpha.
- **Raise `bootstrap_alpha` globally.** Also makes the king more beatable, but by
  loosening the LCB itself — it unlocks real AND noise dethrones. The margin
  unlocks only the already-confirmed-real ones. Keep the noise gate at 0.05.
- **Also arm the increment margin (DEC-CA-0039) on top.** Increment is a SECOND,
  more aggressive beatability lever (it re-prices the bar as a fraction of the
  per-round increment, so the bar goes near-irrelevant for a maturing lineage).
  Stacking it on a flattened ramp overshoots into churn. Pick one lever; the ramp
  flatten is the controllable one. (If increment arms at 9046800 as scheduled,
  re-evaluate this node against increment-mode LCBs before also flattening.)

**Reconciliation with the seed-variance bundle (DEC-CA-0033, armed:
`ema_decay = 0.999`, `gen_seed_mix = 3`).** Those mitigations shrink the
entrant-specific GENERATION-seed noise baked into the checkpoint. But the gate
that actually binds here is the EVAL-WINDOW noise the paired bootstrap resamples
— that is what keeps sub-1% point improvements from ever reaching LCB>0. EMA +
seed-mix do not move that floor; they are why the LCBs used in the replay are the
right (deployed-artifact) statistic. Chasing a sub-0.3% margin on the theory that
"seed noise is now small" is misdirected — the binding floor is the eval-window
LCB, and it is ~1%.

**Not crazy, by construction:** the extra beatability is entirely
bootstrap-confirmed real improvements (LCB>0, ≥1.36%) that the fresh-king ramp
was over-rejecting; the noise gate (`LCB>0` at `alpha=0.05`) and the hysteresis
floor (0.5%) are both untouched. Zero noise flips at every setting in the replay.

**Residual risk + arming gate.** The fresh-king ramp also damps CROWN PING-PONG
between near-equal strong lineages (A ~0.6% better than B's last checkpoint, B
~0.6% better than A's, alternating). Each hop would be real (LCB>0), not noise,
but it churns warm-start lineage every round. Whether such pairs exist in the
field is empirical: the arming gate is (1) re-run `scripts/replay_margin_decay.py
--start 0.005 --ends 0.005 --warmups 8` on the current trail and publish the
flip/near-miss counts (DEC-CA-0016's "evidence before schedule"), and (2) a full
testnet cycle showing no ping-pong. If ping-pong appears, ship the LIGHT ramp
(`win_margin_start = 0.007`, `win_margin_end = 0.005`) instead — just enough
hysteresis to break ties.

**Rollout is the DEC-CA-0016 shape, unchanged:** consensus param, block-gated via
`margin_activation_block` / `win_margin_start_prev` (set the prev to the current
0.01), ARMED AT RELEASE under one coordinated validator-upgrade window — a
config-mixed fleet forks verdicts exactly as the decay flip did
(docs/MARGIN_DECAY_ROLLOUT.md). `chain.toml` is NOT modified by this node; the
param flip is a separate owner-gated change after the replay + testnet gate pass.
