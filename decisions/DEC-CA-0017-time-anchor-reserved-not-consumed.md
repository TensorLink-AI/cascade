---
id: DEC-CA-0017
type: decision
title: "Time anchor (start, freq): reserved in the carrier now, consumed by nothing — the fixed model is calendar-free by pinned architecture"
status: proposed
date: 2026-08-13
tags: [interface, generator, eval, scoring, trainer, seasonality]
revisit_when: "the calendar-feature ablation (below) shows a material GIFT-Eval gain at 4M — at which point consuming freq becomes an architecture decision with measured value; or a trainer-side consumer that needs no arch change is identified (none is known today); or the arch pin moves off released Toto-2.0 (whose input is values-and-mask only), which removes the structural blocker"
relations: {depends_on: DEC-CA-0016, enables: DEC-CA-0021}
---

Gap 1 as framed: "a generator cannot say *hourly, starting Monday, with
weekday seasonality*, yet the eval computes MASE from a seasonal period it
reads out of window metadata — miners are scored on frequency-dependent
behaviour they cannot generate. Largest hole, close to free."

**The framing is disputed on both halves, from the code.** The carrier field
lands (via DEC-CA-0016's reservation); the payload has no consumer in Phase 1
and is deliberately not accepted.

## Why "scored on behaviour they cannot generate" doesn't hold

1. **The eval's frequency use never touches the model.**
   `_resolve_seasonal_period` (`eval/scoring.py:87`) feeds only the MASE
   *denominator* — the seasonal-naive error computed on the real eval
   window's own history. It is a property of the eval data, identical for
   king and challenger, and paired out of the duel statistic. No forecast is
   conditioned on it: `forecast_fn` receives history values and nothing else
   (`scoring.py:120`, `evaluator.py:62`).

2. **The fixed model cannot consume a calendar anywhere.** The input per
   patch is `values ‖ mask` — `patch_embed = nn.Linear(patch_size * 2,
   d_model)` (`trainer/toto2_model.py:327`) — and positions are relative
   (xPos). There is no slot for day-of-week, hour, or absolute time at train
   *or* eval. This is not an accident of our implementation: released
   Toto-2.0 is calendar-free by design, and the arch is pinned to its
   `config.json` (`base_arch_digest`). Consuming `freq` means widening
   `patch_embed` — a new architecture, a `base_arch_digest` recompute, a
   `contract_digest` bump, and a deliberate divergence from the released
   model the whole identity is pinned to.

3. **Frequency-dependent behaviour is already expressible — as shape.** The
   only way this model can perceive "hourly with weekday seasonality" is as
   periodic structure in the values, and a generator can emit period-24 and
   period-168 structure today. What a miner cannot do is *label* it — and
   today the label would reach nothing.

So the hole is real but its edges are elsewhere than claimed: `(start, freq)`
matters for panel alignment (DEC-CA-0020), drift anchoring (DEC-CA-0021),
future-known covariates (DEC-CA-0022's calendars), and any future arch
generation that takes exogenous features. All of those are payload decisions
gated on consumers.

## The mechanism (settled now, via the carrier)

Named record fields, not positional convention — DEC-CA-0016 makes this the
only shape in town:

- `start`: scalar int64, UTC epoch-seconds of the first timestep. No
  timezones, no strings, no leap-second semantics — epoch arithmetic only.
- `freq`: pandas-style frequency string drawn from the
  `eval/seasonality.py:11` vocabulary (plus integer multipliers its
  `_normalise` already parses). Vocabulary-validated at `check_series` time
  so junk can't enter the digest.

Both optional; absent means what today means (no anchor). One channel-shared
anchor per series — per-channel timestamps are ragged data and out of the
carrier's shape (DEC-CA-0016 forecloses that knowingly).

## Change surface, migration, digest

- `chain.toml`: nothing until acceptance; at acceptance, the field joins the
  `[training]`-folded accepted set (DEC-CA-0016 G2 layer 3) → one digest bump.
- `cascade/interface/`: reservation now (rejected); validation rules above at
  acceptance.
- Trainer/eval: **nothing** — no consumer exists, and per DEC-CA-0016's
  refuse-unconsumed-payload rule that is exactly why acceptance waits.
- Migration: none, ever — deployed generators never carry the fields.
- Gaming surface at acceptance: `freq` is a self-declared label the data need
  not match. Any future consumer must treat it as a hint (a data statement to
  learn from), never a trusted fact — the same posture as every other
  miner-supplied byte. Mis-declared frequency mostly self-harms (the model
  learns from the values either way); it becomes adversarial only if a
  consumer *scores* on the declaration, which nothing may do.

## The open question that decides the payload, with its experiment

**Does a 4M-scale Toto2 gain anything from calendar features at all?** Run an
offline ablation on the reference trainer: the pinned 4M versus a variant
whose patch embedding takes `values ‖ mask ‖ sin-cos(hour, dow)` derived from
a synthetic anchor, both trained on the same corpus under the same budget,
scored on GIFT-Eval across its frequency slices. If the delta is noise at 4M
(plausible — 3.3M params is small for exogenous-feature routing), the payload
stays reserved through Phase 1 with a clear conscience and re-opens at the
22M size seam. If it is material, consuming it is an arch decision with a
measured prize, taken through the normal `base_arch_digest` re-pin protocol.

## Rank

Against the proposed ordering (#1 of six): **demoted**. Carrier cost is near
zero — that part ships in Stage 0 as reservation — but value-per-effort of
the *payload* is the lowest of the six until the ablation says otherwise,
because it is the only gap whose consumption is structurally blocked by the
arch pin rather than by eval or plumbing.
