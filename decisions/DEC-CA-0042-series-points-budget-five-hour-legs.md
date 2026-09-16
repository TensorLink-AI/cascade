---
id: DEC-CA-0042
type: decision
title: "Multivariate advantage by construction: batches are 64 SERIES at every width, the token budget is denominated in series-points (a C-channel corpus trains C× the tokens in the same step count), and the wall grows to 5h while the budget stays the 3h figure — the wall stays the law"
status: accepted
date: 2026-09-16
amended: 2026-09-16 21:30 — target_train_hours stays 3.0 (the univariate budget was never meant to grow; only the wall moves to 5h). The first cut shipped 5.0 for one round (9082800, restarted 21:32 with the correction; contract digest f88909e9).
tags: [training, multivariate, budget, contract, trainer, audit, economics]
revisit_when: "the first 5h mainnet rounds report a wide (C ≥ 8) leg's deadline_hit tokens_frac — if wide legs stop at the wall well under budget the 'advantage' is wall-bound and the owner must choose between a width-scaled wall, a per-width step cap, or re-arming the 'sequences' fill; or a from-scratch (generation-0) run ever lands at a width whose wall-bound step count sits inside warmup_fraction (5% of the series-points budget) — the run would never leave warmup; or the worker image is not rebuilt before the first 5h round (an old worker bills C×L and stops wide legs C× early — a mismatched contract, not a slow pod)"
relations: {revises: DEC-CA-0041, depends_on: [DEC-CA-0026, DEC-CA-0001], relates_to: [DEC-CA-0036, DEC-CA-0037]}
---

## Decision

Three `[training]` changes, one cut (owner 2026-09-16):

1. **`batch_denomination = "series"`** — a batch is `batch_size` (64) series at
   every width. A C=32 batch is 64 × 32 sequences: Toto 2's own geometry, and
   the shape DEC-CA-0041's Phase 3 ablation measured as ~20% faster per
   sequence and ~40% lighter than the token-matched univariate batch.
   Reverses the `"sequences"` fill armed 2026-09-09.
2. **`budget_denomination = "series_points"`** (new, digest-bound
   drop-when-default; `"points"` = the legacy C×L billing) — one budget point
   is one TIME-STEP of one series, whatever its width. A `(C, L)` series costs
   `L`. The stream's stop (`element_points`), the trainer's counter
   (`batch_points`) and `cascade-audit`'s re-derivation all apply the one
   rule, because the corpus digest is the rolling digest of the consumed
   prefix and the stop rule therefore decides what the audit replays.
3. **`max_train_seconds = 18000`** (was 10800) with **`target_train_hours`
   unchanged at 3.0**: the wall grows to 5h, the budget stays 40B
   series-points. The wall is unchanged by width. (Amended 2026-09-16 21:30:
   the first cut also raised `target_train_hours` to 5.0, which grew the
   univariate budget to 66.6B — never the intent. Width is what buys tokens;
   hours stay fixed.)

Together: every corpus earns the same optimizer step count
(`budget / (64 × L)` ≈ 152k at the shipped numbers), and a C-channel corpus
trains `C×` the channel tokens per step — univariate 1×, C=32 → 32×. That is
the multivariate advantage the owner asked for, priced in exactly the unit
the miner controls (width), with no discount on the per-channel stream
billing (`[generator] max_total_points` still counts every value).

## Why not keep "sequences"

`"sequences"` (DEC-CA-0041) equalised steps by shrinking the batch to
`64 // C` series, i.e. it held tokens-per-step constant and made width a pure
diversity trade. That was the right call while the question was "does width
hurt"; it makes width worthless once the answer (Phase 3: no, it helps at
token parity) is in. The owner's intent is that a miner who emits coupled
channels gets MORE trained tokens for the same budget, not the same tokens in
a different arrangement. Holding the batch at 64 series and denominating the
budget per time-step is the smallest change that does that, and it is
bit-identical at C = 1 (every historical round; the golden fixture).

## What the wall does to it — stated, not hidden

DEC-CA-0001 stands: `max_train_seconds` is the law. A C-channel step costs
≈ C× the GPU time of a univariate one (Phase 3: linear in C, ~0.83× per
sequence). The series-points budget is the 3h figure (40B), so a C = 1 leg
fills it in ~3h on the reference SKU with 2h of wall to spare; a width-C leg
needs ≈ 0.83·C × 3h and therefore reaches only ≈ 5/(0.83·C·3) of its budget
before the 5h wall stops it:

    C     steps in 5h    tokens_frac    channel tokens vs C=1
    1        ~152k          100% (~3h)        1.0×
    2        ~152k          100% (~5h)        2.0×
    4         ~76k           50%              2.0×
    8         ~38k           25%              2.0×
    32        ~9.5k          6%               2.0×

So on the pinned SKU the realised advantage of width is ~2× the univariate
channel tokens (the extra 2h of wall plus the ~20% per-sequence efficiency),
delivered as C× wider, C× fewer updates — and every leg wider than C≈2 lands
`deadline_hit`. Scaling the wall with C (the owner's stated intent: wall AND
budget in proportion to channels, with a per-leg cap) is a worker change and a
follow-up decision, not part of this cut. That is the intended contract (a stop at the wall is a
first-class outcome, never an infra fault), and it is self-policing: a miner
who over-widens for the SKU trains fewer updates at their own expense. It is
NOT a step-count guarantee. Two consequences the operator must watch:

* **Warmup on a from-scratch run.** `warmup_fraction = 0.05` of the
  series-points budget is ~7.6k steps; a C ≥ 32 generation-0 leg never
  leaves warmup. Warm-started rounds (the live case under wsd) run flat and
  are unaffected. Do not open a new generation at high width.
* **Fleet geometry.** 5h legs fit 2 waves per lane in the 12h grid
  (`duel_waves_that_fit`: (12 − 1.5) // 5), not 3. `duel_seat_all` still seats
  the whole field; the provisioner's fit warning and the funded-round
  `_funded_rent_wait_deadline` follow `max_train_seconds` automatically.
  Miner-side, a funded seat bills up to ~5h (a univariate leg stops at its
  budget after ~3h on a good host) + ~1h bench.

## Rollout

* Pure trainer/worker policy under the declared-contract gate
  (`declared_contract_from_block = 8942400`): none of the four keys is a
  LOCKED term, so validators accept the declared body without a restart. The
  digest still moves (max_train_seconds is hashed; budget_denomination
  enters via drop-when-default) — chain.toml's transition note records the
  new value (f88909e9…).
* **Worker image rebuild is a hard prerequisite.** The v0.8.0 worker parses
  no `budget_denomination`; it would bill C×L and stop every wide leg C×
  early while the orchestrator's manifest declares series-points. Re-pin
  `train_image_digest` from the rebuilt image before the first 5h round.
* Testnet mirrors the pairing (`chain.testnet.toml`), fast-round hours
  untouched.
* `cascade-audit` reads the denomination off the round's own
  `contract_body`; rounds before this cut carry no key and replay under
  `"points"` — unchanged.

## Rejected

* **Scale the wall with C.** Makes the round length miner-chosen (a C=32 leg
  would be 130h) and breaks the epoch grid; the wall is the law.
* **Discount channel billing on the stream** (`max_total_points`). Mints
  cheap points; DEC-CA-0026's channel-spam argument stands. The advantage is
  delivered on the training side only.
* **A per-width step cap** instead of a token budget. Would need a
  width-conditional contract term; revisit only if the first rounds show the
  wall-bound regime is the operating point (see revisit_when).
