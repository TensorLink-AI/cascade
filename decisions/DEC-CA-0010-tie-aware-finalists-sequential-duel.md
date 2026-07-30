---
id: DEC-CA-0010
type: decision
title: "Finalist count responds to the screen's own statistic; tied finalists duel sequentially"
status: active
date: 2026-07-30
tags: [koth, heat, scoring, bootstrap, eval, provision, audit]
revisit_when: "tied sets routinely hit max_finalists after the run-off (the run-off is not separating, so the bar or the run-off size is wrong); or sequential duels routinely reach challenger 3+ (the heat's point-estimate ordering carries no information, so ordering by it does not buy the expected-one-duel property); or the run-off's wall-clock cap starts expiring on real fields, which would mean it is delaying the heat_complete marker and the JIT final rental rather than paying for itself"
relations: {supersedes_behavior_of: DEC-CA-0006}
---
`[round] finalists` stops being a constant. The heat still **ranks** on the
observed `global_geomean` — that part of [[DEC-CA-0006]] is untouched, the
bootstrap still never re-ranks the field — but *how many* entrants advance now
responds to the screen's own decisiveness statistic. A leader the screen
separated advances alone. A leader it did not separate from gets its tied set
re-scored on a much larger eval; whoever still cannot be separated advances too,
capped, and the validator duels them **sequentially** — best point estimate
first, stopping at the first challenger that clears the LCB against the margin.

**Why now.** [[DEC-CA-0006]] declined this and wrote the trigger into its own
`revisit_when:` — "or `[round] finalists` rises above 1 — at which point 'advance
the statistically tied set' becomes a real choice rather than a no-op". It also
supplied the numbers that make the case: at the heat's ~320 window-equivalents
the screen picks the true best **~74%** of the time, ~89% at 2000 windows, and
one extra finalist takes "the true best is somewhere in the advancing set" from
90% to **~98%**. The screen has been logging `p_best` and `leader_lcb` since,
precisely so this decision could be made on evidence rather than taste.

**The tie criterion.** Off the *same* joint bootstrap the heat already runs
(`joint_bag_geomeans` — one shared cluster resample across every entrant, which
is what makes within-bag comparisons paired), for each non-leader entrant `j`:

    lcb_vs[j] = quantile( (bags[j] - bags[0]) / bags[j],  alpha )

The leader **separates** from `j` iff `lcb_vs[j] > 0`; the **tied set** is the
leader plus every `j` it does not separate from. This is character-for-character
the `leader_lcb` computation already in `cascade/eval/heat.py`, with `bags[1]`
generalised to `bags[j]` — no new statistic, no second bootstrap, and
`leader_lcb` survives as `lcb_vs[runner_up_key]` so the DEC-CA-0006 test pin
holds unchanged.

**The bar is 0, not the duel's 2% margin, and that is deliberate.** The margin
exists because the *king* holds ties — incumbency and throne stability. Among
challengers there is no incumbent and no null; the question is not "is `j` better
by a decisive amount" but "can I safely discard `j`", and that is answered when
we are 95% confident the leader is at least as good. This is the same asymmetry
[[DEC-CA-0006]] used to argue the duel's LCB does not transfer as a *ranking*
rule. Used as a *drop* rule it fits, because dropping is exactly the one-sided
decision a one-sided bound is built for.

**Multiplicity is not corrected, and that is a choice.** A 5% one-sided bound
applied against N−1 entrants inflates the tied set: on a ~20-entrant field
expect roughly one false inclusion per round by chance. No Bonferroni-style
tightening is applied, because a bar that depends on field size would let the
same two generators separate or not depending on how many unrelated entrants
showed up that round — non-comparable across rounds, and gameable by padding the
field. The cap plus the run-off are the honest controls instead.

**The run-off is the load-bearing half.** It re-scores only the tied set, only on
the windows it has not already scored, on CPU on the orchestrator, against heat
checkpoints still on local disk — no retraining and no GPU rent.
`RotatingWindowSource.windows_for_round` is a seeded permutation *prefix*, so the
256-window slice is a strict prefix of the 2000-window slice for the same round
seed: the run-off scores windows `[256, 2000)` and concatenates onto what it has,
and pairing holds by construction (which `joint_bag_geomeans` requires — it
raises on mismatched `abs_target`). At the measured screen rate (~30s per entrant
for 256 windows at 100 samples, ~0.117 s/window/entrant) the incremental 1744
windows cost **~3.4 CPU-min per tied entrant**; a 5-way tie is ~17 CPU-min. It
runs under a wall-clock cap mirroring `dedup_phase_seconds`, and on expiry falls
back to the pre-run-off tied set — a screen that cannot finish must not sink the
round it protects.

**Why sequential rather than parallel duels.** Cost becomes bounded by the
outcome rather than by the tie size. The king is evaluated once and reused across
every challenger (the windows are identical — that is what makes them paired), so
each extra challenger costs one extra challenger eval; expected spend is about
one duel, and the case where all of them fail is exactly today's outcome, the
king holds. You never pay for information you did not get. Note the inversion
that makes this comfortable: a wide tie means the screen discriminated nothing
this round, so that is when duel spend is *most* justified, not least — it is the
only stage left that can tell them apart.

**What sequential duelling does NOT bound.** Every advanced finalist still gets a
full `target_train_hours` GPU lane, because the trainer must publish the manifest
before any validator judges it. That cost is bounded only by the cap:

| Cost | Bounded by |
|---|---|
| Validator eval | **outcome** — sequential stop, expected ≈ 1 duel |
| Final GPU training lanes | **the cap only** — `1 + n_advanced`, regardless of verdict |
| Run-off eval (CPU) | tied-set size × incremental windows, wall-clock capped |

This is why the run-off is not optional garnish: it converts GPU lanes into CPU
minutes before the cap is ever reached. On mainnet prices (L40S at $1.30/GPU-hr,
~3h lanes) today's 2-lane duel is ~$8/round; `max_finalists = 3` makes the worst
case ~$16 against a `max_spend_per_round = 120.0` breaker.

**Sequence semantics.** A conclusive loss continues to the next challenger. A
win stops and is the verdict. An `inconclusive` round **stops** and the king
holds — `min_windows` and `min_clusters` are properties of the window slice,
identical for every challenger, so if the first duel is inconclusive every duel
would be, and continuing would be shopping for a decidable opponent. The
public-benchmark gate moves *inside* the loop: a challenger blocked by an
`enforce` gate has not won the round, so the sequence continues; an uncomputable
gate yields inconclusive, which stops. Each duelled challenger must cover the
king's full `throne_sizes` set — one missing a size is dropped rather than judged
on a subset, so every challenger faces the same evidence base.

**Duel order rides in signed bytes.** The manifest's `heat` block is deliberately
unsigned (a discarded heat checkpoint is not reproducible), so the order cannot
ride on the standings, and `_train_remote` does not preserve job order. A
`duel_rank` field on `TrainedEntry`, dropped from the canonical body when `0`,
carries it — the same convention `bench_scores`, `eval_pool_key`, and
`warm_start_ckpt` already use, so every pre-existing manifest hashes
byte-identically, old signatures stay valid, and `MANIFEST_VERSION` does not
move. Non-contiguous ranks are fine (the final content-clone drop can remove
one); the validator sorts rather than indexes.

**No receipt format change.** `VerdictRecord` cannot gain fields — `asdict` goes
verbatim into the signed body, so a new field re-serialises every archived
receipt with bytes that were never signed, and `RECEIPT_VERSION` cannot be bumped
because `load_receipt` rejects any other version. Multiple `EntryScores` records
are per-round data, not a format change, and they already carry `hotkey`. The
audit gets *stronger* rather than weaker: a new `check_duel_sequence` replays
every recorded challenger in `duel_rank` order and asserts that none before the
verdict's cleared the margin — it verifies the **selection**, not just the
verdict, which nothing does today. The deciding challenger is identified by
brute-force reproduction of the recorded `lcb`, the same self-verifying trick
`check_verdict` already uses for `wql_mode`.

**Blast radius beyond the two stages.** The validator currently collapses
challengers by size (`chal_by_size = {e.size: e for e in ...}`), as do
`_pooled_scores` in the audit, `check_transition`, `_maybe_run_benchmarks`, and
`summarize_receipt` — so `finalists > 1` today would silently judge, audit, and
benchmark an arbitrary entrant. Those are fixed as part of this, and the
validator fix is worth shipping on its own merits regardless. `size_fleet` must
size the final fleet off the **cap** so the pre-phased fleet and the
`within_budget` breaker cover the worst case; JIT rental already sizes off the
marker's actual finalist list ([[DEC-CA-0003]] rule 5) and adapts for free. The
heat's `(score, uid)` tiebreak becomes `(score, reveal_block, uid)` — a UID is
not a seniority claim (see [[NOTE-ca-operational-invariants]], and [[DEC-CA-0008]]
on earliest-commit).

**Rollout is ordered, and the order is not cosmetic.** A validator that does not
understand N challengers silently judges an arbitrary one. So the validator,
audit, and manifest field ship FIRST while the trainer still emits one finalist
(a pure no-op — `duel_rank` stays 0); then the trainer run-off, config, and fleet
sizing ship inert on mainnet (`max_finalists = 1`, `tie_runoff_windows = 0` ⇒
byte-identical behaviour, the pattern `dedup_mode`/`gift_gate_mode`/
`cascade_enabled` all use); then testnet arms. Trainer and validator restart
together — not for `contract_digest` (this touches `[round]`/`[scoring]`, not
`[training]`) but because the finalist count and the duel rule are two halves of
one decision, the coupling [[DEC-CA-0009]] called out. Mainnet stays unarmed
until a full testnet round has produced a genuine multi-finalist sequential duel
*and* `cascade-audit` has replayed it.

**Deliberately NOT done.** No change to the metric, the margin, `bootstrap_alpha`,
or the ranking rule — this is a change to *how many advance* and *in what order
they are judged*, nothing else. No multiple-comparisons correction (above). No
uncapped tied set: trusting the run-off to always collapse the field would make
GPU spend a function of how indistinguishable the field happens to be. And the
run-off does not re-rank with a bound — it buys **evidence**, which is what
[[DEC-CA-0006]] measured actually moves selection accuracy, rather than
post-processing the same noisy scores.
