---
id: DEC-CA-0010
type: decision
title: "Finalist count responds to the screen's own statistic; the tied cohort all duel under a family-wise alpha"
status: active
date: 2026-07-30
tags: [koth, heat, scoring, bootstrap, eval, provision, audit]
revisit_when: "tied sets routinely hit max_finalists after the run-off (the run-off is not separating, so the bar or the run-off size is wrong); or the shadow max-statistic diagnostic shows the Bonferroni alpha/k is materially over-tightening under the real cohort correlation (then move the correction onto the joint bootstrap); or the run-off's wall-clock cap starts expiring on real fields, which would mean it is delaying the heat_complete marker and the JIT final rental rather than paying for itself; or cohorts routinely produce more than one margin-clearer, which would mean the run-off is advancing genuinely-separable challengers"
relations: {supersedes_behavior_of: DEC-CA-0006}
---
`[round] finalists` stops being a constant. The heat still **ranks** on the
observed `global_geomean` — that part of [[DEC-CA-0006]] is untouched, the
bootstrap still never re-ranks the field — but *how many* entrants advance now
responds to the screen's own decisiveness statistic. A leader the screen
separated advances alone. A leader it did not separate from gets its tied set
re-scored on a much larger eval; whoever still cannot be separated advances too,
capped. The validator then duels **every** advanced challenger against the king
under a family-wise-corrected alpha and crowns the best margin-clearer.

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

**The screen's bar is 0, not the duel's 2% margin, and that is deliberate.** The
margin exists because the *king* holds ties — incumbency and throne stability.
Among challengers there is no incumbent and no null; the question is not "is `j`
better by a decisive amount" but "can I safely discard `j`", and that is answered
when we are 95% confident the leader is at least as good. This is the same
asymmetry [[DEC-CA-0006]] used to argue the duel's LCB does not transfer as a
*ranking* rule. Used as a *drop* rule it fits, because dropping is exactly the
one-sided decision a one-sided bound is built for.

**Multiplicity is NOT corrected on the screen's drop rule.** A 5% one-sided bound
against N−1 entrants inflates the tied set — on a ~20-entrant field expect
roughly one false inclusion per round. No Bonferroni-style tightening is applied
*here*, because a bar that depends on field size would let the same two
generators separate or not depending on how many unrelated entrants showed up
that round — non-comparable across rounds, and gameable by padding the field. The
cap plus the run-off are the controls. The duel side is the opposite case; see
below.

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

## The duel: judge the whole cohort, do not stop at the first clearer

The design discussion's headline proposal was to duel sequentially — best point
estimate first, stop at the first challenger to clear the margin — so that
judging cost was bounded by the outcome rather than by the tie size. **That is
rejected, on measurement and on two structural grounds.**

**It optimises a cost that is not there.** A full paired duel eval is **106
seconds** of CPU for 2000 windows, both sides (measured 2026-07-30). The king is
evaluated once and reused across the cohort, so each additional challenger costs
~53s. At a cap of 3 the entire "saving" sequential-stop buys is under two
minutes — against a 3h L40S lane per finalist that was already spent to *train*
the checkpoint being judged. Once you have paid to train them, judging all of
them is free.

**It decides the throne by noise.** The premise for advancing more than one
challenger is that the screen could not separate them. So ordering them by point
estimate is ordering by the very noise that motivated advancing them. If two
challengers would both clear the margin, first-to-clear crowns whichever the
coin-flip ordering happened to put first — a fairness artifact invented by the
cost optimisation, on evidence we have explicitly declared insufficient.

**It is incompatible with the multiplicity correction below, and this is the
decisive point.** Correcting per-challenger alpha requires `k`, the number of
challengers tested. Under sequential-stop `k` is outcome-dependent: a cohort
where the first challenger clears tests one, a cohort where it fails tests more.
Choosing to stop early and keep the full alpha does not control family-wise
error — it is the classic optional-stopping failure. Using `k = advanced` instead
makes the correction sound but then sequential-stop saves nothing it did not
already fail to save. Judging the whole cohort makes `k` fixed before any verdict
exists, deterministic, and derivable from signed data. The two amendments are not
independent; the second forces the first.

**The rule.** Evaluate the king once per size. Evaluate every advanced challenger.
Clearers are `{j : LCB_j ≥ margin}` at the corrected alpha. Crown the clearer
with the best **observed relative improvement** — equivalently the lowest
`chal_geomean`, since the king's is shared — tiebroken on `(reveal_block, uid)`.

Gate on the bound, select on the point estimate. That split is [[DEC-CA-0006]]'s
own finding applied consistently: clearing the margin is a hypothesis test and a
one-sided bound is the right instrument, but picking the winner among clearers is
a *selection*, and ranking a selection by a bound penalises per-window dispersion
— a property the duel does not score. Crowning by highest LCB would re-import
exactly the error that node measured at −20pp.

**What stays sequential: the public-benchmark gate, and only it.** The gift gate
is a sidecar GIFT-Eval sweep, the one genuinely expensive step per challenger.
So it runs on the best clearer first; if an `enforce` gate blocks it, fall to the
next-best clearer and re-gate. Sequential logic was never wrong in general — it
was applied to the stage where cost was not.

**What stays from the sequential design: the inconclusive short-circuit.**
`min_windows` and `min_clusters` are properties of the window slice, identical for
every challenger, so if the cohort is inconclusive for one it is inconclusive for
all. Detect it once, hold the throne, do not evaluate the rest.

## Family-wise alpha: multiple challengers must not quietly weaken the margin

Each duel is a one-sided test at `bootstrap_alpha = 0.05` against a 2% margin.
Duel `k` challengers per round and a merely-equal cohort gets `k` draws at the 5%
tail, so the king's per-round false-dethrone probability scales roughly with `k`.
Arming tie-aware finalists without addressing this would silently weaken the 2%
margin *every time the screen ties* — which is precisely the rounds where the
feature activates. Throne stability under simultaneous challengers is the product
being protected, so this is where a correction belongs even though it does not
belong on the drop rule.

**Ship Bonferroni: per-challenger alpha becomes `bootstrap_alpha / k`**, with
`k` = advanced challengers actually duelled. At `k = 3`, α goes 0.05 → 0.0167.
Simple, and conservative in the protective direction — under the real correlation
structure (the `k` relative-improvement statistics share the king's scores and
one window draw, and [[DEC-CA-0006]] measured mean pairwise correlation 0.91
among entrant bag geomeans) Bonferroni over-tightens, meaning the king is
*over*-protected rather than under-. For a change whose failure mode is an
illegitimate dethrone, that is the correct side to err on.

**Record the tighter rule as a shadow diagnostic, do not obey it yet.** The
statistically right answer is a simultaneous bound off one shared resample — the
α-quantile of the max-statistic across the cohort, which prices the correlation
instead of assuming independence. The machinery already exists in the pattern
`joint_bag_geomeans` uses. Compute it, log it, and let it decide whether
Bonferroni's conservatism is costing real dethrones. This is exactly the
discipline [[DEC-CA-0006]] used on `p_best`: compute the better statistic, record
it, obey the simpler one until the logs justify the switch.

**Implementation trap worth naming.** Do NOT write the adjusted alpha into
`VerdictRecord.params`. `check_koth_params` asserts the recorded `KothParams`
equal published `chain.toml [scoring]`, so a per-round-mutated alpha would fail
its own audit. Record the unmodified config params; `k` is the count of
challenger entries in the **signed** manifest, so the audit derives
`α_eff = bootstrap_alpha / k` itself and replays. Strictly better than the
`wql_mode` precedent in `check_verdict`, which has to guess among known rules —
here the correction is a deterministic function of signed data.

**Streaks.** Only the crowned challenger's streak advances; every other duelled
challenger's streak resets, including a clearer that was not crowned. A streak is
a claim on the throne and only one challenger can hold that claim per round.
Dormant at `dethrone_cp = 1` but it must be correct.

## No heat-based dethrone-hopelessness gate. Ever.

The tempting saving is a pre-duel gate: score the king's previous checkpoint on
the heat windows (CPU, free) and refuse to advance challengers hopelessly below
it. **Do not build this.** The comparison is structurally biased against
challengers — 1h heat checkpoints on 4090s against a 3h L40S king — so every
skipped duel would be a round where dethroning was impossible by fiat, decided on
cross-hardware evidence the contract explicitly does not trust. That distrust is
the entire reason `expected_gpu` exists. On 2026-07-30 the apparently-hopeless
challenger u86 took the throne with an **8.3% LCB** the moment it got fair
silicon. The duel lane is what makes the throne legitimate; any heat-derived
hopelessness gate trades that legitimacy for a rounding error on the GPU bill.
Excess training of losing challengers is the honest price of a fair contest, and
it is small next to what one void round costs.

## Cost, honestly

| Cost | Bounded by |
|---|---|
| Validator eval | cohort size, at ~53s CPU per extra challenger — negligible |
| Final GPU training lanes | **the cap only** — `1 + n_advanced`, regardless of verdict |
| Run-off eval (CPU) | tied-set size × incremental windows, wall-clock capped |

Judging is not the cost; training is, and only the cap bounds it. That is why the
run-off is not optional garnish — it converts GPU lanes into CPU minutes before
the cap is ever reached. At mainnet prices the extra exposure at `max_finalists =
3` is roughly one to two additional L40S lanes, ~$8/round, against a
`max_spend_per_round` circuit breaker two orders of magnitude larger.

**The numbers must come from the DEPLOYED provisioner config, not this repo's
`deploy/provision.mainnet.toml`.** The deployed file has diverged (`max_spend`
500 vs the template's 120, final `max_price_hr` 3.00 vs 2.60, plus the 2026-07-30
heat-ladder rework). The pre-arm check runs against the deployed file and must
confirm two things at `1 + max_finalists` slots: that `within_budget` still
passes at worst case, and specifically that `final_pods` is **not clamped** by
`max_pods` — a clamped final still completes, but serially, which pushes 3h lanes
into the round's tail. In the template's shape (`gpus_per_pod = 2, max_pods = 2`)
a cap of 3 lands on exactly 2 pods, i.e. saturated with zero headroom.

## Carrying order and verifying it

`duel_rank` on `TrainedEntry`, dropped from the canonical body when `0`, still
ships — but its role is now record order, not outcome. It gives the receipt's
`entry_scores` a stable cross-validator order and the dashboard a meaningful one;
it no longer decides who takes the throne, which is the point of the amendment.
The convention is the one `bench_scores`, `eval_pool_key`, and `warm_start_ckpt`
already use, so every pre-existing manifest hashes byte-identically, old
signatures stay valid, and `MANIFEST_VERSION` does not move. Non-contiguous ranks
are fine (the final content-clone drop can remove one); the validator sorts
rather than indexes.

**No receipt format change.** `VerdictRecord` cannot gain fields — `asdict` goes
verbatim into the signed body, so a new field re-serialises every archived
receipt with bytes that were never signed, and `RECEIPT_VERSION` cannot be bumped
because `load_receipt` rejects any other version. Multiple `EntryScores` records
are per-round data, not a format change, and they already carry `hotkey`.

The audit gets **stronger**, and cohort judging is what makes it so — there is no
path dependence to reconstruct. A new `check_duel_cohort` asserts that every
recorded challenger's LCB reproduces at `α_eff = bootstrap_alpha / k`, that `k`
matches the challenger-entry count in the signed manifest, and that the crowned
challenger is the best observed relative improvement among the clearers. That
verifies the **selection**, not just the verdict, which nothing does today.

## Blast radius beyond the two stages

The validator currently collapses challengers by size (`chal_by_size = {e.size: e
for e in ...}`), as do `_pooled_scores` in the audit, `check_transition`,
`_maybe_run_benchmarks`, and `summarize_receipt` — so `finalists > 1` today would
silently judge, audit, and benchmark an arbitrary entrant. That is the same class
of silent-wrong-attribution bug the provisioner hit twice on 2026-07-30, and the
validator fix ships first regardless of what happens to the rest of this node.
`size_fleet` must size the final fleet off the **cap** so the pre-phased fleet and
the `within_budget` breaker cover the worst case; JIT rental already sizes off the
marker's actual finalist list ([[DEC-CA-0003]] rule 5) and adapts for free. The
heat's `(score, uid)` tiebreak becomes `(score, reveal_block, uid)` — a UID is not
a seniority claim (see [[NOTE-ca-operational-invariants]], and [[DEC-CA-0008]] on
earliest-commit).

## Rollout is ordered, and the order is not cosmetic

A validator that does not understand `k` challengers silently judges an arbitrary
one. So the validator, audit, and manifest field ship FIRST while the trainer
still emits one finalist — a pure no-op: `duel_rank` stays 0, `k = 1`, and
`α/k = α`, so the corrected rule is bit-identical to today's at a cohort of one.
Then the trainer run-off, config, and fleet sizing ship inert on mainnet
(`max_finalists = 1`, `tie_runoff_windows = 0`, the pattern
`dedup_mode`/`gift_gate_mode`/`cascade_enabled` all use). Then testnet arms.
Trainer and validator restart together — not for `contract_digest` (this touches
`[round]`/`[scoring]`, not `[training]`) but because the finalist count and the
duel rule are two halves of one decision, the coupling [[DEC-CA-0009]] called
out. Mainnet stays unarmed until a full testnet round has produced a genuine
multi-finalist cohort duel *and* `cascade-audit` has replayed it — the 2026-07-30
JIT incident existed precisely because a config change live-exercised a path that
had never run on mainnet.

## Deliberately NOT done

- **Sequential duelling / first-to-clear.** Measured at ~53s per extra
  challenger against 3h GPU lanes; decides ties by ordering noise; and cannot
  carry a sound family-wise correction (optional stopping). Recorded here because
  it was the discussion's headline proposal and the reasoning that killed it is
  worth keeping.
- **Crowning by highest LCB.** Ranking a selection by a bound penalises
  per-window dispersion, which the duel does not score — [[DEC-CA-0006]] measured
  that at −20pp in the adverse case. Gate on the bound, select on the point
  estimate.
- **A multiplicity correction on the screen's drop rule** (above: it would make
  the bar field-size-dependent and padding-gameable).
- **Any heat-based dethrone-hopelessness gate** (above: structurally biased,
  legitimacy cost, u86).
- **An uncapped tied set.** Trusting the run-off to always collapse the field
  would make GPU spend a function of how indistinguishable the field happens to
  be.
- **Any change to the metric, the margin, or the ranking rule.** This changes how
  many advance, and the alpha the cohort is judged at. Nothing else.
