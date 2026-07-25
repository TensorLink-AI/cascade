---
id: DEC-CA-0006
type: decision
title: "Heat screen: LCB machinery is a diagnostic, never the selection rule"
status: active
date: 2026-07-25
tags: [koth, heat, scoring, bootstrap, eval]
revisit_when: "logged heat diagnostics show leader_lcb routinely at or below 0 (the screen is not separating first from second), or [round] finalists rises above 1 — at which point 'advance the statistically tied set' becomes a real choice rather than a no-op"
relations: {}
---
The duel's paired-bootstrap LCB is NOT extended into a selection rule for the
heat. The heat keeps ranking on the observed `global_geomean` and advancing the
top `[round] finalists`. The bootstrap is computed anyway and recorded as a
shadow diagnostic (`p_best` per entrant, `leader_lcb` vs the runner-up) so the
screen's decisiveness is finally observable.

**Why the LCB does not transfer.** The duel is a hypothesis test — is the
challenger better than the king by at least a margin, with the king holding on
ties. That asymmetric null is what makes a one-sided lower bound the right
instrument. The heat is a *selection*: pick k of N, no incumbent, no null. With
`finalists = 1` the pick is the argmin whatever bound you wrap around it.

**Why "rank by lowest upper confidence bound" specifically was rejected.** It
reads like the conservative choice — prefer the entrant whose worst case is best
— and it targets a real problem (the heat has a genuine winner's curse: argmin
over N noisy scores systematically promotes the luckiest draw). But simulation
on the heat's actual structure (N entrants, one shared window slice from
clustered feeds, multiplicative scores) says it does not pay:

- Mean pairwise correlation between entrants' bootstrap draws is **0.91** —
  ~91% of an entrant's marginal variance is shared window difficulty, identical
  for everyone. The bound shifts the whole field by nearly the same amount; the
  spread of per-entrant SEs is ~4.5%. It barely re-ranks.
- Where it does bite, it penalises **per-window dispersion** — which is not what
  the duel scores. A generator whose model is erratic across windows but equally
  good in expectation gets demoted in the heat for a property the final ignores.
- Measured over 6 seeds x 400 trials, picking the truly-best entrant:
  homogeneous field −0.5pp vs the point estimate; dispersion independent of
  quality +1.4pp (inside noise); dispersion correlated with quality **−20pp**.
  A wash on average with a large downside tail.
- Making it *paired* (contrast against the field mean within each bag, removing
  the shared term) does not rescue it — it sharpens the dispersion penalty and
  gets worse, −32pp in the adverse case.

`p_best` from a joint bootstrap is the rule that matches the objective and is
the only one never meaningfully negative (+4.0pp in the adverse case, neutral
elsewhere), but the gain is too small to justify overriding the observed
ranking. So it is recorded, not obeyed.

**What actually moves heat selection accuracy** is evidence, not
post-processing: at the heat's current 320-ish window-equivalents the screen
picks the true best ~74% of the time; at 2000 windows ~89%. One extra finalist
takes "the true best is somewhere in the advancing set" from 90% to ~98%. Both
cost real resources — screening windows are CPU (see the sequential-screening
caveat in `_build_screen_fn`), an extra finalist is a full final at every size
and collides with the fleet sizing DEC-CA-0003 does at the `heat_complete`
marker. Neither is taken here; the logged diagnostics are what should decide it.

**Related fix, shipped with this.** `eval.scoring.global_geomean` aggregated MASE
*arithmetically* while `bootstrap._bag_geomeans` aggregated it *geometrically*,
so the heat ranked on (and the receipt reported) the heavy-tail-sensitive form
the duel deliberately rejected — worst exactly where the heat lives, on a
fraction of the windows and samples. They are now the same function:
`global_geomean` equals one bootstrap bag on the identity resample, pinned by
test. This changes the reported `king_geomean`/`chal_geomean` on the round
receipt (regenerated golden fixture); it does NOT change any decision — the
dethrone LCB never called `global_geomean` — and `RECEIPT_VERSION` stays 3
because the wire format is unchanged.
