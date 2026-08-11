# X announcement — warm-start promotes a set (DEC-CA-0013)

Source of truth: `decisions/DEC-CA-0013-warm-start-top-k-propose-and-verify.md`,
shipped in #191. Mainnet (netuid 91) is armed: `cascade_enabled = true`,
`cascade_top_k = 3`, `cascade_reign_rounds = 5`, `cascade_quality_epsilon = 0.05`.

Claims below are checked against that node and `chain.toml`. Anything not
checkable there was cut.

---

## Option A — thread (7 posts)

**1/** (192 chars)

> Promoting the single best checkpoint is the obvious design.
>
> It's also how you funnel an entire subnet down one trajectory.
>
> cascade's warm-start now promotes a *set*. Live on SN91.

**2/** (271 chars)

> Context: cascade holds the model fixed — a Toto2 backbone — and scores the
> synthetic data generators feeding it. Every round trains from scratch, so the
> only variable is data quality.
>
> Warm-start changes where "scratch" starts: progress compounds across
> generations.

**3/** (259 chars)

> The naive version promotes one winner. Every future round then descends from
> that one checkpoint.
>
> One lucky init and the whole search is in a corner, with no second opinion
> left anywhere in the population. You've Goodharted the aggregate you selected on.

**4/** (277 chars)

> So promotion now carries up to 3 members, and rounds rotate across them.
>
> The field trains 3 parallel lineages at zero extra GPU. Each round still
> trains from ONE shared init — the controlled experiment is untouched. The
> parallelism is across rounds; the next generation prunes across lineages.

**5/** (269 chars)

> Members are drawn from BOTH sides of the duel — king and challengers.
> Different generators are genuinely different data distributions, which is the
> deepest diversity on offer.
>
> Promotion pays the checkpoint's owner nothing. The floor just moves, for
> everyone at once.

**6/** (278 chars)

> The trust model flipped too. Validators used to re-derive the winner from the
> trainer's own signed numbers — checking the trainer's arithmetic on the
> trainer's data.
>
> Now the trainer proposes a signed record and validators verify an envelope:
> provenance, quality floor, reign ripeness, set cap.

**7/** (276 chars)

> Frontier TSFMs are won on the synthetic prior — Chronos-2, FlowState,
> TempoPFN, DynaMix all point the same way. cascade makes that prior an open
> competition.
>
> Keeping several lineages alive is how a search over priors stays a search,
> instead of a hill climb into the nearest local max.

---

## Option B — single post (270 chars)

> cascade's warm-start no longer promotes a single winner.
>
> After a 5-round reign, the top 3 checkpoints — king *and* challengers — become
> the next generation's starting set. Rounds rotate across them: 3 parallel
> lineages, zero extra GPU.
>
> A search over data priors, not a hill climb.

---

## Option C — single post, blunter (243 chars)

> Promote one winner and every future round descends from one checkpoint.
> That's not a search, it's a hill climb.
>
> cascade's warm-start now promotes a top-3 set. The field trains 3 lineages in
> parallel, at zero extra GPU cost.
>
> Live on SN91.

---

## Notes before posting

- k is **3** on mainnet, 2 on testnet. "Lots of trajectories" overstates it —
  the drafts say three.
- The lineages are parallel across **rounds**, not simultaneous within a round.
  Every round still pins one `warm_start_ckpt`. Post 4 says so; don't cut that
  line, it's the claim someone will check.
- SOTA is stated as the goal, not a result. No benchmark number is claimed here
  that isn't already on the public scoreboard.
- DEC-CA-0014 (shadow scratch control, reseed valve) is **not built**. Don't
  reference it, even though it's the natural answer to "doesn't warm-start let
  errors compound forever?" The honest reply to that in-thread is: quality floor
  (every member within 5% of the best verifiable reign score) plus the challenger
  side of the pool; the scratch control is planned, not shipped.
