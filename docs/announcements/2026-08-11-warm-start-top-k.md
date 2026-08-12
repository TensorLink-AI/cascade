# X announcement — warm-start promotes a set (DEC-CA-0013)

Source of truth: `decisions/DEC-CA-0013-warm-start-top-k-propose-and-verify.md`,
shipped in #191. Mainnet (netuid 91) is armed: `cascade_enabled = true`,
`cascade_top_k = 3`, `cascade_reign_rounds = 5`, `cascade_quality_epsilon = 0.05`.
Supporting research claims are sourced from `README.md`.

Primary draft is the **investor thread**. The technical thread at the bottom is
the same news for people who already know the mechanism.

---

## Option A — investor thread (8 posts, zero assumed knowledge)

No third-party model is named in this thread. Outside research is backup for
replies, not opening material — see the notes.

**1/**

> For months we've run an open competition to find the best training data for
> forecasting AI.
>
> It had one flaw. Every round started from zero. Winners got paid, then the
> next round began from a blank model. Nothing accumulated.
>
> That changed this week.

**2/**

> Why this is worth caring about.
>
> Every company forecasts something: demand, load, prices, staffing, risk.
> Nearly all of it still runs on models built one at a time, by hand, by
> expensive people.
>
> The prize is one model that forecasts anything out of the box.

**3/**

> Our bet is that the winner won't be decided by architecture. It'll be decided
> by what the model is trained on.
>
> So we freeze the model — identical every time — and compete the data. Anyone
> can submit a generator. We train that same model on each one and score the
> forecasts.

**4/**

> That gives a clean read on data quality, and it's been running and paying out.
>
> But with no memory, the competition could only measure. It couldn't build.
> Round 100 started from the same blank model as round 1.

**5/**

> Now winners carry forward. The best models from each stretch become the
> starting point for the next round of competition.
>
> Every generation of data begins from everything the last one learned. The
> tournament became a compounding asset.

**6/**

> One trap to avoid: promote a single winner, and every future model descends
> from one ancestor.
>
> Nobody can tell in advance which direction is the right one. Betting the whole
> network on the current leader is the move you can't reverse.

**7/**

> So we promote three, not one.
>
> Three lineages advance in parallel, and later generations select across all of
> them. It costs zero extra compute — the network was training these models
> anyway.
>
> A portfolio, not a single bet.

**8/**

> Live now on Bittensor subnet 91. Every round is signed and published —
> training records, scores, checkpoints — so the whole history is auditable
> rather than asserted.

---

## Option B — investor single post

> We've run an open competition for AI training data for months. It had one
> flaw: every round started from zero. Nothing accumulated.
>
> Fixed this week. Winners now seed the next generation — three lineages in
> parallel, no extra compute.
>
> A tournament became a compounding asset.

---

## Option D — technical thread (for the dev/miner audience)

**1/** Promoting the single best checkpoint is the obvious design. It's also how
you funnel an entire subnet down one trajectory. cascade's warm-start now
promotes a *set*. Live on SN91.

**2/** Context: cascade holds the model fixed — a Toto2 backbone — and scores the
synthetic data generators feeding it. Every round trains from scratch, so the
only variable is data quality. Warm-start changes where "scratch" starts.

**3/** The naive version promotes one winner. Every future round then descends
from that one checkpoint. One lucky init and the search is in a corner, with no
second opinion left in the population.

**4/** So promotion carries up to 3 members, and rounds rotate across them. The
field trains 3 parallel lineages at zero extra GPU. Each round still trains from
ONE shared init — the controlled experiment is untouched. The parallelism is
across rounds; the next generation prunes across lineages.

**5/** Members are drawn from BOTH sides of the duel — king and challengers.
Different generators are genuinely different data distributions, the deepest
diversity on offer. Promotion pays the checkpoint's owner nothing; the floor
moves for everyone at once.

**6/** The trust model flipped too. Validators used to re-derive the winner from
the trainer's own signed numbers. Now the trainer proposes a signed record and
validators verify an envelope: provenance, quality floor, reign ripeness, set cap.

**7/** Frontier TSFMs are won on the synthetic prior — Chronos-2, FlowState,
TempoPFN, DynaMix all point the same way. Keeping several lineages alive is how a
search over priors stays a search, instead of a hill climb into the nearest local
max.

---

## Notes before posting

- **k is 3** on mainnet, 2 on testnet. "Lots of trajectories" overstates it; the
  drafts say three.
- The lineages are parallel across **rounds**, not simultaneous within a round.
  Every round still pins one `warm_start_ckpt`. Don't let an edit blur this — it
  is the claim a technical reader will check.
- "Zero extra compute" is exact: the promoted checkpoints come from training the
  network already performs. Safe to keep.
- SOTA is framed as the goal, never as a result. No benchmark number appears
  that isn't already on the public scoreboard.
- The investor thread carries **no** token, price, emission, or return claim, and
  shouldn't acquire one — the traction story is the mechanism and the public
  audit trail.
- DEC-CA-0014 (shadow scratch control, reseed valve) is **not built**. Don't
  reference it, even though it's the natural answer to "doesn't this let errors
  compound forever?" The shipped answer to that reply: a quality floor keeps
  every promoted member within 5% of the best score of the reign, and challenger
  models are in the candidate pool, so the lineages don't all descend from the
  incumbent.
- **Outside research stays out of the thread.** Earlier drafts opened on DynaMix
  (~10k params beating a model ~1,000x its size). Cut deliberately: it spends the
  opening on someone else's model, argues a premise nobody has challenged, and
  hands a skeptic the question "if a 10k-parameter model already wins, why do you
  need a GPU fleet?" Post 3 states the data-over-architecture bet as ours and
  moves on.
- Keep that research as **reply ammunition** for the two challenges it actually
  answers, both cited with links in `README.md`:
  - *"Why believe data decides it?"* → Chronos-2's purely-synthetic ablation
    lands within ~1 point of the full model; Toto 2.0 is 57.5% synthetic with
    zero public series and tops GIFT-Eval.
  - *"Why keep three lineages instead of backing the leader?"* → DynaMix trained
    on nothing but 34 chaotic dynamical systems and beats Chronos zero-shot on
    unseen traffic and weather ([arXiv 2505.13192](https://arxiv.org/abs/2505.13192)).
    No one reasons their way to that recipe, which is the case for not
    collapsing the search.
