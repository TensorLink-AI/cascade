# X announcement — warm-start promotes a set (DEC-CA-0013)

Source of truth: `decisions/DEC-CA-0013-warm-start-top-k-propose-and-verify.md`,
shipped in #191. Mainnet (netuid 91) is armed: `cascade_enabled = true`,
`cascade_top_k = 3`, `cascade_reign_rounds = 5`, `cascade_quality_epsilon = 0.05`.
Supporting research claims are sourced from `README.md`.

Primary draft is the **investor thread**. The technical thread at the bottom is
the same news for people who already know the mechanism.

---

## Option A — investor thread (9 posts, zero assumed knowledge)

**1/**

> A forecasting model with about 10,000 parameters recently beat one roughly a
> thousand times its size.
>
> Not a better architecture. Better training data.
>
> That result is the whole thesis behind what we're building. This week it got
> considerably more powerful.

**2/**

> Start from the beginning.
>
> Every company forecasts something: demand, load, prices, staffing, risk.
> Almost all of it still runs on bespoke models, built one at a time, by hand,
> by people who are expensive.

**3/**

> The prize is one model that forecasts anything out of the box — point it at a
> series it has never seen and get a usable answer.
>
> The same shift LLMs brought to text, aimed at the numbers businesses actually
> run on.

**4/**

> The field has quietly converged on how you get there, and it isn't size or
> architecture. It's the synthetic data you train on.
>
> Amazon trained a version of Chronos-2 on *purely synthetic* data. It landed
> within ~1 point of the full model.

**5/**

> So cascade makes that the competition.
>
> We freeze the model. Identical architecture, identical training, every time.
> Anyone can submit a data generator, and we train that same model on each one.
>
> Only the data varies. A controlled experiment, with money on it.

**6/**

> Until now there was a ceiling on it.
>
> Every round started from a blank model. Winners won, got paid, and the next
> round began again from zero.
>
> A tournament with no memory. The competition was real, but nothing
> accumulated.

**7/**

> That's what changed this week.
>
> Winning models now become the starting point for the next round. Each
> generation of data begins from everything the last one learned.
>
> The tournament became a compounding asset.

**8/**

> With one trap we deliberately avoided.
>
> The easy build promotes the single best winner — and then every future model
> descends from one ancestor. One lucky result and the network is locked into a
> dead end, with nothing left to compare against.

**9/**

> So we promote three, not one.
>
> Three lineages advance in parallel and later generations select across all of
> them. It costs zero extra compute; the network was training these models
> anyway.
>
> A portfolio, not a single bet.

**10/** (close — optional, drop if the thread runs long)

> Live now on Bittensor subnet 91. Every round is signed and published —
> training records, scores, checkpoints — so the whole history is auditable
> rather than asserted.
>
> Open competition on the one input that decides who wins.

---

## Option B — investor single post

> A 10,000-parameter forecasting model recently beat one ~1,000x its size. The
> difference was the training data.
>
> cascade freezes the model and competes the data. Winners now carry forward —
> three lineages in parallel, so progress compounds instead of restarting each
> round.

---

## Option C — investor single post, harder edge

> We ran an open competition for AI training data for months. It had one flaw:
> every round started from zero. Nothing accumulated.
>
> That's fixed. Winners now seed the next generation — three parallel lineages,
> no extra compute.
>
> A tournament just became a compounding asset.

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
- Research claims (10k-parameter model, purely-synthetic Chronos-2 ablation) are
  cited with links in `README.md` — worth attaching as a reply if the thread
  gets traction.
