# Scope: a sampled warm-start init pool

Status: **proposal, not decided.** Written 2026-08-10 against `main` (Cascade
armed on mainnet 2026-08-05: `cascade_enabled = true`, `cascade_reign_rounds = 5`,
12h rounds ⇒ a promotion roughly every 2.5 days).

Two variants are on the table. They differ enough that they should be judged
separately:

- **Variant A — sampled promotion.** When a Cascade fires, draw the promoted
  init from the reign's top `k` instead of taking the argmin. The init is still
  fixed between promotions. Scoped in §7; the short version is that it is weak,
  because the pool is one miner's own checkpoints.
- **Variant B — a standing pool, drawn per round.** Maintain a pool of `K`
  checkpoints. Every round, draw one; that init is the shared starting point for
  every generator in that round (heat and final alike). The init rotates
  round-to-round.

**B is the real proposal and this document is mostly about B.** It is a
substantially better idea than A, and it is also a bigger change: it converts
Cascade from a ratchet with one moving floor into a population with a rotating
one, and it is paid for in model depth.

---

## 1. What B changes and what it doesn't

**Unchanged — the controlled-experiment invariant holds.** Within a round, every
generator still trains from one identical init (`trainer/loop.py:1973-2027`
already applies the same `warm_start` to heat *and* final — the screen must rank
on the init the final trains at). The duel stays a paired within-round
comparison, so a varying init cannot bias king-vs-challenger.

**Unchanged — the wire format.** `warm_start_ckpt` / `warm_start_size` are
already **per-manifest** fields (`shared/manifest.py:334`), and `cascade-audit`
already re-derives a round from whatever pointer that manifest pins
(`audit/rederive.py:220`). A per-round varying init is already representable.
No `contract_digest` change, no receipt-format change — which matters, because
DEC-CA-0009 established that receipt-format churn breaks the signed audit trail.

**Unchanged — the enforcement seam.** `_check_warm_start`
(`validator/loop.py:216`) already gates every manifest on "the signed
`warm_start_ckpt` equals the init this validator's own deterministic promotion
installed". Under B that generalises to "equals the draw from the published pool
under this round's `base_seed`". The gate exists; only its `expected` expression
changes.

**Changed — where the init comes from.** Today: one pointer file written by the
validator's Cascade install, read by the co-hosted trainer (`_load_warm_start`,
`trainer/loop.py:1611`). Under B: a published pool plus a per-round draw that
both sides compute independently.

## 2. Why B is a real anti-gaming mechanism (where A is not)

Variant A's pool is a single reign's checkpoints — all produced by the reigning
king, one round apart, same lineage. Randomising among artifacts the adversary
wholly owns reduces its precision, not its power.

B is different on every count that matters:

- **The pool spans lineages and owners**, so no single miner controls it.
- **It defeats standing-init specialisation, not just lead time.** With a fixed
  init held for 5 rounds, a generator can be tuned to that specific checkpoint's
  quirks — produce exactly the data that patches *this* model's weaknesses. That
  is a stable, learnable exploit. Under B the generator must be good across the
  whole pool, in expectation, without knowing which draw it faces.
- **The timing already makes the draw unguessable, for free.** `cascade deploy`
  targets its timelock reveal at `epoch boundary − reveal_margin_blocks` (25
  blocks, `chain.toml:264`). The round's `base_seed` is
  `seed_from_block_hash(epoch_block_hash)` at the boundary block — which does not
  exist when the generator is revealed. **A miner therefore commits its generator
  before the init it will be trained on is knowable.** No new mechanism is needed
  to get this property; it falls out of the existing schedule.
- **It shrinks the blast radius of a bad init** from "every round until the next
  promotion" (5 rounds) to "1 round in `K`". A checkpoint that got into the pool
  on benchmark luck poisons a fraction of rounds rather than all of them.

### The strongest argument is actually about attribution, not gaming

The repo's stated premise is that holding the model fixed turns "is this model
good?" into "is this *data* good?" (`README.md`, `docs/ARCHITECTURE.md:8`).
Warm-start already softens that: with a fixed init, what is measured drifts
toward "is this data good *for this particular checkpoint*" — data quality
confounded with init affinity.

Drawing the init per round marginalises over the pool and pulls the measurement
back toward the thing the subnet claims to measure. That is a principled
argument grounded in the design's own premise, and it is stronger than the
anti-gaming framing. It should be the headline justification if B is written up
as a decision.

## 3. The cost: this trades model depth for robustness, directly

Cascade's purpose per its own docstring is a rising floor — "one king held the
throne long enough that its best checkpoint should become the new floor the whole
field trains up from". A rotating pool has no single floor.

The arithmetic is unforgiving. Total training compute is fixed at 1× — one round
per epoch. If the pool holds `K` independently-advancing lineages, each advances
on ~1/`K` of rounds, so mean pool depth grows `K`× slower than a single chain.
**You cannot have 10 deep lineages on 1× compute.** At `K = 10` the ratchet
effectively stops; the subnet would be running a population search instead of
accumulating a pretrained model, and the pretraining-completion framing on the
website stops meaning what it says.

Two further dynamics that are easy to miss:

- **Score-based eviction collapses the pool.** If members are evicted worst-first
  on `cascade_score`, and score correlates with depth (it will), the deepest
  lineage's descendants survive and everything else is evicted. The pool
  converges to near-clones of one lineage and you are back to a fixed init with
  extra machinery. Genuine diversity needs eviction that is *not* purely
  score-based — age-based, or a per-lineage cap.
- **Naive promotion silently `K`×-slows the ratchet.** A checkpoint trained from
  a stale, shallow init scores worse in absolute terms and never gets promoted —
  so promotions only ever happen on rounds that happened to draw the deepest
  member. Fix: score a checkpoint for pool admission by its **improvement over
  the init it trained from**, not its absolute bench score. Without that
  refinement, B degrades to "the ratchet advances 1/`K` as often".

## 4. Pool construction is the whole design (and there are two families)

**(i) Parallel lineages — a true population.** `K` independent chains, each
advanced when drawn. Maximum diversity, maximum depth cost, needs `K`× compute
to keep pace with today's ratchet. Not affordable.

**(ii) Ancestral — the last `K` promoted checkpoints.** Zero extra compute. The
ratchet keeps running exactly as today (promotion still selects the best
checkpoint of a reign); the pool is `{floor, floor−1, …, floor−K+1}`. Diversity
is ancestral rather than parallel — an init from five promotions ago is
genuinely a different model, not a perturbation. The cost is that a fraction of
rounds train from a stale init.

**(ii) is the practical family, and it has a clean knob.** Weight the draw
toward recent promotions — geometric weights, say — so the current floor is drawn
most of the time and older members occasionally. One parameter moves the whole
depth-vs-diversity trade continuously, and it **degrades to today's exact
behaviour** when all mass sits on the floor. That makes it shippable dark and
tunable in production, which a hard `K`-way uniform draw is not.

Recommendation on size: **`K` = 2–3, not 10.** Marginalising over three inits
already breaks per-init specialisation — an exploit has to work on all of them —
while `K` = 10 pays ~10× in depth for diminishing attribution return.

## 4a. Who is eligible: king only, or challengers too?

Today only the reigning king's checkpoint is logged (`_record_king_checkpoint`,
`validator/loop.py:586`). But the trainer benches **both duel entries** each
round and both land in the signed bench report, so challenger checkpoints are
already available at no new compute.

**Include them.** A pool drawn only from kings is, over a long reign, one
miner's lineage — which is exactly the objection that sinks Variant A. Multi-owner
membership is what gives the pool its anti-gaming property, and the duel verdict
is answering a different question anyway: the duel ranks *generators* on the
private eval pool, while pool membership asks "is this a good place to start
from", measured on public suites. A challenger that lost its duel can still be a
good init.

Two consequences to go in with eyes open:

- **A thin new attack surface — thinner than it first looks.** Pool membership
  carries **no reward**: the weight vector is `decayed_share_vector` over the
  current king and `former_kings` (`shared/chain.py:37`,
  `[scoring] reward_prior_kings`), and nothing in it references the warm-start
  pool. So there is no direct payoff for getting a checkpoint in. The residual
  path is indirect: a generator co-adapted to init X wins duels it would
  otherwise lose on the ~1/`K` of rounds that draw X, and collects through the
  ordinary king/court channel. That path is mostly self-closing — the only way
  into the pool is through the duel, the only way into the duel is winning the
  heat, and the heat ranks on the same eval metric the duel decides on. A miner
  cannot land a *deliberately odd* checkpoint while being odd; the mechanism
  admits it only for being good, and a good checkpoint is a normal-good init.
  Worth noting, not worth designing around.
- **It makes the reign clock vestigial.** If membership is continuous admission
  from every benched checkpoint, nothing is left for the reign clock to decide —
  "when has a king reigned long enough to promote" stops being a question. That
  deletes the subject matter of a live, armed mechanism (DEC-CA-0004). It may
  well be a simplification worth having, but it must be decided deliberately,
  not absorbed as a side effect of a pool change.

### 4a-bis. Should pool membership be rewarded at all?

It isn't today, and that absence is doing real work. It is *why* §4a's attack
surface is thin: with no payoff attached to membership, there is nothing to
manipulate membership *for*. **The lack of a reward channel is a security
property, not an oversight** — and any proposal to add one should be read as
manufacturing the gaming pressure this whole design exists to remove.

The argument on the other side is real enough to write down. Miners are paid
purely for round-local duel wins; nobody is paid for producing a good *init*, and
the init is the artifact that compounds into the thing the project measures
itself on (a deep pretrained model). Pool quality is therefore an
**unincentivised byproduct**. Mechanically, paying for it would be easy —
`former_kings` already establishes a decayed multi-recipient court, so a "pool
court" would slot straight into `decayed_share_vector`.

**Lean: don't add it.** The correlation between "wins duels" and "is a good
init" supplies the alignment for free, and paying for membership would trade that
free alignment for a new gameable surface.

But name the assumption, because it is load-bearing: **pool quality is free only
while duel-winning and init-quality stay correlated.** If they diverge — if
generators that win on the private eval pool start producing checkpoints that are
poor bases for further training — the pool degrades silently, with no signal
anywhere in the system. That is a thing to watch for, not a thing to fix now.

## 4b. What criterion picks members: rank, or metric orthogonality?

The tension is real: top-`N` by `cascade_score` gives `N` checkpoints that are
similar *because* they were ranked on one aggregate, and B's whole argument
needs them to differ. But selecting for orthogonality across the six bench
numbers is the wrong fix, for four reasons in ascending order of severity.

1. **The six numbers are not six dimensions.** They are a 3×2 grid — suite
   (GIFT-Eval / BOOM / TIME) × metric type (CRPS = distributional calibration,
   MASE = point accuracy) — and within a suite both are computed from the same
   forecasts over the same datasets, so they move together. The honest axes are
   "domain" and "calibration vs accuracy": two, and both sit under a dominant
   overall-quality component.
2. **The sample is too small to estimate what the method needs.** A covariance
   structure in six dimensions from the ~10 candidates a pool would ever see is
   not an estimate.
3. **Selecting on the orthogonal residual selects on noise.** This is
   [[DEC-CA-0006]]'s finding wearing different clothes: there, ~91% of marginal
   variance was shared window difficulty and ranking on the residual penalised
   dispersion the duel doesn't score. Here the leading component is "this
   checkpoint is better"; what remains after removing it is largely bench noise —
   and with one bench run per checkpoint there is no repeated measurement to
   separate residual signal from residual noise.
4. **Decisive: "be unusual" is far easier to game than "be good."** The criterion
   is being chosen for anti-gaming reasons, and it is *more* gameable than the one
   it replaces. Ranking top-`N` by score requires producing a good checkpoint;
   being an outlier in metric space just requires being weird — a deliberately
   narrow predictive distribution buys a strong MASE with a terrible CRPS for
   free. An orthogonality rule pays miners to submit outliers and then seeds the
   entire field from them.

**Do diversity by construction, not by estimation.** Admit on quality — a floor
(within x% of the best known) or top-`N` by `cascade_score` — then enforce
diversity on axes that can be *checked* rather than *inferred*:

- **per-owner cap** (1–2 members per miner hotkey) — justified on *diversity*
  grounds alone (a pool of near-clones from one owner defeats the marginalisation
  that is B's entire point), not as an attack mitigation; per §4a there is
  little to mitigate;
- **per-generation / lineage cap** — members must trace to different warm-start
  ancestors. An init several generations back is genuinely a different model,
  and this is free;
- **recency weighting** — the §4 (ii) knob.

These are exactly enforceable, they cost nothing to compute, and they are not
gameable by manufacturing an odd bench profile. Registration cost makes a fake
second hotkey expensive; a fake second lineage is not available at all.

**If metric-space complementarity is still wanted**, the only tractable version
is one interpretable bucket split — CRPS-leaning vs MASE-leaning (calibration vs
point accuracy) — applied *under* the quality floor. One bit, checkable,
explainable. Not a PCA over six aggregates.

**Measure before deciding.** Compute the correlation matrix of the six metrics
across the checkpoints already benched. If the first component explains the great
majority of the variance — the expectation — the orthogonality idea is settled
and this note should record it, the same way DEC-CA-0006 settled the LCB question
by simulating it first.

Worth knowing for the long run: the sidecar computes **per-dataset rows** (GIFT-Eval's
configs, BOOM's datasets) and only then aggregates to the shifted geometric mean
(`benchmarks/cascade_benchmark/aggregate.py`). A profile with real dimensionality
therefore exists — it is just not published; only the six aggregates are signed
into the bench report. If a genuine diversity signal is ever wanted, publishing
per-dataset profiles in the bench report is the enabling change, and the bench
report is a separate signed document, so it is far cheaper to extend than the
receipt.

## 5. Consensus severity — a correction

`_check_warm_start` is a **hard reject gate**, not an observation. A validator
whose expected init differs from the manifest's rejects *every round* with
`warm_start_mismatch`. So:

- Any disagreement about the pool or the draw is a fleet split, not a cosmetic
  divergence. (An earlier draft of this note understated this.)
- **The pool must be a pure function of signed public data** — manifest history
  plus trainer-signed bench reports plus the round's `epoch_block_hash` — not of
  locally-accumulated state. Today's reign log is locally accumulated and lossy:
  a validator that misses a bench report within `BENCH_REPORT_RETRY_ROUNDS` (5)
  has a different log. Argmin tolerates that (a missed non-best round changes
  nothing); **a modular draw does not** — a missing member shifts the pool and
  changes the result, silently and every time.

That reconstruction is DEC-CA-0005 item 2(b), still unbuilt. It is a **hard
prerequisite** for either variant, and it is worth building regardless: today
nothing verifies that the promoted init was legitimately selected — the install
is trusted, not checked. Making the pool derivable from signed public data makes
the current argmin verifiable too.

The pool must also be **published** (trainer-side, `status/`-shaped, per the
DEC-CA-0011 pattern). Miners need to know what they are being asked to be robust
across; a private pool would be an information asymmetry favouring the operator.

## 6. Implementation sketch for B (assuming §5 lands first)

Small on the trainer side, moderate on the validator side.

**Pool derivation (new, shared — `cascade/shared/` so trainer, validator and
audit all use one implementation):**
- `warm_start_pool(history) -> tuple[PoolMember, ...]` — the last `K` promotions,
  derived from signed manifests + bench reports. Primary size only
  (`throne_sizes[0]`); the trainer ignores a warm start whose size doesn't match
  the arch preset (`trainer/loop.py:2123`), so an off-size member would silently
  disable warm-start for that round.
- `draw_init(pool, base_seed) -> PoolMember` —
  `blake2b(base_seed || "cascade-warm-start" || "|".join(ids))`, weighted by the
  recency schedule. Binding the digest to the member ids means a divergent pool
  produces a divergent, *loggable* draw rather than accidental agreement.

**Trainer (`cascade/trainer/loop.py`):** `_load_warm_start` becomes
"derive pool, draw with this round's seed". `base_seed` is already in scope at
the call site (`loop.py:1965-1973`) — this is a handful of lines.

**Validator (`cascade/validator/loop.py`):** `_check_warm_start`'s `expected`
becomes the same derivation instead of reading the pointer file. Same gate, same
reject reason.

**Audit (`cascade/audit/`):** a check that re-derives pool + draw from the
receipt's `epoch_block_hash` and asserts the manifest's `warm_start_ckpt` matches.
`seed_from_block_hash` is already verified offline (`audit/checks.py:131`), so
this composes with what exists.

**Config (`[scoring]`):** `cascade_pool_size` (default **1** = today's exact
behaviour) and the recency-weight parameter. Consensus-critical — same warning
banner as `cascade_reign_rounds`. Not in `contract_digest`, but **trainer and
validator must deploy together** (the DEC-CA-0009 precedent).

**Tests:** pool of 1 is byte-identical to today; `len(pool) < K`; tie-stable
ordering; same seed ⇒ same draw across processes; draw distribution matches the
recency weights; a member of the wrong size never reaches the trainer.

## 7. Variant A, for completeness

Draw from the reign's top `k` at promotion time; init still fixed between
promotions. Implementation is smaller (`select_winner`, `cascade.py:277`, plus
the seed plumbing), but:

- The pool is one king's own checkpoints, so it constrains that miner's precision
  and not its power — the counter is to make all `k` the artifact you wanted.
- "Each round is different" isn't what happens: promotions fire every 5 rounds,
  so it changes *which* of ≤5, not how often.
- What it does buy: it removes the few hours of lead time between the last bench
  report landing and the next round (today anyone can compute the argmin and
  start tuning), and it is some protection against the argmin of ~5 point
  estimates on a 6-number geomean with no error bars being a draw for luckiest
  bench noise.

A is not wrong, it is just small. If the pool were widened to include challenger
checkpoints — already benched, already in the signed bench report at no new
compute (`chain.testnet.toml:205-212`) — A stops being owner-controlled and gets
most of B's anti-gaming property without B's depth cost. That is worth
considering as a middle option.

## 8. Recommendation

1. **Build the signed-data pool derivation now, unconditionally**, with
   `cascade_pool_size = 1`. Zero behaviour change; makes today's promotion
   verifiable for the first time; unblocks everything else. This is the
   DEC-CA-0005 2(b) obligation.
2. **Then take B in the ancestral form** (§4 (ii)) with a recency-weighted draw,
   `K` = 2–3, and admission scored on improvement-over-init (§3). Frame it as
   *attribution* — marginalising the init out of the measurement — not as
   anti-gaming; that is the argument that survives scrutiny and it is the one
   grounded in the subnet's own stated premise.
3. **Do not ship `K` = 10 with a uniform draw.** At 1× compute that is a
   population search wearing a ratchet's clothes, and it stalls the pretraining
   accumulation the project measures itself on.

Rollout note: **testnet cannot exercise any of this as configured** —
`chain.testnet.toml:265` sets `cascade_reign_days = 1`, so there is one
checkpoint per reign and any pool is size 1. Testing needs the testnet reign
length raised first.

## 9. Open questions for the owner

1. How much depth are we willing to spend? That single number picks `K` and the
   recency weights, and everything else follows.
2. Does the pretraining-completion goal (per-size budget vs Toto2) bind? If it
   does, ancestral-with-recency is the only affordable family.
3. Are challenger checkpoints eligible (§4a)? It is the difference between a
   multi-owner pool and one miner's lineage — and it makes the reign clock
   vestigial, which is its own decision.
4. Is the middle option (§7: A, with challenger checkpoints in the pool) enough
   for the anti-gaming goal at a fraction of the cost?
5. Who runs the six-metric correlation check (§4b) — it settles the
   orthogonality question one way or the other in an afternoon.
