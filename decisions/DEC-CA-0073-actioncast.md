---
id: DEC-CA-0073
type: decision
title: "ActionCast (`act:<scenario>`): a lane-C competition for decision software that calls our per-vertical fine-tunes (plus Engy LLMs and decisionmodels.io decision models), judged forward-only against a reference harness on the same model"
status: proposed
date: 2026-11-24
tags: [lane-c, competitions, actioncast, harness, applications, v2]
revisit_when: "the owner settles Q1–Q10; or a scenario cannot supply enough forward episodes per settlement; or slots stay unpaid for weeks because decision-score noise swamps the margin; or Engy / decisionmodels.io cannot pin versions or return reproducible replies"
relations: {depends_on: [DEC-CA-0056, DEC-CA-0064, DEC-CA-0063], relates_to: [DEC-CA-0015, DEC-CA-0053, DEC-CA-0069]}
---

**Naming (owner, 2026-11-24):** the Cast set.

* **Models are `<Vertical>Cast`:** MarketCast, EnergyCast, InventoryCast,
  and so on, one fine-tune competition per vertical.
* **Software is always ActionCast.** There is one software competition, and
  every vertical joins it as an `act:<scenario>` target. A new vertical adds
  a `<Vertical>Cast` and an ActionCast scenario, never a new software
  competition.

**Design:** [docs/ACTIONCAST.md](../docs/ACTIONCAST.md) has the system
diagram, the lifecycle, the kit, the scenarios, the scoring, the
implementation map, the config and the rollout. This record holds what is
decided and what is still open. Nothing is built or armed.

## Context

Lane C pays for better forecasting models (MarketCast, DEC-CA-0064). Nobody
is paid to turn those forecasts into decisions, which is where they earn
their value. We fine-tune a model per vertical. Each vertical should also
get open, competitive decision software built on that model.

## Decision

Owner rulings (2026-11-24) are marked **(owner)**. The rest is proposed.

1. **ActionCast is a lane-C competition kind, `act`.** A target is
   `act:<scenario>`. It uses the lane-C plugin interface
   (`docs/LANE_C_COMPETITIONS.md`) and has its own weight in
   `[lane_c.competitions.act]`.
2. **Miners submit a harness.** That is deterministic Python decision code
   on the shared kit, plus `harness.json`. It is not trained. It runs
   sandboxed with no network.
3. **It uses our fine-tune for the vertical (owner).** Each scenario is
   paired with its vertical's fine-tune target and always calls that
   target's current slot holder, or its pinned base before the first
   holder: `act:perp` ↔ MarketCast (`mkt:market`), `act:bess` ↔ EnergyCast
   (`nrg:`), `act:inventory` ↔ InventoryCast (`inv:`). An `act:` scenario arms only after its fine-tune
   target is configured.
4. **LLMs come from Engy, decision models from decisionmodels.io (owner).**
   A harness reaches both, and the fine-tune, only through one model
   gateway on the main validator. The gateway pins versions, enforces
   budgets, caches replies and records every request and reply (and Engy's
   proofs) in a transcript. There are no provider-side web tools.
5. **Forward-only, paired judging.** Episodes are simulated on real data
   recorded after commit. Each entry runs against the owner's reference
   harness and against its own naive-forecast run, on the same episodes and
   the same model. It passes on a composite gain LCB ≥ 0.01, with no term
   confidently worse and a positive model lift. After that, the DEC-CA-0056
   Bonferroni → pending → confirm → slot rules apply unchanged.
6. **Trading is scored on risk management, not profit (owner).** `act:perp`
   forces a mandate: basket, directions and weights. It scores vol
   tracking, tail control, an FZ0 risk-forecast loss and costs. PnL is a
   diagnostic only.
7. **Creative use of forecasts is paid for (owner).** The rules provide:
   * an open forecast API under a per-episode budget;
   * a **novel-use slot** per scenario, for a qualifier whose gains are
     decorrelated from the main holder's and that uses the forecasts at
     least as much as the reference does;
   * seeded variants and hard tiers.

   Creativity is never judged directly.
8. **One primitive kit for every vertical (owner).** `cascade.harness` has
   four layers: data, forecasts, decisions, environments. It has three
   templates: Store, Position, Commit. It is versioned and additive-only. A
   new vertical needs a data adapter plus a scenario spec, not a new
   competition.
9. **Auditable without trust.** The auditor replays harnesses on CPU from
   the transcripts, checks Engy proofs, re-queries a sample of
   decision-model calls and re-derives every score.
10. **Separate product.** Winning harnesses and their transcripts are
    published. Copies are not paid: exact or behavioural duplicates lose to
    the earlier commitment.
11. **Actions are explained to the customer, factually (owner).** Every
    material action has an explanation: claims tied to transcript evidence,
    plus plain text where every sentence cites a claim. Explanations are
    checked without an LLM judge:
    * claims against the transcript;
    * numbers against the cited claims;
    * counterfactuals by replaying the harness;
    * drivers against measured sensitivity.

    A pinned Engy model acts only as a *reader* that must predict the
    harness's actions from the explanations. False or unsupported
    statements fail the entry. Explanation quality may not fall below the
    reference's. Design: ACTIONCAST.md §10.

## Consequences

- **New code:**
  * the kit;
  * the episode runner;
  * the model gateway with Engy and decisionmodels.io clients;
  * forward recorders for each vertical's data;
  * the trainer and validator halves of the plugin;
  * an audit replay;
  * the EnergyCast (`nrg:`) and InventoryCast (`inv:`) fine-tune competitions that the energy and
    inventory apps depend on.
- **Main-validator load** grows: episode runs, plus metered external calls.
  The proposal is that miners pay for external calls (Q10).
- **Shared fate with the fine-tunes.** A harness slot is re-confirmed on each
  new fine-tune holder, so the two competitions move together.
- **Payout dilution.** Lane C's share is split with MarketCast and future
  fine-tune kinds (Q7).

## Open (owner to answer)

- **Q1. Forecaster change.** Proposed in ACTIONCAST.md §5: the holder
  keeps its slot, and its next confirm runs on the new model. Confirm this,
  or choose to re-score every harness holder at each fine-tune change.
- **Q2. Model-lift gate.** Should a harness be *required* to beat its own
  naive-forecast run? Requiring it keeps the competition about the model.
  Dropping it widens the field to any good policy.
- **Q3. Score per scenario.** Use the oracle-normalised capture or regret
  ratio, or the raw cost ratio? For `act:perp`: the term weights, σ* range,
  `min_util`, and whether execution / client-flow steps are in v1 or later.
- **Q4. The energy and demand fine-tunes.** Resolved in principle: each
  vertical uses its own fine-tune (owner, 2026-11-24). Still open: the pinned
  base and suites for `nrg:` and `inv:` (Timestral-28M again?), and whether
  each app scenario arms with its fine-tune or one era after it.
- **Q5. Noise budget.** Before any `from_block`, run the reference harness
  against small perturbations of itself on recorded data. Measure how many
  episodes a 1 % gain needs to pass at α/k. This decides
  `episodes_per_verdict`. For `act:perp`, measure each score term's noise
  separately; a term too noisy to move the composite is dropped or
  down-weighted.
- **Q6. Who pays for eval.** Should the main validator pay for forward
  episodes, as MarketCast forecasting does, with the miner's CPU leg
  qualifier as the spam filter? Or should the miner also fund eval
  forecasts?
- **Q7. Weight.** Should the app competition take part of lane C's 25 %, and
  how much (0.5 in the sketch above, against MarketCast's 1.0)?

- **Q8. Creativity terms.** Agree the novel-use slot (`novel_share`,
  `novel_max_corr`). Should forecast value also count in the main composite,
  or stay a novel-use gate? And how large should the forecast budget be? Too
  small blocks new ideas; too large makes the main validator's GPU cost
  unpredictable.

- **Q9. Generalist slots.** Should there be `act:store` / `act:position`
  slots scored across every scenario of a template, including one added after
  commit? And what weight should they get against the per-scenario slots?

- **Q10. Engy and decisionmodels.io.** Which models do we pin from each? Do
  both support version pinning and deterministic replies, and does every
  Engy reply carry a proof? Who pays: metered to the miner, as proposed, or
  absorbed by us? Is there a commercial or partner agreement (rate limits,
  data terms)? Should miner-shipped decision weights stay as a second path?
- **Q11. Explanations.** Agree the factual-gate rates and the driver
  coverage. Should the explanation score only guard (not worse than the
  reference, as proposed), or also add to the composite gain? Which
  locales come after English? Is "material action" the right trigger, or
  should every step be explained?

## Rollout

See [docs/ACTIONCAST.md §15](../docs/ACTIONCAST.md#15-rollout). In short:

1. Build the kit and the gateway.
2. Run `act:perp` end to end on `mkt:market`, which already exists.
3. Build EnergyCast and `act:bess` together.
4. Run a noise study per scenario before each scenario's `from_block`.
