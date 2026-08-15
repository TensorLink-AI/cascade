# Tenure-decaying dethrone margin — release-then-activate plan (DEC-CA-0016)

Status: **design approved; NOTHING deploys to mainnet without explicit owner
sign-off.** The mechanism is pure config (`margin_for_tenure`,
`cascade/eval/koth.py:79`, is already an affine schedule); this document is the
rollout protocol that makes flipping the config safe on a subnet with 6
EXTERNAL validators.

## What changes

`chain.toml [scoring]`, validator-side only:

```toml
win_margin_start     = 0.02      # unchanged — a fresh king defends the full margin
win_margin_end       = 0.00X     # NEW floor, reached at margin_warmup_rounds of tenure
margin_warmup_rounds = N         # rounds of tenure over which the margin decays
```

With `end < start` the existing warmup schedule runs in reverse: an entrenched
king must be beaten by progressively *less* as its tenure grows, floored at
`win_margin_end`. Facts the design rests on (all verified in code):

- `margin_for_tenure` is an affine ramp from `win_margin_start` (tenure 0) to
  `win_margin_end` at `margin_warmup_rounds`, clamped after. Nothing in it
  assumes `end >= start`.
- `ChampionState.tenure_rounds` advances every round the king holds
  (inconclusive included) and resets **only** on a real dethrone or a
  king-resync re-crown — NOT on a cascade warm-start promotion (live proof:
  tenure=9 at the r19 verdict while the reign clock had fired at 5). Decay
  therefore deepens across a whole hold, which is the intent: the longer the
  hold, the cheaper the escape.
- Receipts already record `king_tenure_rounds` and the full `verdict.params`,
  so `cascade-audit` replays every archived round under the rule that judged
  it. **No receipt format change.**
- The trainer never reads the margin; this is a validator-only value.

## Hard guardrail

`win_margin_end` must stay **strictly > 0**. The margin is the improvement bar
*above* the LCB noise gate (`win = LCB >= margin`); at a 0 floor any
statistically nonzero improvement dethrones, and `bootstrap_alpha` alone does
not price repeated attempts across rounds. The replay harness and the config
review both enforce this.

## Step 0 — evidence (before any schedule is picked)

Run the replay harness over the full mainnet receipt trail:

```
uv run python scripts/replay_margin_decay.py --fetch ./receipts_cache \
    --chain-toml chain.toml --verify --json margin_replay.json
```

It re-judges every scored receipt under the grid `start=0.02,
end ∈ {0.005, 0.008, 0.010}, warmup ∈ {8, 12, 16}` using only recorded fields
(pure `margin_for_tenure` + recorded tenure + recorded LCBs — the bootstrap
quantile is untouched by a margin change, so recorded LCBs are exactly the
right statistic). Output: which past verdicts flip (all flips are loss→win;
decay can never revoke a recorded win), plus near-misses within 50bp of each
candidate margin. **First-order caveat** (printed by the tool): a flip at
round r would have changed the king/tenure/field for every later round —
counts measure how binding the flat margin was, not an alternate timeline.

The owner picks the schedule from this report. Recommendation to start the
discussion: `end=0.008, warmup=12` (floor at 40% of the fresh-king margin,
reached in ~6 days of 12h rounds — one full cascade promotion cycle is 5
rounds, so a king past two promotion cycles defends the floor).

## Step 1 — testnet (netuid 259)

`chain.testnet.toml` now carries the candidate schedule (this branch). One
full cycle before mainnet:

1. Restart testnet validator + trainer with the schedule (no digest change —
   `[scoring]` is not in `contract_digest`; no lockstep needed on a
   single-operator testnet).
2. Let a king accumulate tenure past `margin_warmup_rounds`; confirm on
   receipts that `verdict.margin` walks down the ramp exactly
   (`margin_for_tenure` at the recorded tenure) and that a promotion re-crown
   does NOT reset it.
3. Run `cascade-audit` on a decayed-margin receipt — the verdict must
   recompute from the receipt alone.
4. Run the replay harness over the testnet trail as a sanity loop (recorded
   margins now vary with tenure; the consistency gate must stay quiet).

## Step 2 — mainnet release-then-activate

**Verdict-fork blast radius — read before any deploy step.** The margin is
computed independently by each validator from its own `chain.toml` at verdict
time. During any window where validators disagree on the schedule, every round
whose decided LCB lands in the disagreement band
`[margin_decayed, margin_flat)` splits the fleet: upgraded validators record a
dethrone, stale ones record a hold. Champion state then **diverges and stays
diverged** — there is no automatic reconvergence mechanism; a diverged
validator only re-joins when a subsequent round dethrones its king on both
branches (or its state is manually resynced). Divergent kings mean divergent
weight vectors (different reward sets on chain), a forked receipt trail, and a
trainer whose receipt-anchored reign clock follows its pinned anchor validator
while others disagree. This is the same class of hazard as the 2026-07-28
`epoch_blocks` change and is why this is release-then-activate, never a local
flip. The band is small (≤ `start − end` ≈ 120bp of LCB) and only exists for
tenured kings, but the replay report quantifies exactly how often historic
rounds landed in it — publish that number in the upgrade announcement so
operators know the real per-round risk of lagging.

Sequence:

1. **Ship**: the schedule lands in `chain.toml` in a tagged release
   (`vX.Y.Z`). Release notes name the activation round and the disagreement
   band, and state the guardrail (`end > 0`).
2. **Upgrade window**: all 6 external validators (plus the owner's) upgrade
   and restart within one announced round window. Config applies at process
   start, and a round's verdict is computed when its manifest lands — so the
   safe restart window for round `R` activation is **after round `R−1`'s
   verdict is recorded and before round `R`'s manifest publishes** (~the
   final-training hours of round `R−1`). Operators confirm in-channel;
   the receipt trail is the check — every validator's
   `receipts/<hotkey>/round-R.json` must record the same `margin` for the
   same tenure.
3. **Activation**: round `R`'s verdicts are the first judged under decay. No
   code path needs a flag day beyond the restart — the schedule is stateless
   config over recorded tenure.
4. **Monitor**: for the first `margin_warmup_rounds` rounds, compare
   per-validator receipts (they are namespaced per hotkey) for margin/verdict
   agreement each round. Any mismatch ⇒ the lagging operator restarts before
   the next verdict; if a split verdict already landed, treat it as an
   incident (manual state resync per VALIDATOR.md) — do not wait for organic
   reconvergence.
5. **Rollback**: restarting with the flat schedule is safe only while no
   decayed-margin verdict has flipped a round that the flat schedule would
   have held. After the first genuine decay-dethrone, rolling back forks
   state exactly like a partial upgrade — roll forward instead.

## Interactions checked

- **Cascade warm-start (DEC-CA-0013/0014)**: promotion does not reset tenure,
  so decay and the reign clock run on independent counters. No coupling: the
  reign clock keys off dethrone verdicts, which decay only makes more likely
  (a dethrone resets both).
- **Cohort duel (DEC-CA-0012)**: the alpha/k correction tightens the
  bootstrap *quantile*; the margin stays one flat bar for the whole cohort.
  Decay lowers that bar identically for every cohort member — no interaction
  with the multiplicity math. The replay harness handles cohort receipts via
  `cohort_lcbs`.
- **Gift gate**: AND-ed after the margin decision; unaffected (off on
  mainnet).
- **Audit**: replays from receipt-recorded params — archived receipts stay
  verifiable, decayed receipts verify the same way.
- **`min_windows`/`min_clusters`**: unchanged; inconclusive rounds still
  advance tenure, which is correct — a king is not "defending" on a round
  with no decision, but its hold is still lengthening.
