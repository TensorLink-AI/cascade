---
id: DEC-CA-0068
type: decision
title: "Shared-series horizon ladder: every scored horizon judges the SAME 4000 series, NaN-padded where a series is too short — decided on chain by validator readiness (DEC-CA-0045), typed block = owner override"
status: accepted
date: 2026-11-12
tags: [eval, validator, consensus, windows, activation]
revisit_when: "the first ~10 settlements after the draw applies — per-horizon CRPS/MASE spreads, how many padded windows each rung carries, and whether the pool should grow its short domains (sales 423, transport 589, healthcare 597) so the even split needs no padding"
relations: {depends_on: [DEC-CA-0032, DEC-CA-0045, DEC-CA-0049, DEC-CA-0067], relates_to: [DEC-CA-0016, DEC-CA-0038]}
---

## Decision

Owner request 2026-11-12: "I want the same number of series, i.e. 2000 or 4000,
for each horizon", "4000 series with each step", "series that fit — pad with
NaN", "roll it out quickly".

* **Draw.** `[eval] shared_ladder_windows = 4000`. From the gate, a round's
  verdict windows are ONE seeded even-by-domain pick of 4000 series over ALL
  of the snapshot's raw series (the DEC-CA-0032 scarce-domain rule: a domain
  that cannot fill an even share gives everything it has), and every rung of
  `[eval] scored_horizons` (64 / 256 / 720) cuts one window from each picked
  series. Rung sizes are equal by construction, so the pooled geomean weights
  horizons equally exactly as DEC-CA-0038's ladder intended, and the three
  rungs now measure the same series rather than three different draws.
  Window ids stay `h<horizon>-s<pool_index>`.
* **Padding.** A series with at least `horizon + 64` steps is cut exactly as
  today (the last `horizon` steps are the target, up to 4096 before them the
  history). A shorter series keeps its first 64 steps as history and its
  whole remaining tail as the target, NaN-padded to the horizon. The scorer
  masks the padded steps: the pinball components and MASE are computed over
  the observed steps only (`cascade.eval.crps.pinball_components`,
  `cascade.eval.mase.mase`), and a finite target takes the pre-existing code
  path unchanged (bit-identical). The model still forecasts the full horizon
  from the history it is given; only the targets differ. The 2026-11-12
  snapshot has 4525 series, minimum length 320, none shorter than 784 in two
  of seven domains: at 4000 every domain contributes (sales all 423, the
  rest ~590–605 each) and only the 720 rung carries padded windows.
* **Batched decode.** From the same gate the quantile head decodes the
  verdict windows in FIXED batches of `SHARED_LADDER_QUANTILE_BATCH = 32`:
  window indices in order, cut into runs of at most 32 consecutive
  univariate windows sharing one horizon (`quantile_chunks`; a multichannel
  window is its own chunk). Every history is padded to the full 4096-step
  context and each horizon is one forward pass, so a chunk is one GPU call.
  Measured on an RTX 5090 with the king checkpoint (2026-11-12, 256 windows
  per horizon): 26 ms → 2.5 ms per window (10.1–10.7×), peak 5.4 GB,
  bit-identical run to run. Batch-32 numerics differ from batch 1 (up to
  0.4–1.7 % of a series' scale on single quantiles) and depend on the
  batch-mates at ~1e-5 — the MoE has no capacity limit, so routing is
  per token and only kernel rounding moves — hence the size and composition
  are rule, hashed into the vote name (`shared-ladder/2|…|batch=32`), and
  every round before the gate keeps the batch-1 decode byte-identical. The
  offload request uses a new schema (`cascade-private-eval/3-quantiles-
  batched`): a GPU host on older code refuses it and the round holds; it is
  never silently decoded at batch 1. The validator, the audit replay and the
  trainer's lane-B self-judge all read the batch from the same gate.
* **What it replaces.** The per-rung ladder (`n_windows` or the DEC-CA-0067
  v2 budget split across rungs, each rung its own draw, rung sizes equalised
  to the scarcest) stays the rule for every round before the gate and for
  audit replay of those rounds. The count is the rule's own — `[eval]
  n_windows` and `[v2] eval_n_windows*` no longer size the ladder once it is
  in force.
* **Activation.** `shared_ladder_from_block = 0` = decided on chain. A
  validator on a release carrying the rule adds one segment to its readiness
  note — `lad-<sha256("shared-ladder/2|4000|64,256,720|batch=32")[:8]>` =
  `lad-556b1b98` — after the DEC-CA-0067 `v2b-` segment. At the first
  boundary where 51% of eligible validator stake has signalled it LOCKS IN
  (one-way, `activation_shared_ladder_state.json`) and the draw applies from
  the first ERA START after the lock-in boundary (`FeatureSpec.align =
  "era"`, the DEC-CA-0049 rule): every settlement of one era draws one way.
  The trainer resolves the same tally (never signals) so its lane-B
  self-judge draws what the fleet scores. A typed `shared_ladder_from_block`
  (an era start at or after `scored_from_block`) is the owner override.
  Changing the count or the horizons renames the feature (a fresh vote).
* **Receipts and audit.** The validator stamps `shared_ladder_block` on every
  receipt it judges under a chain-decided block (drop-when-default, so older
  signatures stay valid); `cascade-audit` replays the receipt's windows under
  it (`apply_receipt_activation`, `check_shared_ladder`), the
  `check_margin_v2` shape.

## Why

Three independent per-horizon draws of ~666 series each (today's 1998-window
verdict) measure three different subsets of the pool. The owner wants one
population judged at every horizon, and more of it: 4000 series at every
rung is 12 000 scored windows per model against 1998 today. Private scoring
runs on the offloaded GPU host (`--private-eval-hosts`): 1998 windows took
~3.5–8 min per model (Nov 2026), so 12 000 windows (6×) is ~20–47 min per
model — close to the default `--private-eval-timeout` of 3600 s per model.
Measure one 12 000-window evaluation on the private host before the vote can
lock, and raise the timeout on any validator that needs it (operator-local,
not consensus). The quantile-head verdict (DEC-CA-0067, in force from era
2637) ships quantiles, not 100 sample paths: ~0.3 GB per model at 12 000
windows, below today's sample-path transfer.

NaN padding is what makes "the same 4000 at every horizon" possible: only
3455 pool series are long enough for a 720-step target, and refusing the
rest would skew the even-by-domain split (sales would fall to 258). Padding
keeps every series in every rung and judges each on the steps it actually
reached. The alternative — growing the pool's short domains first — is the
`revisit_when` item, not a blocker.

## Blast radius

* CONSENSUS: the draw is part of the verdict. A validator that is not on the
  release when the block arrives scores different windows and forks its
  verdict from that block on — exactly why this is an on-chain vote and not
  a typed block: lock-in needs 51% of stake on the release first, and the
  rollover is a full era boundary later.
* The note budget: the on-chain readiness note is one 128-byte field. With
  the mainnet segments (`rolling-era-king`, `prelaunch-95-r2`, `v2b-69630f92`)
  and this one all LOCKED at 7-digit blocks the note is 126 bytes — it fits
  for any lock-in before block 10 000 000 (≈ January 2027). The segment is
  the newest, so it is the first trimmed if the note ever overflows (the
  validator WARNS at startup and on every write). Typing the already-decided
  `[v2] prelaunch_burn_from_block = 9486900` into the release chain.toml frees
  33 bytes if headroom is ever needed.
* Fork guard: the rule exists only where a node's chain.toml carries
  `shared_ladder_windows`, so — like `[v2] image_decoupled_feature` — the
  config pins `[eval] shared_ladder_feature = "lad-556b1b98"`. Wherever the
  ladder can apply (`[activation]` on, `scored_horizons` set) a missing key or
  a different computed name REFUSES the validator, trainer and audit at start
  (other processes log an ERROR). Without it, a node on the release with an
  older deployed chain.toml would neither signal nor apply the lock-in and
  would keep the per-rung draw after the switch, silently. Deploying the
  release therefore means adding the three `[eval] shared_ladder_*` keys to
  every deployed chain.toml (the main validator's and the trainer's included).
* Scorer: NaN-masked branches run only on a padded target; every finite
  target takes the unchanged code path, pinned by test.
* Not touched: the training contract, seeds, `contract_digest`, the bench
  gate, the king-check rule, the bootstrap, the margin schedule. Nothing
  retrains; the running seed epoch continues.
