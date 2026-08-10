# Scope: sampled top-k warm-start promotion

Status: **proposal, not decided.** Written 2026-08-10 against `main` (Cascade
armed on mainnet 2026-08-05: `cascade_enabled = true`, `cascade_reign_rounds = 5`,
12h rounds ⇒ a promotion roughly every 2.5 days).

The idea: instead of promoting the reign's single lowest-`cascade_score`
checkpoint, take the top `k` and draw one from that pool, so the promoted
warm-start init is not knowable in advance.

This document scopes it — what it touches, what it actually buys, what it
costs, and what has to be true before it can ship. It does not implement it.

---

## 1. Where selection happens today

`cascade/validator/cascade.py:277` — one function, five lines:

```python
def select_winner(state: CascadeState) -> CheckpointRecord | None:
    if not state.checkpoints:
        return None
    return min(state.checkpoints, key=lambda r: (r.score, r.checkpoint_id))
```

The surrounding shape matters more than the function:

- **The pool is one king's own checkpoints.** `_record_king_checkpoint`
  (`cascade/validator/loop.py:586`) logs exactly one entry per round — the
  reigning champion's, at the primary throne size — and `crown()` clears the log
  on every re-crown, including the re-crown a fired Cascade performs. So at
  `cascade_reign_rounds = 5` the pool is **at most 5 candidates, all produced by
  the same miner, one round apart, from the same lineage.**
- **Missing bench reports shrink it.** The bench runs post-publish; a round whose
  signed report never lands within `BENCH_REPORT_RETRY_ROUNDS` (5) contributes
  nothing (`_drain_pending_bench`, `loop.py:622`). A real reign can be 3–5 deep,
  not always 5.
- **Scores come from signed trainer data**, not local re-eval: the round's
  bench report is the authoritative source, joined on the exact
  `trained_pointer` (`_report_bench_scores`, `loop.py:569`).
- **Only one validator's selection actually drives training.** The install
  writes `warm_start_init.json` (`_warm_start_installer`, `loop.py:1621`) and the
  co-hosted trainer reads it (`trainer/loop.py:1611`). Other validators compute a
  selection that nothing consumes. Divergence between validators is therefore not
  a fork today — it is an *audit* problem (see §5).

Downstream: the trainer pins `warm_start_ckpt` + `warm_start_size` into the
signed manifest (`trainer/loop.py:2060`), applies it only when the size matches
the arch preset (`trainer/loop.py:2123`), and `cascade-audit` re-derives the
round from that pinned checkpoint (`audit/rederive.py:220`).

## 2. What the change actually is

Three sub-decisions hide inside "sample the top k":

1. **Pool rule** — fixed `k`, or a *band* (every checkpoint within x% of the
   best score), or `k` with a band guard.
2. **Draw rule** — uniform over the pool, or weighted by score.
3. **Seed** — this is the whole ballgame. See §4.

Everything else is plumbing.

## 3. Honest read: what it buys, what it doesn't

### It does not meaningfully constrain the adversary who controls the pool

Every candidate in the pool belongs to the reigning king. If the concern is
"the king engineers which artifact becomes the field's init", randomising among
its own checkpoints reduces its *precision*, not its *power* — and the counter is
cheap: make all `k` the artifact you want promoted. The king trains a checkpoint
every round of the reign anyway; it costs nothing to have the top of the pool be
homogeneous. **Top-k sampling over a single-owner pool is weak as an anti-gaming
device against that owner.**

Nor does it do anything about benchmark overfitting: all `k` candidates are
scored on the same six public numbers (GIFT-Eval / BOOM / TIME CRPS+MASE), and a
checkpoint tuned to those numbers stays tuned to them whichever one is drawn.

### It does buy three real things

- **Kills pre-promotion lead time.** Today, once the last bench report of a ripe
  reign lands (~an hour after the manifest), *anyone* can compute the argmin and
  start tuning a generator against the init the field will use next round. With
  an unpredictable seed, nobody — king included — knows which of `k` it is until
  the promotion round's block exists. That is a genuine, if modest, edge removed:
  a few hours of head start, and it currently accrues to whoever is watching the
  bench store most attentively.
- **Multiplies the cost of pre-tuning by `k`.** To keep the head start, a miner
  must tune against `k` inits instead of one.
- **Winner's-curse robustness.** The argmin of ~5 point estimates on a
  6-number geomean, with no repeated measurement and no error bars, is partly a
  draw for "luckiest bench noise" rather than "best checkpoint". Refusing to
  over-trust the max of a small noisy sample is the same instinct as
  [[DEC-CA-0006]] — with the important difference that DEC-CA-0006 *rejected*
  turning that instinct into a selection rule. Any proposal here has to argue
  why promotion is different from the heat. (It plausibly is: the heat is a
  competitive ranking with a prize attached, promotion is a shared-baseline
  choice where being wrong costs everyone equally and nobody is being paid on
  the outcome. That argument should be made explicitly, not assumed.)

### Two framing corrections

- **"Each round is different" is not what this does.** Promotions fire every
  `cascade_reign_rounds` = 5 rounds, not every round. The init changes ~every 2.5
  days regardless of this change; sampling changes *which* of 5, not *how often*.
- **The pool's diversity is narrow.** Five checkpoints from one reign are the
  same miner's generator, one round apart. They are far more alike than
  "a different model each time" suggests. Whatever variety this buys, it is
  variety within one lineage.

### If the anti-gaming goal is the real goal, the bigger lever is the pool

The trainer already benches **both duel entries** each round, not just the king
(`cascade_bench_max_series` docs, `chain.testnet.toml:205-212`), and both land in the
signed bench report. Widening the promotion pool to include challenger
checkpoints would mean the pool is not owner-controlled by any single miner —
which is the property "harder to game" actually wants — **at no new bench
compute**. It is a bigger change (it means promoting a round-loser's checkpoint,
which needs its own incentive argument), and it should be scoped as its own
decision rather than smuggled in as a sampling tweak. Worth knowing it is cheap
and available.

## 4. The seed is the hard part

A promoted init must be derivable identically by every validator and by an
offline auditor. So the "randomness" has to be verifiable randomness, not
`random.choice`.

**Do not** seed from anything the pool contributes (checkpoint digests, scores,
ids). Miners control checkpoint bytes and can grind them to steer a modular
index — cheaply, since they are retraining every round anyway.

**Use the promotion round's `base_seed`.** It already exists, is already
published, and is already audited:

- `base_seed = seed_from_block_hash(epoch_block_hash)` (`shared/chain.py:130`)
- it is a field of the signed round receipt (`shared/receipt.py:257`)
- `cascade-audit` already verifies the derivation offline
  (`audit/checks.py:131`) and checks the recorded hash against chain
  (`checks.py:193`)

The promotion round's epoch block does not exist when the reign's checkpoints
were trained, so no miner can predict or grind the draw. Auditors recompute it
with no chain connection. This is the right primitive and it is already built.

**Plumbing gap:** `_cascade_round` (`loop.py:649`) currently has only
`_epoch_start_block(manifest)`; the hash/`base_seed` is fetched separately in the
receipt path (`loop.py:916-921`, best-effort, may be `""`). The seed must be
threaded into the cascade step. And "hash unavailable" must **hold the
promotion for a round**, never fall back to argmin — a silent fallback is exactly
the divergence class this design is supposed to remove. Holding is already a
supported state (a ripe clock with no checkpoint holds, `cascade.py:483`).

## 5. The cost nobody will notice until it bites: reign-log divergence

This is the finding that should gate the whole change.

Today, a validator that missed one round's bench report *still promotes the same
checkpoint* as everyone else, as long as the missed round wasn't the best one.
Argmin is robust to a lossy log.

Under top-k sampling with a modular draw, **missing any member of the top-k
shifts the pool and changes the drawn index.** Two validators with logs differing
by one absent bench report promote different inits. The failure is silent and
100% of the time, not occasionally.

Because only the trainer-cohosted validator's pointer file drives training, this
does not fork weights today — but it does break the property that a third party
can verify the promoted init was legitimately selected. And it is exactly the
class of bug that DEC-CA-0005 sequenced work around.

**Prerequisite:** reconstruct the reign log from signed public data (manifest
history + bench reports) at promotion time, rather than from locally-accumulated
state. That is DEC-CA-0005 item 2(b), still unbuilt. With it, the pool is a pure
function of signed artifacts and the divergence class disappears.

Note this prerequisite is worth building on its own merits: **today nothing
verifies that the promoted init was actually the reign's best.** The install is
trusted, not checked. Making the pool derivable from public signed data makes
argmin verifiable too — so the audit work pays for itself whether or not
sampling ships.

## 6. Implementation sketch (assuming §5 lands first)

Roughly 200–300 lines across four files, plus tests.

**`cascade/validator/cascade.py`**
- `select_pool(state, k, band=None) -> tuple[CheckpointRecord, ...]` — sort by
  `(score, checkpoint_id)` (the existing tie-break, already consensus-safe),
  truncate to `min(k, len)`, optionally drop anything worse than
  `best_score * (1 + band)`.
- `draw_winner(pool, seed) -> CheckpointRecord` —
  `blake2b(seed || "cascade-promote" || "|".join(ids))` → index. Binding the
  digest to the pool ids means a divergent pool produces a divergent, *loggable*
  draw rather than an accidental agreement.
- `select_winner(state, *, k=1, seed=None)` keeps today's exact behaviour at
  `k=1` / `seed=None`. No behaviour change until config says so.
- `CascadeEvent` gains `pool: tuple[CheckpointRecord, ...]` and `seed: int`.
- `cascade_check(*, block, seed, now)` — signature change; one call site.

**`cascade/validator/loop.py`**
- Thread the promotion round's `base_seed` into `_cascade_round`; hold the
  promotion when it is unavailable.
- `_warm_start_installer` writes `pool` (ids + scores), `seed`, and
  `selected_index` alongside the winner, so an operator can see the draw.

**`cascade/shared/config.py` + `chain.toml`**
- `[scoring] cascade_promote_top_k` (default **1** = today), optional
  `cascade_promote_band`. Consensus-relevant — same warning banner as
  `cascade_reign_rounds`. `[scoring]` is not in `contract_digest`, so no digest
  churn, but the validator must restart; the trainer does not read these keys.

**`cascade/audit/`**
- New check: given the reign's public bench reports and the promotion round's
  `epoch_block_hash`, re-derive the pool and the draw and assert the manifest's
  `warm_start_ckpt` matches. This is the deliverable that makes the change
  defensible rather than merely random.

**Edge cases that need explicit tests**
- `len(pool) < k` (common: a 3-deep reign at k=3 ⇒ sampling is a no-op over the
  whole reign).
- Ties in score — must not perturb pool order across validators.
- Mixed `size` in the pool: `_current_king_entry` falls back to a non-primary
  size when the king didn't train the primary one, and the trainer *ignores* a
  warm-start whose size doesn't match the arch preset (`trainer/loop.py:2123`).
  Drawing a non-primary-size checkpoint silently disables warm-start for that
  cycle. Either filter the pool to the primary size, or accept and log it.
- `k=1` byte-identical to current behaviour (regression guard).

## 7. Rollout is awkward and needs a config change first

**Testnet cannot exercise this as configured.** `chain.testnet.toml:265` sets
`cascade_reign_days = 1` — one round per reign — so the pool is always size 1 and
top-k is inert. Testing requires raising testnet's reign length to ≥3 first, and
then waiting a few reigns.

Suggested order:

1. Ship the signed-data reign-log reconstruction (§5) and the audit check, with
   `cascade_promote_top_k = 1`. Verifiable argmin, zero behaviour change.
2. **Measure before choosing `k`.** Replay the mainnet reign logs
   (`cascade_state.json` + the published bench reports) and compute, per reign,
   the score spread between rank 1 and ranks 2/3. If the top three are within
   ~1% the draw costs nothing and the argument is easy; if rank 1 is clearly
   separated, sampling is paying real init quality for ~1.6 bits of
   unpredictability, and the band variant is the better shape. This mirrors how
   DEC-CA-0006 was settled — simulate first, then decide.
3. Raise testnet `cascade_reign_rounds` to ≥3, arm `cascade_promote_top_k` there,
   observe a full promotion cycle.
4. Mainnet only after a clean testnet cycle.

## 8. Recommendation

Split the idea in two:

- **Build now, unconditionally:** derive the promotion pool from signed public
  data and add the audit check. It closes a real hole (the promoted init is
  currently unverifiable), it is a stated DEC-CA-0005 obligation, and it is the
  hard prerequisite for anything else here.
- **Then decide sampling on evidence.** Frame it as *robustness against the
  argmin of a small noisy sample* — that argument survives scrutiny. The
  anti-gaming framing does not survive contact with the fact that the pool is
  entirely owner-controlled; if that goal is the real driver, scope the wider
  pool (challenger checkpoints, already benched, already signed) instead, as its
  own decision.

If sampling does ship, the specifics that matter: seed from the promotion
round's `base_seed`, bind the draw digest to the pool ids, hold rather than
fall back when the seed is unavailable, default `k = 1`, and treat
`cascade_promote_top_k` as consensus-critical config.

## 9. Open questions for the owner

1. Which goal is primary — unpredictability, or robustness to bench noise? They
   point at different designs (fixed `k` vs. a statistical band).
2. Is a wider pool (challenger checkpoints too) on the table? It is the version
   of this idea with actual anti-gaming teeth.
3. Is `k` worth spending init quality on at all, before the step-2 measurement
   says how much quality that is?
4. Are we willing to raise testnet's reign length purely to exercise this?
