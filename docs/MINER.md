# Miner guide

You submit a **data generator**: deterministic code that produces synthetic
time series. The operator trains a fixed Toto2-4M forecaster on your data and
scores it against the reigning king on a private, rotating held-out set. Beat
the king by the round's margin and you take the throne. No GPU needed; you pay
for your own training leg with a [Lium](https://lium.io) API key.

The submission contract (what the code must be) is [INTERFACE.md](INTERFACE.md).
This guide is the end-to-end walkthrough.

```
write a generator → cascade verify → cascade score (optional)
  → register a hotkey → cascade submit (private) or cascade deploy + cascade fund (public)
  → watch cascade queue / cascade round → read cascade duel
```

Rounds run every 12 h (boundaries ≈ 08:30 and 20:30 UTC). Mainnet is netuid 91.
The intake is `https://submissions.cascadesub.net`.

## 1. Install

```bash
git clone https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[hippius,chain]'
```

Use this repo's pinned environment to commit on-chain (`bittensor==10.5.0`).
Other SDK versions write reveals the validators cannot decode; your commit
lands but you are silently skipped every round.

## 2. Write a generator

Start from a shipped example and edit:

```bash
cp -r scripts/example_generator my-generator      # minimal trend + seasonal + AR(1)
# richer priors: gen_changepoint, gen_chaotic, gen_garch, base_generator
```

Your repo must contain:

```
generator.py        # class Generator(DataGenerator)
config.json         # any JSON your generator reads
requirements.txt    # hash-locked deps from the allowlist (numpy, scipy, torch, …)
```

Rules that matter:

- **Deterministic.** `generate()` must be a pure function of the `seed` given
  to `__init__`. Seed every RNG from it; never use `hash()`, wall-clock, or the
  network. Two runs at one seed must produce byte-identical corpora.
- **Allowlisted imports only.** `socket`, `subprocess`, `pickle`,
  `multiprocessing` and friends are blocked (`chain.toml [static_guard]`).
- **Series shape.** Each yield is a float array of shape `(L,)` or `(C, L)`
  with `64 ≤ L ≤ 4096` and `C ≤ 32`. Values must be finite.
- **Speed is scored.** Your generator streams during training and the compute
  budget is fixed. A slow generator feeds the model less data inside the wall,
  and that is the score, not a bug. Pin BLAS to one thread, prefer shorter
  series for expensive priors (GP draws scale ~cubically in length), and use a
  relative-jitter Cholesky instead of SVD fallbacks. Details in
  [INTERFACE.md](INTERFACE.md).
- **Numerics.** Series with extreme magnitudes can make training diverge. A
  leg whose loss goes non-finite is rejected as your fault and your submission
  is spent. Keep values in a sane float32 range.

### Multivariate series

A `(C, L)` yield is one series with `C` coupled channels (C ≤ 32).

- A channel costs what it trains: `(C, L)` bills `C×L` points of the budget.
- Step count does not depend on C. Batches hold `batch_size // C` series, so a
  wide corpus gets the same number of optimizer steps as a univariate one.
- From block 9068400 (Mon 2026-09-14 ~20:30 UTC) multivariate eval windows are
  scored jointly: all channels forecast in one pass, the window counts once.
- Only coupled channels teach the variate layers anything. Stacking unrelated
  series into one array is legal and useless.
- Eval windows have at most 8 channels regardless of your C.

## 3. Verify and score locally

`cascade verify` runs every check the trainer runs (layout, import guard,
hash-locked deps, determinism):

```bash
cascade verify ./my-generator
# → OK: generator would be accepted by the trainer.   [deterministic]
```

`cascade score` trains the fixed model on your data at a cheap budget and
scores it on a pool you control. Needs the `[train]` extra and ideally a GPU.
Compare against the king on the same pool with the same init:

```bash
cascade fetch king --out ./king
cascade score ./king   --pool-dir ./my-heldout --warm-start live
cascade score ./my-gen --pool-dir ./my-heldout --warm-start live
```

The number is directional. Validators score on a private pool you never see,
so use the local score to hill-climb, not as the verdict, and rotate your pool
so you do not overfit it. `--warm-start live` trains from the init the current
round uses; without it you train from random init, which ranks generators
differently.

## 4. Register a hotkey

```bash
btcli wallet new-coldkey --wallet-name my-miner
btcli wallet new-hotkey  --wallet-name my-miner --wallet-hotkey gen1
btcli subnets register --netuid 91 --network finney --wallet-name my-miner --wallet-hotkey gen1
```

One hotkey = one submission. A hotkey is spent once its submission has been
judged in a duel (or failed through its own fault). To submit again, register
a fresh hotkey.

## 5. Submit

Two paths. Both end with a funded entry in the queue.

### Private (recommended): `cascade submit`

Your code goes straight to the operator's private vault. It is never
published unless it takes the throne.

```bash
export LIUM_API_KEY=sk-...            # env only, never an argument
cascade submit ./my-generator https://submissions.cascadesub.net \
    --wallet-name my-miner --wallet-hotkey gen1 --label my-gen-v3
# → verifies, ZIPs, stores privately, commits vault/direct@sha256:… on-chain,
#   funds the leg; it queues when the reveal lands
```

Flags: `--no-fund` (fund later), `--no-commit` (print the chain payload only),
`--reveal-now` / `--next-epoch` / `--blocks-until-reveal N` (reveal timing).

### Public: `cascade deploy` + `cascade fund`

Your code is pushed to a public Hippius Hub repo and committed on-chain, then
you fund it.

```bash
export HIPPIUS_HUB_USERNAME=... HIPPIUS_HUB_PASSWORD=...   # or HIPPIUS_HUB_TOKEN
cascade deploy ./my-generator --hub-namespace my-namespace \
    --wallet-name my-miner --wallet-hotkey gen1
export LIUM_API_KEY=sk-...
cascade fund https://submissions.cascadesub.net --ref <repo@digest> \
    --wallet-name my-miner --wallet-hotkey gen1 --label my-gen-v3
```

The Hippius project must be **public** or the trainer cannot pull it
(`generator_artifact_unreachable`). `--hub-namespace` gives each deploy a
random repo name so nobody can watch your namespace. If the Hub is down, add
`--hf-repo <ns/name>` (needs `HF_TOKEN`) to mirror to HuggingFace.

### Reveal timing

Only a commitment **revealed strictly before** the epoch boundary enters that
round. `submit` and `deploy` default to a timed reveal a few minutes before
the next boundary, so nobody can copy your fresh entry into the same round.
A reveal that lands late rolls into the following round, nothing is spent.
Confirm with `cascade reveal-status <hotkey> --watch`, and watch the deadline
with `cascade round`.

### Labels

`--label` attaches a display name (≤ 32 chars, `[A-Za-z0-9._-]`) that shows
beside your hotkey in `cascade queue`, the published roster and the heat
standings. Re-funding with a new label renames; without one keeps the old.
Labels are cosmetic: never identity, never in signed records.

### Your Lium key

- Sent once, signed by your hotkey over the key hash, a timestamp and your
  ref. Always use `https://`.
- Held at most 36 h in a sealed vault, used only to rent and tear down your
  pod, forgotten on withdraw. Submit or fund within 36 h of your reveal.
- Keep about **4 h of the round's GPU price** on the account: ~3 h training
  plus ~1 h benching your own checkpoint afterwards.

## 6. What happens in a round

1. **Seats.** At the boundary, funded entries seat in reveal order, up to the
   round's cap (8 in the shipped config), clamped to live GPU capacity. Unseated
   entries wait, unspent, keeping their place.
2. **One GPU type per round**, the most available of RTX4090, RTX3090, L40S,
   L40, A6000. King and every challenger train on the same type.
3. **Training.** Each seated entry rents a pod on its own key and trains the
   full budget (~3 h) from the round's shared init. There is no heat screen:
   every seated entry duels the king.
4. **Manifest and verdict.** The operator signs a manifest (~3.5–4 h after
   the boundary). Validators score king and challengers on the same private
   windows and set weights (~4 h).
5. **Bench.** Your pod benches your checkpoint on GIFT-Eval / BOOM / TIME
   (~1 h), then is torn down. The operator re-benches the best-reported
   challenger on its own pod; only operator numbers are signed.

An unfunded boundary runs no round; the king holds.

### Watching it

```bash
cascade queue --intake https://submissions.cascadesub.net --hotkey <you>   # live queue + last roster
cascade round                 # deadline countdown, stage, dethrone bar, revealed submissions
cascade heat --hotkey <you>   # who seated / who waits this round
cascade duel                  # the settled verdict: margin, geomeans, per-domain win rates
```

Every leg also streams a public JSONL training log (`logs/round-<id>/…`) with
per-step loss, throughput and `data_wait_frac`, plus a `host` record for the
pod. High `data_wait_frac` means training waited on your generator.

## 7. How the verdict works

- **Metric.** Geometric mean of CRPS and MASE over the private windows, on a
  horizon ladder of 64 / 256 / 720 steps.
- **Dethrone rule.** The paired-bootstrap lower confidence bound of your
  advantage over the king must clear the round's margin. The margin is 1%
  against a fresh king and decays to a 0.5% floor over 8 held rounds. With
  several challengers, a cohort-wide correction keeps the king's false-dethrone
  risk fixed. `cascade round` prints the live bar; each receipt records the
  rule it was judged under.
- **Rewards.** The king and up to 4 prior kings share weights with geometric
  decay 0.5 (≈ 52 / 26 / 13 / 6 / 3 %). Losing challengers earn nothing.
- **Warm-start lineage.** When a king holds 5 consecutive rounds, up to 3 of
  the reign's best checkpoints (king's or challengers') become the next
  generation's shared init, picked by public benchmark score and error
  diversity. Later rounds rotate through them. You are improving the strongest
  lineage, not teaching from zero; data that adds regimes the lineage is weak
  on beats data that re-teaches what it knows. `cascade round` shows the init
  in use.

## 8. Failure classes

What happens to your entry when a leg does not produce a judged checkpoint:

| class | examples | your submission |
|---|---|---|
| infrastructure | dead pod, sold-out market, rate limit, harvest transport failure | **kept**, re-queued; one bounded attempt burned (sold-out and rate limits burn nothing) |
| stall | your generator produced no series for 30 min | first time: treated as infrastructure; second time: yours |
| auth | invalid or revoked Lium key | released as `auth`; fix the key, fund again |
| generator | your code raised, a series failed the checks, the loss went non-finite | **spent** |
| tamper | pod replaced under the same name, checkpoint altered | **spent** |
| ref_mismatch / burned / funding_expired | you re-revealed a different ref / hotkey already used / entry outlived the 36 h key TTL | terminal; re-fund (a burned hotkey needs a fresh one) |

`cascade queue` shows the class beside a failed entry. A rate-limit streak
longer than 6 h turns terminal; raise the key's limits and fund again.

## 9. Study the competition

The king's code is public the moment it takes the throne (`champions/` on
the manifest bucket; private losers stay private forever):

```bash
cascade fetch king --out ./king          # the reigning king
cascade fetch <uid|hotkey|repo@digest>   # a public competitor
```

Copying is unrewarding by construction: the earliest reveal owns a ref, and a
byte-identical corpus is dropped before it trains. You win by improving on
the visible best.

Rotated-out eval windows are published with a lag to
[Tensor-Link/cascade-eval-pool](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool).
Use them to replay past verdicts locally; live rounds always score on windows
that were never published.

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| `verify` fails determinism | an unseeded RNG, `hash()`, wall-clock, set iteration order |
| `blocked_import` | banned import; see `chain.toml [static_guard]` |
| `requirement_not_hash_locked` | every `requirements.txt` line needs `--hash=sha256:…`, allowlisted packages only |
| `403 not_registered` | register the hotkey first |
| `403 not_revealed` | the ref is not a revealed commitment for this hotkey; wait for the reveal, then fund |
| `400 bad_label` | label too long or has characters outside `[A-Za-z0-9._-]` |
| `generator_artifact_unreachable` | your Hippius project is private; make it public |
| funded but never seated | more senior reveals filled the seats, or the GPU market is thin; you wait unspent |
| `failed [generator]` | your code failed on the pod or training diverged; check the training log, fix, submit from a fresh hotkey |
| `failed [rate_limited]` | your Lium key was rate-limited for 6 h; raise limits, fund again |
| loses every duel | the pool is broad real-world data; widen the prior rather than fitting one shape. `cascade duel` shows which domains you lost |
