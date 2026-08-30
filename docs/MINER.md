# Miner guide — submit a data generator

You compete by submitting a **data generator**: purely-algorithmic code that
produces synthetic time series. The owner's trainer trains a fixed Toto2-4M
forecaster on your data from the round's **shared init** — random at
generation 0, the promoted cascade checkpoint once promotions fire (the live
case today; see [Warm-started rounds](#the-cascade--warm-started-rounds)).
You win rounds when your data trains a
better forecaster than the reigning king's, scored on a private, rotating
held-out set you never see. No GPU, no shipped weights — you compete the
*prior*. The submission contract (what the code must be) is
[`INTERFACE.md`](INTERFACE.md); this is the end-to-end operator walkthrough.

At a glance:

```
fork a generator → cascade verify → make a wallet → register on the subnet
   → set Hippius creds → cascade deploy → confirm it competes in a round
```

## 0. Install

Miners need no GPU. Install the core package with the Hippius (registry push)
and chain (on-chain commit) extras:

```bash
git clone https://github.com/TensorLink-AI/cascade && cd cascade
pip install -e '.[hippius,chain]'      # numpy/scipy + hippius-hub + boto3 + bittensor
```

## 1. Write (or fork) your generator

Start from a reference and edit — the shipped examples are real, deployable
generators:

```bash
cp -r scripts/example_generator my-generator      # minimal trend+seasonal+AR(1)
# or one of the richer priors: gen_changepoint, gen_chaotic, gen_garch, base_generator
```

Your repo directory must contain:

```
generator.py        # exposes `class Generator(DataGenerator)`
config.json         # any JSON your generator reads (band lengths, weights, …)
requirements.txt    # hash-locked deps from the allowlist (numpy, scipy, torch, …)
```

The one hard rule that trips people up: **determinism**. `generate()` must be a
pure function of the `seed` passed to `__init__` — two runs at the same seed
produce byte-identical corpora. Seed every RNG (numpy, torch, `random`) from it,
avoid `hash()`/wall-clock/network. See `INTERFACE.md` for the full contract and
the dependency allowlist (`chain.toml [dependencies]`).

## 1a. Compute-heavy priors (GP / kernel families): making them affordable

`gpytorch`, `scikit-learn`, and `networkx` are on the dependency allowlist
precisely so GP/kernel/graph priors can compete — but a naive implementation
prices itself out. Three facts to design around (DEC-CA-0031):

* **The generation budget is CPU-seconds, and threads sum into it.** The
  sandbox enforces `max_generate_seconds` as `RLIMIT_CPU`, which accumulates
  across every thread; `multiprocessing` is a blocked import. A multi-core
  BLAS therefore burns the budget *faster* instead of buying you more — pin
  your linear algebra to one thread (`OMP_NUM_THREADS=1` is set for you on
  lane-pinned pods; do not fight it) and spend the cores you don't have on a
  cheaper factorisation instead.

* **The corpus budget is denominated in points, not series.** With
  `[generator] corpus_target_points` armed (it ships armed at 67,108,864 =
  16384 × 4096), the materialised drain stops when your corpus reaches the
  target points — the series *count* is free. GP draws scale roughly
  cubically in series length, so this is the lever that matters: at L=1024
  instead of 4096 a kernel-synthesis prior costs ~64× less per series, and
  the same total corpus lands near the time budget instead of two orders of
  magnitude over it. Emit as many shorter series as your prior can afford;
  the length band `[min_length, max_length]` is the only shape constraint.
  (In the live `stream_cpu` feed the same freedom has always existed — the
  stream stops at the training token budget, not at a series count.)

* **Factorise with relative jitter, not SVD fallbacks.** The standard
  "Cholesky, and on failure fall back to SVD" pattern is the single biggest
  cost in a GP draw: the fallback is ~an order of magnitude slower and fires
  constantly on near-singular kernels (long lengthscales, periodic kernels at
  short lengths). Replace it with a *relative*-jitter Cholesky — scale the
  diagonal nudge to the kernel's own magnitude and escalate until it
  factorises:

  ```python
  def stable_cholesky(K, rng_dtype=np.float64, max_tries=6):
      # jitter proportional to the mean diagonal — an absolute epsilon is
      # either uselessly small or distribution-warping, depending on scale
      base = np.mean(np.diag(K))
      for i in range(max_tries):
          try:
              return np.linalg.cholesky(K + (1e-9 * 10**i) * base * np.eye(len(K)))
          except np.linalg.LinAlgError:
              continue
      raise np.linalg.LinAlgError("kernel not factorisable at max jitter")
  ```

  Measured on our GP priors this alone was ~5× per draw and took SVD
  fallbacks from 47 per corpus to 0. It is pure CPU-side algorithm choice —
  no contract implication, and the jitter is deterministic, so your digest
  stays reproducible.

One honest caveat on the live economics: in the deployed `stream_cpu` feed
your generator streams *during* training, and throughput is a compute
multiplier (DEC-CA-0001 — "the wall is the law"). A prior that generates
slowly feeds the trainer less data inside the round's wall regardless of the
drain budgets above, so the per-series cost you save with shorter lengths and
a better factorisation converts directly into more training data for your own
run. Budget accordingly: measure `seconds/series × (points needed / points
per series)` against the round budget before you deploy, with
`cascade score` as the ground truth.

## 2. Verify locally

`cascade verify` runs **every check the trainer runs** — layout, the static
import guard, hash-locked deps, and the determinism check (it builds your corpus
twice and compares digests). Fix anything it flags *before* you spend a
registration:

```bash
cascade verify ./my-generator --chain-toml chain.testnet.toml
# → OK: generator would be accepted by the trainer.
#   corpus_digest (seed=0): 3ff20660d2fd1c55…  [deterministic]
```

A green `[deterministic]` line means the trainer will accept it.

## 2b. Score it locally (the fast iteration loop)

`verify` proves your generator is *valid*; `cascade score` tells you if it's
*good* — without deploying, spending TAO, or waiting out a ~12h round. It
trains the fixed model on your data at the cheap **heat** budget and scores it
on a pool you control, entirely offline (needs the `[train]` extra + ideally a
GPU):

```bash
cascade score ./my-generator --pool-dir ./my-heldout --device cuda
# → score: geomean=0.412  (lower is better)
#     pool:    dir:./my-heldout  (256 windows)
#     corpus:  1024 series, digest c29ae1caa6b3…
#     trained: 92s
```

The tight loop for a human or an agent:

```bash
cascade fetch king --out ./king                          # pull the current best
cascade score ./king   --pool-dir ./my-heldout           # baseline to beat
cascade score ./my-gen --pool-dir ./my-heldout           # your candidate
# keep editing my-gen until it beats the king's number, THEN deploy
```

Two caveats worth internalising:
- **Directional, not the verdict.** You score on *your* pool; the validator
  scores on its private, rotating pool. Use the local number to hill-climb, not
  as truth — and use real held-out data (`--pool-dir`), since the default
  offline synthetic sample is only a smoke signal.
- **Don't overfit your pool.** A generator tuned to ace one fixed local set is
  exactly what the private rotating eval punishes. Rotate/expand your pool.
- **Local scoring trains from random init.** Live rounds train from the
  promoted warm-start init once a cascade generation is live (the case today),
  so your absolute local numbers won't match live heat scores — the *relative*
  comparison against `cascade score ./king` on the same pool is what carries.

## 3. Make a wallet and register

You need a bittensor wallet (a coldkey + a hotkey) and a UID on the subnet.

```bash
# create keys (skip if you already have a wallet)
btcli wallet new-coldkey --wallet-name my-miner
btcli wallet new-hotkey  --wallet-name my-miner --wallet-hotkey gen1

# register on the subnet (burns a small amount of test/real TAO for the slot)
btcli subnets register --netuid 259 --network test \
  --wallet-name my-miner --wallet-hotkey gen1
# mainnet: --netuid 91 --network finney
```

`btcli subnet list --network test` shows the current registration cost. One
hotkey = one UID = one competing generator; register more hotkeys to run several
priors in parallel.

## 4. Set your Hippius credentials

`cascade deploy` pushes your generator to the Hippius Hub registry, which needs
registry auth in the environment (never in `chain.toml`):

```bash
export HIPPIUS_HUB_USERNAME=...     # or a token: HIPPIUS_HUB_TOKEN=...
export HIPPIUS_HUB_PASSWORD=...
```

You do **not** need S3 credentials — those are the trainer/validator's.

Optionally, set a HuggingFace token if you want the outage fallback in
[§5b](#5b-if-the-hippius-hub-is-down) — it is **only** used when the Hub is down:

```bash
export HF_TOKEN=hf_...               # only needed for `--hf-repo`
```

## 5. Deploy

`cascade deploy` re-verifies locally, pushes the generator to the registry
(content-addressed by `repo@digest`), and commits the on-chain pointer
`metro-v1:gen:hippius:<repo>@<digest>`:

```bash
cascade deploy ./my-generator \
  --chain-toml chain.testnet.toml --network test \
  --wallet-name my-miner --wallet-hotkey gen1 \
  --hub-repo my-namespace/my-generator
# → pushed to Hippius Hub: my-namespace/my-generator@sha256:…
#   timed reveal: payload decrypts ~block 48575 (4820 blocks from now, 25 blocks
#   before the epoch boundary at 48600) — hidden until the field locks.
#   committed: metro-v1:gen:hippius:my-namespace/my-generator@sha256:…
```

Re-deploy any time to submit a new version — the latest pre-cutoff reveal per
hotkey is the one that competes.

> **⚠️ Your Hippius project must be PUBLIC.** The trainer pulls your generator
> anonymously; a private Harbor project returns `401 Unauthorized` and your
> submission is rejected every round as `generator_artifact_unreachable`
> (observed live for several miners). New Hippius Harbor projects can default
> to private — after your first push, open the Hippius Hub UI and set the
> project's visibility to public. Self-check (should print `200`):
>
> ```bash
> REPO=my-namespace/my-generator DIGEST=sha256:...   # from `cascade deploy` output
> TOK=$(curl -s "https://registry.hippius.com/service/token?service=harbor-registry&scope=repository:${REPO}:pull" | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')
> curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $TOK" \
>   "https://registry.hippius.com/v2/${REPO}/manifests/${DIGEST}"
> ```

> **⚠️ SDK version matters — use this repo's environment to commit.** The
> on-chain pointer travels through bittensor's timelock commit-reveal, and the
> reveal ENCODING differs across SDK lines: older `set_reveal_commitment`
> variants (and `subtensor.commit()` / `publish_metadata` / raw btcli
> commitments) write reveals the subnet's decoder cannot read — your commit
> lands, but you are silently **skipped every round** with
> `revealed-commitment decode failed: non-hexadecimal number found in
> fromhex()`. This repo pins `bittensor==10.5.0` (what the validators run);
> `cascade deploy` on this environment is the known-good path. Pass the plain
> pointer string — do NOT pre-hex `data=` yourself. To self-check after a
> deploy: `sub.get_revealed_commitment(netuid, <your uid>)` must return
> `(block, "metro-v1:gen:hippius:…")` as a clean string. If you were affected,
> simply re-deploy from this environment — the newest commit wins.

### 5a. Protecting your submission (timed reveal)

**Threat model.** Everything on cascade is public *after* a round locks — that's
what makes every round independently re-derivable, and studying (or forking) the
reigning king is the intended game. What should **not** be possible is a
competitor copying your *fresh* submission into the **same round**, free-riding
on work that hasn't even been evaluated yet. Two things could leak it early:

1. **The on-chain pointer.** Deploy defaults to a **timed reveal**: the
   timelock-encrypted commit decrypts `[round] reveal_margin_blocks` (~5 min)
   before the epoch boundary. Your pointer is hidden for its whole submission
   window; by the time it's readable, a copier can no longer land their own
   reveal before the cutoff (eligibility requires the *reveal* — not the commit —
   to be strictly before the boundary). Flags:
   - `--reveal-now` — reveal immediately (the old behaviour). Your ref is public
     and copyable for the rest of the window; committing someone's exact ref
     needs no upload at all.
   - `--next-epoch` — target the *following* boundary when you'd rather sit out
     the imminent round than deploy inside the margin.
   - `--blocks-until-reveal N` — full manual control.
   Don't try to out-tune the margin by hand: reveal timing jitters by a few
   blocks, and a reveal landing at/after the boundary misses the round entirely.
2. **The generator content itself.** The upload to your Hub repo happens at
   deploy time, *before* the reveal — and a predictable repo name (or a public
   HuggingFace mirror, which lists your whole account) lets a competitor watch
   your namespace and copy the content without ever reading the chain. Use
   `--hub-namespace my-namespace` instead of `--hub-repo`: each deploy then goes
   to a fresh, non-guessable `my-namespace/gen-<random>` repo, so the content is
   only discoverable through the (still-hidden) on-chain ref. Avoid the
   `--hf-repo` fallback for competitive submissions; it exists for Hub outages.

Copying is also unrewarding by construction, at both levels the trainer can see:

- **Same ref** (someone commits your exact `repo@digest` string — needs no
  upload): the **earliest reveal** keeps the slot; the copy is dropped before
  any training.
- **Same content** (someone re-uploads your generator bytes under their own
  repo — a different ref): under the round's shared seed an identical generator
  produces an identical **corpus digest**, and the trainer drops the clone —
  before screening in the heat (`duplicate` in the standings), and from the
  final's entries — again keeping the earliest reveal. A clone can therefore
  never tie you and steal your slot on a tiebreak; a challenger whose corpus is
  byte-identical to the king's is discarded outright.

What remains possible inside the ~5-minute margin is a *modified* fork — which
is just the ordinary (allowed) forking game played with almost no time to
actually improve anything.

#### After you deploy: confirm the reveal, and what a miss means

Reveal timing jitters by a few blocks, so don't assume — confirm. `deploy`
prints the exact command:

```bash
cascade reveal-status <your-hotkey-ss58> --network test \
    --expect-boundary <printed-boundary> --watch
# → revealed at block 48577 — eligible for the round locking at block 48600 …
# or, loudly:
# → ⚠ MISSED the targeted boundary 48600: the reveal landed 3 block(s) at/after it.
```

If the reveal **misses** its boundary:

- The submission **auto-rolls into the next round** — no re-commit needed
  (eligibility is just "revealed before that round's boundary").
- It has **not** consumed your one-submission budget: the burn is persisted
  only after a round's heat stage actually screened the field, and a missed
  reveal never entered one.
- The cost is secrecy, not eligibility: the ref is public until the next
  boundary. The same-ref and same-content rules above still keep your slot
  yours; if you'd rather enter *fresh, improved* content hidden, just
  re-deploy — the **latest reveal per hotkey** is the one that competes.

**Fat-fingered a deploy?** Same mechanism: re-deploy the corrected generator
before the boundary. Both commitments will eventually reveal; the later reveal
wins. Two timed deploys in one window may target the same reveal block — if you
need the replacement to be unambiguous, give it `--next-epoch` (or a later
explicit `--blocks-until-reveal`) so its reveal strictly follows the original's.

Two empirical assumptions behind this scheme can (and should) be re-checked
against the live network:

```bash
# is the margin big enough? (needs a throwaway testnet hotkey; ~minutes)
python scripts/measure_reveal_jitter.py --chain-toml chain.testnet.toml \
    --network test --wallet-name probe --wallet-hotkey probe1

# is content behind a random repo name actually undiscoverable? (~seconds)
python scripts/probe_hub_enumeration.py
```

If the registry turns out to be enumerable, random repo names don't hide
content and the fallback is *sealed submissions* (upload ciphertext; the
decryption key + plaintext digest ride in the timelocked payload).

### 5b. If the Hippius Hub is down

Miner submission uploads to the Hippius **Hub** (the OCI registry) — a different
service from Hippius **S3** (which only the trainer/validator use). If the Hub is
having an outage, the upload fails with `registry upload failed: …` (exit 4). Pass
`--hf-repo` to mirror your generator to HuggingFace instead so you can still
submit:

```bash
cascade deploy ./my-generator \
  --chain-toml chain.testnet.toml --network test \
  --wallet-name my-miner --wallet-hotkey gen1 \
  --hub-repo my-namespace/my-generator \
  --hf-repo  my-hf-namespace/my-generator      # fallback, needs HF_TOKEN
# → Hippius Hub upload failed (…); falling back to HuggingFace mirror …
#   mirrored to HuggingFace: my-hf-namespace/my-generator@hf:<sha>
#   committed: metro-v1:gen:hippius:my-hf-namespace/my-generator@hf:<sha>
```

How it works, and what to know:

- **Hippius is priority one.** The Hub is *always* tried first; HF engages **only**
  if that push fails. `--hub-repo` is required — you cannot submit straight to HF
  while the Hub is healthy. (`--hf-repo` alone is refused.)
- **It's a real submission.** The chain commit records `repo@hf:<sha>`, and the
  trainer/validators/auditors fetch, train, and score it exactly like a Hub one —
  the `hf:` ref just tells them to fetch from HuggingFace.
- **Keep the HF repo public and don't delete it** while that commit is your active
  submission — the trainer fetches it anonymously, so a private/deleted repo means
  it can't be evaluated. (A newly-created repo is public by default.)
- **The commit stays on HF until you replace it.** When the Hub recovers it does
  *not* auto-migrate — re-deploy with just `--hub-repo` to move your submission back
  onto the content-addressed Hub (the preferred, audit-anchored form).

### 5c. Time your submission — `cascade round`

Only commits revealed **strictly before** the epoch boundary enter the next
round; commit at or after it and you wait a whole extra round (~12h). `cascade
round` is a live round dashboard: the countdown to that deadline, where the
round roughly is, and the revealed submissions — run it before you deploy so
you don't commit into the wrong round, and keep it running to see your own
commit land:

```bash
cascade round --network test --chain-toml chain.testnet.toml
# cascade round — network: test
#   current block   4,321,004
#   round (epoch)   600  ·  started at block 4,320,000
#   next round      epoch 601 at block 4,327,200
#   progress        [████░░░░░░░░░░░░░░░░░░░░░░░░]  13.9%  (1,004 / 7,200 blocks)
#   countdown       20h 39m 12s until next round  (~12.0s/block)
#   deadline        commit strictly before block 4,327,200 to enter epoch 601
#   eta             2026-07-12 03:51 UTC (estimated)
#   stage           heat ▸ [DUEL] ▸ validation ▸ settled
#                   king vs finalists training at the full budget — 3h 20m 48s into the round (est.)
#   last round      king held (uid 3)
#   dethrone bar    LCB > 0.875% this round  (king tenure 6; floor 0.50% at tenure 8)
#   submissions     4 in this round · 1 committed for the next
#     uid   47  5F3sab…8kQz  my-ns/my-generator@ab12cd34…      block 4,320,100  → next round   ● new
#     uid   12  5DkPcd…1mVx  other/gen@77aabb01…               block 4,319,882  in this round
#     …
```

It ticks every second, re-syncing to the real block height every `--refresh`
seconds (default 30); Ctrl+C exits. `--once` prints a single snapshot instead
(piped output does this automatically, with no escape codes), which is handy
in scripts. The block numbers are on-chain-exact; the wall-clock countdown and
ETA are estimates from the configured cadence (`[round] round_hours` over
`epoch_blocks`, ~12s/block). Read-only — no wallet needed. Don't cut it to the
last block: leave margin for the upload plus commit inclusion.

What the live sections mean:

- **stage** — where the current round is: `heat` (every challenger trained
  cheaply and screened), `duel` (king vs the surviving finalists at the full
  budget), `validation` (validators scoring the duel and setting weights),
  `settled` (this round's receipt is public — the line shows the verdict:
  king held, dethroned, or rejected). The trainer's internal progress isn't
  public, so the pre-settle stages are wall-clock **estimates** from the
  configured budgets (marked `est.`); `settled` is confirmed from the public
  receipt index and needs no credentials. `last round` shows the previous
  round's verdict while the current one is still in flight.
- **dethrone bar** — the LCB margin a challenger must clear to take the throne
  **this round**. The margin decays with the king's tenure (an affine ramp from
  `[scoring] win_margin_start` to `win_margin_end` over `margin_warmup_rounds`
  held rounds, then floored), so the number in the last settled receipt is
  already one step stale — this line derives the live bar from the public
  receipt index (consecutive holds by the current king) and the configured
  schedule. Shown only while the round is in flight; once it settles, the
  receipt states the margin it was actually judged at.
- **heat** — this round's screening standings, shown from the moment the heat
  settles (the trainer publishes them then, not with the round's receipt): every
  entrant's rank, its gap to the best entrant, its raw CRPS/MASE, and whether it
  advanced. Pass `--hotkey <your-ss58|uid>` and your row is marked `← you` and
  always shown, however far down you placed. Full view:
  [`cascade heat`](#your-heat-result--cascade-heat).
- **submissions** — the revealed on-chain commitments, newest first: who is
  competing in the current round vs committed for the next one (relative to
  the epoch boundary). In watch mode the field re-polls about once a minute,
  and a commit that appears while you watch is flagged `● new` — after
  `cascade deploy`, that flag on your UID is the confirmation your submission
  is on chain and which round it will enter.

## 6. Confirm it's competing

Your commit is now on chain. The quickest check is `cascade round` — your UID
appears in its **submissions** feed (flagged `● new` if you were already
watching), tagged with the round it enters ([§5c](#5c-time-your-submission--cascade-round)).
To verify from first principles instead:

```bash
# it shows in the revealed commitments for the netuid …
# (the trainer reads these before each epoch boundary)
python - <<'PY'
from cascade.shared.chain import ChainClient
from cascade.shared.config import load_chain_config
cfg = load_chain_config('chain.testnet.toml')
c = ChainClient(netuid=cfg.netuid, network="test")
for cm in c.poll_commitments():
    print(cm.uid, cm.hotkey[:10], cm.payload[:60])
PY
```

Then watch the **public round receipts** (or the dashboard): revealed *before*
the epoch boundary, your generator enters the next round's **heat**, gets
trained and scored, and appears in that round's receipt participant set with
your `gen_ref`. If it wins the heat it advances to the full final against the
king — and when the screen cannot statistically separate the top entrants, the
tied cohort advances together (capped at `[round] max_finalists`, 3), the
validators judging the whole cohort and crowning the best margin-clearer.
Verify any round independently with `cascade-audit latest` (see
[`AUDIT.md`](AUDIT.md)).

### Your heat result — `cascade heat`

The heat is the only place a non-winning submission is ever scored, and you do
**not** have to wait for the round to settle to read it: the trainer publishes
the standings the moment the heat settles — before the duel trains, hours before
a validator signs the receipt, and even for a round that is later rejected at a
gate.

```bash
cascade heat --hotkey <your-hotkey-ss58> --network test --chain-toml chain.testnet.toml
# cascade heat — round 4321000  ·  epoch start block 4,320,000
#   published       2026-07-30T04:12:19+00:00
#   field           5 entrants · 1 advanced · screened at tiny-24m
#   advancing       top 1 to the duel against the king
#   decisiveness    leader LCB +0.0310 — the screen separated 1st from 2nd (n_windows=120, feeds=9)
#
#     #1  uid    2  5FcCso…yfsw  carol/gen-a@cccccccc…    best  crps   0.4123  mase   1.021  ▲ advanced
#     #2  uid   47  5F3sab…8kQz  my-ns/my-gen@dddddddd…  +4.8%  crps   0.5100  mase   1.200  screened   ← you
#      —  uid    4  5Gzzzz…qwer  frank/gen@ffffffff…         —  crps        —  mase       —  did not train
```

What you get per entrant: your **rank**, your heat score *relative to the best
entrant* (`+4.8%` = 4.8% worse than the leader), your raw **CRPS** and **MASE**
on that round's held-out eval-pool slice, and your **standing** — advanced,
screened out, `duplicate` (byte-identical corpus to an earlier reveal), did not
train, or screen error. `--round <id>` reads an archived round, `--history`
lists what has been published. Read-only: no wallet, no chain call, no
credentials.

`cascade round` shows the same standings inline as soon as this round's heat
lands (pass `--hotkey` there too and your row is marked `← you` and always
shown, however far down you placed), and the web dashboard's **Heat** panel
switches to the live standings the moment they are published — labelled *Live*
while the duel is still training. Everything here is informational and
**unsigned** (it rides the manifest as a presentational block and is mirrored to
`status/heat.json` + `heats/round-<id>.json`), so it never affects the signed
verdict.

### The duel verdict — `cascade duel`

Once a round settles, `cascade round` compresses the result into one line
("DETHRONED" / "king held"). `cascade duel` prints the full verdict behind it,
straight from the validators' public receipts:

```bash
cascade duel                       # latest settled round (--round <id> for an older one)
# cascade duel — round 13527411684103147578  ·  epoch start block 8809200
#   outcome        DETHRONED — challenger uid 124 took the throne
#   king           uid   31  5HpRNH…tQeJ  geomean 0.24836  cascade-private/gen…@05e741f7…
#   challenger     uid  124  5FdwqF…5scc  geomean 0.23881  (3.85% better than the king)  garuda-labs/gen-04f…@hf:1ebecb70…
#   margin         LCB +0.0220 vs +0.0200 required — challenger cleared the bar
#   evidence       challenger won 60.8% of windows (1945 windows, 852 feeds) · wilcoxon p=2.7e-27
#   bootstrap      Δ p50 +0.0376 / p95 +0.0576
#   heat           25 entrants · 1 advanced · leader p_best 0.977
#   rewards        uids [124, 31, 49, 246, 158]
#   per-domain win rate  (right of centre = challenger ahead)
#     web_cloudops   0.74  n=  440  ············██████
#     healthcare     0.46  n=  147             █············
#   validators     2 scored (lcb +0.0220/+0.0220) · 5C8W9P…LJDD rejected (contract_digest_mismatch)
```

The dethrone rule in one line: the challenger takes the throne when the
paired-bootstrap **LCB** of its advantage clears the round's **margin** —
which is not flat: it decays with the king's tenure (2% against a fresh king,
ramping to a 0.5% floor over 8 held rounds under the default `[scoring]`
schedule), so an entrenched king is progressively cheaper to challenge. Each
receipt records the margin *that round* was judged at plus the king's tenure,
and `cascade round` shows the live bar for the round in flight. The
per-domain table then shows *where* the duel was won or lost (a win rate
above 0.50 means the challenger beat the king on that domain's windows).
`--history` lists every settled round's outcome. Like `cascade heat` this is
read-only — no wallet, no chain call, no credentials — but unlike the heat
standings it reads the **signed** receipt index, one row per validator, so you
also see whether the validators agreed.

### Reading the training log — was it your generator, or the pod?

Every run streams a JSONL log to the public logs bucket at
`logs/round-<id>/<role>.jsonl` — `heat-<your-hotkey>.jsonl` for a heat entry,
`king-<size>` / `challenger-<size>` for a final. When `[wandb] enabled`, the
same records mirror live into the public wandb project while the run is in
flight. Per-step rows carry `loss`, `lr`, `tokens`, `throughput_tokens_per_s`,
`steps_per_s`, and `data_wait_frac`; the closing `summary` row carries those
totals plus `tokens_frac` and `deadline_hit`.

The wall prices generator speed on purpose — a slow generator gets less compute
and that *is* the score, not a bug (`decisions/DEC-CA-0001`). But the fleet is
rented per round, sometimes across providers, so a run can also be slow for
reasons you don't control. Each run therefore publishes a `host` record (also
merged onto the `summary` row) describing the pod it landed on:

| field | what it tells you |
|---|---|
| `host_bench_tokens_per_s` | a **fixed** calibration workload, timed before your generator starts. Identical on every pod and independent of your submission, so it is the one number directly comparable host to host. |
| `host_bench_cpu_tokens_per_s` | the CPU leg alone — per-core speed of the core this lane got. |
| `host_bench_d2d_gb_s` | the GPU's achieved memory bandwidth (catches a throttled or cut-down card). |
| `host_lane_index` / `host_lane_count` | the pod's GPU fan-out. `host_gen_cpu_slice` is the core count your generator was confined to. |
| `host_cpu_count`, `host_cpu_model` | cores this run could actually use, and the CPU it ran on. |
| `host_gpu_name`, `host_gpu_sm_count`, `host_pcie_gen`, `host_pcie_width` | the device, its SM count, and the PCIe link it negotiated. |
| `host_id`, `host_boot_id` | opaque ids — the pod, and the physical machine. Two runs sharing `host_boot_id` but not `host_id` were **co-tenants on one box**. |

Use them to tell the two causes apart. High `data_wait_frac` means training sat
waiting on your generator — that is yours to fix. Low `data_wait_frac` *and* a
`host_bench_tokens_per_s` well below the round's other entries means the pod was
slow, not the generator. `host_bench_spec` tags the workload version; only
compare numbers that share it, and read the bench as a host *index* — it is a
different workload from a training step, so its absolute value is not comparable
to `throughput_tokens_per_s`.

This is observability, not an appeal channel: these fields are not signed, not
in the manifest, and never re-weight a score. They exist so a claim about host
variance can be checked against numbers instead of inferred.

## The cascade — warm-started rounds

Rounds no longer always train from random init. When a king survives
`[scoring] cascade_reign_rounds` (5) consecutive rounds undethroned, the
trainer **promotes** up to `cascade_top_k` (3) of that reign's best duel
checkpoints — within `cascade_quality_epsilon` (5%) of the reign's best public
benchmark score, picked for error diversity — as the next **warm-start
generation**. Subsequent rounds rotate through the members, and **every run in
a round — heat and final — trains from that same init**, so the controlled
experiment is untouched: both sides still share one init and your data is
still the only variable. Promotions have fired on mainnet; warm-started
rounds are the live case, not a future feature.

What it means for you as a miner:

- **You are improving the strongest lineage, not teaching from zero.** After
  generation 0 the model already forecasts when your data arrives; corpora
  that add regimes the lineage is weak on beat corpora that re-teach what it
  already knows. The per-domain win-rate table in `cascade duel` shows where
  the current lineage is weak.
- **Losing challengers' checkpoints are promotable too.** The candidate pool
  is every benched duel checkpoint of the reign — the king's *and* the
  challengers'. Promotion pays nothing, but your data can end up shaping the
  init every later round trains from.
- **The promotion can't ratchet downhill.** A ripe reign whose best
  checkpoint benches worse than the live generation's best member holds until
  it produces an equal-or-better one. The king persists through a promotion —
  only a genuine dethrone changes the throne.
- **Where to see it**: `cascade round` shows a
  `warm start — this round trains from …` line with the generation while the
  round is in flight; `cascade heat` prints the same plus the rotation's
  scheduled pick for the next round; and the web dashboard's warm-start
  panel shows the member rotation and which round each init came from. The
  init is pinned in each round's signed manifest, so `cascade-audit`
  verifies it like everything else.

## Study the competition

Every committed generator is content-addressed and **public** — that's what
makes the eval re-derivable, and it makes the current best openly studyable.
Pull the reigning king (or any competitor) and read its code:

```bash
cascade fetch king --network test --chain-toml chain.testnet.toml
# → fetched king-uid3: cascade/testnet-smoothgp@sha256:…
#   inspect it, or fork + improve it:  cascade verify ./fetched-king-uid3

cascade fetch 13 --out ./chal13      # a specific UID
cascade fetch 5Haf…                  # a specific hotkey (ss58)
cascade fetch namespace/repo@sha256:…  --verify   # a raw ref; --verify runs the checks
```

This is the game: the best generator is visible, and you win by **improving**
on it, not hiding — a byte-identical copy of the king is dropped before it
trains (it can only tie), so you have to genuinely beat it. Read-only; no wallet
needed, just Hub read credentials.

**Prior eval windows are public too.** The private eval pool rotates every
round; once a round's windows have rotated out they are published (on a lag,
so nothing live is ever revealed) to the
[`Tensor-Link/cascade-eval-pool`](https://huggingface.co/datasets/Tensor-Link/cascade-eval-pool)
dataset on HuggingFace. Use them to see exactly what past duels were scored
on, replay a verdict against your own generator locally (download a round's
windows and point `cascade score --pool-dir` at them), or study which domains
and horizons the pool actually exercises.
Training against them directly is pointless by design — the live rounds are
always scored on windows that have never been published.

## Common failures

| symptom | cause |
|---|---|
| `cascade verify` fails determinism | an unseeded RNG, `hash()`, wall-clock, or set iteration order — make `generate()` pure in `seed` |
| `blocked_import` | a banned import (`socket`, `subprocess`, `pickle`, …); see `chain.toml [static_guard]` |
| `requirement_not_hash_locked` | every `requirements.txt` line needs `--hash=sha256:…`; only allowlisted packages |
| deploy: Hub auth error | `HIPPIUS_HUB_USERNAME`/`PASSWORD` (or `HIPPIUS_HUB_TOKEN`) not exported |
| `registry upload failed` (Hub outage) | the Hippius Hub is down — retry, or add `--hf-repo` + `HF_TOKEN` to submit via the HuggingFace fallback ([§5b](#5b-if-the-hippius-hub-is-down)) |
| committed but never in a receipt | committed *at/after* the epoch boundary → it competes next round (check the deadline with `cascade round`, [§5c](#5c-time-your-submission--cascade-round)); or it failed to train (heat drops it — `cascade heat` shows it as `did not train`) |
| loses every heat | expected while you iterate — the pool is broad real-world data; widen your prior (mix families) rather than fitting one shape. `cascade heat --hotkey <you>` shows how far off you were, published as soon as each heat settles |
