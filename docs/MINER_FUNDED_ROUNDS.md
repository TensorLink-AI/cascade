# Miner-funded rounds (DEC-CA-0036) — rollout runbook

Challenger training legs billed to the submitting miner's own Lium API key;
king, confirmation, and eval legs stay on the operator's account. Rounds stay
rounds; cadence becomes elastic (fire only funded boundaries, up to the epoch
grid). Everything below shipped **inert** — `[round] funded_mode = "off"` —
and arms in the order given. The decision node
(`decisions/DEC-CA-0036-miner-funded-elastic-rounds.md`) carries the design
rationale; this file is the how.

## Pieces

| piece | where | state |
|---|---|---|
| Payer-key vault (TTL, 0600, hydrate-for-teardown) | `cascade/funding/vault.py` | built |
| Rent-fault taxonomy (auth/rate/capacity/infra, no-burn rules) | `cascade/funding/faults.py` | built |
| Funded queue (reveal-block seniority, 1 live entry/hotkey) | `cascade/funding/queue.py` | built |
| Intake service (`cascade-intake`, signed `X-Lium-Api-Key` POST) | `cascade/funding/intake.py`, `main.py` | built |
| Per-payer rentals + payer-attributed ledger + payer-key reap | `cascade/provision/funded.py`, `core.py`, `state.py` | built |
| `[round] funded_mode` field selection + boundary skip | `cascade/trainer/loop.py`, `shared/config.py` | built |
| Per-payer funded pods (`[round] funded_pods = "rent"`: rent → dispatch → verified teardown, write-ahead ledger, boundary + startup sweep) | `cascade/trainer/loop.py`, `provision/funded.py` | built, live-validated (testnet 2026-09-02) |
| Elastic no-heat field (`funded_field_cap`, capacity probe/reserve clamp, whole seated field duels) | `cascade/trainer/loop.py` | built, live-validated |
| Per-round GPU-type choice (`funded_pod_skus` preference list, most-available wins; JIT operator king pod via `funded_king_rent`) | `cascade/trainer/loop.py` | built, live-validated |
| Transparency (public roster `funded/round-<id>.json`, `cascade queue`, tier-0 `funded-roster` audit check) | `trainer/loop.py`, `miner/cli.py`, `audit/checks.py` | built |
| Miner command (`cascade fund` / `--withdraw` / `cascade queue`) | `cascade/miner/cli.py` | built |
| Scoped per-pod credentials: payer pods are ISOLATED (nothing from the orchestrator env) and get a per-pod push-only Hub robot, revoked at teardown (`cascade/funding/robots.py`) | `funding/robots.py`, `trainer/loop.py`, `trainer/remote.py` | built (needs a project-admin Hub login or a static funded robot — see "Credentials on payer pods") |
| Worker image rebuilt from this branch (vault-ref fetch on funded pods) + image/orchestrator `[training]` budget parity | — | **gates arming** |
| Confirmation-leg wiring (operator retrain on provisional dethrone) | — | **gates arming** |
| Tenure decay re-denominated to wall-clock/blocks (DEC-CA-0016 amendment) | — | **gates "required" + elastic grid** |
| Retrain-noise measurement (scratch-shadow paired evals) | — | gates confirmation margin choice |

## Operator: bringing it up

1. **Intake.** `cascade-intake --queue-path <work_root>/funded_queue.json
   --vault-dir /root/.cascade/payer-vault --chain-toml chain.toml` behind a
   TLS terminator. `--queue-path` is REQUIRED and must be the exact path the
   trainer resolves for `[round] funded_queue_path` (work_root-relative) —
   there is deliberately no default, because a CWD-relative fallback silently
   split-brains the queue (funds 202 but never enter). The queue writes are
   flock-serialised via a sibling `funded_queue.json.lock` — keep both files
   on one local filesystem (flock over NFS is not a lock). Loopback bind is
   the default; the key header must never cross the wire in clear. The vault
   dir stays off git, off rsync'd pod trees, and off backups that leave the
   box.
2. **Shadow.** Set `[round] funded_mode = "shadow"` (trainer restart; no
   digest impact). Rounds behave identically; logs show `funded shadow:
   N/M eligible challengers are funded`. Watch adoption.
3. **Required (testnet first).** `funded_mode = "required"` +
   `skip_unfunded_rounds = true`. The field is now the funded queue,
   `finalist_cap` at a time, earliest reveal first; the heat's
   fits-the-cap fast path means no screen GPU is spent. Unfunded boundaries
   run nothing.
3a. **Per-payer pods + the elastic no-heat field.** `funded_pods = "rent"`
   moves every funded challenger leg onto ITS payer's own Lium key (one pod
   per seat, verified teardown, write-ahead ledger `_train_work/
   funded_pods.json`, swept at every round entry and trainer startup).
   `funded_field_cap = N` seats up to N funded challengers per round — the
   WHOLE seated field advances to the duel (no heat, ever; alpha/k splits
   over the cohort), wall-clock stays one leg because seats parallelize
   across payer pods. With `funded_capacity_probe`, admission clamps to the
   live 1-GPU market minus `funded_capacity_reserve` (the king's own
   rental); held-back seats wait unburned and un-expired. `funded_pod_skus`
   lists the allowed GPU types preference-ordered: each boundary probes all
   of them and the round runs ENTIRELY on the most-available type — king
   included via `funded_king_rent` (JIT operator-billed king pod at the
   chosen SKU; requires `[training] expected_gpu = ""`, enforced at trainer
   launch). Pod names are deployment-scoped (`cascade-n<netuid>-…`) so
   co-hosted deployments on a shared payer account never reap each other,
   and the provisioner's reaper never touches them.
3b. **Transparency.** Every required round publishes
   `funded/round-<id>.json` + `funded/latest.json` (admission cap, market
   capacities, seated in reveal-block seniority with on-chain blocks,
   waiting, outcomes). Miners read it with `cascade queue` (add `--intake
   <url>` for the live queue); the tier-0 `funded-roster` audit check
   cross-checks it against the signed manifest — "the queue was jumped" is
   a named WARN anyone can reproduce.
4. **Elastic grid (the one consensus-coordinated step).** Shrink
   `epoch_blocks` to the max cadence (6h grid = 4 rounds/day ceiling) via
   the scheduled switch: publish `epoch_blocks_prev` = old,
   `epoch_activation_block` = the flip block, new `epoch_blocks` — every
   validator floors each round on the grid in force at its
   `created_block`, so no fork. This is DEC-CA-0016/0019-class: externals
   upgrade before the activation block.
5. **Mainnet** repeats 2→4 after a full testnet cycle has exercised: a
   funded round end-to-end, an infra requeue on a miner key (no burn), an
   auth-class failure (entry released, not burned), a skipped unfunded
   boundary, and a payer-key teardown after a provisioner restart
   (vault hydrate → `teardown_funded`).

## GO-LIVE 2026-09-04 (mainnet, block-gated — owner 2026-09-02)

The shipped `chain.toml` flips at ONE block; the epoch grid stays 12h
(3600 blocks) for now:

| what | block | projected UTC |
|---|---|---|
| funded machinery live (`[round] funded_activation_block`) | 8992800 | Fri 2026-09-04 ~08:30 |
| scored horizon ladder `[64, 256, 720]` (`[eval] scored_from_block`, PR #241) | 8992800 | Fri 2026-09-04 ~08:30 |

Why this block: it is the first 12h boundary after the last legacy round.
Legacy rounds (6–8h) keep running right up to the flip — the last one
starts at the Thu 20:30 boundary (8989200) and ends ~02:30–04:30, leaving
the usual gap; at 8992800 the funded field and the ladder activate together.
No extra rounds, no hold, no overlap. (Projections assume 12s blocks —
verify against the live chain the day before; keep any moved block a
multiple of 3600.) A later move to a 3h grid is its own scheduled
`epoch_activation_block` switch (a coordinated validator update).

Miners can `cascade fund` as soon as the intake is up — entries queue by
reveal block and the first funded round at 8992800 seats them. With
`one_submission_per_hotkey = true` a hotkey that competed in a legacy
round is spent, so the funded era's entrants are FRESH hotkeys whose
reveal lands after the Thu 20:30 cutoff (an earlier reveal competes in
the last legacy round and burns). Funded entries burn at the DUEL settle,
only when judged or on a generator failure — never on a requeue or an
auth fault. Operator checklist, in order:

1. **Before block 8989200 (Thu ~20:30 UTC, the last legacy boundary):**
   deploy this release — WITH PR #241 merged — to the trainer AND every
   validator (the `expected_gpu = ""` unpin changes `contract_digest`, and
   the horizon ladder forks verdicts from 8992800 on any validator without
   it) — the standard coordinated window; announce to externals. Verify the current pool's
   series-length eligibility at the 720 rung (>= 784 steps) before the block.
2. **Set the payer-pod credential source** on the orchestrator: either
   `CASCADE_HUB_ADMIN_USERNAME/PASSWORD` (a project-admin Hub user — per-pod
   robots) or `CASCADE_FUNDED_HUB_USERNAME/PASSWORD` (a hand-made push-only
   robot). Without one, every funded leg skips (fail-closed).
3. **Start `cascade-intake`** on the orchestrator (see "bringing it up") with
   `--vault-dir` matching `[round] payer_vault_dir` and the queue path the
   trainer resolves; front it with TLS and publish the intake URL to miners.
   NO `--trust-refs` and NO `--no-require-signature` on mainnet, ever.
   Start it before the Thu 20:30 legacy round so miners can pre-fund.
4. **Retire the provisioner's final stage** at the seam: with
   `funded_king_rent = true` the trainer rents/ledgers/sweeps the king pod
   itself; a standing final fleet would idle-bill and the provisioner must
   never touch `cascade-n91-…` pods.
5. **Announce to miners** (docs/MINER.md §6b, llms.txt): funding required
   from block 8992800; `cascade fund` after reveal; registered hotkey; keep
   ~3h × chosen-GPU balance on the Lium key.
6. **Watch the first rounds**: `cascade queue`, `funded/latest.json`,
   `cascade-audit latest` (funded-roster check), and the trainer log's
   admission/SKU lines. Rollback = set `funded_activation_block` far future
   + restart trainer (legacy rounds resume).

Direct submissions stay OFF at go-live (`submission_vault_dir = ""`): the
pinned worker image predates the vault fetch — arming them is a separate
image rebuild + budget-parity release.

## Fronting the intake (DoS posture)

The intake runs on the orchestrator — the box holding the trainer wallet and
the private eval pool — so exposing it raw is not an option. The division of
labour:

**The front proxy does volumetric defence.** Terminate TLS in nginx/caddy (or
put the DNS behind an edge like Cloudflare) and configure, at minimum:

- per-IP connection and request-rate limits (nginx: `limit_conn` /
  `limit_req`; the fund/withdraw endpoints are a few requests per miner per
  day — single-digit req/s per IP is generous),
- header/body read timeouts (`client_header_timeout`, `client_body_timeout`),
- a body size limit matching the intake's ZIP cap (`client_max_body_size` =
  `--max-zip-mb`, or ~1 MB if the intake is funding-only),
- proxy buffering ON, so a slow client dribbles at the proxy, not at a
  handler thread.

**The intake keeps in-app backstops** so a missing or misconfigured proxy
degrades gracefully instead of taking the orchestrator down:

- `--request-timeout` (default 30 s): per-socket read/write timeout — a
  slowloris client gets its connection closed, not a parked thread (the
  stdlib serves with NO timeout by default).
- `--max-connections` (default 64): concurrent-handler cap; over it, new
  connections get an immediate `503` + `Retry-After` and are closed without
  spawning a thread, so a connection flood costs one small write each.
- `--chain-timeout` (default 30 s): deadline on the reveal poll — a hung
  substrate websocket serves the stale reveal table instead of pinning every
  handler slot behind one dead chain connection.
- `--request-deadline` (default 120 s): whole-connection wall clock enforced
  by a reaper — a drip-feed client sending one byte per interval defeats a
  per-op timeout but not this.
- `--max-uploads` (default 4): concurrent submit-body buffers (each is up to
  the ZIP cap in RAM).
- Identity floors: submits (and funds) require a hotkey REGISTERED on the
  subnet (metagraph oracle, fail-closed 503 when the chain is unreachable);
  each hotkey gets `--max-hotkey-mb` (default 256 MiB) of stored
  submissions; ZIPs are capped by bytes, decompressed bytes, AND member
  count; signed timestamps are NaN-proof, strictly increasing per (action,
  hotkey), and the v2 signature binds the sha256 of the key header, so a
  captured request cannot be replayed with a swapped key.
- Structural caps already in the request path: declared-length `413` before
  any body byte is read, the signature gate before body buffering, and the
  streaming decompression cap in the store.

None of this rate-limits per identity — an attacker with many IPs can still
saturate `--max-connections` at the proxy-less intake. That is the front
proxy's job; the backstops only guarantee the box stays responsive and the
trainer keeps its CPU.

## Credentials on payer pods (PRISM-level trust model)

A funded pod runs on the MINER's Lium account: the payer has console access,
so everything in the worker's environment is theirs to read. Rules, enforced
in code:

- The pod is **isolated**: `RemoteHost.isolated` drops every `forward_env`
  and the dispatcher's global extras (`WANDB_API_KEY`); nothing from the
  orchestrator's environment travels. Pulls are anonymous (the repos are
  public by design), so the only credential the worker needs is a Hub push.
- That push credential is a **Harbor robot** scoped to the checkpoint project
  with `repository:push` only — no delete, no other project, no S3, no HF:
  1. **Per-pod robot (preferred):** minted at rent, revoked at teardown and by
     every sweep, Harbor expiry `[round] funded_robot_duration_days` as the
     backstop. Harbor forbids robots managing robots, and the operator's
     everyday Hub identity IS a project robot (`robot$cascade+cascade-bot`),
     so minting needs a project-admin USER login on the orchestrator:
     `CASCADE_HUB_ADMIN_USERNAME` / `CASCADE_HUB_ADMIN_PASSWORD` (used only
     by the minter; never forwarded anywhere).
  2. **Static funded robot (fallback):** a push-only robot you create in the
     Hippius UI, handed to the trainer as `CASCADE_FUNDED_HUB_USERNAME` /
     `CASCADE_FUNDED_HUB_PASSWORD`; rotate it by hand. A user login in that
     slot is refused.
  3. **Neither → the leg fails CLOSED** (skipped, unburned). The operator's
     own Hub login is never an option.
- **Pod identity is pinned for the leg.** Names are owner-chosen and
  reusable, so the trainer records the platform's pod id at rent and the
  container's SSH host key (generated at first boot), dispatches with
  `StrictHostKeyChecking=yes` against that key only, and re-checks the
  platform id before dispatch and when the leg returns. A pod relaunched
  under the same name, or a different container answering at its address,
  settles the entry as **`tamper`** — miner fault, terminal, hotkey spent —
  never as infra. (Verified on live pods: the owner cannot ssh/exec into a
  custom-image pod — only the rent caller's keys are injected — and backups
  are confined to `/workspace`, so replacing the pod is the remaining move,
  and this catches it.)
- **A checkpoint is data, never code.** Every scorer (validator verdict,
  king-pod bench, audit replay, heat screen) used to import and execute the
  `forecast_wrapper.py` / `model.py` shipped INSIDE the checkpoint and build
  the model from its `config.json` — sound while only operator pods produced
  checkpoints, arbitrary code execution on every validator once a miner's pod
  does. `cascade/eval/checkpoint_guard.py` now requires the two files to be
  byte-identical to the release's own copies (any other `.py` is refused),
  `config.json` to equal the contract's model config, and the safetensors
  HEADER (names/dtypes/shapes) plus file size to match the pinned model —
  before a tensor is allocated. The trainer runs the same guard on every
  funded leg's checkpoint before it can enter the manifest (a deviation
  settles as `tamper`); validators/audit/bench run it again on their side.
  Honest checkpoints pass byte-for-byte; nothing about which checkpoints
  bench or promote changes.
- Still open vs PRISM: their pod holds NO credential at all (the master
  SSH-harvests the checkpoint through a secure receive). Ours needs the
  pinned worker to gain a local-only mode — the next worker-image release —
  after which payer pods carry zero credentials. Also PRISM seals payer keys
  at rest with a key file; our vault is plaintext 0600 (operator-local).

## What the provisioner must NOT do on funded pods

- Rent them on the operator key (`provision/funded.py` has no such path —
  keep it that way).
- Forward shared credentials (`HIPPIUS_*`, `HF_TOKEN`, `WANDB_*`): the payer
  owns the Lium account and can console into their own pod. Funded hosts are
  `isolated` and carry only their per-pod robot (above).
- Auto-retry a rent in place — every attempt spends the miner's budget.
  Retry cadence lives on queue requeues (`should_recover` + attempt caps).

## Direct submissions (private code, champion-only publication)

With `--submission-dir` on the intake and `[round] submission_vault_dir` +
`champion_publish` on the trainer, code never touches a miner-hosted repo:

- `POST /v1/submit` (body = the ZIP, signature over its sha256, optionally
  `X-Lium-Api-Key` in the SAME request) stores it in the operator-private
  store and returns a `vault/direct@sha256:<hex>` ref + the exact chain
  payload — a byte-ordinary hippius commit, so no validator changes.
  A submit-with-key parks `pending_reveal` and auto-queues when the chain
  reveal resolves (the intake sweeps on every request).
- Ownership: earliest upload owns a digest; another hotkey committing your
  digest is dropped at field entry; byte-copies still die at dedup.
- Publication: ONLY thrones publish (`champions/<digest>.zip` + index,
  public-read on the manifest bucket) per `champion_publish`: `crown` /
  `delay` (after `champion_publish_delay_rounds`) / `dethrone`. A deposed
  vault king always reveals at hand-off; losers never do.
- Pods: a dispatch must stage exactly ONE entry's ZIP
  (`SubmissionStore.stage_for_dispatch` → the pod's `$CASCADE_VAULT_DIR`);
  the king's published code also resolves via `$CASCADE_CHAMPION_BASE`
  (`{s3_endpoint}/{manifest_bucket}`). Staging wiring is an ARMING GATE —
  never rsync the whole store anywhere.
- Reveal timing: a submit-with-key entry parks `pending_reveal` and its key
  lives in the vault under the same TTL, so submit (or fund) within the key
  TTL of your reveal — a reveal timed beyond it fails the entry because the
  key it needs is already gone (re-submit+fund closer to reveal). Default
  36h TTL vs a ≤12–24h timed reveal leaves ample headroom; keep
  `[round] funded_entry_ttl_hours` = `cascade-intake --ttl-hours`.

## Miner flow

```
# Direct (one request: code + funding; private until it takes the throne):
export LIUM_API_KEY=sk-…               # your key, env only — never argv
cascade submit ./my-generator https://<intake> --wallet-name w --wallet-hotkey h
# → stores privately, chain-commits the vault ref, auto-funds on reveal

# Or the classic Hub path + explicit funding:
cascade deploy …                       # unchanged: upload + commit/reveal
cascade fund https://<intake> --ref <repo@digest> \
    --wallet-name w --wallet-hotkey h
# queue position: GET <intake>/v1/queue (or `cascade round`)
cascade fund https://<intake> --ref <repo@digest> --withdraw \
    --wallet-name w --wallet-hotkey h   # while still queued only
cascade fetch king                     # published champions resolve anonymously
```

Your hotkey must be **registered on the subnet** before the intake accepts a
submission or fund (403 `not_registered` otherwise) — reveal + registration
are what give an entry seniority to claim.

Failure semantics, as a miner experiences them: a dead pod / sold-out market
/ 429 **requeues your entry without burning it** (sold-out waits as long as
it takes; a rate-limit streak longer than 6h turns terminal — fix the key's
limits and fund again; infra faults get bounded retries on your kept key);
an invalid or revoked key fails your entry `auth` — fix the key and fund
again; your
generator crashing is your run, spent as ever. Three more terminal classes
exist so a dead entry can never squat in the queue: `ref_mismatch` (you
re-revealed a different ref — fund the new one), `burned` (the hotkey
already used its one submission), and `funding_expired` (the entry outlived
the key TTL without entering a round). All are re-fundable immediately.
Your key is held at most 36h (TTL), forgotten on withdraw, and never stored
anywhere but the operator's sealed vault.

## Cost model after arming

Operator: one king leg per fired round + one confirmation leg per
provisional dethrone + CPU evals + the intake box — O(rounds + dethrones).
Miners: their own leg, win or lose. Nobody pays for screening, because
nothing is screened: the queue caps the cohort and everyone who enters
duels.
