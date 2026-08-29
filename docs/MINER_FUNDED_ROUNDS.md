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
| Miner command (`cascade fund` / `--withdraw`) | `cascade/miner/cli.py` | built |
| Scoped per-pod credentials for funded pods | — | **gates arming** |
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

## What the provisioner must NOT do on funded pods

- Rent them on the operator key (`provision/funded.py` has no such path —
  keep it that way).
- Forward shared credentials (`HIPPIUS_*`, `HF_TOKEN`, `WANDB_*`): the payer
  owns the Lium account and can console into their own pod. Until scoped
  per-pod credentials exist, funded legs must run credential-free (artifacts
  pulled by the orchestrator, not pushed by the pod).
- Auto-retry a rent in place — every attempt spends the miner's budget.
  Retry cadence lives on queue requeues (`should_recover` + attempt caps).

## Miner flow

```
cascade deploy …                       # unchanged: upload + commit/reveal
export LIUM_API_KEY=sk-…               # your key, env only — never argv
cascade fund https://<intake> --ref <repo@digest> \
    --wallet-name w --wallet-hotkey h
# queue position: GET <intake>/v1/queue (or `cascade round`)
cascade fund https://<intake> --ref <repo@digest> --withdraw \
    --wallet-name w --wallet-hotkey h   # while still queued only
```

Failure semantics, as a miner experiences them: a dead pod / sold-out market
/ 429 **requeues your entry without burning it** (sold-out waits as long as
it takes; infra faults get bounded retries on your kept key); an invalid or
revoked key fails your entry `auth` — fix the key and fund again; your
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
