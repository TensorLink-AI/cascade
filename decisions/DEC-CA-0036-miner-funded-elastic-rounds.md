---
id: DEC-CA-0036
type: decision
title: "Miner-funded challenger legs under elastic-cadence rounds; king and confirmation legs stay operator-funded"
status: proposed
date: 2026-08-28
tags: [funding, provisioner, round, cadence, intake, security, cost]
revisit_when: "the funded queue routinely overflows max cadence × finalist_cap for days (the k-cap or the grid is the wrong bottleneck); or shadow mode shows most of the field funding but the screen still deciding outcomes (retire the heat formally); or scratch-shadow retrain-noise measurements put same-generator retrain variance above the decayed margin floor (the confirmation gate needs the un-decayed margin, or the floor moves); or Lium reliably lacks the pinned final SKU so funded entries live in capacity-requeue purgatory (multi-provider BYOK or per-size pins, DEC-CA-0027)"
relations: {builds_on: [DEC-CA-0003, DEC-CA-0008, DEC-CA-0012, DEC-CA-0016], external: "PRISM (subnet 100), tensorlink-dev/base: prism-lium-payer / lium-rent-pool / prism-challenge"}
---

Challenger training legs are billed to the **submitting miner's own Lium API
key**; the operator keeps every other kind of control — the rental, the pinned
image and SKU, the orchestration, the eval. Costs scale with the field and
land on the miners creating them. The pattern is PRISM's (subnet 100): miner
sends `X-Lium-Api-Key` with an authenticated request, operator re-executes on
a pod rented with that key, fail-closed. Cascade keeps its ROUND structure —
the round model is what makes duels paired (one king leg, one draw, one
cohort) — and gets demand-responsiveness from **elastic cadence** instead of
per-submission pipelines.

## The shape

1. **Funding channel.** One new HTTP surface (`cascade-intake`): a
   hotkey-signed POST binding the miner's Lium key to their *revealed* commit
   ref. Keys live in a memory-first, TTL-sealed vault (0600 files so a
   restart can still tear down; never in any shared store, never logged —
   the queue, standings, and manifests carry hotkeys and refs only). An API
   key can never ride the chain commit: a reveal is public forever.
2. **Funded queue = the demand signal.** Entries order by reveal block
   (earliest-commit seniority, per [[DEC-CA-0008]] and the UID invariant),
   one live entry per hotkey. `[round] funded_mode = "required"` makes the
   queue the round's field: at most `finalist_cap` funded entries enter,
   which trips the heat's fits-the-cap fast path — **no GPU screen runs;
   every funded entrant duels**. The screen thereby retires operationally
   without touching consensus or [[DEC-CA-0012]]'s statistics: per-round k
   stays capped where the α/k correction was sized, and demand raises the
   NUMBER of rounds, never k.
3. **Elastic cadence rides existing machinery.** Validators never schedule
   rounds — they poll `read_latest_manifest` and score what appears — so
   scale-DOWN is pure trainer policy (`skip_unfunded_rounds`: an unfunded
   boundary runs nothing; the king holds, consensus-invisibly). Scale-UP is
   the one consensus-touching move: set the epoch grid to the maximum
   cadence via the existing scheduled `epoch_blocks` switch
   (`epoch_blocks_prev` / `epoch_activation_block` — built for exactly
   this), then skip empty boundaries. 1–4 rounds/day = a 6h grid plus the
   floor.
4. **Per-payer provisioning.** One `LiumProvider` per payer key (PRISM's
   per-submission backend): the key enters only the rent subprocess's env,
   never argv or our environ; pods are ledgered with `payer_hotkey` because
   teardown and orphan-reap on a miner's account REQUIRE the miner's key
   (the operator's key cannot even see those pods). Rent failures classify
   auth / rate-limited / no-capacity / infra; **an infra fault never burns
   an entry** — it requeues on the miner's kept key (bounded attempts);
   sold-out waits unboundedly; auth is the miner's to fix. A rent is never
   auto-retried in-place: each attempt spends the miner's budget.
5. **The king's leg (and the reference) stay operator-funded, every round.**
   Owner decision 2026-08-28. And on a provisional dethrone, the operator
   **retrains the winner's generator on the operator's own account** — same
   init, contract, seeds — and the crown stands only if the operator-produced
   checkpoint clears the margin again. A Lium account owner can always
   console into their own pod, so nothing trained on a miner-funded pod is
   ever the artifact consensus anchors on; the miner-funded leg is a paid
   screen, the confirmation leg is the binding one. Operator GPU cost is
   O(rounds + dethrones), not O(field).

## Consequences owned

- **No shared operator credentials on miner-funded pods** (HIPPIUS_*,
  HF_TOKEN, WANDB_*): stdin-piped env is readable by root on the box, and
  the payer is root-adjacent by account ownership. Funded legs get scoped
  per-pod credentials or push artifacts through the orchestrator.
- **Tenure decay ([[DEC-CA-0016]]) must re-denominate to wall-clock/blocks**
  before "required" arms: with elastic cadence, tenure-in-rounds is
  demand-coupled — an attacker could fund junk entries to raise cadence and
  grind the margin to its floor on demand.
- **No balance pre-check** (PRISM's choice too): a key is validated by use;
  underfunded surfaces as a classified rent failure that requeues.
- **Dedup stays pre-duel.** Anyone can now pay in, so [[DEC-CA-0008]]'s
  content screen matters more, not less.
- **Registration economics can relax**: the compute funding is the per-entry
  cost, so `one_submission_per_hotkey`'s lifetime bar can move toward
  PRISM's one-live-entry-per-hotkey (the queue already enforces that shape).

## Direct submissions + champion-only publication (second half, same node)

Code POSTs straight to the gateway — the SAME request that carries the Lium
key — and nothing is miner-hosted. This kills the account-watching and
delete-my-repo-after-losing problems in one move:

1. **The vault ref rides the existing grammar.** The intake stores the ZIP
   privately (content-addressed sha256, ownership recorded) and the miner
   chain-commits ``metro-v1:gen:hippius:vault/direct@sha256:<hex>`` — a
   byte-ordinary payload every deployed validator already parses, so
   participant sets cannot fork and no coordinated validator release is
   needed. Only the FETCH path branches (``fetch_from_hub`` → the private
   store / staged pod ZIP / the published champion object).
2. **Ownership is enforced, not assumed.** A digest is owned by its earliest
   uploader; a copied digest (the published champion's, most obviously)
   committed by another hotkey is dropped at field entry. Byte-copies
   arriving as fresh uploads still die at [[DEC-CA-0008]]'s dedup screen.
3. **Only thrones publish** (the sn100/PRISM ``top-model/`` pattern):
   ``champions/<digest>.zip`` + index land public-read on the manifest
   bucket per ``[round] champion_publish`` — ``crown`` (immediately),
   ``delay`` (after N reign rounds), ``dethrone`` (at hand-off). Every
   policy reveals a deposed vault king at latest, so history stays fully
   re-derivable; only the LIVE king's privacy varies, and losers never
   publish. ``cascade fetch king`` resolves the published object
   anonymously; ``cascade submit`` is the one-request miner flow.
4. **Accepted trade, stated plainly:** third-party re-derivation of
   non-champion entries is gone (their code is operator-private), and under
   ``delay``/``dethrone`` the live king's is too until reveal. The audit
   trail for every PUBLISHED reign is intact; auditing a private entry means
   trusting the operator or a disclosure agreement. This is PRISM's exact
   posture and the price of killing copy-watching.

Additional arming gate beyond the list below: per-dispatch ZIP staging
(``SubmissionStore.stage_for_dispatch`` exists; the remote-dispatch wiring
that places exactly ONE entry's ZIP on its pod, and ``$CASCADE_VAULT_DIR`` /
``$CASCADE_CHAMPION_BASE`` in the pod environment, are deploy work) — a
funded pod's payer can read the box, so the whole store must never ship.

## Built (this node's landing, all inert at shipped defaults)

`cascade/funding/` (vault, fault taxonomy, queue, intake + `cascade-intake`),
per-payer `LiumProvider` + `payer_hotkey` ledger attribution + funded
rent/teardown/reconcile (`cascade/provision/funded.py`), `[round]
funded_mode/funded_queue_path/skip_unfunded_rounds/max_rounds_per_day`
(digest-inert; `"off"` shipped), the trainer's funded field filter +
settle marking + boundary skip, and `cascade fund` / `fund --withdraw`
miner commands. Arming order: intake deployed → shadow on testnet →
required on testnet (with the grid change) → mainnet release-then-activate.
NOT built here: scoped pod credentials, the confirmation-leg wiring, the
tenure re-denomination, and the retrain-noise measurement — each gates
arming, none gates landing.

## Amendment 2026-09-02 — elastic no-heat field, per-round GPU choice, review hardening

Built and live-validated on testnet 259 (rounds 9997246590844856043,
9598728707958414075), extending the landing above; all still inert at
shipped defaults:

* **Per-payer pods armed-able**: `[round] funded_pods = "rent"` wires
  `provision/funded.py` into the trainer — rent on the payer's key →
  dispatch → verified teardown, a write-ahead ledger
  (`_train_work/funded_pods.json`, intent row BEFORE `lium up`), a sweep at
  every round entry + trainer startup, and the burn point moved from heat
  settle to DUEL settle (`_settle_funded`: manifest entry → done; auth/rc=3
  → terminal; else requeue per taxonomy — a leg lost to operator infra can
  no longer burn a paid entry).
* **Elastic no-heat field**: `funded_field_cap` seats up to N funded
  challengers (0 = legacy finalist_cap); the whole seated field advances to
  the duel (the heat cap yields to the funded field — no screen ever runs
  on an all-funded field); `funded_capacity_probe`/`funded_capacity_reserve`
  clamp admission to the live 1-GPU market; held-back seats are touched so
  they never TTL-expire while waiting.
* **Per-round GPU type**: `funded_pod_skus` (preference-ordered allow-list)
  — each boundary probes every type and the round runs entirely on the
  most-available one, king included via `funded_king_rent` (JIT
  operator-billed king pod, ledgered payer_hotkey="", swept next boundary).
  Requires `[training] expected_gpu = ""` (enforced at trainer launch).
* **Transparency**: public per-round roster `funded/round-<id>.json` +
  `funded/latest.json` (cap, market capacities, seniority order with
  on-chain reveal blocks, outcomes), the `cascade queue` miner command, and
  the WARN-only tier-0 `funded-roster` audit check ("the queue was jumped"
  is a named, reproducible warning).
* **Review hardening (2026-09-02 pre-deploy review)**: intake requires
  subnet REGISTRATION (fail-closed metagraph oracle) + per-hotkey stored
  quota + ZIP member-count cap + NaN-proof, strictly-increasing signed
  timestamps + v2 canonical message binding the key header's sha256 +
  whole-connection deadline + bounded upload buffers; funded pod names are
  deployment-scoped (`cascade-n<netuid>-…`) and OFF the provisioner reaper
  scheme (kill-the-live-king / cross-deployment-reap classes); the
  settled-retry path restores funded state from the heat_complete marker
  (bills stayed payer-side, entries settle-able); rate-limit streaks turn
  terminal past the 6h recovery window; numeric fault markers match only
  standalone tokens (vanity-hotkey steering).

Owner direction this amendment serves: **no heat — miners provide keys, any
number of challengers, all rounds ~3h; coordination = same-GPU-type
capacity, chosen per round from the five cu124-compatible types**
(`["RTX4090", "RTX3090", "L40S", "L40", "A6000"]`; Blackwell waits on the
torch cu12.8 re-pin, a coordinated contract change).

## Amendment 2026-09-02 (2) — block-gated mainnet go-live Fri 2026-09-04

Owner directive: live Friday morning UTC, legacy rounds running right up to
the flip, epoch grid STAYING 12h for now. Shipped in `chain.toml`: `[round]
funded_activation_block` (new key — before it every funded_* knob reads
"off"; from it the armed config applies) and PR #241's `[eval]
scored_from_block` (ladder [64, 256, 720]) BOTH at block 8992800 (≈ Fri
08:30 UTC) — the first 12h boundary after the last legacy round (Thu 20:30,
6–8h, ~4h gap as today): no extra rounds, no overlap, no hold. The 3h grid
is a later scheduled `epoch_activation_block` switch. `expected_gpu`
unpinned ("" — contract change, coordinated deploy before Thu 20:30).
Direct submissions stay off (image gate). Go-live checklist:
docs/MINER_FUNDED_ROUNDS.md; miner-facing: docs/MINER.md §6b + llms.txt.
Accepted risk, owner-directed: the confirmation-leg wiring ships OPEN — a
provisional dethrone crowns without an operator retrain. Revisit within
the first week live.
