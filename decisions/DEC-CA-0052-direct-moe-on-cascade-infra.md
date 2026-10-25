---
id: DEC-CA-0052
type: decision
title: "Direct ~220M MoE runs on cascade's infrastructure: miner-chosen Lium/Vast/RunPod pods on the payer's own key, SSH dispatch with orchestrator harvest to Hippius, emission rewards, and validators that check the trainer's signed scores instead of evaluating"
status: proposed
date: 2026-10-17
tags: [v2, direct-moe, compute, funding, validation, consensus]
revisit_when: "the live era grid length changes (direct rounds drift until a new journal); or 3-day rounds prove too short for miners to iterate or too long for checkpoint progress; or a trainer is seen stamping scoring blocks late (move the block stamp to a chain-anchored journal head); or the stale-king window proves too short for real cohort durations; or the owner network-config key is lost or must rotate (validators pin it by flag); or a marketplace changes its key scoping (re-check that a miner can withhold billing/withdraw scope); or a cascade-audit --reproduce run FAILs a decision (trainer dispute: decide the remedy, e.g. journal correction or validator spot checks); or private panels cannot be rotated fast enough to reveal them; or real recovery qualification on any of the three marketplaces fails"
relations: {depends_on: [DEC-CA-0036, DEC-CA-0050], relates_to: [DEC-CA-0043, DEC-CA-0051]}
---

## Decision

Owner (2026-10-17): "we don't want AWS, we use Lium / Hippius"; "add vast ai
runpod as compute providers and for miners to be able select which provider
(and supply current key) — if they allow the same scoping that Lium does";
"validators don't eval at all, they just check scores provided by the trainer".

1. **Compute.** A direct job rents ONE pod holding its whole GPU gang on the
   marketplace its payer chose, billed to the payer's key — the DEC-CA-0036
   shape (`cascade/direct/compute.py` over `cascade/provision/{core,vast,runpod}.py`).
   The pod name derives from the reservation key and is outside the rolling
   reaper's pattern; a launch is submitted at most once per key; pod id and SSH
   host key are pinned on first sight. The EC2/STS/IAM path is deleted.
2. **Keys.** `POST /v2/direct/credentials` (hotkey-signed; the signature binds
   `sha256("<provider>:<key>")`) stores the key in that marketplace's vault.
   Lium keeps the rolling vault; Vast/RunPod use subdirectories, so the rolling
   trainer never sees a non-Lium key. Admission requires a key on file for
   every provider the signed terms name.
3. **Scoping.** Lium keys are full-account. Vast offers category-scoped keys
   (`instance_read/instance_write/misc`, no billing/user); RunPod's Restricted
   keys need GraphQL read/write for pods but drop Serverless/AI scopes. Both are
   at least as narrow as Lium, which was the owner's bar.
4. **Transport and storage.** Pods carry no credentials: the orchestrator stages
   inputs by digest over SSH, launches the existing supervisor detached, and
   harvests checkpoints (tar, `data` filter), re-hashing every object before it
   reaches Hippius S3 and before the controller records the pointer. SSM/S3/KMS
   are deleted.
5. **Rewards.** Emissions, never transfers: `settle_boundary` splits a payout
   boundary's continuation pool pro rata across earned, unpaid entitlements in
   the journal reducer itself; validators turn the boundary into weights. The
   Balances-transfer outbox and nonce coordinator are deleted.
6. **Validation.** Validators fetch the trainer's sr25519-signed journal from
   Hippius, verify it against the configured trainer hotkey and replay every
   event and decision from the trainer's per-case scores. They load no
   checkpoint and run no model.
7. **Trust model: reproduce to verify.** Owner (2026-10-17): "we will have a
   trust model where people reproduce results to verify they are true with our
   audit system". Validators prove the journal is internally consistent;
   whether its scores are TRUE is proven by reproduction through `cascade-audit`:
   - every decision carries a `scoring` binding — both checkpoint manifests
     (digest-checked against the judged ids in `replay_decision`), the
     architecture, the private/public panel digests (or sidecar dataset
     revisions), samples, seed, device and scorer version;
   - checkpoints are written public-read (`storage.public_read`), so anyone can
     fetch them credential-free;
   - a private panel stays private while it scores; once the operator rotates
     it out, `python -m cascade.direct --reveal-panel PANEL.json` publishes it to
     `direct/reveal/<digest>.{json,npz}` (refused for the live panel);
   - `cascade-audit direct-moe --journal-bucket B --trainer-hotkey T --reproduce`
     re-scores every decision: PASS when every per-case score reproduces within
     `--tolerance` and the re-derived verdict matches; FAIL on any divergent
     score/verdict, a binding that names other checkpoints, or a missing/corrupt
     bound object; WARN while a panel is unrevealed or no sidecar is configured.
   **Accepted risk:** a false score is caught after the panel is revealed, not
   when it pays; between decision and reveal the public-suite half is the only
   live check. The per-case tolerance (default 1e-3 relative) is not yet
   calibrated across GPU types.

8. **Owner network config: the main validator sets every rule.** Owner
   (2026-10-18): "we need it so the main validator can change anything and
   they check our config and score or set weights". The owner hotkey signs a
   hash-chained config (`cascade/direct/network.py`, scope `network-config`,
   `direct/network/latest.json`) naming the trainer hotkey, journal location,
   burn uid, an all-burn switch, the ledger and reward policies and allow-lists
   of guard and private-policy digests. Validators pin only the owner hotkey
   and the config bucket (`--direct-owner-hotkey`, `--direct-config-bucket`),
   re-read it each poll, refuse a lower `seq`, and refuse a journal that used
   a rule the config does not list. Changing a direct-MoE rule is a republish,
   never a validator update or a `chain.toml` edit; a change to the replay CODE
   still needs a validator release. **Accepted risk:** the owner key now
   decides direct-MoE weights outright (burn_all, trainer rotation); its loss
   or compromise is a subnet-wide incident, so keep it offline.
9. **Validators enforce auditability; the config is anchored on chain.** Owner
   (2026-10-18) approved the checks that keep every round reproducible without
   validators running a model, and asked for an on-chain commitment.
   - The owner hotkey (a dedicated registered hotkey: one plain commitment per
     hotkey, never one that posts DEC-CA-0045 readiness notes) commits
     `cascade-direct:1:<digest(config body)>` at every publish. Validators
     accept only the committed config; the chain's block for that commitment
     is its timestamp.
   - `private_panels` is a schedule `[{panel, from_block}]`, one panel active
     at a time. A panel added or re-timed after a validator's last accepted
     config must start at or after the commitment block, so a panel cannot be
     listed after its scores were known. The schedule is append-only: dropping
     an entry re-maps old decisions and refuses the journal.
   - Every decision must carry its `scoring` binding including `block` (the
     chain block at scoring time, stamped by the evaluator from `chain` in the
     trainer config), not in the future, and use the panel active at it.
   - Both bound checkpoints must answer a credential-free HEAD on Hippius.
   - A private panel first used `reveal_after` seconds or more before the
     boundary being paid must be revealed, or the validator burns that
     boundary until it is. The trainer must rotate panels at least that often.
   **Accepted risk:** a validator seeing the config for the first time trusts
   the schedule it finds (it persists the last accepted config under
   `--cache-dir`, so restarts do not reset this); the scoring block is stamped
   by the trainer, so a trainer can claim a later block within an already
   committed window but cannot pick a panel listed after the fact. The reveal
   check is existence only; `cascade-audit` verifies the content.
10. **Stale-king burn.** Owner (2026-10-18): "we want to work on 1 checkpoint,
   no rotation, only promotion. If their generator doesn't keep improving the
   checkpoint they get a 90% burn of their emissions". `RewardPolicy` gains
   `stale_after` (journal clock) and `stale_burn` (fraction): at a payout
   boundary, a king whose own last promotion (`king_history`) is
   `stale_after` or older has `stale_burn` of the throne share burned
   (`allocate_boundary` records it as `stale_burn`). Royalties to past kings and
   continuation entitlements are unaffected. Defaults (0, "0") leave old
   policies replaying unchanged; the network config pins the live values.
11. **Rounds on the era clock.** Owner (2026-10-18): "a new round every 2-3
   days, and when a king's own checkpoint doesn't beat the last there is no
   progression"; "we need enough time miners can optimise". Agreed: one round
   = 6 eras (3 days), on the live era grid. The ledger policy gains optional
   `round_origin`, `round_blocks`, `era_blocks` (all or none; `round_blocks` a
   whole number of eras; the owner publish command checks the origin is a live
   era start). A round journal stamps the chain block on every cohort, settle,
   advance and payout event (non-decreasing; validators refuse one past the
   chain head), and the reducer enforces:
   - one cohort per round, opened only after the previous round's winner
     advanced;
   - a round's winner advances the checkpoint only after the round closes; a
     round with no winner (the king's own block did not beat the checkpoint
     and no challenger did) leaves the checkpoint where it is;
   - at most one payout per era;
   - the stale-king burn (§10) by rounds instead of `stale_after`: a king
     with no promotion of its own in the last full round loses `stale_burn`
     (90%) of the throne share until its next promotion.
   Journals without the round fields replay unchanged. **Accepted risk:**
   `era_blocks` is fixed per journal; a later change of the live grid length
   drifts the direct rounds off the era grid until a new journal starts.

Activation stays refused (`DirectConfig.validate()`); nothing here rents a pod
or sets a weight on mainnet.
